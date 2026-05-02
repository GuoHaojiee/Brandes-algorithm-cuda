/*
 * solution_mpi.cpp — 分布式 Brandes BC（GPU辅助 + 批量源节点）
 *
 * 核心优化：每轮同时处理 BATCH_SIZE 个源节点的 BFS
 *   - GPU 一次调用扩展 BATCH_SIZE 个 frontier（bc_gpu_expand_batch）
 *   - 正向/反向各只需 1 次 MPI 集合操作覆盖 BATCH_SIZE 个源
 *   - MPI 调用次数 ≈ n/BATCH_SIZE × 层数，比逐源处理少 BATCH_SIZE 倍
 *
 * 内存：O(BATCH_SIZE × n/p) = O(n/p)（BATCH_SIZE 是常数）
 */

#include "defs.h"
#include <vector>
#include <cstdio>
#include <algorithm>
using namespace std;

static const int BATCH_SIZE = 64;

/* GPU 函数（brandes_gpu.cu）*/
extern "C" void bc_gpu_init(int local_n, int local_m, int batch_size,
                             const int* offset, const int* dest, int gpu_device);
extern "C" int  bc_gpu_expand_batch(
    const int* all_front, const int* front_offsets, const long long* front_sigma,
    int batch_sz, int total_fz, int v0_global,
    int* out_b, int* out_src, int* out_dst, long long* out_sig);
extern "C" void bc_gpu_cleanup(void);

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int loc_n = (int)G->local_n;
    const int loc_m = (int)G->local_m;
    /* 本进程第一个本地顶点的全局 ID（顶点连续分配，local l → global v0+l）*/
    const int v0    = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    /* ---- 上传本地 CSR 到 GPU ---- */
    {
        vector<int> loff(loc_n + 1), ldst(loc_m > 0 ? loc_m : 1);
        for (int i = 0; i <= loc_n; i++) loff[i] = (int)G->rowsIndices[i];
        for (int i = 0; i < loc_m;  i++) ldst[i] = (int)G->endV[i];
        bc_gpu_init(loc_n, loc_m, BATCH_SIZE, loff.data(), ldst.data(), rank % 2);
    }

    /* ================================================================
     * BFS 状态：BATCH_SIZE 份，每份 O(n/p)
     * 外层下标 b = 当前批次中的 slot（0 .. batch_sz-1）
     * ================================================================ */
    vector<vector<int> >       bd    (BATCH_SIZE, vector<int>      (loc_n));
    vector<vector<long long> > bsigma(BATCH_SIZE, vector<long long>(loc_n));
    vector<vector<double> >    bdelta(BATCH_SIZE, vector<double>   (loc_n));

    /* 前驱列表：local_preds[b][w] = 本地前驱 v_local 列表 */
    vector<vector<vector<int> > > bpreds_l(
        BATCH_SIZE, vector<vector<int> >(loc_n));
    /* 远程前驱：remote_preds[b][w] = (v_global, sigma_v) 列表 */
    vector<vector<vector<pair<int,long long> > > > bpreds_r(
        BATCH_SIZE, vector<vector<pair<int,long long> > >(loc_n));

    /* 每层访问的本地顶点（反向传播用）*/
    vector<vector<vector<int> > > blevels(BATCH_SIZE);
    /* 当前层 frontier */
    vector<vector<int> > bcur(BATCH_SIZE);
    /* 下一层 frontier（循环内复用）*/
    vector<vector<int> > bnext(BATCH_SIZE);

    /* ---- GPU 输出缓冲（最大 BATCH_SIZE × local_m）---- */
    int max_out = BATCH_SIZE * max(loc_m, 1);
    vector<int>       go_b  (max_out), go_src(max_out), go_dst(max_out);
    vector<long long> go_sig(max_out);

    /* ---- GPU 输入缓冲（逐级复用）---- */
    vector<int>       all_front   (loc_n);          /* 合并 frontier */
    vector<int>       front_off   (BATCH_SIZE + 1); /* 分段偏移 */
    vector<long long> front_sigma (loc_n);          /* 对应 sigma */

    /* ---- MPI 通信元数据（逐级复用）---- */
    vector<int> scnt(nproc), rcnt(nproc), sdisp(nproc+1), rdisp(nproc+1);

    /* 正向 BFS 消息暂存（按目标进程）：batch_id / v_global / w_global / sigma_v */
    vector<vector<int> >       fb(nproc), fv(nproc), fw(nproc);
    vector<vector<long long> > fs(nproc);

    /* 反向传播消息暂存（按目标进程）：batch_id / v_global / contribution */
    vector<vector<int> >    qb(nproc), qv(nproc);
    vector<vector<double> > qc(nproc);

    for (int i = 0; i < loc_n; i++) result[i] = 0.0;
    double t0 = MPI_Wtime();

    /* ================================================================
     * 主循环：每次处理 BATCH_SIZE 个源节点
     * ================================================================ */
    for (int s_start = 0; s_start < n; s_start += BATCH_SIZE) {
        int batch_sz = min(BATCH_SIZE, n - s_start);

        /* ---- 初始化本批 BFS 状态 ---- */
        for (int b = 0; b < batch_sz; b++) {
            int s_gl = s_start + b;
            fill(bd[b].begin(), bd[b].end(), -1);
            fill(bsigma[b].begin(), bsigma[b].end(), 0LL);
            for (int i = 0; i < loc_n; i++) {
                bpreds_l[b][i].clear();
                bpreds_r[b][i].clear();
            }
            blevels[b].clear();
            bcur[b].clear();

            /* 只有 s_gl 的 owner 才初始化种子 */
            if (VERTEX_OWNER((vertex_id_t)s_gl, G->n, G->nproc) == rank) {
                int s_lc = s_gl - v0;
                bd[b][s_lc]     = 0;
                bsigma[b][s_lc] = 1LL;
                bcur[b].push_back(s_lc);
                blevels[b].push_back(vector<int>(1, s_lc));
            } else {
                blevels[b].push_back(vector<int>());
            }
        }

        int cur_level = 0;

        /* ============================================================
         * 正向 BFS：BATCH_SIZE 个源节点同步逐层推进
         * 每层：
         *   1. 合并所有批次 frontier → 一次 GPU expand_batch
         *   2. CPU 区分本地/远程邻居
         *   3. 一次 MPI 交换覆盖所有批次
         *   4. 处理收到的消息
         * ============================================================ */
        while (true) {

            /* ---- 合并 frontier ---- */
            int total_fz = 0;
            front_off[0] = 0;
            for (int b = 0; b < batch_sz; b++) {
                for (int i = 0; i < (int)bcur[b].size(); i++) {
                    all_front  [total_fz] = bcur[b][i];
                    front_sigma[total_fz] = bsigma[b][bcur[b][i]];
                    total_fz++;
                }
                front_off[b + 1] = total_fz;
            }

            /* ---- GPU：批量扩展所有批次的 frontier ---- */
            int ne = bc_gpu_expand_batch(
                all_front.data(), front_off.data(), front_sigma.data(),
                batch_sz, total_fz, v0,
                go_b.data(), go_src.data(), go_dst.data(), go_sig.data()
            );

            /* ---- 区分本地/远程 ---- */
            for (int p = 0; p < nproc; p++) {
                fb[p].clear(); fv[p].clear(); fw[p].clear(); fs[p].clear();
            }
            for (int b = 0; b < batch_sz; b++) bnext[b].clear();

            for (int e = 0; e < ne; e++) {
                int       b     = go_b  [e];
                int       w_gl  = go_dst[e];
                int       w_own = VERTEX_OWNER((vertex_id_t)w_gl, G->n, G->nproc);
                long long sig_v = go_sig[e];
                int       v_lc  = go_src[e] - v0;

                if (w_own == rank) {
                    int w_lc = w_gl - v0;
                    if (bd[b][w_lc] == -1) {
                        bd[b][w_lc]     = cur_level + 1;
                        bsigma[b][w_lc] = sig_v;
                        bpreds_l[b][w_lc].push_back(v_lc);
                        bnext[b].push_back(w_lc);
                    } else if (bd[b][w_lc] == cur_level + 1) {
                        bsigma[b][w_lc] += sig_v;
                        bpreds_l[b][w_lc].push_back(v_lc);
                    }
                } else {
                    fb[w_own].push_back(b);
                    fv[w_own].push_back(go_src[e]);  /* v_global */
                    fw[w_own].push_back(w_gl);
                    fs[w_own].push_back(sig_v);
                }
            }

            /* ---- 一次 MPI 交换：覆盖所有批次的远程消息 ---- */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)fs[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            vector<int>       s_b(ts), r_b(tr);
            vector<int>       s_v(ts), r_v(tr);
            vector<int>       s_w(ts), r_w(tr);
            vector<long long> s_s(ts), r_s(tr);
            for (int p = 0, pos = 0; p < nproc; p++)
                for (int i = 0; i < scnt[p]; i++, pos++) {
                    s_b[pos] = fb[p][i];
                    s_v[pos] = fv[p][i];
                    s_w[pos] = fw[p][i];
                    s_s[pos] = fs[p][i];
                }

            MPI_Alltoallv(s_b.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_b.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
            MPI_Alltoallv(s_v.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_v.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
            MPI_Alltoallv(s_w.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_w.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
            MPI_Alltoallv(s_s.data(), scnt.data(), sdisp.data(), MPI_LONG_LONG,
                          r_s.data(), rcnt.data(), rdisp.data(), MPI_LONG_LONG, MPI_COMM_WORLD);

            /* ---- 处理收到的消息 ---- */
            for (int i = 0; i < tr; i++) {
                int       b     = r_b[i];
                int       v_gl  = r_v[i];
                int       w_gl  = r_w[i];
                long long sig_v = r_s[i];
                int       w_lc  = w_gl - v0;

                if (bd[b][w_lc] == -1) {
                    bd[b][w_lc]     = cur_level + 1;
                    bsigma[b][w_lc] = sig_v;
                    bpreds_r[b][w_lc].push_back(make_pair(v_gl, sig_v));
                    bnext[b].push_back(w_lc);
                } else if (bd[b][w_lc] == cur_level + 1) {
                    bsigma[b][w_lc] += sig_v;
                    bpreds_r[b][w_lc].push_back(make_pair(v_gl, sig_v));
                }
            }

            /* ---- 全局终止检测（所有批次的 next_front 均为空才停）---- */
            int local_sz = 0;
            for (int b = 0; b < batch_sz; b++) local_sz += (int)bnext[b].size();
            int global_sz = 0;
            MPI_Allreduce(&local_sz, &global_sz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (global_sz == 0) break;

            for (int b = 0; b < batch_sz; b++) {
                blevels[b].push_back(bnext[b]);
                bcur[b].swap(bnext[b]);
            }
            cur_level++;
        }

        /* ============================================================
         * 反向传播：逆层累积 delta，所有批次共享一次 MPI 交换
         * Brandes 公式：delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
         * ============================================================ */
        for (int b = 0; b < batch_sz; b++)
            fill(bdelta[b].begin(), bdelta[b].end(), 0.0);

        /* 所有进程、所有批次中的最大层数 */
        int max_lev = 0;
        for (int b = 0; b < batch_sz; b++)
            max_lev = max(max_lev, (int)blevels[b].size() - 1);
        int global_max_lev = 0;
        MPI_Allreduce(&max_lev, &global_max_lev, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int lev = global_max_lev; lev >= 1; lev--) {
            for (int p = 0; p < nproc; p++) {
                qb[p].clear(); qv[p].clear(); qc[p].clear();
            }

            for (int b = 0; b < batch_sz; b++) {
                if (lev >= (int)blevels[b].size()) continue;
                for (int wi = 0; wi < (int)blevels[b][lev].size(); wi++) {
                    int w_lc = blevels[b][lev][wi];
                    if (bsigma[b][w_lc] == 0) continue;
                    double coeff = (1.0 + bdelta[b][w_lc]) / (double)bsigma[b][w_lc];

                    /* 本地前驱：直接读最终 sigma */
                    for (int vi = 0; vi < (int)bpreds_l[b][w_lc].size(); vi++) {
                        int v_lc = bpreds_l[b][w_lc][vi];
                        bdelta[b][v_lc] += (double)bsigma[b][v_lc] * coeff;
                    }

                    /* 远程前驱：发送 delta 贡献给 owner */
                    for (int pi = 0; pi < (int)bpreds_r[b][w_lc].size(); pi++) {
                        int       v_gl  = bpreds_r[b][w_lc][pi].first;
                        long long sig_v = bpreds_r[b][w_lc][pi].second;
                        int v_own = VERTEX_OWNER((vertex_id_t)v_gl, G->n, G->nproc);
                        qb[v_own].push_back(b);
                        qv[v_own].push_back(v_gl);
                        qc[v_own].push_back((double)sig_v * coeff);
                    }
                }
            }

            /* 一次 MPI 交换：覆盖所有批次的 delta 贡献 */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)qc[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            vector<int>    s_qb(ts), r_qb(tr);
            vector<int>    s_qv(ts), r_qv(tr);
            vector<double> s_qc(ts), r_qc(tr);
            for (int p = 0, pos = 0; p < nproc; p++)
                for (int i = 0; i < scnt[p]; i++, pos++) {
                    s_qb[pos] = qb[p][i];
                    s_qv[pos] = qv[p][i];
                    s_qc[pos] = qc[p][i];
                }

            MPI_Alltoallv(s_qb.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_qb.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
            MPI_Alltoallv(s_qv.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_qv.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
            MPI_Alltoallv(s_qc.data(), scnt.data(), sdisp.data(), MPI_DOUBLE,
                          r_qc.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            /* 应用收到的 delta 贡献 */
            for (int i = 0; i < tr; i++)
                bdelta[r_qb[i]][r_qv[i] - v0] += r_qc[i];
        }

        /* ---- 累积 BC（跳过各自的源节点）---- */
        for (int b = 0; b < batch_sz; b++) {
            int s_gl = s_start + b;
            for (int i = 0; i < loc_n; i++)
                if ((v0 + i) != s_gl)
                    result[i] += bdelta[b][i];
        }

    } /* end for s_start */

    bc_gpu_cleanup();

    /* 无向图：每条最短路被双向计数 */
    for (int i = 0; i < loc_n; i++) result[i] /= 2.0;

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);
}
