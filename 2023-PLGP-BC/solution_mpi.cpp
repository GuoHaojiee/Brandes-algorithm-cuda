/*
 * solution_mpi.cpp — 分布式 Brandes 介数中心性（GPU 辅助版）
 *
 * 内存模型：每个进程只持有本地图分片，所有数组大小 O(n/p)。
 *
 * 流程：
 *   1. 把本地 CSR 上传到 GPU（只上传一次）
 *   2. 对每个源节点 s（所有进程同步参与）：
 *      a. 正向 BFS：逐层推进
 *         - GPU 并行扩展本地 frontier 的出边
 *         - CPU 区分本地邻居（直接更新）和远程邻居（MPI_Alltoallv 发送）
 *         - 接收其他进程发来的消息，更新本地 d/sigma
 *      b. 反向传播：逆层累积 delta
 *         - 本地前驱：直接累加
 *         - 远程前驱：MPI 发送 delta 贡献
 *   3. 汇总 result，除以 2（无向图）
 */

#include "defs.h"
#include <vector>
#include <cstdio>
#include <cstring>
using namespace std;

/* GPU 函数声明（实现在 brandes_gpu.cu）*/
extern "C" void bc_gpu_init(int local_n, int local_m,
                             const int* offset, const int* dest, int gpu_device);
extern "C" int  bc_gpu_expand(const int* frontier, const long long* front_sigma,
                               int fz, int v0_global,
                               int* out_src, int* out_dst, long long* out_sig);
extern "C" void bc_gpu_cleanup(void);

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int loc_n = (int)G->local_n;
    const int loc_m = (int)G->local_m;

    /*
     * 本进程第一个本地顶点的全局 ID。
     * 顶点按进程连续分配，所以 local vertex l 的 global ID = v0 + l。
     */
    const int v0 = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    /* ----------------------------------------------------------------
     * 把本地 CSR 转为 int 类型并上传 GPU（G 内部用 edge_id_t/vertex_id_t）
     * 内存大小：O(n/p + m/p)
     * ---------------------------------------------------------------- */
    {
        vector<int> loff(loc_n + 1), ldst(loc_m > 0 ? loc_m : 1);
        for (int i = 0; i <= loc_n; i++) loff[i] = (int)G->rowsIndices[i];
        for (int i = 0; i < loc_m;  i++) ldst[i] = (int)G->endV[i];
        bc_gpu_init(loc_n, loc_m, loff.data(), ldst.data(), rank % 2);
    }

    /* ----------------------------------------------------------------
     * BFS 状态数组，全部 O(n/p)
     * ---------------------------------------------------------------- */
    vector<int>       d(loc_n);
    vector<long long> sigma(loc_n);   /* 用 long long 避免大图路径数溢出 */
    vector<double>    delta(loc_n);

    /*
     * local_preds[w]  = w 的本地前驱列表（存 local ID）
     * remote_preds[w] = w 的远程前驱列表（存 {v_global, sigma_v}）
     *                   sigma_v 在发送时已是 v 的最终 sigma 值（BFS 层序保证）
     */
    vector<vector<int>>                  local_preds(loc_n);
    vector<vector<pair<int,long long>>>  remote_preds(loc_n);

    /* 记录每层的本地顶点（反向传播时用）*/
    vector<vector<int>> bfs_levels;

    /* GPU 输出缓冲（最大 = 本地所有边，即 O(m/p)）*/
    int safe_m = (loc_m > 0) ? loc_m : 1;
    vector<int>       go_src(safe_m), go_dst(safe_m);
    vector<long long> go_sig(safe_m);

    /* MPI 通信元数据（每层复用）*/
    vector<int> scnt(nproc), rcnt(nproc), sdisp(nproc+1), rdisp(nproc+1);

    /* 正向 BFS 消息暂存（按目标进程分组）*/
    vector<vector<int>>       fwd_src(nproc), fwd_dst(nproc);
    vector<vector<long long>> fwd_sig(nproc);

    /* 反向传播消息暂存（按目标进程分组）*/
    vector<vector<int>>    bwd_vtx(nproc);
    vector<vector<double>> bwd_val(nproc);

    /* 初始化结果 */
    for (int i = 0; i < loc_n; i++) result[i] = 0.0;
    double t0 = MPI_Wtime();

    /* ================================================================
     * 主循环：对所有 n 个源节点执行分布式 BFS
     * 所有进程同步参与每次迭代；
     * 仅源节点的 owner 初始化 cur_front，其他进程在本次 BFS 中
     * 通过接收 MPI 消息参与计算并累积本地顶点的 delta。
     * ================================================================ */
    for (int s_global = 0; s_global < n; s_global++) {

        /* ---- 初始化 BFS 数组 ---- */
        fill(d.begin(), d.end(), -1);
        fill(sigma.begin(), sigma.end(), 0LL);
        for (int i = 0; i < loc_n; i++) {
            local_preds[i].clear();
            remote_preds[i].clear();
        }
        bfs_levels.clear();

        /* 只有源节点的 owner 往 cur_front 里放种子 */
        vector<int> cur_front;
        if (VERTEX_OWNER((vertex_id_t)s_global, G->n, G->nproc) == rank) {
            int s_lc       = s_global - v0;
            d[s_lc]        = 0;
            sigma[s_lc]    = 1LL;
            cur_front.push_back(s_lc);
            bfs_levels.push_back({s_lc});
        } else {
            bfs_levels.push_back({});   /* 占位，保持 bfs_levels 下标一致 */
        }
        int cur_level = 0;

        /* ============================================================
         * 正向 BFS
         * ============================================================ */
        while (true) {
            vector<int> next_front;

            /* ---- GPU 并行扩展本地 frontier 的出边 ---- */
            int ne = 0;
            if (!cur_front.empty()) {
                /* 收集 frontier 顶点的 sigma（只传 fz 个，不传整个数组）*/
                vector<long long> fsig(cur_front.size());
                for (int i = 0; i < (int)cur_front.size(); i++)
                    fsig[i] = sigma[cur_front[i]];

                ne = bc_gpu_expand(
                    cur_front.data(), fsig.data(),
                    (int)cur_front.size(), v0,
                    go_src.data(), go_dst.data(), go_sig.data()
                );
            }

            /* ---- 区分本地邻居和远程邻居 ---- */
            for (int p = 0; p < nproc; p++) {
                fwd_src[p].clear(); fwd_dst[p].clear(); fwd_sig[p].clear();
            }

            for (int e = 0; e < ne; e++) {
                int       w_gl  = go_dst[e];
                int       w_own = VERTEX_OWNER((vertex_id_t)w_gl, G->n, G->nproc);
                long long sig_v = go_sig[e];
                int       v_lc  = go_src[e] - v0;

                if (w_own == rank) {
                    /* 本地顶点：直接更新 d/sigma/preds */
                    int w_lc = w_gl - v0;
                    if (d[w_lc] == -1) {
                        d[w_lc]     = cur_level + 1;
                        sigma[w_lc] = sig_v;
                        local_preds[w_lc].push_back(v_lc);
                        next_front.push_back(w_lc);
                    } else if (d[w_lc] == cur_level + 1) {
                        sigma[w_lc] += sig_v;
                        local_preds[w_lc].push_back(v_lc);
                    }
                } else {
                    /* 远程顶点：放入对应进程的发送队列 */
                    fwd_src[w_own].push_back(go_src[e]);  /* v_global */
                    fwd_dst[w_own].push_back(w_gl);       /* w_global */
                    fwd_sig[w_own].push_back(sig_v);
                }
            }

            /* ---- MPI_Alltoallv：交换边界顶点消息 ---- */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)fwd_sig[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            vector<int>       s_src(ts), r_src(tr);
            vector<int>       s_dst(ts), r_dst(tr);
            vector<long long> s_sig(ts), r_sig(tr);

            /* 打包发送缓冲 */
            for (int p = 0, pos = 0; p < nproc; p++)
                for (int i = 0; i < scnt[p]; i++, pos++) {
                    s_src[pos] = fwd_src[p][i];
                    s_dst[pos] = fwd_dst[p][i];
                    s_sig[pos] = fwd_sig[p][i];
                }

            MPI_Alltoallv(s_src.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_src.data(), rcnt.data(), rdisp.data(), MPI_INT,
                          MPI_COMM_WORLD);
            MPI_Alltoallv(s_dst.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_dst.data(), rcnt.data(), rdisp.data(), MPI_INT,
                          MPI_COMM_WORLD);
            MPI_Alltoallv(s_sig.data(), scnt.data(), sdisp.data(), MPI_LONG_LONG,
                          r_sig.data(), rcnt.data(), rdisp.data(), MPI_LONG_LONG,
                          MPI_COMM_WORLD);

            /* ---- 处理收到的消息 ---- */
            for (int i = 0; i < tr; i++) {
                int       w_lc  = r_dst[i] - v0;
                int       v_gl  = r_src[i];
                long long sig_v = r_sig[i];

                if (d[w_lc] == -1) {
                    d[w_lc]     = cur_level + 1;
                    sigma[w_lc] = sig_v;
                    remote_preds[w_lc].push_back({v_gl, sig_v});
                    next_front.push_back(w_lc);
                } else if (d[w_lc] == cur_level + 1) {
                    sigma[w_lc] += sig_v;
                    remote_preds[w_lc].push_back({v_gl, sig_v});
                }
            }

            /* ---- 全局终止检测 ---- */
            int lsz = (int)next_front.size(), gsz = 0;
            MPI_Allreduce(&lsz, &gsz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (gsz == 0) break;

            bfs_levels.push_back(next_front);
            cur_front.swap(next_front);
            cur_level++;
        }

        /* ============================================================
         * 反向传播：从最深层向根层逐层累积 delta
         *
         * Brandes 公式：delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
         * ============================================================ */
        fill(delta.begin(), delta.end(), 0.0);
        int max_lev = (int)bfs_levels.size() - 1;

        for (int lev = max_lev; lev >= 1; lev--) {
            for (int p = 0; p < nproc; p++) {
                bwd_vtx[p].clear(); bwd_val[p].clear();
            }

            for (int w_lc : bfs_levels[lev]) {
                if (sigma[w_lc] == 0) continue;
                double coeff = (1.0 + delta[w_lc]) / (double)sigma[w_lc];

                /* 本地前驱：直接读最终 sigma（不用快照）*/
                for (int v_lc : local_preds[w_lc])
                    delta[v_lc] += (double)sigma[v_lc] * coeff;

                /* 远程前驱：把贡献值发给 owner 进程 */
                for (int pi = 0; pi < (int)remote_preds[w_lc].size(); pi++) {
                    int       v_gl  = remote_preds[w_lc][pi].first;
                    long long sig_v = remote_preds[w_lc][pi].second;
                    int v_own = VERTEX_OWNER((vertex_id_t)v_gl, G->n, G->nproc);
                    bwd_vtx[v_own].push_back(v_gl);
                    bwd_val[v_own].push_back((double)sig_v * coeff);
                }
            }

            /* MPI 交换 delta 贡献 */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)bwd_vtx[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            vector<int>    sv(ts), rv(tr);
            vector<double> sd(ts), rd(tr);
            for (int p = 0, pos = 0; p < nproc; p++)
                for (int i = 0; i < scnt[p]; i++, pos++) {
                    sv[pos] = bwd_vtx[p][i];
                    sd[pos] = bwd_val[p][i];
                }

            MPI_Alltoallv(sv.data(), scnt.data(), sdisp.data(), MPI_INT,
                          rv.data(), rcnt.data(), rdisp.data(), MPI_INT,
                          MPI_COMM_WORLD);
            MPI_Alltoallv(sd.data(), scnt.data(), sdisp.data(), MPI_DOUBLE,
                          rd.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE,
                          MPI_COMM_WORLD);

            /* 应用收到的 delta 贡献 */
            for (int i = 0; i < tr; i++)
                delta[rv[i] - v0] += rd[i];
        }

        /* ---- 累积 BC（跳过源节点本身）---- */
        for (int i = 0; i < loc_n; i++)
            if ((v0 + i) != s_global)
                result[i] += delta[i];

    } /* end for s_global */

    bc_gpu_cleanup();

    /* 无向图：每条最短路被双向计数，除以 2 */
    for (int i = 0; i < loc_n; i++) result[i] /= 2.0;

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);
}
