/*
 * solution_mpi.cpp — v3.6: v3.5 + RCM 重排序（环境变量 BC_USE_RCM 切换）
 *
 * 相对 v3.5 的改动（仅 4 处，逻辑零修改）：
 *   1. #include "rcm.h"
 *   2. 入口构造 RcmReorderer，调用 apply()
 *   3. GPU 初始化数据源切到 reorderer.offset()/dest()，loc_n/loc_m 也用 reorderer
 *   4. 算法主循环把结果写到 new_bc（局部 vector），末尾 unpermute_result 还原到 result
 *
 * v3.5 已有的优化保留：
 *   - 合并 Alltoallv（FwdMsg 24B / BpMsg 16B）
 *   - BATCH_SIZE = 128
 *   - send/recv 缓冲循环外预声明
 *
 * 内存：主算法循环期间严格 O(n/p)；apply() 期间峰值 O(n+m)（预处理一次性）
 *
 * 用法：
 *   BC_USE_RCM=1 mpiexec -n 4 ./solution_mpi ...   # 启用 RCM
 *   BC_USE_RCM=0 mpiexec -n 4 ./solution_mpi ...   # 不启用（恒等映射，作对比基线）
 */

#include "defs.h"
#include "rcm.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;

static const int BATCH_SIZE = 128;

/* GPU 函数（brandes_gpu.cu，未改动）*/
extern "C" void bc_gpu_init(int local_n, int local_m, int batch_size,
                             const int* offset, const int* dest, int gpu_device);
extern "C" int  bc_gpu_expand_batch(
    const int* all_front, const int* front_offsets, const long long* front_sigma,
    int batch_sz, int total_fz, int v0_global,
    int* out_b, int* out_src, int* out_dst, long long* out_sig);
extern "C" void bc_gpu_cleanup(void);

/* ---- 合并的消息结构体 ---- */
struct FwdMsg {
    int       b;
    int       v_gl;
    int       w_gl;
    int       _pad;
    long long sig;
};   /* 24 字节 */

struct BpMsg {
    int    b;
    int    v_gl;
    double contrib;
};   /* 16 字节 */

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int v0    = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    /* ============ [新增 #1] RCM 重排序（或恒等）============
     * 重排前后 v0/local_n 不变（分区不变），只有边集变了。
     */
    RcmReorderer R;
    R.apply(G, RcmReorderer::enabled_by_env());

    const int loc_n = R.local_n();
    const int loc_m = R.local_m();
    /* ====================================================== */

    /* 注册 MPI 数据类型 */
    MPI_Datatype mpi_fwd_t, mpi_bp_t;
    MPI_Type_contiguous(sizeof(FwdMsg), MPI_BYTE, &mpi_fwd_t);
    MPI_Type_commit(&mpi_fwd_t);
    MPI_Type_contiguous(sizeof(BpMsg), MPI_BYTE, &mpi_bp_t);
    MPI_Type_commit(&mpi_bp_t);

    /* ============ [改动 #2] 上传重排后的本地 CSR 到 GPU ============ */
    bc_gpu_init(loc_n, loc_m, BATCH_SIZE,
                R.offset(), R.dest(), rank % 2);
    /* ============================================================== */

    /* BFS 状态：BATCH_SIZE 份，每份 O(n/p) */
    vector<vector<int> >       bd    (BATCH_SIZE, vector<int>      (loc_n));
    vector<vector<long long> > bsigma(BATCH_SIZE, vector<long long>(loc_n));
    vector<vector<double> >    bdelta(BATCH_SIZE, vector<double>   (loc_n));

    vector<vector<vector<int> > > bpreds_l(
        BATCH_SIZE, vector<vector<int> >(loc_n));
    vector<vector<vector<pair<int,long long> > > > bpreds_r(
        BATCH_SIZE, vector<vector<pair<int,long long> > >(loc_n));

    vector<vector<vector<int> > > blevels(BATCH_SIZE);
    vector<vector<int> > bcur(BATCH_SIZE);
    vector<vector<int> > bnext(BATCH_SIZE);

    /* GPU 输出缓冲 */
    int max_out = BATCH_SIZE * max(loc_m, 1);
    vector<int>       go_b  (max_out), go_src(max_out), go_dst(max_out);
    vector<long long> go_sig(max_out);

    /* GPU 输入缓冲 */
    vector<int>       all_front   (BATCH_SIZE * loc_n);
    vector<int>       front_off   (BATCH_SIZE + 1);
    vector<long long> front_sigma (BATCH_SIZE * loc_n);

    /* MPI 通信元数据 */
    vector<int> scnt(nproc), rcnt(nproc), sdisp(nproc+1), rdisp(nproc+1);

    /* 远程消息暂存 */
    vector<vector<FwdMsg> > fwd_buf(nproc);
    vector<vector<BpMsg> >  bp_buf(nproc);

    /* 合并 send/recv 缓冲 */
    vector<FwdMsg> fwd_send, fwd_recv;
    vector<BpMsg>  bp_send,  bp_recv;

    /* ============ [改动 #3] 算法在新编号下跑，结果写到 new_bc ============ */
    vector<double> new_bc(loc_n, 0.0);
    /* ===================================================================== */

    double t0 = MPI_Wtime();

    /* ================================================================
     * 主循环：每次 BATCH_SIZE 个源节点
     * ================================================================ */
    for (int s_start = 0; s_start < n; s_start += BATCH_SIZE) {
        int batch_sz = min(BATCH_SIZE, n - s_start);

        /* 初始化本批 BFS 状态 */
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
         * 正向 BFS
         * ============================================================ */
        while (true) {

            /* 合并 frontier */
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

            /* GPU 批量扩展 */
            int ne = bc_gpu_expand_batch(
                all_front.data(), front_off.data(), front_sigma.data(),
                batch_sz, total_fz, v0,
                go_b.data(), go_src.data(), go_dst.data(), go_sig.data()
            );

            /* 区分本地/远程；远程打包成 FwdMsg */
            for (int p = 0; p < nproc; p++) fwd_buf[p].clear();
            for (int b = 0; b < batch_sz; b++) bnext[b].clear();

            for (int e = 0; e < ne; e++) {
                int       b     = go_b  [e];
                int       w_gl  = go_dst[e];
                int       w_own = VERTEX_OWNER((vertex_id_t)w_gl, G->n, G->nproc);
                long long sig_v = go_sig[e];
                int       v_gl  = go_src[e];

                if (w_own == rank) {
                    int w_lc = w_gl - v0;
                    int v_lc = v_gl - v0;
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
                    FwdMsg m;
                    m.b    = b;
                    m.v_gl = v_gl;
                    m.w_gl = w_gl;
                    m._pad = 0;
                    m.sig  = sig_v;
                    fwd_buf[w_own].push_back(m);
                }
            }

            /* size exchange */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)fwd_buf[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            /* 打包 send 缓冲 */
            fwd_send.resize(ts);
            for (int p = 0, pos = 0; p < nproc; p++) {
                int cnt = (int)fwd_buf[p].size();
                if (cnt > 0) {
                    memcpy(&fwd_send[pos], fwd_buf[p].data(),
                           (size_t)cnt * sizeof(FwdMsg));
                    pos += cnt;
                }
            }

            /* 一次合并的 Alltoallv */
            fwd_recv.resize(tr);
            MPI_Alltoallv(
                fwd_send.data(), scnt.data(), sdisp.data(), mpi_fwd_t,
                fwd_recv.data(), rcnt.data(), rdisp.data(), mpi_fwd_t,
                MPI_COMM_WORLD);

            /* 处理收到的消息 */
            for (int i = 0; i < tr; i++) {
                const FwdMsg& m = fwd_recv[i];
                int       b     = m.b;
                int       v_gl  = m.v_gl;
                int       w_gl  = m.w_gl;
                long long sig_v = m.sig;
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

            /* 全局终止检测 */
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
         * 反向传播
         * ============================================================ */
        for (int b = 0; b < batch_sz; b++)
            fill(bdelta[b].begin(), bdelta[b].end(), 0.0);

        int max_lev = 0;
        for (int b = 0; b < batch_sz; b++)
            max_lev = max(max_lev, (int)blevels[b].size() - 1);
        int global_max_lev = 0;
        MPI_Allreduce(&max_lev, &global_max_lev, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int lev = global_max_lev; lev >= 1; lev--) {
            for (int p = 0; p < nproc; p++) bp_buf[p].clear();

            for (int b = 0; b < batch_sz; b++) {
                if (lev >= (int)blevels[b].size()) continue;
                for (int wi = 0; wi < (int)blevels[b][lev].size(); wi++) {
                    int w_lc = blevels[b][lev][wi];
                    if (bsigma[b][w_lc] == 0) continue;
                    double coeff = (1.0 + bdelta[b][w_lc]) / (double)bsigma[b][w_lc];

                    for (int vi = 0; vi < (int)bpreds_l[b][w_lc].size(); vi++) {
                        int v_lc = bpreds_l[b][w_lc][vi];
                        bdelta[b][v_lc] += (double)bsigma[b][v_lc] * coeff;
                    }

                    for (int pi = 0; pi < (int)bpreds_r[b][w_lc].size(); pi++) {
                        int       v_gl  = bpreds_r[b][w_lc][pi].first;
                        long long sig_v = bpreds_r[b][w_lc][pi].second;
                        int v_own = VERTEX_OWNER((vertex_id_t)v_gl, G->n, G->nproc);
                        BpMsg m;
                        m.b       = b;
                        m.v_gl    = v_gl;
                        m.contrib = (double)sig_v * coeff;
                        bp_buf[v_own].push_back(m);
                    }
                }
            }

            for (int p = 0; p < nproc; p++) scnt[p] = (int)bp_buf[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            bp_send.resize(ts);
            for (int p = 0, pos = 0; p < nproc; p++) {
                int cnt = (int)bp_buf[p].size();
                if (cnt > 0) {
                    memcpy(&bp_send[pos], bp_buf[p].data(),
                           (size_t)cnt * sizeof(BpMsg));
                    pos += cnt;
                }
            }

            bp_recv.resize(tr);
            MPI_Alltoallv(
                bp_send.data(), scnt.data(), sdisp.data(), mpi_bp_t,
                bp_recv.data(), rcnt.data(), rdisp.data(), mpi_bp_t,
                MPI_COMM_WORLD);

            for (int i = 0; i < tr; i++) {
                const BpMsg& m = bp_recv[i];
                bdelta[m.b][m.v_gl - v0] += m.contrib;
            }
        }

        /* ============ [改动 #4a] 累积到 new_bc 而不是 result ============ */
        for (int b = 0; b < batch_sz; b++) {
            int s_gl = s_start + b;
            for (int i = 0; i < loc_n; i++)
                if ((v0 + i) != s_gl)
                    new_bc[i] += bdelta[b][i];
        }
        /* =============================================================== */

    } /* end for s_start */

    bc_gpu_cleanup();

    /* 无向图：每条最短路被双向计数 */
    for (int i = 0; i < loc_n; i++) new_bc[i] /= 2.0;

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);

    /* ============ [改动 #4b] 把基于新编号的 BC 还原到 framework 期望顺序 ============ */
    R.unpermute_result(new_bc, result);
    /* ============================================================================ */

    MPI_Type_free(&mpi_fwd_t);
    MPI_Type_free(&mpi_bp_t);
}
