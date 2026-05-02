/*
 * solution_mpi.cpp — 分布式 Brandes BC（Direction B: GPU-resident state）
 *
 * 每个进程只持有 O(n/p) 的本地图分区。
 * BFS 状态 d[], sigma[], delta[] 全部驻留在 GPU 显存中。
 * GPU 使用 atomicCAS/atomicAdd 处理所有本地顶点更新。
 * CPU 只负责 MPI 跨进程通信（打包/解包消息）。
 *
 * 批量处理：每次同时处理 BATCH_SIZE 个源节点，摊销 MPI 调用开销。
 */

#include "defs.h"
#include <vector>
#include <cstdio>
#include <algorithm>
using namespace std;

static const int BATCH_SIZE = 64;

/* GPU 函数（brandes_gpu.cu）*/
extern "C" void bc_gpu_init(int local_n, int local_m, int batch_size,
                             const int* offset, const int* dest,
                             int v0_global, int gpu_device);
extern "C" void bc_gpu_reset_batch(int batch_sz, const int* src_lc_h);
extern "C" int  bc_gpu_bfs_expand(const int* af_h, const int* foff_h,
                                   int batch_sz, int total_fz, int cur_lev,
                                   int* out_b, int* out_wgl, long long* out_sig);
extern "C" void bc_gpu_apply_remote_fwd(const int* b_h, const int* wlc_h,
                                         const long long* sig_h,
                                         int count, int disc_lev);
extern "C" int  bc_gpu_collect_next(int batch_sz, int target_lev,
                                     int* h_nv, int* h_ncnt);
extern "C" int  bc_gpu_backprop_level(int batch_sz, int lev,
                                       int* out_b, int* out_ugl, double* out_coeff);
extern "C" void bc_gpu_apply_remote_bp(const int* b_h, const int* ulc_h,
                                        const double* coeff_h,
                                        int count, int lev);
extern "C" void bc_gpu_get_delta(int batch_sz, double* h_delta);
extern "C" void bc_gpu_cleanup(void);

/* ---- 辅助：执行一次 MPI_Alltoallv（3 数组版本）---- */
static void alltoallv3(
    const vector<int>& scnt, const vector<int>& sdisp,
    const vector<int>& rcnt, const vector<int>& rdisp,
    vector<int>& sb, vector<int>& rb,
    vector<int>& sv, vector<int>& rv,
    vector<long long>& ss, vector<long long>& rs)
{
    MPI_Alltoallv(sb.data(), scnt.data(), sdisp.data(), MPI_INT,
                  rb.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoallv(sv.data(), scnt.data(), sdisp.data(), MPI_INT,
                  rv.data(), rcnt.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoallv(ss.data(), scnt.data(), sdisp.data(), MPI_LONG_LONG,
                  rs.data(), rcnt.data(), rdisp.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
}

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int loc_n = (int)G->local_n;
    const int loc_m = (int)G->local_m;
    const int v0    = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    /* ---- 上传本地 CSR 到 GPU ---- */
    {
        vector<int> loff(loc_n + 1), ldst(max(loc_m, 1));
        for (int i = 0; i <= loc_n; i++) loff[i] = (int)G->rowsIndices[i];
        for (int i = 0; i < loc_m;  i++) ldst[i] = (int)G->endV[i];
        bc_gpu_init(loc_n, loc_m, BATCH_SIZE,
                    loff.data(), ldst.data(), v0, rank % 2);
    }

    /* ---- 预分配 CPU 缓冲区 ---- */
    int max_out = BATCH_SIZE * max(loc_m, 1);

    /* frontier 输入（本地，逐层复用）*/
    vector<int>  all_front(BATCH_SIZE * loc_n);
    vector<int>  front_off(BATCH_SIZE + 1);

    /* GPU expand 输出（远程消息）*/
    vector<int>       go_b  (max_out), go_wgl(max_out);
    vector<long long> go_sig(max_out);

    /* collect_next 中间缓冲 */
    vector<int> h_nv  (BATCH_SIZE * loc_n);
    vector<int> h_ncnt(BATCH_SIZE);

    /* GPU backprop 输出（远程贡献）*/
    vector<int>    bp_b  (max_out), bp_ugl(max_out);
    vector<double> bp_coeff(max_out);

    /* delta 下载缓冲 */
    vector<double> h_delta(BATCH_SIZE * loc_n);

    /* ---- MPI 通信元数据（逐级复用）---- */
    vector<int> scnt(nproc), rcnt(nproc), sdisp(nproc + 1), rdisp(nproc + 1);

    /* 正向 BFS 消息暂存：batch_id / w_global / sigma_v */
    vector<vector<int> >       fwd_b(nproc), fwd_w(nproc);
    vector<vector<long long> > fwd_s(nproc);

    /* 反向传播消息暂存：batch_id / u_global / coeff */
    vector<vector<int> >    bpq_b(nproc), bpq_u(nproc);
    vector<vector<double> > bpq_c(nproc);

    for (int i = 0; i < loc_n; i++) result[i] = 0.0;
    double t0 = MPI_Wtime();

    /* ================================================================
     * 主循环：每次处理 BATCH_SIZE 个源节点
     * ================================================================ */
    for (int s_start = 0; s_start < n; s_start += BATCH_SIZE) {
        int batch_sz = min(BATCH_SIZE, n - s_start);

        /* ---- 重置 GPU 状态，设置种子 ---- */
        {
            vector<int> src_lc(batch_sz);
            for (int b = 0; b < batch_sz; b++) {
                int s_gl = s_start + b;
                if (VERTEX_OWNER((vertex_id_t)s_gl, G->n, G->nproc) == rank)
                    src_lc[b] = s_gl - v0;
                else
                    src_lc[b] = -1;
            }
            bc_gpu_reset_batch(batch_sz, src_lc.data());
        }

        int cur_level = 0;

        /* ---- 预取 level-0 frontier（种子）---- */
        bc_gpu_collect_next(batch_sz, 0, h_nv.data(), h_ncnt.data());

        /* ============================================================
         * 正向 BFS
         * h_nv / h_ncnt 在每次迭代开始时已包含 cur_level 层的顶点
         * （由上一迭代末尾的 collect_next 填充，或由上方预取填充）
         * ============================================================ */
        while (true) {

            /* ---- 从缓存的 h_nv/h_ncnt 构建 all_front ---- */
            front_off[0] = 0;
            int fi = 0;
            for (int b = 0; b < batch_sz; b++) {
                for (int i = 0; i < h_ncnt[b]; i++)
                    all_front[fi++] = h_nv[b * loc_n + i];
                front_off[b + 1] = fi;
            }
            int total_fz = fi;

            /* ---- GPU：扩展 frontier，返回远程消息 ---- */
            int ne = bc_gpu_bfs_expand(
                all_front.data(), front_off.data(),
                batch_sz, total_fz, cur_level,
                go_b.data(), go_wgl.data(), go_sig.data());

            /* ---- 路由远程消息 ---- */
            for (int p = 0; p < nproc; p++) {
                fwd_b[p].clear(); fwd_w[p].clear(); fwd_s[p].clear();
            }
            for (int e = 0; e < ne; e++) {
                int p = VERTEX_OWNER((vertex_id_t)go_wgl[e], G->n, G->nproc);
                fwd_b[p].push_back(go_b[e]);
                fwd_w[p].push_back(go_wgl[e]);
                fwd_s[p].push_back(go_sig[e]);
            }

            /* ---- 打包发送缓冲 ---- */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)fwd_s[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p + 1] = sdisp[p] + scnt[p];
                rdisp[p + 1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            vector<int>       s_b(ts), r_b(tr);
            vector<int>       s_w(ts), r_w(tr);
            vector<long long> s_s(ts), r_s(tr);
            for (int p = 0, pos = 0; p < nproc; p++)
                for (int i = 0; i < scnt[p]; i++, pos++) {
                    s_b[pos] = fwd_b[p][i];
                    s_w[pos] = fwd_w[p][i];
                    s_s[pos] = fwd_s[p][i];
                }

            alltoallv3(scnt, sdisp, rcnt, rdisp,
                       s_b, r_b, s_w, r_w, s_s, r_s);

            /* ---- 应用收到的消息：转换 w_global → w_local ---- */
            if (tr > 0) {
                vector<int>       recv_b  (tr), recv_wlc(tr);
                vector<long long> recv_sig(tr);
                for (int i = 0; i < tr; i++) {
                    recv_b  [i] = r_b[i];
                    recv_wlc[i] = r_w[i] - v0;
                    recv_sig[i] = r_s[i];
                }
                bc_gpu_apply_remote_fwd(
                    recv_b.data(), recv_wlc.data(), recv_sig.data(),
                    tr, cur_level + 1);
            }

            /* ---- 收集下一层并检测全局终止 ---- */
            /* h_nv/h_ncnt 被更新为 cur_level+1 层，下轮直接使用 */
            int local_next = bc_gpu_collect_next(batch_sz, cur_level + 1,
                                                  h_nv.data(), h_ncnt.data());
            int global_next = 0;
            MPI_Allreduce(&local_next, &global_next, 1, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD);
            if (global_next == 0) break;

            cur_level++;
        } /* end forward BFS */

        /* ============================================================
         * 反向传播
         * ============================================================ */
        int global_max_lev = 0;
        MPI_Allreduce(&cur_level, &global_max_lev, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);

        for (int lev = global_max_lev; lev >= 1; lev--) {

            /* ---- GPU：计算本层本地 delta，输出远程贡献请求 ---- */
            int ne_bp = bc_gpu_backprop_level(
                batch_sz, lev,
                bp_b.data(), bp_ugl.data(), bp_coeff.data());

            /* ---- 路由远程贡献 ---- */
            for (int p = 0; p < nproc; p++) {
                bpq_b[p].clear(); bpq_u[p].clear(); bpq_c[p].clear();
            }
            for (int e = 0; e < ne_bp; e++) {
                int p = VERTEX_OWNER((vertex_id_t)bp_ugl[e], G->n, G->nproc);
                bpq_b[p].push_back(bp_b[e]);
                bpq_u[p].push_back(bp_ugl[e]);
                bpq_c[p].push_back(bp_coeff[e]);
            }

            for (int p = 0; p < nproc; p++) scnt[p] = (int)bpq_c[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p + 1] = sdisp[p] + scnt[p];
                rdisp[p + 1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            vector<int>    s_qb(ts), r_qb(tr);
            vector<int>    s_qu(ts), r_qu(tr);
            vector<double> s_qc(ts), r_qc(tr);
            for (int p = 0, pos = 0; p < nproc; p++)
                for (int i = 0; i < scnt[p]; i++, pos++) {
                    s_qb[pos] = bpq_b[p][i];
                    s_qu[pos] = bpq_u[p][i];
                    s_qc[pos] = bpq_c[p][i];
                }

            MPI_Alltoallv(s_qb.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_qb.data(), rcnt.data(), rdisp.data(), MPI_INT,
                          MPI_COMM_WORLD);
            MPI_Alltoallv(s_qu.data(), scnt.data(), sdisp.data(), MPI_INT,
                          r_qu.data(), rcnt.data(), rdisp.data(), MPI_INT,
                          MPI_COMM_WORLD);
            MPI_Alltoallv(s_qc.data(), scnt.data(), sdisp.data(), MPI_DOUBLE,
                          r_qc.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE,
                          MPI_COMM_WORLD);

            /* ---- 应用收到的 delta 贡献 ---- */
            if (tr > 0) {
                vector<int>    recv_b  (tr), recv_ulc(tr);
                vector<double> recv_c  (tr);
                for (int i = 0; i < tr; i++) {
                    recv_b  [i] = r_qb[i];
                    recv_ulc[i] = r_qu[i] - v0;
                    recv_c  [i] = r_qc[i];
                }
                bc_gpu_apply_remote_bp(
                    recv_b.data(), recv_ulc.data(), recv_c.data(),
                    tr, lev);
            }
        } /* end back_prop */

        /* ---- 下载 delta，累积 BC ---- */
        bc_gpu_get_delta(batch_sz, h_delta.data());
        for (int b = 0; b < batch_sz; b++) {
            int s_gl = s_start + b;
            for (int i = 0; i < loc_n; i++)
                if ((v0 + i) != s_gl)
                    result[i] += h_delta[b * loc_n + i];
        }

    } /* end for s_start */

    bc_gpu_cleanup();

    /* 无向图：每条最短路被双向计数 */
    for (int i = 0; i < loc_n; i++) result[i] /= 2.0;

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);
}
