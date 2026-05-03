/*
 * solution_mpi.cpp — v3.5: 合并 Alltoallv + 增大 BATCH + 缓冲复用
 *
 * 相比 v3 的改动：
 *   1. 正向 BFS：4 次 Alltoallv → 1 次（24 字节 FwdMsg 结构体）
 *   2. 反向 BP： 3 次 Alltoallv → 1 次（16 字节 BpMsg 结构体）
 *   3. BATCH_SIZE 64 → 128（GPU 显存仍在 P100 16GB 内）
 *   4. send/recv 缓冲循环外预声明，自动复用 vector capacity
 *
 * 不变：
 *   - 内存仍是 O(BATCH × n/p) = O(n/p) ✓
 *   - GPU 接口完全没改，brandes_gpu.cu 不需要重新编译
 *   - 算法语义完全一致，正确性不受影响
 */

#include "defs.h"
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

/* ---- 合并的消息结构体 ----
 * 用 MPI_Type_contiguous 包成 MPI 数据类型，
 * 一次 Alltoallv 取代原来的 4 次（forward）/ 3 次（backward）
 */
struct FwdMsg {
    int       b;        /* batch_id */
    int       v_gl;     /* 远程前驱 global ID（反向传播时需要）*/
    int       w_gl;     /* 目标顶点 global ID */
    int       _pad;     /* 8 字节对齐 */
    long long sig;      /* sigma_v */
};   /* 24 字节 */

struct BpMsg {
    int    b;
    int    v_gl;
    double contrib;
};   /* 16 字节，double 自然 8 字节对齐 */

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int loc_n = (int)G->local_n;
    const int loc_m = (int)G->local_m;
    const int v0    = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    /* ---- 注册 MPI 数据类型 ---- */
    MPI_Datatype mpi_fwd_t, mpi_bp_t;
    MPI_Type_contiguous(sizeof(FwdMsg), MPI_BYTE, &mpi_fwd_t);
    MPI_Type_commit(&mpi_fwd_t);
    MPI_Type_contiguous(sizeof(BpMsg), MPI_BYTE, &mpi_bp_t);
    MPI_Type_commit(&mpi_bp_t);

    /* ---- 上传本地 CSR 到 GPU ---- */
    {
        vector<int> loff(loc_n + 1), ldst(loc_m > 0 ? loc_m : 1);
        for (int i = 0; i <= loc_n; i++) loff[i] = (int)G->rowsIndices[i];
        for (int i = 0; i < loc_m;  i++) ldst[i] = (int)G->endV[i];
        bc_gpu_init(loc_n, loc_m, BATCH_SIZE, loff.data(), ldst.data(), rank % 2);
    }

    /* ---- BFS 状态：BATCH_SIZE 份，每份 O(n/p) ---- */
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

    /* ---- GPU 输出缓冲 ---- */
    int max_out = BATCH_SIZE * max(loc_m, 1);
    vector<int>       go_b  (max_out), go_src(max_out), go_dst(max_out);
    vector<long long> go_sig(max_out);

    /* ---- GPU 输入缓冲 ---- */
    vector<int>       all_front   (BATCH_SIZE * loc_n);
    vector<int>       front_off   (BATCH_SIZE + 1);
    vector<long long> front_sigma (BATCH_SIZE * loc_n);

    /* ---- MPI 通信元数据（循环外预分配，每层复用）---- */
    vector<int> scnt(nproc), rcnt(nproc), sdisp(nproc+1), rdisp(nproc+1);

    /* ---- 远程消息暂存（按目标进程分桶）---- */
    vector<vector<FwdMsg> > fwd_buf(nproc);
    vector<vector<BpMsg> >  bp_buf(nproc);

    /* ---- 合并后的 send/recv 缓冲（每层 resize，capacity 自动保留）---- */
    vector<FwdMsg> fwd_send, fwd_recv;
    vector<BpMsg>  bp_send,  bp_recv;

    for (int i = 0; i < loc_n; i++) result[i] = 0.0;
    double t0 = MPI_Wtime();

    /* ================================================================
     * 主循环：每次 BATCH_SIZE 个源节点
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

            /* ---- GPU 批量扩展 ---- */
            int ne = bc_gpu_expand_batch(
                all_front.data(), front_off.data(), front_sigma.data(),
                batch_sz, total_fz, v0,
                go_b.data(), go_src.data(), go_dst.data(), go_sig.data()
            );

            /* ---- 区分本地 / 远程；远程打包成 FwdMsg ---- */
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

            /* ---- size exchange ---- */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)fwd_buf[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            /* ---- 打包 send 缓冲（连续内存）---- */
            fwd_send.resize(ts);
            for (int p = 0, pos = 0; p < nproc; p++) {
                int cnt = (int)fwd_buf[p].size();
                if (cnt > 0) {
                    memcpy(&fwd_send[pos], fwd_buf[p].data(),
                           (size_t)cnt * sizeof(FwdMsg));
                    pos += cnt;
                }
            }

            /* ---- 一次合并的 Alltoallv（取代原来 4 次）---- */
            fwd_recv.resize(tr);
            MPI_Alltoallv(
                fwd_send.data(), scnt.data(), sdisp.data(), mpi_fwd_t,
                fwd_recv.data(), rcnt.data(), rdisp.data(), mpi_fwd_t,
                MPI_COMM_WORLD);

            /* ---- 处理收到的消息 ---- */
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

            /* ---- 全局终止检测 ---- */
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

                    /* 本地前驱 */
                    for (int vi = 0; vi < (int)bpreds_l[b][w_lc].size(); vi++) {
                        int v_lc = bpreds_l[b][w_lc][vi];
                        bdelta[b][v_lc] += (double)bsigma[b][v_lc] * coeff;
                    }

                    /* 远程前驱：打包成 BpMsg */
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

            /* size exchange */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)bp_buf[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            /* 打包 send */
            bp_send.resize(ts);
            for (int p = 0, pos = 0; p < nproc; p++) {
                int cnt = (int)bp_buf[p].size();
                if (cnt > 0) {
                    memcpy(&bp_send[pos], bp_buf[p].data(),
                           (size_t)cnt * sizeof(BpMsg));
                    pos += cnt;
                }
            }

            /* 一次合并的 Alltoallv（取代原来 3 次）*/
            bp_recv.resize(tr);
            MPI_Alltoallv(
                bp_send.data(), scnt.data(), sdisp.data(), mpi_bp_t,
                bp_recv.data(), rcnt.data(), rdisp.data(), mpi_bp_t,
                MPI_COMM_WORLD);

            /* 应用 */
            for (int i = 0; i < tr; i++) {
                const BpMsg& m = bp_recv[i];
                bdelta[m.b][m.v_gl - v0] += m.contrib;
            }
        }

        /* ---- 累积 BC（跳过源节点）---- */
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

    MPI_Type_free(&mpi_fwd_t);
    MPI_Type_free(&mpi_bp_t);

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);
}