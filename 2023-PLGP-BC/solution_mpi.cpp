/*
 * solution_mpi.cpp — v3.7: v3.5 + MPI 通信-计算重叠
 *
 * 优化（教师列表第 1 项："Отправка и прием сообщений в разных программных потоках"）
 * ───────────────────────────────────────────────────────────────────────────
 * 利用 MPI_Ialltoallv 非阻塞集合通信实现通信-计算重叠。Spectrum MPI 的
 * 非阻塞集合操作由内部独立 progress thread 推进，等价于"在不同程序线程中
 * 发送/接收消息"。同时主线程在 MPI_Ialltoallv 与 MPI_Wait 之间执行本地
 * 拥有顶点的状态更新，实现真正的通信-计算并行。
 *
 * 重叠窗口：
 *   正向 BFS：issue Ialltoallv → 处理本地边（更新 bd/bsigma/bpreds_l/bnext）
 *             → Wait → 处理远程消息
 *   反向传播：issue Ialltoallv → 处理本地前驱（累加 bdelta） → Wait
 *             → 处理远程贡献
 *
 * 开关（环境变量）：BC_USE_OVERLAP
 *   =1 (default)  — 启用通信-计算重叠
 *   =0            — 阻塞 MPI_Alltoallv + 即时本地处理（v3.5 行为，作 baseline）
 *
 * v3.5 已有的优化保留：
 *   - 合并 Alltoallv（FwdMsg 24B / BpMsg 16B）
 *   - BATCH_SIZE = 128
 *   - send/recv 缓冲循环外预声明
 *
 * 内存：严格 O(n/p)
 *
 * 用法：
 *   BC_USE_OVERLAP=1 mpiexec -n 4 ./solution_mpi ...   # 优化版
 *   BC_USE_OVERLAP=0 mpiexec -n 4 ./solution_mpi ...   # baseline 对比
 */

#include "defs.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
using namespace std;

static const int BATCH_SIZE = 128;

/* GPU 函数（brandes_gpu.cu，未改动） */
extern "C" void bc_gpu_init(int local_n, int local_m, int batch_size,
                             const int* offset, const int* dest, int gpu_device);
extern "C" int  bc_gpu_expand_batch(
    const int* all_front, const int* front_offsets, const long long* front_sigma,
    int batch_sz, int total_fz, int v0_global,
    int* out_b, int* out_src, int* out_dst, long long* out_sig);
extern "C" void bc_gpu_cleanup(void);

/* ---- 合并的消息结构体（v3.5 沿用） ---- */
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

/* ---- 重叠模式下暂存本地工作的结构体 ---- */
struct LocalEdge {           /* 正向 BFS：暂存本地边 */
    int       b;
    int       w_lc;
    int       v_lc;
    int       _pad;
    long long sig;
};   /* 24 字节 */

struct LocalBp {             /* 反向传播：暂存本地前驱贡献 */
    int    b;
    int    v_lc;
    double contrib;
};   /* 16 字节 */

/* 读环境变量 BC_USE_OVERLAP（默认 1） */
static bool overlap_enabled_by_env() {
    const char* e = getenv("BC_USE_OVERLAP");
    return (e == NULL) || (e[0] != '0');
}

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int loc_n = (int)G->local_n;
    const int loc_m = (int)G->local_m;
    const int v0    = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    const bool USE_OVERLAP = overlap_enabled_by_env();
    if (rank == 0) {
        printf("[v3.7] Communication-computation overlap: %s\n",
               USE_OVERLAP ? "ENABLED (MPI_Ialltoallv)"
                           : "DISABLED (blocking baseline)");
    }

    /* 用本地数组累积 BC（与 v3.6 identity 模式等价），最后一次性覆盖写入
     * result。这样无论 framework 传入的 result 起点如何、是否多次调用
     * run() 共享同一 result，都能保证正确。 */

    /* 注册 MPI 数据类型 */
    MPI_Datatype mpi_fwd_t, mpi_bp_t;
    MPI_Type_contiguous(sizeof(FwdMsg), MPI_BYTE, &mpi_fwd_t);
    MPI_Type_commit(&mpi_fwd_t);
    MPI_Type_contiguous(sizeof(BpMsg), MPI_BYTE, &mpi_bp_t);
    MPI_Type_commit(&mpi_bp_t);

    /* 上传本地 CSR 到 GPU（把 framework 的整型数组转 int） */
    {
        vector<int> off_int(loc_n + 1);
        for (int i = 0; i <= loc_n; i++) off_int[i] = (int)G->rowsIndices[i];
        vector<int> dst_int(loc_m > 0 ? loc_m : 1);
        for (int i = 0; i < loc_m; i++)  dst_int[i] = (int)G->endV[i];
        bc_gpu_init(loc_n, loc_m, BATCH_SIZE,
                    off_int.data(), dst_int.data(), rank % 2);
    }

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

    /* 重叠模式下：本地工作的暂存 buffer */
    vector<LocalEdge> local_edges;   /* 正向 BFS 用 */
    vector<LocalBp>   local_bps;     /* 反向传播用 */

    /* 本地 BC 累积器（v3.6 identity 风格，最后一次性覆盖 result） */
    vector<double> bc(loc_n, 0.0);

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

            /* ===== 阶段 A：扫描 GPU 输出，分类边 =====
             * USE_OVERLAP=1：本地边只暂存到 local_edges，留待重叠窗口处理
             * USE_OVERLAP=0：本地边立即处理（v3.5 baseline 行为）
             */
            for (int p = 0; p < nproc; p++) fwd_buf[p].clear();
            for (int b = 0; b < batch_sz; b++) bnext[b].clear();
            local_edges.clear();

            for (int e = 0; e < ne; e++) {
                int       b     = go_b  [e];
                int       w_gl  = go_dst[e];
                int       w_own = VERTEX_OWNER((vertex_id_t)w_gl, G->n, G->nproc);
                long long sig_v = go_sig[e];
                int       v_gl  = go_src[e];

                if (w_own == rank) {
                    int w_lc = w_gl - v0;
                    int v_lc = v_gl - v0;
                    if (USE_OVERLAP) {
                        /* 暂存本地边，留到重叠窗口里处理 */
                        LocalEdge le;
                        le.b    = b;
                        le.w_lc = w_lc;
                        le.v_lc = v_lc;
                        le._pad = 0;
                        le.sig  = sig_v;
                        local_edges.push_back(le);
                    } else {
                        /* baseline：立即处理（v3.5 行为） */
                        if (bd[b][w_lc] == -1) {
                            bd[b][w_lc]     = cur_level + 1;
                            bsigma[b][w_lc] = sig_v;
                            bpreds_l[b][w_lc].push_back(v_lc);
                            bnext[b].push_back(w_lc);
                        } else if (bd[b][w_lc] == cur_level + 1) {
                            bsigma[b][w_lc] += sig_v;
                            bpreds_l[b][w_lc].push_back(v_lc);
                        }
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

            /* ===== 阶段 B：size exchange（阻塞，因后续 displacement 计算依赖） ===== */
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

            /* ===== 阶段 C：发起 payload 通信（非阻塞或阻塞） ===== */
            fwd_recv.resize(tr);
            MPI_Request req = MPI_REQUEST_NULL;
            if (USE_OVERLAP) {
                MPI_Ialltoallv(
                    fwd_send.data(), scnt.data(), sdisp.data(), mpi_fwd_t,
                    fwd_recv.data(), rcnt.data(), rdisp.data(), mpi_fwd_t,
                    MPI_COMM_WORLD, &req);
            } else {
                MPI_Alltoallv(
                    fwd_send.data(), scnt.data(), sdisp.data(), mpi_fwd_t,
                    fwd_recv.data(), rcnt.data(), rdisp.data(), mpi_fwd_t,
                    MPI_COMM_WORLD);
            }

            /* ===== 阶段 D：重叠窗口 — 处理本地边（仅 OVERLAP 模式） =====
             * 在 MPI 通信进行的同时执行的纯本地工作。
             * baseline 模式下 local_edges 为空，此循环退化为空操作。
             */
            for (size_t i = 0; i < local_edges.size(); i++) {
                const LocalEdge& le = local_edges[i];
                if (bd[le.b][le.w_lc] == -1) {
                    bd[le.b][le.w_lc]     = cur_level + 1;
                    bsigma[le.b][le.w_lc] = le.sig;
                    bpreds_l[le.b][le.w_lc].push_back(le.v_lc);
                    bnext[le.b].push_back(le.w_lc);
                } else if (bd[le.b][le.w_lc] == cur_level + 1) {
                    bsigma[le.b][le.w_lc] += le.sig;
                    bpreds_l[le.b][le.w_lc].push_back(le.v_lc);
                }
            }

            /* ===== 阶段 E：等通信完成 ===== */
            if (USE_OVERLAP) MPI_Wait(&req, MPI_STATUS_IGNORE);

            /* ===== 阶段 F：处理收到的远程消息 ===== */
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
            /* ===== 阶段 A：扫描层 lev 的顶点，分类前驱贡献 ===== */
            for (int p = 0; p < nproc; p++) bp_buf[p].clear();
            local_bps.clear();

            for (int b = 0; b < batch_sz; b++) {
                if (lev >= (int)blevels[b].size()) continue;
                for (int wi = 0; wi < (int)blevels[b][lev].size(); wi++) {
                    int w_lc = blevels[b][lev][wi];
                    if (bsigma[b][w_lc] == 0) continue;
                    double coeff = (1.0 + bdelta[b][w_lc]) / (double)bsigma[b][w_lc];

                    /* 本地前驱 */
                    for (int vi = 0; vi < (int)bpreds_l[b][w_lc].size(); vi++) {
                        int    v_lc    = bpreds_l[b][w_lc][vi];
                        double contrib = (double)bsigma[b][v_lc] * coeff;
                        if (USE_OVERLAP) {
                            /* 暂存，留到重叠窗口处理 */
                            LocalBp lb;
                            lb.b       = b;
                            lb.v_lc    = v_lc;
                            lb.contrib = contrib;
                            local_bps.push_back(lb);
                        } else {
                            /* baseline：立即累加 */
                            bdelta[b][v_lc] += contrib;
                        }
                    }

                    /* 远程前驱 */
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

            /* ===== 阶段 B：size exchange ===== */
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

            /* ===== 阶段 C：发起 payload 通信 ===== */
            bp_recv.resize(tr);
            MPI_Request req = MPI_REQUEST_NULL;
            if (USE_OVERLAP) {
                MPI_Ialltoallv(
                    bp_send.data(), scnt.data(), sdisp.data(), mpi_bp_t,
                    bp_recv.data(), rcnt.data(), rdisp.data(), mpi_bp_t,
                    MPI_COMM_WORLD, &req);
            } else {
                MPI_Alltoallv(
                    bp_send.data(), scnt.data(), sdisp.data(), mpi_bp_t,
                    bp_recv.data(), rcnt.data(), rdisp.data(), mpi_bp_t,
                    MPI_COMM_WORLD);
            }

            /* ===== 阶段 D：重叠窗口 — 累加本地贡献 ===== */
            for (size_t i = 0; i < local_bps.size(); i++) {
                const LocalBp& lb = local_bps[i];
                bdelta[lb.b][lb.v_lc] += lb.contrib;
            }

            /* ===== 阶段 E：等通信完成 ===== */
            if (USE_OVERLAP) MPI_Wait(&req, MPI_STATUS_IGNORE);

            /* ===== 阶段 F：处理收到的远程贡献 ===== */
            for (int i = 0; i < tr; i++) {
                const BpMsg& m = bp_recv[i];
                bdelta[m.b][m.v_gl - v0] += m.contrib;
            }
        }

        /* 累积到本地 bc */
        for (int b = 0; b < batch_sz; b++) {
            int s_gl = s_start + b;
            for (int i = 0; i < loc_n; i++)
                if ((v0 + i) != s_gl)
                    bc[i] += bdelta[b][i];
        }

    } /* end for s_start */

    bc_gpu_cleanup();

    /* 无向图：每条最短路被双向计数。
     * 注意：覆盖式赋值给 result（不是 +=），与 v3.6 unpermute_result identity
     * 行为一致——这样不依赖 framework 是否清零 result。 */
    for (int i = 0; i < loc_n; i++) result[i] = bc[i] / 2.0;

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);

    MPI_Type_free(&mpi_fwd_t);
    MPI_Type_free(&mpi_bp_t);
}