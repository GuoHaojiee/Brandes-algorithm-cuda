/*
 * solution_mpi.cpp — v3.5 + 内存测量模式 (mem-instrumented)
 *
 * 在 v3.5 基础上仅添加内存测量代码，算法逻辑完全不变。
 *
 * 添加内容：
 *   1. mem_report_theoretical()  — 启动时打印各核心数据结构的理论占用（基于
 *      BATCH × loc_n × ... 公式），让读者一眼看到哪些是 O(n/p)。
 *   2. mem_report_dynamic()      — 首批 batch 完成后扫描 bpreds_l/r 的实际
 *      capacity，量化 vector 数据 + 元数据的动态内存。
 *   3. mem_report_actual()       — 结束时解析 /proc/self/status 拿到 VmHWM
 *      (peak resident memory) 和 VmRSS，MPI_Gather 到 rank 0 汇总。
 *
 * 这三处打印的目的：用 nproc = 1 / 2 / 3 / 4 跑同一个图，对比 HWM_max 这一列：
 *   - O(n)   实现：HWM_max 几乎不随 p 变化（每节点存全图）
 *   - O(n/p) 实现：HWM_max 随 p 显著下降（接近 1/p）
 *
 * 算法部分未做任何改动；如需关闭内存报告，把对 mem_report_* 的三处调用
 * 注释掉即可。
 */

#include "defs.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

/* ================================================================
 *                      内存测量辅助函数
 * ================================================================ */

/* 解析 /proc/self/status 中某行（如 "VmHWM:"）的 KB 数值
 * VmHWM = peak resident set size, VmRSS = current resident set size */
static size_t mem_parse_proc_status_kb(const char* key)
{
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    int klen = (int)strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, klen) == 0) {
            sscanf(line + klen, "%zu", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}

/* 启动时打印每个核心数据结构的理论占用（仅 rank 0 打印） */
static void mem_report_theoretical(int rank, int nproc, int n, int loc_n,
                                   int loc_m, int batch)
{
    if (rank != 0) return;

    int safe_m = (loc_m > 0) ? loc_m : 1;

    size_t csr_bytes      = ((size_t)loc_n + 1 + safe_m) * sizeof(int);
    size_t bd_bytes       = (size_t)batch * loc_n * sizeof(int);
    size_t bsigma_bytes   = (size_t)batch * loc_n * sizeof(long long);
    size_t bdelta_bytes   = (size_t)batch * loc_n * sizeof(double);
    size_t gpu_in_bytes   = (size_t)batch * loc_n
                            * (sizeof(int) + sizeof(long long));
    size_t gpu_out_bytes  = (size_t)batch * safe_m
                            * (3 * sizeof(int) + sizeof(long long));
    size_t bpreds_meta    = (size_t)batch * loc_n
                            * (sizeof(vector<int>)
                               + sizeof(vector<pair<int,long long> >));
    size_t total_static   = csr_bytes + bd_bytes + bsigma_bytes + bdelta_bytes
                            + gpu_in_bytes + gpu_out_bytes + bpreds_meta;

    const double MB = 1024.0 * 1024.0;

    printf("\n");
    printf("=========== 内存预算 (per process, BATCH=%d) ===========\n", batch);
    printf("  全局图:        n     = %d 顶点,  分布于 %d 个进程\n", n, nproc);
    printf("  本进程 rank 0: loc_n = %d 顶点,  loc_m = %d 边\n", loc_n, loc_m);
    printf("  ------------------------------------------------------\n");
    printf("  核心常驻数据 (这部分是 O(n/p)):\n");
    printf("    本地 CSR    (offset+dest):           %9.2f MB\n", csr_bytes/MB);
    printf("    bd      [BATCH][loc_n] int:          %9.2f MB\n", bd_bytes/MB);
    printf("    bsigma  [BATCH][loc_n] long long:    %9.2f MB\n", bsigma_bytes/MB);
    printf("    bdelta  [BATCH][loc_n] double:       %9.2f MB\n", bdelta_bytes/MB);
    printf("    GPU 输入  (BATCH×loc_n):             %9.2f MB\n", gpu_in_bytes/MB);
    printf("    GPU 输出  (BATCH×loc_m):             %9.2f MB\n", gpu_out_bytes/MB);
    printf("    bpreds_l/r 元数据 (BATCH×loc_n vec): %9.2f MB\n", bpreds_meta/MB);
    printf("  ------------------------------------------------------\n");
    printf("  合计 (核心常驻):                       %9.2f MB\n", total_static/MB);
    printf("  注: bpreds_l/r 实际数据 + MPI 通信缓冲为动态，与图结构相关\n");
    printf("======================================================\n");
    fflush(stdout);
}

/* 首批 batch 完成后扫描前驱表实际占用（数据 + 元数据） */
static void mem_report_dynamic(int rank, int nproc, int loc_n, int batch,
    const vector<vector<vector<int> > >& bpreds_l,
    const vector<vector<vector<pair<int,long long> > > >& bpreds_r)
{
    unsigned long my_lp_data = 0, my_rp_data = 0;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < loc_n; i++) {
            my_lp_data += (unsigned long)bpreds_l[b][i].capacity()
                          * (unsigned long)sizeof(int);
            my_rp_data += (unsigned long)bpreds_r[b][i].capacity()
                          * (unsigned long)sizeof(pair<int,long long>);
        }
    }
    unsigned long my_lp_meta = (unsigned long)batch * (unsigned long)loc_n
                               * (unsigned long)sizeof(vector<int>);
    unsigned long my_rp_meta = (unsigned long)batch * (unsigned long)loc_n
                               * (unsigned long)sizeof(vector<pair<int,long long> >);

    unsigned long mine[4] = { my_lp_data, my_lp_meta, my_rp_data, my_rp_meta };
    unsigned long agg_max[4] = {0, 0, 0, 0};
    unsigned long agg_sum[4] = {0, 0, 0, 0};
    MPI_Reduce(mine, agg_max, 4, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(mine, agg_sum, 4, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double MB = 1024.0 * 1024.0;
        printf("\n");
        printf("=========== 前驱表动态内存 (首批 batch 完成后) ===========\n");
        printf("  bpreds_l 数据      (本地前驱):  最大 %7.2f MB,  全局 %7.2f MB\n",
               agg_max[0]/MB, agg_sum[0]/MB);
        printf("  bpreds_l 元数据    (vec 头部):  最大 %7.2f MB,  全局 %7.2f MB\n",
               agg_max[1]/MB, agg_sum[1]/MB);
        printf("  bpreds_r 数据      (远程前驱):  最大 %7.2f MB,  全局 %7.2f MB\n",
               agg_max[2]/MB, agg_sum[2]/MB);
        printf("  bpreds_r 元数据    (vec 头部):  最大 %7.2f MB,  全局 %7.2f MB\n",
               agg_max[3]/MB, agg_sum[3]/MB);
        printf("==========================================================\n");
        fflush(stdout);
    }
}

/* 程序结束时收集所有进程的 VmHWM/VmRSS，rank 0 汇总打印 */
static void mem_report_actual(const char* tag, int rank, int nproc)
{
    unsigned long my_hwm = (unsigned long)mem_parse_proc_status_kb("VmHWM:");
    unsigned long my_rss = (unsigned long)mem_parse_proc_status_kb("VmRSS:");

    vector<unsigned long> all_hwm, all_rss;
    if (rank == 0) {
        all_hwm.resize(nproc);
        all_rss.resize(nproc);
    }
    MPI_Gather(&my_hwm, 1, MPI_UNSIGNED_LONG,
               (rank == 0) ? all_hwm.data() : NULL,
               1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_rss, 1, MPI_UNSIGNED_LONG,
               (rank == 0) ? all_rss.data() : NULL,
               1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        unsigned long hmax = 0, hsum = 0;
        unsigned long hmin = (unsigned long)-1;
        for (int p = 0; p < nproc; p++) {
            hsum += all_hwm[p];
            if (all_hwm[p] > hmax) hmax = all_hwm[p];
            if (all_hwm[p] < hmin) hmin = all_hwm[p];
        }
        printf("\n");
        printf("=========== 实测峰值物理内存 [%s] ===========\n", tag);
        printf("  nproc = %d   (VmHWM = peak resident, VmRSS = current resident)\n",
               nproc);
        printf("  ------------------------------------------------------\n");
        for (int p = 0; p < nproc; p++) {
            printf("  rank %2d:  VmHWM = %9.2f MB    VmRSS = %9.2f MB\n",
                   p, all_hwm[p] / 1024.0, all_rss[p] / 1024.0);
        }
        printf("  ------------------------------------------------------\n");
        printf("  HWM 最大 :  %9.2f MB     <--- 单节点显存压力上界\n",
               hmax / 1024.0);
        printf("  HWM 最小 :  %9.2f MB\n", hmin / 1024.0);
        printf("  HWM 总和 :  %9.2f MB     <--- 集群总内存使用\n",
               hsum / 1024.0);
        printf("  HWM 平均 :  %9.2f MB\n", (hsum / (double)nproc) / 1024.0);
        if (hmax > 0) {
            printf("  负载均衡 (min/max): %.1f %%   (100%% = 完美均衡)\n",
                   (double)hmin / (double)hmax * 100.0);
        }
        printf("  ------ 扩展性判读 ------\n");
        printf("  若实现是 O(n/p): 加大 p 时 HWM_max 应显著下降（接近 1/p）\n");
        printf("  若实现是 O(n)  : 加大 p 时 HWM_max 几乎不变（每节点存全图）\n");
        printf("============================================================\n");
        fflush(stdout);
    }
}

/* ================================================================
 *                          主算法
 * ================================================================ */

void run(graph_t *G, double *result)
{
    const int rank  = G->rank;
    const int nproc = G->nproc;
    const int n     = (int)G->n;
    const int loc_n = (int)G->local_n;
    const int loc_m = (int)G->local_m;
    const int v0    = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank);

    /* === [内存] 启动时打印理论预算 === */
    mem_report_theoretical(rank, nproc, n, loc_n, loc_m, BATCH_SIZE);

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

    /* ---- MPI 通信元数据 ---- */
    vector<int> scnt(nproc), rcnt(nproc), sdisp(nproc+1), rdisp(nproc+1);

    vector<vector<FwdMsg> > fwd_buf(nproc);
    vector<vector<BpMsg> >  bp_buf(nproc);

    vector<FwdMsg> fwd_send, fwd_recv;
    vector<BpMsg>  bp_send,  bp_recv;

    for (int i = 0; i < loc_n; i++) result[i] = 0.0;
    double t0 = MPI_Wtime();

    int first_batch_reported = 0;   /* 首批后报告动态内存的标志 */

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

            /* ---- 打包 send 缓冲 ---- */
            fwd_send.resize(ts);
            for (int p = 0, pos = 0; p < nproc; p++) {
                int cnt = (int)fwd_buf[p].size();
                if (cnt > 0) {
                    memcpy(&fwd_send[pos], fwd_buf[p].data(),
                           (size_t)cnt * sizeof(FwdMsg));
                    pos += cnt;
                }
            }

            /* ---- 一次合并的 Alltoallv ---- */
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
            MPI_Allreduce(&local_sz, &global_sz, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD);
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
        MPI_Allreduce(&max_lev, &global_max_lev, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);

        for (int lev = global_max_lev; lev >= 1; lev--) {
            for (int p = 0; p < nproc; p++) bp_buf[p].clear();

            for (int b = 0; b < batch_sz; b++) {
                if (lev >= (int)blevels[b].size()) continue;
                for (int wi = 0; wi < (int)blevels[b][lev].size(); wi++) {
                    int w_lc = blevels[b][lev][wi];
                    if (bsigma[b][w_lc] == 0) continue;
                    double coeff = (1.0 + bdelta[b][w_lc])
                                   / (double)bsigma[b][w_lc];

                    /* 本地前驱 */
                    for (int vi = 0; vi < (int)bpreds_l[b][w_lc].size(); vi++) {
                        int v_lc = bpreds_l[b][w_lc][vi];
                        bdelta[b][v_lc] += (double)bsigma[b][v_lc] * coeff;
                    }

                    /* 远程前驱：打包成 BpMsg */
                    for (int pi = 0; pi < (int)bpreds_r[b][w_lc].size(); pi++) {
                        int       v_gl  = bpreds_r[b][w_lc][pi].first;
                        long long sig_v = bpreds_r[b][w_lc][pi].second;
                        int v_own = VERTEX_OWNER((vertex_id_t)v_gl,
                                                 G->n, G->nproc);
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

            /* 一次合并的 Alltoallv */
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

        /* === [内存] 首批 batch 完成后报告动态前驱内存 === */
        if (!first_batch_reported) {
            mem_report_dynamic(rank, nproc, loc_n, batch_sz,
                               bpreds_l, bpreds_r);
            first_batch_reported = 1;
        }

    } /* end for s_start */

    bc_gpu_cleanup();

    /* 无向图：每条最短路被双向计数 */
    for (int i = 0; i < loc_n; i++) result[i] /= 2.0;

    MPI_Type_free(&mpi_fwd_t);
    MPI_Type_free(&mpi_bp_t);

    if (rank == 0)
        printf("[Total] 计算时间: %.4f 秒\n", MPI_Wtime() - t0);

    /* === [内存] 程序结束前打印实测峰值 === */
    mem_report_actual("End of run()", rank, nproc);
}