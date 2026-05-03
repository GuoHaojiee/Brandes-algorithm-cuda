/*
 * solution_mpi.cpp — v3.6: 隐式前驱（按层扁平存储）+ 终止检测合并
 *
 * 相比 v3.5 的改动：
 *   1. 反向 BP 用按层组织的扁平数组 local_preds_per_lev / remote_preds_per_lev
 *      取代 bpreds_l / bpreds_r 这两个 vector<vector<vector<...>>>。
 *      → 元数据从 O(BATCH × n/p × 48B) 降到 O(level_count × 48B)
 *      → 每个 batch 初始化跳过 BATCH × n/p 次 vector::clear()
 *      → 反向 BP 顺序遍历扁平数组，cache 友好
 *   2. 正向 BFS 终止检测的 MPI_Allreduce 合并到 size exchange 的 MPI_Alltoall：
 *      每对进程多发 1 个 int 携带"本地是否还活跃"标志，接收端 OR 一下
 *      即得全局活跃标志。每层省一次集合通信。
 *   3. 反向 BP 预计算 bcoeff[b][w_lc]，避免每条前驱重复做除法。
 *
 * 不变：
 *   - GPU 接口完全没改，brandes_gpu.cu 无需重新编译
 *   - 算法语义完全一致（终止合并最多多 1 层空迭代，开销可忽略）
 *   - 内存仍是 O(BATCH × n/p) = O(n/p)，但常数显著更小
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

/* ---- 跨进程消息结构体 ---- */
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
};   /* 16 字节 */

/* ---- 隐式前驱（按层扁平存储） ---- */
struct LocalPred {
    int b;
    int w_lc;
    int v_lc;
};   /* 12 字节 */

struct RemotePred {
    int       b;
    int       w_lc;
    int       v_gl;
    int       _pad;
    long long sig_v;
};   /* 24 字节 */

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
    vector<vector<double> >    bcoeff(BATCH_SIZE, vector<double>   (loc_n));

    /* blevels 仍保留：反向 BP 第一阶段需遍历每层的 w 算 coeff */
    vector<vector<vector<int> > > blevels(BATCH_SIZE);
    vector<vector<int> > bcur(BATCH_SIZE);
    vector<vector<int> > bnext(BATCH_SIZE);

    /* 隐式前驱：每层一个扁平数组，跨 batch 复用 capacity */
    vector<vector<LocalPred> >  local_preds_per_lev;
    vector<vector<RemotePred> > remote_preds_per_lev;

    /* ---- GPU 输出缓冲 ---- */
    int max_out = BATCH_SIZE * max(loc_m, 1);
    vector<int>       go_b  (max_out), go_src(max_out), go_dst(max_out);
    vector<long long> go_sig(max_out);

    /* ---- GPU 输入缓冲 ---- */
    vector<int>       all_front   (BATCH_SIZE * loc_n);
    vector<int>       front_off   (BATCH_SIZE + 1);
    vector<long long> front_sigma (BATCH_SIZE * loc_n);

    /* ---- MPI 元数据（循环外预分配，每层复用）---- */
    vector<int> scnt (nproc),     rcnt (nproc);
    vector<int> sdisp(nproc + 1), rdisp(nproc + 1);
    vector<int> scnt2(2 * nproc), rcnt2(2 * nproc);   /* 终止合并版的 size+flag */

    vector<vector<FwdMsg> > fwd_buf(nproc);
    vector<vector<BpMsg> >  bp_buf (nproc);
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
            fill(bd[b].begin(),     bd[b].end(),     -1);
            fill(bsigma[b].begin(), bsigma[b].end(), 0LL);
            /* bdelta 在反向 BP 前再清零 */
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

        /* 第 0 层无前驱 */
        if (local_preds_per_lev.empty())  local_preds_per_lev .resize(1);
        else                              local_preds_per_lev [0].clear();
        if (remote_preds_per_lev.empty()) remote_preds_per_lev.resize(1);
        else                              remote_preds_per_lev[0].clear();

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

            /* ---- 准备新层 (cur_level+1) 的前驱容器（复用 capacity）---- */
            int new_lev = cur_level + 1;
            if ((int)local_preds_per_lev.size()  <= new_lev)
                local_preds_per_lev .resize(new_lev + 1);
            if ((int)remote_preds_per_lev.size() <= new_lev)
                remote_preds_per_lev.resize(new_lev + 1);
            vector<LocalPred>&  lp_target = local_preds_per_lev [new_lev];
            vector<RemotePred>& rp_target = remote_preds_per_lev[new_lev];
            lp_target.clear();
            rp_target.clear();

            /* ---- 区分本地/远程；远程打包成 FwdMsg ---- */
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
                    int dw   = bd[b][w_lc];
                    if (dw == -1) {
                        bd[b][w_lc]     = new_lev;
                        bsigma[b][w_lc] = sig_v;
                        LocalPred lp = { b, w_lc, v_lc };
                        lp_target.push_back(lp);
                        bnext[b].push_back(w_lc);
                    } else if (dw == new_lev) {
                        bsigma[b][w_lc] += sig_v;
                        LocalPred lp = { b, w_lc, v_lc };
                        lp_target.push_back(lp);
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

            /* ---- 终止检测合并到 size exchange ----
             *
             * my_flag 等价于"本进程本层有任何工作要扩散到其他进程或本地新染色"。
             * 全局 OR 后若为 0：
             *   - 所有进程 fwd_buf 全空 ⇒ 没人发任何消息
             *   - 所有进程 bnext 全空   ⇒ 本地也没新染色
             *   - 远程消息处理后 bnext 仍为空 ⇒ 下层 frontier 全空 ⇒ 终止
             * 比原方案最多多 1 层空迭代（GPU ne=0、Alltoall(2x)），开销忽略。
             */
            int local_active = 0;
            for (int b = 0; b < batch_sz; b++) {
                if (!bnext[b].empty()) { local_active = 1; break; }
            }
            int total_send = 0;
            for (int p = 0; p < nproc; p++) {
                scnt[p]    = (int)fwd_buf[p].size();
                total_send += scnt[p];
            }
            int my_flag = (local_active != 0) || (total_send != 0);

            for (int p = 0; p < nproc; p++) {
                scnt2[2*p]     = scnt[p];
                scnt2[2*p + 1] = my_flag;
            }
            MPI_Alltoall(scnt2.data(), 2, MPI_INT,
                         rcnt2.data(), 2, MPI_INT, MPI_COMM_WORLD);

            int global_active = my_flag;
            for (int p = 0; p < nproc; p++) {
                rcnt[p]        = rcnt2[2*p];
                global_active |= rcnt2[2*p + 1];
            }
            if (!global_active) break;   /* 全局终止，无需 Alltoallv */

            /* ---- Alltoallv ---- */
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            fwd_send.resize(ts);
            for (int p = 0, pos = 0; p < nproc; p++) {
                int cnt = (int)fwd_buf[p].size();
                if (cnt > 0) {
                    memcpy(&fwd_send[pos], fwd_buf[p].data(),
                           (size_t)cnt * sizeof(FwdMsg));
                    pos += cnt;
                }
            }
            fwd_recv.resize(tr);
            MPI_Alltoallv(
                fwd_send.data(), scnt.data(), sdisp.data(), mpi_fwd_t,
                fwd_recv.data(), rcnt.data(), rdisp.data(), mpi_fwd_t,
                MPI_COMM_WORLD);

            /* ---- 处理收到的远程消息 ---- */
            for (int i = 0; i < tr; i++) {
                const FwdMsg& m = fwd_recv[i];
                int       b     = m.b;
                int       v_gl  = m.v_gl;
                int       w_gl  = m.w_gl;
                long long sig_v = m.sig;
                int       w_lc  = w_gl - v0;
                int       dw    = bd[b][w_lc];

                if (dw == -1) {
                    bd[b][w_lc]     = new_lev;
                    bsigma[b][w_lc] = sig_v;
                    RemotePred rp = { b, w_lc, v_gl, 0, sig_v };
                    rp_target.push_back(rp);
                    bnext[b].push_back(w_lc);
                } else if (dw == new_lev) {
                    bsigma[b][w_lc] += sig_v;
                    RemotePred rp = { b, w_lc, v_gl, 0, sig_v };
                    rp_target.push_back(rp);
                }
            }

            /* ---- 推进到下一层 ---- */
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

            /* 阶段 1: 预计算本层每个 (b, w_lc) 的 coeff
             * bcoeff 不需要清零：阶段 2/3 只读取阶段 1 写过的位置 */
            for (int b = 0; b < batch_sz; b++) {
                if (lev >= (int)blevels[b].size()) continue;
                const vector<int>& Lw = blevels[b][lev];
                for (int wi = 0; wi < (int)Lw.size(); wi++) {
                    int       w_lc = Lw[wi];
                    long long sw   = bsigma[b][w_lc];
                    bcoeff[b][w_lc] = (sw == 0) ? 0.0
                        : (1.0 + bdelta[b][w_lc]) / (double)sw;
                }
            }

            /* 阶段 2: 本地前驱 — 顺序扫扁平数组 */
            if (lev < (int)local_preds_per_lev.size()) {
                const vector<LocalPred>& lps = local_preds_per_lev[lev];
                int sz = (int)lps.size();
                for (int i = 0; i < sz; i++) {
                    const LocalPred& lp = lps[i];
                    bdelta[lp.b][lp.v_lc] +=
                        (double)bsigma[lp.b][lp.v_lc] * bcoeff[lp.b][lp.w_lc];
                }
            }

            /* 阶段 3: 远程前驱 → BpMsg */
            for (int p = 0; p < nproc; p++) bp_buf[p].clear();
            if (lev < (int)remote_preds_per_lev.size()) {
                const vector<RemotePred>& rps = remote_preds_per_lev[lev];
                int sz = (int)rps.size();
                for (int i = 0; i < sz; i++) {
                    const RemotePred& rp = rps[i];
                    int v_own = VERTEX_OWNER((vertex_id_t)rp.v_gl, G->n, G->nproc);
                    BpMsg m;
                    m.b       = rp.b;
                    m.v_gl    = rp.v_gl;
                    m.contrib = (double)rp.sig_v * bcoeff[rp.b][rp.w_lc];
                    bp_buf[v_own].push_back(m);
                }
            }

            /* 阶段 4: size exchange */
            for (int p = 0; p < nproc; p++) scnt[p] = (int)bp_buf[p].size();
            MPI_Alltoall(scnt.data(), 1, MPI_INT,
                         rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
            sdisp[0] = rdisp[0] = 0;
            for (int p = 0; p < nproc; p++) {
                sdisp[p+1] = sdisp[p] + scnt[p];
                rdisp[p+1] = rdisp[p] + rcnt[p];
            }
            int ts = sdisp[nproc], tr = rdisp[nproc];

            /* 阶段 5: 打包 + Alltoallv */
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

            /* 阶段 6: 应用 */
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