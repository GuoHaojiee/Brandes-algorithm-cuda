/*
 * rcm.cpp — Reverse Cuthill-McKee 实现
 *
 * 步骤：
 *   1. Allgatherv 收集全局 CSR（每进程都拿到一份）
 *   2. rank 0 跑 CM-BFS，反转得到 order[new_id] = old_id
 *   3. Bcast order 到所有进程
 *   4. 各进程根据 order/perm 重建本地 CSR，并保留 inv_perm_local_
 *   5. 释放全局表（vector 析构）
 */

#include "rcm.h"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <algorithm>
#include <climits>
#include <utility>

using namespace std;

/* ================================================================
 * 环境变量开关
 * ================================================================ */
bool RcmReorderer::enabled_by_env() {
    const char* e = getenv("BC_USE_RCM");
    return (e != NULL && e[0] == '1');
}

/* ================================================================
 * rank 0 上的 RCM 计算
 * 输入：全局 CSR (off [n+1], dest [m])
 * 输出：order[new_id] = old_id（即 inv_perm，新→旧）
 *
 * TODO: 起始点用伪外周（pseudo-peripheral）会更优；
 *       这里用最简单的"未访问中度数最小者"，对 R-MAT 通常已足够好。
 * ================================================================ */
static vector<int> compute_rcm_on_rank0(int n,
                                         const vector<int>& off,
                                         const vector<int>& dest)
{
    vector<int> degree(n);
    for (int i = 0; i < n; i++) degree[i] = off[i+1] - off[i];

    vector<char> visited(n, 0);
    vector<int>  order;
    order.reserve(n);

    /* 处理多个连通分量：每分量重启一次 BFS */
    while ((int)order.size() < n) {
        int start   = -1;
        int min_deg = INT_MAX;
        for (int v = 0; v < n; v++) {
            if (!visited[v] && degree[v] < min_deg) {
                min_deg = degree[v];
                start   = v;
                if (min_deg == 0) break;   /* 孤立点直接选 */
            }
        }
        if (start < 0) break;

        /* 标准 Cuthill-McKee BFS：邻居按度数升序入队 */
        queue<int> Q;
        Q.push(start);
        visited[start] = 1;
        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            order.push_back(v);

            int deg_v = off[v+1] - off[v];
            vector<pair<int,int> > nbrs;   /* (degree, vertex) */
            nbrs.reserve(deg_v);
            for (int e = off[v]; e < off[v+1]; e++) {
                int u = dest[e];
                if (!visited[u]) nbrs.push_back(make_pair(degree[u], u));
            }
            sort(nbrs.begin(), nbrs.end());
            for (size_t k = 0; k < nbrs.size(); k++) {
                int u = nbrs[k].second;
                if (!visited[u]) {
                    visited[u] = 1;
                    Q.push(u);
                }
            }
        }
    }

    /* RCM = 反转 CM */
    reverse(order.begin(), order.end());
    return order;
}

/* ================================================================
 * apply：核心入口
 * ================================================================ */
void RcmReorderer::apply(const graph_t* G, bool use_rcm)
{
    rank_       = G->rank;
    nproc_      = G->nproc;
    n_          = (int)G->n;
    int local_n = (int)G->local_n;
    int local_m = (int)G->local_m;
    v0_         = (int)VERTEX_TO_GLOBAL(0, G->n, G->nproc, rank_);

    /* ============ 恒等模式：直接复制原 CSR，作对比基线 ============ */
    if (!use_rcm) {
        identity_     = true;
        new_local_n_  = local_n;
        new_local_m_  = local_m;
        new_offset_.assign(local_n + 1, 0);
        for (int i = 0; i <= local_n; i++)
            new_offset_[i] = (int)G->rowsIndices[i];
        new_dest_.assign(local_m > 0 ? local_m : 1, 0);
        for (int i = 0; i < local_m; i++)
            new_dest_[i] = (int)G->endV[i];
        inv_perm_local_.assign(local_n, 0);
        for (int i = 0; i < local_n; i++)
            inv_perm_local_[i] = v0_ + i;
        if (rank_ == 0)
            printf("[RCM] DISABLED (BC_USE_RCM!=1) — using identity ordering\n");
        return;
    }

    identity_ = false;
    double t0 = MPI_Wtime();

    /* ============ Step 1: Allgatherv 收集全局 CSR ============ */

    /* 1a. 各 rank 的 local_n（顺便算 displacement = v0(r)） */
    vector<int> all_local_n(nproc_);
    MPI_Allgather(&local_n, 1, MPI_INT,
                  all_local_n.data(), 1, MPI_INT, MPI_COMM_WORLD);
    vector<int> v_disp(nproc_ + 1, 0);
    for (int r = 0; r < nproc_; r++) v_disp[r+1] = v_disp[r] + all_local_n[r];
    /* sanity: v_disp[nproc_] == n_ */

    /* 1b. 收集全局度数（以本地 deg = rowsIndices[i+1]-rowsIndices[i] 为单位） */
    vector<int> my_deg(local_n);
    for (int i = 0; i < local_n; i++)
        my_deg[i] = (int)(G->rowsIndices[i+1] - G->rowsIndices[i]);

    vector<int> global_deg(n_);
    MPI_Allgatherv(my_deg.data(), local_n, MPI_INT,
                   global_deg.data(),
                   all_local_n.data(), v_disp.data(),
                   MPI_INT, MPI_COMM_WORLD);

    /* 1c. 全局 offset = exclusive prefix sum */
    vector<int> global_off(n_ + 1, 0);
    for (int i = 0; i < n_; i++) global_off[i+1] = global_off[i] + global_deg[i];

    /* 1d. 收集全部边（dest），按 rank 顺序拼接 */
    vector<int> all_local_m(nproc_);
    MPI_Allgather(&local_m, 1, MPI_INT,
                  all_local_m.data(), 1, MPI_INT, MPI_COMM_WORLD);
    vector<int> e_disp(nproc_ + 1, 0);
    for (int r = 0; r < nproc_; r++) e_disp[r+1] = e_disp[r] + all_local_m[r];
    int total_m = e_disp[nproc_];

    vector<int> my_dest(local_m > 0 ? local_m : 1);
    for (int i = 0; i < local_m; i++) my_dest[i] = (int)G->endV[i];

    vector<int> global_dest(total_m > 0 ? total_m : 1);
    MPI_Allgatherv(my_dest.data(), local_m, MPI_INT,
                   global_dest.data(),
                   all_local_m.data(), e_disp.data(),
                   MPI_INT, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    /* ============ Step 2: rank 0 跑 RCM，Bcast 排列 ============ */
    vector<int> order(n_);   /* order[new_id] = old_id */
    if (rank_ == 0) {
        order = compute_rcm_on_rank0(n_, global_off, global_dest);
    }
    MPI_Bcast(order.data(), n_, MPI_INT, 0, MPI_COMM_WORLD);

    /* perm[old_id] = new_id */
    vector<int> perm(n_);
    for (int i = 0; i < n_; i++) perm[order[i]] = i;

    double t2 = MPI_Wtime();

    /* ============ Step 3: 重建本地 CSR（新编号下） ============ *
     *
     * 本进程在新编号下负责 [v0_, v0_+local_n)（分区不变）。
     * 对每个新本地 i：
     *   new_g = v0_ + i
     *   old_g = order[new_g]                      （旧 ID）
     *   邻居 new_nbr = perm[ global_dest[ global_off[old_g] .. ] ]
     */
    new_local_n_ = local_n;
    new_offset_.assign(new_local_n_ + 1, 0);
    for (int i = 0; i < new_local_n_; i++) {
        int new_g = v0_ + i;
        int old_g = order[new_g];
        int deg   = global_off[old_g + 1] - global_off[old_g];
        new_offset_[i+1] = new_offset_[i] + deg;
    }
    new_local_m_ = new_offset_[new_local_n_];
    new_dest_.assign(new_local_m_ > 0 ? new_local_m_ : 1, 0);

    for (int i = 0; i < new_local_n_; i++) {
        int new_g   = v0_ + i;
        int old_g   = order[new_g];
        int dst_pos = new_offset_[i];
        for (int e = global_off[old_g]; e < global_off[old_g + 1]; e++) {
            int old_nbr = global_dest[e];
            new_dest_[dst_pos++] = perm[old_nbr];   /* 转为新 global ID */
        }
    }

    /* ============ Step 4: 仅保留本进程的 inv_perm 段（O(n/p)） ============ */
    inv_perm_local_.assign(new_local_n_, 0);
    for (int i = 0; i < new_local_n_; i++)
        inv_perm_local_[i] = order[v0_ + i];

    /* global_off / global_dest / order / perm 在 apply() 返回时随栈释放 */

    double t3 = MPI_Wtime();
    if (rank_ == 0) {
        printf("[RCM] Allgather=%.3fs  Compute+Bcast=%.3fs  Rebuild=%.3fs  (total=%.3fs)\n",
               t1 - t0, t2 - t1, t3 - t2, t3 - t0);
    }
}

/* ================================================================
 * unpermute_result：把新编号下的 BC 还原到旧编号本地索引
 *
 * new_bc[i] = BC of new global ID (v0_ + i)
 *           = BC of old global ID inv_perm_local_[i]
 *
 * 目标：old_bc[i] = BC of old global ID (v0_ + i)
 *
 * 流程：每条 (old_g, bc_value) 发到 VERTEX_OWNER(old_g) → 按 old_g - v0_ 写入
 * ================================================================ */
void RcmReorderer::unpermute_result(const vector<double>& new_bc,
                                     double* old_bc) const
{
    /* 恒等模式快速路径 */
    if (identity_) {
        for (int i = 0; i < new_local_n_; i++) old_bc[i] = new_bc[i];
        return;
    }

    struct Msg {
        int    old_g;
        int    _pad;     /* 对齐到 8 字节 */
        double bc;
    };   /* sizeof = 16 */

    /* 按目标进程分桶 */
    vector<vector<Msg> > bucket(nproc_);
    for (int i = 0; i < new_local_n_; i++) {
        int old_g = inv_perm_local_[i];
        int owner = (int)VERTEX_OWNER((vertex_id_t)old_g, (vertex_id_t)n_, nproc_);
        Msg m;
        m.old_g = old_g;
        m._pad  = 0;
        m.bc    = new_bc[i];
        bucket[owner].push_back(m);
    }

    /* size exchange */
    vector<int> scnt(nproc_), rcnt(nproc_);
    vector<int> sdisp(nproc_ + 1, 0), rdisp(nproc_ + 1, 0);
    for (int p = 0; p < nproc_; p++) scnt[p] = (int)bucket[p].size();
    MPI_Alltoall(scnt.data(), 1, MPI_INT,
                 rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int p = 0; p < nproc_; p++) {
        sdisp[p+1] = sdisp[p] + scnt[p];
        rdisp[p+1] = rdisp[p] + rcnt[p];
    }
    int ts = sdisp[nproc_], tr = rdisp[nproc_];

    /* 打包发送缓冲 */
    vector<Msg> sbuf(ts > 0 ? ts : 1), rbuf(tr > 0 ? tr : 1);
    int pos = 0;
    for (int p = 0; p < nproc_; p++) {
        int cnt = (int)bucket[p].size();
        if (cnt > 0) {
            memcpy(&sbuf[pos], bucket[p].data(), (size_t)cnt * sizeof(Msg));
            pos += cnt;
        }
    }

    /* 用 MPI_BYTE 派生类型一次发完 */
    MPI_Datatype mpi_msg_t;
    MPI_Type_contiguous(sizeof(Msg), MPI_BYTE, &mpi_msg_t);
    MPI_Type_commit(&mpi_msg_t);
    MPI_Alltoallv(sbuf.data(), scnt.data(), sdisp.data(), mpi_msg_t,
                  rbuf.data(), rcnt.data(), rdisp.data(), mpi_msg_t,
                  MPI_COMM_WORLD);
    MPI_Type_free(&mpi_msg_t);

    /* 写入 old_bc：按 old_g - v0_ 索引 */
    for (int i = 0; i < new_local_n_; i++) old_bc[i] = 0.0;
    for (int i = 0; i < tr; i++) {
        int local_idx = rbuf[i].old_g - v0_;
        old_bc[local_idx] = rbuf[i].bc;
    }
}
