/*
 * solution_mpi.cpp
 * MPI + GPU 分布式 Brandes 介数中心性算法
 *
 * 整体流程：
 *   1. MPI_Allgatherv：各进程协作，在每个进程本地组装完整图 CSR
 *   2. 绑定 GPU（每节点 2 块 P100，按 rank 交替）
 *   3. 以本进程负责的顶点为源节点，调用 GPU 计算部分 BC
 *   4. MPI_Allreduce：汇总各进程的部分 BC，得到完整 BC
 *   5. 提取本进程本地顶点的 BC 值，除以 2（无向图修正）
 *
 * 内存可扩展性说明：
 *   - 各进程平时只持有 local_n 个顶点的图数据（O(N/P)）
 *   - 完整图 CSR（O(N)）只在 GPU 计算前临时组装，函数返回后自动释放
 *   - bc_partial（O(N)）同样是临时缓冲，用完后释放
 *   - result 数组始终只有 local_n 大小（O(N/P)）
 */

#include "defs.h"
#include <vector>
#include <cstdio>

using namespace std;

/* 声明 GPU 计算函数（在 brandes_gpu.cu 中实现） */
extern "C" void computeBC_gpu(
    int     n,
    int*    offset,
    int*    dest,
    int     total_edges,
    int*    sources,
    int     n_src,
    double* bc_out,
    int     gpu_device
);

/* 计算进程 rank 拥有的本地顶点数（与 defs.h get_local_n 逻辑一致） */
static inline int local_n_of_rank(int n_total, int nproc, int rank)
{
    int q = n_total / nproc;
    int r = n_total % nproc;
    return (rank < r) ? (q + 1) : q;
}

/*
 * run() — 框架入口
 * result[i] 存储本进程第 i 个本地顶点的完整 BC 值（i 为本地编号）
 */
void run(graph_t *G, double *result)
{
    int rank  = G->rank;
    int nproc = G->nproc;
    int n     = (int)G->n;
    int loc_n = (int)G->local_n;
    int loc_m = (int)G->local_m;  /* 本进程边数（int 范围内安全，scale<=25） */

    /* ================================================================
     * 第一步：MPI 通信，各进程协作组装完整图 CSR
     * ================================================================ */
    double t_comm0 = MPI_Wtime();

    /* 1a. 收集各进程 local_m，计算 endV 的位移 */
    vector<int> all_lm(nproc);
    MPI_Allgather(&loc_m, 1, MPI_INT,
                  all_lm.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> endv_disp(nproc + 1, 0);
    for (int p = 0; p < nproc; p++)
        endv_disp[p + 1] = endv_disp[p] + all_lm[p];
    int total_edges = endv_disp[nproc];  /* 全图边数（双向，等于 G->m） */

    /* 1b. Allgatherv：汇集所有进程的 endV（全局顶点编号） */
    vector<unsigned> full_endV_u(total_edges);
    MPI_Allgatherv(G->endV, loc_m, MPI_UNSIGNED,
                   full_endV_u.data(), all_lm.data(), endv_disp.data(),
                   MPI_UNSIGNED, MPI_COMM_WORLD);

    /* 1c. 计算各进程的本地顶点数和全局顶点起始编号
     *     顶点采用块分布（连续块），g_starts[p] = 进程 p 的第一个全局顶点编号 */
    vector<int> all_ln(nproc);
    vector<int> row_disp(nproc + 1, 0);  /* 同时作为 Allgatherv 位移和全局顶点偏移 */
    for (int p = 0; p < nproc; p++) {
        all_ln[p]       = local_n_of_rank(n, nproc, p);
        row_disp[p + 1] = row_disp[p] + all_ln[p];
    }
    /* row_disp[p] 即 g_starts[p]（各进程全局顶点起始编号） */

    /* 1d. 收集各进程的 rowsIndices[0..local_n-1]（本地相对偏移，转为 int）
     *     rowsIndices[i] = 本进程前 i 个顶点的边数之和，从 0 开始 */
    vector<int> my_rows(loc_n);
    for (int i = 0; i < loc_n; i++)
        my_rows[i] = (int)G->rowsIndices[i];

    vector<int> all_rows(n);  /* 收集后：all_rows[row_disp[p]+l] = 进程p的rowsIndices[l] */
    MPI_Allgatherv(my_rows.data(), loc_n, MPI_INT,
                   all_rows.data(), all_ln.data(), row_disp.data(),
                   MPI_INT, MPI_COMM_WORLD);

    /* 1e. 构建完整 full_offset[n+1]
     *     对进程 p 的第 l 个本地顶点（全局编号 row_disp[p]+l）：
     *     full_offset[row_disp[p]+l] = endv_disp[p] + rowsIndices_p[l]
     *     即：全局边偏移 + 本地内部偏移 */
    vector<int> full_offset(n + 1);
    {
        int gv = 0;
        for (int p = 0; p < nproc; p++) {
            int base = endv_disp[p];          /* 进程 p 的边全局偏移 */
            int roff = row_disp[p];           /* 进程 p 的 all_rows 起始位置 */
            for (int l = 0; l < all_ln[p]; l++)
                full_offset[gv++] = base + all_rows[roff + l];
        }
        full_offset[n] = total_edges;
    }

    /* 将 endV 从 unsigned 转为 int（顶点编号 < n <= 2^31，安全） */
    vector<int> full_dest(total_edges);
    for (int i = 0; i < total_edges; i++)
        full_dest[i] = (int)full_endV_u[i];

    double t_comm1 = MPI_Wtime();
    if (rank == 0)
        printf("[MPI] 图组装通信时间: %.4f 秒\n", t_comm1 - t_comm0);

    /* ================================================================
     * 第二步：绑定 GPU
     * 每节点 2 块 P100，偶数 rank 用 GPU 0，奇数 rank 用 GPU 1
     * ================================================================ */
    int gpu_device = rank % 2;

    /* ================================================================
     * 第三步：GPU 计算
     * 本进程以其 local_n 个顶点为源节点，计算这些源的 BC 贡献
     * bc_partial[v] = 来自本进程源节点、对顶点 v 的 BC 贡献之和（部分值）
     * ================================================================ */
    double t_gpu0 = MPI_Wtime();

    /* 构建源节点列表：本地编号 → 全局编号 */
    vector<int> sources(loc_n);
    for (int i = 0; i < loc_n; i++)
        sources[i] = (int)VERTEX_TO_GLOBAL((vertex_id_t)i, G->n, G->nproc, G->rank);

    /* GPU 输出：大小 n，记录本进程源节点对各顶点的部分 BC 贡献 */
    vector<double> bc_partial(n, 0.0);

    computeBC_gpu(
        n,
        full_offset.data(),
        full_dest.data(),
        total_edges,
        sources.data(),
        loc_n,
        bc_partial.data(),
        gpu_device
    );

    /* 完整图临时数据在此之后不再需要，vector 析构时自动释放内存 */
    { vector<unsigned>().swap(full_endV_u); }
    { vector<int>().swap(full_dest);        }
    { vector<int>().swap(full_offset);      }

    double t_gpu1 = MPI_Wtime();
    if (rank == 0)
        printf("[GPU] 计算时间: %.4f 秒\n", t_gpu1 - t_gpu0);

    /* ================================================================
     * 第四步：MPI Allreduce，汇总各进程的部分 BC
     * 每个进程只计算了自己源节点集合的贡献；
     * Allreduce(SUM) 后，bc_partial[v] = 所有源节点对 v 的完整 BC 贡献
     * ================================================================ */
    double t_red0 = MPI_Wtime();

    MPI_Allreduce(MPI_IN_PLACE, bc_partial.data(), n,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double t_red1 = MPI_Wtime();
    if (rank == 0)
        printf("[MPI] Allreduce 时间: %.4f 秒\n", t_red1 - t_red0);

    /* 将完整 BC 中本进程负责的本地顶点部分写入 result */
    for (int i = 0; i < loc_n; i++)
        result[i] = bc_partial[sources[i]];

    /* ================================================================
     * 第五步：无向图结果除以 2
     * 无向图中每条最短路径被正向和反向各计数一次，需除以 2 修正
     * ================================================================ */
    for (int i = 0; i < loc_n; i++)
        result[i] /= 2.0;
}
