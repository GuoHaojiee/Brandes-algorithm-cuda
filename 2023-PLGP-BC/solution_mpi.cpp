
#include "defs.h"
#include <vector>
#include <cstdio>

using namespace std;

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

static inline int local_n_of_rank(int n_total, int nproc, int rank)
{
    int q = n_total / nproc;
    int r = n_total % nproc;
    return (rank < r) ? (q + 1) : q;
}


void run(graph_t *G, double *result)
{
    int rank  = G->rank;
    int nproc = G->nproc;
    int n     = (int)G->n;
    int loc_n = (int)G->local_n;
    int loc_m = (int)G->local_m;  

    double t_comm0 = MPI_Wtime();

   
    vector<int> all_lm(nproc);
    MPI_Allgather(&loc_m, 1, MPI_INT,
                  all_lm.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> endv_disp(nproc + 1, 0);
    for (int p = 0; p < nproc; p++)
        endv_disp[p + 1] = endv_disp[p] + all_lm[p];
    int total_edges = endv_disp[nproc];  /* 全图边数（双向，等于 G->m） */

    vector<unsigned> full_endV_u(total_edges);
    MPI_Allgatherv(G->endV, loc_m, MPI_UNSIGNED,
                   full_endV_u.data(), all_lm.data(), endv_disp.data(),
                   MPI_UNSIGNED, MPI_COMM_WORLD);

    vector<int> all_ln(nproc);
    vector<int> row_disp(nproc + 1, 0);  /* 同时作为 Allgatherv 位移和全局顶点偏移 */
    for (int p = 0; p < nproc; p++) {
        all_ln[p]       = local_n_of_rank(n, nproc, p);
        row_disp[p + 1] = row_disp[p] + all_ln[p];
    }
    vector<int> my_rows(loc_n);
    for (int i = 0; i < loc_n; i++)
        my_rows[i] = (int)G->rowsIndices[i];

    vector<int> all_rows(n);  /* 收集后：all_rows[row_disp[p]+l] = 进程p的rowsIndices[l] */
    MPI_Allgatherv(my_rows.data(), loc_n, MPI_INT,
                   all_rows.data(), all_ln.data(), row_disp.data(),
                   MPI_INT, MPI_COMM_WORLD);

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

    vector<int> full_dest(total_edges);
    for (int i = 0; i < total_edges; i++)
        full_dest[i] = (int)full_endV_u[i];

    double t_comm1 = MPI_Wtime();
    if (rank == 0)
        printf("[MPI] 图组装通信时间: %.4f 秒\n", t_comm1 - t_comm0);

    int gpu_device = rank % 2;

    double t_gpu0 = MPI_Wtime();

    vector<int> sources(loc_n);
    for (int i = 0; i < loc_n; i++)
        sources[i] = (int)VERTEX_TO_GLOBAL((vertex_id_t)i, G->n, G->nproc, G->rank);

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

    { vector<unsigned>().swap(full_endV_u); }
    { vector<int>().swap(full_dest);        }
    { vector<int>().swap(full_offset);      }

    double t_gpu1 = MPI_Wtime();
    if (rank == 0)
        printf("[GPU] 计算时间: %.4f 秒\n", t_gpu1 - t_gpu0);

    double t_red0 = MPI_Wtime();

    MPI_Allreduce(MPI_IN_PLACE, bc_partial.data(), n,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double t_red1 = MPI_Wtime();
    if (rank == 0)
        printf("[MPI] Allreduce 时间: %.4f 秒\n", t_red1 - t_red0);

    for (int i = 0; i < loc_n; i++)
        result[i] = bc_partial[sources[i]];

    for (int i = 0; i < loc_n; i++)
        result[i] /= 2.0;
}
