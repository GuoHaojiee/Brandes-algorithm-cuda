#include "defs.h"

using namespace std;

/* function returns size of edges block for current process */
static edge_id_t get_local_m(graph_t *G)
{
    return G->local_n * G->avg_vertex_degree;
}

static unsigned long long my_next = 1;

static double my_rand()
{
    my_next = my_next * 1365781351523LL + 12345;
    return (double)((my_next / (1LL << 32)) % (1LL << 31)) / (1LL << 31);
}

static void my_srand(unsigned seed)
{
    my_next = seed;
}

/* distributed RMAT graph generator */
void gen_RMAT_graph_MPI(graph_t *G) 
{
    /* init */
    vertex_id_t n, local_n;
    edge_id_t local_m, num_dir_edges;
    edge_id_t offset;
    double a, b, c, d;
    double av, bv, cv, dv, S, p;
    int SCALE;
    double var;
    vertex_id_t step;
    vertex_id_t *permV, tmpVal;
    bool permute_vertices;
    vertex_id_t u, v;
    int seed;
    vertex_id_t *edges;
    vertex_id_t *degree;
    int size, rank, lgsize;
    G->a = 0.45;
    G->b = 0.25;
    G->c = 0.15;
    G->permute_vertices = true;
    G->avg_vertex_degree = AVG_VERTEX_DEGREE;
    G->n = (vertex_id_t)1 << G->scale;
    G->m = G->n * (edge_id_t)G->avg_vertex_degree;
    a = G->a;
    b = G->b;
    c = G->c;
    assert(a + b + c < 1);
    d = 1 - (a + b + c);
    permute_vertices = G->permute_vertices;
    n = G->n;
    unsigned TotVertices;
    TotVertices = G->n;
    
    MPI_Datatype MPI_VERTEX_ID_T;
    MPI_Type_contiguous(sizeof(vertex_id_t), MPI_BYTE, &MPI_VERTEX_ID_T);
    MPI_Type_commit(&MPI_VERTEX_ID_T);
    
    rank = G->rank;
    size = G->nproc;
    for (lgsize = 0; lgsize < size; ++lgsize) {
        if ((1 << lgsize) == size) {
            break;
        }
    }
    
    /* get size of vertices and edges blocks for current processes */
    local_n = get_local_n(G);
    G->local_n = local_n;
    local_m = get_local_m(G);
    num_dir_edges = local_m;
    local_m = 2 * local_m;
    G->local_m = local_m;
    
    edges = new vertex_id_t[2 * num_dir_edges];
    assert(edges != NULL);
    degree = new vertex_id_t[local_n];
    memset(degree, 0, sizeof(vertex_id_t) * local_n);
    assert(degree != NULL);
    
    seed = 2387 + rank;
    srand48(seed);
    SCALE = G->scale;
    /* generate edges */
    for (edge_id_t i = 0; i < num_dir_edges; i++) {
        u = 1;
        v = 1;
        step = n / 2;
        
        av = a;
        bv = b;
        cv = c;
        dv = d;
        p = drand48();
        if (p < av) {
            /* Do nothing */
        } else if ((p >= av) && (p < av + bv)) {
            v += step;
        } else if ((p >= av + bv) && (p < av + bv + cv)) {
            u += step;
        } else {
            u += step;
            v += step;
        }
        
        for (int j = 1; j < SCALE; j++) {
            step = step / 2;
            /* Vary a, b, c, d by up to 10% */
            var = 0.1;
            av *= 0.95 + var * drand48();
            bv *= 0.95 + var * drand48();
            cv *= 0.95 + var * drand48();
            dv *= 0.95 + var * drand48();
            
            S = av + bv + cv + dv;
            av = av / S;
            bv = bv / S;
            cv = cv / S;
            dv = dv / S;
            
            /* choose partition */
            p = drand48();
            if (p < av) {
                /* Do nothing */
            } else if ((p >= av) && (p < av + bv)) {
                v += step;
            } else if ((p >= av + bv) && (p < av + bv + cv)) {
                u += step;
            } else {
                u += step;
                v += step;
            }
        }
        
        edges[2 * i + 0] = u - 1;
        edges[2 * i + 1] = v - 1;
    }
    
    /* reshuffle */
    if (permute_vertices) {
        permV = new vertex_id_t[n];
        assert(permV != NULL);
        
        my_srand(4791);
        for (vertex_id_t i = 0; i < n; i++) {
            permV[i] = i;
        }
        
        for (vertex_id_t i = 0; i < n; i++) {
            vertex_id_t j = n * my_rand();
            tmpVal = permV[i];
            permV[i] = permV[j];
            permV[j] = tmpVal;
        }
        
        for (edge_id_t i = 0; i < num_dir_edges; i++) {
            edges[2 * i + 0] = permV[edges[2 * i + 0]];
            edges[2 * i + 1] = permV[edges[2 * i + 1]];
        }
        
        delete[] permV;
    }
    
    vertex_id_t *send_edges = new vertex_id_t[2 * local_m];
    assert(send_edges != NULL);
    vertex_id_t *recv_edges;
    int *send_counts = new int[size];
    assert(send_counts != NULL);
    memset(send_counts, 0, sizeof(int) * size);
    int *recv_counts = new int[size];
    assert(recv_counts != NULL);
    memset(recv_counts, 0, sizeof(int) * size);
    edge_id_t *send_offsets_edge = new edge_id_t[size];
    assert(send_offsets_edge != NULL);
    memset(send_offsets_edge, 0, sizeof(edge_id_t) * size);
    edge_id_t *recv_offsets_edge = new edge_id_t[size];
    assert(recv_offsets_edge != NULL);
    memset(recv_offsets_edge, 0, sizeof(edge_id_t) * size);
    /* calc count of data in each process */
    for (edge_id_t i = 0; i < num_dir_edges; i++) {
        int proc_id = VERTEX_OWNER(edges[2 * i + 0], TotVertices, size);
        send_counts[proc_id]++;
        proc_id = VERTEX_OWNER(edges[2 * i + 1], TotVertices, size);
        send_counts[proc_id]++;
    }
    
    /* calc offsets */
    for (int i = 1; i < size; i++) {
        send_offsets_edge[i] = send_offsets_edge[i - 1] + 2 * send_counts[i - 1];
    }
    
    /* clear send_counts for next using */
    for (int i = 0; i < size; i++) {
        send_counts[i] = 0;
    }
    
    /* copy edges to send_data */
    for (edge_id_t i = 0; i < num_dir_edges; i++) {
        int proc_id = VERTEX_OWNER(edges[2 * i + 0], TotVertices, size);
        offset = send_offsets_edge[proc_id] + 2 * send_counts[proc_id];
        send_edges[offset + 0] = edges[2 * i + 0];
        send_edges[offset + 1] = edges[2 * i + 1];
        send_counts[proc_id]++;
        proc_id = VERTEX_OWNER(edges[2 * i + 1], TotVertices, size);
        offset = send_offsets_edge[proc_id] + 2 * send_counts[proc_id];
        send_edges[offset + 0] = edges[2 * i + 1];
        send_edges[offset + 1] = edges[2 * i + 0];
        send_counts[proc_id]++;
    }
    
    delete[] edges;
    MPI_Request request[size];
    MPI_Status status[size];
    /* report counts to each process */
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    /* calc offsets and number of elements for MPI_Send */
    edge_id_t counts = 0;
    for(int i = 0; i < size; i++) {
        counts += recv_counts[i];
    }
    
    /* calc offsets and number of elements for the next MPI_Send */
    for (int i = 0; i < size; i++) {
        recv_counts[i] = 2 * recv_counts[i];
        send_counts[i] = 2 * send_counts[i]; 
    }
    
    for (int i = 1; i < size; i++) {
        recv_offsets_edge[i] = recv_offsets_edge[i - 1] + recv_counts[i - 1];
    }
    
    recv_edges = new vertex_id_t[2 * counts];
    assert(recv_edges != NULL);
    /* send edges to each process */
    for (int i = 0; i < size; i++) {
        MPI_Irecv(&recv_edges[recv_offsets_edge[i]], recv_counts[i], MPI_VERTEX_ID_T, i, G->rank, MPI_COMM_WORLD, &request[i]);
    }
    
    for (int i = 0; i < size; i++) {
        MPI_Send(&send_edges[send_offsets_edge[i]], send_counts[i], MPI_VERTEX_ID_T, i, i, MPI_COMM_WORLD);
    }
    
    MPI_Waitall(size, request, status);
    /* saving new value for local_m */
    local_m = counts;
    G->local_m = local_m;
    /* undirected graph, each edge is stored twice; if edge is (u, v), then it's
     *stored at the vertex u and at the vertex v */
    G->m *= 2;
    delete[] send_edges;
    delete[] recv_offsets_edge;
    delete[] send_offsets_edge;
    delete[] recv_counts;
    delete[] send_counts;
    
    for (edge_id_t i = 0; i < 2 * G->local_m; i = i + 2) {
        degree[VERTEX_LOCAL(recv_edges[i], TotVertices, size, rank)]++;
    }
    
    /* update graph data structure */
    G->endV = new vertex_id_t[G->local_m];
    assert(G->endV != NULL);
    memset(G->endV, 0, sizeof(vertex_id_t) * G->local_m);
    G->rowsIndices = new edge_id_t[local_n + 1];
    assert(G->rowsIndices != NULL);
    G->rowsIndices[0] = 0; 
    for (vertex_id_t i = 1; i <= G->local_n; i++) {
        G->rowsIndices[i] = G->rowsIndices[i - 1] + degree[i - 1];
    }
    
    for (edge_id_t i = 0; i < 2 * G->local_m; i = i + 2) {
        u = VERTEX_LOCAL(recv_edges[i + 0], TotVertices, size, rank);
        v = recv_edges[i + 1];
        offset = degree[u]--;
        G->endV[G->rowsIndices[u] + offset - 1] = v;
    }
    
    delete[] recv_edges;
    delete[] degree;
}
