#ifndef __GRAPH_HPC_DEFS_H
#define __GRAPH_HPC_DEFS_H

#include <mpi.h>
#include <cstdio>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <ctime>
#include <sys/types.h>
#include <iostream>
#include <queue>
#include <cstring>
#include <cfloat>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#define AVG_VERTEX_DEGREE 16
#define FILENAME_LEN 256
#define eps 1e-6
#define UINT32_MAX 0xFFFFFFFF

typedef unsigned vertex_id_t;
typedef unsigned long long edge_id_t;

/* The graph data structure*/
typedef struct
{
    /***
     The minimal graph repesentation consists of:
     n        -- the number of vertices
     m        -- the number of edges
     endV     -- an array of size m that stores the
                 destination ID of an edge <src->dest>.
     rowsIndices -- an array of size (n + 1) that stores pointers to the endV array (CRS format).
                 The degree of vertex i is given by rowsIndices[i + 1] - rowsIndices[i], and the
                 edges out of i are stored in the contiguous block
                 endV[rowsIndices[i] .. rowsIndices[i + 1] - 1].
     Vertices are numbered from 0 in our internal representation.
     ***/
    vertex_id_t n;
    edge_id_t m;
    edge_id_t *rowsIndices;
    vertex_id_t *endV;
    
    /* other graph parameters */
    int scale; /* log2 of vertices number */
    int avg_vertex_degree; /* relation m / n */
    
    /* RMAT graph parameters */
    double a, b, c;
    bool permute_vertices;
    
    /* Distributed version variables */
    int nproc, rank;
    vertex_id_t local_n; /* local vertices number */
    edge_id_t local_m; /* local edges number */
    edge_id_t *num_edges_of_any_process;
} graph_t;

/* write graph to file */
void writeGraph(graph_t *G, char *filename);

/* read graph from file */
void readGraph(graph_t *G, char *filename);
void readGraph_singleFile_MPI(graph_t *G, char *filename);

/* print text graph in std out */
void printGraph(graph_t *G);

/* free graph */
void freeGraph(graph_t *G);

/* algorithm */
void run(graph_t *G, double *result);

/* generators */
void gen_RMAT_graph_MPI(graph_t *G);
void gen_random_graph_MPI(graph_t *G);

#define MOD_SIZE(v) ((v) % size)
#define DIV_SIZE(v) ((v) / size)
#define MUL_SIZE(x) ((x) * size)

/* returns number of vertex owner, v - the global vertex number, TotVertices - the global number of vertices, size - the number of processes */
inline int VERTEX_OWNER(const vertex_id_t v, const vertex_id_t TotVertices, const int size)
{
    vertex_id_t mod_size = MOD_SIZE(TotVertices);
    vertex_id_t div_size = DIV_SIZE(TotVertices);
    if (!mod_size) {
        return v / div_size;
    } else {
        if (v / (div_size + 1) < mod_size) {
            return v / (div_size + 1);
        } else {
            return (v - mod_size * (div_size + 1)) / div_size + mod_size;
        }
    }
}

/* returns local vertex number, v - the global vertex number, TotVertices - the global number of vertices, size - the number of processes, rank - the process number */
inline vertex_id_t VERTEX_LOCAL(const vertex_id_t v, const vertex_id_t TotVertices, const int size, const int rank) 
{
    if (MOD_SIZE(TotVertices) <= (unsigned int)rank) {
        return ((v - MOD_SIZE(TotVertices) * (DIV_SIZE(TotVertices) + 1)) % DIV_SIZE(TotVertices));
    } else {
        return (v % (DIV_SIZE(TotVertices) + 1));
    }
}

/* returns global vertex number, v_local - the local vertex number, TotVertices - the global number of vertices, size - the number of processes, rank - the process number */
inline vertex_id_t VERTEX_TO_GLOBAL(const vertex_id_t v_local, const vertex_id_t TotVertices, const int size, const int rank)
{
    if (MOD_SIZE(TotVertices) > (unsigned int)rank) {
        return ((DIV_SIZE(TotVertices) + 1) * rank + (vertex_id_t)v_local);
    } else {
        return (MOD_SIZE(TotVertices) * (DIV_SIZE(TotVertices) + 1) + DIV_SIZE(TotVertices) * (rank - MOD_SIZE(TotVertices)) + v_local);
    }
}

/* calculate local vertex count (is using in distributed graph generators) */
inline vertex_id_t get_local_n(graph_t *G)
{
    vertex_id_t TotVertices = G->n;
    unsigned size = G->nproc;
    unsigned rank = G->rank;
    vertex_id_t mod_size = MOD_SIZE(TotVertices);
    vertex_id_t div_size = DIV_SIZE(TotVertices);
    if (!mod_size) {
        return div_size;
    } else {
        if (rank < mod_size) {
            return div_size + 1;
        } else {
            return div_size;
        }
    }
}

#endif
