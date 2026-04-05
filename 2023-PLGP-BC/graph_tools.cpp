#include "defs.h"

using namespace std;

/* read graph from the file */
void readGraph(graph_t *G, char *filename)
{
    unsigned char align;
    FILE *f = fopen(filename, "rb");
    assert(f != NULL);
    assert(fread(&G->n, sizeof(vertex_id_t), 1, f) == 1);
    G->scale = log(G->n) / log(2);
    assert(fread(&G->m, sizeof(edge_id_t), 1, f) == 1);
    assert(fread(&align, sizeof(unsigned char), 1, f) == 1);
    G->rowsIndices = new edge_id_t[G->n + 1];
    assert(G->rowsIndices != NULL);
    assert(fread(G->rowsIndices, sizeof(edge_id_t), G->n + 1, f) == (G->n + 1));
    G->endV = new vertex_id_t[G->m];
    assert(G->endV != NULL);
    assert(fread(G->endV, sizeof(vertex_id_t), G->m, f) == G->m);    
    fclose(f);
}

/* write graph to the file */
void writeGraph(graph_t *G, char *filename)
{
    FILE *f = fopen(filename, "wb");
    assert(f != NULL);
    assert(fwrite(&G->n, sizeof(vertex_id_t), 1, f) == 1);
    assert(fwrite(&G->m, sizeof(edge_id_t), 1, f) == 1);
    unsigned char align = 0;
    assert(fwrite(&align, sizeof(unsigned char), 1, f) == 1);
    assert(fwrite(G->rowsIndices, sizeof(edge_id_t), G->n + 1, f) == G->n + 1);
    assert(fwrite(G->endV, sizeof(vertex_id_t), G->m, f) == G->m);
    fclose(f);
}

/* all processes read graph from the file */
void readGraph_singleFile_MPI(graph_t *G, char *filename)
{
    unsigned char align;
    int rank, size;
    MPI_Offset offset, offset_row, offset_col;
    edge_id_t my_edges[2];
    int local_n = 0;
    int local_m = 0;
    int k;
    unsigned TotVertices;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    G->rank = rank;
    G->nproc = size;
    int lgsize; 
    for (lgsize = 0; lgsize < size; ++lgsize) {
        if ((1 << lgsize) == size) {
            break;
        }
    }
    
    MPI_File fh;
    MPI_Status status;    
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read(fh, &G->n, 1, MPI_UNSIGNED, &status);
    offset = sizeof(vertex_id_t);
    TotVertices = G->n;
    
    MPI_File_read_at(fh, offset, &G->m, 1, MPI_UNSIGNED_LONG_LONG, &status);
    offset += sizeof(edge_id_t);
    
    MPI_File_read_at(fh, offset, &align, 1, MPI_UNSIGNED_CHAR, &status);
    offset += sizeof(unsigned char);
    offset_row = offset;
    
    for (unsigned i = 0; i < G->n; i++) {
        if (rank == VERTEX_OWNER(i, TotVertices, size)) {
            MPI_File_read_at(fh, offset_row + i * sizeof(edge_id_t), &my_edges[0], 2, MPI_UNSIGNED_LONG_LONG, &status);
            local_n++;
            local_m += my_edges[1] - my_edges[0];
        }
    }
    
    G->local_n = local_n;
    G->local_m = local_m;
    offset_col = offset_row + (G->n + 1) * sizeof(edge_id_t);
    
    G->rowsIndices = new edge_id_t[G->local_n + 1];
    assert(G->rowsIndices != NULL);
    G->endV = new vertex_id_t[G->local_m];
    assert(G->endV != NULL);
    G->rowsIndices[0] = 0;
    k = 1;
    
    for (unsigned i = 0; i < G->n; i++) {
        if (rank == VERTEX_OWNER(i, TotVertices, size)) {
            MPI_File_read_at(fh, offset_row + i * sizeof(edge_id_t), &my_edges[0], 2, MPI_UNSIGNED_LONG_LONG, &status);
            G->rowsIndices[k] = G->rowsIndices[k - 1] + my_edges[1] - my_edges[0];
            MPI_File_read_at(fh, offset_col + my_edges[0] * sizeof(vertex_id_t), &G->endV[G->rowsIndices[k - 1]],
                G->rowsIndices[k] - G->rowsIndices[k - 1], MPI_UNSIGNED, &status);
            k++;
        }
    }
    
    MPI_File_close(&fh);
}

/* free graph */
void freeGraph(graph_t *G)
{
    delete[] G->rowsIndices;
    delete[] G->endV;
}
