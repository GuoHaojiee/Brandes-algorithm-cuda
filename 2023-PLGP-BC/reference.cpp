#include "defs.h"

using namespace std;

/* Max. number of vertices of the graph */
#define N 4096

/* class for the square matrices */
class matrix
{
private:
    unsigned **A;
    /* size is length of the side of the matrix, for example A[n][n] --> size = n */
    vertex_id_t size;
public:
    /* constructor (allocate memory for the matrix) */
    matrix(vertex_id_t n = 0)
    {
        assert(n != 0);
        size = n;
        A = new unsigned *[size];
        assert(A != NULL);
        for (vertex_id_t i = 0; i < size; i++) {
            A[i] = new unsigned[size];
            assert(A[i] != NULL);
        }
    }
    
    /* get one byte and fill all matrix */
    void fill(unsigned char val)
    {
        for (vertex_id_t i = 0; i < size; i++) {
            memset(A[i], val, size * sizeof(unsigned));
        }
    }
    
    /* setter */
    void set(vertex_id_t i, vertex_id_t j, unsigned val)
    {
        A[i][j] = val;
    }
    
    /* getter */
    unsigned get(vertex_id_t i, vertex_id_t j)
    {
        return A[i][j];
    }
    
    /* inc */
    void inc(vertex_id_t i, vertex_id_t j, unsigned val)
    {
        A[i][j] += val;
    }
    
    /* A = B (deep copy) */
    matrix& operator=(matrix &B)
    {
        vertex_id_t n = this->size;
        for (vertex_id_t i = 0; i < n; i++) {
            for (vertex_id_t j = 0; j < n; j++) {
                A[i][j] = B.get(i, j);
            }
        }
        
        return *this;
    }
    
    /* A *= B */
    matrix& operator*=(matrix &B)
    {
        vertex_id_t n = this->size;
        matrix C(n);
        C.fill(0);
        for (vertex_id_t i = 0; i < n; i++) {
            for (vertex_id_t j = 0; j < n; j++) {
                for (vertex_id_t k = 0; k < n; k++) {
                    C.inc(i, j, A[i][k] * B.get(k, j));
                }
            }
        }
        
        *this = C;
        
        return *this;
    }
    
    /* check, if matrix is null then return true, else - false */
    bool is_null()
    {
        vertex_id_t n = this->size;
        for (vertex_id_t i = 0; i < n; i++) {
            for (vertex_id_t j = 0; j < n; j++) {
                if (A[i][j]) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /* destructor */
    ~matrix()
    {
        for (vertex_id_t i = 0; i < size; i++) {
            delete[] A[i];
        }
        
        delete[] A;
    }
};

/* algorithm */
void run(graph_t *G, double *result)
{
    vertex_id_t n = G->n;
    
    /* reference works with small graphs only */
    if (n > N) {
        cout << "Fail, too large graph! Max. number of vertices of the graph = " << N << " (in the reference)" << endl;
        return;
    }
    
    /* init */
    /* A - adjacency matrix, cur_matrix - some power of A */
    matrix shortest_path(n), shortest_paths_cnt(n), cur_matrix(n), A(n);
    shortest_path.fill(0xFF);
    A.fill(0);
    for (vertex_id_t i = 0; i < n; i++) {
        for (edge_id_t j = G->rowsIndices[i]; j < G->rowsIndices[i + 1]; j++) {
            A.inc(i, G->endV[j], 1);
        }
    }
    
    /* cur_matrix = I = A^0 */
    cur_matrix.fill(0);
    for (vertex_id_t i = 0; i < n; i++) {
        cur_matrix.set(i, i, 1);
    }
    
    /* main part */
    unsigned dist = 0;
    while (true) {
        bool flag = false;
        for (vertex_id_t i = 0; i < n; i++) {
            for (vertex_id_t j = 0; j < n; j++) {
                if (shortest_path.get(i, j) == UINT32_MAX && cur_matrix.get(i, j)) {
                    shortest_path.set(i, j, dist);
                    shortest_paths_cnt.set(i, j, cur_matrix.get(i, j));
                    flag = true;
                }
            }
        }
        
        if (!flag) {
            break;
        }
        
        dist++;
        cur_matrix *= A;
    }
    
    /* calc */
    for (vertex_id_t u = 0; u < n; u++) {
        result[u] = 0;
        for (vertex_id_t i = 0; i < n; i++) {
            if (i == u) {
                continue;
            }
            
            for (vertex_id_t j = i + 1; j < n; j++) {
                if (j == u) {
                    continue;
                }
                
                /* if one of the shortest paths between i and j contains vertex u */
                if (shortest_path.get(i, u) + shortest_path.get(u, j) == shortest_path.get(i, j)) {
                    result[u] += (double)shortest_paths_cnt.get(i, u) * shortest_paths_cnt.get(u, j) / shortest_paths_cnt.get(i, j);
                }
            }
        }
    }
}
