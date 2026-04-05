#include "defs.h"

using namespace std;

/* Max. number of vertices of the graph */
#define N 4096

/* bfs from the vertex start_v */
void bfs(graph_t *G, vertex_id_t start_v, unsigned **shortest_path, unsigned **shortest_paths_cnt)
{
    /* init */
    unsigned d[N], cnt[N];
    memset(d, 0xFF, sizeof(d));
    memset(cnt, 0, sizeof(cnt));
    
    /* distance to start_v from start_v is 0 */
    d[start_v] = 0;
    
    /* number of the shortest paths to start_v from start_v is 1 */
    cnt[start_v] = 1;
    
    /* queue for the bfs */
    queue<vertex_id_t> q;
    q.push(start_v);
    
    /* main part */
    while (!q.empty()) {
        vertex_id_t u = q.front();
        q.pop();
        for (edge_id_t j = G->rowsIndices[u]; j < G->rowsIndices[u + 1]; j++) {
            vertex_id_t v = G->endV[j];
            if (d[u] + 1 < d[v]) {
                d[v] = d[u] + 1;
                cnt[v] = cnt[u];
                q.push(v);
            } else if (d[u] + 1 == d[v]) {
                cnt[v] += cnt[u];
            }
        }
    }
    
    /* fill input arrays */
    for (vertex_id_t i = 0; i < G->n; i++) {
        shortest_path[start_v][i] = d[i];
        shortest_paths_cnt[start_v][i] = cnt[i];
    }
}

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
    unsigned **shortest_path = new unsigned *[n], **shortest_paths_cnt = new unsigned *[n];
    assert(shortest_path != NULL);
    assert(shortest_paths_cnt != NULL);
    for (vertex_id_t i = 0; i < n; i++) {
        shortest_path[i] = new unsigned[n];
        assert(shortest_path[i] != NULL);
        shortest_paths_cnt[i] = new unsigned[n];
        assert(shortest_paths_cnt[i] != NULL);
    }
    
    /* bfs for calculating shortest_path and shortest_paths_cnt for all pairs of vertices */
    for (vertex_id_t i = 0; i < n; i++) {
        bfs(G, i, shortest_path, shortest_paths_cnt);
    }
    
    /* main part */
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
                if (shortest_path[i][u] + shortest_path[u][j] == shortest_path[i][j]) {
                    result[u] += (double)shortest_paths_cnt[i][u] * shortest_paths_cnt[u][j] / shortest_paths_cnt[i][j];
                }
            }
        }
    }
    
    /* free */
    for (vertex_id_t i = 0; i < n; i++) {
        delete[] shortest_path[i];
        delete[] shortest_paths_cnt[i];
    }
    
    delete[] shortest_path;
    delete[] shortest_paths_cnt;
}
