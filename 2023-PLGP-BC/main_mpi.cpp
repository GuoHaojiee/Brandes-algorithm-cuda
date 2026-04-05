#include "defs.h"

using namespace std;

char inFilename[FILENAME_LEN];
char outFilename[FILENAME_LEN];

int nIters = 1;

/* helper */
void usage(int argc, char **argv)
{
    printf("Usage:\n");
    printf("%s -in <input> [other options]\n", argv[0]);
    printf("Options:\n");
    printf("    -in <input> -- input graph filename\n");
    printf("    --generate <graph type> -- generate graph. Supported options of <type> are: RMAT, random\n");
    printf("    -s <scale> -- number of vertices is 2^<scale>, in case of --generate option\n");
    printf("    -nIters <nIters> -- number of iterations\n");
	printf("    -out <output filename>, file for the result storage\n");
    MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
}

/* initialization */
void init(int argc, char **argv, graph_t *G)
{
    MPI_Init(&argc, &argv);
    G->scale = -1;
    bool should_read_graph = false;
    bool should_gen_rmat = false;
    bool should_gen_random = false;
    int l;
    MPI_Comm_size(MPI_COMM_WORLD, &G->nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &G->rank);
    
    if (argc == 1) {
        usage(argc, argv);
    }
    
	bool no_outFilename = true;
	
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--generate")) {
            if (!strcmp(argv[i + 1], "RMAT")) {
                should_gen_rmat = true;
            } else if (!strcmp(argv[i + 1], "random")) {
                should_gen_random = true;
            } else {
                usage(argc, argv);
            }
            
            ++i;
        }
        
        if (!strcmp(argv[i], "-in")) {
            should_read_graph = true;
            l = strlen(argv[++i]);
            strncpy(inFilename, argv[i], (l > FILENAME_LEN - 1 ? FILENAME_LEN - 1 : l));
        }
        
        if (!strcmp(argv[i], "-s")) {
            G->scale = (int)atoi(argv[++i]);
            G->n = (vertex_id_t)1 << G->scale;
        }
        
        if (!strcmp(argv[i], "-nIters")) {
            nIters = (int)atoi(argv[++i]);
        }
		
		if (!strcmp(argv[i], "-out")) {
            no_outFilename = false;
            sprintf(outFilename, argv[++i]);
        }
    }
    
    if (should_gen_rmat && G->scale != -1) {
        gen_RMAT_graph_MPI(G);
    } else if (should_gen_random && G->scale != -1) {
        gen_random_graph_MPI(G);
    } else {
        if (should_read_graph) {
            readGraph_singleFile_MPI(G, inFilename);
        } else {
            usage(argc, argv);
        }
    }
    
    if (no_outFilename) {
        if (should_read_graph) {
            sprintf(outFilename, "%s.res", inFilename);
        } else if (should_gen_random) {
            sprintf(outFilename,"random-%d.res", G->scale);
        } else {
            sprintf(outFilename,"rmat-%d.res", G->scale);
        }
    }
}

/* write result to the file */
void write_output_information(double *result, vertex_id_t local_n, char *filename)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        /* create file if it doesn't exist */
        FILE *f = fopen(filename, "w");
        fclose(f);
        
        f = fopen(filename, "ab");
        assert(f != NULL);
        assert(fwrite(result, sizeof(double), local_n, f) == local_n);
        fclose(f);
        
        if (rank != size - 1) {
            int trash = 0;
            MPI_Send(&trash, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        int trash;
        MPI_Status status;
        MPI_Recv(&trash, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        
        FILE *f = fopen(filename, "ab");
        assert(f != NULL);
        assert(fwrite(result, sizeof(double), local_n, f) == local_n);
        fclose(f);
        
        if (rank != size - 1) {
            int trash = 0;
            MPI_Send(&trash, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }
}

void print0(int rank, const char *s)
{
    if (rank == 0) {
        cout << s << endl;
    }
}

int main(int argc, char **argv)
{
    graph_t g;
    double start_ts, finish_ts;
    double *alg_time;
    /* initializing */
    init(argc, argv, &g);
    
    alg_time = new double[nIters];
    assert(alg_time != NULL);
    double *result = new double[g.local_n];
    assert(result != NULL);
    
    print0(g.rank, "start algorithm...");
    for (int i = 0; i < nIters; i++) {
        if (g.rank == 0) {
            cout << "iter number = " << i;
        }
        
        memset(result, 0, g.local_n * sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        start_ts = MPI_Wtime();
        run(&g, result);
        MPI_Barrier(MPI_COMM_WORLD);
        finish_ts = MPI_Wtime();
        double time = finish_ts - start_ts;
        alg_time[i] = time;
        if (g.rank == 0) {
            cout.precision(5);
            cout << " finished. Time is " << fixed << time << " sec." << endl;
        }
    }
    
    print0(g.rank, "algorithm iterations finished.");
    write_output_information(result, g.local_n, outFilename);
    
    /* final print */
    double min_time, max_time, avg_time;
    double global_min_time, global_max_time, global_avg_time;
    max_time = avg_time = 0;
    min_time = DBL_MAX;     
    for (int i = 0; i < nIters; ++i) {  
        avg_time += alg_time[i];
        if (alg_time[i] < min_time) {
            min_time = alg_time[i];
        }
        
        if (alg_time[i] > max_time) {
            max_time = alg_time[i];
        }
    }
    
    avg_time /= nIters;
    
    MPI_Reduce(&min_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_time, &global_avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (g.rank == 0) {
        cout << inFilename << " vertices = " << g.n << ", edges = " << g.m << ", nIters = " << nIters 
            << ", time: min = " << global_min_time << ", avg = " << global_avg_time / g.nproc << ", max = " << global_max_time << endl;
        // print average time
        cout << "Time = " << global_avg_time / g.nproc << " sec." << endl;
    }
    
    delete[] alg_time;
    delete[] result;
    freeGraph(&g);
    
    MPI_Finalize();
    return 0;
}
