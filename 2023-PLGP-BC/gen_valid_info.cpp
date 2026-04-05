#include "defs.h"

char inFilename[FILENAME_LEN];
char outFilename[FILENAME_LEN];

using namespace std;

/* helper */
void usage(int argc, char **argv)
{
    printf("%s\n", argv[0]);
    printf("Usage:\n");
    printf("%s -in <graph filename>\n", argv[0]);
    printf("Options:\n");
    printf("    -in <graph filename>\n");
    printf("    -out <output filename for the answer>\n");
    exit(1);
}

/* initialization */
void init(int argc, char **argv)
{
    int l;
    bool no_inFilename = true;
    bool no_outFilename = true;
    inFilename[0] = outFilename[0] = '\0';
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-in")) {
            l = strlen(argv[++i]);
            strncpy(inFilename, argv[i], (l > FILENAME_LEN - 1 ? FILENAME_LEN - 1 : l));
            no_inFilename = false;
        }
        
        if (!strcmp(argv[i], "-out")) {
            l = strlen(argv[++i]);
            strncpy(outFilename, argv[i], (l > FILENAME_LEN - 1 ? FILENAME_LEN - 1 : l));
            no_outFilename = false;
        }
    }
    
    if (no_inFilename) {
        usage(argc, argv);
    }
    
    if (no_outFilename) {
        sprintf(outFilename, "%s.ans", inFilename);
    }
}

int main(int argc, char **argv)
{
    graph_t g;
    init(argc, argv);
    
    /* read graph from the file */
    readGraph(&g, inFilename);
    
    double *answer = new double[g.n];
    assert(answer != NULL);
    /* get the right answer */
    run(&g, answer);
    
    FILE *f = fopen(outFilename, "wb");
    assert(f != NULL);
    /* write the answer */
    assert(fwrite(answer, sizeof(double), g.n, f) == g.n);
    fclose(f);
    
    delete[] answer;
    freeGraph(&g);
    
    return 0;
}
