#include "defs.h"

char ansFilename[FILENAME_LEN];
char resFilename[FILENAME_LEN];

using namespace std;

/* helper */
void usage(int argc, char **argv)
{
    printf("%s\n", argv[0]);
    printf("Usage:\n");
    printf("%s -ans <right answer> -res <result>\n", argv[0]);
    printf("Options:\n");
    printf("    -ans <right answer> -- right answer filename\n");
    printf("    -res <result> -- result filename\n");
    exit(1);
}

/* initialization */
void init(int argc, char **argv)
{
    int l;
    bool no_ansFilename = true;
    bool no_resFilename = true;
    ansFilename[0] = resFilename[0] = '\0';
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-ans")) {
            l = strlen(argv[++i]);
            strncpy(ansFilename, argv[i], (l > FILENAME_LEN - 1 ? FILENAME_LEN - 1 : l));
            no_ansFilename = false;
        }
        
        if (!strcmp(argv[i], "-res")) {
            l = strlen(argv[++i]);
            strncpy(resFilename, argv[i], (l > FILENAME_LEN - 1 ? FILENAME_LEN - 1 : l));
            no_resFilename = false;
        }
    }
    
    if (no_ansFilename || no_resFilename) {
        usage(argc, argv);
    }
}

int main(int argc, char **argv)
{
    init(argc, argv);
    
    FILE *f;
    f = fopen(ansFilename, "rb");
    assert(f != NULL);
    vector<double> answer;
    /* read the right answer */
    double val;
    while (fread(&val, sizeof(double), 1, f) == 1) {
        answer.push_back(val);
    }
    
    fclose(f);
    
    int n = answer.size();
    f = fopen(resFilename, "rb");
    assert(f != NULL);
    double *result = new double[n];
    assert(result != NULL);
    /* read the result */
    assert(fread(result, sizeof(double), n, f) == n);
    fclose(f);
    
    bool right_answer = true;
    /* first check */
    for (vertex_id_t i = 0; i < n; i++) {
        switch (fpclassify(result[i])) {
            case FP_INFINITE:
                right_answer = false;
                break;
            case FP_NAN:
                right_answer = false;
                break;
            case FP_SUBNORMAL:
                right_answer = false;
                break;
            default:
                break;
        }
    }
    
    if (!right_answer) {
        /* red color */
        cout << "\033[1;31mWrong answer\033[0m\n";
        
        delete[] result;
        
        return 0;
    }
    
    /* comparison the result with right answer */
    for (vertex_id_t i = 0; i < n; i++) {
        if (!(fabs(answer[i] - result[i]) < eps || fabs(answer[i] - result[i]) / answer[i] < eps)) {
            /* red color */
            cout << "\033[1;31mWrong answer\033[0m\n";
            
            delete[] result;
            
            return 0;
        }
    }
    
    /* green color */
    cout << "\033[1;32mAccepted\033[0m\n";
    
    delete[] result;
    
    return 0;
}
