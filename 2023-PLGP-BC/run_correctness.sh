#!/bin/bash
#BSUB -J bc_correct
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o correct_%J.out
#BSUB -e correct_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

PASS=0
FAIL=0

for np in 1 2 4; do
    for ov in 0 1; do
        case $np in
            1) map="" ;;
            2) map="--map-by ppr:1:node" ;;
            4) map="--map-by ppr:2:node" ;;
        esac

        RES=rmat-12_${np}p_ov${ov}.res
        echo ""
        echo "============ rmat-12 / np=${np} / OVERLAP=${ov} ============"

        BC_USE_OVERLAP=$ov mpiexec -n $np $map \
            ./solution_mpi -in rmat-12 -out $RES

        # Validate output against reference answer
        echo "--- validation result ---"
        VAL_OUT=$(./validation -ans rmat-12.ans -res $RES 2>&1)
        echo "$VAL_OUT"

        if echo "$VAL_OUT" | grep -q "Accepted"; then
            echo ">>> VERDICT: ACCEPTED"
            PASS=$((PASS+1))
        else
            echo ">>> VERDICT: FAILED"
            FAIL=$((FAIL+1))
        fi

        rm -f $RES
    done
done

echo ""
echo "================ SUMMARY ================"
echo "Passed: $PASS / 6    Failed: $FAIL / 6"
if [ $FAIL -eq 0 ]; then
    echo "ALL CORRECTNESS CHECKS PASSED"
else
    echo "SOME CHECKS FAILED - investigate before running performance tests"
fi