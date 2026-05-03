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

for np in 1 2 4; do
    for ov in 0 1; do
        case $np in
            1) map="" ;;
            2) map="--map-by ppr:1:node" ;;
            4) map="--map-by ppr:2:node" ;;
        esac
        echo ""
        echo "============ rmat-12 / np=${np} / OVERLAP=${ov} ============"
        BC_USE_OVERLAP=$ov mpiexec -n $np $map \
            ./solution_mpi -in rmat-12 -out /tmp/r12_${np}p_ov${ov}.res
    done
done