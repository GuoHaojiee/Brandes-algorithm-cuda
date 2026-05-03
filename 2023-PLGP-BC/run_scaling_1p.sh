#!/bin/bash
#BSUB -J bc_big_1p
#BSUB -W 01:00
#BSUB -q normal 
#BSUB -n 1
#BSUB -m "polus-c3-ib"
#BSUB -gpu "num=1:mode=shared"
#BSUB -o big_1p_%J.out
#BSUB -e big_1p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

for graph in rmat-14 random-14; do
    echo "############# 图：${graph} (nproc=1) #############"
    for i in {1..3}; do
        echo "--- Run #$i ---"
        mpiexec -n 1 ./solution_mpi -in ${graph} -out ${graph}-1p.res
    done
done