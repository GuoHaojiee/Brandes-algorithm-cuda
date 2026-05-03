#!/bin/bash
#BSUB -J bc_big_4p
#BSUB -W 01:00
#BSUB -q normal 
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -m "polus-c3-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o big_4p_%J.out
#BSUB -e big_4p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

for graph in rmat-14 random-14; do
    echo "############# 图：${graph} (nproc=4) #############"
    for i in {1..3}; do
        echo "--- Run #$i ---"
        mpiexec -n 4 ./solution_mpi -in ${graph} -out ${graph}-4p.res
    done
done