#!/bin/bash
#BSUB -J bc_big_2p_1node
#BSUB -W 01:00
#BSUB -q normal 
#BSUB -n 2
#BSUB -m "polus-c3-ib"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o big_2p_1n_%J.out
#BSUB -e big_2p_1n_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

for graph in rmat-14 random-14; do
    echo "############# 图：${graph} (nproc=2) #############"
    for i in {1..3}; do
        echo "--- Run #$i ---"
        # 使用单节点内的 2 个 GPU
        mpiexec -n 2 ./solution_mpi -in ${graph} -out ${graph}-2p.res
    done
done