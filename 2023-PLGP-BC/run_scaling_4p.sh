#!/bin/bash
#BSUB -J bc_big_all
#BSUB -W 01:30
#BSUB -q normal
#BSUB -n 40
#BSUB -R "span[ptile=20]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o big_all_%J.out
#BSUB -e big_all_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

# ============ 1 进程（单节点） ============
for graph in rmat-14 random-14; do
    echo ""
    echo "############# 图：${graph} (nproc=1) #############"
    if [ ! -f "${graph}" ]; then continue; fi
    mpiexec -n 1 ./solution_mpi -in ${graph} -out ${graph}-1p.res
done

# ============ 2 进程（单节点） ============
for graph in rmat-14 random-14; do
    echo ""
    echo "############# 图：${graph} (nproc=2) #############"
    if [ ! -f "${graph}" ]; then continue; fi
    mpiexec -n 2 ./solution_mpi -in ${graph} -out ${graph}-2p.res
done

# ============ 4 进程（2 节点，每节点 2 进程） ============
for graph in rmat-14 random-14; do
    echo ""
    echo "############# 图：${graph} (nproc=4) #############"
    if [ ! -f "${graph}" ]; then continue; fi
    mpiexec -n 4 --map-by ppr:2:node ./solution_mpi -in ${graph} -out ${graph}-4p.res
done