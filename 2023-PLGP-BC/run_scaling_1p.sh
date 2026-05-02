#!/bin/bash
#BSUB -J bc_scale_1p_r14
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=1:mode=shared"
#BSUB -o scaling_1p_r14_%J.out
#BSUB -e scaling_1p_r14_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

NPROC=1
GRAPH=rmat-14
echo "=== 强扩展性：nproc=${NPROC} / 1 节点 / 1 GPU ==="
echo "图：${GRAPH}，开始：$(date)"

for i in 1 2 3; do
    echo "--- Run $i ---"
    mpiexec -n ${NPROC} ./solution_mpi -in ${GRAPH} -out ${GRAPH}-${NPROC}p-r${i}.res
done

echo "SCALING nproc=${NPROC} graph=${GRAPH} done at $(date)"