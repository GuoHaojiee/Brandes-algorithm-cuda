#!/bin/bash
#BSUB -J bc_scale_6p
#BSUB -W 00:30
#BSUB -n 6
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_6p_%J.out
#BSUB -e scaling_6p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

NPROC=6
echo "=== 强可扩展性：nproc=${NPROC} / 3 节点 ==="
echo "图：rmat-14，开始：$(date)"

for i in 1 2 3; do
    echo "--- Run $i ---"
    mpiexec -n ${NPROC} ./solution_mpi -in rmat-12 -out rmat-12-${NPROC}p-r${i}.res
done

echo "SCALING nproc=${NPROC} graph=rmat-14 done at $(date)"