#!/bin/bash
#BSUB -J bc_scale_4p
#BSUB -W 01:00
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_4p_%J.out
#BSUB -e scaling_4p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

NPROC=4
echo "=== 强可扩展性：nproc=${NPROC} / 2 节点 ==="
echo "图：rmat-14，开始：$(date)"

for i in 1 2 3; do
    echo "--- Run $i ---"
    mpiexec -n ${NPROC} ./solution_mpi -in rmat-14 -out rmat-14-${NPROC}p-r${i}.res
done

echo "SCALING nproc=${NPROC} graph=rmat-14 done at $(date)"