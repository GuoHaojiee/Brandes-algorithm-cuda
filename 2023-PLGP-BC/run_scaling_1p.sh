#!/bin/bash
#BSUB -J bc_scale_1p
#BSUB -W 01:00
#BSUB -n 1
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -o scaling_1p_%J.out
#BSUB -e scaling_1p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

NPROC=1
echo "=== 强可扩展性测试：nproc=${NPROC} ==="
echo "图：rmat-14，开始时间：$(date)"

mpiexec -n ${NPROC} ./solution_mpi -in rmat-14 -out rmat-14-${NPROC}p.res

# grep 标记行，方便后续 grep SCALING *.out 汇总结果
echo "SCALING nproc=${NPROC} graph=rmat-14 done at $(date)"

echo "完成时间：$(date)"
