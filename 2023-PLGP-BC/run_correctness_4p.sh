#!/bin/bash
#BSUB -J bc_correct_4p
#BSUB -W 00:20
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o correctness_4p_%J.out
#BSUB -e correctness_4p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

echo "=== 正确性测试：4 进程 / 2 节点 ==="
echo "图：rmat-12，开始时间：$(date)"

mpiexec -n 4 ./solution_mpi -in rmat-12 -out rmat-12-4p.res

echo "--- validation 结果 ---"
./validation -ans rmat-12.ans -res rmat-12-4p.res

echo "完成时间：$(date)"
