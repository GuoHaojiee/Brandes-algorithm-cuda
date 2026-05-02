#!/bin/bash
#BSUB -J bc_rcm_correct_2p
#BSUB -W 00:30
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o correctness_rcm_2p_%J.out
#BSUB -e correctness_rcm_2p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

echo "=== RCM 正确性测试：2 进程 / 1 节点，rmat-12 ==="
echo "开始时间：$(date)"

echo ""
echo "--- [1/2] BC_USE_RCM=0（identity 模式，应等价 v3.5）---"
BC_USE_RCM=0 mpiexec -n 2 ./solution_mpi -in rmat-12 -out rmat-12-2p-noRcm.res
./validation -ans rmat-12.ans -res rmat-12-2p-noRcm.res

echo ""
echo "--- [2/2] BC_USE_RCM=1（启用 RCM）---"
BC_USE_RCM=1 mpiexec -n 2 ./solution_mpi -in rmat-12 -out rmat-12-2p-rcm.res
./validation -ans rmat-12.ans -res rmat-12-2p-rcm.res

echo ""
echo "完成时间：$(date)"