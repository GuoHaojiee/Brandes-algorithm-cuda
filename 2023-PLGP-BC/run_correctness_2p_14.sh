#!/bin/bash
#BSUB -J bc_rcm_perf_4p
#BSUB -W 00:40
#BSUB -n 40
#BSUB -R "span[ptile=20]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o perf_rcm_4p_%J.out
#BSUB -e perf_rcm_4p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

echo "=== RCM 性能对比：4 进程 / 2 节点 / rmat-14 ==="
echo "节点：$(hostname)，开始：$(date)"

for run in 1 2 3; do
  echo ""
  echo "============ Run $run ============"
  echo "--- identity ---"
  BC_USE_RCM=0 mpiexec -n 4 ./solution_mpi -in rmat-14 -out /tmp/r${run}_id.res
  echo "--- RCM ---"
  BC_USE_RCM=1 mpiexec -n 4 ./solution_mpi -in rmat-14 -out /tmp/r${run}_rcm.res
done

echo ""
echo "完成：$(date)"