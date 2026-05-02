#!/bin/bash
#BSUB -J bc_rcm_perf_2p
#BSUB -W 00:30
#BSUB -n 40
#BSUB -R "span[ptile=20]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o perf_rcm_2p_%J.out
#BSUB -e perf_rcm_2p_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

echo "=== RCM 性能对比：2 进程 / 1 节点，rmat-12（独占）==="
echo "开始：$(date)"

for run in 1 2 3; do
  echo ""
  echo "============ Run $run ============"
  echo "--- identity ---"
  BC_USE_RCM=0 mpiexec -n 2 ./solution_mpi -in rmat-12 -out rmat-12-id-r${run}.res
  echo "--- RCM ---"
  BC_USE_RCM=1 mpiexec -n 2 ./solution_mpi -in rmat-12 -out rmat-12-rcm-r${run}.res
done

echo ""
echo "完成：$(date)"