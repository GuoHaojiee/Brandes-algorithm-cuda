#!/bin/bash
#BSUB -J bc_overlap_perf_4p
#BSUB -W 00:30
#BSUB -n 40
#BSUB -R "span[ptile=20]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -o perf_overlap_4p_%J.out
#BSUB -e perf_overlap_4p_%J.err

# 关键 LSF 设置说明：
#   -n 40 + ptile=20 + -m "c3 c4"  →  独占 c3、c4 两个节点（防节点争用）
#   实际只用 4 进程，剩下 36 slot 浪费但保证独占
#   --map-by ppr:2:node           →  强制 2 进程/节点分布（跨节点通信走 IB）

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

GRAPH=rmat-14
NREPS=2

echo "============================================================"
echo "  OVERLAP 性能对比：4 进程 / 2 独占节点 / $GRAPH / 各 $NREPS 次"
echo "============================================================"
echo "开始时间：$(date)"
echo ""

for mode in 0 1; do
    if [ "$mode" = "0" ]; then
        LABEL="DISABLED (blocking, v3.5 baseline)"
    else
        LABEL="ENABLED (MPI_Ialltoallv overlap)"
    fi

    echo "############################################################"
    echo "###  BC_USE_OVERLAP=$mode  —  $LABEL"
    echo "############################################################"

    for run in $(seq 1 $NREPS); do
        echo ""
        echo "------ run $run / $NREPS ------"
        BC_USE_OVERLAP=$mode mpiexec --map-by ppr:2:node -n 4 \
            ./solution_mpi -in $GRAPH -out ${GRAPH}-m${mode}-r${run}.res
        ./validation -ans ${GRAPH}.ans -res ${GRAPH}-m${mode}-r${run}.res
    done
    echo ""
done

echo ""
echo "完成时间：$(date)"
echo ""
echo "提取计算时间汇总（grep 行号便于人工核对）："
echo "--- mode=0 (baseline) ---"
grep -n "计算时间" perf_overlap_4p_${LSB_JOBID}.out 2>/dev/null | head -$NREPS
echo "--- mode=1 (overlap) ---"
grep -n "计算时间" perf_overlap_4p_${LSB_JOBID}.out 2>/dev/null | tail -$NREPS
