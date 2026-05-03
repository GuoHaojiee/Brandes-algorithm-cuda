#!/bin/bash
#BSUB -J bc_diagnose
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -o diagnose_%J.out
#BSUB -e diagnose_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

# 全部用 OVERLAP=0（baseline，等价 v3.5），排除重叠优化的影响
# 目标：找到 Wrong Answer 的触发边界
export BC_USE_OVERLAP=0

run_test() {
    local graph=$1
    local np=$2
    local layout=$3   # "1node" or "2nodes"

    echo ""
    echo "============================================================"
    echo "  GRAPH=$graph  NP=$np  layout=$layout"
    echo "============================================================"

    if [ "$layout" = "2nodes" ]; then
        MAP_FLAG="--map-by ppr:$((np/2)):node"
    else
        MAP_FLAG=""
    fi

    mpiexec $MAP_FLAG -n $np ./solution_mpi \
        -in $graph -out diag-${graph}-${np}p.res
    ./validation -ans ${graph}.ans -res diag-${graph}-${np}p.res
}

echo "诊断 Brandes BC v3.7 OVERLAP=0 (= v3.5 等价) 在不同配置下的正确性"
echo "开始时间：$(date)"

# 实验 1: 已知工作的基线（应 Accepted）
run_test rmat-12 2 "1node"

# 实验 2: 同图改 4 进程跨节点 → 隔离"进程数/跨节点"维度
run_test rmat-12 4 "2nodes"

# 实验 3: rmat-13 / 4 进程 → 中等规模
run_test rmat-13 4 "2nodes"

# 实验 4: rmat-14 / 2 进程 → 大图但 2 进程
run_test rmat-14 2 "1node"

# 实验 5: rmat-14 / 4 进程跨节点 → 已知失败的基线
run_test rmat-14 4 "2nodes"

echo ""
echo "完成时间：$(date)"
echo ""
echo "===== 结果总结：检查 Accepted/Wrong answer ====="
grep -E "(GRAPH=|Accepted|Wrong)" diagnose_${LSB_JOBID}.out 2>/dev/null
