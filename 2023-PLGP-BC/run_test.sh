#!/bin/bash
#BSUB -J bc_ans_check
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -o ans_check_%J.out
#BSUB -e ans_check_%J.err

cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

echo "============================================================"
echo "  rmat-14.ans 完整性诊断"
echo "============================================================"
echo "开始时间：$(date)"

# === 步骤 0：备份现有 .ans 文件 ===
echo ""
echo "--- [0] 备份现有 rmat-14.ans ---"
if [ -f rmat-14.ans ]; then
    cp rmat-14.ans rmat-14.ans.OLD
    ls -la rmat-14.ans rmat-14.ans.OLD
    md5sum rmat-14.ans rmat-14.ans.OLD 2>/dev/null
fi

# === 步骤 1：用 framework 自带工具重新生成参考答案 ===
echo ""
echo "--- [1] 用 gen_valid_info 重新生成 rmat-14.ans ---"
echo "（这是 framework 的串行参考 BFS，跑出来的就是 ground truth）"
time ./gen_valid_info -in rmat-14 -out rmat-14.ans
echo ""
echo "新生成的 .ans 文件大小："
ls -la rmat-14.ans

# === 步骤 2：与旧 .ans 对比 ===
echo ""
echo "--- [2] 与旧 .ans 对比 ---"
if [ -f rmat-14.ans.OLD ]; then
    if cmp -s rmat-14.ans rmat-14.ans.OLD; then
        echo "✓ 新旧 .ans 文件完全一致 → 旧 .ans 没问题"
    else
        echo "✗ 新旧 .ans 不一致！"
        echo "   说明旧 .ans 有问题，bug 不在我们代码里。"
        echo "   下面用新 .ans 重测："
    fi
fi

# === 步骤 3：用新 .ans 重测我们的程序 ===
echo ""
echo "--- [3] 用新生成的 .ans 测 np=2（同节点）---"
BC_USE_OVERLAP=0 mpiexec -n 2 ./solution_mpi \
    -in rmat-14 -out rmat-14-test-2p.res
./validation -ans rmat-14.ans -res rmat-14-test-2p.res

echo ""
echo "--- [4] 用新生成的 .ans 测 np=4（跨节点）---"
BC_USE_OVERLAP=0 mpiexec --map-by ppr:2:node -n 4 ./solution_mpi \
    -in rmat-14 -out rmat-14-test-4p.res
./validation -ans rmat-14.ans -res rmat-14-test-4p.res

# === 步骤 5：单进程 sanity（如果支持 np=1）===
echo ""
echo "--- [5] 单进程测试（最强 sanity check：无 MPI 通信）---"
BC_USE_OVERLAP=0 mpiexec -n 1 ./solution_mpi \
    -in rmat-14 -out rmat-14-test-1p.res
./validation -ans rmat-14.ans -res rmat-14-test-1p.res

echo ""
echo "完成时间：$(date)"
echo ""
echo "===== 解读 ====="
echo "  - 步骤 [2] 不一致 → bug 是旧 ans 文件，重新生成后用新 ans"
echo "  - 步骤 [5] np=1 Accepted、np>1 Wrong → 算法在多进程时 bug"
echo "  - 步骤 [5] np=1 Wrong → 算法核心或 ans 都可能错，进一步分析"
