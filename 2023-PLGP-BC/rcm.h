/*
 * rcm.h — Reverse Cuthill-McKee 顶点重排序
 *
 * 用法：
 *   RcmReorderer R;
 *   R.apply(G, RcmReorderer::enabled_by_env());   // 重排 or 恒等
 *   bc_gpu_init(R.local_n(), R.local_m(), ..., R.offset(), R.dest(), ...);
 *   // ...在新编号下跑算法...
 *   R.unpermute_result(new_bc, result);            // 写回 framework 期望顺序
 *
 * 内存策略：
 *   - apply() 调用期间峰值 O(n+m)（gather 全图 + perm 全表）
 *   - 返回后仅保留本地 CSR + inv_perm_local_（O(n/p)）
 *   - 主算法循环期间严格 O(n/p) ✓
 *
 * 分区不变：新 global ID k 仍归 VERTEX_OWNER(k, n, nproc) 所有，
 *           即 v0 / local_n 重排前后相同。只有边集（dest 数组）变了。
 */

#ifndef RCM_H_
#define RCM_H_

#include "defs.h"
#include <vector>

class RcmReorderer {
public:
    /* 入口：use_rcm=true 跑 RCM，false 仅复制原 CSR（恒等模式，便于对比实验） */
    void apply(const graph_t* G, bool use_rcm);

    /* 重排后的本地 CSR；algorithm code 应当用这些访问图，而非 G->rowsIndices/endV */
    int local_n() const { return new_local_n_; }
    int local_m() const { return new_local_m_; }
    const int* offset() const { return new_offset_.data(); }
    const int* dest()   const { return new_dest_.data();   }

    /* 把基于新编号的 BC 还原为基于旧编号的本地 BC（写入 framework 期望的 result[]）
     * new_bc[i] : BC of new global ID (v0 + i)
     * old_bc[i] : BC of old global ID (v0 + i)
     * 通过一次 MPI_Alltoallv 完成（恒等模式下退化为 memcpy）
     */
    void unpermute_result(const std::vector<double>& new_bc,
                          double* old_bc) const;

    /* 读环境变量 BC_USE_RCM；为 "1" 即启用 */
    static bool enabled_by_env();

private:
    int n_           = 0;
    int rank_        = 0;
    int nproc_       = 0;
    int v0_          = 0;
    int new_local_n_ = 0;
    int new_local_m_ = 0;
    bool identity_   = true;   /* true：未做 RCM，unpermute 走快速路径 */

    std::vector<int> new_offset_;       /* [new_local_n_+1] */
    std::vector<int> new_dest_;         /* [new_local_m_]，新 global ID */
    std::vector<int> inv_perm_local_;   /* [new_local_n_]，新本地 i 对应的旧 global ID */
};

#endif /* RCM_H_ */
