/*
 * brandes_gpu.cu — GPU 辅助：本地 CSR 上传 + frontier 边扩展
 *
 * 每个 MPI 进程只把自己的本地子图（O(n/p) 顶点，O(m/p) 边）上传到 GPU。
 * bc_gpu_expand 在每次 BFS 层推进时被调用，GPU 并行枚举 frontier 顶点的出边，
 * 结果返回给 CPU，由 CPU + MPI 完成跨进程通信。
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误 [%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* ---- 进程级 GPU 状态（bc_gpu_init 后保持直到 bc_gpu_cleanup） ---- */
static int        s_local_n   = 0;
static int        s_local_m   = 0;
static int*       s_d_offset  = nullptr;   /* 本地 CSR offset[local_n+1] */
static int*       s_d_dest    = nullptr;   /* 本地 CSR dest[local_m]（全局顶点 ID）*/
static int*       s_d_front   = nullptr;   /* 当前层 frontier（local vertex IDs）*/
static long long* s_d_fsigma  = nullptr;   /* frontier 顶点对应的 sigma 值 */
static int*       s_d_out_src = nullptr;   /* 输出：源顶点全局 ID */
static int*       s_d_out_dst = nullptr;   /* 输出：目标顶点全局 ID */
static long long* s_d_out_sig = nullptr;   /* 输出：sigma_v */
static int*       s_d_cnt     = nullptr;   /* 原子输出计数器 */

/*
 * expand_kernel
 * 每个线程负责 frontier 中的一个顶点 v（本地编号）：
 *   对 v 的每条出边 e → w_global，原子地写入输出数组一条记录
 *   (v_global, w_global, sigma_v)
 */
__global__ void expand_kernel(
    const int*       frontier,     /* [fz] 本地顶点 ID */
    const long long* front_sigma,  /* [fz] 对应的 sigma 值 */
    int              fz,
    const int*       offset,       /* 本地 CSR offset */
    const int*       dest,         /* 本地 CSR dest（全局 ID）*/
    int              v0_global,    /* 本进程第一个顶点的全局 ID */
    int*             out_src,
    int*             out_dst,
    long long*       out_sig,
    int*             cnt           /* 原子计数器 */
)
{
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= fz) return;

    int       v_lc  = frontier[fi];
    long long sig_v = front_sigma[fi];
    int       v_gl  = v0_global + v_lc;  /* 顶点连续分配，v_global = v0 + v_local */

    for (int e = offset[v_lc]; e < offset[v_lc + 1]; e++) {
        int pos    = atomicAdd(cnt, 1);
        out_src[pos] = v_gl;
        out_dst[pos] = dest[e];
        out_sig[pos] = sig_v;
    }
}

/* ----------------------------------------------------------------
 * bc_gpu_init
 * 把本地 CSR 上传到 GPU，分配 expand 所需的输出缓冲区。
 * 每个进程调用一次（run() 开始时）。
 * ---------------------------------------------------------------- */
extern "C" void bc_gpu_init(
    int        local_n,
    int        local_m,
    const int* local_offset,
    const int* local_dest,
    int        gpu_device
)
{
    CUDA_CHECK(cudaSetDevice(gpu_device));
    s_local_n = local_n;
    s_local_m = local_m;

    int safe_m = (local_m > 0) ? local_m : 1;  /* 避免 0 大小分配 */

    CUDA_CHECK(cudaMalloc(&s_d_offset,   (size_t)(local_n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_dest,     (size_t)safe_m        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_front,    (size_t)local_n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&s_d_fsigma,  (size_t)local_n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_d_out_src,  (size_t)safe_m        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_out_dst,  (size_t)safe_m        * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&s_d_out_sig,  (size_t)safe_m * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_d_cnt,      sizeof(int)));

    /* 上传本地 CSR（只上传一次，后续 expand 复用）*/
    CUDA_CHECK(cudaMemcpy(s_d_offset, local_offset,
                          (size_t)(local_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (local_m > 0)
        CUDA_CHECK(cudaMemcpy(s_d_dest, local_dest,
                              (size_t)local_m * sizeof(int), cudaMemcpyHostToDevice));
}

/* ----------------------------------------------------------------
 * bc_gpu_expand
 * 对当前 BFS frontier 执行边扩展，返回输出边数。
 *
 * 输出数组 out_src/out_dst/out_sig 由调用方分配，大小需 >= local_m。
 * ---------------------------------------------------------------- */
extern "C" int bc_gpu_expand(
    const int*       frontier,     /* [fz] host 端 frontier 本地顶点 ID */
    const long long* front_sigma,  /* [fz] host 端对应 sigma */
    int              fz,
    int              v0_global,
    int*             out_src,      /* host 端输出缓冲（大小 >= local_m）*/
    int*             out_dst,
    long long*       out_sig
)
{
    if (fz <= 0) return 0;

    /* 上传 frontier 和 sigma（大小均为 fz，远小于 local_n）*/
    CUDA_CHECK(cudaMemcpy(s_d_front,  frontier,    (size_t)fz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_d_fsigma, front_sigma, (size_t)fz * sizeof(long long),
                          cudaMemcpyHostToDevice));

    /* 重置原子计数器 */
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(s_d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (fz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    expand_kernel<<<blocks, BLOCK_SIZE>>>(
        s_d_front, s_d_fsigma, fz,
        s_d_offset, s_d_dest, v0_global,
        s_d_out_src, s_d_out_dst, s_d_out_sig, s_d_cnt
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, s_d_cnt, sizeof(int), cudaMemcpyDeviceToHost));

    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(out_src, s_d_out_src,
                              (size_t)count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_dst, s_d_out_dst,
                              (size_t)count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_sig, s_d_out_sig,
                              (size_t)count * sizeof(long long), cudaMemcpyDeviceToHost));
    }
    return count;
}

/* ---- 释放 GPU 内存 ---- */
extern "C" void bc_gpu_cleanup(void)
{
    cudaFree(s_d_offset);
    cudaFree(s_d_dest);
    cudaFree(s_d_front);
    cudaFree(s_d_fsigma);
    cudaFree(s_d_out_src);
    cudaFree(s_d_out_dst);
    cudaFree(s_d_out_sig);
    cudaFree(s_d_cnt);
    s_d_offset = s_d_dest = s_d_front = nullptr;
    s_d_fsigma = nullptr;
    s_d_out_src = s_d_out_dst = s_d_cnt = nullptr;
    s_d_out_sig = nullptr;
}
