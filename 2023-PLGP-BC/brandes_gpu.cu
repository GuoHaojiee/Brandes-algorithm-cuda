/*
 * brandes_gpu.cu — GPU 辅助：本地 CSR 上传 + 批量 frontier 边扩展
 *
 * bc_gpu_expand_batch：一次调用处理 batch_sz 个 BFS frontier，
 * 每个线程负责合并 frontier 中的一个顶点，输出 (b, v_global, w_global, sigma_v)。
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

/* ---- 进程级 GPU 状态 ---- */
static int        s_local_n     = 0;
static int        s_local_m     = 0;
static int        s_batch_size  = 0;
static int*       s_d_offset    = nullptr;   /* 本地 CSR offset[local_n+1] */
static int*       s_d_dest      = nullptr;   /* 本地 CSR dest[local_m] */
static int*       s_d_all_front = nullptr;   /* 合并 frontier [local_n] */
static int*       s_d_front_off = nullptr;   /* frontier 分段偏移 [batch_size+1] */
static long long* s_d_front_sig = nullptr;   /* frontier 顶点 sigma [local_n] */
static int*       s_d_out_b     = nullptr;   /* 输出：batch_id */
static int*       s_d_out_src   = nullptr;   /* 输出：v_global */
static int*       s_d_out_dst   = nullptr;   /* 输出：w_global */
static long long* s_d_out_sig   = nullptr;   /* 输出：sigma_v */
static int*       s_d_cnt       = nullptr;   /* 原子计数器 */

/*
 * expand_batch_kernel
 * 每线程处理合并 frontier 中的一个位置 fi。
 * 通过二分查找 front_offsets 确定 fi 属于第 b 个源节点的 BFS。
 * 对顶点 v_lc 的每条出边输出一条记录 (b, v_global, w_global, sigma_v)。
 */
__global__ void expand_batch_kernel(
    const int*       all_front,      /* [total_fz] 合并后的 frontier 本地 ID */
    const int*       front_offsets,  /* [batch_sz+1] 各 BFS 在 all_front 中的起止 */
    const long long* front_sigma,    /* [total_fz] 与 all_front 一一对应的 sigma */
    int              batch_sz,
    int              total_fz,
    const int*       offset,         /* 本地 CSR offset */
    const int*       dest,           /* 本地 CSR dest（全局顶点 ID）*/
    int              v0_global,
    int*             out_b,
    int*             out_src,
    int*             out_dst,
    long long*       out_sig,
    int*             cnt
)
{
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= total_fz) return;

    /* 二分查找：找到 fi 所属的 batch slot b */
    int lo = 0, hi = batch_sz - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (front_offsets[mid + 1] <= fi) lo = mid + 1;
        else hi = mid;
    }
    int b = lo;

    int       v_lc  = all_front[fi];
    long long sig_v = front_sigma[fi];
    int       v_gl  = v0_global + v_lc;

    for (int e = offset[v_lc]; e < offset[v_lc + 1]; e++) {
        int pos    = atomicAdd(cnt, 1);
        out_b[pos]   = b;
        out_src[pos] = v_gl;
        out_dst[pos] = dest[e];
        out_sig[pos] = sig_v;
    }
}

/* ----------------------------------------------------------------
 * bc_gpu_init
 * 上传本地 CSR，分配批量 expand 所需缓冲区。每个进程调用一次。
 * ---------------------------------------------------------------- */
extern "C" void bc_gpu_init(
    int        local_n,
    int        local_m,
    int        batch_size,   /* 同时处理的源节点数（= BATCH_SIZE）*/
    const int* local_offset,
    const int* local_dest,
    int        gpu_device
)
{
    CUDA_CHECK(cudaSetDevice(gpu_device));
    s_local_n    = local_n;
    s_local_m    = local_m;
    s_batch_size = batch_size;

    int safe_m  = (local_m > 0) ? local_m : 1;
    int max_out = batch_size * safe_m;   /* 输出缓冲最大条数 */

    CUDA_CHECK(cudaMalloc(&s_d_offset,    (size_t)(local_n + 1)    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_dest,      (size_t)safe_m           * sizeof(int)));
    /* total_fz 最大 = batch_size × local_n，需按批量大小分配 */
    CUDA_CHECK(cudaMalloc(&s_d_all_front, (size_t)batch_size * local_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_front_off, (size_t)(batch_size + 1)     * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&s_d_front_sig, (size_t)batch_size * local_n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_d_out_b,     (size_t)max_out          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_out_src,   (size_t)max_out          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_d_out_dst,   (size_t)max_out          * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&s_d_out_sig, (size_t)max_out    * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_d_cnt,       sizeof(int)));

    CUDA_CHECK(cudaMemcpy(s_d_offset, local_offset,
                          (size_t)(local_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (local_m > 0)
        CUDA_CHECK(cudaMemcpy(s_d_dest, local_dest,
                              (size_t)local_m * sizeof(int), cudaMemcpyHostToDevice));
}

/* ----------------------------------------------------------------
 * bc_gpu_expand_batch
 * 一次处理 batch_sz 个 BFS frontier 的边扩展。
 * 返回输出边数；out_* 缓冲由调用方分配，大小需 >= batch_size * local_m。
 * ---------------------------------------------------------------- */
extern "C" int bc_gpu_expand_batch(
    const int*       all_front,      /* [total_fz] 合并 frontier 本地顶点 ID */
    const int*       front_offsets,  /* [batch_sz+1] 分段偏移 */
    const long long* front_sigma,    /* [total_fz] 对应 sigma */
    int              batch_sz,
    int              total_fz,
    int              v0_global,
    int*             out_b,          /* 调用方缓冲，大小 >= batch_size * local_m */
    int*             out_src,
    int*             out_dst,
    long long*       out_sig
)
{
    if (total_fz <= 0) return 0;

    CUDA_CHECK(cudaMemcpy(s_d_all_front, all_front,
                          (size_t)total_fz    * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_d_front_off, front_offsets,
                          (size_t)(batch_sz+1) * sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_d_front_sig, front_sigma,
                          (size_t)total_fz    * sizeof(long long), cudaMemcpyHostToDevice));

    int zero = 0;
    CUDA_CHECK(cudaMemcpy(s_d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (total_fz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    expand_batch_kernel<<<blocks, BLOCK_SIZE>>>(
        s_d_all_front, s_d_front_off, s_d_front_sig,
        batch_sz, total_fz,
        s_d_offset, s_d_dest, v0_global,
        s_d_out_b, s_d_out_src, s_d_out_dst, s_d_out_sig, s_d_cnt
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, s_d_cnt, sizeof(int), cudaMemcpyDeviceToHost));

    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(out_b,   s_d_out_b,   (size_t)count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_src, s_d_out_src, (size_t)count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_dst, s_d_out_dst, (size_t)count * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_sig, s_d_out_sig, (size_t)count * sizeof(long long),
                              cudaMemcpyDeviceToHost));
    }
    return count;
}

/* ---- 释放 GPU 内存 ---- */
extern "C" void bc_gpu_cleanup(void)
{
    cudaFree(s_d_offset);    cudaFree(s_d_dest);
    cudaFree(s_d_all_front); cudaFree(s_d_front_off);
    cudaFree(s_d_front_sig);
    cudaFree(s_d_out_b);     cudaFree(s_d_out_src);
    cudaFree(s_d_out_dst);   cudaFree(s_d_out_sig);
    cudaFree(s_d_cnt);
    s_d_offset = s_d_dest = s_d_all_front = s_d_front_off = nullptr;
    s_d_front_sig = nullptr;
    s_d_out_b = s_d_out_src = s_d_out_dst = s_d_cnt = nullptr;
    s_d_out_sig = nullptr;
}
