/*
 * brandes_gpu.cu
 * GPU 加速的 Brandes 介数中心性算法（第一阶段：正确性优先）
 *
 * 架构目标：NVIDIA Tesla P100，-arch=sm_60
 *
 * 算法结构：
 *   bfs_kernel      — 正向 BFS，计算距离 d[] 和最短路径数 sigma[]
 *   back_prop_kernel — 反向传播，累积依存值 delta[]，汇总到 bc_out[]
 *   computeBC_gpu   — 主入口，管理内存与批次调度
 *
 * 显存占用（BATCH_SIZE=64，n 个顶点）：
 *   图 CSR     : (n+1 + total_edges) * 4 字节（常驻）
 *   d[]        : n * 64 * 4 字节
 *   sigma[]    : n * 64 * 8 字节
 *   delta[]    : n * 64 * 8 字节
 *   stack[]    : n * 64 * 4 字节
 *   对 n=4096  : 约 20MB，对 n=1M：约 1.5GB，P100 16GB 均可容纳
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256   /* 每个 block 的线程数 */
#define BATCH_SIZE 64    /* 每批处理的源节点数，控制显存占用 */

/* CUDA 错误检查宏 */
#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA 错误 [%s:%d]: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

/*
 * 正向 BFS kernel
 * 每个 block 处理一个源节点，block 内 BLOCK_SIZE 线程并行处理出边。
 * 输出：
 *   d[src_idx*n .. src_idx*n+n-1]     — 到源的 BFS 距离，-1 表示不可达
 *   sigma[src_idx*n .. ]              — 最短路径数
 *   stack[src_idx*n .. ]              — 按 BFS 访问顺序排列的顶点
 *   stack_size[src_idx]               — 实际入队顶点数
 */
__global__ void bfs_kernel(
    int              n,
    const int* __restrict__ offset,     /* CSR 偏移，大小 n+1 */
    const int* __restrict__ dest,       /* CSR 目标，大小 total_edges */
    const int* __restrict__ sources,    /* 本批源节点全局编号 */
    int              batch_size,
    int*             d,                 /* 输出：BFS 距离，batch_size*n */
    long long*       sigma,             /* 输出：最短路径数，batch_size*n */
    int*             stack,             /* 输出：BFS 顺序，batch_size*n */
    int*             stack_size         /* 输出：入队数，batch_size */
) {
    int src_idx = blockIdx.x;
    if (src_idx >= batch_size) return;

    int src = sources[src_idx];
    int tid = threadIdx.x;

    /* 本源节点对应的数组基地址 */
    int*       my_d     = d     + (long long)src_idx * n;
    long long* my_sigma = sigma + (long long)src_idx * n;
    int*       my_stack = stack + (long long)src_idx * n;
    int*       my_ss    = stack_size + src_idx;

    /* 初始化：全部未访问，最短路径数为 0 */
    for (int v = tid; v < n; v += BLOCK_SIZE) {
        my_d[v]     = -1;
        my_sigma[v] = 0LL;
    }
    __syncthreads();

    /* 初始化源节点 */
    if (tid == 0) {
        my_d[src]     = 0;
        my_sigma[src] = 1LL;
        my_stack[0]   = src;
        *my_ss        = 1;
    }
    __syncthreads();

    /* 层同步 BFS：共享变量追踪当前层的队列范围 */
    __shared__ int lvl_start, lvl_end;
    if (tid == 0) { lvl_start = 0; lvl_end = 1; }
    __syncthreads();

    while (lvl_start < lvl_end) {
        /*
         * 第一遍：发现下一层新顶点
         * 用 atomicCAS 保证每个顶点只被入队一次
         */
        for (int i = lvl_start + tid; i < lvl_end; i += BLOCK_SIZE) {
            int v   = my_stack[i];
            int lvl = my_d[v];
            for (int e = offset[v]; e < offset[v + 1]; e++) {
                int w = dest[e];
                if (atomicCAS(&my_d[w], -1, lvl + 1) == -1) {
                    /* w 首次被发现，加入队列 */
                    int pos = atomicAdd(my_ss, 1);
                    my_stack[pos] = w;
                }
            }
        }
        __syncthreads();

        /*
         * 第二遍：累加最短路径数 sigma
         * 必须在第一遍结束后执行，确保 d[w] 已全部设置
         * 对于当前层顶点 v，将 sigma[v] 累加到其下一层邻居 w
         */
        for (int i = lvl_start + tid; i < lvl_end; i += BLOCK_SIZE) {
            int v   = my_stack[i];
            int lvl = my_d[v];
            for (int e = offset[v]; e < offset[v + 1]; e++) {
                int w = dest[e];
                if (my_d[w] == lvl + 1) {
                    atomicAdd((unsigned long long*)&my_sigma[w],
                              (unsigned long long)my_sigma[v]);
                }
            }
        }
        __syncthreads();

        /* 推进到下一层 */
        if (tid == 0) {
            lvl_start = lvl_end;
            lvl_end   = *my_ss;
        }
        __syncthreads();
    }
}

/*
 * 反向传播 kernel（Brandes 算法第二阶段）
 * 每个 block 处理一个源节点。
 * 按 BFS 层逆序（从叶到根）计算依存值 delta，累加到全局 bc_out。
 *
 * Brandes 公式：对每个非源顶点 w（按距离从大到小处理）：
 *   delta[u] += sigma[u] / sigma[w] * (1 + delta[w])  对所有前驱 u
 *   bc_out[w] += delta[w]
 * 其中前驱 u 满足：d[u] == d[w]-1 且 (u,w) 是边
 */
__global__ void back_prop_kernel(
    int              n,
    const int* __restrict__ offset,
    const int* __restrict__ dest,
    const int* __restrict__ sources,
    int              batch_size,
    const int*       d,
    const long long* sigma,
    const int*       stack,
    const int*       stack_size,
    double*          delta,     /* 工作缓冲区：依存值，batch_size*n */
    double*          bc_out     /* 输出：BC 累加，大小 n，跨批次保留 */
) {
    int src_idx = blockIdx.x;
    if (src_idx >= batch_size) return;

    int src = sources[src_idx];
    int tid = threadIdx.x;

    const int*       my_d     = d     + (long long)src_idx * n;
    const long long* my_sigma = sigma + (long long)src_idx * n;
    int              my_ss    = stack_size[src_idx];
    double*          my_delta = delta + (long long)src_idx * n;

    /* 初始化依存值 */
    for (int v = tid; v < n; v += BLOCK_SIZE)
        my_delta[v] = 0.0;
    __syncthreads();

    /*
     * 找最大 BFS 层号
     * stack 中最后一个元素在最深层（BFS 按层顺序入队）
     */
    __shared__ int max_lvl;
    if (tid == 0)
        max_lvl = (my_ss > 1) ? my_d[stack[(long long)src_idx * n + my_ss - 1]] : 0;
    __syncthreads();

    /*
     * 按层逆序传播 delta（从最深层到第 1 层）
     * 每层：
     *   1) 所有该层顶点 v 将 delta 贡献给前驱 u（原子加，处理并发写）
     *   2) 所有该层顶点 v 将 delta[v] 汇入 bc_out[v]
     * __syncthreads() 保证层间顺序正确
     */
    for (int lvl = max_lvl; lvl >= 1; lvl--) {

        /* 传播：v 的前驱 u 获得 sigma[u]/sigma[v]*(1+delta[v]) 的贡献 */
        for (int v = tid; v < n; v += BLOCK_SIZE) {
            if (my_d[v] == lvl && my_sigma[v] > 0LL) {
                double coeff = (1.0 + my_delta[v]) / (double)my_sigma[v];
                for (int e = offset[v]; e < offset[v + 1]; e++) {
                    int u = dest[e];
                    if (my_d[u] == lvl - 1) {
                        /* 原子加：同层多个 v 可能共享同一前驱 u */
                        atomicAdd(&my_delta[u], (double)my_sigma[u] * coeff);
                    }
                }
            }
        }
        __syncthreads();

        /* 将本层 delta 汇入全局 BC（跨 block 并发，需原子加） */
        for (int v = tid; v < n; v += BLOCK_SIZE) {
            if (my_d[v] == lvl && v != src) {
                atomicAdd(&bc_out[v], my_delta[v]);
            }
        }
        __syncthreads();
    }
}

/*
 * computeBC_gpu — GPU Brandes BC 计算主入口
 *
 * 参数：
 *   n           — 全局顶点数
 *   offset      — 主机端 CSR 偏移（int），大小 n+1
 *   dest        — 主机端 CSR 目标（int，全局顶点编号），大小 total_edges
 *   total_edges — dest 数组长度
 *   sources     — 本组源节点全局编号，大小 n_src
 *   n_src       — 源节点数量
 *   bc_out      — 主机端输出数组，大小 n；函数返回后为部分 BC（仅本组源节点贡献）
 *   gpu_device  — 绑定的 GPU 编号（0 或 1）
 */
extern "C" void computeBC_gpu(
    int     n,
    int*    offset,
    int*    dest,
    int     total_edges,
    int*    sources,
    int     n_src,
    double* bc_out,
    int     gpu_device
) {
    CUDA_CHECK(cudaSetDevice(gpu_device));

    /* 上传完整图 CSR 到 GPU（在整个计算过程中常驻显存） */
    int *d_offset, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_offset, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dest,   (size_t)total_edges * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_offset, offset, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dest,   dest,   (size_t)total_edges * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* BC 累加数组，跨批次保留，初始化为 0 */
    double *d_bc;
    CUDA_CHECK(cudaMalloc(&d_bc, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_bc, 0, (size_t)n * sizeof(double)));

    /* 源节点缓冲（每批复用） */
    int *d_sources;
    CUDA_CHECK(cudaMalloc(&d_sources, (size_t)BATCH_SIZE * sizeof(int)));

    /* 每批工作数组（BATCH_SIZE 个源节点，每个 n 个元素） */
    long long batch_n = (long long)BATCH_SIZE * n;
    int*       d_d;
    long long* d_sigma;
    double*    d_delta;
    int*       d_stack;
    int*       d_stack_size;
    CUDA_CHECK(cudaMalloc(&d_d,          batch_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sigma,      batch_n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_delta,      batch_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_stack,      batch_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stack_size, (size_t)BATCH_SIZE * sizeof(int)));

    /* 分批处理所有源节点 */
    for (int batch_start = 0; batch_start < n_src; batch_start += BATCH_SIZE) {
        int batch_sz = BATCH_SIZE;
        if (batch_start + batch_sz > n_src)
            batch_sz = n_src - batch_start;

        /* 上传本批源节点编号 */
        CUDA_CHECK(cudaMemcpy(d_sources, sources + batch_start,
                              (size_t)batch_sz * sizeof(int),
                              cudaMemcpyHostToDevice));

        /* 正向 BFS：batch_sz 个 block，每 block 处理一个源节点 */
        bfs_kernel<<<batch_sz, BLOCK_SIZE>>>(
            n, d_offset, d_dest, d_sources, batch_sz,
            d_d, d_sigma, d_stack, d_stack_size
        );
        CUDA_CHECK(cudaGetLastError());

        /* 反向传播：同样 batch_sz 个 block */
        back_prop_kernel<<<batch_sz, BLOCK_SIZE>>>(
            n, d_offset, d_dest, d_sources, batch_sz,
            d_d, d_sigma, d_stack, d_stack_size,
            d_delta, d_bc
        );
        CUDA_CHECK(cudaGetLastError());
    }

    /* 等待所有 kernel 完成 */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 将 BC 结果拷贝回主机 */
    CUDA_CHECK(cudaMemcpy(bc_out, d_bc, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    /* 释放显存 */
    cudaFree(d_offset);
    cudaFree(d_dest);
    cudaFree(d_bc);
    cudaFree(d_sources);
    cudaFree(d_d);
    cudaFree(d_sigma);
    cudaFree(d_delta);
    cudaFree(d_stack);
    cudaFree(d_stack_size);
}
