
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256  
#define BATCH_SIZE 64   

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA 错误 [%s:%d]: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)


__global__ void bfs_kernel(
    int              n,
    const int* __restrict__ offset,     
    const int* __restrict__ dest,       
    const int* __restrict__ sources,    
    int              batch_size,
    int*             d,                 
    long long*       sigma,            
    int*             stack,            
    int*             stack_size        
) {
    int src_idx = blockIdx.x;
    if (src_idx >= batch_size) return;

    int src = sources[src_idx];
    int tid = threadIdx.x;

    int*       my_d     = d     + (long long)src_idx * n;
    long long* my_sigma = sigma + (long long)src_idx * n;
    int*       my_stack = stack + (long long)src_idx * n;
    int*       my_ss    = stack_size + src_idx;

    for (int v = tid; v < n; v += BLOCK_SIZE) {
        my_d[v]     = -1;
        my_sigma[v] = 0LL;
    }
    __syncthreads();

    if (tid == 0) {
        my_d[src]     = 0;
        my_sigma[src] = 1LL;
        my_stack[0]   = src;
        *my_ss        = 1;
    }
    __syncthreads();

    __shared__ int lvl_start, lvl_end;
    if (tid == 0) { lvl_start = 0; lvl_end = 1; }
    __syncthreads();

    while (lvl_start < lvl_end) {
        
        for (int i = lvl_start + tid; i < lvl_end; i += BLOCK_SIZE) {
            int v   = my_stack[i];
            int lvl = my_d[v];
            for (int e = offset[v]; e < offset[v + 1]; e++) {
                int w = dest[e];
                if (atomicCAS(&my_d[w], -1, lvl + 1) == -1) {
                    int pos = atomicAdd(my_ss, 1);
                    my_stack[pos] = w;
                }
            }
        }
        __syncthreads();

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

     
        if (tid == 0) {
            lvl_start = lvl_end;
            lvl_end   = *my_ss;
        }
        __syncthreads();
    }
}


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
    double*          delta,     
    double*          bc_out     
) {
    int src_idx = blockIdx.x;
    if (src_idx >= batch_size) return;

    int src = sources[src_idx];
    int tid = threadIdx.x;

    const int*       my_d     = d     + (long long)src_idx * n;
    const long long* my_sigma = sigma + (long long)src_idx * n;
    int              my_ss    = stack_size[src_idx];
    double*          my_delta = delta + (long long)src_idx * n;


    for (int v = tid; v < n; v += BLOCK_SIZE)
        my_delta[v] = 0.0;
    __syncthreads();


    __shared__ int max_lvl;
    if (tid == 0)
        max_lvl = (my_ss > 1) ? my_d[stack[(long long)src_idx * n + my_ss - 1]] : 0;
    __syncthreads();


    for (int lvl = max_lvl; lvl >= 1; lvl--) {

        for (int v = tid; v < n; v += BLOCK_SIZE) {
            if (my_d[v] == lvl && my_sigma[v] > 0LL) {
                double coeff = (1.0 + my_delta[v]) / (double)my_sigma[v];
                for (int e = offset[v]; e < offset[v + 1]; e++) {
                    int u = dest[e];
                    if (my_d[u] == lvl - 1) {
                    
                        atomicAdd(&my_delta[u], (double)my_sigma[u] * coeff);
                    }
                }
            }
        }
        __syncthreads();

        for (int v = tid; v < n; v += BLOCK_SIZE) {
            if (my_d[v] == lvl && v != src) {
                atomicAdd(&bc_out[v], my_delta[v]);
            }
        }
        __syncthreads();
    }
}

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


    int *d_offset, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_offset, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dest,   (size_t)total_edges * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_offset, offset, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dest,   dest,   (size_t)total_edges * sizeof(int),
                          cudaMemcpyHostToDevice));


    double *d_bc;
    CUDA_CHECK(cudaMalloc(&d_bc, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_bc, 0, (size_t)n * sizeof(double)));


    int *d_sources;
    CUDA_CHECK(cudaMalloc(&d_sources, (size_t)BATCH_SIZE * sizeof(int)));


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


    for (int batch_start = 0; batch_start < n_src; batch_start += BATCH_SIZE) {
        int batch_sz = BATCH_SIZE;
        if (batch_start + batch_sz > n_src)
            batch_sz = n_src - batch_start;


        CUDA_CHECK(cudaMemcpy(d_sources, sources + batch_start,
                              (size_t)batch_sz * sizeof(int),
                              cudaMemcpyHostToDevice));

        bfs_kernel<<<batch_sz, BLOCK_SIZE>>>(
            n, d_offset, d_dest, d_sources, batch_sz,
            d_d, d_sigma, d_stack, d_stack_size
        );
        CUDA_CHECK(cudaGetLastError());

        back_prop_kernel<<<batch_sz, BLOCK_SIZE>>>(
            n, d_offset, d_dest, d_sources, batch_sz,
            d_d, d_sigma, d_stack, d_stack_size,
            d_delta, d_bc
        );
        CUDA_CHECK(cudaGetLastError());
    }


    CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(cudaMemcpy(bc_out, d_bc, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));

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
