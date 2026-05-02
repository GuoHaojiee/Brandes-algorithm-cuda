/*
 * brandes_gpu.cu — Direction B: GPU-resident d/sigma/delta
 *
 * All per-vertex BFS state (d, sigma, delta) lives on GPU.
 * CPU only handles MPI messages between processes.
 *
 * Key functions:
 *   bc_gpu_init          — allocate GPU state + upload CSR
 *   bc_gpu_reset_batch   — reset d/sigma/delta, plant seeds
 *   bc_gpu_bfs_expand    — expand frontier, update local state atomically,
 *                          return remote messages for MPI
 *   bc_gpu_apply_remote_fwd — apply received BFS discoveries to local state
 *   bc_gpu_collect_next  — scan d[] to collect frontier at a given level
 *   bc_gpu_backprop_level — compute delta for one level, return remote requests
 *   bc_gpu_apply_remote_bp  — apply received delta contributions
 *   bc_gpu_get_delta     — download delta[] to host
 *   bc_gpu_cleanup       — free all GPU memory
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
using std::max;

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA[%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* ---- persistent GPU state ---- */
static int        s_n, s_m, s_bs;     /* local_n, local_m, batch_size */
static int        s_v0, s_vend;       /* local vertex global range [v0, vend) */

static int*       s_off;              /* CSR offset  [local_n+1]         */
static int*       s_dst;              /* CSR dest    [local_m] global IDs */

static int*       s_dd;              /* d[bs*n]                          */
static long long* s_ds;              /* sigma[bs*n]                      */
static double*    s_ddel;            /* delta[bs*n]                      */

/* frontier input buffers (reused each level) */
static int*       s_af;              /* all_front   [bs*n]               */
static int*       s_foff;            /* front_off   [bs+1]               */

/* forward BFS remote output [bs*m] */
static int*       s_fb, *s_fw;      /* b, w_global                      */
static long long* s_fsig;
static int*       s_fcnt;

/* frontier collect per-batch slices [bs*n] */
static int*       s_nv;
static int*       s_ncnt;           /* [bs] per-batch counts            */

/* back_prop remote output [bs*m] */
static int*       s_bb, *s_bu;     /* b, u_global                      */
static double*    s_bc;             /* coeff                            */
static int*       s_bcnt;

/* apply buffers for remote receives [bs*m] — temporary uploads */
static int*       s_ra_b, *s_ra_x;  /* b, w_lc or u_lc                 */
static long long* s_ra_sig;
static double*    s_ra_coeff;

/* ============================================================
 * Kernels
 * ============================================================ */

__global__ void k_reset(int* d, long long* sig, double* del, int tot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tot) { d[i] = -1; sig[i] = 0LL; del[i] = 0.0; }
}

__global__ void k_seed(int* d, long long* sig, int bs, int n, const int* src_lc)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= bs) return;
    int s = src_lc[b];
    if (s >= 0 && s < n) {
        d[b * n + s]   = 0;
        sig[b * n + s] = 1LL;
    }
}

/* Expand frontier: for each fi, iterate CSR of v_lc.
 * Local w: atomicCAS on d[], atomicAdd on sigma[].
 * Remote w: output (b, w_gl, sig_v) for MPI. */
__global__ void k_expand(
    const int* af, const int* foff, int bs, int fz, int lev,
    int n, int v0, int ve,
    const int* coff, const int* cdst,
    int* dd, long long* ds,
    int* ob, int* ow, long long* os, int* cnt)
{
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= fz) return;

    /* binary search: find batch b owning position fi */
    int lo = 0, hi = bs - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (foff[mid + 1] <= fi) lo = mid + 1; else hi = mid;
    }
    int b = lo;
    int vlc = af[fi];
    long long sv = ds[b * n + vlc];

    for (int e = coff[vlc]; e < coff[vlc + 1]; e++) {
        int wg = cdst[e];
        if (wg >= v0 && wg < ve) {
            int wlc = wg - v0;
            int old = atomicCAS(&dd[b * n + wlc], -1, lev + 1);
            if (old == -1 || old == lev + 1)
                atomicAdd(&ds[b * n + wlc], sv);
        } else {
            int pos = atomicAdd(cnt, 1);
            ob[pos] = b; ow[pos] = wg; os[pos] = sv;
        }
    }
}

/* Apply received BFS messages: (b, w_lc, sig_v) at discovery level disc_lev */
__global__ void k_apply_fwd(
    const int* ab, const int* awlc, const long long* asig,
    int cnt, int dlev, int n,
    int* dd, long long* ds)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;
    int b = ab[i], wlc = awlc[i];
    long long sv = asig[i];
    int old = atomicCAS(&dd[b * n + wlc], -1, dlev);
    if (old == -1 || old == dlev)
        atomicAdd(&ds[b * n + wlc], sv);
}

/* Collect vertices at target level into per-batch slices of s_nv.
 * Each batch b uses slice [b*n .. b*n+cnt[b]). */
__global__ void k_collect(
    const int* dd,
    int bs, int n, int tlev,
    int* ov, int* bcnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * n) return;
    int b = idx / n, v = idx % n;
    if (dd[idx] == tlev) {
        int p = atomicAdd(&bcnt[b], 1);
        ov[b * n + p] = v;
    }
}

/* Back_prop for one level:
 * For each local vertex w at level lev (all batches):
 *   coeff = (1 + delta[w]) / sigma[w]
 *   For each CSR neighbor u of w:
 *     local u (d[u]==lev-1): atomicAdd delta[u], sigma[u]*coeff
 *     remote u:              output (b, u_global, coeff) for MPI */
__global__ void k_backprop(
    const int* dd, const long long* ds, double* del,
    int bs, int n, int lev,
    int v0, int ve,
    const int* coff, const int* cdst,
    int* ob, int* ou, double* oc, int* cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * n) return;
    int b = idx / n, w = idx % n;
    if (dd[idx] != lev || ds[idx] == 0LL) return;
    double coeff = (1.0 + del[idx]) / (double)ds[idx];
    for (int e = coff[w]; e < coff[w + 1]; e++) {
        int ug = cdst[e];
        if (ug >= v0 && ug < ve) {
            int ulc = ug - v0;
            if (dd[b * n + ulc] == lev - 1)
                atomicAdd(&del[b * n + ulc],
                          (double)ds[b * n + ulc] * coeff);
        } else {
            int pos = atomicAdd(cnt, 1);
            ob[pos] = b; ou[pos] = ug; oc[pos] = coeff;
        }
    }
}

/* Apply received back_prop contributions: (b, u_lc, coeff).
 * Only applies if d[u] == lev-1 (lev passed as parameter). */
__global__ void k_apply_bp(
    const int* ab, const int* aulc, const double* acoeff,
    int cnt, int lev, int n,
    const int* dd, const long long* ds, double* del)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;
    int b = ab[i], ulc = aulc[i];
    if (dd[b * n + ulc] == lev - 1)
        atomicAdd(&del[b * n + ulc],
                  (double)ds[b * n + ulc] * acoeff[i]);
}

/* ============================================================
 * C API
 * ============================================================ */

extern "C" void bc_gpu_init(
    int local_n, int local_m, int batch_size,
    const int* loff, const int* ldst,
    int v0_global, int gpu_device)
{
    CUDA_CHECK(cudaSetDevice(gpu_device));
    s_n  = local_n;
    s_m  = local_m;
    s_bs = batch_size;
    s_v0   = v0_global;
    s_vend = v0_global + local_n;

    int sm    = max(local_m, 1);
    int state = batch_size * local_n;
    int out   = batch_size * sm;

    CUDA_CHECK(cudaMalloc(&s_off,  (local_n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_dst,  sm             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_dd,   state * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_ds,   state * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_ddel, state * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&s_af,   state * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_foff, (batch_size + 1) * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&s_fb,   out * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_fw,   out * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_fsig, out * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_fcnt, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&s_nv,   state * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_ncnt, batch_size * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&s_bb,   out * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_bu,   out * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_bc,   out * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s_bcnt, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&s_ra_b,     out * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_ra_x,     out * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s_ra_sig,   out * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&s_ra_coeff, out * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(s_off, loff, (local_n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (local_m > 0)
        CUDA_CHECK(cudaMemcpy(s_dst, ldst, local_m * sizeof(int),
                              cudaMemcpyHostToDevice));
}

/* Reset d/sigma/delta; set seeds (src_lc[b] = local ID, or -1 if not local) */
extern "C" void bc_gpu_reset_batch(int batch_sz, const int* src_lc_h)
{
    int tot  = batch_sz * s_n;
    int blk  = (tot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_reset<<<blk, BLOCK_SIZE>>>(s_dd, s_ds, s_ddel, tot);

    int* d_src;
    CUDA_CHECK(cudaMalloc(&d_src, batch_sz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, src_lc_h, batch_sz * sizeof(int),
                          cudaMemcpyHostToDevice));
    int sblk = (batch_sz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_seed<<<sblk, BLOCK_SIZE>>>(s_dd, s_ds, batch_sz, s_n, d_src);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_src);
}

/* Expand frontier; returns remote message count.
 * af_h[total_fz]: local vertex IDs (compact, per batch)
 * foff_h[batch_sz+1]: frontier offsets per batch */
extern "C" int bc_gpu_bfs_expand(
    const int* af_h, const int* foff_h,
    int batch_sz, int total_fz, int cur_lev,
    int* out_b, int* out_wgl, long long* out_sig)
{
    if (total_fz <= 0) return 0;

    CUDA_CHECK(cudaMemcpy(s_af,  af_h,  total_fz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_foff, foff_h, (batch_sz + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(s_fcnt, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blk = (total_fz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_expand<<<blk, BLOCK_SIZE>>>(
        s_af, s_foff, batch_sz, total_fz, cur_lev,
        s_n, s_v0, s_vend,
        s_off, s_dst, s_dd, s_ds,
        s_fb, s_fw, s_fsig, s_fcnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int cnt = 0;
    CUDA_CHECK(cudaMemcpy(&cnt, s_fcnt, sizeof(int), cudaMemcpyDeviceToHost));
    if (cnt > 0) {
        CUDA_CHECK(cudaMemcpy(out_b,   s_fb,   cnt * sizeof(int),       cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_wgl, s_fw,   cnt * sizeof(int),       cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_sig, s_fsig, cnt * sizeof(long long), cudaMemcpyDeviceToHost));
    }
    return cnt;
}

/* Apply received forward BFS messages: (b[], w_lc[], sig[], count, disc_lev) */
extern "C" void bc_gpu_apply_remote_fwd(
    const int* b_h, const int* wlc_h, const long long* sig_h,
    int count, int disc_lev)
{
    if (count <= 0) return;
    CUDA_CHECK(cudaMemcpy(s_ra_b,   b_h,   count * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_ra_x,   wlc_h, count * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_ra_sig, sig_h, count * sizeof(long long), cudaMemcpyHostToDevice));

    int blk = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_apply_fwd<<<blk, BLOCK_SIZE>>>(
        s_ra_b, s_ra_x, s_ra_sig, count, disc_lev, s_n, s_dd, s_ds);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Collect vertices at target_lev into per-batch slices.
 * h_nv[b*local_n .. b*local_n+h_ncnt[b]): local IDs for batch b.
 * Returns total vertex count across all batches. */
extern "C" int bc_gpu_collect_next(
    int batch_sz, int target_lev,
    int* h_nv, int* h_ncnt)
{
    CUDA_CHECK(cudaMemset(s_ncnt, 0, batch_sz * sizeof(int)));

    int tot = batch_sz * s_n;
    int blk = (tot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_collect<<<blk, BLOCK_SIZE>>>(s_dd, batch_sz, s_n, target_lev, s_nv, s_ncnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_ncnt, s_ncnt, batch_sz * sizeof(int), cudaMemcpyDeviceToHost));
    int total = 0;
    for (int b = 0; b < batch_sz; b++) total += h_ncnt[b];
    if (total > 0)
        CUDA_CHECK(cudaMemcpy(h_nv, s_nv, batch_sz * s_n * sizeof(int),
                              cudaMemcpyDeviceToHost));
    return total;
}

/* Compute back_prop for one level; returns remote contribution count.
 * out_b[], out_ugl[], out_coeff[]: remote messages to send via MPI. */
extern "C" int bc_gpu_backprop_level(
    int batch_sz, int lev,
    int* out_b, int* out_ugl, double* out_coeff)
{
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(s_bcnt, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int tot = batch_sz * s_n;
    int blk = (tot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_backprop<<<blk, BLOCK_SIZE>>>(
        s_dd, s_ds, s_ddel,
        batch_sz, s_n, lev,
        s_v0, s_vend,
        s_off, s_dst,
        s_bb, s_bu, s_bc, s_bcnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int cnt = 0;
    CUDA_CHECK(cudaMemcpy(&cnt, s_bcnt, sizeof(int), cudaMemcpyDeviceToHost));
    if (cnt > 0) {
        CUDA_CHECK(cudaMemcpy(out_b,     s_bb, cnt * sizeof(int),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_ugl,   s_bu, cnt * sizeof(int),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_coeff, s_bc, cnt * sizeof(double), cudaMemcpyDeviceToHost));
    }
    return cnt;
}

/* Apply received back_prop contributions: (b, u_lc, coeff) at level lev */
extern "C" void bc_gpu_apply_remote_bp(
    const int* b_h, const int* ulc_h, const double* coeff_h,
    int count, int lev)
{
    if (count <= 0) return;
    CUDA_CHECK(cudaMemcpy(s_ra_b,     b_h,    count * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_ra_x,     ulc_h,  count * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_ra_coeff, coeff_h,count * sizeof(double), cudaMemcpyHostToDevice));

    int blk = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_apply_bp<<<blk, BLOCK_SIZE>>>(
        s_ra_b, s_ra_x, s_ra_coeff, count, lev, s_n,
        s_dd, s_ds, s_ddel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Download delta[batch_sz * local_n] to host */
extern "C" void bc_gpu_get_delta(int batch_sz, double* h_delta)
{
    CUDA_CHECK(cudaMemcpy(h_delta, s_ddel,
                          (size_t)batch_sz * s_n * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

/* Free all GPU memory */
extern "C" void bc_gpu_cleanup(void)
{
    cudaFree(s_off);   cudaFree(s_dst);
    cudaFree(s_dd);    cudaFree(s_ds);    cudaFree(s_ddel);
    cudaFree(s_af);    cudaFree(s_foff);
    cudaFree(s_fb);    cudaFree(s_fw);    cudaFree(s_fsig);  cudaFree(s_fcnt);
    cudaFree(s_nv);    cudaFree(s_ncnt);
    cudaFree(s_bb);    cudaFree(s_bu);    cudaFree(s_bc);    cudaFree(s_bcnt);
    cudaFree(s_ra_b);  cudaFree(s_ra_x);
    cudaFree(s_ra_sig); cudaFree(s_ra_coeff);
    s_off = s_dst = s_dd = s_af = s_foff = nullptr;
    s_ds = s_fsig = s_ra_sig = nullptr;
    s_ddel = s_bc = s_ra_coeff = nullptr;
    s_fb = s_fw = s_fcnt = s_nv = s_ncnt = nullptr;
    s_bb = s_bu = s_bcnt = s_ra_b = s_ra_x = nullptr;
}
