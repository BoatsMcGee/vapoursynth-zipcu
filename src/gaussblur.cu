/*
GaussBlur — separable Gaussian (fused-smem small path + two-pass large path).
Parity: mirror reflects, ascending-k tap order, f32 pipeline — do not alter.
Defines: SMALL, BLK_X/Y or BX/BY/R/RH, W/H/STRIDE/KLEN/RAD/VRT (small), BITS/HALF, IDX_LONG.
io: integer = raw codes (__float2int_rn + sat); f16 = cvt.rn.f16.f32; f32 identity.
Weights stay in global memory (const __restrict__), not __constant__ (edge-divergent indices).
*/

#ifndef BITS
#define BITS 32
#endif
#ifndef HALF
#define HALF 0
#endif
#ifndef IDX_LONG
#define IDX_LONG 0
#endif
#if IDX_LONG
typedef long long idx_t;
#else
typedef int idx_t;
#endif

#if BITS == 16 && HALF
static __device__ __forceinline__ float half_widen(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
static __device__ __forceinline__ unsigned short half_narrow_rte(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
#endif

#if BITS == 32
typedef float io_t;
#define LOADI(p, i) ((p)[i])
#define STOREI(p, i, x) ((p)[i] = (x))
#elif BITS == 16 && HALF
typedef unsigned short io_t;
#define LOADI(p, i) half_widen((p)[i])
#define STOREI(p, i, x) ((p)[i] = half_narrow_rte(x))
#elif BITS == 16
typedef unsigned short io_t;
#define LOADI(p, i) ((float)((p)[i]))
#define STOREI(p, i, x) ((p)[i] = (unsigned short)min(max(__float2int_rn(x), 0), 65535))
#else
typedef unsigned char io_t;
#define LOADI(p, i) ((float)((p)[i]))
#define STOREI(p, i, x) ((p)[i] = (unsigned char)min(max(__float2int_rn(x), 0), 255))
#endif

#if SMALL
/* Fused V-then-H. Mirror: single fold, edge not repeated (-i / 2*(dim-1)-i). */
extern "C" __global__ void __launch_bounds__(BLK_X * BLK_Y)
gauss_blur(io_t * __restrict__ dst, const io_t * __restrict__ src,
           const float * __restrict__ blur) {
    __shared__ float vblur[VRT * BLK_Y][BLK_X + 2 * RAD];
    const int lx = threadIdx.x, ly = threadIdx.y;
    const int gx0 = blockIdx.x * BLK_X;
    const int gy0 = blockIdx.y * (VRT * BLK_Y);

    for (int ry = ly; ry < VRT * BLK_Y; ry += BLK_Y) {
        const int y = gy0 + ry;
        if (y >= H) continue;
        for (int cj = lx; cj < BLK_X + 2 * RAD; cj += BLK_X) {
            int cx = gx0 - RAD + cj;
            if (cx < 0) cx = -cx;
            else if (cx >= W) cx = 2 * (W - 1) - cx;
            cx = min(max(cx, 0), W - 1);
            float vsum = 0.0f;
            #pragma unroll
            for (int k = 0; k < KLEN; ++k) {
                int sy = y + k - RAD;
                if (sy < 0) sy = -sy;
                else if (sy >= H) sy = 2 * (H - 1) - sy;
                vsum += LOADI(src, (idx_t)sy * STRIDE + cx) * blur[k];
            }
            vblur[ry][cj] = vsum;
        }
    }
    __syncthreads();

    const int x = gx0 + lx;
    if (x >= W) return;
    for (int r = 0; r < VRT; ++r) {
        const int y = gy0 + ly + r * BLK_Y;
        if (y >= H) return;
        const int lc = ly + r * BLK_Y;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KLEN; ++k)
            sum += vblur[lc][lx + k] * blur[k];
        STOREI(dst, (idx_t)y * STRIDE + x, sum);
    }
}

#else

/* R consecutive outputs; rolling weight window. Bit-exact vs per-pixel ascending-k. */
extern "C" __global__ void __launch_bounds__(BX * BY)
vertical_blur(float * __restrict__ dst, const io_t * __restrict__ src,
              const float * __restrict__ blur_kernel, int kernel_len,
              const int w, const int h, const int stride) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y0 = (blockIdx.y * BY + threadIdx.y) * R;
    if (x >= w || y0 >= h) return;
    const int radius = kernel_len / 2;
    float sum[R], wreg[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) { sum[j] = 0.0f; wreg[j] = 0.0f; }
    /* wreg[j]==0 for OOB taps; sum+=v*0 is IEEE no-op — same ascending-k FMAs. */
    for (int k = 0; k < kernel_len + (R - 1); ++k) {
        #pragma unroll
        for (int j = R - 1; j >= 1; --j) wreg[j] = wreg[j - 1];
        wreg[0] = (k < kernel_len) ? blur_kernel[k] : 0.0f;
        int sy = y0 + k - radius;
        if (sy < 0) sy = -sy;
        else if (sy >= h) sy = 2 * (h - 1) - sy;
        sy = min(max(sy, 0), h - 1);
        const float v = LOADI(src, (idx_t)sy * stride + x);
        #pragma unroll
        for (int j = 0; j < R; ++j)
            sum[j] += v * wreg[j];
    }
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        if (y0 + j < h) dst[(idx_t)(y0 + j) * stride + x] = sum[j];
    }
}

#ifndef RH
#define RH R
#endif
extern "C" __global__ void __launch_bounds__(BX * BY)
horizontal_blur(io_t * __restrict__ dst, const float * __restrict__ src,
                const float * __restrict__ blur_kernel, int kernel_len,
                const int w, const int h, const int stride) {
    const int x0 = (blockIdx.x * BX + threadIdx.x) * RH;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x0 >= w || y >= h) return;
    const int radius = kernel_len / 2;
    const idx_t row = (idx_t)y * stride;
    float sum[RH], wreg[RH];
    #pragma unroll
    for (int j = 0; j < RH; ++j) { sum[j] = 0.0f; wreg[j] = 0.0f; }
    for (int k = 0; k < kernel_len + (RH - 1); ++k) {
        #pragma unroll
        for (int j = RH - 1; j >= 1; --j) wreg[j] = wreg[j - 1];
        wreg[0] = (k < kernel_len) ? blur_kernel[k] : 0.0f;
        int sx = x0 + k - radius;
        if (sx < 0) sx = -sx;
        else if (sx >= w) sx = 2 * (w - 1) - sx;
        sx = min(max(sx, 0), w - 1);
        const float v = src[row + sx];
        #pragma unroll
        for (int j = 0; j < RH; ++j)
            sum[j] += v * wreg[j];
    }
    #pragma unroll
    for (int j = 0; j < RH; ++j) {
        if (x0 + j < w) STOREI(dst, row + x0 + j, sum[j]);
    }
}
#endif /* SMALL */
