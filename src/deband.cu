/*
Deband — libplacebo pl_shader_deband + pl_shader_dither. Do not "simplify":
  * __cosf/__sinf (SFU; precise cos/sin would NOT match)
  * int UNORM load: v * (1.0f/PEAK), NOT divide (div.rn differs on ~half the codes)
  * f16 store: cvt.rz.f16.f32 (RTZ), not rte
  * db_store8/16: NVIDIA UNORM (fixed-point RTZ intermediate + half-up rescale)
Defines: ITER, GRAIN_ON, BITS, HALF, DITHERK, DMODE, DB_BX, DB_BY.
*/

#define TWO_PI 6.283185f
#define INV_U32 0x1p-32f

#ifndef DB_BX
#define DB_BX 16
#endif
#ifndef DB_BY
#define DB_BY 8
#endif
#ifndef ITER
#define ITER 1
#endif
#ifndef GRAIN_ON
#define GRAIN_ON 1
#endif
#ifndef BITS
#define BITS 32
#endif
#ifndef HALF
#define HALF 0
#endif
#ifndef DITHERK
#define DITHERK 0
#endif
#ifndef DMODE
#define DMODE 0
#endif

#if BITS == 16 && HALF
static __device__ __forceinline__ float half_widen(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
/* r16f render-target store: RTZ, NOT rte. */
static __device__ __forceinline__ unsigned short half_narrow_rtz(float f) {
    unsigned short h;
    asm("cvt.rz.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
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
#define STOREI(p, i, x) ((p)[i] = half_narrow_rtz(x))
#elif BITS == 16
typedef unsigned short io_t;
#define LOADI(p, i) ((float)((p)[i]) * (1.0f/65535.0f))
#define STOREI(p, i, x) ((p)[i] = db_store16(x))
#else
typedef unsigned char io_t;
#define LOADI(p, i) ((float)((p)[i]) * (1.0f/255.0f))
#define STOREI(p, i, x) ((p)[i] = db_store8(x))
#endif

#if BITS != 32 && !HALF
/* NVIDIA UNORM store: (n+4)-bit RTZ intermediate, scale with half-DOWN. */
static __device__ __forceinline__ unsigned char db_store8(float x) {
    unsigned int k = (unsigned int)(fminf(fmaxf(x, 0.0f), 1.0f) * 4096.0f);
    return (unsigned char)((k * 255u + 2047u) >> 12);
}
static __device__ __forceinline__ unsigned short db_store16(float x) {
    unsigned int k = (unsigned int)(fminf(fmaxf(x, 0.0f), 1.0f) * 1048576.0f);
    return (unsigned short)(((unsigned long long)k * 65535ull + 524287ull) >> 20);
}
#endif

static __device__ __forceinline__ uint3 db_pcg3d(uint3 v) {
    v.x = v.x * 1664525u + 1013904223u;
    v.y = v.y * 1664525u + 1013904223u;
    v.z = v.z * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.x ^= v.x >> 16;
    v.y ^= v.y >> 16;
    v.z ^= v.z >> 16;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

static __device__ __forceinline__ float db_get(const io_t * __restrict__ src,
                                               int W, int H, int STRIDE,
                                               int px, int py, float ox, float oy) {
    /* Nearest + clamp-to-edge; deliberately asymmetric at half-integer offsets —
     * do NOT reuse a negated offset or "correct" the rounding. */
    int cx = min(max((int)floorf((float)px + 0.5f + ox), 0), W - 1);
    int cy = min(max((int)floorf((float)py + 0.5f + oy), 0), H - 1);
    return LOADI(src, cy * STRIDE + cx);
}

static __device__ __forceinline__ float db_core(const io_t * __restrict__ src,
                                                int W, int H, int STRIDE,
                                                float threshold, float radius, float grain,
                                                unsigned int zseed, int gx, int gy) {
    uint3 state = make_uint3((unsigned int)gx, (unsigned int)gy, zseed);
    float res = LOADI(src, gy * STRIDE + gx);

#if ITER > 0
    #pragma unroll
    for (int i = 1; i <= ITER; ++i) {
        state = db_pcg3d(state);
        float rx = (float)state.x * INV_U32;
        float ry = (float)state.y * INV_U32;
        float dist  = rx * ((float)i * radius);
        float theta = ry * TWO_PI;
        float dx = dist * __cosf(theta);       /* SFU trig — required for parity */
        float dy = dist * __sinf(theta);
        float avg = 0.0f;                      /* exact 4-tap order + *0.25 */
        avg += db_get(src, W, H, STRIDE, gx, gy,  dx,  dy);
        avg += db_get(src, W, H, STRIDE, gx, gy, -dx,  dy);
        avg += db_get(src, W, H, STRIDE, gx, gy, -dx, -dy);
        avg += db_get(src, W, H, STRIDE, gx, gy,  dx, -dy);
        avg *= 0.25f;
        float diff  = fabsf(res - avg);
        float bound = threshold / (float)i;
        res = (diff > bound) ? res : avg;
    }
#endif

#if GRAIN_ON
    state = db_pcg3d(state);
    float gx0 = (float)state.x * INV_U32;
    float strength = fminf(fabsf(res), grain);   /* grain_neutral == 0 */
    res += strength * (gx0 - 0.5f);
#endif

    return res;
}

/* Fused: gridDim.z=n_proc; per-plane: gridDim.z=1, geom pre-offset, zbase carries rank.
 * zseed = (zbase + z) & 0xFF reproduces (n*P + rank) & 0xFF. geom[z*6+..] =
 * {w,h,stride,off_src,off_dst,dith}. */
extern "C" __global__ void __launch_bounds__(DB_BX * DB_BY)
deband(io_t * __restrict__ dst_b, const io_t * __restrict__ src_b,
       const int * __restrict__ geom,
       float threshold, float radius, float grain, unsigned int zbase
#if DITHERK && (DMODE == 0 || DMODE == 1)
       , const float * __restrict__ dlut
#endif
       ) {
    const int z = blockIdx.z;
    const int W = geom[z*6+0], H = geom[z*6+1], STRIDE = geom[z*6+2];
    const int gx = blockIdx.x * DB_BX + threadIdx.x;
    const int gy = blockIdx.y * DB_BY + threadIdx.y;
    if (gx >= W || gy >= H) return;
    const io_t * __restrict__ src = src_b + geom[z*6+3];
    io_t * __restrict__ dst = dst_b + geom[z*6+4];
    const unsigned int zseed = (zbase + (unsigned int)z) & 0xFFu;
    float res = db_core(src, W, H, STRIDE, threshold, radius, grain, zseed, gx, gy);

#if DITHERK
    if (geom[z*6+5]) {  /* 8-bit only, first processed plane (vs-placebo) */
        float bias;
#if DMODE == 0 || DMODE == 1
        bias = dlut[((gy & 63) << 6) | (gx & 63)];
#elif DMODE == 2
        /* PL_DITHER_ORDERED_FIXED: 16x16 morton + bitwise reversal, verbatim. */
        unsigned int bx = ((unsigned int)gx & 15u) ^ ((unsigned int)gy & 15u);
        unsigned int by = (unsigned int)gy & 15u;
        bx = (bx | (bx << 2)) & 0x33333333u;
        by = (by | (by << 2)) & 0x33333333u;
        bx = (bx | (bx << 1)) & 0x55555555u;
        by = (by | (by << 1)) & 0x55555555u;
        unsigned int b = bx + (by << 1);
        b = (b * 0x0802u & 0x22110u) | (b * 0x8020u & 0x88440u);
        b = 0x10101u * b;
        b = (b >> 16) & 0xFFu;
        bias = (float)b * (1.0f / 256.0f);
#else
        /* PL_DITHER_WHITE_NOISE: temporal=false -> z=0. */
        uint3 ws = db_pcg3d(make_uint3((unsigned int)gx, (unsigned int)gy, 0u));
        bias = (float)ws.x * INV_U32;
#endif
        res = (fabsf(res) < 1e-5f) ? 0.0f : res;    /* don't lift true black off zero */
        /* PINNED fma(255, res, bias) — any other contraction flips pixels. */
        float q = floorf(__fmaf_rn(255.0f, res, bias));
        dst[gy * STRIDE + gx] = (unsigned char)min(max(__float2int_rz(q), 0), 255);
        return;
    }
#endif
    STOREI(dst, gy * STRIDE + gx, res);
}
