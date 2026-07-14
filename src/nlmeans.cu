/*
NLMeans (KNLMeansCL spatio-temporal on plain buffers). Do not reorder/re-associate:
math, tap order, zero-border, q-batched sweep, joint-channel distance.
'YUV' requires 4:4:4 (one pixel lattice). Integer LOADU1 uses division (UNORM decode);
f16 via PTX cvt; u2 as float2/float3/float4.
*/

#ifndef IDX_LONG
#define IDX_LONG 0
#endif
#if IDX_LONG
typedef long long idx_t;
#else
typedef int idx_t;
#endif
#ifndef BITS
#define BITS 32
#endif
#ifndef HALF
#define HALF 0
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
#define LOADU1(i)     (u1[i])
#define STOREU1(i, x) (u1z[i] = (x))
#elif BITS == 16 && HALF
typedef unsigned short io_t;
#define LOADU1(i)     half_widen(u1[i])
#define STOREU1(i, x) (u1z[i] = half_narrow_rte(x))
#elif BITS == 16
typedef unsigned short io_t;
#define LOADU1(i)     ((float)(u1[i]) / 65535.0f)
#define STOREU1(i, x) (u1z[i] = (unsigned short)min(max(__float2int_rn((x) * 65535.0f), 0), 65535))
#else
typedef unsigned char io_t;
#define LOADU1(i)     ((float)(u1[i]) / 255.0f)
#define STOREU1(i, x) (u1z[i] = (unsigned char)min(max(__float2int_rn((x) * 255.0f), 0), 255))
#endif

#define NLM_NORM        (255.0f*255.0f)
#define NLM_LEGACY      (3.0f)
#define NLM_S_SIZE      ((2*NLM_S+1)*(2*NLM_S+1))
#define NLM_H2_INV_NORM (NLM_NORM/(NLM_LEGACY*NLM_H*NLM_H*NLM_S_SIZE))
#define U1_LAYER        ((idx_t)PSTRIDE*PH)
#define U1_PLANE        ((idx_t)(2*NLM_D+1)*U1_LAYER)
#define U4_LAYER        ((idx_t)STRIDE*VI_DIM_Y)
#define NPIX            ((idx_t)STRIDE*VI_DIM_Y)

/* Joint multi-channel patch distance; constants and operand order are load-bearing. */
static __device__ __forceinline__ float nlm_pix_dist(const io_t * __restrict__ u1,
                                                     const int t, const int qz,
                                                     const int xx, const int y,
                                                     const int qx, const int qy) {
    const idx_t ac = (idx_t)t*U1_LAYER + (idx_t)(y+PAD)*PSTRIDE + (xx+PAD);
    const idx_t bc = (idx_t)(t+qz)*U1_LAYER + (idx_t)(y+qy+PAD)*PSTRIDE + (xx+qx+PAD);
#if   NLM_REF == 0   /* LUMA */
    const float d0 = LOADU1(ac) - LOADU1(bc);
    return 3.0f * (d0 * d0);
#elif NLM_REF == 1   /* CHROMA (U,V) */
    const float du = LOADU1(ac) - LOADU1(bc);
    const float dv = LOADU1(U1_PLANE + ac) - LOADU1(U1_PLANE + bc);
    return 1.5f * (du*du + dv*dv);
#elif NLM_REF == 2   /* YUV (4:4:4 only) */
    const float dy = LOADU1(ac) - LOADU1(bc);
    const float du = LOADU1(U1_PLANE + ac) - LOADU1(U1_PLANE + bc);
    const float dv = LOADU1(2*U1_PLANE + ac) - LOADU1(2*U1_PLANE + bc);
    return dy*dy + du*du + dv*dv;
#else                /* RGB */
    const float ar = LOADU1(ac), br = LOADU1(bc);
    const float dr = ar - br;
    const float dg = LOADU1(U1_PLANE + ac) - LOADU1(U1_PLANE + bc);
    const float db = LOADU1(2*U1_PLANE + ac) - LOADU1(2*U1_PLANE + bc);
    const float m_red = (ar + br) / 6.0f;
    return (2.0f/3.0f + m_red) * (dr*dr) + (4.0f/3.0f) * (dg*dg) + (1.0f - m_red) * (db*db);
#endif
}

/* Q-batched: blockIdx.z picks pass; (t,qx,qy,qz,slot) from wq. */
extern "C" __global__ void __launch_bounds__(BLK_X * BLK_Y)
nlmWeight(const io_t * __restrict__ u1, float * __restrict__ u4a,
          const int * __restrict__ wq, const int pass_base) {
    const int prow = (pass_base + (int)blockIdx.z) * 8;
    const int t = wq[prow+0], qx = wq[prow+1], qy = wq[prow+2], qz = wq[prow+3];
    const int slot = wq[prow+4];
    /* Odd-padded row strides: even natural strides 2-way bank-conflict. Layout only. */
#define DIST_W ((BLK_X + 2*NLM_S) | 1)
#define HSUM_W (BLK_X | 1)
    __shared__ float dist[VRT_RESULT*BLK_Y + 2*NLM_S][DIST_W];
    __shared__ float hsum[VRT_RESULT*BLK_Y + 2*NLM_S][HSUM_W];
    const int lx = threadIdx.x, ly = threadIdx.y;
    const int gx0 = blockIdx.x * BLK_X;
    const int gy0 = blockIdx.y * (VRT_RESULT*BLK_Y);
    for (int ry = ly; ry < VRT_RESULT*BLK_Y + 2*NLM_S; ry += BLK_Y) {
        const int yy = gy0 - NLM_S + ry;
        for (int cx = lx; cx < BLK_X + 2*NLM_S; cx += BLK_X) {
            const int xx = gx0 - NLM_S + cx;
            dist[ry][cx] = (xx >= 0 && xx < VI_DIM_X && yy >= 0 && yy < VI_DIM_Y)
                           ? nlm_pix_dist(u1, t, qz, xx, yy, qx, qy) : 0.0f;
        }
    }
    __syncthreads();
    for (int ry = ly; ry < VRT_RESULT*BLK_Y + 2*NLM_S; ry += BLK_Y) {
        float sh = 0.0f;
        #pragma unroll
        for (int i = 0; i <= 2*NLM_S; ++i) sh += dist[ry][lx + i];
        hsum[ry][lx] = sh;
    }
    __syncthreads();
    const int x = gx0 + lx;
    if (x >= VI_DIM_X) return;
    int y = gy0 + ly, lc = ly;
    for (int r = 0; r < VRT_RESULT; ++r, y += BLK_Y, lc += BLK_Y) {
        if (y >= VI_DIM_Y) return;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i <= 2*NLM_S; ++i) sum += hsum[lc + i][lx];
        /* PINNED multiply: must not fuse into fdim/exp; weight needs rounded product. */
        const float arg = __fmul_rn(sum, NLM_H2_INV_NORM);
        float w;
#if   WMODE == 0
        w = expf(-arg);
#elif WMODE == 1
        w = fdimf(1.0f, arg);
#elif WMODE == 2
        { float c = fdimf(1.0f, arg); w = c*c; }
#else
        { float c = fdimf(1.0f, arg); c = c*c; c = c*c; w = c*c; }
#endif
        u4a[(idx_t)slot*U4_LAYER + y*STRIDE + x] = w;
    }
}

/* Batch of +q/-q; per-pixel sweep order is part of the result. */
extern "C" __global__ void __launch_bounds__(BLK_X * BLK_Y)
nlmAccumulation(const io_t * __restrict__ u1, float * __restrict__ u2,
                const float * __restrict__ u4a, float * __restrict__ u5,
                const int * __restrict__ aq, const int q_base, const int nb,
                const int first) {
    const int x = blockIdx.x * BLK_X + threadIdx.x;
    const int y = blockIdx.y * BLK_Y + threadIdx.y;
    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    const int g = y*STRIDE + x;
    const int cx = x+PAD, cy = y+PAD;
    /* first batch starts from frame-init constants (replaces u2/u5 memsets). */
    float u5v = first ? 0x1p-23f : u5[g];   /* FLT_EPS seed keeps den > 0 */
#if   NLM_CHANNELS == 1
    float2 acc = first ? make_float2(0.0f, 0.0f) : reinterpret_cast<float2*>(u2)[g];
#elif NLM_CHANNELS == 2
    float3 acc = first ? make_float3(0.0f, 0.0f, 0.0f) : reinterpret_cast<float3*>(u2)[g];
#else
    float4 acc = first ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : reinterpret_cast<float4*>(u2)[g];
#endif
    for (int b = 0; b < nb; ++b) {
        const int r = (q_base + b) * 8;
        const int qx = aq[r+0], qy = aq[r+1], qz = aq[r+2];
        const int sc = aq[r+3], sm = aq[r+4];
        const float u4 = u4a[(idx_t)sc*U4_LAYER + g];
        const int xm = x-qx, ym = y-qy;
        const float u4_mq = (xm < 0 || xm >= VI_DIM_X || ym < 0 || ym >= VI_DIM_Y)
                            ? 0.0f : u4a[(idx_t)sm*U4_LAYER + ym*STRIDE + xm];
        u5v = fmaxf(u4, fmaxf(u4_mq, u5v));
        const idx_t pq = (idx_t)(NLM_D+qz)*U1_LAYER + (idx_t)(cy+qy)*PSTRIDE + (cx+qx);
        const idx_t mq = (idx_t)(NLM_D-qz)*U1_LAYER + (idx_t)(cy-qy)*PSTRIDE + (cx-qx);
#if   NLM_CHANNELS == 1
        const float pq0 = LOADU1(pq), mq0 = LOADU1(mq);
        acc.x += (u4*pq0) + (u4_mq*mq0);
        acc.y += (u4 + u4_mq);
#elif NLM_CHANNELS == 2
        const float pq0 = LOADU1(pq),            mq0 = LOADU1(mq);
        const float pq1 = LOADU1(U1_PLANE + pq), mq1 = LOADU1(U1_PLANE + mq);
        acc.x += (u4*pq0) + (u4_mq*mq0);
        acc.y += (u4*pq1) + (u4_mq*mq1);
        acc.z += (u4 + u4_mq);
#else
        const float pq0 = LOADU1(pq),              mq0 = LOADU1(mq);
        const float pq1 = LOADU1(U1_PLANE + pq),   mq1 = LOADU1(U1_PLANE + mq);
        const float pq2 = LOADU1(2*U1_PLANE + pq), mq2 = LOADU1(2*U1_PLANE + mq);
        acc.x += (u4*pq0) + (u4_mq*mq0);
        acc.y += (u4*pq1) + (u4_mq*mq1);
        acc.z += (u4*pq2) + (u4_mq*mq2);
        acc.w += (u4 + u4_mq);
#endif
    }
#if   NLM_CHANNELS == 1
    reinterpret_cast<float2*>(u2)[g] = acc;
#elif NLM_CHANNELS == 2
    reinterpret_cast<float3*>(u2)[g] = acc;
#else
    reinterpret_cast<float4*>(u2)[g] = acc;
#endif
    u5[g] = u5v;
}

/* den==0 unguarded: FLT_EPS seed in u5 keeps den > 0. */
extern "C" __global__ void __launch_bounds__(BLK_X * BLK_Y)
nlmFinish(const io_t * __restrict__ u1, io_t * __restrict__ u1z,
          const float * __restrict__ u2, const float * __restrict__ u5) {
    const int x = blockIdx.x * BLK_X + threadIdx.x;
    const int y = blockIdx.y * BLK_Y + threadIdx.y;
    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    const int g = y*STRIDE + x;
    /* PINNED: m must stay rounded (not fused into den). */
    const float m = __fmul_rn(NLM_WREF, u5[g]);
    const idx_t uc = (idx_t)NLM_D*U1_LAYER + (idx_t)(y+PAD)*PSTRIDE + (x+PAD);
#if   NLM_CHANNELS == 1
    float2 acc = reinterpret_cast<const float2*>(u2)[g];
    const float den = m + acc.y;
    STOREU1(g, __fmaf_rn(LOADU1(uc), m, acc.x) / den);
#elif NLM_CHANNELS == 2
    float3 acc = reinterpret_cast<const float3*>(u2)[g];
    const float den = m + acc.z;
    STOREU1(g,        (LOADU1(uc)*m            + acc.x) / den);
    STOREU1(NPIX + g, (LOADU1(U1_PLANE + uc)*m + acc.y) / den);
#else
    float4 acc = reinterpret_cast<const float4*>(u2)[g];
    const float den = m + acc.w;
    STOREU1(g,          (LOADU1(uc)*m              + acc.x) / den);
    STOREU1(NPIX + g,   (LOADU1(U1_PLANE + uc)*m   + acc.y) / den);
    STOREU1(2*NPIX + g, (LOADU1(2*U1_PLANE + uc)*m + acc.z) / den);
#endif
}
