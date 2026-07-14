/*
EEDI3 CUDA kernels. Math, index logic, DP combine order, tie-break chains and barrier
placement are bit-exactness contracts against the OpenCL reference — do not "simplify".

io parity: int LOAD uses (float)v*(1/PEAK) not `/` (div.rn differs); STORE __float2int_rn;
f16 via PTX cvt.f32.f16 / cvt.rn.f16.f32.
*/

#define TPMAX 85  /* tpitch_max + 4 sentinels (K=2 fused DP reads tid..tid+4) */
#define FLTMAX9 3.0e38f
#ifndef CN
#define CN 2
#endif
#ifndef MDIS
#define MDIS 20
#endif
#ifndef BX
#define BX 8
#endif
#define TP  (2*MDIS + 1)
#define TPH (4*MDIS + 1)

#ifndef BITS
#define BITS 32
#endif
#ifndef HALF
#define HALF 0
#endif

typedef signed char schar;

#ifndef LWSF
#define LWSF 128
#endif
#define CEIL32(v) (((v) + 31) / 32 * 32)
#define LWS_NH (CEIL32(TP)  > LWSF ? CEIL32(TP)  : LWSF)
#define LWS_HP (CEIL32(TPH) > LWSF ? CEIL32(TPH) : LWSF)
#ifndef MINB_NH
#define MINB_NH 1
#endif
#ifndef MINB_HP
#define MINB_HP 1
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
typedef float raw_t;
#define LOAD_IO(p, i) ((p)[i])
#define STORE_IO(p, i, x) ((p)[i] = (x))
#elif BITS == 16 && HALF
typedef unsigned short io_t;
typedef unsigned short raw_t;
#define LOAD_IO(p, i) half_widen((p)[i])
#define STORE_IO(p, i, x) ((p)[i] = half_narrow_rte(x))
#elif BITS == 16
typedef unsigned short io_t;
typedef unsigned short raw_t;
#define LOAD_IO(p, i) ((float)((p)[i]) * (1.0f/65535.0f))
#define STORE_IO(p, i, x) ((p)[i] = (unsigned short)min(max(__float2int_rn((x) * 65535.0f), 0), 65535))
#else
typedef unsigned char io_t;
typedef unsigned char raw_t;
#define LOAD_IO(p, i) ((float)((p)[i]) * (1.0f/255.0f))
#define STORE_IO(p, i, x) ((p)[i] = (unsigned char)min(max(__float2int_rn((x) * 255.0f), 0), 255))
#endif

static __device__ int refl(int i, const int w) {
    if (w == 1) return 0;
    while (i < 0 || i >= w) { if (i < 0) i = -i; if (i >= w) i = 2*(w-1) - i; }
    return i;
}

extern "C" __global__ void pad_src(float *out, const io_t *src,
                                   const int w, const int stride, const int pstride,
                                   const int pad, const int src_h, const int soff) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= pad + w + pad || y >= src_h) return;
    out[(long long)y*pstride + px] = LOAD_IO(src, (long long)soff + (long long)y*stride + refl(px - pad, w));
}

extern "C" __global__ void pad_hp(float *hpout, const float *srcpad,
                                  const int pstride, const int src_h, const int pad, const int w) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int pw = pad + w + pad;
    if (px >= pw || y >= src_h) return;
    const float *r = srcpad + (long long)y*pstride;
    float v = (px >= 1 && px + 2 <= pw - 1)
        ? 0.5625f*(r[px]+r[px+1]) - 0.0625f*(r[px-1]+r[px+2])
        : r[px];
    hpout[(long long)y*pstride + px] = v;
}

static __device__ float conn_cost(const float * __restrict__ r3p, const float * __restrict__ r1p,
                                  const float * __restrict__ r1n, const float * __restrict__ r3n,
                                  const int x, const int u, const int w, const int nrad,
                                  const int pad,
                                  const float alpha, const float beta, const float one_minus_ab) {
    const int two_u = 2*u;
    const int xp = x + pad;
    float sw = 0.0f;
    #pragma unroll
    for (int k = -CN; k <= CN; k++) {
        int a0 = xp + u + k,    b0 = a0 - two_u;
        int a1 = xp + k,        b1 = a1 - two_u;
        int a2 = xp + two_u + k, b2 = a2 - two_u;
        sw += fabsf(r3p[a0]-r1p[b0]) + fabsf(r1p[a0]-r1n[b0]) + fabsf(r1n[a0]-r3n[b0]);
        sw += fabsf(r3p[a1]-r1p[b1]) + fabsf(r1p[a1]-r1n[b1]) + fabsf(r1n[a1]-r3n[b1]);
        sw += fabsf(r3p[a2]-r1p[b2]) + fabsf(r1p[a2]-r1n[b2]) + fabsf(r1n[a2]-r3n[b2]);
    }
    float ip = (r1p[xp+u] + r1n[xp-u]) * 0.5f;
    float v = fabsf(r1p[xp] - ip) + fabsf(r1n[xp] - ip);
    return alpha * sw + beta * (float)abs(u) + one_minus_ab * v;
}

#define CW (2*CN + 1)
#ifndef RUN
#define RUN 8
#endif
#ifndef RUN_HP
#define RUN_HP 4
#endif

static __device__ float h_pos(const float * __restrict__ r3p, const float * __restrict__ r1p,
                              const float * __restrict__ r1n, const float * __restrict__ r3n,
                              const int c, const int two_u) {
    const int b = c - two_u;
    return fabsf(r3p[c]-r1p[b]) + fabsf(r1p[c]-r1n[b]) + fabsf(r1n[c]-r3n[b]);
}
static __device__ float h_neg(const float * __restrict__ r3p, const float * __restrict__ r1p,
                              const float * __restrict__ r1n, const float * __restrict__ r3n,
                              const int c, const int two_u) {
    const int b = c + two_u;
    return fabsf(r3p[c]-r1p[b]) + fabsf(r1p[c]-r1n[b]) + fabsf(r1n[c]-r3n[b]);
}
static __device__ float2 h_pair(const float * __restrict__ r3p, const float * __restrict__ r1p,
                                const float * __restrict__ r1n, const float * __restrict__ r3n,
                                const int c, const int two_u) {
    const float A = r3p[c], B = r1p[c], C = r1n[c];
    const float Bm = r1p[c-two_u], Cm = r1n[c-two_u], Dm = r3n[c-two_u];
    const float Bp = r1p[c+two_u], Cp = r1n[c+two_u], Dp = r3n[c+two_u];
    return make_float2(fabsf(A-Bm) + fabsf(B-Cm) + fabsf(C-Dm),
                       fabsf(A-Bp) + fabsf(B-Cp) + fabsf(C-Dp));
}

static __device__ void cost_pair_run(const float * __restrict__ r3p, const float * __restrict__ r1p,
                                     const float * __restrict__ r1n, const float * __restrict__ r3n,
                                     float *co, const int xs, const int rl, const int u,
                                     const int pad, const float alpha, const float beta,
                                     const float one_minus_ab)
{
    const int two_u = 2*u;
    const int x0p = xs + pad;
    float wp1[CW], wp0[CW], wp2[CW], wm1[CW], wm0[CW], wm2[CW];
    #pragma unroll
    for (int t = 0; t < CW; t++) {
        const int c = x0p - CN + t;
        const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, two_u);
        wp0[t] = hc.x;  wm0[t] = hc.y;
        wp1[t] = h_pos(r3p,r1p,r1n,r3n, c + u,     two_u);
        wp2[t] = h_pos(r3p,r1p,r1n,r3n, c + two_u, two_u);
        wm1[t] = h_neg(r3p,r1p,r1n,r3n, c - u,     two_u);
        wm2[t] = h_neg(r3p,r1p,r1n,r3n, c - two_u, two_u);
    }
    #pragma unroll
    for (int t = 0; t < RUN; t++) {
        if (t >= rl) break;
        const int xp = x0p + t;
        float swp = 0.0f, swm = 0.0f;
        #pragma unroll
        for (int k = 0; k < CW; k++) {
            swp += wp1[k]; swp += wp0[k]; swp += wp2[k];
            swm += wm1[k]; swm += wm0[k]; swm += wm2[k];
        }
        const float b1pc = r1p[xp], b1nc = r1n[xp];
        float ipp = (r1p[xp+u] + r1n[xp-u]) * 0.5f;
        float vp = fabsf(b1pc - ipp) + fabsf(b1nc - ipp);
        float ipm = (r1p[xp-u] + r1n[xp+u]) * 0.5f;
        float vm = fabsf(b1pc - ipm) + fabsf(b1nc - ipm);
        co[t*TP + (MDIS + u)] = alpha * swp + beta * (float)abs(u) + one_minus_ab * vp;
        co[t*TP + (MDIS - u)] = alpha * swm + beta * (float)abs(u) + one_minus_ab * vm;
        if (t + 1 < rl) {
            #pragma unroll
            for (int k = 0; k < CW-1; k++) {
                wp1[k]=wp1[k+1]; wp0[k]=wp0[k+1]; wp2[k]=wp2[k+1];
                wm1[k]=wm1[k+1]; wm0[k]=wm0[k+1]; wm2[k]=wm2[k+1];
            }
            const int c = xp + 1 + CN;
            const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, two_u);
            wp0[CW-1] = hc.x;  wm0[CW-1] = hc.y;
            wp1[CW-1] = h_pos(r3p,r1p,r1n,r3n, c + u,     two_u);
            wp2[CW-1] = h_pos(r3p,r1p,r1n,r3n, c + two_u, two_u);
            wm1[CW-1] = h_neg(r3p,r1p,r1n,r3n, c - u,     two_u);
            wm2[CW-1] = h_neg(r3p,r1p,r1n,r3n, c - two_u, two_u);
        }
    }
}

extern "C" __global__ void copy_kept(raw_t *dst, const raw_t *src,
                                     const int w, const int stride, const int dh,
                                     const int field, const int src_h, const int dst_h,
                                     raw_t *dst2, const int dual,
                                     const int soff, const int doff) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= dst_h) return;
    int is_interp, sy = -1;
    if (dh) {
        is_interp = ((y & 1) == field);
        if (!is_interp) sy = (y - (1 - field)) >> 1;
    } else {
        is_interp = ((y & 1) == field);
        if (!is_interp) sy = y;
    }
    if (!is_interp) {
        const raw_t v = src[(long long)soff + (long long)sy*stride + x];
        dst[(long long)doff + (long long)y*stride + x] = v;
        if (dual) dst2[(long long)doff + (long long)y*stride + x] = v;
    }
}

extern "C" __global__ void __launch_bounds__(LWS_NH, MINB_NH)
interp(io_t * __restrict__ dst, const float * __restrict__ srcpad,
                                  const int * __restrict__ rowidx, const int * __restrict__ dst_y,
                                  schar * __restrict__ pbackt, int * __restrict__ dmap,
                                  const int w, const int stride, const int pstride, const int pad,
                                  const int mdis, const int nrad,
                                  const float alpha, const float beta, const float gamma,
                                  const float one_minus_ab,
                                  io_t * __restrict__ dst2, const int dual,
                                  const int doff, const int moff) {
    const int off = blockIdx.y;
    const int tid = threadIdx.x;
    const int lsz = blockDim.x;

    const float * __restrict__ r3p = srcpad + (long long)rowidx[off*4+0]*pstride;
    const float * __restrict__ r1p = srcpad + (long long)rowidx[off*4+1]*pstride;
    const float * __restrict__ r1n = srcpad + (long long)rowidx[off*4+2]*pstride;
    const float * __restrict__ r3n = srcpad + (long long)rowidx[off*4+3]*pstride;
    const int pb_pitch = (w * TP + 15) & ~15;
    schar *pb = pbackt + (long long)off * pb_pitch;

    __shared__ float pc[2][TPMAX];
    __shared__ __align__(16) float cst[BX * TP];
    const int u = tid - MDIS;
    const int active = (tid < TP);

    if (tid == 0) {
        pc[0][0]=FLTMAX9; pc[0][1]=FLTMAX9; pc[1][0]=FLTMAX9; pc[1][1]=FLTMAX9;
        pc[0][TP+2]=FLTMAX9; pc[0][TP+3]=FLTMAX9; pc[1][TP+2]=FLTMAX9; pc[1][TP+3]=FLTMAX9;
    }
    __syncthreads();

    int ping = 0;
    if (active) pc[ping][tid+2] = conn_cost(r3p,r1p,r1n,r3n, 0, u, w, nrad, pad, alpha, beta, one_minus_ab);
    __syncthreads();

    #define NUNIT (MDIS + 1)
    for (int x0 = 1; x0 < w; x0 += BX) {
        const int bn = min(BX, w - x0);
        const int nrun = (bn + RUN - 1) / RUN;
        for (int i = tid; i < nrun*NUNIT; i += lsz) {
            const int r  = i / NUNIT;
            const int j  = i - r*NUNIT;
            const int xs = x0 + r*RUN;
            const int rl = min(RUN, bn - r*RUN);
            float *co = cst + (r*RUN)*TP;
            if (j == 0) {
                for (int t = 0; t < rl; t++)
                    co[t*TP + MDIS] = conn_cost(r3p,r1p,r1n,r3n, xs+t, 0, w, nrad, pad, alpha, beta, one_minus_ab);
            } else {
                cost_pair_run(r3p,r1p,r1n,r3n, co, xs, rl, j, pad, alpha, beta, one_minus_ab);
            }
        }
        __syncthreads();
        int dx = 0;
        for (; dx + 1 < bn; dx += 2) {
            const int pong = ping ^ 1;
            if (active) {
                const float p0 = pc[ping][tid];
                const float p1 = pc[ping][tid+1];
                const float p2 = pc[ping][tid+2];
                const float p3 = pc[ping][tid+3];
                const float p4 = pc[ping][tid+4];
                const float cL = cst[dx*TP + max(tid-1, 0)];
                const float cC = cst[dx*TP + tid];
                const float cR = cst[dx*TP + min(tid+1, TP-1)];
                const float cN = cst[(dx+1)*TP + tid];
                float lft = p0 + gamma, cnt = p1, rgt = p2 + gamma;
                float bL = cnt; if (lft < bL) bL = lft; if (rgt < bL) bL = rgt;
                const float qL = (tid == 0) ? FLTMAX9 : fminf(bL + cL, FLTMAX9);
                lft = p1 + gamma; cnt = p2; rgt = p3 + gamma;
                float bC = cnt; schar bdC = 0;
                if (lft < bC) { bC = lft; bdC = -1; }
                if (rgt < bC) { bC = rgt; bdC =  1; }
                const float qC = fminf(bC + cC, FLTMAX9);
                pb[(x0+dx-1)*TP + tid] = bdC;
                lft = p2 + gamma; cnt = p3; rgt = p4 + gamma;
                float bR = cnt; if (lft < bR) bR = lft; if (rgt < bR) bR = rgt;
                const float qR = (tid == TP-1) ? FLTMAX9 : fminf(bR + cR, FLTMAX9);
                lft = qL + gamma; cnt = qC; rgt = qR + gamma;
                float bN = cnt; schar bdN = 0;
                if (lft < bN) { bN = lft; bdN = -1; }
                if (rgt < bN) { bN = rgt; bdN =  1; }
                pc[pong][tid+2] = fminf(bN + cN, FLTMAX9);
                pb[(x0+dx)*TP + tid] = bdN;
            }
            __syncthreads();
            ping = pong;
        }
        if (dx < bn) {
            const int pong = ping ^ 1;
            if (active) {
                const float cost = cst[dx*TP + tid];
                float left  = pc[ping][tid+1] + gamma;
                float cent  = pc[ping][tid+2];
                float right = pc[ping][tid+3] + gamma;
                float bval = cent; schar bd = 0;
                if (left  < bval) { bval = left;  bd = -1; }
                if (right < bval) { bval = right; bd =  1; }
                pc[pong][tid+2] = fminf(bval + cost, FLTMAX9);
                pb[(x0+dx-1)*TP + tid] = bd;
            }
            __syncthreads();
            ping = pong;
        }
    }

    __syncthreads();
    int *dm = dmap + moff + (long long)off*stride;
    {
        schar *stg = (schar *)cst;
        int f = 0;
        if (tid == 0) dm[w-1] = 0;
        for (int k = (w - 2) / (4*BX); k >= 0; k--) {
            const int xlo = k * (4*BX);
            const int xhi = min(xlo + 4*BX - 1, w - 2);
            const int nb = (xhi - xlo + 1) * TP;
#if (4*BX*TP) % 16 == 0
            const uint4 *gsrc = (const uint4 *)(pb + xlo*TP);
            uint4 *sdst = (uint4 *)stg;
            const int nv = nb >> 4;
            for (int i = tid; i < nv; i += lsz) sdst[i] = gsrc[i];
            for (int i = (nv << 4) + tid; i < nb; i += lsz) stg[i] = pb[xlo*TP + i];
#else
            for (int i = tid; i < nb; i += lsz) stg[i] = pb[xlo*TP + i];
#endif
            __syncthreads();
            if (tid == 0) {
                for (int bx = xhi; bx >= xlo; bx--) {
                    f += (int)stg[(bx - xlo)*TP + (MDIS + f)];
                    dm[bx] = f;
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();

    io_t *drow = dst + doff + (long long)dst_y[off]*stride;
    io_t *drow2 = dst2 + doff + (long long)dst_y[off]*stride;
    for (int x = tid; x < w; x += lsz) {
        const int dir = dm[x];
        int ad = abs(dir);
        const int xp = x + pad;
        float val;
        if (x >= ad*3 && x + ad*3 <= w-1) {
            val = 0.5625f * (r1p[xp+dir] + r1n[xp-dir])
                - 0.0625f * (r3p[xp+dir*3] + r3n[xp-dir*3]);
        } else {
            val = (r1p[xp+dir] + r1n[xp-dir]) * 0.5f;
        }
        STORE_IO(drow, x, val);
        if (dual) STORE_IO(drow2, x, val);
    }
}

#define TPMAX_HP 165
static __device__ __forceinline__ float R(const float * __restrict__ row, int j) { return row[j]; }

static __device__ float conn_cost_hp(const float * __restrict__ r3p, const float * __restrict__ r1p,
                                     const float * __restrict__ r1n, const float * __restrict__ r3n,
                                     const float * __restrict__ hp3p, const float * __restrict__ hp1p,
                                     const float * __restrict__ hp1n, const float * __restrict__ hp3n,
                                     const int x, const int u, const int w, const int nrad,
                                     const float alpha3, const float beta255, const float one_minus_ab) {
    const int uh = u >> 1;
    const int odd = u & 1;
    const int lo0 = odd ? (-uh - 1) : (-uh);
    float s0=0.0f, s1=0.0f, s2=0.0f;
    #pragma unroll
    for (int k=-CN; k<=CN; k++) {
        int xk = x+k, hi = x+uh+k, lo = x+lo0+k, xu = x+u+k, xmu = x-u+k;
        s1 += fabsf(R(r3p,xk)-R(r1p,xmu)) + fabsf(R(r1p,xk)-R(r1n,xmu)) + fabsf(R(r1n,xk)-R(r3n,xmu));
        s2 += fabsf(R(r3p,xu)-R(r1p,xk)) + fabsf(R(r1p,xu)-R(r1n,xk)) + fabsf(R(r1n,xu)-R(r3n,xk));
        if (odd)
            s0 += fabsf(hp3p[hi]-hp1p[lo]) + fabsf(hp1p[hi]-hp1n[lo]) + fabsf(hp1n[hi]-hp3n[lo]);
        else
            s0 += fabsf(R(r3p,hi)-R(r1p,lo)) + fabsf(R(r1p,hi)-R(r1n,lo)) + fabsf(R(r1n,hi)-R(r3n,lo));
    }
    float Bxuh = odd ? hp1p[x+uh] : R(r1p,x+uh);
    float Cxlo = odd ? hp1n[x+lo0] : R(r1n,x+lo0);
    float ip = (Bxuh + Cxlo) * 0.5f;
    float v = fabsf(R(r1p,x)-ip) + fabsf(R(r1n,x)-ip);
    return alpha3*(s0+s1+s2) + beta255*(float)abs(u)*0.5f + one_minus_ab*v;
}

static __device__ void cost_hp_pair_run(
        const float * __restrict__ r3p, const float * __restrict__ r1p,
        const float * __restrict__ r1n, const float * __restrict__ r3n,
        const float * __restrict__ S3p, const float * __restrict__ S1p,
        const float * __restrict__ S1n, const float * __restrict__ S3n,
        const float * __restrict__ Sel1p, const float * __restrict__ Sel1n,
        float *co, const int xs, const int rl, const int u,
        const float alpha3, const float beta255, const float one_minus_ab)
{
    const int uh = u >> 1;
    const int lo0 = (u & 1) ? (-uh - 1) : (-uh);
    float V0p[CW], V0m[CW], V1p[CW], V1m[CW], G0p[CW], G0m[CW];
    #pragma unroll
    for (int t = 0; t < CW; t++) {
        const int c = xs - CN + t;
        const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, u);
        V0p[t] = hc.x;  V0m[t] = hc.y;
        V1p[t] = h_pos(r3p,r1p,r1n,r3n, c + u, u);
        V1m[t] = h_neg(r3p,r1p,r1n,r3n, c - u, u);
        G0p[t] = h_pos(S3p,S1p,S1n,S3n, c + uh,  u);
        G0m[t] = h_neg(S3p,S1p,S1n,S3n, c + lo0, u);
    }
    #pragma unroll
    for (int t = 0; t < RUN_HP; t++) {
        if (t >= rl) break;
        const int x = xs + t;
        float s0p=0.0f, s1p=0.0f, s2p=0.0f;
        float s0m=0.0f, s1m=0.0f, s2m=0.0f;
        #pragma unroll
        for (int k = 0; k < CW; k++) {
            s1p += V0p[k]; s2p += V1p[k];
            s1m += V0m[k]; s2m += V1m[k];
            s0p += G0p[k]; s0m += G0m[k];
        }
        const float b1pc = R(r1p,x), b1nc = R(r1n,x);
        float Bxuh_p = Sel1p[x+uh];
        float Cxlo_p = Sel1n[x+lo0];
        float ip_p = (Bxuh_p + Cxlo_p) * 0.5f;
        float v_p = fabsf(b1pc-ip_p) + fabsf(b1nc-ip_p);
        float Bxuh_m = Sel1p[x+lo0];
        float Cxlo_m = Sel1n[x+uh];
        float ip_m = (Bxuh_m + Cxlo_m) * 0.5f;
        float v_m = fabsf(b1pc-ip_m) + fabsf(b1nc-ip_m);
        co[t*TPH + (2*MDIS + u)] = alpha3*(s0p+s1p+s2p) + beta255*(float)abs(u)*0.5f + one_minus_ab*v_p;
        co[t*TPH + (2*MDIS - u)] = alpha3*(s0m+s1m+s2m) + beta255*(float)abs(u)*0.5f + one_minus_ab*v_m;
        if (t + 1 < rl) {
            #pragma unroll
            for (int k = 0; k < CW-1; k++) {
                V0p[k]=V0p[k+1]; V0m[k]=V0m[k+1]; V1p[k]=V1p[k+1];
                V1m[k]=V1m[k+1]; G0p[k]=G0p[k+1]; G0m[k]=G0m[k+1];
            }
            const int c = x + 1 + CN;
            const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, u);
            V0p[CW-1] = hc.x;  V0m[CW-1] = hc.y;
            V1p[CW-1] = h_pos(r3p,r1p,r1n,r3n, c + u, u);
            V1m[CW-1] = h_neg(r3p,r1p,r1n,r3n, c - u, u);
            G0p[CW-1] = h_pos(S3p,S1p,S1n,S3n, c + uh,  u);
            G0m[CW-1] = h_neg(S3p,S1p,S1n,S3n, c + lo0, u);
        }
    }
}

extern "C" __global__ void __launch_bounds__(LWS_HP, MINB_HP)
interp_hp(io_t * __restrict__ dst, const float * __restrict__ srcpad,
                                     const int * __restrict__ rowidx, const int * __restrict__ dst_y,
                                     schar * __restrict__ pbackt, int * __restrict__ dmap,
                                     const int w, const int stride, const int pstride, const int pad,
                                     const int mdis, const int nrad,
                                     const float alpha3, const float beta255, const float gamma255,
                                     const float one_minus_ab,
                                     const float * __restrict__ hpsrcpad,
                                     io_t * __restrict__ dst2, const int dual,
                                     const int doff, const int moff) {
    const int off = blockIdx.y;
    const int tid = threadIdx.x;
    const int lsz = blockDim.x;
    const int cen = 2*MDIS;
    const float * __restrict__ r3p = srcpad + (long long)rowidx[off*4+0]*pstride + pad;
    const float * __restrict__ r1p = srcpad + (long long)rowidx[off*4+1]*pstride + pad;
    const float * __restrict__ r1n = srcpad + (long long)rowidx[off*4+2]*pstride + pad;
    const float * __restrict__ r3n = srcpad + (long long)rowidx[off*4+3]*pstride + pad;
    const float * __restrict__ hp3p = hpsrcpad + (long long)rowidx[off*4+0]*pstride + pad;
    const float * __restrict__ hp1p = hpsrcpad + (long long)rowidx[off*4+1]*pstride + pad;
    const float * __restrict__ hp1n = hpsrcpad + (long long)rowidx[off*4+2]*pstride + pad;
    const float * __restrict__ hp3n = hpsrcpad + (long long)rowidx[off*4+3]*pstride + pad;
    const int pb_pitch = (w * TPH + 15) & ~15;
    schar *pb = pbackt + (long long)off * pb_pitch;
    __shared__ float pc[2][TPMAX_HP];
    __shared__ __align__(16) float cst[BX * TPH];
    const int u = tid - cen;
    const int active = (tid < TPH);
    if (tid == 0) { for (int b=0;b<2;b++){ pc[b][0]=FLTMAX9; pc[b][1]=FLTMAX9; pc[b][TPH+2]=FLTMAX9; pc[b][TPH+3]=FLTMAX9; } }
    __syncthreads();
    int ping = 0;
    if (active) pc[ping][tid+2] = conn_cost_hp(r3p,r1p,r1n,r3n, hp3p,hp1p,hp1n,hp3n, 0, u, w, nrad, alpha3, beta255, one_minus_ab);
    __syncthreads();
    const float g1 = gamma255*0.5f, g2 = gamma255;
    for (int x0 = 1; x0 < w; x0 += BX) {
        const int bn = min(BX, w - x0);
        #define NEVU (MDIS + 1)
        #define NODU (MDIS)
        const int nrun = (bn + RUN_HP - 1) / RUN_HP;
        for (int i = tid; i < nrun*NEVU; i += lsz) {
            const int r = i / NEVU;
            const int j = i - r*NEVU;
            const int xs = x0 + r*RUN_HP;
            const int rl = min(RUN_HP, bn - r*RUN_HP);
            float *co = cst + (r*RUN_HP)*TPH;
            if (j == 0) {
                for (int t = 0; t < rl; t++)
                    co[t*TPH + cen] = conn_cost_hp(r3p,r1p,r1n,r3n, hp3p,hp1p,hp1n,hp3n, xs+t, 0, w, nrad, alpha3, beta255, one_minus_ab);
            } else {
                cost_hp_pair_run(r3p,r1p,r1n,r3n, r3p,r1p,r1n,r3n, r1p,r1n,
                                 co, xs, rl, 2*j, alpha3, beta255, one_minus_ab);
            }
        }
        for (int i = tid; i < nrun*NODU; i += lsz) {
            const int r = i / NODU;
            const int j = i - r*NODU;
            const int xs = x0 + r*RUN_HP;
            const int rl = min(RUN_HP, bn - r*RUN_HP);
            cost_hp_pair_run(r3p,r1p,r1n,r3n, hp3p,hp1p,hp1n,hp3n, hp1p,hp1n,
                             cst + (r*RUN_HP)*TPH, xs, rl, 2*j + 1, alpha3, beta255, one_minus_ab);
        }
        __syncthreads();
        for (int dx = 0; dx < bn; dx++) {
            const int pong = ping ^ 1;
            if (active) {
                const float cost = cst[dx*TPH + tid];
                float c_m2 = fminf(pc[ping][tid+0]+g2, FLTMAX9);
                float c_m1 = fminf(pc[ping][tid+1]+g1, FLTMAX9);
                float c_0  = fminf(pc[ping][tid+2],    FLTMAX9);
                float c_p1 = fminf(pc[ping][tid+3]+g1, FLTMAX9);
                float c_p2 = fminf(pc[ping][tid+4]+g2, FLTMAX9);
                float bval = c_m2; schar bd = -2;
                if (c_m1 < bval) { bval = c_m1; bd = -1; }
                if (c_0  < bval) { bval = c_0;  bd =  0; }
                if (c_p1 < bval) { bval = c_p1; bd =  1; }
                if (c_p2 < bval) { bval = c_p2; bd =  2; }
                pc[pong][tid+2] = fminf(bval + cost, FLTMAX9);
                pb[(x0+dx-1)*TPH + tid] = bd;
            }
            __syncthreads();
            ping = pong;
        }
    }
    __syncthreads();
    int *dm = dmap + moff + (long long)off*stride;
    {
        schar *stg = (schar *)cst;
        int f = 0;
        if (tid == 0) dm[w-1] = 0;
        for (int k = (w - 2) / (4*BX); k >= 0; k--) {
            const int xlo = k * (4*BX);
            const int xhi = min(xlo + 4*BX - 1, w - 2);
            const int nb = (xhi - xlo + 1) * TPH;
#if (4*BX*TPH) % 16 == 0
            const uint4 *gsrc = (const uint4 *)(pb + xlo*TPH);
            uint4 *sdst = (uint4 *)stg;
            const int nv = nb >> 4;
            for (int i = tid; i < nv; i += lsz) sdst[i] = gsrc[i];
            for (int i = (nv << 4) + tid; i < nb; i += lsz) stg[i] = pb[xlo*TPH + i];
#else
            for (int i = tid; i < nb; i += lsz) stg[i] = pb[xlo*TPH + i];
#endif
            __syncthreads();
            if (tid == 0) {
                for (int bx = xhi; bx >= xlo; bx--) {
                    f += (int)stg[(bx - xlo)*TPH + (cen + f)];
                    dm[bx] = f;
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();
    io_t *drow = dst + doff + (long long)dst_y[off]*stride;
    io_t *drow2 = dst2 + doff + (long long)dst_y[off]*stride;
    for (int x = tid; x < w; x += lsz) {
        const int dir = dm[x];
        float val;
        if ((dir & 1) == 0) {
            int d2 = dir >> 1; int ad = abs(d2);
            if (x >= ad*3 && x + ad*3 <= w-1)
                val = 0.5625f*(R(r1p,x+d2)+R(r1n,x-d2)) - 0.0625f*(R(r3p,x+3*d2)+R(r3n,x-3*d2));
            else
                val = (R(r1p,x+d2)+R(r1n,x-d2))*0.5f;
        } else {
            int d20 = dir>>1, d21 = (dir+1)>>1, d30 = (dir*3)>>1, d31 = (dir*3+1)>>1;
            int ad = max(abs(d30), abs(d31));
            if (x >= ad && x + ad <= w-1) {
                float c0 = R(r3p,x+d30)+R(r3p,x+d31);
                float c1 = R(r1p,x+d20)+R(r1p,x+d21);
                float c2 = R(r1n,x-d20)+R(r1n,x-d21);
                float c3 = R(r3n,x-d30)+R(r3n,x-d31);
                val = 0.28125f*(c1+c2) - 0.03125f*(c0+c3);
            } else {
                val = (R(r1p,x+d20)+R(r1p,x+d21)+R(r1n,x-d20)+R(r1n,x-d21))*0.25f;
            }
        }
        STORE_IO(drow, x, val);
        if (dual) STORE_IO(drow2, x, val);
    }
}

extern "C" __global__ void __launch_bounds__(VC_WG)
vcheck(io_t *out_b, const io_t *dst_b, const io_t *src_b,
       const int *rowidx0, const int *rowidx1, const int *rowidx2,
       const int *dmap_b, const io_t *scp_b,
       const int *geom, const int field,
       const int vmode, const int use_scp, const int hp,
       const float rcp0, const float rcp1, const float rcp2, const float vthresh2,
       const int nplanes) {
    const int p = blockIdx.x;
    if (p >= nplanes) return;
    const int w = geom[p*6+0], stride = geom[p*6+1], dst_h = geom[p*6+2];
    io_t *out = out_b + geom[p*6+3];
    const io_t *dst = dst_b + geom[p*6+3];
    const io_t *src = src_b + geom[p*6+4];
    const int *dmap = dmap_b + geom[p*6+5];
    const io_t *scp = scp_b + geom[p*6+3];
    const int *rowidx = (p == 0) ? rowidx0 : ((p == 1) ? rowidx1 : rowidx2);
    const int n_interp = (dst_h - field + 1) / 2;
    const int lid = threadIdx.x;
    const int lsz = blockDim.x;
    for (int off = 1; off + 1 < n_interp; ++off) {
        const int y = field + 2*off;
        if (y >= 2 && y + 2 < dst_h) {
            const io_t *drow = dst + (long long)y*stride;
            const io_t *d1p = dst + (long long)(y-1)*stride;
            const io_t *d2p = out + (long long)(y-2)*stride;
            const io_t *d1n = dst + (long long)(y+1)*stride;
            const io_t *d2n = dst + (long long)(y+2)*stride;
            const io_t *d3p = src + (long long)rowidx[off*4+0]*stride;
            const io_t *d3n = src + (long long)rowidx[off*4+3]*stride;
            for (int x = lid; x < w; x += lsz) {
                const int dirc = dmap[(long long)off*stride + x];
                float cint = use_scp ? LOAD_IO(scp, (long long)y*stride + x) : (0.5625f*(LOAD_IO(d1p,x)+LOAD_IO(d1n,x)) - 0.0625f*(LOAD_IO(d3p,x)+LOAD_IO(d3n,x)));
                int dirt = dmap[(long long)(off-1)*stride + x];
                int dirb = dmap[(long long)(off+1)*stride + x];
                int maxoff = hp ? ((dirc & 1) == 0 ? abs(dirc>>1) : max(abs(dirc>>1), abs((dirc+1)>>1))) : abs(dirc);
                if (dirc == 0 || max(dirc*dirt, dirc*dirb) < 0 || (dirt==dirb && dirt==0)
                    || x + maxoff >= w || x - maxoff < 0) {
                    STORE_IO(out, (long long)y*stride + x, cint);
                    continue;
                }
                float it, ib, vt, vb;
                int dabs;
                if (hp && (dirc & 1) != 0) {
                    int d20 = dirc>>1, d21 = (dirc+1)>>1;
                    int xp0 = x+d20, xp1 = x+d21, xm0 = x-d20, xm1 = x-d21;
                    float s2psum = LOAD_IO(d2p,xp0)+LOAD_IO(d2p,xp1), s1psum = LOAD_IO(d1p,xp0)+LOAD_IO(d1p,xp1);
                    float pa0 = LOAD_IO(drow,xp0)+LOAD_IO(drow,xp1), ps0 = LOAD_IO(drow,xm0)+LOAD_IO(drow,xm1);
                    float s1nsum = LOAD_IO(d1n,xm0)+LOAD_IO(d1n,xm1), s2nsum = LOAD_IO(d2n,xm0)+LOAD_IO(d2n,xm1);
                    it = (s2psum + ps0)*0.25f;
                    vt = (fabsf(s2psum-s1psum) + fabsf(pa0-s1psum))*0.5f;
                    ib = (pa0 + s2nsum)*0.25f;
                    vb = (fabsf(s2nsum-s1nsum) + fabsf(ps0-s1nsum))*0.5f;
                    dabs = abs(dirc) >> 1;
                } else {
                    int offh = hp ? (dirc>>1) : dirc;
                    int xpd = x+offh, xmd = x-offh;
                    it = (LOAD_IO(d2p,xpd) + LOAD_IO(drow,xmd)) * 0.5f;
                    ib = (LOAD_IO(drow,xpd) + LOAD_IO(d2n,xmd)) * 0.5f;
                    vt = fabsf(LOAD_IO(d2p,xpd)-LOAD_IO(d1p,xpd)) + fabsf(LOAD_IO(drow,xpd)-LOAD_IO(d1p,xpd));
                    vb = fabsf(LOAD_IO(d2n,xmd)-LOAD_IO(d1n,xmd)) + fabsf(LOAD_IO(drow,xmd)-LOAD_IO(d1n,xmd));
                    dabs = hp ? (abs(dirc)>>1) : abs(dirc);
                }
                float vc = fabsf(LOAD_IO(drow,x)-LOAD_IO(d1p,x)) + fabsf(LOAD_IO(drow,x)-LOAD_IO(d1n,x));
                float d0 = fabsf(it-LOAD_IO(d1p,x)), d1 = fabsf(ib-LOAD_IO(d1n,x)), d2 = fabsf(vt-vc), d3 = fabsf(vb-vc);
                float mdiff0 = (vmode==1)?fminf(d0,d1):(vmode==2)?(d0+d1)*0.5f:fmaxf(d0,d1);
                float mdiff1 = (vmode==1)?fminf(d2,d3):(vmode==2)?(d2+d3)*0.5f:fmaxf(d2,d3);
                float a0 = mdiff0*rcp0, a1 = mdiff1*rcp1;
                float a2 = fmaxf((vthresh2 - (float)dabs)*rcp2, 0.0f);
                float a = fminf(fmaxf(a0, fmaxf(a1,a2)), 1.0f);
                /* Blend shape pinned: fma(1-a, drow, a*cint). Other contraction drifts 1 ulp. */
                STORE_IO(out, (long long)y*stride + x, __fmaf_rn(1.0f-a, LOAD_IO(drow,x), __fmul_rn(a, cint)));
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void transpose(raw_t *out, const raw_t *in,
                                     const int in_w, const int in_h, const int in_stride, const int out_stride,
                                     const int soff, const int doff) {
    __shared__ raw_t tile[16][17];
    const int lx = threadIdx.x, ly = threadIdx.y;
    const int c0 = blockIdx.x * 16, r0 = blockIdx.y * 16;
    const int c = c0 + lx, r = r0 + ly;
    if (c < in_w && r < in_h) tile[ly][lx] = in[(long long)soff + (long long)r*in_stride + c];
    __syncthreads();
    const int oc = c0 + ly, orr = r0 + lx;
    if (oc < in_w && orr < in_h) out[(long long)doff + (long long)oc*out_stride + orr] = tile[lx][ly];
}
