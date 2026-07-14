// -fmad=false: prescreener no contract; predictor only at explicit __fmaf_rn.

#define MARGIN_H 24
#define MARGIN_V 3
#define FS (XDIM * YDIM)

#if PIXEL_TYPE == 0
#define pix_t unsigned char
#define INT_FMT 1
#elif PIXEL_TYPE == 1
#define pix_t unsigned short
#define INT_FMT 1
#elif PIXEL_TYPE == 2
#define pix_t unsigned short /* f16 bit pattern; PTX convert */
#define INT_FMT 0
#else
#define pix_t float
#define INT_FMT 0
#endif

__device__ __forceinline__ float loadPix(pix_t v) {
#if PIXEL_TYPE == 2
    float f;
    unsigned short u = v;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(u));
    return f;
#else
    return (float) v;
#endif
}

__device__ __forceinline__ pix_t storePix(float v) {
#if INT_FMT
    return (pix_t) (unsigned int) rintf(fminf(fmaxf(v, 0.0f), (float) PEAK));
#elif PIXEL_TYPE == 2
    unsigned short u;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(u) : "f"(v));
    return u;
#else
    return v;
#endif
}

#if EXP_APPROX
#define EXPF(x) __expf(x)
#else
#define EXPF(x) expf(x)
#endif
#if DIV_APPROX
#define DIVF(a, b) __fdividef((a), (b))
#else
#define DIVF(a, b) ((a) / (b))
#endif
#if SQRT_APPROX
__device__ __forceinline__ float sqrt_approx(float x) { float r; asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
#define SQRTF(x) sqrt_approx(x)
#else
#define SQRTF(x) sqrtf(x)
#endif
#if RCP_APPROX
__device__ __forceinline__ float rcp_approx(float x) { float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
#define RCPF(x) rcp_approx(x)
#else
#define RCPF(x) (1.0f / (x))
#endif

// Prescreener Elliott: always IEEE divide.
__device__ __forceinline__ float elliottPre(float x) {
    return x / (1.0f + fabsf(x));
}

__device__ __forceinline__ float elliott(float x) {
    return DIVF(x, 1.0f + fabsf(x));
}

__device__ __forceinline__ void computeMstd(float mean, float variance, float & mstd0, float & mstd1, float & mstd2) {
    mstd0 = mean;
    if (variance < 1.1920929e-7f) {
        mstd1 = 0.0f;
        mstd2 = 0.0f;
    } else {
        mstd1 = SQRTF(variance);
        mstd2 = RCPF(mstd1);
    }
}

__device__ __forceinline__ void wae5Accum(float actS, float actE, float & vsum, float & wsum) {
    const float e = EXPF(fminf(fmaxf(actS, -80.0f), 80.0f));
    vsum = __fmaf_rn(e, elliott(actE), vsum);
    wsum += e;
}

__device__ __forceinline__ float wae5Blend(float vsum, float wsum, float mstd0, float mstd1) {
    return wsum > 1e-10f ? __fmaf_rn(DIVF(5.0f * vsum, wsum), mstd1, mstd0) : mstd0;
}

__device__ __forceinline__ void windowOrigin(unsigned int idx, int width, int & col0, int & row0) {
    const int r = (int) idx / width;
    const int x = (int) idx % width;
    col0 = x - (XDIM / 2 - 1) + MARGIN_H;
    row0 = r + (YDIM == 6 ? 0 : 1);
}

extern "C" __global__ void pad(
    pix_t * __restrict__ padBuf,
    const pix_t * __restrict__ field,
    int width,
    int rows,
    int fieldStride,
    int padStride,
    int fp
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int padH = rows + MARGIN_V * 2;
    const int padW = width + MARGIN_H * 2;
    if (x >= padW || i >= padH)
        return;
    const int f = min(max(i - (MARGIN_V - fp), 0), rows - 1);
    const int c = min(max(x - MARGIN_H, 0), width - 1);
    padBuf[i * padStride + x] = field[f * fieldStride + c];
}

__device__ __forceinline__ float loadPad(const pix_t * __restrict__ padBuf, int padStride, int row, int col) {
    return loadPix(padBuf[row * padStride + col]);
}

__device__ __forceinline__ float cubic(const pix_t * __restrict__ padBuf, int padStride, int r, int col) {
    float acc = 0.0f;
    acc += (-3.0f / 32.0f) * loadPad(padBuf, padStride, r + 1, col);
    acc += (19.0f / 32.0f) * loadPad(padBuf, padStride, r + 2, col);
    acc += (19.0f / 32.0f) * loadPad(padBuf, padStride, r + 3, col);
    acc += (-3.0f / 32.0f) * loadPad(padBuf, padStride, r + 4, col);
    return acc;
}

extern "C" __global__ void prescreen(
    pix_t * __restrict__ dstBuf,
    const pix_t * __restrict__ padBuf,
    const float4 * __restrict__ psW4,
    unsigned int * __restrict__ listBuf,
    unsigned int * __restrict__ predCount,
    int width,
    int rows,
    int padStride
) {
#if PSCRN == 1
    const int P = 1;
#else
    const int P = 4;
#endif
    const int xg = (width + P - 1) / P;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = gid / xg;
    const int xbase = (gid % xg) * P;
    const bool valid = r < rows;

    unsigned int needMask = 0u;

    if (valid) {
#if PSCRN == 1
        float st0 = 0.0f, st1 = 0.0f, st2 = 0.0f, st3 = 0.0f;
        for (int k = 0; k < 48; k++) {
            const float v = loadPad(padBuf, padStride, r + 1 + k / 12, xbase - 5 + k % 12 + MARGIN_H);
            const float4 w = psW4[k];
            st0 += w.x * v;
            st1 += w.y * v;
            st2 += w.z * v;
            st3 += w.w * v;
        }

        const float4 b0 = psW4[48];
        float st[8];
        st[0] = st0 + b0.x;
        st[1] = elliottPre(st1 + b0.y);
        st[2] = elliottPre(st2 + b0.z);
        st[3] = elliottPre(st3 + b0.w);

        const float4 b1 = psW4[53];
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            const float4 w = psW4[49 + n];
            float acc = 0.0f;
            acc += w.x * st[0];
            acc += w.y * st[1];
            acc += w.z * st[2];
            acc += w.w * st[3];
            const float bn = n == 0 ? b1.x : n == 1 ? b1.y : n == 2 ? b1.z : b1.w;
            st[4 + n] = elliottPre(acc + bn);
        }

        const float4 b2 = psW4[62];
        float l2[4];
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            const float4 wa = psW4[54 + n * 2];
            const float4 wb = psW4[55 + n * 2];
            float acc = 0.0f;
            acc += wa.x * st[0];
            acc += wa.y * st[1];
            acc += wa.z * st[2];
            acc += wa.w * st[3];
            acc += wb.x * st[4];
            acc += wb.y * st[5];
            acc += wb.z * st[6];
            acc += wb.w * st[7];
            const float bn = n == 0 ? b2.x : n == 1 ? b2.y : n == 2 ? b2.z : b2.w;
            l2[n] = acc + bn;
        }

        if (fmaxf(l2[2], l2[3]) > fmaxf(l2[0], l2[1]))
            needMask = 1u;
#else
        float st0 = 0.0f, st1 = 0.0f, st2 = 0.0f, st3 = 0.0f;
        for (int k = 0; k < 64; k++) {
            const float v = loadPad(padBuf, padStride, r + 1 + k / 16, xbase - 6 + k % 16 + MARGIN_H);
            const float4 w = psW4[k];
            st0 += w.x * v;
            st1 += w.y * v;
            st2 += w.z * v;
            st3 += w.w * v;
        }

        const float4 b0 = psW4[64];
        float st[4];
        st[0] = elliottPre(st0 + b0.x);
        st[1] = elliottPre(st1 + b0.y);
        st[2] = elliottPre(st2 + b0.z);
        st[3] = elliottPre(st3 + b0.w);

        const float4 b1 = psW4[69];
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            const float4 w = psW4[65 + n];
            float acc = 0.0f;
            acc += w.x * st[0];
            acc += w.y * st[1];
            acc += w.z * st[2];
            acc += w.w * st[3];
            const float bn = n == 0 ? b1.x : n == 1 ? b1.y : n == 2 ? b1.z : b1.w;
            if (!(acc + bn > 0.0f))
                needMask |= 1u << n;
        }
#endif
    }

    int cnt = 0;
    unsigned int pixIdx[4];
    if (valid) {
        #pragma unroll
        for (int n = 0; n < P; n++) {
            const int x = xbase + n;
            if (x >= width)
                break;
            const unsigned int idx = (unsigned int) r * (unsigned int) width + (unsigned int) x;
            if ((needMask & (1u << n)) != 0u)
                pixIdx[cnt++] = idx;
            else
                dstBuf[idx] = storePix(cubic(padBuf, padStride, r, x + MARGIN_H));
        }
    }

    const unsigned int active = 0xffffffffu;
    unsigned int total = (unsigned int) cnt;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        total += __shfl_xor_sync(active, total, o);
    unsigned int base = 0u;
    if ((threadIdx.x & 31u) == 0u && total > 0u)
        base = atomicAdd(predCount, total);
    base = __shfl_sync(active, base, 0);
    unsigned int off = (unsigned int) cnt;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned int t = __shfl_up_sync(active, off, o);
        if ((threadIdx.x & 31u) >= (unsigned int) o)
            off += t;
    }
    off = base + off - (unsigned int) cnt;
    for (int c = 0; c < cnt; c++)
        listBuf[off + c] = pixIdx[c];
}

#define SGSIZE 32
#define PPL ((NNS + SGSIZE - 1) / SGSIZE)
#define EPL ((FS + SGSIZE - 1) / SGSIZE)
#define PX 4
#define WARPS 4

__device__ __forceinline__ float warpAdd(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__ __launch_bounds__(WARPS * 32) void predict(
    pix_t * __restrict__ dstBuf,
    const pix_t * __restrict__ padBuf,
    const float2 * __restrict__ pdW,
    const float2 * __restrict__ pdB,
    const unsigned int * __restrict__ listBuf,
    const unsigned int * __restrict__ predCount,
    int width,
    int rows,
    int padStride
) {
    __shared__ float4 shIn[WARPS * (PX / 4) * FS];

    const unsigned int npix = listBuf != 0 ? *predCount : (unsigned int) (width * rows);
    const int warp = (int) (threadIdx.x / 32u);
    const unsigned int firstPix = ((unsigned int) blockIdx.x * WARPS + warp) * PX;
    if (firstPix >= npix)
        return;
    const int npx = (int) min((unsigned int) PX, npix - firstPix);

    const int lane = (int) (threadIdx.x & 31u);

    unsigned int idxs[PX];
    float mstd0[PX], mstd1[PX], mstd2[PX], result[PX];

    #pragma unroll
    for (int px = 0; px < PX; px++) {
        const unsigned int li = min(firstPix + (unsigned int) px, npix - 1u);
        const unsigned int idx = listBuf != 0 ? listBuf[li] : li;
        idxs[px] = idx;

        int col0, row0;
        windowOrigin(idx, width, col0, row0);

        const int shBase = (warp * (PX / 4) + px / 4) * FS;
        float v[EPL];
        float sum = 0.0f;
        #pragma unroll
        for (int t = 0; t < EPL; t++) {
            const int e = lane + t * SGSIZE;
            if (FS % SGSIZE != 0 && e >= FS)
                continue;
            v[t] = loadPad(padBuf, padStride, row0 + e / XDIM, col0 + e % XDIM);
            sum += v[t];
        }
        const float mean = DIVF(warpAdd(sum), (float) FS);

        float s2 = 0.0f;
        #pragma unroll
        for (int t = 0; t < EPL; t++) {
            const int e = lane + t * SGSIZE;
            if (FS % SGSIZE != 0 && e >= FS)
                continue;
            const float d = v[t] - mean;
            s2 += d * d;
        }
        const float variance = DIVF(warpAdd(s2), (float) FS);

        computeMstd(mean, variance, mstd0[px], mstd1[px], mstd2[px]);
        result[px] = 0.0f;

        #pragma unroll
        for (int t = 0; t < EPL; t++) {
            const int e = lane + t * SGSIZE;
            if (FS % SGSIZE != 0 && e >= FS)
                continue;
            ((float *) &shIn[shBase + e])[px % 4] = v[t] * mstd2[px];
        }
    }
    __syncwarp();

    #pragma unroll
    for (int q = 0; q < QUAL; q++) {
        const int bBase = q * 2 * NNS;
        const int wBase = q * FS * NNS;
        const int shSub = warp * (PX / 4) * FS;

        float4 accSa[PPL], accEa[PPL];
        #pragma unroll
        for (int t = 0; t < PPL; t++) {
            accSa[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            accEa[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        for (int k = 0; k < FS; k++) {
            const float4 xa = shIn[shSub + k];
            #pragma unroll
            for (int t = 0; t < PPL; t++) {
                const int p = lane + t * SGSIZE;
                if (NNS % SGSIZE != 0 && p >= NNS)
                    continue;
                const float2 w = pdW[wBase + k * NNS + p];
                accSa[t].x = __fmaf_rn(w.x, xa.x, accSa[t].x);
                accSa[t].y = __fmaf_rn(w.x, xa.y, accSa[t].y);
                accSa[t].z = __fmaf_rn(w.x, xa.z, accSa[t].z);
                accSa[t].w = __fmaf_rn(w.x, xa.w, accSa[t].w);
                accEa[t].x = __fmaf_rn(w.y, xa.x, accEa[t].x);
                accEa[t].y = __fmaf_rn(w.y, xa.y, accEa[t].y);
                accEa[t].z = __fmaf_rn(w.y, xa.z, accEa[t].z);
                accEa[t].w = __fmaf_rn(w.y, xa.w, accEa[t].w);
            }
        }

        #pragma unroll
        for (int px = 0; px < PX; px++) {
            if (px >= npx)
                break;

            float vsum = 0.0f, wsum = 0.0f;
            #pragma unroll
            for (int t = 0; t < PPL; t++) {
                const int p = lane + t * SGSIZE;
                if (NNS % SGSIZE != 0 && p >= NNS)
                    continue;
                const float2 bias = pdB[bBase + p];
                const float actS = (px == 0 ? accSa[t].x : px == 1 ? accSa[t].y : px == 2 ? accSa[t].z : accSa[t].w) + bias.x;
                const float actE = (px == 0 ? accEa[t].x : px == 1 ? accEa[t].y : px == 2 ? accEa[t].z : accEa[t].w) + bias.y;
                wae5Accum(actS, actE, vsum, wsum);
            }
            vsum = warpAdd(vsum);
            wsum = warpAdd(wsum);

            result[px] += wae5Blend(vsum, wsum, mstd0[px], mstd1[px]);
        }
    }

    if (lane == 0)
        #pragma unroll
        for (int px = 0; px < PX; px++) {
            if (px >= npx)
                break;
            dstBuf[idxs[px]] = storePix(DIVF(result[px], (float) QUAL));
        }
}

#if MMA
#define KT (FS / 16)
#define NT (2 * NNS / 8)
#define NTC (NT < 8 ? NT : 8)
// +4 is deliberate (not +1): bank-conflict-free A-fragment loads.
#define SH_STRIDE (FS / 2 + 4)

extern "C" __global__ __launch_bounds__(128) void predict_mma(
    pix_t * __restrict__ dstBuf,
    const pix_t * __restrict__ padBuf,
    const unsigned int * __restrict__ pdWmma,
    const float2 * __restrict__ pdB,
    const unsigned int * __restrict__ listBuf,
    const unsigned int * __restrict__ predCount,
    int width,
    int rows,
    int padStride
) {
    __shared__ unsigned int shWin[4][16 * SH_STRIDE];
    __shared__ float shMean[4][16];
    __shared__ float shScale[4][16];
    __shared__ float shM0[4][16];
    __shared__ float shM1[4][16];
    __shared__ unsigned int shIdx[4][16];

    const unsigned int npix = listBuf != 0 ? *predCount : (unsigned int) (width * rows);
    const int warp = (int) (threadIdx.x / 32u);
    const unsigned int firstPix = ((unsigned int) blockIdx.x * 4 + warp) * 16;
    if (firstPix >= npix)
        return;
    const int lane = (int) (threadIdx.x & 31u);

    {
        const int px = lane / 2;
        const int half = lane & 1;
        const unsigned int li = min(firstPix + (unsigned int) px, npix - 1u);
        const unsigned int idx = listBuf != 0 ? listBuf[li] : li;
        int col0, row0;
        windowOrigin(idx, width, col0, row0);

        float sum = 0.0f;
        for (int e = half; e < FS; e += 2)
            sum += loadPad(padBuf, padStride, row0 + e / XDIM, col0 + e % XDIM);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        const float mean = DIVF(sum, (float) FS);

        float s2 = 0.0f;
        for (int e = half; e < FS; e += 2) {
            const float d = loadPad(padBuf, padStride, row0 + e / XDIM, col0 + e % XDIM) - mean;
            s2 += d * d;
        }
        s2 += __shfl_xor_sync(0xffffffffu, s2, 1);

        float m0, m1, m2;
        computeMstd(mean, DIVF(s2, (float) FS), m0, m1, m2);
        if (half == 0) {
            shMean[warp][px] = mean;
            shScale[warp][px] = m2;
            shM0[warp][px] = m0;
            shM1[warp][px] = m1;
            shIdx[warp][px] = idx;
        }
        for (int e2 = half; e2 < FS / 2; e2 += 2) {
            const int e = e2 * 2;
            const float f0 = (loadPad(padBuf, padStride, row0 + (e + 0) / XDIM, col0 + (e + 0) % XDIM) - mean) * m2;
            const float f1 = (loadPad(padBuf, padStride, row0 + (e + 1) / XDIM, col0 + (e + 1) % XDIM) - mean) * m2;
            unsigned int packed;
            asm("{ .reg .f16 lo, hi; cvt.rn.f16.f32 lo, %1; cvt.rn.f16.f32 hi, %2; mov.b32 %0, {lo, hi}; }"
                : "=r"(packed) : "f"(f0), "f"(f1));
            shWin[warp][px * SH_STRIDE + e2] = packed;
        }
    }
    __syncwarp();

    const int r1 = lane / 4;
    const int r2 = r1 + 8;
    const int cpair = lane % 4;

    float res1 = 0.0f, res2 = 0.0f;

    #pragma unroll
    for (int q = 0; q < QUAL; q++) {
        const int bBase = q * 2 * NNS;
        float vs1 = 0.0f, ws1 = 0.0f, vs2 = 0.0f, ws2 = 0.0f;

        for (int nc = 0; nc < NT; nc += NTC) {
            float c[NTC][4];
            #pragma unroll
            for (int j = 0; j < NTC; j++) {
                c[j][0] = 0.0f; c[j][1] = 0.0f; c[j][2] = 0.0f; c[j][3] = 0.0f;
            }

            for (int kt = 0; kt < KT; kt++) {
                unsigned int a0 = shWin[warp][r1 * SH_STRIDE + kt * 8 + cpair];
                unsigned int a1 = shWin[warp][r2 * SH_STRIDE + kt * 8 + cpair];
                unsigned int a2 = shWin[warp][r1 * SH_STRIDE + kt * 8 + 4 + cpair];
                unsigned int a3 = shWin[warp][r2 * SH_STRIDE + kt * 8 + 4 + cpair];

                const unsigned int * bbase = &pdWmma[(((unsigned int) q * KT + kt) * NT + nc) * 64 + (unsigned int) lane * 2];
                #pragma unroll
                for (int j = 0; j < NTC; j++) {
                    const unsigned int b0 = bbase[j * 64 + 0];
                    const unsigned int b1 = bbase[j * 64 + 1];
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }

            const float corr1 = shMean[warp][r1] * shScale[warp][r1];
            const float corr2 = shMean[warp][r2] * shScale[warp][r2];
            #pragma unroll
            for (int j = 0; j < NTC; j++) {
                const int p = (nc + j) * 4 + cpair;
                const float2 bias = pdB[bBase + p];
                const float2 rowSum = pdB[bBase + NNS + p];

                const float actS1 = c[j][0] + __fmaf_rn(rowSum.x, corr1, bias.x);
                const float actE1 = c[j][1] + __fmaf_rn(rowSum.y, corr1, bias.y);
                wae5Accum(actS1, actE1, vs1, ws1);

                const float actS2 = c[j][2] + __fmaf_rn(rowSum.x, corr2, bias.x);
                const float actE2 = c[j][3] + __fmaf_rn(rowSum.y, corr2, bias.y);
                wae5Accum(actS2, actE2, vs2, ws2);
            }
        }

        #pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            vs1 += __shfl_xor_sync(0xffffffffu, vs1, o);
            ws1 += __shfl_xor_sync(0xffffffffu, ws1, o);
            vs2 += __shfl_xor_sync(0xffffffffu, vs2, o);
            ws2 += __shfl_xor_sync(0xffffffffu, ws2, o);
        }
        res1 += wae5Blend(vs1, ws1, shM0[warp][r1], shM1[warp][r1]);
        res2 += wae5Blend(vs2, ws2, shM0[warp][r2], shM1[warp][r2]);
    }

    if (cpair == 0) {
        if (firstPix + (unsigned int) r1 < npix)
            dstBuf[shIdx[warp][r1]] = storePix(DIVF(res1, (float) QUAL));
        if (firstPix + (unsigned int) r2 < npix)
            dstBuf[shIdx[warp][r2]] = storePix(DIVF(res2, (float) QUAL));
    }
}
#endif // MMA
