// No -use_fast_math (-ftz flushes denormals); ascending temporal sum only (float add not associative).

#define TEMPORAL_STRIDE (HEIGHT * STRIDE)
#define TEMPORAL_WIDTH (2 * RADIUS + 1)
#define MAX_TW 33

struct AggSrc {
    const float *p[MAX_TW];
    int z[MAX_TW];
};

// BY VALUE: MODULE layout must match bm3d.zig — mismatch = silent wrong pointers.
static_assert(sizeof(AggSrc) == 400, "AggSrc ABI must match bm3d.zig");
static_assert(alignof(AggSrc) == 8, "AggSrc ABI must match bm3d.zig");
static_assert(sizeof(((AggSrc *)0)->p) == 264, "AggSrc ABI must match bm3d.zig");
static_assert(sizeof(((AggSrc *)0)->z) == 132, "AggSrc ABI must match bm3d.zig");

extern "C" __global__ __launch_bounds__(256) void vaggregate(
    /* [NUM_PLANES, HEIGHT, STRIDE] */
    float *__restrict__ dst,
    const AggSrc s) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    // Skipped sigma planes: do not write 0/0 NaN.
    const int plane = blockIdx.z;
    if (!((PROC_MASK >> plane) & 1)) {
        return;
    }

    const int i = y * STRIDE + x;
    const int plane_base = plane * TEMPORAL_WIDTH * 2 * TEMPORAL_STRIDE;

    float sum_v = 0.0f;
    float sum_w = 0.0f;
#pragma unroll
    for (int t = 0; t < TEMPORAL_WIDTH; ++t) {
        const float *wdst = s.p[t] + plane_base + s.z[t] * 2 * TEMPORAL_STRIDE;
        sum_v += wdst[i];
        sum_w += wdst[TEMPORAL_STRIDE + i];
    }

    dst[plane * TEMPORAL_STRIDE + i] = __fdiv_rn(sum_v, sum_w);
}
