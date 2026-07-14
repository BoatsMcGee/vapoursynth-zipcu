/*
Bilateral kernel (VapourSynth-BilateralGPU). Do not alter math or border logic.
SMEM_W pads shared-memory pitch for bank conflicts; host sizes dynamic smem the same way.
*/

#define SMEM_W_RAW (2 * radius + BLOCK_X)
#if BLOCK_X < 32
#define SMEM_W (SMEM_W_RAW + ((((int)BLOCK_X) - (SMEM_W_RAW & 31) + 32) & 31))
#else
#define SMEM_W SMEM_W_RAW
#endif

__device__ static const dim3 BlockDim = dim3(BLOCK_X, BLOCK_Y);

extern "C"
__global__
__launch_bounds__(BLOCK_X * BLOCK_Y)
void bilateral(
    float * __restrict__ dst, const float * __restrict__ src
) {

    const int x = threadIdx.x + blockIdx.x * BLOCK_X;
    const int y = threadIdx.y + blockIdx.y * BLOCK_Y;

    float num {};
    float den {};

    if constexpr (use_shared_memory) {
        extern __shared__ float buffer[
            /* (1 + has_ref) * (2 * radius + BLOCK_Y) * SMEM_W */];

        for (int cy = threadIdx.y; cy < 2 * radius + BLOCK_Y; cy += BLOCK_Y) {
            int sy = min(max(cy - static_cast<int>(threadIdx.y) - radius + y, 0), height - 1);
            for (int cx = threadIdx.x; cx < 2 * radius + BLOCK_X; cx += BLOCK_X) {
                int sx = min(max(cx - static_cast<int>(threadIdx.x) - radius + x, 0), width - 1);
                buffer[cy * SMEM_W + cx] = src[sy * stride + sx];
            }
        }

        if constexpr (has_ref) {
            for (int cy = threadIdx.y; cy < 2 * radius + BLOCK_Y; cy += BLOCK_Y) {
                int sy = min(max(cy - static_cast<int>(threadIdx.y) - radius + y, 0), height - 1);
                for (int cx = threadIdx.x; cx < 2 * radius + BLOCK_X; cx += BLOCK_X) {
                    int sx = min(max(cx - static_cast<int>(threadIdx.x) - radius + x, 0), width - 1);
                    buffer[(2 * radius + BLOCK_Y + cy) * SMEM_W + cx] = src[(height + sy) * stride + sx];
                }
            }
        }

        __syncthreads();

        if (x >= width || y >= height)
            return;

        const float center = buffer[
            (has_ref * (2 * radius + BLOCK_Y) + radius + threadIdx.y) * SMEM_W +
            radius + threadIdx.x
        ];

        for (int cy = -radius; cy <= radius; ++cy) {
            int sy = cy + radius + threadIdx.y;

            for (int cx = -radius; cx <= radius; ++cx) {
                int sx = cx + radius + threadIdx.x;

                float value = buffer[(has_ref * (2 * radius + BLOCK_Y) + sy) * SMEM_W + sx];

                float space = cy * cy + cx * cx;
                float range = (value - center) * (value - center);

                float weight = exp2f(space * sigma_spatial_scaled + range * sigma_color_scaled);

                if constexpr (has_ref) {
                    value = buffer[sy * SMEM_W + sx];
                }

                num += weight * value;
                den += weight;
            }
        }
    } else {
        if (x >= width || y >= height)
            return;

        const float center = src[(has_ref * height + y) * stride + x];

        for (int cy = max(y - radius, 0); cy <= min(y + radius, height - 1); ++cy) {
            for (int cx = max(x - radius, 0); cx <= min(x + radius, width - 1); ++cx) {
                float value = src[(has_ref * height + cy) * stride + cx];

                float space = (y - cy) * (y - cy) + (x - cx) * (x - cx);
                float range = (value - center) * (value - center);

                float weight = exp2f(space * sigma_spatial_scaled + range * sigma_color_scaled);

                if constexpr (has_ref) {
                    value = src[cy * stride + cx];
                }

                num += weight * value;
                den += weight;
            }
        }
    }

    dst[y * stride + x] = num / den;
}
