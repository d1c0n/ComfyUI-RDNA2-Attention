// RMS norm for fp16 rows with fp32 math, reading and writing each value once.
#include <hip/hip_runtime.h>
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
#define THREADS 256

// TPR threads share a row and hold it in registers. Rows can be strided, so Q and K are read straight from the linear output.
// F holds the 2x2 rotation of each pair of values, for one token after the other, as in core's rope.
template <int TPR, int CHUNKS>
__device__ __forceinline__ void norm_row(const _Float16* __restrict__ X, const float* __restrict__ W, _Float16* __restrict__ Y,
                                         int rows, int D, int L, int H, int sb, int sh, int sl, float eps, const f4* __restrict__ F = nullptr, int fb = 0) {
    __shared__ float red[THREADS / 32];
    const int t = threadIdx.x % TPR, n = blockIdx.x * (THREADS / TPR) + threadIdx.x / TPR;
    if (n >= rows) return;
    // Heads go first, so a block reads neighbouring heads of one token instead of rows a whole pitch apart.
    const int h = n % H, l = (n / H) % L, b = n / H / L;
    const h8* x = (const h8*)(X + (size_t)b * sb + (size_t)h * sh + (size_t)l * sl);
    h8 v[CHUNKS];
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; c++) {
        const int i = t + c * TPR;
        if (i < D / 8) {
            v[c] = x[i];
#pragma unroll
            for (int j = 0; j < 8; j++) sum += (float)v[c][j] * (float)v[c][j];
        }
    }
#pragma unroll
    for (int o = (TPR < 32 ? TPR : 32) / 2; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
    if (TPR > 32) {
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
        __syncthreads();
        sum = 0.f;
#pragma unroll
        for (int i = 0; i < THREADS / 32; i++) sum += red[i];
    }
    const float rstd = rsqrtf(sum / D + eps);
    h8* y = (h8*)(Y + ((size_t)(b * H + h) * L + l) * D);
#pragma unroll
    for (int c = 0; c < CHUNKS; c++) {
        const int i = t + c * TPR;
        if (i < D / 8) {
            const f4 lo = ((const f4*)W)[i * 2], hi = ((const f4*)W)[i * 2 + 1];
            h8 o;
#pragma unroll
            for (int j = 0; j < 8; j++) o[j] = (_Float16)((float)v[c][j] * rstd * ((j < 4 ? lo[j & 3] : hi[j & 3]) + 1.f));
            if (F) {
#pragma clang fp contract(off)
                const f4* f = F + (size_t)b * fb + (size_t)l * (D / 2) + i * 4;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const f4 m = f[j];
                    const float x0 = (float)o[j * 2], x1 = (float)o[j * 2 + 1];
                    float a = __builtin_fmaf(m[1], x1, m[0] * x0), c = __builtin_fmaf(m[3], x1, m[2] * x0);
                    // Rounding to fp32 first is what kitchen's rope does. The compiler would otherwise round once, straight to fp16.
                    asm volatile("" : "+v"(a), "+v"(c));
                    o[j * 2] = (_Float16)a;
                    o[j * 2 + 1] = (_Float16)c;
                }
            }
            y[i] = o;
        }
    }
}

extern "C" __global__ __launch_bounds__(THREADS) void rms_norm_head(const _Float16* X, const float* W, _Float16* Y, int rows, int D, int L, int H, int sb, int sh, int sl, float eps) {
    norm_row<16, 1>(X, W, Y, rows, D, L, H, sb, sh, sl, eps);
}

extern "C" __global__ __launch_bounds__(THREADS) void rms_norm_row(const _Float16* X, const float* W, _Float16* Y, int rows, int D, int L, int H, int sb, int sh, int sl, float eps) {
    norm_row<THREADS, 4>(X, W, Y, rows, D, L, H, sb, sh, sl, eps);
}

extern "C" __global__ __launch_bounds__(THREADS) void rms_norm_rope(const _Float16* X, const float* W, _Float16* Y, int rows, int D, int L, int H, int sb, int sh, int sl, float eps, const f4* F, int fb) {
    norm_row<16, 1>(X, W, Y, rows, D, L, H, sb, sh, sl, eps, F, fb);
}
