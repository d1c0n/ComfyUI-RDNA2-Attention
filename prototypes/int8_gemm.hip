#include <hip/hip_runtime.h>
typedef unsigned u32;
typedef u32 u4 __attribute__((ext_vector_type(4)));
typedef u32 s16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef const __attribute__((address_space(4))) s16* cs16;
typedef const __attribute__((address_space(4))) u32* cu32;
typedef const __attribute__((address_space(4))) char* cchar;
#ifndef WAVES
#define WAVES 8
#endif
#ifndef WARM_AT
#define WARM_AT 8
#endif
#define WARM_LINES (64 / WAVES)
#define FENCE __builtin_amdgcn_sched_barrier(0)
__device__ __forceinline__ int d4(u32 a, u32 b, int c) { return __builtin_amdgcn_sdot4((int)a, (int)b, c, false); }
struct Chunk { s16 a, b; };
__device__ __forceinline__ void ld(Chunk& c, cchar p, int n) { c.a = *(cs16)(p + n * 128); c.b = *(cs16)(p + n * 128 + 64); }
// chunk n: column group n >> 1 (4 columns), dword half n & 1 (8 dwords of the 16 in this k step)
__device__ __forceinline__ void mac(const Chunk& c, const u32 (&a)[2][16], int (&acc)[2][64], int n) {
    const int g = (n >> 1) * 4, d = (n & 1) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            acc[r][g] = d4(c.a[e], a[r][d + e], acc[r][g]);
            acc[r][g + 1] = d4(c.a[8 + e], a[r][d + e], acc[r][g + 1]);
            acc[r][g + 2] = d4(c.b[e], a[r][d + e], acc[r][g + 2]);
            acc[r][g + 3] = d4(c.b[8 + e], a[r][d + e], acc[r][g + 3]);
        }
    }
}
__device__ __forceinline__ void load_a(u32 (&a)[2][16], const u4* ap, int kc) {
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            u4 v = ap[((kc * 2 + r) * 4 + q) * 32];
            a[r][q * 4] = v.x; a[r][q * 4 + 1] = v.y; a[r][q * 4 + 2] = v.z; a[r][q * 4 + 3] = v.w;
        }
}
extern "C" __global__ __launch_bounds__(WAVES * 32) void gemm(const u4* __restrict__ Ap, const u32* Bp, const float* __restrict__ as, _Float16* __restrict__ C,
    int M, int N, int KCs, float ws)
{
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int wt = blockIdx.x * WAVES + wave, nt = blockIdx.y;
    const u4* ap = Ap + (size_t)wt * KCs * 256 + lane;
    cchar p = (cchar)(Bp + (size_t)nt * KCs * 1024);
    int acc[2][64];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int i = 0; i < 64; i++) acc[r][i] = 0;
    u32 a[2][16], an[2][16];
    load_a(a, ap, 0);
    Chunk X, Y;
    ld(X, p, 0);
    for (int kc = 0; kc < KCs; kc++) {
        FENCE; __builtin_amdgcn_s_barrier(); FENCE;
        if (kc + 1 < KCs) load_a(an, ap, kc + 1);
        FENCE;
#pragma unroll
        for (int n = 0; n < 32; n += 2) {
            u32 w[WARM_LINES];
            if (n == WARM_AT) {
#pragma unroll
                for (int i = 0; i < WARM_LINES; i++) w[i] = *(cu32)(p + 4096 + (wave * WARM_LINES + i) * 64);
            }
            ld(Y, p, n + 1); FENCE;
            mac(X, a, acc, n); FENCE;
            if (n == WARM_AT) {
#pragma unroll
                for (int i = 0; i < WARM_LINES; i++) asm volatile("" :: "s"(w[i]));
            }
            ld(X, p, n + 2); FENCE;
            mac(Y, a, acc, n + 1); FENCE;
        }
        p += 4096;
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int i = 0; i < 16; i++) a[r][i] = an[r][i];
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        int row = wt * 64 + r * 32 + lane;
        if (row < M) {
            float s = as[row] * ws;
            _Float16* cp = C + (size_t)row * N + nt * 64;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                h8 v;
#pragma unroll
                for (int j = 0; j < 8; j++) v[j] = (_Float16)((float)acc[r][i * 8 + j] * s);
                *(h8*)(cp + i * 8) = v;
            }
        }
    }
}
