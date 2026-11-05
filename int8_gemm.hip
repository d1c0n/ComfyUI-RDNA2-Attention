// INT8 matmul for RDNA2 (gfx1030).
//
// LDS bandwidth holds tiled int8 kernels at about 40% of the dot4 peak. Like hip_attention.hip, lanes own rows,
// here 2 each, and weights stream through the scalar cache as SGPR operands of v_dot4c_i32_i8.
#include <hip/hip_runtime.h>
typedef unsigned u32;
typedef u32 u4 __attribute__((ext_vector_type(4)));
typedef u32 s16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef const __attribute__((address_space(4))) s16* cs16;
typedef const __attribute__((address_space(4))) u32* cu32;
typedef const __attribute__((address_space(4))) char* cchar;
#define WAVES 8
#define MAX_ROT_K 16384
#define WARM_AT 8
#define WARM_LINES (64 / WAVES)
#define FENCE __builtin_amdgcn_sched_barrier(0)
template <bool I4> __device__ __forceinline__ int dot(u32 a, u32 b, int c) {
    return I4 ? __builtin_amdgcn_sdot8((int)a, (int)b, c, false) : __builtin_amdgcn_sdot4((int)a, (int)b, c, false);
}
// Quantizes a row like triton_quantize_rowwise and writes it as gemm reads it: u4 index ((kc * 2 + r) * 4 + q) * 32 + lane.
__device__ __forceinline__ void quant_row(const h8* x, float mx, float* red, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    const int row = blockIdx.x, t = threadIdx.x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((t & 31) == 0) red[t >> 5] = mx;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; i++) mx = fmaxf(mx, red[i]);
    const float scale = fmaxf(mx / 127.f, 1e-30f);
    if (t == 0) as[row] = scale;

    const int KCs = K / 64, wt = row >> 6, r = (row >> 5) & 1, lane = row & 31;
    u4* out = Ap + (size_t)wt * KCs * 256 + r * 128 + lane;
    for (int g = t; g < K / 16; g += 256) {
        h8 lo = x[g * 2], hi = x[g * 2 + 1];
        u4 o;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            u32 w = 0;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                int i = d * 4 + b;
                float f = (float)(i < 8 ? lo[i & 7] : hi[i & 7]);
                int q = (int)fminf(fmaxf(floorf(f / scale + 0.5f), -128.f), 127.f);
                w |= (u32)(q & 255) << (b * 8);
            }
            o[d] = w;
        }
        out[(g >> 2) * 256 + (g & 3) * 32] = o;
    }
}

// quant_row for W4A4 activations: symmetric in [-7, 7], 8 values per dword, low nibble first, as v_dot8_i32_i4 reads them.
__device__ __forceinline__ void quant_row_i4(const h8* x, float mx, float* red, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    const int row = blockIdx.x, t = threadIdx.x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((t & 31) == 0) red[t >> 5] = mx;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; i++) mx = fmaxf(mx, red[i]);
    const float scale = fmaxf(mx, 1e-10f) / 7.f;
    if (t == 0) as[row] = scale;

    const int KCs = K / 128, wt = row >> 6, r = (row >> 5) & 1, lane = row & 31;
    u4* out = Ap + (size_t)wt * KCs * 256 + r * 128 + lane;
    for (int g = t; g < K / 32; g += 256) {
        u4 o;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            h8 v = x[g * 4 + d];
            u32 w = 0;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                int q = (int)fminf(fmaxf(rintf((float)v[b] / scale), -7.f), 7.f);
                w |= (u32)(q & 15) << (b * 4);
            }
            o[d] = w;
        }
        out[(g >> 2) * 256 + (g & 3) * 32] = o;
    }
}

extern "C" __global__ __launch_bounds__(256) void quant_pack_a(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    __shared__ float red[8];
    const h8* x = (const h8*)(X + (size_t)blockIdx.x * K);
    float mx = 0.f;
    for (int i = threadIdx.x; i < K / 8; i += 256) {
        h8 v = x[i];
#pragma unroll
        for (int j = 0; j < 8; j++) mx = fmaxf(mx, fabsf((float)v[j]));
    }
    quant_row(x, mx, red, Ap, as, K);
}

// The ConvRot matrix is a Kronecker power of a 4x4 Hadamard H4, so each base 4 digit of the index is one H4 step.
__device__ __forceinline__ void h4(float& a, float& b, float& c, float& d) {
    float s = a + b + c + d, a0 = a, b0 = b;
    a = s - 2.f * d; b = s - 2.f * c; c = s - 2.f * b0; d = s - 2.f * a0;
}

__device__ __forceinline__ void h16(float (&v)[16]) {
#pragma unroll
    for (int i = 0; i < 16; i += 4) h4(v[i], v[i + 1], v[i + 2], v[i + 3]);
#pragma unroll
    for (int i = 0; i < 4; i++) h4(v[i], v[i + 4], v[i + 8], v[i + 12]);
}

__device__ __forceinline__ float soft_tanh(float x) {
    float ax = fabsf(x), t = __builtin_amdgcn_exp2f(-2.885390082f * ax), y = (1.f - t) / (1.f + t);
    if (ax < 0.02f) y = ax * (1.f - ax * ax * (1.f / 3.f - ax * ax * (2.f / 15.f)));
    return x < 0.f ? -y : y;
}

__device__ __forceinline__ float situ(float hidden, float gate) {
    float sigmoid = 1.f / (1.f + __builtin_amdgcn_exp2f(-1.442695041f * gate));
    return (float)(_Float16)(4.f * soft_tanh(gate * 0.25f) * sigmoid * 25.f * soft_tanh(hidden * 0.04f));
}

// quant_pack_a after a ConvRot rotation of every 256 elements. 16 lanes share a group: two digits in registers, an LDS transpose, two more.
// SITU reads a 2K wide [hidden | gate] row and applies the activation first, so the K wide activation never goes through VRAM.
template <bool SITU, bool I4>
__device__ __forceinline__ void rot_quant(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    __shared__ float red[8];
    __shared__ float swap[8][2][16 * 17];
    __shared__ __attribute__((aligned(16))) _Float16 rotated[MAX_ROT_K];
    const int wave = threadIdx.x >> 5, half = (threadIdx.x >> 4) & 1, c = threadIdx.x & 15, groups = K / 256;
    const h8* x = (const h8*)(X + (size_t)blockIdx.x * K * (SITU ? 2 : 1));
    float* sw = swap[wave][half];
    float mx = 0.f;
    for (int g = wave * 2 + half; g < (groups + 1) / 2 * 2; g += 16) {
        const bool active = g < groups;
        float v[16];
        if (active) {
            h8 lo = x[g * 32 + c * 2], hi = x[g * 32 + c * 2 + 1];
            if (SITU) {
                h8 glo = x[K / 8 + g * 32 + c * 2], ghi = x[K / 8 + g * 32 + c * 2 + 1];
#pragma unroll
                for (int i = 0; i < 8; i++) { v[i] = situ((float)lo[i], (float)glo[i]); v[8 + i] = situ((float)hi[i], (float)ghi[i]); }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) { v[i] = (float)lo[i]; v[8 + i] = (float)hi[i]; }
            }
            h16(v);
#pragma unroll
            for (int i = 0; i < 16; i++) sw[c * 17 + i] = v[i];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        if (active) {
#pragma unroll
            for (int i = 0; i < 16; i++) v[i] = sw[i * 17 + c];
            h16(v);
#pragma unroll
            for (int i = 0; i < 16; i++) {
                _Float16 y = (_Float16)(v[i] * 0.0625f);
                rotated[g * 256 + i * 16 + c] = y;
                mx = fmaxf(mx, fabsf((float)y));
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    __syncthreads();
    if (I4) quant_row_i4((const h8*)rotated, mx, red, Ap, as, K);
    else quant_row((const h8*)rotated, mx, red, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(256) void rot_quant_pack_a(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<false, false>(X, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(256) void situ_rot_quant_pack_a(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<true, false>(X, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(256) void rot_quant_pack_a_i4(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<false, true>(X, Ap, as, K);
}

// Reorders weights into the stream gemm reads. Runs on every call: a temporary copy is far cheaper than keeping a second model in VRAM.
extern "C" __global__ __launch_bounds__(256) void pack_b(const u32* __restrict__ B, u4* __restrict__ Bp, int K) {
    const int kc = blockIdx.x, nt = blockIdx.y, t = threadIdx.x, KCs = gridDim.x;
    const int g = t >> 4, h = (t >> 3) & 1, j = (t >> 1) & 3, e = (t & 1) * 4;
    const int col = nt * 64 + g * 4 + j, kd = kc * 16 + h * 8 + e;
    Bp[((size_t)nt * KCs + kc) * 256 + t] = *(const u4*)(B + (size_t)col * (K / 4) + kd);
}

struct Chunk { s16 a, b; };
__device__ __forceinline__ void ld(Chunk& c, cchar p, int n) { c.a = *(cs16)(p + n * 128); c.b = *(cs16)(p + n * 128 + 64); }
// chunk n: column group n >> 1 (4 columns), dword half n & 1 (8 dwords of the 16 in this k step)
template <bool I4>
__device__ __forceinline__ void mac(const Chunk& c, const u32 (&a)[2][16], int (&acc)[2][64], int n) {
    const int g = (n >> 1) * 4, d = (n & 1) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            acc[r][g] = dot<I4>(c.a[e], a[r][d + e], acc[r][g]);
            acc[r][g + 1] = dot<I4>(c.a[8 + e], a[r][d + e], acc[r][g + 1]);
            acc[r][g + 2] = dot<I4>(c.b[e], a[r][d + e], acc[r][g + 2]);
            acc[r][g + 3] = dot<I4>(c.b[8 + e], a[r][d + e], acc[r][g + 3]);
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
template <bool I4>
__device__ __forceinline__ void gemm_body(const u4* __restrict__ Ap, const u32* Bp, const float* __restrict__ as, const float* __restrict__ ws, const _Float16* __restrict__ bias, _Float16* __restrict__ C,
    int M, int N, int KCs)
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
            mac<I4>(X, a, acc, n); FENCE;
            if (n == WARM_AT) {
#pragma unroll
                for (int i = 0; i < WARM_LINES; i++) asm volatile("" :: "s"(w[i]));
            }
            ld(X, p, n + 2); FENCE;
            mac<I4>(Y, a, acc, n + 1); FENCE;
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
            float s = as[row];
            _Float16* cp = C + (size_t)row * N + nt * 64;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                h8 v;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    int n = nt * 64 + i * 8 + j;
                    float y = (float)acc[r][i * 8 + j] * s * ws[n];
                    v[j] = (_Float16)(bias ? y + (float)bias[n] : y);
                }
                *(h8*)(cp + i * 8) = v;
            }
        }
    }
}

// KCs counts 64 byte steps of a row, so an int4 row of K values is simply a row of K / 2 bytes here.
#define GEMM_ARGS const u4* __restrict__ Ap, const u32* Bp, const float* __restrict__ as, const float* __restrict__ ws, const _Float16* __restrict__ bias, _Float16* __restrict__ C, int M, int N, int KCs
extern "C" __global__ __launch_bounds__(WAVES * 32) void gemm(GEMM_ARGS) { gemm_body<false>(Ap, Bp, as, ws, bias, C, M, N, KCs); }
extern "C" __global__ __launch_bounds__(WAVES * 32) void gemm_i4(GEMM_ARGS) { gemm_body<true>(Ap, Bp, as, ws, bias, C, M, N, KCs); }
