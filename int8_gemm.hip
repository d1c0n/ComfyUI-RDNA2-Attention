// INT8 matmul for RDNA2 (gfx1030).
//
// LDS bandwidth holds tiled int8 kernels at about 40% of the dot4 peak. Like hip_attention.hip, lanes own rows,
// here 2 each, and weights stream through the scalar cache as SGPR operands of v_dot4c_i32_i8.
#include <hip/hip_runtime.h>
typedef unsigned u32;
typedef u32 u4 __attribute__((ext_vector_type(4)));
typedef u32 s16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef u32 s8 __attribute__((ext_vector_type(8)));
typedef const __attribute__((address_space(4))) s16* cs16;
typedef const __attribute__((address_space(4))) s8* cs8;
typedef const __attribute__((address_space(4))) u32* cu32;
typedef const __attribute__((address_space(4))) char* cchar;
#define WAVES 8
#define ROT_WAVES 4
#define ROT_THREADS (ROT_WAVES * 32)
#define WARM_AT 8
#define WARM_AT_RAW 24
#define WARM_LINES (64 / WAVES)
#define FENCE __builtin_amdgcn_sched_barrier(0)
enum Act { NO_ACT, SITU, SWIGLU, GATE, MOD };
// The RMS norm and modulation in front of a Krea 2 block's linears. s is 1 + scale, s and t hold one row per batch.
struct Mod { const float* w; const _Float16* s; const _Float16* t; int L; float eps; };
template <bool I4> __device__ __forceinline__ int dot(u32 a, u32 b, int c) {
    return I4 ? __builtin_amdgcn_sdot8((int)a, (int)b, c, false) : __builtin_amdgcn_sdot4((int)a, (int)b, c, false);
}
// Quantizes a row like triton_quantize_rowwise and writes it as gemm reads it: u4 index ((kc * 2 + r) * 4 + q) * 32 + lane.
template <int NW>
__device__ __forceinline__ void quant_row(const h8* x, float mx, float* red, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    const int row = blockIdx.x, t = threadIdx.x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((t & 31) == 0) red[t >> 5] = mx;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NW; i++) mx = fmaxf(mx, red[i]);
    const float scale = fmaxf(mx / 127.f, 1e-30f);
    if (t == 0) as[row] = scale;

    const int KCs = K / 64, wt = row >> 6, r = (row >> 5) & 1, lane = row & 31;
    u4* out = Ap + (size_t)wt * KCs * 256 + r * 128 + lane;
    for (int g = t; g < K / 16; g += NW * 32) {
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
    for (int i = 0; i < ROT_WAVES; i++) mx = fmaxf(mx, red[i]);
    const float scale = fmaxf(mx, 1e-10f) / 7.f;
    if (t == 0) as[row] = scale;

    const int KCs = K / 128, wt = row >> 6, r = (row >> 5) & 1, lane = row & 31;
    u4* out = Ap + (size_t)wt * KCs * 256 + r * 128 + lane;
    for (int g = t; g < K / 32; g += ROT_THREADS) {
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
    quant_row<8>(x, mx, red, Ap, as, K);
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

// Rounds to fp16 at the same points as F.silu(gate).mul_(up).
__device__ __forceinline__ float swiglu(float gate, float up) {
    return (float)(_Float16)((float)(_Float16)(gate / (1.f + expf(-gate))) * up);
}

// Rounds to fp16 at the same points as x * F.sigmoid(gate).
__device__ __forceinline__ float sigmoid_gate(float x, float gate) {
    return (float)(_Float16)(x * (float)(_Float16)(1.f / (1.f + expf(-gate))));
}

// Rounds to fp16 at the same points as s * rms_norm(x) + t.
__device__ __forceinline__ float modulate(float x, float rstd, float w, float s, float t) {
#pragma clang fp contract(off)
    return (float)(_Float16)((float)(_Float16)((float)(_Float16)(x * rstd * (w + 1.f)) * s) + t);
}

__device__ __forceinline__ void wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// quant_pack_a after a ConvRot rotation of every 256 elements. 16 lanes share a group: two digits in registers, an LDS transpose, two more.
// SITU reads a 2K wide [hidden | gate] row and SWIGLU a gate row from X and an up row from U. Both apply the activation first, so the K wide
// activation never goes through VRAM.
template <Act ACT, bool I4>
__device__ __forceinline__ void rot_quant(const _Float16* __restrict__ X, const _Float16* __restrict__ U, u4* __restrict__ Ap, float* __restrict__ as, int K, Mod mod = {}) {
    __shared__ float red[ROT_WAVES * 2];
    __shared__ float swap[ROT_WAVES][2][16 * 17];
    // Sized to K at launch. A fixed maximum would leave room for far fewer blocks per WGP.
    extern __shared__ __attribute__((aligned(16))) _Float16 rotated[];
    const int wave = threadIdx.x >> 5, half = (threadIdx.x >> 4) & 1, c = threadIdx.x & 15, groups = K / 256;
    const h8* x = (const h8*)(X + (size_t)blockIdx.x * K * (ACT == SITU ? 2 : 1));
    const h8* u = (const h8*)(U + (size_t)blockIdx.x * K);
    const h8* ms = (const h8*)(mod.s + (size_t)(blockIdx.x / (mod.L ? mod.L : 1)) * K);
    const h8* mt = (const h8*)(mod.t + (size_t)(blockIdx.x / (mod.L ? mod.L : 1)) * K);
    float* sw = swap[wave][half];
    float rstd = 0.f;
    if (ACT == MOD) {
        // The sums run in the order of rms_norm_row, whose 256 threads this block covers two at a time.
        float sum[2] = {0.f, 0.f};
#pragma unroll
        for (int v = 0; v < 2; v++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int idx = threadIdx.x + v * ROT_THREADS + i * 256;
                if (idx < K / 8) {
                    h8 row = x[idx];
                    ((h8*)rotated)[idx] = row;
#pragma unroll
                    for (int j = 0; j < 8; j++) sum[v] += (float)row[j] * (float)row[j];
                }
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { sum[0] += __shfl_xor(sum[0], o, 32); sum[1] += __shfl_xor(sum[1], o, 32); }
        if ((threadIdx.x & 31) == 0) { red[wave] = sum[0]; red[wave + ROT_WAVES] = sum[1]; }
        __syncthreads();
        float total = 0.f;
#pragma unroll
        for (int i = 0; i < ROT_WAVES * 2; i++) total += red[i];
        rstd = rsqrtf(total / K + mod.eps);
        __syncthreads();
    }
    float mx = 0.f;
    for (int g = wave * 2 + half; g < (groups + 1) / 2 * 2; g += ROT_WAVES * 2) {
        const bool active = g < groups;
        float v[16];
        if (active) {
            const h8* src = ACT == MOD ? (const h8*)rotated : x;
            h8 lo = src[g * 32 + c * 2], hi = src[g * 32 + c * 2 + 1];
            if (ACT == SITU) {
                h8 glo = x[K / 8 + g * 32 + c * 2], ghi = x[K / 8 + g * 32 + c * 2 + 1];
#pragma unroll
                for (int i = 0; i < 8; i++) { v[i] = situ((float)lo[i], (float)glo[i]); v[8 + i] = situ((float)hi[i], (float)ghi[i]); }
            } else if (ACT == SWIGLU || ACT == GATE) {
                h8 ulo = u[g * 32 + c * 2], uhi = u[g * 32 + c * 2 + 1];
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    v[i] = ACT == SWIGLU ? swiglu((float)lo[i], (float)ulo[i]) : sigmoid_gate((float)lo[i], (float)ulo[i]);
                    v[8 + i] = ACT == SWIGLU ? swiglu((float)hi[i], (float)uhi[i]) : sigmoid_gate((float)hi[i], (float)uhi[i]);
                }
            } else if (ACT == MOD) {
                h8 slo = ms[g * 32 + c * 2], shi = ms[g * 32 + c * 2 + 1], tlo = mt[g * 32 + c * 2], thi = mt[g * 32 + c * 2 + 1];
                const float* w = mod.w + g * 256 + c * 16;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    v[i] = modulate((float)lo[i], rstd, w[i], (float)slo[i], (float)tlo[i]);
                    v[8 + i] = modulate((float)hi[i], rstd, w[8 + i], (float)shi[i], (float)thi[i]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) { v[i] = (float)lo[i]; v[8 + i] = (float)hi[i]; }
            }
            h16(v);
#pragma unroll
            for (int i = 0; i < 16; i++) sw[c * 17 + i] = v[i];
        }
        wave_sync();
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
        wave_sync();
    }
    __syncthreads();
    if (I4) quant_row_i4((const h8*)rotated, mx, red, Ap, as, K);
    else quant_row<ROT_WAVES>((const h8*)rotated, mx, red, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(ROT_THREADS) void rot_quant_pack_a(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<NO_ACT, false>(X, nullptr, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(ROT_THREADS) void situ_rot_quant_pack_a(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<SITU, false>(X, nullptr, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(ROT_THREADS) void swiglu_rot_quant_pack_a(const _Float16* __restrict__ X, const _Float16* __restrict__ U, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<SWIGLU, false>(X, U, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(ROT_THREADS) void gate_rot_quant_pack_a(const _Float16* __restrict__ X, const _Float16* __restrict__ U, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<GATE, false>(X, U, Ap, as, K);
}

extern "C" __global__ __launch_bounds__(ROT_THREADS) void mod_rot_quant_pack_a(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K,
    const float* __restrict__ W, const _Float16* __restrict__ S, const _Float16* __restrict__ T, int L, float eps) {
    rot_quant<MOD, false>(X, nullptr, Ap, as, K, Mod{W, S, T, L, eps});
}

extern "C" __global__ __launch_bounds__(ROT_THREADS) void rot_quant_pack_a_i4(const _Float16* __restrict__ X, u4* __restrict__ Ap, float* __restrict__ as, int K) {
    rot_quant<NO_ACT, true>(X, nullptr, Ap, as, K);
}

// Reorders weights into the stream gemm reads. Runs on every call: a temporary copy is far cheaper than keeping a second model in VRAM.
extern "C" __global__ __launch_bounds__(256) void pack_b(const u32* __restrict__ B, u4* __restrict__ Bp, int K) {
    const int kc = blockIdx.x, nt = blockIdx.y, t = threadIdx.x, KCs = gridDim.x;
    const int g = t >> 4, h = (t >> 3) & 1, j = (t >> 1) & 3, e = (t & 1) * 4;
    const int col = nt * 64 + g * 4 + j, kd = kc * 16 + h * 8 + e;
    Bp[((size_t)nt * KCs + kc) * 256 + t] = *(const u4*)(B + (size_t)col * (K / 4) + kd);
}

struct Chunk { s16 a, b; };
// A row pitch of RB reads a raw (N, K) weight in place. RB of 0 reads the pack_b stream.
template <int RB> __device__ __forceinline__ void ld(Chunk& c, cchar p, int n) {
    if (RB) {
        const int col = (n >> 1) * 4, h = (n & 1) * 32;
        s8 w0 = *(cs8)(p + col * RB + h), w1 = *(cs8)(p + (col + 1) * RB + h), w2 = *(cs8)(p + (col + 2) * RB + h), w3 = *(cs8)(p + (col + 3) * RB + h);
        c.a = __builtin_shufflevector(w0, w1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        c.b = __builtin_shufflevector(w2, w3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else {
        c.a = *(cs16)(p + n * 128); c.b = *(cs16)(p + n * 128 + 64);
    }
}
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
// Activations are read once per column tile, so they shouldn't push weights, which every row block shares, out of the cache.
__device__ __forceinline__ void load_a(u32 (&a)[2][16], const u4* ap, int kc) {
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            u4 v = __builtin_nontemporal_load(&ap[((kc * 2 + r) * 4 + q) * 32]);
            a[r][q * 4] = v.x; a[r][q * 4 + 1] = v.y; a[r][q * 4 + 2] = v.z; a[r][q * 4 + 3] = v.w;
        }
}
template <bool I4, int RB>
__device__ __forceinline__ void gemm_body(const u4* __restrict__ Ap, const u32* Bp, const float* __restrict__ as, const float* __restrict__ ws, const _Float16* __restrict__ bias, _Float16* __restrict__ C,
    int M, int N, int KCs, const _Float16* __restrict__ R, const _Float16* __restrict__ G, int L)
{
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int wt = blockIdx.x * WAVES + wave, nt = blockIdx.y;
    // Rows past M only exist to fill the last block. Leaving frees their dot slots for the waves that have work.
    if (wt * 64 >= M) return;
    const u4* ap = Ap + (size_t)wt * KCs * 256 + lane;
    // Integer sums are exact in any order, so column tiles start at different k to spread reads over the memory channels.
    const int step = RB ? 64 : 4096;
    cchar base = (cchar)Bp + (size_t)nt * (RB ? 64 * RB : KCs * 4096);
    int kc = nt % KCs;
    cchar p = base + kc * step;
    int acc[2][64];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int i = 0; i < 64; i++) acc[r][i] = 0;
    u32 a[2][16], an[2][16];
    load_a(a, ap, kc);
    Chunk X, Y;
    ld<RB>(X, p, 0);
    for (int i = 0; i < KCs; i++) {
        const int next = kc + 1 == KCs ? 0 : kc + 1;
        cchar pn = base + next * step;
        FENCE; __builtin_amdgcn_s_barrier(); FENCE;
        load_a(an, ap, next);
        FENCE;
#pragma unroll
        for (int n = 0; n < 32; n += 2) {
            u32 w[WARM_LINES];
            if (n == (RB ? WARM_AT_RAW : WARM_AT)) {
#pragma unroll
                for (int i = 0; i < WARM_LINES; i++) w[i] = *(cu32)(pn + (wave * WARM_LINES + i) * (RB ? RB : 64));
            }
            ld<RB>(Y, p, n + 1); FENCE;
            mac<I4>(X, a, acc, n); FENCE;
            if (n == (RB ? WARM_AT_RAW : WARM_AT)) {
#pragma unroll
                for (int i = 0; i < WARM_LINES; i++) asm volatile("" :: "s"(w[i]));
            }
            if (n == 30) ld<RB>(X, pn, 0);
            else ld<RB>(X, p, n + 2);
            FENCE;
            mac<I4>(Y, a, acc, n + 1); FENCE;
        }
        p = pn;
        kc = next;
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int i = 0; i < 16; i++) a[r][i] = an[r][i];
    }
    // Lanes own rows, so storing from registers writes 16 bytes to each of 32 rows, and power of two row pitches
    // put those on few memory channels. Going through LDS lets each store write whole 128 byte row segments.
    __shared__ h8 tile[WAVES][32][9];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int base = wt * 64 + r * 32, row = base + lane;
        const float s = row < M ? as[row] : 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            h8 v;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int n = nt * 64 + i * 8 + j;
                float y = (float)acc[r][i * 8 + j] * s * ws[n];
                v[j] = (_Float16)(bias ? y + (float)bias[n] : y);
            }
            tile[wave][lane][i] = v;
        }
        wave_sync();
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int local = i * 4 + (lane >> 3), col = lane & 7;
            if (base + local < M) {
                const size_t at = (size_t)(base + local) * N + nt * 64 + col * 8;
                h8 y = tile[wave][local][col];
                if (R) {
#pragma clang fp contract(off)
                    // A block's residual, R + G * y, with the rounding of the separate ops.
                    const h8 res = *(const h8*)(R + at), gate = *(const h8*)(G + (size_t)((base + local) / L) * N + nt * 64 + col * 8);
#pragma unroll
                    for (int j = 0; j < 8; j++) y[j] = (_Float16)((float)res[j] + (float)(_Float16)((float)gate[j] * (float)y[j]));
                }
                *(h8*)(C + at) = y;
            }
        }
        wave_sync();
    }
}

// KCs counts 64 byte steps of a row, so an int4 row of K values is simply a row of K / 2 bytes here.
#define GEMM_ARGS const u4* __restrict__ Ap, const u32* Bp, const float* __restrict__ as, const float* __restrict__ ws, const _Float16* __restrict__ bias, _Float16* __restrict__ C, int M, int N, int KCs, \
    const _Float16* __restrict__ R, const _Float16* __restrict__ G, int L
extern "C" __global__ __launch_bounds__(WAVES * 32) void gemm(GEMM_ARGS) { gemm_body<false, 0>(Ap, Bp, as, ws, bias, C, M, N, KCs, R, G, L); }
extern "C" __global__ __launch_bounds__(WAVES * 32) void gemm_i4(GEMM_ARGS) { gemm_body<true, 0>(Ap, Bp, as, ws, bias, C, M, N, KCs, R, G, L); }
// pack_b is not needed where the row pitch spreads over the memory channels. A power of two pitch does not, so K = 16384 stays on the stream.
extern "C" __global__ __launch_bounds__(WAVES * 32) void gemm_k6144(GEMM_ARGS) { gemm_body<false, 6144>(Ap, Bp, as, ws, bias, C, M, N, KCs, R, G, L); }
