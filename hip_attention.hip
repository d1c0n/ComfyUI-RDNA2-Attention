// Fused attention for RDNA2 (gfx1030), fp16, head dim 128.
//
// RDNA2 has no matrix cores, and LDS bandwidth caps tiled kernels at about half of the dot2 peak.
// Each lane owns one query row instead, with Q and O in VGPRs. K and V are the same for the whole wave,
// so they stream through the scalar cache and are read as SGPR operands of v_dot2c_f32_f16.
#include <hip/hip_runtime.h>

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef unsigned u32;
typedef u32 u4 __attribute__((ext_vector_type(4)));
typedef u32 s16 __attribute__((ext_vector_type(16)));
typedef const __attribute__((address_space(4))) s16* cs16;
typedef const __attribute__((address_space(4))) u32* cu32;
typedef const __attribute__((address_space(4))) char* cchar;

#define WAVES 8
#define WARM_LINES (64 / WAVES)
#define WARM_AT 8
#define SCHED_FENCE __builtin_amdgcn_sched_barrier(0)

__device__ __forceinline__ float dot2(u32 a, u32 b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}

__device__ __forceinline__ u32 pack(float a, float b) {
    h2 v = {(_Float16)a, (_Float16)b};
    return __builtin_bit_cast(u32, v);
}

// Writes groups of 16 keys, 2048 dwords each, in the order attn consumes them.
// K: 32 chunks of 4 keys x 8 dwords. V: 8 rows of 128 dwords holding (V[2k][d], V[2k + 1][d]).
extern "C" __global__ __launch_bounds__(256) void pack_kv(const _Float16* __restrict__ K, const _Float16* __restrict__ V, u32* __restrict__ P,
    int Lk, int groups, int H, int skb, int skh, int skl, int svb, int svh, int svl)
{
    const int g = blockIdx.x, bh = blockIdx.y, b = bh / H, h = bh % H, t = threadIdx.x;
    u32* out = P + ((size_t)bh * (groups + 1) + g) * 2048;
    const _Float16* kb = K + (size_t)b * skb + (size_t)h * skh;
    const _Float16* vb = V + (size_t)b * svb + (size_t)h * svh;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = t + i * 256, key = idx >> 6, kd = idx & 63;
        u32 v = 0;
        if (g * 16 + key < Lk) v = *(const u32*)(kb + (size_t)(g * 16 + key) * skl + kd * 2);
        out[((key >> 2) * 8 + (kd >> 3)) * 32 + (key & 3) * 8 + (kd & 7)] = v;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = t + i * 256, kp = idx >> 7, d = idx & 127, key = g * 16 + kp * 2;
        _Float16 x = key < Lk ? vb[(size_t)key * svl + d] : (_Float16)0;
        _Float16 y = key + 1 < Lk ? vb[(size_t)(key + 1) * svl + d] : (_Float16)0;
        h2 v = {x, y};
        out[1024 + idx] = __builtin_bit_cast(u32, v);
    }
}

// Stream of groups of 16 keys for attn_i8, 6208 bytes each: 16 float K scales, 2KB of int8 K in chunks of 4 keys x 8 dwords, then V as in pack_kv.
#define I8_GROUP 1552
#define I8_V 528
extern "C" __global__ __launch_bounds__(256) void pack_kv_i8(const _Float16* __restrict__ K, const _Float16* __restrict__ V, u32* __restrict__ P,
    int Lk, int groups, int H, int skb, int skh, int skl, int svb, int svh, int svl)
{
    const int g = blockIdx.x, bh = blockIdx.y, b = bh / H, h = bh % H, t = threadIdx.x;
    u32* out = P + ((size_t)bh * (groups + 1) + g) * I8_GROUP;
    const _Float16* kb = K + (size_t)b * skb + (size_t)h * skh;
    const _Float16* vb = V + (size_t)b * svb + (size_t)h * svh;

    const int key = t >> 4, sub = t & 15;
    float f[8];
    float mx = 0.f;
    if (g * 16 + key < Lk) {
        u4 v = *(const u4*)(kb + (size_t)(g * 16 + key) * skl + sub * 8);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 w = v[i]; // bit_cast directly on a vector element always reads element 0
            h2 x = __builtin_bit_cast(h2, w);
            f[i * 2] = (float)x.x; f[i * 2 + 1] = (float)x.y;
            mx = fmaxf(mx, fmaxf(fabsf(f[i * 2]), fabsf(f[i * 2 + 1])));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) f[i] = 0.f;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    const float scale = fmaxf(mx / 127.f, 1e-30f);
    if (sub == 0) ((float*)out)[key] = scale;
#pragma unroll
    for (int d = 0; d < 2; d++) {
        u32 w = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) w |= (u32)((int)rintf(f[d * 4 + i] / scale) & 255) << (i * 8);
        const int kd = sub * 2 + d;
        out[16 + ((key >> 2) * 4 + (kd >> 3)) * 32 + (key & 3) * 8 + (kd & 7)] = w;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = t + i * 256, kp = idx >> 7, d = idx & 127, k2 = g * 16 + kp * 2;
        _Float16 x = k2 < Lk ? vb[(size_t)k2 * svl + d] : (_Float16)0;
        _Float16 y = k2 + 1 < Lk ? vb[(size_t)(k2 + 1) * svl + d] : (_Float16)0;
        h2 v = {x, y};
        out[I8_V + idx] = __builtin_bit_cast(u32, v);
    }
}

struct Chunk { s16 a, b; };

__device__ __forceinline__ void load_chunk(Chunk& c, cchar p, int n) {
    c.a = *(cs16)(p + n * 128);
    c.b = *(cs16)(p + n * 128 + 64);
}

// A new line costs a full miss, and s_waitcnt can only wait for every scalar load. Each wave touches
// its slice of the next 4KB, so the workgroup pays that latency once instead of once per chunk.
__device__ __forceinline__ void warm(cchar p, int wave, u32* w) {
#pragma unroll
    for (int i = 0; i < WARM_LINES; i++) w[i] = *(cu32)(p + (wave * WARM_LINES + i) * 64);
}

__device__ __forceinline__ void warm_done(const u32* w) {
#pragma unroll
    for (int i = 0; i < WARM_LINES; i++) asm volatile("" :: "s"(w[i]));
}

__device__ __forceinline__ void qk(const Chunk& c, const u32* q, float* s, int n) {
    const int jg = (n >> 3) * 4, d = (n & 7) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        s[jg] = dot2(c.a[e], q[d + e], s[jg]);
        s[jg + 1] = dot2(c.a[8 + e], q[d + e], s[jg + 1]);
        s[jg + 2] = dot2(c.b[e], q[d + e], s[jg + 2]);
        s[jg + 3] = dot2(c.b[8 + e], q[d + e], s[jg + 3]);
    }
}

__device__ __forceinline__ void pv(const Chunk& c, const u32* p, float* o, int n) {
    const int d = (n & 3) * 32;
#pragma unroll
    for (int e = 0; e < 16; e++) {
        o[d + e] = dot2(c.a[e], p[n >> 2], o[d + e]);
        o[d + 16 + e] = dot2(c.b[e], p[n >> 2], o[d + 16 + e]);
    }
}

__device__ __forceinline__ void load_chunk_i8(Chunk& c, cchar p, int n) {
    c.a = *(cs16)(p + 64 + n * 128);
    c.b = *(cs16)(p + 128 + n * 128);
}

__device__ __forceinline__ void qk_i8(const Chunk& c, const u32* q, int* s, int n) {
    const int jg = (n >> 2) * 4, d = (n & 3) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        s[jg] = __builtin_amdgcn_sdot4((int)c.a[e], (int)q[d + e], s[jg], false);
        s[jg + 1] = __builtin_amdgcn_sdot4((int)c.a[8 + e], (int)q[d + e], s[jg + 1], false);
        s[jg + 2] = __builtin_amdgcn_sdot4((int)c.b[e], (int)q[d + e], s[jg + 2], false);
        s[jg + 3] = __builtin_amdgcn_sdot4((int)c.b[8 + e], (int)q[d + e], s[jg + 3], false);
    }
}

// attn with int8 QK^T. PV stays fp16: quantizing P destroys the small probabilities that make up most of the output.
extern "C" __global__ __launch_bounds__(WAVES * 32) void attn_i8(const _Float16* __restrict__ Q, const u32* P, _Float16* __restrict__ O,
    int Lq, int Lk, int groups, int H, int sqb, int sqh, int sql, int sob, int soh, int sol, float scale_log2)
{
    const int bh = blockIdx.y, b = bh / H, h = bh % H;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int row = (blockIdx.x * WAVES + wave) * 32 + (threadIdx.x & 31);
    const int q_row = row < Lq ? row : Lq - 1;

    const _Float16* qp = Q + (size_t)b * sqb + (size_t)h * sqh + (size_t)q_row * sql;
    float qmax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        u4 v = *(const u4*)(qp + i * 8);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            u32 w = v[j];
            h2 x = __builtin_bit_cast(h2, w);
            qmax = fmaxf(qmax, fmaxf(fabsf((float)x.x), fabsf((float)x.y)));
        }
    }
    const float q_scale = fmaxf(qmax / 127.f, 1e-30f), score_scale = q_scale * scale_log2;
    u32 q[32];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        u4 v = *(const u4*)(qp + i * 8);
#pragma unroll
        for (int d = 0; d < 2; d++) {
            u32 w0 = v[d * 2], w1 = v[d * 2 + 1];
            h2 x = __builtin_bit_cast(h2, w0), y = __builtin_bit_cast(h2, w1);
            q[i * 2 + d] = ((u32)((int)rintf((float)x.x / q_scale) & 255)) | ((u32)((int)rintf((float)x.y / q_scale) & 255) << 8)
                | ((u32)((int)rintf((float)y.x / q_scale) & 255) << 16) | ((u32)((int)rintf((float)y.y / q_scale) & 255) << 24);
        }
    }
    float o[128];
#pragma unroll
    for (int i = 0; i < 128; i++) o[i] = 0.f;
    float m = -1e30f, l = 0.f;

    cchar p = (cchar)(P + (size_t)bh * (groups + 1) * I8_GROUP);
    Chunk A, B;
    load_chunk_i8(A, p, 0);
    for (int g = 0; g < groups; g++) {
        SCHED_FENCE; __builtin_amdgcn_s_barrier(); SCHED_FENCE;

        int si[16];
#pragma unroll
        for (int j = 0; j < 16; j++) si[j] = 0;
        s16 ks;
#pragma unroll
        for (int n = 0; n < 16; n += 2) {
            u32 w[WARM_LINES];
            if (n == WARM_AT) warm(p + 2112, wave, w);
            load_chunk_i8(B, p, n + 1); SCHED_FENCE;
            qk_i8(A, q, si, n); SCHED_FENCE;
            if (n == WARM_AT) warm_done(w);
            if (n == 14) ks = *(cs16)p;
            load_chunk_i8(A, p, n + 2); SCHED_FENCE;
            qk_i8(B, q, si, n + 1); SCHED_FENCE;
        }

        float s[16];
#pragma unroll
        for (int j = 0; j < 16; j++) {
            u32 w = ks[j];
            s[j] = (float)si[j] * score_scale * __builtin_bit_cast(float, w);
        }
        const int valid = Lk - g * 16;
        if (valid < 16) {
#pragma unroll
            for (int j = 0; j < 16; j++) if (j >= valid) s[j] = -1e30f;
        }
        float mx = s[0];
#pragma unroll
        for (int j = 1; j < 16; j++) mx = fmaxf(mx, s[j]);
        if (__any(mx > m + 8.f)) {
            float mn = fmaxf(m, mx), alpha = __builtin_amdgcn_exp2f(m - mn);
            m = mn;
            l *= alpha;
#pragma unroll
            for (int i = 0; i < 128; i++) o[i] *= alpha;
        }
        u32 pr[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            float p0 = __builtin_amdgcn_exp2f(s[2 * u] - m), p1 = __builtin_amdgcn_exp2f(s[2 * u + 1] - m);
            l += p0 + p1;
            pr[u] = pack(p0, p1);
        }
        SCHED_FENCE;

#pragma unroll
        for (int n = 0; n < 32; n += 2) {
            u32 w[WARM_LINES];
            if (n == WARM_AT) warm(p + 6208, wave, w);
            load_chunk_i8(B, p, 16 + n + 1); SCHED_FENCE;
            pv(A, pr, o, n); SCHED_FENCE;
            if (n == WARM_AT) warm_done(w);
            if (n == 30) load_chunk_i8(A, p + 6208, 0);
            else load_chunk_i8(A, p, 16 + n + 2);
            SCHED_FENCE;
            pv(B, pr, o, n + 1); SCHED_FENCE;
        }
        p += 6208;
    }

    if (row < Lq) {
        float inv = 1.f / l;
        _Float16* op = O + (size_t)b * sob + (size_t)h * soh + (size_t)row * sol;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            u4 v = {pack(o[i * 8] * inv, o[i * 8 + 1] * inv), pack(o[i * 8 + 2] * inv, o[i * 8 + 3] * inv),
                    pack(o[i * 8 + 4] * inv, o[i * 8 + 5] * inv), pack(o[i * 8 + 6] * inv, o[i * 8 + 7] * inv)};
            *(u4*)(op + i * 8) = v;
        }
    }
}

extern "C" __global__ __launch_bounds__(WAVES * 32) void attn(const _Float16* __restrict__ Q, const u32* P, _Float16* __restrict__ O,
    int Lq, int Lk, int groups, int H, int sqb, int sqh, int sql, int sob, int soh, int sol, float scale_log2)
{
    const int bh = blockIdx.y, b = bh / H, h = bh % H;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int row = (blockIdx.x * WAVES + wave) * 32 + (threadIdx.x & 31);
    const int q_row = row < Lq ? row : Lq - 1;

    u32 q[64];
    const _Float16* qp = Q + (size_t)b * sqb + (size_t)h * sqh + (size_t)q_row * sql;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        u4 v = *(const u4*)(qp + i * 8);
        q[i * 4] = v.x; q[i * 4 + 1] = v.y; q[i * 4 + 2] = v.z; q[i * 4 + 3] = v.w;
    }
    float o[128];
#pragma unroll
    for (int i = 0; i < 128; i++) o[i] = 0.f;
    float m = -1e30f, l = 0.f;

    cchar p = (cchar)(P + (size_t)bh * (groups + 1) * 2048);
    // Scalar loads return out of order, so only one chunk may be in flight while the other is used.
    Chunk A, B;
    load_chunk(A, p, 0);
    for (int g = 0; g < groups; g++) {
        // Keeps the waves together so that they share the lines of the scalar cache.
        SCHED_FENCE; __builtin_amdgcn_s_barrier(); SCHED_FENCE;

        float s[16];
#pragma unroll
        for (int j = 0; j < 16; j++) s[j] = 0.f;
#pragma unroll
        for (int n = 0; n < 32; n += 2) {
            u32 w[WARM_LINES];
            if (n == WARM_AT) warm(p + 4096, wave, w);
            load_chunk(B, p, n + 1); SCHED_FENCE;
            qk(A, q, s, n); SCHED_FENCE;
            if (n == WARM_AT) warm_done(w);
            load_chunk(A, p, n + 2); SCHED_FENCE;
            qk(B, q, s, n + 1); SCHED_FENCE;
        }

        const int valid = Lk - g * 16;
        if (valid < 16) {
#pragma unroll
            for (int j = 0; j < 16; j++) if (j >= valid) s[j] = -1e30f;
        }
        float mx = s[0];
#pragma unroll
        for (int j = 1; j < 16; j++) mx = fmaxf(mx, s[j]);
        mx *= scale_log2;
        // Rescaling o costs 128 multiplies, skip it unless p would grow past 2^8.
        if (__any(mx > m + 8.f)) {
            float mn = fmaxf(m, mx), alpha = __builtin_amdgcn_exp2f(m - mn);
            m = mn;
            l *= alpha;
#pragma unroll
            for (int i = 0; i < 128; i++) o[i] *= alpha;
        }
        u32 pr[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            float p0 = __builtin_amdgcn_exp2f(fmaf(s[2 * u], scale_log2, -m)), p1 = __builtin_amdgcn_exp2f(fmaf(s[2 * u + 1], scale_log2, -m));
            l += p0 + p1;
            pr[u] = pack(p0, p1);
        }
        SCHED_FENCE;

#pragma unroll
        for (int n = 0; n < 32; n += 2) {
            u32 w[WARM_LINES];
            if (n == WARM_AT) warm(p + 8192, wave, w);
            load_chunk(B, p, 32 + n + 1); SCHED_FENCE;
            pv(A, pr, o, n); SCHED_FENCE;
            if (n == WARM_AT) warm_done(w);
            load_chunk(A, p, 32 + n + 2); SCHED_FENCE;
            pv(B, pr, o, n + 1); SCHED_FENCE;
        }
        p += 8192;
    }

    if (row < Lq) {
        float inv = 1.f / l;
        _Float16* op = O + (size_t)b * sob + (size_t)h * soh + (size_t)row * sol;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            u4 v = {pack(o[i * 8] * inv, o[i * 8 + 1] * inv), pack(o[i * 8 + 2] * inv, o[i * 8 + 3] * inv),
                    pack(o[i * 8 + 4] * inv, o[i * 8 + 5] * inv), pack(o[i * 8 + 6] * inv, o[i * 8 + 7] * inv)};
            *(u4*)(op + i * 8) = v;
        }
    }
}
