// Single token linear layers for text generation on gfx1030.
// One row of activations makes these purely bandwidth bound, so the weights are read once, straight from
// their stored int8 or bf16 form. Activations stay fp32, which is what generate() computes in on this card.
// Each wave owns ROWS output rows and walks them together, so one read of x from LDS feeds ROWS rows.
#include <hip/hip_runtime.h>

#define WAVES 8
#define ROWS 2
#define STEP 512  // int8 values a wave reads from one row per step, 16 per lane.
#define BF16_STEP 256  // bf16 values, 8 per lane.
#define SPLITS 32  // Blocks per KV head in attn_decode.
#define GROUPS 4  // Query heads per KV head.
#define HEAD 128

typedef unsigned int u32;
typedef u32 u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

__device__ inline float wave_sum(float v) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v += __shfl_xor(v, i, 32);
    return v;
}

__device__ inline float dot4(u32 w, f32x4 x, float acc) {
    #pragma unroll
    for (int i = 0; i < 4; i++) acc = fmaf((float)((int)(w << (24 - i * 8)) >> 24), x[i], acc);
    return acc;
}

// x @ H for one row. H is a Kronecker power of a 4x4 regular Hadamard, so each step is y[i] = sum - 2 * x[3 - i].
extern "C" __global__ __launch_bounds__(256) void rotate(const float* x, float* out, int group) {
    extern __shared__ float s[];
    const float* in = x + (size_t)blockIdx.x * group;
    for (int i = threadIdx.x; i < group; i += 256) s[i] = in[i];
    __syncthreads();
    for (int stride = 1; stride < group; stride *= 4) {
        for (int q = threadIdx.x; q < group / 4; q += 256) {
            int i = (q / stride) * stride * 4 + q % stride;
            float a = s[i], b = s[i + stride], c = s[i + stride * 2], d = s[i + stride * 3];
            float sum = a + b + c + d;
            s[i] = sum - 2.0f * d;
            s[i + stride] = sum - 2.0f * c;
            s[i + stride * 2] = sum - 2.0f * b;
            s[i + stride * 3] = sum - 2.0f * a;
        }
        __syncthreads();
    }
    float norm = rsqrtf((float)group);
    float* o = out + (size_t)blockIdx.x * group;
    for (int i = threadIdx.x; i < group; i += 256) o[i] = s[i] * norm;
}

// out[n] = scale[n] * dot(w[n], x) + bias[n]. K is a multiple of STEP and N of WAVES * ROWS.
extern "C" __global__ __launch_bounds__(256) void gemv_i8(const float* x, const signed char* w, const float* scale, const float* bias, float* out, int K) {
    extern __shared__ f32x4 xs[];
    // Stored in reading order, so a wave reads LDS without bank conflicts.
    for (int i = threadIdx.x; i < K / 4; i += 256)
        xs[((i / 128) * 4 + i % 4) * 32 + (i % 128) / 4] = ((const f32x4*)x)[i];
    __syncthreads();

    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = (blockIdx.x * WAVES + wave) * ROWS;
    const signed char* base = w + (size_t)row * K + lane * 16;
    float acc[ROWS] = {};
    // Waves start at different columns: with a power of two K they would otherwise all sit on one memory channel.
    const int steps = K / STEP;
    int step = (blockIdx.x * WAVES + wave) % steps;
    for (int i = 0; i < steps; i++, step = step + 1 == steps ? 0 : step + 1) {
        u32x4 v[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) v[r] = __builtin_nontemporal_load((const u32x4*)(base + (size_t)r * K + step * STEP));
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            f32x4 xv = xs[(step * 4 + j) * 32 + lane];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) acc[r] = dot4(v[r][j], xv, acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float sum = wave_sum(acc[r]) * scale[row + r];
        if (lane == 0) out[row + r] = bias ? sum + bias[row + r] : sum;
    }
}

// out[n] = dot(w[n], x) for a bf16 table. K is a multiple of BF16_STEP and N of WAVES * ROWS.
extern "C" __global__ __launch_bounds__(256) void gemv_bf16(const float* x, const u32* w, float* out, int K) {
    extern __shared__ f32x4 xs[];
    for (int i = threadIdx.x; i < K / 4; i += 256)
        xs[((i / 64) * 2 + i % 2) * 32 + (i % 64) / 2] = ((const f32x4*)x)[i];
    __syncthreads();

    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = (blockIdx.x * WAVES + wave) * ROWS;
    float acc[ROWS] = {};
    for (int step = 0; step < K / BF16_STEP; step++) {
        u32x4 v[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) v[r] = __builtin_nontemporal_load((const u32x4*)(w + ((size_t)(row + r) * K + step * BF16_STEP + lane * 8) / 2));
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            f32x4 xv = xs[(step * 2 + j) * 32 + lane];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                u32 a = v[r][j * 2], b = v[r][j * 2 + 1];
                acc[r] = fmaf(__builtin_bit_cast(float, a << 16), xv[0], acc[r]);
                acc[r] = fmaf(__builtin_bit_cast(float, a & 0xffff0000u), xv[1], acc[r]);
                acc[r] = fmaf(__builtin_bit_cast(float, b << 16), xv[2], acc[r]);
                acc[r] = fmaf(__builtin_bit_cast(float, b & 0xffff0000u), xv[3], acc[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float sum = wave_sum(acc[r]);
        if (lane == 0) out[row + r] = sum;
    }
}

// Attention for one new token over the fixed KV cache, split by position so that a few KV heads still fill the GPU.
// The grid is fixed and the length read here, so this can be replayed from a graph as the cache grows.
// One wave takes one position at a time: a K or V row of a head is 512 bytes, 16 per lane, shared by GROUPS query heads.
extern "C" __global__ __launch_bounds__(256) void attn_decode(const float* q, const float* k, const float* v, const int* lengths, float* partial, float* stats, int capacity, int kv_heads, float scale) {
    __shared__ float acc_s[WAVES][GROUPS][HEAD];
    __shared__ float max_s[WAVES][GROUPS], sum_s[WAVES][GROUPS];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int b = blockIdx.y / kv_heads, kv_head = blockIdx.y % kv_heads, split = blockIdx.x;
    const int length = lengths[b];
    const int chunk = (length + SPLITS - 1) / SPLITS;
    const int end = min((split + 1) * chunk, length);
    const int first_head = b * kv_heads * GROUPS + kv_head * GROUPS;

    f32x4 qv[GROUPS], acc[GROUPS];
    float mx[GROUPS], sum[GROUPS];
    #pragma unroll
    for (int g = 0; g < GROUPS; g++) {
        qv[g] = ((const f32x4*)(q + (size_t)(first_head + g) * HEAD))[lane] * scale;
        acc[g] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
        mx[g] = -1e30f;
        sum[g] = 0.0f;
    }
    for (int j = split * chunk + wave; j < end; j += WAVES) {
        const size_t row = ((size_t)(b * capacity + j) * kv_heads + kv_head) * HEAD;
        f32x4 kv = ((const f32x4*)(k + row))[lane], vv = ((const f32x4*)(v + row))[lane];
        #pragma unroll
        for (int g = 0; g < GROUPS; g++) {
            f32x4 d = qv[g] * kv;
            float score = wave_sum(d[0] + d[1] + d[2] + d[3]);
            float next = fmaxf(mx[g], score);
            float keep = expf(mx[g] - next), p = expf(score - next);
            sum[g] = sum[g] * keep + p;
            acc[g] = acc[g] * keep + vv * p;
            mx[g] = next;
        }
    }
    #pragma unroll
    for (int g = 0; g < GROUPS; g++) {
        ((f32x4*)acc_s[wave][g])[lane] = acc[g];
        if (lane == 0) { max_s[wave][g] = mx[g]; sum_s[wave][g] = sum[g]; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < GROUPS * HEAD; i += 256) {
        const int g = i / HEAD, d = i % HEAD;
        float top = max_s[0][g];
        #pragma unroll
        for (int w = 1; w < WAVES; w++) top = fmaxf(top, max_s[w][g]);
        float value = 0.0f, total = 0.0f;
        #pragma unroll
        for (int w = 0; w < WAVES; w++) {
            float keep = expf(max_s[w][g] - top);
            value += acc_s[w][g][d] * keep;
            total += sum_s[w][g] * keep;
        }
        const size_t at = (size_t)(first_head + g) * SPLITS + split;
        partial[at * HEAD + d] = value;
        if (d == 0) { stats[at * 2] = top; stats[at * 2 + 1] = total; }
    }
}

// Joins the splits of attn_decode. One block per query head, one thread per value.
extern "C" __global__ __launch_bounds__(HEAD) void attn_merge(const float* partial, const float* stats, float* out) {
    const size_t head = blockIdx.x;
    const float* st = stats + head * SPLITS * 2;
    float top = st[0];
    for (int s = 1; s < SPLITS; s++) top = fmaxf(top, st[s * 2]);
    float value = 0.0f, total = 0.0f;
    for (int s = 0; s < SPLITS; s++) {
        float keep = expf(st[s * 2] - top);
        value += partial[(head * SPLITS + s) * HEAD + threadIdx.x] * keep;
        total += st[s * 2 + 1] * keep;
    }
    out[head * HEAD + threadIdx.x] = value / total;
}
