//! ops_scalar.c - the RV64GC scalar kernel baseline for TensorFold's RISC-V backend.
//
// Portable C11 with no ISA intrinsics: this file is the correctness oracle every
// vectorized family (RVV, IME2) is tested against, and the fallback a binary
// uses when the CPU reports no usable vector unit.
//
// Numeric contract: accumulations are float32 except where a comment says
// double; the tests in zig/src/riscv/kernel_test.zig check each kernel against
// a float64 reference with a tolerance that grows with the reduction length.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "tf_riscv.h"

#include <math.h>
#include <string.h>

static inline float bf16_to_f32(uint16_t b) {
    union {
        uint32_t u;
        float f;
    } v = {.u = (uint32_t)b << 16};
    return v.f;
}

static inline float half_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else { // subnormal: normalize
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp -= 1;
            }
            man &= 0x3ffu;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    union {
        uint32_t u;
        float f;
    } v = {.u = out};
    return v.f;
}

void tf_riscv_rmsnorm_scalar(const float *x, const float *w, float *out, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// rotate_half(x) = [-x[half..], x[..half]]; out = x*cos + rotate_half(x)*sin
// with cos/sin held once per half (the HF Qwen3 layout).
static inline void rope_head(float *h, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        double freq = pow((double)theta, -2.0 * (double)i / (double)head_dim);
        float c = (float)cos((double)pos * freq);
        float s = (float)sin((double)pos * freq);
        float a = h[i];
        float b = h[half + i];
        h[i] = a * c - b * s;
        h[half + i] = b * c + a * s;
    }
}

void tf_riscv_rope_scalar(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta) {
    for (int h = 0; h < q_heads; h++) rope_head(q + (size_t)h * head_dim, head_dim, pos, theta);
    for (int h = 0; h < kv_heads; h++) rope_head(k + (size_t)h * head_dim, head_dim, pos, theta);
}

// Table + apply (the scalar oracle stays libm; the layout matches the RVV
// family exactly so the families agree bit-for-bit in the tests).
void tf_riscv_rope_table_scalar(int pos, float *s, float *c, int heads, int head_dim, float theta) {
    const size_t half = (size_t)head_dim / 2;
    for (int h = 0; h < heads; h++) {
        double step = pow((double)theta, -2.0 / (double)head_dim);
        double freq = 1.0;
        for (size_t i = 0; i < half; i++) {
            s[(size_t)h * half + i] = (float)sin((double)pos * freq);
            c[(size_t)h * half + i] = (float)cos((double)pos * freq);
            freq *= step;
        }
    }
}

void tf_riscv_rope_apply_scalar(float *q, float *k, const float *s, const float *c, int q_heads, int kv_heads, int head_dim) {
    const size_t half = (size_t)head_dim / 2;
    for (int h = 0; h < q_heads; h++) {
        for (size_t i = 0; i < half; i++) {
            const float a = q[(size_t)h * head_dim + i];
            const float b = q[(size_t)h * head_dim + half + i];
            const float cs = c[(size_t)h * half + i];
            const float sn = s[(size_t)h * half + i];
            q[(size_t)h * head_dim + i] = a * cs - b * sn;
            q[(size_t)h * head_dim + half + i] = b * cs + a * sn;
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (size_t i = 0; i < half; i++) {
            const float a = k[(size_t)h * head_dim + i];
            const float b = k[(size_t)h * head_dim + half + i];
            const float cs = c[(size_t)h * half + i];
            const float sn = s[(size_t)h * half + i];
            k[(size_t)h * head_dim + i] = a * cs - b * sn;
            k[(size_t)h * head_dim + half + i] = b * cs + a * sn;
        }
    }
}

void tf_riscv_softmax_scalar(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void tf_riscv_silu_mul_scalar(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

void tf_riscv_add_scalar(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void tf_riscv_matmul_f32_scalar(const float *a, const float *b, float *out, int m, int n, int k, int trans_b) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            if (trans_b) {
                const float *brow = b + (size_t)j * k;
                for (int t = 0; t < k; t++) s += a[(size_t)i * k + t] * brow[t];
            } else {
                for (int t = 0; t < k; t++) s += a[(size_t)i * k + t] * b[(size_t)t * n + j];
            }
            out[(size_t)i * n + j] = s;
        }
    }
}

void tf_riscv_matmul_bf16_scalar(const float *x, const uint16_t *w, float *out, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        const float *xrow = x + (size_t)i * k;
        for (int j = 0; j < n; j++) {
            const uint16_t *wrow = w + (size_t)j * k;
            float s = 0.0f;
            for (int t = 0; t < k; t++) s += xrow[t] * bf16_to_f32(wrow[t]);
            out[(size_t)i * n + j] = s;
        }
    }
}

// Weight rows reused across a 64-row output block: prefill reads each weight
// row ceil(m/64) times instead of m times (the m>1 weight-traffic fix).
#define TF_RISCV_ROW_BLOCK 64

static inline float dot_bf16_scalar(const float *x, const uint16_t *w, int k) {
    float s = 0.0f;
    for (int t = 0; t < k; t++) s += x[t] * bf16_to_f32(w[t]);
    return s;
}

static inline float dot_f32_scalar(const float *a, const float *b, int k) {
    float s = 0.0f;
    for (int t = 0; t < k; t++) s += a[t] * b[t];
    return s;
}

void tf_riscv_matmul_bf16_shard_scalar(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1) {
    if (m == 1) {
        for (int j = j0; j < j1; j++) out[j] = dot_bf16_scalar(x, w + (size_t)j * k, k);
        return;
    }
    for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
        int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
        for (int j = j0; j < j1; j++) {
            const uint16_t *wrow = w + (size_t)j * k;
            for (int i = 0; i < im; i++) {
                out[(size_t)(i0 + i) * n + j] = dot_bf16_scalar(x + (size_t)(i0 + i) * k, wrow, k);
            }
        }
    }
}

void tf_riscv_matmul_f32_shard_scalar(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1) {
    if (trans_b) {
        if (m == 1) {
            for (int j = j0; j < j1; j++) out[j] = dot_f32_scalar(a, b + (size_t)j * k, k);
            return;
        }
        for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
            int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
            for (int j = j0; j < j1; j++) {
                const float *brow = b + (size_t)j * k;
                for (int i = 0; i < im; i++) {
                    out[(size_t)(i0 + i) * n + j] = dot_f32_scalar(a + (size_t)(i0 + i) * k, brow, k);
                }
            }
        }
    } else {
        // [k,n] rows stream into the shard's columns only.
        for (int i = 0; i < m; i++) {
            float *orow = out + (size_t)i * n;
            for (int j = j0; j < j1; j++) orow[j] = 0.0f;
            for (int t = 0; t < k; t++) {
                float av = a[(size_t)i * k + t];
                const float *brow = b + (size_t)t * n;
                for (int j = j0; j < j1; j++) orow[j] += av * brow[j];
            }
        }
    }
}

void tf_riscv_attention_f32_scalar(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch) {
    // One dot per key row into the caller's score buffer, then exp and mix.
    float max = -INFINITY;
    for (int t = 0; t < rows; t++) {
        const float *krow = k + (size_t)t * head_dim;
        double s = 0.0;
        for (int i = 0; i < head_dim; i++) s += (double)q[i] * (double)krow[i];
        float score = (float)((double)scale * s);
        scratch[t] = score;
        if (score > max) max = score;
    }
    double sum = 0.0;
    for (int t = 0; t < rows; t++) {
        scratch[t] = expf(scratch[t] - max);
        sum += scratch[t];
    }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < head_dim; i++) out[i] = 0.0f;
    for (int t = 0; t < rows; t++) {
        float p = scratch[t] * inv;
        const float *vrow = v + (size_t)t * head_dim;
        for (int i = 0; i < head_dim; i++) out[i] += p * vrow[i];
    }
}

// GGUF q8_0: 32 xi8 + f16 scale per block.
static inline float dot_q8_0_scalar(const float *x, const uint8_t *row, int k) {
    const int qk = 32;
    double s = 0.0;
    for (int b = 0; b < k / qk; b++) {
        uint16_t sh;
        memcpy(&sh, row + b * 34, 2);
        float d = half_to_f32(sh);
        const int8_t *q = (const int8_t *)(row + b * 34 + 2);
        for (int i = 0; i < qk; i++) s += (double)x[b * qk + i] * ((double)d * (double)q[i]);
    }
    return (float)s;
}

void tf_riscv_gemv_q8_0_scalar(const float *x, const void *w, float *out, int n, int k) {
    const uint8_t *wb = (const uint8_t *)w;
    for (int j = 0; j < n; j++) out[j] = dot_q8_0_scalar(x, wb + (size_t)j * (k / 32) * 34, k);
}

// GGUF q4_0: 16 packed bytes (32 u4, lo nibble first) + f16 scale (18 bytes);
// value = scale * (q - 8).
static inline float dot_q4_0_scalar(const float *x, const uint8_t *row, int k) {
    const int qk = 32;
    double s = 0.0;
    for (int b = 0; b < k / qk; b++) {
        uint16_t sh;
        memcpy(&sh, row + b * 18, 2);
        float d = half_to_f32(sh);
        const uint8_t *q = row + b * 18 + 2;
        for (int i = 0; i < qk / 2; i++) {
            uint8_t packed = q[i];
            s += (double)x[b * qk + 2 * i] * ((double)d * ((double)(packed & 0x0fu) - 8.0));
            s += (double)x[b * qk + 2 * i + 1] * ((double)d * ((double)(packed >> 4) - 8.0));
        }
    }
    return (float)s;
}

void tf_riscv_gemv_q4_0_scalar(const float *x, const void *w, float *out, int n, int k) {
    const uint8_t *wb = (const uint8_t *)w;
    for (int j = 0; j < n; j++) out[j] = dot_q4_0_scalar(x, wb + (size_t)j * (k / 32) * 18, k);
}

// Quantized shards: weight rows reused across 64-row output blocks.
void tf_riscv_matmul_q8_0_shard_scalar(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    const uint8_t *wb = (const uint8_t *)w;
    const size_t row_bytes = (size_t)(k / 32) * 34;
    for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
        int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
        for (int j = j0; j < j1; j++) {
            const uint8_t *wrow = wb + (size_t)j * row_bytes;
            for (int i = 0; i < im; i++) {
                out[(size_t)(i0 + i) * n + j] = dot_q8_0_scalar(x + (size_t)(i0 + i) * k, wrow, k);
            }
        }
    }
}

void tf_riscv_matmul_q4_0_shard_scalar(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    const uint8_t *wb = (const uint8_t *)w;
    const size_t row_bytes = (size_t)(k / 32) * 18;
    for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
        int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
        for (int j = j0; j < j1; j++) {
            const uint8_t *wrow = wb + (size_t)j * row_bytes;
            for (int i = 0; i < im; i++) {
                out[(size_t)(i0 + i) * n + j] = dot_q4_0_scalar(x + (size_t)(i0 + i) * k, wrow, k);
            }
        }
    }
}
