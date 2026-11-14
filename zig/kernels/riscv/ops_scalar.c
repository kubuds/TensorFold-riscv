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

void tf_riscv_attention_f32_scalar(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale) {
    // Scores in a caller-provided window would bound memory; this baseline keeps
    // one VLA-free pass with two loops over the cache instead (no allocation).
    float max = -INFINITY;
    for (int t = 0; t < rows; t++) {
        const float *krow = k + (size_t)t * head_dim;
        double s = 0.0;
        for (int i = 0; i < head_dim; i++) s += (double)q[i] * (double)krow[i];
        float score = (float)((double)scale * s);
        if (score > max) max = score;
    }
    double sum = 0.0;
    for (int t = 0; t < rows; t++) {
        const float *krow = k + (size_t)t * head_dim;
        double s = 0.0;
        for (int i = 0; i < head_dim; i++) s += (double)q[i] * (double)krow[i];
        sum += exp((double)scale * s - (double)max);
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < head_dim; i++) out[i] = 0.0f;
    for (int t = 0; t < rows; t++) {
        const float *krow = k + (size_t)t * head_dim;
        double s = 0.0;
        for (int i = 0; i < head_dim; i++) s += (double)q[i] * (double)krow[i];
        double p = exp((double)scale * s - (double)max) * inv;
        const float *vrow = v + (size_t)t * head_dim;
        for (int i = 0; i < head_dim; i++) out[i] += (float)(p * (double)vrow[i]);
    }
}

// GGUF q8_0: 32 int8 + f16 scale, 34-byte blocks.
void tf_riscv_gemv_q8_0_scalar(const float *x, const void *w, float *out, int n, int k) {
    const int qk = 32;
    const uint8_t *wb = (const uint8_t *)w;
    for (int j = 0; j < n; j++) {
        const uint8_t *row = wb + (size_t)j * (k / qk) * 34;
        double s = 0.0;
        for (int b = 0; b < k / qk; b++) {
            uint16_t sh;
            memcpy(&sh, row + b * 34, 2);
            float d = half_to_f32(sh);
            const int8_t *q = (const int8_t *)(row + b * 34 + 2);
            for (int i = 0; i < qk; i++) s += (double)x[b * qk + i] * ((double)d * (double)q[i]);
        }
        out[j] = (float)s;
    }
}

// GGUF q4_0: 16 bytes of u4 (32 values, lo nibble first) + f16 scale, 18-byte blocks.
// Dequantized value = scale * (q - 8), the llama.cpp convention.
void tf_riscv_gemv_q4_0_scalar(const float *x, const void *w, float *out, int n, int k) {
    const int qk = 32;
    const uint8_t *wb = (const uint8_t *)w;
    for (int j = 0; j < n; j++) {
        const uint8_t *row = wb + (size_t)j * (k / qk) * 18;
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
        out[j] = (float)s;
    }
}
