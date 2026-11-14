//! ops_rvv.c - the RVV 1.0 kernel family for TensorFold's RISC-V backend.
//
// Targets SpacemiT K3's X100 (VLEN=256) and A100 (VLEN=1024): every loop uses
// scalable vectors and vsetvl, so one binary serves both, and any other RVV
// 1.0 implementation. Compile this file with the V extension on; dispatch.c
// only calls it when the CPU reports HWCAP_V at runtime.
//
// Numeric contract matches ops_scalar.c (float32 products; the reference in
// zig/src/riscv/reference.zig is the float64 oracle the tests compare to).
// Where ops_scalar accumulates in double (attention scores) this file widens
// to float64 as well, so the families agree within f32 rounding of the
// elementwise ops only.
//
// Micro-optimization ledger (kept honest, not silently applied): exp/sin/cos
// still call libm per element (a polynomial RVV exp is a later change); GEMV
// rows stream without prefetch; the m>1 matmul re-reads weight rows once per
// output row; q4_0 dequantizes a block into a scratch buffer before the dot.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "tf_riscv.h"

#include <riscv_vector.h>
#include <math.h>
#include <string.h>

static inline float half_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
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

// Horizontal sums and maxima: the reduction returns an m1 vector whose
// element 0 is the result (the overloads' operand and result are m1).
static inline double hsum_f64(vfloat64m4_t acc, size_t vl) {
    vfloat64m1_t one = __riscv_vfmv_v_f_f64m1(0.0, 1);
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum(acc, one, vl));
}

static inline float hsum_f32(vfloat32m4_t acc, size_t vl) {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum(acc, one, vl));
}

static inline float hmax_f32(vfloat32m4_t acc, size_t vl) {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(-__builtin_inff(), 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax(acc, one, vl));
}

// bf16 u16 lanes widened into f32 vector lanes (bit pattern is f32 >> 16).
static inline vfloat32m2_t load_bf16_f32m2(const uint16_t *p, size_t vl) {
    vuint16m1_t h = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t w = __riscv_vzext_vf2_u32m2(h, vl);
    w = __riscv_vsll_vx_u32m2(w, 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(w);
}

// sum(a * b) in float64 widening accumulators (score parity with the scalar).
static inline double dot_f64(const float *a, const float *b, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat64m4_t acc = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    size_t i = 0, vl;
    while (i < n) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t va = __riscv_vle32_v_f32m2(a + i, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + i, vl);
        acc = __riscv_vfwmacc(acc, va, vb, vl);
        i += vl;
    }
    return hsum_f64(acc, vlmax);
}

static inline float dot_f32(const float *a, const float *b, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0, vl;
    while (i < n) {
        vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, va, vb, vl);
        i += vl;
    }
    return hsum_f32(acc, vlmax);
}

static inline float dot_bf16(const float *x, const uint16_t *w, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0, vl;
    while (i < n) {
        vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vuint16m2_t vh = __riscv_vle16_v_u16m2(w + i, vl);
        vuint32m4_t vw = __riscv_vzext_vf2_u32m4(vh, vl);
        vw = __riscv_vsll_vx_u32m4(vw, 16, vl);
        vfloat32m4_t wf = __riscv_vreinterpret_v_u32m4_f32m4(vw);
        acc = __riscv_vfmacc_vv_f32m4(acc, vx, wf, vl);
        i += vl;
    }
    return hsum_f32(acc, vlmax);
}

void tf_riscv_rmsnorm_rvv(const float *x, const float *w, float *out, int n, float eps) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0, vl;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, v, v, vl);
        i += vl;
    }
    float ss = hsum_f32(acc, vlmax);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    i = 0;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        v = __riscv_vfmul_vf_f32m4(v, inv, vl);
        if (w) {
            vfloat32m4_t vw = __riscv_vle32_v_f32m4(w + i, vl);
            v = __riscv_vfmul_vv_f32m4(v, vw, vl);
        }
        __riscv_vse32_v_f32m4(out + i, v, vl);
        i += vl;
    }
}

// rotate_half over one head: freq[i] via one powf and a recurrence, cos/sin
// per element (libm), the rotate itself on vectors (ledger: cos/sin tables).
static inline void rope_head_rvv(float *h, int head_dim, int pos, float theta) {
    const size_t half = (size_t)head_dim / 2;
    const double step = pow((double)theta, -2.0 / (double)head_dim); // double:
    double freq = 1.0; // the recurrence keeps |freq error| under i * 2^-53
    size_t i = 0, vl;
    while (i < half) {
        vl = __riscv_vsetvl_e32m4(half - i);
        float cbuf[32], sbuf[32];
        if (vl > 32) vl = 32;
        for (size_t j = 0; j < vl; j++) {
            cbuf[j] = (float)cos((double)pos * freq);
            sbuf[j] = (float)sin((double)pos * freq);
            freq *= step;
        }
        vfloat32m4_t va = __riscv_vle32_v_f32m4(h + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(h + half + i, vl);
        vfloat32m4_t vc = __riscv_vle32_v_f32m4(cbuf, vl);
        vfloat32m4_t vs = __riscv_vle32_v_f32m4(sbuf, vl);
        vfloat32m4_t oa = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(va, vc, vl), vb, vs, vl); // a*c - b*s
        vfloat32m4_t ob = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(vb, vc, vl), va, vs, vl);  // b*c + a*s
        __riscv_vse32_v_f32m4(h + i, oa, vl);
        __riscv_vse32_v_f32m4(h + half + i, ob, vl);
        i += vl;
    }
}

void tf_riscv_rope_rvv(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta) {
    for (int h = 0; h < q_heads; h++) rope_head_rvv(q + (size_t)h * head_dim, head_dim, pos, theta);
    for (int h = 0; h < kv_heads; h++) rope_head_rvv(k + (size_t)h * head_dim, head_dim, pos, theta);
}

void tf_riscv_softmax_rvv(float *x, int n) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(-INFINITY, vlmax);
    size_t i = 0, vl;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        vmax = __riscv_vfmax_vv_f32m4(vmax, v, vl);
        i += vl;
    }
    float max = hmax_f32(vmax, vlmax);
    for (int j = 0; j < n; j++) x[j] = expf(x[j] - max); // libm per element: the ledger's exp item
    vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    i = 0;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        vsum = __riscv_vfadd_vv_f32m4(vsum, v, vl);
        i += vl;
    }
    float inv = 1.0f / hsum_f32(vsum, vlmax);
    i = 0;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        __riscv_vse32_v_f32m4(x + i, __riscv_vfmul_vf_f32m4(v, inv, vl), vl);
        i += vl;
    }
}

void tf_riscv_silu_mul_rvv(const float *gate, const float *up, float *out, int n) {
    size_t i = 0, vl;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        float sbuf[32];
        if (vl > 32) vl = 32;
        for (size_t j = 0; j < vl; j++) sbuf[j] = gate[i + j] / (1.0f + expf(-gate[i + j]));
        vfloat32m4_t vs = __riscv_vle32_v_f32m4(sbuf, vl);
        vfloat32m4_t vu = __riscv_vle32_v_f32m4(up + i, vl);
        __riscv_vse32_v_f32m4(out + i, __riscv_vfmul_vv_f32m4(vs, vu, vl), vl);
        i += vl;
    }
}

void tf_riscv_add_rvv(const float *a, const float *b, float *out, int n) {
    size_t i = 0, vl;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
        __riscv_vse32_v_f32m4(out + i, __riscv_vfadd_vv_f32m4(va, vb, vl), vl);
        i += vl;
    }
}

void tf_riscv_matmul_f32_rvv(const float *a, const float *b, float *out, int m, int n, int k, int trans_b) {
    for (int i = 0; i < m; i++) {
        const float *arow = a + (size_t)i * k;
        if (trans_b) {
            for (int j = 0; j < n; j++) {
                out[(size_t)i * n + j] = dot_f32(arow, b + (size_t)j * k, (size_t)k);
            }
        } else {
            // [k,n] rows stream into an output row: one macc pass per t.
            float *orow = out + (size_t)i * n;
            for (int j = 0; j < n; j++) orow[j] = 0.0f;
            for (int t = 0; t < k; t++) {
                float av = arow[t];
                const float *brow = b + (size_t)t * n;
                size_t j = 0, vl;
                while (j < (size_t)n) {
                    vl = __riscv_vsetvl_e32m4((size_t)n - j);
                    vfloat32m4_t vo = __riscv_vle32_v_f32m4(orow + j, vl);
                    vfloat32m4_t vb = __riscv_vle32_v_f32m4(brow + j, vl);
                    vo = __riscv_vfmacc_vf_f32m4(vo, av, vb, vl);
                    __riscv_vse32_v_f32m4(orow + j, vo, vl);
                    j += vl;
                }
            }
        }
    }
}

void tf_riscv_matmul_bf16_rvv(const float *x, const uint16_t *w, float *out, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        const float *xrow = x + (size_t)i * k;
        for (int j = 0; j < n; j++) {
            out[(size_t)i * n + j] = dot_bf16(xrow, w + (size_t)j * k, (size_t)k);
        }
    }
}

void tf_riscv_attention_f32_rvv(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale) {
    // Pass 1: the row max (scores in f64 widening like the scalar family).
    float max = -INFINITY;
    for (int t = 0; t < rows; t++) {
        float score = (float)(scale * dot_f64(q, k + (size_t)t * head_dim, (size_t)head_dim));
        if (score > max) max = score;
    }
    // Pass 2: the exponentials' sum (exp per element: the ledger's exp item).
    double sum = 0.0;
    for (int t = 0; t < rows; t++) {
        float score = (float)(scale * dot_f64(q, k + (size_t)t * head_dim, (size_t)head_dim));
        sum += exp((double)score - (double)max);
    }
    double inv = 1.0 / sum;
    // Pass 3: the value mix on vectors.
    for (int i = 0; i < head_dim; i++) out[i] = 0.0f;
    for (int t = 0; t < rows; t++) {
        float score = (float)(scale * dot_f64(q, k + (size_t)t * head_dim, (size_t)head_dim));
        float p = (float)(exp((double)score - (double)max) * inv);
        const float *vrow = v + (size_t)t * head_dim;
        size_t i = 0, vl;
        while (i < (size_t)head_dim) {
            vl = __riscv_vsetvl_e32m4((size_t)head_dim - i);
            vfloat32m4_t vo = __riscv_vle32_v_f32m4(out + i, vl);
            vfloat32m4_t vv = __riscv_vle32_v_f32m4(vrow + i, vl);
            vo = __riscv_vfmacc_vf_f32m4(vo, p, vv, vl);
            __riscv_vse32_v_f32m4(out + i, vo, vl);
            i += vl;
        }
    }
}

// GGUF q8_0 blocks: 32 xi8 + f16 scale (34 bytes). The i8 dot widens to f64.
void tf_riscv_gemv_q8_0_rvv(const float *x, const void *w, float *out, int n, int k) {
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
            size_t vl = __riscv_vsetvl_e8m1(qk);
            vint8m1_t vq = __riscv_vle8_v_i8m1(q, vl);
            vfloat32m4_t vqf = __riscv_vfcvt_f(__riscv_vsext_vf4(vq, vl), vl);
            vfloat32m4_t vxf = __riscv_vle32_v_f32m4(x + (size_t)b * qk, vl);
            vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(0.0, vl);
            acc = __riscv_vfwmacc(acc, vxf, vqf, vl);
            vfloat64m1_t one = __riscv_vfmv_v_f_f64m1(0.0, 1);
            s += (double)d * __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum(acc, one, vl));
        }
        out[j] = (float)s;
    }
}

// GGUF q4_0 blocks: 16 packed bytes (32 u4, lo nibble first) + f16 scale
// (18 bytes); value = scale * (q - 8). The block dequantizes to a scratch
// row (the ledger keeps this simple; the decode path never sees q4_0 today).
void tf_riscv_gemv_q4_0_rvv(const float *x, const void *w, float *out, int n, int k) {
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
            float block[32];
            for (int i = 0; i < qk / 2; i++) {
                uint8_t packed = q[i];
                block[2 * i] = d * (float)((int)(packed & 0x0fu) - 8); // int: u4 - 8 goes negative
                block[2 * i + 1] = d * (float)((int)(packed >> 4) - 8);
            }
            s += dot_f64(x + (size_t)b * qk, block, qk);
        }
        out[j] = (float)s;
    }
}
