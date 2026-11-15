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

// Fast libm replacements for the family's per-element transcendentals: a
// decode token calls exp ~370k times and sin/cos ~170k times through softmax,
// SwiGLU and RoPE, and glibc's correctly-rounded calls dominate that time.
// Accuracy: exp ~2 ulp (2^k * degree-6 poly on [-1/2,1/2]), sin/cos ~2 ulp
// (double range reduction, degree-9/8 Taylor on [-pi,pi]); the kernel tests
// hold the 1e-5 family tolerances against the float64 reference. The scalar
// family keeps libm and stays the oracle.
static inline float fast_expf(float x) {
    if (!(x > -87.0f)) return 0.0f; // NaNs fall through to the scaling path
    if (x > 88.0f) return expf(x);
    const float log2e = 1.4426950408889634f;
    float t = x * log2e;
    float k = roundf(t);
    float r = t - k;
    float p = 1.0f + r * (0.6931471805599453f + r * (0.2402265069591007f + r * (0.0555041086648216f + r * (0.0096181291076284f + r * (0.0013333558146428f + r * 0.0001540353039338f)))));
    return ldexpf(p, (int)k);
}

static inline void fast_sincosf(double x, float *s, float *c) {
    // The angle stays double end to end (a float angle alone costs 2e-4 at
    // pos 4096) and the range reduces to a quadrant: the Taylor series on
    // [-pi/4, pi/4] is accurate to ~1e-12, while on [-pi, pi] its truncation
    // is 5e-5 - which measured on K3 as rope drift.
    const double half_pi = 1.5707963267948966;
    double k4 = round(x / half_pi);
    double r = x - k4 * half_pi; // [-pi/4, pi/4]
    double r2 = r * r;
    double sp = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362880.0)))));
    double cp = 1.0 + r2 * (-1.0 / 2.0 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40320.0))));
    long q = (long)k4 & 3;
    if (k4 < 0) q = (q + 4) & 3;
    switch (q) {
        case 0:
            *s = (float)sp;
            *c = (float)cp;
            break;
        case 1:
            *s = (float)cp;
            *c = (float)-sp;
            break;
        case 2:
            *s = (float)-sp;
            *c = (float)-cp;
            break;
        default:
            *s = (float)-cp;
            *c = (float)sp;
            break;
    }
}

// bf16 u16 lanes widened into f32 vector lanes (bit pattern is f32 >> 16).
static inline vfloat32m2_t load_bf16_f32m2(const uint16_t *p, size_t vl) {
    vuint16m1_t h = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t w = __riscv_vzext_vf2_u32m2(h, vl);
    w = __riscv_vsll_vx_u32m2(w, 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(w);
}

// bf16 u16 lanes at m4 width (dot_bf16's unrolled body).
static inline vfloat32m4_t load_bf16_f32m4(const uint16_t *p, size_t vl) {
    vuint16m2_t h = __riscv_vle16_v_u16m2(p, vl);
    vuint32m4_t w = __riscv_vzext_vf2_u32m4(h, vl);
    w = __riscv_vsll_vx_u32m4(w, 16, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(w);
}

// sum(a * b) in float64 widening accumulators (score parity with the scalar).
// Four independent accumulator chains: one vfmacc chain per vector chunk keeps
// the FMA pipes busy while the loads stream (the latency-hiding unroll the
// IPC=0.61 profile called for).
static inline double dot_f64(const float *a, const float *b, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat64m4_t acc0 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t acc1 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t acc2 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t acc3 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    size_t i = 0;
    while (n - i >= 4 * vlmax) {
        vfloat32m2_t a0 = __riscv_vle32_v_f32m2(a + i, vlmax);
        vfloat32m2_t b0 = __riscv_vle32_v_f32m2(b + i, vlmax);
        vfloat32m2_t a1 = __riscv_vle32_v_f32m2(a + i + vlmax, vlmax);
        vfloat32m2_t b1 = __riscv_vle32_v_f32m2(b + i + vlmax, vlmax);
        vfloat32m2_t a2 = __riscv_vle32_v_f32m2(a + i + 2 * vlmax, vlmax);
        vfloat32m2_t b2 = __riscv_vle32_v_f32m2(b + i + 2 * vlmax, vlmax);
        vfloat32m2_t a3 = __riscv_vle32_v_f32m2(a + i + 3 * vlmax, vlmax);
        vfloat32m2_t b3 = __riscv_vle32_v_f32m2(b + i + 3 * vlmax, vlmax);
        acc0 = __riscv_vfwmacc(acc0, a0, b0, vlmax);
        acc1 = __riscv_vfwmacc(acc1, a1, b1, vlmax);
        acc2 = __riscv_vfwmacc(acc2, a2, b2, vlmax);
        acc3 = __riscv_vfwmacc(acc3, a3, b3, vlmax);
        i += 4 * vlmax;
    }
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t va = __riscv_vle32_v_f32m2(a + i, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + i, vl);
        acc0 = __riscv_vfwmacc(acc0, va, vb, vl);
        i += vl;
    }
    acc0 = __riscv_vfadd_vv_f64m4(acc0, acc1, vlmax);
    acc2 = __riscv_vfadd_vv_f64m4(acc2, acc3, vlmax);
    acc0 = __riscv_vfadd_vv_f64m4(acc0, acc2, vlmax);
    return hsum_f64(acc0, vlmax);
}

// Four accumulator chains (see dot_f64 for why).
static inline float dot_f32(const float *a, const float *b, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    while (n - i >= 4 * vlmax) {
        vfloat32m4_t a0 = __riscv_vle32_v_f32m4(a + i, vlmax);
        vfloat32m4_t b0 = __riscv_vle32_v_f32m4(b + i, vlmax);
        vfloat32m4_t a1 = __riscv_vle32_v_f32m4(a + i + vlmax, vlmax);
        vfloat32m4_t b1 = __riscv_vle32_v_f32m4(b + i + vlmax, vlmax);
        vfloat32m4_t a2 = __riscv_vle32_v_f32m4(a + i + 2 * vlmax, vlmax);
        vfloat32m4_t b2 = __riscv_vle32_v_f32m4(b + i + 2 * vlmax, vlmax);
        vfloat32m4_t a3 = __riscv_vle32_v_f32m4(a + i + 3 * vlmax, vlmax);
        vfloat32m4_t b3 = __riscv_vle32_v_f32m4(b + i + 3 * vlmax, vlmax);
        acc0 = __riscv_vfmacc_vv_f32m4(acc0, a0, b0, vlmax);
        acc1 = __riscv_vfmacc_vv_f32m4(acc1, a1, b1, vlmax);
        acc2 = __riscv_vfmacc_vv_f32m4(acc2, a2, b2, vlmax);
        acc3 = __riscv_vfmacc_vv_f32m4(acc3, a3, b3, vlmax);
        i += 4 * vlmax;
    }
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
        acc0 = __riscv_vfmacc_vv_f32m4(acc0, va, vb, vl);
        i += vl;
    }
    acc0 = __riscv_vfadd_vv_f32m4(acc0, acc1, vlmax);
    acc2 = __riscv_vfadd_vv_f32m4(acc2, acc3, vlmax);
    acc0 = __riscv_vfadd_vv_f32m4(acc0, acc2, vlmax);
    return hsum_f32(acc0, vlmax);
}

// bf16 lanes widened on the fly; four accumulator chains (see dot_f64).
static inline float dot_bf16(const float *x, const uint16_t *w, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    size_t i = 0;
    while (n - i >= 4 * vlmax) {
        vfloat32m4_t x0 = __riscv_vle32_v_f32m4(x + i, vlmax);
        vfloat32m4_t w0 = load_bf16_f32m4(w + i, vlmax);
        vfloat32m4_t x1 = __riscv_vle32_v_f32m4(x + i + vlmax, vlmax);
        vfloat32m4_t w1 = load_bf16_f32m4(w + i + vlmax, vlmax);
        vfloat32m4_t x2 = __riscv_vle32_v_f32m4(x + i + 2 * vlmax, vlmax);
        vfloat32m4_t w2 = load_bf16_f32m4(w + i + 2 * vlmax, vlmax);
        vfloat32m4_t x3 = __riscv_vle32_v_f32m4(x + i + 3 * vlmax, vlmax);
        vfloat32m4_t w3 = load_bf16_f32m4(w + i + 3 * vlmax, vlmax);
        acc0 = __riscv_vfmacc_vv_f32m4(acc0, x0, w0, vlmax);
        acc1 = __riscv_vfmacc_vv_f32m4(acc1, x1, w1, vlmax);
        acc2 = __riscv_vfmacc_vv_f32m4(acc2, x2, w2, vlmax);
        acc3 = __riscv_vfmacc_vv_f32m4(acc3, x3, w3, vlmax);
        i += 4 * vlmax;
    }
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t wf = load_bf16_f32m4(w + i, vl);
        acc0 = __riscv_vfmacc_vv_f32m4(acc0, vx, wf, vl);
        i += vl;
    }
    acc0 = __riscv_vfadd_vv_f32m4(acc0, acc1, vlmax);
    acc2 = __riscv_vfadd_vv_f32m4(acc2, acc3, vlmax);
    acc0 = __riscv_vfadd_vv_f32m4(acc0, acc2, vlmax);
    return hsum_f32(acc0, vlmax);
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

// The per-position (sin, cos) table, built once per token. The angles are
// reduced in double (a float-only reduction costs 2e-4 at pos 4096, measured
// as rope drift on the K3), the polynomials run as f32 vectors, and the
// quadrant choice is a vector merge. Same math as fast_sincosf modulo f32
// polynomial rounding (~1e-7, inside the family tolerance).
void tf_riscv_rope_table_rvv(int pos, float *s, float *c, int heads, int head_dim, float theta) {
    const size_t half = (size_t)head_dim / 2;
    const double step = pow((double)theta, -2.0 / (double)head_dim);
    const double two_over_pi = 2.0 / 3.14159265358979323846;
    const double half_pi = 1.5707963267948966;
    for (int h = 0; h < heads; h++) {
        double freq = 1.0;
        float *sp = s + (size_t)h * half;
        float *cp = c + (size_t)h * half;
        size_t i = 0;
        while (i < half) {
            size_t vl = __riscv_vsetvl_e32m4(half - i);
            float rbuf[32], qbuf[32];
            if (vl > 32) vl = 32;
            for (size_t j = 0; j < vl; j++) {
                const double angle = (double)pos * freq;
                const double k4 = round(angle * two_over_pi);
                rbuf[j] = (float)(angle - k4 * half_pi);
                qbuf[j] = (float)((long)k4 & 3);
                freq *= step;
            }
            vfloat32m4_t r = __riscv_vle32_v_f32m4(rbuf, vl);
            vfloat32m4_t r2 = __riscv_vfmul_vv_f32m4(r, r, vl);
            const float c1 = 1.0f, c2 = -1.0f / 6.0f, c3 = 1.0f / 120.0f, c4 = -1.0f / 5040.0f, c5 = 1.0f / 362880.0f;
            const float d1 = 1.0f, d2 = -1.0f / 2.0f, d3 = 1.0f / 24.0f, d4 = -1.0f / 720.0f, d5 = 1.0f / 40320.0f;
            // Horner: each step is prev*r2 + coeff, i.e. vfmadd_vv with the
            // coefficient broadcast as a vector (the _vf variant adds instead
            // of multiplying the accumulated value).
            vfloat32m4_t spoly = __riscv_vfmv_v_f_f32m4(c5, vl);
            spoly = __riscv_vfmadd_vv_f32m4(spoly, r2, __riscv_vfmv_v_f_f32m4(c4, vl), vl);
            spoly = __riscv_vfmadd_vv_f32m4(spoly, r2, __riscv_vfmv_v_f_f32m4(c3, vl), vl);
            spoly = __riscv_vfmadd_vv_f32m4(spoly, r2, __riscv_vfmv_v_f_f32m4(c2, vl), vl);
            spoly = __riscv_vfmadd_vv_f32m4(spoly, r2, __riscv_vfmv_v_f_f32m4(c1, vl), vl);
            spoly = __riscv_vfmul_vv_f32m4(spoly, r, vl);
            vfloat32m4_t cpoly = __riscv_vfmv_v_f_f32m4(d5, vl);
            cpoly = __riscv_vfmadd_vv_f32m4(cpoly, r2, __riscv_vfmv_v_f_f32m4(d4, vl), vl);
            cpoly = __riscv_vfmadd_vv_f32m4(cpoly, r2, __riscv_vfmv_v_f_f32m4(d3, vl), vl);
            cpoly = __riscv_vfmadd_vv_f32m4(cpoly, r2, __riscv_vfmv_v_f_f32m4(d2, vl), vl);
            cpoly = __riscv_vfmadd_vv_f32m4(cpoly, r2, __riscv_vfmv_v_f_f32m4(d1, vl), vl);
            vfloat32m4_t qv = __riscv_vle32_v_f32m4(qbuf, vl);
            vbool8_t q0 = __riscv_vmfeq_vf_f32m4_b8(qv, 0.0f, vl);
            vbool8_t q1 = __riscv_vmfeq_vf_f32m4_b8(qv, 1.0f, vl);
            vbool8_t q2 = __riscv_vmfeq_vf_f32m4_b8(qv, 2.0f, vl);
            vbool8_t q3 = __riscv_vmfeq_vf_f32m4_b8(qv, 3.0f, vl);
            // vmerge_vvm(false_val, true_val, mask): mask selects the SECOND
            // operand (probed on the K3: all-true mask returns op2).
            vfloat32m4_t negsp = __riscv_vfneg_v_f32m4(spoly, vl);
            vfloat32m4_t negcp = __riscv_vfneg_v_f32m4(cpoly, vl);
            vfloat32m4_t sn = spoly;
            sn = __riscv_vmerge_vvm_f32m4(sn, cpoly, q1, vl);
            sn = __riscv_vmerge_vvm_f32m4(sn, negsp, q2, vl);
            sn = __riscv_vmerge_vvm_f32m4(sn, negcp, q3, vl);
            vfloat32m4_t cs = cpoly;
            cs = __riscv_vmerge_vvm_f32m4(cs, negsp, q1, vl);
            cs = __riscv_vmerge_vvm_f32m4(cs, negcp, q2, vl);
            cs = __riscv_vmerge_vvm_f32m4(cs, spoly, q3, vl);
            __riscv_vse32_v_f32m4(sp + i, sn, vl);
            __riscv_vse32_v_f32m4(cp + i, cs, vl);
            i += vl;
        }
    }
}

// Apply the precomputed table: pure vector rotation, no transcendentals.
void tf_riscv_rope_apply_rvv(float *q, float *k, const float *s, const float *c, int q_heads, int kv_heads, int head_dim) {
    const size_t half = (size_t)head_dim / 2;
    for (int h = 0; h < q_heads; h++) {
        float *hh = q + (size_t)h * head_dim;
        const float *sp = s + (size_t)h * half;
        const float *cp = c + (size_t)h * half;
        size_t i = 0;
        while (i < half) {
            size_t vl = __riscv_vsetvl_e32m4(half - i);
            vfloat32m4_t va = __riscv_vle32_v_f32m4(hh + i, vl);
            vfloat32m4_t vb = __riscv_vle32_v_f32m4(hh + half + i, vl);
            vfloat32m4_t vc = __riscv_vle32_v_f32m4(cp + i, vl);
            vfloat32m4_t vs = __riscv_vle32_v_f32m4(sp + i, vl);
            vfloat32m4_t oa = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(va, vc, vl), vb, vs, vl);
            vfloat32m4_t ob = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(vb, vc, vl), va, vs, vl);
            __riscv_vse32_v_f32m4(hh + i, oa, vl);
            __riscv_vse32_v_f32m4(hh + half + i, ob, vl);
            i += vl;
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        float *hh = k + (size_t)h * head_dim;
        const float *sp = s + (size_t)h * half;
        const float *cp = c + (size_t)h * half;
        size_t i = 0;
        while (i < half) {
            size_t vl = __riscv_vsetvl_e32m4(half - i);
            vfloat32m4_t va = __riscv_vle32_v_f32m4(hh + i, vl);
            vfloat32m4_t vb = __riscv_vle32_v_f32m4(hh + half + i, vl);
            vfloat32m4_t vc = __riscv_vle32_v_f32m4(cp + i, vl);
            vfloat32m4_t vs = __riscv_vle32_v_f32m4(sp + i, vl);
            vfloat32m4_t oa = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(va, vc, vl), vb, vs, vl);
            vfloat32m4_t ob = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(vb, vc, vl), va, vs, vl);
            __riscv_vse32_v_f32m4(hh + i, oa, vl);
            __riscv_vse32_v_f32m4(hh + half + i, ob, vl);
            i += vl;
        }
    }
}

// The combined entry keeps the old signature for the tests: table + apply.
void tf_riscv_rope_rvv(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta) {
    const int heads = q_heads > kv_heads ? q_heads : kv_heads;
    const size_t half = (size_t)head_dim / 2;
    float *sbuf = (float *)tf_riscv_scratch((size_t)heads * half * 2 * sizeof(float));
    float *cbuf = sbuf + (size_t)heads * half;
    tf_riscv_rope_table_rvv(pos, sbuf, cbuf, heads, head_dim, theta);
    tf_riscv_rope_apply_rvv(q, k, sbuf, cbuf, q_heads, kv_heads, head_dim);
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
    for (int j = 0; j < n; j++) x[j] = fast_expf(x[j] - max);
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
        for (size_t j = 0; j < vl; j++) sbuf[j] = gate[i + j] / (1.0f + fast_expf(-gate[i + j]));
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

// One output-column shard for the multithreaded path; m > 1 re-reads each
// weight row every 64 output rows (the m>1 weight-traffic fix).
#define TF_RISCV_ROW_BLOCK 64

void tf_riscv_matmul_bf16_shard_rvv(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1) {
    if (m == 1) {
        for (int j = j0; j < j1; j++) {
            if (j + 1 < j1) __builtin_prefetch(w + (size_t)(j + 1) * k, 0, 1); // next row
            out[j] = dot_bf16(x, w + (size_t)j * k, (size_t)k);
        }
        return;
    }
    for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
        int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
        for (int j = j0; j < j1; j++) {
            if (j + 1 < j1) __builtin_prefetch(w + (size_t)(j + 1) * k, 0, 1);
            const uint16_t *wrow = w + (size_t)j * k;
            for (int i = 0; i < im; i++) {
                out[(size_t)(i0 + i) * n + j] = dot_bf16(x + (size_t)(i0 + i) * k, wrow, (size_t)k);
            }
        }
    }
}

void tf_riscv_matmul_f32_shard_rvv(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1) {
    if (trans_b) {
        if (m == 1) {
            for (int j = j0; j < j1; j++) {
                if (j + 1 < j1) __builtin_prefetch(b + (size_t)(j + 1) * k, 0, 1);
                out[j] = dot_f32(a, b + (size_t)j * k, (size_t)k);
            }
            return;
        }
        for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
            int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
            for (int j = j0; j < j1; j++) {
                const float *brow = b + (size_t)j * k;
                for (int i = 0; i < im; i++) {
                    out[(size_t)(i0 + i) * n + j] = dot_f32(a + (size_t)(i0 + i) * k, brow, (size_t)k);
                }
            }
        }
    } else {
        for (int i = 0; i < m; i++) {
            float *orow = out + (size_t)i * n;
            for (int j = j0; j < j1; j++) orow[j] = 0.0f;
            for (int t = 0; t < k; t++) {
                float av = a[(size_t)i * k + t];
                const float *brow = b + (size_t)t * n;
                size_t j = (size_t)j0, vl;
                while (j < (size_t)j1) {
                    vl = __riscv_vsetvl_e32m4((size_t)j1 - j);
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

void tf_riscv_attention_f32_rvv(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch) {
    // One dot per key row (f64 widening like the scalar family), then the
    // exponentials and the value mix from the score buffer.
    float max = -INFINITY;
    for (int t = 0; t < rows; t++) {
        float score = (float)(scale * dot_f64(q, k + (size_t)t * head_dim, (size_t)head_dim));
        scratch[t] = score;
        if (score > max) max = score;
    }
    float sum = 0.0f;
    for (int t = 0; t < rows; t++) {
        scratch[t] = fast_expf(scratch[t] - max);
        sum += scratch[t];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < head_dim; i++) out[i] = 0.0f;
    for (int t = 0; t < rows; t++) {
        float p = scratch[t] * inv;
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

// GGUF q8_0 blocks: 32 xi8 + f16 scale (34 bytes). The i8 lanes widen once
// per block into an f32 accumulator that runs across blocks: one reduce per
// row, not one per block (the per-block f64 reduce was the slow path).
static inline float dot_q8_0(const float *x, const uint8_t *row, int k) {
    const int qk = 32;
    size_t vl = __riscv_vsetvl_e32m4(qk);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int b = 0; b < k / qk; b++) {
        uint16_t sh;
        memcpy(&sh, row + b * 34, 2);
        float d = half_to_f32(sh);
        const int8_t *q = (const int8_t *)(row + b * 34 + 2);
        vint8m1_t vq = __riscv_vle8_v_i8m1(q, __riscv_vsetvl_e8m1(qk));
        vfloat32m4_t vqf = __riscv_vfcvt_f(__riscv_vsext_vf4(vq, __riscv_vsetvl_e8m1(qk)), vl);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + (size_t)b * qk, vl);
        vfloat32m4_t vd = __riscv_vfmul_vf_f32m4(vx, d, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, vd, vqf, vl);
    }
    return hsum_f32(acc, __riscv_vsetvl_e32m4(qk));
}

void tf_riscv_gemv_q8_0_rvv(const float *x, const void *w, float *out, int n, int k) {
    const uint8_t *wb = (const uint8_t *)w;
    for (int j = 0; j < n; j++) out[j] = dot_q8_0(x, wb + (size_t)j * (k / 32) * 34, k);
}

// GGUF q4_0 blocks: 16 packed bytes (32 u4, lo nibble first) + f16 scale
// (18 bytes); value = scale * (q - 8). The block dequantizes to a scratch
// row; the f32 accumulator runs across blocks like q8_0's.
static inline float dot_q4_0(const float *x, const uint8_t *row, int k) {
    const int qk = 32;
    size_t vl = __riscv_vsetvl_e32m4(qk);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int b = 0; b < k / qk; b++) {
        uint16_t sh;
        memcpy(&sh, row + b * 18, 2);
        float d = half_to_f32(sh);
        const uint8_t *q = row + b * 18 + 2;
        float block[32];
        for (int i = 0; i < qk / 2; i++) {
            uint8_t packed = q[i];
            block[2 * i] = (float)((int)(packed & 0x0fu) - 8); // int: u4 - 8 goes negative
            block[2 * i + 1] = (float)((int)(packed >> 4) - 8);
        }
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(block, vl);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + (size_t)b * qk, vl);
        vfloat32m4_t vd = __riscv_vfmul_vf_f32m4(vx, d, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, vd, vb, vl);
    }
    return hsum_f32(acc, __riscv_vsetvl_e32m4(qk));
}

void tf_riscv_gemv_q4_0_rvv(const float *x, const void *w, float *out, int n, int k) {
    const uint8_t *wb = (const uint8_t *)w;
    for (int j = 0; j < n; j++) out[j] = dot_q4_0(x, wb + (size_t)j * (k / 32) * 18, k);
}

// Quantized shards: weight rows reused across 64-row output blocks.
void tf_riscv_matmul_q8_0_shard_rvv(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    const uint8_t *wb = (const uint8_t *)w;
    const size_t row_bytes = (size_t)(k / 32) * 34;
    for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
        int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
        for (int j = j0; j < j1; j++) {
            if (j + 1 < j1) __builtin_prefetch(wb + (size_t)(j + 1) * row_bytes, 0, 1);
            const uint8_t *wrow = wb + (size_t)j * row_bytes;
            for (int i = 0; i < im; i++) {
                out[(size_t)(i0 + i) * n + j] = dot_q8_0(x + (size_t)(i0 + i) * k, wrow, k);
            }
        }
    }
}

void tf_riscv_matmul_q4_0_shard_rvv(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    const uint8_t *wb = (const uint8_t *)w;
    const size_t row_bytes = (size_t)(k / 32) * 18;
    for (int i0 = 0; i0 < m; i0 += TF_RISCV_ROW_BLOCK) {
        int im = m - i0 < TF_RISCV_ROW_BLOCK ? m - i0 : TF_RISCV_ROW_BLOCK;
        for (int j = j0; j < j1; j++) {
            if (j + 1 < j1) __builtin_prefetch(wb + (size_t)(j + 1) * row_bytes, 0, 1);
            const uint8_t *wrow = wb + (size_t)j * row_bytes;
            for (int i = 0; i < im; i++) {
                out[(size_t)(i0 + i) * n + j] = dot_q4_0(x + (size_t)(i0 + i) * k, wrow, k);
            }
        }
    }
}
