//! attention_block.c - causal attention for a block of query rows.
//
// The per-row kernel (ops_*.c tf_riscv_attention_f32) scans the whole K/V
// panel once per query row: for a T-token prefill that re-reads the panel T
// times per head (measured: attention was 3.6 s of the 7.2 s pp512 run on the
// K3). This kernel takes nb query rows at once and walks the key rows in the
// OUTER loop: each K and V row is read once per block.
//
// The score matrix is a transposed product: for one key row all nb query dots
// land in a single vector (k's elements broadcast against the columns of the
// transposed query block), so there is no per-pair reduction - the pair form
// paid a vfredusum + scalar move per score and measured ~50 cycles each.
// Scores live in key-major layout [rows][nb]: one contiguous store per key
// row, then the softmax works row-wise against per-column max/sum vectors and
// the value mix reads the probabilities as scalars. Rows past a query's
// causal prefix hold -inf, which makes the max, the sum and the mix skip them
// for free.
//
// Causal mask exact: query i (0-based in the block) attends key rows
// [0, rows - nb + i].
//
// Score dots accumulate in f32 and every kernel family runs this same code,
// so the families agree exactly; the tests hold it against the float64
// reference like the per-row kernel.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "tf_riscv.h"

#include <math.h>
#include <string.h>

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

// The same degree-6 exp polynomial as the RVV family (~2 ulp), so all three
// kernel families produce identical attention for identical inputs.
static inline float ab_expf(float x) {
    if (!(x > -87.0f)) return 0.0f;
    if (x > 88.0f) return expf(x);
    const float log2e = 1.4426950408889634f;
    float t = x * log2e;
    float kf = roundf(t);
    float r = t - kf;
    float p = 1.0f + r * (0.6931471805599453f + r * (0.2402265069591007f + r * (0.0555041086648216f + r * (0.0096181291076284f + r * (0.0013333558146428f + r * 0.0001540353039338f)))));
    return ldexpf(p, (int)kf);
}

#if defined(__riscv) && defined(__riscv_vector)
// Vector exp over n floats in place: the same polynomial, but the 2^k
// scaling goes through the float bit pattern instead of libm's ldexpf
// (measured 105 ns per scalar call on the A100 tiles - 1.5 s of pp512).
void tf_riscv_expf_vec(float *x, int n) {
    const float log2e = 1.4426950408889634f;
    size_t i = 0;
    while (i < (size_t)n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
        vbool8_t small = __riscv_vmfle_vf_f32m4_b8(v, -87.0f, vl);
        vfloat32m4_t t = __riscv_vfmul_vf_f32m4(v, log2e, vl);
        vint32m4_t k = __riscv_vfcvt_x_f_v_i32m4(t, vl);
        vfloat32m4_t r = __riscv_vfsub_vv_f32m4(t, __riscv_vfcvt_f_x_v_f32m4(k, vl), vl);
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(0.0001540353039338f, vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.0013333558146428f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.0096181291076284f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.0555041086648216f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.2402265069591007f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.6931471805599453f, vl), vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 1.0f, vl);
        vint32m4_t bits = __riscv_vreinterpret_v_f32m4_i32m4(p);
        bits = __riscv_vadd_vv_i32m4(bits, __riscv_vsll_vx_i32m4(k, 23, vl), vl);
        vfloat32m4_t e = __riscv_vreinterpret_v_i32m4_f32m4(bits);
        e = __riscv_vfmerge_vfm_f32m4(e, 0.0f, small, vl);
        __riscv_vse32_v_f32m4(x + i, e, vl);
        i += vl;
    }
}

// out = silu(gate) * up = gate / (1 + exp(-gate)) * up, fully vector: the
// scalar family's per-element exp measured 1.8 s of the 5.6 s pp512 run.
void tf_riscv_silu_mul_vec(const float *gate, const float *up, float *out, int n) {
    const float log2e = 1.4426950408889634f;
    size_t i = 0;
    while (i < (size_t)n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)n - i);
        vfloat32m4_t g = __riscv_vle32_v_f32m4(gate + i, vl);
        vfloat32m4_t t = __riscv_vfmul_vf_f32m4(__riscv_vfneg_v_f32m4(g, vl), log2e, vl);
        vint32m4_t k = __riscv_vfcvt_x_f_v_i32m4(t, vl);
        vfloat32m4_t r = __riscv_vfsub_vv_f32m4(t, __riscv_vfcvt_f_x_v_f32m4(k, vl), vl);
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(0.0001540353039338f, vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.0013333558146428f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.0096181291076284f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.0555041086648216f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.2402265069591007f, vl), vl);
        p = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), __riscv_vfmv_v_f_f32m4(0.6931471805599453f, vl), vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 1.0f, vl);
        vint32m4_t bits = __riscv_vreinterpret_v_f32m4_i32m4(p);
        bits = __riscv_vadd_vv_i32m4(bits, __riscv_vsll_vx_i32m4(k, 23, vl), vl);
        vfloat32m4_t e = __riscv_vreinterpret_v_i32m4_f32m4(bits);
        vfloat32m4_t s = __riscv_vfdiv_vv_f32m4(g, __riscv_vfadd_vf_f32m4(e, 1.0f, vl), vl);
        vfloat32m4_t u = __riscv_vle32_v_f32m4(up + i, vl);
        __riscv_vse32_v_f32m4(out + i, __riscv_vfmul_vv_f32m4(s, u, vl), vl);
        i += vl;
    }
}
#else
void tf_riscv_expf_vec(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = ab_expf(x[i]);
}
void tf_riscv_silu_mul_vec(const float *gate, const float *up, float *out, int n) {
    for (int j = 0; j < n; j++) out[j] = (gate[j] / (1.0f + expf(-gate[j]))) * up[j];
}
#endif

float tf_riscv_attention_block_f32_dot(const float *a, const float *b, int n);

#if defined(__riscv) && defined(__riscv_vector)
// Generic f32 dot, used by the small-nb path below.
static float ab_dot(const float *a, const float *b, int n) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t c1 = c0;
    size_t i = 0;
    while (i + 2 * vlmax <= (size_t)n) {
        c0 = __riscv_vfmacc_vv_f32m2(c0, __riscv_vle32_v_f32m2(a + i, vlmax), __riscv_vle32_v_f32m2(b + i, vlmax), vlmax);
        c1 = __riscv_vfmacc_vv_f32m2(c1, __riscv_vle32_v_f32m2(a + i + vlmax, vlmax), __riscv_vle32_v_f32m2(b + i + vlmax, vlmax), vlmax);
        i += 2 * vlmax;
    }
    while (i < (size_t)n) {
        size_t vl = __riscv_vsetvl_e32m2((size_t)n - i);
        c0 = __riscv_vfmacc_vv_f32m2(c0, __riscv_vle32_v_f32m2(a + i, vl), __riscv_vle32_v_f32m2(b + i, vl), vl);
        i += vl;
    }
    c0 = __riscv_vfadd_vv_f32m2(c0, c1, vlmax);
    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(c0, z, vlmax));
}
#else
static float ab_dot(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

float tf_riscv_attention_block_f32_dot(const float *a, const float *b, int n) { return ab_dot(a, b, n); }

// The KV cache writes copy head_dim floats per (token, head): going through
// the Zig runtime's variable-length memcpy measured 826 ns per 256-byte copy
// at pp512 (95 ms of the run for 115 k copies). This is one vector pair per
// register width.
void tf_riscv_copy_f32(float *dst, const float *src, int n) {
#if defined(__riscv) && defined(__riscv_vector)
    size_t i = 0;
    while (i < (size_t)n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)n - i);
        __riscv_vse32_v_f32m4(dst + i, __riscv_vle32_v_f32m4(src + i, vl), vl);
        i += vl;
    }
#else
    for (int j = 0; j < n; j++) dst[j] = src[j];
#endif
}

// `scratch` holds nb*rows + head_dim*nb floats: the score block plus the
// transposed query block. The caller allocates it once per shard (the pool's
// TCM arena inside an A100 tile, heap elsewhere).
void tf_riscv_attention_block_f32(const float *q, int q_stride, const float *k, const float *v, float *out, int out_stride,
                                  int nb, int rows, int head_dim, float scale, float *scratch) {
    if (nb <= 0 || rows < nb || !scratch) return;
    float *restrict skey = scratch;                       // [rows][nb], key-major
    float *restrict qT = scratch + (size_t)nb * rows;     // [head_dim][nb]

#if defined(__riscv) && defined(__riscv_vector)
    const size_t v32 = __riscv_vsetvlmax_e32m1();
    if (nb >= 8 && head_dim <= 128 && (size_t)nb <= v32) {
        // ---- fast path: transposed scores, row-wise softmax, grouped mix ----
        const size_t vl = __riscv_vsetvl_e32m1((size_t)nb);
        // Q transposed to [head_dim][nb]: the score inner loop streams one
        // 128-byte row per k element and broadcasts k's scalar.
        for (int d = 0; d < head_dim; d++) {
            vfloat32m1_t vv = __riscv_vlse32_v_f32m1(q + d, (long)q_stride * 4, vl);
            __riscv_vse32_v_f32m1(qT + (size_t)d * nb, vv, vl);
        }
        // Scores: four key rows per group so the fma chains stay independent
        // (a single chain is latency-bound; the pair form was load-bound).
        int kk = 0;
        for (; kk + 8 <= rows; kk += 8) {
            // Eight independent chains: the accumulator dependency distance
            // grows to eight d-steps, which keeps the fma pipe full (the
            // four-chain form was latency-bound at ~10 ns per pair).
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            for (int d = 0; d < head_dim; d++) {
                vfloat32m1_t qv = __riscv_vle32_v_f32m1(qT + (size_t)d * nb, vl);
                a0 = __riscv_vfmacc_vf_f32m1(a0, k[(size_t)(kk + 0) * head_dim + d], qv, vl);
                a1 = __riscv_vfmacc_vf_f32m1(a1, k[(size_t)(kk + 1) * head_dim + d], qv, vl);
                a2 = __riscv_vfmacc_vf_f32m1(a2, k[(size_t)(kk + 2) * head_dim + d], qv, vl);
                a3 = __riscv_vfmacc_vf_f32m1(a3, k[(size_t)(kk + 3) * head_dim + d], qv, vl);
                a4 = __riscv_vfmacc_vf_f32m1(a4, k[(size_t)(kk + 4) * head_dim + d], qv, vl);
                a5 = __riscv_vfmacc_vf_f32m1(a5, k[(size_t)(kk + 5) * head_dim + d], qv, vl);
                a6 = __riscv_vfmacc_vf_f32m1(a6, k[(size_t)(kk + 6) * head_dim + d], qv, vl);
                a7 = __riscv_vfmacc_vf_f32m1(a7, k[(size_t)(kk + 7) * head_dim + d], qv, vl);
            }
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 0) * nb, a0, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 1) * nb, a1, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 2) * nb, a2, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 3) * nb, a3, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 4) * nb, a4, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 5) * nb, a5, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 6) * nb, a6, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 7) * nb, a7, vl);
        }
        for (; kk + 4 <= rows; kk += 4) {
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a1 = a0, a2 = a0, a3 = a0;
            for (int d = 0; d < head_dim; d++) {
                vfloat32m1_t qv = __riscv_vle32_v_f32m1(qT + (size_t)d * nb, vl);
                a0 = __riscv_vfmacc_vf_f32m1(a0, k[(size_t)(kk + 0) * head_dim + d], qv, vl);
                a1 = __riscv_vfmacc_vf_f32m1(a1, k[(size_t)(kk + 1) * head_dim + d], qv, vl);
                a2 = __riscv_vfmacc_vf_f32m1(a2, k[(size_t)(kk + 2) * head_dim + d], qv, vl);
                a3 = __riscv_vfmacc_vf_f32m1(a3, k[(size_t)(kk + 3) * head_dim + d], qv, vl);
            }
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 0) * nb, a0, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 1) * nb, a1, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 2) * nb, a2, vl);
            __riscv_vse32_v_f32m1(skey + (size_t)(kk + 3) * nb, a3, vl);
        }
        for (; kk < rows; kk++) {
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (int d = 0; d < head_dim; d++) {
                a0 = __riscv_vfmacc_vf_f32m1(a0, k[(size_t)kk * head_dim + d], __riscv_vle32_v_f32m1(qT + (size_t)d * nb, vl), vl);
            }
            __riscv_vse32_v_f32m1(skey + (size_t)kk * nb, a0, vl);
        }
        // The attention scale belongs on the scores before the softmax.
        {
            vfloat32m1_t sv = __riscv_vfmv_v_f_f32m1(scale, vl);
            for (int k2 = 0; k2 < rows; k2++) {
                float *row = skey + (size_t)k2 * nb;
                __riscv_vse32_v_f32m1(row, __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(row, vl), sv, vl), vl);
            }
        }
        // Causal mask: key row kk is invalid for query lanes b < kk-rows+nb.
        // Those entries become -inf so the max, the sum and the mix skip them.
        {
            uint32_t idx[32];
            for (int j = 0; j < 32; j++) idx[j] = (uint32_t)j;
            vuint32m1_t lv = __riscv_vle32_v_u32m1(idx, vl);
            for (int k2 = 0; k2 < rows; k2++) {
                const int t = k2 - rows + nb;
                if (t <= 0) continue;
                float *row = skey + (size_t)k2 * nb;
                vfloat32m1_t rv = __riscv_vle32_v_f32m1(row, vl);
                vbool32_t bad = __riscv_vmsltu_vx_u32m1_b32(lv, (uint32_t)t, vl);
                rv = __riscv_vfmerge_vfm_f32m1(rv, -INFINITY, bad, vl);
                __riscv_vse32_v_f32m1(row, rv, vl);
            }
        }
        // Per-column max and sum as 32-lane accumulators over the rows: the
        // whole softmax is row-wise, no strided passes at all.
        vfloat32m1_t maxv = __riscv_vfmv_v_f_f32m1(-INFINITY, vl);
        for (int k2 = 0; k2 < rows; k2++) {
            maxv = __riscv_vfmax_vv_f32m1(maxv, __riscv_vle32_v_f32m1(skey + (size_t)k2 * nb, vl), vl);
        }
        vfloat32m1_t sumv = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        for (int k2 = 0; k2 < rows; k2++) {
            float *row = skey + (size_t)k2 * nb;
            vfloat32m1_t rv = __riscv_vle32_v_f32m1(row, vl);
            rv = __riscv_vfsub_vv_f32m1(rv, maxv, vl);
            __riscv_vse32_v_f32m1(row, rv, vl);
        }
        for (int k2 = 0; k2 < rows; k2++) {
            float *row = skey + (size_t)k2 * nb;
            tf_riscv_expf_vec(row, nb);
            sumv = __riscv_vfadd_vv_f32m1(sumv, __riscv_vle32_v_f32m1(row, vl), vl);
        }
        {
            float sums[32];
            __riscv_vse32_v_f32m1(sums, sumv, vl);
            float inv[32];
            for (int b = 0; b < nb; b++) inv[b] = sums[b] > 0.0f ? 1.0f / sums[b] : 0.0f;
            // Scale the probabilities in place: one vector multiply per key
            // row instead of a scalar multiply per (key row, query row) in
            // the mix below (43 M scalar ops per pp512 tile otherwise).
            vfloat32m1_t invv = __riscv_vle32_v_f32m1(inv, vl);
            for (int k2 = 0; k2 < rows; k2++) {
                float *row = skey + (size_t)k2 * nb;
                __riscv_vse32_v_f32m1(row, __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(row, vl), invv, vl), vl);
            }
        }
        // Value mix: four query rows per group share each V row's load and
        // keep their outputs in registers across the key rows.
        for (int b0 = 0; b0 < nb; b0 += 4) {
            const int bw = (nb - b0) < 4 ? (nb - b0) : 4;
            // Chunk count follows the actual register width: vsetvlmax is 64
            // lanes at the A100's VLEN=1024 but 16 at VLEN=256, and a fixed
            // two chunks covered only half the output there (caught by the
            // qemu VLEN=256 gate).
            const size_t v64 = __riscv_vsetvlmax_e32m2();
            for (size_t c = 0; (size_t)c * v64 < (size_t)head_dim; c++) {
                const size_t vlc = __riscv_vsetvl_e32m2((size_t)head_dim - c * v64);
                vfloat32m2_t o0 = __riscv_vfmv_v_f_f32m2(0.0f, vlc);
                vfloat32m2_t o1 = o0, o2 = o0, o3 = o0;
                for (int k2 = 0; k2 < rows; k2++) {
                    const float *vr = v + (size_t)k2 * head_dim + (size_t)c * v64;
                    const float *pr = skey + (size_t)k2 * nb + b0;
                    vfloat32m2_t vv = __riscv_vle32_v_f32m2(vr, vlc);
                    o0 = __riscv_vfmacc_vf_f32m2(o0, pr[0], vv, vlc);
                    if (bw > 1) o1 = __riscv_vfmacc_vf_f32m2(o1, pr[1], vv, vlc);
                    if (bw > 2) o2 = __riscv_vfmacc_vf_f32m2(o2, pr[2], vv, vlc);
                    if (bw > 3) o3 = __riscv_vfmacc_vf_f32m2(o3, pr[3], vv, vlc);
                }
                __riscv_vse32_v_f32m2(out + (size_t)(b0 + 0) * out_stride + (size_t)c * v64, o0, vlc);
                if (bw > 1) __riscv_vse32_v_f32m2(out + (size_t)(b0 + 1) * out_stride + (size_t)c * v64, o1, vlc);
                if (bw > 2) __riscv_vse32_v_f32m2(out + (size_t)(b0 + 2) * out_stride + (size_t)c * v64, o2, vlc);
                if (bw > 3) __riscv_vse32_v_f32m2(out + (size_t)(b0 + 3) * out_stride + (size_t)c * v64, o3, vlc);
            }
        }
        return;
    }
#endif

    // ---- generic path (small nb or no RVV): per-pair dots ----
    for (int kk = 0; kk < rows; kk++) {
        const float *kr = k + (size_t)kk * head_dim;
        for (int b = 0; b < nb; b++) {
            if (kk > rows - nb + b) continue;
            scratch[(size_t)b * rows + kk] = scale * ab_dot(q + (size_t)b * q_stride, kr, head_dim);
        }
    }
    for (int b = 0; b < nb; b++) {
        const int prefix = rows - nb + b + 1;
        float *sc = scratch + (size_t)b * rows;
        float max = -INFINITY;
        for (int kk = 0; kk < prefix; kk++)
            if (sc[kk] > max) max = sc[kk];
        for (int kk = 0; kk < prefix; kk++) sc[kk] -= max;
        tf_riscv_expf_vec(sc, prefix);
        float sum = 0.0f;
        for (int kk = 0; kk < prefix; kk++) sum += sc[kk];
        const float inv = 1.0f / sum;
        float *o = out + (size_t)b * out_stride;
        for (int i = 0; i < head_dim; i++) o[i] = 0.0f;
        for (int kk = 0; kk < prefix; kk++) sc[kk] *= inv;
        for (int kk = 0; kk < prefix; kk++) {
            const float p = sc[kk];
            const float *vr = v + (size_t)kk * head_dim;
            for (int i = 0; i < head_dim; i++) o[i] += p * vr[i];
        }
    }
}
