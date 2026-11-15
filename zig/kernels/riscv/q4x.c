//! q4x.c - TensorFold's 4-bit weight family for the SpacemiT K3.
//
// q4x is packed for SpacemiT's IME2 dot instruction (smt.vmadot: an 8x8x16
// int8 -> int32 tensor-core MAC, 1024 MACs per instruction) while keeping a
// portable RVV float path over the same bytes.
//
// ## Weight layout (8-row tiles)
//
// A row block is 8 output rows. Per 32-k chunk one tile is 128 bytes:
//
//   byte t (t = 0..127): low nibble  = w[r0 + t/16][k0 + t%16]
//                        high nibble = w[r0 + t/16][k0 + 16 + t%16]
//
// One whole-register load plus `vand 0xF` yields the 8x16 int8 tile for
// k0..k0+15 and `vsrl 4` yields k0+16..k0+31: two vmadot operands per load,
// no shuffles (see q4x_ime2.c). Both nibble planes are also contiguous in k,
// so the float path reads its activation vector without a de-interleave.
// Nibbles store (q + 8); `vadd -8` makes them signed.
//
// Per row block: scales, then tiles.
//   scales: (k/256) groups x 8 rows x f16     (w ~= q_signed * d)
//   tiles:  (k/32) x 128 bytes
// Row-block stride = 4k + k/16: 4.0625 bits per weight at k=1024.
//
// ## Activation layout (the A operand)
//
// Activations are quantized to int8 with one f32 scale per 256-value group and
// tiled the way vmadot wants them: for each 8-row block and 16-k chunk, 128
// bytes = 8 rows x 16 k, plus a per-row group scale array. Decode pads the tile
// to eight rows with zeros.
//
// The IME2 kernels live in q4x_ime2.c (VLEN=1024 only: X100's IME1 dot has a
// narrower operand and traps on vmadot). This file is the portable float
// family, the scalar oracle the tests compare against, and the dispatcher.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "tf_riscv.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#define Q4X_GROUP 256
#define Q4X_ROWS 8
#define Q4X_TILE 128

size_t tf_riscv_q4x_block_bytes(int k) { return (size_t)4 * (size_t)k + (size_t)k / 16; }
size_t tf_riscv_q4x_row_bytes(int k) { return tf_riscv_q4x_block_bytes(k) / Q4X_ROWS; }
int tf_riscv_q4x_ok(int k) { return k > 0 && (k % Q4X_GROUP) == 0; }
int tf_riscv_q4x_perm_k(int u) { return u; } // the tile layout has no k permutation

// ---- shared scalar pieces --------------------------------------------------

static float q4x_half_to_f32(uint16_t h) {
#if defined(__riscv_zfh) || defined(__riscv_zfhmin)
    _Float16 f;
    memcpy(&f, &h, 2);
    return (float)f;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = 0;
            do {
                man <<= 1;
                e++;
            } while ((man & 0x400) == 0);
            man &= 0x3ff;
            bits = sign | (uint32_t)(127 - 15 - e) << 23 | man << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | man << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | man << 13;
    }
    union {
        uint32_t u;
        float f;
    } v = {.u = bits};
    return v.f;
#endif
}

// Signed nibble of a 32-k tile for output row `r`, k offset `u` (0..31).
static inline int q4x_tile_val(const uint8_t *tile, int r, int u) {
    const uint8_t b = tile[r * 16 + (u & 15)];
    const int v = (u < 16) ? (int)(b & 0x0f) : (int)(b >> 4);
    return v - 8;
}

static inline int8_t q4x_round_i8(float v, float inv) {
    float q = lrintf(v * inv);
    if (q > 127.0f) q = 127.0f;
    if (q < -127.0f) q = -127.0f;
    return (int8_t)q;
}

// Activation tile of (activation row i, 16-k chunk kc) inside the packed A.
static inline const int8_t *q4x_a_tile(const int8_t *xa, int i, int k, int kc) {
    return xa + ((size_t)(i / Q4X_ROWS) * (k / 16) + kc) * Q4X_TILE + (i % Q4X_ROWS) * 16;
}

// ---- scalar oracle ---------------------------------------------------------

void tf_riscv_quant_a_scalar(const float *x, int8_t *xa, float *xs, int m, int k) {
    const int groups = k / Q4X_GROUP;
    const int mblocks = (m + Q4X_ROWS - 1) / Q4X_ROWS;
    memset(xa, 0, (size_t)mblocks * Q4X_ROWS * k);
    for (int i = 0; i < m; i++) {
        for (int g = 0; g < groups; g++) {
            const float *src = x + (size_t)i * k + g * Q4X_GROUP;
            float amax = 0.0f;
            for (int j = 0; j < Q4X_GROUP; j++) amax = fmaxf(amax, fabsf(src[j]));
            const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            xs[(size_t)i * groups + g] = d;
            for (int c = 0; c < Q4X_GROUP / 16; c++) {
                const int kc = g * (Q4X_GROUP / 16) + c;
                int8_t *tile = (int8_t *)q4x_a_tile(xa, i, k, kc);
                for (int u = 0; u < 16; u++) tile[u] = q4x_round_i8(src[c * 16 + u], inv);
            }
        }
    }
}

// out[i][j] = sum_g d_w[j][g] * d_a[i][g] * sum_k q_w * q_a for one (i, j).
static float q4x_dot_row_scalar(const int8_t *xa, const float *xs, const uint8_t *block, int i, int r, int k) {
    const int groups = k / Q4X_GROUP;
    const uint8_t *tiles = block + (size_t)groups * Q4X_ROWS * 2;
    float acc = 0.0f;
    for (int g = 0; g < groups; g++) {
        const uint16_t *sc = (const uint16_t *)(block + (size_t)g * Q4X_ROWS * 2);
        const float dw = q4x_half_to_f32(sc[r]);
        int32_t isum = 0;
        for (int c = 0; c < Q4X_GROUP / 32; c++) {
            const int kc = g * (Q4X_GROUP / 32) + c;
            const uint8_t *tile = tiles + (size_t)kc * Q4X_TILE;
            const int8_t *a0 = q4x_a_tile(xa, i, k, 2 * kc);
            const int8_t *a1 = q4x_a_tile(xa, i, k, 2 * kc + 1);
            for (int u = 0; u < 16; u++) {
                isum += q4x_tile_val(tile, r, u) * a0[u];
                isum += q4x_tile_val(tile, r, 16 + u) * a1[u];
            }
        }
        acc += (float)isum * dw * xs[(size_t)i * groups + g];
    }
    return acc;
}

void tf_riscv_gemv_q4x_scalar(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k) {
    tf_riscv_gemv_q4x_shard_scalar(xa, xs, w, out, n, k, 0, n);
}

void tf_riscv_gemv_q4x_shard_scalar(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1) {
    (void)n;
    const size_t stride = tf_riscv_q4x_block_bytes(k);
    for (int j = j0; j < j1; j++) {
        const uint8_t *block = (const uint8_t *)w + (size_t)(j / Q4X_ROWS) * stride;
        out[j] = q4x_dot_row_scalar(xa, xs, block, 0, j % Q4X_ROWS, k);
    }
}

void tf_riscv_gemm_q4x_shard_scalar(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    const size_t stride = tf_riscv_q4x_block_bytes(k);
    for (int j = j0; j < j1; j++) {
        const uint8_t *block = (const uint8_t *)w + (size_t)(j / Q4X_ROWS) * stride;
        for (int i = 0; i < m; i++) out[(size_t)i * n + j] = q4x_dot_row_scalar(xa, xs, block, i, j % Q4X_ROWS, k);
    }
}

// ---- RVV family (portable float path) --------------------------------------

#if defined(__riscv) && defined(__riscv_vector)

void tf_riscv_quant_a_rvv(const float *x, int8_t *xa, float *xs, int m, int k) {
    const int groups = k / Q4X_GROUP;
    const int mblocks = (m + Q4X_ROWS - 1) / Q4X_ROWS;
    memset(xa, 0, (size_t)mblocks * Q4X_ROWS * k);
    for (int i = 0; i < m; i++) {
        for (int g = 0; g < groups; g++) {
            const float *src = x + (size_t)i * k + g * Q4X_GROUP;
            float amax = 0.0f;
            // The stride must follow vl: at VLEN=256 e32m4 is 32 lanes, and a
            // fixed 64-step sampled only half the group (the group scale then
            // differed from the scalar oracle whenever the max sat in the
            // skipped half).
            for (int j = 0; j < Q4X_GROUP;) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(Q4X_GROUP - j));
                vfloat32m4_t v = __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(src + j, vl), vl);
                vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
                vfloat32m1_t mx = __riscv_vfredmax_vs_f32m4_f32m1(v, z, vl);
                amax = fmaxf(amax, __riscv_vfmv_f_s_f32m1_f32(mx));
                j += (int)vl;
            }
            const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            xs[(size_t)i * groups + g] = d;
            for (int c = 0; c < Q4X_GROUP / 16; c++) {
                const int kc = g * (Q4X_GROUP / 16) + c;
                int8_t *tile = (int8_t *)q4x_a_tile(xa, i, k, kc);
                const float *p = src + c * 16;
                // Narrowing needs the source register group to be twice the
                // destination LMUL, so the whole 16-lane chunk runs at m2:
                // i32m2 (16 lanes at VLEN=256) -> i16m1 -> i8mf2. The old
                // i32m1/mf4 chain only stored 8 bytes per chunk at VLEN=256:
                // the half-zero activation that made the X100 quant differ
                // from the scalar oracle.
                size_t vl = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v = __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(p, vl), inv, vl);
                vint32m2_t iv = __riscv_vfcvt_x_f_v_i32m2(v, vl);
                iv = __riscv_vmax_vx_i32m2(iv, -127, vl);
                iv = __riscv_vmin_vx_i32m2(iv, 127, vl);
                vint16m1_t mid = __riscv_vncvt_x_x_w_i16m1(iv, vl);
                __riscv_vse8_v_i8mf2(tile, __riscv_vncvt_x_x_w_i8mf2(mid, vl), vl);
            }
        }
    }
}

// One weight row over the tile layout. Both nibble planes are contiguous 16-k
// halves, and the activation arrives int8-tiled, so a tile iteration is one
// 16-byte weight load, one 16-byte activation load and two fused convert-mul-
// accumulate chains. Used on CPUs without IME2 (X100: IME1 has no 8x8x16 dot).
static inline float q4x_dot_row_rvv(const int8_t *xa, const float *xs, const uint8_t *block, int i, int r, int k) {
    const int groups = k / Q4X_GROUP;
    const uint8_t *tiles = block + (size_t)groups * Q4X_ROWS * 2;
    const size_t vlm = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
    for (int g = 0; g < groups; g++) {
        const uint16_t *sc = (const uint16_t *)(block + (size_t)g * Q4X_ROWS * 2);
        const float dw = q4x_half_to_f32(sc[r]);
        const float da = xs[(size_t)i * groups + g];
        const float d = dw * da;
        for (int c = 0; c < Q4X_GROUP / 32; c++) {
            const int kc = g * (Q4X_GROUP / 32) + c;
            const uint8_t *row = tiles + (size_t)kc * Q4X_TILE + r * 16;
            size_t vl = __riscv_vsetvl_e8m1(16);
            vuint8m1_t v = __riscv_vle8_v_u8m1(row, vl);
            const int8_t *a0 = q4x_a_tile(xa, i, k, 2 * kc);
            const int8_t *a1 = q4x_a_tile(xa, i, k, 2 * kc + 1);
            vint8m1_t wlo = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vadd_vx_u8m1(__riscv_vand_vx_u8m1(v, 0x0f, vl), (uint8_t)-8, vl));
            vint8m1_t whi = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vadd_vx_u8m1(__riscv_vsrl_vx_u8m1(v, 4, vl), (uint8_t)-8, vl));
            // int8 x int8 -> f32 through a product of two int->f32 conversions;
            // fma with the group scale folded into the activation side.
            vfloat32m4_t fw0 = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf4_i32m4(wlo, vl), vl);
            vfloat32m4_t fw1 = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf4_i32m4(whi, vl), vl);
            vfloat32m4_t fa0 = __riscv_vfcvt_f_x_v_f32m4(
                __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(a0, vl), vl), vl);
            vfloat32m4_t fa1 = __riscv_vfcvt_f_x_v_f32m4(
                __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(a1, vl), vl), vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, __riscv_vfmul_vf_f32m4(fw0, d, vl), fa0, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, __riscv_vfmul_vf_f32m4(fw1, d, vl), fa1, vl);
        }
    }
    __riscv_vsetvl_e32m4(vlm);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vlm);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

void tf_riscv_gemv_q4x_float(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1) {
    (void)n;
    const size_t stride = tf_riscv_q4x_block_bytes(k);
    for (int j = j0; j < j1; j++) {
        const uint8_t *block = (const uint8_t *)w + (size_t)(j / Q4X_ROWS) * stride;
        out[j] = q4x_dot_row_rvv(xa, xs, block, 0, j % Q4X_ROWS, k);
    }
}

void tf_riscv_gemm_q4x_float(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    const size_t stride = tf_riscv_q4x_block_bytes(k);
    for (int j = j0; j < j1; j++) {
        const uint8_t *block = (const uint8_t *)w + (size_t)(j / Q4X_ROWS) * stride;
        for (int i = 0; i < m; i++) out[(size_t)i * n + j] = q4x_dot_row_rvv(xa, xs, block, i, j % Q4X_ROWS, k);
    }
}

#endif // __riscv && __riscv_vector

// ---- runtime entry points --------------------------------------------------

static int q4x_scalar(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TF_RISCV_Q4X");
        cached = (env && strcmp(env, "scalar") == 0) ? 1 : 0;
    }
    return cached;
}

static int q4x_ime2(void) {
#if defined(__riscv) && defined(__riscv_vector)
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TF_RISCV_Q4X");
        if (env && strcmp(env, "float") == 0) {
            cached = 0;
        } else {
            unsigned long vlenb = 0;
            __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
            cached = vlenb == 128; // vmadot's operands are 128-byte registers
        }
    }
    return cached;
#else
    return 0;
#endif
}

void tf_riscv_quant_a(const float *x, int8_t *xa, float *xs, int m, int k) {
#if defined(__riscv) && defined(__riscv_vector)
    if (!q4x_scalar()) {
        tf_riscv_quant_a_rvv(x, xa, xs, m, k);
        return;
    }
#endif
    tf_riscv_quant_a_scalar(x, xa, xs, m, k);
}

void tf_riscv_gemv_q4x(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k) {
    tf_riscv_gemv_q4x_shard(xa, xs, w, out, n, k, 0, n);
}

void tf_riscv_gemv_q4x_shard(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1) {
    if (q4x_scalar()) {
        tf_riscv_gemv_q4x_shard_scalar(xa, xs, w, out, n, k, j0, j1);
        return;
    }
    if (q4x_ime2()) {
        tf_riscv_gemv_q4x_ime2(xa, xs, w, out, n, k, j0, j1);
        return;
    }
#if defined(__riscv) && defined(__riscv_vector)
    tf_riscv_gemv_q4x_float(xa, xs, w, out, n, k, j0, j1);
#else
    tf_riscv_gemv_q4x_shard_scalar(xa, xs, w, out, n, k, j0, j1);
#endif
}

void tf_riscv_gemm_q4x_shard(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    if (q4x_scalar()) {
        tf_riscv_gemm_q4x_shard_scalar(xa, xs, w, out, m, n, k, j0, j1);
        return;
    }
    if (q4x_ime2()) {
        tf_riscv_gemm_q4x_ime2(xa, xs, w, out, m, n, k, j0, j1);
        return;
    }
#if defined(__riscv) && defined(__riscv_vector)
    tf_riscv_gemm_q4x_float(xa, xs, w, out, m, n, k, j0, j1);
#else
    tf_riscv_gemm_q4x_shard_scalar(xa, xs, w, out, m, n, k, j0, j1);
#endif
}
