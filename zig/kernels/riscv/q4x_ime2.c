//! q4x_ime2.c - the IME2 (vmadot) kernels for TensorFold's q4x weights.
//
// SpacemiT's IME2 dot instruction, `smt.vmadot`, is a 8x8x16 int8 -> int32
// tensor-core MAC: 1024 MACs per instruction, measured at 1.15 ns per tile on
// the K3's A100 cores (7 TMAC/s across eight tiles, tools/riscv/probe_k3.c).
// Its operands and accumulator are vector registers at VLEN=1024:
//
//   vmadot vd, vs1(A), vs2(B)   [e8,m1; VL=128 bytes]
//     vd[m*8 + n] += sum_{k=0..15} A[m*16 + k] * B[n*16 + k]
//   A: 8 activation rows x 16 k, B: 8 output rows x 16 k, vd: 64 int32.
//   The encoding carries vd/2, so vd must be 4-register aligned.
//
// q4x's weight tiles are shaped for exactly this (see q4x.c): one 128-byte
// load yields B for k0..k0+15 in the low nibbles and k0+16..k0+31 in the high
// nibbles, so a 32-k tile is two vmadot operands with no shuffles.
//
// The accumulator lives in fixed vector registers across the inner loop's asm
// blocks: RVV vector types are sizeless (no arrays) and there is no way to pin
// an intrinsic value to a named register, so the asm blocks name their
// registers and declare them clobbered - the compiler then keeps live values
// out of them while the hardware keeps accumulating.
//
// The int32 accumulator is folded into f32 once per 256-value group, where the
// weight group scale (per output row) and the activation group scale (per
// activation row) apply.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "tf_riscv.h"

#include <math.h>
#include <string.h>

#include "pool.h"

#if !defined(__riscv) || !defined(__riscv_vector)

// No vector unit in this build: the IME2 entry points fall back to the
// portable scalar product so the dispatch in q4x.c stays uniform.
void tf_riscv_gemv_q4x_ime2(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0,
                            int j1) {
    tf_riscv_gemv_q4x_shard_scalar(xa, xs, w, out, n, k, j0, j1);
}
void tf_riscv_gemm_q4x_ime2(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0,
                            int j1) {
    tf_riscv_gemm_q4x_shard_scalar(xa, xs, w, out, m, n, k, j0, j1);
}

#else

#include <riscv_vector.h>

#define Q4X_GROUP 256
#define Q4X_ROWS 8
#define Q4X_TILE 128

// custom-2 opcode 0x2b, funct7 0x71, signed x signed mode bits [12:13],
// destination pair vd/2 at [8:11], vs1 at [15:19], vs2 at [20:24].
#define TF_VMADOT(vd, vs1, vs2) \
    (0xE200302Bu | (((unsigned)(vd) / 2u) << 8) | ((unsigned)(vs1) << 15) | ((unsigned)(vs2) << 20))

// Byte address of (activation row i, 16-k chunk kc) inside the packed A.
static inline const int8_t *a_tile(const int8_t *xa, int i, int k, int kc) {
    return xa + ((size_t)(i / Q4X_ROWS) * (k / 16) + kc) * Q4X_TILE + (i % Q4X_ROWS) * 16;
}

static inline float half_to_f32(uint16_t h) {
    _Float16 f;
    memcpy(&f, &h, 2);
    return (float)f;
}

// Fold one group's int32 accumulator lanes (0..7 = the eight output rows) into
// the f32 accumulator of the block being computed.
static inline vfloat32m1_t fold_group(vfloat32m1_t facc, const int32_t *iacc, const float *dw, float da,
                                      size_t v8) {
    vfloat32m1_t iv = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(iacc, v8), v8);
    vfloat32m1_t wf = __riscv_vle32_v_f32m1(dw, v8);
    iv = __riscv_vfmul_vv_f32m1(iv, wf, v8);
    return __riscv_vfmacc_vf_f32m1(facc, da, iv, v8);
}

// ---- decode: one activation row, eight output rows per pass ----------------

void tf_riscv_gemv_q4x_ime2(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0,
                            int j1) {
    const int groups = k / Q4X_GROUP;
    const size_t stride = tf_riscv_q4x_block_bytes(k);
    const size_t v8 = __riscv_vsetvl_e32m1(Q4X_ROWS);
    // Stage the activation row in the tile's TCM (one 8 KiB copy per pass,
    // then every weight tile reads its A operands from local SRAM; measured
    // 12.7 -> 15.4 GB/s on the lm_head shape).
    const size_t a_bytes = (size_t)Q4X_ROWS * k;
    int8_t *atcm = (int8_t *)tf_riscv_scratch(a_bytes);
    if (atcm) memcpy(atcm, xa, a_bytes);
    const int8_t *xa_d = atcm ? atcm : xa;
    int32_t iacc_stack[64] __attribute__((aligned(64)));
    int32_t *iacc = iacc_stack;
    for (int jb = j0; jb < j1; jb += Q4X_ROWS) {
        int cols = j1 - jb;
        if (cols > Q4X_ROWS) cols = Q4X_ROWS;
        const uint8_t *block = (const uint8_t *)w + (size_t)(jb / Q4X_ROWS) * stride;
        const uint8_t *tiles = block + (size_t)groups * Q4X_ROWS * 2;
        vfloat32m1_t facc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        for (int g = 0; g < groups; g++) {
            __asm__ volatile("vsetvli t0, zero, e32, m2 \n\t vmv.v.i v4, 0 \n\t vmv.v.i v8, 0" ::: "t0", "v4", "v5", "v8", "v9", "memory");
            // Two accumulator pairs: even tiles accumulate into v4/v5, odd
            // tiles into v8/v9, so consecutive vmadots do not chain on one
            // dependency (the decode loop is IPC-bound, not bandwidth-bound).
            for (int c = 0; c < Q4X_GROUP / 32; c += 2) {
                const int kc0 = g * (Q4X_GROUP / 32) + c;
                __builtin_prefetch(tiles + (size_t)(kc0 + 8) * Q4X_TILE, 0, 1);
                const uint8_t *t0p = tiles + (size_t)kc0 * Q4X_TILE;
                const uint8_t *t1p = t0p + Q4X_TILE;
                const int8_t *a00 = a_tile(xa_d, 0, k, 2 * kc0);
                const int8_t *a01 = a_tile(xa_d, 0, k, 2 * kc0 + 1);
                const int8_t *a10 = a_tile(xa_d, 0, k, 2 * kc0 + 2);
                const int8_t *a11 = a_tile(xa_d, 0, k, 2 * kc0 + 3);
                __asm__ volatile(
                    "vsetvli t0, zero, e8, m1  \n\t"
                    "vle8.v  v0, (%[t0])       \n\t"
                    "vle8.v  v10, (%[t1])      \n\t"
                    "vand.vi v1, v0, 0xF       \n\t"
                    "vand.vi v11, v10, 0xF     \n\t"
                    "vsrl.vi v6, v0, 4         \n\t"
                    "vsrl.vi v12, v10, 4       \n\t"
                    "vadd.vi v1, v1, -8        \n\t"
                    "vadd.vi v11, v11, -8      \n\t"
                    "vadd.vi v6, v6, -8        \n\t"
                    "vadd.vi v12, v12, -8      \n\t"
                    "vle8.v  v2, (%[a00])      \n\t"
                    ".4byte %[m0]              \n\t"
                    "vle8.v  v2, (%[a01])      \n\t"
                    ".4byte %[m1]              \n\t"
                    "vle8.v  v2, (%[a10])      \n\t"
                    ".4byte %[m2]              \n\t"
                    "vle8.v  v2, (%[a11])      \n\t"
                    ".4byte %[m3]              \n\t"
                    :
                    : [t0] "r"(t0p), [t1] "r"(t1p), [a00] "r"(a00), [a01] "r"(a01), [a10] "r"(a10), [a11] "r"(a11),
                      [m0] "i"(TF_VMADOT(4, 2, 1)), [m1] "i"(TF_VMADOT(4, 2, 6)),
                      [m2] "i"(TF_VMADOT(8, 2, 11)), [m3] "i"(TF_VMADOT(8, 2, 12))
                    : "t0", "v0", "v1", "v2", "v6", "v10", "v11", "v12", "v4", "v5", "v8", "v9", "memory");
            }
            __asm__ volatile(
                "vsetvli t0, zero, e32, m2 \n\t"
                "vadd.vv v4, v4, v8        \n\t"
                "vse32.v v4, (%[o])        \n\t"
                :
                : [o] "r"(iacc)
                : "t0", "v4", "v5", "memory");
            float dw[Q4X_ROWS];
            const uint16_t *sc = (const uint16_t *)(block + (size_t)g * Q4X_ROWS * 2);
            for (int r = 0; r < Q4X_ROWS; r++) dw[r] = half_to_f32(sc[r]);
            facc = fold_group(facc, iacc, dw, xs[g], v8);
        }
        __riscv_vse32_v_f32m1(out + jb, facc, __riscv_vsetvl_e32m1((size_t)cols));
    }
}

// ---- prefill: seven activation 8-row blocks per pass -----------------------
//
// Weight traffic is what bounds prefill here (each pass streams the shard's
// weights once), so the pass covers 56 activation rows and a weight tile is
// consumed by 14 vmadots: 8 output rows x 32 k x 56 rows per 128 bytes.

enum { GEMM_BLOCKS = 7 };

void tf_riscv_gemm_q4x_ime2(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0,
                            int j1) {
    const int groups = k / Q4X_GROUP;
    const size_t stride = tf_riscv_q4x_block_bytes(k);
    const size_t v8 = __riscv_vsetvl_e32m1(Q4X_ROWS);
    const size_t v64 = __riscv_vsetvl_e32m2(64);
    const long ostride = (long)n * 4;
    const int mpass = Q4X_ROWS * GEMM_BLOCKS;
    const int full = (m / mpass) * mpass;

    for (int jb = j0; jb < j1; jb += Q4X_ROWS) {
        const int cols = (j1 - jb) < Q4X_ROWS ? (j1 - jb) : Q4X_ROWS;
        const uint8_t *block = (const uint8_t *)w + (size_t)(jb / Q4X_ROWS) * stride;
        const uint8_t *tiles = block + (size_t)groups * Q4X_ROWS * 2;
        for (int m0 = 0; m0 < full; m0 += mpass) {
            // A pass's activation block (56 rows x k int8) is read once per
            // weight tile but only once per pass in total, so staging it in
            // TCM costs more than it saves here (measured: 45 -> 41 GMAC/s);
            // the decode path, which re-reads the row for every tile, does
            // stage it (12.7 -> 15.5 GB/s).
            const int8_t *xa_m = xa + (size_t)m0 * k;
            float facc[GEMM_BLOCKS][64] __attribute__((aligned(64)));
            memset(facc, 0, sizeof(facc));
            for (int g = 0; g < groups; g++) {
                int32_t iacc[GEMM_BLOCKS][64] __attribute__((aligned(64)));
                __asm__ volatile(
                    "vsetvli t0, zero, e32, m2 \n\t"
                    "vmv.v.i v4, 0             \n\t"
                    "vmv.v.i v8, 0             \n\t"
                    "vmv.v.i v12, 0            \n\t"
                    "vmv.v.i v16, 0            \n\t"
                    "vmv.v.i v20, 0            \n\t"
                    "vmv.v.i v24, 0            \n\t"
                    "vmv.v.i v28, 0            \n\t"
                    :
                    :
                    : "t0", "v4", "v5", "v8", "v9", "v12", "v13", "v16", "v17", "v20", "v21", "v24", "v25", "v28",
                      "v29", "memory");
                for (int c = 0; c < Q4X_GROUP / 32; c++) {
                    const int kc = g * (Q4X_GROUP / 32) + c;
                    if (c + 2 < Q4X_GROUP / 32) __builtin_prefetch(tiles + (size_t)(kc + 2) * Q4X_TILE, 0, 1);
                    const uint8_t *tile = tiles + (size_t)kc * Q4X_TILE;
                    const int8_t *a0 = a_tile(xa_m, 0, k, 2 * kc);
                    const int8_t *a1 = a_tile(xa_m, 0, k, 2 * kc + 1);
                    __asm__ volatile(
                        "vsetvli t0, zero, e8, m1  \n\t"
                        "vle8.v  v0, (%[t])        \n\t"
                        "vand.vi v1, v0, 0xF       \n\t"
                        "vsrl.vi v6, v0, 4         \n\t"
                        "vadd.vi v1, v1, -8        \n\t"
                        "vadd.vi v6, v6, -8        \n\t"
                        // block 0
                        "vle8.v  v2, (%[a0])       \n\t"
                        ".4byte %[b0l]             \n\t"
                        "vle8.v  v2, (%[a1])       \n\t"
                        ".4byte %[b0h]             \n\t"
                        // block 1
                        "vle8.v  v2, (%[a0b1])     \n\t"
                        ".4byte %[b1l]             \n\t"
                        "vle8.v  v2, (%[a1b1])     \n\t"
                        ".4byte %[b1h]             \n\t"
                        // block 2
                        "vle8.v  v2, (%[a0b2])     \n\t"
                        ".4byte %[b2l]             \n\t"
                        "vle8.v  v2, (%[a1b2])     \n\t"
                        ".4byte %[b2h]             \n\t"
                        // block 3
                        "vle8.v  v2, (%[a0b3])     \n\t"
                        ".4byte %[b3l]             \n\t"
                        "vle8.v  v2, (%[a1b3])     \n\t"
                        ".4byte %[b3h]             \n\t"
                        // block 4
                        "vle8.v  v2, (%[a0b4])     \n\t"
                        ".4byte %[b4l]             \n\t"
                        "vle8.v  v2, (%[a1b4])     \n\t"
                        ".4byte %[b4h]             \n\t"
                        // block 5
                        "vle8.v  v2, (%[a0b5])     \n\t"
                        ".4byte %[b5l]             \n\t"
                        "vle8.v  v2, (%[a1b5])     \n\t"
                        ".4byte %[b5h]             \n\t"
                        // block 6
                        "vle8.v  v2, (%[a0b6])     \n\t"
                        ".4byte %[b6l]             \n\t"
                        "vle8.v  v2, (%[a1b6])     \n\t"
                        ".4byte %[b6h]             \n\t"
                        :
                        : [t] "r"(tile), [a0] "r"(a0), [a1] "r"(a1),
                          [a0b1] "r"(a0 + (size_t)(k / 16) * Q4X_TILE), [a1b1] "r"(a1 + (size_t)(k / 16) * Q4X_TILE),
                          [a0b2] "r"(a0 + 2 * (size_t)(k / 16) * Q4X_TILE), [a1b2] "r"(a1 + 2 * (size_t)(k / 16) * Q4X_TILE),
                          [a0b3] "r"(a0 + 3 * (size_t)(k / 16) * Q4X_TILE), [a1b3] "r"(a1 + 3 * (size_t)(k / 16) * Q4X_TILE),
                          [a0b4] "r"(a0 + 4 * (size_t)(k / 16) * Q4X_TILE), [a1b4] "r"(a1 + 4 * (size_t)(k / 16) * Q4X_TILE),
                          [a0b5] "r"(a0 + 5 * (size_t)(k / 16) * Q4X_TILE), [a1b5] "r"(a1 + 5 * (size_t)(k / 16) * Q4X_TILE),
                          [a0b6] "r"(a0 + 6 * (size_t)(k / 16) * Q4X_TILE), [a1b6] "r"(a1 + 6 * (size_t)(k / 16) * Q4X_TILE),
                          [b0l] "i"(TF_VMADOT(4, 2, 1)), [b0h] "i"(TF_VMADOT(4, 2, 6)),
                          [b1l] "i"(TF_VMADOT(8, 2, 1)), [b1h] "i"(TF_VMADOT(8, 2, 6)),
                          [b2l] "i"(TF_VMADOT(12, 2, 1)), [b2h] "i"(TF_VMADOT(12, 2, 6)),
                          [b3l] "i"(TF_VMADOT(16, 2, 1)), [b3h] "i"(TF_VMADOT(16, 2, 6)),
                          [b4l] "i"(TF_VMADOT(20, 2, 1)), [b4h] "i"(TF_VMADOT(20, 2, 6)),
                          [b5l] "i"(TF_VMADOT(24, 2, 1)), [b5h] "i"(TF_VMADOT(24, 2, 6)),
                          [b6l] "i"(TF_VMADOT(28, 2, 1)), [b6h] "i"(TF_VMADOT(28, 2, 6))
                        : "t0", "v0", "v1", "v2", "v6", "v4", "v5", "v8", "v9", "v12", "v13", "v16", "v17", "v20",
                          "v21", "v24", "v25", "v28", "v29", "memory");
                }
                __asm__ volatile(
                    "vsetvli t0, zero, e32, m2  \n\t"
                    "vse32.v v4,  (%[o0])       \n\t"
                    "vse32.v v8,  (%[o1])       \n\t"
                    "vse32.v v12, (%[o2])       \n\t"
                    "vse32.v v16, (%[o3])       \n\t"
                    "vse32.v v20, (%[o4])       \n\t"
                    "vse32.v v24, (%[o5])       \n\t"
                    "vse32.v v28, (%[o6])       \n\t"
                    :
                    : [o0] "r"(iacc[0]), [o1] "r"(iacc[1]), [o2] "r"(iacc[2]), [o3] "r"(iacc[3]), [o4] "r"(iacc[4]),
                      [o5] "r"(iacc[5]), [o6] "r"(iacc[6])
                    : "t0", "memory");
                // Fold this group: lane (m*8 + n) carries output row n of
                // activation row m0 + 8*b + m, scaled by d_w[n][g] * d_a[m][g].
                float dw[Q4X_ROWS];
                const uint16_t *sc = (const uint16_t *)(block + (size_t)g * Q4X_ROWS * 2);
                for (int r = 0; r < Q4X_ROWS; r++) dw[r] = half_to_f32(sc[r]);
                // The 64-lane weight-scale vector repeats dw every 8 lanes;
                // build it once per group instead of 64 scalar stores per
                // block (the previous inner loop was most of the epilogue).
                float dw64[64] __attribute__((aligned(64)));
                for (int r = 0; r < 64; r++) dw64[r] = dw[r & 7];
                vfloat32m2_t wv = __riscv_vle32_v_f32m2(dw64, v64);
                for (int b = 0; b < GEMM_BLOCKS; b++) {
                    vfloat32m2_t fv = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(iacc[b], v64), v64);
                    vfloat32m2_t fa = __riscv_vle32_v_f32m2(facc[b], v64);
                    fv = __riscv_vfmul_vv_f32m2(fv, wv, v64);
                    fa = __riscv_vfmacc_vf_f32m2(fa, xs[(size_t)(m0 / Q4X_ROWS + b) * groups + g], fv, v64);
                    __riscv_vse32_v_f32m2(facc[b], fa, v64);
                }
            }
            // Store: for output column n, lanes {n, n+8, ...} are the eight
            // activation rows of that column, written to out[row][jb + n].
            for (int b = 0; b < GEMM_BLOCKS; b++) {
                const int row0 = m0 + b * Q4X_ROWS;
                for (int nc = 0; nc < cols; nc++) {
                    vint32m1_t v = __riscv_vlse32_v_i32m1((const int32_t *)(facc[b] + nc), 32, v8);
                    __riscv_vsse32_v_i32m1((int32_t *)(out + (size_t)row0 * n + (jb + nc)), ostride, v, v8);
                }
            }
        }
    }
    // Rows past the last full pass go through the portable float kernel: the
    // tail starts on an 8-row boundary, so the packed activation and scale
    // arrays shift by whole rows.
    if (full < m) {
        tf_riscv_gemm_q4x_float(xa + (size_t)full * k, xs + (size_t)full * groups, w, out + (size_t)full * n, m - full, n,
                                k, j0, j1);
    }
}

#endif // __riscv && __riscv_vector
