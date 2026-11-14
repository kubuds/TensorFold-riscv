//! ime2.c - the IME2 kernel family for TensorFold's RISC-V backend (K3 A100).
//
// SpacemiT's IME 2.0 is the dual-core-shared MMA of the A100 (VLEN=1024):
// bf16 (8x8)x(8x8)->fp32 and i8 (8x16)x(16x8)->i32 tiles on vector registers.
// This family tiles the backend's products through that MMA contract:
//
//   - matmul_bf16 with m >= 8 runs tiled MMA (weight tiles transposed into
//     B[k,n] layout per the D = A.B + C definition in the K3 paper);
//   - single-row GEMV (decode) and every non-matrix op stay on the RVV
//     family: decode is bandwidth-bound and the MMA cannot shorten the
//     weight stream (the paper's own decode analysis is read-bandwidth
//     limited). This split is deliberate and measured, not accidental.
//
// The MMA itself comes from three places, in this order:
//   1. a vendor runtime library (TF_RISCV_IME2=/path/lib.so) implementing
//      ime2_abi.h - the drop-in point for SpacemiT's SDK;
//   2. TF_RISCV_IME2_INLINE builds: the reference MMA compiled in, so the
//      family is exercised by the regression tests under qemu (VLEN=1024);
//   3. none: tf_riscv_ime2_available() returns 0 and dispatch never picks
//      this family.
// The reference MMA computes the same products in float64 accumulation;
// real hardware multiplies bf16 lanes, so on-hardware tolerances are
// re-established with the vendor library (docs/riscv-performance.md).
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "tf_riscv.h"
#include "ime2_abi.h"

#include <stdlib.h>
#include <string.h>

// The RVV family this file forwards to (ops_rvv.c, same -mcpu).
void tf_riscv_rmsnorm_rvv(const float *x, const float *w, float *out, int n, float eps);
void tf_riscv_rope_rvv(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta);
void tf_riscv_softmax_rvv(float *x, int n);
void tf_riscv_silu_mul_rvv(const float *gate, const float *up, float *out, int n);
void tf_riscv_add_rvv(const float *a, const float *b, float *out, int n);
void tf_riscv_matmul_f32_rvv(const float *a, const float *b, float *out, int m, int n, int k, int trans_b);
void tf_riscv_matmul_bf16_rvv(const float *x, const uint16_t *w, float *out, int m, int n, int k);
void tf_riscv_attention_f32_rvv(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale);
void tf_riscv_gemv_q8_0_rvv(const float *x, const void *w, float *out, int n, int k);
void tf_riscv_gemv_q4_0_rvv(const float *x, const void *w, float *out, int n, int k);

// --- the reference MMA (TF_RISCV_IME2_INLINE or the tools/riscv shim) ------

static inline float bf16_to_f32(uint16_t b) {
    union {
        uint32_t u;
        float f;
    } v = {.u = (uint32_t)b << 16};
    return v.f;
}

void tf_riscv_ime2_mma_bf16_ref(const uint16_t *a, const uint16_t *b, const float *c, float *d) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double s = (double)c[i * 8 + j];
            for (int t = 0; t < 8; t++) {
                s += (double)bf16_to_f32(a[i * 8 + t]) * (double)bf16_to_f32(b[t * 8 + j]);
            }
            d[i * 8 + j] = (float)s;
        }
    }
}

void tf_riscv_ime2_mma_i8_ref(const int8_t *a, const int8_t *b, const int32_t *c, int32_t *d) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int32_t s = c[i * 8 + j];
            for (int t = 0; t < 16; t++) s += (int32_t)a[i * 16 + t] * (int32_t)b[t * 8 + j];
            d[i * 8 + j] = s;
        }
    }
}

// --- the seam ---------------------------------------------------------------

static const tf_riscv_ime2_abi *vendor = NULL;
static int loaded = 0;
static tf_riscv_ime2_abi inline_abi = {NULL, NULL, NULL};

#if defined(TF_RISCV_IME2_INLINE)
static const tf_riscv_ime2_abi emulated = {NULL, tf_riscv_ime2_mma_bf16_ref, tf_riscv_ime2_mma_i8_ref};
static int emulated_ok(void) { return 1; }
#endif

#if defined(__riscv)
// The vendor library's entry: it hands back its vtable (ime2_abi.h).
typedef const tf_riscv_ime2_abi *(*ime2_entry_fn)(void);
#include <dlfcn.h>

static void load_vendor(void) {
    if (loaded) return;
    loaded = 1;
    const char *path = getenv("TF_RISCV_IME2");
    if (!path || strcmp(path, "off") == 0) {
#if defined(TF_RISCV_IME2_INLINE)
        inline_abi = emulated;
        inline_abi.available = emulated_ok;
        vendor = &inline_abi;
#endif
        return;
    }
    void *h = dlopen(path, RTLD_NOW);
    if (!h) return;
    ime2_entry_fn entry = (ime2_entry_fn)dlsym(h, "tf_riscv_ime2_abi_v1");
    if (!entry) return;
    vendor = entry();
}
#else
static void load_vendor(void) {
    if (loaded) return;
    loaded = 1;
#if defined(TF_RISCV_IME2_INLINE)
    inline_abi = emulated;
    inline_abi.available = emulated_ok;
    vendor = &inline_abi;
#endif
}
#endif

int tf_riscv_ime2_available(void) {
    load_vendor();
    if (!vendor || !vendor->mma_bf16 || !vendor->mma_i8) return 0;
    return vendor->available ? vendor->available() : 1;
}

// --- the family -------------------------------------------------------------

void tf_riscv_rmsnorm_ime2(const float *x, const float *w, float *out, int n, float eps) { tf_riscv_rmsnorm_rvv(x, w, out, n, eps); }
void tf_riscv_rope_ime2(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta) {
    tf_riscv_rope_rvv(q, k, pos, q_heads, kv_heads, head_dim, theta);
}
void tf_riscv_softmax_ime2(float *x, int n) { tf_riscv_softmax_rvv(x, n); }
void tf_riscv_silu_mul_ime2(const float *gate, const float *up, float *out, int n) { tf_riscv_silu_mul_rvv(gate, up, out, n); }
void tf_riscv_add_ime2(const float *a, const float *b, float *out, int n) { tf_riscv_add_rvv(a, b, out, n); }
void tf_riscv_matmul_f32_ime2(const float *a, const float *b, float *out, int m, int n, int k, int trans_b) {
    tf_riscv_matmul_f32_rvv(a, b, out, m, n, k, trans_b); // f32 x f32 is not an IME2 shape
}
void tf_riscv_attention_f32_ime2(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale) {
    tf_riscv_attention_f32_rvv(q, k, v, out, rows, head_dim, scale);
}
void tf_riscv_gemv_q8_0_ime2(const float *x, const void *w, float *out, int n, int k) { tf_riscv_gemv_q8_0_rvv(x, w, out, n, k); }
void tf_riscv_gemv_q4_0_ime2(const float *x, const void *w, float *out, int n, int k) { tf_riscv_gemv_q4_0_rvv(x, w, out, n, k); }

// f32 -> bf16 lanes (round to nearest even on the low bit).
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7fffu + lsb) >> 16);
}

// y[m,n] = x[m,k] @ w[n,k]^T through 8x8 bf16 MMA tiles; m < 8 or no vendor
// runtime: the RVV GEMV path (decode rows never benefit from the MMA).
void tf_riscv_matmul_bf16_ime2(const float *x, const uint16_t *w, float *out, int m, int n, int k) {
    load_vendor();
    if (m < 8 || !vendor || !vendor->mma_bf16 || (vendor->available && !vendor->available())) {
        tf_riscv_matmul_bf16_rvv(x, w, out, m, n, k);
        return;
    }
    for (int i0 = 0; i0 < m; i0 += 8) {
        int mi = m - i0 < 8 ? m - i0 : 8;
        for (int j0 = 0; j0 < n; j0 += 8) {
            int nj = n - j0 < 8 ? n - j0 : 8;
            float acc[64];
            memset(acc, 0, sizeof(acc));
            for (int t0 = 0; t0 < k; t0 += 8) {
                int kt = k - t0 < 8 ? k - t0 : 8;
                uint16_t at[64], bt[64]; // A[8,8] rows of x, B[8,8] = w^T tile
                memset(at, 0, sizeof(at));
                memset(bt, 0, sizeof(bt));
                for (int i = 0; i < mi; i++) {
                    for (int t = 0; t < kt; t++) at[i * 8 + t] = f32_to_bf16(x[(size_t)(i0 + i) * k + t0 + t]);
                }
                for (int j = 0; j < nj; j++) {
                    for (int t = 0; t < kt; t++) bt[t * 8 + j] = w[(size_t)(j0 + j) * k + t0 + t];
                }
                vendor->mma_bf16(at, bt, acc, acc);
            }
            for (int i = 0; i < mi; i++) {
                for (int j = 0; j < nj; j++) out[(size_t)(i0 + i) * n + j0 + j] = acc[i * 8 + j];
            }
        }
    }
}
