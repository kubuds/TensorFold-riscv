//! tf_riscv.h - TensorFold's RISC-V CPU backend kernel ABI.
//
// Row-major f32 compute; bf16 weights travel as uint16 lanes (IEEE 754 binary16
// layout stored in the high half of an f32). The ABI is plain C11 so the Zig
// runtime, the tests and any future engine adapter share one surface.
//
// Kernel families (selected at runtime, see tf_riscv_select_kernels):
//   scalar  - portable C for RV64GC, the correctness baseline
//   rvv     - RVV 1.0 intrinsics (VLEN-agnostic: 256-bit X100, 1024-bit A100)
//   ime2    - SpacemiT-IME 2.0 MMA paths (A100), RVV fallback for the rest
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#ifndef TF_RISCV_H
#define TF_RISCV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Capability bits reported by tf_riscv_caps.
enum {
    TF_RISCV_CAP_RVV = 1u << 0,  // RVV 1.0 usable (HWCAP_V)
    TF_RISCV_CAP_IME1 = 1u << 1, // SpacemiT-IME 1.0 present (probe)
    TF_RISCV_CAP_IME2 = 1u << 2, // SpacemiT-IME 2.0 present (probe)
};

// Process-wide capabilities and the selected kernel family.
uint32_t tf_riscv_caps(void);
uint32_t tf_riscv_vlen_bits(void); // architectural VLEN in bits; 0 without RVV
const char *tf_riscv_active_kernels(void); // "scalar", "rvv" or "ime2"

// Picks a kernel family: "auto", "scalar", "rvv" or "ime2".
// Returns 0 on success, -1 when the name is unknown or not usable here.
int tf_riscv_select_kernels(const char *name);

// Pin this process to a CPU list ("0-7", "8-15", "0,2,4"): K3 schedules AI
// work on the A100 set (8-15) and general work on the X100 set (0-7).
// Returns 0 on success, -1 on a bad spec or when the platform refuses.
int tf_riscv_bind_cpus(const char *spec);

// out = x * rsqrt(mean(x*x) + eps) * w, n elements (w may be NULL for plain norm).
void tf_riscv_rmsnorm(const float *x, const float *w, float *out, int n, float eps);

// Rotary embeddings (HF Qwen3 "rotate_half", full head_dim) applied in place at
// position `pos`: q holds q_heads*head_dim, k holds kv_heads*head_dim.
void tf_riscv_rope(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta);

// In-place row softmax over n elements.
void tf_riscv_softmax(float *x, int n);

// out = silu(gate) * up over n elements.
void tf_riscv_silu_mul(const float *gate, const float *up, float *out, int n);

// out = a + b over n elements.
void tf_riscv_add(const float *a, const float *b, float *out, int n);

// out[m,n] = a[m,k] @ b[k,n] (trans_b == 0), or a[m,k] @ b[n,k]^T (trans_b != 0).
void tf_riscv_matmul_f32(const float *a, const float *b, float *out, int m, int n, int k, int trans_b);

// out[m,n] = x[m,k] @ w[n,k]^T with bf16 weights (Hugging Face linear layout).
// The decode hot path: GEMV when m == 1.
void tf_riscv_matmul_bf16(const float *x, const uint16_t *w, float *out, int m, int n, int k);

// One query row's attention over `rows` cached keys/values: softmax(scale*q.k)
// then the value mix, per head of head_dim elements.
void tf_riscv_attention_f32(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale);

// Quantized GGUF block GEMV: out[n] = x[k] . w[n,k]^T, dequantizing blocks.
// q8_0 blocks: 32 xi8 + f16 scale; q4_0 blocks: 16 bytes (32 u4) + f16 scale.
void tf_riscv_gemv_q8_0(const float *x, const void *w, float *out, int n, int k);
void tf_riscv_gemv_q4_0(const float *x, const void *w, float *out, int n, int k);

#ifdef __cplusplus
}
#endif

#endif // TF_RISCV_H
