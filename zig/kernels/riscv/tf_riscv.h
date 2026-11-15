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

// One output-column shard [j0,j1) of the two products above: the multithreaded
// path splits rows of w across threads, each streaming its own sequential
// weight range (the K3 paper's thread-sharded sequential read pattern).
// m > 1 re-reads each weight row only every 64 output rows (prefill traffic).
void tf_riscv_matmul_bf16_shard(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_f32_shard(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1);

// One query row's attention over `rows` cached keys/values: softmax(scale*q.k)
// then the value mix, per head of head_dim elements. `scratch` holds `rows`
// floats (the score buffer): one dot pass per key row, not three.
void tf_riscv_attention_f32(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch);

// Quantized GGUF block GEMV: out[n] = x[k] . w[n,k]^T, dequantizing blocks.
// q8_0 blocks: 32 xi8 + f16 scale; q4_0 blocks: 16 bytes (32 u4) + f16 scale.
void tf_riscv_gemv_q8_0(const float *x, const void *w, float *out, int n, int k);
void tf_riscv_gemv_q4_0(const float *x, const void *w, float *out, int n, int k);

// Quantized projections for m rows over output columns [j0,j1) (the
// multithreaded and prefill paths); blocks follow the two GEMV formats.
void tf_riscv_matmul_q8_0_shard(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_q4_0_shard(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);

// ---- q4x: 4-bit weights in IME2 dot tiles --------------------------------
//
// Weights are packed in 8-output-row blocks for SpacemiT's IME2 dot (vmadot,
// 8x8x16 int8 -> int32). Per 32-k chunk a tile is 128 bytes: byte t holds
// w[r0 + t/16][k0 + t%16] in its low nibble and w[r0 + t/16][k0 + 16 + t%16]
// in its high nibble, so one load gives both vmadot operands. Per block the
// scales come first: (k/256) groups x 8 rows x f16, value w = q_signed * d.
// Block stride is 4k + k/16 bytes (4.0625 bits per weight at k=1024).
//
// Activations are quantized to int8 with one f32 scale per 256-value group and
// tiled as vmadot's A operand: per 8-row block and 16-k chunk, 128 bytes.
size_t tf_riscv_q4x_block_bytes(int k);
size_t tf_riscv_q4x_row_bytes(int k);
int tf_riscv_q4x_ok(int k);
int tf_riscv_q4x_perm_k(int u);

// int8 activation tiles + per-row group scales (tutorial: qwen3.zig quantizeQ4x).
void tf_riscv_quant_a(const float *x, int8_t *xa, float *xs, int m, int k);
void tf_riscv_quant_a_scalar(const float *x, int8_t *xa, float *xs, int m, int k);

// Decode (m == 1) and prefill (m rows): out[m][n] = x[m][k] @ w[n][k]^T.
void tf_riscv_gemv_q4x(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k);
void tf_riscv_gemv_q4x_shard(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1);
void tf_riscv_gemm_q4x_shard(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1);

// The two kernel families behind the dispatch, plus the scalar oracle.
void tf_riscv_gemv_q4x_ime2(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1);
void tf_riscv_gemm_q4x_ime2(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_gemv_q4x_float(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1);
void tf_riscv_gemm_q4x_float(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_gemv_q4x_shard_scalar(const int8_t *xa, const float *xs, const void *w, float *out, int n, int k, int j0, int j1);
void tf_riscv_gemm_q4x_shard_scalar(const int8_t *xa, const float *xs, const void *w, float *out, int m, int n, int k, int j0, int j1);

// ---- worker scratch and K3 local-memory placement -------------------------
//
// tf_riscv_scratch returns per-worker scratch that is stable until the next
// call: inside an A100 tile it is carved from TCM (the K3 AI cores' 393 KiB
// local SRAM), otherwise it is a per-thread malloc buffer.
void *tf_riscv_scratch(size_t bytes);
size_t tf_riscv_tcm_bytes(void);   // TCM arena this process obtained (0 = none)
size_t tf_riscv_tcm_used(void);    // peak bytes handed out by tf_riscv_scratch

// Weight arena: anonymous pages with MADV_HUGEPAGE (the vendor's HPAGE
// backend), 2 MiB aligned. tf_riscv_hpage_available reports whether THP is
// enabled on this kernel.
void *tf_riscv_hpage_alloc(size_t bytes);
void tf_riscv_hpage_free(void *p, size_t bytes);
int tf_riscv_hpage_available(void);
size_t tf_riscv_hpage_backed(const void *p);
size_t tf_riscv_hpage_bytes(void);

// ---- A100 dispatch --------------------------------------------------------
//
// tf_riscv_pool_mode("auto"|"x100"|"a100") picks where the sharded kernels
// run: the X100 cores through a pthread pool, or the A100 AI cores through
// SpacemiT's spine runtime (libspert), eight tiles per call. "auto" is
// TF_RISCV_A100=1/0 or x100 by default.
int tf_riscv_pool_mode(const char *mode);
const char *tf_riscv_pool_mode_name(void);
int tf_riscv_a100_available(void);

#ifdef __cplusplus
}
#endif

#endif // TF_RISCV_H
