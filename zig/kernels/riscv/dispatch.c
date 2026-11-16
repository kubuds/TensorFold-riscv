//! dispatch.c - runtime kernel-family selection for TensorFold's RISC-V backend.
//
// One process-wide table of kernel entry points, chosen once:
//   TF_RISCV_KERNELS=auto|scalar|rvv|ime2   (default auto)
// "auto" takes the richest family the CPU reports: ime2 > rvv > scalar.
// Compile-time: this file carries the same -mcpu as ops_rvv.c so the RVV family
// is visible exactly when the build enabled the V extension; a pure RV64GC
// build links the scalar family only.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE // cpu_set_t and CPU_SET for sched_setaffinity: first line
#endif
#include "tf_riscv.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    const char *name;
    void (*rmsnorm)(const float *, const float *, float *, int, float);
    void (*rope)(float *, float *, int, int, int, int, float);
    void (*rope_table)(int, float *, float *, int, int, float);
    void (*rope_apply)(float *, float *, const float *, const float *, int, int, int);
    void (*softmax)(float *, int);
    void (*silu_mul)(const float *, const float *, float *, int);
    void (*add)(const float *, const float *, float *, int);
    void (*matmul_f32)(const float *, const float *, float *, int, int, int, int);
    void (*matmul_bf16)(const float *, const uint16_t *, float *, int, int, int);
    void (*matmul_bf16_shard)(const float *, const uint16_t *, float *, int, int, int, int, int);
    void (*matmul_f32_shard)(const float *, const float *, float *, int, int, int, int, int, int);
    void (*attention_f32)(const float *, const float *, const float *, float *, int, int, float, float *);
    void (*attention_block)(const float *, int, const float *, const float *, float *, int, int, int, int, float, float *);
    void (*gemv_q8_0)(const float *, const void *, float *, int, int);
    void (*gemv_q4_0)(const float *, const void *, float *, int, int);
    void (*matmul_q8_0_shard)(const float *, const void *, float *, int, int, int, int, int);
    void (*matmul_q4_0_shard)(const float *, const void *, float *, int, int, int, int, int);
} tf_riscv_ops;

void tf_riscv_rmsnorm_scalar(const float *x, const float *w, float *out, int n, float eps);
void tf_riscv_rope_scalar(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta);
void tf_riscv_rope_table_scalar(int pos, float *s, float *c, int heads, int head_dim, float theta);
void tf_riscv_rope_apply_scalar(float *q, float *k, const float *s, const float *c, int q_heads, int kv_heads, int head_dim);
void tf_riscv_softmax_scalar(float *x, int n);
void tf_riscv_silu_mul_scalar(const float *gate, const float *up, float *out, int n);
void tf_riscv_add_scalar(const float *a, const float *b, float *out, int n);
void tf_riscv_matmul_f32_scalar(const float *a, const float *b, float *out, int m, int n, int k, int trans_b);
void tf_riscv_matmul_bf16_scalar(const float *x, const uint16_t *w, float *out, int m, int n, int k);
void tf_riscv_matmul_bf16_shard_scalar(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_f32_shard_scalar(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1);
void tf_riscv_attention_f32_scalar(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch);

// One shared block-attention implementation for every family (attention_block.c):
// identical numbers whichever family runs it.
void tf_riscv_attention_block_f32(const float *q, int q_stride, const float *k, const float *v, float *out, int out_stride,
                                  int nb, int rows, int head_dim, float scale, float *scratch);
void tf_riscv_gemv_q8_0_scalar(const float *x, const void *w, float *out, int n, int k);
void tf_riscv_matmul_q8_0_shard_scalar(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_q4_0_shard_scalar(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_gemv_q4_0_scalar(const float *x, const void *w, float *out, int n, int k);

static const tf_riscv_ops scalar_ops = {
    "scalar",
    tf_riscv_rmsnorm_scalar,
    tf_riscv_rope_scalar,
    tf_riscv_rope_table_scalar,
    tf_riscv_rope_apply_scalar,
    tf_riscv_softmax_scalar,
    tf_riscv_silu_mul_scalar,
    tf_riscv_add_scalar,
    tf_riscv_matmul_f32_scalar,
    tf_riscv_matmul_bf16_scalar,
    tf_riscv_matmul_bf16_shard_scalar,
    tf_riscv_matmul_f32_shard_scalar,
    tf_riscv_attention_f32_scalar,
    tf_riscv_attention_block_f32,
    tf_riscv_gemv_q8_0_scalar,
    tf_riscv_gemv_q4_0_scalar,
    tf_riscv_matmul_q8_0_shard_scalar,
    tf_riscv_matmul_q4_0_shard_scalar,
};

#if defined(__riscv) && defined(__riscv_vector)
void tf_riscv_rmsnorm_rvv(const float *x, const float *w, float *out, int n, float eps);
void tf_riscv_rope_rvv(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta);
void tf_riscv_rope_table_rvv(int pos, float *s, float *c, int heads, int head_dim, float theta);
void tf_riscv_rope_apply_rvv(float *q, float *k, const float *s, const float *c, int q_heads, int kv_heads, int head_dim);
void tf_riscv_softmax_rvv(float *x, int n);
void tf_riscv_silu_mul_rvv(const float *gate, const float *up, float *out, int n);
void tf_riscv_add_rvv(const float *a, const float *b, float *out, int n);
void tf_riscv_matmul_f32_rvv(const float *a, const float *b, float *out, int m, int n, int k, int trans_b);
void tf_riscv_matmul_bf16_rvv(const float *x, const uint16_t *w, float *out, int m, int n, int k);
void tf_riscv_matmul_bf16_shard_rvv(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_f32_shard_rvv(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1);
void tf_riscv_attention_f32_rvv(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch);
void tf_riscv_gemv_q8_0_rvv(const float *x, const void *w, float *out, int n, int k);
void tf_riscv_matmul_q8_0_shard_rvv(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_q4_0_shard_rvv(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_gemv_q4_0_rvv(const float *x, const void *w, float *out, int n, int k);

static const tf_riscv_ops rvv_ops = {
    "rvv",
    tf_riscv_rmsnorm_rvv,
    tf_riscv_rope_rvv,
    tf_riscv_rope_table_rvv,
    tf_riscv_rope_apply_rvv,
    tf_riscv_softmax_rvv,
    tf_riscv_silu_mul_rvv,
    tf_riscv_add_rvv,
    tf_riscv_matmul_f32_rvv,
    tf_riscv_matmul_bf16_rvv,
    tf_riscv_matmul_bf16_shard_rvv,
    tf_riscv_matmul_f32_shard_rvv,
    tf_riscv_attention_f32_rvv,
    tf_riscv_attention_block_f32,
    tf_riscv_gemv_q8_0_rvv,
    tf_riscv_gemv_q4_0_rvv,
    tf_riscv_matmul_q8_0_shard_rvv,
    tf_riscv_matmul_q4_0_shard_rvv,
};
#endif

// The IME2 family (A100's tensor core): same entry names in ime2.c, which is
// only linked when the build asks for it (-DTF_RISCV_WITH_IME2).
#if defined(TF_RISCV_WITH_IME2)
int tf_riscv_ime2_available(void); // ime2.c: vendor runtime or emulated MMA present
void tf_riscv_rmsnorm_ime2(const float *x, const float *w, float *out, int n, float eps);
void tf_riscv_rope_ime2(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta);
void tf_riscv_rope_table_ime2(int pos, float *s, float *c, int heads, int head_dim, float theta);
void tf_riscv_rope_apply_ime2(float *q, float *k, const float *s, const float *c, int q_heads, int kv_heads, int head_dim);
void tf_riscv_softmax_ime2(float *x, int n);
void tf_riscv_silu_mul_ime2(const float *gate, const float *up, float *out, int n);
void tf_riscv_add_ime2(const float *a, const float *b, float *out, int n);
void tf_riscv_matmul_f32_ime2(const float *a, const float *b, float *out, int m, int n, int k, int trans_b);
void tf_riscv_matmul_bf16_ime2(const float *x, const uint16_t *w, float *out, int m, int n, int k);
void tf_riscv_matmul_bf16_shard_ime2(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_f32_shard_ime2(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1);
void tf_riscv_attention_f32_ime2(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch);
void tf_riscv_gemv_q8_0_ime2(const float *x, const void *w, float *out, int n, int k);
void tf_riscv_matmul_q8_0_shard_ime2(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_matmul_q4_0_shard_ime2(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1);
void tf_riscv_gemv_q4_0_ime2(const float *x, const void *w, float *out, int n, int k);

static const tf_riscv_ops ime2_ops = {
    "ime2",
    tf_riscv_rmsnorm_ime2,
    tf_riscv_rope_ime2,
    tf_riscv_rope_table_ime2,
    tf_riscv_rope_apply_ime2,
    tf_riscv_softmax_ime2,
    tf_riscv_silu_mul_ime2,
    tf_riscv_add_ime2,
    tf_riscv_matmul_f32_ime2,
    tf_riscv_matmul_bf16_ime2,
    tf_riscv_matmul_bf16_shard_ime2,
    tf_riscv_matmul_f32_shard_ime2,
    tf_riscv_attention_f32_ime2,
    tf_riscv_attention_block_f32,
    tf_riscv_gemv_q8_0_ime2,
    tf_riscv_gemv_q4_0_ime2,
    tf_riscv_matmul_q8_0_shard_ime2,
    tf_riscv_matmul_q4_0_shard_ime2,
};
#endif

static const tf_riscv_ops *ops = &scalar_ops;

#if defined(__riscv)
#include <sys/auxv.h>
#ifndef COMPAT_HWCAP_ISA_V
#define COMPAT_HWCAP_ISA_V (1 << ('V' - 'A'))
#endif

uint32_t tf_riscv_caps(void) {
    uint32_t caps = 0;
    if (getauxval(AT_HWCAP) & COMPAT_HWCAP_ISA_V) caps |= TF_RISCV_CAP_RVV;
#if defined(TF_RISCV_WITH_IME2)
    if (tf_riscv_ime2_available()) caps |= TF_RISCV_CAP_IME2 | TF_RISCV_CAP_IME1;
#endif
    return caps;
}

uint32_t tf_riscv_vlen_bits(void) {
#if defined(__riscv_vector)
    unsigned long vlenb = 0;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    return (uint32_t)(vlenb * 8);
#else
    return 0;
#endif
}
#else
uint32_t tf_riscv_caps(void) { return 0; }
uint32_t tf_riscv_vlen_bits(void) { return 0; }
#endif

const char *tf_riscv_active_kernels(void) { return ops->name; }

// "0-7,9" -> cpuset; Linux-only, the K3 deployment target.
#if defined(__linux__)
#include <sched.h>
int tf_riscv_bind_cpus(const char *spec) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (!spec || !*spec) return -1;
    const char *p = spec;
    while (*p) {
        char *end;
        long a = strtol(p, &end, 10);
        if (end == p || a < 0 || a >= CPU_SETSIZE) return -1;
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            if (end == p || b < a || b >= CPU_SETSIZE) return -1;
            p = end;
        }
        for (long c = a; c <= b; c++) CPU_SET((int)c, &set);
        if (*p == ',') p++;
        else if (*p) return -1;
    }
    return sched_setaffinity(0, sizeof(set), &set) == 0 ? 0 : -1;
}
#else
int tf_riscv_bind_cpus(const char *spec) {
    (void)spec;
    return -1;
}
#endif

static const tf_riscv_ops *find(const char *name) {
    if (strcmp(name, "scalar") == 0) return &scalar_ops;
#if defined(__riscv) && defined(__riscv_vector)
    if (strcmp(name, "rvv") == 0) return (tf_riscv_caps() & TF_RISCV_CAP_RVV) ? &rvv_ops : NULL;
#endif
#if defined(TF_RISCV_WITH_IME2)
    if (strcmp(name, "ime2") == 0) return (tf_riscv_caps() & TF_RISCV_CAP_IME2) ? &ime2_ops : NULL;
#endif
    return NULL;
}

int tf_riscv_select_kernels(const char *name) {
    if (name == NULL || strcmp(name, "auto") == 0) {
        const tf_riscv_ops *pick = &scalar_ops;
#if defined(__riscv) && defined(__riscv_vector)
        if (tf_riscv_caps() & TF_RISCV_CAP_RVV) pick = &rvv_ops;
#endif
#if defined(TF_RISCV_WITH_IME2)
        if (tf_riscv_caps() & TF_RISCV_CAP_IME2) pick = &ime2_ops;
#endif
        ops = pick;
        return 0;
    }
    const tf_riscv_ops *pick = find(name);
    if (!pick) return -1;
    ops = pick;
    return 0;
}

__attribute__((constructor)) static void init(void) {
    const char *env = getenv("TF_RISCV_KERNELS");
    tf_riscv_select_kernels(env ? env : "auto");
}

// Public entry points: forward to the selected family.
void tf_riscv_rmsnorm(const float *x, const float *w, float *out, int n, float eps) { ops->rmsnorm(x, w, out, n, eps); }
void tf_riscv_rope(float *q, float *k, int pos, int q_heads, int kv_heads, int head_dim, float theta) {
    ops->rope(q, k, pos, q_heads, kv_heads, head_dim, theta);
}
void tf_riscv_rope_table(int pos, float *s, float *c, int heads, int head_dim, float theta) {
    ops->rope_table(pos, s, c, heads, head_dim, theta);
}
void tf_riscv_rope_apply(float *q, float *k, const float *s, const float *c, int q_heads, int kv_heads, int head_dim) {
    ops->rope_apply(q, k, s, c, q_heads, kv_heads, head_dim);
}
void tf_riscv_softmax(float *x, int n) { ops->softmax(x, n); }
void tf_riscv_silu_mul(const float *gate, const float *up, float *out, int n) { ops->silu_mul(gate, up, out, n); }
void tf_riscv_add(const float *a, const float *b, float *out, int n) { ops->add(a, b, out, n); }
void tf_riscv_matmul_f32(const float *a, const float *b, float *out, int m, int n, int k, int trans_b) {
    ops->matmul_f32(a, b, out, m, n, k, trans_b);
}
void tf_riscv_matmul_bf16(const float *x, const uint16_t *w, float *out, int m, int n, int k) {
    ops->matmul_bf16(x, w, out, m, n, k);
}
void tf_riscv_matmul_bf16_shard(const float *x, const uint16_t *w, float *out, int m, int n, int k, int j0, int j1) {
    ops->matmul_bf16_shard(x, w, out, m, n, k, j0, j1);
}
void tf_riscv_matmul_f32_shard(const float *a, const float *b, float *out, int m, int n, int k, int trans_b, int j0, int j1) {
    ops->matmul_f32_shard(a, b, out, m, n, k, trans_b, j0, j1);
}
void tf_riscv_attention_f32(const float *q, const float *k, const float *v, float *out, int rows, int head_dim, float scale, float *scratch) {
    ops->attention_f32(q, k, v, out, rows, head_dim, scale, scratch);
}
void tf_riscv_gemv_q8_0(const float *x, const void *w, float *out, int n, int k) { ops->gemv_q8_0(x, w, out, n, k); }
void tf_riscv_gemv_q4_0(const float *x, const void *w, float *out, int n, int k) { ops->gemv_q4_0(x, w, out, n, k); }
void tf_riscv_matmul_q8_0_shard(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    ops->matmul_q8_0_shard(x, w, out, m, n, k, j0, j1);
}
void tf_riscv_matmul_q4_0_shard(const float *x, const void *w, float *out, int m, int n, int k, int j0, int j1) {
    ops->matmul_q4_0_shard(x, w, out, m, n, k, j0, j1);
}
