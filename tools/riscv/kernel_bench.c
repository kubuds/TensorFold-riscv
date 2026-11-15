//! kernel_bench.c - per-kernel throughput for the RISC-V backend: the micro
//  receipts behind docs/riscv-performance.md (GB/s for GEMV, GFLOP/s for GEMM).
//  Build: zig cc -target riscv64-linux-musl -mcpu=generic_riv64+a+m+c+d+f+v \
//           -O2 -I zig/kernels/riscv tools/riscv/kernel_bench.c \
//           zig/kernels/riscv/ops_scalar.c zig/kernels/riscv/ops_rvv.c \
//           zig/kernels/riscv/dispatch.c zig/kernels/riscv/pool.c -lm -o kernel_bench
//  Usage: kernel_bench [family] [threads]   (family: scalar|rvv|ime2|auto)
//  SPDX-License-Identifier: Apache-2.0
#include "tf_riscv.h"
#include "pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct {
    const float *x;
    const uint16_t *w;
    float *out;
    int n, k;
} GemvCtx;

static int gemm_rows = 1; // set before the m>1 case

static void gemv_shard(void *ctx, int begin, int end) {
    const GemvCtx *g = (const GemvCtx *)ctx;
    tf_riscv_matmul_bf16_shard(g->x, g->w, g->out, gemm_rows, g->n, g->k, begin, end);
}

int main(int argc, char **argv) {
    if (argc > 1 && tf_riscv_select_kernels(argv[1]) != 0) {
        printf("unknown family %s\n", argv[1]);
        return 2;
    }
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    if (threads < 1) threads = 1;
    threads = tf_riscv_pool_init(threads);
    printf("family=%s vlen=%u threads=%d\n", tf_riscv_active_kernels(), tf_riscv_vlen_bits(), threads);

    // Decode GEMV shapes of Qwen3-0.6B (m=1): the bandwidth-bound path.
    const int shapes[][2] = {{1024, 1024}, {2048, 1024}, {1024, 2048}, {3072, 1024}, {1024, 3072}, {151936, 1024}};
    for (int s = 0; s < 6; s++) {
        const int n = shapes[s][0], k = shapes[s][1];
        float *x = (float *)malloc((size_t)k * 4);
        float *out = (float *)malloc((size_t)n * 4);
        uint16_t *w = (uint16_t *)malloc((size_t)n * k * 2);
        for (int i = 0; i < k; i++) x[i] = 0.01f * (float)(i % 100);
        for (size_t i = 0; i < (size_t)n * k; i++) w[i] = (uint16_t)((i * 2654435761u) & 0xBFFFu);
        const int iters = (size_t)n * k > 200000000u ? 3 : 20;
        GemvCtx ctx = {x, w, out, n, k};
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            for (int i = 0; i < iters; i++) {
                if (threads > 1) tf_riscv_pool_run(gemv_shard, &ctx, n);
                else tf_riscv_matmul_bf16(x, w, out, 1, n, k);
            }
            double dt = (now_s() - t0) / iters;
            if (dt < best) best = dt;
        }
        double bytes = (double)n * k * 2.0;
        printf("gemv_bf16 %6dx%-5d %8.2f ms  %7.2f GB/s  %6.2f GMAC/s\n", n, k, best * 1e3, bytes / best / 1e9, (double)n * k / best / 1e9);
        free(x);
        free(out);
        free(w);
    }

    // GEMM (prefill): 64 rows, single call (weight rows reused every 64 rows).
    {
        const int m = 64, n = 1024, k = 1024;
        float *x = (float *)malloc((size_t)m * k * 4);
        float *out = (float *)malloc((size_t)m * n * 4);
        uint16_t *w = (uint16_t *)malloc((size_t)n * k * 2);
        for (size_t i = 0; i < (size_t)m * k; i++) x[i] = 0.01f * (float)(i % 100);
        for (size_t i = 0; i < (size_t)n * k; i++) w[i] = (uint16_t)((i * 2654435761u) & 0xBFFFu);
        GemvCtx ctx = {x, w, out, n, k};
        gemm_rows = m;
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            if (threads > 1) tf_riscv_pool_run(gemv_shard, &ctx, n);
            else tf_riscv_matmul_bf16(x, w, out, m, n, k);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("gemm_bf16 %6dx%-5d %8.2f ms  %7.2f GFLOP/s\n", m, n, best * 1e3, 2.0 * m * n * k / best / 1e9);
        free(x);
        free(out);
        free(w);
    }
    tf_riscv_pool_stop();
    return 0;
}
