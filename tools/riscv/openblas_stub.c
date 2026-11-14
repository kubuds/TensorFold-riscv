//! openblas_stub.c - a tiny cblas_sgemm stand-in so the OpenBLAS code path can
//  be correctness-tested without the real library (tools/riscv/parity.sh links
//  it and points TF_RISCV_BLAS at it). Not used in production; trivially slow.
//  SPDX-License-Identifier: Apache-2.0
#include <stddef.h>

void openblas_set_num_threads(int n) { (void)n; }

// Row-major C = A(m,k) * B(n,k)^T, the only shape the backend calls.
void cblas_sgemm(int order, int transa, int transb, int m, int n, int k,
                 float alpha, const float *a, int lda, const float *b, int ldb,
                 float beta, float *c, int ldc) {
    (void)order;
    (void)transa;
    (void)transb;
    (void)lda;
    (void)ldb;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int t = 0; t < k; t++) s += a[(size_t)i * k + t] * b[(size_t)j * k + t];
            c[(size_t)i * n + j] = alpha * s + beta * c[(size_t)i * n + j];
        }
    }
}
