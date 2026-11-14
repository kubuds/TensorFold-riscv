//! ime2_shim.c - a reference implementation of the IME2 vendor runtime ABI
//  (zig/kernels/riscv/ime2_abi.h) so the whole vendor path - dlopen, vtable
//  lookup, tiled MMA - is regression-tested before SpacemiT's SDK exists.
//  On K3 this library is replaced by the vendor one that drives the real
//  MMA instructions; nothing else in the backend changes.
//
//  The reference MMA computes the tiles' products in float64 like the inline
//  emulation; real IME2 multiplies bf16 lanes (the backend already rounds its
//  A operand to bf16, so the deviation is bounded by the same input rounding).
//  SPDX-License-Identifier: Apache-2.0
//  (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "ime2_abi.h"

static int available(void) { return 1; }

static void mma_bf16(const uint16_t *a, const uint16_t *b, const float *c, float *d) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double s = (double)c[i * 8 + j];
            for (int t = 0; t < 8; t++) {
                union {
                    uint32_t u;
                    float f;
                } x = {.u = (uint32_t)a[i * 8 + t] << 16}, y = {.u = (uint32_t)b[t * 8 + j] << 16};
                s += (double)x.f * (double)y.f;
            }
            d[i * 8 + j] = (float)s;
        }
    }
}

static void mma_i8(const int8_t *a, const int8_t *b, const int32_t *c, int32_t *d) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int32_t s = c[i * 8 + j];
            for (int t = 0; t < 16; t++) s += (int32_t)a[i * 16 + t] * (int32_t)b[t * 8 + j];
            d[i * 8 + j] = s;
        }
    }
}

static const tf_riscv_ime2_abi abi = {available, mma_bf16, mma_i8};

// The entry dispatch.c looks up after dlopen.
const tf_riscv_ime2_abi *tf_riscv_ime2_abi_v1(void) { return &abi; }
