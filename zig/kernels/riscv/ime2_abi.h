//! ime2_abi.h - the vendor-runtime seam for SpacemiT's IME 2.0 tensor unit.
//
// The K3 paper (SpacemiT K3: A RVA23 RISC-V AI CPU with 60 TOPS AI Compute)
// defines IME 2.0's MMA shapes but the preview does not publish instruction
// encodings, so the backend reaches the unit through a tiny vendor runtime
// library with this ABI (TF_RISCV_IME2=/path/lib.so). When SpacemiT ships the
// SDK, only that library changes; the kernels, their tiling and every test
// stay. tools/riscv/ime2_shim.c implements this ABI with reference math so
// the whole path is regression-tested today.
//
// Matrix contract (the paper's Table 3):
//   mma_bf16: D[8,8] fp32 = A[8,8] bf16 * B[8,8] bf16 + C[8,8] fp32
//   mma_i8:   D[8,8] i32  = A[8,16] i8  * B[16,8] i8  + C[8,8] i32
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#ifndef TF_RISCV_IME2_ABI_H
#define TF_RISCV_IME2_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // 1 when the hardware path is live; the shim returns 1 as well.
    int (*available)(void);
    // A and B are bf16 lanes (uint16); C and D are fp32, row-major 8x8.
    void (*mma_bf16)(const uint16_t *a, const uint16_t *b, const float *c, float *d);
    // A is 8x16 i8, B is 16x8 i8, C and D are i32 8x8, row-major.
    void (*mma_i8)(const int8_t *a, const int8_t *b, const int32_t *c, int32_t *d);
} tf_riscv_ime2_abi;

#ifdef __cplusplus
}
#endif

#endif // TF_RISCV_IME2_ABI_H
