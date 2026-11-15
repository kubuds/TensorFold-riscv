//! pool.h - the sharded-work thread pool behind the backend's multithreaded
//  projections. C11 + pthreads so every kernel family, every libc and the
//  Zig runtime share one pool without pulling an Io instance into the math.
//
//  SPDX-License-Identifier: Apache-2.0
//  (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#ifndef TF_RISCV_POOL_H
#define TF_RISCV_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

// One shard of a parallel-for over [0, items): the pool calls fn(ctx, begin, end).
typedef void (*tf_riscv_task)(void *ctx, int begin, int end);

// Create the pool with `threads` participants (the caller plus threads-1
// workers). threads <= 1 or a second call: the pool stops and restarts.
// Returns the participant count actually running (>= 1).
int tf_riscv_pool_init(int threads);

// Run fn over [0, items) sharded across the pool, blocking until done.
// Runs inline when the pool is single-threaded or items is small.
void tf_riscv_pool_run(tf_riscv_task fn, void *ctx, int items);

// Stop the workers (idempotent).
void tf_riscv_pool_stop(void);

// Participants of the current placement (threads, or A100 tiles).
int tf_riscv_pool_active(void);

#ifdef __cplusplus
}
#endif

#endif // TF_RISCV_POOL_H
