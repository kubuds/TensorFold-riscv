//! pool.c - the sharded-work thread pool (see pool.h).
//
// Two placements for the same shard callbacks:
//
//   x100  a pthread pool on the general-purpose X100 cores. Workers spin on
//         an atomic generation counter instead of sleeping on a condition
//         variable: a decode token makes ~200 projection calls, and a
//         condvar wake per call costs more than the projection itself.
//   a100  SpacemiT's spine runtime (libspert.so.1) hands the same callbacks to
//         eight A100 AI-core tiles (RV64GCV, VLEN=1024). The stream is checked
//         out once for the process, each tile gets worker-local TCM (the K3
//         AI cores' 393 KiB local SRAM) as its scratch arena, and a call is
//         one synchronous spine_parallel_dispatch_1d grid.
//
// The ABI follows SpacemiT's published MIT source: spert_abi.h ships with
// spacemit-runtime 0.6.3 in /usr/include, and spacemit-llama.cpp's
// ggml/src/ggml-cpu/spacemit adapter (MIT, (c) SpacemiT) documents the call
// sequence. No vendor code is copied; the runtime is dlopen'd, like OpenBLAS.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "pool.h"
#include "tf_riscv.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <dlfcn.h>
#include <sched.h>
#endif

// ---- placement -------------------------------------------------------------

enum { PLACE_AUTO = -1, PLACE_X100 = 0, PLACE_A100 = 1 };
static volatile int placement = PLACE_AUTO;

// ---- spine runtime (A100 tiles) -------------------------------------------

#if defined(__linux__)
// The C subset of /usr/include/spert_abi.h (which is C++-only).
typedef int64_t (*spert_require_fn)(void);
typedef void (*spert_release_fn)(int64_t);
typedef void (*spert_dispatch_fn)(int64_t, void *, void *, int64_t);
typedef int64_t (*spert_grid_fn)(int64_t, int64_t);
typedef void *(*spert_tcm_malloc_fn)(int64_t, size_t);
typedef void (*spert_tcm_free_fn)(int64_t, void *);
typedef int64_t (*spert_i64_fn)(void);

static struct {
    void *handle;
    spert_require_fn require_stream;
    spert_release_fn release_stream;
    spert_dispatch_fn dispatch_1d;
    spert_grid_fn grid;
    spert_tcm_malloc_fn tcm_malloc;
    spert_tcm_free_fn tcm_free;
    spert_i64_fn num_cores;
    spert_i64_fn vlen;
    spert_i64_fn tcm_size;
    int cores;
} spert;

static int spert_load(void) {
    static int tried = 0;
    if (tried) return spert.handle != NULL;
    tried = 1;
    spert.handle = dlopen("libspert.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!spert.handle) {
        fprintf(stderr, "tf_riscv_pool: libspert.so.1 unavailable (%s)\n", dlerror());
        return 0;
    }
    spert.require_stream = (spert_require_fn)dlsym(spert.handle, "spine_require_stream");
    spert.release_stream = (spert_release_fn)dlsym(spert.handle, "spine_release_stream");
    spert.dispatch_1d = (spert_dispatch_fn)dlsym(spert.handle, "spine_parallel_dispatch_1d");
    spert.grid = (spert_grid_fn)dlsym(spert.handle, "spine_grid");
    spert.tcm_malloc = (spert_tcm_malloc_fn)dlsym(spert.handle, "spine_thread_tcm_malloc");
    spert.tcm_free = (spert_tcm_free_fn)dlsym(spert.handle, "spine_thread_tcm_free");
    spert.num_cores = (spert_i64_fn)dlsym(spert.handle, "spine_get_num_cores");
    spert.vlen = (spert_i64_fn)dlsym(spert.handle, "spine_get_ai_core_vlen");
    spert.tcm_size = (spert_i64_fn)dlsym(spert.handle, "spine_get_ai_core_tcm_size");
    spert.cores = spert.num_cores ? (int)spert.num_cores() : 8;
    if (!spert.require_stream || !spert.release_stream || !spert.dispatch_1d || !spert.grid) {
        spert.handle = NULL;
        return 0;
    }
    return 1;
}

static int64_t spert_stream = 0;
static int spert_stream_tiles = 0;

static void spert_release_held(void) {
    if (spert_stream && spert.release_stream) {
        spert.release_stream(spert_stream);
        spert_stream = 0;
        spert_stream_tiles = 0;
    }
}
#endif // __linux__

// Per-tile / per-thread state. tls_tile_ctx is nonzero only inside an A100
// tile; tls_tcm is that tile's local-memory arena.
static __thread int64_t tls_tile_ctx = 0;
static __thread unsigned char *tls_tcm = NULL;
static __thread size_t tls_tcm_cap = 0;
static __thread size_t tls_tcm_used = 0;
static __thread unsigned char *tls_heap = NULL;
static __thread size_t tls_heap_cap = 0;
static __thread unsigned long tls_epoch = 0;
static size_t tcm_peak_used = 0;

static void *thread_scratch(size_t bytes) {
    if (tls_tcm && tls_tile_ctx) {
        const size_t aligned = (bytes + 63) & ~(size_t)63;
        if (tls_tcm_used + aligned <= tls_tcm_cap) {
            void *p = tls_tcm + tls_tcm_used;
            tls_tcm_used += aligned;
            if (tls_tcm_used > tcm_peak_used) tcm_peak_used = tls_tcm_used;
            return p;
        }
    }
    if (tls_heap_cap < bytes) {
        unsigned char *p = (unsigned char *)realloc(tls_heap, bytes);
        if (!p) return NULL;
        tls_heap = p;
        tls_heap_cap = bytes;
    }
    return tls_heap;
}

void *tf_riscv_scratch(size_t bytes) { return thread_scratch(bytes); }
size_t tf_riscv_tcm_bytes(void) { return tls_tcm_cap; }
size_t tf_riscv_tcm_used(void) { return tcm_peak_used; }

#if defined(__linux__)
static size_t tcm_want(void) {
    static size_t cached = (size_t)-1;
    if (cached == (size_t)-1) {
        const char *env = getenv("TF_RISCV_TCM_KB");
        long kb = env ? strtol(env, NULL, 10) : 320;
        if (kb <= 0) kb = 0;
        const size_t max = spert.tcm_size ? (size_t)spert.tcm_size() : 0;
        size_t bytes = (size_t)kb * 1024;
        if (max && (bytes + 4096 > max)) bytes = max > 4096 ? max - 4096 : 0;
        cached = bytes;
    }
    return cached;
}

static void tcm_release_tile(void) {
    if (tls_tcm && spert.tcm_free && tls_tile_ctx) spert.tcm_free(tls_tile_ctx, tls_tcm);
    tls_tcm = NULL;
    tls_tcm_cap = 0;
    tls_tcm_used = 0;
}

// Once per tile execution: obtain (or keep) the TCM arena and reset the bump
// pointer. Cached across dispatches by default (an alloc/free pair measured
// ~2.4 us); TF_RISCV_TCM_CACHE=0 releases it after every tile.
static void tcm_enter(int64_t ctx) {
    tls_tile_ctx = ctx;
    tls_tcm_used = 0;
    tls_epoch++;
    const char *off = getenv("TF_RISCV_TCM");
    if (off && strcmp(off, "0") == 0) return;
    if (tls_tcm || !spert.tcm_malloc) return;
    size_t want = tcm_want();
    if (want < 4096) return;
    void *p = spert.tcm_malloc(ctx, want);
    if (!p) {
        want /= 4; // some firmware splits TCM unevenly across tiles
        p = spert.tcm_malloc(ctx, want);
    }
    if (!p) return;
    tls_tcm = (unsigned char *)p;
    tls_tcm_cap = want;
}

static void tcm_leave(void) {
    const char *cache = getenv("TF_RISCV_TCM_CACHE");
    if (cache && strcmp(cache, "0") == 0) tcm_release_tile();
    tls_tile_ctx = 0;
}

typedef struct {
    tf_riscv_task fn;
    void *ctx;
    int items;
    int tiles;
} SpertTask;

static void spert_wrap(int64_t tile_ctx, void *args) {
    const SpertTask *t = (const SpertTask *)args;
    const int ith = (int)spert.grid(tile_ctx, 0);
    if (ith < 0 || ith >= t->tiles) return;
    tcm_enter(tile_ctx);
    const int begin = (int)((long)t->items * ith / t->tiles);
    const int end = (int)((long)t->items * (ith + 1) / t->tiles);
    if (end > begin) t->fn(t->ctx, begin, end);
    tcm_leave();
}

static void spert_run(tf_riscv_task fn, void *ctx, int items) {
    const int tiles = items < spert.cores ? items : spert.cores;
    if (tiles <= 1 || spert.cores <= 1) {
        fn(ctx, 0, items);
        return;
    }
    SpertTask t = {fn, ctx, items, tiles};
    if (!spert_stream || spert_stream_tiles != tiles) {
        spert_release_held();
        spert_stream = spert.require_stream();
        spert_stream_tiles = spert_stream ? tiles : 0;
    }
    if (!spert_stream) {
        fn(ctx, 0, items); // no AI core free right now: correct, just slower
        return;
    }
    spert.dispatch_1d(spert_stream, (void *)spert_wrap, &t, tiles);
}
#endif // __linux__

static int placement_is_a100(void) {
    int p = placement;
    if (p == PLACE_AUTO) {
#if defined(__linux__)
        const char *env = getenv("TF_RISCV_A100");
        if (env && strcmp(env, "0") != 0 && spert_load()) p = PLACE_A100;
        else p = PLACE_X100;
#else
        p = PLACE_X100;
#endif
        placement = p;
    }
    return p == PLACE_A100;
}

int tf_riscv_a100_available(void) {
#if defined(__linux__)
    return spert_load();
#else
    return 0;
#endif
}

int tf_riscv_pool_mode(const char *mode) {
    if (!mode || strcmp(mode, "auto") == 0) {
        placement = PLACE_AUTO;
        return placement_is_a100() ? PLACE_A100 : PLACE_X100;
    }
    if (strcmp(mode, "a100") == 0) {
        if (!tf_riscv_a100_available()) return -1;
        placement = PLACE_A100;
        return PLACE_A100;
    }
    if (strcmp(mode, "x100") == 0) {
        placement = PLACE_X100;
        return PLACE_X100;
    }
    return -1;
}

const char *tf_riscv_pool_mode_name(void) { return placement_is_a100() ? "a100" : "x100"; }

// ---- pthread pool (X100) ---------------------------------------------------

typedef struct {
    pthread_t *workers;
    int n;               // participants: n-1 workers plus the caller
    volatile int gen;    // bumped by the caller to publish a task
    volatile int done;   // workers that finished the current generation
    volatile int stop;
    tf_riscv_task fn;
    void *ctx;
    int items;
} Pool;

static Pool *pool = NULL;

// Workers spin for the next generation: a serving deployment runs back-to-back
// projections, so the wake latency would dominate. A long spin yields so an
// idle pool does not hold a core.
static void *worker_main(void *arg) {
    const long idx = (long)arg;
    Pool *p = pool;
    int seen = 0;
    for (;;) {
        int spins = 0;
        while (!p->stop && p->gen == seen) {
            if (++spins < 50000) {
                __asm__ volatile("" ::: "memory");
            } else {
                sched_yield();
            }
        }
        if (p->stop) return NULL;
        seen = p->gen;
        __sync_synchronize();
        tf_riscv_task fn = p->fn;
        void *ctx = p->ctx;
        const int items = p->items;
        const int n = p->n;
        const int begin = (int)((long)items * idx / n);
        const int end = (int)((long)items * (idx + 1) / n);
        if (end > begin) fn(ctx, begin, end);
        __sync_fetch_and_add(&p->done, 1);
    }
}

int tf_riscv_pool_init(int threads) {
    tf_riscv_pool_stop();
    if (threads <= 1) return 1;
    Pool *p = (Pool *)calloc(1, sizeof(Pool));
    if (!p) return 1;
    p->n = threads;
    p->workers = (pthread_t *)calloc((size_t)threads - 1, sizeof(pthread_t));
    if (!p->workers) {
        free(p);
        return 1;
    }
    pool = p;
    for (long i = 1; i < threads; i++) {
        if (pthread_create(&p->workers[i - 1], NULL, worker_main, (void *)i) != 0) {
            p->n = (int)i; // fewer workers than asked: still correct
            break;
        }
    }
    return p->n;
}

void tf_riscv_pool_run(tf_riscv_task fn, void *ctx, int items) {
#if defined(__linux__)
    if (placement_is_a100()) {
        if (spert_load()) {
            spert_run(fn, ctx, items);
            return;
        }
        placement = PLACE_X100;
    }
#endif
    Pool *p = pool;
    if (!p || p->n <= 1 || items < p->n) {
        fn(ctx, 0, items);
        return;
    }
    p->fn = fn;
    p->ctx = ctx;
    p->items = items;
    p->done = 0;
    __sync_synchronize();
    __sync_fetch_and_add(&p->gen, 1);

    const int end = items / p->n;
    if (end > 0) fn(ctx, 0, end);

    int spins = 0;
    while (p->done < p->n - 1) {
        if (++spins < 500000) {
            __asm__ volatile("" ::: "memory");
        } else {
            sched_yield();
        }
    }
}

void tf_riscv_pool_stop(void) {
#if defined(__linux__)
    if (spert.handle) spert_release_held();
#endif
    Pool *p = pool;
    if (!p) return;
    pool = NULL;
    p->stop = 1;
    __sync_synchronize();
    for (int i = 1; i < p->n; i++) pthread_join(p->workers[i - 1], NULL);
    free(p->workers);
    free(p);
}

// The tile-execution epoch: scratch cached by a kernel is valid only for the
// epoch it was taken in (the TCM bump resets every dispatch).
unsigned long tf_riscv_scratch_epoch(void) { return tls_epoch; }

int tf_riscv_pool_active(void) {
    if (placement_is_a100()) {
#if defined(__linux__)
        if (spert.cores > 0) return spert.cores;
#endif
    }
    return pool ? pool->n : 1;
}
