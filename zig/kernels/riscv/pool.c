//! pool.c - the sharded-work thread pool (see pool.h).
//
// Workers wait on a condition variable, take their fixed shard of the next
// task and wake the caller when the last shard lands. Shards are disjoint
// output ranges, so results are bit-identical to a serial run.
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#include "pool.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- SpacemiT spine runtime (libspert.so.1): AI-core dispatch ------------
//
// TF_RISCV_SPERT=1 runs the shards on the A100 AI cores through the spine
// runtime's stable C ABI instead of the pthread pool. The ABI shape follows
// SpacemiT's own MIT-licensed adapter (spacemit-llama.cpp,
// ggml/src/ggml-cpu/spacemit/spacemit_spert_adapter.cpp): a stream checks
// out N AI cores from the SHM arbiter, spine_parallel_dispatch_1d runs one
// callback per tile (synchronously), and the tile index comes from
// spine_grid(ctx, 0). No vendor code is copied; the call sequence is
// reimplemented from the published source (MIT, (c) SpacemiT).
#if defined(__linux__)
#include <dlfcn.h>

typedef int64_t (*spert_require_fn)(int);
typedef int64_t (*spert_require_cfg_fn)(int, const int64_t *);
typedef void (*spert_release_fn)(int64_t);
typedef void (*spert_dispatch_fn)(int64_t, void (*)(int64_t, void *), void *, int);
typedef int64_t (*spert_grid_fn)(int64_t, int);

static struct {
    void *handle;
    spert_require_fn require_stream;
    spert_release_fn release_stream;
    spert_dispatch_fn dispatch_1d;
    spert_grid_fn grid;
    int cores; // AI cores the runtime reports
} spert = {0};

static void spert_release_held(void);

static int spert_load(void) {
    static int tried = 0;
    if (tried) return spert.handle != NULL;
    tried = 1;
    const char *want = getenv("TF_RISCV_SPERT");
    if (!want || strcmp(want, "1") != 0) return 0;
    spert.handle = dlopen("libspert.so.1", RTLD_NOW);
    if (!spert.handle) {
        fprintf(stderr, "tf_riscv_pool: libspert.so.1 unavailable (%s)\n", dlerror());
        return 0;
    }
    spert.require_stream = (spert_require_fn)dlsym(spert.handle, "spine_require_stream");
    spert.release_stream = (spert_release_fn)dlsym(spert.handle, "spine_release_stream");
    spert.dispatch_1d = (spert_dispatch_fn)dlsym(spert.handle, "spine_parallel_dispatch_1d");
    spert.grid = (spert_grid_fn)dlsym(spert.handle, "spine_grid");
    typedef int (*spert_cores_fn)(void);
    spert_cores_fn cores = (spert_cores_fn)dlsym(spert.handle, "spine_get_num_cores");
    spert.cores = cores ? cores() : 8;
    if (!spert.require_stream || !spert.release_stream || !spert.dispatch_1d || !spert.grid) {
        spert.handle = NULL;
        return 0;
    }
    fprintf(stderr, "tf_riscv_pool: AI-core dispatch via libspert (%d cores)\n", spert.cores);
    atexit(spert_release_held);
    return 1;
}

// The stream is checked out once and held: per-call require/release measured
// ~1.2 ms of arbiter round trip on K3 (fine for a graph compute, fatal for the
// ~200 dispatches a decode token makes). Released at pool_stop/atexit; a
// serving deployment that shares the AI cores with other backends should set
// TF_RISCV_SPERT_SCOPED=1 to release after every dispatch like the vendor
// adapter does.
static int64_t spert_stream = 0;
static int spert_stream_tiles = 0;

typedef struct {
    tf_riscv_task fn;
    void *ctx;
    int items;
    int tiles;
} SpertTask;

static void spert_wrap(int64_t tile_ctx, void *args) {
    const SpertTask *t = (const SpertTask *)args;
    int ith = (int)spert.grid(tile_ctx, 0);
    if (ith < 0 || ith >= t->tiles) return;
    int begin = (int)((long)t->items * ith / t->tiles);
    int end = (int)((long)t->items * (ith + 1) / t->tiles);
    if (end > begin) t->fn(t->ctx, begin, end);
}

static void spert_release_held(void) {
    if (spert_stream) {
        spert.release_stream(spert_stream);
        spert_stream = 0;
        spert_stream_tiles = 0;
    }
}

static void spert_run(tf_riscv_task fn, void *ctx, int items) {
    int tiles = items < spert.cores ? items : spert.cores;
    if (tiles <= 1) {
        fn(ctx, 0, items);
        return;
    }
    SpertTask t = {fn, ctx, items, tiles};
    const char *scoped = getenv("TF_RISCV_SPERT_SCOPED");
    int hold = !(scoped && strcmp(scoped, "1") == 0);
    if (!hold || !spert_stream || spert_stream_tiles != tiles) {
        spert_release_held();
        spert_stream = spert.require_stream(tiles);
        spert_stream_tiles = tiles;
    }
    if (!spert_stream) {
        fn(ctx, 0, items); // no AI core free right now: correct, just slower
        return;
    }
    spert.dispatch_1d(spert_stream, spert_wrap, &t, tiles);
    if (!hold) spert_release_held();
}
#endif

typedef struct {
    pthread_t *workers;
    int n; // participants: n-1 workers plus the caller
    pthread_mutex_t mu;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    tf_riscv_task fn;
    void *ctx;
    int items;
    int pending;
    int gen; // task generation: a worker runs each generation once
    int stop;
} Pool;

static Pool *pool = NULL;

static void *worker_main(void *arg) {
    const long idx = (long)arg;
    Pool *p = pool;
    int seen_gen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        // Wait for a generation this worker has not run yet.
        while (!p->stop && (p->fn == NULL || p->gen == seen_gen)) pthread_cond_wait(&p->work_cv, &p->mu);
        if (p->stop) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        seen_gen = p->gen;
        tf_riscv_task fn = p->fn;
        void *ctx = p->ctx;
        const int items = p->items;
        const int n = p->n;
        pthread_mutex_unlock(&p->mu);
        int begin = (int)((long)items * idx / n);
        int end = (int)((long)items * (idx + 1) / n);
        if (end > begin) fn(ctx, begin, end);
        pthread_mutex_lock(&p->mu);
        if (--p->pending == 0) {
            p->fn = NULL;
            pthread_cond_signal(&p->done_cv);
        }
        pthread_mutex_unlock(&p->mu);
    }
}

int tf_riscv_pool_init(int threads) {
    tf_riscv_pool_stop();
    if (threads <= 1) return 1;
    Pool *p = (Pool *)calloc(1, sizeof(Pool));
    if (!p) return 1;
    p->n = threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->work_cv, NULL);
    pthread_cond_init(&p->done_cv, NULL);
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
    if (spert_load()) {
        spert_run(fn, ctx, items);
        return;
    }
#endif
    Pool *p = pool;
    if (!p || p->n <= 1 || items < p->n) {
        fn(ctx, 0, items);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->items = items;
    p->pending = p->n;
    p->gen += 1;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mu);

    int begin = 0;
    int end = items / p->n;
    if (end > begin) fn(ctx, begin, end);

    pthread_mutex_lock(&p->mu);
    if (--p->pending == 0) {
        p->fn = NULL;
        pthread_cond_signal(&p->done_cv);
    }
    while (p->pending > 0) pthread_cond_wait(&p->done_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

void tf_riscv_pool_stop(void) {
#if defined(__linux__)
    if (spert.handle) spert_release_held();
#endif
    Pool *p = pool;
    if (!p) return;
    pool = NULL;
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->workers[i - 1], NULL);
    pthread_cond_destroy(&p->work_cv);
    pthread_cond_destroy(&p->done_cv);
    pthread_mutex_destroy(&p->mu);
    free(p->workers);
    free(p);
}
