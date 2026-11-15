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
#include <stdlib.h>

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
