// probe_k3.c - K3 hardware probes behind the RISC-V backend's measurements.
//
// Answers, on the machine, the questions the A100/IME/TCM port depends on:
//   mem       streaming read bandwidth: X100 pthreads vs A100 spine tiles,
//             4 KB pages vs MADV_HUGEPAGE, 1..8 workers;
//   dispatch  cost of spine_parallel_dispatch_1d, sync and async tokens;
//   tile      per-tile VLEN, clock, plain-RVV vs vmadot instruction rates, and
//             the vmadot lane semantics in the A100 (VLEN=1024) context.
//
// Build on the K3 (Bianbu 4.0.8, GCC 15.2, libspert 0.6.3):
//   gcc -O3 -march=rv64gcv_xsmtvdotii -mtune=spacemit-a100 probe_k3.c \
//       -o probe_k3 -lspert -lpthread
// Run: ./probe_k3 mem|dispatch|tile [chunk_MiB]
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <riscv_vector.h>

// The spine ABI header (/usr/include/spert_abi.h) is C++-only; the symbols
// themselves are extern "C". Declared here as the C subset this probe uses.
int64_t spine_get_num_cores(void);
int64_t spine_get_ai_core_vlen(void);
int64_t spine_get_ai_core_tcm_size(void);
int64_t spine_require_stream(void);
void spine_release_stream(int64_t stream);
void spine_parallel_dispatch_1d(int64_t stream, void *function, void *function_args, int64_t grid_x_size);
int64_t spine_parallel_dispatch_1d_async(int64_t stream, void *function, void *function_args, int64_t grid_x_size);
void spine_parallel_sync(int64_t stream, int64_t *tile, int64_t tile_size);
int64_t spine_grid(int64_t ctx, int64_t axis);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

// ---- streaming read --------------------------------------------------------

typedef struct {
    const uint8_t *base;
    size_t bytes;
    int iters;
    volatile uint64_t sink;
} ReadJob;

static inline uint64_t stream_read(const uint8_t *p, size_t bytes) {
    size_t vmax = __riscv_vsetvlmax_e8m8();
    vuint8m8_t acc = __riscv_vmv_v_x_u8m8(0, vmax);
    size_t i = 0;
    for (; i + vmax <= bytes; i += vmax) {
        vuint8m8_t v = __riscv_vle8_v_u8m8(p + i, vmax);
        acc = __riscv_vadd_vv_u8m8(acc, v, vmax);
    }
    for (; i + 128 <= bytes; i += 128) { // tail: does not affect the steep part
        size_t vl = __riscv_vsetvl_e8m8(bytes - i);
        vuint8m8_t v = __riscv_vle8_v_u8m8(p + i, vl);
        acc = __riscv_vadd_vv_u8m8(acc, v, vl);
    }
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(0, __riscv_vsetvlmax_e8m1());
    vuint8m1_t red = __riscv_vredsum_vs_u8m8_u8m1(acc, zero, vmax);
    return (uint64_t)__riscv_vmv_x_s_u8m1_u8(red);
}

static void *x100_worker(void *arg) {
    ReadJob *j = (ReadJob *)arg;
    uint64_t s = 0;
    for (int t = 0; t < j->iters; t++) s += stream_read(j->base, j->bytes);
    j->sink = s;
    return NULL;
}

static void nlane_dummy(int64_t ctx, void *args) {
    (void)args;
    (void)spine_grid(ctx, 0);
}

static void a100_tile(int64_t ctx, void *args) {
    (void)ctx;
    ReadJob *j = (ReadJob *)args;
    uint64_t s = 0;
    for (int t = 0; t < j->iters; t++) s += stream_read(j->base, j->bytes);
    j->sink = s;
}

// ---- per-tile probes -------------------------------------------------------

typedef struct {
    int64_t vlen_max_e8;   // vsetvlmax_e8m1: bytes per register
    uint64_t cycle0, cycle1, cycle2, cycle3;
    uint64_t sink;
} TileInfo;

static TileInfo tiles[8];

// A plain-RVV control loop (vadd.vv, LMUL=8) and a vmadot loop, both timed
// with rdcycle, so per-tile frequency and per-instruction cost separate the
// instruction rate from dispatch overhead.
static void tile_probe(int64_t ctx, void *args) {
    (void)args;
    int ith = (int)spine_grid(ctx, 0);
    if (ith < 0 || ith > 7) return;
    TileInfo *ti = &tiles[ith];
    static int8_t a[128], b[128];
    for (int i = 0; i < 128; i++) { a[i] = (int8_t)(i * 3); b[i] = (int8_t)(i * 5); }

    size_t vl = __riscv_vsetvlmax_e8m1();
    ti->vlen_max_e8 = (int64_t)vl;

    uint64_t s = 0;
    const int iters = 20000;
    vuint8m8_t acc8 = __riscv_vmv_v_x_u8m8(0, __riscv_vsetvlmax_e8m8());
    uint64_t c0 = rdcycle();
    for (int it = 0; it < iters; it++) {
        acc8 = __riscv_vadd_vv_u8m8(acc8, acc8, __riscv_vsetvlmax_e8m8());
    }
    uint64_t c1 = rdcycle();
    s += (uint64_t)__riscv_vmv_x_s_u8m1_u8(__riscv_vredsum_vs_u8m8_u8m1(acc8, __riscv_vmv_v_x_u8m1(0, __riscv_vsetvlmax_e8m1()),
                                                                     __riscv_vsetvlmax_e8m8()));

    // vmadot: vd must be a multiple of 4 (assembler), 4 registers wide.
    uint64_t c2 = rdcycle();
    for (int it = 0; it < iters; it++) {
        __asm__ volatile(
            "vsetvli t0, %[vl], e8, m1  \n\t"
            "vle8.v  v8, (%[pa])        \n\t"
            "vle8.v  v9, (%[pb])        \n\t"
            "vsetvli t0, zero, e8, m1   \n\t"
            "vmadot  v4, v8, v9         \n\t"
            :
            : [pa] "r"(a), [pb] "r"(b), [vl] "r"(vl)
            : "t0", "v4", "v5", "v6", "v7", "v8", "v9", "memory");
    }
    uint64_t c3 = rdcycle();
    ti->cycle0 = c0;
    ti->cycle1 = c1;
    ti->cycle2 = c2;
    ti->cycle3 = c3;
    ti->sink = s;

    // semantics: a[i]=i, b=1, VL=min(vl,128); store vd as e32,m4.
    static int32_t out[512];
    static int8_t sa[128];
    for (int i = 0; i < 128; i++) { sa[i] = (int8_t)i; a[i] = 1; }
    size_t vl2 = vl < 128 ? vl : 128;
    __asm__ volatile(
        "vsetvli t0, %[vl], e8, m1   \n\t"
        "vle8.v  v8, (%[pa])         \n\t"
        "vle8.v  v9, (%[pa2])        \n\t"
        "vsetvli t0, zero, e32, m4   \n\t"
        "vmv.v.i v4, 0               \n\t"
        "vsetvli t0, zero, e8, m1    \n\t"
        "vmadot  v4, v8, v9          \n\t"
        "vsetvli t0, %[vl], e32, m4  \n\t"
        "vse32.v v4, (%[po])         \n\t"
        :
        : [pa] "r"(sa), [pa2] "r"(a), [po] "r"(out), [vl] "r"(vl2)
        : "t0", "v4", "v5", "v6", "v7", "v8", "v9", "memory");
    printf(" tile %d: vlen/b=e8 1reg=%ld  vmadot out:", ith, (long)vl);
    for (int i = 0; i < 12; i++) printf(" %d", out[i & 31]);
    printf("   (out[32]=%d out[64]=%d out[96]=%d) sum_out=%d\n", out[32], out[64], out[96],
           out[0] + out[1] + out[2] + out[3]);
}

// ---- main ------------------------------------------------------------------

static uint8_t *alloc_stream(size_t bytes, int huge) {
    uint8_t *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    if (huge && madvise(p, bytes, MADV_HUGEPAGE) != 0) perror("madvise(HUGEPAGE)");
    memset(p, 1, bytes);
    return p;
}

static void bind_x100(void) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 8; i++) CPU_SET(i, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity 0-7");
}

static void mem_probe(size_t mib) {
    size_t bytes = mib << 20;
    for (int huge = 0; huge < 2; huge++) {
        uint8_t *buf = alloc_stream(bytes, huge);
        for (int nw = 1; nw <= 8; nw *= 8) {
            ReadJob jobs[8];
            pthread_t th[8];
            size_t per = bytes / nw;
            bind_x100();
            double t0 = now_s();
            for (int i = 0; i < nw; i++) {
                jobs[i] = (ReadJob){buf + i * per, per, 4, 0};
                if (i == 0) {
                    x100_worker(&jobs[0]);
                } else if (pthread_create(&th[i], NULL, x100_worker, &jobs[i]) != 0) {
                    perror("pthread_create");
                }
            }
            for (int i = 1; i < nw; i++) pthread_join(th[i], NULL);
            double dt = now_s() - t0;
            printf("x100 pthread read %d worker(s) %-4s: %6.2f GB/s\n", nw, huge ? "huge" : "4k",
                   (double)(bytes * 4) / dt / 1e9);
        }
        munmap(buf, bytes);
    }

    for (int huge = 0; huge < 2; huge++) {
        uint8_t *buf = alloc_stream(bytes, huge);
        int64_t stream = spine_require_stream();
        ReadJob jobs[8];
        size_t per = bytes / 8;
        for (int i = 0; i < 8; i++) jobs[i] = (ReadJob){buf + i * per, per, 4, 0};
        double t0 = now_s();
        spine_parallel_dispatch_1d(stream, (void *)a100_tile, jobs, 8);
        double dt = now_s() - t0;
        printf("a100 spine read 8 tiles %-4s: %6.2f GB/s\n", huge ? "huge" : "4k", (double)(bytes * 4) / dt / 1e9);
        spine_release_stream(stream);
        munmap(buf, bytes);
    }
}

static void dispatch_probe(void) {
    ReadJob zero = {0, 0, 1, 0};
    int64_t stream = spine_require_stream();
    double t0 = now_s();
    for (int i = 0; i < 200; i++) spine_parallel_dispatch_1d(stream, (void *)nlane_dummy, &zero, 8);
    double sync_us = (now_s() - t0) / 200 * 1e6;
    t0 = now_s();
    for (int i = 0; i < 200; i++) {
        int64_t tok = spine_parallel_dispatch_1d_async(stream, (void *)nlane_dummy, &zero, 8);
        spine_parallel_sync(stream, &tok, 1);
    }
    double async_us = (now_s() - t0) / 200 * 1e6;
    printf("spine empty 8-tile dispatch: sync %.2f us, async+sync %.2f us\n", sync_us, async_us);
    spine_release_stream(stream);
}

int main(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "mem";
    size_t mib = argc > 2 ? (size_t)atoi(argv[2]) : 256;
    printf("probe_k3 %s: cpus %ld, spert cores %lld vlen %lld bytes tcm %lld bytes\n", what,
           sysconf(_SC_NPROCESSORS_ONLN), (long long)spine_get_num_cores(), (long long)spine_get_ai_core_vlen(),
           (long long)spine_get_ai_core_tcm_size());
    if (strcmp(what, "mem") == 0) mem_probe(mib);
    else if (strcmp(what, "dispatch") == 0) dispatch_probe();
    else if (strcmp(what, "tile") == 0) {
        int64_t stream = spine_require_stream();
        double t0 = now_s();
        spine_parallel_dispatch_1d(stream, (void *)tile_probe, NULL, 8);
        double wall = now_s() - t0;
        spine_release_stream(stream);
        for (int i = 0; i < 8; i++) {
            TileInfo *t = &tiles[i];
            double add_cyc = (double)(t->cycle1 - t->cycle0) / 20000.0;
            double dot_cyc = (double)(t->cycle3 - t->cycle2) / 20000.0;
            printf("tile %d: vadd.vv %6.2f cyc/instr, vmadot+2loads+2vsetvl %6.2f cyc/iter\n", i, add_cyc, dot_cyc);
        }
        printf("8-tile probe wall %.3f ms\n", wall * 1e3);
    } else {
        printf("usage: probe_k3 mem|dispatch|tile [MiB]\n");
        return 2;
    }
    return 0;
}
