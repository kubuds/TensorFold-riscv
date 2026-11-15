//! mem.c - weight arenas for the K3 backend.
//
// HPAGE: the vendor llama.cpp's mem_backend=HPAGE maps its weight arena with
// huge pages (spacemit-llama.cpp spine_mem_pool.cpp: mmap + MADV_HUGEPAGE,
// or /dev/hugetlb_1g). Measured on the K3 (tools/riscv/probe_k3.c mem), eight
// A100 tiles streaming a 256 MiB buffer read 30.0 GB/s with 4 KiB pages and
// 40.5 GB/s with THP: the TLB is the difference between a decode token at
// 40 and at 55 tok/s. This file is TensorFold's own arena: anonymous mmap,
// 2 MiB aligned, MADV_HUGEPAGE, and a record of how much of it the kernel
// actually backed with huge pages (reported in `caps`).
//
// SPDX-License-Identifier: Apache-2.0
// (c) 2026 TensorFold contributors; see NOTICE and LICENSE.
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE // MADV_HUGEPAGE, MAP_POPULATE: must precede any header
#endif
#include "tf_riscv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static size_t hpage_total = 0;
static size_t hpage_allocations = 0;

int tf_riscv_hpage_available(void) {
#if defined(__linux__)
    FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (!f) return 0;
    char buf[128] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    (void)n;
    // "[always]" or "[madvise]" mean THP can back this arena; "[never]" cannot.
    return strstr(buf, "[always]") != NULL || strstr(buf, "[madvise]") != NULL;
#else
    return 0;
#endif
}

void *tf_riscv_hpage_alloc(size_t bytes) {
#if defined(__linux__)
    size_t page = 2u << 20; // the K3's THP size is 2 MiB (probe: smaps)
    size_t size = (bytes + page - 1) & ~(page - 1);
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    (void)madvise(p, size, MADV_HUGEPAGE);
    hpage_total += size;
    hpage_allocations++;
    return p;
#else
    (void)bytes;
    return NULL;
#endif
}

void tf_riscv_hpage_free(void *p, size_t bytes) {
#if defined(__linux__)
    if (!p) return;
    size_t page = 2u << 20;
    size_t size = (bytes + page - 1) & ~(page - 1);
    munmap(p, size);
    if (hpage_total >= size) hpage_total -= size;
    if (hpage_allocations) hpage_allocations--;
#else
    (void)p;
    (void)bytes;
#endif
}

// Backed huge-page bytes for the arena: /proc/self/smaps AnonHugePages of the
// mapping that contains `p`. 0 when the kernel backed it with 4 KiB pages.
size_t tf_riscv_hpage_backed(const void *p) {
#if defined(__linux__)
    if (!p) return 0;
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    char line[512];
    unsigned long long start = 0, end = 0;
    const unsigned long long want = (unsigned long long)(uintptr_t)p;
    int inside = 0;
    size_t backed = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long a, b;
        if (sscanf(line, "%llx-%llx", &a, &b) == 2) {
            inside = (want >= a && want < b);
            start = a;
            end = b;
            (void)start;
            (void)end;
            continue;
        }
        if (inside && strncmp(line, "AnonHugePages:", 14) == 0) {
            unsigned long long kb = 0;
            sscanf(line + 14, "%llu kB", &kb);
            backed = (size_t)kb * 1024;
            break;
        }
    }
    fclose(f);
    return backed;
#else
    (void)p;
    return 0;
#endif
}

size_t tf_riscv_hpage_bytes(void) { return hpage_total; }

// Synchronously collapse the arena's eligible pages into transparent huge
// pages (MADV_COLLAPSE, Linux 6.1+). MADV_HUGEPAGE only marks the VMA for
// khugepaged, which on the K3 had not promoted a 3.4 GiB arena by the time
// the first bench run started (hpage_backed read 0); the synchronous collapse
// makes --mem hpage mean what the vendor's mem_backend=HPAGE means. Returns
// the number of collapsed bytes, 0 when the kernel does not support it.
size_t tf_riscv_hpage_collapse(void) {
#if defined(__linux__) && defined(MADV_COLLAPSE)
    size_t total = 0;
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        unsigned long long a = 0, b = 0;
        if (sscanf(line, "%llx-%llx", &a, &b) != 2) continue;
        void *p = (void *)(uintptr_t)a;
        size_t len = (size_t)(b - a);
        if (madvise(p, len, MADV_COLLAPSE) == 0) total += len;
    }
    fclose(f);
    return total;
#else
    return 0;
#endif
}
