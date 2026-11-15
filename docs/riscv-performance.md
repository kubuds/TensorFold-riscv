# RISC-V backend performance report (SpacemiT K3)

All numbers were measured on the K3 test machine `k3pico16g.local` (Bianbu
4.0.8, Linux 6.18.3, 16 GB LPDDR5) with the committed harness and receipts
in `~/receipts/` on that machine. **Correction notice**: the first version of
this report compared against a self-built upstream llama.cpp. The baseline of
record is the SpacemiT vendor-optimized llama.cpp; both of its sources are
now measured (section 2), and the comparison below is redone against it.

## 1. Summary

Qwen3-0.6B, decode tg128 (the serving metric), all on the K3 SoC:

| engine | placement | prefill pp512 (tok/s) | decode tg128 (tok/s) |
| --- | --- | --- | --- |
| **vendor llama.cpp apt 0.2.2, Q4_K_M** | A100 + IME2 + TCM/HPAGE | 326.5 | **40.9** |
| vendor llama.cpp apt 0.2.2, Q8_0 | A100 + IME2 | 302.2 | 32.0 |
| vendor llama.cpp apt 0.2.2, BF16 | A100 + IME2 | 2.7 | 1.9 |
| spacemit-llama.cpp source 27acf9d, Q4_K_M | X100 CPU path | 92.4 | 4.2 |
| TensorFold q8_0 (this backend) | X100, RVV | 15.1 | 11.7 |
| TensorFold bf16 | X100, RVV | 15.2 | 8.7 |
| upstream llama.cpp (self-built), Q4_K_M | X100 CPU path | 92.6 | 4.1 |

**Status against the goal (beat the vendor-native llama.cpp): not yet.**
Decode reaches 29% of the vendor Q4_K_M rate (11.7 vs 40.9); prefill 5%.
What the gap is made of is measured in section 4 - it is not ISA-luck: the
vendor path reads 3.7x fewer bytes per token (Q4_K_M repack) at about 2x
the effective memory rate (TCM/HPAGE tuned streams) and computes prefill
through IME2 INT MMA tiles. Section 5 is the concrete path to parity.

Where we do win (same Qwen3-0.6B, same machine): decode is 4.5x the vendor
binary's own BF16 path (8.7 vs 1.9 tok/s) and 2.8x the upstream llama.cpp at
equal format; the vendor's speed comes entirely from its quantized+IME2
paths, its BF16 fallback is slower than ours.

## 2. The two vendor sources (as requested)

1. **apt baseline**: `llama.cpp-tools-spacemit` 0.2.2 (archive.spacemit.com,
   resolute-porting/main), binaries `/usr/bin/llama-bench` etc., builder
   51c81f90f (10056). Startup log: `use_ime2=1`, `cpu_mask: ff00`
   (compute pinned to the A100 set 8-15), `mem_backend: HPAGE`, TCM
   available (8 x 393 KiB), device reports `SPACEMIT`. This is the baseline
   of record. Full matrix (~/receipts/llama-vendor-bench.txt):

   | quant | pp64 t4 | tg64 t4 | pp512 t8 | tg128 t8 |
   | --- | --- | --- | --- | --- |
   | Q4_K_M | 214.9 | 37.3 | 326.5 | 40.9 |
   | Q8_0 | 200.1 | 30.7 | 302.2 | 32.0 |
   | BF16 | 1.4 | 1.0 | 2.7 | 1.9 |

2. **source** (kernels' origin): github.com/spacemit-com/llama.cpp, branch
   mtmd-backend @ 27acf9d (2026-08-05), built natively on K3
   (~/receipts/llama-source-bench.txt). This commit builds the plain CPU
   path (no spine runtime): Q4_K_M pp64/tg64 t4 = 64.2/18.6, pp512/tg128
   t8 = 92.4/4.2. Its newer master carries the spine/IME2 adapter
   (ggml/src/ggml-cpu/spacemit/*) that the apt package ships.

   For direct comparison at the user's methodology (-p 64 -n 64 -t 4):
   vendor 37.3 tg64, source-build 18.6, TensorFold q8_0 15.3
   (~/receipts/tensorfold-t4.txt).

Method notes: `llama-bench` defaults to its own thread/core placement (the
vendor binary checks out A100 cores through the spine runtime); TensorFold
was pinned with --cpus 0-7 on X100 unless noted. A run with the vendor
binary confirms its workers execute on CPUs 8-15 while the process affinity
mask still reads 0-7 (threads are placed by the spine runtime, not by
sched_setaffinity).

## 3. Correctness (unchanged, all on the machine)

14/14 tests native on K3 and under qemu (rv64gc + rv64gcv at VLEN 256/1024);
numpy-oracle parity OK for --quant none/q8_0/q4_0 with identical greedy
tokens; bf16 and q8_0 greedy text runs agree. The hardware gate also caught
and fixed a real sin/cos precision bug the qemu budget had hidden (2.1e-4
rope drift -> 1.2e-7).

## 4. Where the gap comes from (measured)

Per token, Qwen3-0.6B streams: bf16 1.18 GB, q8_0 0.64 GB, Q4_K_M 0.37 GB.

1. **Bytes per token**: our best decode (q8_0) reads 1.7x the vendor's
   Q4_K_M. Our q4_0 path is correct but slow (scalar nibble dequant: 6.5
   tok/s vs q8_0's 11.7), so the deployable quant today is q8_0.
2. **Effective bandwidth**: our decode reads ~7.5 GB/s end to end (8
   threads) while the vendor sustains ~15 GB/s (their HPAGE/TCM-tuned
   streams). Our kernel micro-rate is 15-20 GB/s on big shapes, so the loss
   is end-to-end shape mix, serial non-GEMV work and per-call overhead -
   not the raw stream. Together (1) and (2) explain the 3.5x decode gap
   almost exactly (1.7 x 2.0 = 3.4).
3. **Prefill**: the vendor computes through IME2 INT MMA tiles on repacked
   weights (their pp512 Q4_K_M 326 tok/s ~= 395 GFLOP/s equivalent), we use
   row-dot RVV GEMM (~18 GFLOP/s aggregate). This is a kernel-class gap.
4. **A100 placement alone does not fix decode** (measured, section 5): it is
   bandwidth-bound on the same LPDDR5.

## 5. Spine/A100 dispatch experiment (done, documented)

`zig/kernels/riscv/pool.c` gained a SpacemiT spine mode (TF_RISCV_SPERT=1):
the sharded projections dispatch to the A100 cores through libspert.so.1's
stable C ABI (spine_require_stream / spine_parallel_dispatch_1d /
spine_grid), reimplemented from SpacemiT's own MIT-licensed adapter
(spacemit-llama.cpp, spacemit_spert_adapter.cpp) with attribution. Findings
on the machine:

- The mechanism works: a probe shows tile i executing on CPU 8+i (VLEN=1024),
  and our kernels run there unchanged (they are VLEN-agnostic).
- Per-shape on A100: large GEMM is 1.7x faster than the X100 pool (22.1 vs
  12.8 GFLOP/s), the lm_head GEMV 1.3x (19.8 vs 15.5 GB/s); small shapes are
  equal. But `spine_parallel_dispatch_1d` costs ~0.6-1.2 ms per call
  (arbiter round trip; holding the stream does not remove it), and a decode
  token makes ~197 projection calls, so end-to-end decode got slower
  (4.7 vs 11.7 tok/s for q8_0).
- Conclusion: A100 dispatch pays only when calls are fused. The follow-up is
  a fused per-layer projection task (q/k/v share one dispatch, gate/up one:
  8 -> 4 dispatches per layer) or one fused task per layer block. Decode
  would still be bandwidth-bound at ~the same rate; prefill/lm_head are
  where A100+fusion wins.
- What actually powers the vendor decode is their INT4/INT8 repacked kernels
  plus TCM/HPAGE streams, not the core placement (their BF16 path on the
  same A100 cores is 20x slower than their Q4_K_M).

## 6. Path to the goal (measured plan, each step has a target number)

1. **Q4-class quantization with vectorized dequant** (ours is q8_0 today;
   their Q4_K_M is 1.7x fewer bytes): target >= 15 tok/s decode closing the
   bytes gap at our bandwidth.
2. **Vendor-kernel bridge**: the MIT-licensed spacemit sources
   (ggml/src/ggml-cpu/spacemit/{ime2_kernels.cpp,repack.cpp,rvv_kernels.cpp})
   contain the repacked INT GEMM and IME2 asm. Compiling them behind our
   kernel seam (the libime2bridge.so approach) plus the spine dispatch is
   the direct route to the vendor's compute class; license-compatible
   (MIT into Apache-2.0 with attribution).
3. **Fused per-layer dispatch + A100** for prefill (section 5): target
   prefill 40-60 tok/s from the 1.7x GEMM gain with 4x fewer dispatches.
4. **Bandwidth efficiency** (shape mix, prefetch, serial work): target the
   15 GB/s end-to-end the vendor sustains.

## 7. A100 / IME2 access notes

Normal processes cannot bind to CPUs 8-15 (sched_setaffinity/pthread return
EINVAL, Cpus_allowed_list 0-7; the kernel's AI-domain policy). The spine
runtime (libspert.so.1) is the sanctioned route and it works (section 5);
IME2 instructions are only reachable through SpacemiT's kernels, which the
public MIT sources provide. qemu gates keep VLEN=1024 and the IME2-family
tiling regression-tested in CI regardless.
