# K3 at 110 % of the vendor SOTA: **not reached** - status and trigger

This is the placeholder for report 3 of the K3 A100 port. Per the project's
own rule it is written as the performance report **only after** every axis
passes 110 % of the SpacemiT llama.cpp baseline; today it records how far
the port is from that and what is measured to close it. Nothing in this
file claims an achievement that was not measured.

Baseline (re-measured this session, `llama.cpp-tools-spacemit` build
`51c81f90f`, receipts `~/receipts/k3-a100-ime2/round3/vendor-t{4,8}.txt`),
resource-matched at `-t 8` against TensorFold's 8 A100 tiles:

| metric | vendor `-t 8` | 110 % trigger | TensorFold now | of SOTA |
| --- | --- | --- | --- | --- |
| pp64 | 367.2 | 403.9 | 247.9 | 67.5 % |
| tg64 | 41.5 | 45.7 | 36.6 | 88.2 % |
| pp512 | 329.8 | 362.8 | 253.6 | 76.9 % |
| tg128 | 40.9 | 45.0 | 25.5 | 62.4 % |

(For reference against the review's four-thread "user method" numbers,
pp64 is at 115 % and tg64 at 97 %; that is not the 110 % trigger.)

## What is measured to close it

Each step carries the number it has to move, from
[the 80 % report](riscv-k3-performance.md#3-where-the-remaining-gap-lives):

1. GEMM fold moved out of the per-group loop + two weight blocks per asm
   body (2.5 -> 1.8 instructions per vmadot): GEMM 0.87 -> ~0.6 s of pp512,
   **pp512 ~275 (83 %)**.
2. Wider m-pass (56 -> 112 rows) so each projection's weights stream once
   per pass instead of twice: **pp512 ~310 (94 %)**.
3. INT4 activations through `vmadotu`'s i4 mode (the vendor's own A-operand
   trick: split int8 into nibbles, 2x the k per dot instruction) for the
   score GEMM and the outer k steps. This is the step that can double the
   GEMM again - and it needs its own accuracy study (attention scores feed
   softmax) before it lands behind a flag.
4. Decode fusion: one spine call per layer behind a `spine_ctx_sync`
   barrier (917 ns measured) and the block-attention kernel's nb=1
   variant for the single query row: **tg128 ~35 (86 %)**.
5. The lm_head (78 MiB of the 298 MiB per token) queued behind the last
   layer's projections: **tg128 ~38-40**, which is where 110 % for decode
   (44.9) becomes a question of the head's own GEMV rate (measured
   15.4 GB/s against a 40.5-51.1 GB/s streaming ceiling).

When all four rows cross their trigger, this file becomes report 3: the
same measurement method as [report 2](riscv-k3-performance.md#5-reproducing-these-numbers),
three runs per shape, receipts in `~/receipts/k3-a100-ime2/`.
