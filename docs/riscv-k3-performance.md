# K3 prefill against SpacemiT's llama.cpp: the 80 % comparison

Reports 2 and 3 of the K3 A100 port (report 1 is the
[reproduction guide](riscv-k3-ime2-port.md)). This document compares
TensorFold's q4x path (A100 tiles + IME2 `vmadot` + TCM + HPAGE) against
SpacemiT's own llama.cpp on the same Qwen3-0.6B checkpoint and the same
board. Every vendor number in it was **re-measured this session** with the
board's `llama.cpp-tools-spacemit` (build `51c81f90f`), and every TensorFold
number is the median of three runs with the receipts in
`~/receipts/k3-a100-ime2/round3/` on the machine.

## 1. The comparison

The vendor publishes a four-thread "user method" pair and an eight-thread
serving pair. TensorFold runs 8 A100 tiles, so the `-t 8` rows are the
resource-matched baseline; the `-t 4` rows are shown because they are the
review's headline numbers.

| shape | vendor `-t 4` | vendor `-t 8` | TensorFold (8 tiles) | of `-t 8` | of `-t 4` |
| --- | --- | --- | --- | --- | --- |
| pp64 | 215.1 | 367.2 | **247.9** | 67.5 % | **115.3 %** |
| tg64 | 37.8 | 41.5 | **36.6** | 88.2 % | **96.9 %** |
| pp512 | — | 329.8 | **253.6** | 76.9 % | — |
| tg128 | — | 40.9 | **25.5** | 62.4 % | — |

All figures are tok/s, Qwen3-0.6B, Q4_K_M (vendor) vs q4x 4.06 bits/weight
(TensorFold), `pp` = prefill tokens/s, `tg` = decode tokens/s after that
prefill - the llama-bench method on both sides.

### Gate status

The project set two gates: 80 % of the vendor SOTA and 110 %. Per metric,
against the resource-matched `-t 8` baseline:

| metric | 80 % gate | 110 % gate | TensorFold | status |
| --- | --- | --- | --- | --- |
| tg64 | 33.2 | 45.7 | 36.6 | **80 % met (88 %)** |
| pp512 | 263.8 | 362.8 | 253.6 | 77 % - 4 % short of the 80 % gate |
| pp64 | 293.8 | 403.9 | 247.9 | 68 % - and 115 % of the `-t 4` number |
| tg128 | 32.7 | 45.0 | 25.5 | 62 % - the long-context decode gap |

**Neither gate is met across the board.** What is met: tg64 at 88 % of the
resource-matched SOTA (and 97 % of the review's headline number), pp64 at
115 % of the headline number. What is not: pp512 (77 %) and tg128 (62 %).
Report 3 (the 110 % report) is therefore **not due**; section 4 carries its
measured plan instead.

## 2. What changed to get here

Round 2 ended at pp512 57.2 / pp64 39.0 / tg64 28.7 / tg128 20.0 tok/s.
Round 3 moved prefill 4.4-6.4x. The phase profile (a cumulative per-phase
timer in `Session.forward`, `TF_RISCV_PROFILE=1`) named the costs; each fix
below has its measured effect on the board:

| cost (pp512) | round 2 | round 3 | what moved it |
| --- | --- | --- | --- |
| projection GEMMs | 3.0 s | 0.87 s | the GEMM pass stages its activation in TCM once per pass (was: every weight block re-streamed 56 KiB through a shared L2 - 81 us per (pass, block) on the tiles vs 4.7 us on an X100 core), the output block goes through TCM and blits once per pass (the direct scatter was 29 % of the kernel), the scale-vector builds stop compiling into 64-lane gathers (40 %+38 % of cycles), q/k/v and gate/up fuse into one dispatch per set |
| attention | 3.6 s | 0.6 s | block kernel: transposed score product (one vector of 32 query dots per key row, no per-pair reductions), key-major scores, row-wise vector exp (the libm `ldexpf` path was 105 ns/call = 1.5 s), value mix with outputs in registers and 4 query rows per V load |
| activation quant | 0.57 s | 0.11 s | q/k/v and gate/up quantize their shared activation once (was once per projection); the quantizer's per-chunk work is 4-way unrolled |
| silu + rmsnorm + rope | 1.2 s | 0.45 s | vector silu (44 M per-element exps were 1.8 s), rope table built once per token with a vector polynomial |
| kv cache writes | 0.1 s | 0.09 s | hoisted panel pointers + a vector copy (the runtime `@memcpy` charged 826 ns per 256-byte copy) |

The single kernel-level result: the gate projection's GEMM
(m=512, n=3072, k=1024) went from 35.9 ms to 4.0 ms (45 -> 397 GMAC/s) over
the session - TCM staging, then the two gather fixes, then the output block.

Two accuracy fixes came out of the same work and are in report 1's failure
list: the GEMM fold applied one activation scale per 8-row block instead of
per row, and the block attention's mix loop assumed the A100's register
width (the qemu VLEN=256 gate caught the second).

## 3. Where the remaining gap lives

### 3.1 pp512: 253.6 vs 329.8 (77 %)

Per token the prefill costs 3.94 ms (253.6 tok/s) of which the phase profile
puts 0.87 s of the 2.0 s run in the GEMMs, 0.6 s in attention and 0.45 s in
norms/rope/silu. The GEMM is now balanced (perf annotate shows vmadots at
~25 % of cycles, the fold at 20 %, the stores at 8 %, no dominant stall),
but it runs at roughly 10-20 ns of exposed cost per vector op - a
load-to-use bound on the spine-scheduled tiles that the vendor's kernels
avoid by carrying their accumulators in int32 across whole k blocks and by
never spilling a per-group epilogue into the tile's memory.

The two named gaps to close for the 80 % gate (263.2):

1. **the GEMM's per-group fold and per-tile prologue** (measured 25 %+ ~30 %
   of the kernel): fold the scales once per pass instead of per 256-k group
   (the group scale is constant across the 8 k-blocks of a group, so the
   fold can move out of the tile loop entirely), and process two weight
   blocks per asm body so the dequant and the A loads amortize across them
   (2.5 -> ~1.8 instructions per vmadot in the inner loop). Target: GEMM
   0.87 -> 0.6 s, pp512 -> ~275 tok/s (83 %).
2. **the tail rows**: m=512 runs 9 full passes plus one 8-row pass through
   the 7-block loop, i.e. 8.6 % of the GEMM's vmadots and A loads compute
   zero-padded rows. A 1-block asm body for the last pass (2 vmadots per
   tile instead of 14) removes it; measured cost today is ~4 % of the GEMM.

### 3.2 tg128: 25.5 vs 40.9 (62 %)

Decode at 512-token context runs 40 ms/token of which the projections are
~23 ms (at the same rate as tg64) and attention grows to ~6 ms with the
context (512 dots + exps + mix per head per layer). The vendor's decode is a
fused per-layer spine call; ours is 140 dispatches per token. The measured
path: the attention score GEMM for the single query row can use the same
transposed product with the key row held in registers (nb=1 variant of the
block kernel), and the per-layer dispatches can collapse to one fused
layer call behind a `spine_ctx_sync` barrier (917 ns measured).

### 3.3 pp64: 247.9 vs 367.2 at -t 8 (68 %) but 115 % of -t 4

The `-t 8` vendor number scales prefill with eight threads on the A100 set;
our 8 tiles stream the same weights but pay the per-tile startup for
much shorter passes (11 passes at 56 rows vs their per-thread row ranges).
Closing this one is the same GEMM work as 3.1 plus a wider m-pass (56 ->
112 rows per pass halves the weight re-reads per projection).

## 4. The 110 % plan (report 3's trigger)

Report 3 is written when every axis passes 110 % (pp512 362.8, tg128 45.0,
pp64 403.9, tg64 45.7 at the `-t 8` baseline);
[its status file](riscv-k3-performance-110.md) tracks the distance. The
measured building blocks:

1. Section 3.1's fold/prologue work: pp512 ~275 (83 %).
2. A wider m-pass (112 rows) with the weights read once per pass instead of
   per 56 rows: prefill's weight traffic drops 2x, pp512 ~310 (94 %).
3. INT4 activations through `vmadotu`'s i4 mode for the score GEMM and the
   outer k steps (the vendor's own pattern: their A operand is split into
   nibbles and the i4-mode dot covers 2x the k per instruction) - this is
   the step that can double the GEMM again, and it needs its own accuracy
   study before it lands.
4. The decode-side fusion of section 3.2: tg128 ~35 (86 %), and the lm_head
   (78 MiB of the 298 MiB per token) queued behind the last layer's
   projections for the rest.

## 5. Reproducing these numbers

Vendor (on the board):

```sh
cd ~/models/gguf
llama-bench -m Qwen3-0.6B-Q4_K_M.gguf -p 64  -n 64  -t 4
llama-bench -m Qwen3-0.6B-Q4_K_M.gguf -p 64  -n 64  -t 8
llama-bench -m Qwen3-0.6B-Q4_K_M.gguf -p 512 -n 128 -t 8
```

TensorFold:

```sh
cd ~/TensorFold   # or scp zig-out/bin/tensorfold-riscv after
                  # zig build riscv-cli -Dtarget=riscv64-linux-gnu
./zig-out/bin/tensorfold-riscv bench ~/models/qwen3-06b --quant q4x \
  --prefill 512 --decode 128 --context 1024 --pool a100 --mem hpage
./zig-out/bin/tensorfold-riscv bench ~/models/qwen3-06b --quant q4x \
  --prefill 64  --decode 64  --context 256  --pool a100 --mem hpage
```

Receipts: `~/receipts/k3-a100-ime2/round3/pp512-{1,2,3}.json` and
`pp64-{1,2,3}.json`; the vendor's rows are in
`~/receipts/k3-a100-ime2/round3/vendor-t4.txt` and `vendor-t8.txt`.
