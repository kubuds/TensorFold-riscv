# K3 performance against SpacemiT's llama.cpp: where the port stands

Reports 2 and 3 of the K3 A100 port (report 1 is the
[reproduction guide](riscv-k3-ime2-port.md)). The baseline of record is
SpacemiT's own llama.cpp: `llama.cpp-tools-spacemit 0.2.2` from
archive.spacemit.com, device `SPACEMIT`, `use_ime2=1`, compute on the A100 set
(`cpu_mask: ff00`), `mem_backend: HPAGE`, TCM available. Both engines run the
same Qwen3-0.6B checkpoint on the same machine (`k3pico16g.local`).

## 1. Where the port stands (measured)

Decode `tg` and prefill `pp`, tokens per second, Qwen3-0.6B:

| engine | quant | placement | pp64 (t4) | tg64 (t4) | pp512 (t8) | tg128 (t8) |
| --- | --- | --- | --- | --- | --- | --- |
| **vendor llama.cpp** | Q4_K_M | A100 + IME2 + TCM/HPAGE | **214.9** | **37.3** | **326.5** | **40.9** |
| vendor llama.cpp | Q8_0 | A100 + IME2 | 200.1 | 30.7 | 302.2 | 32.0 |
| vendor llama.cpp | BF16 | A100 + IME2 | 1.4 | 1.0 | 2.7 | 1.9 |
| **TensorFold q4x** | 4-bit IME2 tiles | A100 tiles + IME2 + TCM + THP | 31.1 | **24.6** | 42.3 | **18.3** |
| TensorFold q4x, portable float | 4-bit IME2 tiles | A100 tiles, no vmadot | 6.8 | 4.7 | — | — |
| TensorFold q4x, portable float | 4-bit IME2 tiles | X100, 8 threads | — | — | 7.6 | 3.6 |
| TensorFold q8_0 (previous round) | 8-bit blocks | X100, RVV | — | — | 15.2 | 11.7 |

Against the goal:

| metric | SOTA (vendor) | 80 % gate | 110 % gate | TensorFold now | share of SOTA |
| --- | --- | --- | --- | --- | --- |
| tg64, user method (-t 4) | 37.3 | 29.8 | 41.0 | 24.6 | 66 % |
| tg128, serving method (-t 8) | 40.9 | 32.7 | 45.0 | 18.3 | 45 % |
| pp64 | 214.9 | 171.9 | 236.4 | 31.1 | 14 % |
| pp512 | 326.5 | 261.2 | 359.2 | 42.3 | 13 % |

**Neither the 80 % nor the 110 % gate is met yet.** What follows is the
measured decomposition of the gap and the ordered plan, not a claim.

Two honest wins in the same run:

- the decode kernel (a single projection) now streams **15.4 GB/s** on the
  `lm_head` shape (151936×1024) and 14.4 GB/s on 3072×1024, against an
  effective ~16 GB/s for the vendor's whole decode — i.e. **the kernel is at
  ~95 % of the vendor's streaming rate**, and the remaining decode gap is
  outside the weight stream;
- the previous round's best decode was 11.7 tok/s (q8_0 on the X100); the
  IME2 path is **2.1× that**, and 5.2× the same 4-bit model on the portable
  float path.

## 2. Performance against the vendor baseline

### 2.1 Decode: kernel is close, orchestration is not

Per token the model streams 298 MiB at q4x (4.06 bits per weight plus the
8-bit embedding table). At the measured 14.4 GB/s kernel rate that is 21.7 ms,
i.e. **46 tok/s** from the kernels alone. Measured end to end: 24.6 tok/s
(40.6 ms per token at a 64-token context), 18.3 tok/s at a 512-token context.
The 19 ms difference is orchestration, and the pieces are measurable:

| cost per token | measured / estimated | source |
| --- | --- | --- |
| weight stream (kernels) | 21.7 ms | 298 MiB ÷ 14.4 GB/s |
| A100 dispatch, 196 projection calls | 2.6 ms | `probe_k3 dispatch`: 13.4 µs per 8-tile call |
| attention (X100 RVV, 28 layers × 16 heads) | 1.5–6 ms | grows with context (the tg128 run is 6 ms/token slower than tg64) |
| `quant_a`, 196 calls | 0.2 ms | scalar quantizer (report 1 §4.6) |
| norms, RoPE, SwiGLU, KV writes | ~1 ms | |

That is 27–32 ms of 40.6 ms accounted for; the remainder is per-call fixed
cost inside the GEMV for small shards (each of the 8 tiles runs a
`spine_thread_tcm_malloc`-backed shard, and the 1024-row projections split into
16 eight-row blocks per tile, where the loop's prologue matters).

The vendor does the same work in 24.4 ms (40.9 tok/s, 372 MiB of Q4_K_M). Its
edge is that its kernels and its graph are fused into one spine dispatch per
layer while this round still dispatches per projection, and that it reads 20 %
fewer bytes (4.85 vs 4.06 bits costs us *less*, so the byte count is not its
advantage — its advantage is that it does not pay 196 dispatch round trips nor
a separate attention pass on the X100).

### 2.2 Prefill: the GEMM is the gap

| projection shape | measured | share of vmadot peak |
| --- | --- | --- |
| m=512, n=3072, k=1024 (gate) | 45.2 GMAC/s | 0.6 % of 7 TMAC/s |
| m=56, n=3072, k=1024 | 56.8 GMAC/s | 0.8 % |

The vendor's `pp512` (326.5 tok/s) is ~390 GFLOP/s ≈ 195 GMAC/s, i.e. 4.3× the
port's GEMM. pp512 for this model is 305 GMAC total, so 42.3 tok/s means the
port spends 12.1 s where the vendor spends 1.57 s.

The GEMM is not limited by the tensor unit (7 TMAC/s measured), nor by DRAM
(the shard's weight stream is under 1 GB/s effective because a pass re-reads
its weight block from L1/L2 across 9 m-passes). It is limited by per-tile
overhead: each 128-byte weight tile carries 14 vmadots (14336 MACs) but the
loop around it pays for 7 A-operand loads, the per-group epilogue (strided
int32 read-back, 64-lane scale folding) and the per-pass store. Measured
2.6 µs per weight tile against the ~0.6 µs the same tile costs in the decode
loop, which does one vmadot per 128 bytes and no M-block epilogue.

### 2.3 What the port fixed relative to the previous round

| | previous round | now |
| --- | --- | --- |
| decode (tg128) | 11.7 tok/s (q8_0, X100 RVV) | 18.3 tok/s (q4x + IME2 + TCM + THP) |
| bytes per token | 0.64 GiB (q8_0) | 0.30 GiB (q4x) |
| A100 dispatch | 4.7 tok/s (spine, per-call stream) | 24.6 tok/s (stream held, TCM scratch, vmadot) |
| IME2 | seam with reference math only | real `smt.vmadot` kernels, 7 TMAC/s validated |
| memory | 4 KiB pages | `MADV_HUGEPAGE` arena, 26.6-30.0 → 40.5-51.1 GB/s (probe) |

## 3. Path to 80 % and 110 %, with measured targets

Ordered by measured payoff, each with the number it has to move. The 80 %
gates are tg128 ≥ 32.7 and pp512 ≥ 261; the 110 % gates are ≥ 45.0 and ≥ 359.

### Step 1 — fuse the per-layer projection set into one dispatch (decode)

The port makes 7 dispatches per layer (q, k, v, o, gate, up, down) plus a
shard call for the head. Every dispatch costs 13.4 µs (measured) and every
shard pays a prologue. Fusing q/k/v and gate/up into single calls (they share
the activation) and batching the layer's elementwise ops into the same call
removes 3 of 7 dispatches per layer (196 → 112 calls per token, −1.1 ms) and
amortizes the prologue.

- measured target: **+8–12 % decode** (24.6 → 27 tok/s at 64 tokens).
- next lever if that is not enough: one dispatch per layer with a grid-wide
  `spine_ctx_sync` barrier (917 ns measured) between stages, 28 calls/token.

### Step 2 — move attention to the A100 tiles and keep the KV cache in TCM

Attention is 1.5 ms at 64 tokens but 6 ms at 512 tokens, and it runs on the
X100 while the A100 tiles idle. The A100 has 1024-bit vectors and 393 KiB of
TCM per tile; a head's 512-row KV panel is 512×128×4 = 256 KiB for K and the
same for V, so half a tile's TCM holds one head's K at fp32, or a whole head's
K+V at bf16.

- measured target: **−4.5 ms per token at 512 tokens**, i.e. tg128
  18.3 → 22 tok/s.

### Step 3 — fix the prefill GEMM's per-tile epilogue (prefill)

Three changes, each measurable on the m=512 gate shape (45.2 GMAC/s today):

1. keep the f32 accumulators in registers instead of a stack array and fold
   the group scale once per pass instead of once per group (the 64-lane scale
   vector is currently rebuilt with 64 scalar stores per block per group);
2. write the output tiles with one contiguous store per 8×8 block into a
   tiled output buffer and transpose once per projection (the strided
   `vsse32` per column is 8 scattered stores);
3. hoist the A-operand pointer arithmetic out of the loop (12 runtime
   `k/16`-scaled offsets are recomputed per tile).

- measured target: the decode loop proves the tile path can carry 14336 MACs
  per 128 bytes at 0.6 µs; closing to 2× that would put the GEMM at
  ~150 GMAC/s → **pp512 ≈ 140 tok/s**; reaching the same per-tile efficiency
  as decode gives ~300 GMAC/s → **pp512 ≈ 280 tok/s** (the 80 % gate).

### Step 4 — 110 %: fewer bytes and a wider m-pass

With the GEMM at the decode loop's efficiency the tensor unit still has 40×
headroom, so the 110 % prefill target needs the weight stream, not the MACs:
widen the m-pass from 56 to 168 rows (register-pressure trade: 7 → 3
accumulator-pair groups in TCM), which cuts the weight re-reads per
projection from 9 to 3, and keep q4x at 4.06 bits per weight against the
vendor's 4.85. The measured streaming ceiling is 40.5-51.1 GB/s on the A100 tiles
(probe, two runs), so pp512 is DRAM-bound at 0.30 GiB × (512/168) = 0.9 GiB → 22 ms,
while the MACs take 305 GMAC / 7 TMAC/s = 44 ms: **pp512 ≈ 380 tok/s**, over
the 110 % gate (359).

For decode the 110 % gate (45 tok/s) needs 22.2 ms per token, which is the
weight stream at 14.4 GB/s with *all* orchestration inside 0.5 ms. The port's
kernel is at 95 % of the vendor's rate and 4.06 bits beats its 4.85, so the
gate is reachable by Steps 1 and 2 plus one more: **queue the head's GEMV
behind the last layer's projections** (the head is 78 MiB of the 298 MiB per
token and does not depend on the KV cache), which overlaps 6.2 ms of the
stream with the attention of the final layer.

## 4. Reproducing these numbers

See [report 1 §3](riscv-k3-ime2-port.md#3-reproducing-a-measurement) for the
commands, the receipt fields and the vendor baseline files
(`~/receipts/llama-vendor-bench.txt`, `llama-source-bench.txt`,
`tfv4-*.json`, `probe_k3-*.txt` on the machine).
