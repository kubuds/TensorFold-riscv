# K3 A100 + IME2 + TCM/HPAGE port: reproduction report

This is the first of three reports for the SpacemiT K3 A100 port. It covers
what was built, how to reproduce every gate on the machine, and the receipts.
The performance comparison against SpacemiT's own llama.cpp is
[report 2](riscv-k3-performance.md#2-performance-against-the-vendor-baseline)
(state at the 64 %/14 % mark) and
[report 3](riscv-k3-performance.md#3-path-to-80-and-110-with-measured-targets)
(the measured path to 80 % and 110 %).

Machine of record: `k3pico16g.local`, Bianbu 4.0.8, Linux 6.18.3, 16 GB
LPDDR5, 8×X100 (VLEN=256) + 8×A100 (VLEN=1024), `spacemit-runtime 0.6.3`
(`/usr/lib/libspert.so.1`), `spacemit-tcm 3.0.1`, THP `[madvise]`.

## 1. What the port adds

The backend keeps its portable RVV family (X100 and any RVV 1.0 CPU) and gains
a K3-specific path:

| piece | where | what it does |
| --- | --- | --- |
| A100 dispatch | `zig/kernels/riscv/pool.c` | `--pool a100` hands the shard callbacks to eight A100 tiles through the spine runtime (`spine_require_stream` once per process, `spine_parallel_dispatch_1d` per call). The ABI is SpacemiT's published `spert_abi.h` (MIT), the runtime is `dlopen`ed like OpenBLAS. |
| IME2 dot kernels | `zig/kernels/riscv/q4x_ime2.c` | `smt.vmadot` (8×8×16 int8 → int32, 1024 MACs per instruction) for the decode GEMV and the prefill GEMM. The instruction encoding is emitted as `.4byte` words, so no SpacemiT assembler fork is needed. |
| TCM | `zig/kernels/riscv/pool.c` (`tf_riscv_scratch`) | Each A100 tile takes up to 320 KiB of its worker-local TCM via `spine_thread_tcm_malloc` and carves kernel scratch from it. The decode kernel stages the quantized activation row there (measured 12.7 → 15.4 GB/s). |
| HPAGE | `zig/kernels/riscv/mem.c`, `zig/src/riscv/hpage.zig` | The model arena is anonymous memory mapped 2 MiB-aligned with `MADV_HUGEPAGE` (the vendor's `mem_backend=HPAGE`), `--mem hpage\|plain`. |
| q4x format | `zig/src/riscv/qwen3.zig`, `zig/kernels/riscv/q4x.c` | 4-bit weights in vmadot tiles: 8 output rows × 32 k per 128 bytes, `(k/256)` groups × 8 rows of f16 scales per block, 4.0625 bits per weight. Activations are int8 per 256-value group (the vmadot A operand). |
| CLI | `zig/src/riscv/main.zig` | `--quant q4x`, `--pool auto\|x100\|a100`, `--mem hpage\|plain`, `--tcm-kb N`; `bench`/`run` reports carry `pool`, `hpage_arena_bytes`, `tcm_peak_bytes`. |

Env switches for experiments: `TF_RISCV_Q4X=scalar|float|ime2` (kernel family),
`TF_RISCV_A100=1` (placement without the flag), `TF_RISCV_TCM=0`,
`TF_RISCV_TCM_KB`, `TF_RISCV_TCM_CACHE=0`.

### 1.1 The IME2 instruction contract, as measured

`tools/riscv/probe_k3.c` answers the questions the port depends on. On the K3:

```
probe_k3 vmadot: spert cores 8 vlen 128 bytes tcm 393216 bytes
vmadot 8x8x16 check: maxdiff=0 bad=0/64
vmadot perf nacc=7: 0.403 ms -> 7116.5 GMAC/s aggregate, 1.15 ns/vmadot/tile
```

That is `vd[m*8+n] += sum_k A[m*16+k] * B[n*16+k]` with 128-byte operands and a
64×int32 accumulator pair (`vd` 4-register aligned, encoding carries `vd/2`),
1024 MACs per instruction at 7 TMAC/s over the eight tiles. X100 traps on it
(IME1 only), so the float family is the fallback and the dispatch checks the
architecture's VLEN (`vlenb == 128`).

The same probe measured the memory system the port has to feed:

```
x100 pthread read 8 worker(s) 4k  :   9.20 GB/s     (second run: 11.67)
x100 pthread read 8 worker(s) huge:   9.68 GB/s     (second run: 12.37)
a100 spine read 8 tiles 4k        :  30.02 GB/s     (second run: 26.58)
a100 spine read 8 tiles huge      :  40.53 GB/s     (second run: 51.05)
spine empty 8-tile dispatch: sync 13.36 us, async+sync 9.75 us  (second run: 12.83 / 9.72)
```

Both runs are in `~/receipts/k3-a100-ime2/probe-mem.txt` and
`probe-dispatch.txt`; the spread between them is board-level variance (the
machine is shared), so the port quotes ranges rather than single values.

Two facts drive the design: the A100 tiles stream 4.2× the X100 cores, and THP
is worth 35 % of that (30.0 → 40.5 GB/s) — which is what `--mem hpage` buys.

## 2. Correctness gates (all reproducible on the machine)

| gate | command | result |
| --- | --- | --- |
| kernel + model tests (host) | `zig build riscv-test` | 16/16 |
| RVV at VLEN=256 and VLEN=1024 | `zig build riscv-qemu-test -Dqemu=…/qemu-riscv64` | 16/16 each (qemu runs `TF_RISCV_Q4X=float`; it has no vmadot) |
| deployment ABIs | `zig build riscv-cross` | glibc and musl, scalar and RVV |
| numpy oracle (f32, q8_0, q4_0) | `python3 tools/riscv/qwen3_parity.py --bin … --quant none\|q8_0\|q4_0` | logits within 1.4e-2 and greedy tokens equal |
| numpy oracle (q4x) | `… --quant q4x --spec q4x` | greedy tokens equal; logits within the int8-activation noise (see §2.1) |
| same tokens on all three q4x families, on the A100 | `TF_RISCV_Q4X=scalar\|float\|ime2 ./tensorfold-riscv run …` | `[52, 82, 65, 42, 55, 80, 82, 65]` for all three |
| vmadot vs scalar vs float kernels | `kchk.c` (in `~/receipts/k3-a100-ime2/`) | `maxdiff 0` for GEMV and GEMM, m ∈ {1, 9}, n ∈ {8, 16}, k ∈ {512, 768} |
| vmadot lane semantics, on hardware | `probe-vmadot.txt` | `vmadot 8x8x16 on A100 tile: maxdiff=0 bad=0/64` |

### 2.1 What the q4x oracle can and cannot pin down

q4x quantizes the activation to int8 per 256-value group (the vmadot A operand)
in addition to 4-bit weights. The numpy oracle models both rules, but it
rounds the group scale and the LSB in float64 while the kernel rounds in f32
through `lrintf`, so its logits carry an independent int8 round trip:
`max|zig - numpy| = 5.1e-1` on a 1.3e+1 scale for the 2-layer synthetic
checkpoint, with identical greedy tokens. The parity gate therefore holds q4x
to `5e-2 · scale` and keeps token equality as the hard check; the scalar,
float and IME2 kernel families agree with each other exactly
(`maxdiff 0`, table above), which is the sharper statement.

## 3. Reproducing a measurement

```sh
# on the K3, with a Hugging Face Qwen3-0.6B directory in ~/models/qwen3-06b
cd ~/TensorFold
zig build riscv-cli -Dtarget=riscv64-linux-gnu        # or scp the binary

# capabilities and placement
./zig-out/bin/tensorfold-riscv caps --pool a100 --threads 8

# decode/prefill, IME2 on the A100 tiles with TCM + THP weights
./zig-out/bin/tensorfold-riscv bench ~/models/qwen3-06b \
  --quant q4x --prefill 64 --decode 64 --context 256 \
  --pool a100 --mem hpage --report receipt.json

# the same workload on the portable path, and the previous best
TF_RISCV_Q4X=float ./zig-out/bin/tensorfold-riscv bench … --pool a100
./zig-out/bin/tensorfold-riscv bench … --quant q8_0 --pool x100 --threads 8
```

`receipt.json` records `pool`, `hpage_arena_bytes`, `tcm_peak_bytes` and the
throughputs, so a report line can be traced to the run that produced it.

Measured receipts (round 1, then round 2, `~/receipts/k3-a100-ime2/` and
`round2/` on the machine):

```json
{"command":"bench","kernels":"rvv","vlen_bits":256,"threads":8,"quant":"q4x",
 "pool":"a100","hpage_arena_bytes":3397386240,"tcm_peak_bytes":24576,
 "prefill_tokens":64,"prefill_tok_s":38.98,"decode_tokens":64,"decode_tok_s":28.72}

{"command":"bench","kernels":"rvv","vlen_bits":256,"threads":8,"quant":"q4x",
 "pool":"a100","hpage_arena_bytes":3397386240,"tcm_peak_bytes":24576,
 "prefill_tokens":512,"prefill_tok_s":57.20,"decode_tokens":128,"decode_tok_s":20.00}
```

## 4. Failures found on the way (kept, because they were real)

1. **`rdcycle` traps inside an A100 tile.** Time per tile with `clock_gettime`,
   not cycle counters (the spine runtime's sandbox leaves `mcounteren` closed
   for A100 tiles).
2. **`vpack.vv` / `vnpack4.vv` trap on the A100.** They are IME1-era
   instructions; the port uses only `vmadot`/`vmadotsu`/`vmadotu` plus RVV 1.0.
3. **`vmadot` requires a 4-register-aligned `vd`** (the assembler rejects
   anything else: `vmadot v3, …` is an error). The encoder stores `vd/2`.
4. **A C++-only header.** `/usr/include/spert_abi.h` includes `<cassert>`, so
   the C port declares the four symbols it uses.
5. **Packing bugs caught by the gates**: the q4x packer double-counted the
   256-group offset for k > 256 (caught by a round-trip test that decodes a
   unit activation), the GEMM's activation blocks were addressed one tile
   apart instead of one block apart, and the prefill tail was recomputed once
   per output block (400 ms → 36 ms on the gate projection once moved out).
6. **The RVV activation quantizer silently differed from the scalar one**;
   the difference was traced to two VLEN-dependent bugs, not rounding:
   - the narrowing chain `i32m1 → i16mf2 → i8mf4` holds only **8 lanes at
     VLEN=256**, so every second 16-k chunk of the quantized activation was
     zero on the X100 (the A100's VLEN=1024 masked the bug: 32 lanes fit);
   - the amax loop stepped by a fixed 64 while `e32m4` holds **32 lanes at
     VLEN=256**, so the group scale sampled half the group.
   Both are fixed with lane-count-correct register groups (`i32m2 → i16m1 →
   i8mf2`) and a `j += vl` stride; the quantizers are now byte-identical.
7. **The rope table's vector polynomial was wrong twice**: the Horner chain
   used `vfmadd_vf` (which adds) where each step must multiply, and the
   `vmerge_vvm` operand order was inverted (probed on hardware: an all-true
   mask returns the *second* operand). Fixed; the table now agrees with libm
   to 3.6e-8 and every gate that had silently passed before actually fails
   when either regression is reintroduced.
8. **The decode fusion shared the q4x path with unquantized weights** and
   clamped shard ranges incorrectly at projection boundaries (`b - base`
   could be negative); both caught by the host oracle, which the K3-only
   gates had missed.

## 5. What is not done yet

- The decode path is at 77 % of the vendor's `tg64` and 49 % of its `tg128`
  (report 2 has the decomposition); the prefill path is at 18 %/17.5 %.
- The prefill GEMM is instruction/latency bound at 45 GMAC/s, far under the
  7 TMAC/s the tensor unit sustains: the per-tile epilogue (strided read-back
  and scale folding) and the A-operand reload dominate. Report 3 lists the
  ordered fixes and the measured target of each.
- HPAGE is applied and THP-eligible (`hpage_arena_bytes` is the arena the
  loader mapped); `hpage_backed_bytes` samples `AnonHugePages` of the first
  mapping and read 0 at report time, so the arena's backing is still being
  filled by khugepaged. The kernel effect of THP is measured with the probe
  on a 256 MiB buffer: 26.6-30.0 → 40.5-51.1 GB/s.
