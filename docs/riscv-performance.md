# RISC-V backend performance report (SpacemiT K3)

Status: measurement plan and local correctness receipts are complete; the K3
numbers fill in on hardware (see "K3 runbook"). No performance number in this
document is claimed without the machine, command and receipt that produced it.

## 1. Summary

| Item | State |
| --- | --- |
| Correctness (scalar / RVV / IME2 vs float64 oracle) | done, see `zig build riscv-qemu-test` receipts |
| End-to-end Qwen3 vs independent numpy forward | done, max `|logits|` 1.7e-06 (scalar/RVV) and 1.9e-06 (IME2 path), greedy tokens identical |
| Kernel/decode throughput on K3 X100 (VLEN=256) | pending hardware |
| Kernel/decode throughput on K3 A100 (VLEN=1024 + IME2) | pending hardware |
| Comparison vs SpacemiT's native llama.cpp | pending hardware |
| Target | beat the SpacemiT-native llama.cpp on the same model and core set |

## 2. What is measured and how

`tensorfold-riscv bench` is the harness: seeded random token ids (the same
workload on every machine and kernel family), JSON receipts per run:

```sh
# on K3, A100 cores (AI set), RVV + IME2:
tensorfold-riscv bench MODEL --prefill 512 --decode 128 \
    --cpus 8-15 --kernels auto --report bench-a100.json
# on K3, X100 cores, RVV only:
tensorfold-riscv bench MODEL --prefill 512 --decode 128 \
    --cpus 0-7 --kernels rvv --report bench-x100.json
```

The receipt records `prefill_tok_s`, `decode_tok_s`, the kernel family, VLEN
and the BLAS in use. Run each configuration 5 times, keep the median, and
attach the receipts (they are the evidence; this document quotes them).

Why these metrics: decode is a full-model weight stream per token, so
`decode_tok_s` is a read-bandwidth measurement on K3 (the paper's own LLM
analysis: parameters stream from LPDDR5 in thread-sharded sequential reads).
Prefill is compute: `prefill_tok_s` is where IME2 and OpenBLAS matter.

Model set for the comparison (lightweight Qwen3):
- Qwen3-0.6B bf16 safetensors (the runner's native format),
- optionally Qwen3-0.6B Q4_K_M GGUF for llama.cpp parity of memory footprint
  (the runner's quantized loader is q8_0/q4_0; Q4_K is a ledger item).

llama.cpp baseline (built for the same cores, same model):

```sh
cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build -j
numactl --cpunodebind=... ./build/bin/llama-bench -m MODEL.gguf -p 512 -n 128
```

Record llama.cpp's `pp` and `tg` numbers for the same thread counts; compare
like-for-like (same cores, same threads, same prompt/decode lengths).

## 3. K3 runbook (fill-in)

Machine: SpacemiT K3, LPDDR5, Linux (RVA23). Scheduling per the K3 paper:
X100 = CPUs 0-7 (general), A100 = CPUs 8-15 (AI work must pin there with
`--cpus 8-15`).

### 3.1 X100, RVV VLEN=256

| Metric | threads | scalar | rvv | rvv + OpenBLAS | llama.cpp |
| --- | --- | --- | --- | --- | --- |
| prefill tok/s (512) | 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| prefill tok/s (512) | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| decode tok/s (128) | 1 | _TBD_ | _TBD_ | n/a | _TBD_ |
| decode tok/s (128) | 8 | _TBD_ | _TBD_ | n/a | _TBD_ |

### 3.2 A100, RVV VLEN=1024 + IME2

| Metric | threads | rvv | ime2 | ime2 + OpenBLAS | llama.cpp |
| --- | --- | --- | --- | --- | --- |
| prefill tok/s (512) | 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| prefill tok/s (512) | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| decode tok/s (128) | 1 | _TBD_ | _TBD_ | n/a | _TBD_ |
| decode tok/s (128) | 8 | _TBD_ | _TBD_ | n/a | _TBD_ |

(Decode rows stay on the RVV GEMV by design: the weight stream is the
bottleneck and the MMA cannot shorten it. The ime2 column shows the same
decode numbers as rvv unless a future w8a8 path changes that - the table
proves the claim instead of asserting it.)

### 3.3 Receipts to attach

- `tensorfold-riscv caps` on both core sets (kernels, vlen_bits, blas).
- every `bench` JSON receipt, plus the llama.cpp `llama-bench` output.
- `tools/riscv/qwen3_parity.py` run natively on K3 for both families.

## 4. Local verification receipts (this workstation, qemu - correctness only)

qemu-user numbers are not performance data; they are recorded to show the
workloads execute correctly in every configuration that K3 will run.

| Gate | Result |
| --- | --- |
| `zig build riscv-test` (host, scalar) | 12/12 |
| `zig build riscv-qemu-test` rv64gc baseline | 12/12 |
| `zig build riscv-qemu-test` rv64gcv VLEN=256 (rvv + scalar-forced) | 12/12, 12/12 |
| `zig build riscv-qemu-test` rv64gcv VLEN=1024 (rvv + scalar-forced) | 12/12, 12/12 |
| IME2 family (inline MMA) under qemu, VLEN=256 / VLEN=1024 | 12/12 both |
| numpy parity, scalar/RVV at VLEN=256 | logits 1.7e-06 on scale 3.2, tokens match |
| numpy parity, A100 config (VLEN=1024 + vendor IME2 shim, 9-token prefill hitting MMA tiles) | logits 1.9e-06 on scale 4.2, tokens match |
| numpy parity through the OpenBLAS stub | logits 1.5e-06, tokens match |
| `zig build riscv-cross` (gnu/musl x baseline/RVV) | 5/5 compile checks |
| riscv64 vs host binary on the synthetic checkpoint | bit-identical logits |

## 5. Analysis and expectations from the K3 paper

- Decode is read-bandwidth bound: the paper measures ~33 GB/s effective
  A100 read bandwidth (LPDDR5, thread-sharded sequential streams). At
  Qwen3-0.6B bf16 (~1.2 GB of weights) that ceiling is around 27 tok/s per
  bandwidth-saturated socket share regardless of ISA - so decode tuning is
  threading and stream layout, not FLOPs. The runner's GEMV streams weight
  rows once per token; the sharded-GEMV pool (ledger item) is the lever.
- VLEN=1024 (A100) vs 256 (X100): the paper's norm/attention notes (Scale
  broadcast kept in registers, FlashAttention-style accumulation resident in
  2-4 vector registers) favor A100 for attention and norms; the kernels are
  VLEN-agnostic and the tables above will quantify it.
- IME2 MMA helps prefill (compute-bound), not decode (bandwidth-bound). The
  bf16 contract rounding of the MMA A operand is bounded and tested
  (kernel tests give the family its documented tolerance).
- OpenBLAS: expected win on prefill GEMM (their riscv64 kernels and
  threading), neutral on decode. Measured in the tables, not assumed.

## 6. Optimization ledger (each item needs a receipt when it lands)

1. Sharded-GEMV thread pool (kernel decode path; OpenBLAS covers GEMM only).
2. Prefill k-outer bf16 GEMM tiling (or bf16->f32 weight cache for OpenBLAS).
3. Polynomial RVV exp/sin/cos (softmax, SwiGLU, RoPE currently call libm).
4. w8a8 quantized GEMM through IME2 mma_i8 (q8_0/q4_0 tiles), plus Q4_K.
5. Cache-panel prefetch hints on the decode weight stream.
6. Engine-adapter integration (engine_api/lanes) for `tensorfold serve`.
