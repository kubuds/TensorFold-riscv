# RISC-V CPU backend (SpacemiT K3)

TensorFold's RISC-V CPU backend serves Qwen3 checkpoints from Hugging Face
safetensors through `tensorfold-riscv`, a self-contained runner built on the
backend's kernel ABI (`zig/kernels/riscv/tf_riscv.h`). Target hardware is the
SpacemiT K3: the X100 cores (RV64GC + RVV 1.0, VLEN=256) and the A100 AI-CPU
cores (RVV 1.0, VLEN=1024, SpacemiT-IME 1.0/2.0 tensor MMA). One binary serves
all of it; the kernel family is chosen at runtime.

Status: correctness-complete for dense Qwen3 (bf16/f32/f16 checkpoints), with
scalar, RVV and IME2 kernel families and an independent numpy oracle. Serving
through the engine API (lanes, server) is the follow-up integration step; see
the ledger at the end.

## Layout

| Path | Role |
| --- | --- |
| `zig/kernels/riscv/tf_riscv.h` | kernel ABI: RMSNorm, RoPE (HF Qwen3 rotate_half), softmax, SwiGLU, add, f32/bf16 matmul, one-row attention, GGUF q8_0/q4_0 GEMV |
| `zig/kernels/riscv/ops_scalar.c` | RV64GC baseline family (portable C, the oracle for the others) |
| `zig/kernels/riscv/ops_rvv.c` | RVV 1.0 family, VLEN-agnostic (256 and 1024 tested) |
| `zig/kernels/riscv/ime2.c` + `ime2_abi.h` | IME2 family (K3 A100 MMA tiling) and the vendor-runtime seam |
| `zig/kernels/riscv/dispatch.c` | family selection (TF_RISCV_KERNELS), caps/VLEN, CPU pinning |
| `zig/src/riscv/` | Zig bindings, f64 reference, Qwen3 loader/forward, CLI, tests |
| `tools/riscv/` | toolchain bootstrap, numpy oracle + parity gates, IME2 shim, OpenBLAS stub |
| `docs/riscv-performance.md` | the performance report and K3 measurement runbook |

## Build and test

```sh
tools/riscv/bootstrap-toolchain.sh tools-local   # Zig 0.17 + qemu-user 10
export PATH="$PWD/tools-local/zig-x86_64-linux-0.17.0:$PATH"

zig build riscv-test        # kernel families + model tests on the host
zig build riscv-qemu-test -Dqemu=tools-local/qemu-user/usr/bin/qemu-riscv64
                            # rv64gc baseline + rv64gcv at VLEN=256 and 1024
zig build riscv-cross       # K3 ABIs: riscv64-linux-gnu and -musl, baseline + RVV
zig build riscv-cli         # host runner; -Dtarget=riscv64-linux-gnu for K3
zig build riscv-parity      # the numpy oracle over the fresh binary
tools/riscv/parity.sh       # kernel path + OpenBLAS path gates
```

Serve and measure:

```sh
tensorfold-riscv caps
tensorfold-riscv run MODEL --prompt "..." --max-tokens 128 --report run.json
tensorfold-riscv bench MODEL --prefill 512 --decode 128 --cpus 8-15 --report bench.json
```

`--cpus LIST` pins the process (K3: `0-7` X100, `8-15` A100); `--threads N`
sets OpenBLAS workers; `--kernels scalar|rvv|ime2` forces a family;
`TF_RISCV_IME2=/path/lib.so` points at the vendor MMA runtime.

## Conventions

- RoPE is HF Qwen3 "rotate_half" over the full head_dim at `rope_theta`;
  q/k per-head RMSNorm weights apply before RoPE; GQA attention scales by
  head_dim^-0.5. The numpy oracle (`tools/riscv/qwen3_parity.py`) is written
  from the same public math and compares logits and greedy tokens.
- Kernel families must agree with the float64 reference within the tested
  tolerances; the IME2 family alone carries bf16-input rounding by contract
  (its MMA multiplies bf16 lanes) and has its documented tolerance.
- Refusals by name: a checkpoint whose tensors mismatch the config is
  rejected with the tensor name and both shapes.

## License notes

The backend is original Apache-2.0 code (TensorFold's license). Optional
runtime dependencies are dlopen'd, never linked or vendored:
- OpenBLAS (BSD-3-Clause) for multithreaded f32 GEMM (`TF_RISCV_BLAS`).
- the SpacemiT IME2 vendor runtime (`TF_RISCV_IME2`, `ime2_abi.h`); the
  reference shim in `tools/riscv/ime2_shim.c` is original code.
The K3 paper informed the MMA shapes and the CPU-affinity rules; no code was
copied from it or any vendor SDK.

## Ledger (honest next steps)

- Sharded-GEMV thread pool for the kernel decode path (OpenBLAS only
  threads GEMM; decode is single-threaded today).
- Prefill GEMM batching for bf16 weights (the m>1 path re-reads weight rows;
  a k-outer tiling or a bf16->f32 weight cache feeding OpenBLAS).
- Polynomial RVV exp/sin/cos (libm per element today; the ledger lives in
  `ops_rvv.c`'s header).
- q8_0/q4_0 w8a8 GEMM through IME2 mma_i8 (quantized GEMV stays RVV).
- Engine-adapter integration (engine_api/lanes) so `tensorfold serve` runs
  on the backend instead of the standalone runner.
