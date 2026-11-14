#!/bin/sh
# parity.sh - the numeric gates for tensorfold-riscv: the numpy oracle over
# the kernel path and over the OpenBLAS path (a stub cblas_sgemm proves the
# code path; the real library is measured on K3).
# Usage: tools/riscv/parity.sh [BIN]   (default zig-out/bin/tensorfold-riscv;
# BIN must run on this machine - rebuild with -Dtarget for K3 before measuring)
set -eu
BIN="${1:-zig-out/bin/tensorfold-riscv}"
HERE="$(dirname "$0")"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
python3 tools/riscv/qwen3_parity.py --bin "$BIN" --work .zig-cache/riscv-parity
STUB=.zig-cache/libtfblas_stub.so
zig cc -shared -fPIC -O2 tools/riscv/openblas_stub.c -o "$STUB"
TF_RISCV_BLAS="$ROOT/$STUB" python3 tools/riscv/qwen3_parity.py --bin "$BIN" --work .zig-cache/riscv-parity-blas
echo "parity: kernel path and OpenBLAS path both OK"
