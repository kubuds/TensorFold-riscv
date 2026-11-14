//! blas.zig - optional OpenBLAS acceleration for the RISC-V CPU backend.
//!
//! OpenBLAS (BSD-3-Clause, https://github.com/OpenMathLib/OpenBLAS) is loaded
//! at runtime with dlopen and never linked or vendored, so TensorFold's
//! Apache-2.0 license is unaffected and the binary works without it. It brings
//! multithreaded GEMM tuned per microarchitecture (including SpacemiT's
//! X100/A100 targets in its riscv64 kernels), which speeds the runner's
//! multi-row products: prefill chunks and the lm_head logits row.
//!
//! Selection: TF_RISCV_BLAS=/path/to/libopenblas.so overrides the sonames
//! tried below; TF_RISCV_BLAS=off disables the lookup entirely.
const std = @import("std");

/// CBLAS enums and the subset of the interface the backend calls.
const CblasRowMajor: c_int = 101;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;

const SgemmFn = *const fn (c_int, c_int, c_int, c_int, c_int, c_int, f32, [*]const f32, c_int, [*]const f32, c_int, f32, [*]f32, c_int) callconv(.c) void;
const SetThreadsFn = *const fn (c_int) callconv(.c) void;

var lib: ?std.DynLib = null;
var sgemm_fn: ?SgemmFn = null;
var set_threads_fn: ?SetThreadsFn = null;
var tried: bool = false;

fn lookup() void {
    if (tried) return;
    tried = true;
    const forced = std.c.getenv("TF_RISCV_BLAS");
    if (forced != null) {
        const path = std.mem.span(forced.?);
        if (std.mem.eql(u8, path, "off")) return;
        if (std.DynLib.open(path)) |l| {
            lib = l;
        } else |_| return;
    } else {
        for ([_][:0]const u8{ "libopenblas.so.0", "libopenblas.so", "libblastrack.so" }) |soname| {
            if (std.DynLib.open(soname)) |l| {
                lib = l;
                break;
            } else |_| {}
        }
    }
    var l = lib orelse return;
    sgemm_fn = l.lookup(SgemmFn, "cblas_sgemm");
    set_threads_fn = l.lookup(SetThreadsFn, "openblas_set_num_threads");
}

/// True when an OpenBLAS cblas_sgemm is usable here.
pub fn available() bool {
    lookup();
    return sgemm_fn != null;
}

/// OpenBLAS worker threads (its own default when the library has no setter).
pub fn setThreads(n: c_int) void {
    lookup();
    if (set_threads_fn) |f| f(n);
}

/// out[m,n] = a[m,k] @ b[n,k]^T in row-major f32 (the backend's project form).
/// Returns false when OpenBLAS is absent and the caller must use its kernels.
pub fn gemmRowMajorTN(m: usize, n: usize, k: usize, a: [*]const f32, b: [*]const f32, out: [*]f32) bool {
    lookup();
    const f = sgemm_fn orelse return false;
    f(CblasRowMajor, CblasNoTrans, CblasTrans, @intCast(m), @intCast(n), @intCast(k), 1.0, a, @intCast(k), b, @intCast(k), 0.0, out, @intCast(n));
    return true;
}

/// A short identity for receipts: "openblas" or "none".
pub fn name() []const u8 {
    return if (available()) "openblas" else "none";
}
