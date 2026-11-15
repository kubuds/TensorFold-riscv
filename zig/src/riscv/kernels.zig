//! kernels.zig - Zig bindings over the RISC-V kernel ABI (zig/kernels/riscv/tf_riscv.h).
//! The C families (scalar RV64GC, RVV 1.0, IME2) are linked into the same
//! module; this file only declares the ABI and reports what the runtime chose.
const std = @import("std");

pub const c = struct {
    pub extern "C" fn tf_riscv_caps() u32;
    pub extern "C" fn tf_riscv_vlen_bits() u32;
    pub extern "C" fn tf_riscv_active_kernels() [*:0]const u8;
    pub extern "C" fn tf_riscv_select_kernels(name: [*:0]const u8) c_int;
    pub extern "C" fn tf_riscv_bind_cpus(spec: [*:0]const u8) c_int;

    pub extern "C" fn tf_riscv_rmsnorm(x: [*]const f32, w: ?[*]const f32, out: [*]f32, n: c_int, eps: f32) void;
    pub extern "C" fn tf_riscv_rope(q: [*]f32, k: [*]f32, pos: c_int, q_heads: c_int, kv_heads: c_int, head_dim: c_int, theta: f32) void;
    pub extern "C" fn tf_riscv_softmax(x: [*]f32, n: c_int) void;
    pub extern "C" fn tf_riscv_silu_mul(gate: [*]const f32, up: [*]const f32, out: [*]f32, n: c_int) void;
    pub extern "C" fn tf_riscv_add(a: [*]const f32, b: [*]const f32, out: [*]f32, n: c_int) void;
    pub extern "C" fn tf_riscv_matmul_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, m: c_int, n: c_int, k: c_int, trans_b: c_int) void;
    pub extern "C" fn tf_riscv_matmul_bf16(x: [*]const f32, w: [*]const u16, out: [*]f32, m: c_int, n: c_int, k: c_int) void;
    pub extern "C" fn tf_riscv_matmul_bf16_shard(x: [*]const f32, w: [*]const u16, out: [*]f32, m: c_int, n: c_int, k: c_int, j0: c_int, j1: c_int) void;
    pub extern "C" fn tf_riscv_matmul_f32_shard(a: [*]const f32, b: [*]const f32, out: [*]f32, m: c_int, n: c_int, k: c_int, trans_b: c_int, j0: c_int, j1: c_int) void;
    pub extern "C" fn tf_riscv_attention_f32(q: [*]const f32, k: [*]const f32, v: [*]const f32, out: [*]f32, rows: c_int, head_dim: c_int, scale: f32, scratch: [*]f32) void;
    pub extern "C" fn tf_riscv_gemv_q8_0(x: [*]const f32, w: *const anyopaque, out: [*]f32, n: c_int, k: c_int) void;
    pub extern "C" fn tf_riscv_gemv_q4_0(x: [*]const f32, w: *const anyopaque, out: [*]f32, n: c_int, k: c_int) void;
    pub extern "C" fn tf_riscv_matmul_q8_0_shard(x: [*]const f32, w: *const anyopaque, out: [*]f32, m: c_int, n: c_int, k: c_int, j0: c_int, j1: c_int) void;
    pub extern "C" fn tf_riscv_matmul_q4_0_shard(x: [*]const f32, w: *const anyopaque, out: [*]f32, m: c_int, n: c_int, k: c_int, j0: c_int, j1: c_int) void;

    // q4x: 4-bit weights in IME2 dot tiles (zig/kernels/riscv/q4x.c).
    pub extern "C" fn tf_riscv_q4x_block_bytes(k: c_int) usize;
    pub extern "C" fn tf_riscv_q4x_row_bytes(k: c_int) usize;
    pub extern "C" fn tf_riscv_q4x_ok(k: c_int) c_int;
    pub extern "C" fn tf_riscv_quant_a(x: [*]const f32, xa: [*]i8, xs: [*]f32, m: c_int, k: c_int) void;
    pub extern "C" fn tf_riscv_gemv_q4x(xa: [*]const i8, xs: [*]const f32, w: *const anyopaque, out: [*]f32, n: c_int, k: c_int) void;
    pub extern "C" fn tf_riscv_gemv_q4x_shard(xa: [*]const i8, xs: [*]const f32, w: *const anyopaque, out: [*]f32, n: c_int, k: c_int, j0: c_int, j1: c_int) void;
    pub extern "C" fn tf_riscv_gemm_q4x_shard(xa: [*]const i8, xs: [*]const f32, w: *const anyopaque, out: [*]f32, m: c_int, n: c_int, k: c_int, j0: c_int, j1: c_int) void;
    pub extern "C" fn tf_riscv_gemv_q4x_shard_scalar(xa: [*]const i8, xs: [*]const f32, w: *const anyopaque, out: [*]f32, n: c_int, k: c_int, j0: c_int, j1: c_int) void;
    pub extern "C" fn tf_riscv_gemm_q4x_shard_scalar(xa: [*]const i8, xs: [*]const f32, w: *const anyopaque, out: [*]f32, m: c_int, n: c_int, k: c_int, j0: c_int, j1: c_int) void;

    // K3 placement: A100 tiles through libspert, TCM scratch, HPAGE weights.
    pub extern "C" fn tf_riscv_pool_mode(mode: [*:0]const u8) c_int;
    pub extern "C" fn tf_riscv_pool_mode_name() [*:0]const u8;
    pub extern "C" fn tf_riscv_pool_active() c_int;
    pub extern "C" fn tf_riscv_a100_available() c_int;
    pub extern "C" fn tf_riscv_tcm_bytes() usize;
    pub extern "C" fn tf_riscv_tcm_used() usize;
    pub extern "C" fn tf_riscv_hpage_alloc(bytes: usize) ?*anyopaque;
    pub extern "C" fn tf_riscv_hpage_free(p: ?*anyopaque, bytes: usize) void;
    pub extern "C" fn tf_riscv_hpage_available() c_int;
    pub extern "C" fn tf_riscv_hpage_backed(p: ?*const anyopaque) usize;
    pub extern "C" fn tf_riscv_hpage_bytes() usize;
};

pub const Cap = struct {
    pub const rvv: u32 = 1 << 0;
    pub const ime1: u32 = 1 << 1;
    pub const ime2: u32 = 1 << 2;
};

pub fn caps() u32 {
    return c.tf_riscv_caps();
}

/// Architectural VLEN in bits; 0 when the build or CPU has no RVV.
pub fn vlenBits() u32 {
    return c.tf_riscv_vlen_bits();
}

pub fn activeKernels() []const u8 {
    return std.mem.span(c.tf_riscv_active_kernels());
}

/// "auto", "scalar", "rvv" or "ime2"; false when the family is not usable here.
pub fn select(name: [:0]const u8) bool {
    return c.tf_riscv_select_kernels(name.ptr) == 0;
}
