//! pool.zig - bindings for the sharded-work thread pool
//! (zig/kernels/riscv/pool.c, pthreads). Projections split their output
//! columns across the pool so each worker streams its own sequential weight
//! range: the thread-sharded sequential read pattern the K3 paper measures
//! as the way to saturate LPDDR5 during LLM decode.
const std = @import("std");

pub const Task = *const fn (ctx: ?*anyopaque, begin: c_int, end: c_int) callconv(.c) void;

pub extern "C" fn tf_riscv_pool_init(threads: c_int) c_int;
pub extern "C" fn tf_riscv_pool_run(task: Task, ctx: ?*anyopaque, items: c_int) void;
pub extern "C" fn tf_riscv_pool_stop() void;
pub extern "C" fn tf_riscv_pool_mode(mode: [*:0]const u8) c_int;
pub extern "C" fn tf_riscv_pool_mode_name() [*:0]const u8;
pub extern "C" fn tf_riscv_pool_active() c_int;

var threads: c_int = 1;

/// Start (or resize) the pool; `n` is total participants, caller included.
/// Returns the participants actually running.
pub fn init(n: usize) usize {
    threads = tf_riscv_pool_init(@intCast(n));
    return @intCast(threads);
}

pub fn active() usize {
    return @intCast(tf_riscv_pool_active());
}

/// Placement of the shard work: "x100" pthreads, or "a100" through libspert.
pub fn setMode(name: [:0]const u8) bool {
    return tf_riscv_pool_mode(name.ptr) >= 0;
}

pub fn modeName() []const u8 {
    return std.mem.span(tf_riscv_pool_mode_name());
}

/// Parallel-for over [0, items) in disjoint shards; inline when the pool is
/// single-threaded. Shards never overlap, so results match a serial run bit
/// for bit.
pub fn run(task: Task, ctx: ?*anyopaque, items: usize) void {
    tf_riscv_pool_run(task, ctx, @intCast(items));
}
