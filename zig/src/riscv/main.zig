//! main.zig - tensorfold-riscv: the RISC-V CPU backend's command line.
//! Stage 1: capabilities and kernel selection; qwen3 serving and benchmarks
//! land in the next stages of the RISC-V backend work.
const std = @import("std");
const kernels = @import("kernels.zig");
const build_info = @import("riscv_build");

const usage =
    \\usage: tensorfold-riscv caps [--kernels scalar|rvv|ime2|auto]
    \\       tensorfold-riscv --version
    \\
    \\Subcommands:
    \\  caps           print the RISC-V capabilities receipt (RVV, VLEN, IME, kernels)
    \\
    \\Environment:
    \\  TF_RISCV_KERNELS  kernel family to select at startup (scalar|rvv|ime2|auto)
    \\
;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("tensorfold-riscv: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const out = std.Io.File.stdout();

    if (args.len >= 2 and std.mem.eql(u8, args[1], "--version")) {
        try out.writeStreamingAll(init.io, "tensorfold-riscv " ++ build_info.version ++ "\n");
        return 0;
    }
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "help")) {
        try out.writeStreamingAll(init.io, usage);
        return if (args.len < 2) 2 else 0;
    }
    if (std.mem.eql(u8, args[1], "caps")) return capsCommand(init, args[2..]);
    try out.writeStreamingAll(init.io, usage);
    return 2;
}

fn capsCommand(init: std.process.Init, args: []const [:0]const u8) !u8 {
    var pick: ?[:0]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--kernels") and i + 1 < args.len) {
            i += 1;
            pick = args[i]; // args are already null-terminated slices
        } else fail("caps: unknown argument {s}", .{args[i]});
    }
    if (pick) |p| {
        if (!kernels.select(p)) fail("kernel family {s} is not usable on this CPU/build", .{p});
    }
    const caps = kernels.caps();
    var buf: [512]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf,
        \\version: {s}
        \\kernels: {s}
        \\vlen_bits: {d}
        \\cap_rvv: {}
        \\cap_ime1: {}
        \\cap_ime2: {}
        \\
    , .{
        build_info.version,
        kernels.activeKernels(),
        kernels.vlenBits(),
        caps & kernels.Cap.rvv != 0,
        caps & kernels.Cap.ime1 != 0,
        caps & kernels.Cap.ime2 != 0,
    });
    try std.Io.File.stdout().writeStreamingAll(init.io, text);
    return 0;
}
