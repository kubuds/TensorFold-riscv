//! riscv.zig - build steps for the RISC-V CPU backend (zig/kernels/riscv, zig/src/riscv).
//!
//! Stages: the RV64GC scalar baseline and the RVV 1.0 family (VLEN-agnostic:
//! 256-bit X100, 1024-bit A100), IME2 hooking in when the build enables it.
//!
//! Steps (all lazy; nothing runs unless asked):
//!   zig build riscv-test          - kernel + model tests on the build host
//!   zig build riscv-qemu-test     - the same tests cross-built for riscv64 and
//!                                   run under qemu-user: the rv64gc baseline,
//!                                   and the rv64gcv build at VLEN=256 and
//!                                   VLEN=1024 (each also with the scalar
//!                                   family forced, TF_RISCV_KERNELS=scalar)
//!   zig build riscv-cross         - compile checks for the K3 deployment ABIs
//!                                   (riscv64-linux-gnu glibc and -musl, both
//!                                   the baseline and the RVV build)
//!   zig build riscv-cli           - tensorfold-riscv; -Dtarget=riscv64-linux-gnu
//!                                   makes the K3 binary (RVA23: rv64gcv)
//!   zig build riscv-parity        - tools/riscv/qwen3_parity.py over the binary
//!
//! Options: -Dqemu=PATH (emulator for riscv-qemu-test; default "qemu-riscv64").
const std = @import("std");

/// C flags for every RISC-V kernel translation unit (the repo's C conventions;
/// no -Werror: this host's clang warns about its dual GCC installs at :1:1).
const cflags = [_][]const u8{ "-std=c11", "-Wall", "-Wextra", "-ffp-contract=off" };

/// cflags plus the marker that this translation unit carries the IME2 kernels.
const ime2_kernel_cflags = cflags ++ .{"-DTF_RISCV_Q4X_IME2"};

/// rv64gc (scalar baseline) or rv64gcv (RVV 1.0) for `abi`'s Linux.
fn riscvTarget(b: *std.Build, v: bool, abi: std.Target.Abi) std.Build.ResolvedTarget {
    const feats = if (v)
        std.Target.riscv.featureSet(&.{ .a, .c, .d, .f, .m, .v, .zicsr, .zifencei })
    else
        std.Target.riscv.featureSet(&.{ .a, .c, .d, .f, .m, .zicsr, .zifencei });
    return b.resolveTargetQuery(.{
        .cpu_arch = .riscv64,
        .os_tag = .linux,
        .abi = abi,
        .cpu_features_add = feats,
    });
}

fn wantsV(target: std.Build.ResolvedTarget) bool {
    return target.result.cpu.arch == .riscv64 and
        std.Target.riscv.featureSetHas(target.result.cpu.features, .v);
}

/// The kernel + test module for `target`: scalar and dispatch always, the RVV
/// family when the target's V is on (dispatch.c sees the same -mcpu).
fn kernelModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/riscv/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .pic = true,
    });
    mod.addIncludePath(b.path("zig/kernels/riscv"));
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ops_scalar.c"), .flags = &cflags });
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/pool.c"), .flags = &cflags });
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/mem.c"), .flags = &cflags });
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/q4x.c"), .flags = &cflags });
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/q4x_ime2.c"), .flags = &cflags });
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/attention_block.c"), .flags = &cflags });
    if (wantsV(target)) {
        // IME2 with its inlined reference MMA: the tests exercise the tiling
        // and dispatch under qemu at VLEN=256 and VLEN=1024. dispatch.c must
        // see the same defines to compile the family in.
        const ime2c = cflags ++ .{ "-DTF_RISCV_WITH_IME2", "-DTF_RISCV_IME2_INLINE" };
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/dispatch.c"), .flags = &ime2c });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ops_rvv.c"), .flags = &cflags });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ime2.c"), .flags = &ime2c });
    } else {
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/dispatch.c"), .flags = &cflags });
    }
    return mod;
}

pub fn steps(b: *std.Build, test_step: *std.Build.Step, target: std.Build.ResolvedTarget) void {
    const optimize = b.option(std.builtin.OptimizeMode, "riscv-optimize", "Optimize mode for the RISC-V steps (default ReleaseFast)") orelse .fast;
    const qemu = b.option([]const u8, "qemu", "qemu-user emulator for riscv-qemu-test (default qemu-riscv64)") orelse "qemu-riscv64";

    // Host tests: every kernel family the host build carries, plus the model.
    {
        const mod = kernelModule(b, b.graph.host, .debug);
        const tests = b.addTest(.{ .root_module = mod });
        const run = b.addRunArtifact(tests);
        const step = b.step("riscv-test", "RISC-V kernel and model tests on the build host");
        step.dependOn(&run.step);
        test_step.dependOn(&run.step); // part of zig build test
    }

    // Cross tests under qemu: rv64gc baseline and rv64gcv at two VLENs.
    {
        const step = b.step("riscv-qemu-test", "RISC-V tests under qemu (rv64gc baseline; rv64gcv at VLEN=256/1024)");
        const baseline = kernelModule(b, riscvTarget(b, false, .musl), optimize);
        const base_tests = b.addTest(.{ .root_module = baseline });
        const run_base = b.addSystemCommand(&.{qemu});
        run_base.addArtifactArg(base_tests);
        step.dependOn(&run_base.step);

        const rvv = kernelModule(b, riscvTarget(b, true, .musl), optimize);
        const rvv_tests = b.addTest(.{ .root_module = rvv });
        for ([_][]const u8{ "256", "1024" }) |vlen| {
            const run = b.addSystemCommand(&.{qemu});
            run.addArgs(&.{ "-cpu", b.fmt("rv64,v=true,vlen={s}", .{vlen}) });
            // qemu does not implement SpacemiT's vmadot, so the q4x family runs
            // its portable float path here; the IME2 path is hardware-only.
            run.setEnvironmentVariable("TF_RISCV_Q4X", "float");
            run.addArtifactArg(rvv_tests);
            step.dependOn(&run.step);
            // The scalar family must stay correct inside an RVV build too.
            const run_scalar = b.addSystemCommand(&.{qemu});
            run_scalar.addArgs(&.{ "-cpu", b.fmt("rv64,v=true,vlen={s}", .{vlen}) });
            run_scalar.setEnvironmentVariable("TF_RISCV_KERNELS", "scalar");
            run_scalar.setEnvironmentVariable("TF_RISCV_Q4X", "float");
            run_scalar.addArtifactArg(rvv_tests);
            step.dependOn(&run_scalar.step);
        }
    }

    // K3 deployment ABIs compile: glibc (system linker on K3) and musl, each
    // in the baseline and the RVV ISA build.
    {
        const step = b.step("riscv-cross", "Compile checks for the K3 deployment ABIs (baseline and RVV)");
        for ([_]std.Target.Abi{ .gnu, .musl }) |abi| {
            for ([_]bool{ false, true }) |v| {
                const mod = kernelModule(b, riscvTarget(b, v, abi), optimize);
                const tests = b.addTest(.{ .root_module = mod });
                step.dependOn(&tests.step);
            }
        }
    }

    // tensorfold-riscv: the RISC-V CPU runner (caps, qwen3 run and bench).
    // A riscv64 CLI target carries V: K3 is RVA23 (V is mandatory there) and
    // dispatch picks the family at runtime; the baseline ISA stays in the
    // riscv-qemu-test and riscv-cross steps above.
    {
        const cli_target = if (target.result.cpu.arch == .riscv64 and !wantsV(target))
            riscvTarget(b, true, target.result.abi)
        else
            target;
        const mod = b.createModule(.{
            .root_source_file = b.path("zig/src/riscv/main.zig"),
            .target = cli_target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        });
        mod.addIncludePath(b.path("zig/kernels/riscv"));
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ops_scalar.c"), .flags = &cflags });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/pool.c"), .flags = &cflags });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/mem.c"), .flags = &cflags });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/q4x.c"), .flags = &cflags });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/attention_block.c"), .flags = &cflags });
        const ime2_inline = b.option(bool, "riscv-ime2-inline", "Compile the IME2 reference MMA into the CLI (correctness bring-up; the vendor runtime is the production path)") orelse false;
        if (wantsV(cli_target)) {
            var ime2_flags: std.ArrayListUnmanaged([]const u8) = .empty;
            try_ime2_flags(b, &ime2_flags, ime2_inline);
            mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/dispatch.c"), .flags = ime2_flags.items });
            mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ops_rvv.c"), .flags = &cflags });
            mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ime2.c"), .flags = ime2_flags.items });
            mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/q4x_ime2.c"), .flags = &cflags });
        } else {
            mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/dispatch.c"), .flags = &cflags });
        }
        mod.addOptions("riscv_build", buildOptions(b));
        const tokenizer_mod = b.createModule(.{
            .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"),
            .target = cli_target,
            .optimize = optimize,
            .link_libc = true,
        });
        mod.addImport("tokenizer", tokenizer_mod);
        const exe = b.addExecutable(.{ .name = "tensorfold-riscv", .root_module = mod });
        const step = b.step("riscv-cli", "tensorfold-riscv: the RISC-V CPU runner (caps, qwen3 run, bench)");
        step.dependOn(&b.addInstallArtifact(exe, .{}).step);

        // riscv-parity: the numpy oracle over the freshly built binary.
        const parity = b.addSystemCommand(&.{"python3"});
        parity.addArgs(&.{ "tools/riscv/qwen3_parity.py", "--bin" });
        parity.addArtifactArg(exe);
        const parity_step = b.step("riscv-parity", "tensorfold-riscv against an independent numpy Qwen3 forward");
        parity_step.dependOn(&parity.step);
    }
}

/// cflags plus the IME2 defines for dispatch.c and ime2.c.
fn try_ime2_flags(b: *std.Build, out: *std.ArrayListUnmanaged([]const u8), inline_mma: bool) void {
    out.appendSlice(b.allocator, &cflags) catch @panic("oom");
    out.append(b.allocator, "-DTF_RISCV_WITH_IME2") catch @panic("oom");
    if (inline_mma) out.append(b.allocator, "-DTF_RISCV_IME2_INLINE") catch @panic("oom");
}

/// Build facts the CLI prints in `caps` receipts.
fn buildOptions(b: *std.Build) *std.Build.Step.Options {
    const manifest = @import("../../build.zig.zon");
    const o = b.addOptions();
    o.addOption([]const u8, "version", manifest.version);
    return o;
}
