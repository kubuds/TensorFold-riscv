//! riscv.zig - build steps for the RISC-V CPU backend (zig/kernels/riscv, zig/src/riscv).
//!
//! Stage 1 (this baseline): the RV64GC scalar kernel family only.
//!
//! Steps (all lazy; nothing runs unless asked):
//!   zig build riscv-test          - kernel tests on the build host (scalar family)
//!   zig build riscv-qemu-test     - the same tests cross-built for riscv64 and run
//!                                   under qemu-user (-Dqemu=PATH, default qemu-riscv64)
//!   zig build riscv-cross         - compile checks for the K3 deployment ABIs
//!                                   (riscv64-linux-gnu glibc and riscv64-linux-musl)
//!   zig build riscv-cli           - tensorfold-riscv for the build host;
//!                                   -Dtarget=riscv64-linux-gnu makes the K3 binary
const std = @import("std");

/// C flags for every RISC-V kernel translation unit (the repo's C conventions).
const cflags = [_][]const u8{ "-std=c11", "-Wall", "-Wextra", "-ffp-contract=off" }; // -Werror off: clang warns about this host's dual GCC installs


/// rv64gc: the scalar baseline ISA (RVA22/RVA23 minus V).
fn riscvTarget(b: *std.Build, abi: std.Target.Abi) std.Build.ResolvedTarget {
    return b.resolveTargetQuery(.{
        .cpu_arch = .riscv64,
        .os_tag = .linux,
        .abi = abi,
        .cpu_features_add = std.Target.riscv.featureSet(&.{ .a, .c, .d, .f, .m, .zicsr, .zifencei }),
    });
}

/// The kernel + binding module for `target`: the scalar family and its dispatch.
fn kernelModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/riscv/kernel_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .pic = true,
    });
    mod.addIncludePath(b.path("zig/kernels/riscv"));
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ops_scalar.c"), .flags = &cflags });
    mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/dispatch.c"), .flags = &cflags });
    return mod;
}

pub fn steps(b: *std.Build, test_step: *std.Build.Step, target: std.Build.ResolvedTarget) void {
    const optimize = b.option(std.builtin.OptimizeMode, "riscv-optimize", "Optimize mode for the RISC-V steps (default ReleaseFast)") orelse .fast;
    const qemu = b.option([]const u8, "qemu", "qemu-user emulator for riscv-qemu-test (default qemu-riscv64)") orelse "qemu-riscv64";

    // Host tests: the scalar family's correctness on whatever CPU builds this.
    {
        const mod = kernelModule(b, b.graph.host, .debug);
        const tests = b.addTest(.{ .root_module = mod });
        const run = b.addRunArtifact(tests);
        const step = b.step("riscv-test", "RISC-V kernel tests on the build host (scalar family)");
        step.dependOn(&run.step);
        test_step.dependOn(&run.step); // part of zig build test
    }

    // Cross tests under qemu: the rv64gc baseline build, scalar family.
    {
        const step = b.step("riscv-qemu-test", "RISC-V kernel tests under qemu (rv64gc baseline build)");
        const mod = kernelModule(b, riscvTarget(b, .musl), optimize);
        const tests = b.addTest(.{ .root_module = mod });
        const run = b.addSystemCommand(&.{qemu});
        run.addArtifactArg(tests);
        step.dependOn(&run.step);
    }

    // K3 deployment ABIs compile: glibc (system linker on K3) and musl (static).
    {
        const step = b.step("riscv-cross", "Compile checks for the K3 deployment ABIs (riscv64-linux-gnu and riscv64-linux-musl)");
        for ([_]std.Target.Abi{ .gnu, .musl }) |abi| {
            const mod = kernelModule(b, riscvTarget(b, abi), optimize);
            const tests = b.addTest(.{ .root_module = mod });
            step.dependOn(&tests.step);
        }
    }

    // tensorfold-riscv: the RISC-V CPU runner (caps now; qwen3 and bench follow).
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("zig/src/riscv/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        });
        mod.addIncludePath(b.path("zig/kernels/riscv"));
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/ops_scalar.c"), .flags = &cflags });
        mod.addCSourceFile(.{ .file = b.path("zig/kernels/riscv/dispatch.c"), .flags = &cflags });
        mod.addOptions("riscv_build", buildOptions(b));
        const exe = b.addExecutable(.{ .name = "tensorfold-riscv", .root_module = mod });
        const step = b.step("riscv-cli", "tensorfold-riscv: the RISC-V CPU runner (caps; qwen3 and bench follow)");
        step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}

/// Build facts the CLI prints in `caps` receipts.
fn buildOptions(b: *std.Build) *std.Build.Step.Options {
    const manifest = @import("../../build.zig.zon");
    const o = b.addOptions();
    o.addOption([]const u8, "version", manifest.version);
    return o;
}
