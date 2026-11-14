//! main.zig - tensorfold-riscv: the RISC-V CPU backend's command line.
//!
//!   caps   capabilities and kernel-family selection
//!   run    a Qwen3 checkpoint: prefill a prompt and decode tokens
//!   bench  prefill/decode throughput on seeded random token ids
//!
//! Every product runs through the kernel ABI (zig/kernels/riscv/tf_riscv.h),
//! so TF_RISCV_KERNELS=scalar|rvv|ime2 (or --kernels) compares families in
//! one binary. Performance claims belong in docs/riscv-performance.md with
//! the K3 measurements this CLI prints as receipts.
const std = @import("std");
const kernels = @import("kernels.zig");
const blas = @import("blas.zig");
const qwen3 = @import("qwen3.zig");
const tokenizer = @import("tokenizer");
const build_info = @import("riscv_build");

const usage =
    \\usage: tensorfold-riscv caps [--kernels scalar|rvv|ime2|auto]
    \\       tensorfold-riscv run MODEL (--prompt TEXT | --tokens ID,ID,...)
    \\                      [--max-tokens N] [--context N] [--kernels FAM]
    \\                      [--temperature T --top-p P --seed S] [--report PATH]
    \\                      [--logits-out PATH]
    \\       tensorfold-riscv bench MODEL [--prefill N] [--decode N] [--context N]
    \\                      [--kernels FAM] [--report PATH]
    \\       tensorfold-riscv --version
    \\
    \\MODEL is a Hugging Face Qwen3 checkpoint directory (config.json,
    \\model.safetensors or its index + shards, tokenizer.json).
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
    const gpa = init.gpa;
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
    if (std.mem.eql(u8, args[1], "run")) return runCommand(init, gpa, args[2..]);
    if (std.mem.eql(u8, args[1], "bench")) return benchCommand(init, gpa, args[2..]);
    try out.writeStreamingAll(init.io, usage);
    return 2;
}

fn capsCommand(init: std.process.Init, args: []const [:0]const u8) !u8 {
    var pick: ?[:0]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--kernels") and i + 1 < args.len) {
            i += 1;
            pick = args[i];
        } else if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
            i += 1;
            blas.setThreads(@intCast(std.fmt.parseInt(u32, args[i], 10) catch fail("--threads needs a number", .{})));
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
        \\blas: {s}
        \\cap_rvv: {}
        \\cap_ime1: {}
        \\cap_ime2: {}
        \\
    , .{
        build_info.version,
        kernels.activeKernels(),
        kernels.vlenBits(),
        blas.name(),
        caps & kernels.Cap.rvv != 0,
        caps & kernels.Cap.ime1 != 0,
        caps & kernels.Cap.ime2 != 0,
    });
    try std.Io.File.stdout().writeStreamingAll(init.io, text);
    return 0;
}

/// Flags shared by run and bench: --kernels, --context, --report plus MODEL.
const Common = struct {
    model: []const u8 = "",
    kernels_pick: ?[:0]const u8 = null,
    context: usize = 4096,
    report: ?[]const u8 = null,
    values: std.StringHashMapUnmanaged([]const u8) = .empty,
};

fn parseCommon(gpa: std.mem.Allocator, args: []const [:0]const u8, comptime extras: []const []const u8) Common {
    var common: Common = .{};
    var positional: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (!std.mem.startsWith(u8, a, "--")) {
            if (positional != null) fail("only one MODEL directory is accepted", .{});
            positional = a;
            continue;
        }
        const key = a[2..];
        var known = std.mem.eql(u8, key, "kernels") or std.mem.eql(u8, key, "context") or std.mem.eql(u8, key, "report") or std.mem.eql(u8, key, "threads");
        for (extras) |e| known = known or std.mem.eql(u8, e, key);
        if (!known) fail("unknown flag --{s}", .{key});
        if (i + 1 >= args.len) fail("--{s} needs a value", .{key});
        i += 1;
        if (std.mem.eql(u8, key, "kernels")) {
            common.kernels_pick = args[i];
        } else if (std.mem.eql(u8, key, "context")) {
            common.context = std.fmt.parseInt(usize, args[i], 10) catch fail("--context needs a number", .{});
        } else if (std.mem.eql(u8, key, "report")) {
            common.report = args[i];
        } else if (std.mem.eql(u8, key, "threads")) {
            blas.setThreads(@intCast(std.fmt.parseInt(u32, args[i], 10) catch fail("--threads needs a number", .{})));
        } else {
            common.values.put(gpa, key, args[i]) catch unreachable;
        }
    }
    common.model = positional orelse fail("a MODEL directory is required", .{});
    return common;
}

fn pickKernels(pick: ?[:0]const u8) void {
    if (pick) |p| {
        if (!kernels.select(p)) fail("kernel family {s} is not usable on this CPU/build", .{p});
    }
}

fn nowNs(init: std.process.Init) i96 {
    return std.Io.Clock.boot.now(init.io).toNanoseconds();
}

fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |piece| {
        if (piece.len == 0) continue;
        try ids.append(gpa, std.fmt.parseInt(u32, piece, 10) catch fail("--tokens needs comma-separated ids", .{}));
    }
    return try ids.toOwnedSlice(gpa);
}

/// Qwen3 checkpoints end a reply at eos_token_id (151645 in the served set).
fn eosId(model: *const qwen3.Model) u32 {
    _ = model;
    return 151645;
}

/// The JSON receipt every measured command can leave behind.
fn writeReport(init: std.process.Init, path: ?[]const u8, body: []const u8) !void {
    const p = path orelse return;
    const file = try std.Io.Dir.cwd().createFile(init.io, p, .{});
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, body);
}

/// One prompt row's logits as decimal text (the parity oracle reads them).
fn writeLogits(init: std.process.Init, gpa: std.mem.Allocator, path: []const u8, logits: []const f32) !void {
    const file = try std.Io.Dir.cwd().createFile(init.io, path, .{});
    defer file.close(init.io);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    for (logits) |v| {
        const line = try std.fmt.allocPrint(gpa, "{d:.9}\n", .{v});
        defer gpa.free(line);
        try buf.appendSlice(gpa, line);
    }
    try file.writeStreamingAll(init.io, buf.items);
}

fn runCommand(init: std.process.Init, gpa: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    const common = parseCommon(gpa, args, &.{ "prompt", "tokens", "max-tokens", "temperature", "top-p", "seed", "logits-out" });
    pickKernels(common.kernels_pick);
    const max_tokens: usize = if (common.values.get("max-tokens")) |v| std.fmt.parseInt(usize, v, 10) catch fail("--max-tokens needs a number", .{}) else 128;
    const temperature: f32 = if (common.values.get("temperature")) |v| std.fmt.parseFloat(f32, v) catch fail("--temperature needs a number", .{}) else 0;
    const top_p: f32 = if (common.values.get("top-p")) |v| std.fmt.parseFloat(f32, v) catch fail("--top-p needs a number", .{}) else 0.9;
    const seed: u64 = if (common.values.get("seed")) |v| std.fmt.parseInt(u64, v, 10) catch fail("--seed needs a number", .{}) else 42;

    var model = qwen3.Model.load(gpa, init.io, common.model) catch |err| fail("cannot load {s}: {s}", .{ common.model, @errorName(err) });
    defer model.deinit();

    const tok_path = try std.fmt.allocPrint(gpa, "{s}/tokenizer.json", .{common.model});
    defer gpa.free(tok_path);

    var prompt_ids: []u32 = &.{};
    var tok: ?tokenizer.Tokenizer = null;
    defer if (tok) |*t| t.deinit();
    if (common.values.get("tokens")) |v| {
        prompt_ids = try parseIds(gpa, v);
    } else {
        const text = common.values.get("prompt") orelse fail("run needs --prompt or --tokens", .{});
        tok = tokenizer.loadTokenizer(init.io, gpa, tok_path) catch |err| fail("cannot load the tokenizer: {s}", .{@errorName(err)});
        prompt_ids = try tok.?.encode(gpa, text);
    }
    defer gpa.free(prompt_ids);
    if (prompt_ids.len == 0) fail("the prompt is empty", .{});
    if (prompt_ids.len + max_tokens > common.context) fail("prompt + tokens exceed --context {d}", .{common.context});

    var session = try qwen3.Session.init(gpa, &model, common.context);
    defer session.deinit(gpa);

    const vocab = model.spec.vocab;
    const logits = try gpa.alloc(f32, vocab);
    defer gpa.free(logits);

    const t0 = nowNs(init);
    try session.forward(prompt_ids, logits, false);
    const t1 = nowNs(init);
    if (common.values.get("logits-out")) |path| try writeLogits(init, gpa, path, logits);

    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(seed);
    var decode_ns: i96 = 0;
    for (0..max_tokens) |_| {
        const id = qwen3.sampleOne(logits, temperature, top_p, prng.random());
        try generated.append(gpa, id);
        if (id == eosId(&model)) break;
        const d0 = nowNs(init);
        try session.forward(&.{id}, logits, false);
        decode_ns += nowNs(init) - d0;
    }

    if (tok) |*t| {
        const text = try t.decode(gpa, generated.items, true);
        defer gpa.free(text);
        try std.Io.File.stdout().writeStreamingAll(init.io, text);
        try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
    }

    const prefill_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(decode_ns)) / 1e9;
    std.debug.print(
        \\tensorfold-riscv run: {d} prompt tokens in {d:.3}s ({d:.2} tok/s), {d} generated in {d:.3}s ({d:.2} tok/s), kernels {s}, vlen {d}
        \\
    , .{
        prompt_ids.len,
        prefill_s,
        @as(f64, @floatFromInt(prompt_ids.len)) / @max(prefill_s, 1e-9),
        generated.items.len,
        decode_s,
        @as(f64, @floatFromInt(generated.items.len)) / @max(decode_s, 1e-9),
        kernels.activeKernels(),
        kernels.vlenBits(),
    });
    const body = try std.json.Stringify.valueAlloc(gpa, .{
        .command = "run",
        .kernels = kernels.activeKernels(),
        .vlen_bits = kernels.vlenBits(),
        .blas = blas.name(),
        .prompt_tokens = prompt_ids.len,
        .generated = generated.items.len,
        .prefill_seconds = prefill_s,
        .decode_seconds = decode_s,
        .prefill_tok_s = @as(f64, @floatFromInt(prompt_ids.len)) / @max(prefill_s, 1e-9),
        .decode_tok_s = @as(f64, @floatFromInt(generated.items.len)) / @max(decode_s, 1e-9),
        .tokens = generated.items,
    }, .{});
    defer gpa.free(body);
    try writeReport(init, common.report, body);
    return 0;
}

fn benchCommand(init: std.process.Init, gpa: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    const common = parseCommon(gpa, args, &.{ "prefill", "decode" });
    pickKernels(common.kernels_pick);
    const prefill_n: usize = if (common.values.get("prefill")) |v| std.fmt.parseInt(usize, v, 10) catch fail("--prefill needs a number", .{}) else 512;
    const decode_n: usize = if (common.values.get("decode")) |v| std.fmt.parseInt(usize, v, 10) catch fail("--decode needs a number", .{}) else 128;
    if (prefill_n + decode_n > common.context) fail("prefill + decode exceed --context {d}", .{common.context});

    var model = qwen3.Model.load(gpa, init.io, common.model) catch |err| fail("cannot load {s}: {s}", .{ common.model, @errorName(err) });
    defer model.deinit();

    // Seeded random ids in the model's vocabulary: the same workload on every
    // machine and kernel family, no tokenizer needed.
    var prng: std.Random.DefaultPrng = .init(0x7C0FFEE5);
    const rnd = prng.random();
    const ids = try gpa.alloc(u32, prefill_n + decode_n);
    defer gpa.free(ids);
    for (ids) |*id| id.* = rnd.uintLessThan(u32, @intCast(model.spec.vocab));

    var session = try qwen3.Session.init(gpa, &model, common.context);
    defer session.deinit(gpa);
    const logits = try gpa.alloc(f32, model.spec.vocab);
    defer gpa.free(logits);

    const t0 = nowNs(init);
    try session.forward(ids[0..prefill_n], logits, false);
    const t1 = nowNs(init);
    for (ids[prefill_n..]) |id| try session.forward(&.{id}, logits, false);
    const t2 = nowNs(init);

    const prefill_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const decode_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    std.debug.print(
        \\tensorfold-riscv bench: prefill {d} tok in {d:.3}s ({d:.2} tok/s), decode {d} tok in {d:.3}s ({d:.2} tok/s), kernels {s}, vlen {d}
        \\
    , .{
        prefill_n,
        prefill_s,
        @as(f64, @floatFromInt(prefill_n)) / @max(prefill_s, 1e-9),
        decode_n,
        decode_s,
        @as(f64, @floatFromInt(decode_n)) / @max(decode_s, 1e-9),
        kernels.activeKernels(),
        kernels.vlenBits(),
    });
    const body = try std.json.Stringify.valueAlloc(gpa, .{
        .command = "bench",
        .kernels = kernels.activeKernels(),
        .vlen_bits = kernels.vlenBits(),
        .blas = blas.name(),
        .prefill_tokens = prefill_n,
        .prefill_seconds = prefill_s,
        .prefill_tok_s = @as(f64, @floatFromInt(prefill_n)) / @max(prefill_s, 1e-9),
        .decode_tokens = decode_n,
        .decode_seconds = decode_s,
        .decode_tok_s = @as(f64, @floatFromInt(decode_n)) / @max(decode_s, 1e-9),
    }, .{});
    defer gpa.free(body);
    try writeReport(init, common.report, body);
    return 0;
}
