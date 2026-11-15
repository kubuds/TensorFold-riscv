//! model_test.zig - a synthetic Qwen3 through the loader and the forward:
//! determinism, finite logits, and agreement across every kernel family the
//! runtime offers. The independent numeric oracle is tools/riscv/qwen3_parity.py.
const std = @import("std");
const qwen3 = @import("qwen3.zig");
const kernels = @import("kernels.zig");
const st = @import("safetensors.zig");

const t = std.testing;

const tiny = qwen3.Spec{
    .hidden = 32,
    .inter = 64,
    .layers = 2,
    .heads = 4,
    .kv_heads = 2,
    .head_dim = 8,
    .vocab = 97,
    .tied = true,
    .eps = 1e-6,
    .theta = 1e6,
};

const Entry = struct {
    name: []const u8,
    rows: usize,
    cols: usize, // 1-D tensors (norms) keep cols == 0
    data: []f32,
};

/// Deterministic f32 weights in [-0.5, 0.5]; same checkpoint on every machine.
fn fill(buf: []f32, seed: u64) void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    for (buf) |*v| v.* = rnd.float(f32) - 0.5;
}

/// Write a one-shard safetensors checkpoint + config.json for `tiny` under dir.
fn writeCheckpoint(gpa: std.mem.Allocator, dir: []const u8) !void {
    const s = tiny;
    var entries: std.ArrayList(Entry) = .empty;
    defer {
        for (entries.items) |e| {
            gpa.free(e.name);
            gpa.free(e.data);
        }
        entries.deinit(gpa);
    }
    var seed: u64 = 1;
    const add = struct {
        fn go(a: std.mem.Allocator, list: *std.ArrayList(Entry), name: []const u8, rows: usize, cols: usize, sd: *u64, norm: bool) !void {
            const n = if (norm) rows else rows * cols;
            const data = try a.alloc(f32, n);
            fill(data, sd.*);
            sd.* += 1;
            if (norm) {
                for (data) |*v| v.* += 1.0; // norm weights stay positive
            }
            try list.append(a, .{ .name = try a.dupe(u8, name), .rows = rows, .cols = if (norm) 0 else cols, .data = data });
        }
    }.go;
    try add(gpa, &entries, "model.embed_tokens.weight", s.vocab, s.hidden, &seed, false);
    try add(gpa, &entries, "model.norm.weight", s.hidden, 1, &seed, true);
    for (0..s.layers) |li| {
        var name_buf: [256]u8 = undefined;
        const names = .{
            .{ "self_attn.q_proj.weight", s.heads * s.head_dim, s.hidden, false },
            .{ "self_attn.k_proj.weight", s.kv_heads * s.head_dim, s.hidden, false },
            .{ "self_attn.v_proj.weight", s.kv_heads * s.head_dim, s.hidden, false },
            .{ "self_attn.o_proj.weight", s.hidden, s.heads * s.head_dim, false },
            .{ "self_attn.q_norm.weight", s.head_dim, 1, true },
            .{ "self_attn.k_norm.weight", s.head_dim, 1, true },
            .{ "input_layernorm.weight", s.hidden, 1, true },
            .{ "post_attention_layernorm.weight", s.hidden, 1, true },
            .{ "mlp.gate_proj.weight", s.inter, s.hidden, false },
            .{ "mlp.up_proj.weight", s.inter, s.hidden, false },
            .{ "mlp.down_proj.weight", s.hidden, s.inter, false },
        };
        inline for (names) |n| {
            const full = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ li, n[0] });
            try add(gpa, &entries, full, n[1], n[2], &seed, n[3]);
        }
    }

    // Assemble the safetensors file: u64 header_len, space-padded JSON, data.
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(gpa);
    try header.append(gpa, '{');
    var offset: usize = 0;
    for (entries.items, 0..) |e, i| {
        if (i > 0) try header.append(gpa, ',');
        const shape = if (e.cols == 0)
            try std.fmt.allocPrint(gpa, "[{d}]", .{e.rows})
        else
            try std.fmt.allocPrint(gpa, "[{d},{d}]", .{ e.rows, e.cols });
        defer gpa.free(shape);
        const piece = try std.fmt.allocPrint(gpa, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ e.name, shape, offset, offset + e.data.len * 4 });
        defer gpa.free(piece);
        try header.appendSlice(gpa, piece);
        offset += e.data.len * 4;
    }
    try header.append(gpa, '}');
    while (header.items.len % 8 != 0) try header.append(gpa, ' ');

    const out = try gpa.alloc(u8, 8 + header.items.len + offset);
    defer gpa.free(out);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    var at = 8 + header.items.len;
    for (entries.items) |e| {
        @memcpy(out[at..][0 .. e.data.len * 4], std.mem.sliceAsBytes(e.data));
        at += e.data.len * 4;
    }

    try std.Io.Dir.cwd().createDirPath(t.io, dir);
    {
        const path = try std.fmt.allocPrint(gpa, "{s}/model.safetensors", .{dir});
        defer gpa.free(path);
        const file = try std.Io.Dir.cwd().createFile(t.io, path, .{});
        defer file.close(t.io);
        try file.writeStreamingAll(t.io, out);
    }
    {
        const path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(path);
        const config = try std.fmt.allocPrint(gpa, "{{\"architectures\":[\"Qwen3ForCausalLM\"],\"hidden_size\":{d},\"intermediate_size\":{d},\"num_hidden_layers\":{d},\"num_attention_heads\":{d},\"num_key_value_heads\":{d},\"head_dim\":{d},\"vocab_size\":{d},\"tie_word_embeddings\":true,\"rms_norm_eps\":1e-6,\"rope_theta\":1e6}}", .{ s.hidden, s.inter, s.layers, s.heads, s.kv_heads, s.head_dim, s.vocab });
        defer gpa.free(config);
        const file = try std.Io.Dir.cwd().createFile(t.io, path, .{});
        defer file.close(t.io);
        try file.writeStreamingAll(t.io, config);
    }
}

test "riscv qwen3: the synthetic checkpoint loads and decodes deterministically" {
    const gpa = t.allocator;
    const dir = ".zig-cache/riscv-tiny-a";
    try writeCheckpoint(gpa, dir);

    var model = try qwen3.Model.load(gpa, t.io, dir);
    defer model.deinit();
    try t.expectEqual(tiny.hidden, model.spec.hidden);
    try t.expectEqual(@as(?bool, true), model.spec.tied);

    const ids = [_]u32{ 1, 5, 90, 42 };
    const vocab = model.spec.vocab;
    const logits_a = try gpa.alloc(f32, vocab * ids.len);
    defer gpa.free(logits_a);
    const logits_b = try gpa.alloc(f32, vocab * ids.len);
    defer gpa.free(logits_b);
    {
        var s = try qwen3.Session.init(gpa, &model, 64);
        defer s.deinit(gpa);
        try s.forward(&ids, logits_a, true);
    }
    {
        var s = try qwen3.Session.init(gpa, &model, 64);
        defer s.deinit(gpa);
        try s.forward(&ids, logits_b, true);
    }
    try t.expectEqualSlices(f32, logits_a, logits_b);
    for (logits_a) |v| try t.expect(std.math.isFinite(v));
}

test "riscv qwen3: every kernel family agrees on the synthetic forward" {
    const gpa = t.allocator;
    const dir = ".zig-cache/riscv-tiny-b";
    try writeCheckpoint(gpa, dir);
    var model = try qwen3.Model.load(gpa, t.io, dir);
    defer model.deinit();

    const ids = [_]u32{ 3, 14, 1, 5, 9 };
    const vocab = model.spec.vocab;
    const logits = try gpa.alloc(f32, vocab * ids.len);
    defer gpa.free(logits);
    var first: ?[]f32 = null;
    var first_name: []const u8 = "";
    defer if (first) |f| gpa.free(f);

    for ([_][:0]const u8{ "scalar", "rvv", "ime2" }) |fam| {
        if (!kernels.select(fam)) continue;
        var s = try qwen3.Session.init(gpa, &model, 64);
        defer s.deinit(gpa);
        @memset(logits, 0);
        try s.forward(&ids, logits, true);
        if (first == null) {
            first = try gpa.dupe(f32, logits);
            first_name = fam;
            continue;
        }
        // Family products agree to f32 accumulation error, not bit for bit.
        for (logits, first.?) |got, want| {
            const tol = 1e-4 * @abs(want) + 1e-5;
            if (@abs(got - want) > tol) {
                std.debug.print("family {s} differs from {s}: {e} vs {e}\n", .{ fam, first_name, got, want });
                return error.TestExpectedEqual;
            }
        }
    }
}

test "riscv qwen3: the thread pool matches the serial forward bit for bit" {
    const gpa = t.allocator;
    const dir = ".zig-cache/riscv-tiny-c";
    try writeCheckpoint(gpa, dir);
    var model = try qwen3.Model.load(gpa, t.io, dir);
    defer model.deinit();

    const ids = [_]u32{ 3, 14, 1, 5, 9, 77, 4, 2 };
    const vocab = model.spec.vocab;
    const serial = try gpa.alloc(f32, vocab * ids.len);
    defer gpa.free(serial);
    const threaded = try gpa.alloc(f32, vocab * ids.len);
    defer gpa.free(threaded);
    {
        var s = try qwen3.Session.init(gpa, &model, 64);
        defer s.deinit(gpa);
        try s.forward(&ids, serial, true);
    }
    const pool = @import("pool.zig");
    const ran = pool.init(4);
    defer _ = pool.init(1);
    try t.expectEqual(@as(usize, 4), ran);
    {
        var s = try qwen3.Session.init(gpa, &model, 64);
        defer s.deinit(gpa);
        try s.forward(&ids, threaded, true);
    }
    // Disjoint output shards: the pool must be bit-identical to the serial run.
    try t.expectEqualSlices(f32, serial, threaded);
}

test "riscv qwen3: safetensors refuses a truncated file" {
    const path = ".zig-cache/riscv-bad.safetensors";
    {
        const file = try std.Io.Dir.cwd().createFile(t.io, path, .{});
        defer file.close(t.io);
        try file.writeStreamingAll(t.io, "\x20\x00\x00\x00\x00\x00\x00\x00");
    }
    try t.expectError(error.BadSafetensors, st.File.open(t.allocator, t.io, path));
}
