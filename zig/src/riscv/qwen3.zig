//! qwen3.zig - a Qwen3 dense decoder on the RISC-V CPU backend.
//!
//! Serves Hugging Face safetensors checkpoints (bf16 or f32; f16 loads as
//! f32) with the tokenizer.json that ships beside them. Every product runs
//! through the kernel ABI (zig/kernels/riscv/tf_riscv.h): RMSNorm, RoPE in
//! the HF Qwen3 "rotate_half" convention, GQA attention over a per-head KV
//! cache, SwiGLU MLP and bf16/f32 matmuls.
//!
//! Correctness contract: tools/riscv/qwen3_parity.py runs an independent
//! numpy forward over the same weights and compares logits and greedy
//! tokens; zig/src/riscv/model_test.zig keeps a deterministic in-memory
//! smoke check in `zig build riscv-test`.
const std = @import("std");
const st = @import("safetensors.zig");
const kernels = @import("kernels.zig");
const blas = @import("blas.zig");
const pool = @import("pool.zig");
const hpage = @import("hpage.zig");

/// Place the model arena in the K3 huge-page weight arena (--mem hpage).
pub var hpage_weights = true;

const c = kernels.c;

pub const Spec = struct {
    hidden: usize,
    inter: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    tied: bool,
    eps: f32,
    theta: f32,
};

/// Weight storage: f32/bf16 lanes (row-major [rows, cols]), GGUF q8_0/q4_0
/// blocks (k/32 blocks per row, k % 32 == 0), or q4x - 4-bit weights packed in
/// SpacemiT's IME2 dot tiles (8 output rows x 32 k per 128 bytes, k % 256).
pub const Kind = enum { f32, bf16, q8_0, q4_0, q4x };

/// Load-time quantization for the projections (norms stay f32). q8_0: d =
/// amax/127 rounded to f16, q = round(w/d) in [-127,127]; q4_0: d = amax/8
/// to f16, q = round(w/d)+8 in [0,15]; q4x: one f16 scale per 256 weights,
/// d = amax/7, q = round(w/d)+8 in [-7,7]. tools/riscv/qwen3_parity.py applies
/// the same convention and compares logits.
pub const Quant = enum { none, q8_0, q4_0, q4x };

/// The quantization `Model.load` applies to 2-D projection weights.
pub var quant: Quant = .none;

pub const Weight = struct {
    rows: usize,
    cols: usize,
    kind: Kind,
    /// rows*cols elements: f32 lanes when kind == .f32, u16 lanes when .bf16.
    mem: []const u8,

    pub fn f32Lanes(w: Weight) []const f32 {
        return @as([*]const f32, @ptrCast(@alignCast(w.mem.ptr)))[0 .. w.rows * w.cols];
    }

    pub fn bf16Lanes(w: Weight) []const u16 {
        return @as([*]const u16, @ptrCast(@alignCast(w.mem.ptr)))[0 .. w.rows * w.cols];
    }

    /// Block bytes per weight row for the quantized kinds.
    pub fn rowBytes(w: Weight) usize {
        return switch (w.kind) {
            .q8_0 => (w.cols / 32) * 34,
            .q4_0 => (w.cols / 32) * 18,
            .q4x => c.tf_riscv_q4x_row_bytes(@intCast(w.cols)),
            else => w.cols * 4,
        };
    }
};

pub const Layer = struct {
    input_norm: []const f32,
    post_norm: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    spec: Spec,
    embed: Weight,
    layers: []Layer,
    final_norm: []const f32,
    /// lm_head, or the embedding when the checkpoint ties them.
    head: Weight,

    pub fn deinit(m: *Model) void {
        m.arena.deinit();
        m.* = undefined;
    }

    /// out[m,n] = x[m,k] @ w[n,k]^T: OpenBLAS when it is loaded and the
    /// product is f32 with several rows, else the backend's kernels - sharded
    /// over the output columns when the thread pool is up (each worker streams
    /// its own weight rows; results are bit-identical to the serial kernel).
    pub fn project(m: *const Model, w: Weight, x: []const f32, out: []f32, rows: usize) void {
        _ = m;
        switch (w.kind) {
            .q4x => {
                // q4x: weights live in IME2 dot tiles and activations are
                // quantized to int8 per 256-value group, so decode (one row)
                // and prefill (m rows) run the same vmadot inner loop.
                const raw: [*]const u8 = @ptrCast(@alignCast(w.mem.ptr));
                const pad = q4xActivation(w.cols, rows);
                var t0 = nowNs();
                c.tf_riscv_quant_a(x.ptr, pad.xa.ptr, pad.xs.ptr, @intCast(rows), @intCast(w.cols));
                profAccum(profilePhaseForProject(), 1, t0);
                t0 = nowNs();
                if (rows == 1) {
                    if (pool.active() > 1) {
                        var ctx: GemvQ4xCtx = .{ .xa = pad.xa.ptr, .xs = pad.xs.ptr, .raw = raw, .out = out.ptr, .n = w.rows, .k = w.cols };
                        pool.run(gemvQ4xShard, &ctx, w.rows);
                        profAccum(profilePhaseForProject(), 6, t0);
                        return;
                    }
                    c.tf_riscv_gemv_q4x(pad.xa.ptr, pad.xs.ptr, raw, out.ptr, @intCast(w.rows), @intCast(w.cols));
                    profAccum(profilePhaseForProject(), 6, t0);
                    return;
                }
                if (pool.active() > 1) {
                    var ctx: GemmQ4xCtx = .{ .xa = pad.xa.ptr, .xs = pad.xs.ptr, .raw = raw, .out = out.ptr, .m = rows, .n = w.rows, .k = w.cols };
                    pool.run(gemmQ4xShard, &ctx, w.rows);
                    profAccum(profilePhaseForProject(), 6, t0);
                    return;
                }
                c.tf_riscv_gemm_q4x_shard(pad.xa.ptr, pad.xs.ptr, raw, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols), 0, @intCast(w.rows));
                profAccum(profilePhaseForProject(), 6, t0);
            },
            .bf16 => {
                if (pool.active() > 1) {
                    var ctx: ProjectCtx = .{ .x = x.ptr, .bf16_lanes = w.bf16Lanes().ptr, .f32_lanes = null, .out = out.ptr, .m = rows, .n = w.rows, .k = w.cols, .bf16 = true };
                    pool.run(projectShard, &ctx, w.rows);
                    return;
                }
                c.tf_riscv_matmul_bf16(x.ptr, w.bf16Lanes().ptr, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols));
            },
            .q8_0, .q4_0 => {
                const raw: [*]const u8 = @ptrCast(@alignCast(w.mem.ptr));
                if (pool.active() > 1) {
                    var ctx: ProjectCtx = .{ .x = x.ptr, .bf16_lanes = null, .f32_lanes = null, .raw = raw, .out = out.ptr, .m = rows, .n = w.rows, .k = w.cols, .bf16 = false, .q8 = w.kind == .q8_0, .quant = true };
                    pool.run(projectShard, &ctx, w.rows);
                    return;
                }
                if (w.kind == .q8_0)
                    c.tf_riscv_matmul_q8_0_shard(x.ptr, raw, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols), 0, @intCast(w.rows))
                else
                    c.tf_riscv_matmul_q4_0_shard(x.ptr, raw, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols), 0, @intCast(w.rows));
            },
            .f32 => {
                if (rows >= 4 and blas.gemmRowMajorTN(rows, w.rows, w.cols, x.ptr, w.f32Lanes().ptr, out.ptr)) return;
                if (pool.active() > 1) {
                    var ctx: ProjectCtx = .{ .x = x.ptr, .bf16_lanes = null, .f32_lanes = w.f32Lanes().ptr, .out = out.ptr, .m = rows, .n = w.rows, .k = w.cols, .bf16 = false };
                    pool.run(projectShard, &ctx, w.rows);
                    return;
                }
                c.tf_riscv_matmul_f32(x.ptr, w.f32Lanes().ptr, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols), 1);
            },
        }
    }

    /// One row of a [rows, cols] weight table into f32 (embedding lookup;
    /// the embedding table is never quantized).
    pub fn tableRow(m: *const Model, w: Weight, row: usize, out: []f32) void {
        _ = m;
        switch (w.kind) {
            .q8_0, .q4_0, .q4x => unreachable, // quantize() never touches the table
            .f32 => {
                const src = w.f32Lanes()[row * w.cols ..][0..w.cols];
                @memcpy(out, src);
                if (qwen3_dump) {
                    std.debug.print("TR row {d} src {*} dst {*} src0 {e} dst0 {e}\n", .{ row, src.ptr, out.ptr, src[0], out[0] });
                }
            },
            .bf16 => {
                const lanes = w.bf16Lanes()[row * w.cols ..][0..w.cols];
                for (lanes, out) |v, *o| o.* = @as(f32, @bitCast(@as(u32, v) << 16));
            },
        }
    }

    /// Load a Hugging Face Qwen3 checkpoint directory: config.json, shards and
    /// their weight map. Every tensor's shape is checked against the config;
    /// a mismatch refuses the checkpoint by name.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Model {
        const backing = if (hpage_weights and hpage.available()) hpage.allocator() else gpa;
        var arena: std.heap.ArenaAllocator = .init(backing);
        errdefer arena.deinit();
        const a = arena.allocator();

        const config_path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir});
        const config_json = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 26));
        const spec = try readSpec(gpa, config_json);
        if (spec.heads % spec.kv_heads != 0) return error.BadConfig;

        // Resolve every needed tensor name to its shard.
        var wanted: std.StringHashMapUnmanaged([]const u8) = .empty; // name -> shard file
        defer wanted.deinit(gpa);
        const index_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{dir});
        const index_json = std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26)) catch null;
        if (index_json) |json| {
            // Single-file checkpoints have no index; a download failure can
            // leave a non-JSON page behind. Neither refuses the checkpoint:
            // without a usable weight_map the loader reads model.safetensors.
            if (std.json.parseFromSlice(std.json.Value, gpa, json, .{})) |parsed| {
                defer parsed.deinit();
                if (parsed.value == .object) {
                    if (parsed.value.object.get("weight_map")) |map| {
                        if (map == .object) {
                            var it = map.object.iterator();
                            while (it.next()) |e| try wanted.put(gpa, try a.dupe(u8, e.key_ptr.*), try a.dupe(u8, e.value_ptr.string));
                        }
                    }
                }
            } else |_| {
                std.debug.print("tensorfold-riscv: {s} is not a weight index; using model.safetensors\n", .{index_path});
            }
        }
        const single = "model.safetensors";

        var m: Model = .{
            .gpa = gpa,
            .arena = arena,
            .spec = spec,
            .embed = .{ .rows = 0, .cols = 0, .kind = .f32, .mem = &.{} },
            .layers = try a.alloc(Layer, spec.layers),
            .final_norm = &.{},
            .head = .{ .rows = 0, .cols = 0, .kind = .f32, .mem = &.{} },
        };
        @memset(std.mem.sliceAsBytes(m.layers), 0);
        var shards: std.StringHashMapUnmanaged(void) = .empty;
        defer shards.deinit(gpa);
        if (wanted.count() == 0) try shards.put(gpa, single, {}) else {
            var it = wanted.valueIterator();
            while (it.next()) |shard| try shards.put(gpa, shard.*, {});
        }

        // Read each shard once, extracting the tensors the model needs.
        var shard_it = shards.iterator();
        while (shard_it.next()) |shard| {
            const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, shard.key_ptr.* });
            var file = try st.File.open(gpa, io, path);
            defer file.deinit(gpa, io);
            try m.take(a, &file);
        }
        if (!spec.tied and m.head.mem.len == 0) {
            std.debug.print("tensorfold-riscv: the checkpoint has no lm_head.weight and does not tie embeddings\n", .{});
            return error.MissingTensor;
        }
        if (m.head.mem.len == 0) {
            // Tied embeddings: the head is the embedding, quantized for the
            // logits projection when quantization is on (its own copy).
            m.head = switch (quant) {
                .none => m.embed,
                .q8_0, .q4_0, .q4x => try quantize(a, m.embed, quant),
            };
        }
        m.arena = arena; // refresh the copy: `a` kept allocating into the local
        try m.checkComplete();
        return m;
    }

    /// Every tensor the config promises is present (undefined never escapes).
    fn checkComplete(m: *const Model) !void {
        const s = m.spec;
        if (m.embed.mem.len == 0 or m.final_norm.len == 0) return error.MissingTensor;
        for (m.layers, 0..) |l, i| {
            const ok = l.input_norm.len == s.hidden and l.post_norm.len == s.hidden and
                l.q_norm.len == s.head_dim and l.k_norm.len == s.head_dim and
                l.q.rows == s.heads * s.head_dim and l.q.cols == s.hidden and
                l.k.rows == s.kv_heads * s.head_dim and l.k.cols == s.hidden and
                l.v.rows == s.kv_heads * s.head_dim and l.v.cols == s.hidden and
                l.o.rows == s.hidden and l.o.cols == s.heads * s.head_dim and
                l.gate.rows == s.inter and l.gate.cols == s.hidden and
                l.up.rows == s.inter and l.up.cols == s.hidden and
                l.down.rows == s.hidden and l.down.cols == s.inter;
            if (!ok) {
                std.debug.print("tensorfold-riscv: layer {d} is missing tensors\n", .{i});
                return error.MissingTensor;
            }
        }
    }

    /// Names this model reads from the checkpoint.
    fn needed(spec: Spec, name: []const u8) bool {
        if (std.mem.startsWith(u8, name, "model.layers.")) return true;
        if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return true;
        if (std.mem.eql(u8, name, "model.norm.weight")) return true;
        if (std.mem.eql(u8, name, "lm_head.weight")) return true;
        _ = spec;
        return false;
    }

    fn take(m: *Model, a: std.mem.Allocator, file: *const st.File) !void {
        for (file.names.items) |name| {
            if (!needed(m.spec, name)) continue;
            const t = file.get(name).?;
            const s = m.spec;
            if (std.mem.eql(u8, name, "model.norm.weight")) {
                m.final_norm = try normLanes(a, t, s.hidden, name);
                continue;
            }
            if (layerIndex(name)) |li| {
                if (li >= s.layers) return error.UnexpectedTensor;
                const l = &m.layers[li];
                const prefix_len = std.fmt.count("model.layers.{d}.", .{li});
                const tail = name[prefix_len..];
                if (std.mem.eql(u8, tail, "self_attn.q_norm.weight")) {
                    l.q_norm = try normLanes(a, t, s.head_dim, name);
                    continue;
                } else if (std.mem.eql(u8, tail, "self_attn.k_norm.weight")) {
                    l.k_norm = try normLanes(a, t, s.head_dim, name);
                    continue;
                } else if (std.mem.eql(u8, tail, "input_layernorm.weight")) {
                    l.input_norm = try normLanes(a, t, s.hidden, name);
                    continue;
                } else if (std.mem.eql(u8, tail, "post_attention_layernorm.weight")) {
                    l.post_norm = try normLanes(a, t, s.hidden, name);
                    continue;
                }
                const w = try normalize(a, t);
                try m.takeLayer(a, l, li, tail, name, w);
                continue;
            }
            // The embedding stays in its checkpoint lanes (a lookup table);
            // lm_head is a projection and quantizes like the layer weights.
            const raw = try normalizeRaw(a, t);
            if (std.mem.eql(u8, name, "model.embed_tokens.weight")) {
                try expectShape(name, raw, s.vocab, s.hidden);
                m.embed = raw;
            } else if (std.mem.eql(u8, name, "lm_head.weight")) {
                try expectShape(name, raw, s.vocab, s.hidden);
                m.head = try normalize(a, t);
            } else return error.UnexpectedTensor;
        }
    }

    /// The layer's projection weights; `tail` is the name without its prefix.
    fn takeLayer(m: *Model, a: std.mem.Allocator, l: *Layer, li: usize, tail: []const u8, name: []const u8, w: Weight) !void {
        _ = a;
        _ = li;
        const s = m.spec;
        if (std.mem.eql(u8, tail, "self_attn.q_proj.weight")) {
            try expectShape(name, w, s.heads * s.head_dim, s.hidden);
            l.q = w;
        } else if (std.mem.eql(u8, tail, "self_attn.k_proj.weight")) {
            try expectShape(name, w, s.kv_heads * s.head_dim, s.hidden);
            l.k = w;
        } else if (std.mem.eql(u8, tail, "self_attn.v_proj.weight")) {
            try expectShape(name, w, s.kv_heads * s.head_dim, s.hidden);
            l.v = w;
        } else if (std.mem.eql(u8, tail, "self_attn.o_proj.weight")) {
            try expectShape(name, w, s.hidden, s.heads * s.head_dim);
            l.o = w;
        } else if (std.mem.eql(u8, tail, "mlp.gate_proj.weight")) {
            try expectShape(name, w, s.inter, s.hidden);
            l.gate = w;
        } else if (std.mem.eql(u8, tail, "mlp.up_proj.weight")) {
            try expectShape(name, w, s.inter, s.hidden);
            l.up = w;
        } else if (std.mem.eql(u8, tail, "mlp.down_proj.weight")) {
            try expectShape(name, w, s.hidden, s.inter);
            l.down = w;
        } else return error.UnexpectedTensor;
    }

    /// tensor name "model.layers.N.rest" -> N
    fn layerIndex(name: []const u8) ?usize {
        const head = "model.layers.";
        if (!std.mem.startsWith(u8, name, head)) return null;
        const rest = name[head.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        return std.fmt.parseInt(usize, rest[0..dot], 10) catch null;
    }
};

fn expectShape(name: []const u8, w: Weight, rows: usize, cols: usize) !void {
    if (w.rows != rows or w.cols != cols) {
        std.debug.print("tensorfold-riscv: {s} is [{d},{d}], the config needs [{d},{d}]\n", .{ name, w.rows, w.cols, rows, cols });
        return error.MismatchedShape;
    }
}

/// Copy a tensor into owned lanes: f32 or bf16 (u16), f16 promoted to f32;
/// projections quantize further when `quant` is set (norms never do).
fn normalize(a: std.mem.Allocator, t: st.Tensor) !Weight {
    const w = try normalizeRaw(a, t);
    return switch (quant) {
        .none => w,
        .q8_0 => quantize(a, w, .q8_0),
        .q4_0 => quantize(a, w, .q4_0),
        .q4x => quantize(a, w, .q4x),
    };
}

/// f32/bf16 lanes to GGUF blocks (k % 32 == 0; q4_0 needs k % 32 == 0 too).
fn quantize(a: std.mem.Allocator, w: Weight, mode: Quant) !Weight {
    const rows = w.rows;
    const k = w.cols;
    if (mode == .q4x) {
        if (k % 256 != 0) return error.QuantShape;
        return quantizeQ4x(a, w, rows, k);
    }
    if (k % 32 != 0) return error.QuantShape;
    const blocks = rows * (k / 32);
    const bytes = switch (mode) {
        .none => unreachable,
        .q8_0 => blocks * 34,
        .q4_0 => blocks * 18,
        .q4x => unreachable,
    };
    const mem = try a.alignedAlloc(u8, .@"8", bytes);
    const src_f32 = w.kind == .f32;
    for (0..rows) |r| {
        for (0..k / 32) |b| {
            var amax: f32 = 0;
            for (0..32) |i| {
                const v = lane(w, r * k + b * 32 + i, src_f32);
                amax = @max(amax, @abs(v));
            }
            if (mode == .q8_0) {
                const d = amax / 127.0;
                const d16: u16 = @bitCast(@as(f16, @floatCast(d)));
                const dst = mem.ptr + (r * (k / 32) + b) * 34;
                @as(*u16, @ptrCast(@alignCast(dst))).* = d16;
                for (0..32) |i| {
                    const v = lane(w, r * k + b * 32 + i, src_f32);
                    var q: i32 = 0;
                    if (d > 0) q = @intFromFloat(@round(v / d));
                    q = @max(-127, @min(127, q));
                    dst[2 + i] = @bitCast(@as(i8, @intCast(q)));
                }
            } else {
                const d = amax / 8.0;
                const d16: u16 = @bitCast(@as(f16, @floatCast(d)));
                const dst = mem.ptr + (r * (k / 32) + b) * 18;
                @as(*u16, @ptrCast(@alignCast(dst))).* = d16;
                for (0..16) |i| {
                    var lo: i32 = 8;
                    var hi: i32 = 8;
                    if (d > 0) {
                        lo = @as(i32, @intFromFloat(@round(lane(w, r * k + b * 32 + 2 * i, src_f32) / d))) + 8;
                        hi = @as(i32, @intFromFloat(@round(lane(w, r * k + b * 32 + 2 * i + 1, src_f32) / d))) + 8;
                    }
                    lo = @max(0, @min(15, lo));
                    hi = @max(0, @min(15, hi));
                    dst[2 + i] = @intCast(lo | (hi << 4));
                }
            }
        }
    }
    const kind: Kind = switch (mode) {
        .q8_0 => .q8_0,
        .q4_0 => .q4_0,
        .q4x => .q4x,
        .none => unreachable,
    };
    return .{ .rows = rows, .cols = k, .kind = kind, .mem = mem };
}

/// q4x: packed for SpacemiT's IME2 dot, tile by tile (zig/kernels/riscv/q4x.c).
/// A row block is 8 output rows: (k/256) groups x (8 rows x f16) scales, then
/// k/32 tiles of 128 bytes, where byte t holds w[r0 + t/16][k0 + t%16] in the
/// low nibble and w[r0 + t/16][k0 + 16 + t%16] in the high nibble. Values are
/// signed 4-bit (q in [-7,7], stored as q+8) with d = amax/7 per 256-value
/// group per row, so one 128-byte load is exactly vmadot's two B operands.
fn quantizeQ4x(a: std.mem.Allocator, w: Weight, rows: usize, k: usize) !Weight {
    const groups = k / 256;
    const block_bytes = 4 * k + k / 16;
    // Rows are packed in blocks of eight (one IME2 dot tile row group); a
    // vocabulary that is not a multiple of eight gets zero padding rows that
    // the kernels compute and the caller never reads.
    const packed_rows = ((rows + 7) / 8) * 8;
    const bytes = (packed_rows / 8) * block_bytes;
    const mem = try a.alignedAlloc(u8, .@"64", bytes);
    @memset(mem, 0);
    const src_f32 = w.kind == .f32;
    for (0..packed_rows / 8) |rb| {
        const block = mem.ptr + rb * block_bytes;
        if (rb * 8 >= rows) break; // zero padding rows stay zero
        const scales: [*]u16 = @ptrCast(@alignCast(block));
        const tiles = block + groups * 8 * 2;
        for (0..groups) |g| {
            for (0..8) |r| {
                if (rb * 8 + r >= rows) break;
                const rowbase = (rb * 8 + r) * k;
                const rowoff = rowbase + g * 256;
                var amax: f32 = 0;
                for (0..256) |i| amax = @max(amax, @abs(lane(w, rowoff + i, src_f32)));
                const d = amax / 7.0;
                scales[g * 8 + r] = @bitCast(@as(f16, @floatCast(d)));
                const inv: f32 = if (d > 0) 1.0 / d else 0;
                for (0..8) |t| {
                    const kc = g * 8 + t; // 32-k chunk
                    const tile = tiles + kc * 128;
                    for (0..16) |u| {
                        var lo: i32 = 0;
                        var hi: i32 = 0;
                        if (d > 0) {
                            lo = q4xRound(lane(w, rowbase + kc * 32 + u, src_f32), inv);
                            hi = q4xRound(lane(w, rowbase + kc * 32 + 16 + u, src_f32), inv);
                        }
                        lo = @max(-8, @min(7, lo)) + 8;
                        hi = @max(-8, @min(7, hi)) + 8;
                        tile[r * 16 + u] = @intCast(lo | (hi << 4));
                    }
                }
            }
        }
    }
    return .{ .rows = rows, .cols = k, .kind = .q4x, .mem = mem };
}

fn q4xRound(v: f32, inv: f32) i32 {
    return @as(i32, @intFromFloat(@round(v * inv)));
}


fn lane(w: Weight, idx: usize, src_f32: bool) f32 {
    return if (src_f32) w.f32Lanes()[idx] else @as(f32, @bitCast(@as(u32, w.bf16Lanes()[idx]) << 16));
}

fn normalizeRaw(a: std.mem.Allocator, t: st.Tensor) !Weight {
    if (t.shape.len != 2) return error.BadShape;
    const rows = t.shape[0];
    const cols = t.shape[1];
    switch (t.dtype) {
        .f32 => {
            const mem = try a.alignedAlloc(u8, .@"8", rows * cols * 4);
            @memcpy(mem, t.bytes[0 .. rows * cols * 4]);
            return .{ .rows = rows, .cols = cols, .kind = .f32, .mem = mem };
        },
        .bf16 => {
            const mem = try a.alignedAlloc(u8, .@"8", rows * cols * 2);
            @memcpy(mem, t.bytes[0 .. rows * cols * 2]);
            return .{ .rows = rows, .cols = cols, .kind = .bf16, .mem = mem };
        },
        .f16, .f64, .i32, .i16, .u8 => {
            const f = try st.toF64(a, t);
            const mem = try a.alignedAlloc(u8, .@"8", rows * cols * 4);
            const lanes = @as([*]f32, @ptrCast(@alignCast(mem.ptr)))[0 .. rows * cols];
            for (f, lanes) |v, *o| o.* = @floatCast(v);
            return .{ .rows = rows, .cols = cols, .kind = .f32, .mem = mem };
        },
    }
}

/// A norm's weight as f32 lanes whatever the checkpoint stores.
fn normLanes(a: std.mem.Allocator, t: st.Tensor, n: usize, name: []const u8) ![]const f32 {
    if (t.shape.len != 1 or t.shape[0] != n) {
        std.debug.print("tensorfold-riscv: {s} has an unexpected shape\n", .{name});
        return error.MismatchedShape;
    }
    const f = try st.toF64(a, t);
    const out = try a.alloc(f32, n);
    for (f, out) |v, *o| o.* = @floatCast(v);
    return out;
}

fn readSpec(gpa: std.mem.Allocator, json: []const u8) !Spec {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const get = struct {
        fn num(obj: std.json.ObjectMap, key: []const u8) !f64 {
            const v = obj.get(key) orelse return error.BadConfig;
            return switch (v) {
                .integer => |i| @floatFromInt(i),
                .float => |f| f,
                else => error.BadConfig,
            };
        }
        fn flag(obj: std.json.ObjectMap, key: []const u8) bool {
            const v = obj.get(key) orelse return false;
            return v == .bool and v.bool;
        }
    };
    const hidden: usize = @intFromFloat(try get.num(o, "hidden_size"));
    const heads: usize = @intFromFloat(try get.num(o, "num_attention_heads"));
    const kv_heads: usize = if (o.get("num_key_value_heads") != null)
        @intFromFloat(try get.num(o, "num_key_value_heads"))
    else
        heads;
    const head_dim: usize = if (o.get("head_dim") != null)
        @intFromFloat(try get.num(o, "head_dim"))
    else
        hidden / heads;
    return .{
        .hidden = hidden,
        .inter = @intFromFloat(try get.num(o, "intermediate_size")),
        .layers = @intFromFloat(try get.num(o, "num_hidden_layers")),
        .heads = heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .vocab = @intFromFloat(try get.num(o, "vocab_size")),
        .tied = get.flag(o, "tie_word_embeddings"),
        .eps = @floatCast(if (o.get("rms_norm_eps")) |_| try get.num(o, "rms_norm_eps") else 1e-6),
        .theta = @floatCast(try get.num(o, "rope_theta")),
    };
}

/// One layer's attention for the pool's head shards.
const AttnCtx = struct {
    session: *Session,
    layer: usize,
    q: []const f32,
    attn: []f32,
    T: usize,
    scale: f32,
};

fn attnShard(ctx: ?*anyopaque, begin: c_int, end: c_int) callconv(.c) void {
    const a: *AttnCtx = @ptrCast(@alignCast(ctx.?));
    attnShardRun(a, begin, end);
}

fn attnShardRun(a: *AttnCtx, begin: c_int, end: c_int) void {
    const s = a.session;
    const spec = s.model.spec;
    const qw = spec.heads * spec.head_dim;
    const hd = spec.head_dim;
    for (@as(usize, @intCast(begin))..@as(usize, @intCast(end))) |h| {
        const kvh = h / (spec.heads / spec.kv_heads);
        const scratch = s.scratch_rows[h * s.ctx ..][0..s.ctx];
        for (0..a.T) |t| {
            const rows = s.pos + t + 1;
            const krows = s.panel(s.k_cache, a.layer, kvh)[0 .. rows * hd];
            const vrows = s.panel(s.v_cache, a.layer, kvh)[0 .. rows * hd];
            c.tf_riscv_attention_f32(a.q[t * qw + h * hd ..].ptr, krows.ptr, vrows.ptr, a.attn[t * qw + h * hd ..].ptr, @intCast(rows), @intCast(hd), a.scale, scratch.ptr);
        }
    }
}

/// Decode (m == 1) q4x shard: output rows [begin, end) of one projection.
const GemvQ4xCtx = struct {
    xa: [*]const i8,
    xs: [*]const f32,
    raw: [*]const u8,
    out: [*]f32,
    n: usize,
    k: usize,
};

fn gemvQ4xShard(ctx: ?*anyopaque, begin: c_int, end: c_int) callconv(.c) void {
    const p: *GemvQ4xCtx = @ptrCast(@alignCast(ctx.?));
    c.tf_riscv_gemv_q4x_shard(p.xa, p.xs, p.raw, p.out, @intCast(p.n), @intCast(p.k), begin, end);
}

/// Prefill (m > 1) q4x shard: output rows [begin, end) of one projection.
const GemmQ4xCtx = struct {
    xa: [*]const i8,
    xs: [*]const f32,
    raw: [*]const u8,
    out: [*]f32,
    m: usize,
    n: usize,
    k: usize,
};

fn gemmQ4xShard(ctx: ?*anyopaque, begin: c_int, end: c_int) callconv(.c) void {
    const p: *GemmQ4xCtx = @ptrCast(@alignCast(ctx.?));
    c.tf_riscv_gemm_q4x_shard(p.xa, p.xs, p.raw, p.out, @intCast(p.m), @intCast(p.n), @intCast(p.k), begin, end);
}

/// Decode fusion: several projections that share one activation run as a
/// single sharded dispatch. The shard index range maps onto the sub-
/// projections' output rows in order, so one pool run replaces N runs and
/// the activation quantizes once for the whole set.
const FusedProj = struct {
    raw: [*]const u8,
    out: [*]f32,
    n: usize,
};

const FusedCtx = struct {
    xa: [*]const i8,
    xs: [*]const f32,
    k: usize,
    projs: []const FusedProj,
};

fn fusedShard(ctx: ?*anyopaque, begin: c_int, end: c_int) callconv(.c) void {
    const f: *FusedCtx = @ptrCast(@alignCast(ctx.?));
    const b: usize = @intCast(begin);
    const e: usize = @intCast(end);
    var base: usize = 0;
    for (f.projs) |p| {
        const lo = @max(b, base); // clamp: the shard range may start before
        const hi = @min(e, base + p.n); // or end after this projection
        if (hi > lo) {
            const j0: c_int = @intCast(lo - base);
            const j1: c_int = @intCast(hi - base);
            c.tf_riscv_gemv_q4x_shard(f.xa, f.xs, p.raw, p.out, @intCast(p.n), @intCast(f.k), j0, j1);
        }
        base += p.n;
    }
}

/// Decode: run `projs` projections over one activation in one dispatch.
fn fusedProject(x: []const f32, k: usize, projs: []const FusedProj) void {
    const pad = q4xActivation(k, 1);
    c.tf_riscv_quant_a(x.ptr, pad.xa.ptr, pad.xs.ptr, 1, @intCast(k));
    var total: usize = 0;
    for (projs) |p| total += p.n;
    if (pool.active() > 1) {
        var ctx: FusedCtx = .{ .xa = pad.xa.ptr, .xs = pad.xs.ptr, .k = k, .projs = projs };
        pool.run(fusedShard, &ctx, total);
        return;
    }
    var f: FusedCtx = .{ .xa = pad.xa.ptr, .xs = pad.xs.ptr, .k = k, .projs = projs };
    fusedShard(&f, 0, @intCast(total));
}

/// Module-level scratch for the quantized activation (int8 tiles plus the
/// per-row group scales). One session per process, so it is never shared.
const Activation = struct { xa: []i8, xs: []f32 };
var act_buf: []align(64) u8 = &.{};

fn q4xActivation(k: usize, rows: usize) Activation {
    const blocks = (rows + 7) / 8;
    const need = blocks * 8 * k + rows * (k / 256) * @sizeOf(f32);
    if (act_buf.len < need) {
        if (act_buf.len > 0) std.heap.page_allocator.free(act_buf);
        act_buf = std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(64), need) catch &.{};
    }
    const xa = act_buf[0 .. blocks * 8 * k];
    const xs: []f32 = @alignCast(@as([*]f32, @ptrCast(@alignCast(act_buf.ptr + blocks * 8 * k)))[0 .. rows * (k / 256)]);
    return .{ .xa = @alignCast(@as([*]i8, @ptrCast(xa.ptr))[0 .. xa.len]), .xs = xs };
}

/// One projection for the pool's shard trampoline.
const ProjectCtx = struct {
    x: [*]const f32,
    bf16_lanes: ?[*]const u16,
    f32_lanes: ?[*]const f32,
    raw: ?[*]const u8 = null,
    out: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    bf16: bool,
    q8: bool = false,
    quant: bool = false,
};

/// A pool shard: output columns [begin, end) of one projection.
fn projectShard(ctx: ?*anyopaque, begin: c_int, end: c_int) callconv(.c) void {
    const p: *ProjectCtx = @ptrCast(@alignCast(ctx.?));
    if (p.quant) {
        if (p.q8) {
            c.tf_riscv_matmul_q8_0_shard(p.x, p.raw.?, p.out, @intCast(p.m), @intCast(p.n), @intCast(p.k), begin, end);
        } else {
            c.tf_riscv_matmul_q4_0_shard(p.x, p.raw.?, p.out, @intCast(p.m), @intCast(p.n), @intCast(p.k), begin, end);
        }
    } else if (p.bf16) {
        c.tf_riscv_matmul_bf16_shard(p.x, p.bf16_lanes.?, p.out, @intCast(p.m), @intCast(p.n), @intCast(p.k), begin, end);
    } else {
        c.tf_riscv_matmul_f32_shard(p.x, p.f32_lanes.?, p.out, @intCast(p.m), @intCast(p.n), @intCast(p.k), 1, begin, end);
    }
}

// ---- phase profiling (TF_RISCV_PROFILE=1) ---------------------------------
// Cumulated nanoseconds per phase, printed at each forward() return. The
// categories are the ones the performance report's gap decomposition claims:
// verify them, do not trust them. Phase 0 is the first (prefill) call, 1 all
// decode calls after it.
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
var profile_on: bool = false;
var profile_calls: [2]u64 = .{ 0, 0 };
var profile_ns: [2][8]u64 = .{ .{ 0, 0, 0, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 } };
var profile_aux: [2]u64 = .{ 0, 0 }; // projection calls counted

extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;
const Timespec = extern struct { tv_sec: isize, tv_nsec: isize };

inline fn nowNs() u64 {
    var ts: Timespec = .{ .tv_sec = 0, .tv_nsec = 0 };
    _ = clock_gettime(1, &ts); // CLOCK_MONOTONIC
    return @as(u64, @bitCast(ts.tv_sec)) *% 1_000_000_000 +% @as(u64, @bitCast(ts.tv_nsec));
}

inline fn profAccum(phase: usize, cat: usize, t0: u64) void {
    if (profile_on) profile_ns[phase][cat] +%= nowNs() -% t0;
}

var profile_phase_now: usize = 0;

pub var qwen3_dump = false;

const FileC = opaque {};
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*FileC;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, n: usize, f: *FileC) usize;
extern "c" fn fclose(f: *FileC) c_int;

fn dumpBytes(path: []const u8, xs: []const u8) void {
    var z: [65:0]u8 = undefined;
    if (path.len > 64) return;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const f = fopen(&z, "wb") orelse return;
    _ = fwrite(xs.ptr, 1, xs.len, f);
    _ = fclose(f);
}

fn dumpTensor(path: []const u8, xs: []const f32) void {
    var z: [65:0]u8 = undefined;
    if (path.len > 64) return;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const f = fopen(&z, "wb") orelse return;
    _ = fwrite(xs.ptr, 4, xs.len, f);
    _ = fclose(f);
}

/// Decode (session has tokens cached) or prefill, for the profiler.
fn profilePhaseForProject() usize {
    return profile_phase_now;
}

pub fn profileEnable() void {
    profile_on = getenv("TF_RISCV_PROFILE") != null;
}

fn profPrint() void {
    if (!profile_on) return;
    const names = [8][]const u8{ "project(qkvo/gud)", "quant_a", "attention", "norms+rope+silu", "kv-cache-writes", "embed/head", "project-dispatch", "other" };
    for (0..2) |ph| {
        if (profile_calls[ph] == 0) continue;
        const denom: f64 = @floatFromInt(profile_calls[ph]);
        std.debug.print("PROFILE phase {d} calls {d}: ", .{ ph, profile_calls[ph] });
        for (0..8) |cat| std.debug.print("{s}={d:.2}us ", .{ names[cat], @as(f64, @floatFromInt(profile_ns[ph][cat])) / denom / 1e3 });
        std.debug.print("| proj_calls={d}\n", .{profile_aux[ph]});
    }
}

/// One generation session: per-layer, per-KV-head caches the kernel
/// attention reads as contiguous [rows, head_dim] panels.
pub const Session = struct {
    model: *const Model,
    /// Cache capacity in tokens (positions).
    ctx: usize,
    pos: usize = 0,
    k_cache: []f32,
    v_cache: []f32,
    /// Per-head score rows for attention (heads * ctx floats).
    scratch_rows: []f32,
    scratch: std.heap.ArenaAllocator,

    pub fn init(gpa: std.mem.Allocator, model: *const Model, ctx: usize) !Session {
        const s = model.spec;
        const per = s.kv_heads * ctx * s.head_dim;
        const total = s.layers * per;
        return .{
            .model = model,
            .ctx = ctx,
            .k_cache = try gpa.alloc(f32, total),
            .v_cache = try gpa.alloc(f32, total),
            .scratch_rows = try gpa.alloc(f32, s.heads * ctx),
            .scratch = .init(gpa),
        };
    }

    pub fn deinit(s: *Session, gpa: std.mem.Allocator) void {
        s.scratch.deinit();
        gpa.free(s.k_cache);
        gpa.free(s.v_cache);
        gpa.free(s.scratch_rows);
        s.* = undefined;
    }

    /// The cache rows of (layer, kv head) so far: contiguous [rows, head_dim].
    fn panel(s: *const Session, cache: []f32, li: usize, kvh: usize) []f32 {
        const per = s.model.spec.kv_heads * s.ctx * s.model.spec.head_dim;
        return cache[li * per + kvh * s.ctx * s.model.spec.head_dim ..][0 .. s.ctx * s.model.spec.head_dim];
    }

    /// ids through every layer; logits rows land in out (vocab per row).
    /// logits_all writes one row per input id, else only the last row.
    pub fn forward(s: *Session, ids: []const u32, out: []f32, logits_all: bool) !void {
        const phase: usize = if (s.pos == 0) 0 else 1;
        profile_calls[phase] += 1;
        profile_phase_now = phase;
        const m = s.model;
        const spec = m.spec;
        if (ids.len == 0 or s.pos + ids.len > s.ctx) return error.ContextFull;
        const T = ids.len;
        const qw = spec.heads * spec.head_dim;
        const kvw = spec.kv_heads * spec.head_dim;

        // Fresh scratch per call (freed at the next call or deinit).
        _ = s.scratch.reset(.retain_capacity);
        const a = s.scratch.allocator();
        const x = try a.alloc(f32, T * spec.hidden);
        const normed = try a.alloc(f32, T * spec.hidden);
        const q = try a.alloc(f32, T * qw);
        const k = try a.alloc(f32, T * kvw);
        const v = try a.alloc(f32, T * kvw);
        const attn = try a.alloc(f32, T * qw);
        const o = try a.alloc(f32, T * spec.hidden);
        const gate = try a.alloc(f32, T * spec.inter);
        const up = try a.alloc(f32, T * spec.inter);
        const act = try a.alloc(f32, T * spec.inter);

        if (qwen3_dump) {
            dumpTensor("embed-row.bin", m.embed.f32Lanes()[3 * spec.hidden ..][0..spec.hidden]);
            const er = m.embed.f32Lanes();
            std.debug.print("DUMP embed ptr {*} first4 {e} {e} {e} {e}\n", .{ m.embed.mem.ptr, er[768], er[769], er[770], er[771] });
        }
        for (ids, 0..) |id, t| {
            if (id >= spec.vocab) return error.BadToken;
            m.tableRow(m.embed, id, x[t * spec.hidden ..][0..spec.hidden]);
        }
        if (qwen3_dump) {
            std.debug.print("DUMP x ptr {*} first4 {e} {e} {e} {e}\n", .{ x.ptr, x[0], x[1], x[2], x[3] });
        }

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(spec.head_dim)));
        for (m.layers, 0..) |*l, li| {
            var t = nowNs();
            for (0..T) |t_| c.tf_riscv_rmsnorm(x[t_ * spec.hidden ..].ptr, l.input_norm.ptr, normed[t_ * spec.hidden ..].ptr, @intCast(spec.hidden), spec.eps);
            if (qwen3_dump and li == 0 and s.pos == 0) {
                dumpTensor("normed.bin", normed);
                dumpTensor("x0.bin", x);
            }
            m.project(l.q, normed, q, T);
            if (qwen3_dump and li == 0 and s.pos == 0) {
                dumpTensor("qout.bin", q);
                dumpBytes("qweight.bin", l.q.mem);
            }
            m.project(l.k, normed, k, T);
            m.project(l.v, normed, v, T);
            for (0..T) |t_| {
                const pos: u32 = @intCast(s.pos + t_);
                for (0..spec.heads) |h| c.tf_riscv_rmsnorm(q[t_ * qw + h * spec.head_dim ..].ptr, l.q_norm.ptr, q[t_ * qw + h * spec.head_dim ..].ptr, @intCast(spec.head_dim), spec.eps);
                for (0..spec.kv_heads) |h| c.tf_riscv_rmsnorm(k[t_ * kvw + h * spec.head_dim ..].ptr, l.k_norm.ptr, k[t_ * kvw + h * spec.head_dim ..].ptr, @intCast(spec.head_dim), spec.eps);
                c.tf_riscv_rope(q[t_ * qw ..].ptr, k[t_ * kvw ..].ptr, @intCast(pos), @intCast(spec.heads), @intCast(spec.kv_heads), @intCast(spec.head_dim), spec.theta);
            }
            profAccum(phase, 3, t);
            t = nowNs();
            for (0..T) |t_| {
                for (0..spec.kv_heads) |h| {
                    const pos: u32 = @intCast(s.pos + t_);
                    const dst_k = s.panel(s.k_cache, li, h)[pos * spec.head_dim ..][0..spec.head_dim];
                    const dst_v = s.panel(s.v_cache, li, h)[pos * spec.head_dim ..][0..spec.head_dim];
                    @memcpy(dst_k, k[t_ * kvw + h * spec.head_dim ..][0..spec.head_dim]);
                    @memcpy(dst_v, v[t_ * kvw + h * spec.head_dim ..][0..spec.head_dim]);
                }
            }
            profAccum(phase, 4, t);
            // Attention over the cache: sharded across heads when the pool is
            // up (each head is independent), one score row per head in the
            // session's scratch.
            var actx: AttnCtx = .{
                .session = s,
                .layer = li,
                .q = q,
                .attn = attn,
                .T = T,
                .scale = scale,
            };
            t = nowNs();
            if (pool.active() > 1) {
                pool.run(attnShard, &actx, spec.heads);
            } else {
                attnShardRun(&actx, 0, @intCast(spec.heads));
            }
            profAccum(phase, 2, t);
            t = nowNs();
            if (qwen3_dump and li == 0 and s.pos == 0) dumpTensor("attn-in.bin", attn);
            m.project(l.o, attn, o, T);
            if (qwen3_dump and li == 0 and s.pos == 0) {
                dumpTensor("o-out.bin", o);
                dumpBytes("o-w.bin", l.o.mem);
            }
            for (0..T) |t_| c.tf_riscv_add(x[t_ * spec.hidden ..].ptr, o[t_ * spec.hidden ..].ptr, x[t_ * spec.hidden ..].ptr, @intCast(spec.hidden));

            for (0..T) |t_| c.tf_riscv_rmsnorm(x[t_ * spec.hidden ..].ptr, l.post_norm.ptr, normed[t_ * spec.hidden ..].ptr, @intCast(spec.hidden), spec.eps);
            if (qwen3_dump and li == 0 and s.pos == 0) {
                dumpTensor("gate-in.bin", normed);
                dumpBytes("gate-w.bin", l.gate.mem);
            }
            if (T == 1 and l.gate.kind == .q4x) {
                var projs: [2]FusedProj = undefined;
                projs[0] = .{ .raw = @ptrCast(@alignCast(l.gate.mem.ptr)), .out = gate.ptr, .n = l.gate.rows };
                projs[1] = .{ .raw = @ptrCast(@alignCast(l.up.mem.ptr)), .out = up.ptr, .n = l.up.rows };
                fusedProject(normed, spec.hidden, &projs);
            } else {
                m.project(l.gate, normed, gate, T);
                m.project(l.up, normed, up, T);
            }
            if (qwen3_dump and li == 0 and s.pos == 0) dumpTensor("gate-out.bin", gate);
            for (0..T) |t_| c.tf_riscv_silu_mul(gate[t_ * spec.inter ..].ptr, up[t_ * spec.inter ..].ptr, act[t_ * spec.inter ..].ptr, @intCast(spec.inter));
            m.project(l.down, act, o, T);
            for (0..T) |t_| c.tf_riscv_add(x[t_ * spec.hidden ..].ptr, o[t_ * spec.hidden ..].ptr, x[t_ * spec.hidden ..].ptr, @intCast(spec.hidden));
            profAccum(phase, 0, t);
        }

        // Final norm and the head, on every row or just the chunk's last.
        const t = nowNs();
        const rows_out: usize = if (logits_all) T else 1;
        for (0..rows_out) |t_| {
            const src = if (logits_all) t_ else T - 1;
            const h = try a.alloc(f32, spec.hidden);
            c.tf_riscv_rmsnorm(x[src * spec.hidden ..].ptr, m.final_norm.ptr, h.ptr, @intCast(spec.hidden), spec.eps);
            m.project(m.head, h, out[t_ * spec.vocab ..][0..spec.vocab], 1);
        }
        profAccum(phase, 5, t);
        s.pos += T;
        profPrint();
    }
};

/// Greedy or sampled pick from one logits row./// Greedy or sampled pick from one logits row.
pub fn sampleOne(logits: []const f32, temperature: f32, top_p: f32, rng: std.Random) u32 {
    if (temperature <= 0) {
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[best]) best = i;
        }
        return @intCast(best);
    }
    // Temperature then nucleus over a scratch copy (allocations here are the
    // CLI's convenience path; the hot loops never allocate).
    const a = std.heap.page_allocator;
    const scaled = a.alloc(f32, logits.len) catch return 0;
    defer a.free(scaled);
    const inv: f32 = 1.0 / temperature;
    var max: f32 = -std.math.floatMax(f32);
    for (logits, scaled) |v, *o| {
        o.* = v * inv;
        max = @max(max, o.*);
    }
    var sum: f64 = 0;
    for (scaled) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    // Nucleus: keep the smallest prefix with mass >= top_p (sorted desc).
    const order = a.alloc(u32, logits.len) catch return 0;
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, scaled, struct {
        fn less(ctx: []const f32, x: u32, y: u32) bool {
            return ctx[x] > ctx[y];
        }
    }.less);
    var kept: usize = order.len;
    var mass: f64 = 0;
    for (order, 0..) |id, i| {
        mass += scaled[id] / sum;
        if (mass >= top_p) {
            kept = i + 1;
            break;
        }
    }
    var roll = rng.float(f64) * mass;
    for (order[0..kept]) |id| {
        roll -= scaled[id] / sum;
        if (roll <= 0) return id;
    }
    return order[kept - 1];
}

test "qwen3 q4x packing round-trips through the kernel dot" {
    const t = std.testing;
    const a = t.allocator;
    const rows = 16;
    const k = 512;
    // A deterministic random weight matrix.
    var prng: std.Random.DefaultPrng = .init(7);
    const rnd = prng.random();
    const src = try a.alloc(f32, rows * k);
    defer a.free(src);
    for (src) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
    const w = Weight{ .rows = rows, .cols = k, .kind = .f32, .mem = std.mem.sliceAsBytes(src) };
    const packed_w = try quantizeQ4x(a, w, rows, k);
    defer a.free(@as([]align(64) u8, @alignCast(@constCast(packed_w.mem))));

    // A unit activation at k selects exactly that dequantized weight.
    const x = try a.alloc(f32, k);
    defer a.free(x);
    const xa = try a.alloc(i8, 8 * k);
    defer a.free(xa);
    const xs = try a.alloc(f32, k / 256);
    defer a.free(xs);
    const out = try a.alloc(f32, rows);
    defer a.free(out);
    for (0..k) |kk| {
        @memset(x, 0);
        x[kk] = 1.0;
        c.tf_riscv_quant_a(x.ptr, xa.ptr, xs.ptr, 1, k);
        c.tf_riscv_gemv_q4x_shard_scalar(xa.ptr, xs.ptr, packed_w.mem.ptr, out.ptr, rows, k, 0, rows);
        for (0..rows) |r| {
            // Expected: the same quantization rule the loader applies.
            var amax: f32 = 0;
            const g0 = (kk / 256) * 256;
            for (src[r * k + g0 ..][0..256]) |v| amax = @max(amax, @abs(v));
            const d: f32 = @floatCast(@as(f16, @floatCast(amax / 7.0)));
            var q: f32 = 0;
            if (amax > 0) q = @round(src[r * k + kk] / (amax / 7.0));
            q = @max(-8, @min(7, q));
            const want = q * d;
            if (@abs(out[r] - want) > 1e-3 * @max(@abs(want), 1.0)) {
                std.debug.print("q4x pack mismatch r={d} k={d}: got {e} want {e}\n", .{ r, kk, out[r], want });
                return error.TestExpectedEqual;
            }
        }
    }
}
