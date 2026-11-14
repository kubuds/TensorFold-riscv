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

/// Weight storage: f32 lanes or bf16 lanes (u16), row-major [rows, cols].
pub const Kind = enum { f32, bf16 };

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
    /// product is f32 with several rows, else the backend's kernels.
    pub fn project(m: *const Model, w: Weight, x: []const f32, out: []f32, rows: usize) void {
        _ = m;
        switch (w.kind) {
            .bf16 => c.tf_riscv_matmul_bf16(x.ptr, w.bf16Lanes().ptr, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols)),
            .f32 => {
                if (rows >= 4 and blas.gemmRowMajorTN(rows, w.rows, w.cols, x.ptr, w.f32Lanes().ptr, out.ptr)) return;
                c.tf_riscv_matmul_f32(x.ptr, w.f32Lanes().ptr, out.ptr, @intCast(rows), @intCast(w.rows), @intCast(w.cols), 1);
            },
        }
    }

    /// One row of a [rows, cols] weight table into f32 (embedding lookup).
    pub fn tableRow(m: *const Model, w: Weight, row: usize, out: []f32) void {
        _ = m;
        switch (w.kind) {
            .f32 => @memcpy(out, w.f32Lanes()[row * w.cols ..][0..w.cols]),
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
        var arena: std.heap.ArenaAllocator = .init(gpa);
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
            const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
            defer parsed.deinit();
            const map = parsed.value.object.get("weight_map") orelse return error.BadIndex;
            var it = map.object.iterator();
            while (it.next()) |e| try wanted.put(gpa, try a.dupe(u8, e.key_ptr.*), try a.dupe(u8, e.value_ptr.string));
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
        if (m.head.mem.len == 0) m.head = m.embed;
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
            const w = try normalize(a, t);
            if (std.mem.eql(u8, name, "model.embed_tokens.weight")) {
                try expectShape(name, w, s.vocab, s.hidden);
                m.embed = w;
            } else if (std.mem.eql(u8, name, "lm_head.weight")) {
                try expectShape(name, w, s.vocab, s.hidden);
                m.head = w;
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

/// Copy a tensor into owned lanes: f32 or bf16 (u16), f16 promoted to f32.
fn normalize(a: std.mem.Allocator, t: st.Tensor) !Weight {
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

/// One generation session: per-layer, per-KV-head caches the kernel
/// attention reads as contiguous [rows, head_dim] panels.
pub const Session = struct {
    model: *const Model,
    /// Cache capacity in tokens (positions).
    ctx: usize,
    pos: usize = 0,
    k_cache: []f32,
    v_cache: []f32,
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
            .scratch = .init(gpa),
        };
    }

    pub fn deinit(s: *Session, gpa: std.mem.Allocator) void {
        s.scratch.deinit();
        gpa.free(s.k_cache);
        gpa.free(s.v_cache);
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

        for (ids, 0..) |id, t| {
            if (id >= spec.vocab) return error.BadToken;
            m.tableRow(m.embed, id, x[t * spec.hidden ..][0..spec.hidden]);
        }

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(spec.head_dim)));
        for (m.layers, 0..) |*l, li| {
            for (0..T) |t| c.tf_riscv_rmsnorm(x[t * spec.hidden ..].ptr, l.input_norm.ptr, normed[t * spec.hidden ..].ptr, @intCast(spec.hidden), spec.eps);
            m.project(l.q, normed, q, T);
            m.project(l.k, normed, k, T);
            m.project(l.v, normed, v, T);
            for (0..T) |t| {
                const pos: u32 = @intCast(s.pos + t);
                for (0..spec.heads) |h| c.tf_riscv_rmsnorm(q[t * qw + h * spec.head_dim ..].ptr, l.q_norm.ptr, q[t * qw + h * spec.head_dim ..].ptr, @intCast(spec.head_dim), spec.eps);
                for (0..spec.kv_heads) |h| c.tf_riscv_rmsnorm(k[t * kvw + h * spec.head_dim ..].ptr, l.k_norm.ptr, k[t * kvw + h * spec.head_dim ..].ptr, @intCast(spec.head_dim), spec.eps);
                c.tf_riscv_rope(q[t * qw ..].ptr, k[t * kvw ..].ptr, @intCast(pos), @intCast(spec.heads), @intCast(spec.kv_heads), @intCast(spec.head_dim), spec.theta);
                for (0..spec.kv_heads) |h| {
                    const dst_k = s.panel(s.k_cache, li, h)[pos * spec.head_dim ..][0..spec.head_dim];
                    const dst_v = s.panel(s.v_cache, li, h)[pos * spec.head_dim ..][0..spec.head_dim];
                    @memcpy(dst_k, k[t * kvw + h * spec.head_dim ..][0..spec.head_dim]);
                    @memcpy(dst_v, v[t * kvw + h * spec.head_dim ..][0..spec.head_dim]);
                }
            }
            for (0..T) |t| {
                const pos: usize = s.pos + t;
                for (0..spec.heads) |h| {
                    const kvh = h / (spec.heads / spec.kv_heads);
                    const rows: usize = pos + 1;
                    const krows = s.panel(s.k_cache, li, kvh)[0 .. rows * spec.head_dim];
                    const vrows = s.panel(s.v_cache, li, kvh)[0 .. rows * spec.head_dim];
                    c.tf_riscv_attention_f32(q[t * qw + h * spec.head_dim ..].ptr, krows.ptr, vrows.ptr, attn[t * qw + h * spec.head_dim ..].ptr, @intCast(rows), @intCast(spec.head_dim), scale);
                }
            }
            m.project(l.o, attn, o, T);
            for (0..T) |t| c.tf_riscv_add(x[t * spec.hidden ..].ptr, o[t * spec.hidden ..].ptr, x[t * spec.hidden ..].ptr, @intCast(spec.hidden));

            for (0..T) |t| c.tf_riscv_rmsnorm(x[t * spec.hidden ..].ptr, l.post_norm.ptr, normed[t * spec.hidden ..].ptr, @intCast(spec.hidden), spec.eps);
            m.project(l.gate, normed, gate, T);
            m.project(l.up, normed, up, T);
            for (0..T) |t| c.tf_riscv_silu_mul(gate[t * spec.inter ..].ptr, up[t * spec.inter ..].ptr, act[t * spec.inter ..].ptr, @intCast(spec.inter));
            m.project(l.down, act, o, T);
            for (0..T) |t| c.tf_riscv_add(x[t * spec.hidden ..].ptr, o[t * spec.hidden ..].ptr, x[t * spec.hidden ..].ptr, @intCast(spec.hidden));
        }

        // Final norm and the head, on every row or just the chunk's last.
        const rows_out: usize = if (logits_all) T else 1;
        for (0..rows_out) |t| {
            const src = if (logits_all) t else T - 1;
            const h = try a.alloc(f32, spec.hidden);
            c.tf_riscv_rmsnorm(x[src * spec.hidden ..].ptr, m.final_norm.ptr, h.ptr, @intCast(spec.hidden), spec.eps);
            m.project(m.head, h, out[t * spec.vocab ..][0..spec.vocab], 1);
        }
        s.pos += T;
    }
};

/// Greedy or sampled pick from one logits row.
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
