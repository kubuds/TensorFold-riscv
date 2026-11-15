//! reference.zig - float64 golden math for the RISC-V kernels, the oracle
//! zig/src/riscv/kernel_test.zig checks every kernel family against.
//! Conventions mirror zig/kernels/riscv/tf_riscv.h exactly (HF Qwen3 math).
const std = @import("std");

pub fn rmsnorm(x: []const f64, w: ?[]const f64, out: []f64, eps: f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, out, 0..) |v, *o, i| o.* = v * inv * (if (w) |ww| ww[i] else 1.0);
}

/// rotate_half over one head: out[i] = x[i]*c[i] - x[half+i]*s[i], etc.
pub fn ropeHead(h: []f64, pos: u32, theta: f64) void {
    const half = h.len / 2;
    for (0..half) |i| {
        const freq = std.math.pow(f64, theta, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(h.len)));
        const c = @cos(@as(f64, @floatFromInt(pos)) * freq);
        const s = @sin(@as(f64, @floatFromInt(pos)) * freq);
        const a = h[i];
        const b = h[half + i];
        h[i] = a * c - b * s;
        h[half + i] = b * c + a * s;
    }
}

pub fn rope(q: []f64, k: []f64, pos: u32, q_heads: usize, kv_heads: usize, head_dim: usize, theta: f64) void {
    for (0..q_heads) |hh| ropeHead(q[hh * head_dim ..][0..head_dim], pos, theta);
    for (0..kv_heads) |hh| ropeHead(k[hh * head_dim ..][0..head_dim], pos, theta);
}

pub fn softmax(x: []f64) void {
    var max: f64 = -std.math.floatMax(f64);
    for (x) |v| max = @max(max, v);
    var sum: f64 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

pub fn siluMul(gate: []const f64, up: []const f64, out: []f64) void {
    for (gate, up, out) |g, u, *o| o.* = (g / (1.0 + @exp(-g))) * u;
}

pub fn add(a: []const f64, b: []const f64, out: []f64) void {
    for (a, b, out) |x, y, *o| o.* = x + y;
}

/// out[m,n] = a[m,k] @ b[k,n], or @ b[n,k]^T when trans_b.
pub fn matmulF32(a: []const f64, b: []const f64, out: []f64, m: usize, n: usize, k: usize, trans_b: bool) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var s: f64 = 0;
            for (0..k) |t| {
                const bv = if (trans_b) b[j * k + t] else b[t * n + j];
                s += a[i * k + t] * bv;
            }
            out[i * n + j] = s;
        }
    }
}

pub fn bf16ToF64(b: u16) f64 {
    const bits: u32 = @as(u32, b) << 16;
    return @as(f32, @bitCast(bits));
}

/// out[m,n] = x[m,k] @ w[n,k]^T with bf16 weights.
pub fn matmulBf16(x: []const f64, w: []const u16, out: []f64, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var s: f64 = 0;
            for (0..k) |t| s += x[i * k + t] * bf16ToF64(w[j * k + t]);
            out[i * n + j] = s;
        }
    }
}

/// One query row's attention over `rows` cached keys/values.
pub fn attentionF32(q: []const f64, k: []const f64, v: []const f64, out: []f64, rows: usize, head_dim: usize, scale: f64) void {
    var scores: std.ArrayListUnmanaged(f64) = .empty;
    defer scores.deinit(std.heap.page_allocator);
    scores.resize(std.heap.page_allocator, rows) catch unreachable;
    for (0..rows) |t| {
        var s: f64 = 0;
        for (0..head_dim) |i| s += q[i] * k[t * head_dim + i];
        scores.items[t] = scale * s;
    }
    softmax(scores.items);
    for (0..head_dim) |i| {
        var o: f64 = 0;
        for (0..rows) |t| o += scores.items[t] * v[t * head_dim + i];
        out[i] = o;
    }
}

pub fn halfToF64(h: u16) f64 {
    return @as(f32, @bitCast(halfToF32Bits(h)));
}

fn halfToF32Bits(h: u16) u32 {
    const sign: u32 = @as(u32, h & 0x8000) << 16;
    const exp: u32 = (h >> 10) & 0x1f;
    const man: u32 = h & 0x3ff;
    if (exp == 0) {
        if (man == 0) return sign;
        var e: u32 = 127 - 15 + 1;
        var m: u32 = man;
        while ((m & 0x400) == 0) : (e -= 1) m <<= 1;
        return sign | (e << 23) | ((m & 0x3ff) << 13);
    } else if (exp == 31) {
        return sign | 0x7f800000 | (man << 13);
    } else {
        return sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
}

/// out[n] = x[k] . w[n,k]^T over GGUF q8_0 blocks (32 xi8 + f16 scale).
pub fn gemvQ80(x: []const f64, w: []const u8, out: []f64, n: usize, k: usize) void {
    const qk = 32;
    for (0..n) |j| {
        const row = w[j * (k / qk) * 34 ..];
        var s: f64 = 0;
        for (0..k / qk) |b| {
            const sh = std.mem.readInt(u16, row[b * 34 ..][0..2], .little);
            const d = halfToF64(sh);
            const q = @as([*]const i8, @ptrCast(@alignCast(&row[b * 34 + 2])));
            for (0..qk) |i| s += x[b * qk + i] * d * @as(f64, @floatFromInt(q[i]));
        }
        out[j] = s;
    }
}

/// out[n] = x[k] . w[n,k]^T over GGUF q4_0 blocks (16 packed bytes + f16 scale).
pub fn gemvQ40(x: []const f64, w: []const u8, out: []f64, n: usize, k: usize) void {
    const qk = 32;
    for (0..n) |j| {
        const row = w[j * (k / qk) * 18 ..];
        var s: f64 = 0;
        for (0..k / qk) |b| {
            const sh = std.mem.readInt(u16, row[b * 18 ..][0..2], .little);
            const d = halfToF64(sh);
            const q = row[b * 18 + 2 ..][0..16];
            for (0..qk / 2) |i| {
                s += x[b * qk + 2 * i] * d * (@as(f64, @floatFromInt(q[i] & 0x0f)) - 8.0);
                s += x[b * qk + 2 * i + 1] * d * (@as(f64, @floatFromInt(q[i] >> 4)) - 8.0);
            }
        }
        out[j] = s;
    }
}

/// out[n] = x[k] . w[n,k]^T over q4x's IME2 tile layout: per 8 output rows,
/// (k/256) groups x 8 f16 scales then k/32 tiles of 128 bytes, where byte t
/// holds w[r0 + t/16][k0 + t%16] (low nibble) and [k0 + 16 + t%16] (high).
/// `row` selects the activation row of `x` the call is for (0 for decode).
pub fn gemvQ4x(x: []const f64, w: []const u8, out: []f64, n: usize, k: usize, row: usize) void {
    const groups = k / 256;
    const block_bytes = 4 * k + k / 16;
    for (0..n) |j| {
        const block = w[(j / 8) * block_bytes ..];
        const tiles = block[groups * 8 * 2 ..];
        const r = j % 8;
        var acc: f64 = 0;
        for (0..groups) |g| {
            const sh = std.mem.readInt(u16, block[g * 16 + r * 2 ..][0..2], .little);
            const d = halfToF64(sh);
            var isum: f64 = 0;
            for (0..8) |c| {
                const tile = tiles[(g * 8 + c) * 128 + r * 16 ..][0..16];
                for (0..16) |u| {
                    const b = tile[u];
                    isum += (@as(f64, @floatFromInt(b & 0x0f)) - 8.0) * x[row * k + (g * 8 + c) * 32 + u];
                    isum += (@as(f64, @floatFromInt(b >> 4)) - 8.0) * x[row * k + (g * 8 + c) * 32 + 16 + u];
                }
            }
            acc += isum * d;
        }
        out[j] = acc;
    }
}
