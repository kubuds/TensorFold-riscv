//! kernel_test.zig - every RISC-V kernel family against the float64 reference.
//! Runs on the host (scalar family) and under qemu or on K3 (scalar, rvv, ime2);
//! families the runtime refuses are skipped, never silently weakened.
const std = @import("std");
const kernels = @import("kernels.zig");
const ref = @import("reference.zig");

const c = kernels.c;
const t = std.testing;

/// Families to check, in dispatch order; each is offered when the runtime has it.
const families: []const [:0]const u8 = &.{ "scalar", "rvv", "ime2" };

fn describe() []const u8 {
    const vlen = kernels.vlenBits();
    if (vlen == 0) return "no RVV";
    return switch (vlen) {
        128 => "VLEN=128",
        256 => "VLEN=256",
        512 => "VLEN=512",
        1024 => "VLEN=1024",
        2048 => "VLEN=2048",
        else => "VLEN=other",
    };
}

/// Pseudo-random floats in [-1, 1] with a fixed seed: same inputs on every machine.
fn fill(buf: []f32, seed: u64) void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
}

fn fillF64(buf: []f64, seed: u64) void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    for (buf) |*v| v.* = rnd.float(f64) * 2.0 - 1.0;
}

/// Largest |got - want| must sit under rel * largest |want| + abs.
fn expectClose(got: []const f32, want: []const f64, rel: f64, abs: f64) !void {
    try t.expectEqual(want.len, got.len);
    var max_want: f64 = 0;
    for (want) |w| max_want = @max(max_want, @abs(w));
    var max_err: f64 = 0;
    for (got, want) |g, w| max_err = @max(max_err, @abs(@as(f64, g) - w));
    if (max_err > rel * max_want + abs) {
        std.debug.print("max_err {e} over tol {e} (rel {e} on scale {e}) family {s}\n", .{ max_err, rel * max_want + abs, rel, max_want, kernels.activeKernels() });
        for (0..@min(4, got.len)) |i| std.debug.print("  got[{d}]={e} want[{d}]={e}\n", .{ i, got[i], i, want[i] });
        return error.TestExpectedEqual;
    }
}

fn toF64(a: std.mem.Allocator, xs: []const f32) ![]f64 {
    const out = try a.alloc(f64, xs.len);
    for (xs, out) |x, *o| o.* = x;
    return out;
}

test "riscv kernels: dispatch surface" {
    try t.expect(kernels.select("scalar"));
    try t.expectEqualStrings("scalar", kernels.activeKernels());
    try t.expect(!kernels.select("no-such-family"));
    _ = c.tf_riscv_select_kernels("auto");
    std.debug.print("kernel tests on {s}: active {s}\n", .{ describe(), kernels.activeKernels() });
}

test "riscv kernels: rmsnorm against the float64 reference" {
    const a = t.allocator;
    for ([_]usize{ 1, 7, 32, 33, 128, 1024 }) |n| {
        const x = try a.alloc(f32, n);
        defer a.free(x);
        const w = try a.alloc(f32, n);
        defer a.free(w);
        fill(x, n);
        fill(w, n + 1);
        const got = try a.alloc(f32, n);
        defer a.free(got);
        const xf = try toF64(a, x);
        defer a.free(xf);
        const wf = try toF64(a, w);
        defer a.free(wf);
        const want = try a.alloc(f64, n);
        defer a.free(want);
        ref.rmsnorm(xf, wf, want, 1e-6);
        for (families) |fam| {
            if (!kernels.select(fam)) continue;
            c.tf_riscv_rmsnorm(x.ptr, w.ptr, got.ptr, @intCast(n), 1e-6);
            try expectClose(got, want, 1e-5, 1e-6);
        }
    }
}

test "riscv kernels: rope against the float64 reference" {
    const a = t.allocator;
    const head_dim = 64;
    const q_heads = 3;
    const kv_heads = 2;
    for ([_]u32{ 0, 1, 7, 1023 }) |pos| {
        const q = try a.alloc(f32, q_heads * head_dim);
        defer a.free(q);
        const k = try a.alloc(f32, kv_heads * head_dim);
        defer a.free(k);
        fill(q, pos + 11);
        fill(k, pos + 29);
        const q0 = try a.dupe(f32, q);
        defer a.free(q0);
        const k0 = try a.dupe(f32, k);
        defer a.free(k0);
        const qf = try toF64(a, q);
        defer a.free(qf);
        const kf = try toF64(a, k);
        defer a.free(kf);
        ref.rope(qf, kf, pos, q_heads, kv_heads, head_dim, 1e6);
        for (families) |fam| {
            if (!kernels.select(fam)) continue;
            // Rope is in place: every family starts from the same rows.
            @memcpy(q, q0);
            @memcpy(k, k0);
            c.tf_riscv_rope(q.ptr, k.ptr, @intCast(pos), q_heads, kv_heads, head_dim, 1e6);
            try expectClose(q, qf, 1e-5, 1e-6);
            try expectClose(k, kf, 1e-5, 1e-6);
        }
    }
}

test "riscv kernels: softmax against the float64 reference" {
    const a = t.allocator;
    for ([_]usize{ 1, 2, 31, 64, 1000 }) |n| {
        const x = try a.alloc(f32, n);
        defer a.free(x);
        fill(x, n * 7 + 3);
        const x0 = try a.dupe(f32, x);
        defer a.free(x0);
        const xf = try toF64(a, x);
        defer a.free(xf);
        ref.softmax(xf);
        for (families) |fam| {
            if (!kernels.select(fam)) continue;
            // Softmax is in place: every family starts from the same row.
            @memcpy(x, x0);
            c.tf_riscv_softmax(x.ptr, @intCast(n));
            try expectClose(x, xf, 1e-5, 1e-7);
        }
    }
}

test "riscv kernels: silu_mul and add against the float64 reference" {
    const a = t.allocator;
    for ([_]usize{ 1, 17, 256, 1023 }) |n| {
        const gate = try a.alloc(f32, n);
        defer a.free(gate);
        const up = try a.alloc(f32, n);
        defer a.free(up);
        fill(gate, n + 5);
        fill(up, n + 6);
        const gf = try toF64(a, gate);
        defer a.free(gf);
        const uf = try toF64(a, up);
        defer a.free(uf);
        const want = try a.alloc(f64, n);
        defer a.free(want);
        ref.siluMul(gf, uf, want);
        const got = try a.alloc(f32, n);
        defer a.free(got);
        const want_add = try a.alloc(f64, n);
        defer a.free(want_add);
        ref.add(gf, uf, want_add);
        for (families) |fam| {
            if (!kernels.select(fam)) continue;
            c.tf_riscv_silu_mul(gate.ptr, up.ptr, got.ptr, @intCast(n));
            try expectClose(got, want, 1e-5, 1e-6);
            c.tf_riscv_add(gate.ptr, up.ptr, got.ptr, @intCast(n));
            try expectClose(got, want_add, 1e-6, 0);
        }
    }
}

test "riscv kernels: matmul_f32 and matmul_bf16 against the float64 reference" {
    const a = t.allocator;
    for ([_]usize{ 1, 2, 8 }) |m| {
        for ([_]usize{ 1, 5, 32 }) |n| {
            for ([_]usize{ 1, 7, 64, 257 }) |k| {
                const x = try a.alloc(f32, m * k);
                defer a.free(x);
                const b = try a.alloc(f32, n * k);
                defer a.free(b);
                fill(x, m * 1000 + n * 100 + k);
                fill(b, m * 1000 + n * 100 + k + 1);
                const got = try a.alloc(f32, m * n);
                defer a.free(got);
                const xf = try toF64(a, x);
                defer a.free(xf);
                const bf = try toF64(a, b);
                defer a.free(bf);
                const want_t = try a.alloc(f64, m * n);
                defer a.free(want_t);
                ref.matmulF32(xf, bf, want_t, m, n, k, true);
                for (families) |fam| {
                    if (!kernels.select(fam)) continue;
                    c.tf_riscv_matmul_f32(x.ptr, b.ptr, got.ptr, @intCast(m), @intCast(n), @intCast(k), 1);
                    try expectClose(got, want_t, 1e-4, 1e-5);
                }
                // [k,n] layout for the non-transposed path
                const b2 = try a.alloc(f32, k * n);
                defer a.free(b2);
                for (0..k) |tt| {
                    for (0..n) |j| b2[tt * n + j] = b[j * k + tt];
                }
                const b2f = try toF64(a, b2);
                defer a.free(b2f);
                const want_n = try a.alloc(f64, m * n);
                defer a.free(want_n);
                ref.matmulF32(xf, b2f, want_n, m, n, k, false);
                for (families) |fam| {
                    if (!kernels.select(fam)) continue;
                    c.tf_riscv_matmul_f32(x.ptr, b2.ptr, got.ptr, @intCast(m), @intCast(n), @intCast(k), 0);
                    try expectClose(got, want_n, 1e-4, 1e-5);
                }
                // bf16 weights: the same u16 lanes feed the reference
                const w = try a.alloc(u16, n * k);
                defer a.free(w);
                var prng: std.Random.DefaultPrng = .init(m * 1000 + n * 100 + k + 2);
                const rnd = prng.random();
                for (w) |*v| v.* = rnd.int(u16) & 0xBFFF; // finite bf16: exponent bit 14 cleared
                const want_b = try a.alloc(f64, m * n);
                defer a.free(want_b);
                ref.matmulBf16(xf, w, want_b, m, n, k);
                for (families) |fam| {
                    if (!kernels.select(fam)) continue;
                    c.tf_riscv_matmul_bf16(x.ptr, w.ptr, got.ptr, @intCast(m), @intCast(n), @intCast(k));
                    // The IME2 MMA multiplies bf16 lanes (its A operand rounds
                    // x to bf16): the family's products carry input rounding
                    // the f32 families do not. That is the IME2 contract from
                    // the K3 paper, not drift - so it gets its own tolerance.
                    if (std.mem.eql(u8, fam, "ime2"))
                        try expectClose(got, want_b, 2e-3, 1e-3) // bf16-input bound
                    else
                        try expectClose(got, want_b, 1e-4, 1e-5);
                }
            }
        }
    }
}

test "riscv kernels: attention against the float64 reference" {
    const a = t.allocator;
    for ([_]usize{ 1, 3, 16, 33 }) |rows| {
        for ([_]usize{ 16, 64 }) |hd| {
            const q = try a.alloc(f32, hd);
            defer a.free(q);
            const k = try a.alloc(f32, rows * hd);
            defer a.free(k);
            const v = try a.alloc(f32, rows * hd);
            defer a.free(v);
            fill(q, rows * 100 + hd);
            fill(k, rows * 100 + hd + 1);
            fill(v, rows * 100 + hd + 2);
            const got = try a.alloc(f32, hd);
            defer a.free(got);
            const qf = try toF64(a, q);
            defer a.free(qf);
            const kf = try toF64(a, k);
            defer a.free(kf);
            const vf = try toF64(a, v);
            defer a.free(vf);
            const want = try a.alloc(f64, hd);
            defer a.free(want);
            ref.attentionF32(qf, kf, vf, want, rows, hd, 0.125);
            for (families) |fam| {
                if (!kernels.select(fam)) continue;
                c.tf_riscv_attention_f32(q.ptr, k.ptr, v.ptr, got.ptr, @intCast(rows), @intCast(hd), 0.125);
                try expectClose(got, want, 1e-5, 1e-6);
            }
        }
    }
}

test "riscv kernels: q8_0 and q4_0 GEMV against the float64 reference" {
    const a = t.allocator;
    for ([_]usize{ 32, 64, 96 }) |k| {
        for ([_]usize{ 1, 5 }) |n| {
            const x = try a.alloc(f32, k);
            defer a.free(x);
            fill(x, k + n);
            const xf = try toF64(a, x);
            defer a.free(xf);
            const got = try a.alloc(f32, n);
            defer a.free(got);

            const w8 = try a.alloc(u8, n * (k / 32) * 34);
            defer a.free(w8);
            var prng: std.Random.DefaultPrng = .init(k * 100 + n);
            var rnd = prng.random();
            for (w8) |*v| v.* = rnd.int(u8);
            for (0..n * (k / 32)) |blk| { // finite f16 scales: exp bits kept below 31
                std.mem.writeInt(u16, w8[blk * 34 ..][0..2], 0x3400 | @as(u16, @intCast(blk & 0xFF)), .little);
            }
            const want8 = try a.alloc(f64, n);
            defer a.free(want8);
            ref.gemvQ80(xf, w8, want8, n, k);
            for (families) |fam| {
                if (!kernels.select(fam)) continue;
                c.tf_riscv_gemv_q8_0(x.ptr, w8.ptr, got.ptr, @intCast(n), @intCast(k));
                try expectClose(got, want8, 1e-4, 1e-5);
            }

            const w4 = try a.alloc(u8, n * (k / 32) * 18);
            defer a.free(w4);
            prng = .init(k * 100 + n + 1);
            rnd = prng.random();
            for (w4) |*v| v.* = rnd.int(u8);
            for (0..n * (k / 32)) |blk| {
                std.mem.writeInt(u16, w4[blk * 18 ..][0..2], 0x3400 | @as(u16, @intCast(blk & 0xFF)), .little);
            }
            const want4 = try a.alloc(f64, n);
            defer a.free(want4);
            ref.gemvQ40(xf, w4, want4, n, k);
            for (families) |fam| {
                if (!kernels.select(fam)) continue;
                c.tf_riscv_gemv_q4_0(x.ptr, w4.ptr, got.ptr, @intCast(n), @intCast(k));
                try expectClose(got, want4, 1e-4, 1e-5);
            }
        }
    }
}
