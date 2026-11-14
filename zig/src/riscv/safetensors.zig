//! safetensors.zig - a minimal reader for the safetensors format
//! (layout documented at https://github.com/huggingface/safetensors; this is an
//! original implementation of that documented layout):
//!   u64 header_len | JSON header | tensor data
//! The JSON maps name -> {dtype, shape, data_offsets}; offsets are byte ranges
//! into the data section. Dtypes read here: F64, F32, F16, BF16, I32, I16, U8.
//! The file is mmap'd read-only; tensors point into the map (copy on use).
const std = @import("std");

pub const DType = enum { f64, f32, f16, bf16, i32, i16, u8 };

pub const Tensor = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    /// The tensor's bytes inside the mapped file's data section.
    bytes: []const u8,
};

pub const File = struct {
    handle: std.Io.File,
    map: std.Io.File.MemoryMap,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    names: std.ArrayListUnmanaged([]const u8) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        const handle = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer handle.close(io);
        const len = std.math.cast(usize, try handle.length(io)) orelse return error.SafetensorsTooLarge;
        if (len < 8) return error.BadSafetensors;
        var map = try std.Io.File.MemoryMap.create(io, handle, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        var f: File = .{ .handle = handle, .map = map };
        parseInto(gpa, &f) catch |err| {
            f.deinitTables(gpa); // the map and handle: the errdefers above
            return err;
        };
        return f;
    }

    fn parseInto(gpa: std.mem.Allocator, f: *File) !void {
        const data = f.map.memory;
        const header_len = std.mem.readInt(u64, data[0..8], .little);
        if (header_len > data.len - 8) return error.BadSafetensors;
        const header = data[8..][0..header_len];
        const data_at = 8 + header_len;
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, header, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadSafetensors;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            const info = entry.value_ptr.*;
            if (info != .object) return error.BadSafetensors;
            const dtype_s = info.object.get("dtype") orelse return error.BadSafetensors;
            const shape_v = info.object.get("shape") orelse return error.BadSafetensors;
            const offs = info.object.get("data_offsets") orelse return error.BadSafetensors;
            if (dtype_s != .string or shape_v != .array or offs != .array) return error.BadSafetensors;
            const dtype = parseDType(dtype_s.string) orelse return error.UnsupportedDtype;
            const shape = try gpa.alloc(usize, shape_v.array.items.len);
            errdefer gpa.free(shape);
            for (shape_v.array.items, shape) |s, *out| {
                if (s != .integer) return error.BadSafetensors;
                out.* = @intCast(s.integer);
            }
            if (offs.array.items.len != 2) return error.BadSafetensors;
            const begin: usize = @intCast(offs.array.items[0].integer);
            const end: usize = @intCast(offs.array.items[1].integer);
            if (end > data.len - data_at or begin > end) return error.BadSafetensors;
            const name = try gpa.dupe(u8, entry.key_ptr.*);
            errdefer gpa.free(name);
            try f.names.append(gpa, name);
            try f.tensors.put(gpa, name, .{
                .name = name,
                .dtype = dtype,
                .shape = shape,
                .bytes = data[data_at..][begin..end],
            });
        }
    }

    /// safetensors dtype names are uppercase; only the reader's subset maps.
    fn parseDType(s: []const u8) ?DType {
        const pairs = .{ .{ "F64", DType.f64 }, .{ "F32", DType.f32 }, .{ "F16", DType.f16 }, .{ "BF16", DType.bf16 }, .{ "I32", DType.i32 }, .{ "I16", DType.i16 }, .{ "U8", DType.u8 } };
        inline for (pairs) |p| {
            if (std.ascii.eqlIgnoreCase(s, p[0])) return p[1];
        }
        return null;
    }

    /// Table memory only (open's error path before the file owns both).
    fn deinitTables(f: *File, gpa: std.mem.Allocator) void {
        var it = f.tensors.iterator();
        while (it.next()) |e| {
            gpa.free(e.key_ptr.*);
            gpa.free(e.value_ptr.shape);
        }
        f.tensors.deinit(gpa);
        f.names.deinit(gpa);
    }

    pub fn deinit(f: *File, gpa: std.mem.Allocator, io: std.Io) void {
        f.deinitTables(gpa);
        f.map.destroy(io);
        f.handle.close(io);
        f.* = undefined;
    }

    pub fn get(f: *const File, name: []const u8) ?Tensor {
        return f.tensors.get(name);
    }
};

/// Elements of a tensor's bytes as f64 (tests and reference use).
pub fn toF64(gpa: std.mem.Allocator, t: Tensor) ![]f64 {
    const width: usize = switch (t.dtype) {
        .f64 => 8,
        .f32, .i32 => 4,
        .f16, .bf16, .i16 => 2,
        .u8 => 1,
    };
    const n = t.bytes.len / width;
    const out = try gpa.alloc(f64, n);
    errdefer gpa.free(out);
    switch (t.dtype) {
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, t.bytes);
            @memcpy(out, src);
        },
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, t.bytes);
            for (src, out) |v, *o| o.* = v;
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, t.bytes);
            for (src, out) |v, *o| o.* = @as(f32, @bitCast(@as(u32, v) << 16));
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(u16, t.bytes);
            for (src, out) |v, *o| o.* = halfToF64(v);
        },
        .i32 => {
            const src = std.mem.bytesAsSlice(i32, t.bytes);
            for (src, out) |v, *o| o.* = @floatFromInt(v);
        },
        .i16 => {
            const src = std.mem.bytesAsSlice(i16, t.bytes);
            for (src, out) |v, *o| o.* = @floatFromInt(v);
        },
        .u8 => {
            for (t.bytes, out) |v, *o| o.* = v;
        },
    }
    return out;
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
