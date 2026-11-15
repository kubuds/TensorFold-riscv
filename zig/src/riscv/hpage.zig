//! hpage.zig - a model allocator backed by the K3's huge-page weight arena.
//!
//! Every allocation becomes its own anonymous, 2 MiB-aligned mapping with
//! MADV_HUGEPAGE (zig/kernels/riscv/mem.c), which is what the vendor
//! llama.cpp's mem_backend=HPAGE does for its weight arena. On the K3 that is
//! worth 30.0 -> 40.5 GB/s of streaming read on the A100 tiles (measured,
//! tools/riscv/probe_k3.c mem), i.e. it is the difference between a decode
//! token at ~40 and ~55 tok/s once the weight format is 4-bit.
//!
//! ArenaAllocator asks for a handful of growing chunks and frees exactly
//! those chunks at deinit, so this allocator keeps a small table of mappings:
//! alloc maps a new region, free unmaps it, resize refuses (the arena then
//! copies, which is correct).

const std = @import("std");
const kernels = @import("kernels.zig");

const c = kernels.c;

pub const HpageAllocator = struct {
    const Entry = struct { ptr: [*]u8, len: usize };
    const max_entries = 64;

    entries: [max_entries]Entry = undefined,
    count: usize = 0,
    bytes: usize = 0,
    fallback: bool = false,

    var instance: HpageAllocator = .{};

    fn make() std.mem.Allocator {
        return .{ .ptr = &instance, .vtable = &vtable };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const self: *HpageAllocator = @ptrCast(@alignCast(ctx));
        const page = 2 * 1024 * 1024;
        const need = @max(len, 1);
        const size = (need + page - 1) & ~@as(usize, page - 1);
        if (self.count < max_entries) {
            if (c.tf_riscv_hpage_alloc(size)) |raw| {
                const p: [*]u8 = @ptrCast(raw);
                std.debug.assert(@intFromPtr(p) % alignment.toByteUnits() == 0);
                self.entries[self.count] = .{ .ptr = p, .len = size };
                self.count += 1;
                self.bytes += size;
                return p;
            }
        }
        // No THP arena (or too many chunks): plain page_allocator memory.
        self.fallback = true;
        // Alignment here is a runtime value; page_allocator's alignedAlloc
        // wants it comptime-known, so over-allocate and align by hand.
        const pad = alignment.toByteUnits();
        const raw = std.heap.page_allocator.alignedAlloc(u8, .of(usize), need + pad) catch return null;
        const aligned = std.mem.alignForward(usize, @intFromPtr(raw.ptr), pad);
        return @ptrFromInt(aligned);
    }

    fn resize(ctx: *anyopaque, buf: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
        _ = ctx;
        return new_len <= buf.len;
    }

    fn remap(ctx: *anyopaque, buf: []u8, _: std.mem.Alignment, new_len: usize, _: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = new_len;
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, _: usize) void {
        const self: *HpageAllocator = @ptrCast(@alignCast(ctx));
        for (self.entries[0..self.count]) |e| {
            if (e.ptr == buf.ptr) {
                c.tf_riscv_hpage_free(@ptrCast(e.ptr), e.len);
                self.bytes -= e.len;
                return;
            }
        }
        std.heap.page_allocator.free(buf);
        _ = alignment;
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };
};

/// The allocator for the model arena (falls back to page_allocator memory
/// when the kernel has no transparent huge pages).
pub fn allocator() std.mem.Allocator {
    return HpageAllocator.make();
}

/// Bytes the model arena holds in huge-page mappings.
pub fn bytes() usize {
    return HpageAllocator.instance.bytes;
}

/// Whether the kernel actually backed the arena with huge pages.
pub fn backed() usize {
    return if (HpageAllocator.instance.count > 0)
        c.tf_riscv_hpage_backed(@ptrCast(HpageAllocator.instance.entries[0].ptr))
    else
        0;
}

pub fn available() bool {
    return c.tf_riscv_hpage_available() != 0;
}
