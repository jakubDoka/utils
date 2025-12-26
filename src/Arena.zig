const Arena = @This();
const std = @import("std");

start: [*]align(page_size) u8,
end: [*]align(page_size) u8,
pos: [*]u8,

const page_size = std.heap.pageSize();

threadlocal var inited: bool = false;
pub threadlocal var scratch: [2]Arena = undefined;

pub const Scratch = struct {
    prev_pos: [*]u8,
    arena: *Arena,

    pub fn deinit(self: *Scratch) void {
        @memset(self.prev_pos[0 .. @intFromPtr(self.arena.pos) - @intFromPtr(self.prev_pos)], undefined);
        self.arena.pos = self.prev_pos;
        self.* = undefined;
    }
};

pub fn initScratch(cap: usize) void {
    if (std.debug.runtime_safety) {
        std.debug.assert(!inited);
        inited = true;
    }
    for (&scratch) |*slt| slt.* = init(cap);
}

pub fn deinitScratch() void {
    if (std.debug.runtime_safety) {
        std.debug.assert(inited);
        inited = false;
    }
    for (&scratch) |*slt| slt.deinit();
}

pub fn resetScratch() void {
    for (&scratch) |*slt| slt.reset();
}

pub fn consumed(arena: *Arena) u64 {
    return @intCast(arena.pos - arena.start);
}

pub fn reset(arena: *Arena) void {
    arena.pos = arena.start;
}

pub fn allocated(self: *Arena) usize {
    return @intFromPtr(self.end) - @intFromPtr(self.pos);
}

pub fn getCapacity(self: *Arena) usize {
    return @intFromPtr(self.end) - @intFromPtr(self.start);
}

pub fn subslice(self: *Arena, capacity: usize) Arena {
    const cap = std.mem.alignBackward(usize, capacity, page_size);
    const ptr = self.allocRaw(page_size, cap).?;
    return .{
        .start = @alignCast(ptr),
        .end = @alignCast(ptr + cap),
        .pos = ptr,
    };
}

pub fn scrath(except: ?*anyopaque) Scratch {
    if (std.debug.runtime_safety) std.debug.assert(inited);
    for (&scratch) |*slt| if (@as(*anyopaque, slt) != except)
        return slt.checkpoint();
    unreachable;
}

pub fn init(cap: usize) Arena {
    const pages = std.mem.alignForward(usize, cap, page_size);
    const ptr = std.heap.page_allocator.rawAlloc(pages, .fromByteUnits(page_size), @returnAddress()).?;
    return .{
        .end = @alignCast(ptr + pages),
        .start = @alignCast(ptr),
        .pos = @alignCast(ptr),
    };
}

pub fn allocator(self: *Arena) std.mem.Allocator {
    const alc_impl = enum {
        fn alloc(ptr: *anyopaque, size: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
            const slf: *Arena = @ptrCast(@alignCast(ptr));
            return slf.allocRaw(alignment.toByteUnits(), size);
        }
        fn free(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {}
        fn remap(ptr: *anyopaque, mem: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            if (@This().resize(ptr, mem, alignment, new_len, ret_addr)) return mem.ptr;
            return null;
        }
        fn resize(ptr: *anyopaque, mem: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
            const slf: *Arena = @ptrCast(@alignCast(ptr));
            if (mem.ptr + mem.len == slf.pos) {
                slf.pos += new_len;
                slf.pos -= mem.len;
                return true;
            }
            return false;
        }
    };

    return .{
        .ptr = self,
        .vtable = &.{
            .alloc = alc_impl.alloc,
            .free = alc_impl.free,
            .remap = alc_impl.remap,
            .resize = alc_impl.resize,
        },
    };
}

pub fn deinit(self: *Arena) void {
    std.heap.page_allocator.rawFree(self.start[0 .. self.end - self.start], .fromByteUnits(page_size), @returnAddress());
    self.* = undefined;
}

pub fn checkpoint(self: *Arena) Scratch {
    return .{ .prev_pos = self.pos, .arena = self };
}

pub fn dupe(self: *Arena, comptime Elem: type, values: []const Elem) []Elem {
    const new = self.alloc(Elem, values.len);
    @memcpy(new, values);
    return new;
}

pub fn dupeZ(self: *Arena, comptime Elem: type, values: []const Elem) [:0]Elem {
    const new = self.alloc(Elem, values.len + 1);
    @memcpy(new[0..values.len], values);
    new[values.len] = 0;
    return new[0..values.len :0];
}

pub fn allocAligned(self: *Arena, comptime T: type, count: usize, comptime alignment: usize) []align(alignment) T {
    const ptr: [*]align(alignment) T = @ptrCast(@alignCast(self.allocRaw(alignment, @sizeOf(T) * count)));
    const mem = ptr[0..count];
    @memset(mem, undefined);
    return mem;
}

pub fn alloc(self: *Arena, comptime T: type, count: usize) []T {
    return self.allocAligned(T, count, @alignOf(T));
}

pub fn allocZ(self: *Arena, comptime T: type, count: usize) [:0]T {
    const ptr: [*]T = @ptrCast(@alignCast(self.allocRaw(@alignOf(T), @sizeOf(T) * (count + 1))));
    ptr[count] = 0;
    return ptr[0..count :0];
}

pub fn allocRaw(self: *Arena, alignment: usize, size: usize) ?[*]u8 {
    self.pos = @ptrFromInt(std.mem.alignForward(usize, @intFromPtr(self.pos), alignment));
    self.pos += size;
    if (@intFromPtr(self.end) < @intFromPtr(self.pos)) return null;
    return self.pos - size;
}

pub fn makeArrayList(self: *Arena, comptime T: type, cap: usize) std.ArrayList(T) {
    return .initBuffer(self.alloc(T, cap));
}

pub fn create(self: *Arena, comptime T: type) *T {
    return &self.alloc(T, 1).ptr[0];
}

pub fn print(self: *Arena, comptime fmt: []const u8, args: anytype) []u8 {
    return std.fmt.allocPrint(self.allocator(), fmt, args) catch unreachable;
}
