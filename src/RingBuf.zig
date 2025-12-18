const std = @import("std");
const root = @import("root.zig");

impl: Impl,
size: usize,
head: usize,
tail: usize,

const RingBuf = @This();

pub fn init(size: usize) !RingBuf {
    std.debug.assert(size % std.heap.pageSize() == 0);
    std.debug.assert(std.math.isPowerOfTwo(size));
    std.debug.assert(size > 0);

    return .{
        .impl = try .init(size),
        .size = size,
        .head = 0,
        .tail = 0,
    };
}

pub fn getReadableBuffer(self: *RingBuf) []u8 {
    return self.impl.addr[self.head & self.size - 1 ..][0 .. self.tail - self.head];
}

pub fn getWritableBuffer(self: *RingBuf) []u8 {
    return self.impl.addr[self.tail & self.size - 1 ..][0 .. self.size -
        (self.tail - self.head)];
}

pub fn advanceRead(self: *RingBuf, by: usize) void {
    std.debug.assert(self.getReadableBuffer().len >= by);
    @memset(self.getReadableBuffer()[0..by], undefined);
    self.head += by;
}

pub fn advanceWrite(self: *RingBuf, by: usize) void {
    std.debug.assert(self.getWritableBuffer().len >= by);
    self.tail += by;
}

pub fn deinit(self: *RingBuf) void {
    self.impl.deinit(self.size);
    self.* = undefined;
}

pub const Impl = switch (@import("builtin").os.tag) {
    .linux, .macos => ImplLinux,
    .windows => WindowsImpl,
    else => @compileError("Unsupported OS"),
};

pub fn tofd(sys_int: usize) std.os.linux.fd_t {
    return @bitCast(@as(u32, @intCast(sys_int)));
}

const WindowsImpl = struct {
    addr: [*]u8,
    handle: std.os.windows.HANDLE,

    extern "kernel32" fn CreateFileMappingA(
        hFile: win.HANDLE,
        lpFileMappingAttributes: ?*win.SECURITY_ATTRIBUTES,
        flProtect: win.DWORD,
        dwMaximumSizeHigh: win.DWORD,
        dwMaximumSizeLow: win.DWORD,
        lpName: ?win.LPCSTR,
    ) callconv(.winapi) ?win.HANDLE;

    extern "kernel32" fn UnmapViewOfFile(
        lpBaseAddress: ?win.LPVOID,
    ) callconv(.winapi) win.BOOL;

    pub const FILE_MAP_ALL_ACCESS = 983_071;
    pub const FILE_MAP_WRITE = 2;
    pub const MEM_REPLACE_PLACEHOLDER = 0x00004000;
    pub const MEM_PRESERVE_PLACEHOLDER = 0x00000002;
    const win = std.os.windows;

    fn init(size: usize) !WindowsImpl {
        var vaule: @TypeOf(tryInit(size)) = undefined;
        for (0..10) |_| {
            vaule = tryInit(size);
            if (vaule) |v| return v else |_| {}
        }
        return vaule;
    }

    const VirtualAlloc2Fn = *const fn (
        ?*anyopaque,
        ?*anyopaque,
        usize,
        u32,
        u32,
        ?*anyopaque,
        usize,
    ) callconv(.winapi) ?*anyopaque;

    const VirtualFreeFn = *const fn (
        ?*anyopaque,
        usize,
        u32,
    ) callconv(.winapi) win.BOOL;

    const MapViewOfFile3Fn = *const fn (
        win.HANDLE,
        win.HANDLE,
        ?*anyopaque,
        u64,
        usize,
        u32,
        u32,
        ?*anyopaque,
        usize,
    ) callconv(.winapi) ?*anyopaque;

    fn tryInit(size: usize) !WindowsImpl {
        var kernelbase = try root.DynLib.init("kernelbase.dll");
        defer kernelbase.deinit();

        const VirtualAlloc2 = kernelbase.lookup(VirtualAlloc2Fn, "VirtualAlloc2") orelse
            return error.VirtualAlloc2Missing;
        const MapViewOfFile3 =
            kernelbase.lookup(MapViewOfFile3Fn, "MapViewOfFile3") orelse
            return error.MapViewOfFile3Missing;

        const addr = VirtualAlloc2(
            win.GetCurrentProcess(),
            null,
            size * 2,
            win.MEM_RESERVE | win.MEM_RESERVE_PLACEHOLDER,
            win.PAGE_NOACCESS,
            null,
            0,
        ) orelse {
            return error.VirtualAllocFailed;
        };

        win.VirtualFree(
            @ptrFromInt(@intFromPtr(addr) + size),
            size,
            win.MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );

        const handle = CreateFileMappingA(
            std.os.windows.INVALID_HANDLE_VALUE,
            null,
            std.os.windows.PAGE_READWRITE,
            0,
            @intCast(size),
            null,
        ) orelse {
            return error.CreateFileMappingFailed;
        };

        errdefer _ = std.os.windows.CloseHandle(handle);

        const first_cpy = MapViewOfFile3(
            handle,
            win.GetCurrentProcess(),
            addr,
            0,
            @intCast(size),
            MEM_REPLACE_PLACEHOLDER,
            win.PAGE_READWRITE,
            null,
            0,
        ) orelse {
            return error.MapViewOfFileFailed1;
        };

        errdefer _ = UnmapViewOfFile(first_cpy);

        if (first_cpy != addr) return error.MapViewOfFileFailed1Miss;

        const first_ptr = @as([*]u8, @ptrCast(addr)) + size;
        const sedond_cpy = MapViewOfFile3(
            handle,
            win.GetCurrentProcess(),
            @as([*]u8, @ptrCast(addr)) + size,
            0,
            @intCast(size),
            MEM_REPLACE_PLACEHOLDER,
            win.PAGE_READWRITE,
            null,
            0,
        ) orelse {
            return error.MapViewOfFileFailed2;
        };

        errdefer _ = UnmapViewOfFile(sedond_cpy);

        if (sedond_cpy != @as(*anyopaque, first_ptr)) return error.MapViewOfFileFailedMiss;

        return .{
            .addr = @ptrCast(addr),
            .handle = handle,
        };
    }

    fn deinit(self: *WindowsImpl, size: usize) void {
        _ = UnmapViewOfFile(self.addr);
        _ = UnmapViewOfFile(@ptrCast(self.addr + size));
        _ = std.os.windows.CloseHandle(self.handle);
    }
};

const ImplLinux = struct {
    addr: [*]u8,
    memefd: std.os.linux.fd_t,

    fn init(size: usize) !ImplLinux {
        const fd = tofd(std.os.linux.memfd_create("ringbuffer", 0));
        if (fd == -1) return error.CreateMemfdFailed;

        const res = tofd(std.os.linux.ftruncate(fd, @intCast(size)));
        if (res == -1) return error.TruncateMemfdFailed;

        const addr = std.os.linux.mmap(
            null,
            size * 2,
            std.os.linux.PROT.NONE,
            .{
                .TYPE = .PRIVATE,
                .ANONYMOUS = true,
            },
            -1,
            0,
        );
        if (addr == std.math.maxInt(usize)) return error.ReservePagesFailed;

        if (std.os.linux.mmap(
            @ptrFromInt(addr),
            size,
            std.os.linux.PROT.READ | std.os.linux.PROT.WRITE,
            .{
                .TYPE = .SHARED,
                .FIXED = true,
            },
            fd,
            0,
        ) == std.math.maxInt(usize))
            return error.MapFirstPageFailed;

        if (std.os.linux.mmap(
            @ptrFromInt(addr + size),
            size,
            std.os.linux.PROT.READ | std.os.linux.PROT.WRITE,
            .{
                .TYPE = .SHARED,
                .FIXED = true,
            },
            fd,
            0,
        ) == std.math.maxInt(usize))
            return error.MapSecondPageFailed;

        @memset(@as([*]u8, @ptrFromInt(addr))[0..size], undefined);

        return .{
            .addr = @ptrFromInt(addr),
            .memefd = fd,
        };
    }

    fn deinit(self: *ImplLinux, size: usize) void {
        _ = std.os.linux.munmap(self.addr, size * 2);
        _ = std.os.linux.close(self.memefd);
    }
};

test RingBuf {
    root.Arena.initScratch(1024 * 4);

    const cap = 1024 * 64;
    var rb = try RingBuf.init(cap);
    defer rb.deinit();

    try std.testing.expectEqual(rb.getReadableBuffer().len, 0);
    try std.testing.expectEqual(rb.getWritableBuffer().len, cap);

    for ([_]usize{ cap / 2, cap / 3, cap / 2, cap / 5, cap / 8 }, 0..) |len, i| {
        @memset(rb.getWritableBuffer()[0..len], @intCast(i));
        rb.advanceWrite(len);
        try std.testing.expectEqual(rb.getReadableBuffer().len, len);
        try std.testing.expectEqual(rb.getWritableBuffer().len, cap - len);

        try std.testing.expect(std.mem.allEqual(u8, rb.getReadableBuffer(), @intCast(i)));
        rb.advanceRead(len);
        try std.testing.expectEqual(rb.getReadableBuffer().len, 0);
        try std.testing.expectEqual(rb.getWritableBuffer().len, cap);
    }
}
