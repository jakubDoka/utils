const std = @import("std");
const lane = @This();
const Arena = @import("Arena.zig");

pub const Barrier = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    waiting: usize = 0,

    pub fn sync(self: *Barrier, up_to: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.waiting += 1;

        if (self.waiting == up_to) {
            self.waiting = 0;
            self.cond.broadcast();
        } else {
            self.cond.wait(&self.mutex);
        }
    }
};

pub const SpinBarrier = struct {
    waiting: std.atomic.Value(usize) = .init(0),
    generation: std.atomic.Value(usize) = .init(0),

    pub fn sync(self: *SpinBarrier, up_to: usize) void {
        const generation = self.generation.load(.acquire);
        const current_waiting = self.waiting.fetchAdd(1, .acquire) + 1;

        if (current_waiting == up_to) {
            self.waiting.store(0, .release);
            self.generation.store(generation + 1, .release);
        } else {
            while (self.generation.load(.acquire) == generation) {}
        }
    }
};

pub const ThreadCtx = struct {
    lane_idx: usize,
    lane_count: usize,
    shared: *SharedCtx,
};

pub const SharedCtx = struct {
    // eliminate needless contention
    _: void align(std.atomic.cache_line) = {},

    broadcast_buffer: u64 = undefined,
    barrier: Barrier = .{},
    spin_barrier: SpinBarrier = .{},
};

pub const single_threaded = @import("builtin").single_threaded;

pub var shared_ = SharedCtx{};
pub threadlocal var ctx: ThreadCtx = if (single_threaded) .{
    .lane_idx = 0,
    .lane_count = 1,
    .shared = &shared_,
} else undefined;

pub fn initSingleThreaded() void {
    ctx = .{
        .lane_idx = 0,
        .lane_count = 1,
        .shared = &shared_,
    };
}

pub inline fn isSingleThreaded() bool {
    return single_threaded or ctx.lane_count == 1;
}

/// Current thread will become the thread 0
pub fn boot(lane_count: usize, cx: anytype, comptime entry: fn (@TypeOf(cx)) void) void {
    if (isSingleThreaded()) {
        entry(cx);
        return;
    }

    var arg_buf: [4096]u8 = undefined;
    var arg_alloc = std.heap.FixedBufferAllocator.init(&arg_buf);

    var shared = SharedCtx{};

    const threads = arg_alloc.allocator().alloc(std.Thread, lane_count) catch unreachable;

    const task = struct {
        pub fn init(idx: usize, cnt: usize, shred: *SharedCtx, c: @TypeOf(cx)) void {
            ctx = .{ .lane_idx = idx, .lane_count = cnt, .shared = shred };
            entry(c);
        }
    };

    for (1..lane_count) |i| {
        threads[i] = std.Thread.spawn(
            .{ .allocator = arg_alloc.allocator() },
            task.init,
            .{ i, lane_count, &shared, cx },
        ) catch unreachable;
    }

    task.init(0, lane_count, &shared, cx);

    for (threads[1..]) |thread| {
        thread.join();
    }
}

pub fn range(values_count: usize) struct { start: usize, end: usize } {
    if (isSingleThreaded()) return .{ .start = 0, .end = values_count };

    const thread_idx = ctx.lane_idx;
    const thread_count = ctx.lane_count;

    const values_per_thread = values_count / thread_count;
    const leftover_values_count = values_count % thread_count;

    const thread_has_leftover = (thread_idx < leftover_values_count);
    const leftovers_before_this_thread_idx =
        if (thread_has_leftover) thread_idx else leftover_values_count;
    const thread_first_value_idx = (values_per_thread * thread_idx +
        leftovers_before_this_thread_idx);
    const thread_opl_value_idx = (thread_first_value_idx + values_per_thread +
        if (thread_has_leftover) @as(usize, 1) else 0);

    return .{ .start = thread_first_value_idx, .end = thread_opl_value_idx };
}

const SyncCtx = struct {
    spin: bool = false,

    pub const spinning = @This(){ .spin = true };
};

pub fn sync(args: SyncCtx) void {
    if (isSingleThreaded()) return;

    if (false and args.spin) {
        ctx.shared.spin_barrier.sync(ctx.lane_count);
    } else {
        ctx.shared.barrier.sync(ctx.lane_count);
    }
}

pub inline fn isRoot() bool {
    if (isSingleThreaded()) return true;

    return ctx.lane_idx == 0;
}

pub inline fn count() usize {
    return ctx.lane_count;
}

pub inline fn index() u16 {
    return @intCast(ctx.lane_idx);
}

pub fn productBroadcast(scratch: *Arena, to_extract: anytype) []@TypeOf(to_extract) {
    if (isSingleThreaded()) {
        const slot = scratch.create(@TypeOf(to_extract));
        slot.* = to_extract;
        return slot[0..1];
    }

    // TODO: this has one extra sync that is not needed
    //
    var buffer: []@TypeOf(to_extract) = undefined;
    if (lane.isRoot()) {
        buffer = scratch.alloc(@TypeOf(to_extract), count());
    }
    broadcast(&buffer, .{});

    buffer[ctx.lane_idx] = to_extract;

    sync(.spinning);

    return buffer;
}

pub fn broadcast(to_sync: anytype, args: struct {
    spin: bool = false,
    from: usize = 0,

    const spinning = @This(){ .spin = true };
}) void {
    if (isSingleThreaded()) return;

    if (@sizeOf(@TypeOf(to_sync.*)) != 8) {
        const to_sync_generic: u64 = @intFromPtr(to_sync);

        if (ctx.lane_idx == args.from) {
            ctx.shared.broadcast_buffer = to_sync_generic;
        }

        sync(.{ .spin = args.spin });

        if (ctx.lane_idx != args.from) {
            to_sync.* = @as(@TypeOf(to_sync), @ptrFromInt(ctx.shared.broadcast_buffer)).*;
        }
    } else {
        const to_sync_generic: *u64 = @ptrCast(to_sync);

        if (ctx.lane_idx == args.from) {
            ctx.shared.broadcast_buffer = to_sync_generic.*;
        }

        sync(.{ .spin = args.spin });

        if (ctx.lane_idx != args.from) {
            to_sync_generic.* = ctx.shared.broadcast_buffer;
        }
    }

    sync(.spinning);
}

pub const FixedQueue = struct {
    cursor: std.atomic.Value(usize) = .init(0),

    pub fn next(self: *FixedQueue, max: usize) ?usize {
        const cursor = self.cursor.fetchAdd(1, .acquire);
        return if (cursor < max) cursor else null;
    }
};

pub fn WorkStealingQueue(comptime Elem: type) type {
    return struct {
        queues: if (!single_threaded) []SubQueue else void,

        wait_mutex: std.Thread.Mutex = .{},
        wait_cond: std.Thread.Condition = .{},
        wait_count: if (!single_threaded) usize else void = if (!single_threaded) 0,

        const SubQueue = struct {
            _: void align(std.atomic.cache_line) = {},

            // TODO: maybe we can go wild and make this wait free
            tasks: std.ArrayList(Elem) = .empty,
            lock: std.Thread.Mutex = .{},
        };

        const Self = @This();

        pub fn init(scratch: *Arena, lane_cap: usize) Self {
            errdefer unreachable;

            if (isSingleThreaded()) return undefined;

            const queues = scratch.alloc(SubQueue, ctx.lane_count);
            for (queues) |*queue| queue.* = .{ .tasks = try .initCapacity(scratch.allocator(), lane_cap) };
            return .{ .queues = queues };
        }

        pub fn push(self: *Self, task: []Elem) usize {
            if (isSingleThreaded()) return task.len;

            var pushed: usize = 0;
            defer for (0..pushed) |_| self.wait_cond.signal();

            const queue = &self.queues[lane.ctx.lane_idx];
            queue.lock.lock();
            defer queue.lock.unlock();

            const max_push = @min(task.len, queue.tasks.capacity - queue.tasks.items.len);

            queue.tasks.appendSliceAssumeCapacity(task[task.len - max_push ..]);

            pushed = max_push;

            return task.len - max_push;
        }

        pub fn pop(self: *Self) ?Elem {
            if (isSingleThreaded()) return null;

            const queue = &self.queues[lane.ctx.lane_idx];
            {
                queue.lock.lock();
                defer queue.lock.unlock();

                if (queue.tasks.pop()) |task| {
                    @branchHint(.likely);
                    return task;
                }
            }

            // TODO: we should make a better access pattern, if that actually matters
            while (true) {
                // steal
                for (self.queues) |*other| {
                    if (other == queue) continue;

                    if (other.lock.tryLock()) {
                        defer other.lock.unlock();

                        if (other.tasks.pop()) |task| {
                            return task;
                        }
                    }
                }

                // steal harder
                for (self.queues) |*other| {
                    if (other == queue) continue;

                    other.lock.lock();
                    defer other.lock.unlock();

                    if (other.tasks.pop()) |task| {
                        return task;
                    }
                }

                // wait for more work, possibly terminate
                self.wait_mutex.lock();
                defer self.wait_mutex.unlock();

                self.wait_count += 1;

                if (self.wait_count == self.queues.len) {
                    self.wait_cond.broadcast();
                    return null;
                } else {
                    self.wait_cond.wait(&self.wait_mutex);

                    if (self.wait_count == self.queues.len) {
                        return null;
                    }

                    self.wait_count -= 1;
                }
            }
        }
    };
}

pub const Queue = WorkStealingQueue(u8);

pub fn share(scratch: *Arena, value: anytype) *@TypeOf(value) {
    var vl: *@TypeOf(value) = undefined;

    if (lane.isRoot()) {
        vl = scratch.create(@TypeOf(value));
        vl.* = value;
    }
    lane.broadcast(&vl, .{});

    return vl;
}

pub const max_groups = 64;

pub const Group = struct {
    prev: ThreadCtx = undefined,
    group_idx: usize = 0,
    alive_groups: std.bit_set.IntegerBitSet(max_groups) = .initEmpty(),

    /// Restores previous grouping of the thread, should be called on all
    /// members of the group unconditionally.
    pub fn deinit(self: Group, args: SyncCtx) void {
        if (isSingleThreaded()) return;
        ctx = self.prev;

        sync(args);
    }

    /// This account for dead groups that happne if thread count is
    /// insufficient to encompass all groups .eg if only 1 thread is available,
    /// all groups get folded to it.
    ///
    /// Compared to lane.isRoot this is true for all members of the group.
    pub fn is(self: Group, id: usize) bool {
        if (isSingleThreaded()) return true;
        if (self.group_idx == id) return true;
        return !self.alive_groups.isSet(id) and id > self.group_idx;
    }
};

/// Split the current thread group into multiple goups where the ratio is
/// specified by the `groups` slice, the current group is partitioned in a best
/// effort manner, some groups may be too small and dont get assigned any
/// threads, you should use `Group.isRoot` to determine if you are in the group
/// which acouts for empty groups
///
/// ther must be at least 2 groups and at most 64 groups (hopefully enough for everyone)
pub fn splitHeter(groups: []const usize, scratch: *Arena, args: SyncCtx) Group {
    return splitWithStrategy(groups, projectHeter, scratch, args);
}

pub fn splitWithStrategy(
    cx: anytype,
    strategy: fn (@TypeOf(cx), usize, usize) Projection,
    scratch: *Arena,
    args: SyncCtx,
) Group {
    var group = Group{};

    if (isSingleThreaded()) {
        group.alive_groups.set(0);
        return group;
    }

    group.prev = ctx;

    const projection = strategy(cx, ctx.lane_idx, ctx.lane_count);

    std.debug.assert(projection.group_count > 1);
    std.debug.assert(projection.group_count <= max_groups);

    group.group_idx = projection.group;

    var new_ctxes: []?*SharedCtx = undefined;
    if (isRoot()) {
        // we dont need this memory beyond this scope but deallocating it
        // here would require extra sync which is not worth it
        new_ctxes = scratch.alloc(?*SharedCtx, projection.group_count);
        @memset(new_ctxes, null);
    }

    broadcast(&new_ctxes, .{ .spin = args.spin });

    if (projection.idx == 0) {
        const new_shared = scratch.create(SharedCtx);
        new_shared.* = .{};
        new_ctxes[projection.group] = new_shared;
    }

    sync(.spinning);

    for (new_ctxes, 0..) |slt, i| {
        if (slt != null) group.alive_groups.set(i);
    }

    ctx.shared = new_ctxes[projection.group].?;
    ctx.lane_idx = projection.idx;
    ctx.lane_count = projection.count;

    return group;
}

const Projection = struct { idx: usize, count: usize, group: usize, group_count: usize };

pub fn projectHeter(
    groups: []const usize,
    cur: usize,
    cur_total: usize,
) Projection {
    var total: usize = 0;
    for (groups) |g| total += g;

    var last_group_transition: usize = 0;
    var last_group: usize = 0;
    var group_accum: usize = groups[last_group] * cur_total;

    for (0..cur + 1) |i| {
        if (i * total >= group_accum) {
            last_group_transition = i;
            last_group += 1;
            group_accum += groups[last_group] * cur_total;
        }
    }

    const group_start = last_group_transition;
    const final_goup = last_group;

    for (cur + 1..cur_total) |i| {
        if (i * total >= group_accum) {
            last_group_transition = i;
            last_group += 1;
            group_accum += groups[last_group] * cur_total;
            break;
        }
    } else {
        last_group_transition = cur_total;
    }

    return Projection{
        .idx = cur - group_start,
        .count = last_group_transition - group_start,
        .group = final_goup,
        .group_count = groups.len,
    };
}

pub const Lobby = struct {
    lock: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},

    pub fn wait(self: *Lobby) void {
        self.lock.lock();
        defer self.lock.unlock();

        self.cond.wait(&self.lock);
    }

    pub fn signal(self: *Lobby) void {
        self.lock.lock();
        defer self.lock.unlock();

        self.cond.signal();
    }

    pub fn done(self: *Lobby) void {
        self.lock.lock();
        defer self.lock.unlock();

        self.cond.broadcast();
    }
};

pub const _queue = RollingQueue(u8);

pub fn RollingQueue(comptime Elem: type) type {
    return struct {
        finished: std.atomic.Value(usize) = .init(0),
        reading: std.atomic.Value(usize) = .init(0),
        red: std.atomic.Value(usize) = .init(0),
        writing: std.atomic.Value(usize) = .init(0),
        written: std.atomic.Value(usize) = .init(0),

        read_futex: std.atomic.Value(u32) = .init(0),

        buffer: []Elem,

        const futex = std.Thread.Futex;

        const Self = @This();

        pub fn init(scratch: *Arena, cap: usize) Self {
            return .initBuffer(scratch.alloc(Elem, cap));
        }

        pub fn initBuffer(buffer: []Elem) Self {
            std.debug.assert(std.math.isPowerOfTwo(buffer.len));
            return .{ .buffer = buffer };
        }

        pub fn pop(self: *Self) ?Elem {
            while (true) switch (self.tryPop()) {
                .ready => |elem| return elem,
                .exhaused => return null,
                .retry => {
                    futex.wait(&self.read_futex, 0);
                },
            };
        }

        pub fn complete(self: *Self) bool {
            const final = self.finished.fetchAdd(1, .release) + 1;
            return final == self.written.load(.unordered);
        }

        pub fn tryPop(self: *Self) union(enum) { ready: Elem, exhaused, retry } {
            if (self.finished.load(.unordered) == self.written.load(.unordered)) {
                if (true) unreachable;
                return .exhaused;
            }

            const to_read = while (true) {
                const reading = self.reading.load(.unordered);
                const written = self.written.load(.unordered);

                if (reading == written) {
                    return .retry;
                }

                std.debug.assert(reading < written);

                if (self.reading.cmpxchgWeak(reading, reading + 1, .monotonic, .monotonic) != null) {
                    continue;
                }

                break reading;
            };

            std.debug.assert(std.math.isPowerOfTwo(self.buffer.len));
            const red = self.buffer[to_read % self.buffer.len];
            self.buffer[to_read % self.buffer.len] = undefined;

            while (self.red.cmpxchgWeak(to_read, to_read + 1, .monotonic, .monotonic) != null) {}

            return .{ .ready = red };
        }

        pub fn push(self: *Self, elem: Elem) void {
            std.debug.assert(self.tryPush(elem));
        }

        pub fn tryPush(self: *Self, elem: Elem) bool {
            const to_write = while (true) {
                const red = self.red.load(.unordered);
                const writing = self.writing.load(.unordered);

                if (writing == self.buffer.len + red) {
                    return false;
                }

                if (self.writing.cmpxchgWeak(writing, writing + 1, .monotonic, .monotonic) != null) {
                    continue;
                }

                break writing;
            };

            std.debug.assert(std.math.isPowerOfTwo(self.buffer.len));
            self.buffer[to_write % self.buffer.len] = elem;

            while (self.written.cmpxchgWeak(to_write, to_write + 1, .monotonic, .monotonic) != null) {}

            futex.wake(&self.read_futex, std.math.maxInt(u32));

            return true;
        }
    };
}

test "lane.RollingQueue.sanity" {
    if (true) return;

    var buff: [16]u8 = undefined;
    var buffer = RollingQueue(u8).initBuffer(&buff);

    buffer.push(1);
    try std.testing.expectEqual(1, buffer.pop());

    for (0..10) |_| {
        for (0..13) |i| {
            buffer.push(@intCast(i));
        }

        for (0..13) |i| {
            try std.testing.expectEqual(@as(u8, @intCast(i)), buffer.pop().?);
        }
    }

    const threads = 16;

    lane.boot(threads, {}, struct {
        pub fn entry(_: void) void {
            Arena.initScratch(1024 * 1024 * 16);
            defer Arena.deinitScratch();

            var tmp = Arena.scrath(null);
            defer tmp.deinit();

            var queue_slot: RollingQueue(u8) = undefined;
            if (lane.isRoot()) {
                queue_slot = .init(tmp.arena, threads);
            }

            var queue = &queue_slot;
            lane.broadcast(&queue, .{});

            queue.push(@intCast(lane.ctx.lane_idx));

            for (0..10000) |_| {
                const byte = queue.pop() orelse {
                    std.debug.print("{}\n", .{queue});

                    unreachable;
                };
                while (!queue.tryPush(byte)) {}
            }

            lane.sync(.{});

            if (lane.isRoot()) {
                var slots: [threads]bool = undefined;
                for (0..threads) |i| {
                    slots[queue_slot.buffer[i]] = true;
                }

                for (0..threads) |i| {
                    std.debug.assert(slots[i]);
                }
            }
        }
    }.entry);
}

test "lane.project" {
    const TestCase = struct {
        groups: []const usize,
        expected: []const usize,
    };

    const test_cases = [_]TestCase{ .{
        .groups = &.{ 1, 1 },
        .expected = &.{0},
    }, .{
        .groups = &.{ 1, 1 },
        .expected = &.{ 0, 1 },
    }, .{
        .groups = &.{ 2, 1 },
        .expected = &.{ 0, 0 },
    }, .{
        .groups = &.{ 3, 1 },
        .expected = &.{ 0, 0 },
    }, .{
        .groups = &.{ 4, 4 },
        .expected = &.{ 0, 1 },
    }, .{
        .groups = &.{ 4, 2 },
        .expected = &.{ 0, 0, 1 },
    }, .{
        .groups = &.{ 4, 3 },
        .expected = &.{ 0, 0, 1 },
    }, .{
        .groups = &.{ 3, 2 },
        .expected = &.{ 0, 0, 0, 0, 1, 1 },
    }, .{
        .groups = &.{ 3, 3, 3 },
        .expected = &.{ 0, 0, 1, 1, 2, 2 },
    } };

    for (test_cases, 0..) |tc, i| {
        var last_group: usize = tc.expected[0];
        var last_boundary: usize = 0;
        for (tc.expected, 0..) |vl, t| {
            errdefer std.debug.print("testcase {}:{} failed: {}\n", .{ i, t, tc });
            const proj = lane.projectHeter(tc.groups, t, tc.expected.len);

            if (proj.group != last_group) {
                last_boundary = t;
                last_group = proj.group;
            }

            try std.testing.expectEqual(vl, proj.group);
            try std.testing.expectEqual(t - last_boundary, proj.idx);

            var group_count: usize = 0;
            for (tc.expected) |g| {
                if (g == proj.group) group_count += 1;
            }
            try std.testing.expect(group_count == proj.count);
        }
    }
}

test "lane.sanity" {
    lane.boot(16, {}, struct {
        pub fn entry(_: void) void {
            Arena.initScratch(1024 * 1024 * 16);
            defer Arena.deinitScratch();

            var tmp = Arena.scrath(null);
            defer tmp.deinit();

            var dataset: []usize = undefined;
            if (lane.isRoot()) {
                dataset = tmp.arena.alloc(usize, 1024 * 1024);
            }
            lane.broadcast(&dataset, .spinning);

            const rnge = lane.range(dataset.len);

            for (dataset[rnge.start..rnge.end], rnge.start..) |*item, i| {
                item.* = i;
            }

            var root_sum: std.atomic.Value(usize) = .init(0);
            var root_sum_ref = &root_sum;
            lane.broadcast(&root_sum_ref, .{});

            var local_count: usize = 0;
            for (dataset[rnge.start..rnge.end]) |item| {
                local_count += item;
            }

            _ = root_sum_ref.fetchAdd(local_count, .acq_rel);

            lane.sync(.{});

            if (lane.isRoot()) {
                var sum: usize = 0;
                for (0..dataset.len) |item| {
                    sum += item;
                }
                std.testing.expectEqual(sum, root_sum.load(.unordered)) catch unreachable;
            }
        }
    }.entry);
}

test "lane.sanity.split" {
    lane.boot(16, {}, struct {
        pub fn entry(_: void) void {
            Arena.initScratch(1024 * 1024 * 16);
            defer Arena.deinitScratch();

            var tmp = Arena.scrath(null);
            defer tmp.deinit();

            {
                const group = lane.splitHeter(&.{ 2, 1, 1 }, tmp.arena, .{});
                defer group.deinit(.{});

                for (0..3) |i| {
                    if (group.is(i)) {
                        var counter: std.atomic.Value(usize) = .init(0);
                        var counter_ref = &counter;

                        lane.broadcast(&counter_ref, .{});

                        _ = counter_ref.fetchAdd(1, .acq_rel);

                        lane.sync(.spinning);

                        if (lane.isRoot()) {
                            std.testing.expectEqual(lane.count(), counter.load(.unordered)) catch unreachable;
                        }
                    }
                }
            }
        }
    }.entry);
}
