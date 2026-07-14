//! Refcounted device frame cache. Caller contracts:
//! - acquire() claims a whole window all-or-nothing (never blocks holding a partial claim).
//! - abandon claimed-but-unpublished slots on error (else later windows hang on the condvar).
//! - drain streams before releasing slots (released slots are evictable under queued work).

const std = @import("std");
const cu = @import("cu.zig");

const allocator = std.heap.c_allocator;
const threaded_io: std.Io = std.Io.Threaded.global_single_threaded.io();

pub const CacheSlot = struct {
    key: i64 = -1,
    buf: cu.DeviceBuffer = .{},
    ev: cu.Event = .{},
    refs: u32 = 0,
    stamp: u64 = 0,
    ready: bool = false,
};

pub const FrameCache = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    slots: []CacheSlot = &.{},
    clock: u64 = 0,
    frame_elems: usize = 0,

    pub fn deinit(self: *FrameCache) void {
        for (self.slots) |*s| {
            s.buf.deinit();
            if (s.ev.handle != null) s.ev.deinit();
        }
        if (self.slots.len > 0) allocator.free(self.slots);
        self.slots = &.{};
    }

    /// Reserve every key all-or-nothing; `load[i]` means this worker must fill the slot.
    pub fn acquire(self: *FrameCache, keys: []const i64, idx: []usize, load: []bool) void {
        self.mutex.lockUncancelable(threaded_io);
        defer self.mutex.unlock(threaded_io);

        outer: while (true) {
            for (keys) |k| {
                for (self.slots) |*s| {
                    if (s.key == k and !s.ready) {
                        self.cond.waitUncancelable(threaded_io, &self.mutex);
                        continue :outer;
                    }
                }
            }

            var claimed: usize = 0;
            while (claimed < keys.len) : (claimed += 1) {
                const k = keys[claimed];
                load[claimed] = false;

                var found: ?usize = null;
                for (self.slots, 0..) |*s, si| {
                    if (s.key == k) {
                        found = si;
                        break;
                    }
                }
                if (found == null) {
                    var victim: ?usize = null;
                    for (self.slots, 0..) |*s, si| {
                        if (s.refs != 0) continue;
                        if (victim == null or s.stamp < self.slots[victim.?].stamp) victim = si;
                    }
                    if (victim) |v| {
                        self.slots[v].key = k;
                        self.slots[v].ready = false;
                        found = v;
                        load[claimed] = true;
                    } else {
                        for (0..claimed) |j| {
                            self.slots[idx[j]].refs -= 1;
                            if (load[j]) self.slots[idx[j]].key = -1;
                        }
                        self.cond.broadcast(threaded_io);
                        self.cond.waitUncancelable(threaded_io, &self.mutex);
                        continue :outer;
                    }
                }
                idx[claimed] = found.?;
                self.clock += 1;
                self.slots[found.?].refs += 1;
                self.slots[found.?].stamp = self.clock;
            }
            return;
        }
    }

    pub fn publish(self: *FrameCache, si: usize) void {
        self.mutex.lockUncancelable(threaded_io);
        self.slots[si].ready = true;
        self.mutex.unlock(threaded_io);
        self.cond.broadcast(threaded_io);
    }

    /// Undo a claim that never published. Safe no-op on published slots.
    pub fn abandon(self: *FrameCache, si: usize) void {
        self.mutex.lockUncancelable(threaded_io);
        if (!self.slots[si].ready) self.slots[si].key = -1;
        self.mutex.unlock(threaded_io);
        self.cond.broadcast(threaded_io);
    }

    pub fn release(self: *FrameCache, idx: []const usize) void {
        self.mutex.lockUncancelable(threaded_io);
        for (idx) |si| self.slots[si].refs -= 1;
        self.mutex.unlock(threaded_io);
        self.cond.broadcast(threaded_io);
    }
};
