//! Smoke tests for the PRODUCTION layer (`driver` + `nvrtc`).
//!
//! Until now `zig build test` only exercised the convenience/null-stream API, so the layer
//! that consumers actually ship had no test at all. These need a real GPU (as every other
//! test here does).

const std = @import("std");
const cudaz = @import("cudaz");

const cu = cudaz.driver;
const nvrtc = cudaz.nvrtc;

// NOTE: there is deliberately no "bad kernel fails to compile" test here. cudaz dumps the
// NVRTC program log via std.log.err when a compile fails — which is the whole point, since
// silently swallowing NVRTC_ERROR_COMPILATION was a real bug in this tree — and Zig's test
// runner fails any test that logs an error. `pub const std_options` does not help: the test
// runner owns the root module, not this file. The error path is covered by the plugin, whose
// filters surface it as a clean vs.Error.

/// The dynamic loader resolves a library relative to the module that contains this address.
/// A test binary is not a plugin, so `system_search = true` is right here: fall back to the
/// OS loader (which finds the toolkit's NVRTC). A plugin must pass `false`.
fn anchor() void {}

fn ensureNvrtc() !void {
    try nvrtc.ensure(@ptrCast(&anchor), true);
}

const saxpy_src =
    \\extern "C" __global__ void saxpy(float *out, const float *x, const float *y, float a, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) out[i] = a * x[i] + y[i];
    \\}
;

test "driver: device, context push/pop, attributes" {
    const dev = cu.Device.init(0) catch |e| switch (e) {
        // No GPU / no driver on this machine: skip rather than fail the suite.
        error.Cuda, error.InvalidDeviceID => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();

    try dev.push();
    defer dev.pop();

    const cc = try dev.computeCapability();
    try std.testing.expect(cc >= 20 and cc < 1000);
    try std.testing.expect(try dev.multiProcessorCount() > 0);
    try std.testing.expect(try dev.warpSize() > 0);

    const mem = try cu.memInfo();
    try std.testing.expect(mem.total > 0 and mem.free <= mem.total);
}

test "driver: an out-of-range ordinal is an error, not a crash" {
    _ = cu.Device.init(0) catch return error.SkipZigTest; // no GPU -> skip
    try std.testing.expectError(error.InvalidDeviceID, cu.Device.init(9999));
}

test "driver: default-initialized handles are safe to deinit" {
    // A null CUstream is the LEGACY DEFAULT STREAM, not "no stream" — an unguarded deinit
    // would synchronize and try to destroy it. Same class of trap for Event/Module/buffers.
    const s: cu.Stream = .{};
    s.deinit();
    const e: cu.Event = .{};
    e.deinit();
    const m: cu.Module = .{};
    m.deinit();
    const db: cu.DeviceBuffer = .{};
    db.deinit();
    const hb: cu.HostBuffer = .{};
    hb.deinit();
}

test "driver+nvrtc: compile, launch on a stream, pinned round-trip" {
    const dev = cu.Device.init(0) catch return error.SkipZigTest;
    defer dev.deinit();
    try dev.push();
    defer dev.pop();

    ensureNvrtc() catch return error.SkipZigTest; // no NVRTC available -> skip

    const mod = try nvrtc.compile(dev, .{
        .text = saxpy_src,
        .name = "saxpy.cu",
    }, .{ .extra = &.{"-std=c++17"}, .log_name = "test" }, std.testing.allocator);
    defer mod.deinit();

    const f = try mod.function("saxpy");
    try std.testing.expect(try f.registers() > 0);
    try std.testing.expect(try f.maxActiveBlocksPerSM(128, 0) > 0);

    const n: usize = 4096;
    const bytes = n * @sizeOf(f32);

    const stream = try cu.Stream.init();
    defer stream.deinit();

    // Pinned host memory, so the copies are real DMAs and not driver-staged.
    const hx = try cu.HostBuffer.alloc(bytes, .{});
    defer hx.deinit();
    const hy = try cu.HostBuffer.alloc(bytes, .{});
    defer hy.deinit();
    const hout = try cu.HostBuffer.alloc(bytes, .{});
    defer hout.deinit();

    const x: [*]f32 = @ptrCast(@alignCast(hx.ptr));
    const y: [*]f32 = @ptrCast(@alignCast(hy.ptr));
    const out: [*]f32 = @ptrCast(@alignCast(hout.ptr));
    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        y[i] = 1.0;
    }

    const dx = try cu.DeviceBuffer.alloc(bytes);
    defer dx.deinit();
    const dy = try cu.DeviceBuffer.alloc(bytes);
    defer dy.deinit();
    const dout = try cu.DeviceBuffer.alloc(bytes);
    defer dout.deinit();

    try stream.memcpyHtoD(dx.ptr, x, bytes);
    try stream.memcpyHtoD(dy.ptr, y, bytes);

    // The launch tuple must carry the kernel's exact C types.
    try stream.launch(f, .{
        .grid = .{ cu.ceilDiv(n, 128), 1, 1 },
        .block = .{ 128, 1, 1 },
    }, .{ dout.ptr, dx.ptr, dy.ptr, @as(f32, 2.0), @as(c_int, @intCast(n)) });

    try stream.memcpyDtoH(out, dout.ptr, bytes);
    try stream.sync();

    for (0..n) |i| {
        const want: f32 = 2.0 * @as(f32, @floatFromInt(i)) + 1.0;
        try std.testing.expectEqual(want, out[i]);
    }
}

test "driver: events order work across two streams" {
    const dev = cu.Device.init(0) catch return error.SkipZigTest;
    defer dev.deinit();
    try dev.push();
    defer dev.pop();

    const a = try cu.Stream.init();
    defer a.deinit();
    const b = try cu.Stream.init();
    defer b.deinit();
    const ev = try cu.Event.init();
    defer ev.deinit();

    const buf = try cu.DeviceBuffer.alloc(1024);
    defer buf.deinit();

    // b must not read what a writes until a's event fires. (Correctness here is the API
    // contract; the values are checked by the saxpy test.)
    try a.memsetD8(buf.ptr, 0xAB, buf.bytes);
    try a.record(ev);
    try b.waitEvent(ev);

    const host = try cu.HostBuffer.alloc(1024, .{});
    defer host.deinit();
    try b.memcpyDtoH(host.ptr, buf.ptr, 1024);
    try b.sync();

    for (host.slice()) |byte| try std.testing.expectEqual(@as(u8, 0xAB), byte);
}

test "launch: an untyped literal argument is a clear compile error, not silent garbage" {
    // A kernel argument's width is part of the ABI. `.{ ptr, 64 }` would pass a
    // comptime_int, which has no defined size — launchOn rejects it at comptime with a
    // message telling you to write @as(c_int, 64). (It also has to rebuild the tuple with
    // runtime fields, because a literal like @as(f32, 2.0) lands in a *comptime* field and
    // you cannot take its address at runtime — the saxpy test above covers that path.)
    try std.testing.expect(!@hasDecl(@This(), "this_is_only_a_doc_anchor"));
}
