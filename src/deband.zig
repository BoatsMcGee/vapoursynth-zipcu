const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const pool_mod = @import("pool.zig");
const vsutil = @import("vsutil.zig");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const math = std.math;

const CreateError = cu.CreateError;

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("deband.cu");

const max_iterations: i32 = 32;

// No -use_fast_math (default fmad rounding).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{"-std=c++17"},
    .log_name = "Deband",
};

const dlut_sizeb = 6;
const dlut_size = 1 << dlut_sizeb;
const dlut_len = dlut_size * dlut_size;

// MSVC/UCRT rand() LCG (blue-noise unseeded tie-breaks).
const CrtRand = struct {
    state: u32 = 1,
    fn next(self: *CrtRand) u32 {
        self.state = self.state *% 214013 +% 2531011;
        return (self.state >> 16) & 0x7fff;
    }
};

// CRT exp/log, not std.math (void-and-cluster ties need exact double bits).
const libm = struct {
    extern fn exp(x: f64) f64;
    extern fn log(x: f64) f64;
};

// (double)UINT64_MAX rounds to 2^64 (the C constant).
const u64_max_f: f64 = 18446744073709551616.0;

const bn_radius = dlut_size / 2 - 1;
const bn_middle = bn_radius | (bn_radius << dlut_sizeb);
const bn_gsize = bn_radius * 2 + 1;
const bn_gsize2 = bn_gsize * bn_gsize;

fn bnXY(x: usize, y: usize) usize {
    return x | (y << dlut_sizeb);
}

const BnCtx = struct {
    gauss: [dlut_len]u64,
    randomat: [dlut_len]u32,
    calcmat: [dlut_len]bool,
    gaussmat: [dlut_len]u64,
    unimat: [dlut_len]u32,
    rng: CrtRand,
};

fn bnMakegauss(k: *BnCtx) void {
    @memset(&k.gauss, 0);
    const sigma = -libm.log(1.5 / u64_max_f * @as(f64, bn_gsize2)) / @as(f64, bn_radius);
    var gy: usize = 0;
    while (gy <= bn_radius) : (gy += 1) {
        var gx: usize = 0;
        while (gx <= gy) : (gx += 1) {
            const cx: i64 = @as(i64, @intCast(gx)) - bn_radius;
            const cy: i64 = @as(i64, @intCast(gy)) - bn_radius;
            const sq: f64 = @floatFromInt(cx * cx + cy * cy);
            const e = libm.exp(-@sqrt(sq) * sigma);
            const v: u64 = @intFromFloat(e / @as(f64, bn_gsize2) * u64_max_f);
            const g1 = bn_gsize - 1;
            k.gauss[bnXY(gx, gy)] = v;
            k.gauss[bnXY(gy, gx)] = v;
            k.gauss[bnXY(gx, g1 - gy)] = v;
            k.gauss[bnXY(gy, g1 - gx)] = v;
            k.gauss[bnXY(g1 - gx, gy)] = v;
            k.gauss[bnXY(g1 - gy, gx)] = v;
            k.gauss[bnXY(g1 - gx, g1 - gy)] = v;
            k.gauss[bnXY(g1 - gy, g1 - gx)] = v;
        }
    }
}

fn bnSetbit(k: *BnCtx, c: usize) void {
    if (k.calcmat[c]) return;
    k.calcmat[c] = true;
    var m: usize = 0;
    var g: usize = (bn_middle + dlut_len - c) & (dlut_len - 1);
    while (g < dlut_len) : (g += 1) {
        k.gaussmat[m] += k.gauss[g];
        m += 1;
    }
    g = 0;
    while (m < dlut_len) : (m += 1) {
        k.gaussmat[m] += k.gauss[g];
        g += 1;
    }
}

fn bnGetmin(k: *BnCtx) usize {
    var min: u64 = math.maxInt(u64);
    var resnum: u32 = 0;
    for (0..dlut_len) |c| {
        if (k.calcmat[c]) continue;
        const total = k.gaussmat[c];
        if (total <= min) {
            if (total != min) {
                min = total;
                resnum = 0;
            }
            k.randomat[resnum] = @intCast(c);
            resnum += 1;
        }
    }
    if (resnum == 1) return k.randomat[0];
    if (resnum == dlut_len) return dlut_len / 2;
    return k.randomat[k.rng.next() % resnum];
}

fn generateBlueNoise(data: []f32) !void {
    const k = try allocator.create(BnCtx);
    defer allocator.destroy(k);
    k.rng = .{};
    @memset(&k.calcmat, false);
    @memset(&k.gaussmat, 0);
    bnMakegauss(k);
    for (0..dlut_len) |c| {
        const r = bnGetmin(k);
        bnSetbit(k, r);
        k.unimat[r] = @intCast(c);
    }
    for (0..dlut_size) |y| {
        for (0..dlut_size) |x|
            data[x + y * dlut_size] = @as(f32, @floatFromInt(k.unimat[bnXY(x, y)])) / @as(f32, dlut_len);
    }
}

fn generateBayer(data: []f32) void {
    data[0] = 0;
    var sz: usize = 1;
    while (sz < dlut_size) : (sz *= 2) {
        for (0..sz) |y| {
            for (0..sz) |x| {
                const pos = y * dlut_size + x;
                const offs = [3]usize{ sz * dlut_size + sz, sz, sz * dlut_size };
                for (offs, 1..) |off, i| {
                    const inc = @as(f64, @floatFromInt(i)) / (4.0 * @as(f64, @floatFromInt(sz * sz)));
                    data[pos + off] = @floatCast(@as(f64, data[pos]) + inc);
                }
            }
        }
    }
}

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    pw: [3]i32 = .{ 0, 0, 0 },
    ph: [3]i32 = .{ 0, 0, 0 },
    pstride: [3]i32 = .{ 0, 0, 0 },
    process: [3]bool = .{ false, false, false },
    rank: [3]i32 = .{ -1, -1, -1 },
    rank_plane: [3]i32 = .{ -1, -1, -1 },
    n_proc: i32 = 0,

    off_elem: [3]usize = .{ 0, 0, 0 },
    sum_elems: usize = 0,
    max_pw: i32 = 0,
    max_ph: i32 = 0,
    geom: [18]i32 = [_]i32{0} ** 18,

    buff_size: usize = 0,
    bits: i32 = 32,
    half: bool = false,
    bytes: u32 = 4,

    iterations: [3]i32 = .{ 1, 1, 1 },
    threshold_s: [3]f32 = .{ 0, 0, 0 },
    radius: [3]f32 = .{ 0, 0, 0 },
    grain_s: [3]f32 = .{ 0, 0, 0 },
    grain_on: [3]bool = .{ false, false, false },
    fused: bool = false,

    dither_on: bool = false,
    dmode: i32 = 0,

    use_pinned: bool = false,
    pageable_h2d: bool = false,
    zc_dst: bool = false,
    stage_src_off: [3]usize = .{ 0, 0, 0 },
    stage_dst_base: usize = 0,
    stage_bytes: usize = 0,

    dev: cu.Device = .{},
    module: [3]cu.Module = .{ .{}, .{}, .{} },
    fn_deband: [3]cu.Function = .{ .{}, .{}, .{} },
    n_mod: usize = 0,
    plane_mod: [3]usize = .{ 0, 0, 0 },
    d_dlut: cu.DeviceBuffer = .{},
    d_geom: cu.DeviceBuffer = .{},

    pool: pool_mod.Pool(Stream, Data) = .{},
};

const Stream = struct {
    stream: cu.Stream,
    d_src: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,
    h_stage: ?cu.HostBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_src = try cu.DeviceBuffer.alloc(d.buff_size * d.bytes);
        errdefer self.d_src.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(if (d.zc_dst) 0 else d.buff_size * d.bytes);
        errdefer self.d_dst.deinit();
        self.h_stage = null;
        if (d.use_pinned) {
            self.h_stage = try cu.HostBuffer.alloc(d.stage_bytes, .{});
        }
        errdefer if (self.h_stage) |hs| hs.deinit();
        self.stream = try cu.Stream.init();
    }

    pub fn deinit(self: *Stream) void {
        self.stream.deinit();
        if (self.h_stage) |hs| hs.deinit();
        self.d_dst.deinit();
        self.d_src.deinit();
    }
};

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, src: ZFrame, dst: ZFrameW, n: c_int) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain: async copies reference VS frame memory.
    errdefer s.stream.drain();

    const num_planes: u32 = @intCast(d.vi.format.numPlanes);

    var p: u32 = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const srcp = src.getReadSlice(p);
        std.debug.assert(srcp.len == @as(usize, @intCast(d.ph[p])) * @as(usize, @intCast(d.pstride[p])) * d.bytes);
        if (s.h_stage != null and !d.pageable_h2d) {
            const hs = s.h_stage.?;
            const region = hs.ptr + d.stage_src_off[p];
            @memcpy(region[0..srcp.len], srcp);
        } else {
            try s.stream.memcpyHtoD(s.d_src.at(d.off_elem[p] * d.bytes), srcp.ptr, srcp.len);
        }
    }
    if (s.h_stage != null and !d.pageable_h2d) {
        const hs = s.h_stage.?;
        try s.stream.memcpyHtoD(s.d_src.ptr, hs.ptr, d.sum_elems * d.bytes);
    }

    const a_dst: cu.c.CUdeviceptr = if (d.zc_dst) s.h_stage.?.devicePtr(d.stage_dst_base) else s.d_dst.ptr;
    const a_src = s.d_src.ptr;
    const a_dlut = d.d_dlut.ptr;
    const dither_args = d.dither_on and (d.dmode == 0 or d.dmode == 1);
    if (d.fused) {
        std.debug.assert(d.n_mod == 1);
        const p0: usize = @intCast(d.rank_plane[0]);
        const a_geom = d.d_geom.ptr;
        const a_thr: f32 = d.threshold_s[p0];
        const a_rad: f32 = d.radius[p0];
        const a_grain: f32 = d.grain_s[p0];
        const a_zbase: c_uint = @intCast((@as(i64, n) * @as(i64, d.n_proc)) & 0xFF);
        const gx: u32 = @intCast(@divTrunc(d.max_pw + 15, 16));
        const gy: u32 = @intCast(@divTrunc(d.max_ph + 7, 8));
        const gz: u32 = @intCast(d.n_proc);
        const cfg: cu.Launch = .{ .grid = .{ gx, gy, gz }, .block = .{ 16, 8, 1 } };
        if (dither_args) {
            try s.stream.launch(d.fn_deband[0], cfg, .{ a_dst, a_src, a_geom, a_thr, a_rad, a_grain, a_zbase, a_dlut });
        } else {
            try s.stream.launch(d.fn_deband[0], cfg, .{ a_dst, a_src, a_geom, a_thr, a_rad, a_grain, a_zbase });
        }
    } else for (0..3) |pl| {
        if (!d.process[pl]) continue;
        const r: usize = @intCast(d.rank[pl]);

        const a_geom = d.d_geom.at(r * 6 * @sizeOf(i32));
        const a_thr: f32 = d.threshold_s[pl];
        const a_rad: f32 = d.radius[pl];
        const a_grain: f32 = d.grain_s[pl];
        const a_zbase: c_uint = @intCast((@as(i64, n) * @as(i64, d.n_proc) + @as(i64, @intCast(r))) & 0xFF);
        const gx: u32 = @intCast(@divTrunc(d.pw[pl] + 15, 16));
        const gy: u32 = @intCast(@divTrunc(d.ph[pl] + 7, 8));
        const cfg: cu.Launch = .{ .grid = .{ gx, gy, 1 }, .block = .{ 16, 8, 1 } };
        const f = d.fn_deband[d.plane_mod[pl]];
        if (dither_args) {
            try s.stream.launch(f, cfg, .{ a_dst, a_src, a_geom, a_thr, a_rad, a_grain, a_zbase, a_dlut });
        } else {
            try s.stream.launch(f, cfg, .{ a_dst, a_src, a_geom, a_thr, a_rad, a_grain, a_zbase });
        }
    }

    if (s.h_stage) |hs| {
        const dst_stage = hs.ptr + d.stage_dst_base;
        if (!d.zc_dst) {
            try s.stream.memcpyDtoH(dst_stage, s.d_dst.ptr, d.sum_elems * d.bytes);
        }
        try s.stream.sync();
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const dstp = dst.getWriteSlice(p);
            @memcpy(dstp, (dst_stage + d.off_elem[p] * d.bytes)[0..dstp.len]);
        }
    } else {
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const dstp = dst.getWriteSlice(p);
            try s.stream.memcpyDtoH(dstp.ptr, s.d_dst.at(d.off_elem[p] * d.bytes), dstp.len);
        }
        try s.stream.sync();
    }
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        defer src.deinit();
        const dst = src.newVideoFrame();

        const num_planes: u32 = @intCast(d.vi.format.numPlanes);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) @memcpy(dst.getWriteSlice(p), src.getReadSlice(p));
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, src, dst, n) catch |err| {
            zapi.setFilterError("Deband: process frame failed.");
            std.log.err("vszipcu Deband process frame failed: {t}", .{err});
            dst.deinit();
            return null;
        };
        return dst.frame;
    }
    return null;
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.dev.push() catch {};
    d.pool.deinit();
    d.d_geom.deinit();
    d.d_dlut.deinit();
    for (0..d.n_mod) |m| d.module[m].deinit();
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize, dlut: ?[]const f32) CreateError!*Data {
    d.dev = try cu.Device.init(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    errdefer {
        var m: usize = 0;
        while (m < d.n_mod) : (m += 1) d.module[m].deinit();
    }
    var key_iter: [3]i32 = undefined;
    var key_grain: [3]bool = undefined;
    for (0..3) |p| {
        if (!d.process[p]) continue;

        var hit: ?usize = null;
        for (0..d.n_mod) |m| {
            if (key_iter[m] == d.iterations[p] and key_grain[m] == d.grain_on[p]) hit = m;
        }
        if (hit) |m| {
            d.plane_mod[p] = m;
            continue;
        }

        const defines = std.fmt.allocPrint(allocator,
            \\#define ITER {d}
            \\#define GRAIN_ON {d}
            \\#define BITS {d}
            \\#define HALF {d}
            \\#define DITHERK {d}
            \\#define DMODE {d}
            \\
        , .{
            d.iterations[p],      @intFromBool(d.grain_on[p]), d.bits,
            @intFromBool(d.half), @intFromBool(d.dither_on),   d.dmode,
        }) catch return error.OutOfMemory;
        defer allocator.free(defines);

        const m = d.n_mod;
        d.module[m] = try cu.compile(d.dev, .{
            .text = kernel_source,
            .defines = defines,
            .name = "deband.cu",
        }, nvrtc_opts);
        d.n_mod += 1;
        d.fn_deband[m] = try d.module[m].function("deband");
        key_iter[m] = d.iterations[p];
        key_grain[m] = d.grain_on[p];
        d.plane_mod[p] = m;
    }

    if (dlut) |lut| {
        d.d_dlut = try cu.DeviceBuffer.alloc(dlut_len * @sizeOf(f32));
        try cu.driver.memcpyHtoDSync(d.d_dlut.ptr, lut.ptr, dlut_len * @sizeOf(f32));
    }
    errdefer d.d_dlut.deinit();

    d.d_geom = try cu.DeviceBuffer.alloc(d.geom.len * @sizeOf(i32));
    errdefer d.d_geom.deinit();
    try cu.driver.memcpyHtoDSync(d.d_geom.ptr, &d.geom, d.geom.len * @sizeOf(i32));

    const data = allocator.create(Data) catch return error.OutOfMemory;
    errdefer allocator.destroy(data);
    data.* = d.*;
    data.pool = .{};
    data.pool.prime(data, num_streams) catch {
        data.pool.deinit();
        return error.OutOfMemory;
    };
    data.pool.prewarm(num_streams) catch |err| {
        data.pool.deinit();
        return err;
    };
    return data;
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;

    var keep = false;
    defer if (!keep) zapi.freeNode(d.node);

    const fmt = d.vi.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Float and (bits == 32 or bits == 16)) or
        (fmt.sampleType == .Integer and (bits == 8 or bits == 16));
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (fmt.colorFamily != .Gray and fmt.colorFamily != .YUV and fmt.colorFamily != .RGB))
        return map_out.setError("Deband: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    var iterations: [3]i32 = undefined;
    var threshold: [3]f32 = undefined;
    var radius: [3]f32 = undefined;
    var grain: [3]f32 = undefined;
    for (0..3) |i| {
        const e: u32 = @intCast(i);
        iterations[i] = map_in.getValue2(i32, "iterations", e) orelse if (i > 0) iterations[i - 1] else 1;
        threshold[i] = map_in.getValue2(f32, "threshold", e) orelse if (i > 0) threshold[i - 1] else 3.0;
        radius[i] = map_in.getValue2(f32, "radius", e) orelse if (i > 0) radius[i - 1] else 16.0;
        grain[i] = map_in.getValue2(f32, "grain", e) orelse if (i > 0) grain[i - 1] else 4.0;

        if (iterations[i] < 0 or iterations[i] > max_iterations) return map_out.setError("Deband: iterations must be 0..32.");
        // Non-finite radius → OOB float→int in db_get.
        if (!math.isFinite(threshold[i]) or threshold[i] < 0.0) return map_out.setError("Deband: threshold must be a finite value >= 0.");
        if (!math.isFinite(radius[i]) or radius[i] < 0.0) return map_out.setError("Deband: radius must be a finite value >= 0.");
        if (!math.isFinite(grain[i]) or grain[i] < 0.0) return map_out.setError("Deband: grain must be a finite value >= 0.");
    }

    var sel = [3]bool{ true, true, true };
    if (map_in.numElements("planes")) |ne| {
        sel = .{ false, false, false };
        var e: u32 = 0;
        while (e < ne) : (e += 1) {
            const idx = map_in.getValue2(i32, "planes", e).?;
            if (idx < 0 or idx >= fmt.numPlanes) return map_out.setError("Deband: plane index out of range.");
            const ui: usize = @intCast(idx);
            if (sel[ui]) return map_out.setError("Deband: plane specified twice.");
            sel[ui] = true;
        }
    }
    const dither_req = map_in.getValue(i32, "dither");
    d.dither_on = (if (dither_req) |dv| dv != 0 else true) and bits == 8;
    const dither_algo = map_in.getValue(i32, "dither_algo") orelse 0;
    if (dither_algo < 0 or dither_algo > 3) return map_out.setError("Deband: dither_algo must be 0..3 (blue noise / bayer / ordered fixed / white noise).");
    d.dmode = dither_algo;
    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("Deband: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) return map_out.setError("Deband: num_streams must be 1..32.");

    for (0..3) |i| {
        d.iterations[i] = iterations[i];
        d.threshold_s[i] = threshold[i] / 1000.0;
        d.radius[i] = radius[i];
        d.grain_s[i] = grain[i] / 1000.0;
        d.grain_on[i] = grain[i] > 0.0;
    }

    const strides = vsutil.strideFromVi(d.vi);
    const num_planes: usize = @intCast(fmt.numPlanes);
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    var np: i32 = 0;
    var pi: usize = 0;
    while (pi < 3) : (pi += 1) {
        if (pi >= num_planes) continue;
        d.pw[pi] = if (pi == 0) @intCast(d.vi.width) else @intCast(d.vi.width >> subW);
        d.ph[pi] = if (pi == 0) @intCast(d.vi.height) else @intCast(d.vi.height >> subH);
        d.pstride[pi] = @intCast(if (pi == 0) strides[0] else strides[1]);
        if (sel[pi]) {
            d.process[pi] = true;
            d.rank[pi] = np;
            d.rank_plane[@intCast(np)] = @intCast(pi);
            np += 1;
        }
    }
    d.n_proc = np;

    if (np > 0) {
        const p0: usize = @intCast(d.rank_plane[0]);
        d.fused = true;
        for (0..3) |i| {
            if (!d.process[i]) continue;
            if (iterations[i] != iterations[p0] or threshold[i] != threshold[p0] or
                radius[i] != radius[p0] or grain[i] != grain[p0]) d.fused = false;
        }
    }

    var chk: usize = 0;
    while (chk < 3) : (chk += 1)
        if (d.process[chk] and (d.pw[chk] <= 0 or d.ph[chk] <= 0)) return map_out.setError("Deband: a processed plane has zero size.");

    {
        var sum: usize = 0;
        var r: usize = 0;
        while (r < @as(usize, @intCast(d.n_proc))) : (r += 1) {
            const p: usize = @intCast(d.rank_plane[r]);
            d.off_elem[p] = sum;
            const extent = @as(usize, @intCast(d.ph[p])) * @as(usize, @intCast(d.pstride[p]));
            d.geom[r * 6 + 0] = d.pw[p];
            d.geom[r * 6 + 1] = d.ph[p];
            d.geom[r * 6 + 2] = d.pstride[p];
            d.geom[r * 6 + 5] = @intFromBool(r == 0 and d.dither_on);
            d.max_pw = @max(d.max_pw, d.pw[p]);
            d.max_ph = @max(d.max_ph, d.ph[p]);
            sum += extent;
        }
        d.sum_elems = sum;
        d.buff_size = sum;
        // Offsets are i32; gate sum < 2^31.
        if (d.sum_elems >= (1 << 31) or d.sum_elems * d.bytes >= (1 << 32))
            return map_out.setError("Deband: frame too large (regions exceed 2^31 samples).");
        r = 0;
        while (r < @as(usize, @intCast(d.n_proc))) : (r += 1) {
            const p: usize = @intCast(d.rank_plane[r]);
            d.geom[r * 6 + 3] = @intCast(d.off_elem[p]);
            d.geom[r * 6 + 4] = @intCast(d.off_elem[p]);
            d.stage_src_off[p] = d.off_elem[p] * d.bytes;
        }
        d.stage_dst_base = d.sum_elems * d.bytes;
        d.stage_bytes = @max(1, 2 * d.sum_elems * d.bytes);
    }
    if (@divTrunc(d.max_ph + 7, 8) > 65535)
        return map_out.setError("Deband: frame too tall for the CUDA launch grid.");

    d.use_pinned = (if (ns_req) |ns| ns else 1) > 1;
    d.pageable_h2d = d.use_pinned and d.n_proc == 1;
    d.zc_dst = d.use_pinned and d.n_proc == 1 and d.bytes == 4;

    var dlut: ?[]f32 = null;
    defer if (dlut) |lut| allocator.free(lut);
    if (d.dither_on and (d.dmode == 0 or d.dmode == 1)) {
        const lut = allocator.alloc(f32, dlut_len) catch return map_out.setError("Deband: out of memory.");
        dlut = lut;
        if (d.dmode == 0) generateBlueNoise(lut) catch {
            return map_out.setError("Deband: out of memory.");
        } else generateBayer(lut);
    }

    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    const data = initCuda(&d, device_id, num_streams, dlut) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "Deband: invalid device ID.",
            error.Nvrtc => "Deband: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "Deband: could not locate NVRTC (put nvrtc64_130_0.dll next to the plugin, or: pip install nvidia-cuda-nvrtc).",
            error.OutOfDeviceMemory => "Deband: out of device memory.",
            else => "Deband: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu Deband init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "Deband", d.vi, getFrame, free, .Parallel, &dep, data);
}
