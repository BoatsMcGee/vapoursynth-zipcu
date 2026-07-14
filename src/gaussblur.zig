const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const pool_mod = @import("pool.zig");
const vsutil = @import("vsutil.zig");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const math = std.math;

const CreateError = cu.CreateError;
const ceilDiv = cu.ceilDiv;

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("gaussblur.cu");

const FLT_EPSILON: f32 = 1.19209290e-7;

const BLK_X: u32 = 16;
const BLK_Y: u32 = 8;
const VRT: u32 = 3;
const LARGE_R: u32 = 8;
const LARGE_THRESHOLD: i32 = 32;
const large_rh: u32 = 8;

// No -use_fast_math (default fmad contraction).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{"-std=c++17"},
    .log_name = "GaussBlur",
};

const Mode = enum { small, large };

const Config = struct {
    key: Key,
    ksize: i32,
    radius: i32,
    mode: Mode,
    weights: []const f32,

    const Key = struct { w: i32, h: i32, stride: i32, sigma: f32 };

    fn extent(self: *const Config) usize {
        return @as(usize, @intCast(self.key.stride)) * @as(usize, @intCast(self.key.h));
    }
};

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    bits: i32 = 32,
    half: bool = false,
    bytes: u32 = 4,

    process: [3]bool = .{ false, false, false },
    plane_cfg: [3]usize = .{ 0, 0, 0 },
    configs: [3]Config = undefined,
    n_cfg: usize = 0,
    any_large: bool = false,

    off_elem: [3]usize = .{ 0, 0, 0 },
    sum_elems: usize = 0,
    max_extent: usize = 0,

    zc_dst: bool = false,

    dev: cu.Device = .{},
    small_mod: [3]cu.Module = .{ .{}, .{}, .{} },
    fn_small: [3]cu.Function = .{ .{}, .{}, .{} },
    large_mod: cu.Module = .{},
    fn_v: cu.Function = .{},
    fn_h: cu.Function = .{},
    n_small: usize = 0,
    d_weights: [3]cu.DeviceBuffer = .{ .{}, .{}, .{} },
    n_weights: usize = 0,

    pool: pool_mod.Pool(Stream, Data) = .{},
};

const Stream = struct {
    stream: cu.Stream,
    cstream: cu.Stream,
    ev_up: [3]cu.Event,
    ev_k: [3]cu.Event,
    n_ev: usize,
    d_src: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,
    d_tmp: cu.DeviceBuffer,
    h_stage: ?cu.HostBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_src = try cu.DeviceBuffer.alloc(d.sum_elems * d.bytes);
        errdefer self.d_src.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(if (d.zc_dst) 0 else d.sum_elems * d.bytes);
        errdefer self.d_dst.deinit();
        self.h_stage = if (d.zc_dst) try cu.HostBuffer.alloc(d.sum_elems * d.bytes, .{}) else null;
        errdefer if (self.h_stage) |hs| hs.deinit();
        self.d_tmp = try cu.DeviceBuffer.alloc(if (d.any_large) d.max_extent * 4 else 0);
        errdefer self.d_tmp.deinit();

        self.n_ev = 0;
        errdefer self.destroyEvents();
        for (0..3) |i| {
            self.ev_up[i] = try cu.Event.init();
            self.n_ev += 1;
        }
        for (0..3) |i| {
            self.ev_k[i] = try cu.Event.init();
            self.n_ev += 1;
        }
        self.stream = try cu.Stream.init();
        errdefer self.stream.deinit();
        self.cstream = try cu.Stream.init();
    }

    fn destroyEvents(self: *Stream) void {
        var i: usize = self.n_ev;
        while (i > 0) {
            i -= 1;
            (if (i >= 3) self.ev_k[i - 3] else self.ev_up[i]).deinit();
        }
        self.n_ev = 0;
    }

    pub fn deinit(self: *Stream) void {
        self.stream.deinit();
        self.cstream.deinit();
        self.destroyEvents();
        if (self.h_stage) |hs| hs.deinit();
        self.d_tmp.deinit();
        self.d_dst.deinit();
        self.d_src.deinit();
    }
};

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, src: ZFrame, dst: ZFrameW) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain: async copies reference VS frame memory.
    errdefer {
        s.cstream.drain();
        s.stream.drain();
    }

    const num_planes: u32 = @intCast(d.vi.format.numPlanes);

    var p: u32 = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const cfg = &d.configs[d.plane_cfg[p]];
        const srcp = src.getReadSlice(p);
        std.debug.assert(srcp.len == cfg.extent() * d.bytes);
        try s.cstream.memcpyHtoD(s.d_src.at(d.off_elem[p] * d.bytes), srcp.ptr, srcp.len);
        try s.cstream.record(s.ev_up[p]);
    }

    p = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const ci = d.plane_cfg[p];
        const cfg = &d.configs[ci];
        try s.stream.waitEvent(s.ev_up[p]);

        const a_dst: cu.c.CUdeviceptr = if (s.h_stage) |hs|
            hs.devicePtr(d.off_elem[p] * d.bytes)
        else
            s.d_dst.at(d.off_elem[p] * d.bytes);
        const a_src = s.d_src.at(d.off_elem[p] * d.bytes);
        const a_wts = d.d_weights[ci].ptr;
        const w: u32 = @intCast(cfg.key.w);
        const h: u32 = @intCast(cfg.key.h);
        switch (cfg.mode) {
            .small => try s.stream.launch(d.fn_small[ci], .{
                .grid = .{ ceilDiv(w, BLK_X), ceilDiv(h, VRT * BLK_Y), 1 },
                .block = .{ BLK_X, BLK_Y, 1 },
            }, .{ a_dst, a_src, a_wts }),
            .large => {
                const a_tmp = s.d_tmp.ptr;
                const a_klen: c_int = cfg.ksize;
                const a_w: c_int = cfg.key.w;
                const a_h: c_int = cfg.key.h;
                const a_stride: c_int = cfg.key.stride;
                try s.stream.launch(d.fn_v, .{
                    .grid = .{ ceilDiv(w, BLK_X), ceilDiv(ceilDiv(h, LARGE_R), BLK_Y), 1 },
                    .block = .{ BLK_X, BLK_Y, 1 },
                }, .{ a_tmp, a_src, a_wts, a_klen, a_w, a_h, a_stride });

                const h_args = .{ a_dst, a_tmp, a_wts, a_klen, a_w, a_h, a_stride };
                try s.stream.launch(d.fn_h, .{
                    .grid = .{ ceilDiv(ceilDiv(w, large_rh), BLK_X), ceilDiv(h, BLK_Y), 1 },
                    .block = .{ BLK_X, BLK_Y, 1 },
                }, h_args);
            },
        }
        try s.stream.record(s.ev_k[p]);
    }

    if (!d.zc_dst) {
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const dstp = dst.getWriteSlice(p);
            try s.cstream.waitEvent(s.ev_k[p]);
            try s.cstream.memcpyDtoH(dstp.ptr, s.d_dst.at(d.off_elem[p] * d.bytes), dstp.len);
        }
    }

    try s.cstream.sync();
    try s.stream.sync();

    if (s.h_stage) |hs| {
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const dstp = dst.getWriteSlice(p);
            @memcpy(dstp, (hs.ptr + d.off_elem[p] * d.bytes)[0..dstp.len]);
        }
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

        process(d, s, src, dst) catch |err| {
            zapi.setFilterError("GaussBlur: process frame failed.");
            std.log.err("vszipcu GaussBlur process frame failed: {t}", .{err});
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
    freeDeviceObjects(d);
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn freeDeviceObjects(d: *Data) void {
    var i: usize = d.n_weights;
    while (i > 0) {
        i -= 1;
        d.d_weights[i].deinit();
    }
    d.n_weights = 0;
    i = d.n_small;
    while (i > 0) {
        i -= 1;
        d.small_mod[i].deinit();
    }
    d.n_small = 0;
    d.large_mod.deinit();
    d.large_mod = .{};
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize) CreateError!*Data {
    d.dev = try cu.initDevice(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    errdefer freeDeviceObjects(d);

    const idx_long: u8 = @intFromBool(d.max_extent >= (1 << 31));

    if (d.any_large) {
        const defines = std.fmt.allocPrint(allocator,
            \\#define SMALL 0
            \\#define BX {d}
            \\#define BY {d}
            \\#define R {d}
            \\#define RH {d}
            \\#define IDX_LONG {d}
            \\#define BITS {d}
            \\#define HALF {d}
            \\
        , .{ BLK_X, BLK_Y, LARGE_R, large_rh, idx_long, d.bits, @intFromBool(d.half) }) catch return error.OutOfMemory;
        defer allocator.free(defines);
        d.large_mod = try cu.compile(d.dev, .{
            .text = kernel_source,
            .defines = defines,
            .name = "gaussblur.cu",
        }, nvrtc_opts);
        d.fn_v = try d.large_mod.function("vertical_blur");
        d.fn_h = try d.large_mod.function("horizontal_blur");
    }
    for (0..d.n_cfg) |ci| {
        const cfg = &d.configs[ci];
        if (cfg.mode == .small) {
            const defines = std.fmt.allocPrint(allocator,
                \\#define SMALL 1
                \\#define W {d}
                \\#define H {d}
                \\#define STRIDE {d}
                \\#define KLEN {d}
                \\#define RAD {d}
                \\#define BLK_X {d}
                \\#define BLK_Y {d}
                \\#define VRT {d}
                \\#define IDX_LONG {d}
                \\#define BITS {d}
                \\#define HALF {d}
                \\
            , .{ cfg.key.w, cfg.key.h, cfg.key.stride, cfg.ksize, cfg.radius, BLK_X, BLK_Y, VRT, idx_long, d.bits, @intFromBool(d.half) }) catch return error.OutOfMemory;
            defer allocator.free(defines);
            d.small_mod[ci] = try cu.compile(d.dev, .{
                .text = kernel_source,
                .defines = defines,
                .name = "gaussblur.cu",
            }, nvrtc_opts);
            d.n_small = ci + 1;
            d.fn_small[ci] = try d.small_mod[ci].function("gauss_blur");
        } else {
            d.small_mod[ci] = .{};
            d.n_small = ci + 1;
        }
        d.d_weights[ci] = try cu.DeviceBuffer.alloc(cfg.weights.len * @sizeOf(f32));
        d.n_weights = ci + 1;
        try cu.driver.memcpyHtoDSync(d.d_weights[ci].ptr, cfg.weights.ptr, cfg.weights.len * @sizeOf(f32));
    }

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
    {
        return map_out.setError("GaussBlur: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    }
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("GaussBlur: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("GaussBlur: num_streams must be 1..32.");
    };

    // Chroma default: sqrt/div in DOUBLE then narrow.
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    var sigma: [3]f32 = undefined;
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            if (map_in.getValue2(f32, "sigma", i)) |given| {
                if (!math.isFinite(given) or given < 0)
                    return map_out.setError("GaussBlur: sigma must be a finite value >= 0.");
                sigma[i] = given;
            } else if (i == 0) {
                sigma[i] = 0.5;
            } else if (i == 1) {
                const prod = (@as(u32, 1) << subH) * (@as(u32, 1) << subW);
                const sub_factor = @sqrt(@as(f64, @floatFromInt(prod)));
                sigma[i] = @floatCast(@as(f64, sigma[0]) / sub_factor);
            } else {
                sigma[i] = sigma[i - 1];
            }
        }
    }

    const num_planes: usize = @intCast(fmt.numPlanes);
    var any_proc = false;
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            d.process[i] = i < num_planes and sigma[i] >= FLT_EPSILON;
            if (d.process[i]) any_proc = true;
        }
    }
    if (!any_proc) {
        return map_out.setError("GaussBlur: all planes have sigma < FLT_EPSILON (nothing to process).");
    }

    const strides = vsutil.strideFromVi(d.vi);
    d.n_cfg = 0;
    defer {
        var wi: usize = 0;
        while (wi < d.n_cfg) : (wi += 1) allocator.free(d.configs[wi].weights);
    }
    {
        var pi: usize = 0;
        while (pi < num_planes) : (pi += 1) {
            if (!d.process[pi]) continue;
            const key: Config.Key = .{
                .w = if (pi == 0) d.vi.width else d.vi.width >> subW,
                .h = if (pi == 0) d.vi.height else d.vi.height >> subH,
                .stride = @intCast(if (pi == 0) strides[0] else strides[1]),
                .sigma = sigma[pi],
            };
            var ci: usize = 0;
            while (ci < d.n_cfg) : (ci += 1) {
                if (std.meta.eql(key, d.configs[ci].key)) break;
            }
            if (ci == d.n_cfg) {
                if (key.sigma > @as(f32, @floatFromInt(@min(key.w, key.h)))) {
                    return map_out.setError("GaussBlur: sigma too large for plane (radius >= dimension).");
                }
                const weights = getGaussKernel(key.sigma) catch return map_out.setError("GaussBlur: out of memory.");
                const ksize: i32 = @intCast(weights.len);
                const radius = @divTrunc(ksize, 2);
                // Single-fold reflect requires radius <= dim-1.
                if (radius > key.w - 1 or radius > key.h - 1) {
                    allocator.free(weights);
                    return map_out.setError("GaussBlur: sigma too large for plane (radius >= dimension).");
                }
                d.configs[ci] = .{
                    .key = key,
                    .ksize = ksize,
                    .radius = radius,
                    .mode = if (radius > LARGE_THRESHOLD) .large else .small,
                    .weights = weights,
                };
                d.n_cfg += 1;
            }
            d.plane_cfg[pi] = ci;
        }
    }

    d.any_large = false;
    {
        var sum: usize = 0;
        var pi: usize = 0;
        while (pi < num_planes) : (pi += 1) {
            if (!d.process[pi]) continue;
            const cfg = &d.configs[d.plane_cfg[pi]];
            d.off_elem[pi] = sum;
            sum += cfg.extent();
            d.max_extent = @max(d.max_extent, cfg.extent());
            if (cfg.mode == .large) d.any_large = true;
        }
        d.sum_elems = sum;
        std.debug.assert(sum > 0);
    }
    if (d.max_extent >= (1 << 31) or d.sum_elems * d.bytes >= (1 << 32))
        return map_out.setError("GaussBlur: frame too large (a plane exceeds 2^31 samples).");
    if (@divTrunc(d.vi.height + 7, 8) > 65535)
        return map_out.setError("GaussBlur: frame too tall for the CUDA launch grid.");

    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 1;

    d.zc_dst = num_streams >= 2 and d.bytes == 4 and !d.any_large;

    const data = initCuda(&d, device_id, num_streams) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "GaussBlur: invalid device ID.",
            error.Nvrtc => "GaussBlur: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "GaussBlur: could not locate NVRTC (put nvrtc64_130_0.dll next to the plugin, or: pip install nvidia-cuda-nvrtc).",
            error.OutOfDeviceMemory => "GaussBlur: out of device memory.",
            else => "GaussBlur: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu GaussBlur init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "GaussBlur", d.vi, getFrame, free, .Parallel, &dep, data);
}

fn getGaussKernel(sigma: f32) ![]f32 {
    var taps: usize = @intFromFloat(@ceil(sigma * 6 + 1));
    if (taps % 2 == 0) {
        taps += 1;
    }

    var kernel = try std.ArrayList(f64).initCapacity(allocator, taps);
    defer kernel.deinit(allocator);

    const half_taps = @divFloor(taps, 2);
    var x: usize = 0;
    while (x < half_taps) : (x += 1) {
        const x_f64 = @as(f64, @floatFromInt(x));
        const value = 1.0 / (@sqrt(2.0 * math.pi) * sigma) *
            @exp(-(x_f64 * x_f64) / (2 * sigma * sigma));
        try kernel.append(allocator, value);
    }

    const first_value = kernel.items[0];
    for (kernel.items[1..]) |*item| {
        item.* *= 1 / first_value;
    }
    kernel.items[0] = 1;

    var full_kernel = try std.ArrayList(f64).initCapacity(allocator, taps);
    defer full_kernel.deinit(allocator);

    var i: usize = kernel.items.len;
    while (i > 0) : (i -= 1) {
        try full_kernel.append(allocator, kernel.items[i - 1]);
    }
    try full_kernel.appendSlice(allocator, kernel.items[1..]);

    var sum: f64 = 0;
    for (full_kernel.items) |v| sum += v;

    const out_kernel = try allocator.alloc(f32, full_kernel.items.len);
    for (out_kernel, full_kernel.items) |*s, f| {
        s.* = @floatCast(f / sum);
    }

    return out_kernel;
}
