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

const kernel_source = @embedFile("fgrain.cu");

// UCRT libm (not std.math) for host tables.
extern "c" fn logf(f32) f32;
extern "c" fn expf(f32) f32;

// -use_fast_math (parity).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{ "-use_fast_math", "-std=c++17" },
    .log_name = "FGrain",
};

// MSVC mt19937 + normal_distribution<float> (not standardized — any deviation changes grain).

const Mt19937 = struct {
    mt: [624]u32,
    idx: usize,

    fn init(seed: u32) Mt19937 {
        var self: Mt19937 = .{ .mt = undefined, .idx = 624 };
        self.mt[0] = seed;
        var i: u32 = 1;
        while (i < 624) : (i += 1) {
            self.mt[i] = 1812433253 *% (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) +% i;
        }
        return self;
    }

    fn gen(self: *Mt19937) u32 {
        if (self.idx >= 624) {
            for (0..624) |i| {
                const y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF);
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1);
                if (y & 1 != 0) self.mt[i] ^= 0x9908B0DF;
            }
            self.idx = 0;
        }
        var y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;
        return y;
    }
};

fn nrandF32(g: *Mt19937) f32 {
    return @as(f32, @floatFromInt(g.gen() >> 8)) * 0x1p-24;
}

// Marsaglia polar in FLOAT + Sx<=1e-4 rescale (skipping rescale desyncs the stream).
const NormalF32 = struct {
    sigma: f32,
    valid: bool = false,
    xx2: f32 = 0,

    fn next(self: *NormalF32, g: *Mt19937) f32 {
        var res: f32 = undefined;
        if (self.valid) {
            res = self.xx2;
            self.valid = false;
        } else {
            var v1: f32 = undefined;
            var v2: f32 = undefined;
            var sx: f32 = undefined;
            while (true) {
                v1 = 2.0 * nrandF32(g) - 1.0;
                v2 = 2.0 * nrandF32(g) - 1.0;
                sx = v1 * v1 + v2 * v2;
                if (sx < 1.0 and v1 != 0.0 and v2 != 0.0) break;
            }
            var logsx: f32 = undefined;
            if (sx > 1e-4) {
                logsx = logf(sx);
            } else {
                const ln2: f32 = 0.69314718055994530941723212145818;
                const maxabs = @max(@abs(v1), @abs(v2));
                const expmax = math.ilogb(maxabs);
                v1 = math.scalbn(v1, -expmax);
                v2 = math.scalbn(v2, -expmax);
                sx = v1 * v1 + v2 * v2;
                logsx = logf(sx) + @as(f32, @floatFromInt(expmax)) * (ln2 * 2.0);
            }
            const fx = @sqrt(-2.0 * logsx / sx);
            self.xx2 = fx * v2;
            self.valid = true;
            res = fx * v1;
        }
        return res * self.sigma;
    }
};

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    num_iterations: i32 = 800,
    grain_radius_mean: f32 = 0.1,
    grain_radius_std: f32 = 0.0,
    sigma: f32 = 0.8,
    seed: i32 = 0,

    w: i32 = 0,
    h: i32 = 0,
    stride: i32 = 0,
    plane_bytes: usize = 0,

    dev: cu.Device = .{},
    module: cu.Module = .{},
    fn_grain: cu.Function = .{},
    d_lambda: cu.DeviceBuffer = .{},
    d_exp_lambda: cu.DeviceBuffer = .{},
    d_x_gaussian: cu.DeviceBuffer = .{},
    d_y_gaussian: cu.DeviceBuffer = .{},

    pool: pool_mod.Pool(Stream, Data) = .{},
};

const Stream = struct {
    stream: cu.Stream,
    d_src: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_src = try cu.DeviceBuffer.alloc(d.plane_bytes);
        errdefer self.d_src.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(d.plane_bytes);
        errdefer self.d_dst.deinit();
        self.stream = try cu.Stream.init();
    }

    pub fn deinit(self: *Stream) void {
        self.stream.deinit();
        self.d_dst.deinit();
        self.d_src.deinit();
    }
};

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, src: ZFrame, dst: ZFrameW, seed: i32) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain: async H2D references VS frame memory.
    errdefer s.stream.drain();

    const srcp = src.getReadSlice(0);
    std.debug.assert(srcp.len == d.plane_bytes);
    try s.stream.memcpyHtoD(s.d_src.ptr, srcp.ptr, srcp.len);

    const a_dst = s.d_dst.ptr;
    const a_src = s.d_src.ptr;
    const a_w: c_int = d.w;
    const a_h: c_int = d.h;
    const a_stride: c_int = d.stride;
    const a_iters: c_int = d.num_iterations;
    const a_seed: c_int = seed;
    const args = .{
        a_dst,          a_src,               a_w,                a_h,                a_stride,
        a_iters,        d.grain_radius_mean, d.grain_radius_std, d.sigma,            a_seed,
        d.d_lambda.ptr, d.d_exp_lambda.ptr,  d.d_x_gaussian.ptr, d.d_y_gaussian.ptr,
    };
    try s.stream.launch(d.fn_grain, .{
        .grid = .{ ceilDiv(@intCast(d.w), 32), ceilDiv(@intCast(d.h), 4), 1 },
        .block = .{ 32, 4, 1 },
    }, args);

    try s.stream.sync();

    const dstp = dst.getWriteSlice(0);
    try s.stream.memcpyDtoH(dstp.ptr, s.d_dst.ptr, dstp.len);
    try s.stream.sync();
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

        const seed_offset: i32 = src.getPropertiesRO().getValue(i32, "FGRAIN_SEED_OFFSET") orelse 0;

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, src, dst, d.seed +% seed_offset) catch |err| {
            zapi.setFilterError("FGrain: process frame failed.");
            std.log.err("vszipcu FGrain process frame failed: {t}", .{err});
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
    d.d_y_gaussian.deinit();
    d.d_x_gaussian.deinit();
    d.d_exp_lambda.deinit();
    d.d_lambda.deinit();
    d.module.deinit();
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize) CreateError!*Data {
    d.dev = try cu.initDevice(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    const defines = "#define SKIP_FAR_CELLS 1\n";

    d.module = try cu.compile(d.dev, .{
        .text = kernel_source,
        .defines = defines,
        .name = "fgrain.cu",
    }, nvrtc_opts);
    errdefer {
        d.module.deinit();
        d.module = .{};
    }
    d.fn_grain = try d.module.function("fgrain");

    errdefer {
        d.d_y_gaussian.deinit();
        d.d_x_gaussian.deinit();
        d.d_exp_lambda.deinit();
        d.d_lambda.deinit();
    }
    {
        var lambda: [256]f32 = undefined;
        var exp_lambda: [256]f32 = undefined;
        // grain_radius_std enters only lambda density; ag in DOUBLE then narrow.
        const ag: f32 = @floatCast(1.0 / @ceil(1.0 / @as(f64, d.grain_radius_mean)));
        const pi_f: f32 = 0x1.921fb6p+1;
        for (0..256) |i| {
            const frac = @as(f32, @floatFromInt(255 - @as(i32, @intCast(i)))) / 255.1;
            lambda[i] = -((ag * ag) / (pi_f *
                (d.grain_radius_mean * d.grain_radius_mean + d.grain_radius_std * d.grain_radius_std))) *
                logf(frac);
            exp_lambda[i] = expf(-lambda[i]);
        }
        d.d_lambda = try cu.DeviceBuffer.alloc(256 * 4);
        try cu.driver.memcpyHtoDSync(d.d_lambda.ptr, &lambda, 256 * 4);
        d.d_exp_lambda = try cu.DeviceBuffer.alloc(256 * 4);
        try cu.driver.memcpyHtoDSync(d.d_exp_lambda.ptr, &exp_lambda, 256 * 4);
    }
    {
        const n: usize = @intCast(d.num_iterations);
        const xg = allocator.alloc(f32, n) catch return error.OutOfMemory;
        defer allocator.free(xg);
        const yg = allocator.alloc(f32, n) catch return error.OutOfMemory;
        defer allocator.free(yg);
        // Kernel multiplies by sigma again (effective displacement stddev = sigma^2).
        var rng = Mt19937.init(@bitCast(d.seed));
        var dist: NormalF32 = .{ .sigma = d.sigma };
        for (0..n) |i| {
            xg[i] = dist.next(&rng);
            yg[i] = dist.next(&rng);
        }
        d.d_x_gaussian = try cu.DeviceBuffer.alloc(n * 4);
        try cu.driver.memcpyHtoDSync(d.d_x_gaussian.ptr, xg.ptr, n * 4);
        d.d_y_gaussian = try cu.DeviceBuffer.alloc(n * 4);
        try cu.driver.memcpyHtoDSync(d.d_y_gaussian.ptr, yg.ptr, n * 4);
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
    if (fmt.colorFamily != .Gray or fmt.sampleType != .Float or fmt.bitsPerSample != 32 or
        d.vi.width <= 0 or d.vi.height <= 0)
    {
        return map_out.setError("FGrain: only constant-format GRAYS input is supported (like vs-fgrain-cuda).");
    }

    const iters = map_in.getValue(i32, "num_iterations") orelse 800;
    if (iters < 1 or iters > (1 << 20)) return map_out.setError("FGrain: num_iterations must be 1..1048576.");
    d.num_iterations = iters;
    const mean = map_in.getValue(f64, "grain_radius_mean") orelse 0.1;
    if (!math.isFinite(mean) or mean <= 0.0 or mean > 100.0)
        return map_out.setError("FGrain: grain_radius_mean must be a finite value in (0, 100].");
    d.grain_radius_mean = @floatCast(mean);
    const gstd = map_in.getValue(f64, "grain_radius_std") orelse 0.0;
    if (!math.isFinite(gstd) or gstd < 0.0)
        return map_out.setError("FGrain: grain_radius_std must be a finite value >= 0.");
    d.grain_radius_std = @floatCast(gstd);
    const sigma = map_in.getValue(f64, "sigma") orelse 0.8;
    if (!math.isFinite(sigma) or sigma < 0.0)
        return map_out.setError("FGrain: sigma must be a finite value >= 0.");
    d.sigma = @floatCast(sigma);
    d.seed = map_in.getValue(i32, "seed") orelse 0;

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("FGrain: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("FGrain: num_streams must be 1..32.");
    };
    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 1;

    const strides = vsutil.strideFromVi(d.vi);
    d.w = d.vi.width;
    d.h = d.vi.height;
    d.stride = @intCast(strides[0]);
    d.plane_bytes = @as(usize, @intCast(d.stride)) * @as(usize, @intCast(d.h)) * 4;
    if (d.plane_bytes >= (1 << 31))
        return map_out.setError("FGrain: frame too large (plane exceeds 2^31 bytes).");
    if (d.h > 65535) // grid.y limit
        return map_out.setError("FGrain: frame too tall for the CUDA launch grid.");

    const data = initCuda(&d, device_id, num_streams) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "FGrain: invalid device ID.",
            error.Nvrtc => "FGrain: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "FGrain: could not locate NVRTC (put nvrtc64_130_0.dll next to the plugin, or: pip install nvidia-cuda-nvrtc).",
            error.OutOfDeviceMemory => "FGrain: out of device memory.",
            else => "FGrain: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu FGrain init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "FGrain", d.vi, getFrame, free, .Parallel, &dep, data);
}
