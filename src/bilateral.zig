const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const pool_mod = @import("pool.zig");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const math = std.math;

const CreateError = cu.CreateError || error{BadBlockDims};

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("kernel.cu");

// -use_fast_math (exp2f → ex2.approx); -modify-stack-limit=false.
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{ "-use_fast_math", "-std=c++17", "-modify-stack-limit=false" },
    .log_name = "Bilateral",
};

const FLT_EPSILON: f32 = 1.19209290e-7;

const SMEM_BUDGET: usize = 48 * 1024;

fn smemPitch(radius: usize, block_x: u32) usize {
    const raw = 2 * radius + block_x;
    if (block_x >= 32) return raw;
    const target: usize = block_x;
    return raw + (((target + 32) - (raw & 31)) & 31);
}

const Config = struct {
    w: i32,
    h: i32,
    radius: i32,
    sp: f32,
    sc: f32,
    use_sm: bool,
    smem: u32,
};

const Data = struct {
    node: ?*vs.Node,
    ref_node: ?*vs.Node,
    vi: *const vs.VideoInfo,

    process: [3]bool,
    has_ref: bool,

    bits: i32,
    half: bool,

    block_x: u32,
    block_y: u32,

    configs: [3]Config,
    n_cfg: usize,
    plane_cfg: [3]usize,

    d_pitch: usize,
    d_stride: usize,
    off_src_row: [3]usize,
    off_dst_row: [3]usize,
    buf_rows: usize,
    dst_rows: usize,

    zc_dst: bool,
    stage_dst_base: usize,
    pin_up: bool,

    dev: cu.Device,
    modules: [3]cu.Module,
    functions: [3]cu.Function,

    pool: pool_mod.Pool(Stream, Data),
};

const Stream = struct {
    d_src: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,
    h_buffer: cu.HostBuffer,
    stream: cu.Stream,
    cstream: cu.Stream,
    ev_up: [3]cu.Event,
    ev_k: [3]cu.Event,
    n_ev: usize,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_src = try cu.DeviceBuffer.alloc(d.buf_rows * d.d_pitch);
        errdefer self.d_src.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(if (d.zc_dst) 0 else d.dst_rows * d.d_pitch);
        errdefer self.d_dst.deinit();
        const src_span: usize = if (d.bits != 32 or d.pin_up) d.buf_rows else 0;
        const dst_span: usize = if (d.bits != 32 or d.zc_dst) d.dst_rows else 0;
        self.h_buffer = try cu.HostBuffer.alloc((src_span + dst_span) * d.d_pitch, .{});
        errdefer self.h_buffer.deinit();
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

    fn hbuf(self: *const Stream) [*]f32 {
        return @ptrCast(@alignCast(self.h_buffer.ptr));
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
        self.h_buffer.deinit();
        self.d_dst.deinit();
        self.d_src.deinit();
    }
};

// `{x}` = hexfloat; decimal f32 round-trip is a parity bug.
fn compileConfig(cfg: Config, d: *const Data) CreateError!struct { cu.Module, cu.Function } {
    const defines = std.fmt.allocPrint(allocator,
        \\#define width {d}
        \\#define height {d}
        \\#define stride {d}
        \\#define sigma_spatial_scaled ((float) {x})
        \\#define sigma_color_scaled ((float) {x})
        \\#define radius {d}
        \\#define use_shared_memory {s}
        \\#define BLOCK_X {d}
        \\#define BLOCK_Y {d}
        \\#define has_ref {s}
        \\
    , .{
        cfg.w,
        cfg.h,
        d.d_stride,
        cfg.sp,
        cfg.sc,
        cfg.radius,
        if (cfg.use_sm) "true" else "false",
        d.block_x,
        d.block_y,
        if (d.has_ref) "true" else "false",
    }) catch return error.OutOfMemory;
    defer allocator.free(defines);

    const module = try cu.compile(d.dev, .{
        .text = kernel_source,
        .defines = defines,
        .name = "bilateral.cu",
    }, nvrtc_opts);
    errdefer module.deinit();
    const function = try module.function("bilateral");

    return .{ module, function };
}

const inv255: f32 = @floatCast(@as(f64, 1.0) / 255.0);
const inv65535: f32 = @floatCast(@as(f64, 1.0) / 65535.0);

comptime {
    std.debug.assert(@as(u32, @bitCast(inv255)) == 0x3B808081);
    std.debug.assert(@as(u32, @bitCast(inv65535)) == 0x37800080);
}

fn packPlane(d: *const Data, hb: [*]f32, srcp: []const u8, src_stride: usize, w: usize, h: usize) void {
    var y: usize = 0;
    while (y < h) : (y += 1) {
        const dst_row = hb[y * d.d_stride ..][0..w];
        const src_row = srcp[y * src_stride ..];
        if (d.bits == 32) {
            const row: []const f32 = @as([*]const f32, @ptrCast(@alignCast(src_row.ptr)))[0..w];
            @memcpy(dst_row, row);
        } else if (d.half) {
            const row: []const f16 = @as([*]const f16, @ptrCast(@alignCast(src_row.ptr)))[0..w];
            for (dst_row, row) |*o, v| o.* = @floatCast(v);
        } else if (d.bits == 16) {
            const row: []const u16 = @as([*]const u16, @ptrCast(@alignCast(src_row.ptr)))[0..w];
            for (dst_row, row) |*o, v| o.* = @as(f32, @floatFromInt(v)) * inv65535;
        } else {
            for (dst_row, src_row[0..w]) |*o, v| o.* = @as(f32, @floatFromInt(v)) * inv255;
        }
    }
}

// clamp before @intFromFloat (out-of-range is UB).
fn unpackPlane(d: *const Data, dstp: []u8, hb: [*]const f32, dst_stride: usize, w: usize, h: usize) void {
    var y: usize = 0;
    while (y < h) : (y += 1) {
        const src_row = hb[y * d.d_stride ..][0..w];
        const dst_row = dstp[y * dst_stride ..];
        if (d.bits == 32) {
            const row: []f32 = @as([*]f32, @ptrCast(@alignCast(dst_row.ptr)))[0..w];
            @memcpy(row, src_row);
        } else if (d.half) {
            const row: []f16 = @as([*]f16, @ptrCast(@alignCast(dst_row.ptr)))[0..w];
            for (row, src_row) |*o, v| o.* = @floatCast(v);
        } else if (d.bits == 16) {
            const row: []u16 = @as([*]u16, @ptrCast(@alignCast(dst_row.ptr)))[0..w];
            for (row, src_row) |*o, v| o.* = @intFromFloat(math.clamp(@round(v * 65535.0), 0.0, 65535.0));
        } else {
            for (dst_row[0..w], src_row) |*o, v| o.* = @intFromFloat(math.clamp(@round(v * 255.0), 0.0, 255.0));
        }
    }
}

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, src: ZFrame, ref: ?ZFrame, dst: ZFrameW) !void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain: async copies reference VS memory / pinned buffer.
    errdefer {
        s.cstream.drain();
        s.stream.drain();
    }

    const num_planes: u32 = @intCast(d.vi.format.numPlanes);

    var p: u32 = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const cfg = d.configs[d.plane_cfg[p]];
        const w: usize = @intCast(cfg.w);
        const h: usize = @intCast(cfg.h);
        std.debug.assert(src.getWidthSigned(p) == cfg.w and src.getHeightSigned(p) == cfg.h);
        const dev_src = s.d_src.at(d.off_src_row[p] * d.d_pitch);

        if (d.bits == 32 and !d.pin_up) {
            try s.cstream.memcpy2D(.{
                .src = .{ .host = src.getReadSlice(p).ptr },
                .src_pitch = src.getStride(p),
                .dst = .{ .device = dev_src },
                .dst_pitch = d.d_pitch,
                .width_bytes = w * 4,
                .height = h,
            });
            if (ref) |r| {
                try s.cstream.memcpy2D(.{
                    .src = .{ .host = r.getReadSlice(p).ptr },
                    .src_pitch = r.getStride(p),
                    .dst = .{ .device = dev_src + h * d.d_pitch },
                    .dst_pitch = d.d_pitch,
                    .width_bytes = w * 4,
                    .height = h,
                });
            }
        } else {
            const hb = s.hbuf() + d.off_src_row[p] * d.d_stride;
            packPlane(d, hb, src.getReadSlice(p), src.getStride(p), w, h);
            if (ref) |r| packPlane(d, hb + h * d.d_stride, r.getReadSlice(p), r.getStride(p), w, h);
            const rows: usize = (1 + @as(usize, @intFromBool(d.has_ref))) * h;
            try s.cstream.memcpy2D(.{
                .src = .{ .host = hb },
                .src_pitch = d.d_pitch,
                .dst = .{ .device = dev_src },
                .dst_pitch = d.d_pitch,
                .width_bytes = w * 4,
                .height = rows,
            });
        }
        try s.cstream.record(s.ev_up[p]);
    }

    p = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const cfg = d.configs[d.plane_cfg[p]];
        const w: usize = @intCast(cfg.w);
        const h: usize = @intCast(cfg.h);
        try s.stream.waitEvent(s.ev_up[p]);

        const d_dst: cu.c.CUdeviceptr = if (d.zc_dst)
            s.h_buffer.devicePtr((d.stage_dst_base + d.off_dst_row[p]) * d.d_pitch)
        else
            s.d_dst.at(d.off_dst_row[p] * d.d_pitch);
        const d_src: cu.c.CUdeviceptr = s.d_src.at(d.off_src_row[p] * d.d_pitch);
        const grid_x: u32 = @intCast((w - 1) / d.block_x + 1);
        const grid_y: u32 = @intCast((h - 1) / d.block_y + 1);
        try s.stream.launch(d.functions[d.plane_cfg[p]], .{
            .grid = .{ grid_x, grid_y, 1 },
            .block = .{ d.block_x, d.block_y, 1 },
            .shared_mem = cfg.smem,
        }, .{ d_dst, d_src });
        try s.stream.record(s.ev_k[p]);
    }

    if (!d.zc_dst) {
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const cfg = d.configs[d.plane_cfg[p]];
            const w: usize = @intCast(cfg.w);
            const h: usize = @intCast(cfg.h);
            const dev_dst = s.d_dst.at(d.off_dst_row[p] * d.d_pitch);
            try s.cstream.waitEvent(s.ev_k[p]);
            if (d.bits == 32) {
                try s.cstream.memcpy2D(.{
                    .src = .{ .device = dev_dst },
                    .src_pitch = d.d_pitch,
                    .dst = .{ .host = dst.getWriteSlice(p).ptr },
                    .dst_pitch = dst.getStride(p),
                    .width_bytes = w * 4,
                    .height = h,
                });
            } else {
                const hb = s.hbuf() + (d.stage_dst_base + d.off_dst_row[p]) * d.d_stride;
                try s.cstream.memcpy2D(.{
                    .src = .{ .device = dev_dst },
                    .src_pitch = d.d_pitch,
                    .dst = .{ .host = hb },
                    .dst_pitch = d.d_pitch,
                    .width_bytes = w * 4,
                    .height = h,
                });
            }
        }
    }

    try s.cstream.sync();
    try s.stream.sync();

    if (d.bits != 32 or d.zc_dst) {
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const cfg = d.configs[d.plane_cfg[p]];
            const hb = s.hbuf() + (d.stage_dst_base + d.off_dst_row[p]) * d.d_stride;
            unpackPlane(d, dst.getWriteSlice(p), hb, dst.getStride(p), @intCast(cfg.w), @intCast(cfg.h));
        }
    }
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node);
        if (d.has_ref) zapi.requestFrameFilter(n, d.ref_node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        defer src.deinit();
        const ref: ?ZFrame = if (d.has_ref) zapi.initZFrame(d.ref_node, n) else null;
        defer if (ref) |r| r.deinit();

        const dst = src.newVideoFrame2(d.process);

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, src, ref, dst) catch |err| {
            zapi.setFilterError("vszipcu.Bilateral: processing frame failed.");
            std.log.err("vszipcu Bilateral process frame failed: {t}", .{err});
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
    for (d.modules[0..d.n_cfg]) |m| m.deinit();
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.ref_node);
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn sameFormat(a: *const vs.VideoInfo, b: *const vs.VideoInfo) bool {
    const fa = a.format;
    const fb = b.format;
    return fa.colorFamily == fb.colorFamily and fa.sampleType == fb.sampleType and
        fa.bitsPerSample == fb.bitsPerSample and fa.subSamplingW == fb.subSamplingW and
        fa.subSamplingH == fb.subSamplingH and a.width == b.width and a.height == b.height;
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize) CreateError!*Data {
    d.dev = try cu.Device.init(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    const max_threads = try d.dev.attribute(cu.c.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    if (d.block_x * d.block_y > @as(u32, @intCast(max_threads))) return error.BadBlockDims;

    const ssw: u5 = @intCast(d.vi.format.subSamplingW);
    const ssh: u5 = @intCast(d.vi.format.subSamplingH);
    const max_w: usize = @intCast(if (d.process[0]) d.vi.width else d.vi.width >> ssw);
    const max_h: usize = @intCast(if (d.process[0]) d.vi.height else d.vi.height >> ssh);
    {
        const pitch = try cu.driver.probePitch(max_w * @sizeOf(f32), max_h, 4);
        d.d_pitch = pitch;
        d.d_stride = pitch / @sizeOf(f32);
    }

    d.buf_rows = 0;
    d.dst_rows = 0;
    var max_plane_rows: usize = 0;
    {
        const np: usize = @intCast(d.vi.format.numPlanes);
        var p: usize = 0;
        while (p < np) : (p += 1) {
            if (!d.process[p]) continue;
            const cfg = d.configs[d.plane_cfg[p]];
            const h: usize = @intCast(cfg.h);
            const rows = (1 + @as(usize, @intFromBool(d.has_ref))) * h;
            d.off_src_row[p] = d.buf_rows;
            d.off_dst_row[p] = d.dst_rows;
            d.buf_rows += rows;
            d.dst_rows += h;
            max_plane_rows = @max(max_plane_rows, rows);
        }
    }

    d.stage_dst_base = if (d.bits != 32 or d.pin_up) d.buf_rows else 0;

    // Kernel uses 32-bit int indices within a plane region.
    if (max_plane_rows * d.d_stride >= (1 << 31) or max_plane_rows * d.d_pitch >= (1 << 32)) return error.FrameTooLarge;

    var loaded: usize = 0;
    errdefer for (d.modules[0..loaded]) |m| {
        m.deinit();
    };
    while (loaded < d.n_cfg) {
        d.modules[loaded], d.functions[loaded] = try compileConfig(d.configs[loaded], d);
        loaded += 1;
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
    var d: Data = undefined;

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;
    d.ref_node = map_in.getNode("ref");
    d.has_ref = d.ref_node != null;

    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.ref_node);
        zapi.freeNode(d.node);
    };

    const fmt = d.vi.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Float and (bits == 32 or bits == 16)) or
        (fmt.sampleType == .Integer and (bits == 8 or bits == 16));
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (fmt.colorFamily != .Gray and fmt.colorFamily != .YUV and fmt.colorFamily != .RGB))
    {
        return map_out.setError("vszipcu.Bilateral: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB, constant format.");
    }
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;

    if (d.has_ref) {
        const ref_vi = zapi.getVideoInfo(d.ref_node);
        if (!sameFormat(d.vi, ref_vi) or d.vi.numFrames != ref_vi.numFrames) {
            return map_out.setError("vszipcu.Bilateral: \"ref\" must be of the same format as \"clip\".");
        }
    }

    const ssw: u5 = @intCast(fmt.subSamplingW);
    const ssh: u5 = @intCast(fmt.subSamplingH);
    var sigma_spatial: [3]f32 = undefined;
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        if (map_in.getValue2(f32, "sigma_spatial", i)) |given| {
            if (given < 0) return map_out.setError("vszipcu.Bilateral: sigma_spatial must be non-negative.");
            sigma_spatial[i] = given;
        } else if (i == 0) {
            sigma_spatial[i] = 3.0;
        } else if (i == 1) {
            // DOUBLE then narrow (f32 path is ~1 ULP off).
            const prod = (@as(u32, 1) << ssh) * (@as(u32, 1) << ssw);
            const sub_factor = @sqrt(@as(f64, @floatFromInt(prod)));
            sigma_spatial[i] = @floatCast(@as(f64, sigma_spatial[0]) / sub_factor);
        } else {
            sigma_spatial[i] = sigma_spatial[i - 1];
        }
    }

    var sigma_color: [3]f32 = undefined;
    i = 0;
    while (i < 3) : (i += 1) {
        if (map_in.getValue2(f32, "sigma_color", i)) |given| {
            if (given < 0) return map_out.setError("vszipcu.Bilateral: sigma_color must be non-negative.");
            sigma_color[i] = given;
        } else if (i == 0) {
            sigma_color[i] = 0.02;
        } else {
            sigma_color[i] = sigma_color[i - 1];
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        d.process[i] = sigma_spatial[i] >= FLT_EPSILON and sigma_color[i] >= FLT_EPSILON;
    }

    const log2e: f32 = math.log2e;
    var sig_sp_scaled: [3]f32 = undefined;
    var sig_col_scaled: [3]f32 = undefined;
    i = 0;
    while (i < 3) : (i += 1) {
        sig_sp_scaled[i] = -0.5 / (sigma_spatial[i] * sigma_spatial[i]) * log2e;
        sig_col_scaled[i] = if (sigma_color[i] >= FLT_EPSILON)
            (-0.5 / (sigma_color[i] * sigma_color[i])) * log2e
        else
            0;
    }

    var radius: [3]i32 = undefined;
    i = 0;
    while (i < 3) : (i += 1) {
        if (map_in.getValue2(i32, "radius", i)) |given| {
            if (given <= 0 or given > 1_000_000) return map_out.setError("vszipcu.Bilateral: radius must be 1..1000000.");
            radius[i] = given;
        } else {
            // Clamp before @intFromFloat (UB otherwise).
            const r_f = @min(@round(sigma_spatial[i] * 3.0), 1_000_000.0);
            radius[i] = @max(1, @as(i32, @intFromFloat(r_f)));
        }
    }

    const device_id = map_in.getValue(i32, "device_id") orelse 0;

    const num_streams = map_in.getValue(i32, "num_streams") orelse 4;
    if (num_streams < 1 or num_streams > 32) {
        return map_out.setError("vszipcu.Bilateral: num_streams must be 1..32.");
    }

    const use_shared_memory = (map_in.getValue(i32, "use_shared_memory") orelse 1) != 0;

    var n_proc: u32 = 0;
    for (0..@as(usize, @intCast(fmt.numPlanes))) |pp| n_proc += @intFromBool(d.process[pp]);
    const subsampled = fmt.subSamplingW + fmt.subSamplingH > 0;
    d.zc_dst = d.bits != 32 or n_proc <= 1 or subsampled or num_streams <= 2;
    d.pin_up = d.bits == 32 and fmt.colorFamily == .Gray and n_proc == 1 and !d.has_ref and num_streams == 2;

    // smem border semantics flip at 48 KiB; block_x=16 matches bilateralgpu defaults.
    const block_x = map_in.getValue(i32, "block_x") orelse 32;
    const block_y = map_in.getValue(i32, "block_y") orelse 8;
    if (block_x < 1 or block_x > 1024 or block_y < 1 or block_y > 1024 or block_x * block_y > 1024) {
        return map_out.setError("vszipcu.Bilateral: block_x/block_y must be positive with block_x*block_y <= 1024.");
    }
    d.block_x = @intCast(block_x);
    d.block_y = @intCast(block_y);

    d.n_cfg = 0;
    {
        var p: usize = 0;
        while (p < @as(usize, @intCast(fmt.numPlanes))) : (p += 1) {
            if (!d.process[p]) continue;
            const rr: usize = @intCast(radius[p]);
            const pitch = smemPitch(rr, d.block_x);
            const smem_bytes: usize = (1 + @as(usize, @intFromBool(d.has_ref))) *
                (2 * rr + d.block_y) * pitch * @sizeOf(f32);
            const use_sm = use_shared_memory and smem_bytes < SMEM_BUDGET;
            const cfg: Config = .{
                .w = if (p == 0) d.vi.width else d.vi.width >> ssw,
                .h = if (p == 0) d.vi.height else d.vi.height >> ssh,
                .radius = radius[p],
                .sp = sig_sp_scaled[p],
                .sc = sig_col_scaled[p],
                .use_sm = use_sm,
                .smem = if (use_sm) @intCast(smem_bytes) else 0,
            };
            // Spatial sqdist is C++ int; overflow → silent NaN pixels.
            const dx: i64 = @min(@as(i64, radius[p]), @as(i64, cfg.w) - 1);
            const dy: i64 = @min(@as(i64, radius[p]), @as(i64, cfg.h) - 1);
            if (dx * dx + dy * dy > math.maxInt(i32)) {
                return map_out.setError("vszipcu.Bilateral: radius too large for this frame size.");
            }
            var ci: usize = 0;
            while (ci < d.n_cfg) : (ci += 1) {
                if (std.meta.eql(cfg, d.configs[ci])) break;
            }
            if (ci == d.n_cfg) {
                d.configs[ci] = cfg;
                d.n_cfg += 1;
            }
            d.plane_cfg[p] = ci;
        }
    }

    const data = initCuda(&d, device_id, @intCast(num_streams)) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "vszipcu.Bilateral: invalid device ID.",
            error.BadBlockDims => "vszipcu.Bilateral: block_x*block_y exceeds the device's max threads per block.",
            error.FrameTooLarge => "vszipcu.Bilateral: frame too large (a plane exceeds 2^31 samples).",
            error.Nvrtc => "vszipcu.Bilateral: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "vszipcu.Bilateral: could not locate NVRTC (wheel should ship nvrtc64_130_0.dll next to the plugin).",
            error.OutOfDeviceMemory => "vszipcu.Bilateral: out of device memory.",
            else => "vszipcu.Bilateral: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu Bilateral init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
        .{ .source = d.ref_node, .requestPattern = .StrictSpatial },
    };
    const deps: []const vs.FilterDependency = if (d.has_ref) dep[0..2] else dep[0..1];

    zapi.createVideoFilter(out, "Bilateral", d.vi, getFrame, free, .Parallel, deps, data);
}
