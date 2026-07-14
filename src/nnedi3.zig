const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const pool_mod = @import("pool.zig");
const vsutil = @import("vsutil.zig");

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const CreateError = cu.CreateError;
const ceilDiv = cu.ceilDiv;

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("nnedi3.cu");
const weights_bin = @embedFile("nnedi3_weights.bin");

const MARGIN_H = 24;
const MARGIN_V = 3;

const NNEDI3_XDIM = [7]u32{ 8, 16, 32, 48, 8, 16, 32 };
const NNEDI3_YDIM = [7]u32{ 6, 6, 6, 6, 4, 4, 4 };
const NNEDI3_NNS = [5]u32{ 16, 32, 64, 128, 256 };

// -fmad=false (prescreener precise; predictor only at explicit __fmaf_rn).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{ "-fmad=false", "-std=c++17" },
    .log_name = "NNEDI3",
};

const PrescreenerOld = struct {
    kernel_l0: [4][48]f32,
    bias_l0: [4]f32,
    kernel_l1: [4][4]f32,
    bias_l1: [4]f32,
    kernel_l2: [4][8]f32,
    bias_l2: [4]f32,
};

const PrescreenerNew = struct {
    kernel_l0: [4][64]f32,
    bias_l0: [4]f32,
    kernel_l1: [4][4]f32,
    bias_l1: [4]f32,
};

const Model = struct {
    xdim: u32,
    ydim: u32,
    nns: u32,
    softmax_q1: []f32,
    elliott_q1: []f32,
    softmax_bias_q1: []f32,
    elliott_bias_q1: []f32,
    softmax_q2: []f32,
    elliott_q2: []f32,
    softmax_bias_q2: []f32,
    elliott_bias_q2: []f32,

    fn deinit(self: *Model) void {
        allocator.free(self.softmax_q1);
        allocator.free(self.elliott_q1);
        allocator.free(self.softmax_bias_q1);
        allocator.free(self.elliott_bias_q1);
        allocator.free(self.softmax_q2);
        allocator.free(self.elliott_q2);
        allocator.free(self.softmax_bias_q2);
        allocator.free(self.elliott_bias_q2);
    }
};

fn vecMean(buf: []const f32) f64 {
    var acc: f64 = 0.0;
    for (buf) |v| acc += v;
    return acc / @as(f64, @floatFromInt(buf.len));
}

fn subtractMeanPs(comptime T: type, ps: *T, pixel_half: f64) void {
    for (0..4) |n| {
        const m = vecMean(&ps.kernel_l0[n]);
        for (&ps.kernel_l0[n]) |*x| {
            x.* = @floatCast((x.* - m) / pixel_half);
        }
    }
}

fn subtractMeanModel(m: *Model) !void {
    const fs: usize = m.xdim * m.ydim;
    const nns: usize = m.nns;

    const softmax_means = try allocator.alloc(f64, nns);
    defer allocator.free(softmax_means);
    const elliott_means = try allocator.alloc(f64, nns);
    defer allocator.free(elliott_means);
    const mean_filter = try allocator.alloc(f64, fs);
    defer allocator.free(mean_filter);

    const one_pass = struct {
        fn run(softmax: []f32, ell: []f32, softmax_bias: []f32, fs_: usize, nns_: usize, sm_means: []f64, el_means: []f64, mf: []f64) void {
            @memset(mf, 0.0);
            for (0..nns_) |nn| {
                sm_means[nn] = vecMean(softmax[nn * fs_ ..][0..fs_]);
                el_means[nn] = vecMean(ell[nn * fs_ ..][0..fs_]);
                for (0..fs_) |k| {
                    mf[k] += softmax[nn * fs_ + k] - sm_means[nn];
                }
            }
            for (0..fs_) |k| mf[k] /= @as(f64, @floatFromInt(nns_));
            const mean_bias = vecMean(softmax_bias);
            for (0..nns_) |nn| {
                for (0..fs_) |k| {
                    softmax[nn * fs_ + k] -= @floatCast(sm_means[nn] + mf[k]);
                    ell[nn * fs_ + k] -= @as(f32, @floatCast(el_means[nn]));
                }
                softmax_bias[nn] -= @as(f32, @floatCast(mean_bias));
            }
        }
    }.run;

    one_pass(m.softmax_q1, m.elliott_q1, m.softmax_bias_q1, fs, nns, softmax_means, elliott_means, mean_filter);
    one_pass(m.softmax_q2, m.elliott_q2, m.softmax_bias_q2, fs, nns, softmax_means, elliott_means, mean_filter);
}

fn readWeights(nsize: usize, nns_sel: usize, etype: usize, ps_old: *PrescreenerOld, ps_new: *[3]PrescreenerNew, model: *Model) !void {
    const data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, weights_bin));
    var ptr: usize = 0;
    const read = struct {
        fn go(d: []const f32, p: *usize, dst: []f32) void {
            @memcpy(dst, d[p.*..][0..dst.len]);
            p.* += dst.len;
        }
    }.go;

    for (0..4) |n| read(data, &ptr, &ps_old.kernel_l0[n]);
    read(data, &ptr, &ps_old.bias_l0);
    for (0..4) |n| read(data, &ptr, &ps_old.kernel_l1[n]);
    read(data, &ptr, &ps_old.bias_l1);
    for (0..4) |n| read(data, &ptr, &ps_old.kernel_l2[n]);
    read(data, &ptr, &ps_old.bias_l2);

    for (0..3) |i| {
        var l0s: [4 * 64]f32 = undefined;
        var l1s: [4 * 4]f32 = undefined;
        read(data, &ptr, &l0s);
        read(data, &ptr, &ps_new[i].bias_l0);
        read(data, &ptr, &l1s);
        read(data, &ptr, &ps_new[i].bias_l1);
        for (0..4) |n| {
            for (0..64) |k| {
                ps_new[i].kernel_l0[n][k] = l0s[(k / 8) * 32 + n * 8 + k % 8];
            }
            for (0..4) |k| {
                ps_new[i].kernel_l1[n][k] = l1s[k * 4 + n];
            }
        }
    }

    for (0..2) |m_| {
        for (0..5) |i| {
            for (0..7) |j| {
                const nns: usize = NNEDI3_NNS[i];
                const fs: usize = NNEDI3_XDIM[j] * NNEDI3_YDIM[j];
                if (m_ == etype and i == nns_sel and j == nsize) {
                    model.xdim = NNEDI3_XDIM[j];
                    model.ydim = NNEDI3_YDIM[j];
                    model.nns = @intCast(nns);
                    model.softmax_q1 = try allocator.alloc(f32, nns * fs);
                    model.elliott_q1 = try allocator.alloc(f32, nns * fs);
                    model.softmax_bias_q1 = try allocator.alloc(f32, nns);
                    model.elliott_bias_q1 = try allocator.alloc(f32, nns);
                    model.softmax_q2 = try allocator.alloc(f32, nns * fs);
                    model.elliott_q2 = try allocator.alloc(f32, nns * fs);
                    model.softmax_bias_q2 = try allocator.alloc(f32, nns);
                    model.elliott_bias_q2 = try allocator.alloc(f32, nns);
                    read(data, &ptr, model.softmax_q1);
                    read(data, &ptr, model.elliott_q1);
                    read(data, &ptr, model.softmax_bias_q1);
                    read(data, &ptr, model.elliott_bias_q1);
                    read(data, &ptr, model.softmax_q2);
                    read(data, &ptr, model.elliott_q2);
                    read(data, &ptr, model.softmax_bias_q2);
                    read(data, &ptr, model.elliott_bias_q2);
                } else {
                    ptr += 4 * nns * fs + 4 * nns;
                }
            }
        }
    }
    std.debug.assert(ptr == data.len);
}

fn floatToHalf(f: f32) u16 {
    return @bitCast(@as(f16, @floatCast(f)));
}

const Geom = struct {
    process: bool = false,
    w: i32 = 0,
    h: i32 = 0,
    rows: i32 = 0,
    pad_stride: i32 = 0,
    pad_h: i32 = 0,
    field_off: usize = 0,
    pad_off: usize = 0,
    dst_off: usize = 0,
    list_off: usize = 0,
    cnt_off: usize = 0,
};

const Data = struct {
    node: ?*vs.Node = null,
    vi_in: vs.VideoInfo = undefined,
    vi_out: vs.VideoInfo = undefined,

    field: i32 = 0,
    dh: bool = false,
    qual: i32 = 1,
    pscrn: i32 = 2,
    peak: i32 = 0,
    bytes: u32 = 4,
    is_float: bool = true,
    use_mma: bool = false,
    nns: u32 = 32,
    fs: u32 = 128,
    xdim_: u32 = 32,
    ydim_: u32 = 4,

    geom: [3]Geom = .{ .{}, .{}, .{} },
    field_bytes: usize = 0,
    pad_bytes: usize = 0,
    dst_bytes: usize = 0,
    list_bytes: usize = 0,

    dev: cu.Device = .{},
    module: cu.Module = .{},
    fn_pad: cu.Function = .{},
    fn_prescreen: cu.Function = .{},
    fn_predict: cu.Function = .{},
    fn_predict_mma: cu.Function = .{},
    d_psw: cu.DeviceBuffer = .{},
    d_pdw: cu.DeviceBuffer = .{},
    d_pdb: cu.DeviceBuffer = .{},

    pool: pool_mod.Pool(Stream, Data) = .{},
    spool: pool_mod.Pool(NStage, Data) = .{},
};

const NStage = struct {
    h: cu.HostBuffer,

    pub fn init(self: *NStage, d: *Data) !void {
        self.h = try cu.HostBuffer.alloc(d.field_bytes + d.dst_bytes, .{});
    }

    pub fn deinit(self: *NStage) void {
        self.h.deinit();
    }
};

const Stream = struct {
    stream: cu.Stream,
    stream2: cu.Stream,
    ev_up: [3]cu.Event,
    n_ev: usize,
    d_field: cu.DeviceBuffer,
    d_pad: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,
    d_list: cu.DeviceBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_field = try cu.DeviceBuffer.alloc(d.field_bytes);
        errdefer self.d_field.deinit();
        self.d_pad = try cu.DeviceBuffer.alloc(d.pad_bytes);
        errdefer self.d_pad.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(d.dst_bytes);
        errdefer self.d_dst.deinit();
        self.d_list = try cu.DeviceBuffer.alloc(if (d.pscrn > 0) d.list_bytes else 0);
        errdefer self.d_list.deinit();
        self.n_ev = 0;
        errdefer while (self.n_ev > 0) {
            self.n_ev -= 1;
            self.ev_up[self.n_ev].deinit();
        };
        for (0..3) |i| {
            self.ev_up[i] = try cu.Event.init();
            self.n_ev = i + 1;
        }
        self.stream = try cu.Stream.init();
        errdefer self.stream.deinit();
        self.stream2 = try cu.Stream.init();
    }

    pub fn deinit(self: *Stream) void {
        self.stream2.deinit();
        self.stream.deinit();
        while (self.n_ev > 0) {
            self.n_ev -= 1;
            self.ev_up[self.n_ev].deinit();
        }
        self.d_list.deinit();
        self.d_dst.deinit();
        self.d_pad.deinit();
        self.d_field.deinit();
    }
};

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, stg: *NStage, parity: i32) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    errdefer {
        s.stream2.drain();
        s.stream.drain();
    }

    const fp: i32 = 1 - parity;
    const num_planes: u32 = @intCast(d.vi_in.format.numPlanes);

    var p: u32 = 0;
    while (p < num_planes) : (p += 1) {
        const g = &d.geom[p];
        if (!g.process) continue;
        const fbytes = @as(usize, @intCast(g.w * g.rows)) * d.bytes;
        try s.stream2.memcpyHtoD(s.d_field.at(g.field_off), stg.h.ptr + g.field_off, fbytes);
        try s.stream2.record(s.ev_up[p]);
    }

    p = 0;
    while (p < num_planes) : (p += 1) {
        const g = &d.geom[p];
        if (!g.process) continue;
        const ks = if (p == 0) s.stream else s.stream2;
        if (p == 0) try ks.waitEvent(s.ev_up[p]);

        const a_pad = s.d_pad.at(g.pad_off);
        const a_field = s.d_field.at(g.field_off);
        const a_w: c_int = g.w;
        const a_rows: c_int = g.rows;
        const a_fstride: c_int = g.w;
        const a_pstride: c_int = g.pad_stride;
        const a_fp: c_int = fp;
        const pad_w: u32 = @intCast(g.w + MARGIN_H * 2);
        const pad_h: u32 = @intCast(g.pad_h);
        try ks.launch(d.fn_pad, .{
            .grid = .{ ceilDiv(pad_w, 32), ceilDiv(pad_h, 8), 1 },
            .block = .{ 32, 8, 1 },
        }, .{ a_pad, a_field, a_w, a_rows, a_fstride, a_pstride, a_fp });

        const a_dst: cu.c.CUdeviceptr = s.d_dst.at(g.dst_off);
        const a_list = if (d.pscrn > 0) s.d_list.at(g.list_off) else 0;
        const a_cnt = if (d.pscrn > 0) s.d_list.at(g.cnt_off) else 0;
        const npix: u32 = @intCast(g.w * g.rows);

        if (d.pscrn > 0) {
            try ks.memsetD32(s.d_list.at(g.cnt_off), 0, 1);
            const px_per_thread: u32 = if (d.pscrn == 1) 1 else 4;
            const threads: u32 = @as(u32, @intCast(g.rows)) * ceilDiv(@intCast(g.w), px_per_thread);
            try ks.launch(d.fn_prescreen, .{
                .grid = .{ ceilDiv(threads, 128), 1, 1 },
                .block = .{ 128, 1, 1 },
            }, .{ a_dst, a_pad, d.d_psw.ptr, a_list, a_cnt, a_w, a_rows, a_pstride });
        }

        if (d.use_mma) {
            try ks.launch(d.fn_predict_mma, .{
                .grid = .{ ceilDiv(npix, 64), 1, 1 },
                .block = .{ 128, 1, 1 },
            }, .{ a_dst, a_pad, d.d_pdw.ptr, d.d_pdb.ptr, a_list, a_cnt, a_w, a_rows, a_pstride });
        } else {
            try ks.launch(d.fn_predict, .{
                .grid = .{ ceilDiv(npix, 16), 1, 1 },
                .block = .{ 128, 1, 1 },
            }, .{ a_dst, a_pad, d.d_pdw.ptr, d.d_pdb.ptr, a_list, a_cnt, a_w, a_rows, a_pstride });
        }
    }

    try s.stream.sync();
    try s.stream2.sync();

    p = 0;
    while (p < num_planes) : (p += 1) {
        const g = &d.geom[p];
        if (!g.process) continue;
        const bytes = @as(usize, @intCast(g.w)) * @as(usize, @intCast(g.rows)) * d.bytes;
        try s.stream2.memcpyDtoH(stg.h.ptr + d.field_bytes + g.dst_off, s.d_dst.at(g.dst_off), bytes);
    }
    try s.stream2.sync();
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    const sn: c_int = if (d.field > 1) @divTrunc(n, 2) else n;

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(sn, d.node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, sn);
        defer src.deinit();

        const dst = if (d.dh)
            src.newVideoFrame3(.{ .height = d.vi_out.height })
        else
            src.newVideoFrame2(.{ d.geom[0].process, d.geom[1].process, d.geom[2].process });

        const props = src.getPropertiesRO();
        const default_parity: i32 = if (d.field == 0 or d.field == 2) 1 else 0;
        var parity: i32 = undefined;
        if (d.dh) {
            parity = props.getValue(i32, "_Field") orelse default_parity;
        } else if (d.field > 1) {
            const fb = props.getValue(i32, "_FieldBased") orelse -1;
            parity = if (fb == 1) 1 else if (fb == 2) 0 else default_parity;
            if (@mod(n, 2) == 1) parity = 1 - parity;
        } else {
            parity = if (d.field == 0) 1 else 0;
        }
        parity = if (parity != 0) 1 else 0;

        const num_planes: u32 = @intCast(d.vi_in.format.numPlanes);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            const g = &d.geom[p];
            if (!g.process) continue;
            const srcp = src.getReadSlice(p);
            const dstp = dst.getWriteSlice(p);
            const src_stride = srcp.len / @as(usize, @intCast(if (d.dh) g.rows else g.h));
            const dst_stride = dstp.len / @as(usize, @intCast(g.h));
            const row_bytes = @as(usize, @intCast(g.w)) * d.bytes;
            const fstart: usize = if (d.dh) 0 else @as(usize, @intCast(parity)) * src_stride;
            const fpitch: usize = src_stride * (if (d.dh) @as(usize, 1) else 2);
            var r: usize = 0;
            while (r < @as(usize, @intCast(g.rows))) : (r += 1) {
                @memcpy(
                    dstp[(@as(usize, @intCast(parity)) + 2 * r) * dst_stride ..][0..row_bytes],
                    srcp[fstart + r * fpitch ..][0..row_bytes],
                );
            }
        }

        const stg = d.spool.acquire();
        defer d.spool.release(stg);
        p = 0;
        while (p < num_planes) : (p += 1) {
            const g = &d.geom[p];
            if (!g.process) continue;
            const srcp = src.getReadSlice(p);
            const src_h: usize = @intCast(if (d.dh) g.rows else g.h);
            const src_stride = srcp.len / src_h;
            const row_bytes = @as(usize, @intCast(g.w)) * d.bytes;
            const fstart: usize = if (d.dh) 0 else @as(usize, @intCast(parity)) * src_stride;
            const fpitch: usize = src_stride * (if (d.dh) @as(usize, 1) else 2);
            var r: usize = 0;
            while (r < @as(usize, @intCast(g.rows))) : (r += 1) {
                @memcpy(
                    (stg.h.ptr + g.field_off + r * row_bytes)[0..row_bytes],
                    srcp[fstart + r * fpitch ..][0..row_bytes],
                );
            }
        }

        {
            const s = d.pool.acquire();
            defer d.pool.release(s);

            process(d, s, stg, parity) catch |err| {
                zapi.setFilterError("NNEDI3: process frame failed.");
                std.log.err("vszipcu NNEDI3 process frame failed: {t}", .{err});
                dst.deinit();
                return null;
            };
        }

        p = 0;
        while (p < num_planes) : (p += 1) {
            const g = &d.geom[p];
            if (!g.process) continue;
            const dstp = dst.getWriteSlice(p);
            const dst_stride = dstp.len / @as(usize, @intCast(g.h));
            const row_bytes = @as(usize, @intCast(g.w)) * d.bytes;
            const rb = stg.h.ptr + d.field_bytes + g.dst_off;
            const off0: usize = @intCast(1 - parity);
            var r: usize = 0;
            while (r < @as(usize, @intCast(g.rows))) : (r += 1) {
                @memcpy(dstp[(off0 + 2 * r) * dst_stride ..][0..row_bytes], rb[r * row_bytes ..][0..row_bytes]);
            }
        }

        const wprops = dst.getPropertiesRW();
        wprops.setInt("_FieldBased", 0, .Replace);
        wprops.deleteKey("_Field");
        if (d.field > 1) {
            const dur_num = wprops.getValue(i64, "_DurationNum");
            const dur_den = wprops.getValue(i64, "_DurationDen");
            if (dur_num != null and dur_den != null) {
                var num_v = dur_num.?;
                var den_v = dur_den.?;
                vsh.muldivRational(&num_v, &den_v, 1, 2);
                wprops.setInt("_DurationNum", num_v, .Replace);
                wprops.setInt("_DurationDen", den_v, .Replace);
            }
        }

        return dst.frame;
    }

    return null;
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.dev.push() catch {};
    d.pool.deinit();
    d.spool.deinit();
    d.d_pdb.deinit();
    d.d_pdw.deinit();
    d.d_psw.deinit();
    d.module.deinit();
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize, ps_blob: []const f32, pdw_blob: []const u8, pdb_blob: []const f32) CreateError!*Data {
    d.dev = try cu.Device.init(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    const cc = try d.dev.computeCapability();
    if (d.use_mma and cc < 80) d.use_mma = false;

    const pixel_type: u8 = if (d.is_float and d.bytes == 4) 3 else if (d.is_float) 2 else if (d.bytes == 2) 1 else 0;

    const lower: [4]u8 = if (d.use_mma) .{ 1, 1, 1, 1 } else .{ 0, 0, 0, 0 };

    var defines_buf: [768]u8 = undefined;
    const defines = std.fmt.bufPrint(&defines_buf,
        \\#define PIXEL_TYPE {d}
        \\#define PSCRN {d}
        \\#define XDIM {d}
        \\#define YDIM {d}
        \\#define NNS {d}
        \\#define QUAL {d}
        \\#define PEAK {d}
        \\#define MMA {d}
        \\#define EXP_APPROX {d}
        \\#define DIV_APPROX {d}
        \\#define SQRT_APPROX {d}
        \\#define RCP_APPROX {d}
        \\
    , .{ pixel_type, d.pscrn, d.xdim_, d.ydim_, d.nns, d.qual, d.peak, @intFromBool(d.use_mma), lower[0], lower[1], lower[2], lower[3] }) catch unreachable;

    d.module = try cu.compile(d.dev, .{
        .text = kernel_source,
        .defines = defines,
        .name = "nnedi3.cu",
    }, nvrtc_opts);
    errdefer {
        d.module.deinit();
        d.module = .{};
    }
    d.fn_pad = try d.module.function("pad");
    if (d.pscrn > 0) d.fn_prescreen = try d.module.function("prescreen");
    if (d.use_mma) {
        d.fn_predict_mma = try d.module.function("predict_mma");
    } else {
        d.fn_predict = try d.module.function("predict");
    }

    errdefer {
        d.d_pdb.deinit();
        d.d_pdw.deinit();
        d.d_psw.deinit();
    }
    d.d_psw = try cu.DeviceBuffer.alloc(ps_blob.len * 4);
    try cu.driver.memcpyHtoDSync(d.d_psw.ptr, ps_blob.ptr, ps_blob.len * 4);
    d.d_pdw = try cu.DeviceBuffer.alloc(pdw_blob.len);
    try cu.driver.memcpyHtoDSync(d.d_pdw.ptr, pdw_blob.ptr, pdw_blob.len);
    d.d_pdb = try cu.DeviceBuffer.alloc(pdb_blob.len * 4);
    try cu.driver.memcpyHtoDSync(d.d_pdb.ptr, pdb_blob.ptr, pdb_blob.len * 4);

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
    const n_stage = num_streams + 2;
    data.spool.prime(data, n_stage) catch {
        data.pool.deinit();
        data.spool.deinit();
        return error.OutOfMemory;
    };
    data.spool.prewarm(n_stage) catch |err| {
        data.pool.deinit();
        data.spool.deinit();
        return err;
    };
    return data;
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi_ptr = map_in.getNodeVi("clip").?;
    d.vi_in = vi_ptr.*;
    d.vi_out = vi_ptr.*;

    var keep = false;
    defer if (!keep) zapi.freeNode(d.node);

    const fmt = d.vi_in.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Integer and bits >= 8 and bits <= 16) or
        (fmt.sampleType == .Float and (bits == 16 or bits == 32));
    if (!depth_ok or d.vi_in.width <= 0 or d.vi_in.height <= 0) {
        return map_out.setError("NNEDI3: only constant format 8-16 bit integer and 16/32 bit float input supported.");
    }
    d.bytes = @intCast(fmt.bytesPerSample);
    d.is_float = fmt.sampleType == .Float;
    // PEAK only for integer (bits=32 float shift is UB).
    d.peak = if (d.is_float) 0 else (@as(i32, 1) << @intCast(bits)) - 1;

    d.field = map_in.getValue(i32, "field") orelse 0;
    if (d.field < 0 or d.field > 3) return map_out.setError("NNEDI3: field must be 0, 1, 2, or 3.");
    d.dh = (map_in.getValue(i32, "dh") orelse 0) != 0;
    if (d.dh and d.field > 1) return map_out.setError("NNEDI3: field must be 0 or 1 when dh=True.");

    var proc = [3]bool{ true, true, true };
    if (map_in.numElements("planes")) |ne| {
        proc = .{ false, false, false };
        var e: u32 = 0;
        while (e < ne) : (e += 1) {
            const idx = map_in.getValue2(i32, "planes", e).?;
            if (idx < 0 or idx >= fmt.numPlanes) return map_out.setError("NNEDI3: plane index out of range.");
            if (proc[@intCast(idx)]) return map_out.setError("NNEDI3: plane specified twice.");
            proc[@intCast(idx)] = true;
        }
    }

    const nsize = map_in.getValue(i32, "nsize") orelse 6;
    const nns_sel = map_in.getValue(i32, "nns") orelse 1;
    d.qual = map_in.getValue(i32, "qual") orelse 1;
    const etype = map_in.getValue(i32, "etype") orelse 0;
    d.pscrn = map_in.getValue(i32, "pscrn") orelse 2;
    if (nsize < 0 or nsize > 6) return map_out.setError("NNEDI3: nsize must be between 0 and 6 (inclusive).");
    if (nns_sel < 0 or nns_sel > 4) return map_out.setError("NNEDI3: nns must be between 0 and 4 (inclusive).");
    if (d.qual < 1 or d.qual > 2) return map_out.setError("NNEDI3: qual must be 1 or 2.");
    if (etype < 0 or etype > 1) return map_out.setError("NNEDI3: etype must be 0 or 1.");
    if (d.pscrn < 0 or d.pscrn > 4) return map_out.setError("NNEDI3: pscrn must be between 0 and 4 (inclusive).");

    if (!d.dh) {
        var pi: usize = 0;
        while (pi < @as(usize, @intCast(fmt.numPlanes))) : (pi += 1) {
            const ph = d.vi_in.height >> @intCast(if (pi > 0) fmt.subSamplingH else 0);
            if (proc[pi] and (ph & 1) != 0)
                return map_out.setError("NNEDI3: plane's height must be mod 2 when dh=False.");
        }
    }

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("NNEDI3: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("NNEDI3: num_streams must be 1..32.");
    };
    const yuv420_small_net = fmt.numPlanes == 3 and fmt.subSamplingW == 1 and
        fmt.subSamplingH == 1 and nns_sel <= 2;
    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else if (yuv420_small_net) 4 else 2;

    if (d.field > 1) {
        if (d.vi_out.numFrames > std.math.maxInt(i32) / 2)
            return map_out.setError("NNEDI3: resulting clip is too long.");
        d.vi_out.numFrames *= 2;
        var fn_ = d.vi_out.fpsNum;
        var fd_ = d.vi_out.fpsDen;
        vsh.muldivRational(&fn_, &fd_, 2, 1);
        d.vi_out.fpsNum = fn_;
        d.vi_out.fpsDen = fd_;
    }
    if (d.dh) d.vi_out.height *= 2;

    var ps_old: PrescreenerOld = undefined;
    var ps_new: [3]PrescreenerNew = undefined;
    var model: Model = undefined;
    readWeights(@intCast(nsize), @intCast(nns_sel), @intCast(etype), &ps_old, &ps_new, &model) catch
        return map_out.setError("NNEDI3: out of memory.");
    defer model.deinit();

    const pixel_half: f64 = if (d.is_float) 0.5 else @as(f64, @floatFromInt(d.peak)) / 2.0;
    if (d.pscrn == 1) {
        subtractMeanPs(PrescreenerOld, &ps_old, pixel_half);
    } else if (d.pscrn >= 2) {
        subtractMeanPs(PrescreenerNew, &ps_new[@intCast(d.pscrn - 2)], pixel_half);
    }
    subtractMeanModel(&model) catch return map_out.setError("NNEDI3: out of memory.");

    d.nns = model.nns;
    d.fs = model.xdim * model.ydim;
    d.xdim_ = model.xdim;
    d.ydim_ = model.ydim;

    d.use_mma = true;

    var ps_blob: []f32 = &.{};
    defer if (ps_blob.len > 0) allocator.free(ps_blob);
    if (d.pscrn == 1) {
        ps_blob = allocator.alloc(f32, @sizeOf(PrescreenerOld) / 4) catch return map_out.setError("NNEDI3: out of memory.");
        var w: usize = 0;
        for (0..48) |k| {
            for (0..4) |n| {
                ps_blob[w] = ps_old.kernel_l0[n][k];
                w += 1;
            }
        }
        for (ps_old.bias_l0) |v| {
            ps_blob[w] = v;
            w += 1;
        }
        for (0..4) |n| for (ps_old.kernel_l1[n]) |v| {
            ps_blob[w] = v;
            w += 1;
        };
        for (ps_old.bias_l1) |v| {
            ps_blob[w] = v;
            w += 1;
        }
        for (0..4) |n| for (ps_old.kernel_l2[n]) |v| {
            ps_blob[w] = v;
            w += 1;
        };
        for (ps_old.bias_l2) |v| {
            ps_blob[w] = v;
            w += 1;
        }
    } else if (d.pscrn >= 2) {
        const ps = &ps_new[@intCast(d.pscrn - 2)];
        ps_blob = allocator.alloc(f32, @sizeOf(PrescreenerNew) / 4) catch return map_out.setError("NNEDI3: out of memory.");
        var w: usize = 0;
        for (0..64) |k| {
            for (0..4) |n| {
                ps_blob[w] = ps.kernel_l0[n][k];
                w += 1;
            }
        }
        for (ps.bias_l0) |v| {
            ps_blob[w] = v;
            w += 1;
        }
        for (0..4) |n| for (ps.kernel_l1[n]) |v| {
            ps_blob[w] = v;
            w += 1;
        };
        for (ps.bias_l1) |v| {
            ps_blob[w] = v;
            w += 1;
        }
    } else {
        ps_blob = allocator.alloc(f32, 1) catch return map_out.setError("NNEDI3: out of memory.");
        ps_blob[0] = 0;
    }

    const fs: usize = d.fs;
    const N: usize = d.nns;
    const numQ: usize = @intCast(d.qual);

    const pdb = allocator.alloc(f32, numQ * 4 * N) catch return map_out.setError("NNEDI3: out of memory.");
    defer allocator.free(pdb);
    for (0..numQ) |q| {
        const sm = if (q != 0) model.softmax_q2 else model.softmax_q1;
        const el = if (q != 0) model.elliott_q2 else model.elliott_q1;
        const smB = if (q != 0) model.softmax_bias_q2 else model.softmax_bias_q1;
        const elB = if (q != 0) model.elliott_bias_q2 else model.elliott_bias_q1;
        for (0..N) |pn| {
            pdb[(q * 2 * N + pn) * 2 + 0] = smB[pn];
            pdb[(q * 2 * N + pn) * 2 + 1] = elB[pn];
            var sm_sum: f64 = 0.0;
            var el_sum: f64 = 0.0;
            for (0..fs) |k| {
                sm_sum += sm[pn * fs + k];
                el_sum += el[pn * fs + k];
            }
            pdb[(q * 2 * N + N + pn) * 2 + 0] = @floatCast(sm_sum);
            pdb[(q * 2 * N + N + pn) * 2 + 1] = @floatCast(el_sum);
        }
    }

    var pdw_blob: []u8 = &.{};
    defer if (pdw_blob.len > 0) allocator.free(pdw_blob);
    if (d.use_mma) {
        const KT = fs / 16;
        const NT = 2 * N / 8;
        const words = numQ * KT * NT * 64;
        const buf = allocator.alloc(u32, words) catch return map_out.setError("NNEDI3: out of memory.");
        for (0..numQ) |q| {
            const sm = if (q != 0) model.softmax_q2 else model.softmax_q1;
            const el = if (q != 0) model.elliott_q2 else model.elliott_q1;
            for (0..KT) |kt| {
                for (0..NT) |nt| {
                    for (0..32) |lane| {
                        for (0..2) |reg| {
                            const ncol = nt * 8 + lane / 4;
                            const p_ = ncol / 2;
                            const w_src = if (ncol % 2 == 0) sm else el;
                            const k0 = kt * 16 + (lane % 4) * 2 + reg * 8;
                            const h0 = floatToHalf(w_src[p_ * fs + k0]);
                            const h1 = floatToHalf(w_src[p_ * fs + k0 + 1]);
                            buf[(((q * KT + kt) * NT + nt) * 32 + lane) * 2 + reg] =
                                @as(u32, h0) | (@as(u32, h1) << 16);
                        }
                    }
                }
            }
        }
        pdw_blob = std.mem.sliceAsBytes(buf);
    } else {
        const buf = allocator.alloc(f32, numQ * fs * N * 2) catch return map_out.setError("NNEDI3: out of memory.");
        for (0..numQ) |q| {
            const sm = if (q != 0) model.softmax_q2 else model.softmax_q1;
            const el = if (q != 0) model.elliott_q2 else model.elliott_q1;
            for (0..fs) |k| {
                for (0..N) |pn| {
                    buf[((q * fs + k) * N + pn) * 2 + 0] = sm[pn * fs + k];
                    buf[((q * fs + k) * N + pn) * 2 + 1] = el[pn * fs + k];
                }
            }
        }
        pdw_blob = std.mem.sliceAsBytes(buf);
    }

    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    {
        var field_sum: usize = 0;
        var pad_sum: usize = 0;
        var dst_sum: usize = 0;
        var list_sum: usize = 0;
        var pi: usize = 0;
        while (pi < @as(usize, @intCast(fmt.numPlanes))) : (pi += 1) {
            const g = &d.geom[pi];
            g.process = proc[pi];
            g.w = if (pi == 0) d.vi_out.width else d.vi_out.width >> subW;
            g.h = if (pi == 0) d.vi_out.height else d.vi_out.height >> subH;
            if (!g.process) continue;
            g.rows = @divTrunc(g.h, 2);
            g.pad_stride = (g.w + MARGIN_H * 2 + 15) & ~@as(i32, 15);
            g.pad_h = g.rows + MARGIN_V * 2;

            g.field_off = field_sum;
            g.pad_off = pad_sum;
            g.dst_off = dst_sum;
            field_sum += @as(usize, @intCast(g.w * g.rows)) * d.bytes;
            pad_sum += @as(usize, @intCast(g.pad_stride * g.pad_h)) * d.bytes;
            dst_sum += @as(usize, @intCast(g.w * g.rows)) * d.bytes;
            g.list_off = list_sum;
            list_sum += @as(usize, @intCast(g.w * g.rows)) * 4;
            g.cnt_off = list_sum;
            list_sum += 16;

            if (@as(usize, @intCast(g.w * g.rows)) >= (1 << 31) or g.rows < 1 or g.w < 1)
                return map_out.setError("NNEDI3: plane geometry out of range.");
        }
        if (dst_sum == 0) return map_out.setError("NNEDI3: no planes to process.");
        d.field_bytes = field_sum;
        d.pad_bytes = pad_sum;
        d.dst_bytes = dst_sum;
        d.list_bytes = list_sum;
    }

    const data = initCuda(&d, device_id, num_streams, ps_blob, pdw_blob, pdb) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "NNEDI3: invalid device ID.",
            error.Nvrtc => "NNEDI3: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "NNEDI3: could not locate NVRTC (wheel should ship nvrtc64_130_0.dll next to the plugin).",
            error.OutOfDeviceMemory => "NNEDI3: out of device memory.",
            else => "NNEDI3: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu NNEDI3 init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = if (d.field > 1) .General else .StrictSpatial },
    };
    zapi.createVideoFilter(out, "NNEDI3", &d.vi_out, getFrame, free, .Parallel, &dep, data);
}
