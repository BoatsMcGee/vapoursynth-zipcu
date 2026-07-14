const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const pool_mod = @import("pool.zig");
const vsutil = @import("vsutil.zig");

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;
const math = std.math;

const ceilDiv = cu.ceilDiv;

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("eedi3.cu");

const mdis_max = 40;
const max_cfg = 2;

const spool_spare: usize = 3;

const BX: u32 = 48;
const LWS_FLOOR: u32 = 128;

const CreateError = cu.CreateError || error{MdisTooLarge};

// No -use_fast_math; default fmad (OpenCL reference contraction).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{"-std=c++17"},
    .log_name = "EEDI3",
};

const Config = struct {
    w: u32,
    src_h: u32,
    dst_h: u32,
    stride: u32,
    pstride: u32,
    n_interp_max: u32,
    in_w: u32 = 0,
    in_h: u32 = 0,
    out_w: u32 = 0,
    in_stride: u32 = 0,
    out_stride: u32 = 0,
};

const Data = struct {
    node: ?*vs.Node = null,
    sclip: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,

    horizontal: bool = false,
    hp: bool = false,
    bits: i32 = 32,
    half: bool = false,
    bytes: u32 = 4,
    field: u8 = 0,
    dh: bool = false,
    mdis: u8 = 0,
    nrad: u8 = 0,
    alpha: f32 = 0,
    beta: f32 = 0,
    gamma: f32 = 0,
    one_minus_ab: f32 = 0,
    vcheck: u8 = 0,
    rcp0: f32 = 0,
    rcp1: f32 = 0,
    rcp2: f32 = 0,
    vthresh2: f32 = 0,

    configs: [max_cfg]Config = undefined,
    n_cfg: usize = 0,
    plane_cfg: [3]usize = .{ 0, 0, 0 },
    off_io: [3]u32 = .{ 0, 0, 0 },
    off_src: [3]u32 = .{ 0, 0, 0 },
    off_dmap: [3]u32 = .{ 0, 0, 0 },
    sum_io: usize = 0,
    sum_src: usize = 0,
    sum_dmap: usize = 0,
    vc_geom: [18]i32 = [_]i32{0} ** 18,
    rowidx_host: [max_cfg][2][]i32 = undefined,
    dsty_host: [max_cfg][2][]i32 = undefined,

    tpitch: u32 = 0,
    lws: u32 = 0,
    vc_wg: u32 = 1024,
    pad: u32 = 0,

    sz_src: usize = 0,
    sz_srcpad: usize = 0,
    sz_hpsrcpad: usize = 0,
    sz_dst: usize = 0,
    sz_pbackt: usize = 0,
    parallel_v: bool = false,
    sz_srcpad2: usize = 8,
    sz_hpsrcpad2: usize = 8,
    sz_pbackt2: usize = 8,
    sz_dmap: usize = 0,
    sz_dst2: usize = 0,
    sz_scp: usize = 0,
    sz_inframe: usize = 0,
    sz_outframe: usize = 0,
    sz_scpframe: usize = 0,
    stage_src_off: [3]usize = .{ 0, 0, 0 },
    stage_scp_off: [3]usize = .{ 0, 0, 0 },
    src_stage_bytes: usize = 0,

    dev: cu.Device = .{},
    module: cu.Module = .{},
    fn_pad: cu.Function = .{},
    fn_pad_hp: cu.Function = .{},
    fn_copy: cu.Function = .{},
    fn_interp: cu.Function = .{},
    fn_vcheck: cu.Function = .{},
    fn_transpose: cu.Function = .{},
    d_rowidx: [max_cfg][2]cu.DeviceBuffer = .{ .{ .{}, .{} }, .{ .{}, .{} } },
    d_dsty: [max_cfg][2]cu.DeviceBuffer = .{ .{ .{}, .{} }, .{ .{}, .{} } },
    d_vcgeom: cu.DeviceBuffer = .{},
    n_tables: usize = 0,

    pool: pool_mod.Pool(Stream, Data) = .{},
    spool: pool_mod.Pool(SrcStage, Data) = .{},
};

const SrcStage = struct {
    buf: cu.HostBuffer,

    pub fn init(self: *SrcStage, d: *Data) !void {
        self.buf = try cu.HostBuffer.alloc(d.src_stage_bytes, .{});
    }

    pub fn deinit(self: *SrcStage) void {
        self.buf.deinit();
    }
};

const Stream = struct {
    stream: cu.Stream,
    cstream: cu.Stream,
    kstream2: cu.Stream,
    ev_up: [3]cu.Event,
    ev_k: [3]cu.Event,
    n_ev: usize,
    d_src: cu.DeviceBuffer,
    d_srcpad: cu.DeviceBuffer,
    d_hpsrcpad: cu.DeviceBuffer,
    d_srcpad2: cu.DeviceBuffer,
    d_hpsrcpad2: cu.DeviceBuffer,
    d_pbackt2: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,
    d_pbackt: cu.DeviceBuffer,
    d_dmap: cu.DeviceBuffer,
    d_dst2: cu.DeviceBuffer,
    d_scp: cu.DeviceBuffer,
    d_inframe: cu.DeviceBuffer,
    d_outframe: cu.DeviceBuffer,
    d_scpframe: cu.DeviceBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_src = try cu.DeviceBuffer.alloc(d.sz_src);
        errdefer self.d_src.deinit();
        self.d_srcpad = try cu.DeviceBuffer.alloc(d.sz_srcpad);
        errdefer self.d_srcpad.deinit();
        self.d_hpsrcpad = try cu.DeviceBuffer.alloc(d.sz_hpsrcpad);
        errdefer self.d_hpsrcpad.deinit();
        self.d_srcpad2 = try cu.DeviceBuffer.alloc(d.sz_srcpad2);
        errdefer self.d_srcpad2.deinit();
        self.d_hpsrcpad2 = try cu.DeviceBuffer.alloc(d.sz_hpsrcpad2);
        errdefer self.d_hpsrcpad2.deinit();
        self.d_pbackt2 = try cu.DeviceBuffer.alloc(d.sz_pbackt2);
        errdefer self.d_pbackt2.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(d.sz_dst);
        errdefer self.d_dst.deinit();
        self.d_pbackt = try cu.DeviceBuffer.alloc(d.sz_pbackt);
        errdefer self.d_pbackt.deinit();
        self.d_dmap = try cu.DeviceBuffer.alloc(d.sz_dmap);
        errdefer self.d_dmap.deinit();
        self.d_dst2 = try cu.DeviceBuffer.alloc(d.sz_dst2);
        errdefer self.d_dst2.deinit();
        self.d_scp = try cu.DeviceBuffer.alloc(d.sz_scp);
        errdefer self.d_scp.deinit();
        self.d_inframe = try cu.DeviceBuffer.alloc(d.sz_inframe);
        errdefer self.d_inframe.deinit();
        self.d_outframe = try cu.DeviceBuffer.alloc(d.sz_outframe);
        errdefer self.d_outframe.deinit();
        self.d_scpframe = try cu.DeviceBuffer.alloc(d.sz_scpframe);
        errdefer self.d_scpframe.deinit();
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
        errdefer self.cstream.deinit();
        self.kstream2 = .{};
        if (d.parallel_v) {
            self.kstream2 = try cu.Stream.init();
        }
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
        if (self.kstream2.handle != null) self.kstream2.deinit();
        self.destroyEvents();
        self.d_scpframe.deinit();
        self.d_outframe.deinit();
        self.d_inframe.deinit();
        self.d_scp.deinit();
        self.d_dst2.deinit();
        self.d_dmap.deinit();
        self.d_pbackt.deinit();
        self.d_dst.deinit();
        self.d_pbackt2.deinit();
        self.d_hpsrcpad2.deinit();
        self.d_srcpad2.deinit();
        self.d_hpsrcpad.deinit();
        self.d_srcpad.deinit();
        self.d_src.deinit();
    }
};

fn compileModule(d: *const Data) CreateError!cu.Module {
    const defines = std.fmt.allocPrint(allocator,
        \\#define CN {d}
        \\#define MDIS {d}
        \\#define BX {d}
        \\#define BITS {d}
        \\#define HALF {d}
        \\#define VC_WG {d}
        \\#define LWSF {d}
        \\#define MINB_NH 4
        \\#define RUN 16
        \\
    , .{
        d.nrad, d.mdis, BX, d.bits, @intFromBool(d.half), d.vc_wg, LWS_FLOOR,
    }) catch return error.OutOfMemory;
    defer allocator.free(defines);

    return cu.compile(d.dev, .{
        .text = kernel_source,
        .defines = defines,
        .name = "eedi3.cu",
    }, nvrtc_opts);
}

fn reflectRow(y: i32, h: i32) u32 {
    if (h == 1) return 0;
    var r = y;
    while (r < 0 or r >= h) {
        if (r < 0) r = -r;
        if (r >= h) r = 2 * (h - 1) - r;
    }
    return @intCast(r);
}

fn stencilRow(yy: i32, sh: i32, dh: bool) i32 {
    return @intCast(if (dh) reflectRow(yy, 2 * sh) / 2 else reflectRow(yy, sh));
}

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn runTranspose(d: *Data, s: *Stream, dst: cu.c.CUdeviceptr, src: cu.c.CUdeviceptr, in_w: u32, in_h: u32, in_stride: u32, out_stride: u32, soff: u32, doff: u32) CreateError!void {
    const a_iw: c_int = @intCast(in_w);
    const a_ih: c_int = @intCast(in_h);
    const a_is: c_int = @intCast(in_stride);
    const a_os: c_int = @intCast(out_stride);
    const a_so: c_int = @intCast(soff);
    const a_do: c_int = @intCast(doff);
    try s.stream.launch(d.fn_transpose, .{
        .grid = .{ ceilDiv(in_w, 16), ceilDiv(in_h, 16), 1 },
        .block = .{ 16, 16, 1 },
    }, .{ dst, src, a_iw, a_ih, a_is, a_os, a_so, a_do });
}

fn uploadPlane(d: *Data, s: *Stream, stg: [*]const u8, ci: usize, plane: usize, srcp: []const u8) CreateError!void {
    const cfg = &d.configs[ci];
    const off_src: u32 = d.off_src[plane];
    const up: [*]const u8 = stg + d.stage_src_off[plane];
    if (d.horizontal) {
        try s.stream.memcpyHtoD(s.d_inframe.ptr, up, srcp.len);
        try runTranspose(d, s, s.d_src.ptr, s.d_inframe.ptr, cfg.in_w, cfg.in_h, cfg.in_stride, cfg.stride, 0, off_src);
    } else {
        // Widen before multiply (u32 product wraps for >=4 GiB f32 regions).
        try s.cstream.memcpyHtoD(s.d_src.at(@as(usize, off_src) * d.bytes), up, srcp.len);
        try s.cstream.record(s.ev_up[plane]);
    }
}

fn processPlane(d: *Data, s: *Stream, stg: [*]const u8, ci: usize, plane: usize, scpp: ?[]const u8, field: u8) CreateError!void {
    const cfg = &d.configs[ci];
    const n_interp: u32 = (cfg.dst_h - field + 1) / 2;

    const off_io: u32 = d.off_io[plane];
    const off_src: u32 = d.off_src[plane];
    const off_dm: u32 = d.off_dmap[plane];
    const fb: usize = field & 1;

    const pv = d.parallel_v and plane == 2;
    const ks: cu.Stream = if (pv) s.kstream2 else s.stream;
    const sp_srcpad: cu.c.CUdeviceptr = if (pv) s.d_srcpad2.ptr else s.d_srcpad.ptr;
    const sp_hp: cu.c.CUdeviceptr = if (pv) s.d_hpsrcpad2.ptr else s.d_hpsrcpad.ptr;
    const sp_pb: cu.c.CUdeviceptr = if (pv) s.d_pbackt2.ptr else s.d_pbackt.ptr;

    if (!d.horizontal) {
        try ks.waitEvent(s.ev_up[plane]);
    }

    const pw: u32 = 2 * d.pad + cfg.w;
    {
        const a_w: c_int = @intCast(cfg.w);
        const a_stride: c_int = @intCast(cfg.stride);
        const a_pstride: c_int = @intCast(cfg.pstride);
        const a_pad: c_int = @intCast(d.pad);
        const a_src_h: c_int = @intCast(cfg.src_h);
        const a_soff: c_int = @intCast(off_src);
        try ks.launch(d.fn_pad, .{
            .grid = .{ ceilDiv(pw, 16), ceilDiv(cfg.src_h, 8), 1 },
            .block = .{ 16, 8, 1 },
        }, .{ sp_srcpad, s.d_src.ptr, a_w, a_stride, a_pstride, a_pad, a_src_h, a_soff });
    }
    if (d.hp) {
        const a_pstride: c_int = @intCast(cfg.pstride);
        const a_src_h: c_int = @intCast(cfg.src_h);
        const a_pad: c_int = @intCast(d.pad);
        const a_w: c_int = @intCast(cfg.w);
        try ks.launch(d.fn_pad_hp, .{
            .grid = .{ ceilDiv(pw, 16), ceilDiv(cfg.src_h, 8), 1 },
            .block = .{ 16, 8, 1 },
        }, .{ sp_hp, sp_srcpad, a_pstride, a_src_h, a_pad, a_w });
    }

    {
        const a_w: c_int = @intCast(cfg.w);
        const a_stride: c_int = @intCast(cfg.stride);
        const a_dh: c_int = @intFromBool(d.dh);
        const a_field: c_int = field;
        const a_src_h: c_int = @intCast(cfg.src_h);
        const a_dst_h: c_int = @intCast(cfg.dst_h);
        const a_dual: c_int = @intFromBool(d.vcheck > 0);
        const a_soff: c_int = @intCast(off_src);
        const a_doff: c_int = @intCast(off_io);
        try ks.launch(d.fn_copy, .{
            .grid = .{ ceilDiv(cfg.w, 16), ceilDiv(cfg.dst_h, 8), 1 },
            .block = .{ 16, 8, 1 },
        }, .{
            s.d_dst.ptr,  s.d_src.ptr, a_w,     a_stride,
            a_dh,         a_field,     a_src_h, a_dst_h,
            s.d_dst2.ptr, a_dual,      a_soff,  a_doff,
        });
    }

    if (n_interp > 0) {
        const a_rowidx = d.d_rowidx[ci][fb].ptr;
        const a_dsty = d.d_dsty[ci][fb].ptr;
        const a_w: c_int = @intCast(cfg.w);
        const a_stride: c_int = @intCast(cfg.stride);
        const a_pstride: c_int = @intCast(cfg.pstride);
        const a_pad: c_int = @intCast(d.pad);
        const a_mdis: c_int = d.mdis;
        const a_nrad: c_int = d.nrad;
        const a_alpha: f32 = d.alpha;
        const a_beta: f32 = d.beta;
        const a_gamma: f32 = d.gamma;
        const a_oma: f32 = d.one_minus_ab;
        const a_dual: c_int = @intFromBool(d.vcheck > 0);
        const a_doff: c_int = @intCast(off_io);
        const a_moff: c_int = @intCast(off_dm);
        const lcfg: cu.Launch = .{ .grid = .{ 1, n_interp, 1 }, .block = .{ d.lws, 1, 1 } };
        if (d.hp) {
            try ks.launch(d.fn_interp, lcfg, .{
                s.d_dst.ptr, sp_srcpad,    a_rowidx, a_dsty,
                sp_pb,       s.d_dmap.ptr, a_w,      a_stride,
                a_pstride,   a_pad,        a_mdis,   a_nrad,
                a_alpha,     a_beta,       a_gamma,  a_oma,
                sp_hp,       s.d_dst2.ptr, a_dual,   a_doff,
                a_moff,
            });
        } else {
            try ks.launch(d.fn_interp, lcfg, .{
                s.d_dst.ptr,  sp_srcpad,    a_rowidx, a_dsty,
                sp_pb,        s.d_dmap.ptr, a_w,      a_stride,
                a_pstride,    a_pad,        a_mdis,   a_nrad,
                a_alpha,      a_beta,       a_gamma,  a_oma,
                s.d_dst2.ptr, a_dual,       a_doff,   a_moff,
            });
        }
    }

    if (!d.horizontal) {
        try ks.record(s.ev_k[plane]);
    }

    if (d.vcheck > 0) {
        if (scpp) |sp| {
            const scp_up: [*]const u8 = stg + d.stage_scp_off[plane];
            if (d.horizontal) {
                try s.stream.memcpyHtoD(s.d_scpframe.ptr, scp_up, sp.len);
                try runTranspose(d, s, s.d_scp.ptr, s.d_scpframe.ptr, cfg.out_w, cfg.in_h, cfg.out_stride, cfg.stride, 0, off_io);
            } else {
                try s.stream.memcpyHtoD(s.d_scp.at(@as(usize, off_io) * d.bytes), scp_up, sp.len);
            }
        }
    }
}

fn vcheckFused(d: *Data, s: *Stream, field: u8) CreateError!void {
    const fb: usize = field & 1;
    const np: u32 = @intCast(d.vi.format.numPlanes);
    var rix: [3]cu.c.CUdeviceptr = undefined;
    for (0..3) |p| {
        const ci = d.plane_cfg[if (p < np) p else 0];
        rix[p] = d.d_rowidx[ci][fb].ptr;
    }
    const a_field: c_int = field;
    const a_vmode: c_int = d.vcheck;
    const a_use_scp: c_int = @intFromBool(d.sclip != null);
    const a_hp: c_int = @intFromBool(d.hp);
    const a_rcp0: f32 = d.rcp0;
    const a_rcp1: f32 = d.rcp1;
    const a_rcp2: f32 = d.rcp2;
    const a_vth2: f32 = d.vthresh2;
    const a_np: c_int = @intCast(np);
    try s.stream.launch(d.fn_vcheck, .{
        .grid = .{ np, 1, 1 },
        .block = .{ d.vc_wg, 1, 1 },
    }, .{
        s.d_dst2.ptr, s.d_dst.ptr, s.d_src.ptr,
        rix[0],       rix[1],      rix[2],
        s.d_dmap.ptr, s.d_scp.ptr, d.d_vcgeom.ptr,
        a_field,      a_vmode,     a_use_scp,
        a_hp,         a_rcp0,      a_rcp1,
        a_rcp2,       a_vth2,      a_np,
    });
}

fn downloadPlane(d: *Data, s: *Stream, ci: usize, plane: usize, dstp: []u8) CreateError!void {
    const cfg = &d.configs[ci];
    const off_io: u32 = d.off_io[plane];
    const result: cu.DeviceBuffer = if (d.vcheck > 0) s.d_dst2 else s.d_dst;
    if (d.horizontal) {
        try runTranspose(d, s, s.d_outframe.ptr, result.ptr, cfg.w, cfg.dst_h, cfg.stride, cfg.out_stride, off_io, 0);
        try s.stream.memcpyDtoH(dstp.ptr, s.d_outframe.ptr, dstp.len);
    } else {
        const gate = if (d.vcheck > 0) s.ev_k[0] else s.ev_k[plane];
        try s.cstream.waitEvent(gate);
        try s.cstream.memcpyDtoH(dstp.ptr, result.at(@as(usize, off_io) * d.bytes), dstp.len);
    }
}

fn getFrame(n: c_int, ar: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);
    const src_n: c_int = if (d.field > 1) @divTrunc(n, 2) else n;

    if (ar == .Initial) {
        zapi.requestFrameFilter(src_n, d.node);
        if (d.vcheck > 0 and d.sclip != null) zapi.requestFrameFilter(n, d.sclip);
    } else if (ar == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, src_n);
        defer src.deinit();
        const scp = if (d.vcheck > 0 and d.sclip != null) zapi.initZFrame(d.sclip, n) else null;
        defer if (scp) |sc| sc.deinit();
        const dst = if (d.horizontal) src.newVideoFrame3(.{ .width = d.vi.width }) else src.newVideoFrame3(.{ .height = d.vi.height });
        const dst_props = dst.getPropertiesRW();

        var field: u8 = d.field & 1;
        switch (dst_props.getFieldBased() orelse .PROGRESSIVE) {
            .BOTTOM => field = 0,
            .TOP => field = 1,
            else => {},
        }
        if (d.field > 1) field = @as(u8, @intCast(n & 1)) ^ field;

        const stg: *SrcStage = d.spool.acquire();
        defer d.spool.release(stg);
        {
            const nplanes: usize = @intCast(d.vi.format.numPlanes);
            for (0..nplanes) |p| {
                const sp = src.getReadSlice(@intCast(p));
                @memcpy((stg.buf.ptr + d.stage_src_off[p])[0..sp.len], sp);
            }
            if (scp) |sc| {
                for (0..nplanes) |p| {
                    const sp = sc.getReadSlice(@intCast(p));
                    @memcpy((stg.buf.ptr + d.stage_scp_off[p])[0..sp.len], sp);
                }
            }
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, stg.buf.ptr, src, scp, dst, field) catch |err| {
            zapi.setFilterError("EEDI3: process failed.");
            std.log.err("vszipcu EEDI3 process failed: {t}", .{err});
            dst.deinit();
            return null;
        };

        dst_props.setFieldBased(.PROGRESSIVE);
        if (d.field > 1) {
            var dn = dst_props.getDurationNum();
            var dd = dst_props.getDurationDen();
            if (dn != null and dd != null) {
                vsh.muldivRational(&dn.?, &dd.?, 1, 2);
                dst_props.setDurationNum(dn.?);
                dst_props.setDurationDen(dd.?);
            }
        }
        return dst.frame;
    }
    return null;
}

fn process(d: *Data, s: *Stream, stg: [*]const u8, src: ZFrame, scp: ?ZFrame, dst: ZFrameW, field: u8) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain: async copies reference pinned staging / VS memory.
    errdefer {
        s.cstream.drain();
        s.stream.drain();
        if (s.kstream2.handle != null) s.kstream2.drain();
    }

    const nplanes: u32 = @intCast(d.vi.format.numPlanes);

    var plane: u32 = 0;
    while (plane < nplanes) : (plane += 1) {
        const ci = d.plane_cfg[plane];
        const srcp = src.getReadSlice(plane);
        std.debug.assert(src.getStride(plane) ==
            (if (d.horizontal) d.configs[ci].in_stride else d.configs[ci].stride) * d.bytes);
        try uploadPlane(d, s, stg, ci, plane, srcp);
    }
    plane = 0;
    while (plane < nplanes) : (plane += 1) {
        const scpp: ?[]const u8 = if (scp) |sc| sc.getReadSlice(plane) else null;
        try processPlane(d, s, stg, d.plane_cfg[plane], plane, scpp, field);
        if (d.horizontal and d.vcheck == 0) {
            try downloadPlane(d, s, d.plane_cfg[plane], plane, dst.getWriteSlice(plane));
        }
    }
    if (d.vcheck > 0) {
        if (d.parallel_v and nplanes == 3) {
            try s.stream.waitEvent(s.ev_k[2]);
        }
        try vcheckFused(d, s, field);
        if (!d.horizontal) try s.stream.record(s.ev_k[0]);
    }
    if (!d.horizontal or d.vcheck > 0) {
        plane = 0;
        while (plane < nplanes) : (plane += 1) {
            try downloadPlane(d, s, d.plane_cfg[plane], plane, dst.getWriteSlice(plane));
        }
    }

    try s.cstream.sync();
    try s.stream.sync();
    if (s.kstream2.handle != null) try s.kstream2.sync();
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.dev.push() catch {};
    d.pool.deinit();
    d.spool.deinit();
    freeTables(d);
    d.module.deinit();
    d.dev.pop();
    d.dev.deinit();
    freeStencilHost(d);
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.sclip);
    allocator.destroy(d);
}

fn freeTables(d: *Data) void {
    var i: usize = d.n_tables;
    while (i > 0) {
        i -= 1;
        d.d_dsty[i / 2][i % 2].deinit();
        d.d_rowidx[i / 2][i % 2].deinit();
    }
    d.n_tables = 0;
    d.d_vcgeom.deinit();
    d.d_vcgeom = .{};
}

fn freeStencilHost(d: *Data) void {
    for (0..d.n_cfg) |ci| {
        for (0..2) |f| {
            allocator.free(d.rowidx_host[ci][f]);
            allocator.free(d.dsty_host[ci][f]);
        }
    }
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize) CreateError!*Data {
    d.dev = try cu.initDevice(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    const max_threads = try d.dev.attribute(cu.c.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    d.vc_wg = @min(@as(u32, 1024), @as(u32, @intCast(max_threads)));
    d.lws = @max(@as(u32, @intCast(vsh.ceilN(d.tpitch, 32))), LWS_FLOOR);
    if (d.lws > @as(u32, @intCast(max_threads))) d.lws = @intCast(vsh.ceilN(d.tpitch, 32));
    if (d.lws > @as(u32, @intCast(max_threads))) return error.MdisTooLarge;

    d.module = try compileModule(d);
    errdefer d.module.deinit();
    d.fn_pad = try d.module.function("pad_src");
    d.fn_pad_hp = try d.module.function("pad_hp");
    d.fn_copy = try d.module.function("copy_kept");
    d.fn_interp = try d.module.function(if (d.hp) "interp_hp" else "interp");
    d.fn_vcheck = try d.module.function("vcheck");
    d.fn_transpose = try d.module.function("transpose");

    errdefer freeTables(d);
    for (0..d.n_cfg) |ci| {
        for (0..2) |fb| {
            const ri = d.rowidx_host[ci][fb];
            const dy = d.dsty_host[ci][fb];
            d.d_rowidx[ci][fb] = try cu.DeviceBuffer.alloc(ri.len * 4);
            errdefer d.d_rowidx[ci][fb].deinit();
            try cu.driver.memcpyHtoDSync(d.d_rowidx[ci][fb].ptr, ri.ptr, ri.len * 4);
            d.d_dsty[ci][fb] = try cu.DeviceBuffer.alloc(dy.len * 4);
            errdefer d.d_dsty[ci][fb].deinit();
            try cu.driver.memcpyHtoDSync(d.d_dsty[ci][fb].ptr, dy.ptr, dy.len * 4);
            d.n_tables = ci * 2 + fb + 1;
        }
    }
    if (d.vcheck > 0) {
        d.d_vcgeom = try cu.DeviceBuffer.alloc(d.vc_geom.len * 4);
        try cu.driver.memcpyHtoDSync(d.d_vcgeom.ptr, &d.vc_geom, d.vc_geom.len * 4);
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
    const n_stage = num_streams + spool_spare;
    data.spool.prime(data, n_stage) catch {
        data.pool.deinit();
        return error.OutOfMemory;
    };
    data.spool.prewarm(n_stage) catch |err| {
        data.spool.deinit();
        data.pool.deinit();
        return err;
    };
    return data;
}

pub fn createEEDI3(in: ?*const vs.Map, out: ?*vs.Map, ud: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createImpl(in, out, ud, core_ptr, vsapi, false);
}
pub fn createEEDI3H(in: ?*const vs.Map, out: ?*vs.Map, ud: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createImpl(in, out, ud, core_ptr, vsapi, true);
}

fn createImpl(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API, horizontal: bool) void {
    var d: Data = .{};
    d.horizontal = horizontal;
    const zapi = ZAPI.init(vsapi, core_ptr, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi_in = map_in.getNodeVi("clip").?;
    d.vi = vi_in.*;

    const vcheck = map_in.getValue(i32, "vcheck") orelse 2;
    d.sclip = if (vcheck > 0) map_in.getNode("sclip") else null;

    var keep = false;
    var stencil_built = false;
    defer if (!keep) {
        if (stencil_built) freeStencilHost(&d);
        zapi.freeNode(d.node);
        zapi.freeNode(d.sclip);
    };

    const cf = d.vi.format.colorFamily;
    const fmt = d.vi.format;
    const io_bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Integer and (io_bits == 8 or io_bits == 16)) or
        (fmt.sampleType == .Float and (io_bits == 16 or io_bits == 32));
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (cf != .Gray and cf != .YUV and cf != .RGB))
    {
        return map_out.setError("EEDI3: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    }
    d.bits = io_bits;
    d.half = fmt.sampleType == .Float and io_bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    const field = map_in.getValue(i32, "field") orelse 0;
    const mdis = map_in.getValue(i32, "mdis") orelse 20;
    const nrad = map_in.getValue(i32, "nrad") orelse 2;
    d.alpha = map_in.getValue(f32, "alpha") orelse 0.2;
    d.beta = map_in.getValue(f32, "beta") orelse 0.25;
    d.gamma = map_in.getValue(f32, "gamma") orelse 20.0;
    d.dh = map_in.getBool("dh") orelse false;
    d.hp = map_in.getBool("hp") orelse false;
    const ns_req = map_in.getValue(i32, "num_streams");

    const interp_axis: i32 = if (horizontal) d.vi.width else d.vi.height;

    if (field < 0 or field > 3) return map_out.setError("EEDI3: field must be 0..3.");
    if (d.dh and field > 1) return map_out.setError("EEDI3: field must be 0 or 1 when dh=True.");
    if (!d.dh and (interp_axis & 1) != 0) return map_out.setError("EEDI3: interpolated axis must be mod 2 when dh=False.");
    if (d.alpha < 0 or d.alpha > 1) return map_out.setError("EEDI3: alpha 0..1.");
    if (d.beta < 0 or d.beta > 1) return map_out.setError("EEDI3: beta 0..1.");
    if (d.alpha + d.beta > 1) return map_out.setError("EEDI3: alpha+beta must be <= 1.");
    if (d.gamma < 0) return map_out.setError("EEDI3: gamma >= 0.");
    if (nrad < 0 or nrad > 3) return map_out.setError("EEDI3: nrad 0..3.");
    if (mdis < 1 or mdis > mdis_max) return map_out.setError("EEDI3: mdis 1..40.");
    if (vcheck < 0 or vcheck > 3) return map_out.setError("EEDI3: vcheck 0..3.");
    if (ns_req) |ns| {
        if (ns < 1 or ns > 32) return map_out.setError("EEDI3: num_streams must be 1..32.");
    }
    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("EEDI3: invalid device ID.");

    if (field > 1) {
        if (d.vi.numFrames > math.maxInt(i32) / 2) return map_out.setError("EEDI3: clip too long.");
        d.vi.numFrames *= 2;
        vsh.muldivRational(&d.vi.fpsNum, &d.vi.fpsDen, 2, 1);
    }
    if (d.dh) {
        if (horizontal) d.vi.width *= 2 else d.vi.height *= 2;
    }

    if (d.sclip) |sc| {
        const svi = zapi.getVideoInfo(sc);
        if (!vsh.isSameVideoInfo(svi, &d.vi) or svi.numFrames != d.vi.numFrames) {
            return map_out.setError("EEDI3: sclip's format/dimensions/length must match the output.");
        }
    }

    d.field = @intCast(field);
    d.mdis = @intCast(mdis);
    d.nrad = @intCast(nrad);
    d.vcheck = @intCast(vcheck);
    d.one_minus_ab = 1.0 - d.alpha - d.beta;
    d.alpha /= 3.0;
    d.beta /= 255.0;
    d.gamma /= 255.0;

    const vthresh0 = (map_in.getValue(f32, "vthresh0") orelse 32.0) / 255.0;
    const vthresh1 = (map_in.getValue(f32, "vthresh1") orelse 64.0) / 255.0;
    d.vthresh2 = map_in.getValue(f32, "vthresh2") orelse 4.0;
    if (vcheck > 0 and (vthresh0 <= 0 or vthresh1 <= 0 or d.vthresh2 <= 0)) return map_out.setError("EEDI3: vthresh* must be > 0.");
    d.rcp0 = 1.0 / vthresh0;
    d.rcp1 = 1.0 / vthresh1;
    d.rcp2 = 1.0 / d.vthresh2;

    d.tpitch = @intCast(if (d.hp) 4 * mdis + 1 else 2 * mdis + 1);
    d.pad = @intCast(3 * mdis + nrad + 2);

    {
        const strides_out = vsutil.strideFromVi(&d.vi);
        const strides_in = vsutil.strideFromVi(vi_in);
        const nplanes: usize = @intCast(d.vi.format.numPlanes);
        var p: usize = 0;
        while (p < nplanes) : (p += 1) {
            const cat: usize = if (p == 0) 0 else 1;
            const sw: u5 = if (p == 0) 0 else @intCast(d.vi.format.subSamplingW);
            const sh: u5 = if (p == 0) 0 else @intCast(d.vi.format.subSamplingH);
            var cfg: Config = .{ .w = 0, .src_h = 0, .dst_h = 0, .stride = 0, .pstride = 0, .n_interp_max = 0 };
            if (horizontal) {
                cfg.in_w = @as(u32, @intCast(vi_in.width)) >> sw;
                cfg.in_h = @as(u32, @intCast(vi_in.height)) >> sh;
                cfg.out_w = @as(u32, @intCast(d.vi.width)) >> sw;
                cfg.w = cfg.in_h;
                cfg.src_h = cfg.in_w;
                cfg.dst_h = cfg.out_w;
                cfg.in_stride = strides_in[cat];
                cfg.out_stride = strides_out[cat];
                const n_align: u32 = @divExact(vsutil.vsFrameAlignment(), @as(u32, @intCast(d.vi.format.bytesPerSample)));
                cfg.stride = @max(strides_out[cat], @as(u32, @intCast(vsh.ceilN(@as(usize, cfg.w), n_align))));
            } else {
                cfg.w = @as(u32, @intCast(vi_in.width)) >> sw;
                cfg.src_h = @as(u32, @intCast(vi_in.height)) >> sh;
                cfg.dst_h = @as(u32, @intCast(d.vi.height)) >> sh;
                cfg.stride = strides_out[cat];
            }
            cfg.pstride = @intCast(vsh.ceilN(@as(usize, d.pad) * 2 + cfg.w, 8));
            cfg.n_interp_max = (cfg.dst_h + 1) / 2;
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

    {
        var acc_io: u64 = 0;
        var acc_src: u64 = 0;
        var acc_dm: u64 = 0;
        const nplanes: usize = @intCast(d.vi.format.numPlanes);
        for (0..nplanes) |p| {
            const cfg = &d.configs[d.plane_cfg[p]];
            acc_io += @as(u64, cfg.stride) * cfg.dst_h;
            acc_src += @as(u64, cfg.stride) * cfg.src_h;
            acc_dm += @as(u64, cfg.n_interp_max) * cfg.stride;
        }
        const plane0: u64 = @as(u64, d.configs[0].stride) * d.configs[0].dst_h;
        const max_extent = @max(plane0, @max(acc_io, @max(acc_src, acc_dm)));
        if (max_extent >= (1 << 31)) {
            return map_out.setError("EEDI3: frame too large (a plane exceeds 2^31 samples).");
        }
        // ZFrame u32 plane-slice math must not overflow for f32 planes.
        if (plane0 * d.bytes >= (1 << 32)) {
            return map_out.setError("EEDI3: frame too large (a plane exceeds 2^32 bytes).");
        }
        // CUDA gridDim.y cap 65535.
        if (d.configs[0].n_interp_max > 65535 or @max(d.configs[0].src_h, d.configs[0].dst_h) / 8 + 1 > 65535) {
            return map_out.setError("EEDI3: frame too tall for the CUDA launch grid.");
        }
        {
            var io: u64 = 0;
            var sr: u64 = 0;
            var dm: u64 = 0;
            for (0..nplanes) |p| {
                const cfg = &d.configs[d.plane_cfg[p]];
                d.off_io[p] = @intCast(io);
                d.off_src[p] = @intCast(sr);
                d.off_dmap[p] = @intCast(dm);
                d.vc_geom[p * 6 + 0] = @intCast(cfg.w);
                d.vc_geom[p * 6 + 1] = @intCast(cfg.stride);
                d.vc_geom[p * 6 + 2] = @intCast(cfg.dst_h);
                d.vc_geom[p * 6 + 3] = @intCast(io);
                d.vc_geom[p * 6 + 4] = @intCast(sr);
                d.vc_geom[p * 6 + 5] = @intCast(dm);
                io += @as(u64, cfg.stride) * cfg.dst_h;
                sr += @as(u64, cfg.stride) * cfg.src_h;
                dm += @as(u64, cfg.n_interp_max) * cfg.stride;
            }
        }
        d.sum_io = @intCast(acc_io);
        d.sum_src = @intCast(acc_src);
        d.sum_dmap = @intCast(acc_dm);
    }

    {
        const c0 = &d.configs[0];
        const bytes: usize = d.bytes;
        const src_elems: usize = d.sum_src;
        const io_elems: usize = d.sum_io;
        const dmap_elems: usize = d.sum_dmap;
        const have_scp = d.sclip != null and d.vcheck > 0;
        d.sz_src = src_elems * bytes;
        d.sz_srcpad = @as(usize, c0.pstride) * c0.src_h * 4;
        d.sz_hpsrcpad = if (d.hp) d.sz_srcpad else 8;
        d.sz_dst = io_elems * bytes;
        // Must match kernel `(w*TP + 15) & ~15` (uint4 backtrack staging).
        d.sz_pbackt = @as(usize, c0.n_interp_max) * vsh.ceilN(@as(usize, c0.w) * d.tpitch, 16);

        d.parallel_v = !horizontal and d.vi.format.numPlanes == 3;
        if (d.parallel_v) {
            const c2 = &d.configs[d.plane_cfg[2]];
            d.sz_srcpad2 = @as(usize, c2.pstride) * c2.src_h * 4;
            d.sz_hpsrcpad2 = if (d.hp) d.sz_srcpad2 else 8;
            d.sz_pbackt2 = @as(usize, c2.n_interp_max) * vsh.ceilN(@as(usize, c2.w) * d.tpitch, 16);
        }
        d.sz_dmap = dmap_elems * 4;
        d.sz_dst2 = if (d.vcheck > 0) io_elems * bytes else 8;
        d.sz_scp = if (have_scp) io_elems * bytes else 8;
        d.sz_inframe = if (horizontal) @as(usize, c0.in_stride) * c0.in_h * bytes else 8;
        d.sz_outframe = if (horizontal) @as(usize, c0.out_stride) * c0.in_h * bytes else 8;
        d.sz_scpframe = if (have_scp and horizontal) @as(usize, c0.out_stride) * c0.in_h * bytes else 8;

        var soff: usize = 0;
        const nplanes: usize = @intCast(d.vi.format.numPlanes);
        for (0..nplanes) |p| {
            const cfg = &d.configs[d.plane_cfg[p]];
            const up: usize = if (horizontal) @as(usize, cfg.in_stride) * cfg.in_h * bytes else @as(usize, cfg.stride) * cfg.src_h * bytes;
            const down: usize = if (horizontal) @as(usize, cfg.out_stride) * cfg.in_h * bytes else @as(usize, cfg.stride) * cfg.dst_h * bytes;
            d.stage_src_off[p] = soff;
            soff += up;
            d.stage_scp_off[p] = soff;
            if (have_scp) soff += down;
        }
        d.src_stage_bytes = @max(1, soff);
    }

    for (0..d.n_cfg) |ci| {
        const cfg = &d.configs[ci];
        const sh_i: i32 = @intCast(cfg.src_h);
        for (0..2) |f| {
            const n: u32 = (cfg.dst_h - @as(u32, @intCast(f)) + 1) / 2;
            const ri = allocator.alloc(i32, @as(usize, n) * 4) catch unreachable;
            const dy = allocator.alloc(i32, n) catch unreachable;
            var off: u32 = 0;
            var iy: i32 = @intCast(f);
            while (off < n) : (off += 1) {
                ri[off * 4 + 0] = stencilRow(iy - 3, sh_i, d.dh);
                ri[off * 4 + 1] = stencilRow(iy - 1, sh_i, d.dh);
                ri[off * 4 + 2] = stencilRow(iy + 1, sh_i, d.dh);
                ri[off * 4 + 3] = stencilRow(iy + 3, sh_i, d.dh);
                dy[off] = iy;
                iy += 2;
            }
            d.rowidx_host[ci][f] = ri;
            d.dsty_host[ci][f] = dy;
        }
    }
    stencil_built = true;

    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    const data = initCuda(&d, device_id, num_streams) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "EEDI3: invalid device ID.",
            error.MdisTooLarge => "EEDI3: device max threads per block is too small for this mdis.",
            error.FrameTooLarge => "EEDI3: frame too large.",
            error.Nvrtc => "EEDI3: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "EEDI3: could not locate NVRTC (put nvrtc64_130_0.dll next to the plugin, or: pip install nvidia-cuda-nvrtc).",
            error.OutOfDeviceMemory => "EEDI3: out of device memory.",
            else => "EEDI3: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu EEDI3 init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep_buf: [2]vs.FilterDependency = undefined;
    dep_buf[0] = .{ .source = d.node, .requestPattern = if (d.field > 1) .General else .StrictSpatial };
    var ndeps: usize = 1;
    if (d.sclip) |sc| {
        dep_buf[ndeps] = .{ .source = sc, .requestPattern = .StrictSpatial };
        ndeps += 1;
    }
    zapi.createVideoFilter(out, if (horizontal) "EEDI3H" else "EEDI3", &data.vi, getFrame, free, .Parallel, dep_buf[0..ndeps], data);
}
