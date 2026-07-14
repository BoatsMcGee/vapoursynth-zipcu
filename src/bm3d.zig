const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const pool_mod = @import("pool.zig");
const vsutil = @import("vsutil.zig");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const cuda = cu.c;
const math = std.math;

const CreateError = cu.CreateError;
const check = cu.check;
const ceilDiv = cu.ceilDiv;

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("bm3d.cu");

// -use_fast_math; aggregate uses __fdiv_rn. -modify-stack-limit=false.
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{ "-use_fast_math", "-std=c++17", "-modify-stack-limit=false" },
    .log_name = "BM3D",
};

// Fused VAggregate: no -use_fast_math (-ftz flushes denormals).
const vagg_source = @embedFile("bm3d_vagg.cu");
const vagg_opts: cu.nvrtc.Options = .{
    .extra = &.{ "-std=c++17", "-modify-stack-limit=false" },
    .log_name = "BM3D VAggregate",
};

const AggSrc = extern struct {
    p: [MAX_TW]cuda.CUdeviceptr = @splat(0),
    z: [MAX_TW]c_int = @splat(0),
};

// BY VALUE: MODULE layout must match — mismatch = silent wrong pointers.
comptime {
    std.debug.assert(@sizeOf(AggSrc) == 400);
    std.debug.assert(@alignOf(AggSrc) == 8);
    std.debug.assert(@offsetOf(AggSrc, "p") == 0);
    std.debug.assert(@offsetOf(AggSrc, "z") == 264);
}

fn aggZ(i: usize, n: i32, nframes: i32, radius: i32) i32 {
    return @min(@max(2 * radius - @as(i32, @intCast(i)), n - nframes + 1 + radius), n + radius);
}

const FLT_EPSILON: f32 = 1.19209290e-7;

const MAX_RADIUS: i32 = 16;
const MAX_TW: usize = 2 * @as(usize, MAX_RADIUS) + 1;

const DEFAULT_WARPS: u32 = 2;

const ModKey = struct {
    w: i32,
    h: i32,
    stride: i32,
    sigma: [3]f32,
    block_step: i32,
    bm_range: i32,
    ps_num: i32,
    ps_range: i32,
    proc_mask: u32,
    bm_error: BmError,
    t2d: Transform,
    t1d: Transform,
};

const BmError = enum {
    ssd,
    sad,
    zssd,
    zsad,
    ssd_norm,

    fn parse(s: []const u8) ?BmError {
        if (eqlLower(s, "ssd")) return .ssd;
        if (eqlLower(s, "sad")) return .sad;
        if (eqlLower(s, "zssd")) return .zssd;
        if (eqlLower(s, "zsad")) return .zsad;
        if (eqlLower(s, "ssd/norm")) return .ssd_norm;
        return null;
    }
};

const Transform = enum {
    dct,
    haar,
    wht,
    bior1_5,

    fn parse(s: []const u8) ?Transform {
        if (eqlLower(s, "dct")) return .dct;
        if (eqlLower(s, "haar")) return .haar;
        if (eqlLower(s, "wht")) return .wht;
        if (eqlLower(s, "bior1.5")) return .bior1_5;
        return null;
    }
};

fn eqlLower(s: []const u8, lit: []const u8) bool {
    if (s.len != lit.len) return false;
    for (s, lit) |a, b| {
        if (std.ascii.toLower(a) != b) return false;
    }
    return true;
}

// Deadlock freedom: reserve whole window all-or-nothing; never block while holding slots.
const framecache = @import("framecache.zig");
const CacheSlot = framecache.CacheSlot;
const FrameCache = framecache.FrameCache;

const Entry = struct {
    key: ModKey,
    mod_idx: usize,

    n_planes: u32,
    planes: [3]u8,
    plane_extent: usize,

    cache_off: usize,
    src_off: usize,
    src_elems: usize,
    res_off: usize,
    res_elems: usize,
    dst_off: usize,
    dst_elems: usize,
};

const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    vi_out: vs.VideoInfo = undefined,

    radius: i32 = 0,
    tw: usize = 1,

    fused: bool = false,

    acc: FrameCache = .{},
    vagg_mods: [3]cu.Module = .{ .{}, .{}, .{} },
    fn_vagg: [3]cu.Function = undefined,
    n_vagg: usize = 0,

    extractor: f32 = 0,
    warps: u32 = DEFAULT_WARPS,
    chroma: bool = false,
    final: bool = false,
    zero_init: bool = true,
    process: [3]bool = .{ false, false, false },

    n_entries: usize = 0,
    entries: [3]Entry = undefined,

    sum_src: usize = 0,
    sum_res: usize = 0,
    sum_dst: usize = 0,

    pin_up: bool = false,
    pin_down: bool = false,
    zc_dst: bool = false,

    stage_down_off: usize = 0,
    stage_elems: usize = 0,

    scache: bool = false,
    cache: FrameCache = .{},
    n_clips: usize = 1,

    dev: cu.Device = .{},
    mods: [3]cu.Module = .{ .{}, .{}, .{} },
    fn_bm3d: [3]cu.Function = .{ .{}, .{}, .{} },
    fn_agg: [3]cu.Function = .{ .{}, .{}, .{} },
    n_mods: usize = 0,

    pool: pool_mod.Pool(Stream, Data) = .{},

    fn normalOut(self: *const Data) bool {
        return self.radius == 0 or self.fused;
    }
};

const Stream = struct {
    stream: cu.Stream,
    cstream: cu.Stream,
    ev_up: [3]cu.Event,
    ev_k: [3]cu.Event,
    n_ev: usize,
    d_src: cu.DeviceBuffer,
    d_res: cu.DeviceBuffer,
    d_dst: cu.DeviceBuffer,
    h_stage: ?cu.HostBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_src = try cu.DeviceBuffer.alloc(d.sum_src * 4);
        errdefer self.d_src.deinit();
        self.d_res = try cu.DeviceBuffer.alloc(if (d.fused) 0 else d.sum_res * 4);
        errdefer self.d_res.deinit();
        self.d_dst = try cu.DeviceBuffer.alloc(if (d.sum_dst > 0 and !d.zc_dst) d.sum_dst * 4 else 0);
        errdefer self.d_dst.deinit();
        self.h_stage = if (d.stage_elems > 0) try cu.HostBuffer.alloc(d.stage_elems * 4, .{}) else null;
        errdefer if (self.h_stage) |hs| hs.deinit();
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
        self.d_dst.deinit();
        self.d_res.deinit();
        self.d_src.deinit();
    }
};

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

const Window = struct {
    f: [2][MAX_TW]ZFrame,
    idx: [MAX_TW]i64,
    n_clips: usize,
    tw: usize,

    fn deinit(self: *const Window) void {
        for (0..self.n_clips) |c| {
            for (0..self.tw) |t| self.f[c][t].deinit();
        }
    }
};

const MAX_SPAN: usize = 2 * MAX_TW - 1;

const FusedWin = struct {
    f: [2][MAX_SPAN]ZFrame,
    lo: i32,
    cnt: usize,
    n_clips: usize,

    fn deinit(self: *const FusedWin) void {
        for (0..self.n_clips) |c| {
            for (0..self.cnt) |j| self.f[c][j].deinit();
        }
    }
};

fn process(d: *Data, s: *Stream, win: *const Window, dst: ZFrameW) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();

    const tw = d.tw;
    const n_clips = win.n_clips;

    var slot_idx: [2 * MAX_TW]usize = undefined;
    var slot_load: [2 * MAX_TW]bool = undefined;
    var n_slots: usize = 0;

    // Drain → abandon unpublished → release (never release before drain; LIFO defers).
    defer if (n_slots > 0) d.cache.release(slot_idx[0..n_slots]);
    errdefer for (0..n_slots) |ki| {
        if (slot_load[ki]) d.cache.abandon(slot_idx[ki]);
    };
    errdefer {
        s.cstream.drain();
        s.stream.drain();
    }

    if (d.scache) {
        var keys: [2 * MAX_TW]i64 = undefined;
        for (0..n_clips) |c| {
            for (0..tw) |t| keys[c * tw + t] = (@as(i64, @intCast(c)) << 40) | win.idx[t];
        }
        const nk = n_clips * tw;
        d.cache.acquire(keys[0..nk], slot_idx[0..nk], slot_load[0..nk]);
        n_slots = nk;

        const hs = s.h_stage.?.ptr;
        for (0..nk) |ki| {
            if (!slot_load[ki]) continue;
            const c = ki / tw;
            const t = ki % tw;
            const slot = &d.cache.slots[slot_idx[ki]];
            for (0..d.n_entries) |ei| {
                const e = &d.entries[ei];
                for (0..e.n_planes) |i| {
                    const srcp = win.f[c][t].getReadSlice(e.planes[i]);
                    const coff = e.cache_off + i * e.plane_extent;
                    const hoff = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                    @memcpy((hs + hoff * 4)[0..srcp.len], srcp);
                    try s.cstream.memcpyHtoD(slot.buf.at(coff * 4), hs + hoff * 4, srcp.len);
                }
            }
            try s.cstream.record(slot.ev);
            d.cache.publish(slot_idx[ki]);
        }

        for (0..nk) |ki| {
            try s.cstream.waitEvent(d.cache.slots[slot_idx[ki]].ev);
        }
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..nk) |ki| {
                const c = ki / tw;
                const t = ki % tw;
                const slot = &d.cache.slots[slot_idx[ki]];
                for (0..e.n_planes) |i| {
                    const dst_off = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                    const coff = e.cache_off + i * e.plane_extent;
                    try s.cstream.memcpyDtoD(s.d_src.at(dst_off * 4), slot.buf.at(coff * 4), e.plane_extent * 4);
                }
            }
            try s.cstream.record(s.ev_up[ei]);
        }
    } else if (d.pin_up) {
        const hs = s.h_stage.?.ptr;
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..n_clips) |c| {
                for (0..e.n_planes) |i| {
                    const plane = e.planes[i];
                    for (0..tw) |t| {
                        const srcp = win.f[c][t].getReadSlice(plane);
                        const off = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                        @memcpy((hs + off * 4)[0..srcp.len], srcp);
                    }
                }
            }
            try s.cstream.memcpyHtoD(s.d_src.at(e.src_off * 4), hs + e.src_off * 4, e.src_elems * 4);
            try s.cstream.record(s.ev_up[ei]);
        }
    } else {
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..n_clips) |c| {
                for (0..e.n_planes) |i| {
                    const plane = e.planes[i];
                    for (0..tw) |t| {
                        const srcp = win.f[c][t].getReadSlice(plane);
                        const off = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                        try s.cstream.memcpyHtoD(s.d_src.at(off * 4), srcp.ptr, srcp.len);
                    }
                }
            }
            try s.cstream.record(s.ev_up[ei]);
        }
    }

    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        try s.stream.memsetD8(s.d_res.at(e.res_off * 4), 0, e.res_elems * 4);
        try s.stream.waitEvent(s.ev_up[ei]);

        const a_res = s.d_res.at(e.res_off * 4);
        const a_src = s.d_src.at(e.src_off * 4);

        const w: u32 = @intCast(e.key.w);
        const h: u32 = @intCast(e.key.h);
        const bs: u32 = @intCast(e.key.block_step);
        const groups_x = ceilDiv(w, 4 * bs);
        try s.stream.launch(d.fn_bm3d[e.mod_idx], .{
            .grid = .{ ceilDiv(groups_x, d.warps), ceilDiv(h, bs), 1 },
            .block = .{ 32 * d.warps, 1, 1 },
        }, .{ a_res, a_src });

        if (d.radius == 0) {
            const a_dst: cuda.CUdeviceptr = if (d.zc_dst)
                s.h_stage.?.devicePtr((d.stage_down_off + e.dst_off) * 4)
            else
                s.d_dst.at(e.dst_off * 4);
            try s.stream.launch(d.fn_agg[e.mod_idx], .{
                .grid = .{ ceilDiv(w, 32), ceilDiv(h, 8), e.n_planes },
                .block = .{ 32, 8, 1 },
            }, .{ a_dst, a_res });
        }
        try s.stream.record(s.ev_k[ei]);
    }

    try download(d, s, dst);
}

const DlSrc = struct {
    fn dev(dd: *Data, ss: *Stream, e: *const Entry, i: usize, t: usize) cuda.CUdeviceptr {
        return if (dd.normalOut())
            ss.d_dst.at((e.dst_off + i * e.plane_extent) * 4)
        else
            ss.d_res.at((e.res_off + i * t * 2 * e.plane_extent) * 4);
    }
    fn stageOff(dd: *Data, e: *const Entry, i: usize, t: usize) usize {
        return dd.stage_down_off + if (dd.normalOut())
            e.dst_off + i * e.plane_extent
        else
            e.res_off + i * t * 2 * e.plane_extent;
    }
    // Skipped-sigma planes: download would clobber passthrough with garbage/NaN.
    fn skip(e: *const Entry, i: usize) bool {
        return (e.key.proc_mask >> @intCast(i)) & 1 == 0;
    }
};

fn download(d: *Data, s: *Stream, dst: ZFrameW) CreateError!void {
    const tw = d.tw;

    if (d.zc_dst) {
        try s.cstream.sync();
        try s.stream.sync();
    } else if (d.pin_down) {
        try s.stream.sync();
        const hs = s.h_stage.?.ptr;
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..e.n_planes) |i| {
                if (DlSrc.skip(e, i)) continue;
                const off = DlSrc.stageOff(d, e, i, tw);
                const len = dst.getWriteSlice(e.planes[i]).len;
                try s.cstream.memcpyDtoH(hs + off * 4, DlSrc.dev(d, s, e, i, tw), len);
            }
        }
        try s.cstream.sync();
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..e.n_planes) |i| {
                if (DlSrc.skip(e, i)) continue;
                const dstp = dst.getWriteSlice(e.planes[i]);
                @memcpy(dstp, (hs + DlSrc.stageOff(d, e, i, tw) * 4)[0..dstp.len]);
            }
        }
        return;
    } else {
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            try s.cstream.waitEvent(s.ev_k[ei]);
            for (0..e.n_planes) |i| {
                if (DlSrc.skip(e, i)) continue;
                const dstp = dst.getWriteSlice(e.planes[i]);
                try s.cstream.memcpyDtoH(dstp.ptr, DlSrc.dev(d, s, e, i, tw), dstp.len);
            }
        }
        try s.cstream.sync();
        try s.stream.sync();
        return;
    }

    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        for (0..e.n_planes) |i| {
            if (DlSrc.skip(e, i)) continue;
            const dstp = dst.getWriteSlice(e.planes[i]);
            @memcpy(dstp, (s.h_stage.?.ptr + (d.stage_down_off + e.dst_off + i * e.plane_extent) * 4)[0..dstp.len]);
        }
    }
}

// Lock order: source cache → acc cache. Each must fit one worker's whole window.
fn processFused(d: *Data, s: *Stream, fw: *const FusedWin, n: i32, dst: ZFrameW) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();

    const tw = d.tw;
    const r = d.radius;
    const nf: i32 = d.vi.numFrames;
    const n_clips = fw.n_clips;
    const span = fw.cnt;
    const fe = d.cache.frame_elems;

    var skeys: [2 * MAX_SPAN]i64 = undefined;
    var sidx: [2 * MAX_SPAN]usize = undefined;
    var sload: [2 * MAX_SPAN]bool = undefined;
    const nk = n_clips * span;
    for (0..n_clips) |c| {
        for (0..span) |j| {
            skeys[c * span + j] = (@as(i64, @intCast(c)) << 40) | (fw.lo + @as(i64, @intCast(j)));
        }
    }
    d.cache.acquire(skeys[0..nk], sidx[0..nk], sload[0..nk]);

    var akeys: [MAX_TW]i64 = undefined;
    var aidx: [MAX_TW]usize = undefined;
    var aload: [MAX_TW]bool = undefined;
    var centres: [MAX_TW]i32 = undefined;
    var acc_held = false;

    // Drain → abandon/release acc → abandon/release source (never release before drain).
    defer d.cache.release(sidx[0..nk]);
    errdefer for (0..nk) |ki| {
        if (sload[ki]) d.cache.abandon(sidx[ki]);
    };
    defer if (acc_held) d.acc.release(aidx[0..tw]);
    errdefer if (acc_held) for (0..tw) |i| {
        if (aload[i]) d.acc.abandon(aidx[i]);
    };
    errdefer {
        s.cstream.drain();
        s.stream.drain();
    }

    const hs = s.h_stage.?.ptr;
    const stage_cap = @max(d.sum_src / @max(fe, 1), 1);
    var staged: usize = 0;
    for (0..nk) |ki| {
        if (!sload[ki]) continue;
        if (staged == stage_cap) {
            try s.cstream.sync();
            staged = 0;
        }
        const c = ki / span;
        const j = ki % span;
        const slot = &d.cache.slots[sidx[ki]];
        const hbase = staged * fe;
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..e.n_planes) |i| {
                const srcp = fw.f[c][j].getReadSlice(e.planes[i]);
                const coff = e.cache_off + i * e.plane_extent;
                @memcpy((hs + (hbase + coff) * 4)[0..srcp.len], srcp);
                try s.cstream.memcpyHtoD(slot.buf.at(coff * 4), hs + (hbase + coff) * 4, srcp.len);
            }
        }
        try s.cstream.record(slot.ev);
        d.cache.publish(sidx[ki]);
        staged += 1;
    }
    for (0..nk) |ki| {
        try s.cstream.waitEvent(d.cache.slots[sidx[ki]].ev);
    }

    for (0..tw) |i| {
        centres[i] = @min(@max(n - r + @as(i32, @intCast(i)), 0), nf - 1);
        akeys[i] = centres[i];
    }
    d.acc.acquire(akeys[0..tw], aidx[0..tw], aload[0..tw]);
    acc_held = true;

    var prev_ev: ?cu.Event = null;
    for (0..tw) |i| {
        if (!aload[i]) continue;
        const m = centres[i];
        const slot = &d.acc.slots[aidx[i]];

        if (prev_ev) |pe| try s.cstream.waitEvent(pe);

        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..n_clips) |c| {
                for (0..tw) |t| {
                    const fidx = @min(@max(m - r + @as(i32, @intCast(t)), 0), nf - 1);
                    const sk = c * span + @as(usize, @intCast(fidx - fw.lo));
                    const ssl = &d.cache.slots[sidx[sk]];
                    for (0..e.n_planes) |p| {
                        const dst_off = e.src_off + ((c * e.n_planes + p) * tw + t) * e.plane_extent;
                        const coff = e.cache_off + p * e.plane_extent;
                        try s.cstream.memcpyDtoD(
                            s.d_src.at(dst_off * 4),
                            ssl.buf.at(coff * 4),
                            e.plane_extent * 4,
                        );
                    }
                }
            }
            try s.cstream.record(s.ev_up[ei]);
        }

        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            try s.stream.memsetD8(slot.buf.at(e.res_off * 4), 0, e.res_elems * 4);
            try s.stream.waitEvent(s.ev_up[ei]);

            const w: u32 = @intCast(e.key.w);
            const h: u32 = @intCast(e.key.h);
            const bs: u32 = @intCast(e.key.block_step);
            const groups_x = ceilDiv(w, 4 * bs);
            try s.stream.launch(d.fn_bm3d[e.mod_idx], .{
                .grid = .{ ceilDiv(groups_x, d.warps), ceilDiv(h, bs), 1 },
                .block = .{ 32 * d.warps, 1, 1 },
            }, .{ slot.buf.at(e.res_off * 4), s.d_src.at(e.src_off * 4) });
        }
        try s.stream.record(slot.ev);
        d.acc.publish(aidx[i]);
        prev_ev = slot.ev;
    }

    for (0..tw) |i| {
        try s.stream.waitEvent(d.acc.slots[aidx[i]].ev);
    }
    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        var as: AggSrc = .{};
        for (0..tw) |i| {
            as.p[i] = d.acc.slots[aidx[i]].buf.at(e.res_off * 4);
            as.z[i] = @intCast(aggZ(i, n, nf, r));
        }
        const a_dst: cuda.CUdeviceptr = if (d.zc_dst)
            s.h_stage.?.devicePtr((d.stage_down_off + e.dst_off) * 4)
        else
            s.d_dst.at(e.dst_off * 4);

        const w: u32 = @intCast(e.key.w);
        const h: u32 = @intCast(e.key.h);
        try s.stream.launch(d.fn_vagg[ei], .{
            .grid = .{ ceilDiv(w, 32), ceilDiv(h, 8), e.n_planes },
            .block = .{ 32, 8, 1 },
        }, .{ a_dst, as });
        try s.stream.record(s.ev_k[ei]);
    }

    try download(d, s, dst);
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    const r: i32 = d.radius;
    const ni: i32 = @intCast(n);
    const nf: i32 = d.vi.numFrames;

    if (activation_reason == .Initial) {
        const req: i32 = if (d.fused) 2 * r else r;
        var i: i32 = @max(ni - req, 0);
        const end: i32 = @min(ni + req, nf - 1);
        while (i <= end) : (i += 1) {
            zapi.requestFrameFilter(@intCast(i), d.node);
            if (d.ref_node) |rn| zapi.requestFrameFilter(@intCast(i), rn);
        }
    } else if (activation_reason == .AllFramesReady) {
        if (d.fused) {
            var fw: FusedWin = undefined;
            fw.n_clips = if (d.final) 2 else 1;
            fw.lo = @max(ni - 2 * r, 0);
            fw.cnt = @intCast(@min(ni + 2 * r, nf - 1) - fw.lo + 1);

            for (0..fw.cnt) |j| {
                const idx: c_int = @intCast(fw.lo + @as(i32, @intCast(j)));
                if (d.final) {
                    fw.f[0][j] = zapi.initZFrame(d.ref_node, idx);
                    fw.f[1][j] = zapi.initZFrame(d.node, idx);
                } else {
                    fw.f[0][j] = zapi.initZFrame(d.node, idx);
                }
            }
            defer fw.deinit();

            const centre = fw.f[fw.n_clips - 1][@intCast(ni - fw.lo)];
            const dst = centre.newVideoFrame2(d.process);

            const s = d.pool.acquire();
            defer d.pool.release(s);

            processFused(d, s, &fw, ni, dst) catch |err| {
                zapi.setFilterError("BM3Dv2: process frame failed.");
                std.log.err("vszipcu BM3Dv2 process frame failed: {t}", .{err});
                dst.deinit();
                return null;
            };
            return dst.frame;
        }

        var win: Window = undefined;
        win.tw = d.tw;
        win.n_clips = if (d.final) 2 else 1;

        var t: usize = 0;
        while (t < d.tw) : (t += 1) {
            const idx: c_int = @intCast(@min(@max(ni - r + @as(i32, @intCast(t)), 0), nf - 1));
            win.idx[t] = idx;
            if (d.final) {
                win.f[0][t] = zapi.initZFrame(d.ref_node, idx);
                win.f[1][t] = zapi.initZFrame(d.node, idx);
            } else {
                win.f[0][t] = zapi.initZFrame(d.node, idx);
            }
        }
        defer win.deinit();

        const center = win.f[win.n_clips - 1][@intCast(r)];

        const dst = if (d.radius == 0)
            center.newVideoFrame2(d.process)
        else
            center.newVideoFrame3(.{ .height = d.vi_out.height });

        if (d.radius > 0) {
            if (d.zero_init) {
                const np: u32 = @intCast(d.vi.format.numPlanes);
                var p: u32 = 0;
                while (p < np) : (p += 1) {
                    if (!d.process[p]) @memset(dst.getWriteSlice(p), 0);
                }
            }
            const props = dst.getPropertiesRW();
            props.setInt("BM3D_V_radius", d.radius, .Replace);
            props.setIntArray("BM3D_V_process", &[_]i64{
                @intFromBool(d.process[0]), @intFromBool(d.process[1]), @intFromBool(d.process[2]),
            });
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, &win, dst) catch |err| {
            zapi.setFilterError("BM3D: process frame failed.");
            std.log.err("vszipcu BM3D process frame failed: {t}", .{err});
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
    d.cache.deinit();
    d.acc.deinit();
    freeDeviceObjects(d);
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.node);
    if (d.ref_node) |rn| vsapi.?.freeNode.?(rn);
    allocator.destroy(d);
}

fn freeDeviceObjects(d: *Data) void {
    var i: usize = d.n_mods;
    while (i > 0) {
        i -= 1;
        d.mods[i].deinit();
        d.mods[i] = .{};
    }
    d.n_mods = 0;
    var v: usize = d.n_vagg;
    while (v > 0) {
        v -= 1;
        d.vagg_mods[v].deinit();
        d.vagg_mods[v] = .{};
    }
    d.n_vagg = 0;
}

const AccShort = struct { need_mib: usize, slots: usize, slot_mib: usize, free_mib: usize };

fn initCuda(d: *Data, device_id: i32, num_streams: usize, acc_short: *?AccShort) CreateError!*Data {
    d.dev = try cu.Device.init(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    errdefer freeDeviceObjects(d);

    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        var mi: usize = 0;
        while (mi < d.n_mods) : (mi += 1) {
            if (std.meta.eql(d.entries[mi].key, e.key)) break;
        }
        if (mi < d.n_mods) {
            e.mod_idx = mi;
            continue;
        }

        const extractor = d.extractor;
        // `{x}` = hexfloat; decimal f32 round-trip is a parity bug.
        const defines = std.fmt.allocPrint(allocator,
            \\#define WIDTH {d}
            \\#define HEIGHT {d}
            \\#define STRIDE {d}
            \\#define SIGMA_Y {x}f
            \\#define SIGMA_U {x}f
            \\#define SIGMA_V {x}f
            \\#define BLOCK_STEP {d}
            \\#define BM_RANGE {d}
            \\#define RADIUS {d}
            \\#define PS_NUM {d}
            \\#define PS_RANGE {d}
            \\#define TEMPORAL {d}
            \\#define CHROMA {d}
            \\#define FINAL {d}
            \\#define EXTRACTOR {x}f
            \\#define WARPS {d}
            \\#define PROC_MASK {d}
            \\#define BM_ERROR {s}
            \\#define TRANSFORM_2D {s}
            \\#define TRANSFORM_1D {s}
            \\
        , .{
            e.key.w,                  e.key.h,               e.key.stride,
            e.key.sigma[0],           e.key.sigma[1],        e.key.sigma[2],
            e.key.block_step,         e.key.bm_range,        d.radius,
            e.key.ps_num,             e.key.ps_range,        @intFromBool(d.radius > 0),
            @intFromBool(d.chroma),   @intFromBool(d.final), extractor,
            d.warps,                  e.key.proc_mask,
            @tagName(e.key.bm_error), @tagName(e.key.t2d),   @tagName(e.key.t1d),
        }) catch return error.OutOfMemory;
        defer allocator.free(defines);

        d.mods[d.n_mods] = try cu.compile(d.dev, .{
            .text = kernel_source,
            .defines = defines,
            .name = "bm3d.cu",
        }, nvrtc_opts);
        e.mod_idx = d.n_mods;
        d.n_mods += 1;
        d.fn_bm3d[e.mod_idx] = try d.mods[e.mod_idx].function("bm3d");
        d.fn_agg[e.mod_idx] = try d.mods[e.mod_idx].function("aggregate");
    }

    if (d.fused) {
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            const defines = std.fmt.allocPrint(allocator,
                \\#define WIDTH {d}
                \\#define HEIGHT {d}
                \\#define STRIDE {d}
                \\#define RADIUS {d}
                \\#define PROC_MASK {d}
                \\
            , .{ e.key.w, e.key.h, e.key.stride, d.radius, e.key.proc_mask }) catch return error.OutOfMemory;
            defer allocator.free(defines);

            d.vagg_mods[ei] = try cu.compile(d.dev, .{
                .text = vagg_source,
                .defines = defines,
                .name = "bm3d_vagg.cu",
            }, vagg_opts);
            d.n_vagg += 1;
            d.fn_vagg[ei] = try d.vagg_mods[ei].function("vaggregate");
        }
    }

    const data = allocator.create(Data) catch return error.OutOfMemory;
    errdefer allocator.destroy(data);
    data.* = d.*;

    errdefer data.cache.deinit();
    if (d.scache) {
        // Deadlock-free min: one worker's whole window (fused source = 4r+1).
        const win_frames = if (d.fused) 2 * d.tw - 1 else d.tw;
        const min_slots = d.n_clips * win_frames;
        const want = min_slots + num_streams + 2 * @as(usize, @intCast(d.radius));
        data.cache = .{ .frame_elems = d.cache.frame_elems };
        data.cache.slots = allocator.alloc(CacheSlot, want) catch return error.OutOfMemory;
        for (data.cache.slots) |*sl| sl.* = .{};
        var made: usize = 0;
        for (data.cache.slots) |*sl| {
            sl.buf = cu.DeviceBuffer.alloc(data.cache.frame_elems * 4) catch |err| {
                if (made >= min_slots and (err == error.OutOfDeviceMemory)) break;
                return err;
            };
            sl.ev = try cu.Event.init();
            made += 1;
        }
        if (made < data.cache.slots.len) {
            data.cache.slots = allocator.realloc(data.cache.slots, made) catch data.cache.slots[0..made];
        }
    }

    data.pool = .{};
    data.pool.prime(data, num_streams) catch {
        data.pool.deinit();
        return error.OutOfMemory;
    };
    data.pool.prewarm(num_streams) catch |err| {
        data.pool.deinit();
        return err;
    };

    errdefer data.acc.deinit();
    if (d.fused) {
        errdefer data.pool.deinit();

        // Deadlock-free min: one worker's (2r+1) acc window.
        const min_slots = d.tw;
        const want = d.tw + num_streams + 2 * @as(usize, @intCast(d.radius));
        const slot_bytes = d.sum_res * 4;

        const mem = try cu.memInfo();
        const margin: usize = 256 << 20;
        const budget = if (mem.free > margin) mem.free - margin else 0;
        var cap = budget / @max(slot_bytes, 1);
        cap = @min(cap, want);

        const short: AccShort = .{
            .need_mib = (min_slots * slot_bytes) >> 20,
            .slots = min_slots,
            .slot_mib = slot_bytes >> 20,
            .free_mib = mem.free >> 20,
        };

        if (cap < min_slots) {
            acc_short.* = short;
            return error.OutOfDeviceMemory;
        }

        data.acc = .{ .frame_elems = d.sum_res };
        data.acc.slots = allocator.alloc(CacheSlot, cap) catch return error.OutOfMemory;
        for (data.acc.slots) |*sl| sl.* = .{};
        var made: usize = 0;
        for (data.acc.slots) |*sl| {
            sl.buf = cu.DeviceBuffer.alloc(slot_bytes) catch |err| {
                if (err != error.OutOfDeviceMemory) return err;
                if (made >= min_slots) break;
                acc_short.* = short;
                return err;
            };
            sl.ev = try cu.Event.init();
            made += 1;
        }
        if (made < data.acc.slots.len) {
            data.acc.slots = allocator.realloc(data.acc.slots, made) catch data.acc.slots[0..made];
        }
        std.log.info("vszipcu BM3Dv2: FUSED, {d} accumulator slots, {d} MiB{s}", .{
            made,
            (made * slot_bytes) >> 20,
            if (made <= min_slots)
                " (at the minimum: concurrent frames will serialise on the cache)"
            else
                "",
        });
    }
    return data;
}

fn perPlane(comptime T: type, map_in: anytype, comptime key: [:0]const u8, base: T) [3]T {
    var v: [3]T = undefined;
    for (0..3) |i| {
        if (map_in.getValue2(T, key, i)) |given| {
            v[i] = given;
        } else {
            v[i] = if (i == 0) base else v[i - 1];
        }
    }
    return v;
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createInner(in, out, core, vsapi, false);
}

fn createInner(
    in: ?*const vs.Map,
    out: ?*vs.Map,
    core: ?*vs.Core,
    vsapi: ?*const vs.API,
    fused_req: bool,
) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;

    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        if (d.ref_node) |rn| zapi.freeNode(rn);
    };

    const fmt = d.vi.format;
    if (d.vi.width <= 0 or d.vi.height <= 0 or fmt.sampleType != .Float or fmt.bitsPerSample != 32) {
        return map_out.setError("BM3D: only constant format 32 bit float input supported.");
    }

    if (map_in.getNode("ref")) |rn| {
        d.ref_node = rn;
        const rvi = zapi.getVideoInfo(rn);
        if (rvi.format.colorFamily != fmt.colorFamily or rvi.format.sampleType != fmt.sampleType or
            rvi.format.bitsPerSample != fmt.bitsPerSample or rvi.format.subSamplingW != fmt.subSamplingW or
            rvi.format.subSamplingH != fmt.subSamplingH)
        {
            return map_out.setError("BM3D: \"ref\" must be of the same format as \"clip\".");
        }
        if (rvi.width != d.vi.width or rvi.height != d.vi.height) {
            return map_out.setError("BM3D: \"ref\" must be of the same dimensions as \"clip\".");
        }
        if (rvi.numFrames != d.vi.numFrames) {
            return map_out.setError("BM3D: \"ref\" must be of the same number of frames as \"clip\".");
        }
        d.final = true;
    }

    var sigma = perPlane(f32, map_in, "sigma", 3.0);
    for (sigma) |sv| {
        if (!(sv >= 0)) return map_out.setError("BM3D: \"sigma\" must be non-negative.");
    }
    for (0..3) |i| d.process[i] = !(sigma[i] < FLT_EPSILON);

    // Exact f32 of stepwise C++ product (comptime fold is 1 ULP off).
    const sigma_factor: f32 = if (d.final)
        @bitCast(@as(u32, 0x3e40c0c1))
    else
        @bitCast(@as(u32, 0x3f021bb6));
    for (&sigma) |*sv| sv.* *= sigma_factor;

    const block_step = perPlane(i32, map_in, "block_step", 8);
    for (block_step) |v| {
        if (v <= 0 or v > 8) return map_out.setError("BM3D: \"block_step\" must be in range [1, 8].");
    }
    const bm_range = perPlane(i32, map_in, "bm_range", 9);
    for (bm_range) |v| {
        if (v <= 0) return map_out.setError("BM3D: \"bm_range\" must be positive.");
    }
    const ps_num = perPlane(i32, map_in, "ps_num", 2);
    for (ps_num) |v| {
        if (v <= 0 or v > 8) return map_out.setError("BM3D: \"ps_num\" must be in range [1, 8].");
    }
    const ps_range = perPlane(i32, map_in, "ps_range", 4);
    for (ps_range) |v| {
        if (v <= 0) return map_out.setError("BM3D: \"ps_range\" must be positive.");
    }

    d.radius = map_in.getValue(i32, "radius") orelse 0;
    if (d.radius < 0) return map_out.setError("BM3D: \"radius\" must be non-negative.");
    if (d.radius > MAX_RADIUS) return map_out.setError("BM3D: \"radius\" must be <= 16.");
    d.tw = @intCast(2 * d.radius + 1);
    d.fused = fused_req and d.radius > 0;

    d.chroma = (map_in.getValue(i32, "chroma") orelse 0) != 0;
    if (d.chroma and (fmt.colorFamily != .YUV or fmt.subSamplingW != 0 or fmt.subSamplingH != 0)) {
        return map_out.setError("BM3D: clip format must be YUV444 when \"chroma\" is true.");
    }

    d.zero_init = (map_in.getValue(i32, "zero_init") orelse 1) != 0;

    const extractor_exp = map_in.getValue(i32, "extractor_exp") orelse 0;
    d.extractor = if (extractor_exp != 0) math.ldexp(@as(f32, 1.0), extractor_exp) else 0.0;

    var bm_error: [3]BmError = .{ .ssd, .ssd, .ssd };
    var t2d: [3]Transform = .{ .dct, .dct, .dct };
    var t1d: [3]Transform = .{ .dct, .dct, .dct };
    for (0..3) |i| {
        const idx: i32 = @intCast(i);
        if (map_in.getData("bm_error_s", idx)) |sv| {
            bm_error[i] = BmError.parse(sv) orelse return map_out.setError("BM3D: invalid \"bm_error_s\".");
        } else if (i > 0) bm_error[i] = bm_error[i - 1];
        if (map_in.getData("transform_2d_s", idx)) |sv| {
            t2d[i] = Transform.parse(sv) orelse return map_out.setError("BM3D: invalid \"transform_2d_s\".");
        } else if (i > 0) t2d[i] = t2d[i - 1];
        if (map_in.getData("transform_1d_s", idx)) |sv| {
            t1d[i] = Transform.parse(sv) orelse return map_out.setError("BM3D: invalid \"transform_1d_s\".");
        } else if (i > 0) t1d[i] = t1d[i - 1];
    }

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("BM3D: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("BM3D: num_streams must be 1..32.");
    };
    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 4;

    const num_planes: usize = @intCast(fmt.numPlanes);
    var any_proc = false;
    for (0..num_planes) |i| {
        if (d.process[i]) any_proc = true;
    }
    if (!any_proc) return map_out.setError("BM3D: all planes have sigma < FLT_EPSILON (nothing to process).");

    const strides = vsutil.strideFromVi(d.vi);
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);

    d.n_entries = 0;
    var sum_src: usize = 0;
    var sum_res: usize = 0;
    var sum_dst: usize = 0;
    var cache_elems: usize = 0;

    if (d.chroma) {
        const w: i32 = d.vi.width;
        const h: i32 = d.vi.height;
        const stride: i32 = @intCast(strides[0]);
        const pe: usize = @as(usize, @intCast(stride)) * @as(usize, @intCast(h));
        var pm: u32 = 0;
        for (0..3) |i| pm |= @as(u32, @intFromBool(d.process[i])) << @intCast(i);
        d.entries[0] = .{
            .key = .{
                .w = w,
                .h = h,
                .stride = stride,
                .sigma = sigma,
                .block_step = block_step[0],
                .bm_range = bm_range[0],
                .ps_num = ps_num[0],
                .ps_range = ps_range[0],
                .proc_mask = pm,
                .bm_error = bm_error[0],
                .t2d = t2d[0],
                .t1d = t1d[0],
            },
            .mod_idx = 0,
            .n_planes = 3,
            .planes = .{ 0, 1, 2 },
            .plane_extent = pe,
            .cache_off = 0,
            .src_off = 0,
            .src_elems = (if (d.final) @as(usize, 2) else 1) * 3 * d.tw * pe,
            .res_off = 0,
            .res_elems = 3 * d.tw * 2 * pe,
            .dst_off = 0,
            .dst_elems = if (d.normalOut()) 3 * pe else 0,
        };
        sum_src = d.entries[0].src_elems;
        sum_res = d.entries[0].res_elems;
        sum_dst = d.entries[0].dst_elems;
        cache_elems = 3 * pe;
        d.n_entries = 1;
    } else {
        for (0..num_planes) |p| {
            if (!d.process[p]) continue;
            const w: i32 = if (p == 0) d.vi.width else d.vi.width >> subW;
            const h: i32 = if (p == 0) d.vi.height else d.vi.height >> subH;
            const stride: i32 = @intCast(if (p == 0) strides[0] else strides[1]);
            if (w < 8 or h < 8) return map_out.setError("BM3D: every processed plane must be at least 8x8.");
            const pe: usize = @as(usize, @intCast(stride)) * @as(usize, @intCast(h));
            const ei = d.n_entries;
            d.entries[ei] = .{
                .key = .{
                    .w = w,
                    .h = h,
                    .stride = stride,
                    .sigma = .{ sigma[p], sigma[p], sigma[p] },
                    .block_step = block_step[p],
                    .bm_range = bm_range[p],
                    .ps_num = ps_num[p],
                    .ps_range = ps_range[p],
                    .proc_mask = 1,
                    .bm_error = bm_error[p],
                    .t2d = t2d[p],
                    .t1d = t1d[p],
                },
                .mod_idx = 0,
                .n_planes = 1,
                .planes = .{ @intCast(p), 0, 0 },
                .plane_extent = pe,
                .cache_off = cache_elems,
                .src_off = sum_src,
                .src_elems = (if (d.final) @as(usize, 2) else 1) * d.tw * pe,
                .res_off = sum_res,
                .res_elems = d.tw * 2 * pe,
                .dst_off = sum_dst,
                .dst_elems = if (d.normalOut()) pe else 0,
            };
            sum_src += d.entries[ei].src_elems;
            sum_res += d.entries[ei].res_elems;
            sum_dst += d.entries[ei].dst_elems;
            cache_elems += pe;
            d.n_entries += 1;
        }
    }
    if (d.chroma and (d.vi.width < 8 or d.vi.height < 8)) {
        return map_out.setError("BM3D: every processed plane must be at least 8x8.");
    }
    d.sum_src = sum_src;
    d.sum_res = sum_res;
    d.sum_dst = sum_dst;
    d.cache.frame_elems = cache_elems;

    // Kernel uses int32 indices; stacked planes must fit u32 slice lengths.
    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        if (e.src_elems >= (1 << 31) or e.res_elems >= (1 << 31)) {
            return map_out.setError("BM3D: frame/radius too large (a device region exceeds 2^31 samples).");
        }
        if (e.plane_extent * d.tw * 2 * 4 >= (1 << 32)) {
            return map_out.setError("BM3D: stacked output plane too large (exceeds 4 GiB).");
        }
        const gy = ceilDiv(@intCast(e.key.h), @intCast(e.key.block_step));
        if (gy > 65535) return map_out.setError("BM3D: frame too tall for the CUDA launch grid.");
    }

    d.pin_up = num_streams >= 2;
    d.pin_down = num_streams >= 2 and !d.normalOut();
    d.zc_dst = num_streams >= 2 and d.normalOut();
    d.zc_dst = d.zc_dst and d.normalOut();
    if (d.zc_dst) d.pin_down = false;

    d.scache = d.radius > 0;
    if (d.fused) d.scache = true; // fused gathers from source cache; required

    if (d.scache) d.pin_up = true;
    d.n_clips = if (d.final) 2 else 1;

    {
        const down_span: usize = if (d.zc_dst or d.pin_down)
            (if (d.normalOut()) d.sum_dst else d.sum_res)
        else
            0;
        d.stage_down_off = if (d.pin_up) d.sum_src else 0;
        d.stage_elems = (if (d.pin_up) d.sum_src else 0) + down_span;
    }

    d.vi_out = d.vi.*;
    if (d.radius > 0 and !d.fused) {
        const mult: i64 = 2 * (2 * @as(i64, d.radius) + 1);
        if (@as(i64, d.vi.height) * mult > math.maxInt(i32)) {
            return map_out.setError("BM3D: clip too tall for the stacked output.");
        }
        d.vi_out.height = @intCast(@as(i64, d.vi.height) * mult);
    }

    var acc_short: ?AccShort = null;
    const data = initCuda(&d, device_id, num_streams, &acc_short) catch |err| {
        if (acc_short) |s| {
            var buf: [320]u8 = undefined;
            const msg: [:0]const u8 = std.fmt.bufPrintZ(
                &buf,
                "BM3Dv2: fast_fused needs at least {d} MiB of accumulator cache ({d} slots x " ++
                    "{d} MiB) but only {d} MiB of device memory is free. Use fast_fused=False " ++
                    "(composed BM3D + VAggregate, much less VRAM), or lower num_streams/radius.",
                .{ s.need_mib, s.slots, s.slot_mib, s.free_mib },
            ) catch "BM3Dv2: fast_fused: not enough device memory for the accumulator cache. Use fast_fused=False.";
            map_out.setError(msg);
            return;
        }
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "BM3D: invalid device ID.",
            error.Nvrtc => "BM3D: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "BM3D: could not locate NVRTC (put nvrtc64_130_0.dll next to the plugin, or: pip install nvidia-cuda-nvrtc).",
            error.OutOfDeviceMemory => "BM3D: out of device memory.",
            else => "BM3D: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu BM3D init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep_buf: [2]vs.FilterDependency = undefined;
    var ndeps: usize = 0;
    const rp: vs.RequestPattern = if (d.radius > 0) .General else .StrictSpatial;
    dep_buf[ndeps] = .{ .source = d.node, .requestPattern = rp };
    ndeps += 1;
    if (d.ref_node) |rn| {
        dep_buf[ndeps] = .{ .source = rn, .requestPattern = rp };
        ndeps += 1;
    }
    const name = if (d.fused) "BM3Dv2" else "BM3D";
    zapi.createVideoFilter(out, name, &data.vi_out, getFrame, free, .Parallel, dep_buf[0..ndeps], data);
}

const V = std.simd.suggestVectorLength(f32) orelse 8;
const Vec = @Vector(V, f32);

const AggData = struct {
    node: ?*vs.Node = null,
    src_node: ?*vs.Node = null,
    src_vi: *const vs.VideoInfo = undefined,
    radius: i32 = 0,
    process: [3]bool = .{ false, false, false },
};

fn aggPlane(dstp: []f32, srcps: []const []const f32, w: usize, h: usize, stride: usize, radius: i32, n: i32, nframes: i32) void {
    const tw: usize = @intCast(2 * radius + 1);
    for (0..h) |y| {
        var rows: [MAX_TW][]const f32 = undefined;
        var wts: [MAX_TW][]const f32 = undefined;
        for (0..tw) |i| {
            const z = aggZ(i, n, nframes, radius);
            const base = (@as(usize, @intCast(z)) * 2 * h + y) * stride;
            rows[i] = srcps[i][base..][0..w];
            wts[i] = srcps[i][base + h * stride ..][0..w];
        }

        const drow = dstp[y * stride ..][0..w];
        var x: usize = 0;
        while (x + V <= w) : (x += V) {
            var acc: Vec = @splat(0);
            var acw: Vec = @splat(0);
            for (0..tw) |i| {
                acc += @as(Vec, rows[i][x..][0..V].*);
                acw += @as(Vec, wts[i][x..][0..V].*);
            }
            drow[x..][0..V].* = acc / acw;
        }
        while (x < w) : (x += 1) {
            var acc: f32 = 0;
            var acw: f32 = 0;
            for (0..tw) |i| {
                acc += rows[i][x];
                acw += wts[i][x];
            }
            drow[x] = acc / acw;
        }
    }
}

fn aggGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *AggData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    const r = d.radius;
    const ni: i32 = @intCast(n);
    const nf: i32 = d.src_vi.numFrames;

    if (activation_reason == .Initial) {
        var i: i32 = @max(ni - r, 0);
        const end: i32 = @min(ni + r, nf - 1);
        while (i <= end) : (i += 1) zapi.requestFrameFilter(@intCast(i), d.node);
        zapi.requestFrameFilter(n, d.src_node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.src_node, n);
        defer src.deinit();

        const tw: usize = @intCast(2 * r + 1);
        var stack: [MAX_TW]ZFrame = undefined;
        for (0..tw) |i| {
            const idx: c_int = @intCast(@min(@max(ni - r + @as(i32, @intCast(i)), 0), nf - 1));
            stack[i] = zapi.initZFrame(d.node, idx);
        }
        defer for (0..tw) |i| stack[i].deinit();

        const dst = src.newVideoFrame2(d.process);

        const num_planes: u32 = @intCast(d.src_vi.format.numPlanes);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const w, const h, const stride_b = src.getDimensions(p);
            const stride: usize = stride_b / 4;

            var srcps: [MAX_TW][]const f32 = undefined;
            for (0..tw) |i| srcps[i] = stack[i].getReadSlice2(f32, p);

            aggPlane(dst.getWriteSlice2(f32, p), srcps[0..tw], w, h, stride, r, ni, nf);
        }

        return dst.frame;
    }

    return null;
}

fn aggFree(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *AggData = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.src_node);
    allocator.destroy(d);
}

pub fn createVAggregate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: AggData = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        zapi.freeNode(d.src_node);
    };

    const node, const vi = map_in.getNodeVi("clip").?;
    d.node = node;
    const src_node, const src_vi = map_in.getNodeVi("src").?;
    d.src_node = src_node;
    d.src_vi = src_vi;

    if (src_vi.height <= 0 or src_vi.format.sampleType != .Float or src_vi.format.bitsPerSample != 32) {
        return map_out.setError("VAggregate: \"src\" must be 32 bit float.");
    }
    if (vi.format.colorFamily != src_vi.format.colorFamily or
        vi.format.sampleType != src_vi.format.sampleType or
        vi.format.bitsPerSample != src_vi.format.bitsPerSample or
        vi.format.subSamplingW != src_vi.format.subSamplingW or
        vi.format.subSamplingH != src_vi.format.subSamplingH or
        vi.numFrames != src_vi.numFrames)
    {
        return map_out.setError("VAggregate: \"clip\" must have the same format and frame count as \"src\".");
    }
    const ratio = @divTrunc(vi.height, src_vi.height);
    if (ratio < 6 or @rem(ratio - 2, 4) != 0 or @rem(vi.height, src_vi.height) != 0 or
        vi.width != src_vi.width)
    {
        return map_out.setError("VAggregate: \"clip\" is not a BM3D stacked clip of \"src\" (radius >= 1).");
    }
    d.radius = @divTrunc(ratio - 2, 4);
    if (d.radius > MAX_RADIUS) return map_out.setError("VAggregate: radius too large.");

    var i: usize = 0;
    while (map_in.getValue2(i32, "planes", i)) |pl| : (i += 1) {
        if (pl < 0 or pl > 2) return map_out.setError("VAggregate: \"planes\" must be 0..2.");
        d.process[@intCast(pl)] = true;
    }
    if (i == 0) return map_out.setError("VAggregate: \"planes\" is required.");

    const data = allocator.create(AggData) catch {
        return map_out.setError("VAggregate: out of memory.");
    };
    data.* = d;
    keep = true;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .General },
        .{ .source = d.src_node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "VAggregate", d.src_vi, aggGetFrame, aggFree, .Parallel, &deps, data);
}

pub fn createV2(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    var proc: [3]bool = .{ true, true, true };
    for (0..3) |i| {
        if (map_in.getValue2(f32, "sigma", i)) |sv| {
            proc[i] = !(sv < FLT_EPSILON);
        } else if (i > 0) {
            proc[i] = proc[i - 1];
        }
    }

    const src, const src_vi = map_in.getNodeVi("clip").?;
    defer zapi.freeNode(src);

    var skip = true;
    for (0..@intCast(src_vi.format.numPlanes)) |i| {
        if (proc[i]) skip = false;
    }
    if (skip) {
        _ = map_out.setNode("clip", src, .Replace);
        return;
    }

    const plugin = zapi.getPluginByID("com.julek.vszipcu") orelse {
        return map_out.setError("BM3Dv2: could not find the vszipcu plugin.");
    };

    const radius = map_in.getValue(i32, "radius") orelse 0;

    // fast_fused: never silent-downgrade if requested.
    if (radius > 0 and (map_in.getValue(i32, "fast_fused") orelse 0) != 0) {
        createInner(in, out, core, vsapi, true);
        return;
    }

    const bm3d_ret = map_in.invoke(plugin, "BM3D");
    defer bm3d_ret.free();
    if (bm3d_ret.getError()) |msg| {
        return map_out.setError(msg);
    }

    if (radius == 0) {
        const node = bm3d_ret.getNode("clip");
        defer zapi.freeNode(node);
        _ = map_out.setNode("clip", node, .Replace);
        return;
    }

    const agg_args = zapi.createZMap();
    defer agg_args.free();
    const bm3d_node = bm3d_ret.getNode("clip");
    _ = agg_args.consumeNode("clip", bm3d_node, .Replace);
    _ = agg_args.setNode("src", src, .Replace);
    for (0..3) |i| {
        if (proc[i]) agg_args.setInt("planes", @intCast(i), .Append);
    }

    const agg_ret = agg_args.invoke(plugin, "VAggregate");
    defer agg_ret.free();
    if (agg_ret.getError()) |msg| {
        return map_out.setError(msg);
    }

    const node = agg_ret.getNode("clip");
    defer zapi.freeNode(node);
    _ = map_out.setNode("clip", node, .Replace);
}
