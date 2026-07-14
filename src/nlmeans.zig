const std = @import("std");
const framecache = @import("framecache.zig");
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

const kernel_source = @embedFile("nlmeans.cu");

const REF_LUMA: u8 = 0;
const REF_CHROMA: u8 = 1;
const REF_YUV: u8 = 2;
const REF_RGB: u8 = 3;

const BLK_X: u32 = 16;
const BLK_Y: u32 = 8;
const VRT: u32 = 6;
const nlm_qb_small: u32 = 16;
const nlm_qb_large: u32 = 4;

// No -use_fast_math (expf/fdimf correctly-rounded libdevice).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{"-std=c++17"},
    .log_name = "NLMeans",
};

const Variant = struct {
    w_base: u32,
    q_base: u32,
    q_cnt: u32,
    w_boff: []u32,
};

const Data = struct {
    node: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,

    ref_node: ?*vs.Node = null,
    has_ref: bool = false,

    d: u8 = 0,
    a: u8 = 0,
    s: u8 = 0,
    h: f32 = 0,
    wref: f32 = 0,
    wmode: u8 = 0,

    ref: u8 = REF_LUMA,
    chans: u8 = 1,
    plane0: u8 = 0,

    bits: i32 = 32,
    half: bool = false,
    wbytes: u32 = 4,

    w: u32 = 0,
    h_: u32 = 0,
    stride: u32 = 0,
    pad: u32 = 0,
    pstride: u32 = 0,
    ph: u32 = 0,

    use_pinned: bool = false,
    zc_dst: bool = false,
    qb: u32 = nlm_qb_small,

    wq_host: []i32 = &.{},
    aq_host: []i32 = &.{},
    variants: []Variant = &.{},

    use_cache: bool = false,
    cache: framecache.FrameCache = .{},
    slot_bytes: usize = 0,

    dev: cu.Device = .{},
    module: cu.Module = .{},
    fn_weight: cu.Function = .{},
    fn_acc: cu.Function = .{},
    fn_finish: cu.Function = .{},
    d_wq: cu.DeviceBuffer = .{},
    d_aq: cu.DeviceBuffer = .{},

    pool: pool_mod.Pool(Stream, Data) = .{},
};

fn freeTables(d: *Data) void {
    for (d.variants) |v| allocator.free(v.w_boff);
    allocator.free(d.variants);
    allocator.free(d.aq_host);
    allocator.free(d.wq_host);
    d.variants = &.{};
    d.aq_host = &.{};
    d.wq_host = &.{};
}

const Stream = struct {
    stream: cu.Stream,
    cstream: cu.Stream,
    ev_up: cu.Event,
    ev_k: cu.Event,
    n_ev: usize,
    d_u1: cu.DeviceBuffer,
    d_u1r: cu.DeviceBuffer,
    d_u1z: cu.DeviceBuffer,
    d_u2: cu.DeviceBuffer,
    d_u4a: cu.DeviceBuffer,
    d_u5: cu.DeviceBuffer,
    h_stage: ?cu.HostBuffer,
    h_stage_r: ?cu.HostBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        const layers: usize = 2 * @as(usize, d.d) + 1;
        const c: usize = d.chans;
        const wb: usize = d.wbytes;
        const u1_bytes = @as(usize, d.pstride) * d.ph * layers * c * wb;
        const npix = @as(usize, d.stride) * d.h_;
        const slots: usize = if (d.d == 0) d.qb else 2 * @as(usize, d.qb);

        self.d_u1 = try cu.DeviceBuffer.alloc(u1_bytes);
        errdefer self.d_u1.deinit();
        self.d_u1r = .{};
        if (d.has_ref) self.d_u1r = try cu.DeviceBuffer.alloc(u1_bytes);
        errdefer self.d_u1r.deinit();
        self.d_u1z = try cu.DeviceBuffer.alloc(npix * c * wb);
        errdefer self.d_u1z.deinit();
        self.d_u2 = try cu.DeviceBuffer.alloc(npix * (c + 1) * 4);
        errdefer self.d_u2.deinit();
        self.d_u4a = try cu.DeviceBuffer.alloc(npix * slots * 4);
        errdefer self.d_u4a.deinit();
        self.d_u5 = try cu.DeviceBuffer.alloc(npix * 4);
        errdefer self.d_u5.deinit();

        self.h_stage = null;
        self.h_stage_r = null;
        if (d.use_pinned) {
            if (cu.HostBuffer.alloc(u1_bytes + npix * c * wb, .{})) |hb| {
                self.h_stage = hb;
                @memset(hb.slice(), 0);
            } else |_| {
                std.log.warn("NLMeans: pinned staging alloc failed ({d} MB); falling back to pageable transfers.", .{u1_bytes / (1 << 20)});
            }
            if (d.has_ref and self.h_stage != null) {
                if (cu.HostBuffer.alloc(u1_bytes, .{})) |hb| {
                    self.h_stage_r = hb;
                    @memset(hb.slice(), 0);
                } else |_| {}
            }
        }
        errdefer if (self.h_stage) |hp| hp.deinit();
        errdefer if (self.h_stage_r) |hp| hp.deinit();

        self.n_ev = 0;
        errdefer self.destroyEvents();
        self.ev_up = try cu.Event.init();
        self.n_ev = 1;
        self.ev_k = try cu.Event.init();
        self.n_ev = 2;
        self.stream = try cu.Stream.init();
        errdefer self.stream.deinit();
        self.cstream = try cu.Stream.init();
        errdefer self.cstream.deinit();

        try self.d_u1.fill(0, u1_bytes);
        if (self.d_u1r.ptr != 0) try self.d_u1r.fill(0, u1_bytes);

        // m=0 leaves part of d_u4a unwritten; unwritten slots must read 0.
        try self.d_u1z.fill(0, npix * c * wb);
        try self.d_u2.fill(0, npix * (c + 1) * 4);
        try self.d_u4a.fill(0, npix * slots * 4);
        try self.d_u5.fill(0, npix * 4);
    }

    fn destroyEvents(self: *Stream) void {
        if (self.n_ev >= 2) self.ev_k.deinit();
        if (self.n_ev >= 1) self.ev_up.deinit();
        self.n_ev = 0;
    }

    pub fn deinit(self: *Stream) void {
        self.stream.deinit();
        self.cstream.deinit();
        self.destroyEvents();
        if (self.h_stage_r) |hp| hp.deinit();
        if (self.h_stage) |hp| hp.deinit();
        self.d_u5.deinit();
        self.d_u4a.deinit();
        self.d_u2.deinit();
        self.d_u1z.deinit();
        self.d_u1r.deinit();
        self.d_u1.deinit();
    }
};

fn compileModule(d: *const Data) CreateError!cu.Module {
    // Past 2^31 samples kernels need 64-bit indices.
    const layers: u64 = 2 * @as(u64, d.d) + 1;
    const slots: u64 = if (d.d == 0) d.qb else 2 * @as(u64, d.qb);
    const npix64 = @as(u64, d.stride) * @as(u64, d.h_);
    const idx_max = @max(
        @as(u64, d.pstride) * @as(u64, d.ph) * layers * @as(u64, d.chans),
        @max(npix64 * slots, npix64 * @as(u64, d.chans)),
    );
    const idx_long: u8 = @intFromBool(idx_max >= (1 << 31));

    // `{x}` = hexfloat; decimal f32 round-trip is a parity bug.
    const defines = std.fmt.allocPrint(allocator,
        \\#define VI_DIM_X {d}
        \\#define VI_DIM_Y {d}
        \\#define STRIDE {d}
        \\#define PSTRIDE {d}
        \\#define PAD {d}
        \\#define PH {d}
        \\#define NLM_S {d}
        \\#define NLM_D {d}
        \\#define NLM_REF {d}
        \\#define NLM_CHANNELS {d}
        \\#define WMODE {d}
        \\#define BLK_X {d}
        \\#define BLK_Y {d}
        \\#define VRT_RESULT {d}
        \\#define NLM_H {x}f
        \\#define NLM_WREF {x}f
        \\#define IDX_LONG {d}
        \\#define BITS {d}
        \\#define HALF {d}
        \\
    , .{
        d.w,      d.h_,    d.stride,             d.pstride, d.pad, d.ph, d.s, d.d,
        d.ref,    d.chans, d.wmode,              BLK_X,     BLK_Y, VRT,  d.h, d.wref,
        idx_long, d.bits,  @intFromBool(d.half),
    }) catch return error.OutOfMemory;
    defer allocator.free(defines);

    return cu.compile(d.dev, .{
        .text = kernel_source,
        .defines = defines,
        .name = "nlmeans.cu",
    }, nvrtc_opts);
}

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;

fn uploadWindow(d: *Data, s: *Stream, buf: cu.DeviceBuffer, host_opt: ?cu.HostBuffer, srcps: []const []const u8, k_start: i32, k_end: i32) CreateError!void {
    const C: usize = d.chans;
    const center: i32 = @intCast(d.d);
    const layers: usize = 2 * @as(usize, d.d) + 1;
    const wb: usize = d.wbytes;
    const pp: usize = d.pstride;
    const lay: usize = pp * @as(usize, d.ph);
    const pad: usize = d.pad;
    const w: usize = d.w;
    var fi: usize = 0;
    var k: i32 = k_start;
    while (k <= k_end) : (k += 1) {
        const t_layer: usize = @intCast(center + k);
        var c: usize = 0;
        while (c < C) : (c += 1) {
            const src = srcps[fi * C + c];
            std.debug.assert(src.len == @as(usize, d.stride) * d.h_ * wb);
            if (host_opt) |hb| {
                const host = hb.ptr;
                const base = ((c * layers + t_layer) * lay + pad * pp + pad) * wb;
                var y: usize = 0;
                while (y < d.h_) : (y += 1) {
                    @memcpy(host[base + y * pp * wb ..][0 .. w * wb], src[y * @as(usize, d.stride) * wb ..][0 .. w * wb]);
                }
            } else {
                try s.cstream.memcpy2D(.{
                    .src = .{ .host = src.ptr },
                    .src_pitch = @as(usize, d.stride) * wb,
                    .dst = .{ .device = buf.at(((c * layers + t_layer) * lay + pad * pp + pad) * wb) },
                    .dst_pitch = pp * wb,
                    .width_bytes = w * wb,
                    .height = d.h_,
                });
            }
        }
        fi += 1;
    }
    if (host_opt) |hb| {
        const host = hb.ptr;
        const t0: usize = @intCast(center + k_start);
        const nlay: usize = @intCast(k_end - k_start + 1);
        var c: usize = 0;
        while (c < C) : (c += 1) {
            const off = (c * layers + t0) * lay * wb;
            try s.cstream.memcpyHtoD(buf.at(off), host + off, nlay * lay * wb);
        }
    }
}

// Source then rclip; one acquire covers both (all-or-nothing, deadlock-free).
const CacheWin = struct {
    n: usize = 0,
    count: usize = 0,
    keys: [66]i64 = undefined,
    idx: [66]usize = undefined,
    load: [66]bool = undefined,
    published: [66]bool = undefined,
};

fn uploadSlot(d: *Data, s: *Stream, slot: *framecache.CacheSlot, host_opt: ?cu.HostBuffer, srcps: []const []const u8, fi: usize, t_layer: usize) CreateError!void {
    const C: usize = d.chans;
    const layers: usize = 2 * @as(usize, d.d) + 1;
    const wb: usize = d.wbytes;
    const pp: usize = d.pstride;
    const lay: usize = pp * @as(usize, d.ph);
    const pad: usize = d.pad;
    const w: usize = d.w;
    var c: usize = 0;
    while (c < C) : (c += 1) {
        const src = srcps[fi * C + c];
        std.debug.assert(src.len == @as(usize, d.stride) * d.h_ * wb);
        if (host_opt) |hb| {
            const host = hb.ptr;
            const base = ((c * layers + t_layer) * lay + pad * pp + pad) * wb;
            var y: usize = 0;
            while (y < d.h_) : (y += 1) {
                @memcpy(host[base + y * pp * wb ..][0 .. w * wb], src[y * @as(usize, d.stride) * wb ..][0 .. w * wb]);
            }
            try s.cstream.memcpyHtoD(slot.buf.at(c * lay * wb), host + (c * layers + t_layer) * lay * wb, lay * wb);
        } else {
            try s.cstream.memcpy2D(.{
                .src = .{ .host = src.ptr },
                .src_pitch = @as(usize, d.stride) * wb,
                .dst = .{ .device = slot.buf.at((c * lay + pad * pp + pad) * wb) },
                .dst_pitch = pp * wb,
                .width_bytes = w * wb,
                .height = d.h_,
            });
        }
    }
}

fn process(d: *Data, s: *Stream, dstps: []const []u8, srcps: []const []const u8, refps: ?[]const []const u8, k_end: i32, win: ?*CacheWin) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain before unwind and before cache release.
    errdefer {
        s.cstream.drain();
        s.stream.drain();
    }

    const npix: usize = @as(usize, d.stride) * d.h_;
    const C: usize = d.chans;
    const wb: usize = d.wbytes;
    const k_start = -k_end;

    if (win) |cw| {
        const lay: usize = @as(usize, d.pstride) * @as(usize, d.ph);
        const layers: usize = 2 * @as(usize, d.d) + 1;
        for (0..cw.n) |e| {
            if (!cw.load[e]) continue;
            const slot = &d.cache.slots[cw.idx[e]];
            const is_ref = e >= cw.count;
            const fi = if (is_ref) e - cw.count else e;
            const t_layer: usize = @intCast(@as(i32, @intCast(d.d)) + k_start + @as(i32, @intCast(fi)));
            const host_opt = if (is_ref) s.h_stage_r else s.h_stage;
            try uploadSlot(d, s, slot, host_opt, if (is_ref) refps.? else srcps, fi, t_layer);
            try s.cstream.record(slot.ev);
            cw.published[e] = true;
            d.cache.publish(cw.idx[e]);
        }
        for (0..cw.n) |e| {
            if (!cw.load[e]) try s.cstream.waitEvent(d.cache.slots[cw.idx[e]].ev);
        }
        for (0..cw.n) |e| {
            const is_ref = e >= cw.count;
            const fi = if (is_ref) e - cw.count else e;
            const t_layer: usize = @intCast(@as(i32, @intCast(d.d)) + k_start + @as(i32, @intCast(fi)));
            const buf = if (is_ref) s.d_u1r else s.d_u1;
            const slot = &d.cache.slots[cw.idx[e]];
            var c: usize = 0;
            while (c < C) : (c += 1) {
                try s.cstream.memcpyDtoD(
                    buf.at((c * layers + t_layer) * lay * wb),
                    slot.buf.at(c * lay * wb),
                    lay * wb,
                );
            }
        }
    } else {
        try uploadWindow(d, s, s.d_u1, s.h_stage, srcps, k_start, k_end);
        if (refps) |rp| try uploadWindow(d, s, s.d_u1r, s.h_stage_r, rp, k_start, k_end);
    }
    try s.cstream.record(s.ev_up);
    _ = cu.c.cuStreamQuery(s.cstream.handle);

    try s.stream.waitEvent(s.ev_up);

    const v = &d.variants[@intCast(k_end)];
    const gx = ceilDiv(d.w, BLK_X);
    const gy_pix = ceilDiv(d.h_, BLK_Y);
    const gy_w = ceilDiv(d.h_, VRT * BLK_Y);
    const qb: usize = d.qb;
    var q0: usize = 0;
    var bi: usize = 0;
    while (q0 < v.q_cnt) : ({
        q0 += qb;
        bi += 1;
    }) {
        const nb: usize = @min(qb, @as(usize, v.q_cnt) - q0);
        const p0 = v.w_boff[bi];
        const p1 = v.w_boff[bi + 1];
        {
            const a_u1 = if (s.d_u1r.ptr != 0) s.d_u1r.ptr else s.d_u1.ptr;
            const a_u4a = s.d_u4a.ptr;
            const a_wq = d.d_wq.ptr;
            const a_base: c_int = @intCast(v.w_base + p0);
            try s.stream.launch(d.fn_weight, .{
                .grid = .{ gx, gy_w, p1 - p0 },
                .block = .{ BLK_X, BLK_Y, 1 },
            }, .{ a_u1, a_u4a, a_wq, a_base });
        }
        {
            const a_u1 = s.d_u1.ptr;
            const a_u2 = s.d_u2.ptr;
            const a_u4a = s.d_u4a.ptr;
            const a_u5 = s.d_u5.ptr;
            const a_aq = d.d_aq.ptr;
            const a_qbase: c_int = @intCast(@as(usize, v.q_base) + q0);
            const a_nb: c_int = @intCast(nb);
            const a_first: c_int = @intFromBool(q0 == 0);
            try s.stream.launch(d.fn_acc, .{
                .grid = .{ gx, gy_pix, 1 },
                .block = .{ BLK_X, BLK_Y, 1 },
            }, .{ a_u1, a_u2, a_u4a, a_u5, a_aq, a_qbase, a_nb, a_first });
        }
    }

    const u1_bytes = @as(usize, d.pstride) * d.ph * (2 * @as(usize, d.d) + 1) * C * wb;
    const zc = d.zc_dst and s.h_stage != null;
    {
        const a_u1 = s.d_u1.ptr;
        const a_u1z: cu.c.CUdeviceptr = if (zc) s.h_stage.?.devicePtr(u1_bytes) else s.d_u1z.ptr;
        const a_u2 = s.d_u2.ptr;
        const a_u5 = s.d_u5.ptr;
        try s.stream.launch(d.fn_finish, .{
            .grid = .{ gx, gy_pix, 1 },
            .block = .{ BLK_X, BLK_Y, 1 },
        }, .{ a_u1, a_u1z, a_u2, a_u5 });
    }
    try s.stream.record(s.ev_k);
    _ = cu.c.cuStreamQuery(s.stream.handle);

    var c: usize = 0;
    if (!zc) {
        try s.cstream.waitEvent(s.ev_k);
        if (s.h_stage) |hs| {
            try s.cstream.memcpyDtoH(hs.ptr + u1_bytes, s.d_u1z.ptr, npix * C * wb);
        } else {
            while (c < C) : (c += 1) {
                try s.cstream.memcpyDtoH(dstps[c].ptr, s.d_u1z.at(c * npix * wb), dstps[c].len);
            }
        }
    }

    try s.cstream.sync();
    try s.stream.sync();

    if (s.h_stage) |hs| {
        c = 0;
        while (c < C) : (c += 1) {
            @memcpy(dstps[c], (hs.ptr + u1_bytes + c * npix * wb)[0..dstps[c].len]);
        }
    }
}

fn getFrame(n: c_int, ar: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);

    const dd: i32 = @intCast(d.d);
    const ni: i32 = @intCast(n);
    const nf: i32 = @intCast(d.vi.numFrames);
    const m: i32 = @min(dd, ni);
    const k_start: i32 = -m;
    const k_end: i32 = m;

    if (ar == .Initial) {
        var k: i32 = k_start;
        while (k <= k_end) : (k += 1) {
            const idx: c_int = @intCast(@min(@max(ni + k, 0), nf - 1));
            zapi.requestFrameFilter(idx, d.node);
            if (d.ref_node) |rn| zapi.requestFrameFilter(idx, rn);
        }
    } else if (ar == .AllFramesReady) {
        const C: usize = d.chans;
        const plane0: u32 = d.plane0;
        const numPlanes: u32 = @intCast(d.vi.format.numPlanes);
        const count: usize = @intCast(k_end - k_start + 1);

        const frames = allocator.alloc(ZFrame, count) catch {
            zapi.setFilterError("NLMeans: out of memory.");
            return null;
        };
        defer allocator.free(frames);
        const srcps = allocator.alloc([]const u8, count * C) catch {
            zapi.setFilterError("NLMeans: out of memory.");
            return null;
        };
        defer allocator.free(srcps);
        const dstps = allocator.alloc([]u8, C) catch {
            zapi.setFilterError("NLMeans: out of memory.");
            return null;
        };
        defer allocator.free(dstps);

        var rframes: []ZFrame = &.{};
        var refps: ?[]const []const u8 = null;
        if (d.ref_node) |_| {
            rframes = allocator.alloc(ZFrame, count) catch {
                zapi.setFilterError("NLMeans: out of memory.");
                return null;
            };
            const rslices = allocator.alloc([]const u8, count * C) catch {
                allocator.free(rframes);
                zapi.setFilterError("NLMeans: out of memory.");
                return null;
            };
            refps = rslices;
        }
        defer if (refps) |rp| allocator.free(rp);
        defer allocator.free(rframes);

        var win: CacheWin = .{ .count = count, .n = if (d.ref_node != null) count * 2 else count };
        var fi: usize = 0;
        var k: i32 = k_start;
        while (k <= k_end) : (k += 1) {
            const idx: c_int = @intCast(@min(@max(ni + k, 0), nf - 1));
            win.keys[fi] = idx;
            if (d.ref_node != null) win.keys[count + fi] = (@as(i64, 1) << 40) | @as(i64, idx);
            frames[fi] = zapi.initZFrame(d.node, idx);
            var c: usize = 0;
            while (c < C) : (c += 1) srcps[fi * C + c] = frames[fi].getReadSlice(plane0 + @as(u32, @intCast(c)));
            if (d.ref_node) |rn| {
                rframes[fi] = zapi.initZFrame(rn, idx);
                c = 0;
                while (c < C) : (c += 1) @constCast(refps.?)[fi * C + c] = rframes[fi].getReadSlice(plane0 + @as(u32, @intCast(c)));
            }
            fi += 1;
        }
        defer for (frames) |f| f.deinit();
        defer if (d.ref_node) |_| for (rframes) |f| f.deinit();

        // Abandon unpublished claims (else keys wait forever).
        if (d.use_cache) {
            d.cache.acquire(win.keys[0..win.n], win.idx[0..win.n], win.load[0..win.n]);
            for (0..win.n) |e| win.published[e] = false;
        }
        defer if (d.use_cache) {
            for (0..win.n) |e| {
                if (win.load[e] and !win.published[e]) d.cache.abandon(win.idx[e]);
            }
            d.cache.release(win.idx[0..win.n]);
        };

        const center_frame = frames[@as(usize, @intCast(-k_start))];
        const dst = center_frame.newVideoFrame();
        var c: usize = 0;
        while (c < C) : (c += 1) dstps[c] = dst.getWriteSlice(plane0 + @as(u32, @intCast(c)));

        var p: u32 = 0;
        while (p < numPlanes) : (p += 1) {
            if (p < plane0 or p >= plane0 + @as(u32, @intCast(C))) {
                @memcpy(dst.getWriteSlice(p), center_frame.getReadSlice(p));
            }
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, dstps, srcps, refps, k_end, if (d.use_cache) &win else null) catch |err| {
            zapi.setFilterError("NLMeans: process failed.");
            std.log.err("vszipcu NLMeans process failed: {t}", .{err});
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
    d.d_aq.deinit();
    d.d_wq.deinit();
    d.module.deinit();
    d.dev.pop();
    d.dev.deinit();
    freeTables(d);
    vsapi.?.freeNode.?(d.node);
    if (d.ref_node) |rn| vsapi.?.freeNode.?(rn);
    allocator.destroy(d);
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize, n_threads: usize) CreateError!*Data {
    d.dev = try cu.Device.init(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    d.module = try compileModule(d);
    errdefer d.module.deinit();
    d.fn_weight = try d.module.function("nlmWeight");
    d.fn_acc = try d.module.function("nlmAccumulation");
    d.fn_finish = try d.module.function("nlmFinish");

    d.d_wq = try cu.DeviceBuffer.alloc(d.wq_host.len * 4);
    errdefer d.d_wq.deinit();
    try cu.driver.memcpyHtoDSync(d.d_wq.ptr, d.wq_host.ptr, d.wq_host.len * 4);
    d.d_aq = try cu.DeviceBuffer.alloc(d.aq_host.len * 4);
    errdefer d.d_aq.deinit();
    try cu.driver.memcpyHtoDSync(d.d_aq.ptr, d.aq_host.ptr, d.aq_host.len * 4);

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

    if (data.use_cache) {
        const n_clips: usize = if (data.has_ref) 2 else 1;
        const tw: usize = 2 * @as(usize, data.d) + 1;
        const min_slots = n_clips * tw;
        const want = min_slots + @max(num_streams, n_threads) + 2 * @as(usize, data.d);
        const slots = allocator.alloc(framecache.CacheSlot, want) catch return error.OutOfMemory;
        var n_ok: usize = 0;
        for (slots) |*slot| {
            slot.* = .{};
            slot.buf = cu.DeviceBuffer.alloc(data.slot_bytes) catch break;
            if (slot.buf.fill(0, data.slot_bytes)) |_| {} else |_| {
                slot.buf.deinit();
                slot.buf = .{};
                break;
            }
            slot.ev = cu.Event.init() catch {
                slot.buf.deinit();
                slot.buf = .{};
                break;
            };
            n_ok += 1;
        }
        if (n_ok < min_slots) {
            var i: usize = n_ok;
            while (i > 0) {
                i -= 1;
                slots[i].ev.deinit();
                slots[i].buf.deinit();
            }
            allocator.free(slots);
            data.use_cache = false;
            std.log.warn("vszipcu NLMeans: not enough device memory for the source cache; running uncached.", .{});
        } else {
            data.cache.slots = if (n_ok == want) slots else blk: {
                const shrunk = allocator.realloc(slots, n_ok) catch slots[0..n_ok];
                break :blk shrunk;
            };
            if (n_ok < want) {
                std.log.warn("vszipcu NLMeans: source cache shrunk to {d}/{d} slots (low device memory).", .{ n_ok, want });
            }
        }
    }
    return data;
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core_ptr, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi_in = map_in.getNodeVi("clip").?;
    d.vi = vi_in.*;

    var keep = false;
    var tables_built = false;
    defer if (!keep) {
        if (tables_built) freeTables(&d);
        zapi.freeNode(d.node);
        if (d.ref_node) |rn| zapi.freeNode(rn);
    };

    const fmt = d.vi.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Float and (bits == 32 or bits == 16)) or
        (fmt.sampleType == .Integer and (bits == 8 or bits == 16));
    if (!depth_ok) {
        return map_out.setError("NLMeans: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float).");
    }
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.wbytes = @intCast(fmt.bytesPerSample);
    if (d.vi.width <= 0 or d.vi.height <= 0) {
        return map_out.setError("NLMeans: clip must have constant dimensions.");
    }
    if (d.vi.width > 8192 or d.vi.height > 8192) {
        return map_out.setError("NLMeans: 8192x8192 is the highest supported resolution.");
    }

    const dd = map_in.getValue(i32, "d") orelse 1;
    const a = map_in.getValue(i32, "a") orelse 2;
    const ss = map_in.getValue(i32, "s") orelse 4;
    d.h = map_in.getValue(f32, "h") orelse 1.2;
    const wmode = map_in.getValue(i32, "wmode") orelse 0;
    d.wref = map_in.getValue(f32, "wref") orelse 1.0;
    const chstr = map_in.getData("channels", 0) orelse "auto";
    const ns_req = map_in.getValue(i32, "num_streams");

    if (dd < 0 or dd > 16) return map_out.setError("NLMeans: d must be 0..16.");
    if (map_in.getNodeVi("rclip")) |rv| {
        d.ref_node = rv[0];
        d.has_ref = true;
        const rvi = rv[1];
        const rfmt = rvi.format;
        const same = rfmt.colorFamily == fmt.colorFamily and rfmt.sampleType == fmt.sampleType and
            rfmt.bitsPerSample == fmt.bitsPerSample and rfmt.subSamplingW == fmt.subSamplingW and
            rfmt.subSamplingH == fmt.subSamplingH and rvi.width == d.vi.width and
            rvi.height == d.vi.height and rvi.numFrames == d.vi.numFrames;
        if (!same) return map_out.setError("NLMeans: 'rclip' must match the source clip's format, dimensions and frame count.");
    }
    if (a < 1 or a > 64) return map_out.setError("NLMeans: a must be 1..64.");
    if (ss < 0 or ss > 8) return map_out.setError("NLMeans: s must be 0..8.");
    if (!math.isFinite(d.h) or d.h <= 0) return map_out.setError("NLMeans: h must be > 0.");
    if (wmode < 0 or wmode > 3) return map_out.setError("NLMeans: wmode must be 0..3.");
    if (!math.isFinite(d.wref) or d.wref < 0) return map_out.setError("NLMeans: wref must be >= 0.");
    if (ns_req) |ns| {
        if (ns < 1 or ns > 32) return map_out.setError("NLMeans: num_streams must be 1..32.");
    }
    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("NLMeans: invalid device ID.");

    const eq = std.ascii.eqlIgnoreCase;
    switch (fmt.colorFamily) {
        .Gray => {
            if (!(eq(chstr, "Y") or eq(chstr, "auto"))) {
                return map_out.setError("NLMeans: 'channels' must be 'Y' with Gray.");
            }
            d.ref = REF_LUMA;
            d.chans = 1;
            d.plane0 = 0;
        },
        .YUV => {
            if (eq(chstr, "YUV")) {
                if (fmt.subSamplingW != 0 or fmt.subSamplingH != 0) {
                    return map_out.setError("NLMeans: 'channels'='YUV' requires 4:4:4 (the joint patch distance needs one pixel lattice; run 'Y' and 'UV' passes for subsampled formats).");
                }
                d.ref = REF_YUV;
                d.chans = 3;
                d.plane0 = 0;
            } else if (eq(chstr, "Y") or eq(chstr, "auto")) {
                d.ref = REF_LUMA;
                d.chans = 1;
                d.plane0 = 0;
            } else if (eq(chstr, "UV")) {
                d.ref = REF_CHROMA;
                d.chans = 2;
                d.plane0 = 1;
            } else {
                return map_out.setError("NLMeans: 'channels' must be 'YUV', 'Y' or 'UV' with YUV.");
            }
        },
        .RGB => {
            if (!(eq(chstr, "RGB") or eq(chstr, "auto"))) {
                return map_out.setError("NLMeans: 'channels' must be 'RGB' with RGB.");
            }
            d.ref = REF_RGB;
            d.chans = 3;
            d.plane0 = 0;
        },
        else => return map_out.setError("NLMeans: unsupported color family."),
    }

    const sw: u5 = @intCast(fmt.subSamplingW);
    const sh: u5 = @intCast(fmt.subSamplingH);
    if (d.ref == REF_CHROMA) {
        d.w = @as(u32, @intCast(d.vi.width)) >> sw;
        d.h_ = @as(u32, @intCast(d.vi.height)) >> sh;
    } else {
        d.w = @intCast(d.vi.width);
        d.h_ = @intCast(d.vi.height);
    }

    if (2 * a + 1 > @as(i32, @intCast(d.w)) or 2 * a + 1 > @as(i32, @intCast(d.h_))) {
        return map_out.setError("NLMeans: research window (2*a+1) larger than the frame.");
    }

    d.d = @intCast(dd);
    d.a = @intCast(a);
    d.s = @intCast(ss);
    d.wmode = @intCast(wmode);
    const strides = vsutil.strideFromVi(&d.vi);
    d.stride = if (d.ref == REF_CHROMA) strides[1] else strides[0];
    d.pad = @intCast(a);
    d.pstride = @intCast((@as(usize, d.w) + 2 * @as(usize, d.pad) + 7) & ~@as(usize, 7));
    d.ph = d.h_ + 2 * d.pad;

    d.qb = if (@as(u64, d.stride) * @as(u64, d.h_) <= 1920 * 1152) nlm_qb_small else nlm_qb_large;

    {
        const spt_side: i32 = 2 * a + 1;
        const spt_area: i32 = spt_side * spt_side;
        const center: i32 = dd;
        var wq_list: std.ArrayListUnmanaged(i32) = .empty;
        var aq_list: std.ArrayListUnmanaged(i32) = .empty;
        const variants = allocator.alloc(Variant, @intCast(dd + 1)) catch return map_out.setError("NLMeans: out of memory.");
        var m: i32 = 0;
        while (m <= dd) : (m += 1) {
            const v = &variants[@intCast(m)];
            v.w_base = @intCast(wq_list.items.len / 8);
            v.q_base = @intCast(aq_list.items.len / 8);
            var boff: std.ArrayListUnmanaged(u32) = .empty;
            var q_idx: u32 = 0;
            var kk: i32 = -m;
            while (kk <= 0) : (kk += 1) {
                var j: i32 = -a;
                while (j <= a) : (j += 1) {
                    var i: i32 = -a;
                    while (i <= a) : (i += 1) {
                        if (kk * spt_area + j * spt_side + i < 0) {
                            const b_local: u32 = q_idx % d.qb;
                            if (b_local == 0) boff.append(allocator, @intCast(wq_list.items.len / 8 - v.w_base)) catch unreachable;
                            const slot_c: i32 = if (dd == 0) @intCast(b_local) else 2 * @as(i32, @intCast(b_local));
                            const slot_m: i32 = if (kk != 0) slot_c + 1 else slot_c;
                            wq_list.appendSlice(allocator, &.{ center, i, j, kk, slot_c, 0, 0, 0 }) catch unreachable;
                            if (kk != 0) wq_list.appendSlice(allocator, &.{ center - kk, i, j, kk, slot_m, 0, 0, 0 }) catch unreachable;
                            aq_list.appendSlice(allocator, &.{ i, j, kk, slot_c, slot_m, 0, 0, 0 }) catch unreachable;
                            q_idx += 1;
                        }
                    }
                }
            }
            boff.append(allocator, @intCast(wq_list.items.len / 8 - v.w_base)) catch unreachable;
            v.q_cnt = q_idx;
            v.w_boff = boff.toOwnedSlice(allocator) catch unreachable;
        }
        d.wq_host = wq_list.toOwnedSlice(allocator) catch unreachable;
        d.aq_host = aq_list.toOwnedSlice(allocator) catch unreachable;
        d.variants = variants;
        tables_built = true;
    }

    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    d.use_pinned = num_streams >= 2;
    d.zc_dst = d.use_pinned and d.wbytes == 4;
    d.use_cache = d.d > 0;
    d.slot_bytes = @as(usize, d.chans) * (@as(usize, d.pstride) * @as(usize, d.ph)) * @as(usize, d.wbytes);

    var core_info: vs.CoreInfo = .{};
    zapi.getCoreInfo(core_ptr, &core_info);
    const n_threads: usize = @intCast(@max(core_info.numThreads, 1));

    const data = initCuda(&d, device_id, num_streams, n_threads) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "NLMeans: invalid device ID.",
            error.Nvrtc => "NLMeans: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "NLMeans: could not locate NVRTC (wheel should ship nvrtc64_130_0.dll next to the plugin).",
            error.OutOfDeviceMemory => "NLMeans: out of device memory.",
            else => "NLMeans: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu NLMeans init failed: {t}", .{err});
        return;
    };

    keep = true;

    const rp: vs.RequestPattern = if (d.d > 0) .General else .StrictSpatial;
    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = rp },
        .{ .source = d.ref_node, .requestPattern = rp },
    };
    const deps = if (d.has_ref) dep[0..2] else dep[0..1];
    zapi.createVideoFilter(out, "NLMeans", &data.vi, getFrame, free, .Parallel, deps, data);
}
