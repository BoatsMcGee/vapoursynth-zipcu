const std = @import("std");
const vapoursynth = @import("vapoursynth");
const cu = @import("cu.zig");
const framecache = @import("framecache.zig");
const pool_mod = @import("pool.zig");
const vsutil = @import("vsutil.zig");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const math = std.math;

const CreateError = cu.CreateError;
const ceilDiv = cu.ceilDiv;

const allocator = std.heap.c_allocator;

const kernel_source = @embedFile("dfttest.cu");

// C libm for table math (std.math differs by 1 ulp).
extern "c" fn cos(f64) f64;
extern "c" fn sin(f64) f64;
extern "c" fn pow(f64, f64) f64;

const BS: i32 = 16;

// -use_fast_math (parity).
const nvrtc_opts: cu.nvrtc.Options = .{
    .extra = &.{ "-use_fast_math", "-std=c++17", "-modify-stack-limit=false" },
    .log_name = "DFTTest",
};

// Shewchuk exact sum for wscale (naive `+=` differs).
fn fsum(values: []const f64) f64 {
    var partials: [64]f64 = undefined;
    var n: usize = 0;
    for (values) |item| {
        var x = item;
        var i: usize = 0;
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var y = partials[j];
            if (@abs(x) < @abs(y)) {
                const t = x;
                x = y;
                y = t;
            }
            const hi = x + y;
            const lo = y - (hi - x);
            if (lo != 0.0) {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        std.debug.assert(i < partials.len);
        partials[i] = x;
        n = i + 1;
    }
    if (n == 0) return 0.0;
    var hi = partials[n - 1];
    var lo: f64 = 0.0;
    var k = n - 1;
    while (k > 0) {
        const x = hi;
        const y = partials[k - 1];
        k -= 1;
        hi = x + y;
        const yr = hi - x;
        lo = y - yr;
        if (lo != 0.0) break;
    }
    if (k > 0 and ((lo < 0.0 and partials[k - 1] < 0.0) or (lo > 0.0 and partials[k - 1] > 0.0))) {
        const y2 = lo * 2.0;
        const x2 = hi + y2;
        if (y2 == x2 - hi) hi = x2;
    }
    return hi;
}

fn getWindowValue(location: f64, size: i32, mode: i32, beta: f64) f64 {
    const size_f: f64 = @floatFromInt(size);
    const temp = math.pi * location / size_f;
    return switch (mode) {
        0 => 0.5 * (1.0 - cos(2.0 * temp)),
        1 => 0.53836 - 0.46164 * cos(2.0 * temp),
        2 => 0.42 - 0.5 * cos(2.0 * temp) + 0.08 * cos(4.0 * temp),
        3 => 0.35875 - 0.48829 * cos(2.0 * temp) + 0.14128 * cos(4.0 * temp) - 0.01168 * cos(6.0 * temp),
        4 => blk: {
            const v = 2.0 * location / size_f - 1.0;
            break :blk besselI0(math.pi * beta * @sqrt(1.0 - v * v)) / besselI0(math.pi * beta);
        },
        5 => 0.27105140069342415
            - 0.433297939234486060 * cos(2.0 * temp)
            + 0.218122999543110620 * cos(4.0 * temp)
            - 0.065925446388030898 * cos(6.0 * temp)
            + 0.010811742098372268 * cos(8.0 * temp)
            - 7.7658482522509342e-4 * cos(10.0 * temp)
            + 1.3887217350903198e-5 * cos(12.0 * temp),
        6 => 0.2810639 - 0.5208972 * cos(2.0 * temp) + 0.1980399 * cos(4.0 * temp),
        7 => 1.0,
        8 => 1.0 - 2.0 * @abs(location - size_f / 2.0) / size_f,
        9 => 0.62 - 0.48 * (location / size_f - 0.5) - 0.38 * cos(2.0 * temp),
        10 => 0.355768 - 0.487396 * cos(2.0 * temp) + 0.144232 * cos(4.0 * temp) - 0.012604 * cos(6.0 * temp),
        11 => 0.3635819 - 0.4891775 * cos(2.0 * temp) + 0.1365995 * cos(4.0 * temp) - 0.0106411 * cos(6.0 * temp),
        else => unreachable,
    };
}

fn besselI0(p_in: f64) f64 {
    const p = p_in / 2.0;
    var n: f64 = 1.0;
    var t: f64 = 1.0;
    var d: f64 = 1.0;
    var k: i32 = 1;
    while (true) {
        n *= p;
        d *= @as(f64, @floatFromInt(k));
        const v = n / d;
        t += v * v;
        k += 1;
        if (k >= 15 or v <= 1e-8) break;
    }
    return t;
}

fn normalizeWindow(window: []f64, size: i32, step: i32) void {
    var nw: [16]f64 = @splat(0.0);
    const size_u: usize = @intCast(size);
    std.debug.assert(size_u <= nw.len);
    var q: i32 = 0;
    while (q < size) : (q += 1) {
        var h: i32 = q;
        while (h >= 0) : (h -= step) {
            nw[@intCast(q)] += window[@intCast(h)] * window[@intCast(h)];
        }
        h = q + step;
        while (h < size) : (h += step) {
            nw[@intCast(q)] += window[@intCast(h)] * window[@intCast(h)];
        }
    }
    q = 0;
    while (q < size) : (q += 1) {
        window[@intCast(q)] = window[@intCast(q)] / @sqrt(nw[@intCast(q)]);
    }
}

fn getWindow(radius: i32, block_step: i32, swin: i32, sbeta: f64, twin: i32, tbeta: f64) ![]f64 {
    const tw: usize = @intCast(2 * radius + 1);
    var temporal: [7]f64 = undefined;
    for (0..tw) |i| {
        temporal[i] = getWindowValue(@as(f64, @floatFromInt(i)) + 0.5, 2 * radius + 1, twin, tbeta);
    }
    var spatial: [16]f64 = undefined;
    for (0..16) |i| {
        spatial[i] = getWindowValue(@as(f64, @floatFromInt(i)) + 0.5, BS, swin, sbeta);
    }
    normalizeWindow(spatial[0..16], BS, block_step);

    const window = try allocator.alloc(f64, tw * 256);
    const div = @sqrt(@as(f64, @floatFromInt(2 * radius + 1))) * @as(f64, @floatFromInt(BS));
    var idx: usize = 0;
    for (0..tw) |t| {
        for (0..16) |s1| {
            for (0..16) |s2| {
                window[idx] = (temporal[t] * spatial[s1] * spatial[s2]) / div;
                idx += 1;
            }
        }
    }
    return window;
}

const Complex = struct { re: f64, im: f64 };

fn dftReal(dst: []Complex, dst_stride: usize, src: []const f64, src_stride: usize, n: i32) void {
    const out_num = @divTrunc(n, 2) + 1;
    var i: i32 = 0;
    while (i < out_num) : (i += 1) {
        var sum: Complex = .{ .re = 0.0, .im = 0.0 };
        var j: i32 = 0;
        while (j < n) : (j += 1) {
            const imag = @as(f64, @floatFromInt(-2 * i * j)) * math.pi / @as(f64, @floatFromInt(n));
            const s = src[@as(usize, @intCast(j)) * src_stride];
            sum.re += s * cos(imag);
            sum.im += s * sin(imag);
        }
        dst[@as(usize, @intCast(i)) * dst_stride] = sum;
    }
}

fn dftCplx(dst: []Complex, src: []const Complex, n: i32, stride: usize) void {
    var out: [7]Complex = undefined;
    var out16: [16]Complex = undefined;
    const o: []Complex = if (n <= 7) out[0..@intCast(n)] else out16[0..@intCast(n)];
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        var sum: Complex = .{ .re = 0.0, .im = 0.0 };
        var j: i32 = 0;
        while (j < n) : (j += 1) {
            const imag = @as(f64, @floatFromInt(-2 * i * j)) * math.pi / @as(f64, @floatFromInt(n));
            const wre = cos(imag);
            const wim = sin(imag);
            const s = src[@as(usize, @intCast(j)) * stride];
            sum.re += s.re * wre - s.im * wim;
            sum.im += s.re * wim + s.im * wre;
        }
        o[@intCast(i)] = sum;
    }
    i = 0;
    while (i < n) : (i += 1) {
        dst[@as(usize, @intCast(i)) * stride] = o[@intCast(i)];
    }
}

fn rdftTables(radius: i32, input: []const f64) ![]f64 {
    const tw: usize = @intCast(2 * radius + 1);
    const cols: usize = 9;
    const csize = tw * 16 * cols;

    const output = try allocator.alloc(Complex, csize);
    defer allocator.free(output);
    const output2 = try allocator.alloc(Complex, csize);
    defer allocator.free(output2);

    if (radius == 0) {
        for (0..16) |i| {
            dftReal(output[i * cols ..], 1, input[i * 16 ..], 1, BS);
        }
        for (0..cols) |i| {
            dftCplx(output2[i..], output[i..], BS, cols);
        }
        const ret = try allocator.alloc(f64, csize * 2);
        for (output2, 0..) |v, i| {
            ret[i * 2] = v.re;
            ret[i * 2 + 1] = v.im;
        }
        return ret;
    }

    for (0..tw * 16) |i| {
        dftReal(output[i * cols ..], 1, input[i * 16 ..], 1, BS);
    }
    for (0..tw) |i| {
        for (0..cols) |j| {
            dftCplx(output2[i * 16 * cols + j ..], output[i * 16 * cols + j ..], BS, cols);
        }
    }
    for (0..16 * cols) |i| {
        dftCplx(output[i..], output2[i..], 2 * radius + 1, 16 * cols);
    }
    const ret = try allocator.alloc(f64, csize * 2);
    for (output, 0..) |v, i| {
        ret[i * 2] = v.re;
        ret[i * 2 + 1] = v.im;
    }
    return ret;
}

const Norm = enum { identity, sqrt, cbrt };

fn applyNorm(n: Norm, x: f64) f64 {
    return switch (n) {
        .identity => x,
        .sqrt => @sqrt(x),
        .cbrt => pow(x, 1.0 / 3.0),
    };
}

const SigmaFunc = struct {
    locs: []f64 = &.{},
    sigmas: []f64 = &.{},
    constant: f64 = 0.0,

    fn initConst(n: Norm, sigma: f64) SigmaFunc {
        return .{ .constant = applyNorm(n, sigma) };
    }

    fn initPacks(data: []const f64, n: Norm) !SigmaFunc {
        std.debug.assert(data.len % 2 == 0 and data.len >= 2);
        const cnt = data.len / 2;
        const locs = try allocator.alloc(f64, cnt);
        errdefer allocator.free(locs);
        const sigmas = try allocator.alloc(f64, cnt);
        for (0..cnt) |i| {
            locs[i] = data[i * 2];
            sigmas[i] = data[i * 2 + 1];
        }
        var i: usize = 1;
        while (i < cnt) : (i += 1) {
            const kl = locs[i];
            const ks = sigmas[i];
            var j = i;
            while (j > 0 and locs[j - 1] > kl) : (j -= 1) {
                locs[j] = locs[j - 1];
                sigmas[j] = sigmas[j - 1];
            }
            locs[j] = kl;
            sigmas[j] = ks;
        }
        for (sigmas) |*s| s.* = applyNorm(n, s.*);
        return .{ .locs = locs, .sigmas = sigmas };
    }

    fn deinit(self: *const SigmaFunc) void {
        if (self.locs.len > 0) {
            allocator.free(self.locs);
            allocator.free(self.sigmas);
        }
    }

    fn eval(self: *const SigmaFunc, x: f64) ?f64 {
        if (self.locs.len == 0) return self.constant;
        var i: usize = 0;
        while (i + 1 < self.locs.len) : (i += 1) {
            if (x <= self.locs[i + 1]) {
                const w = (x - self.locs[i]) / (self.locs[i + 1] - self.locs[i]);
                return (1.0 - w) * self.sigmas[i] + w * self.sigmas[i + 1];
            }
        }
        return null;
    }
};

fn getLocation(position: i32, length: i32) f64 {
    if (length == 1) return 0.0;
    const half = @divTrunc(length, 2);
    if (position > half) {
        return @as(f64, @floatFromInt(length - position)) / @as(f64, @floatFromInt(half));
    }
    return @as(f64, @floatFromInt(position)) / @as(f64, @floatFromInt(half));
}

fn getSigma(position: i32, length: i32, func: *const SigmaFunc) ?f64 {
    if (length == 1) return 1.0;
    return func.eval(getLocation(position, length));
}

fn emitF32Array(buf: *std.ArrayList(u8), values: []const f64) !void {
    // `{x}` = hexfloat; decimal f32 round-trip is a parity bug.
    for (values, 0..) |v, i| {
        if (i != 0) try buf.appendSlice(allocator, ",");
        try buf.print(allocator, "{x}", .{@as(f32, @floatCast(v))});
    }
}

const KernelConfig = struct {
    radius: i32,
    block_step: i32,
    warps_per_block: u32,
    tiles_per_warp: u32,
    warp_size: u32,
    sample_type: vs.SampleType,
    bits: i32,
    filter_type: i32,
    zmean: bool,
    sigma_scalar: ?f64,
    sigma_array: []const f64,
    sigma2: f64,
    pmin: f64,
    pmax: f64,
    beta: f64,
    window: []const f64,
    window_freq: []const f64,
};

fn genKernelPrefix(cfg: *const KernelConfig) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const w = &buf;

    try w.print(allocator,
        \\#define RADIUS {d}
        \\#define BLOCK_SIZE {d}
        \\#define BLOCK_STEP {d}
        \\#define IN_PLACE 0
        \\#define WARP_SIZE {d}
        \\#define WARPS_PER_BLOCK {d}
        \\#define TILES_PER_WARP {d}
        \\
    , .{ cfg.radius, BS, cfg.block_step, cfg.warp_size, cfg.warps_per_block, cfg.tiles_per_warp });
    if (cfg.sample_type == .Integer) {
        const type_str: []const u8 = if (cfg.bits <= 8) "unsigned char" else "unsigned short";
        const scale = 1.0 / @as(f64, @floatFromInt(@as(u32, 1) << @intCast(cfg.bits - 8)));
        try w.print(allocator, "#define TYPE {s}\n#define SCALE {x}\n#define PEAK {d}\n", .{
            type_str, scale, (@as(u32, 1) << @intCast(cfg.bits)) - 1,
        });
    } else {
        try w.appendSlice(allocator, "#define TYPE float\n#define SCALE 255.0\n");
    }
    try w.print(allocator,
        \\#define FILTER_TYPE {d}
        \\#define ZERO_MEAN {d}
        \\#define SIGMA_IS_SCALAR {d}
        \\
    , .{ cfg.filter_type, @intFromBool(cfg.zmean), @intFromBool(cfg.sigma_scalar != null) });

    if (cfg.zmean) {
        try w.appendSlice(allocator, "__device__ static const float window_freq[] { ");
        try emitF32Array(w, cfg.window_freq);
        try w.appendSlice(allocator, " };\n");
    }
    try w.appendSlice(allocator, "__device__ static const float window[] { ");
    try emitF32Array(w, cfg.window);
    try w.appendSlice(allocator, " };\n");

    try w.appendSlice(allocator,
        \\
        \\__device__
        \\static void filter(float2 & value, int x, int y, int t) {
        \\#if SIGMA_IS_SCALAR
        \\    float sigma = static_cast<float>(
    );
    if (cfg.sigma_scalar) |s| {
        try w.print(allocator, "{x}", .{@as(f32, @floatCast(s))});
    } else {
        try emitF32Array(w, cfg.sigma_array);
    }
    try w.appendSlice(allocator,
        \\);
        \\#else // SIGMA_IS_SCALAR
        \\    __device__ static const float sigma_array[] {
    );
    if (cfg.sigma_scalar) |s| {
        try w.print(allocator, "{x}", .{@as(f32, @floatCast(s))});
    } else {
        try emitF32Array(w, cfg.sigma_array);
    }
    try w.print(allocator,
        \\ }};
        \\    float sigma = sigma_array[(t * BLOCK_SIZE + y) * (BLOCK_SIZE / 2 + 1) + x];
        \\#endif // SIGMA_IS_SCALAR
        \\    [[maybe_unused]] float sigma2 = static_cast<float>({x});
        \\    [[maybe_unused]] float pmin = static_cast<float>({x});
        \\    [[maybe_unused]] float pmax = static_cast<float>({x});
        \\    [[maybe_unused]] float beta = static_cast<float>({x});
        \\    [[maybe_unused]] float multiplier {{}};
        \\
        \\#if FILTER_TYPE == 2
        \\    value.x *= sigma;
        \\    value.y *= sigma;
        \\    return ;
        \\#endif
        \\
        \\    float psd = value.x * value.x + value.y * value.y;
        \\
        \\#if FILTER_TYPE == 1
        \\    if (psd < sigma) {{
        \\        value.x = 0.0f;
        \\        value.y = 0.0f;
        \\    }}
        \\    return ;
        \\#elif FILTER_TYPE == 0
        \\    multiplier = fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f);
        \\#elif FILTER_TYPE == 3
        \\    if (psd >= pmin && psd <= pmax) {{
        \\        multiplier = sigma;
        \\    }} else {{
        \\        multiplier = sigma2;
        \\    }}
        \\#elif FILTER_TYPE == 4
        \\    multiplier = sigma * sqrtf(psd * (pmax / ((psd + pmin) * (psd + pmax) + 1e-15f)));
        \\#elif FILTER_TYPE == 5
        \\    multiplier = powf(fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f), beta);
        \\#else
        \\    multiplier = sqrtf(fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f));
        \\#endif
        \\
        \\    value.x *= multiplier;
        \\    value.y *= multiplier;
        \\}}
        \\
    , .{
        @as(f32, @floatCast(cfg.sigma2)),
        @as(f32, @floatCast(cfg.pmin)),
        @as(f32, @floatCast(cfg.pmax)),
        @as(f32, @floatCast(cfg.beta)),
    });

    return buf.toOwnedSlice(allocator);
}

fn calcPadSize(size: i32, block_step: i32) i32 {
    const rem = @mod(size, BS);
    return size + (if (rem != 0) BS - rem else 0) + @max(BS - block_step, block_step) * 2;
}

fn calcPadNum(size: i32, block_step: i32) i32 {
    return @divTrunc(calcPadSize(size, block_step) - BS, block_step) + 1;
}

const Geom = struct {
    w: i32 = 0,
    h: i32 = 0,
    stride: i32 = 0,
    pw: i32 = 0,
    ph: i32 = 0,
    hn: i32 = 0,
    vn: i32 = 0,
    padded_off: usize = 0,
    spatial_off: usize = 0,
    out_off: usize = 0,
    cache_off: usize = 0,

    fn padSlice(self: *const Geom) usize {
        return @as(usize, @intCast(self.pw)) * @as(usize, @intCast(self.ph));
    }
};

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    radius: i32 = 0,
    block_step: i32 = 0,
    bytes: u32 = 4,
    process: [3]bool = .{ false, false, false },
    geom: [3]Geom = .{ .{}, .{}, .{} },
    padded_bytes: usize = 0,
    spatial_bytes: usize = 0,
    out_bytes: usize = 0,

    warp_size: u32 = 32,
    wpb: u32 = 1,
    tpw: u32 = 2,
    fused_blocks: u32 = 1,

    direct_d2h: bool = false,

    use_cache: bool = false,
    cache: framecache.FrameCache = .{},
    slot_bytes: usize = 0,

    raw_off: [3]usize = .{ 0, 0, 0 },
    raw_bytes: usize = 0,
    staged: bool = false,

    dev: cu.Device = .{},
    module: cu.Module = .{},
    fn_fused: cu.Function = .{},
    fn_col2im: cu.Function = .{},
    fn_pad: cu.Function = .{},

    pool: pool_mod.Pool(Stream, Data) = .{},
    spool: pool_mod.Pool(PadStage, Data) = .{},
};

const PadStage = struct {
    h: cu.HostBuffer,

    pub fn init(self: *PadStage, d: *Data) !void {
        const tw: usize = @intCast(2 * d.radius + 1);
        self.h = try cu.HostBuffer.alloc(tw * d.raw_bytes, .{});
    }

    pub fn deinit(self: *PadStage) void {
        self.h.deinit();
    }
};

const Stream = struct {
    stream: cu.Stream,
    d_padded: cu.DeviceBuffer,
    d_spatial: cu.DeviceBuffer,
    d_raw: cu.DeviceBuffer,
    d_out: cu.DeviceBuffer,
    h_out: cu.HostBuffer,

    pub fn init(self: *Stream, d: *Data) !void {
        self.d_padded = try cu.DeviceBuffer.alloc(d.padded_bytes);
        errdefer self.d_padded.deinit();
        self.d_spatial = try cu.DeviceBuffer.alloc(d.spatial_bytes);
        errdefer self.d_spatial.deinit();
        self.d_raw = try cu.DeviceBuffer.alloc(d.raw_bytes);
        errdefer self.d_raw.deinit();
        self.d_out = try cu.DeviceBuffer.alloc(d.out_bytes);
        errdefer self.d_out.deinit();
        self.h_out = try cu.HostBuffer.alloc(if (d.direct_d2h) 0 else d.out_bytes, .{});
        errdefer self.h_out.deinit();
        self.stream = try cu.Stream.init();
    }

    pub fn deinit(self: *Stream) void {
        self.stream.deinit();
        self.h_out.deinit();
        self.d_out.deinit();
        self.d_raw.deinit();
        self.d_spatial.deinit();
        self.d_padded.deinit();
    }
};

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

const CacheWin = struct {
    tw: usize = 0,
    keys: [7]i64 = undefined,
    idx: [7]usize = undefined,
    load: [7]bool = undefined,
    published: [7]bool = undefined,
};

const Upload = struct {
    ptr: [7][3][*]const u8 = undefined,
};

fn uploadPadPlane(d: *Data, s: *Stream, g: *const Geom, src: [*]const u8, dst_slice: cu.c.CUdeviceptr, p: u32) CreateError!void {
    const raw_len = @as(usize, @intCast(g.h)) * @as(usize, @intCast(g.stride)) * d.bytes;
    try s.stream.memcpyHtoD(s.d_raw.at(d.raw_off[p]), src, raw_len);
    const a_src = s.d_raw.at(d.raw_off[p]);
    const a_w: c_int = g.w;
    const a_h: c_int = g.h;
    const a_stride: c_int = g.stride;
    try s.stream.launch(d.fn_pad, .{
        .grid = .{ ceilDiv(@intCast(g.pw), 32), ceilDiv(@intCast(g.ph), 8), 1 },
        .block = .{ 32, 8, 1 },
    }, .{ dst_slice, a_src, a_w, a_h, a_stride });
}

fn process(d: *Data, s: *Stream, up: *const Upload, dst: ZFrameW, win: ?*CacheWin) CreateError!void {
    try d.dev.push();
    defer d.dev.pop();
    // Drain before unwind and before cache release.
    errdefer s.stream.drain();

    const tw: usize = @intCast(2 * d.radius + 1);
    const num_planes: u32 = @intCast(d.vi.format.numPlanes);

    if (win) |cw| {
        for (0..tw) |t| {
            if (!cw.load[t]) continue;
            const slot = &d.cache.slots[cw.idx[t]];
            var p: u32 = 0;
            while (p < num_planes) : (p += 1) {
                if (!d.process[p]) continue;
                const g = &d.geom[p];
                try uploadPadPlane(d, s, g, up.ptr[t][p], slot.buf.at(g.cache_off), p);
            }
            try s.stream.record(slot.ev);
            cw.published[t] = true;
            d.cache.publish(cw.idx[t]);
        }
        for (0..tw) |t| {
            if (!cw.load[t]) try s.stream.waitEvent(d.cache.slots[cw.idx[t]].ev);
        }
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const g = &d.geom[p];
            const sb = g.padSlice() * d.bytes;
            for (0..tw) |t| {
                try s.stream.memcpyDtoD(
                    s.d_padded.at(g.padded_off + t * sb),
                    d.cache.slots[cw.idx[t]].buf.at(g.cache_off),
                    sb,
                );
            }
        }
    } else {
        for (0..tw) |t| {
            var p: u32 = 0;
            while (p < num_planes) : (p += 1) {
                if (!d.process[p]) continue;
                const g = &d.geom[p];
                const sb = g.padSlice() * d.bytes;
                try uploadPadPlane(d, s, g, up.ptr[t][p], s.d_padded.at(g.padded_off + t * sb), p);
            }
        }
    }

    var p: u32 = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const g = &d.geom[p];
        const a_spatial = s.d_spatial.at(g.spatial_off);
        const a_padded = s.d_padded.at(g.padded_off);
        const a_out: cu.c.CUdeviceptr = s.d_out.at(g.out_off);
        const a_w: c_int = g.w;
        const a_h: c_int = g.h;
        const a_stride: c_int = g.stride;
        try s.stream.launch(d.fn_fused, .{
            .grid = .{ d.fused_blocks, 1, 1 },
            .block = .{ d.wpb * d.warp_size, 1, 1 },
        }, .{ a_spatial, a_padded, a_w, a_h });
        try s.stream.launch(d.fn_col2im, .{
            .grid = .{ ceilDiv(@intCast(g.pw), d.warp_size), ceilDiv(@intCast(g.ph), d.wpb), 1 },
            .block = .{ d.warp_size, d.wpb, 1 },
        }, .{ a_out, a_spatial, a_w, a_h, a_stride });
    }

    if (d.direct_d2h) {
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const g = &d.geom[p];
            const dstp = dst.getWriteSlice(p);
            std.debug.assert(dstp.len == @as(usize, @intCast(g.h)) * @as(usize, @intCast(g.stride)) * d.bytes);
            try s.stream.memcpyDtoH(dstp.ptr, s.d_out.at(g.out_off), dstp.len);
        }
        try s.stream.sync();
        return;
    }

    try s.stream.sync();

    p = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const g = &d.geom[p];
        const len = @as(usize, @intCast(g.h)) * @as(usize, @intCast(g.stride)) * d.bytes;
        try s.stream.memcpyDtoH(s.h_out.ptr + g.out_off, s.d_out.at(g.out_off), len);
    }
    try s.stream.sync();

    p = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;
        const g = &d.geom[p];
        const dstp = dst.getWriteSlice(p);
        std.debug.assert(dstp.len == @as(usize, @intCast(g.h)) * @as(usize, @intCast(g.stride)) * d.bytes);
        @memcpy(dstp, (s.h_out.ptr + g.out_off)[0..dstp.len]);
    }
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    const r = d.radius;
    const ni: i32 = @intCast(n);
    const nf: i32 = d.vi.numFrames;

    if (activation_reason == .Initial) {
        var i: i32 = @max(ni - r, 0);
        const end: i32 = @min(ni + r, nf - 1);
        while (i <= end) : (i += 1) {
            zapi.requestFrameFilter(@intCast(i), d.node);
        }
    } else if (activation_reason == .AllFramesReady) {
        const tw: usize = @intCast(2 * r + 1);
        var frames: [7]ZFrame = undefined;
        var win: CacheWin = .{ .tw = tw };
        for (0..tw) |t| {
            const idx: i32 = @min(@max(ni - r + @as(i32, @intCast(t)), 0), nf - 1);
            win.keys[t] = idx;
            frames[t] = zapi.initZFrame(d.node, @intCast(idx));
        }
        defer for (0..tw) |t| frames[t].deinit();

        const center = frames[@intCast(r)];
        const dst = center.newVideoFrame2(d.process);

        if (d.use_cache) {
            d.cache.acquire(win.keys[0..tw], win.idx[0..tw], win.load[0..tw]);
            for (0..tw) |t| win.published[t] = false;
        }
        defer if (d.use_cache) {
            // Abandon unpublished claimed slots (else ready=false forever).
            for (0..tw) |t| {
                if (win.load[t] and !win.published[t]) d.cache.abandon(win.idx[t]);
            }
            d.cache.release(win.idx[0..tw]);
        };

        var up: Upload = .{};
        const num_planes: u32 = @intCast(d.vi.format.numPlanes);
        var stg: ?*PadStage = null;
        defer if (stg) |sg| d.spool.release(sg);
        if (d.staged) {
            stg = d.spool.acquire();
            for (0..tw) |t| {
                if (d.use_cache and !win.load[t]) continue;
                var p: u32 = 0;
                while (p < num_planes) : (p += 1) {
                    if (!d.process[p]) continue;
                    const src = frames[t].getReadSlice(p);
                    const dstb = stg.?.h.ptr + t * d.raw_bytes + d.raw_off[p];
                    @memcpy(dstb[0..src.len], src);
                    up.ptr[t][p] = dstb;
                }
            }
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, &up, dst, if (d.use_cache) &win else null) catch |err| {
            zapi.setFilterError("DFTTest: process frame failed.");
            std.log.err("vszipcu DFTTest process frame failed: {t}", .{err});
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
    d.spool.deinit();
    d.cache.deinit();
    d.module.deinit();
    d.dev.pop();
    d.dev.deinit();
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn initCuda(d: *Data, device_id: i32, num_streams: usize, n_threads: usize, cfg: *KernelConfig) CreateError!*Data {
    d.dev = try cu.Device.init(device_id);
    errdefer d.dev.deinit();
    try d.dev.push();
    defer d.dev.pop();

    cfg.warp_size = @intCast(try d.dev.warpSize());
    d.warp_size = cfg.warp_size;
    cfg.warps_per_block = d.wpb;

    const prefix = genKernelPrefix(cfg) catch return error.OutOfMemory;
    defer allocator.free(prefix);

    d.module = try cu.compile(d.dev, .{
        .text = kernel_source,
        .defines = prefix,
        .name = "dfttest.cu",
    }, nvrtc_opts);
    errdefer {
        d.module.deinit();
        d.module = .{};
    }
    d.fn_fused = try d.module.function("fused");
    d.fn_col2im = try d.module.function("col2im");
    d.fn_pad = try d.module.function("reflect_pad");

    const num_sms = try d.dev.multiProcessorCount();
    const max_blocks = try d.fn_fused.maxActiveBlocksPerSM(@intCast(d.wpb * d.warp_size), 0);
    d.fused_blocks = @intCast(num_sms * max_blocks);

    const data = allocator.create(Data) catch return error.OutOfMemory;
    errdefer allocator.destroy(data);
    data.* = d.*;
    data.pool = .{};
    data.spool = .{};
    data.pool.prime(data, num_streams) catch {
        data.pool.deinit();
        return error.OutOfMemory;
    };
    data.pool.prewarm(num_streams) catch |err| {
        data.pool.deinit();
        return err;
    };
    {
        const n_stage = num_streams + 3;
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
    }

    if (data.use_cache) {
        const tw: usize = @intCast(2 * data.radius + 1);
        const want = tw + @max(num_streams, n_threads) + @as(usize, @intCast(2 * data.radius));
        const slots = allocator.alloc(framecache.CacheSlot, want) catch return error.OutOfMemory;
        var n_ok: usize = 0;
        for (slots) |*slot| {
            slot.* = .{};
            slot.buf = cu.DeviceBuffer.alloc(data.slot_bytes) catch break;
            slot.ev = cu.Event.init() catch {
                slot.buf.deinit();
                slot.buf = .{};
                break;
            };
            n_ok += 1;
        }
        if (n_ok < tw) {
            var i: usize = n_ok;
            while (i > 0) {
                i -= 1;
                slots[i].ev.deinit();
                slots[i].buf.deinit();
            }
            allocator.free(slots);
            data.use_cache = false;
            std.log.warn("vszipcu DFTTest: not enough device memory for the padded-source cache; running uncached.", .{});
        } else {
            data.cache.slots = if (n_ok == want) slots else blk: {
                const shrunk = allocator.realloc(slots, n_ok) catch slots[0..n_ok];
                break :blk shrunk;
            };
            if (n_ok < want) {
                std.log.warn("vszipcu DFTTest: padded-source cache shrunk to {d}/{d} slots (low device memory).", .{ n_ok, want });
            }
        }
    }
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
    const depth_ok = (fmt.sampleType == .Float and bits == 32) or
        (fmt.sampleType == .Integer and bits >= 8 and bits <= 16);
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (fmt.colorFamily != .Gray and fmt.colorFamily != .YUV and fmt.colorFamily != .RGB))
    {
        return map_out.setError("DFTTest: input must be 8-16 bit integer or 32 bit float, Gray/YUV/RGB, constant format.");
    }
    d.bytes = @intCast(fmt.bytesPerSample);

    const ftype = map_in.getValue(i32, "ftype") orelse 0;
    if (ftype < 0 or ftype > 4) return map_out.setError("DFTTest: ftype must be 0, 1, 2, 3, or 4.");
    var sigma = map_in.getValue(f64, "sigma") orelse 8.0;
    var sigma2 = map_in.getValue(f64, "sigma2") orelse 8.0;
    var pmin = map_in.getValue(f64, "pmin") orelse 0.0;
    var pmax = map_in.getValue(f64, "pmax") orelse 500.0;
    if (!math.isFinite(sigma) or !math.isFinite(sigma2) or !math.isFinite(pmin) or !math.isFinite(pmax))
        return map_out.setError("DFTTest: sigma/sigma2/pmin/pmax must be finite.");
    const sbsize = map_in.getValue(i32, "sbsize") orelse 16;
    if (sbsize != 16) return map_out.setError("DFTTest: sbsize must be 16 (NVRTC-backend port).");
    const sosize = map_in.getValue(i32, "sosize") orelse 12;
    if (sosize < 0 or sosize > 15) return map_out.setError("DFTTest: sosize must be 0..15.");
    if (sosize > 8 and @mod(sbsize, sbsize - sosize) != 0)
        return map_out.setError("DFTTest: spatial overlap > 50% requires that sbsize-sosize is a divisor of sbsize.");
    const tbsize = map_in.getValue(i32, "tbsize") orelse 3;
    if (tbsize < 1 or tbsize > 7) return map_out.setError("DFTTest: tbsize must be odd, 1..7 (temporal radius 0..3).");
    if (@mod(tbsize, 2) == 0) return map_out.setError("DFTTest: tbsize must be odd (dfttest2 silently aliases even values to tbsize-1).");
    const swin = map_in.getValue(i32, "swin") orelse 0;
    const twin = map_in.getValue(i32, "twin") orelse 7;
    if (swin < 0 or swin > 11 or twin < 0 or twin > 11)
        return map_out.setError("DFTTest: swin/twin must be 0..11.");
    const sbeta = map_in.getValue(f64, "sbeta") orelse 2.5;
    const tbeta = map_in.getValue(f64, "tbeta") orelse 2.5;
    if (!math.isFinite(sbeta) or !math.isFinite(tbeta))
        return map_out.setError("DFTTest: sbeta/tbeta must be finite.");
    const zmean = (map_in.getValue(i32, "zmean") orelse 1) != 0;
    const f0beta = map_in.getValue(f64, "f0beta") orelse 1.0;
    const ssystem = map_in.getValue(i32, "ssystem") orelse 0;
    if (ssystem < 0 or ssystem > 1) return map_out.setError("DFTTest: ssystem must be 0 or 1.");

    const slocation = map_in.getFloatArray("slocation");
    const ssx = map_in.getFloatArray("ssx");
    const ssy = map_in.getFloatArray("ssy");
    const sst = map_in.getFloatArray("sst");
    inline for (.{ slocation, ssx, ssy, sst }, .{ "slocation", "ssx", "ssy", "sst" }) |arr, nm| {
        if (arr) |a| if (a.len % 2 != 0 or a.len < 2)
            return map_out.setError("DFTTest: number of elements in " ++ nm ++ " must be a non-zero multiple of 2.");
    }

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("DFTTest: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("DFTTest: num_streams must be 1..32.");
    };
    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 1;

    d.radius = @divTrunc(tbsize - 1, 2);
    d.block_step = sbsize - sosize;

    const num_planes: usize = @intCast(fmt.numPlanes);
    if (map_in.numElements("planes")) |ne| {
        var e: u32 = 0;
        while (e < ne) : (e += 1) {
            const idx = map_in.getValue2(i32, "planes", e).?;
            if (idx < 0 or idx >= fmt.numPlanes) return map_out.setError("DFTTest: plane index out of range.");
            if (d.process[@intCast(idx)]) return map_out.setError("DFTTest: plane specified twice.");
            d.process[@intCast(idx)] = true;
        }
    } else {
        for (0..num_planes) |pi| d.process[pi] = true;
    }

    // FILTER_TYPE 5: unscaled f0beta (wscale-scaled pmin → NaN at pmin=0).
    var filter_type: i32 = ftype;
    if (ftype == 0) {
        if (@abs(f0beta - 1.0) < 0.00005) {
            filter_type = 0;
        } else if (@abs(f0beta - 0.5) < 0.0005) {
            filter_type = 6;
        } else {
            filter_type = 5;
        }
    }

    const strides = vsutil.strideFromVi(d.vi);
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    const tw: usize = @intCast(2 * d.radius + 1);
    {
        var padded_sum: usize = 0;
        var spatial_sum: usize = 0;
        var out_sum: usize = 0;
        var pi: usize = 0;
        while (pi < num_planes) : (pi += 1) {
            if (!d.process[pi]) continue;
            const g = &d.geom[pi];
            g.w = if (pi == 0) d.vi.width else d.vi.width >> subW;
            g.h = if (pi == 0) d.vi.height else d.vi.height >> subH;
            g.stride = @intCast(if (pi == 0) strides[0] else strides[1]);
            g.pw = calcPadSize(g.w, d.block_step);
            g.ph = calcPadSize(g.h, d.block_step);
            g.hn = calcPadNum(g.w, d.block_step);
            g.vn = calcPadNum(g.h, d.block_step);

            // Single-fold reflect_pad requires pad <= dim-1.
            const ox = @divTrunc(g.pw - g.w, 2);
            const oy = @divTrunc(g.ph - g.h, 2);
            if (ox > g.w - 1 or (g.pw - g.w - ox) > g.w - 1 or oy > g.h - 1 or (g.ph - g.h - oy) > g.h - 1)
                return map_out.setError("DFTTest: a processed plane is too small for the padded block layout.");

            const pad_elems = @as(usize, @intCast(g.pw)) * @as(usize, @intCast(g.ph));
            const nblk = @as(usize, @intCast(g.hn)) * @as(usize, @intCast(g.vn));
            g.padded_off = padded_sum;
            g.spatial_off = spatial_sum;
            g.out_off = out_sum;
            g.cache_off = d.slot_bytes;
            d.raw_off[pi] = d.raw_bytes;
            padded_sum += tw * pad_elems * d.bytes;
            spatial_sum += nblk * 256 * @sizeOf(f32);
            out_sum += @as(usize, @intCast(g.h)) * @as(usize, @intCast(g.stride)) * d.bytes;
            d.slot_bytes += pad_elems * d.bytes;
            d.raw_bytes += @as(usize, @intCast(g.h)) * @as(usize, @intCast(g.stride)) * d.bytes;

            if (tw * pad_elems >= (1 << 31) or nblk * 256 >= (1 << 31))
                return map_out.setError("DFTTest: frame too large (padded plane exceeds 2^31 elements).");
        }
        if (padded_sum == 0) return map_out.setError("DFTTest: no planes to process.");
        d.padded_bytes = padded_sum;
        d.spatial_bytes = spatial_sum;
        d.out_bytes = out_sum;
        if (padded_sum >= (1 << 32) or spatial_sum >= (1 << 32))
            return map_out.setError("DFTTest: frame too large (device buffers exceed 4 GiB).");
    }

    const window = getWindow(d.radius, d.block_step, swin, sbeta, twin, tbeta) catch
        return map_out.setError("DFTTest: out of memory.");
    defer allocator.free(window);

    const sq = allocator.alloc(f64, window.len) catch return map_out.setError("DFTTest: out of memory.");
    defer allocator.free(sq);
    for (sq, window) |*s, wv| s.* = wv * wv;
    const wscale = fsum(sq);

    const sigma_is_scalar = slocation == null and ssx == null and ssy == null and sst == null;
    var sigma_array: []f64 = &.{};
    defer if (sigma_array.len > 0) allocator.free(sigma_array);

    if (!sigma_is_scalar) {
        const norm: Norm = if (slocation != null and ssystem == 1)
            .identity
        else if (tbsize == 1)
            .sqrt
        else
            .cbrt;

        var fx: SigmaFunc = undefined;
        var fy: SigmaFunc = undefined;
        var ft: SigmaFunc = undefined;
        var shared = false;
        if (slocation) |sl| {
            fx = SigmaFunc.initPacks(sl, norm) catch return map_out.setError("DFTTest: out of memory.");
            fy = fx;
            ft = fx;
            shared = true;
        } else {
            fx = if (ssx) |a| SigmaFunc.initPacks(a, norm) catch return map_out.setError("DFTTest: out of memory.") else SigmaFunc.initConst(norm, sigma);
            fy = if (ssy) |a| SigmaFunc.initPacks(a, norm) catch return map_out.setError("DFTTest: out of memory.") else SigmaFunc.initConst(norm, sigma);
            ft = if (sst) |a| SigmaFunc.initPacks(a, norm) catch return map_out.setError("DFTTest: out of memory.") else SigmaFunc.initConst(norm, sigma);
        }
        defer {
            fx.deinit();
            if (!shared) {
                fy.deinit();
                ft.deinit();
            }
        }

        sigma_array = allocator.alloc(f64, tw * 16 * 9) catch return map_out.setError("DFTTest: out of memory.");
        var idx: usize = 0;
        var fail = false;
        if (ssystem == 0) {
            var t: i32 = 0;
            while (t < 2 * d.radius + 1) : (t += 1) {
                const st = getSigma(t, 2 * d.radius + 1, &ft) orelse {
                    fail = true;
                    break;
                };
                var y: i32 = 0;
                while (y < BS and !fail) : (y += 1) {
                    const sy = getSigma(y, BS, &fy) orelse {
                        fail = true;
                        break;
                    };
                    var x: i32 = 0;
                    while (x < @divTrunc(BS, 2) + 1) : (x += 1) {
                        const sx = getSigma(x, BS, &fx) orelse {
                            fail = true;
                            break;
                        };
                        sigma_array[idx] = st * sy * sx;
                        idx += 1;
                    }
                }
                if (fail) break;
            }
        } else {
            const ndim: f64 = if (d.radius > 0) 3.0 else 2.0;
            var t: i32 = 0;
            while (t < 2 * d.radius + 1) : (t += 1) {
                const lt = getLocation(t, 2 * d.radius + 1);
                var y: i32 = 0;
                while (y < BS and !fail) : (y += 1) {
                    const ly = getLocation(y, BS);
                    var x: i32 = 0;
                    while (x < @divTrunc(BS, 2) + 1) : (x += 1) {
                        const lx = getLocation(x, BS);
                        const location = @sqrt((lt * lt + ly * ly + lx * lx) / ndim);
                        sigma_array[idx] = ft.eval(location) orelse {
                            fail = true;
                            break;
                        };
                        idx += 1;
                    }
                }
                if (fail) break;
            }
        }
        if (fail)
            return map_out.setError("DFTTest: slocation/ssx/ssy/sst must cover the full [0, 1] frequency range.");
    }

    if (ftype < 2) {
        if (sigma_is_scalar) {
            sigma *= wscale;
        } else {
            for (sigma_array) |*s| s.* *= wscale;
        }
        sigma2 *= wscale;
    }
    pmin *= wscale;
    pmax *= wscale;

    var window_freq: []f64 = &.{};
    defer if (window_freq.len > 0) allocator.free(window_freq);
    if (zmean) {
        const scaled = allocator.alloc(f64, window.len) catch return map_out.setError("DFTTest: out of memory.");
        defer allocator.free(scaled);
        for (scaled, window) |*s, wv| s.* = wv * 255.0;
        window_freq = rdftTables(d.radius, scaled) catch return map_out.setError("DFTTest: out of memory.");
    }

    d.wpb = 4;

    d.direct_d2h = num_streams == 1;
    d.staged = true;
    d.use_cache = d.radius > 0;

    var kcfg: KernelConfig = .{
        .radius = d.radius,
        .block_step = d.block_step,
        .warps_per_block = d.wpb,
        .tiles_per_warp = d.tpw,
        .warp_size = 32,
        .sample_type = fmt.sampleType,
        .bits = bits,
        .filter_type = filter_type,
        .zmean = zmean,
        .sigma_scalar = if (sigma_is_scalar) sigma else null,
        .sigma_array = sigma_array,
        .sigma2 = sigma2,
        .pmin = pmin,
        .pmax = pmax,
        .beta = f0beta,
        .window = window,
        .window_freq = window_freq,
    };

    var core_info: vs.CoreInfo = .{};
    zapi.getCoreInfo(core, &core_info);
    const n_threads: usize = @intCast(@max(core_info.numThreads, 1));

    const data = initCuda(&d, device_id, num_streams, n_threads, &kcfg) catch |err| {
        map_out.setError(switch (err) {
            error.InvalidDeviceID => "DFTTest: invalid device ID.",
            error.Nvrtc => "DFTTest: CUDA kernel compilation failed (see log).",
            error.NvrtcNotFound => "DFTTest: could not locate NVRTC (put nvrtc64_130_0.dll next to the plugin, or: pip install nvidia-cuda-nvrtc).",
            error.OutOfDeviceMemory => "DFTTest: out of device memory.",
            else => "DFTTest: CUDA initialization failed (see log).",
        });
        std.log.err("vszipcu DFTTest init failed: {t}", .{err});
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = if (d.radius > 0) .General else .StrictSpatial },
    };
    zapi.createVideoFilter(out, "DFTTest", d.vi, getFrame, free, .Parallel, &dep, data);
}
