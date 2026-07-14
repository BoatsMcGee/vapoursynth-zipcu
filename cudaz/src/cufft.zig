//! cuFFT: runtime-loaded bindings + the plan surface a filter actually needs.
//!
//! Status: this exists so a cuFFT-backed filter (vs-dfttest2's `Backend.cuFFT`) can be
//! written without adding a link-time dependency. It covers exactly the surface that
//! backend uses — plan_many (R2C/C2R), work-area control, stream binding, exec, destroy —
//! and nothing else. It is NOT exercised by any shipped filter yet.
//!
//! Types are hand-declared rather than @cImport'd: cufft.h is not needed to BUILD a plugin
//! that only runtime-loads cuFFT, and requiring it would make the toolkit's cufft.h a hard
//! build dependency for every consumer of cudaz. The ABI here is stable and small.
//!
//! Loading follows the same rule as NVRTC: never import-link it (a plugin's DLL search
//! path excludes PATH), load it at first use from the plugin's own directory, then the pip
//! wheel. Upstream dfttest2 solves the same problem on Windows with delay-loading plus a
//! `vsmlrt-cuda\` sidecar probe; the runtime-load path here is the portable equivalent.

const std = @import("std");
const builtin = @import("builtin");
const dynlib = @import("dynlib.zig");
const driver = @import("driver.zig");

const log = std.log.scoped(.cudaz);

pub const Error = error{
    Cufft,
    CufftNotFound,
    OutOfMemory,
};

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

pub const Handle = c_int;

pub const Result = enum(c_int) {
    success = 0,
    invalid_plan = 1,
    alloc_failed = 2,
    invalid_type = 3,
    invalid_value = 4,
    internal_error = 5,
    exec_failed = 6,
    setup_failed = 7,
    invalid_size = 8,
    unaligned_data = 9,
    incomplete_parameter_list = 10,
    invalid_device = 11,
    parse_error = 12,
    no_workspace = 13,
    not_implemented = 14,
    license_error = 15,
    not_supported = 16,
    _,
};

pub const Type = enum(c_int) {
    r2c = 0x2a, // real -> complex (interleaved)
    c2r = 0x2c, // complex (interleaved) -> real
    c2c = 0x29,
    d2z = 0x6a,
    z2d = 0x6c,
    z2z = 0x69,
};

pub const Real = f32;
pub const Complex = extern struct { x: f32, y: f32 };

// ---------------------------------------------------------------------------
// Runtime binding
// ---------------------------------------------------------------------------

const Table = struct {
    // The EXTENSIBLE plan API. `cufftPlanMany` is the legacy combined call: it creates the
    // plan AND allocates a work area in one go, so `cufftSetAutoAllocation` afterwards is
    // too late — NVIDIA specifies it must be called after cufftCreate and before
    // cufftMakePlan*. Hence Create + SetAutoAllocation + MakePlanMany here.
    Create: *const fn (plan: *Handle) callconv(.c) Result,
    // inembed/onembed are `[*c]` (already nullable), NOT `?[*c]` — a non-pointer optional has
    // no guaranteed in-memory representation and Zig rejects it in a C calling convention.
    MakePlanMany: *const fn (
        plan: Handle,
        rank: c_int,
        n: [*c]c_int,
        inembed: [*c]c_int,
        istride: c_int,
        idist: c_int,
        onembed: [*c]c_int,
        ostride: c_int,
        odist: c_int,
        type_: Type,
        batch: c_int,
        work_size: *usize,
    ) callconv(.c) Result,
    Destroy: *const fn (plan: Handle) callconv(.c) Result,
    SetStream: *const fn (plan: Handle, stream: driver.c.CUstream) callconv(.c) Result,
    SetAutoAllocation: *const fn (plan: Handle, auto_allocate: c_int) callconv(.c) Result,
    SetWorkArea: *const fn (plan: Handle, work_area: ?*anyopaque) callconv(.c) Result,
    GetSize: *const fn (plan: Handle, work_size: *usize) callconv(.c) Result,
    ExecR2C: *const fn (plan: Handle, idata: [*]Real, odata: [*]Complex) callconv(.c) Result,
    ExecC2R: *const fn (plan: Handle, idata: [*]Complex, odata: [*]Real) callconv(.c) Result,
    ExecC2C: *const fn (plan: Handle, idata: [*]Complex, odata: [*]Complex, direction: c_int) callconv(.c) Result,
    GetVersion: *const fn (version: *c_int) callconv(.c) Result,
};

const Bind = dynlib.Binder(Table, "cufft");

/// cuFFT's soname tracks the LIBRARY version, not the CUDA major — the CUDA 13 wheel still
/// ships `cufft64_12.dll`. Try the plausible set, newest first.
const names: []const []const u8 = if (builtin.os.tag == .windows)
    &.{ "cufft64_12.dll", "cufft64_11.dll", "cufft64_10.dll" }
else
    &.{ "libcufft.so.12", "libcufft.so.11", "libcufft.so" };

/// CUDA 13 (`nvidia-cufft`) puts it under the shared `nvidia/cu13/...` tree; CUDA 12
/// (`nvidia-cufft-cu12`) gave it its own.
const wheel_subdirs: []const []const u8 = if (builtin.os.tag == .windows)
    &.{ "nvidia\\cu13\\bin\\x86_64", "nvidia\\cufft\\bin" }
else
    &.{ "nvidia/cu13/lib", "nvidia/cufft/lib" };

pub fn ensure(anchor: *const anyopaque, system_search: bool) Error!void {
    Bind.ensure(.{
        .names = names,
        .wheel_subdirs = wheel_subdirs,
        .anchor = anchor,
        .system_search = system_search,
    }) catch {
        log.err("could not locate cuFFT ({s}); try: pip install nvidia-cufft", .{names[0]});
        return error.CufftNotFound;
    };
}

pub fn available() bool {
    return Bind.loaded();
}

fn fns() *const Table {
    return &Bind.table;
}

fn check(r: Result) Error!void {
    if (r == .success) return;
    log.err("cuFFT error: {t}", .{r});
    return error.Cufft;
}

pub fn version() Error!i32 {
    var v: c_int = 0;
    try check(fns().GetVersion(&v));
    return v;
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

pub const PlanConfig = struct {
    /// Transform dimensions, slowest-varying first (rank = dims.len, 1..3).
    dims: []const c_int,
    /// Number of independent transforms. For a tile-based filter this is the tile count.
    batch: c_int,
    type: Type,
    /// Default embedding (null inembed/onembed, stride 1, dist 0) — what dfttest2 uses:
    /// the tiles are contiguous, so cuFFT derives the distances itself.
    embed: ?Embed = null,
    /// Let cuFFT own the scratch buffer.
    ///
    /// Set this FALSE when several plans live in one filter: each would otherwise allocate
    /// its own scratch, which on a large batch is a lot of device memory doing nothing.
    /// Instead query `work_size` on every plan, allocate ONE buffer of the max, and point
    /// them all at it with `setWorkArea()`. That is what dfttest2's cuFFT backend does —
    /// and it is only sound because the plans run **sequentially on one stream**. cuFFT
    /// requires an exclusive work area per *concurrent* execution: two plans running on two
    /// streams must NOT share one buffer.
    auto_allocate: bool = true,

    pub const Embed = struct {
        inembed: []c_int,
        istride: c_int,
        idist: c_int,
        onembed: []c_int,
        ostride: c_int,
        odist: c_int,
    };
};

/// An FFT plan.
///
/// Built with cuFFT's **extensible** API — `cufftCreate` → `cufftSetAutoAllocation` →
/// `cufftMakePlanMany` — and not the legacy `cufftPlanMany`. That ordering is not a style
/// choice: `cufftPlanMany` creates the plan *and* allocates its work area in a single call,
/// so disabling auto-allocation afterwards is too late, and NVIDIA specifies
/// `cufftSetAutoAllocation` must be called after `cufftCreate` and before `cufftMakePlan*`.
pub const Plan = struct {
    handle: Handle = 0,
    /// Scratch cuFFT needs for this plan, as reported by `cufftMakePlanMany`. Zero is normal
    /// for small power-of-two transforms.
    work_size: usize = 0,

    pub fn init(cfg: PlanConfig) Error!Plan {
        std.debug.assert(cfg.dims.len >= 1 and cfg.dims.len <= 3);
        var p: Plan = .{};

        try check(fns().Create(&p.handle));
        errdefer _ = fns().Destroy(p.handle);

        if (!cfg.auto_allocate) {
            try check(fns().SetAutoAllocation(p.handle, 0));
        }

        const dims = cfg.dims;
        const e = cfg.embed;
        try check(fns().MakePlanMany(
            p.handle,
            @intCast(dims.len),
            @constCast(dims.ptr),
            if (e) |x| x.inembed.ptr else null,
            if (e) |x| x.istride else 1,
            if (e) |x| x.idist else 0,
            if (e) |x| x.onembed.ptr else null,
            if (e) |x| x.ostride else 1,
            if (e) |x| x.odist else 0,
            cfg.type,
            cfg.batch,
            &p.work_size,
        ));
        return p;
    }

    pub fn deinit(self: Plan) void {
        if (self.handle != 0) _ = fns().Destroy(self.handle);
    }

    /// Re-query the scratch size. `Plan.work_size` already holds what `cufftMakePlanMany`
    /// reported; this is here for the cases where a later call can change it.
    pub fn workSize(self: Plan) Error!usize {
        var n: usize = 0;
        try check(fns().GetSize(self.handle, &n));
        return n;
    }

    pub fn setWorkArea(self: Plan, buf: driver.DeviceBuffer) Error!void {
        return check(fns().SetWorkArea(self.handle, @ptrFromInt(buf.ptr)));
    }

    pub fn setStream(self: Plan, s: driver.Stream) Error!void {
        return check(fns().SetStream(self.handle, s.handle));
    }

    pub fn execR2C(self: Plan, in: driver.c.CUdeviceptr, out: driver.c.CUdeviceptr) Error!void {
        return check(fns().ExecR2C(self.handle, @ptrFromInt(in), @ptrFromInt(out)));
    }

    pub fn execC2R(self: Plan, in: driver.c.CUdeviceptr, out: driver.c.CUdeviceptr) Error!void {
        return check(fns().ExecC2R(self.handle, @ptrFromInt(in), @ptrFromInt(out)));
    }

    pub const Direction = enum(c_int) { forward = -1, inverse = 1 };

    pub fn execC2C(self: Plan, in: driver.c.CUdeviceptr, out: driver.c.CUdeviceptr, dir: Direction) Error!void {
        return check(fns().ExecC2C(self.handle, @ptrFromInt(in), @ptrFromInt(out), @intFromEnum(dir)));
    }
};
