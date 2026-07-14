//! NVRTC: runtime-loaded, and the compile → module pipeline.
//!
//! NVRTC is never import-linked (see dynlib.zig for why). Types and constants still come
//! from the @cImport of nvrtc.h, so the bound function-pointer types are derived from the
//! header and cannot drift; only the entry points actually used are bound.
//!
//! `compile()` is the pipeline every filter in this repo had copy-pasted: prepend the
//! `#define` block, create the program, pick CUBIN-vs-PTX from the device's compute
//! capability against `nvrtcGetSupportedArchs()`, compile, dump the log on failure, and
//! load the image as a module.
//!
//! Compile OPTIONS are deliberately the caller's business. They are not a detail: for a
//! port that must be bit-exact against a reference plugin, `-use_fast_math` and the fmad
//! contraction mode are part of the parity contract, and they differ per filter. A
//! library that picked them would be picking the output.

const std = @import("std");
const dynlib = @import("dynlib.zig");
const driver = @import("driver.zig");

pub const c = @import("c.zig").nvrtc;

const log = std.log.scoped(.cudaz);

pub const Error = error{
    Nvrtc,
    NvrtcNotFound,
    OutOfMemory,
} || driver.Error;

// ---------------------------------------------------------------------------
// Runtime binding
// ---------------------------------------------------------------------------

fn Fn(comptime name: []const u8) type {
    return *const @TypeOf(@field(c, "nvrtc" ++ name));
}

/// Field names are the nvrtc symbol names minus the `nvrtc` prefix; dynlib.Binder binds
/// each from the loaded library.
const Table = struct {
    CreateProgram: Fn("CreateProgram"),
    DestroyProgram: Fn("DestroyProgram"),
    CompileProgram: Fn("CompileProgram"),
    GetProgramLogSize: Fn("GetProgramLogSize"),
    GetProgramLog: Fn("GetProgramLog"),
    GetPTXSize: Fn("GetPTXSize"),
    GetPTX: Fn("GetPTX"),
    GetCUBINSize: Fn("GetCUBINSize"),
    GetCUBIN: Fn("GetCUBIN"),
    GetNumSupportedArchs: Fn("GetNumSupportedArchs"),
    GetSupportedArchs: Fn("GetSupportedArchs"),
    GetErrorString: Fn("GetErrorString"),
};

const Bind = dynlib.Binder(Table, "nvrtc");

/// The sonames NVRTC ships under, newest first. A missing file just fails to open, so
/// listing the CUDA-12 names too costs nothing and lets a CUDA-12-only install work.
const names: []const []const u8 = if (@import("builtin").os.tag == .windows)
    &.{ "nvrtc64_130_0.dll", "nvrtc64_120_0.dll" }
else
    &.{ "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so" };

/// Wheel layouts, newest first. CUDA 13 (`nvidia-cuda-nvrtc`) puts every component under one
/// `nvidia/cu13/...` tree; CUDA 12 (`nvidia-cuda-nvrtc-cu12`) gave each its own.
const wheel_subdirs: []const []const u8 = if (@import("builtin").os.tag == .windows)
    &.{ "nvidia\\cu13\\bin\\x86_64", "nvidia\\cuda_nvrtc\\bin" }
else
    &.{ "nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib" };

/// Bind NVRTC on first use. `anchor` must be a function in YOUR module — it is how the
/// loader finds your module's own directory (a plugin's directory is on no loader search
/// path, so any relative lookup has to be explicit).
///
/// `system_search` should stay false for a plugin and be set true for an application.
pub fn ensure(anchor: *const anyopaque, system_search: bool) Error!void {
    Bind.ensure(.{
        .names = names,
        .wheel_subdirs = wheel_subdirs,
        .anchor = anchor,
        .system_search = system_search,
    }) catch {
        log.err("could not locate NVRTC ({s}); try: pip install nvidia-cuda-nvrtc", .{names[0]});
        return error.NvrtcNotFound;
    };
}

fn fns() *const Table {
    return &Bind.table;
}

pub fn check(result: c.nvrtcResult) Error!void {
    if (result == c.NVRTC_SUCCESS) return;
    log.err("NVRTC error: {s}", .{fns().GetErrorString(result)});
    return error.Nvrtc;
}

pub fn errorString(result: c.nvrtcResult) [*:0]const u8 {
    return fns().GetErrorString(result);
}

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------

pub const Source = struct {
    /// The kernel text (typically `@embedFile("foo.cu")`).
    text: []const u8,
    /// Prepended verbatim — the `#define` block that bakes this instance's configuration
    /// into the kernel. Constants baked here turn hot loops into compile-time trip counts;
    /// float constants must be printed with `{x}` (Zig's exact hexfloat) or a decimal
    /// round-trip becomes a parity bug.
    defines: []const u8 = "",
    /// Program name, used in NVRTC's diagnostics (e.g. "eedi3.cu").
    name: [*:0]const u8,
};

pub const Options = struct {
    /// Everything AFTER `-arch=...`, which is prepended for you. These are a parity
    /// contract, not a tuning detail — pass exactly what the reference used.
    extra: []const [*:0]const u8 = &.{},
    /// Name used in the "compilation failed" log line.
    log_name: []const u8 = "kernel",
    /// Force PTX even when the device's arch is natively supported (rarely wanted; the
    /// driver would JIT it).
    force_ptx: bool = false,
};

/// Compile `src` for `dev` and load the result as a module.
///
/// Arch selection: if this NVRTC knows the device's compute capability, emit a native
/// CUBIN (`-arch=sm_XX`, no driver JIT). If the device is NEWER than anything this NVRTC
/// knows, emit PTX for the newest arch it does know (`-arch=compute_XX`) and let the
/// driver JIT it — that is what lets a plugin built against CUDA 13 still run on a future
/// GPU without shipping fatbins.
pub fn compile(dev: driver.Device, src: Source, opts: Options, allocator: std.mem.Allocator) Error!driver.Module {
    const source = std.fmt.allocPrintSentinel(allocator, "{s}{s}", .{ src.defines, src.text }, 0) catch return error.OutOfMemory;
    defer allocator.free(source);

    var program: c.nvrtcProgram = null;
    try check(fns().CreateProgram(&program, source.ptr, src.name, 0, null, null));
    defer _ = fns().DestroyProgram(&program);

    const cc = try dev.computeCapability();

    var num_archs: c_int = 0;
    try check(fns().GetNumSupportedArchs(&num_archs));
    if (num_archs <= 0) {
        // Should never happen, but the alternative is indexing an empty slice.
        log.err("NVRTC reports no supported architectures", .{});
        return error.Nvrtc;
    }
    const archs = allocator.alloc(c_int, @intCast(num_archs)) catch return error.OutOfMemory;
    defer allocator.free(archs);
    try check(fns().GetSupportedArchs(archs.ptr)); // ascending

    // A native CUBIN requires that this NVRTC knows THIS EXACT arch. `cc <= newest` is not
    // the same test: NVIDIA's list has gaps, and `-arch=sm_<gap>` is a hard compile error,
    // not a graceful fallback. When the arch is unknown — whether it sits in a gap or is
    // newer than anything this NVRTC has heard of — emit PTX for the newest supported arch
    // *at or below* it and let the driver JIT forward. That is what keeps a plugin built
    // against one toolkit running on a future GPU.
    var exact = false;
    var ptx_arch = archs[0];
    for (archs) |a| {
        if (a == cc) exact = true;
        if (a <= cc) ptx_arch = a;
    }
    const cubin = !opts.force_ptx and exact;

    var arch_buf: [32]u8 = undefined;
    const arch_opt = std.fmt.bufPrintSentinel(&arch_buf, "-arch={s}{d}", .{
        if (cubin) @as([]const u8, "sm_") else "compute_",
        if (cubin) cc else ptx_arch,
    }, 0) catch unreachable;

    var argv = std.ArrayList([*c]const u8).initCapacity(allocator, opts.extra.len + 1) catch return error.OutOfMemory;
    defer argv.deinit(allocator);
    argv.appendAssumeCapacity(arch_opt.ptr);
    for (opts.extra) |o| argv.appendAssumeCapacity(o);

    if (fns().CompileProgram(program, @intCast(argv.items.len), argv.items.ptr) != c.NVRTC_SUCCESS) {
        // The program log is the only way a user can report a kernel problem; always emit it.
        var log_size: usize = 0;
        if (fns().GetProgramLogSize(program, &log_size) == c.NVRTC_SUCCESS and log_size > 1) {
            if (allocator.allocSentinel(u8, log_size, 0) catch null) |buf| {
                defer allocator.free(buf);
                if (fns().GetProgramLog(program, buf.ptr) == c.NVRTC_SUCCESS) {
                    log.err("{s} kernel compilation failed:\n{s}", .{ opts.log_name, buf });
                }
            }
        }
        return error.Nvrtc;
    }

    var image_size: usize = 0;
    try check(if (cubin)
        fns().GetCUBINSize(program, &image_size)
    else
        fns().GetPTXSize(program, &image_size));

    const image = allocator.alloc(u8, image_size) catch return error.OutOfMemory;
    defer allocator.free(image);
    try check(if (cubin)
        fns().GetCUBIN(program, image.ptr)
    else
        fns().GetPTX(program, image.ptr));

    return dev.loadModule(image.ptr);
}
