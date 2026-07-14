//! cudaz — CUDA for Zig.
//!
//! Two layers, on purpose.
//!
//! **`driver` / `nvrtc` / `cufft`** — the stream-centric layer, and what a production
//! filter should use. It is thin and 1:1 with the driver calls it names: nothing here
//! synchronizes, allocates, or reorders behind your back, because in a GPU media filter
//! the exact sequence of async calls IS the performance, and often the numerics too.
//! `nvrtc` and `cufft` are loaded at RUNTIME and never import-linked, so a plugin loaded
//! by a Python host — whose DLL search path excludes PATH — can still find them.
//!
//! ```zig
//! const cu = @import("cudaz").driver;
//! const dev = try cu.Device.init(0);
//! try dev.push();
//! defer dev.pop();
//! const s = try cu.Stream.init();
//! try s.memcpyHtoD(buf.ptr, src.ptr, src.len);
//! try s.launch(f, .{ .grid = .{ gx, gy, 1 }, .block = .{ 32, 8, 1 } }, .{ buf.ptr, w, h });
//! try s.sync();
//! ```
//!
//! **`Device` / `Cudaslice` / `Compile` / `Rng`** (top level) — the original convenience
//! layer, for standalone programs and the examples. It owns the context, runs on the null
//! stream, and allocates per call: fine for a script, not what you want in a plugin.
//!
//! Both sit on `CAPI`, the raw @cImport of cuda.h / nvrtc.h, which stays public — no
//! wrapper should ever be the reason you cannot reach a driver call.

const wrappers = @import("wrappers.zig");
const utils = @import("utils.zig");

// --- the stream-centric layer -------------------------------------------------------
pub const driver = @import("driver.zig");
pub const nvrtc = @import("nvrtc.zig");
pub const cufft = @import("cufft.zig");
pub const dynlib = @import("dynlib.zig");

// --- raw C API ----------------------------------------------------------------------
pub const CAPI = @import("c.zig");

// The names most people will reach for resolve to the PRODUCTION types. `cudaz.Device`
// used to be the legacy null-stream device, which is a nasty default: the obvious import
// was the one you must not ship.
pub const Device = driver.Device;
pub const Stream = driver.Stream;
pub const Event = driver.Event;
pub const DeviceBuffer = driver.DeviceBuffer;
pub const HostBuffer = driver.HostBuffer;
pub const check = driver.check;

// --- the original convenience layer -------------------------------------------------
// Null stream, owns the context, allocates per call. Fine for a script or the examples;
// do not ship it inside a plugin.
pub const SimpleDevice = @import("device.zig");
pub const Cudaslice = wrappers.CudaSlice;
pub const SimpleModule = wrappers.Module;
pub const SimpleFunction = wrappers.Function;
pub const Compile = @import("compile.zig");
pub const LaunchConfig = @import("launchconfig.zig").LaunchConfig;
pub const Rng = @import("rng.zig");
pub const DType = utils.DType;

const Error = @import("error.zig");
pub const CudaError = Error.CurandError.Error || Error.NvrtcError.Error;

test {
    const std = @import("std");
    std.testing.refAllDecls(driver);
    std.testing.refAllDecls(nvrtc);
    std.testing.refAllDecls(cufft);
    std.testing.refAllDecls(dynlib);
}
