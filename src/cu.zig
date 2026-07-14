//! Thin re-export of cudaz plus plugin-local NVRTC compile helpers.

const std = @import("std");
const builtin = @import("builtin");
const cudaz = @import("cudaz");

pub const driver = cudaz.driver;
pub const nvrtc = cudaz.nvrtc;
pub const c = cudaz.CAPI.cuda;

pub const Error = driver.Error;
pub const Device = driver.Device;
pub const Stream = driver.Stream;
pub const Event = driver.Event;
pub const DeviceBuffer = driver.DeviceBuffer;
pub const HostBuffer = driver.HostBuffer;
pub const Module = driver.Module;
pub const Function = driver.Function;
pub const Launch = driver.Launch;
pub const Memcpy2D = driver.Memcpy2D;
pub const check = driver.check;
pub const ceilDiv = driver.ceilDiv;
pub const memInfo = driver.memInfo;

pub const CreateError = error{
    Cuda,
    Nvrtc,
    NvrtcNotFound,
    OutOfDeviceMemory,
    OutOfMemory,
    OutOfResources,
    InvalidDeviceID,
    FrameTooLarge,
};

const allocator = std.heap.c_allocator;

// Address used by the runtime loader to locate this plugin DLL (not on PATH).
fn anchor() void {}

/// Check that the CUDA driver (nvcuda.dll / libcuda.so.1) is available before
/// making any CUDA call. On Windows nvcuda.dll is delay-loaded so the plugin
/// DLL loads without it; use a direct LoadLibrary probe to avoid the SEH the
/// delay-load mechanism raises when the library is absent. On every other
/// platform the driver is import-linked and therefore present.
pub fn ensureDriver() CreateError!void {
    if (builtin.os.tag == .windows) {
        const win = std.os.windows;
        const loaded = win.kernel32.LoadLibraryA("nvcuda.dll");
        if (loaded) |h| {
            win.kernel32.FreeLibrary(h);
        } else {
            return error.Cuda;
        }
    }
}

/// Initialise a CUDA device. Thin wrapper over `Device.init` that calls
/// `ensureDriver` first so the absence of the NVIDIA driver produces a clean
/// error rather than a fatal exception from the delay-load thunk.
pub fn initDevice(ordinal: i32) CreateError!Device {
    try ensureDriver();
    return driver.Device.init(ordinal);
}

/// NVRTC-compile and load. `opts.extra` is a parity contract (`-use_fast_math` /
/// `-fmad=false` / default), not a free tuning knob.
pub fn compile(dev: Device, src: nvrtc.Source, opts: nvrtc.Options) CreateError!Module {
    try nvrtc.ensure(@ptrCast(&anchor), false);
    return nvrtc.compile(dev, src, opts, allocator);
}
