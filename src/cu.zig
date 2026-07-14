//! Thin re-export of cudaz plus plugin-local NVRTC compile helpers.

const std = @import("std");
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

/// NVRTC-compile and load. `opts.extra` is a parity contract (`-use_fast_math` /
/// `-fmad=false` / default), not a free tuning knob.
pub fn compile(dev: Device, src: nvrtc.Source, opts: nvrtc.Options) CreateError!Module {
    try nvrtc.ensure(@ptrCast(&anchor), false);
    return nvrtc.compile(dev, src, opts, allocator);
}
