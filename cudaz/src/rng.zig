//! cuRAND convenience generator (the tutorial layer).
//!
//! Allocates through the DRIVER API (`cuMemAlloc`), not the runtime API. The previous
//! version called `cudaMalloc` — which meant (a) the convenience layer dragged in a
//! `cudart` link for one call, and (b) it fed the returned `cudaError_t` into
//! `fromCurandErrorCode`, two different code spaces, so a failed allocation surfaced as
//! whatever cuRAND status happened to share that integer.

const c = @import("c.zig");
const Error = @import("error.zig");
const CudaSlice = @import("wrappers.zig").CudaSlice;
const Device = @import("device.zig");
const std = @import("std");
const DEFAULT_SEED = 0;

rng: c.curand.curandGenerator_t,
device: Device,

/// Destroy the generator. Without this it leaked for the process lifetime.
pub fn deinit(self: @This()) void {
    _ = c.curand.curandDestroyGenerator(self.rng);
}

pub fn default() !@This() {
    var local_rng: c.curand.curandGenerator_t = undefined;
    try Error.fromCurandErrorCode(c.curand.curandCreateGenerator(&local_rng, c.curand.CURAND_RNG_PSEUDO_DEFAULT));
    try Error.fromCurandErrorCode(c.curand.curandSetPseudoRandomGeneratorSeed(local_rng, DEFAULT_SEED));
    return .{ .rng = local_rng, .device = try Device.default() };
}

pub fn init(device: Device, seed: ?u64) !@This() {
    var local_rng: c.curand.curandGenerator_t = undefined;
    try Error.fromCurandErrorCode(c.curand.curandCreateGenerator(&local_rng, c.curand.CURAND_RNG_PSEUDO_DEFAULT));
    try Error.fromCurandErrorCode(c.curand.curandSetPseudoRandomGeneratorSeed(local_rng, seed orelse DEFAULT_SEED));
    return .{ .rng = local_rng, .device = device };
}

pub fn genrandom(self: @This(), size: usize) !CudaSlice(f32) {
    var device_ptr: c.cuda.CUdeviceptr = 0;
    try Error.fromCudaErrorCode(c.cuda.cuMemAlloc(&device_ptr, @sizeOf(f32) * size));
    errdefer _ = c.cuda.cuMemFree(device_ptr);
    // cuRAND takes a plain device pointer; under UVA the driver's CUdeviceptr IS that
    // address, so no runtime-API allocation is needed.
    try Error.fromCurandErrorCode(c.curand.curandGenerateUniform(self.rng, @ptrFromInt(device_ptr), size));
    return .{ .device_ptr = device_ptr, .len = size, .device = self.device };
}
