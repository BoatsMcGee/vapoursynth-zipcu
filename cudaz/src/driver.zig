//! Stream-centric CUDA driver API for Zig.
//!
//! This is the layer production filters use. It is a THIN wrapper: every call maps 1:1
//! onto the driver call it names, in the same order, with the same arguments. There is no
//! hidden synchronization, no hidden allocation, and no implicit stream — because a
//! VapourSynth-style plugin's correctness and performance both live in the exact sequence
//! of `cu*Async` calls it issues (see the repo's mem_copy.md / src_stage.md), and a
//! wrapper that reorders or "helpfully" synchronizes would silently destroy them.
//!
//! What it does provide is the boilerplate that was previously copy-pasted into every
//! filter: error mapping, primary-context lifecycle, stream/event RAII, pinned host
//! memory, device attributes, occupancy, and a comptime-checked kernel launch that
//! removes the hand-rolled `[_]?*anyopaque{...}` argument array (the single most
//! error-prone thing in the driver API: a mismatch there is silent garbage, not an error).

const std = @import("std");
const cuda = @import("c.zig").cuda;

pub const c = cuda;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The only two outcomes a caller ever needs to distinguish. `OutOfDeviceMemory` is split
/// out because a stream pool's prewarm treats it as "make fewer streams", not as failure.
pub const Error = error{
    Cuda,
    OutOfDeviceMemory,
};

const log = std.log.scoped(.cudaz);

/// Map a CUresult, logging the driver's own message. Success is the hot path.
pub fn check(result: cuda.CUresult) Error!void {
    if (result == cuda.CUDA_SUCCESS) return;
    var str: [*c]const u8 = null;
    _ = cuda.cuGetErrorString(result, &str);
    log.err("CUDA error: {s}", .{if (str != null) std.mem.span(str) else "unknown"});
    return if (result == cuda.CUDA_ERROR_OUT_OF_MEMORY) error.OutOfDeviceMemory else error.Cuda;
}

pub fn errorString(result: cuda.CUresult) []const u8 {
    var str: [*c]const u8 = null;
    _ = cuda.cuGetErrorString(result, &str);
    return if (str != null) std.mem.span(str) else "unknown";
}

// ---------------------------------------------------------------------------
// Device + primary context
// ---------------------------------------------------------------------------

pub const InitError = Error || error{InvalidDeviceID};

/// A device plus a retained primary context.
///
/// The primary context is the right choice for a plugin: every filter instance on a
/// device shares one context, so their streams actually overlap on the GPU. Each instance
/// retains once in create() and releases once in free().
pub const Device = struct {
    handle: cuda.CUdevice = 0,
    context: cuda.CUcontext = null,

    /// cuInit + validate the ordinal + retain the primary context. Does NOT make the
    /// context current; call `push()` around the work.
    pub fn init(ordinal: i32) InitError!Device {
        try check(cuda.cuInit(0));
        var count: c_int = 0;
        try check(cuda.cuDeviceGetCount(&count));
        if (ordinal < 0 or ordinal >= count) return error.InvalidDeviceID;

        var d: Device = .{};
        try check(cuda.cuDeviceGet(&d.handle, ordinal));
        try check(cuda.cuDevicePrimaryCtxRetain(&d.context, d.handle));
        return d;
    }

    pub fn deinit(self: Device) void {
        _ = cuda.cuDevicePrimaryCtxRelease(self.handle);
    }

    /// Make the context current on THIS thread. VapourSynth calls getFrame from arbitrary
    /// worker threads, so every frame pushes at the top and pops on the way out.
    pub fn push(self: Device) Error!void {
        return check(cuda.cuCtxPushCurrent(self.context));
    }

    pub fn pop(_: Device) void {
        _ = cuda.cuCtxPopCurrent(null);
    }

    pub fn attribute(self: Device, attr: cuda.CUdevice_attribute) Error!i32 {
        var v: c_int = 0;
        try check(cuda.cuDeviceGetAttribute(&v, attr, self.handle));
        return v;
    }

    /// Compute capability as `major * 10 + minor` (75, 86, 89, ...), the form NVRTC's
    /// `-arch=sm_XX` and `nvrtcGetSupportedArchs()` both speak.
    pub fn computeCapability(self: Device) Error!i32 {
        const major = try self.attribute(cuda.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        const minor = try self.attribute(cuda.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return major * 10 + minor;
    }

    /// Number of DMA engines that can overlap a copy with kernel execution. Consumer
    /// boards commonly report **1**, meaning H2D and D2H never overlap each other — for a
    /// transfer-bound filter that makes the summed per-frame engine time a hard fps
    /// ceiling. Worth querying before designing a transfer path.
    pub fn asyncEngineCount(self: Device) Error!i32 {
        return self.attribute(cuda.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
    }

    pub fn multiProcessorCount(self: Device) Error!i32 {
        return self.attribute(cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    }

    pub fn warpSize(self: Device) Error!i32 {
        return self.attribute(cuda.CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    }

    pub fn loadModule(_: Device, image: [*]const u8) Error!Module {
        var m: cuda.CUmodule = null;
        try check(cuda.cuModuleLoadData(&m, image));
        return .{ .handle = m };
    }

    /// Load a module from a file (an offline `nvcc -cubin` build; the escape hatch for
    /// no-NVRTC deployment and for `-lineinfo` builds that ncu needs for per-line
    /// attribution).
    pub fn loadModuleFile(_: Device, path: [*:0]const u8) Error!Module {
        var m: cuda.CUmodule = null;
        try check(cuda.cuModuleLoad(&m, path));
        return .{ .handle = m };
    }
};

// ---------------------------------------------------------------------------
// Streams and events
// ---------------------------------------------------------------------------

pub const Stream = struct {
    handle: cuda.CUstream = null,

    /// Non-blocking by default: a stream that implicitly synchronizes with the legacy
    /// NULL stream is never what a multi-stream filter wants.
    pub fn init() Error!Stream {
        var s: Stream = .{};
        try check(cuda.cuStreamCreate(&s.handle, cuda.CU_STREAM_NON_BLOCKING));
        return s;
    }

    /// Sync-then-destroy. The guard is not paranoia: a default-initialized `Stream` has a
    /// null handle, and null means the **legacy default stream** — so an unguarded deinit
    /// would synchronize (and try to destroy) it. Filters that keep an optional second
    /// stream hit exactly this.
    pub fn deinit(self: Stream) void {
        if (self.handle == null) return;
        _ = cuda.cuStreamSynchronize(self.handle);
        _ = cuda.cuStreamDestroy(self.handle);
    }

    pub fn sync(self: Stream) Error!void {
        return check(cuda.cuStreamSynchronize(self.handle));
    }

    /// Drain without reporting — for error paths, where the only thing that matters is
    /// that no in-flight async copy still references host memory that is about to die.
    pub fn drain(self: Stream) void {
        _ = cuda.cuStreamSynchronize(self.handle);
    }

    /// Poll. Returns true when everything enqueued so far has completed.
    ///
    /// Also, and less obviously, this is a **submission flush**: on WDDM the driver
    /// batches enqueues, and a query forces them out. NLMeans needs exactly that — without
    /// it the batched launches sit unflushed while another stream waits on an event that
    /// has not been submitted yet, and the GPU idles. Discarding the result is a legitimate
    /// use of this call.
    pub fn query(self: Stream) bool {
        return cuda.cuStreamQuery(self.handle) == cuda.CUDA_SUCCESS;
    }

    pub fn record(self: Stream, ev: Event) Error!void {
        return check(cuda.cuEventRecord(ev.handle, self.handle));
    }

    /// Make this stream wait for `ev` (recorded on another stream). The cross-stream
    /// ordering primitive: record on the producer, wait on the consumer.
    pub fn waitEvent(self: Stream, ev: Event) Error!void {
        return check(cuda.cuStreamWaitEvent(self.handle, ev.handle, 0));
    }

    // -- transfers -----------------------------------------------------------
    // `Async` here means "enqueued on the stream". For PAGEABLE host memory the driver
    // still blocks the calling thread while it stages (fully, for D2H) — an async name on
    // a synchronous call. That is a performance property, not a correctness one, and it
    // is why the copy direction and the pinned/pageable choice are per-filter decisions.

    pub fn memcpyHtoD(self: Stream, dst: cuda.CUdeviceptr, src: *const anyopaque, bytes: usize) Error!void {
        return check(cuda.cuMemcpyHtoDAsync(dst, src, bytes, self.handle));
    }

    pub fn memcpyDtoH(self: Stream, dst: *anyopaque, src: cuda.CUdeviceptr, bytes: usize) Error!void {
        return check(cuda.cuMemcpyDtoHAsync(dst, src, bytes, self.handle));
    }

    pub fn memcpyDtoD(self: Stream, dst: cuda.CUdeviceptr, src: cuda.CUdeviceptr, bytes: usize) Error!void {
        return check(cuda.cuMemcpyDtoDAsync(dst, src, bytes, self.handle));
    }

    /// Row-wise copy between differing pitches (a VS frame stride vs a cuMemAllocPitch
    /// pitch). Prefer the 1-D call when the pitches match — the 2-D path pays per-row
    /// bookkeeping.
    pub fn memcpy2D(self: Stream, p: Memcpy2D) Error!void {
        var d = p.toDriver();
        return check(cuda.cuMemcpy2DAsync(&d, self.handle));
    }

    pub fn memcpy3D(self: Stream, p: *const cuda.CUDA_MEMCPY3D) Error!void {
        return check(cuda.cuMemcpy3DAsync(p, self.handle));
    }

    pub fn memsetD8(self: Stream, dst: cuda.CUdeviceptr, value: u8, bytes: usize) Error!void {
        return check(cuda.cuMemsetD8Async(dst, value, bytes, self.handle));
    }

    pub fn memsetD32(self: Stream, dst: cuda.CUdeviceptr, value: u32, n: usize) Error!void {
        return check(cuda.cuMemsetD32Async(dst, value, n, self.handle));
    }

    /// Launch. See `launch()` — args are packed at comptime.
    pub fn launch(self: Stream, f: Function, cfg: Launch, args: anytype) Error!void {
        return launchOn(f, cfg, self.handle, args);
    }
};

pub const Event = struct {
    handle: cuda.CUevent = null,

    /// Timing is disabled by default: these are ordering primitives on the frame path,
    /// and the timing machinery is not free.
    pub fn init() Error!Event {
        var e: Event = .{};
        try check(cuda.cuEventCreate(&e.handle, cuda.CU_EVENT_DISABLE_TIMING));
        return e;
    }

    pub fn initTiming() Error!Event {
        var e: Event = .{};
        try check(cuda.cuEventCreate(&e.handle, cuda.CU_EVENT_DEFAULT));
        return e;
    }

    pub fn deinit(self: Event) void {
        if (self.handle == null) return;
        _ = cuda.cuEventDestroy(self.handle);
    }

    pub fn sync(self: Event) Error!void {
        return check(cuda.cuEventSynchronize(self.handle));
    }

    /// Milliseconds between two events (both must have been created with timing on).
    pub fn elapsed(start: Event, end: Event) Error!f32 {
        var ms: f32 = 0;
        try check(cuda.cuEventElapsedTime(&ms, start.handle, end.handle));
        return ms;
    }
};

/// A 2-D copy descriptor with the fields that actually get set, so callers stop
/// zero-initializing a 20-field driver struct by hand.
pub const Memcpy2D = struct {
    src: Ptr,
    src_pitch: usize,
    dst: Ptr,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,

    pub const Ptr = union(enum) {
        host: *const anyopaque,
        device: cuda.CUdeviceptr,
    };

    fn toDriver(self: Memcpy2D) cuda.CUDA_MEMCPY2D {
        var d: cuda.CUDA_MEMCPY2D = std.mem.zeroes(cuda.CUDA_MEMCPY2D);
        switch (self.src) {
            .host => |h| {
                d.srcMemoryType = cuda.CU_MEMORYTYPE_HOST;
                d.srcHost = h;
            },
            .device => |p| {
                d.srcMemoryType = cuda.CU_MEMORYTYPE_DEVICE;
                d.srcDevice = p;
            },
        }
        switch (self.dst) {
            .host => |h| {
                d.dstMemoryType = cuda.CU_MEMORYTYPE_HOST;
                d.dstHost = @constCast(h);
            },
            .device => |p| {
                d.dstMemoryType = cuda.CU_MEMORYTYPE_DEVICE;
                d.dstDevice = p;
            },
        }
        d.srcPitch = self.src_pitch;
        d.dstPitch = self.dst_pitch;
        d.WidthInBytes = self.width_bytes;
        d.Height = self.height;
        return d;
    }
};

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Plain device memory. Not stream-ordered (no cuMemAllocAsync): filters allocate once at
/// create() and reuse for the whole instance lifetime, which is both faster and simpler
/// than per-frame allocation.
pub const DeviceBuffer = struct {
    ptr: cuda.CUdeviceptr = 0,
    bytes: usize = 0,

    pub fn alloc(bytes: usize) Error!DeviceBuffer {
        // cuMemAlloc rejects 0; hand out a tiny stub so "unused buffer" needs no branch
        // at every use site.
        const n = if (bytes == 0) 8 else bytes;
        var b: DeviceBuffer = .{ .bytes = n };
        try check(cuda.cuMemAlloc(&b.ptr, n));
        return b;
    }

    pub fn allocZeroed(bytes: usize) Error!DeviceBuffer {
        const b = try alloc(bytes);
        errdefer b.deinit();
        try check(cuda.cuMemsetD8(b.ptr, 0, b.bytes));
        return b;
    }

    pub fn deinit(self: DeviceBuffer) void {
        if (self.ptr != 0) _ = cuda.cuMemFree(self.ptr);
    }

    /// Blocking zero of the whole buffer. For create()-time initialization, not the frame
    /// path (use `Stream.memsetD8` there).
    pub fn zero(self: DeviceBuffer) Error!void {
        return check(cuda.cuMemsetD8(self.ptr, 0, self.bytes));
    }

    /// Blocking fill of a prefix. Same caveat.
    pub fn fill(self: DeviceBuffer, value: u8, bytes: usize) Error!void {
        std.debug.assert(bytes <= self.bytes);
        return check(cuda.cuMemsetD8(self.ptr, value, bytes));
    }

    /// Byte offset. Widen BEFORE multiplying at the call site: `off * elem_size` with two
    /// u32s multiplies in 32 bits and wraps before the 64-bit pointer add.
    pub fn at(self: DeviceBuffer, byte_offset: usize) cuda.CUdeviceptr {
        return self.ptr + byte_offset;
    }
};

/// Page-locked host memory. Required for a DMA that does not go through the driver's
/// internal staging pool, and for zero-copy (the device can write it directly over UVA,
/// which is how a cheap output kernel gets the download off the copy engine entirely).
pub const HostBuffer = struct {
    ptr: [*]u8 = undefined,
    bytes: usize = 0,

    pub const Flags = struct {
        /// Write-combined: fast for host writes that the CPU never reads back, and
        /// catastrophic if it does read back (uncached loads).
        write_combined: bool = false,
    };

    pub fn alloc(bytes: usize, flags: Flags) Error!HostBuffer {
        const n = if (bytes == 0) 8 else bytes;
        var p: ?*anyopaque = null;
        var f: c_uint = 0;
        if (flags.write_combined) f |= cuda.CU_MEMHOSTALLOC_WRITECOMBINED;
        try check(cuda.cuMemHostAlloc(&p, n, f));
        return .{ .ptr = @ptrCast(p.?), .bytes = n };
    }

    /// `bytes` is the allocated-ness flag: a successful `alloc` always sets it to at least
    /// 8, so 0 means "never allocated" and `ptr` is still `undefined`. Freeing that would be
    /// UB, so guard rather than trusting every caller to keep the value in an optional.
    pub fn deinit(self: HostBuffer) void {
        if (self.bytes == 0) return;
        _ = cuda.cuMemFreeHost(self.ptr);
    }

    pub fn slice(self: HostBuffer) []u8 {
        return self.ptr[0..self.bytes];
    }

    /// The device-visible address of this pinned allocation (UVA). Pass THIS as a kernel
    /// pointer argument and the kernel writes host memory directly.
    pub fn devicePtr(self: HostBuffer, byte_offset: usize) cuda.CUdeviceptr {
        return @intCast(@intFromPtr(self.ptr + byte_offset));
    }
};

/// Blocking copies, for create()-time table uploads. Never on the frame path.
pub fn memcpyHtoDSync(dst: cuda.CUdeviceptr, src: *const anyopaque, bytes: usize) Error!void {
    return check(cuda.cuMemcpyHtoD(dst, src, bytes));
}

pub fn memcpyDtoHSync(dst: *anyopaque, src: cuda.CUdeviceptr, bytes: usize) Error!void {
    return check(cuda.cuMemcpyDtoH(dst, src, bytes));
}

/// Probe the pitch the driver would pick for a row of `width_bytes`, without keeping the
/// allocation (BilateralGPU's layout is defined in terms of it).
pub fn probePitch(width_bytes: usize, height: usize, element_size: c_uint) Error!usize {
    var ptr: cuda.CUdeviceptr = 0;
    var pitch: usize = 0;
    try check(cuda.cuMemAllocPitch(&ptr, &pitch, width_bytes, height, element_size));
    _ = cuda.cuMemFree(ptr);
    return pitch;
}

/// A pitched allocation: rows padded to a driver-chosen pitch so every row start is
/// aligned. Note the whole allocation is `pitch * height` bytes, so a plain 1-D copy is
/// only valid when the source shares the pitch — otherwise use a 2-D copy.
pub const Pitched = struct {
    buf: DeviceBuffer,
    pitch: usize,

    pub fn alloc(width_bytes: usize, height: usize, element_size: c_uint) Error!Pitched {
        var p: Pitched = .{ .buf = .{}, .pitch = 0 };
        try check(cuda.cuMemAllocPitch(&p.buf.ptr, &p.pitch, width_bytes, height, element_size));
        p.buf.bytes = p.pitch * height;
        return p;
    }

    pub fn deinit(self: Pitched) void {
        self.buf.deinit();
    }
};

/// Free/total device memory. The honest way for a stream pool to decide how many streams
/// it can actually afford before it starts allocating.
pub fn memInfo() Error!struct { free: usize, total: usize } {
    var f: usize = 0;
    var t: usize = 0;
    try check(cuda.cuMemGetInfo(&f, &t));
    return .{ .free = f, .total = t };
}

/// Page-lock memory you did NOT allocate (e.g. a host frame buffer someone handed you).
///
/// Measured on this repo and NOT worth it per frame: registering 8 MB costs ~0.12 ms but
/// UNregistering costs 0.7-1 ms, and caching registrations over a host frame pool is
/// unsafe (the pool reuses addresses, so a stale "pinned" entry would DMA to unpinned
/// pages). Here because it is occasionally the right tool, not because it usually is.
pub fn hostRegister(ptr: *anyopaque, bytes: usize, flags: c_uint) Error!void {
    return check(cuda.cuMemHostRegister(ptr, bytes, flags));
}

pub fn hostUnregister(ptr: *anyopaque) Error!void {
    return check(cuda.cuMemHostUnregister(ptr));
}

// ---------------------------------------------------------------------------
// Modules, functions, launches
// ---------------------------------------------------------------------------

pub const Function = struct {
    handle: cuda.CUfunction = null,

    pub fn attribute(self: Function, attr: cuda.CUfunction_attribute) Error!i32 {
        var v: c_int = 0;
        try check(cuda.cuFuncGetAttribute(&v, attr, self.handle));
        return v;
    }

    pub fn registers(self: Function) Error!i32 {
        return self.attribute(cuda.CU_FUNC_ATTRIBUTE_NUM_REGS);
    }

    pub fn setAttribute(self: Function, attr: cuda.CUfunction_attribute, value: i32) Error!void {
        return check(cuda.cuFuncSetAttribute(self.handle, attr, value));
    }

    /// Opt in to more than the default 48 KiB of dynamic shared memory per block. Without
    /// this call a launch asking for more simply fails, which reads as a mysterious
    /// CUDA_ERROR_INVALID_VALUE at launch rather than at compile.
    pub fn setMaxDynamicSharedMemory(self: Function, bytes: i32) Error!void {
        return self.setAttribute(cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
    }

    /// Blocks resident per SM at this block size. The grid-sizing input dfttest2 uses, and
    /// the number that reveals when occupancy is capped by the hardware blocks/SM limit
    /// rather than by registers.
    pub fn maxActiveBlocksPerSM(self: Function, block_size: i32, dynamic_smem: usize) Error!i32 {
        var n: c_int = 0;
        try check(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, self.handle, block_size, dynamic_smem));
        return n;
    }
};

pub const Module = struct {
    handle: cuda.CUmodule = null,

    pub fn deinit(self: Module) void {
        if (self.handle != null) _ = cuda.cuModuleUnload(self.handle);
    }

    pub fn function(self: Module, name: [*:0]const u8) Error!Function {
        var f: cuda.CUfunction = null;
        try check(cuda.cuModuleGetFunction(&f, self.handle, name));
        return .{ .handle = f };
    }

    pub fn global(self: Module, name: [*:0]const u8) Error!DeviceBuffer {
        var p: cuda.CUdeviceptr = 0;
        var n: usize = 0;
        try check(cuda.cuModuleGetGlobal(&p, &n, self.handle, name));
        return .{ .ptr = p, .bytes = n };
    }
};

pub const Launch = struct {
    grid: [3]u32 = .{ 1, 1, 1 },
    block: [3]u32 = .{ 1, 1, 1 },
    shared_mem: u32 = 0,
};

/// Launch a kernel with comptime-packed arguments.
///
/// `cuLaunchKernel` wants `void**` — an array of POINTERS TO the argument values, which
/// must stay alive across the call. Hand-rolling that (`var a: c_int = ...;
/// params = [_]?*anyopaque{ @ptrCast(&a), ... }`) is the most dangerous thing in the
/// driver API: a wrong count, order, or width is not an error, it is silent garbage.
/// Here the tuple is copied into an addressable local and the pointer array is built from
/// it at comptime, so the count and order cannot drift from the call site.
///
/// It still cannot check the types against the KERNEL's signature — nothing can — so pass
/// exactly the C types the kernel declares (`c_int`, `f32`, `cuda.CUdeviceptr`, ...).
pub fn launchOn(f: Function, cfg: Launch, stream: cuda.CUstream, args: anytype) Error!void {
    const A = @TypeOf(args);
    const info = @typeInfo(A);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("launch args must be a tuple, e.g. .{ d_dst, d_src, width }");
    }
    const fields = info.@"struct".fields;

    // Rebuild the tuple with RUNTIME fields before taking addresses.
    //
    // A tuple literal like `.{ d_dst, @as(f32, 2.0) }` gives the constant a *comptime*
    // field, and a comptime field has no runtime address — which is precisely what
    // cuLaunchKernel wants. Copying into a `std.meta.Tuple` of the same types materializes
    // every element as an ordinary runtime value. (Without this, passing any literal
    // constant as a kernel argument is a compile error, which is not obvious from the call
    // site.)
    comptime var types: [fields.len]type = undefined;
    inline for (fields, 0..) |fl, i| {
        if (fl.type == comptime_int or fl.type == comptime_float) {
            @compileError("launch arg " ++ fl.name ++ " is an untyped literal; a kernel argument's" ++
                " width is part of the ABI, so give it the kernel's exact C type, e.g." ++
                " @as(c_int, 64) or @as(f32, 1.5)");
        }
        types[i] = fl.type;
    }

    // One addressable copy of every argument, living until the call returns.
    var storage: std.meta.Tuple(&types) = args;
    var params: [fields.len]?*anyopaque = undefined;
    inline for (fields, 0..) |fl, i| {
        params[i] = @ptrCast(&@field(storage, fl.name));
    }

    return check(cuda.cuLaunchKernel(
        f.handle,
        cfg.grid[0],
        cfg.grid[1],
        cfg.grid[2],
        cfg.block[0],
        cfg.block[1],
        cfg.block[2],
        cfg.shared_mem,
        stream,
        if (fields.len == 0) null else &params,
        null,
    ));
}

/// `a` divided by `b`, rounded up — the grid-dimension idiom, in one place.
pub fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}
