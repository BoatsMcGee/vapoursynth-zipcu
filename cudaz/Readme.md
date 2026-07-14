![AI Generated](cuda_zig.jpeg)
# Cuda library for Zig
This library helps to interact with NVIDIA GPUs from zig. Provides high level interface to communicate with GPU. It can detect cuda installation and link to a project's binary on Linux/MacOS. Check [Customization](https://github.com/akhildevelops/cudaz/tree/main#Customization) to give cuda manual path.


## Two layers

**`cudaz.driver` / `cudaz.nvrtc` / `cudaz.cufft`** — the stream-centric layer. Use this for
anything real (and for anything loaded as a *plugin*, e.g. a VapourSynth filter).

- `Device` — primary context, `push()`/`pop()` (not `setCurrent` — a plugin runs on threads
  it does not own), attributes, compute capability, async-engine count.
- `Stream` / `Event` — non-blocking streams, async H2D/D2H/D2D, 2D (pitched) and 3D copies,
  `record`/`waitEvent` for cross-stream ordering, `query()` (which is also a WDDM submission
  flush).
- `DeviceBuffer` / `HostBuffer` — device memory and page-locked host memory, including the
  UVA device pointer that lets a kernel write host memory directly (zero-copy download).
- `Pitched` — `cuMemAllocPitch` + the 2D copy that goes with it.
- `Module` / `Function` — load from CUBIN/PTX image or file, `unload`, function attributes,
  occupancy (`maxActiveBlocksPerSM`).
- `Stream.launch(f, cfg, .{ a, b, c })` — the kernel argument array is packed at **comptime**,
  so its count and order cannot drift from the call site. (Hand-rolling
  `[_]?*anyopaque{ @ptrCast(&a), ... }` is the most dangerous thing in the driver API: a wrong
  count, order, or width is silent garbage, not an error.)
- `nvrtc.compile()` — `#define` block + source → arch selection (native CUBIN when the device
  is known, PTX for the newest known arch otherwise) → program log on failure → `Module`.
- `cufft.Plan` — `plan_many` (R2C/C2R/C2C), a **shared** work area, stream binding, exec.

**NVRTC and cuFFT are loaded at runtime, never import-linked.** A plugin loaded by a Python
host gets a DLL search path that excludes `PATH`, so an `nvrtc64_*.dll` import simply fails to
resolve on most installs. `dynlib.zig` looks next to the calling module, then in the pip wheel
under `<site-packages>/nvidia/...`, and never touches `PATH` or `CUDA_PATH`.

The layer is deliberately **thin**: nothing synchronizes, allocates, or reorders behind your
back, because in a GPU media filter the exact sequence of async calls *is* the performance, and
often the numerics too.

**`cudaz.SimpleDevice` / `Cudaslice` / `Compile` / `Rng`** — the original convenience layer:
owns the context, runs on the null stream, allocates per call. Fine for a script, and what the
examples below use.

(The top-level `cudaz.Device` / `Stream` / `Event` / `DeviceBuffer` / `HostBuffer` names now
alias the **production** types. `cudaz.Device` used to be the convenience one, which was a nasty
default: the obvious import was the one you must not ship.)

Both sit on `cudaz.CAPI`, the raw `@cImport` of `cuda.h` / `nvrtc.h`, which stays public — no
wrapper should ever be the reason you cannot reach a driver call.

Check [test](./test) folder for code samples.

>Scroll below to go through an example of incrementing each value in an array parallely using GPU.

### Install
Download and save the library path in `build.zig.zon` file by running

#### zig 0.16.0
`zig fetch --save https://github.com/akhildevelops/cudaz/archive/0.4.0.tar.gz`

#### zig 0.15.2
`zig fetch --save https://github.com/akhildevelops/cudaz/archive/0.3.1.tar.gz`

#### zig 0.14.1
`zig fetch --save https://github.com/akhildevelops/cudaz/archive/0.2.1.tar.gz`

#### zig 0.13.0
`zig fetch --save https://github.com/akhildevelops/cudaz/archive/0.1.0.tar.gz`


Add cudaz module in your project's `build.zig` file that will link to your project's binary.
```zig
//build.zig
const std = @import("std");

pub fn build(b: *std.Build) !void {
    // exe points to main.zig that uses cudaz
    const exe = b.addExecutable(.{ .name = "main", .root_source_file = .{ .path = "src/main.zig" }, .target = b.host });

    // Point to cudaz dependency
    const cudaz_dep = b.dependency("cudaz", .{});

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.root_module.addImport("cudaz", cudaz_module);

    exe.linkLibC();
    exe.linkSystemLibrary("cuda");

    // Only for the convenience `Compile` layer, and only in a normal EXECUTABLE.
    // The production `cudaz.nvrtc` loads NVRTC at runtime and needs no link at all.
    exe.linkSystemLibrary("nvrtc");

    // Run binary
    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}
```

> **If you are building a PLUGIN** (a shared library loaded by a host — VapourSynth, Python,
> an editor…), link **only `cuda`** (plus `dl` on Linux). Do **not** `linkSystemLibrary("nvrtc")`
> or `"cufft"`: such a host loads plugins with a DLL search path that excludes `PATH`, so the
> import fails to resolve on most machines even with the CUDA toolkit installed, and the
> `*_static` variants drag MSVC's static CRT into Zig's dynamic one. Use `cudaz.nvrtc` /
> `cudaz.cufft`, which bind at first use from your plugin's own directory and then the pip
> wheels (`nvidia-cuda-nvrtc`, `nvidia-cufft`). See `dynlib.zig`.

### Increment Array using GPU
```zig

// src/main.zig

const std = @import("std");
const Cuda = @import("cudaz");
const CuDevice = Cuda.SimpleDevice; // the convenience/null-stream layer
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;

// Cuda Kernel
const increment_kernel =
    \\extern "C" __global__ void increment(float *out)
    \\{
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    out[i] = out[i] + 1;
    \\}
;

pub fn main() !void {
    // Initialize allocator
    var GP = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = GP.deinit();
    const allocator = GP.allocator();

    // Initialize GPU
    const device = try CuDevice.default();
    defer device.deinit();

    // Copy data from host to GPU
    const data = [_]f32{ 1.2, 2.8, 0.123 };
    const cu_slice = try device.htodCopy(f32, &data);
    defer cu_slice.free();

    // Compile and load the Kernel
    const ptx = try CuCompile.cudaText(increment_kernel, .{}, allocator);
    defer allocator.free(ptx);
    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("increment");

    // Run the kernel on the data
    try function.run(.{&cu_slice.device_ptr}, CuLaunchConfig{ .block_dim = .{ 3, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });

    // Retrieve incremented data back to the system
    var incremented_arr = try CuDevice.syncReclaim(f32, allocator, cu_slice);
    defer incremented_arr.deinit(allocator);
}

```
For running above code system refer to the example project: [increment](./example/increment)

## Examples:
- [Incrementing array in GPU](example/increment/)
- [Sending Custom Types to GPU](example/custom_type/)

## Customization
- It is intelligent to identify and link to installed cuda libraries. If needed, provide cuda installation path manually by mentioning build parameter `zig build -DCUDA_PATH=<cuda_folder>`.

Inspired from Rust Cuda library: https://github.com/coreylowman/cudarc/tree/main
