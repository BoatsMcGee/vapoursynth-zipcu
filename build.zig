const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) !void {
    // NVIDIA ships MSVC import libs; gnu ABI cannot link them cleanly on Windows.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .windows) .{
            .cpu_arch = .x86_64,
            .os_tag = .windows,
            .abi = .msvc,
        } else .{},
    });
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.createModule(.{
        .root_source_file = b.path("src/plugin.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "vszipcu",
        .linkage = .dynamic,
        .root_module = mod,
    });

    const cudaz_dep = b.dependency("cudaz", .{});
    mod.addImport("cudaz", cudaz_dep.module("cudaz"));

    const vapoursynth_dep = b.dependency("vapoursynth", .{
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("vapoursynth", vapoursynth_dep.module("vapoursynth"));

    // cuda.lib is a stub that LoadLibrary()s nvcuda.dll. NVRTC is loaded at runtime
    // (VS plugins do not see PATH, so an import of nvrtc64_*.dll would fail).
    mod.link_libc = true;
    mod.linkSystemLibrary("cuda", .{});

    if (target.result.os.tag == .windows) {
        // Delay-load nvcuda.dll (CUDA driver) on Windows so the plugin DLL loads
        // without the NVIDIA driver present. The first CUDA driver function call
        // (triggered by a filter create) goes through the delay-load thunk, which
        // loads the library automatically when the driver IS available.
        lib.addWin32DelayLoadLibrary("nvcuda.dll");
    }

    if (target.result.os.tag == .linux) {
        mod.linkSystemLibrary("dl", .{}); // dynlib.zig uses dladdr
    }

    if (optimize == .ReleaseFast) {
        mod.strip = true;
    }

    b.installArtifact(lib);
}
