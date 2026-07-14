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

    // exe points to main.zig that uses cudaz
    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
        }),
    });

    // Point to cudaz dependency.
    // CUDA toolkit root: set CUDA_PATH env, or pass `.{ .CUDA_PATH = "..." }`, or `-DCUDA_PATH=...`.
    const cudaz_dep = b.dependency("cudaz", .{});

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.root_module.addImport("cudaz", cudaz_module);

    // Dynamically link to libc, cuda, nvrtc
    exe.root_module.link_libc = true;
    exe.root_module.linkSystemLibrary("cuda", .{});
    exe.root_module.linkSystemLibrary("nvrtc", .{});

    // Run binary
    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}
