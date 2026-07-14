// Build file to create executables and small util binaries like clean to remove cached-dirs and default artifact folder.
const std = @import("std");
const utils = @import("test/utils.zig");
const builtin = @import("builtin");
const Context = struct { io: std.Io, allocator: std.mem.Allocator };

fn hasCudaHeader(parent: []const u8, context: Context) bool {
    const cuda_file = std.fs.path.join(context.allocator, &.{ parent, "include", "cuda.h" }) catch return false;
    defer context.allocator.free(cuda_file);
    std.Io.Dir.accessAbsolute(context.io, cuda_file, .{}) catch return false;
    return true;
}

/// Resolve CUDA toolkit root. Priority:
/// 1. explicit path (`-DCUDA_PATH=...` or dependency option)
/// 2. `CUDA_PATH` environment variable
/// 3. common install locations (Linux)
fn getCudaPath(explicit: ?[]const u8, context: Context) ![]const u8 {
    if (explicit) |parent| {
        if (hasCudaHeader(parent, context)) return parent;
        return error.CUDA_INSTALLATION_NOT_FOUND;
    }

    const probable_roots: []const []const u8 = switch (builtin.os.tag) {
        .windows => &.{},
        else => &.{
            "/usr",
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
        },
    };
    for (probable_roots) |parent| {
        if (hasCudaHeader(parent, context)) return parent;
    }
    return error.CUDA_INSTALLATION_NOT_FOUND;
}

pub fn build(b: *std.Build) !void {
    ////////////////////////////////////////////////////////////
    //// Create Context
    var _thread = std.Io.Threaded.init_single_threaded;
    const io = _thread.io();
    const context: Context = .{ .allocator = b.allocator, .io = io };
    ////////////////////////////////////////////////////////////
    //// Creates default options for building the library.
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    // NVIDIA ships MSVC import libs; default to msvc ABI on Windows hosts.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .windows) .{
            .cpu_arch = .x86_64,
            .os_tag = .windows,
            .abi = .msvc,
        } else .{},
    });

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    /////////////////////////////////////////////////////////////
    //// CUDA path: -DCUDA_PATH takes precedence over CUDA_PATH env
    const cuda_path_opt = b.option([]const u8, "CUDA_PATH", "locally installed Cuda's path");
    const cuda_path_env = b.graph.environ_map.get("CUDA_PATH");
    const cuda_path = cuda_path_opt orelse cuda_path_env;

    /////////////////////////////////////////////////////////////
    //// Get Cuda paths
    const cuda_folder = try getCudaPath(cuda_path, context);
    const cuda_include_dir = try std.fs.path.join(b.allocator, &.{ cuda_folder, "include" });

    ////////////////////////////////////////////////////////////
    //// CudaZ Module
    const cudaz_module = b.addModule("cudaz", .{ .root_source_file = b.path("./src/lib.zig") });
    cudaz_module.addIncludePath(.{ .cwd_relative = cuda_include_dir });

    // Prefer arch-specific import lib dirs (Windows ships both lib/x64 and lib/Win32).
    const lib_paths: []const []const u8 = switch (target.result.cpu.arch) {
        .x86_64 => &.{
            "lib/x64",
            "lib/x86_64",
            "lib/x86_64-linux-gnu",
            "lib64",
            "lib64/stubs",
            "lib",
            "targets/x86_64-linux",
            "targets/x86_64-linux/lib",
            "targets/x86_64-linux/lib/stubs",
        },
        .x86 => &.{ "lib/Win32", "lib" },
        else => &.{ "lib64", "lib" },
    };

    for (lib_paths) |lib_path| {
        const path = try std.fs.path.join(b.allocator, &.{ cuda_folder, lib_path });
        std.Io.Dir.accessAbsolute(io, path, .{}) catch {
            b.allocator.free(path);
            continue;
        };
        cudaz_module.addLibraryPath(.{ .cwd_relative = path });
    }

    ////////////////////////////////////////////////////////////
    //// Unit Testing
    // Creates a test binary.
    // Test step is created to be run from commandline i.e, zig build test
    test_blk: {
        const test_file = std.Io.Dir.cwd().openFile(io, "build.zig.zon", .{}) catch {
            break :test_blk;
        };
        defer test_file.close(io);

        const test_file_buffer = try b.allocator.alloc(u8, 1024 * 1024);
        defer b.allocator.free(test_file_buffer);
        const read_bytes = try test_file.readPositionalAll(io, test_file_buffer, 0);
        const test_file_contents = test_file_buffer[0..read_bytes];
        // Hack for identifying if the current root is cudaz project, if not don't register tests.
        if (std.mem.indexOf(u8, test_file_contents, ".name = .cudaz") == null) {
            break :test_blk;
        }
        const test_filter = b.option([]const u8, "test_filter", "Filters Tests") orelse "";

        const test_step = b.step("test", "Run library tests");
        const test_dir = try std.Io.Dir.cwd().openDir(io, "test", .{ .iterate = true });
        defer test_dir.close(io);
        var dir_iterator = try test_dir.walk(b.allocator);
        while (try dir_iterator.next(io)) |item| {
            if (item.kind == .file) {
                const test_path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ "test", item.path });
                const test_module = b.addModule(item.path, .{ .root_source_file = b.path(test_path), .target = target, .optimize = optimize });
                const sub_test = b.addTest(.{ .filters = &[_][]const u8{test_filter}, .name = item.path, .root_module = test_module });
                // Add Module
                sub_test.root_module.addImport("cudaz", cudaz_module);

                // Link libc, cuda and nvrtc libraries
                sub_test.root_module.link_libc = true;
                sub_test.root_module.linkSystemLibrary("cuda", .{});
                sub_test.root_module.linkSystemLibrary("nvrtc", .{});
                sub_test.root_module.linkSystemLibrary("curand", .{});
                sub_test.root_module.linkSystemLibrary("cudart", .{});

                // Creates a run step for test binary
                const run_sub_tests = b.addRunArtifact(sub_test);

                const test_name = try std.fmt.allocPrint(b.allocator, "test-{s}", .{item.path[0 .. item.path.len - 4]});
                // Create a test_step name
                const ind_test_step = b.step(test_name, "Individual Test");
                ind_test_step.dependOn(&run_sub_tests.step);
                test_step.dependOn(&run_sub_tests.step);
            }
        }
    }

    ////////////////////////////////////////////////////////////
    //// Clean the cache folders and artifacts

    // Creates a binary that cleans up zig artifact folders
    const delete_cache_module = b.addModule("clean", .{ .root_source_file = b.path("bin/delete-zig-cache.zig"), .target = target, .optimize = optimize });
    const clean = b.addExecutable(.{ .name = "clean", .root_module = delete_cache_module });

    // Creates a run step
    const clean_step = b.addRunArtifact(clean);

    // Register clean command i.e, zig build clean to cleanup any artifacts and cache
    const clean_cmd = b.step("clean", "Cleans the cache folders");
    clean_cmd.dependOn(&clean_step.step);
}
