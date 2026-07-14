//! Runtime library loading, for libraries a plugin must NOT import-link.
//!
//! Why this exists: a VapourSynth plugin (and anything else loaded as a plugin by a
//! Python host) is loaded with an altered DLL search path that does **not** include
//! PATH. An import-linked `nvrtc64_*.dll` / `cufft64_*.dll` therefore fails to resolve
//! on most installs even with the CUDA toolkit installed, and the `*_static.lib`
//! variants drag in MSVC static-CRT initialization that conflicts with Zig's dynamic
//! CRT. The fix is to bind the entry points with LoadLibrary/GetProcAddress (dlopen/
//! dlsym elsewhere) at first use — the same technique CUDA's own `cuda.lib` stub uses
//! for `nvcuda.dll`.
//!
//! Search order for a soname:
//!   1. the calling module's own directory (drop the library next to the plugin);
//!   2. the matching pip wheel under `<site-packages>/nvidia/<component>/...`, found by
//!      walking up from the module's own path to the `site-packages` component;
//!   3. the OS loader (bare name) — a normal application, not a plugin, wants this.
//!
//! `PATH` and `%CUDA_PATH%` are deliberately NOT searched: a plugin that silently picks
//! up whichever toolkit happens to be on PATH is a support nightmare.

const std = @import("std");
const builtin = @import("builtin");

const windows = std.os.windows;

pub const Handle = if (builtin.os.tag == .windows) windows.HMODULE else *anyopaque;

pub const sep = if (builtin.os.tag == .windows) "\\" else "/";

// std.DynLib dropped Windows support in Zig 0.16, so bind the loader directly. Unused
// externs are pruned per target, so the Windows and POSIX decls coexist harmlessly.
extern "kernel32" fn LoadLibraryW(lpLibFileName: [*:0]const u16) callconv(.winapi) ?windows.HMODULE;
extern "kernel32" fn GetProcAddress(hModule: windows.HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetModuleHandleExW(dwFlags: u32, lpModuleName: ?*const anyopaque, phModule: *?windows.HMODULE) callconv(.winapi) c_int;
extern "kernel32" fn GetModuleFileNameW(hModule: ?windows.HMODULE, lpFilename: [*]u16, nSize: u32) callconv(.winapi) u32;

const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;

/// dladdr maps an address (one of our own functions) back to the shared object that
/// contains it, which is how a plugin learns its own on-disk path. Declared here so we
/// do not depend on std.c exposing it.
const DlInfo = extern struct {
    dli_fname: ?[*:0]const u8,
    dli_fbase: ?*anyopaque,
    dli_sname: ?[*:0]const u8,
    dli_saddr: ?*anyopaque,
};
extern "c" fn dladdr(addr: ?*const anyopaque, info: *DlInfo) c_int;

/// Directory of the module that contains `anchor` (UTF-8/native bytes, no trailing
/// separator), or null. Pass the address of a function in YOUR module — taking it from
/// inside cudaz would still be correct here (cudaz is compiled into the caller), but the
/// explicit anchor keeps that from being an accident.
/// `buf` must hold 3 bytes per UTF-16 code unit of the path.
pub fn moduleDir(anchor: *const anyopaque, buf: *[3072]u8) ?[]const u8 {
    if (builtin.os.tag == .windows) {
        var module: ?windows.HMODULE = null;
        if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            anchor,
            &module,
        ) == 0) return null;
        var wide: [1024]u16 = undefined;
        const n = GetModuleFileNameW(module, &wide, wide.len);
        if (n == 0 or n >= wide.len) return null;
        const end = std.mem.lastIndexOfScalar(u16, wide[0..n], '\\') orelse return null;
        const len = std.unicode.wtf16LeToWtf8(buf, wide[0..end]);
        return buf[0..len];
    } else {
        var info: DlInfo = undefined;
        if (dladdr(anchor, &info) == 0) return null;
        const fname = info.dli_fname orelse return null;
        const path = std.mem.span(fname);
        const end = std.mem.lastIndexOfScalar(u8, path, '/') orelse return null;
        if (end > buf.len) return null;
        @memcpy(buf[0..end], path[0..end]);
        return buf[0..end];
    }
}

pub fn openByPath(path: []const u8) ?Handle {
    if (builtin.os.tag == .windows) {
        var wide: [4096:0]u16 = undefined;
        // Output code units <= input bytes, so this pre-check bounds the conversion.
        if (path.len >= wide.len) return null;
        const n = std.unicode.wtf8ToWtf16Le(&wide, path) catch return null;
        wide[n] = 0;
        return LoadLibraryW(wide[0..n :0].ptr);
    } else {
        var buf: [1024:0]u8 = undefined;
        if (path.len >= buf.len) return null;
        @memcpy(buf[0..path.len], path);
        buf[path.len] = 0;
        return std.c.dlopen(buf[0..path.len :0].ptr, .{ .NOW = true });
    }
}

pub fn getProc(handle: Handle, sym: [*:0]const u8) ?*anyopaque {
    return if (builtin.os.tag == .windows) GetProcAddress(handle, sym) else std.c.dlsym(handle, sym);
}

/// Index one past the LAST `site-packages` path component of `dir`, or null.
/// Component-boundary checked, so e.g. `my-site-packages-backup` does not match.
pub fn sitePackagesEnd(dir: []const u8) ?usize {
    const needle = "site-packages";
    if (dir.len < needle.len) return null;
    var found: ?usize = null;
    var i: usize = 0;
    while (i + needle.len <= dir.len) : (i += 1) {
        if (!std.ascii.eqlIgnoreCase(dir[i .. i + needle.len], needle)) continue;
        const end = i + needle.len;
        const pre_ok = i == 0 or dir[i - 1] == '\\' or dir[i - 1] == '/';
        const post_ok = end == dir.len or dir[end] == '\\' or dir[end] == '/';
        if (pre_ok and post_ok) found = end;
    }
    return found;
}

/// Where to look for a runtime-loaded CUDA library.
pub const Search = struct {
    /// Candidate sonames, tried in order (e.g. `nvrtc64_130_0.dll`, `libnvrtc.so.13`).
    names: []const []const u8,
    /// Subdirectories of the pip wheels, relative to site-packages, tried in order. The
    /// layout is NOT stable across CUDA majors: the CUDA 13 wheels put everything under one
    /// `nvidia/cu13/...` tree, while the CUDA 12 wheels gave each component its own
    /// (`nvidia/cuda_nvrtc/...`, `nvidia/cufft/...`). Empty skips the wheel search.
    wheel_subdirs: []const []const u8 = &.{},
    /// A function in the caller's module, used to find the caller's own directory.
    anchor: *const anyopaque,
    /// Fall back to the OS loader with the bare name. Correct for applications, WRONG
    /// for plugins (which is why it defaults off).
    system_search: bool = false,
};

pub fn open(s: Search) ?Handle {
    var dir_buf: [3072]u8 = undefined;
    var buf: [3200]u8 = undefined;

    if (moduleDir(s.anchor, &dir_buf)) |dir| {
        // 1. Next to the calling module.
        for (s.names) |name| {
            const path = std.fmt.bufPrint(&buf, "{s}" ++ sep ++ "{s}", .{ dir, name }) catch continue;
            if (openByPath(path)) |h| return h;
        }
        // 2. The pip wheels, anchored at site-packages.
        if (sitePackagesEnd(dir)) |sp_end| {
            for (s.wheel_subdirs) |sub| {
                for (s.names) |name| {
                    const path = std.fmt.bufPrint(&buf, "{s}" ++ sep ++ "{s}" ++ sep ++ "{s}", .{ dir[0..sp_end], sub, name }) catch continue;
                    if (openByPath(path)) |h| return h;
                }
            }
        }
    }
    // 3. The OS loader (opt-in).
    if (s.system_search) {
        for (s.names) |name| {
            if (openByPath(name)) |h| return h;
        }
    }
    return null;
}

/// One-shot, thread-safe binding of a symbol table. `Table` is a struct whose fields are
/// function pointers; each field `foo` binds the symbol `prefix ++ "foo"`.
///
/// The library stays loaded for the process lifetime (the pointers are global), which is
/// what you want: unloading NVRTC out from under a compiled module is not recoverable.
pub fn Binder(comptime Table: type, comptime prefix: []const u8) type {
    return struct {
        const Self = @This();

        pub var table: Table = undefined;

        var lock: std.atomic.Mutex = .unlocked;
        var state: enum { unloaded, loaded, failed } = .unloaded;

        /// Bind on first use. Idempotent; safe from several threads (contention is
        /// effectively nil — this is called from filter creation).
        pub fn ensure(search: Search) error{LibraryNotFound}!void {
            while (!lock.tryLock()) std.atomic.spinLoopHint();
            defer lock.unlock();
            switch (state) {
                .loaded => return,
                .failed => return error.LibraryNotFound,
                .unloaded => {},
            }
            bind(search) catch {
                state = .failed;
                return error.LibraryNotFound;
            };
            state = .loaded;
        }

        pub fn loaded() bool {
            return state == .loaded;
        }

        fn bind(search: Search) !void {
            const lib = open(search) orelse {
                std.log.scoped(.cudaz).err(
                    "could not locate {s} next to the module or in the pip wheel",
                    .{search.names[0]},
                );
                return error.LibraryNotFound;
            };
            inline for (@typeInfo(Table).@"struct".fields) |f| {
                const sym = prefix ++ f.name;
                const ptr = getProc(lib, sym) orelse {
                    std.log.scoped(.cudaz).err("{s} is missing symbol {s}", .{ search.names[0], sym });
                    return error.LibraryNotFound;
                };
                @field(table, f.name) = @ptrCast(@alignCast(ptr));
            }
        }
    };
}
