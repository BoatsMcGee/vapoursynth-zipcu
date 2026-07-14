//! VS frame geometry helpers. Stride must match the running core's allocation.

const std = @import("std");
const builtin = @import("builtin");
const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;

/// [luma, chroma] stride in elements.
pub fn strideFromVi(vi: *const vs.VideoInfo) [2]u32 {
    const n: u32 = @divExact(vsFrameAlignment(), @as(u32, @intCast(vi.format.bytesPerSample)));
    const ssw: u3 = @intCast(vi.format.subSamplingW);
    return .{
        @intCast(vsh.ceilN(@intCast(vi.width), n)),
        @intCast(vsh.ceilN(@intCast(vi.width >> ssw), n)),
    };
}

/// VS row alignment: 64 if AVX-512F (with OS XSAVE), else 32 — matches vscore `alignmentHelper`.
pub fn vsFrameAlignment() u32 {
    if (comptime (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86)) {
        const leaf1 = cpuid(1, 0);
        const osxsave_avx: u32 = (1 << 27) | (1 << 28);
        if ((leaf1.ecx & osxsave_avx) != osxsave_avx) return 32;

        const xcr0 = getXCR0();
        if ((xcr0 & 0x06) != 0x06) return 32;

        const leaf7 = cpuid(7, 0);
        if ((leaf7.ebx & (1 << 16)) != 0 and (xcr0 & 0xE0) == 0xE0) return 64;
    }
    return 32;
}

const CpuidLeaf = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuid(leaf: u32, subleaf: u32) CpuidLeaf {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (subleaf),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn getXCR0() u32 {
    return asm volatile (
        \\ xor %%ecx, %%ecx
        \\ xgetbv
        : [_] "={eax}" (-> u32),
        :
        : .{ .edx = true, .ecx = true });
}
