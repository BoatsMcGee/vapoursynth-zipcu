const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const bilateral = @import("bilateral.zig");
const bm3d = @import("bm3d.zig");
const dfttest = @import("dfttest.zig");
const eedi3 = @import("eedi3.zig");
const fgrain = @import("fgrain.zig");
const deband = @import("deband.zig");
const gaussblur = @import("gaussblur.zig");
const nlmeans = @import("nlmeans.zig");
const nnedi3 = @import("nnedi3.zig");

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config(
        "com.julek.vszipcu",
        "vszipcu",
        "VapourSynth Zig Image Process CUDA",
        .{ .major = 0, .minor = 1, .patch = 0 },
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        "Bilateral",
        "clip:vnode;" ++
            "sigma_spatial:float[]:opt;" ++
            "sigma_color:float[]:opt;" ++
            "radius:int[]:opt;" ++
            "device_id:int:opt;" ++
            "num_streams:int:opt;" ++
            "use_shared_memory:int:opt;" ++
            "block_x:int:opt;" ++
            "block_y:int:opt;" ++
            "ref:vnode:opt;",
        "clip:vnode;",
        bilateral.create,
        plugin,
        vspapi,
    );
    const eedi3_sig = "clip:vnode;field:int;dh:int:opt;mdis:int:opt;nrad:int:opt;" ++
        "alpha:float:opt;beta:float:opt;gamma:float:opt;hp:int:opt;vcheck:int:opt;" ++
        "vthresh0:float:opt;vthresh1:float:opt;vthresh2:float:opt;sclip:vnode:opt;" ++
        "device_id:int:opt;num_streams:int:opt;";
    ZAPI.Plugin.function("EEDI3", eedi3_sig, "clip:vnode;", eedi3.createEEDI3, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3H", eedi3_sig, "clip:vnode;", eedi3.createEEDI3H, plugin, vspapi);
    ZAPI.Plugin.function(
        "FGrain",
        "clip:vnode;num_iterations:int:opt;grain_radius_mean:float:opt;" ++
            "grain_radius_std:float:opt;sigma:float:opt;seed:int:opt;" ++
            "device_id:int:opt;num_streams:int:opt;",
        "clip:vnode;",
        fgrain.create,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        "GaussBlur",
        "clip:vnode;sigma:float[]:opt;device_id:int:opt;num_streams:int:opt;",
        "clip:vnode;",
        gaussblur.create,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        "NNEDI3",
        "clip:vnode;field:int;dh:int:opt;planes:int[]:opt;nsize:int:opt;nns:int:opt;" ++
            "qual:int:opt;etype:int:opt;pscrn:int:opt;device_id:int:opt;num_streams:int:opt;",
        "clip:vnode;",
        nnedi3.create,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        "NLMeans",
        "clip:vnode;d:int:opt;a:int:opt;s:int:opt;h:float:opt;channels:data:opt;" ++
            "wmode:int:opt;wref:float:opt;rclip:vnode:opt;device_id:int:opt;num_streams:int:opt;",
        "clip:vnode;",
        nlmeans.create,
        plugin,
        vspapi,
    );
    // fast_fused is BM3Dv2-only; standalone BM3D accepts and ignores it (composed path re-invokes BM3D with this map).
    const bm3d_sig = "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_step:int[]:opt;" ++
        "bm_range:int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;" ++
        "chroma:int:opt;device_id:int:opt;num_streams:int:opt;" ++
        "extractor_exp:int:opt;bm_error_s:data[]:opt;transform_2d_s:data[]:opt;" ++
        "transform_1d_s:data[]:opt;zero_init:int:opt;fast_fused:int:opt;";
    ZAPI.Plugin.function("BM3D", bm3d_sig, "clip:vnode;", bm3d.create, plugin, vspapi);
    ZAPI.Plugin.function("BM3Dv2", bm3d_sig, "clip:vnode;", bm3d.createV2, plugin, vspapi);
    ZAPI.Plugin.function(
        "VAggregate",
        "clip:vnode;src:vnode;planes:int[];",
        "clip:vnode;",
        bm3d.createVAggregate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        "DFTTest",
        "clip:vnode;ftype:int:opt;sigma:float:opt;sigma2:float:opt;pmin:float:opt;" ++
            "pmax:float:opt;sbsize:int:opt;sosize:int:opt;tbsize:int:opt;swin:int:opt;" ++
            "twin:int:opt;sbeta:float:opt;tbeta:float:opt;zmean:int:opt;f0beta:float:opt;" ++
            "slocation:float[]:opt;ssx:float[]:opt;ssy:float[]:opt;sst:float[]:opt;" ++
            "ssystem:int:opt;planes:int[]:opt;device_id:int:opt;num_streams:int:opt;",
        "clip:vnode;",
        dfttest.create,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        "Deband",
        "clip:vnode;" ++
            "iterations:int[]:opt;" ++
            "threshold:float[]:opt;" ++
            "radius:float[]:opt;" ++
            "grain:float[]:opt;" ++
            "planes:int[]:opt;" ++
            "dither:int:opt;" ++
            "dither_algo:int:opt;" ++
            "device_id:int:opt;" ++
            "num_streams:int:opt;",
        "clip:vnode;",
        deband.create,
        plugin,
        vspapi,
    );
}
