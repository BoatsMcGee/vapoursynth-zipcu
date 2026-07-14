# vszipcu

VapourSynth CUDA image processing, written in Zig.

## Requirements

- An NVIDIA GPU. (Only NNEDI3 cares about the architecture: its tensor-core predictor needs
  sm_80+, and it falls back to an fp32 path automatically below that.)
- **NVRTC at runtime:** plugin looks in exactly two places: its **own directory**, then the
  **`nvidia-cuda-nvrtc` wheel** (`pip install nvidia-cuda-nvrtc`, CUDA 13.3+).

## Install

```sh
pip install vapoursynth-vszipcu        # released wheels (Windows / Linux x86_64)
```

The wheel drops the plugin where VapourSynth autoloads it and pulls `nvidia-cuda-nvrtc`
automatically — no toolkit needed at runtime, just the NVIDIA driver.

## Build

Directly from a clone, via Python (no local Zig needed — pip fetches the `ziglang` wheel;
the CUDA **toolkit** must be installed and findable via `CUDA_PATH`):

```sh
git clone https://github.com/dnjulek/vapoursynth-zipcu
cd vapoursynth-zipcu
pip install .                          # build + install into site-packages
```

The zig path leaves the DLL in `zig-out` — copy it to a VapourSynth plugin dir yourself
(and keep NVRTC findable, see Requirements).

## Common arguments

`device_id=0` selects the GPU. `num_streams` sets how many frames the filter processes
concurrently — more streams means more VRAM, raising it past 4 rarely helps.

## Filters

```python
core.vszipcu.Bilateral(clip clip[,
    float[] sigma_spatial=3.0,      # per-plane; chroma default = sigma[0]/sqrt((1<<ssw)*(1<<ssh))
    float[] sigma_color=0.02,       # per-plane; operates on the [0,1] domain at every depth
    int[]   radius,                 # per-plane; default max(1, round(sigma_spatial*3))
    int     device_id=0, int num_streams=4,
    int     use_shared_memory=1,    # shared-memory tile kernel while the tile fits 48 KiB
    int     block_x=32, int block_y=8,
    clip    ref])                   # joint/cross bilateral: weights from ref, values from clip
```
8/16-bit integer, 16-bit half, 32-bit float.

```python
core.vszipcu.BM3Dv2(clip clip[,      # and BM3D, same signature
    clip    ref,                     # empirical Wiener (2nd pass); also drives block matching
    float[] sigma=3.0,               # per-plane; < FLT_EPSILON => plane not processed
    int[]   block_step=8,            # per-plane, 1..8
    int[]   bm_range=9,              # per-plane, > 0
    int     radius=0,                # temporal radius; 0 = spatial only
    int[]   ps_num=2, int[] ps_range=4,
    bint    chroma=False,            # CBM3D; YUV444PS only, block matching on Y
    int     device_id=0, int num_streams=4,
    int     extractor_exp=0,         # >= 3 => deterministic (order-independent) sums
    data[]  bm_error_s="ssd",        # ssd | sad | zssd | zsad | ssd/norm
    data[]  transform_2d_s="dct",    # dct | haar | wht | bior1.5
    data[]  transform_1d_s="dct",
    bint    zero_init=True,          # BM3D, radius>0 only: zero the unprocessed planes of the
                                     #   stacked output instead of leaving them uninitialized
    bint    fast_fused=False])       # see below
core.vszipcu.VAggregate(clip clip, clip src, int[] planes)
```
32-bit float only. `BM3Dv2` is the one-step interface; `BM3D` alone emits the stacked accumulator
clip if you want to drive `VAggregate` yourself. Per-plane arrays follow upstream's rule: element
*i*, when absent, falls back to element *i-1*.

**`fast_fused`** (BM3Dv2, `radius > 0`): runs the collaborative filter and the temporal
aggregation as one kernel chain, keeping the accumulator stack on the GPU. Byte-identical output,
**+63 % to +169 %** faster — but it costs **1.7x to 7.9x** the VRAM (the accumulator cache grows
quadratically with `radius` and is independent of `num_streams`). Hence opt-in. If the cache does
not fit it raises an error telling you what it needed; it never silently falls back.

Note `extractor_exp=0` (the default, matching upstream) makes the result *nondeterministic* —
float `atomicAdd` order is arbitrary. Use `extractor_exp >= 3` if you need reproducibility.

```python
core.vszipcu.Deband(clip clip[,
    int[]   iterations=1,           # debanding passes, 0..32; 0 = grain only
    float[] threshold=3.0,          # cut-off threshold. Higher debands harder, but
                                    #   progressively destroys image detail
    float[] radius=16.0,            # initial sampling radius (grows each iteration).
                                    #   Higher finds more gradients; lower smooths more aggressively
    float[] grain=4.0,              # noise added on top, to cover residual
                                    #   quantization; 0 = none
    int[]   planes,                 # which planes to process at all; default: all
    bint    dither=True,            # 8-bit output only, first processed plane only
    int     dither_algo=0,          # 0=blue noise | 1=bayer | 2=ordered fixed | 3=white noise
    int     device_id=0, int num_streams=1])
```
8/16-bit integer, 16-bit half, 32-bit float.

The four strength parameters are **per-plane arrays**. 
`int[] planes` says *which* planes are touched, the arrays say *how*.

```python
core.vszipcu.DFTTest(clip clip[,
    int     ftype=0,                # 0 = Wiener: mult by max((psd-sigma)/psd, 0) ** f0beta
                                    # 1 = hard threshold: zero the bin when psd < sigma
                                    # 2 = mult every bin by sigma
                                    # 3 = mult by sigma inside [pmin,pmax], else by sigma2
                                    # 4 = mult by sigma*sqrt(psd*pmax / ((psd+pmin)*(psd+pmax)))
    float   sigma=8.0,              # main strength (scaled by the window; see slocation/ssx/..)
    float   sigma2=8.0,             # ftype=3 only: strength outside [pmin,pmax]
    float   pmin=0.0,               # ftype=3/4 only: PSD bounds
    float   pmax=500.0,
    int     sbsize=16,              # spatial block size; must be 16 in this backend
    int     sosize=12,              # spatial overlap, 0..15; >50% needs (sbsize-sosize) | sbsize
    int     tbsize=3,               # temporal block size; ODD, 1..7 (1 = spatial only)
    int     swin=0,                 # spatial window: 0=hanning 1=hamming 2=blackman
    int     twin=7,                 # temporal window: 3=4-term b-harris 4=kaiser-bessel
                                    #   5=7-term b-harris 6=flat top 7=rectangular 8=bartlett
                                    #   9=bartlett-hann 10=nuttall 11=blackman-nuttall
    float   sbeta=2.5,              # kaiser-bessel beta (swin=4 / twin=4)
    float   tbeta=2.5,
    bint    zmean=True,             # subtract the windowed mean before filtering
    float   f0beta=1.0,             # ftype=0 exponent (1.0 = plain Wiener, 0.5 = sqrt)
    float[] slocation,              # frequency-dependent sigma: [freq, sigma, freq, sigma, ...]
    float[] ssx, float[] ssy, float[] sst,   # per-axis variants of slocation
    int     ssystem=0,
    int[]   planes,                 # default: all
    int     device_id=0, int num_streams=1])
```
8-16-bit integer and 32-bit float.

```python
core.vszipcu.EEDI3(clip clip, int field[,   # and EEDI3H, same args, horizontal
    bint    dh=False,               # double the height (keep every source line)
    int     mdis=20,                # max connection radius, 1..40; larger connects shallower
                                    #   lines, but costs speed and risks artifacts
    int     nrad=2,                 # radius used for neighborhood similarity, 0..3
    float   alpha=0.2,              # 0..1 (alpha+beta <= 1): weight given to connecting similar
                                    #   neighborhoods. Larger = more lines/edges connected
    float   beta=0.25,              # 0..1: weight given to the vertical difference created by
                                    #   the interpolation. Larger = fewer edges connected
                                    #   (1.0 = no edge directedness at all)
    float   gamma=20.0,             # >= 0: penalizes changes in interpolation direction.
                                    #   Larger = smoother interpolation field
    bint    hp=False,               # search half-pixel directions too
    int     vcheck=2,               # 0..3: strength of the vertical-consistency check
    float   vthresh0=32.0,          # vcheck thresholds; all must be > 0 when vcheck > 0
    float   vthresh1=64.0,
    float   vthresh2=4.0,
    clip    sclip,                  # source for the vcheck comparison
    int     device_id=0, int num_streams=1])
```
8/16-bit integer, 16-bit half, 32-bit float. `field`: 0/1 pick the field to keep, 2/3 are the
double-rate variants (not allowed with `dh=True`).

```python
core.vszipcu.FGrain(clip clip[,
    int     num_iterations=800,     # Monte-Carlo samples per pixel; quality vs speed
    float   grain_radius_mean=0.1,  # mean grain radius, in pixels
    float   grain_radius_std=0.0,   # grain-radius spread; 0 = every grain the same size
    float   sigma=0.8,              # std dev of the Gaussian in the grain model
    int     seed=0,
    int     device_id=0, int num_streams=1])
```
32-bit float. Physically-based film grain synthesis. For temporally-varying grain, set the
integer frame property `FGRAIN_SEED_OFFSET` on the source frame — it is added to `seed`.

```python
core.vszipcu.GaussBlur(clip clip[,
    float[] sigma=0.5,              # per-plane; chroma default = sigma[0]/sqrt((1<<ssw)*(1<<ssh))
    int     device_id=0, int num_streams=1])
```
8/16-bit integer, 16-bit half, 32-bit float. A plane whose sigma is below `FLT_EPSILON` is copied
through untouched.

```python
core.vszipcu.NLMeans(clip clip[,
    int     d=1,                    # temporal radius; 0 = spatial only
    int     a=2,                    # search-window radius, 1..64
    int     s=4,                    # patch radius, 0..8
    float   h=1.2,                  # > 0: filtering strength
    string  channels="auto",        # auto | Y | UV | YUV | RGB
    int     wmode=0,                # weighting function of the patch distance x, 0..3:
                                    #   0 = exp(-x) | 1 = max(1-x, 0)
                                    #   2 = max(1-x, 0)**2 | 3 = max(1-x, 0)**8
    float   wref=1.0,               # >= 0: weight of the pixel itself
    clip    rclip,                  # weights computed from rclip, values from clip
    int     device_id=0, int num_streams=1])
```
8/16-bit integer, 16-bit half, 32-bit float. `channels="YUV"` requires 4:4:4 — a joint patch
distance needs one shared pixel lattice — so on subsampled clips run a `"Y"` pass and a `"UV"`
pass instead.

```python
core.vszipcu.NNEDI3(clip clip, int field[,
    bint    dh=False,               # double the height (keep every source line)
    int[]   planes,                 # default: all
    int     nsize=6,                # predictor neighborhood, 0..6: 0=8x6 1=16x6 2=32x6
                                    #   3=48x6 4=8x4 5=16x4 6=32x4
    int     nns=1,                  # predictor neurons, 0..4: 0=16 1=32 2=64 3=128 4=256
    int     qual=1,                 # 1 or 2: number of predictor passes averaged
    int     etype=0,                # 0 = weights trained on absolute error, 1 = squared error
    int     pscrn=2,                # prescreener, 0..4: 0=off (predict every pixel)
                                    #   1=original, 2..4=new levels 0..2 (higher = fewer pixels
                                    #   left to cubic interpolation: slower, slightly better)
    int     device_id=0, int num_streams])
```
8/16-bit integer, 16-bit half, 32-bit float. Weights are embedded in the plugin — no external
`nnedi3_weights.bin` file needed. On sm_80+ the predictor runs on tensor cores (fp16 inputs,
fp32 accumulation); older cards fall back to an fp32 path automatically.

## Credits

| filter | ported from |
|---|---|
| Bilateral | [VapourSynth-BilateralGPU](https://github.com/WolframRhodium/VapourSynth-BilateralGPU) (rtc variant), by WolframRhodium |
| BM3D / BM3Dv2 / VAggregate | [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA), by WolframRhodium |
| DFTTest | [vs-dfttest2](https://github.com/AmusementClub/vs-dfttest2) (NVRTC backend), by AmusementClub |
| FGrain | [vs-fgrain-cuda](https://github.com/AmusementClub/vs-fgrain-cuda), by AmusementClub |
| NNEDI3 | [VapourSynth-nnedi3vk](https://github.com/HolyWu/VapourSynth-nnedi3vk), by HolyWu |
| Deband | [vs-placebo](https://github.com/Lypheo/vs-placebo), by Lypheo (libplacebo's `pl_shader_deband` + `pl_shader_dither`) |
| NLMeans | [KNLMeansCL](https://github.com/Khanattila/KNLMeansCL), by Khanattila |
| EEDI3 / EEDI3H | [VapourSynth-EEDI3](https://github.com/HolyWu/VapourSynth-EEDI3) (by HolyWu) and [vapoursynth-zip](https://github.com/dnjulek/vapoursynth-zip) (by dnjulek) — the CPU references |

Built on [cudaz](https://github.com/akhildevelops/cudaz) (CUDA driver API + NVRTC bindings,
vendored in `cudaz/`).
