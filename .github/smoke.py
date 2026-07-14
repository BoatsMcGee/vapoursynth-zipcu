"""Smoke test for the installed wheel on a runner WITHOUT an NVIDIA GPU.

Asserts three things: the wheel put the plugin where VapourSynth autoloads it, the
plugin actually loads and registers (its library resolves — on Linux that exercises the
libcuda stub the workflow stages), and creating a filter fails with a clean VapourSynth
error rather than a crash. On a machine WITH a working driver the creation succeeds and
a frame is rendered instead — both outcomes pass.

Expects `pip install wheelhouse/*.whl` before this runs. VapourSynth auto-loads plugins
from site-packages — do not call LoadPlugin here.
"""
import glob
import os
import site
import sys

import vapoursynth as vs

library_suffixes = {".so", ".dll"}


def installed_plugin_path() -> str:
    for site_dir in site.getsitepackages() + ([site.getusersitepackages()] if site.getusersitepackages() else []):
        plugin_dir = os.path.join(site_dir, "vapoursynth", "plugins", "vszipcu")
        for path in sorted(glob.glob(os.path.join(plugin_dir, "*"))):
            if os.path.isfile(path) and os.path.splitext(path)[1] in library_suffixes:
                return path
    sys.exit("no vszipcu plugin in installed wheel (vapoursynth/plugins/vszipcu/)")


plugin_path = installed_plugin_path()
core = vs.core

if not hasattr(core, "vszipcu"):
    sys.exit(f"vszipcu not auto-loaded from installed wheel ({plugin_path})")

print(f"  using installed wheel: {plugin_path}")
print(f"  registered functions: {sorted(f.name for f in core.vszipcu.functions())}")

src = core.std.BlankClip(width=256, height=256, format=vs.GRAYS, length=2, color=0.5)
try:
    clip = core.vszipcu.GaussBlur(src, sigma=[1.5])
    frame = clip.get_frame(0)
    print("  GPU present: rendered a frame OK")
except vs.Error as e:
    # No driver/device on CI runners — creation must fail CLEANLY, not crash.
    print(f"  no GPU (expected on CI): clean error: {str(e).strip().splitlines()[-1]}")

print("smoke OK")
