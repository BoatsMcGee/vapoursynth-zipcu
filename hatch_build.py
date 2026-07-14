import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

# One entry per shipped wheel — CUDA exists on x86_64 Linux/Windows only. `zig_target`
# pins glibc 2.17 (manylinux) resp. the msvc ABI (NVIDIA ships MSVC import libs; the gnu
# ABI cannot link them); `platform_tag` is the wheel's platform tag; `os` selects the
# shared-library name `zig build` emits (see LIBRARY_NAME).
TARGETS = {
    "x86_64-linux-gnu": {"zig_target": "x86_64-linux-gnu.2.17", "platform_tag": "manylinux_2_17_x86_64", "os": "linux"},
    "x86_64-windows":   {"zig_target": "x86_64-windows-msvc",   "platform_tag": "win_amd64",             "os": "windows"},
}

LIBRARY_NAME = {
    "windows": "vszipcu.dll",
    "linux": "libvszipcu.so",
}

# Host OS -> TARGETS `os`, for a plain (non-ZTARGET) local build.
HOST_OS = "windows" if sys.platform == "win32" else "linux"


class CustomHook(BuildHookInterface[Any]):
    """Compile the plugin with Zig and place its shared library in the wheel's plugin dir.

    Building needs the CUDA toolkit (cuda.h + the libcuda link stub / cuda.lib): set
    `CUDA_PATH`, or rely on the common Linux install locations cudaz probes.
    """

    source_dir = Path("zig-out")
    target_dir = Path("vapoursynth/plugins/vszipcu")

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        self.target_dir.mkdir(parents=True, exist_ok=True)

        zig_cmd = [sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast"]

        # CI passes ZTARGET (e.g. "x86_64-linux-gnu") to build + tag a specific wheel; a bare
        # local build targets this machine and tags with the host platform.
        ztarget = os.environ.get("ZTARGET")
        if ztarget is not None:
            try:
                target = TARGETS[ztarget]
            except KeyError:
                raise ValueError(
                    f"Unsupported ZTARGET {ztarget!r}; expected one of: {', '.join(TARGETS)}"
                ) from None
            build_data["tag"] = f"py3-none-{target['platform_tag']}"
            zig_cmd.append(f"-Dtarget={target['zig_target']}")
            os_name = target["os"]
        else:
            build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"
            os_name = HOST_OS

        subprocess.run(zig_cmd, check=True)

        # Copy exactly this OS's plugin library into the wheel. A missing file fails the build
        # loudly (e.g. no CUDA toolkit to link against) instead of silently shipping an empty,
        # unloadable wheel.
        lib_name = LIBRARY_NAME[os_name]
        matches = sorted(self.source_dir.rglob(lib_name))
        if not matches:
            raise RuntimeError(
                f"Zig build produced no {lib_name} under {self.source_dir}/ — the plugin failed "
                f"to compile/link (CUDA toolkit not found for {os_name}? set CUDA_PATH)."
            )
        shutil.copy2(matches[0], self.target_dir)

        manifest = self.target_dir / "manifest.vs"
        manifest.write_text(f"[VapourSynth Manifest V1]\n{Path(lib_name).stem}\n")

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        # The wheel is already assembled here; drop the whole staged tree (vapoursynth/…) so the
        # source checkout stays clean. parents[1] is "vapoursynth/" (parents[0] is ".../plugins").
        shutil.rmtree(self.target_dir.parents[1], ignore_errors=True)
