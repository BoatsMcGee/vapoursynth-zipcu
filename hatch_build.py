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

# NVRTC is LoadLibrary'd / dlopen'd from the plugin directory (see cudaz dynlib.zig).
# Do NOT ship libcuda / nvcuda — that is the host driver.
WINDOWS_NVRTC = (
    "nvrtc64_130_0.dll",
    "nvrtc-builtins64_130.dll",
)

# Prefer versioned sonames; we also recreate the libnvrtc.so.13 alias the loader opens.
LINUX_NVRTC_GLOBS = (
    "libnvrtc.so*",
    "libnvrtc-builtins.so*",
)


def resolve_cuda_path() -> Path:
    """Toolkit root with include/cuda.h (CUDA_PATH, else common Linux installs)."""
    env = os.environ.get("CUDA_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    if sys.platform != "win32":
        candidates.extend(
            Path(p)
            for p in (
                "/usr/local/cuda-13.0",
                "/usr/local/cuda",
                "/usr/lib/cuda",
                "/opt/cuda",
                "/usr",
            )
        )
    for root in candidates:
        if (root / "include" / "cuda.h").is_file():
            return root
    raise RuntimeError(
        "CUDA toolkit not found (need include/cuda.h). Set CUDA_PATH to the toolkit root."
    )


def _search_dirs(cuda_path: Path, os_name: str) -> list[Path]:
    if os_name == "windows":
        return [
            cuda_path / "bin" / "x64",
            cuda_path / "bin",
        ]
    return [
        cuda_path / "lib64",
        cuda_path / "lib" / "x86_64-linux-gnu",
        cuda_path / "targets" / "x86_64-linux" / "lib",
        cuda_path / "lib",
    ]


def copy_nvrtc_runtime(cuda_path: Path, dest: Path, os_name: str) -> list[str]:
    """Copy NVRTC shared libs into the plugin dir. Returns basenames that were placed."""
    search = [d for d in _search_dirs(cuda_path, os_name) if d.is_dir()]
    if not search:
        raise RuntimeError(f"no CUDA lib/bin dirs under {cuda_path}")

    placed: list[str] = []

    if os_name == "windows":
        for name in WINDOWS_NVRTC:
            src = next((d / name for d in search if (d / name).is_file()), None)
            if src is None:
                raise RuntimeError(
                    f"missing {name} under {cuda_path} (bin/x64). Install the CUDA "
                    f"NVRTC runtime component, or set CUDA_PATH."
                )
            shutil.copy2(src, dest / name)
            placed.append(name)
        return placed

    # Linux: copy every matching real file + symlink so sonames resolve next to the plugin.
    found_any = False
    for pattern in LINUX_NVRTC_GLOBS:
        for d in search:
            for src in sorted(d.glob(pattern)):
                if src.is_dir():
                    continue
                found_any = True
                target = dest / src.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                if src.is_symlink():
                    # Preserve relative soname links (libnvrtc.so.13 -> libnvrtc.so.13.0.88).
                    link = os.readlink(src)
                    target.symlink_to(link)
                else:
                    shutil.copy2(src, target)
                placed.append(src.name)

    if not found_any:
        raise RuntimeError(
            f"no libnvrtc.so* under {cuda_path} (install cuda-nvrtc-13-0 runtime, set CUDA_PATH)."
        )

    # Ensure the name dynlib opens first exists (libnvrtc.so.13).
    primary = dest / "libnvrtc.so.13"
    if not primary.exists() and not primary.is_symlink():
        # Point at the highest versioned real file we just copied.
        reals = sorted(
            (dest / n for n in placed if n.startswith("libnvrtc.so.") and (dest / n).is_file() and not (dest / n).is_symlink()),
            key=lambda p: p.name,
        )
        if not reals:
            raise RuntimeError("copied NVRTC files but found no real libnvrtc.so.* to alias as libnvrtc.so.13")
        primary.symlink_to(reals[-1].name)
        placed.append("libnvrtc.so.13")

    return placed


class CustomHook(BuildHookInterface[Any]):
    """Compile the plugin with Zig, vendor NVRTC next to it, and stage the wheel plugin dir.

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

        cuda_path = resolve_cuda_path()
        placed = copy_nvrtc_runtime(cuda_path, self.target_dir, os_name)
        print(f"hatch_build: vendored NVRTC from {cuda_path}: {', '.join(placed)}")

        # Manifest lists only the VS plugin module stem (not the NVRTC companions).
        manifest = self.target_dir / "manifest.vs"
        manifest.write_text(f"[VapourSynth Manifest V1]\n{Path(lib_name).stem}\n")

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        # The wheel is already assembled here; drop the whole staged tree (vapoursynth/…) so the
        # source checkout stays clean. parents[1] is "vapoursynth/" (parents[0] is ".../plugins").
        shutil.rmtree(self.target_dir.parents[1], ignore_errors=True)
