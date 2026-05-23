"""Profile CUDA render stages for one PyTrace scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda

from src.raytracer.gpu.device_context import to_device
from src.raytracer.gpu.renderer import _clear, _launch
from src.raytracer.render_context import build_render_context
from src.scene.scene import Scene


def _level_scene(level: str) -> Scene:
    scene = Scene.default()
    if level == "low":
        scene.render.width = 800
        scene.render.height = 450
        scene.render.samples = 32
        scene.render.max_bounces = 8
    elif level == "med":
        scene.render.width = 1280
        scene.render.height = 720
        scene.render.samples = 128
        scene.render.max_bounces = 10
    elif level == "high":
        scene.render.width = 1920
        scene.render.height = 1080
        scene.render.samples = 384
        scene.render.max_bounces = 12
    elif level == "ultra":
        scene.render.width = 3840
        scene.render.height = 2160
        scene.render.samples = 768
        scene.render.max_bounces = 16
    return scene


def _time(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["low", "med", "high", "ultra"], default="low")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--bounces", type=int, default=None)
    args = parser.parse_args()

    if not cuda.is_available():
        print("CUDA is not available")
        return 1

    scene = _level_scene(args.level)
    if args.width is not None:
        scene.render.width = args.width
    if args.height is not None:
        scene.render.height = args.height
    if args.samples is not None:
        scene.render.samples = args.samples
    if args.bounces is not None:
        scene.render.max_bounces = args.bounces

    print(
        f"Profiling {scene.render.width}x{scene.render.height}, "
        f"{scene.render.samples} spp, {scene.render.max_bounces} bounces"
    )

    ctx, _ = _time("build render context", lambda: build_render_context(scene))
    dev, _ = _time("upload scene buffers", lambda: to_device(ctx))
    image, _ = _time(
        "allocate output",
        lambda: cuda.device_array((dev.height, dev.width, 3), dtype="float32"),
    )

    print("warming CUDA kernels...")
    _clear(dev, image)
    _launch(dev, image, 0, 1)
    cuda.synchronize()

    _, clear_elapsed = _time("clear output", lambda: (_clear(dev, image), cuda.synchronize()))
    _, kernel_elapsed = _time(
        "render kernel",
        lambda: (_launch(dev, image, 0, dev.samples), cuda.synchronize()),
    )
    _, download_elapsed = _time("download image", lambda: image.copy_to_host())

    total = clear_elapsed + kernel_elapsed + download_elapsed
    print(f"measured GPU stages: {total:.3f}s")
    if total > 0.0:
        print(f"kernel share: {kernel_elapsed / total * 100.0:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
