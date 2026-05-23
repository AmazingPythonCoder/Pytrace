"""Benchmark render time (warm up JIT, then time one headless-quality pass)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.renderer import render
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Sphere
from src.scene.scene import Scene


def _stress_scene(num_spheres: int) -> Scene:
    """Many spheres in a box — BVH helps vs linear scan."""
    s = Scene.default()
    rng = np.random.default_rng(42)
    for i in range(num_spheres):
        pos = rng.uniform(-4.0, 4.0, size=3)
        pos[1] = abs(pos[1]) + 0.2
        r = float(rng.uniform(0.08, 0.25))
        s.objects.append(
            Sphere(
                name=f"Stress {i}",
                position=pos.astype(np.float64),
                radius=r,
                material=DiffuseMaterial(
                    color=rng.uniform(0.2, 0.9, size=3).astype(np.float64),
                    roughness=float(rng.uniform(0.2, 0.9)),
                ),
            )
        )
    s.render.width = 320
    s.render.height = 180
    s.render.samples = 8
    s.render.max_bounces = 4
    return s


def _bench(label: str, scene: Scene, warmup: bool = True) -> float:
    if warmup:
        w = scene.render
        old_w, old_h, old_s = w.width, w.height, w.samples
        w.width, w.height, w.samples = 32, 18, 1
        render(scene, parallel=False, use_gpu=False)
        w.width, w.height, w.samples = old_w, old_h, old_s

    t0 = time.perf_counter()
    render(scene, parallel=False, use_gpu=False)
    elapsed = time.perf_counter() - t0
    n_obj = sum(1 for o in scene.objects if o.visible)
    print(f"{label}: {elapsed:.3f}s  ({n_obj} visible objects, {scene.render.width}x{scene.render.height}, {scene.render.samples} spp)")
    return elapsed


def main() -> None:
    print("PyTrace render benchmark (single-threaded, JIT warmed up)\n")

    default = Scene.default()
    default.render.width = 320
    default.render.height = 180
    default.render.samples = 16
    _bench("Default room (7 spheres + 6 planes, BVH)", default)

    stress = _stress_scene(200)
    _bench("Stress (+200 spheres)", stress)

    print("\nDone.")


if __name__ == "__main__":
    main()
