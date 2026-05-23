"""Compare CPU vs GPU renders (skipped when CUDA unavailable)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda

from src.raytracer.renderer import render
from src.scene.scene import Scene


def main() -> int:
    if not cuda.is_available():
        print("verify_gpu: SKIP (CUDA not available)")
        return 0

    scene = Scene.default()
    scene.render.width = 64
    scene.render.height = 36
    scene.render.samples = 4
    scene.render.max_bounces = 6

    cpu = render(scene, use_gpu=False, parallel=False)
    gpu = render(scene, use_gpu=True)

    assert cpu.shape == gpu.shape == (36, 64, 3)
    diff = np.abs(cpu - gpu)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"CPU vs GPU max diff: {max_diff:.6f}, mean: {mean_diff:.6f}")

    threshold = 1e-2
    if max_diff >= threshold:
        print(f"FAIL: max diff {max_diff} >= {threshold}")
        return 1

    print("verify_gpu: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
