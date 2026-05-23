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
from src.scene.camera import Camera
from src.scene.lights import PointLight
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Sphere
from src.scene.scene import Scene
from src.scene.scene import RenderConfig


def stable_parity_scene() -> Scene:
    """Small diffuse scene with no glass/specular/area-light path randomness."""
    return Scene(
        objects=[
            Sphere(
                name="GPU Check Sphere",
                position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                radius=1.0,
                material=DiffuseMaterial(
                    color=np.array([0.8, 0.2, 0.2], dtype=np.float64),
                    roughness=1.0,
                ),
            )
        ],
        lights=[
            PointLight(
                name="GPU Check Light",
                position=np.array([0.0, 4.0, 4.0], dtype=np.float64),
                intensity=80.0,
            )
        ],
        camera=Camera(
            position=np.array([0.0, 0.0, 5.0], dtype=np.float64),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            fov=35.0,
        ),
        render=RenderConfig(
            width=32,
            height=18,
            samples=64,
            max_bounces=2,
            area_light_samples=1,
        ),
    )


def main() -> int:
    if not cuda.is_available():
        print("verify_gpu: SKIP (CUDA not available)")
        return 0

    scene = stable_parity_scene()

    cpu = render(scene, use_gpu=False, parallel=False)
    gpu = render(scene, use_gpu=True)

    assert cpu.shape == gpu.shape == (18, 32, 3)
    diff = np.abs(cpu - gpu)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    p99_diff = float(np.percentile(diff, 99))
    print(
        f"CPU vs GPU HDR diff: mean={mean_diff:.6f}, "
        f"p99={p99_diff:.6f}, max={max_diff:.6f}"
    )

    mean_threshold = 0.02
    p99_threshold = 0.15
    if mean_diff >= mean_threshold or p99_diff >= p99_threshold:
        print(
            "FAIL: stochastic CPU/GPU parity drift exceeded thresholds "
            f"(mean < {mean_threshold}, p99 < {p99_threshold})"
        )
        return 1

    showcase = Scene.default()
    showcase.render.width = 32
    showcase.render.height = 18
    showcase.render.samples = 1
    showcase.render.max_bounces = 2
    image = render(showcase, use_gpu=True)
    assert image.shape == (18, 32, 3)
    assert np.isfinite(image).all()
    assert float(image.max()) > 0.0

    print("verify_gpu: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
