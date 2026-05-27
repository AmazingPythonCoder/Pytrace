"""Compare CPU vs GPU renders (skipped when CUDA unavailable)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda

from src.raytracer.gpu.renderer import render_gpu
from src.raytracer.render_context import build_render_context
from src.raytracer.renderer import render
from src.scene.camera import Camera
from src.scene.lights import DirectionalLight, PointLight
from src.scene.materials import DiffuseMaterial, EmissiveMaterial
from src.scene.objects import Mesh, Sphere
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


def advanced_gpu_scene(environment_path: Path, render_mode: str = "direct") -> Scene:
    """Exercise GPU-only parity features that used to force CPU fallback."""
    return Scene(
        objects=[
            Mesh.cube(
                name="GPU Cube",
                position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.62, 0.3, 0.18], dtype=np.float64),
                    roughness=0.85,
                ),
            ),
            Sphere(
                name="GPU Emissive",
                position=np.array([0.0, 1.25, 0.0], dtype=np.float64),
                radius=0.22,
                material=EmissiveMaterial(
                    color=np.array([1.0, 0.75, 0.35], dtype=np.float64),
                    strength=2.0,
                ),
            ),
        ],
        lights=[
            DirectionalLight(
                name="GPU Sun",
                direction=np.array([-1.0, -2.0, -1.0], dtype=np.float64),
                color=np.array([0.9, 0.96, 1.0], dtype=np.float64),
                intensity=1.2,
            )
        ],
        camera=Camera(
            position=np.array([0.0, 1.1, 4.0], dtype=np.float64),
            target=np.array([0.0, 0.25, 0.0], dtype=np.float64),
            fov=38.0,
            aperture=0.025,
            focus_distance=4.0,
        ),
        render=RenderConfig(
            width=20,
            height=12,
            samples=2,
            max_bounces=3,
            area_light_samples=1,
            render_mode=render_mode,
            background_mode="environment",
            environment_path=str(environment_path),
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

    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / "gpu_env.png"
        Image.fromarray(
            np.array(
                [
                    [[40, 70, 120], [255, 210, 120], [120, 180, 255], [20, 25, 45]],
                    [[10, 12, 20], [80, 90, 100], [220, 180, 90], [30, 30, 35]],
                ],
                dtype=np.uint8,
            ),
            mode="RGB",
        ).save(env_path)

        advanced = advanced_gpu_scene(env_path, render_mode="direct")
        ctx = build_render_context(advanced)
        assert ctx.gpu_supported
        assert ctx.triangles.shape[0] > 0
        assert ctx.environment.shape == (2, 4, 3)
        assert int(ctx.render_mode) == 0
        assert int(ctx.background_mode) == 2
        assert float(ctx.camera_frame.aperture) > 0.0
        assert bool(np.any(ctx.materials[:, 0] == 3.0))
        assert bool(np.any(ctx.lights[:, 0] == 2.0))
        preview_frames: list[np.ndarray] = []
        advanced_image = render_gpu(ctx, preview_callback=lambda image: preview_frames.append(image.copy()))
        assert advanced_image.shape == (12, 20, 3)
        assert np.isfinite(advanced_image).all()
        assert float(advanced_image.max()) > 0.0
        assert len(preview_frames) == 2
        assert all(frame.shape == advanced_image.shape for frame in preview_frames)
        assert all(np.isfinite(frame).all() for frame in preview_frames)
        assert np.allclose(preview_frames[-1], advanced_image)

        path_scene = advanced_gpu_scene(env_path, render_mode="path")
        path_ctx = build_render_context(path_scene)
        assert path_ctx.gpu_supported
        assert int(path_ctx.render_mode) == 1
        path_image = render_gpu(path_ctx)
        assert path_image.shape == (12, 20, 3)
        assert np.isfinite(path_image).all()
        assert float(path_image.max()) > 0.0

    print("verify_gpu: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
