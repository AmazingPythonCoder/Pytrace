"""Verify CPU tile callbacks and editor-only render features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.render_control import RenderCancelled
from src.raytracer.renderer import render
from src.scene.camera import Camera
from src.scene.lights import DirectionalLight
from src.scene.materials import DiffuseMaterial, EmissiveMaterial
from src.scene.objects import Mesh, Sphere
from src.scene.scene import RenderConfig, Scene


def main() -> None:
    scene = Scene(
        objects=[
            Mesh.cube(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                material=DiffuseMaterial(color=np.array([0.6, 0.25, 0.2], dtype=np.float64)),
            ),
            Sphere(
                name="Glow",
                position=np.array([0.0, 1.2, 0.0], dtype=np.float64),
                radius=0.2,
                material=EmissiveMaterial(color=np.array([1.0, 0.8, 0.4], dtype=np.float64), strength=1.5),
            ),
        ],
        lights=[
            DirectionalLight(
                direction=np.array([-1.0, -2.0, -1.0], dtype=np.float64),
                intensity=1.0,
            )
        ],
        camera=Camera(
            position=np.array([0.0, 1.2, 4.0], dtype=np.float64),
            target=np.array([0.0, 0.2, 0.0], dtype=np.float64),
            aperture=0.03,
            focus_distance=4.0,
        ),
        render=RenderConfig(width=24, height=16, samples=1, max_bounces=2, render_mode="direct"),
    )

    callbacks: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = []
    image = render(
        scene,
        use_gpu=False,
        parallel=False,
        tile_callback=lambda bounds, tile: callbacks.append((bounds, tile.shape)),
    )
    assert image.shape == (16, 24, 3)
    assert callbacks == [((0, 0, 24, 16), (16, 24, 3))]
    assert np.isfinite(image).all()
    assert float(image.max()) > 0.0

    scene.render.render_mode = "path"
    path_image = render(scene, use_gpu=False, parallel=False)
    assert path_image.shape == image.shape
    assert np.isfinite(path_image).all()

    scene.render.width = 80
    scene.render.height = 16
    scene.render.render_mode = "direct"
    progress_calls: list[tuple[int, int]] = []
    try:
        render(
            scene,
            use_gpu=False,
            parallel=False,
            progress_callback=lambda done, total: progress_calls.append((done, total)),
            cancel_callback=lambda: len(progress_calls) >= 1,
        )
    except RenderCancelled:
        pass
    else:
        raise AssertionError("expected sequential CPU render cancellation between tiles")
    assert progress_calls and progress_calls[0] == (1, 2)

    print("Render callbacks OK: tile callback, cancellation, mesh, directional, emissive, DOF, and path mode verified")


if __name__ == "__main__":
    main()
