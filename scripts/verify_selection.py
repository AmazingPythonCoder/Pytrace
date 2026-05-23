"""Verify editor selection math without opening a Pygame/OpenGL window."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.editor.layout import Rect
from src.editor.orbit_camera import OrbitCamera
from src.editor.selection import pick
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Plane, Sphere
from src.scene.scene import Scene


def _camera() -> OrbitCamera:
    return OrbitCamera(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        distance=5.0,
        yaw=0.0,
        pitch=0.0,
        fov=55.0,
    )


def main() -> None:
    viewport = Rect(0, 0, 800, 600)
    camera = _camera()

    near = Sphere(
        name="Near",
        position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        radius=0.5,
        material=DiffuseMaterial(),
    )
    far = Sphere(
        name="Far",
        position=np.array([0.0, 0.0, -2.0], dtype=np.float64),
        radius=0.8,
        material=DiffuseMaterial(),
    )
    scene = Scene(objects=[far, near])
    assert pick(400, 300, viewport, camera, scene) is near
    assert pick(10, 10, viewport, camera, scene) is None

    plane_scene = Scene(
        objects=[
            Plane(
                name="Ground",
                position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                material=DiffuseMaterial(),
            )
        ]
    )
    assert pick(400, 300, viewport, camera, plane_scene) is plane_scene.objects[0]

    print("Selection OK: sphere hit, plane hit, miss, and nearest ordering verified")


if __name__ == "__main__":
    main()

