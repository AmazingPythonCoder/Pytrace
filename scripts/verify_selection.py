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
from src.scene.lights import DirectionalLight, PointLight
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Mesh, Plane, Sphere
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

    mesh_scene = Scene(
        objects=[
            Mesh.cube(
                name="Pick Cube",
                position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                material=DiffuseMaterial(),
            )
        ]
    )
    assert pick(400, 300, viewport, camera, mesh_scene) is mesh_scene.objects[0]

    light_scene = Scene(
        lights=[
            PointLight(
                name="Pick Point",
                position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            ),
            DirectionalLight(
                name="Pick Sun",
                position=np.array([2.0, 0.0, 1.0], dtype=np.float64),
            ),
        ]
    )
    assert pick(400, 300, viewport, camera, light_scene) is light_scene.lights[0]

    camera_scene = Scene()
    camera_scene.camera.position = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    assert pick(400, 300, viewport, camera, camera_scene) is camera_scene.camera

    print("Selection OK: sphere, plane, mesh, light, camera, miss, and nearest ordering verified")


if __name__ == "__main__":
    main()
