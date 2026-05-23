"""Verify Scene JSON round-tripping for editor save/load."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scene.camera import Camera
from src.scene.lights import DirectionalLight
from src.scene.materials import DiffuseMaterial, EmissiveMaterial, GlassMaterial, SpecularMaterial
from src.scene.objects import Mesh, Plane, Sphere
from src.scene.scene import RenderConfig, Scene


def main() -> None:
    scene = Scene.default()
    scene.objects.append(
        Mesh.cube(
            name="Serialized Cube",
            position=np.array([1.0, 0.5, -1.0], dtype=np.float64),
            normals=np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            source_path="fixtures/cube.obj",
            material=EmissiveMaterial(color=np.array([1.0, 0.7, 0.3], dtype=np.float64), strength=2.5),
        )
    )
    scene.lights.append(
        DirectionalLight(
            name="Serializer Sun",
            direction=np.array([-1.0, -2.0, -0.5], dtype=np.float64),
            intensity=1.75,
        )
    )
    scene.camera = Camera(
        position=np.array([1.0, 2.0, 6.0], dtype=np.float64),
        target=np.array([0.0, 0.5, 0.0], dtype=np.float64),
        fov=42.0,
        aperture=0.04,
        focus_distance=5.0,
    )
    scene.render = RenderConfig(
        width=320,
        height=180,
        samples=3,
        max_bounces=4,
        exposure=1.2,
        area_light_samples=2,
        render_mode="path",
        background_mode="environment",
        environment_path="envs/studio.hdr",
        background_color=np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )
    data = scene.to_dict()
    restored = Scene.from_dict(data)

    assert len(restored.objects) == len(scene.objects)
    assert len(restored.lights) == len(scene.lights)
    assert restored.selected is None
    assert np.allclose(restored.camera.position, scene.camera.position)
    assert np.allclose(restored.camera.target, scene.camera.target)
    assert restored.camera.fov == scene.camera.fov
    assert restored.camera.aperture == scene.camera.aperture
    assert restored.camera.focus_distance == scene.camera.focus_distance
    assert restored.render.width == scene.render.width
    assert restored.render.height == scene.render.height
    assert restored.render.samples == scene.render.samples
    assert restored.render.max_bounces == scene.render.max_bounces
    assert restored.render.render_mode == "path"
    assert restored.render.background_mode == "environment"
    assert restored.render.environment_path == "envs/studio.hdr"
    assert np.allclose(restored.render.background_color, scene.render.background_color)

    object_types = {type(obj) for obj in restored.objects}
    material_types = {type(obj.material) for obj in restored.objects}
    assert Sphere in object_types
    assert Plane in object_types
    assert Mesh in object_types
    assert DiffuseMaterial in material_types
    assert SpecularMaterial in material_types
    assert GlassMaterial in material_types
    assert EmissiveMaterial in material_types

    meshes = [obj for obj in restored.objects if isinstance(obj, Mesh)]
    assert meshes
    assert meshes[-1].source_path == "fixtures/cube.obj"
    assert meshes[-1].normals is not None
    assert meshes[-1].normals.shape == (8, 3)
    assert any(isinstance(light, DirectionalLight) for light in restored.lights)

    print("Serializer OK: default and stretch scene fields round-trip")


if __name__ == "__main__":
    main()
