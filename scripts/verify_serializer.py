"""Verify Scene JSON round-tripping for editor save/load."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scene.materials import DiffuseMaterial, GlassMaterial, SpecularMaterial
from src.scene.objects import Plane, Sphere
from src.scene.scene import Scene


def main() -> None:
    scene = Scene.default()
    data = scene.to_dict()
    restored = Scene.from_dict(data)

    assert len(restored.objects) == len(scene.objects)
    assert len(restored.lights) == len(scene.lights)
    assert restored.selected is None
    assert np.allclose(restored.camera.position, scene.camera.position)
    assert np.allclose(restored.camera.target, scene.camera.target)
    assert restored.camera.fov == scene.camera.fov
    assert restored.render.width == scene.render.width
    assert restored.render.height == scene.render.height
    assert restored.render.samples == scene.render.samples
    assert restored.render.max_bounces == scene.render.max_bounces

    object_types = {type(obj) for obj in restored.objects}
    material_types = {type(obj.material) for obj in restored.objects}
    assert Sphere in object_types
    assert Plane in object_types
    assert DiffuseMaterial in material_types
    assert SpecularMaterial in material_types
    assert GlassMaterial in material_types

    print("Serializer OK: default scene round-trips")


if __name__ == "__main__":
    main()

