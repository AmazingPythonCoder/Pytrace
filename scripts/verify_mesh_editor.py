"""Verify mesh/cube scene data, OBJ import, and mesh picking."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.editor.layout import Rect
from src.editor.orbit_camera import OrbitCamera
from src.editor.selection import pick
from src.raytracer.render_context import build_render_context
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Mesh
from src.scene.scene import Scene


def main() -> None:
    cube = Mesh.cube(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        material=DiffuseMaterial(color=np.array([0.7, 0.3, 0.2], dtype=np.float64)),
    )
    scene = Scene(objects=[cube])
    restored = Scene.from_dict(scene.to_dict())
    assert isinstance(restored.objects[0], Mesh)
    assert restored.objects[0].triangles.shape == (12, 3)

    with tempfile.TemporaryDirectory() as tmp:
        obj_path = Path(tmp) / "quad.obj"
        obj_path.write_text(
            "\n".join(
                [
                    "v -0.5 -0.5 0",
                    "v 0.5 -0.5 0",
                    "v 0.5 0.5 0",
                    "v -0.5 0.5 0",
                    "vn 0 0 1",
                    "f 1//1 2//1 3//1 4//1",
                ]
            ),
            encoding="utf-8",
        )
        mesh = Mesh.from_obj(obj_path)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.triangles.shape == (2, 3)
        assert mesh.normals is not None
        assert mesh.normals.shape == (4, 3)
        assert np.allclose(mesh.normals[0], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        assert mesh.source_path == str(obj_path)

    ctx = build_render_context(restored)
    assert ctx.triangles.shape[0] == 12

    camera = OrbitCamera(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov=55.0,
    )
    assert pick(400, 300, Rect(0, 0, 800, 600), camera, restored) is restored.objects[0]
    print("Mesh editor OK: cube round-trip, OBJ import, render packing, and picking verified")


if __name__ == "__main__":
    main()
