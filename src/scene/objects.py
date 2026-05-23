from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.math.vec3 import normalize

from .materials import DiffuseMaterial, Material


@dataclass
class SceneObject:
    name: str = "Object"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    material: Material = field(default_factory=DiffuseMaterial)
    visible: bool = True


@dataclass
class Sphere(SceneObject):
    radius: float = 0.5
    name: str = "Sphere"


@dataclass
class Plane(SceneObject):
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64))
    name: str = "Plane"

    def __post_init__(self) -> None:
        self.normal = normalize(self.normal)


@dataclass
class Mesh(SceneObject):
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    normals: np.ndarray | None = None
    source_path: str = ""
    name: str = "Mesh"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape((-1, 3))
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape((-1, 3))
        if self.normals is not None:
            normals = np.asarray(self.normals, dtype=np.float64).reshape((-1, 3))
            self.normals = np.array([normalize(n) for n in normals], dtype=np.float64)

    @classmethod
    def cube(cls, size: float = 1.0, **kwargs) -> "Mesh":
        name = kwargs.pop("name", "Cube")
        h = float(size) * 0.5
        vertices = np.array(
            [
                [-h, -h, -h],
                [h, -h, -h],
                [h, h, -h],
                [-h, h, -h],
                [-h, -h, h],
                [h, -h, h],
                [h, h, h],
                [-h, h, h],
            ],
            dtype=np.float64,
        )
        triangles = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [1, 2, 6],
                [1, 6, 5],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int64,
        )
        return cls(vertices=vertices, triangles=triangles, name=name, **kwargs)

    @classmethod
    def from_obj(cls, path: str | Path, **kwargs) -> "Mesh":
        path = Path(path)
        name = kwargs.pop("name", path.stem)
        vertices: list[list[float]] = []
        obj_normals: list[list[float]] = []
        normal_accum: list[np.ndarray] = []
        triangles: list[list[int]] = []

        def parse_index(raw: str, count: int) -> int:
            idx = int(raw)
            return idx - 1 if idx > 0 else count + idx

        def parse_face_token(token: str) -> tuple[int, int | None]:
            parts = token.split("/")
            vertex_index = parse_index(parts[0], len(vertices))
            normal_index = None
            if len(parts) >= 3 and parts[2]:
                normal_index = parse_index(parts[2], len(obj_normals))
            return vertex_index, normal_index

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                normal_accum.append(np.zeros(3, dtype=np.float64))
            elif parts[0] == "vn" and len(parts) >= 4:
                obj_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face: list[int] = []
                for token in parts[1:]:
                    if not token.split("/")[0]:
                        continue
                    vertex_index, normal_index = parse_face_token(token)
                    face.append(vertex_index)
                    if normal_index is not None and 0 <= normal_index < len(obj_normals):
                        normal_accum[vertex_index] += np.asarray(obj_normals[normal_index], dtype=np.float64)
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])

        if not vertices or not triangles:
            raise ValueError(f"OBJ file has no usable mesh faces: {path}")

        vertex_normals = None
        if obj_normals and any(float(np.linalg.norm(n)) > 1e-12 for n in normal_accum):
            vertex_normals = np.zeros((len(vertices), 3), dtype=np.float64)
            for i, normal in enumerate(normal_accum):
                length = float(np.linalg.norm(normal))
                if length > 1e-12:
                    vertex_normals[i] = normal / length
            for tri in triangles:
                face_normal = np.cross(
                    np.asarray(vertices[tri[1]], dtype=np.float64) - np.asarray(vertices[tri[0]], dtype=np.float64),
                    np.asarray(vertices[tri[2]], dtype=np.float64) - np.asarray(vertices[tri[0]], dtype=np.float64),
                )
                face_len = float(np.linalg.norm(face_normal))
                if face_len <= 1e-12:
                    continue
                face_normal = face_normal / face_len
                for vertex_index in tri:
                    if float(np.linalg.norm(vertex_normals[vertex_index])) <= 1e-12:
                        vertex_normals[vertex_index] = face_normal

        return cls(
            vertices=np.asarray(vertices, dtype=np.float64),
            triangles=np.asarray(triangles, dtype=np.int64),
            normals=vertex_normals,
            source_path=str(path),
            name=name,
            **kwargs,
        )
