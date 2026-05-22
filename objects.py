from __future__ import annotations

from dataclasses import dataclass, field

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
