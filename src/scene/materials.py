from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Material:
    type: str = "material"


@dataclass
class DiffuseMaterial(Material):
    type: str = "diffuse"
    color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=np.float64))
    roughness: float = 1.0


@dataclass
class SpecularMaterial(Material):
    type: str = "specular"
    color: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.9, 0.9], dtype=np.float64))
    roughness: float = 0.0
    ior: float = 1.5


@dataclass
class GlassMaterial(Material):
    type: str = "glass"
    ior: float = 1.45
    roughness: float = 0.0
    absorption_color: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    tint: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
