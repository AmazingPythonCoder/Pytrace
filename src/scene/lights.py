from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Light:
    type: str = "light"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    intensity: float = 100.0
    name: str = "Light"


@dataclass
class PointLight(Light):
    type: str = "point"
    position: np.ndarray = field(default_factory=lambda: np.array([3.0, 6.0, 3.0], dtype=np.float64))
    name: str = "Point Light"


@dataclass
class AreaLight(Light):
    """Disk area light. ``normal`` controls the disk orientation."""

    type: str = "area"
    position: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=np.float64))
    name: str = "Area Light"
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0], dtype=np.float64))
    radius: float = 1.0

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64)
        length = float(np.linalg.norm(n))
        if length > 1e-12:
            self.normal = n / length
