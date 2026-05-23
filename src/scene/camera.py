from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.raytracer.camera_frame import compute_camera_frame, generate_ray as ray_from_frame
from src.raytracer.ray import Ray


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.0, 8.0], dtype=np.float64))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64))
    fov: float = 55.0
    aperture: float = 0.0
    focus_distance: float | None = None

    def generate_ray(self, px: float, py: float, image_width: int, image_height: int) -> Ray:
        frame = compute_camera_frame(
            self.position,
            self.target,
            self.up,
            self.fov,
            image_width,
            image_height,
            self.aperture,
            self.focus_distance,
        )
        return ray_from_frame(px, py, frame)
