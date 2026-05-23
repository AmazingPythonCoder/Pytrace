from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / length


@dataclass
class OrbitCamera:
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.8, 0.0], dtype=np.float64))
    distance: float = 9.0
    yaw: float = 0.0
    pitch: float = 16.0
    fov: float = 55.0

    @property
    def eye(self) -> np.ndarray:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        x = self.target[0] + self.distance * math.cos(pitch) * math.sin(yaw)
        y = self.target[1] + self.distance * math.sin(pitch)
        z = self.target[2] + self.distance * math.cos(pitch) * math.cos(yaw)
        return np.array([x, y, z], dtype=np.float64)

    @property
    def up(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    @property
    def forward(self) -> np.ndarray:
        return _normalize(self.target - self.eye)

    @property
    def right(self) -> np.ndarray:
        return _normalize(np.cross(self.forward, self.up))

    @property
    def view_up(self) -> np.ndarray:
        return _normalize(np.cross(self.right, self.forward))

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.35
        self.pitch = max(-85.0, min(85.0, self.pitch - dy * 0.35))

    def look(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.16
        self.pitch = max(-85.0, min(85.0, self.pitch - dy * 0.16))

    def pan(self, dx: float, dy: float) -> None:
        speed = self.distance * 0.0015
        self.target -= self.right * dx * speed
        self.target += self.view_up * dy * speed

    def move(self, local_right: float, local_up: float, local_forward: float, amount: float) -> None:
        delta = (
            self.right * local_right
            + self.up * local_up
            + self.forward * local_forward
        ) * amount
        self.target += delta

    def zoom(self, amount: float) -> None:
        factor = 0.88 if amount > 0 else 1.14
        self.distance = max(0.4, min(250.0, self.distance * factor))
