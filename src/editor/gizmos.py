from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

from src.scene.camera import Camera
from src.scene.lights import AreaLight, Light
from src.scene.objects import Plane, SceneObject, Sphere
from src.scene.scene import Scene

from .orbit_camera import OrbitCamera


@dataclass
class _Snapshot:
    position: np.ndarray | None = None
    target: np.ndarray | None = None
    rotation: np.ndarray | None = None
    scale: np.ndarray | None = None
    normal: np.ndarray | None = None
    radius: float | None = None


class TransformController:
    def __init__(self) -> None:
        self.mode: str | None = None
        self.axis: str | None = None
        self.item: object | None = None
        self.snapshot = _Snapshot()
        self.last_mouse: tuple[int, int] | None = None

    def status(self) -> str:
        if self.mode is None:
            return "G move | S scale | R rotate | X delete | F12 render"
        axis = f" {self.axis}" if self.axis else ""
        return f"{self.mode.upper()}{axis}: Enter/LMB confirm, Esc cancel"

    def handle_key(self, event: Any, scene: Scene) -> bool:
        if pygame is None or event.type != pygame.KEYDOWN:
            return False
        key = event.key
        if self.mode is None:
            if scene.selected is None:
                return False
            if key == pygame.K_g:
                self.begin("move", scene.selected)
                return True
            if key == pygame.K_s:
                self.begin("scale", scene.selected)
                return True
            if key == pygame.K_r:
                self.begin("rotate", scene.selected)
                return True
            return False

        if key in (pygame.K_x, pygame.K_y, pygame.K_z):
            self.axis = pygame.key.name(key).upper()
            return True
        if key == pygame.K_RETURN:
            self.confirm()
            return True
        if key == pygame.K_ESCAPE:
            self.cancel()
            return True
        return False

    def handle_mouse_button(self, event: Any) -> bool:
        if pygame is None or self.mode is None:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.confirm()
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self.cancel()
            return True
        return False

    def handle_motion(self, event: Any, orbit_camera: OrbitCamera) -> bool:
        if pygame is None or self.mode is None or event.type != pygame.MOUSEMOTION:
            return False
        if self.last_mouse is None:
            self.last_mouse = event.pos
            return True
        dx = event.pos[0] - self.last_mouse[0]
        dy = event.pos[1] - self.last_mouse[1]
        self.last_mouse = event.pos
        if self.mode == "move":
            self._move(dx, dy, orbit_camera)
        elif self.mode == "scale":
            self._scale(dx, dy, orbit_camera)
        elif self.mode == "rotate":
            self._rotate(dx, dy)
        return True

    def begin(self, mode: str, item: object) -> None:
        self.mode = mode
        self.axis = None
        self.item = item
        self.last_mouse = pygame.mouse.get_pos() if pygame is not None else None
        self.snapshot = self._snapshot(item)

    def confirm(self) -> None:
        self.mode = None
        self.axis = None
        self.item = None
        self.last_mouse = None
        self.snapshot = _Snapshot()

    def cancel(self) -> None:
        if self.item is not None:
            self._restore(self.item, self.snapshot)
        self.confirm()

    def _snapshot(self, item: object) -> _Snapshot:
        snap = _Snapshot()
        if hasattr(item, "position"):
            snap.position = np.asarray(getattr(item, "position"), dtype=np.float64).copy()
        if isinstance(item, Camera):
            snap.target = item.target.copy()
        if isinstance(item, SceneObject):
            snap.rotation = item.rotation.copy()
            snap.scale = item.scale.copy()
        if isinstance(item, (Plane, AreaLight)):
            snap.normal = item.normal.copy()
        if isinstance(item, (Sphere, AreaLight)):
            snap.radius = float(item.radius)
        return snap

    def _restore(self, item: object, snap: _Snapshot) -> None:
        if snap.position is not None and hasattr(item, "position"):
            getattr(item, "position")[:] = snap.position
        if isinstance(item, Camera) and snap.target is not None:
            item.target[:] = snap.target
        if isinstance(item, SceneObject):
            if snap.rotation is not None:
                item.rotation[:] = snap.rotation
            if snap.scale is not None:
                item.scale[:] = snap.scale
        if isinstance(item, (Plane, AreaLight)) and snap.normal is not None:
            item.normal[:] = snap.normal
        if isinstance(item, (Sphere, AreaLight)) and snap.radius is not None:
            item.radius = snap.radius

    def _axis_vector(self, axis: str | None) -> np.ndarray:
        if axis == "X":
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis == "Y":
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if axis == "Z":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def _move(self, dx: float, dy: float, orbit_camera: OrbitCamera) -> None:
        item = self.item
        if item is None or not hasattr(item, "position"):
            return
        speed = orbit_camera.distance * 0.002
        if self.axis:
            delta = self._axis_vector(self.axis) * ((dx - dy) * speed)
        else:
            delta = (-orbit_camera.right * dx + orbit_camera.view_up * dy) * speed
        getattr(item, "position")[:] = getattr(item, "position") + delta
        if isinstance(item, Camera):
            item.target[:] = item.target + delta

    def _scale(self, dx: float, dy: float, orbit_camera: OrbitCamera) -> None:
        item = self.item
        if item is None:
            return
        amount = (dx - dy) * orbit_camera.distance * 0.001
        if isinstance(item, Sphere):
            item.radius = max(0.02, item.radius + amount)
        elif isinstance(item, AreaLight):
            item.radius = max(0.02, item.radius + amount)
        elif isinstance(item, SceneObject):
            item.scale[:] = np.maximum(0.02, item.scale + amount)

    def _rotate(self, dx: float, dy: float) -> None:
        item = self.item
        amount = (dx - dy) * 0.01
        axis = self.axis or "Y"
        if isinstance(item, SceneObject):
            idx = {"X": 0, "Y": 1, "Z": 2}[axis]
            item.rotation[idx] += amount * 30.0
        if isinstance(item, (Plane, AreaLight)):
            item.normal[:] = self._rotate_vector(item.normal, axis, amount)
        if isinstance(item, Camera):
            offset = item.target - item.position
            item.target[:] = item.position + self._rotate_vector(offset, axis, amount)

    def _rotate_vector(self, vector: np.ndarray, axis: str, angle: float) -> np.ndarray:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        x, y, z = vector
        if axis == "X":
            rotated = np.array([x, y * c - z * s, y * s + z * c], dtype=np.float64)
        elif axis == "Z":
            rotated = np.array([x * c - y * s, x * s + y * c, z], dtype=np.float64)
        else:
            rotated = np.array([x * c + z * s, y, -x * s + z * c], dtype=np.float64)
        length = float(np.linalg.norm(rotated))
        return rotated / length if length > 1e-12 else vector

