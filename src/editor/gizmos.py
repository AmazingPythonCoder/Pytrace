from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

pygame = None  # Legacy Pygame event handlers below are kept as no-op compatibility shims.

from src.scene.camera import Camera
from src.scene.lights import AreaLight, DirectionalLight, Light
from src.scene.objects import Plane, SceneObject, Sphere
from src.scene.scene import Scene

from .layout import Rect
from .orbit_camera import OrbitCamera


@dataclass
class _Snapshot:
    position: np.ndarray | None = None
    target: np.ndarray | None = None
    rotation: np.ndarray | None = None
    scale: np.ndarray | None = None
    normal: np.ndarray | None = None
    direction: np.ndarray | None = None
    radius: float | None = None


class TransformController:
    def __init__(self) -> None:
        self.mode: str | None = None
        self.axis: str | None = None
        self.item: object | None = None
        self.snapshot = _Snapshot()
        self.last_mouse: tuple[float, float] | None = None

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

    def handle_gizmo_press(self, event: Any, scene: Scene, orbit_camera: OrbitCamera, viewport: Rect) -> bool:
        if pygame is None or self.mode is not None:
            return False
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1 or scene.selected is None:
            return False
        position = self._item_position(scene.selected)
        if position is None:
            return False
        axis = self._pick_axis(event.pos, position, orbit_camera, viewport)
        if axis is None:
            return False
        self.begin("move", scene.selected)
        self.axis = axis
        return True

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

    def begin(self, mode: str, item: object, mouse_pos: tuple[float, float] | None = None) -> None:
        self.mode = mode
        self.axis = None
        self.item = item
        self.last_mouse = mouse_pos if mouse_pos is not None else (pygame.mouse.get_pos() if pygame is not None else None)
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

    def handle_command(self, command: str, scene: Scene, mouse_pos: tuple[float, float] | None = None) -> bool:
        """Backend-neutral keyboard command handler used by the ImGui editor."""
        key = command.lower()
        if self.mode is None:
            if scene.selected is None:
                return False
            if key == "g":
                self.begin("move", scene.selected, mouse_pos)
                return True
            if key == "s":
                self.begin("scale", scene.selected, mouse_pos)
                return True
            if key == "r":
                self.begin("rotate", scene.selected, mouse_pos)
                return True
            return False

        if key in {"x", "y", "z"}:
            self.axis = key.upper()
            return True
        if key in {"enter", "return"}:
            self.confirm()
            return True
        if key in {"escape", "esc"}:
            self.cancel()
            return True
        return False

    def pick_axis_at(
        self,
        mouse_pos: tuple[float, float],
        scene: Scene,
        orbit_camera: OrbitCamera,
        viewport: Rect,
    ) -> str | None:
        if self.mode is not None or scene.selected is None:
            return None
        position = self._item_position(scene.selected)
        if position is None:
            return None
        return self._pick_axis(mouse_pos, position, orbit_camera, viewport)

    def drag(self, mouse_pos: tuple[float, float], orbit_camera: OrbitCamera, viewport_height: int = 900, speed_scale: float = 1.0) -> bool:
        if self.mode is None:
            return False
        if self.last_mouse is None:
            self.last_mouse = mouse_pos
            return True
        dx = mouse_pos[0] - self.last_mouse[0]
        dy = mouse_pos[1] - self.last_mouse[1]
        self.last_mouse = mouse_pos
        dx = max(-80.0, min(80.0, dx))
        dy = max(-80.0, min(80.0, dy))
        if self.mode == "move":
            self._move(dx, dy, orbit_camera, viewport_height, speed_scale)
        elif self.mode == "scale":
            self._scale(dx, dy, orbit_camera, viewport_height, speed_scale)
        elif self.mode == "rotate":
            self._rotate(dx, dy, speed_scale)
        return True

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
        if isinstance(item, DirectionalLight):
            snap.direction = item.direction.copy()
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
        if isinstance(item, DirectionalLight) and snap.direction is not None:
            item.direction[:] = snap.direction
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

    def _item_position(self, item: object) -> np.ndarray | None:
        if hasattr(item, "position"):
            return np.asarray(getattr(item, "position"), dtype=np.float64)
        return None

    def _project(self, point: np.ndarray, orbit_camera: OrbitCamera, viewport: Rect) -> tuple[float, float] | None:
        rel = point - orbit_camera.eye
        z = float(np.dot(rel, orbit_camera.forward))
        if z <= 1e-6:
            return None
        aspect = viewport.w / viewport.h
        tan_half = float(np.tan(np.radians(orbit_camera.fov * 0.5)))
        ndc_x = float(np.dot(rel, orbit_camera.right)) / (z * tan_half * aspect)
        ndc_y = float(np.dot(rel, orbit_camera.view_up)) / (z * tan_half)
        if abs(ndc_x) > 1.4 or abs(ndc_y) > 1.4:
            return None
        sx = viewport.x + (ndc_x + 1.0) * 0.5 * viewport.w
        sy = viewport.y + (1.0 - ndc_y) * 0.5 * viewport.h
        return sx, sy

    def _distance_to_segment(self, point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        px, py = float(point[0]), float(point[1])
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-9:
            return float(np.hypot(px - ax, py - ay))
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length_sq))
        cx = ax + t * vx
        cy = ay + t * vy
        return float(np.hypot(px - cx, py - cy))

    def _pick_axis(
        self,
        mouse_pos: tuple[float, float],
        position: np.ndarray,
        orbit_camera: OrbitCamera,
        viewport: Rect,
    ) -> str | None:
        origin = self._project(position, orbit_camera, viewport)
        if origin is None:
            return None
        best_axis: str | None = None
        best_distance = 12.0
        for axis, direction in (
            ("X", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        ):
            end = self._project(position + direction * 1.25, orbit_camera, viewport)
            if end is None:
                continue
            distance = self._distance_to_segment(mouse_pos, origin, end)
            if distance < best_distance:
                best_distance = distance
                best_axis = axis
        return best_axis

    def _world_per_pixel(self, orbit_camera: OrbitCamera, viewport_height: int) -> float:
        pixels = max(1.0, float(viewport_height))
        world_height = 2.0 * np.tan(np.radians(orbit_camera.fov) * 0.5) * orbit_camera.distance
        return float(world_height / pixels)

    def _move(self, dx: float, dy: float, orbit_camera: OrbitCamera, viewport_height: int, speed_scale: float) -> None:
        item = self.item
        if item is None or not hasattr(item, "position"):
            return
        speed = self._world_per_pixel(orbit_camera, viewport_height) * speed_scale
        if self.axis:
            delta = self._axis_vector(self.axis) * ((dx - dy) * speed)
        else:
            delta = (-orbit_camera.right * dx + orbit_camera.view_up * dy) * speed
        getattr(item, "position")[:] = getattr(item, "position") + delta
        if isinstance(item, Camera):
            item.target[:] = item.target + delta

    def _scale(self, dx: float, dy: float, orbit_camera: OrbitCamera, viewport_height: int, speed_scale: float) -> None:
        item = self.item
        if item is None:
            return
        amount = np.exp((dx - dy) * 0.006 * speed_scale)
        if isinstance(item, Sphere):
            item.radius = max(0.02, item.radius * amount)
        elif isinstance(item, AreaLight):
            item.radius = max(0.02, item.radius * amount)
        elif isinstance(item, SceneObject):
            if self.axis:
                idx = {"X": 0, "Y": 1, "Z": 2}[self.axis]
                item.scale[idx] = max(0.02, item.scale[idx] * amount)
            else:
                item.scale[:] = np.maximum(0.02, item.scale * amount)

    def _rotate(self, dx: float, dy: float, speed_scale: float = 1.0) -> None:
        item = self.item
        amount = (dx - dy) * 0.006 * speed_scale
        axis = self.axis or "Y"
        if isinstance(item, SceneObject):
            idx = {"X": 0, "Y": 1, "Z": 2}[axis]
            item.rotation[idx] += amount * 30.0
        if isinstance(item, (Plane, AreaLight)):
            item.normal[:] = self._rotate_vector(item.normal, axis, amount)
        if isinstance(item, DirectionalLight):
            item.direction[:] = self._rotate_vector(item.direction, axis, amount)
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
