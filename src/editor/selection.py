from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.scene.camera import Camera
from src.scene.lights import Light
from src.scene.objects import Plane, SceneObject, Sphere
from src.scene.scene import Scene

from .layout import Rect
from .orbit_camera import OrbitCamera


@dataclass(frozen=True)
class PickRay:
    origin: np.ndarray
    direction: np.ndarray


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / length


def unproject_ray(x: int, y: int, viewport: Rect, camera: OrbitCamera) -> PickRay:
    ndc_x = ((x - viewport.x) / viewport.w) * 2.0 - 1.0
    ndc_y = 1.0 - ((y - viewport.y) / viewport.h) * 2.0
    aspect = viewport.w / viewport.h
    tan_half = math.tan(math.radians(camera.fov * 0.5))

    direction = (
        camera.forward
        + camera.right * (ndc_x * aspect * tan_half)
        + camera.view_up * (ndc_y * tan_half)
    )
    return PickRay(origin=camera.eye, direction=_normalize(direction))


def _hit_sphere(ray: PickRay, center: np.ndarray, radius: float) -> float | None:
    oc = ray.origin - center
    a = float(np.dot(ray.direction, ray.direction))
    half_b = float(np.dot(oc, ray.direction))
    c = float(np.dot(oc, oc) - radius * radius)
    discriminant = half_b * half_b - a * c
    if discriminant < 0.0:
        return None
    sqrt_d = math.sqrt(discriminant)
    for t in ((-half_b - sqrt_d) / a, (-half_b + sqrt_d) / a):
        if t > 1e-4:
            return t
    return None


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent = _normalize(np.cross(helper, n))
    bitangent = _normalize(np.cross(n, tangent))
    return tangent, bitangent


def _hit_plane_preview(ray: PickRay, plane: Plane) -> float | None:
    normal = _normalize(plane.normal)
    denom = float(np.dot(ray.direction, normal))
    if abs(denom) < 1e-6:
        return None
    t = float(np.dot(plane.position - ray.origin, normal) / denom)
    if t <= 1e-4:
        return None
    point = ray.origin + ray.direction * t
    tangent, bitangent = _plane_basis(normal)
    rel = point - plane.position
    half_size = max(1.0, float(max(plane.scale[0], plane.scale[2]))) * 4.0
    if abs(float(np.dot(rel, tangent))) <= half_size and abs(float(np.dot(rel, bitangent))) <= half_size:
        return t
    return None


def _object_hit(ray: PickRay, obj: SceneObject) -> float | None:
    if isinstance(obj, Sphere):
        radius = float(obj.radius * max(obj.scale[0], obj.scale[1], obj.scale[2]))
        return _hit_sphere(ray, obj.position, radius)
    if isinstance(obj, Plane):
        return _hit_plane_preview(ray, obj)
    return None


def pick(x: int, y: int, viewport: Rect, orbit_camera: OrbitCamera, scene: Scene) -> object | None:
    if not viewport.contains((x, y)):
        return None

    ray = unproject_ray(x, y, viewport, orbit_camera)
    nearest_item: object | None = None
    nearest_t = math.inf

    for obj in scene.objects:
        if not obj.visible:
            continue
        t = _object_hit(ray, obj)
        if t is not None and t < nearest_t:
            nearest_t = t
            nearest_item = obj

    for light in scene.lights:
        t = _hit_sphere(ray, light.position, 0.24)
        if t is not None and t < nearest_t:
            nearest_t = t
            nearest_item = light

    t = _hit_sphere(ray, scene.camera.position, 0.28)
    if t is not None and t < nearest_t:
        nearest_item = scene.camera

    return nearest_item


def display_name(item: object) -> str:
    if isinstance(item, SceneObject):
        return item.name
    if isinstance(item, Light):
        return getattr(item, "name", item.type.title())
    if isinstance(item, Camera):
        return "Camera"
    return "None"

