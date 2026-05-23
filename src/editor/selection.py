from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.scene.camera import Camera
from src.scene.lights import DirectionalLight, Light
from src.scene.objects import Mesh, Plane, SceneObject, Sphere
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


def _rotation_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(euler_degrees, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


def _mesh_world_vertices(mesh: Mesh) -> np.ndarray:
    if mesh.vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return (mesh.vertices * mesh.scale) @ _rotation_matrix(mesh.rotation).T + mesh.position


def _hit_triangle(ray: PickRay, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float | None:
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(ray.direction, e2)
    a = float(np.dot(e1, h))
    if abs(a) < 1e-8:
        return None
    f = 1.0 / a
    s = ray.origin - v0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * float(np.dot(ray.direction, q))
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * float(np.dot(e2, q))
    return t if t > 1e-4 else None


def _hit_mesh(ray: PickRay, mesh: Mesh) -> float | None:
    vertices = _mesh_world_vertices(mesh)
    nearest: float | None = None
    for tri in mesh.triangles:
        t = _hit_triangle(ray, vertices[int(tri[0])], vertices[int(tri[1])], vertices[int(tri[2])])
        if t is not None and (nearest is None or t < nearest):
            nearest = t
    return nearest


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
    if isinstance(obj, Mesh):
        return _hit_mesh(ray, obj)
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
        radius = 0.34 if isinstance(light, DirectionalLight) else 0.24
        t = _hit_sphere(ray, light.position, radius)
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
