"""Immutable per-render state built once and shared across tile workers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.raytracer.bvh import build_bvh
from src.raytracer.camera_frame import CameraFrame, compute_camera_frame
from src.raytracer.environment import load_environment
from src.scene.lights import AreaLight, DirectionalLight, PointLight
from src.scene.materials import DiffuseMaterial, EmissiveMaterial, GlassMaterial, SpecularMaterial
from src.scene.objects import Mesh, Plane, Sphere
from src.scene.scene import Scene

# Light row layout (12 floats):
#   type 0 point: [0, px, py, pz, cr, cg, cb, intensity, 0, 0, 0, 0]
#   type 1 disk: [1, cx, cy, cz, nx, ny, nz, cr, cg, cb, intensity, radius]
#   type 2 directional: [2, dx, dy, dz, cr, cg, cb, intensity, 0, 0, 0, 0]


@dataclass(frozen=True)
class RenderContext:
    width: int
    height: int
    samples: int
    max_bounces: int
    area_light_samples: int
    render_mode: int
    background_mode: int
    background: np.ndarray
    environment: np.ndarray
    camera_frame: CameraFrame

    # Flat arrays for JIT.
    spheres: np.ndarray  # (N, 5) -> [x, y, z, r, mat_idx]
    planes: np.ndarray  # (N, 7) -> [px, py, pz, nx, ny, nz, mat_idx]
    triangles: np.ndarray  # (N, 13) -> v0, v1, v2, normal, mat_idx
    materials: np.ndarray  # (N, 9) -> [type, r, g, b, roughness/strength, ior, abs_r, abs_g, abs_b]
    lights: np.ndarray  # (N, 12), see layout above
    bvh_nodes: np.ndarray  # (num_nodes, 8) bbox + child indices
    bvh_prims: np.ndarray  # (num_prims, 2) type, index
    gpu_supported: bool
    gpu_fallback_reason: str


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
    scaled = mesh.vertices * np.asarray(mesh.scale, dtype=np.float64)
    rotated = scaled @ _rotation_matrix(mesh.rotation).T
    return rotated + np.asarray(mesh.position, dtype=np.float64)


def _mode_id(value: str, mapping: dict[str, int], default: int) -> int:
    return mapping.get(str(value).strip().lower(), default)


def _normal_matrix(scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    safe_scale = np.asarray(scale, dtype=np.float64).copy()
    safe_scale[np.abs(safe_scale) < 1e-12] = 1.0
    return _rotation_matrix(rotation) @ np.diag(1.0 / safe_scale)


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / length


def _mesh_triangle_normal(mesh: Mesh, tri_index: int, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    normal = np.cross(v1 - v0, v2 - v0)
    if mesh.normals is not None and mesh.normals.size > 0:
        local_normal: np.ndarray | None = None
        tri = mesh.triangles[tri_index]
        if mesh.normals.shape[0] == mesh.vertices.shape[0]:
            local_normal = mesh.normals[[int(tri[0]), int(tri[1]), int(tri[2])]].mean(axis=0)
        elif mesh.normals.shape[0] == mesh.triangles.shape[0]:
            local_normal = mesh.normals[tri_index]
        if local_normal is not None:
            normal = _normal_matrix(mesh.scale, mesh.rotation) @ local_normal
    return _normalize(normal)


def build_render_context(scene: Scene) -> RenderContext:
    w = scene.render.width
    h = scene.render.height
    cam = scene.camera

    mat_list = []
    mat_map: dict[int, int] = {}

    def get_mat_idx(mat):
        idx = id(mat)
        if idx not in mat_map:
            mat_map[idx] = len(mat_list)
            mat_list.append(mat)
        return mat_map[idx]

    for obj in scene.objects:
        if obj.visible:
            get_mat_idx(obj.material)

    mat_arr = np.zeros((max(1, len(mat_list)), 9), dtype=np.float64)
    for i, mat in enumerate(mat_list):
        if isinstance(mat, DiffuseMaterial):
            mat_arr[i] = [0.0, mat.color[0], mat.color[1], mat.color[2], mat.roughness, 1.0, 1.0, 1.0, 1.0]
        elif isinstance(mat, SpecularMaterial):
            mat_arr[i] = [1.0, mat.color[0], mat.color[1], mat.color[2], mat.roughness, mat.ior, 1.0, 1.0, 1.0]
        elif isinstance(mat, GlassMaterial):
            mat_arr[i] = [
                2.0,
                mat.tint[0],
                mat.tint[1],
                mat.tint[2],
                mat.roughness,
                mat.ior,
                mat.absorption_color[0],
                mat.absorption_color[1],
                mat.absorption_color[2],
            ]
        elif isinstance(mat, EmissiveMaterial):
            mat_arr[i] = [3.0, mat.color[0], mat.color[1], mat.color[2], mat.strength, 1.0, 1.0, 1.0, 1.0]

    spheres = []
    planes = []
    triangles = []
    for obj in scene.objects:
        if not obj.visible:
            continue
        midx = float(get_mat_idx(obj.material))
        if isinstance(obj, Sphere):
            radius = float(obj.radius * max(obj.scale[0], obj.scale[1], obj.scale[2]))
            spheres.append([obj.position[0], obj.position[1], obj.position[2], radius, midx])
        elif isinstance(obj, Plane):
            planes.append(
                [
                    obj.position[0],
                    obj.position[1],
                    obj.position[2],
                    obj.normal[0],
                    obj.normal[1],
                    obj.normal[2],
                    midx,
                ]
            )
        elif isinstance(obj, Mesh):
            world_vertices = _mesh_world_vertices(obj)
            for tri_index, tri in enumerate(obj.triangles):
                v0 = world_vertices[int(tri[0])]
                v1 = world_vertices[int(tri[1])]
                v2 = world_vertices[int(tri[2])]
                normal = _mesh_triangle_normal(obj, tri_index, v0, v1, v2)
                if float(np.linalg.norm(normal)) <= 1e-12:
                    continue
                triangles.append(
                    [
                        v0[0],
                        v0[1],
                        v0[2],
                        v1[0],
                        v1[1],
                        v1[2],
                        v2[0],
                        v2[1],
                        v2[2],
                        normal[0],
                        normal[1],
                        normal[2],
                        midx,
                    ]
                )

    spheres_arr = np.array(spheres, dtype=np.float64) if spheres else np.zeros((0, 5), dtype=np.float64)
    planes_arr = np.array(planes, dtype=np.float64) if planes else np.zeros((0, 7), dtype=np.float64)
    triangles_arr = np.array(triangles, dtype=np.float64) if triangles else np.zeros((0, 13), dtype=np.float64)

    lights = []
    for light in scene.lights:
        if isinstance(light, PointLight):
            lights.append(
                [
                    0.0,
                    light.position[0],
                    light.position[1],
                    light.position[2],
                    light.color[0],
                    light.color[1],
                    light.color[2],
                    light.intensity,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        elif isinstance(light, AreaLight):
            lights.append(
                [
                    1.0,
                    light.position[0],
                    light.position[1],
                    light.position[2],
                    light.normal[0],
                    light.normal[1],
                    light.normal[2],
                    light.color[0],
                    light.color[1],
                    light.color[2],
                    light.intensity,
                    light.radius,
                ]
            )
        elif isinstance(light, DirectionalLight):
            lights.append(
                [
                    2.0,
                    light.direction[0],
                    light.direction[1],
                    light.direction[2],
                    light.color[0],
                    light.color[1],
                    light.color[2],
                    light.intensity,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
    lights_arr = np.array(lights, dtype=np.float64) if lights else np.zeros((0, 12), dtype=np.float64)

    env = load_environment(scene.render.environment_path)
    render_mode = _mode_id(scene.render.render_mode, {"direct": 0, "path": 1, "path_tracing": 1}, 0)
    background_mode = _mode_id(scene.render.background_mode, {"solid": 0, "gradient": 1, "environment": 2}, 1)
    bvh_nodes, bvh_prims = build_bvh(spheres_arr, planes_arr, triangles_arr, cam.position)
    return RenderContext(
        width=w,
        height=h,
        samples=max(1, scene.render.samples),
        max_bounces=scene.render.max_bounces,
        area_light_samples=max(1, scene.render.area_light_samples),
        render_mode=render_mode,
        background_mode=background_mode,
        background=np.asarray(scene.render.background_color, dtype=np.float64),
        environment=env,
        camera_frame=compute_camera_frame(
            cam.position,
            cam.target,
            cam.up,
            cam.fov,
            w,
            h,
            cam.aperture,
            cam.focus_distance,
        ),
        spheres=spheres_arr,
        planes=planes_arr,
        triangles=triangles_arr,
        materials=mat_arr,
        lights=lights_arr,
        bvh_nodes=bvh_nodes,
        bvh_prims=bvh_prims,
        gpu_supported=True,
        gpu_fallback_reason="",
    )
