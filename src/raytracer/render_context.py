"""Immutable per-render state built once and shared across tile workers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.raytracer.camera_frame import CameraFrame, compute_camera_frame
from src.scene.scene import Scene


from src.scene.objects import Sphere, Plane
from src.scene.materials import DiffuseMaterial, SpecularMaterial, GlassMaterial
from src.scene.lights import AreaLight, PointLight

# Light row layout (12 floats):
#   type 0 point: [0, px, py, pz, cr, cg, cb, intensity, 0, 0, 0, 0]
#   type 1 disk:  [1, cx, cy, cz, nx, ny, nz, cr, cg, cb, intensity, radius]

@dataclass(frozen=True)
class RenderContext:
    width: int
    height: int
    samples: int
    max_bounces: int
    area_light_samples: int
    background: np.ndarray
    camera_frame: CameraFrame
    
    # Flat arrays for JIT
    spheres: np.ndarray  # (N, 5) -> [x, y, z, r, mat_idx]
    planes: np.ndarray   # (N, 7) -> [px, py, pz, nx, ny, nz, mat_idx]
    materials: np.ndarray # (N, 9) -> [type, r, g, b, roughness, ior, abs_r, abs_g, abs_b]
    lights: np.ndarray    # (N, 12) — see layout above


def build_render_context(scene: Scene) -> RenderContext:
    w = scene.render.width
    h = scene.render.height
    cam = scene.camera
    
    # Compile materials
    mat_list = []
    mat_map = {}
    
    def get_mat_idx(mat):
        idx = id(mat)
        if idx not in mat_map:
            mat_map[idx] = len(mat_list)
            mat_list.append(mat)
        return mat_map[idx]

    # Pre-scan objects to build material table
    for obj in scene.objects:
        if obj.visible:
            get_mat_idx(obj.material)
            
    # Also we don't have materials on lights, but just in case
    mat_arr = np.zeros((max(1, len(mat_list)), 9), dtype=np.float64)
    for i, mat in enumerate(mat_list):
        if isinstance(mat, DiffuseMaterial):
            mat_arr[i] = [0.0, mat.color[0], mat.color[1], mat.color[2], mat.roughness, 1.0, 1.0, 1.0, 1.0]
        elif isinstance(mat, SpecularMaterial):
            mat_arr[i] = [1.0, mat.color[0], mat.color[1], mat.color[2], mat.roughness, mat.ior, 1.0, 1.0, 1.0]
        elif isinstance(mat, GlassMaterial):
            mat_arr[i] = [
                2.0,
                mat.tint[0], mat.tint[1], mat.tint[2],
                mat.roughness,
                mat.ior,
                mat.absorption_color[0], mat.absorption_color[1], mat.absorption_color[2],
            ]

    # Compile objects
    spheres = []
    planes = []
    for obj in scene.objects:
        if not obj.visible:
            continue
        midx = float(get_mat_idx(obj.material))
        if isinstance(obj, Sphere):
            spheres.append([obj.position[0], obj.position[1], obj.position[2], obj.radius, midx])
        elif isinstance(obj, Plane):
            planes.append([obj.position[0], obj.position[1], obj.position[2], obj.normal[0], obj.normal[1], obj.normal[2], midx])

    spheres_arr = np.array(spheres, dtype=np.float64) if spheres else np.zeros((0, 5), dtype=np.float64)
    planes_arr = np.array(planes, dtype=np.float64) if planes else np.zeros((0, 7), dtype=np.float64)

    # Compile lights
    lights = []
    for light in scene.lights:
        if isinstance(light, PointLight):
            lights.append(
                [
                    0.0,
                    light.position[0], light.position[1], light.position[2],
                    light.color[0], light.color[1], light.color[2],
                    light.intensity,
                    0.0, 0.0, 0.0, 0.0,
                ]
            )
        elif isinstance(light, AreaLight):
            lights.append(
                [
                    1.0,
                    light.position[0], light.position[1], light.position[2],
                    light.normal[0], light.normal[1], light.normal[2],
                    light.color[0], light.color[1], light.color[2],
                    light.intensity,
                    light.radius,
                ]
            )
    lights_arr = np.array(lights, dtype=np.float64) if lights else np.zeros((0, 12), dtype=np.float64)

    return RenderContext(
        width=w,
        height=h,
        samples=max(1, scene.render.samples),
        max_bounces=scene.render.max_bounces,
        area_light_samples=max(1, scene.render.area_light_samples),
        background=np.asarray(scene.render.background_color, dtype=np.float64),
        camera_frame=compute_camera_frame(
            cam.position, cam.target, cam.up, cam.fov, w, h
        ),
        spheres=spheres_arr,
        planes=planes_arr,
        materials=mat_arr,
        lights=lights_arr,
    )
