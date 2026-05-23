"""Upload RenderContext arrays to CUDA device."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda

from src.raytracer.render_context import RenderContext


def _f32(value) -> np.float32:
    return np.float32(value)


def _device_f32(array: np.ndarray):
    return cuda.to_device(np.ascontiguousarray(array, dtype=np.float32))


@dataclass
class DeviceRenderContext:
    width: int
    height: int
    samples: int
    max_bounces: int
    area_light_samples: int
    cam_ox: float
    cam_oy: float
    cam_oz: float
    cam_llx: float
    cam_lly: float
    cam_llz: float
    cam_hx: float
    cam_hy: float
    cam_hz: float
    cam_vx: float
    cam_vy: float
    cam_vz: float
    inv_width: float
    inv_height: float
    spheres: cuda.devicearray.DeviceNDArray
    planes: cuda.devicearray.DeviceNDArray
    bvh_nodes: cuda.devicearray.DeviceNDArray
    bvh_prims: cuda.devicearray.DeviceNDArray
    materials: cuda.devicearray.DeviceNDArray
    lights: cuda.devicearray.DeviceNDArray
    background: cuda.devicearray.DeviceNDArray


def to_device(ctx: RenderContext) -> DeviceRenderContext:
    cf = ctx.camera_frame
    return DeviceRenderContext(
        width=ctx.width,
        height=ctx.height,
        samples=ctx.samples,
        max_bounces=ctx.max_bounces,
        area_light_samples=ctx.area_light_samples,
        cam_ox=_f32(cf.origin[0]),
        cam_oy=_f32(cf.origin[1]),
        cam_oz=_f32(cf.origin[2]),
        cam_llx=_f32(cf.lower_left[0]),
        cam_lly=_f32(cf.lower_left[1]),
        cam_llz=_f32(cf.lower_left[2]),
        cam_hx=_f32(cf.horizontal[0]),
        cam_hy=_f32(cf.horizontal[1]),
        cam_hz=_f32(cf.horizontal[2]),
        cam_vx=_f32(cf.vertical[0]),
        cam_vy=_f32(cf.vertical[1]),
        cam_vz=_f32(cf.vertical[2]),
        inv_width=_f32(cf.inv_width),
        inv_height=_f32(cf.inv_height),
        spheres=_device_f32(ctx.spheres),
        planes=_device_f32(ctx.planes),
        bvh_nodes=_device_f32(ctx.bvh_nodes),
        bvh_prims=_device_f32(ctx.bvh_prims),
        materials=_device_f32(ctx.materials),
        lights=_device_f32(ctx.lights),
        background=_device_f32(ctx.background),
    )
