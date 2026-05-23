from __future__ import annotations

import math

import numpy as np
from numba import njit

from src.raytracer.bvh import bvh_any_hit, bvh_find_closest_hit

_SHADOW_TOL = 1e-6

# Hit tuple: (t, px, py, pz, nx, ny, nz, front_face, mat_idx)
# If t < 0, there was no hit.
def make_empty_hit() -> tuple[float, float, float, float, float, float, float, bool, float]:
    return (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, -1.0)


@njit(cache=True)
def _sphere_hit_t(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    t_min: float,
    t_max: float,
) -> float:
    oc_x = ox - center_x
    oc_y = oy - center_y
    oc_z = oz - center_z

    a = dx * dx + dy * dy + dz * dz
    half_b = oc_x * dx + oc_y * dy + oc_z * dz
    c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius

    discriminant = half_b * half_b - a * c
    if discriminant < 0.0:
        return -1.0

    sqrt_d = math.sqrt(discriminant)
    t = (-half_b - sqrt_d) / a
    if t <= t_min or t >= t_max:
        t = (-half_b + sqrt_d) / a
        if t <= t_min or t >= t_max:
            return -1.0
    return t


@njit(cache=True)
def _plane_hit_t(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    px: float,
    py: float,
    pz: float,
    nx: float,
    ny: float,
    nz: float,
    t_min: float,
    t_max: float,
) -> float:
    denom = dx * nx + dy * ny + dz * nz
    if abs(denom) < _SHADOW_TOL:
        return -1.0
    t = ((px - ox) * nx + (py - oy) * ny + (pz - oz) * nz) / denom
    if t <= t_min or t >= t_max:
        return -1.0
    return t


@njit(cache=True)
def any_hit(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    spheres: np.ndarray,
    planes: np.ndarray,
    bvh_nodes: np.ndarray,
    bvh_prims: np.ndarray,
    t_min: float,
    t_max: float,
) -> bool:
    return bvh_any_hit(
        ox, oy, oz, dx, dy, dz,
        spheres, planes, bvh_nodes, bvh_prims,
        t_min, t_max,
    )


@njit(cache=True)
def find_closest_hit(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    spheres: np.ndarray,
    planes: np.ndarray,
    bvh_nodes: np.ndarray,
    bvh_prims: np.ndarray,
    t_min: float = 1e-4,
    t_max: float = 1e9,
) -> tuple[float, float, float, float, float, float, float, bool, float]:
    return bvh_find_closest_hit(
        ox, oy, oz, dx, dy, dz,
        spheres, planes, bvh_nodes, bvh_prims,
        t_min, t_max,
    )
