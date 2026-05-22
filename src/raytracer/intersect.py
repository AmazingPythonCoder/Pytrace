from __future__ import annotations

import math

import numpy as np
from numba import njit

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
    t_min: float,
    t_max: float,
) -> bool:
    for i in range(spheres.shape[0]):
        t = _sphere_hit_t(
            ox, oy, oz, dx, dy, dz,
            spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3],
            t_min, t_max,
        )
        if t > 0.0:
            return True

    for i in range(planes.shape[0]):
        t = _plane_hit_t(
            ox, oy, oz, dx, dy, dz,
            planes[i, 0], planes[i, 1], planes[i, 2],
            planes[i, 3], planes[i, 4], planes[i, 5],
            t_min, t_max,
        )
        if t > 0.0:
            return True

    return False


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
    t_min: float = 1e-4,
    t_max: float = 1e9,
) -> tuple[float, float, float, float, float, float, float, bool, float]:
    closest_t = t_max
    best_hit = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, -1.0)

    for i in range(spheres.shape[0]):
        t = _sphere_hit_t(
            ox, oy, oz, dx, dy, dz,
            spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3],
            t_min, closest_t,
        )
        if t > 0.0:
            closest_t = t
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz

            outward_nx = (px - spheres[i, 0]) / spheres[i, 3]
            outward_ny = (py - spheres[i, 1]) / spheres[i, 3]
            outward_nz = (pz - spheres[i, 2]) / spheres[i, 3]

            ff = (dx * outward_nx + dy * outward_ny + dz * outward_nz) < 0.0
            nx = outward_nx if ff else -outward_nx
            ny = outward_ny if ff else -outward_ny
            nz = outward_nz if ff else -outward_nz

            best_hit = (t, px, py, pz, nx, ny, nz, ff, spheres[i, 4])

    for i in range(planes.shape[0]):
        t = _plane_hit_t(
            ox, oy, oz, dx, dy, dz,
            planes[i, 0], planes[i, 1], planes[i, 2],
            planes[i, 3], planes[i, 4], planes[i, 5],
            t_min, closest_t,
        )
        if t > 0.0:
            closest_t = t
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz

            outward_nx = planes[i, 3]
            outward_ny = planes[i, 4]
            outward_nz = planes[i, 5]

            ff = (dx * outward_nx + dy * outward_ny + dz * outward_nz) < 0.0
            nx = outward_nx if ff else -outward_nx
            ny = outward_ny if ff else -outward_ny
            nz = outward_nz if ff else -outward_nz

            best_hit = (t, px, py, pz, nx, ny, nz, ff, planes[i, 6])

    return best_hit
