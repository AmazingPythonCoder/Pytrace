"""BVH acceleration: build-time bounds + Numba traversal over spheres and planes."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numba import njit

PRIM_SPHERE = 0.0
_SHADOW_TOL = 1e-6
PRIM_PLANE = 1.0

_PLANE_SLAB_EPS = 1e-3
_MAX_STACK = 64


def _expand_bounds(
    bmin: np.ndarray,
    bmax: np.ndarray,
    point: Sequence[float],
) -> None:
    bmin[0] = min(bmin[0], point[0])
    bmin[1] = min(bmin[1], point[1])
    bmin[2] = min(bmin[2], point[2])
    bmax[0] = max(bmax[0], point[0])
    bmax[1] = max(bmax[1], point[1])
    bmax[2] = max(bmax[2], point[2])


def compute_scene_bounds(
    spheres: np.ndarray,
    planes: np.ndarray,
    camera_pos: np.ndarray,
    padding: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Union AABB of spheres, plane anchor points, and camera; then pad."""
    bmin = np.array([math.inf, math.inf, math.inf], dtype=np.float64)
    bmax = np.array([-math.inf, -math.inf, -math.inf], dtype=np.float64)

    for i in range(spheres.shape[0]):
        cx, cy, cz, r = spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3]
        _expand_bounds(bmin, bmax, (cx - r, cy - r, cz - r))
        _expand_bounds(bmin, bmax, (cx + r, cy + r, cz + r))

    for i in range(planes.shape[0]):
        _expand_bounds(bmin, bmax, (planes[i, 0], planes[i, 1], planes[i, 2]))

    _expand_bounds(bmin, bmax, (camera_pos[0], camera_pos[1], camera_pos[2]))

    if not np.isfinite(bmin[0]):
        bmin[:] = (-10.0, -1.0, -10.0)
        bmax[:] = (10.0, 12.0, 10.0)
    else:
        extent = bmax - bmin
        pad = padding * np.maximum(extent, 1.0)
        bmin -= pad
        bmax += pad

    return bmin, bmax


def sphere_aabb(spheres: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz, r = spheres[index, 0], spheres[index, 1], spheres[index, 2], spheres[index, 3]
    bmin = np.array([cx - r, cy - r, cz - r], dtype=np.float64)
    bmax = np.array([cx + r, cy + r, cz + r], dtype=np.float64)
    centroid = np.array([cx, cy, cz], dtype=np.float64)
    return bmin, bmax, centroid


def plane_aabb(
    scene_bmin: np.ndarray,
    scene_bmax: np.ndarray,
    px: float,
    py: float,
    pz: float,
    nx: float,
    ny: float,
    nz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative AABB for the portion of an infinite plane inside scene bounds."""
    corners = [
        (scene_bmin[0], scene_bmin[1], scene_bmin[2]),
        (scene_bmax[0], scene_bmin[1], scene_bmin[2]),
        (scene_bmin[0], scene_bmax[1], scene_bmin[2]),
        (scene_bmax[0], scene_bmax[1], scene_bmin[2]),
        (scene_bmin[0], scene_bmin[1], scene_bmax[2]),
        (scene_bmax[0], scene_bmin[1], scene_bmax[2]),
        (scene_bmin[0], scene_bmax[1], scene_bmax[2]),
        (scene_bmax[0], scene_bmax[1], scene_bmax[2]),
    ]

    bmin = np.array([math.inf, math.inf, math.inf], dtype=np.float64)
    bmax = np.array([-math.inf, -math.inf, -math.inf], dtype=np.float64)

    for cx, cy, cz in corners:
        d = (cx - px) * nx + (cy - py) * ny + (cz - pz) * nz
        qx = cx - d * nx
        qy = cy - d * ny
        qz = cz - d * nz
        _expand_bounds(bmin, bmax, (qx, qy, qz))

    _expand_bounds(bmin, bmax, (px, py, pz))

    bmin[0] -= abs(nx) * _PLANE_SLAB_EPS
    bmin[1] -= abs(ny) * _PLANE_SLAB_EPS
    bmin[2] -= abs(nz) * _PLANE_SLAB_EPS
    bmax[0] += abs(nx) * _PLANE_SLAB_EPS
    bmax[1] += abs(ny) * _PLANE_SLAB_EPS
    bmax[2] += abs(nz) * _PLANE_SLAB_EPS

    centroid = 0.5 * (bmin + bmax)
    return bmin, bmax, centroid


def _merge_aabb(bmin_a, bmax_a, bmin_b, bmax_b):
    bmin = np.minimum(bmin_a, bmin_b)
    bmax = np.maximum(bmax_a, bmax_b)
    return bmin, bmax


def _collect_primitives(
    spheres: np.ndarray,
    planes: np.ndarray,
    scene_bmin: np.ndarray,
    scene_bmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """bvh_prims (P,2), prim_bmin (P,3), prim_bmax (P,3), prim_centroid (P,3)."""
    prims: list[list[float]] = []
    bmins: list[np.ndarray] = []
    bmaxs: list[np.ndarray] = []
    cents: list[np.ndarray] = []

    for i in range(spheres.shape[0]):
        bmin, bmax, cent = sphere_aabb(spheres, i)
        prims.append([PRIM_SPHERE, float(i)])
        bmins.append(bmin)
        bmaxs.append(bmax)
        cents.append(cent)

    for i in range(planes.shape[0]):
        bmin, bmax, cent = plane_aabb(
            scene_bmin,
            scene_bmax,
            planes[i, 0],
            planes[i, 1],
            planes[i, 2],
            planes[i, 3],
            planes[i, 4],
            planes[i, 5],
        )
        prims.append([PRIM_PLANE, float(i)])
        bmins.append(bmin)
        bmaxs.append(bmax)
        cents.append(cent)

    if not prims:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    return (
        np.array(prims, dtype=np.float64),
        np.stack(bmins),
        np.stack(bmaxs),
        np.stack(cents),
    )


def build_bvh(
    spheres: np.ndarray,
    planes: np.ndarray,
    camera_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat BVH. Returns (bvh_nodes, bvh_prims)."""
    scene_bmin, scene_bmax = compute_scene_bounds(spheres, planes, camera_pos)
    bvh_prims, prim_bmin, prim_bmax, prim_centroid = _collect_primitives(
        spheres, planes, scene_bmin, scene_bmax
    )

    n = bvh_prims.shape[0]
    if n == 0:
        return np.zeros((0, 8), dtype=np.float64), bvh_prims

    nodes: list[list[float]] = []
    indices = list(range(n))

    def build_recursive(idxs: list[int]) -> int:
        node_idx = len(nodes)
        nodes.append([0.0] * 8)

        if len(idxs) == 1:
            i = idxs[0]
            nodes[node_idx][0:3] = prim_bmin[i].tolist()
            nodes[node_idx][3:6] = prim_bmax[i].tolist()
            nodes[node_idx][6] = -(i + 1)
            nodes[node_idx][7] = -1.0
            return node_idx

        merged_bmin = prim_bmin[idxs[0]].copy()
        merged_bmax = prim_bmax[idxs[0]].copy()
        for i in idxs[1:]:
            merged_bmin, merged_bmax = _merge_aabb(merged_bmin, merged_bmax, prim_bmin[i], prim_bmax[i])

        extent = merged_bmax - merged_bmin
        axis = int(np.argmax(extent))
        idxs.sort(key=lambda i: prim_centroid[i, axis])

        mid = len(idxs) // 2
        left_idxs = idxs[:mid]
        right_idxs = idxs[mid:]
        if not left_idxs or not right_idxs:
            mid = 1
            left_idxs = idxs[:mid]
            right_idxs = idxs[mid:]

        left_child = build_recursive(left_idxs)
        right_child = build_recursive(right_idxs)

        nodes[node_idx][0:3] = merged_bmin.tolist()
        nodes[node_idx][3:6] = merged_bmax.tolist()
        nodes[node_idx][6] = float(left_child)
        nodes[node_idx][7] = float(right_child)
        return node_idx

    build_recursive(indices)
    return np.array(nodes, dtype=np.float64), bvh_prims


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
def _aabb_hit(
    bmin_x: float,
    bmin_y: float,
    bmin_z: float,
    bmax_x: float,
    bmax_y: float,
    bmax_z: float,
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    t_min: float,
    t_max: float,
) -> bool:
    for axis in range(3):
        if axis == 0:
            bmn, bmx, orig, direc = bmin_x, bmax_x, ox, dx
        elif axis == 1:
            bmn, bmx, orig, direc = bmin_y, bmax_y, oy, dy
        else:
            bmn, bmx, orig, direc = bmin_z, bmax_z, oz, dz

        if abs(direc) < 1e-12:
            if orig < bmn or orig > bmx:
                return False
        else:
            inv_d = 1.0 / direc
            t0 = (bmn - orig) * inv_d
            t1 = (bmx - orig) * inv_d
            if inv_d < 0.0:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max <= t_min:
                return False
    return True


@njit(cache=True)
def _leaf_hit_t(
    prim_idx: int,
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    spheres: np.ndarray,
    planes: np.ndarray,
    bvh_prims: np.ndarray,
    t_min: float,
    t_max: float,
) -> float:
    ptype = bvh_prims[prim_idx, 0]
    pindex = int(bvh_prims[prim_idx, 1])
    if ptype < 0.5:
        return _sphere_hit_t(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            spheres[pindex, 0],
            spheres[pindex, 1],
            spheres[pindex, 2],
            spheres[pindex, 3],
            t_min,
            t_max,
        )
    return _plane_hit_t(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        planes[pindex, 0],
        planes[pindex, 1],
        planes[pindex, 2],
        planes[pindex, 3],
        planes[pindex, 4],
        planes[pindex, 5],
        t_min,
        t_max,
    )


@njit(cache=True)
def _leaf_closest_hit(
    prim_idx: int,
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    spheres: np.ndarray,
    planes: np.ndarray,
    bvh_prims: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[float, float, float, float, float, float, float, bool, float]:
    ptype = bvh_prims[prim_idx, 0]
    pindex = int(bvh_prims[prim_idx, 1])

    if ptype < 0.5:
        t = _sphere_hit_t(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            spheres[pindex, 0],
            spheres[pindex, 1],
            spheres[pindex, 2],
            spheres[pindex, 3],
            t_min,
            t_max,
        )
        if t <= 0.0:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, -1.0
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        outward_nx = (px - spheres[pindex, 0]) / spheres[pindex, 3]
        outward_ny = (py - spheres[pindex, 1]) / spheres[pindex, 3]
        outward_nz = (pz - spheres[pindex, 2]) / spheres[pindex, 3]
        ff = (dx * outward_nx + dy * outward_ny + dz * outward_nz) < 0.0
        nx = outward_nx if ff else -outward_nx
        ny = outward_ny if ff else -outward_ny
        nz = outward_nz if ff else -outward_nz
        return t, px, py, pz, nx, ny, nz, ff, spheres[pindex, 4]

    t = _plane_hit_t(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        planes[pindex, 0],
        planes[pindex, 1],
        planes[pindex, 2],
        planes[pindex, 3],
        planes[pindex, 4],
        planes[pindex, 5],
        t_min,
        t_max,
    )
    if t <= 0.0:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, -1.0
    px = ox + t * dx
    py = oy + t * dy
    pz = oz + t * dz
    outward_nx = planes[pindex, 3]
    outward_ny = planes[pindex, 4]
    outward_nz = planes[pindex, 5]
    ff = (dx * outward_nx + dy * outward_ny + dz * outward_nz) < 0.0
    nx = outward_nx if ff else -outward_nx
    ny = outward_ny if ff else -outward_ny
    nz = outward_nz if ff else -outward_nz
    return t, px, py, pz, nx, ny, nz, ff, planes[pindex, 6]


@njit(cache=True)
def bvh_any_hit(
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
    if bvh_nodes.shape[0] == 0:
        return False

    stack = np.empty(_MAX_STACK, dtype=np.int64)
    stack_ptr = 0
    stack[0] = 0
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]

        if not _aabb_hit(
            bvh_nodes[node, 0],
            bvh_nodes[node, 1],
            bvh_nodes[node, 2],
            bvh_nodes[node, 3],
            bvh_nodes[node, 4],
            bvh_nodes[node, 5],
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            t_min,
            t_max,
        ):
            continue

        left = bvh_nodes[node, 6]
        if left < 0.0:
            prim_idx = int(-left - 1.0)
            t = _leaf_hit_t(
                prim_idx,
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                spheres,
                planes,
                bvh_prims,
                t_min,
                t_max,
            )
            if t > 0.0:
                return True
            continue

        right = int(bvh_nodes[node, 7])
        left_i = int(left)
        if stack_ptr + 2 <= _MAX_STACK:
            stack[stack_ptr] = right
            stack_ptr += 1
            stack[stack_ptr] = left_i
            stack_ptr += 1

    return False


@njit(cache=True)
def bvh_find_closest_hit(
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
) -> tuple[float, float, float, float, float, float, float, bool, float]:
    if bvh_nodes.shape[0] == 0:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, -1.0

    closest_t = t_max
    best_hit = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, -1.0)

    stack = np.empty(_MAX_STACK, dtype=np.int64)
    stack_ptr = 0
    stack[0] = 0
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]

        if not _aabb_hit(
            bvh_nodes[node, 0],
            bvh_nodes[node, 1],
            bvh_nodes[node, 2],
            bvh_nodes[node, 3],
            bvh_nodes[node, 4],
            bvh_nodes[node, 5],
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            t_min,
            closest_t,
        ):
            continue

        left = bvh_nodes[node, 6]
        if left < 0.0:
            prim_idx = int(-left - 1.0)
            hit = _leaf_closest_hit(
                prim_idx,
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                spheres,
                planes,
                bvh_prims,
                t_min,
                closest_t,
            )
            t = hit[0]
            if t > 0.0:
                closest_t = t
                best_hit = hit
            continue

        left_i = int(left)
        right_i = int(bvh_nodes[node, 7])
        if stack_ptr + 2 <= _MAX_STACK:
            stack[stack_ptr] = right_i
            stack_ptr += 1
            stack[stack_ptr] = left_i
            stack_ptr += 1

    return best_hit
