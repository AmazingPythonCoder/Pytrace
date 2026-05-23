"""BVH traversal and primitive intersection on CUDA."""

from __future__ import annotations

import math

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda, float32, int64

_F0 = float32(0.0)
_F05 = float32(0.5)
_F1 = float32(1.0)
_FM1 = float32(-1.0)
_SHADOW_TOL = float32(1e-6)
_PARALLEL_TOL = float32(1e-12)
_MAX_STACK = 64


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _sphere_hit_t(
    ox, oy, oz, dx, dy, dz,
    center_x, center_y, center_z, radius,
    t_min, t_max,
):
    oc_x = ox - center_x
    oc_y = oy - center_y
    oc_z = oz - center_z
    a = dx * dx + dy * dy + dz * dz
    half_b = oc_x * dx + oc_y * dy + oc_z * dz
    c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius
    discriminant = half_b * half_b - a * c
    if discriminant < _F0:
        return _FM1
    sqrt_d = math.sqrt(discriminant)
    t = (-half_b - sqrt_d) / a
    if t <= t_min or t >= t_max:
        t = (-half_b + sqrt_d) / a
        if t <= t_min or t >= t_max:
            return _FM1
    return t


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _plane_hit_t(
    ox, oy, oz, dx, dy, dz,
    px, py, pz, nx, ny, nz,
    t_min, t_max,
):
    denom = dx * nx + dy * ny + dz * nz
    if abs(denom) < _SHADOW_TOL:
        return _FM1
    t = ((px - ox) * nx + (py - oy) * ny + (pz - oz) * nz) / denom
    if t <= t_min or t >= t_max:
        return _FM1
    return t


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _aabb_hit(
    bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z,
    ox, oy, oz, dx, dy, dz,
    t_min, t_max,
):
    for axis in range(3):
        if axis == 0:
            bmn, bmx, orig, direc = bmin_x, bmax_x, ox, dx
        elif axis == 1:
            bmn, bmx, orig, direc = bmin_y, bmax_y, oy, dy
        else:
            bmn, bmx, orig, direc = bmin_z, bmax_z, oz, dz
        if abs(direc) < _PARALLEL_TOL:
            if orig < bmn or orig > bmx:
                return False
        else:
            inv_d = _F1 / direc
            t0 = (bmn - orig) * inv_d
            t1 = (bmx - orig) * inv_d
            if inv_d < _F0:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max <= t_min:
                return False
    return True


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _leaf_hit_t(
    prim_idx, ox, oy, oz, dx, dy, dz,
    spheres, planes, bvh_prims,
    t_min, t_max,
):
    ptype = bvh_prims[prim_idx, 0]
    pindex = int(bvh_prims[prim_idx, 1])
    if ptype < _F05:
        return _sphere_hit_t(
            ox, oy, oz, dx, dy, dz,
            spheres[pindex, 0], spheres[pindex, 1], spheres[pindex, 2], spheres[pindex, 3],
            t_min, t_max,
        )
    return _plane_hit_t(
        ox, oy, oz, dx, dy, dz,
        planes[pindex, 0], planes[pindex, 1], planes[pindex, 2],
        planes[pindex, 3], planes[pindex, 4], planes[pindex, 5],
        t_min, t_max,
    )


@cuda.jit(device=True, fastmath=True, cache=True)
def _leaf_closest_hit(
    prim_idx, ox, oy, oz, dx, dy, dz,
    spheres, planes, bvh_prims,
    t_min, t_max,
):
    ptype = bvh_prims[prim_idx, 0]
    pindex = int(bvh_prims[prim_idx, 1])
    if ptype < _F05:
        t = _sphere_hit_t(
            ox, oy, oz, dx, dy, dz,
            spheres[pindex, 0], spheres[pindex, 1], spheres[pindex, 2], spheres[pindex, 3],
            t_min, t_max,
        )
        if t <= _F0:
            return _FM1, _F0, _F0, _F0, _F0, _F0, _F0, 0, _FM1
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        outward_nx = (px - spheres[pindex, 0]) / spheres[pindex, 3]
        outward_ny = (py - spheres[pindex, 1]) / spheres[pindex, 3]
        outward_nz = (pz - spheres[pindex, 2]) / spheres[pindex, 3]
        ff = 1 if (dx * outward_nx + dy * outward_ny + dz * outward_nz) < _F0 else 0
        nx = outward_nx if ff else -outward_nx
        ny = outward_ny if ff else -outward_ny
        nz = outward_nz if ff else -outward_nz
        return t, px, py, pz, nx, ny, nz, ff, spheres[pindex, 4]
    t = _plane_hit_t(
        ox, oy, oz, dx, dy, dz,
        planes[pindex, 0], planes[pindex, 1], planes[pindex, 2],
        planes[pindex, 3], planes[pindex, 4], planes[pindex, 5],
        t_min, t_max,
    )
    if t <= _F0:
        return _FM1, _F0, _F0, _F0, _F0, _F0, _F0, 0, _FM1
    px = ox + t * dx
    py = oy + t * dy
    pz = oz + t * dz
    outward_nx = planes[pindex, 3]
    outward_ny = planes[pindex, 4]
    outward_nz = planes[pindex, 5]
    ff = 1 if (dx * outward_nx + dy * outward_ny + dz * outward_nz) < _F0 else 0
    nx = outward_nx if ff else -outward_nx
    ny = outward_ny if ff else -outward_ny
    nz = outward_nz if ff else -outward_nz
    return t, px, py, pz, nx, ny, nz, ff, planes[pindex, 6]


@cuda.jit(device=True, fastmath=True, cache=True)
def bvh_any_hit(
    ox, oy, oz, dx, dy, dz,
    spheres, planes, bvh_nodes, bvh_prims,
    t_min, t_max,
):
    n_nodes = bvh_nodes.shape[0]
    if n_nodes == 0:
        return False
    stack = cuda.local.array(_MAX_STACK, dtype=int64)
    stack_ptr = 1
    stack[0] = 0
    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]
        if not _aabb_hit(
            bvh_nodes[node, 0], bvh_nodes[node, 1], bvh_nodes[node, 2],
            bvh_nodes[node, 3], bvh_nodes[node, 4], bvh_nodes[node, 5],
            ox, oy, oz, dx, dy, dz, t_min, t_max,
        ):
            continue
        left = bvh_nodes[node, 6]
        if left < _F0:
            prim_idx = int(-left - _F1)
            t = _leaf_hit_t(
                prim_idx, ox, oy, oz, dx, dy, dz,
                spheres, planes, bvh_prims, t_min, t_max,
            )
            if t > _F0:
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


@cuda.jit(device=True, fastmath=True, cache=True)
def bvh_find_closest_hit(
    ox, oy, oz, dx, dy, dz,
    spheres, planes, bvh_nodes, bvh_prims,
    t_min, t_max,
):
    n_nodes = bvh_nodes.shape[0]
    if n_nodes == 0:
        return _FM1, _F0, _F0, _F0, _F0, _F0, _F0, 0, _FM1
    closest_t = t_max
    best_t = _FM1
    best_px, best_py, best_pz = _F0, _F0, _F0
    best_nx, best_ny, best_nz = _F0, _F0, _F0
    best_ff = 0
    best_mat = _FM1
    stack = cuda.local.array(_MAX_STACK, dtype=int64)
    stack_ptr = 1
    stack[0] = 0
    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]
        if not _aabb_hit(
            bvh_nodes[node, 0], bvh_nodes[node, 1], bvh_nodes[node, 2],
            bvh_nodes[node, 3], bvh_nodes[node, 4], bvh_nodes[node, 5],
            ox, oy, oz, dx, dy, dz, t_min, closest_t,
        ):
            continue
        left = bvh_nodes[node, 6]
        if left < _F0:
            prim_idx = int(-left - _F1)
            t, px, py, pz, nx, ny, nz, ff, mat_idx = _leaf_closest_hit(
                prim_idx, ox, oy, oz, dx, dy, dz,
                spheres, planes, bvh_prims, t_min, closest_t,
            )
            if t > _F0:
                closest_t = t
                best_t, best_px, best_py, best_pz = t, px, py, pz
                best_nx, best_ny, best_nz = nx, ny, nz
                best_ff = ff
                best_mat = mat_idx
            continue
        left_i = int(left)
        right_i = int(bvh_nodes[node, 7])
        if stack_ptr + 2 <= _MAX_STACK:
            stack[stack_ptr] = right_i
            stack_ptr += 1
            stack[stack_ptr] = left_i
            stack_ptr += 1
    return best_t, best_px, best_py, best_pz, best_nx, best_ny, best_nz, best_ff, best_mat

