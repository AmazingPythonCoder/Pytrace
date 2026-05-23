"""Full path tracing on CUDA (parity with CPU shading.py)."""

from __future__ import annotations

import math

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda, float32

from src.raytracer.gpu.cuda_bvh import direct_any_hit, direct_find_closest_hit
from src.raytracer.gpu.cuda_rng import rand_float, rand_in_unit_sphere, rng_seed

_F0 = float32(0.0)
_F001 = float32(0.001)
_F01 = float32(0.01)
_F05 = float32(0.5)
_F1 = float32(1.0)
_F2 = float32(2.0)
_FM1 = float32(-1.0)
_EPS8 = float32(1e-8)
_EPS12 = float32(1e-12)
_RAY_EPS = float32(1e-4)
_TMAX = float32(1e9)


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _reflect(ix, iy, iz, nx, ny, nz):
    dot_in = ix * nx + iy * ny + iz * nz
    return ix - 2.0 * dot_in * nx, iy - 2.0 * dot_in * ny, iz - 2.0 * dot_in * nz


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _schlick(cos_theta, f0_r, f0_g, f0_b):
    one_minus_cos = _F1 - cos_theta
    o5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos
    fr = f0_r + (_F1 - f0_r) * o5
    fg = f0_g + (_F1 - f0_g) * o5
    fb = f0_b + (_F1 - f0_b) * o5
    return fr, fg, fb


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _refract(ux, uy, uz, nx, ny, nz, ior_ratio):
    cos_theta = min(_F1, max(_FM1, -(ux * nx + uy * ny + uz * nz)))
    r_out_perp_x = ior_ratio * (ux + cos_theta * nx)
    r_out_perp_y = ior_ratio * (uy + cos_theta * ny)
    r_out_perp_z = ior_ratio * (uz + cos_theta * nz)
    r_out_perp_len_sq = r_out_perp_x**2 + r_out_perp_y**2 + r_out_perp_z**2
    if r_out_perp_len_sq >= _F1:
        return _F0, _F0, _F0, 0
    r_out_parallel_mag = -math.sqrt(abs(_F1 - r_out_perp_len_sq))
    return (
        r_out_perp_x + r_out_parallel_mag * nx,
        r_out_perp_y + r_out_parallel_mag * ny,
        r_out_perp_z + r_out_parallel_mag * nz,
        1,
    )


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def _disk_basis(nx, ny, nz):
    if abs(nz) < abs(nx):
        ax, ay, az = _F1, _F0, _F0
    else:
        ax, ay, az = _F0, _F0, _F1
    tx = ay * nz - az * ny
    ty = az * nx - ax * nz
    tz = ax * ny - ay * nx
    tlen = math.sqrt(tx * tx + ty * ty + tz * tz)
    tx /= tlen
    ty /= tlen
    tz /= tlen
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx
    return tx, ty, tz, bx, by, bz


@cuda.jit(device=True, fastmath=True, cache=True)
def _random_disk_point(state, cx, cy, cz, nx, ny, nz, radius):
    while True:
        state, u = rand_float(state)
        state, v = rand_float(state)
        u = u * _F2 - _F1
        v = v * _F2 - _F1
        if u * u + v * v <= _F1:
            break
    tx, ty, tz, bx, by, bz = _disk_basis(nx, ny, nz)
    lx = cx + radius * (u * tx + v * bx)
    ly = cy + radius * (u * ty + v * by)
    lz = cz + radius * (u * tz + v * bz)
    return state, lx, ly, lz


@cuda.jit(device=True, fastmath=True, cache=True)
def _add_point_light_contrib(
    px, py, pz, nx, ny, nz,
    color_r, color_g, color_b, roughness,
    ray_dx, ray_dy, ray_dz,
    lx, ly, lz, lr, lg, lb, intensity,
    spheres, planes, bvh_nodes, bvh_prims,
    out_r, out_g, out_b, weight,
):
    to_light_x = lx - px
    to_light_y = ly - py
    to_light_z = lz - pz
    dist_sq = to_light_x * to_light_x + to_light_y * to_light_y + to_light_z * to_light_z
    if dist_sq < _EPS12:
        return out_r, out_g, out_b
    inv_dist = _F1 / math.sqrt(dist_sq)
    ldx = to_light_x * inv_dist
    ldy = to_light_y * inv_dist
    ldz = to_light_z * inv_dist
    shadow_ox = px + nx * _RAY_EPS
    shadow_oy = py + ny * _RAY_EPS
    shadow_oz = pz + nz * _RAY_EPS
    if direct_any_hit(
        shadow_ox, shadow_oy, shadow_oz, ldx, ldy, ldz,
        spheres, planes,
        _RAY_EPS, math.sqrt(dist_sq) - _RAY_EPS,
    ):
        return out_r, out_g, out_b
    attenuation = weight * intensity / dist_sq
    n_dot_l = nx * ldx + ny * ldy + nz * ldz
    wrap = 0.6 * roughness
    ndotl = max(_F0, (n_dot_l + wrap) / (_F1 + wrap))
    ndotl = ndotl * (_F1 - float32(0.45) * roughness)
    specular = _F0
    if roughness < _F1:
        view_dx, view_dy, view_dz = -ray_dx, -ray_dy, -ray_dz
        hx = ldx + view_dx
        hy = ldy + view_dy
        hz = ldz + view_dz
        hlen = math.sqrt(hx * hx + hy * hy + hz * hz)
        if hlen > _EPS8:
            hx /= hlen
            hy /= hlen
            hz /= hlen
            specular_power = _F2 / (roughness * roughness + _F001)
            ndoth = max(_F0, nx * hx + ny * hy + nz * hz)
            spec_intensity = ndoth ** specular_power
            specular = spec_intensity * (_F1 - roughness)
    out_r += (color_r * lr * attenuation * ndotl) + (lr * attenuation * specular)
    out_g += (color_g * lg * attenuation * ndotl) + (lg * attenuation * specular)
    out_b += (color_b * lb * attenuation * ndotl) + (lb * attenuation * specular)
    return out_r, out_g, out_b


@cuda.jit(device=True, fastmath=True, cache=True)
def _shade_diffuse(
    state, px, py, pz, nx, ny, nz, mat_idx,
    ray_dx, ray_dy, ray_dz,
    spheres, planes, bvh_nodes, bvh_prims,
    materials, lights, area_light_samples,
):
    idx = int(mat_idx)
    color_r = materials[idx, 1]
    color_g = materials[idx, 2]
    color_b = materials[idx, 3]
    roughness = materials[idx, 4]
    if py < _F001 and abs(ny) > float32(0.9):
        cos60 = _F05
        sin60 = float32(0.86602540378)
        u = px * cos60 + pz * sin60
        v = -px * sin60 + pz * cos60
        cx_i = int(math.floor(u * _F1))
        cz_i = int(math.floor(v * _F1))
        if (cx_i + cz_i) % 2 == 0:
            color_r, color_g, color_b = float32(0.25), float32(0.25), float32(0.25)
        else:
            color_r, color_g, color_b = float32(0.85), float32(0.85), float32(0.85)
    elif color_r > _F05 and color_g < float32(0.1) and color_b < float32(0.1):
        n1 = math.sin(px * float32(30.0) + py * float32(15.0))
        n2 = math.sin(py * float32(25.0) + pz * float32(20.0))
        n3 = math.sin(pz * float32(18.0) + px * float32(35.0))
        noise = (n1 + n2 + n3) / float32(3.0)
        color_r = color_r * (float32(0.85) + float32(0.15) * noise)
        roughness = min(_F1, max(_F01, roughness + float32(0.15) * abs(noise)))
        freq = float32(30.0)
        bump = float32(0.08)
        nx = nx + bump * math.cos(px * freq) * math.sin(py * freq) * math.sin(pz * freq)
        ny = ny + bump * math.sin(px * freq) * math.cos(py * freq) * math.sin(pz * freq)
        nz = nz + bump * math.sin(px * freq) * math.sin(py * freq) * math.cos(pz * freq)
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= nlen
        ny /= nlen
        nz /= nlen
    out_r, out_g, out_b = _F0, _F0, _F0
    n_lights = lights.shape[0]
    for i in range(n_lights):
        light_type = lights[i, 0]
        if light_type < _F05:
            out_r, out_g, out_b = _add_point_light_contrib(
                px, py, pz, nx, ny, nz,
                color_r, color_g, color_b, roughness,
                ray_dx, ray_dy, ray_dz,
                lights[i, 1], lights[i, 2], lights[i, 3],
                lights[i, 4], lights[i, 5], lights[i, 6],
                lights[i, 7],
                spheres, planes, bvh_nodes, bvh_prims,
                out_r, out_g, out_b, _F1,
            )
        else:
            cx_l, cy_l, cz_l = lights[i, 1], lights[i, 2], lights[i, 3]
            lnx, lny, lnz = lights[i, 4], lights[i, 5], lights[i, 6]
            lr, lg, lb = lights[i, 7], lights[i, 8], lights[i, 9]
            intensity = lights[i, 10]
            radius = lights[i, 11]
            for s in range(area_light_samples):
                state, lx, ly, lz = _random_disk_point(
                    state, cx_l, cy_l, cz_l, lnx, lny, lnz, radius,
                )
                sample_intensity = intensity / float32(area_light_samples)
                out_r, out_g, out_b = _add_point_light_contrib(
                    px, py, pz, nx, ny, nz,
                    color_r, color_g, color_b, roughness,
                    ray_dx, ray_dy, ray_dz,
                    lx, ly, lz, lr, lg, lb, sample_intensity,
                    spheres, planes, bvh_nodes, bvh_prims,
                    out_r, out_g, out_b, _F1,
                )
    t = _F05 * (ny + _F1)
    amb_r = (_F1 - t) * float32(0.12) + t * float32(0.55)
    amb_g = (_F1 - t) * float32(0.16) + t * float32(0.62)
    amb_b = (_F1 - t) * float32(0.18) + t * float32(0.68)
    out_r += float32(0.28) * color_r * amb_r + float32(0.08) * color_r
    out_g += float32(0.28) * color_g * amb_g + float32(0.08) * color_g
    out_b += float32(0.28) * color_b * amb_b + float32(0.08) * color_b
    return state, out_r, out_g, out_b


@cuda.jit(device=True, fastmath=True, cache=True)
def trace_path(
    state, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
    spheres, planes, bvh_nodes, bvh_prims,
    materials, lights, background,
    max_bounces, area_light_samples,
):
    color_r, color_g, color_b = _F0, _F0, _F0
    th_r, th_g, th_b = _F1, _F1, _F1
    for depth in range(max_bounces + 1):
        t, px, py, pz, nx, ny, nz, front_face, mat_idx = direct_find_closest_hit(
            ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            spheres, planes, _RAY_EPS, _TMAX,
        )
        if t < _F0:
            sky_t = _F05 * (ray_dy + _F1)
            sky_r = (_F1 - sky_t) * _F1 + sky_t * _F05
            sky_g = (_F1 - sky_t) * _F1 + sky_t * float32(0.7)
            sky_b = (_F1 - sky_t) * _F1 + sky_t * _F1
            color_r += th_r * sky_r
            color_g += th_g * sky_g
            color_b += th_b * sky_b
            break
        mat_type = materials[int(mat_idx), 0]
        if mat_type == _F0:
            state, dr, dg, db = _shade_diffuse(
                state, px, py, pz, nx, ny, nz, mat_idx,
                ray_dx, ray_dy, ray_dz,
                spheres, planes, bvh_nodes, bvh_prims,
                materials, lights, area_light_samples,
            )
            color_r += th_r * dr
            color_g += th_g * dg
            color_b += th_b * db
            break
        elif mat_type == _F1:
            mr = materials[int(mat_idx), 1]
            mg = materials[int(mat_idx), 2]
            mb = materials[int(mat_idx), 3]
            roughness = materials[int(mat_idx), 4]
            cos_theta = max(_F0, -(ray_dx * nx + ray_dy * ny + ray_dz * nz))
            fr, fg, fb = _schlick(cos_theta, mr, mg, mb)
            th_r *= fr
            th_g *= fg
            th_b *= fb
            ray_dx, ray_dy, ray_dz = _reflect(ray_dx, ray_dy, ray_dz, nx, ny, nz)
            if roughness > _F0:
                state, rx, ry, rz = rand_in_unit_sphere(state)
                ray_dx += roughness * rx
                ray_dy += roughness * ry
                ray_dz += roughness * rz
            L = math.sqrt(ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz)
            if L > _F0:
                ray_dx /= L
                ray_dy /= L
                ray_dz /= L
            dot_nd = ray_dx * nx + ray_dy * ny + ray_dz * nz
            if dot_nd > _F0:
                ray_ox = px + nx * _RAY_EPS
                ray_oy = py + ny * _RAY_EPS
                ray_oz = pz + nz * _RAY_EPS
            else:
                ray_ox = px - nx * _RAY_EPS
                ray_oy = py - ny * _RAY_EPS
                ray_oz = pz - nz * _RAY_EPS
        elif mat_type == _F2:
            roughness = materials[int(mat_idx), 4]
            ior = materials[int(mat_idx), 5]
            tr = materials[int(mat_idx), 1]
            tg = materials[int(mat_idx), 2]
            tb = materials[int(mat_idx), 3]
            ar = materials[int(mat_idx), 6]
            ag = materials[int(mat_idx), 7]
            ab = materials[int(mat_idx), 8]
            sqrt_ar = math.sqrt(ar)
            sqrt_ag = math.sqrt(ag)
            sqrt_ab = math.sqrt(ab)
            ior_ratio = _F1 / ior if front_face else ior
            cos_theta = min(_F1, max(_FM1, -(ray_dx * nx + ray_dy * ny + ray_dz * nz)))
            sin_theta = math.sqrt(max(_F0, _F1 - cos_theta * cos_theta))
            cannot_refract = ior_ratio * sin_theta > _F1
            r0 = ((_F1 - ior_ratio) / (_F1 + ior_ratio)) ** 2
            reflectance = r0 + (_F1 - r0) * ((_F1 - cos_theta) ** 5)
            state, rf = rand_float(state)
            if cannot_refract or rf < reflectance:
                ray_dx, ray_dy, ray_dz = _reflect(ray_dx, ray_dy, ray_dz, nx, ny, nz)
                th_r *= tr
                th_g *= tg
                th_b *= tb
            else:
                rx, ry, rz, ok = _refract(ray_dx, ray_dy, ray_dz, nx, ny, nz, ior_ratio)
                if ok:
                    ray_dx, ray_dy, ray_dz = rx, ry, rz
                    th_r *= tr
                    th_g *= tg
                    th_b *= tb
                    if front_face:
                        th_r *= sqrt_ar
                        th_g *= sqrt_ag
                        th_b *= sqrt_ab
                    else:
                        th_r *= ar
                        th_g *= ag
                        th_b *= ab
                else:
                    ray_dx, ray_dy, ray_dz = _reflect(ray_dx, ray_dy, ray_dz, nx, ny, nz)
                    th_r *= tr
                    th_g *= tg
                    th_b *= tb
            if roughness > _F0:
                state, jx, jy, jz = rand_in_unit_sphere(state)
                ray_dx += roughness * jx
                ray_dy += roughness * jy
                ray_dz += roughness * jz
            L = math.sqrt(ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz)
            if L > _F0:
                ray_dx /= L
                ray_dy /= L
                ray_dz /= L
            dot_nd = ray_dx * nx + ray_dy * ny + ray_dz * nz
            if dot_nd > _F0:
                ray_ox = px + nx * _RAY_EPS
                ray_oy = py + ny * _RAY_EPS
                ray_oz = pz + nz * _RAY_EPS
            else:
                ray_ox = px - nx * _RAY_EPS
                ray_oy = py - ny * _RAY_EPS
                ray_oz = pz - nz * _RAY_EPS
    else:
        color_r += th_r * background[0]
        color_g += th_g * background[1]
        color_b += th_b * background[2]
    return state, color_r, color_g, color_b


@cuda.jit(device=True, fastmath=True, cache=True)
def trace_pixel_cuda(
    px, py, sample_i,
    cam_ox, cam_oy, cam_oz,
    cam_llx, cam_lly, cam_llz,
    cam_hx, cam_hy, cam_hz,
    cam_vx, cam_vy, cam_vz,
    inv_width, inv_height,
    max_bounces, area_light_samples,
    spheres, planes, bvh_nodes, bvh_prims,
    materials, lights, background,
):
    state = rng_seed(px, py, sample_i, 0)
    state, jx = rand_float(state)
    state, jy = rand_float(state)
    jx -= _F05
    jy -= _F05
    s = (px + jx) * inv_width
    t = _F1 - (py + jy) * inv_height
    dx = cam_llx + s * cam_hx + t * cam_vx - cam_ox
    dy = cam_lly + s * cam_hy + t * cam_vy - cam_oy
    dz = cam_llz + s * cam_hz + t * cam_vz - cam_oz
    dlen = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dlen > _F0:
        dx /= dlen
        dy /= dlen
        dz /= dlen
    state, cr, cg, cb = trace_path(
        state, cam_ox, cam_oy, cam_oz, dx, dy, dz,
        spheres, planes, bvh_nodes, bvh_prims,
        materials, lights, background,
        max_bounces, area_light_samples,
    )
    return cr, cg, cb
