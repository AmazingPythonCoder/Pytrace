from __future__ import annotations

import random
import math

import numpy as np
from numba import njit

from src.raytracer.intersect import any_hit, find_closest_hit
from src.raytracer.sampling import random_in_unit_sphere

@njit(cache=True)
def _reflect(ix: float, iy: float, iz: float, nx: float, ny: float, nz: float):
    dot_in = ix*nx + iy*ny + iz*nz
    return ix - 2.0*dot_in*nx, iy - 2.0*dot_in*ny, iz - 2.0*dot_in*nz

@njit(cache=True)
def _schlick(cos_theta: float, f0_r: float, f0_g: float, f0_b: float):
    one_minus_cos = 1.0 - cos_theta
    one_minus_cos_5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos
    fr = f0_r + (1.0 - f0_r) * one_minus_cos_5
    fg = f0_g + (1.0 - f0_g) * one_minus_cos_5
    fb = f0_b + (1.0 - f0_b) * one_minus_cos_5
    return fr, fg, fb


@njit(cache=True)
def _refract(ux: float, uy: float, uz: float, nx: float, ny: float, nz: float, ior_ratio: float):
    cos_theta = min(1.0, max(-1.0, -(ux*nx + uy*ny + uz*nz)))
    r_out_perp_x = ior_ratio * (ux + cos_theta * nx)
    r_out_perp_y = ior_ratio * (uy + cos_theta * ny)
    r_out_perp_z = ior_ratio * (uz + cos_theta * nz)
    r_out_perp_len_sq = r_out_perp_x**2 + r_out_perp_y**2 + r_out_perp_z**2
    if r_out_perp_len_sq >= 1.0:
        return 0.0, 0.0, 0.0, False
    r_out_parallel_mag = -math.sqrt(abs(1.0 - r_out_perp_len_sq))
    return r_out_perp_x + r_out_parallel_mag * nx, r_out_perp_y + r_out_parallel_mag * ny, r_out_perp_z + r_out_parallel_mag * nz, True

@njit(cache=True)
def _disk_basis(nx: float, ny: float, nz: float):
    if abs(nz) < abs(nx):
        ax, ay, az = 1.0, 0.0, 0.0
    else:
        ax, ay, az = 0.0, 0.0, 1.0
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


@njit(cache=True)
def _random_disk_point(cx: float, cy: float, cz: float, nx: float, ny: float, nz: float, radius: float):
    while True:
        u = random.random() * 2.0 - 1.0
        v = random.random() * 2.0 - 1.0
        if u * u + v * v <= 1.0:
            break
    tx, ty, tz, bx, by, bz = _disk_basis(nx, ny, nz)
    lx = cx + radius * (u * tx + v * bx)
    ly = cy + radius * (u * ty + v * by)
    lz = cz + radius * (u * tz + v * bz)
    return lx, ly, lz


@njit(cache=True)
def _background_color(dx: float, dy: float, dz: float, background: np.ndarray, environment: np.ndarray, background_mode: int):
    if background_mode == 2 and environment.shape[0] > 0 and environment.shape[1] > 0:
        u = math.atan2(dz, dx) / (2.0 * math.pi) + 0.5
        v = math.acos(max(-1.0, min(1.0, dy))) / math.pi
        ix = int(max(0, min(environment.shape[1] - 1, int(u * float(environment.shape[1] - 1)))))
        iy = int(max(0, min(environment.shape[0] - 1, int(v * float(environment.shape[0] - 1)))))
        return environment[iy, ix, 0], environment[iy, ix, 1], environment[iy, ix, 2]
    if background_mode == 0:
        return background[0], background[1], background[2]
    sky_t = 0.5 * (dy + 1.0)
    sky_r = (1.0 - sky_t) * 1.0 + sky_t * 0.5
    sky_g = (1.0 - sky_t) * 1.0 + sky_t * 0.7
    sky_b = (1.0 - sky_t) * 1.0 + sky_t * 1.0
    return sky_r, sky_g, sky_b


@njit(cache=True)
def _random_hemisphere_direction(nx: float, ny: float, nz: float):
    rx, ry, rz = random_in_unit_sphere()
    lx = nx + rx
    ly = ny + ry
    lz = nz + rz
    llen = math.sqrt(lx * lx + ly * ly + lz * lz)
    if llen <= 1e-12:
        return nx, ny, nz
    return lx / llen, ly / llen, lz / llen


@njit(cache=True)
def _add_point_light_contrib(
    px: float, py: float, pz: float,
    nx: float, ny: float, nz: float,
    color_r: float, color_g: float, color_b: float,
    roughness: float,
    ray_dx: float, ray_dy: float, ray_dz: float,
    lx: float, ly: float, lz: float,
    lr: float, lg: float, lb: float,
    intensity: float,
    spheres: np.ndarray, planes: np.ndarray,
    triangles: np.ndarray,
    bvh_nodes: np.ndarray, bvh_prims: np.ndarray,
    out_r: float, out_g: float, out_b: float,
    weight: float = 1.0,
) -> tuple[float, float, float]:
    to_light_x = lx - px
    to_light_y = ly - py
    to_light_z = lz - pz

    dist_sq = to_light_x**2 + to_light_y**2 + to_light_z**2
    if dist_sq < 1e-12:
        return out_r, out_g, out_b

    inv_dist = 1.0 / math.sqrt(dist_sq)
    ldx = to_light_x * inv_dist
    ldy = to_light_y * inv_dist
    ldz = to_light_z * inv_dist

    shadow_ox = px + nx * 1e-4
    shadow_oy = py + ny * 1e-4
    shadow_oz = pz + nz * 1e-4

    if any_hit(
        shadow_ox, shadow_oy, shadow_oz,
        ldx, ldy, ldz,
        spheres, planes, triangles, bvh_nodes, bvh_prims,
        1e-4, math.sqrt(dist_sq) - 1e-4,
    ):
        return out_r, out_g, out_b

    attenuation = weight * intensity / dist_sq
    n_dot_l = nx * ldx + ny * ldy + nz * ldz

    wrap = 0.6 * roughness
    ndotl = max(0.0, (n_dot_l + wrap) / (1.0 + wrap))
    ndotl = ndotl * (1.0 - 0.45 * roughness)

    specular = 0.0
    if roughness < 1.0:
        view_dx, view_dy, view_dz = -ray_dx, -ray_dy, -ray_dz
        hx = ldx + view_dx
        hy = ldy + view_dy
        hz = ldz + view_dz
        hlen = math.sqrt(hx**2 + hy**2 + hz**2)
        if hlen > 1e-8:
            hx /= hlen
            hy /= hlen
            hz /= hlen
            specular_power = 2.0 / (roughness**2 + 0.001)
            ndoth = max(0.0, nx * hx + ny * hy + nz * hz)
            spec_intensity = ndoth**specular_power
            specular = spec_intensity * (1.0 - roughness)

    out_r += (color_r * lr * attenuation * ndotl) + (lr * attenuation * specular)
    out_g += (color_g * lg * attenuation * ndotl) + (lg * attenuation * specular)
    out_b += (color_b * lb * attenuation * ndotl) + (lb * attenuation * specular)
    return out_r, out_g, out_b


@njit(cache=True)
def _add_directional_light_contrib(
    px: float, py: float, pz: float,
    nx: float, ny: float, nz: float,
    color_r: float, color_g: float, color_b: float,
    roughness: float,
    ray_dx: float, ray_dy: float, ray_dz: float,
    dir_x: float, dir_y: float, dir_z: float,
    lr: float, lg: float, lb: float,
    intensity: float,
    spheres: np.ndarray, planes: np.ndarray, triangles: np.ndarray,
    bvh_nodes: np.ndarray, bvh_prims: np.ndarray,
    out_r: float, out_g: float, out_b: float,
) -> tuple[float, float, float]:
    ldx = -dir_x
    ldy = -dir_y
    ldz = -dir_z
    llen = math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz)
    if llen <= 1e-12:
        return out_r, out_g, out_b
    ldx /= llen
    ldy /= llen
    ldz /= llen

    shadow_ox = px + nx * 1e-4
    shadow_oy = py + ny * 1e-4
    shadow_oz = pz + nz * 1e-4
    if any_hit(
        shadow_ox, shadow_oy, shadow_oz,
        ldx, ldy, ldz,
        spheres, planes, triangles, bvh_nodes, bvh_prims,
        1e-4, 1e9,
    ):
        return out_r, out_g, out_b

    n_dot_l = nx * ldx + ny * ldy + nz * ldz
    wrap = 0.6 * roughness
    ndotl = max(0.0, (n_dot_l + wrap) / (1.0 + wrap))
    ndotl = ndotl * (1.0 - 0.45 * roughness)

    specular = 0.0
    if roughness < 1.0:
        view_dx, view_dy, view_dz = -ray_dx, -ray_dy, -ray_dz
        hx = ldx + view_dx
        hy = ldy + view_dy
        hz = ldz + view_dz
        hlen = math.sqrt(hx * hx + hy * hy + hz * hz)
        if hlen > 1e-8:
            hx /= hlen
            hy /= hlen
            hz /= hlen
            specular_power = 2.0 / (roughness * roughness + 0.001)
            ndoth = max(0.0, nx * hx + ny * hy + nz * hz)
            specular = (ndoth ** specular_power) * (1.0 - roughness)

    out_r += (color_r * lr * intensity * ndotl) + (lr * intensity * specular)
    out_g += (color_g * lg * intensity * ndotl) + (lg * intensity * specular)
    out_b += (color_b * lb * intensity * ndotl) + (lb * intensity * specular)
    return out_r, out_g, out_b


@njit(cache=True)
def _shade_diffuse(
    px: float, py: float, pz: float,
    nx: float, ny: float, nz: float,
    mat_idx: float,
    ray_dx: float, ray_dy: float, ray_dz: float,
    spheres: np.ndarray, planes: np.ndarray, triangles: np.ndarray,
    bvh_nodes: np.ndarray, bvh_prims: np.ndarray,
    materials: np.ndarray, lights: np.ndarray,
    area_light_samples: int,
):
    idx = int(mat_idx)
    color_r, color_g, color_b = materials[idx, 1], materials[idx, 2], materials[idx, 3]
    roughness = materials[idx, 4]

    if py < 1e-3 and abs(ny) > 0.9:
        scale = 1.0
        cos60 = 0.5
        sin60 = 0.86602540378
        u = px * cos60 + pz * sin60
        v = -px * sin60 + pz * cos60
        cx = int(math.floor(u * scale))
        cz = int(math.floor(v * scale))
        if (cx + cz) % 2 == 0:
            color_r, color_g, color_b = 0.25, 0.25, 0.25
        else:
            color_r, color_g, color_b = 0.85, 0.85, 0.85
            
    elif color_r > 0.5 and color_g < 0.1 and color_b < 0.1:
        # Procedural imperfections for the Red Sphere
        n1 = math.sin(px * 30.0 + py * 15.0)
        n2 = math.sin(py * 25.0 + pz * 20.0)
        n3 = math.sin(pz * 18.0 + px * 35.0)
        noise = (n1 + n2 + n3) / 3.0
        # noise is roughly -1 to 1
        
        # 1. Subtle color variation
        color_r = color_r * (0.85 + 0.15 * noise)
        
        # 2. Roughness variation (smudges/dust)
        roughness = min(1.0, max(0.01, roughness + 0.15 * abs(noise)))
        
        # 3. Micro-bump map (structured dimple pattern)
        # We calculate the gradient of a 3D sine grid to perturb the normals realistically
        freq = 30.0
        bump = 0.08
        nx += bump * math.cos(px * freq) * math.sin(py * freq) * math.sin(pz * freq)
        ny += bump * math.sin(px * freq) * math.cos(py * freq) * math.sin(pz * freq)
        nz += bump * math.sin(px * freq) * math.sin(py * freq) * math.cos(pz * freq)
        nlen = math.sqrt(nx*nx + ny*ny + nz*nz)
        nx /= nlen; ny /= nlen; nz /= nlen

    out_r, out_g, out_b = 0.0, 0.0, 0.0

    for i in range(lights.shape[0]):
        light_type = lights[i, 0]
        if light_type < 0.5:
            out_r, out_g, out_b = _add_point_light_contrib(
                px, py, pz, nx, ny, nz,
                color_r, color_g, color_b, roughness,
                ray_dx, ray_dy, ray_dz,
                lights[i, 1], lights[i, 2], lights[i, 3],
                lights[i, 4], lights[i, 5], lights[i, 6],
                lights[i, 7],
                spheres, planes, triangles, bvh_nodes, bvh_prims, out_r, out_g, out_b,
            )
        elif light_type < 1.5:
            cx, cy, cz = lights[i, 1], lights[i, 2], lights[i, 3]
            lnx, lny, lnz = lights[i, 4], lights[i, 5], lights[i, 6]
            lr, lg, lb = lights[i, 7], lights[i, 8], lights[i, 9]
            intensity = lights[i, 10]
            radius = lights[i, 11]
            for _ in range(area_light_samples):
                lx, ly, lz = _random_disk_point(cx, cy, cz, lnx, lny, lnz, radius)
                sample_intensity = intensity / float(area_light_samples)
                out_r, out_g, out_b = _add_point_light_contrib(
                    px, py, pz, nx, ny, nz,
                    color_r, color_g, color_b, roughness,
                    ray_dx, ray_dy, ray_dz,
                    lx, ly, lz, lr, lg, lb, sample_intensity,
                    spheres, planes, triangles, bvh_nodes, bvh_prims, out_r, out_g, out_b,
                )
        else:
            out_r, out_g, out_b = _add_directional_light_contrib(
                px, py, pz, nx, ny, nz,
                color_r, color_g, color_b, roughness,
                ray_dx, ray_dy, ray_dz,
                lights[i, 1], lights[i, 2], lights[i, 3],
                lights[i, 4], lights[i, 5], lights[i, 6],
                lights[i, 7],
                spheres, planes, triangles, bvh_nodes, bvh_prims,
                out_r, out_g, out_b,
            )

    t = 0.5 * (ny + 1.0)
    amb_r = (1.0 - t) * 0.12 + t * 0.55
    amb_g = (1.0 - t) * 0.16 + t * 0.62
    amb_b = (1.0 - t) * 0.18 + t * 0.68

    out_r += 0.28 * color_r * amb_r + 0.08 * color_r
    out_g += 0.28 * color_g * amb_g + 0.08 * color_g
    out_b += 0.28 * color_b * amb_b + 0.08 * color_b

    return out_r, out_g, out_b

@njit(cache=True)
def trace(
    ray_ox: float, ray_oy: float, ray_oz: float,
    ray_dx: float, ray_dy: float, ray_dz: float,
    spheres: np.ndarray, planes: np.ndarray, triangles: np.ndarray,
    bvh_nodes: np.ndarray, bvh_prims: np.ndarray,
    materials: np.ndarray, lights: np.ndarray,
    background: np.ndarray, environment: np.ndarray,
    max_bounces: int, area_light_samples: int,
    render_mode: int, background_mode: int,
) -> tuple[float, float, float]:
    
    color_r, color_g, color_b = 0.0, 0.0, 0.0
    th_r, th_g, th_b = 1.0, 1.0, 1.0
    
    for depth in range(max_bounces + 1):
        hit = find_closest_hit(
            ray_ox, ray_oy, ray_oz,
            ray_dx, ray_dy, ray_dz,
            spheres, planes, triangles, bvh_nodes, bvh_prims, 1e-4, 1e9,
        )
        
        t, px, py, pz, nx, ny, nz, front_face, mat_idx = hit
        
        if t < 0.0:
            bg_r, bg_g, bg_b = _background_color(ray_dx, ray_dy, ray_dz, background, environment, background_mode)
            color_r += th_r * bg_r
            color_g += th_g * bg_g
            color_b += th_b * bg_b
            break
            
        mat_type = materials[int(mat_idx), 0]
        
        if mat_type == 0.0: # Diffuse
            if render_mode == 1:
                idx = int(mat_idx)
                th_r *= materials[idx, 1]
                th_g *= materials[idx, 2]
                th_b *= materials[idx, 3]
                ray_dx, ray_dy, ray_dz = _random_hemisphere_direction(nx, ny, nz)
                ray_ox = px + nx * 1e-4
                ray_oy = py + ny * 1e-4
                ray_oz = pz + nz * 1e-4
            else:
                dr, dg, db = _shade_diffuse(
                    px, py, pz, nx, ny, nz, mat_idx,
                    ray_dx, ray_dy, ray_dz,
                    spheres, planes, triangles, bvh_nodes, bvh_prims,
                    materials, lights, area_light_samples,
                )
                color_r += th_r * dr
                color_g += th_g * dg
                color_b += th_b * db
                break
            
        elif mat_type == 1.0: # Specular
            mr, mg, mb = materials[int(mat_idx), 1], materials[int(mat_idx), 2], materials[int(mat_idx), 3]
            roughness = materials[int(mat_idx), 4]

            cos_theta = max(0.0, -(ray_dx * nx + ray_dy * ny + ray_dz * nz))
            fr, fg, fb = _schlick(cos_theta, mr, mg, mb)
            th_r *= fr
            th_g *= fg
            th_b *= fb
            
            ray_dx, ray_dy, ray_dz = _reflect(ray_dx, ray_dy, ray_dz, nx, ny, nz)
            if roughness > 0.0:
                rx, ry, rz = random_in_unit_sphere()
                ray_dx += roughness * rx
                ray_dy += roughness * ry
                ray_dz += roughness * rz
            
            L = math.sqrt(ray_dx**2 + ray_dy**2 + ray_dz**2)
            if L > 0:
                ray_dx /= L; ray_dy /= L; ray_dz /= L
            
            dot_nd = ray_dx*nx + ray_dy*ny + ray_dz*nz
            if dot_nd > 0.0:
                ray_ox = px + nx * 1e-4
                ray_oy = py + ny * 1e-4
                ray_oz = pz + nz * 1e-4
            else:
                ray_ox = px - nx * 1e-4
                ray_oy = py - ny * 1e-4
                ray_oz = pz - nz * 1e-4
            
        elif mat_type == 2.0: # Glass
            roughness = materials[int(mat_idx), 4]
            ior = materials[int(mat_idx), 5]
            tr, tg, tb = materials[int(mat_idx), 1], materials[int(mat_idx), 2], materials[int(mat_idx), 3]
            ar, ag, ab = materials[int(mat_idx), 6], materials[int(mat_idx), 7], materials[int(mat_idx), 8]
            sqrt_ar = math.sqrt(ar)
            sqrt_ag = math.sqrt(ag)
            sqrt_ab = math.sqrt(ab)
            
            ior_ratio = 1.0 / ior if front_face else ior
            
            cos_theta = min(1.0, max(-1.0, -(ray_dx*nx + ray_dy*ny + ray_dz*nz)))
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            
            cannot_refract = (ior_ratio * sin_theta > 1.0)
            r0 = ((1.0 - ior_ratio) / (1.0 + ior_ratio)) ** 2
            reflectance = r0 + (1.0 - r0) * ((1.0 - cos_theta) ** 5)
            
            if cannot_refract or random.random() < reflectance:
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

            if roughness > 0.0:
                jx, jy, jz = random_in_unit_sphere()
                ray_dx += roughness * jx
                ray_dy += roughness * jy
                ray_dz += roughness * jz
                    
            L = math.sqrt(ray_dx**2 + ray_dy**2 + ray_dz**2)
            if L > 0:
                ray_dx /= L; ray_dy /= L; ray_dz /= L
            
            dot_nd = ray_dx*nx + ray_dy*ny + ray_dz*nz
            if dot_nd > 0.0:
                ray_ox = px + nx * 1e-4
                ray_oy = py + ny * 1e-4
                ray_oz = pz + nz * 1e-4
            else:
                ray_ox = px - nx * 1e-4
                ray_oy = py - ny * 1e-4
                ray_oz = pz - nz * 1e-4
        elif mat_type == 3.0: # Emissive
            idx = int(mat_idx)
            strength = materials[idx, 4]
            color_r += th_r * materials[idx, 1] * strength
            color_g += th_g * materials[idx, 2] * strength
            color_b += th_b * materials[idx, 3] * strength
            break
    else:
        bg_r, bg_g, bg_b = _background_color(ray_dx, ray_dy, ray_dz, background, environment, background_mode)
        color_r += th_r * bg_r
        color_g += th_g * bg_g
        color_b += th_b * bg_b

    return color_r, color_g, color_b

@njit(cache=True)
def trace_pixel(
    px: float,
    py: float,
    cam_ox: float,
    cam_oy: float,
    cam_oz: float,
    cam_llx: float,
    cam_lly: float,
    cam_llz: float,
    cam_hx: float,
    cam_hy: float,
    cam_hz: float,
    cam_vx: float,
    cam_vy: float,
    cam_vz: float,
    lens_ux: float,
    lens_uy: float,
    lens_uz: float,
    lens_vx: float,
    lens_vy: float,
    lens_vz: float,
    aperture: float,
    inv_width: float,
    inv_height: float,
    samples: int,
    max_bounces: int,
    area_light_samples: int,
    spheres: np.ndarray,
    planes: np.ndarray,
    triangles: np.ndarray,
    bvh_nodes: np.ndarray,
    bvh_prims: np.ndarray,
    materials: np.ndarray,
    lights: np.ndarray,
    background: np.ndarray,
    environment: np.ndarray,
    render_mode: int,
    background_mode: int,
) -> tuple[float, float, float]:
    cr, cg, cb = 0.0, 0.0, 0.0
    inv_samples = 1.0 / samples

    for _ in range(samples):
        jx = random.random() - 0.5
        jy = random.random() - 0.5

        s = (px + jx) * inv_width
        t = 1.0 - (py + jy) * inv_height

        focus_x = cam_llx + s * cam_hx + t * cam_vx
        focus_y = cam_lly + s * cam_hy + t * cam_vy
        focus_z = cam_llz + s * cam_hz + t * cam_vz
        dx = focus_x - cam_ox
        dy = focus_y - cam_oy
        dz = focus_z - cam_oz

        dlen = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dlen > 0.0:
            dx /= dlen
            dy /= dlen
            dz /= dlen

        ray_ox = cam_ox
        ray_oy = cam_oy
        ray_oz = cam_oz
        if aperture > 0.0:
            while True:
                lx = random.random() * 2.0 - 1.0
                ly = random.random() * 2.0 - 1.0
                if lx * lx + ly * ly <= 1.0:
                    break
            lens_x = aperture * lx
            lens_y = aperture * ly
            off_x = lens_ux * lens_x + lens_vx * lens_y
            off_y = lens_uy * lens_x + lens_vy * lens_y
            off_z = lens_uz * lens_x + lens_vz * lens_y
            ray_ox = cam_ox + off_x
            ray_oy = cam_oy + off_y
            ray_oz = cam_oz + off_z
            dx = focus_x - ray_ox
            dy = focus_y - ray_oy
            dz = focus_z - ray_oz
            dlen = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dlen > 0.0:
                dx /= dlen
                dy /= dlen
                dz /= dlen

        tr, tg, tb = trace(
            ray_ox,
            ray_oy,
            ray_oz,
            dx,
            dy,
            dz,
            spheres,
            planes,
            triangles,
            bvh_nodes,
            bvh_prims,
            materials,
            lights,
            background,
            environment,
            max_bounces,
            area_light_samples,
            render_mode,
            background_mode,
        )
        cr += tr
        cg += tg
        cb += tb

    return cr * inv_samples, cg * inv_samples, cb * inv_samples


@njit(cache=True)
def render_tile(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    cam_ox: float,
    cam_oy: float,
    cam_oz: float,
    cam_llx: float,
    cam_lly: float,
    cam_llz: float,
    cam_hx: float,
    cam_hy: float,
    cam_hz: float,
    cam_vx: float,
    cam_vy: float,
    cam_vz: float,
    lens_ux: float,
    lens_uy: float,
    lens_uz: float,
    lens_vx: float,
    lens_vy: float,
    lens_vz: float,
    aperture: float,
    inv_width: float,
    inv_height: float,
    samples: int,
    max_bounces: int,
    area_light_samples: int,
    spheres: np.ndarray,
    planes: np.ndarray,
    triangles: np.ndarray,
    bvh_nodes: np.ndarray,
    bvh_prims: np.ndarray,
    materials: np.ndarray,
    lights: np.ndarray,
    background: np.ndarray,
    environment: np.ndarray,
    render_mode: int,
    background_mode: int,
) -> np.ndarray:
    tile_h = y1 - y0
    tile_w = x1 - x0
    result = np.empty((tile_h, tile_w, 3), dtype=np.float64)

    for row in range(tile_h):
        py = y0 + row
        for col in range(tile_w):
            px = x0 + col
            cr, cg, cb = trace_pixel(
                float(px),
                float(py),
                cam_ox,
                cam_oy,
                cam_oz,
                cam_llx,
                cam_lly,
                cam_llz,
                cam_hx,
                cam_hy,
                cam_hz,
                cam_vx,
                cam_vy,
                cam_vz,
                lens_ux,
                lens_uy,
                lens_uz,
                lens_vx,
                lens_vy,
                lens_vz,
                aperture,
                inv_width,
                inv_height,
                samples,
                max_bounces,
                area_light_samples,
                spheres,
                planes,
                triangles,
                bvh_nodes,
                bvh_prims,
                materials,
                lights,
                background,
                environment,
                render_mode,
                background_mode,
            )
            result[row, col, 0] = cr
            result[row, col, 1] = cg
            result[row, col, 2] = cb

    return result
