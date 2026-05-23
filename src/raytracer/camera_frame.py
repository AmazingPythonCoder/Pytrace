"""Precomputed camera viewport for fast per-ray generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.math.vec3 import cross, normalize
from src.raytracer.ray import Ray


@dataclass(frozen=True)
class CameraFrame:
    origin: np.ndarray
    lower_left: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    lens_u: np.ndarray
    lens_v: np.ndarray
    aperture: float
    focus_distance: float
    inv_width: float
    inv_height: float


def compute_camera_frame(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fov: float,
    image_width: int,
    image_height: int,
    aperture: float = 0.0,
    focus_distance: float | None = None,
) -> CameraFrame:
    aspect = image_width / image_height
    half_h = math.tan(math.radians(fov / 2.0))
    half_w = aspect * half_h

    w = normalize(position - target)
    u = normalize(cross(up, w))
    v = cross(w, u)
    focus = float(focus_distance) if focus_distance is not None else float(np.linalg.norm(position - target))
    focus = max(1e-6, focus)

    lower_left = position - focus * half_w * u - focus * half_h * v - focus * w
    horizontal = 2.0 * focus * half_w * u
    vertical = 2.0 * focus * half_h * v

    return CameraFrame(
        origin=np.asarray(position, dtype=np.float64),
        lower_left=lower_left,
        horizontal=horizontal,
        vertical=vertical,
        lens_u=u,
        lens_v=v,
        aperture=max(0.0, float(aperture)),
        focus_distance=focus,
        inv_width=1.0 / image_width,
        inv_height=1.0 / image_height,
    )


def generate_ray(px: float, py: float, frame: CameraFrame) -> Ray:
    s = px * frame.inv_width
    t = 1.0 - py * frame.inv_height
    direction = normalize(
        frame.lower_left + s * frame.horizontal + t * frame.vertical - frame.origin
    )
    return Ray(origin=frame.origin, direction=direction)
