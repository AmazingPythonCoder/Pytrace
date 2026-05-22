"""3D vector math: explicit functions on numpy (3,) arrays."""

from __future__ import annotations

import numpy as np
from numba import njit

_EPS = 1e-8


@njit(cache=True)
def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Create a float64 vector with shape (3,)."""
    return np.array([x, y, z], dtype=np.float64)


@njit(cache=True)
def _as_vec3(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected shape (3,), got {arr.shape}")
    return arr


@njit(cache=True)
def dot(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _as_vec3(a), _as_vec3(b)
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


@njit(cache=True)
def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _as_vec3(a), _as_vec3(b)
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def length_squared(v: np.ndarray) -> float:
    return dot(v, v)


@njit(cache=True)
def length(v: np.ndarray) -> float:
    return float(np.sqrt(length_squared(v)))


@njit(cache=True)
def normalize(v: np.ndarray) -> np.ndarray:
    x = _as_vec3(v)
    ls = dot(x, x)
    if ls <= _EPS * _EPS:
        return vec3(0.0, 0.0, 0.0)
    inv = 1.0 / np.sqrt(ls)
    return x * inv


@njit(cache=True)
def reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect incident direction about normal: v - 2*dot(v,n)*n."""
    i = _as_vec3(incident)
    n = normalize(normal)
    return i - 2.0 * dot(i, n) * n


@njit(cache=True)
def refract(uv: np.ndarray, normal: np.ndarray, ior_ratio: float) -> np.ndarray | None:
    """Snell's law refraction. Returns None on total internal reflection."""
    unit_normal = normalize(normal)
    cos_theta = min(1.0, max(-1.0, -dot(uv, unit_normal)))
    r_perp = ior_ratio * (uv + cos_theta * unit_normal)
    r_perp_len_sq = dot(r_perp, r_perp)
    if r_perp_len_sq >= 1.0:
        return None
    r_parallel = -np.sqrt(1.0 - r_perp_len_sq) * unit_normal
    return r_perp + r_parallel
