"""Deterministic xorshift64* RNG for CUDA device code."""

from __future__ import annotations

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda, float32

_F0 = float32(0.0)
_F1 = float32(1.0)
_F2 = float32(2.0)
_INV_24BIT = float32(1.0 / 16777216.0)


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def rng_seed(px: float, py: float, sample_i: int, tag: int):
    s = int(px * float32(374761393.0) + py * float32(668265263.0)) & 0x7FFFFFFF
    s = (s ^ (sample_i * 1274126177) ^ (tag * 972163451)) & 0x7FFFFFFF
    if s == 0:
        s = 1
    return s


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def rand_float(state):
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x, float32(x & 0xFFFFFF) * _INV_24BIT


@cuda.jit(device=True, inline=True, fastmath=True, cache=True)
def rand_in_unit_sphere(state):
    while True:
        state, x = rand_float(state)
        state, y = rand_float(state)
        state, z = rand_float(state)
        x = x * _F2 - _F1
        y = y * _F2 - _F1
        z = z * _F2 - _F1
        if x * x + y * y + z * z < _F1:
            return state, x, y, z
    return state, _F0, _F0, _F1
