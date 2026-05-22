"""Random sampling helpers for anti-aliasing and material roughness."""

from __future__ import annotations
import random
import math
import numpy as np
from numba import njit

from src.math.vec3 import normalize, vec3


@njit(cache=True)
def random_in_unit_sphere() -> tuple[float, float, float]:
    while True:
        x = random.random() * 2.0 - 1.0
        y = random.random() * 2.0 - 1.0
        z = random.random() * 2.0 - 1.0
        if x*x + y*y + z*z < 1.0:
            return x, y, z


def random_unit_vector() -> np.ndarray:
    return normalize(random_in_unit_sphere())
