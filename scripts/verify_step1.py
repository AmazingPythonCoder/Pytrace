"""Sanity checks for Step 1 (vec3 math). Run from project root: python scripts/verify_step1.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.math.vec3 import cross, dot, length, normalize, reflect, vec3


def main() -> None:
    v = vec3(3.0, 4.0, 0.0)
    assert abs(dot(v, v) - 25.0) < 1e-9
    assert abs(length(v) - 5.0) < 1e-9

    a = vec3(1.0, 0.0, 0.0)
    b = vec3(0.0, 1.0, 0.0)
    c = cross(a, b)
    assert np.allclose(c, [0.0, 0.0, 1.0])

    n = normalize(vec3(0.0, 3.0, 4.0))
    assert abs(length(n) - 1.0) < 1e-9

    incident = vec3(1.0, -1.0, 0.0)
    normal = vec3(0.0, 1.0, 0.0)
    r = reflect(incident, normal)
    assert np.allclose(r, [1.0, 1.0, 0.0])

    summed = vec3(1.0, 2.0, 3.0) + vec3(4.0, 5.0, 6.0)
    assert np.allclose(summed, [5.0, 7.0, 9.0])

    print("Step 1 OK: vec3 math verified")


if __name__ == "__main__":
    main()
