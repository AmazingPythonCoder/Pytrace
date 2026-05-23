import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda  # numba import MUST come after all of the above
import numpy as np
@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

a = np.ones(1000)
b = np.ones(1000)
c = np.zeros(1000)
add[64, 16](a, b, c)
print(c[:5])
