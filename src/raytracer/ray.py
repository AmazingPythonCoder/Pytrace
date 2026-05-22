from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction
