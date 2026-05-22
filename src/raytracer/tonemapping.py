"""HDR float buffer to displayable sRGB."""

from __future__ import annotations

import numpy as np

_GAMMA = 1.0 / 2.2


def tonemap(color: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Apply exposure, luminance-based Reinhard tone map, and gamma."""
    c = np.asarray(color, dtype=np.float64) * exposure
    
    # Calculate luminance to preserve hue/saturation and avoid white burnout
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    L = np.sum(c * weights, axis=-1, keepdims=True)
    
    c = c / (1.0 + L)
    c = np.clip(c, 0.0, 1.0) ** _GAMMA
    return c


def tonemap_to_uint8(color: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Tonemap and convert to uint8 RGB."""
    mapped = tonemap(color, exposure)
    return (mapped * 255.0 + 0.5).astype(np.uint8)
