"""Environment map loading for CPU rendering.

The renderer keeps environment maps as float RGB arrays. Ordinary LDR image
formats are normalized into 0..1, while Radiance RGBE HDR files retain their
high dynamic range values.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_EMPTY_ENVIRONMENT = np.zeros((0, 0, 3), dtype=np.float64)


def empty_environment() -> np.ndarray:
    return _EMPTY_ENVIRONMENT.copy()


def load_environment(path: str | Path) -> np.ndarray:
    """Load an equirectangular environment image as ``float64`` RGB.

    Supported paths include Radiance ``.hdr``/``.pic`` via the built-in RGBE
    decoder, common LDR formats via Pillow, and any additional formats exposed
    by imageio.
    """

    if not path:
        return empty_environment()
    env_path = Path(path).expanduser()
    if not env_path.exists() or not env_path.is_file():
        return empty_environment()

    suffix = env_path.suffix.lower()
    try:
        if suffix in {".hdr", ".pic"}:
            return _sanitize_rgb(_read_radiance_hdr(env_path), is_hdr=True)
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        data = iio.imread(env_path)
        return _sanitize_rgb(data, is_hdr=np.issubdtype(np.asarray(data).dtype, np.floating))
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(env_path) as image:
            return _sanitize_rgb(np.asarray(image.convert("RGB")), is_hdr=False)
    except Exception:
        return empty_environment()


def _sanitize_rgb(data: np.ndarray, is_hdr: bool) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return empty_environment()

    arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(info.max) if info.max > 0 else 255.0
        rgb = arr.astype(np.float64) / scale
    else:
        rgb = arr.astype(np.float64)
        if not is_hdr and rgb.size and float(np.nanmax(rgb)) > 1.0:
            rgb /= 255.0

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1e6, neginf=0.0)
    return np.maximum(rgb, 0.0).astype(np.float64, copy=False)


def _read_radiance_hdr(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        header: list[str] = []
        exposure = 1.0
        while True:
            raw = fh.readline()
            if raw == b"":
                raise ValueError("HDR file ended before resolution line")
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                break
            header.append(line)
            if line.upper().startswith("EXPOSURE="):
                try:
                    exposure *= float(line.split("=", 1)[1])
                except ValueError:
                    pass

        format_lines = [line for line in header if line.upper().startswith("FORMAT=")]
        if format_lines and "32-bit_rle_rgbe" not in format_lines[-1]:
            raise ValueError("Only Radiance 32-bit_rle_rgbe HDR files are supported")

        resolution = ""
        while not resolution:
            raw = fh.readline()
            if raw == b"":
                raise ValueError("HDR file is missing a resolution line")
            resolution = raw.decode("ascii", errors="ignore").strip()

        y_sign, height, x_sign, width = _parse_resolution(resolution)
        rgbe = _read_rgbe_pixels(fh, width, height)

    if y_sign == "+":
        rgbe = rgbe[::-1, :, :]
    if x_sign == "-":
        rgbe = rgbe[:, ::-1, :]

    rgb = _rgbe_to_float(rgbe)
    if exposure > 0.0 and exposure != 1.0:
        rgb /= exposure
    return rgb


def _parse_resolution(line: str) -> tuple[str, int, str, int]:
    parts = re.findall(r"([+-])([XY])\s+(\d+)", line)
    if len(parts) != 2:
        raise ValueError(f"Unsupported HDR resolution line: {line!r}")
    dims: dict[str, tuple[str, int]] = {axis: (sign, int(size)) for sign, axis, size in parts}
    if "X" not in dims or "Y" not in dims:
        raise ValueError(f"Unsupported HDR resolution line: {line!r}")
    y_sign, height = dims["Y"]
    x_sign, width = dims["X"]
    if width <= 0 or height <= 0:
        raise ValueError("HDR image dimensions must be positive")
    return y_sign, height, x_sign, width


def _read_rgbe_pixels(fh, width: int, height: int) -> np.ndarray:
    data = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        if 8 <= width <= 32767:
            prefix = fh.read(4)
            if len(prefix) != 4:
                raise ValueError("HDR scanline ended unexpectedly")
            expected_width = (prefix[2] << 8) | prefix[3]
            if prefix[0] == 2 and prefix[1] == 2 and (prefix[2] & 0x80) == 0 and expected_width == width:
                scanline = np.zeros((4, width), dtype=np.uint8)
                for channel in range(4):
                    x = 0
                    while x < width:
                        code_raw = fh.read(1)
                        if len(code_raw) != 1:
                            raise ValueError("HDR RLE data ended unexpectedly")
                        code = code_raw[0]
                        if code > 128:
                            count = code - 128
                            value_raw = fh.read(1)
                            if len(value_raw) != 1:
                                raise ValueError("HDR RLE run ended unexpectedly")
                            if x + count > width:
                                raise ValueError("HDR RLE run exceeds scanline width")
                            scanline[channel, x : x + count] = value_raw[0]
                            x += count
                        elif code > 0:
                            values = fh.read(code)
                            if len(values) != code:
                                raise ValueError("HDR RLE literal ended unexpectedly")
                            if x + code > width:
                                raise ValueError("HDR RLE literal exceeds scanline width")
                            scanline[channel, x : x + code] = np.frombuffer(values, dtype=np.uint8)
                            x += code
                        else:
                            raise ValueError("HDR RLE contains a zero-length packet")
                data[y, :, 0] = scanline[0]
                data[y, :, 1] = scanline[1]
                data[y, :, 2] = scanline[2]
                data[y, :, 3] = scanline[3]
                continue

            rest = fh.read(width * 4 - 4)
            if len(rest) != width * 4 - 4:
                raise ValueError("HDR flat scanline ended unexpectedly")
            data[y] = np.frombuffer(prefix + rest, dtype=np.uint8).reshape((width, 4))
        else:
            raw = fh.read(width * 4)
            if len(raw) != width * 4:
                raise ValueError("HDR flat scanline ended unexpectedly")
            data[y] = np.frombuffer(raw, dtype=np.uint8).reshape((width, 4))
    return data


def _rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    rgb = np.zeros(rgbe.shape[:2] + (3,), dtype=np.float64)
    exponent = rgbe[:, :, 3].astype(np.int32)
    mask = exponent > 0
    if not np.any(mask):
        return rgb
    scale = np.exp2(exponent[mask].astype(np.float64) - 136.0)
    rgb[mask] = rgbe[:, :, :3][mask].astype(np.float64) * scale[:, None]
    return rgb
