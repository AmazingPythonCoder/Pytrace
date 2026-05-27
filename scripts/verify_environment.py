"""Verify environment map loading and CPU sky sampling."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.environment import load_environment
from src.raytracer.render_context import build_render_context
from src.raytracer.renderer import render
from src.scene.camera import Camera
from src.scene.scene import RenderConfig, Scene


def _write_flat_hdr(path: Path) -> None:
    # RGBE values decode to [1.0, 0.5, 0.25] and [0.25, 0.5, 1.0].
    pixels = bytes([128, 64, 32, 129, 32, 64, 128, 129])
    path.write_bytes(
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n"
        b"\n"
        b"-Y 1 +X 2\n"
        + pixels
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        png_path = tmp_path / "env.png"
        Image.fromarray(
            np.array([[[255, 0, 0], [0, 128, 255]]], dtype=np.uint8),
            mode="RGB",
        ).save(png_path)
        ldr = load_environment(png_path)
        assert ldr.shape == (1, 2, 3)
        assert np.allclose(ldr[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))

        hdr_path = tmp_path / "env.hdr"
        _write_flat_hdr(hdr_path)
        hdr = load_environment(hdr_path)
        assert hdr.shape == (1, 2, 3)
        assert np.allclose(hdr[0, 0], np.array([1.0, 0.5, 0.25], dtype=np.float64))
        assert np.allclose(hdr[0, 1], np.array([0.25, 0.5, 1.0], dtype=np.float64))

        assert load_environment(tmp_path / "missing.hdr").shape == (0, 0, 3)

        scene = Scene(
            camera=Camera(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                target=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            ),
            render=RenderConfig(
                width=4,
                height=2,
                samples=1,
                max_bounces=1,
                background_mode="environment",
                environment_path=str(hdr_path),
            ),
        )
        ctx = build_render_context(scene)
        assert ctx.environment.shape == (1, 2, 3)
        assert ctx.gpu_supported
        assert ctx.gpu_fallback_reason == ""
        image = render(scene, use_gpu=False, parallel=False)
        assert image.shape == (2, 4, 3)
        assert np.isfinite(image).all()
        assert float(image.max()) > 0.5

    print("Environment OK: LDR, HDR RGBE, missing-file fallback, and sky sampling verified")


if __name__ == "__main__":
    main()
