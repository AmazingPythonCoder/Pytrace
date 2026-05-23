"""Week 2 smoke test: reflections, glass, AA, tone mapping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.raytracer.renderer import render, save_png
from src.scene.materials import GlassMaterial, SpecularMaterial
from src.scene.scene import Scene


def main() -> None:
    scene = Scene.default()
    scene.render.width = 64
    scene.render.height = 36
    scene.render.samples = 4
    scene.render.max_bounces = 6

    image = render(scene, use_gpu=False)
    assert image.shape == (36, 64, 3)
    assert float(image.max()) > 0.0

    bg = scene.render.background_color
    diff_from_bg = np.linalg.norm(image - bg, axis=2)
    assert float(diff_from_bg.max()) > 0.05

    out = ROOT / "output" / "_verify_step3.png"
    save_png(image, out, exposure=scene.render.exposure)
    assert out.exists() and out.stat().st_size > 500

    types = {type(o.material) for o in scene.objects}
    assert GlassMaterial in types
    assert SpecularMaterial in types

    print(f"Step 3 OK: rendered {out}")


if __name__ == "__main__":
    main()
