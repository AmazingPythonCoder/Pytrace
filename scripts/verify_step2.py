"""Week 1 headless render smoke test. Run from project root: python scripts/verify_step2.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.renderer import render, save_png
from src.scene.scene import Scene


def main() -> None:
    scene = Scene.default()
    scene.render.width = 64
    scene.render.height = 36

    image = render(scene)
    assert image.shape == (36, 64, 3)
    assert float(image.max()) > 0.0

    out = ROOT / "output" / "_verify_step2.png"
    save_png(image, out)
    assert out.exists() and out.stat().st_size > 500

    print(f"Step 2 OK: rendered {out}")


if __name__ == "__main__":
    main()
