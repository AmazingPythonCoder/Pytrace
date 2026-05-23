"""Verify render history PNG and gallery index updates."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.history import infer_quality, save_render_history
from src.scene.scene import RenderConfig, Scene


def main() -> None:
    scene = Scene(render=RenderConfig(width=800, height=450, samples=32, max_bounces=8))
    assert infer_quality(scene) == "low"
    scene.render.samples = 33
    assert infer_quality(scene) == "custom"
    scene.render.samples = 32

    image = np.ones((scene.render.height, scene.render.width, 3), dtype=np.float64) * 0.25
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_path = save_render_history(image, scene, root=root)
        second_path = save_render_history(image, scene, root=root)
        assert first_path.exists()
        assert second_path.exists()
        assert first_path != second_path
        assert first_path.parent == root / "output" / "history"

        js_path = root / "output" / "history" / "history.js"
        assert js_path.exists()
        content = js_path.read_text(encoding="utf-8").strip()
        assert content.startswith("const PYTRACE_HISTORY = ")
        data = content.split("const PYTRACE_HISTORY = ", 1)[1].rstrip(";")
        entries = json.loads(data)
        assert len(entries) == 2
        assert entries[0]["filename"].startswith("output/history/render_")
        assert entries[0]["filename"].endswith(".png")
        assert entries[0]["level"] == "low"
        assert entries[0]["width"] == 800
        assert entries[0]["height"] == 450
        assert entries[0]["samples"] == 32
        assert entries[0]["bounces"] == 8

    print("History OK: PNG copies and gallery history.js entries verified")


if __name__ == "__main__":
    main()
