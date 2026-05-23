from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.raytracer.renderer import save_png
from src.scene.scene import Scene


QUALITY_PRESETS = {
    "low": (800, 450, 32, 8),
    "med": (1280, 720, 128, 10),
    "high": (1920, 1080, 384, 12),
    "ultra": (3840, 2160, 768, 16),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_quality(scene: Scene, fallback: str = "custom") -> str:
    render_tuple = (
        scene.render.width,
        scene.render.height,
        scene.render.samples,
        scene.render.max_bounces,
    )
    for quality, preset in QUALITY_PRESETS.items():
        if render_tuple == preset:
            return quality
    return fallback


def save_render_history(
    image: np.ndarray,
    scene: Scene,
    *,
    level: str | None = None,
    root: str | Path | None = None,
) -> Path:
    root_path = Path(root) if root is not None else project_root()
    history_dir = root_path / "output" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    quality = level or infer_quality(scene)
    timestamp = datetime.now()
    timestamp_file = timestamp.strftime("%Y%m%d_%H%M%S")
    history_filename = f"render_{timestamp_file}_{quality}.png"
    history_path = history_dir / history_filename
    counter = 2
    while history_path.exists():
        history_filename = f"render_{timestamp_file}_{quality}_{counter}.png"
        history_path = history_dir / history_filename
        counter += 1

    save_png(image, history_path, exposure=scene.render.exposure)
    _prepend_history_entry(
        history_dir / "history.js",
        {
            "filename": f"output/history/{history_filename}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "level": quality,
            "width": scene.render.width,
            "height": scene.render.height,
            "samples": scene.render.samples,
            "bounces": scene.render.max_bounces,
        },
    )
    return history_path


def _prepend_history_entry(js_path: Path, entry: dict) -> None:
    history_entries = _read_history_entries(js_path)
    history_entries.insert(0, entry)
    js_content = f"const PYTRACE_HISTORY = {json.dumps(history_entries, indent=2)};"
    js_path.write_text(js_content, encoding="utf-8")


def _read_history_entries(js_path: Path) -> list[dict]:
    if not js_path.exists():
        return []
    try:
        content = js_path.read_text(encoding="utf-8").strip()
        prefix = "const PYTRACE_HISTORY = "
        if prefix not in content:
            return []
        json_str = content.split(prefix, 1)[1]
        if json_str.endswith(";"):
            json_str = json_str[:-1]
        data = json.loads(json_str)
        return data if isinstance(data, list) else []
    except Exception:
        return []
