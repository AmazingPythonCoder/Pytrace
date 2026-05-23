"""Import the editor package without opening a window."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.editor.app  # noqa: F401
import src.editor.add_actions  # noqa: F401
import src.editor.gl_texture  # noqa: F401
import src.editor.gizmos  # noqa: F401
import src.editor.imgui_render_window  # noqa: F401
import src.editor.layout  # noqa: F401
import src.editor.orbit_camera  # noqa: F401
import src.editor.preview_gl  # noqa: F401
import src.editor.selection  # noqa: F401

assert "pygame" not in sys.modules


def main() -> None:
    print("Editor imports OK: ImGui runtime modules import without Pygame")


if __name__ == "__main__":
    main()
