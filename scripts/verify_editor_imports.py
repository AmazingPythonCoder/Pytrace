"""Import the editor package without opening a window."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.editor.app  # noqa: F401
import src.editor.gizmos  # noqa: F401
import src.editor.layout  # noqa: F401
import src.editor.orbit_camera  # noqa: F401
import src.editor.outliner  # noqa: F401
import src.editor.properties  # noqa: F401
import src.editor.render_window  # noqa: F401
import src.editor.selection  # noqa: F401
import src.editor.toolbar  # noqa: F401
import src.editor.viewport  # noqa: F401


def main() -> None:
    print("Editor imports OK")


if __name__ == "__main__":
    main()

