"""Verify default CPU worker selection keeps one available core free."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raytracer.renderer import available_cpu_count, default_workers


def main() -> None:
    available = available_cpu_count()
    assert available >= 1
    assert default_workers(reserve=1) == max(1, available - 1)
    assert default_workers(reserve=0) == available
    assert default_workers(reserve=available + 10) == 1
    print(f"CPU workers OK: {default_workers(reserve=1)} worker(s) for {available} available core(s)")


if __name__ == "__main__":
    main()
