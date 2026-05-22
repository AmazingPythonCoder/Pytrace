"""PyTrace entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raytracer.renderer import default_workers, render, save_png
from src.scene.scene import Scene


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTrace — scene editor and raytracer")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render the default scene to a PNG (no editor window)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/render.png",
        help="Output PNG path (relative to project root)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override render width")
    parser.add_argument("--height", type=int, default=None, help="Override render height")
    parser.add_argument("--samples", type=int, default=None, help="Samples per pixel (AA)")
    parser.add_argument("--bounces", type=int, default=None, help="Max ray bounces")
    parser.add_argument(
        "--level",
        choices=["low", "med", "high"],
        default="med",
        help="Rendering level preset: low (800x450, 32 spp), med (1280x720, 128 spp), high (1920x1080, 384 spp) (default: high)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: CPU count minus 1)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Force single-threaded render (for debugging)",
    )
    parser.add_argument(
        "--debug-workers",
        action="store_true",
        help="Print main PID and tile completion hints",
    )
    args = parser.parse_args()

    if not args.headless:
        print("Editor not implemented yet. Defaulting to --headless render.")

    scene = Scene.default()

    # Apply level preset settings
    if args.level == "low":
        # The new "low" is the old "high" (~2 seconds)
        scene.render.width = 800
        scene.render.height = 450
        scene.render.samples = 32
        scene.render.max_bounces = 8
    elif args.level == "med":
        # Target: ~30 seconds (15x more rays than low)
        scene.render.width = 1280
        scene.render.height = 720
        scene.render.samples = 128
        scene.render.max_bounces = 10
    elif args.level == "high":
        # Target: ~2 minutes (60x more rays than low)
        scene.render.width = 1920
        scene.render.height = 1080
        scene.render.samples = 384
        scene.render.max_bounces = 12

    # Command line overrides
    if args.width is not None:
        scene.render.width = args.width
    if args.height is not None:
        scene.render.height = args.height
    if args.samples is not None:
        scene.render.samples = args.samples
    if args.bounces is not None:
        scene.render.max_bounces = args.bounces

    workers = args.workers
    parallel = not args.no_parallel
    if parallel and workers is None:
        workers = default_workers(reserve=1)

    out_path = ROOT / args.output

    import time
    start_time = time.time()

    def progress(done: int, total: int) -> None:
        pct = done / total
        elapsed = time.time() - start_time
        eta = (elapsed / pct - elapsed) if pct > 0 else 0

        def fmt_time(seconds: float) -> str:
            s = int(seconds)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if h > 0: return f"{h}:{m:02d}:{s:02d}"
            if m > 0: return f"{m}:{s:02d}"
            return f"{s}s"

        bar_len = 20
        filled = int(bar_len * pct)
        if filled > 0:
            bar = "=" * (filled - 1) + ">" + " " * (bar_len - filled)
        else:
            bar = " " * bar_len

        print(f"\r[{bar}] {int(pct * 100)}% eta {fmt_time(eta)} elapsed {fmt_time(elapsed)}     ", end="", flush=True)

    mode = f"{workers} workers" if parallel and workers and workers > 1 else "1 thread"
    print(
        f"Rendering {scene.render.width}x{scene.render.height} "
        f"({scene.render.samples} spp, {mode}) ..."
    )
    image = render(
        scene,
        progress_callback=progress,
        workers=workers,
        parallel=parallel,
        debug_workers=args.debug_workers,
    )
    print()
    save_png(image, out_path, exposure=scene.render.exposure)
    print(f"Wrote {out_path}")

    # Save to rendering history
    try:
        import json
        from datetime import datetime
        history_dir = ROOT / "output" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_filename = f"render_{timestamp_str}_{args.level}.png"
        history_path = history_dir / history_filename

        save_png(image, history_path, exposure=scene.render.exposure)
        print(f"Saved history copy to {history_path}")

        # Update history.js database
        js_path = history_dir / "history.js"
        history_entries = []
        if js_path.exists():
            try:
                content = js_path.read_text(encoding="utf-8").strip()
                if "const PYTRACE_HISTORY = " in content:
                    json_str = content.split("const PYTRACE_HISTORY = ", 1)[1]
                    if json_str.endswith(";"):
                        json_str = json_str[:-1]
                    history_entries = json.loads(json_str)
            except Exception:
                history_entries = []

        new_entry = {
            "filename": f"output/history/{history_filename}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": args.level,
            "width": scene.render.width,
            "height": scene.render.height,
            "samples": scene.render.samples,
            "bounces": scene.render.max_bounces
        }
        history_entries.insert(0, new_entry)  # Newest first

        js_content = f"const PYTRACE_HISTORY = {json.dumps(history_entries, indent=2)};"
        js_path.write_text(js_content, encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not save to history: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
