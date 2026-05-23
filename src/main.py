"""PyTrace entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda

from src.raytracer.history import save_render_history
from src.raytracer.renderer import available_cpu_count, default_workers, render, save_png
from src.raytracer.render_context import build_render_context
from src.scene.serializer import load_scene
from src.scene.scene import Scene


def _parse_bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


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
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Load a serialized scene JSON path for the editor or headless render",
    )
    parser.add_argument("--width", type=int, default=None, help="Override render width")
    parser.add_argument("--height", type=int, default=None, help="Override render height")
    parser.add_argument("--samples", type=int, default=None, help="Samples per pixel (AA)")
    parser.add_argument("--bounces", type=int, default=None, help="Max ray bounces")
    parser.add_argument(
        "--level",
        choices=["low", "med", "high", "ultra"],
        default="med",
        help="Rendering level preset: low (800x450, 32 spp), med (1280x720, 128 spp), high (1920x1080, 384 spp), ultra (3840x2160, 768 spp) (default: med)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: available logical CPUs minus 1)",
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
    parser.add_argument(
        "--gpu",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Use CUDA GPU when available (default: true). Use --gpu false for CPU.",
    )
    args = parser.parse_args()

    if args.scene:
        scene_path = Path(args.scene)
        if not scene_path.is_absolute():
            scene_path = ROOT / scene_path
        scene = load_scene(scene_path)
    else:
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
    elif args.level == "ultra":
        # 4K UHD — ~8x ray budget vs high (4x pixels, 2x spp, deeper paths)
        scene.render.width = 3840
        scene.render.height = 2160
        scene.render.samples = 768
        scene.render.max_bounces = 16

    # Command line overrides
    if args.width is not None:
        scene.render.width = args.width
    if args.height is not None:
        scene.render.height = args.height
    if args.samples is not None:
        scene.render.samples = args.samples
    if args.bounces is not None:
        scene.render.max_bounces = args.bounces

    if not args.headless:
        try:
            from src.editor.app import run as run_editor
        except Exception as exc:
            print(f"Could not start editor: {exc}")
            print("Install editor dependencies with: pip install -r requirements.txt")
            return 1
        try:
            return run_editor(scene, use_gpu=args.gpu)
        except RuntimeError as exc:
            print(f"Could not start editor: {exc}")
            print("Install editor dependencies with: pip install -r requirements.txt")
            return 1

    use_gpu = args.gpu
    if use_gpu and not cuda.is_available():
        print("Warning: --gpu requested but CUDA is not available; using CPU.")
        use_gpu = False

    workers = args.workers
    parallel = not args.no_parallel
    if use_gpu:
        parallel = False
    elif parallel and workers is None:
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

    if use_gpu:
        mode = "GPU (CUDA)"
    else:
        mode = f"{workers} workers" if parallel and workers and workers > 1 else "1 thread"
    print(
        f"Rendering {scene.render.width}x{scene.render.height} "
        f"({scene.render.samples} spp, {mode}) ..."
    )
    if use_gpu:
        from src.raytracer.gpu.renderer import prepare_gpu_render, render_gpu

        ctx = build_render_context(scene)
        if ctx.gpu_supported:
            print("Uploading scene to GPU...")
            dev = prepare_gpu_render(ctx)
            start_time = time.time()
            image = render_gpu(ctx, progress_callback=progress, dev=dev)
        else:
            reason = ctx.gpu_fallback_reason or "scene feature is CPU-only"
            print(f"Scene uses CPU-only features ({reason}); using CPU renderer.")
            workers = workers or default_workers(reserve=1)
            print(f"CPU render using {workers} worker(s) from {available_cpu_count()} available core(s).")
            image = render(
                scene,
                progress_callback=progress,
                workers=workers,
                parallel=True,
                debug_workers=args.debug_workers,
                use_gpu=False,
            )
    else:
        if parallel:
            workers = workers or default_workers(reserve=1)
            print(f"CPU render using {workers} worker(s) from {available_cpu_count()} available core(s).")
        image = render(
            scene,
            progress_callback=progress,
            workers=workers,
            parallel=parallel,
            debug_workers=args.debug_workers,
            use_gpu=False,
        )
    print()
    save_png(image, out_path, exposure=scene.render.exposure)
    print(f"Wrote {out_path}")

    try:
        history_path = save_render_history(image, scene, level=args.level, root=ROOT)
        print(f"Saved history copy to {history_path}")
    except Exception as e:
        print(f"Warning: Could not save to history: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
