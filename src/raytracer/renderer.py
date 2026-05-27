from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.raytracer.render_context import RenderContext, build_render_context
from src.raytracer.render_control import RenderCancelled
from src.raytracer.shading import render_tile
from src.raytracer.tonemapping import tonemap_to_uint8
from src.scene.scene import Scene

_TILE_SIZE = 64


def available_cpu_count() -> int:
    """Return CPUs available to this process, falling back to the host count."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return max(1, int(count))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def default_workers(reserve: int = 1) -> int:
    """Use available logical CPUs minus ``reserve``."""
    total = available_cpu_count()
    return max(1, total - reserve)


def _render_tile_args(ctx: RenderContext, x0: int, y0: int, x1: int, y1: int) -> tuple:
    cf = ctx.camera_frame
    return (
        x0,
        y0,
        x1,
        y1,
        cf.origin[0],
        cf.origin[1],
        cf.origin[2],
        cf.lower_left[0],
        cf.lower_left[1],
        cf.lower_left[2],
        cf.horizontal[0],
        cf.horizontal[1],
        cf.horizontal[2],
        cf.vertical[0],
        cf.vertical[1],
        cf.vertical[2],
        cf.lens_u[0],
        cf.lens_u[1],
        cf.lens_u[2],
        cf.lens_v[0],
        cf.lens_v[1],
        cf.lens_v[2],
        cf.aperture,
        cf.inv_width,
        cf.inv_height,
        ctx.samples,
        ctx.max_bounces,
        ctx.area_light_samples,
        ctx.spheres,
        ctx.planes,
        ctx.triangles,
        ctx.bvh_nodes,
        ctx.bvh_prims,
        ctx.materials,
        ctx.lights,
        ctx.background,
        ctx.environment,
        ctx.render_mode,
        ctx.background_mode,
    )


def _render_tile_worker(
    args: tuple[RenderContext, tuple[int, int, int, int], bool],
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    ctx, (x0, y0, x1, y1), debug = args
    if debug:
        print(f"Worker PID: {os.getpid()}", flush=True)
    tile = render_tile(*_render_tile_args(ctx, x0, y0, x1, y1))
    return (x0, y0, x1, y1), tile


def _tile_bounds(width: int, height: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    tiles: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            tiles.append((x0, y0, x1, y1))
    return tiles


def _render_sequential(ctx: RenderContext, progress_callback=None, cancel_callback=None, tile_callback=None) -> np.ndarray:
    tiles = _tile_bounds(ctx.width, ctx.height, _TILE_SIZE)
    image = np.zeros((ctx.height, ctx.width, 3), dtype=np.float64)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        if cancel_callback and cancel_callback():
            raise RenderCancelled()
        tile = render_tile(*_render_tile_args(ctx, x0, y0, x1, y1))
        image[y0:y1, x0:x1] = tile
        if tile_callback:
            tile_callback((x0, y0, x1, y1), tile)
        if progress_callback:
            progress_callback(i + 1, len(tiles))
    if cancel_callback and cancel_callback():
        raise RenderCancelled()
    return image


def _render_parallel(
    ctx: RenderContext,
    workers: int,
    progress_callback=None,
    debug_workers: bool = False,
    cancel_callback=None,
    tile_callback=None,
) -> np.ndarray:
    tiles = _tile_bounds(ctx.width, ctx.height, _TILE_SIZE)
    image = np.zeros((ctx.height, ctx.width, 3), dtype=np.float64)
    tasks = [(ctx, tile, debug_workers) for tile in tiles]

    if debug_workers:
        print(f"Main PID: {os.getpid()}, workers={workers}, tiles={len(tiles)}", flush=True)

    pool = mp.Pool(processes=workers)
    finished = False
    try:
        for i, ((x0, y0, x1, y1), tile) in enumerate(pool.imap_unordered(_render_tile_worker, tasks)):
            if cancel_callback and cancel_callback():
                raise RenderCancelled()
            if debug_workers and i < 3:
                print(f"  Tile {i} done (worker returned)", flush=True)
            image[y0:y1, x0:x1] = tile
            if tile_callback:
                tile_callback((x0, y0, x1, y1), tile)
            if progress_callback:
                progress_callback(i + 1, len(tiles))
        finished = True
    finally:
        if finished:
            pool.close()
        else:
            pool.terminate()
        pool.join()

    return image


def render(
    scene: Scene,
    progress_callback=None,
    workers: int | None = None,
    parallel: bool = True,
    debug_workers: bool = False,
    use_gpu: bool = True,
    cancel_callback=None,
    tile_callback=None,
    preview_callback=None,
) -> np.ndarray:
    """Render scene to HDR float buffer. GPU (CUDA) when use_gpu else CPU tiles."""
    ctx = build_render_context(scene)

    if use_gpu and ctx.gpu_supported:
        from src.raytracer.gpu.renderer import render_gpu

        return render_gpu(
            ctx,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            preview_callback=preview_callback,
        )

    if workers is None:
        workers = default_workers(reserve=1)

    if parallel and workers > 1:
        return _render_parallel(ctx, workers, progress_callback, debug_workers, cancel_callback, tile_callback)
    return _render_sequential(ctx, progress_callback, cancel_callback, tile_callback)


def save_png(image: np.ndarray, path: str | Path, exposure: float = 1.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = tonemap_to_uint8(image, exposure)
    Image.fromarray(rgb, mode="RGB").save(path)
