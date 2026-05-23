"""GPU render orchestration: kernel launch and host image readback."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Any, cast

import numpy as np
from src.raytracer.gpu import cuda_init  # noqa: F401
from numba import cuda, float32
from numba.core.errors import NumbaPerformanceWarning

from src.raytracer.gpu.cuda_trace import trace_pixel_cuda
from src.raytracer.gpu.device_context import DeviceRenderContext, to_device
from src.raytracer.render_context import RenderContext

_CLEAR_BLOCK = 16
_RENDER_BLOCK_X = 8
_RENDER_BLOCK_Y = 8
_RENDER_BLOCK_SAMPLES = 4
_PROGRESS_STEPS = 20
_F0 = float32(0.0)
_cuda: Any = cuda


def _suppress_low_occupancy_warning() -> None:
    warnings.simplefilter("ignore", NumbaPerformanceWarning)


_suppress_low_occupancy_warning()


@cuda.jit(fastmath=True, cache=True)
def _clear_kernel(image, width, height):
    x, y = cuda.grid(2)  # pyright: ignore[reportAttributeAccessIssue]
    if x >= width or y >= height:
        return
    image[y, x, 0] = _F0
    image[y, x, 1] = _F0
    image[y, x, 2] = _F0


@cuda.jit(fastmath=True, cache=True)
def _render_kernel(
    image,
    width,
    height,
    samples,
    sample_start,
    sample_count,
    cam_ox,
    cam_oy,
    cam_oz,
    cam_llx,
    cam_lly,
    cam_llz,
    cam_hx,
    cam_hy,
    cam_hz,
    cam_vx,
    cam_vy,
    cam_vz,
    inv_width,
    inv_height,
    max_bounces,
    area_light_samples,
    spheres,
    planes,
    bvh_nodes,
    bvh_prims,
    materials,
    lights,
    background,
):
    x, y, sample_offset = cuda.grid(3)  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]
    if x >= width or y >= height or sample_offset >= sample_count:
        return
    sample_i = sample_start + sample_offset
    tr, tg, tb = trace_pixel_cuda(
        float32(x),
        float32(y),
        sample_i,
        cam_ox,
        cam_oy,
        cam_oz,
        cam_llx,
        cam_lly,
        cam_llz,
        cam_hx,
        cam_hy,
        cam_hz,
        cam_vx,
        cam_vy,
        cam_vz,
        inv_width,
        inv_height,
        max_bounces,
        area_light_samples,
        spheres,
        planes,
        bvh_nodes,
        bvh_prims,
        materials,
        lights,
        background,
    )
    cuda.atomic.add(image, (y, x, 0), tr)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
    cuda.atomic.add(image, (y, x, 1), tg)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
    cuda.atomic.add(image, (y, x, 2), tb)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]


def _clear_grid(dev: DeviceRenderContext) -> tuple[tuple[int, int], tuple[int, int]]:
    threads_per_block = (_CLEAR_BLOCK, _CLEAR_BLOCK)
    blocks_per_grid_x = int(math.ceil(dev.width / _CLEAR_BLOCK))
    blocks_per_grid_y = int(math.ceil(dev.height / _CLEAR_BLOCK))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return blocks_per_grid, threads_per_block


def _render_grid(
    dev: DeviceRenderContext,
    sample_count: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    threads_per_block = (_RENDER_BLOCK_X, _RENDER_BLOCK_Y, _RENDER_BLOCK_SAMPLES)
    blocks_per_grid = (
        int(math.ceil(dev.width / _RENDER_BLOCK_X)),
        int(math.ceil(dev.height / _RENDER_BLOCK_Y)),
        int(math.ceil(sample_count / _RENDER_BLOCK_SAMPLES)),
    )
    return blocks_per_grid, threads_per_block


def _clear(dev: DeviceRenderContext, image) -> None:
    blocks_per_grid, threads_per_block = _clear_grid(dev)
    cast(Any, _clear_kernel)[blocks_per_grid, threads_per_block](image, dev.width, dev.height)


def _launch(dev: DeviceRenderContext, image, sample_start: int, sample_count: int) -> None:
    blocks_per_grid, threads_per_block = _render_grid(dev, sample_count)
    cast(Any, _render_kernel)[blocks_per_grid, threads_per_block](
        image,
        dev.width,
        dev.height,
        dev.samples,
        sample_start,
        sample_count,
        dev.cam_ox,
        dev.cam_oy,
        dev.cam_oz,
        dev.cam_llx,
        dev.cam_lly,
        dev.cam_llz,
        dev.cam_hx,
        dev.cam_hy,
        dev.cam_hz,
        dev.cam_vx,
        dev.cam_vy,
        dev.cam_vz,
        dev.inv_width,
        dev.inv_height,
        dev.max_bounces,
        dev.area_light_samples,
        dev.spheres,
        dev.planes,
        dev.bvh_nodes,
        dev.bvh_prims,
        dev.materials,
        dev.lights,
        dev.background,
    )


def prepare_gpu_render(ctx: RenderContext) -> DeviceRenderContext:
    """Upload scene data and compile CUDA kernels before the timed render phase."""
    dev = to_device(ctx)
    warmup_dev = replace(
        dev,
        width=1,
        height=1,
        samples=1,
        max_bounces=1,
        area_light_samples=1,
        inv_width=np.float32(1.0),
        inv_height=np.float32(1.0),
    )
    image = _cuda.device_array((1, 1, 3), dtype=np.float32)
    _clear(warmup_dev, image)
    _launch(warmup_dev, image, 0, 1)
    _cuda.synchronize()
    return dev


def render_gpu(
    ctx: RenderContext,
    progress_callback=None,
    dev: DeviceRenderContext | None = None,
) -> np.ndarray:
    """Render on CUDA; returns host float64 HDR image (H, W, 3)."""
    if not _cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --gpu false for CPU rendering.")

    if dev is None:
        dev = to_device(ctx)
    image = _cuda.device_array((dev.height, dev.width, 3), dtype=np.float32)
    chunk_size = (
        max(1, int(math.ceil(dev.samples / _PROGRESS_STEPS)))
        if progress_callback
        else dev.samples
    )
    total_chunks = int(math.ceil(dev.samples / chunk_size))

    if progress_callback:
        progress_callback(0, total_chunks)

    try:
        _clear(dev, image)
        chunks_done = 0
        for sample_start in range(0, dev.samples, chunk_size):
            sample_count = min(chunk_size, dev.samples - sample_start)
            _launch(dev, image, sample_start, sample_count)
            _cuda.synchronize()
            chunks_done += 1
            if progress_callback:
                progress_callback(chunks_done, total_chunks)
    except _cuda.CudaAPIError as e:
        raise RuntimeError(
            f"GPU kernel failed: {e}. Try --gpu false or update CUDA/Numba."
        ) from e

    return image.copy_to_host() / dev.samples
