from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from imgui_bundle import ImVec2, imgui

from src.raytracer.history import save_render_history
from src.raytracer.render_control import RenderCancelled
from src.raytracer.render_context import build_render_context
from src.raytracer.renderer import available_cpu_count, default_workers, render, save_png
from src.raytracer.tonemapping import tonemap_to_uint8
from src.scene.scene import Scene

from .gl_texture import ImageTexture


class ImguiRenderWindow:
    def __init__(self, scene: Scene, use_gpu: bool = True) -> None:
        self.scene = Scene.from_dict(scene.to_dict())
        self.use_gpu_requested = use_gpu
        self.cancel_requested = False
        self.progress = (0, 1)
        self.status = "Preparing render"
        self.error: str | None = None
        self.image: np.ndarray | None = None
        self.thread: threading.Thread | None = None
        self.texture = ImageTexture()
        self.texture_dirty = False
        self.popup_opened = False
        self._lock = threading.Lock()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._worker, name="PyTraceRender", daemon=True)
        self.thread.start()

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.is_running():
            self.status = "Cancelling after current work chunk..."

    def draw(self) -> str | None:
        if not self.popup_opened:
            imgui.open_popup("Render")
            self.popup_opened = True

        imgui.set_next_window_size(ImVec2(980, 700), imgui.Cond_.appearing)
        visible, _ = imgui.begin_popup_modal("Render", None, imgui.WindowFlags_.no_saved_settings)
        if not visible:
            return None

        if imgui.is_key_pressed(imgui.Key.escape):
            if self.is_running():
                self.cancel()
            else:
                imgui.close_current_popup()
                imgui.end_popup()
                return "close"

        done, total = self.progress
        pct = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
        imgui.text("Rendering" if self.is_running() else "Render")
        imgui.text_disabled(self.status)
        imgui.progress_bar(pct, ImVec2(-1, 0), f"{int(pct * 100)}%")
        imgui.separator()

        if self.error:
            imgui.text_colored((1.0, 0.35, 0.35, 1.0), self.error)
        elif self.image is not None:
            texture_id, texture_size = self._preview_texture()
            avail = imgui.get_content_region_avail()
            max_w = max(1.0, avail.x)
            max_h = max(1.0, avail.y - 52.0)
            scale = min(max_w / texture_size[0], max_h / texture_size[1])
            imgui.image(imgui.ImTextureRef(texture_id), ImVec2(texture_size[0] * scale, texture_size[1] * scale))
        else:
            imgui.text_disabled("Waiting for pixels...")

        imgui.separator()
        action: str | None = None
        if self.is_running():
            if imgui.button("Cancel"):
                self.cancel()
        else:
            if self.image is not None:
                if imgui.button("Save PNG"):
                    self._save_dialog()
                imgui.same_line()
            if imgui.button("Close"):
                imgui.close_current_popup()
                action = "close"

        imgui.end_popup()
        return action

    def cleanup(self) -> None:
        self.texture.cleanup()

    def _preview_texture(self) -> tuple[int, tuple[int, int]]:
        with self._lock:
            image = None if self.image is None else self.image.copy()
            dirty = self.texture_dirty
            self.texture_dirty = False
        if image is None:
            return self.texture.texture, self.texture.size
        if dirty or not self.texture.texture:
            rgb = tonemap_to_uint8(image, self.scene.render.exposure)
            self.texture.update_rgb(rgb)
        return self.texture.texture, self.texture.size

    def _worker(self) -> None:
        try:
            ctx = build_render_context(self.scene)
            cuda_available = self._cuda_available()
            use_gpu = bool(self.use_gpu_requested and cuda_available and ctx.gpu_supported)
            cpu_workers = default_workers(reserve=1)
            cpu_count = available_cpu_count()
            if self.use_gpu_requested and not cuda_available:
                self.status = f"CUDA unavailable; CPU {cpu_workers}/{cpu_count} workers"
            elif self.use_gpu_requested and not ctx.gpu_supported:
                reason = ctx.gpu_fallback_reason or "scene feature is CPU-only"
                self.status = f"CPU fallback {cpu_workers}/{cpu_count}: {reason}"
            elif use_gpu:
                self.status = "Rendering on GPU (CUDA)"
            else:
                self.status = f"Rendering on CPU {cpu_workers}/{cpu_count} workers"

            def progress(done: int, total: int) -> None:
                self.progress = (done, max(1, total))

            def tile_callback(tile_bounds: tuple[int, int, int, int], tile: np.ndarray) -> None:
                x0, y0, x1, y1 = tile_bounds
                with self._lock:
                    if self.image is None:
                        self.image = np.zeros(
                            (self.scene.render.height, self.scene.render.width, 3),
                            dtype=np.float64,
                        )
                    self.image[y0:y1, x0:x1] = tile
                    self.texture_dirty = True

            def preview_callback(image: np.ndarray) -> None:
                with self._lock:
                    self.image = image.copy()
                    self.texture_dirty = True

            image = render(
                self.scene,
                progress_callback=progress,
                workers=cpu_workers,
                parallel=True,
                use_gpu=use_gpu,
                cancel_callback=lambda: self.cancel_requested,
                tile_callback=tile_callback,
                preview_callback=preview_callback,
            )
            with self._lock:
                self.image = image
                self.texture_dirty = True
            history_path = save_render_history(image, self.scene)
            self.progress = (1, 1)
            self.status = f"Render complete; added to gallery ({history_path.name})"
        except RenderCancelled:
            self.status = "Render cancelled"
        except Exception as exc:  # pragma: no cover - keeps the editor alive
            self.error = str(exc)
            self.status = "Render failed"

    def _cuda_available(self) -> bool:
        try:
            from src.raytracer.gpu import cuda_init  # noqa: F401
            from numba import cuda

            return bool(cuda.is_available())
        except Exception:
            return False

    def _save_dialog(self) -> None:
        if self.image is None:
            return
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                initialfile="render.png",
            )
            root.destroy()
        except Exception:
            path = ""
        if path:
            save_png(self.image, Path(path), exposure=self.scene.render.exposure)
            self.status = f"Saved {path}"
