from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

import numpy as np

from src.raytracer.render_control import RenderCancelled
from src.raytracer.renderer import default_workers, render, save_png
from src.raytracer.tonemapping import tonemap_to_uint8
from src.scene.scene import Scene

from . import layout


class RenderWindow:
    def __init__(self, scene: Scene, use_gpu: bool = True) -> None:
        self.scene = Scene.from_dict(scene.to_dict())
        self.use_gpu_requested = use_gpu
        self.cancel_requested = False
        self.progress = (0, 1)
        self.status = "Preparing render"
        self.image: np.ndarray | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.preview_surface: Any | None = None
        self.buttons: dict[str, Any] = {}

    def start(self) -> None:
        self.thread = threading.Thread(target=self._worker, name="PyTraceRender", daemon=True)
        self.thread.start()

    def handle(self, event: Any) -> str | None:
        if pygame is None:
            return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.is_running():
                    self.cancel()
                    return None
                return "close"
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        for action, rect in self.buttons.items():
            if rect.collidepoint(event.pos):
                if action == "cancel":
                    self.cancel()
                    return None
                if action == "close":
                    return "close"
                if action == "save":
                    self._save_dialog()
                    return None
        return None

    def draw(self, surface: Any, font: Any, small_font: Any) -> None:
        if pygame is None:
            return
        width, height = surface.get_size()
        overlay = pygame.Rect(0, layout.TOP_BAR_HEIGHT, width, height - layout.TOP_BAR_HEIGHT)
        pygame.draw.rect(surface, (0, 0, 0, 150), overlay)

        card_w = min(980, width - 120)
        card_h = min(680, height - 120)
        card = pygame.Rect((width - card_w) // 2, (height - card_h) // 2 + 20, card_w, card_h)
        pygame.draw.rect(surface, (23, 27, 34, 250), card, border_radius=8)
        pygame.draw.rect(surface, layout.PANEL_EDGE, card, width=1, border_radius=8)

        title = "Rendering" if self.is_running() else "Render"
        self._draw_text(surface, font, title, (card.x + 22, card.y + 18), layout.TEXT)
        self._draw_text(surface, small_font, self.status, (card.x + 22, card.y + 45), layout.TEXT_MUTED)

        done, total = self.progress
        pct = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
        bar = pygame.Rect(card.x + 22, card.y + 76, card.w - 44, 16)
        pygame.draw.rect(surface, layout.FIELD_BG, bar, border_radius=8)
        fill = pygame.Rect(bar.x, bar.y, int(bar.w * pct), bar.h)
        pygame.draw.rect(surface, layout.ACCENT, fill, border_radius=8)
        self._draw_text(surface, small_font, f"{int(pct * 100)}%", (bar.right - 42, bar.y - 1), layout.TEXT)

        image_rect = pygame.Rect(card.x + 22, card.y + 110, card.w - 44, card.h - 180)
        pygame.draw.rect(surface, (8, 10, 13, 255), image_rect)
        if self.image is not None:
            preview = self._preview()
            target = self._fit_rect(preview.get_size(), image_rect)
            scaled = pygame.transform.smoothscale(preview, (target.w, target.h))
            surface.blit(scaled, target)
        elif self.error:
            self._draw_text(surface, font, self.error, (image_rect.x + 18, image_rect.y + 18), layout.DANGER)
        else:
            self._draw_text(surface, font, "Waiting for pixels...", (image_rect.x + 18, image_rect.y + 18), layout.TEXT_MUTED)

        self.buttons = {}
        button_y = card.bottom - 48
        if self.is_running():
            self._button(surface, font, "cancel", "Cancel", card.right - 126, button_y, 104)
        else:
            self._button(surface, font, "close", "Close", card.right - 126, button_y, 104)
            if self.image is not None:
                self._button(surface, font, "save", "Save PNG", card.right - 244, button_y, 108)

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.is_running():
            self.status = "Cancelling after current work chunk..."

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _worker(self) -> None:
        try:
            use_gpu = self._cuda_available() and self.use_gpu_requested
            mode = "GPU" if use_gpu else "CPU"
            self.status = f"Rendering on {mode}"

            def progress(done: int, total: int) -> None:
                self.progress = (done, max(1, total))

            self.image = render(
                self.scene,
                progress_callback=progress,
                workers=default_workers(reserve=1),
                parallel=not use_gpu,
                use_gpu=use_gpu,
                cancel_callback=lambda: self.cancel_requested,
            )
            self.progress = (1, 1)
            self.status = "Render complete"
        except RenderCancelled:
            self.status = "Render cancelled"
        except Exception as exc:  # pragma: no cover - keeps editor alive on render errors
            self.error = str(exc)
            self.status = "Render failed"

    def _cuda_available(self) -> bool:
        try:
            from src.raytracer.gpu import cuda_init  # noqa: F401
            from numba import cuda

            return bool(cuda.is_available())
        except Exception:
            return False

    def _preview(self) -> Any:
        if pygame is None:
            raise RuntimeError("pygame is required")
        if self.preview_surface is None and self.image is not None:
            rgb = tonemap_to_uint8(self.image, self.scene.render.exposure)
            self.preview_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        return self.preview_surface

    def _fit_rect(self, src_size: tuple[int, int], dst: Any) -> Any:
        src_w, src_h = src_size
        scale = min(dst.w / src_w, dst.h / src_h)
        width = max(1, int(src_w * scale))
        height = max(1, int(src_h * scale))
        return pygame.Rect(dst.centerx - width // 2, dst.centery - height // 2, width, height)

    def _button(self, surface: Any, font: Any, action: str, label: str, x: int, y: int, width: int) -> None:
        rect = pygame.Rect(x, y, width, 30)
        pygame.draw.rect(surface, (42, 47, 56, 245), rect, border_radius=6)
        pygame.draw.rect(surface, layout.FIELD_EDGE, rect, width=1, border_radius=6)
        self._draw_text(surface, font, label, (rect.centerx - font.size(label)[0] // 2, rect.y + 6), layout.TEXT)
        self.buttons[action] = rect

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

    def _draw_text(self, surface: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int, int]) -> None:
        img = font.render(text, True, color)
        surface.blit(img, pos)

