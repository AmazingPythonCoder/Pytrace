from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

from src.scene.camera import Camera
from src.scene.lights import DirectionalLight, Light
from src.scene.objects import Mesh, Plane, SceneObject, Sphere
from src.scene.scene import Scene

from . import layout


@dataclass
class _Row:
    item: object
    rect: Any
    toggle_rect: Any | None = None


class Outliner:
    def __init__(self) -> None:
        self.rows: list[_Row] = []

    def handle(self, event: Any, scene: Scene) -> bool:
        if pygame is None or event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        for row in self.rows:
            if row.toggle_rect is not None and row.toggle_rect.collidepoint(event.pos):
                if isinstance(row.item, SceneObject):
                    row.item.visible = not row.item.visible
                return True
            if row.rect.collidepoint(event.pos):
                scene.selected = row.item
                return True
        return False

    def draw(self, surface: Any, scene: Scene, font: Any, small_font: Any) -> None:
        if pygame is None:
            return
        rect = layout.left_panel_rect(surface.get_size())
        panel = pygame.Rect(rect.x, rect.y, rect.w, rect.h)
        pygame.draw.rect(surface, layout.PANEL, panel)
        pygame.draw.line(surface, layout.PANEL_EDGE, (rect.right - 1, rect.y), (rect.right - 1, rect.bottom))

        self.rows = []
        y = rect.y + layout.PANEL_PAD
        self._draw_text(surface, font, "Scene", (rect.x + 14, y), layout.TEXT)
        y += 34

        y = self._section(surface, small_font, "OBJECTS", rect.x + 14, y)
        for obj in scene.objects:
            y = self._row(surface, font, scene, obj, rect.x + 10, y)

        y += 8
        y = self._section(surface, small_font, "LIGHTS", rect.x + 14, y)
        for light in scene.lights:
            y = self._row(surface, font, scene, light, rect.x + 10, y)

        y += 8
        y = self._section(surface, small_font, "CAMERA", rect.x + 14, y)
        self._row(surface, font, scene, scene.camera, rect.x + 10, y)

    def _section(self, surface: Any, font: Any, label: str, x: int, y: int) -> int:
        self._draw_text(surface, font, label, (x, y), layout.TEXT_MUTED)
        return y + 24

    def _row(self, surface: Any, font: Any, scene: Scene, item: object, x: int, y: int) -> int:
        width = layout.LEFT_PANEL_WIDTH - 20
        rect = pygame.Rect(x, y, width, 28)
        selected = scene.selected is item
        if selected:
            pygame.draw.rect(surface, layout.ACCENT_SOFT, rect, border_radius=6)
        else:
            pygame.draw.rect(surface, (255, 255, 255, 12), rect, border_radius=6)

        toggle_rect = None
        label_x = x + 12
        if isinstance(item, SceneObject):
            toggle_rect = pygame.Rect(x + 8, y + 6, 16, 16)
            pygame.draw.rect(surface, layout.SUCCESS if item.visible else layout.FIELD_EDGE, toggle_rect, border_radius=3)
            label_x = x + 34

        self._draw_text(surface, font, f"{self._icon(item)} {self._name(item)}", (label_x, y + 5), layout.TEXT)
        self.rows.append(_Row(item=item, rect=rect, toggle_rect=toggle_rect))
        return y + 31

    def _name(self, item: object) -> str:
        if isinstance(item, Camera):
            return "Camera"
        return str(getattr(item, "name", "Item"))

    def _icon(self, item: object) -> str:
        if isinstance(item, Sphere):
            return "S"
        if isinstance(item, Plane):
            return "P"
        if isinstance(item, Mesh):
            return "M"
        if isinstance(item, DirectionalLight):
            return "D"
        if isinstance(item, Light):
            return "L"
        if isinstance(item, Camera):
            return "C"
        return "-"

    def _draw_text(self, surface: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int, int]) -> None:
        img = font.render(text, True, color)
        surface.blit(img, pos)
