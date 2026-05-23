from __future__ import annotations

from typing import Any

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

from . import layout


class Toolbar:
    def __init__(self) -> None:
        self.buttons: dict[str, Any] = {}

    def handle(self, event: Any) -> str | None:
        if pygame is None or event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None

        for action, rect in self.buttons.items():
            if rect.collidepoint(event.pos):
                return action
        return None

    def draw(self, surface: Any, font: Any, small_font: Any, use_gpu: bool, quality: str) -> None:
        if pygame is None:
            return
        rect = layout.top_bar_rect(surface.get_size())
        pygame.draw.rect(surface, layout.PANEL_DARK, pygame.Rect(rect.x, rect.y, rect.w, rect.h))
        pygame.draw.line(surface, layout.PANEL_EDGE, (0, rect.bottom - 1), (rect.right, rect.bottom - 1))

        self.buttons = {}
        x = 12
        x = self._button(surface, font, "add_menu", "Add", x, 8, 76)
        x = self._button(surface, font, "save", "Save", x, 8, 76)
        x = self._button(surface, font, "load", "Load", x, 8, 76)
        x = self._button(surface, font, "cycle_quality", f"Quality {quality.title()}", x, 8, 118)
        x = self._button(surface, font, "render", "Render F12", x, 8, 114)
        x = self._button(surface, font, "toggle_gpu", f"GPU {'On' if use_gpu else 'Off'}", x, 8, 96)

        title = small_font.render("PyTrace Editor", True, layout.TEXT_MUTED)
        surface.blit(title, (surface.get_width() - title.get_width() - 14, 14))

    def _button(self, surface: Any, font: Any, action: str, label: str, x: int, y: int, width: int) -> int:
        rect = pygame.Rect(x, y, width, 28)
        pygame.draw.rect(surface, (42, 47, 56, 245), rect, border_radius=6)
        pygame.draw.rect(surface, layout.FIELD_EDGE, rect, width=1, border_radius=6)
        text = font.render(label, True, layout.TEXT)
        surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
        self.buttons[action] = rect
        return x + width + 8
