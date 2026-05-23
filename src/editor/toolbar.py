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
        self.menu_items: dict[str, Any] = {}
        self.add_menu_open = False

    def handle(self, event: Any) -> str | None:
        if pygame is None or event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None

        if self.add_menu_open:
            for action, rect in self.menu_items.items():
                if rect.collidepoint(event.pos):
                    self.add_menu_open = False
                    return action
            self.add_menu_open = False

        for action, rect in self.buttons.items():
            if rect.collidepoint(event.pos):
                if action == "add_menu":
                    self.add_menu_open = not self.add_menu_open
                    return None
                return action
        return None

    def draw(self, surface: Any, font: Any, small_font: Any, use_gpu: bool) -> None:
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
        x = self._button(surface, font, "render", "Render F12", x, 8, 114)
        x = self._button(surface, font, "toggle_gpu", f"GPU {'On' if use_gpu else 'Off'}", x, 8, 96)

        title = small_font.render("PyTrace Editor", True, layout.TEXT_MUTED)
        surface.blit(title, (surface.get_width() - title.get_width() - 14, 14))

        if self.add_menu_open:
            self._draw_add_menu(surface, font, x=12, y=layout.TOP_BAR_HEIGHT + 6)

    def _button(self, surface: Any, font: Any, action: str, label: str, x: int, y: int, width: int) -> int:
        rect = pygame.Rect(x, y, width, 28)
        pygame.draw.rect(surface, (42, 47, 56, 245), rect, border_radius=6)
        pygame.draw.rect(surface, layout.FIELD_EDGE, rect, width=1, border_radius=6)
        text = font.render(label, True, layout.TEXT)
        surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
        self.buttons[action] = rect
        return x + width + 8

    def _draw_add_menu(self, surface: Any, font: Any, x: int, y: int) -> None:
        entries = [
            ("add_sphere", "Sphere"),
            ("add_plane", "Plane"),
            ("add_point_light", "Point Light"),
            ("add_area_light", "Area Light"),
        ]
        width = 170
        height = 8 + len(entries) * 30
        bg = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, layout.PANEL_DARK, bg, border_radius=8)
        pygame.draw.rect(surface, layout.PANEL_EDGE, bg, width=1, border_radius=8)

        self.menu_items = {}
        row_y = y + 5
        for action, label in entries:
            rect = pygame.Rect(x + 6, row_y, width - 12, 26)
            pygame.draw.rect(surface, (255, 255, 255, 10), rect, border_radius=5)
            text = font.render(label, True, layout.TEXT)
            surface.blit(text, (rect.x + 10, rect.y + 5))
            self.menu_items[action] = rect
            row_y += 30

