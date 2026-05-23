from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

from . import layout


@dataclass(frozen=True)
class AddEntry:
    action: str
    title: str
    detail: str
    group: str


ENTRIES = (
    AddEntry("add_sphere", "Sphere", "Round primitive with radius control", "Geometry"),
    AddEntry("add_plane", "Plane", "Infinite render plane with preview tile", "Geometry"),
    AddEntry("add_cube", "Cube", "Triangle mesh cube for hard-surface blocking", "Geometry"),
    AddEntry("import_obj", "Import OBJ", "Load a Wavefront mesh from disk", "Geometry"),
    AddEntry("add_point_light", "Point Light", "Small positional light", "Lights"),
    AddEntry("add_directional_light", "Directional Light", "Sun-style light with direction", "Lights"),
    AddEntry("add_area_light", "Area Light", "Disk light for soft shadows", "Lights"),
)


class AddMenu:
    def __init__(self) -> None:
        self.open = False
        self.cards: dict[str, Any] = {}

    def show(self) -> None:
        self.open = True

    def hide(self) -> None:
        self.open = False

    def handle(self, event: Any) -> str | None:
        if pygame is None or not self.open:
            return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.hide()
                return "handled"
            return "handled"
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return "handled"

        for action, rect in self.cards.items():
            if rect.collidepoint(event.pos):
                self.hide()
                return action
        self.hide()
        return "handled"

    def draw(self, surface: Any, font: Any, small_font: Any) -> None:
        if pygame is None or not self.open:
            return
        width, height = surface.get_size()
        pygame.draw.rect(surface, (0, 0, 0, 150), pygame.Rect(0, layout.TOP_BAR_HEIGHT, width, height - layout.TOP_BAR_HEIGHT))

        card_w = min(920, width - 96)
        card_h = min(580, height - 112)
        card = pygame.Rect((width - card_w) // 2, (height - card_h) // 2 + 24, card_w, card_h)
        pygame.draw.rect(surface, (23, 27, 34, 250), card, border_radius=8)
        pygame.draw.rect(surface, layout.PANEL_EDGE, card, width=1, border_radius=8)

        title = font.render("Add To Scene", True, layout.TEXT)
        surface.blit(title, (card.x + 24, card.y + 20))
        hint = small_font.render("Esc closes", True, layout.TEXT_MUTED)
        surface.blit(hint, (card.right - hint.get_width() - 24, card.y + 23))

        self.cards = {}
        x = card.x + 24
        y = card.y + 66
        content_w = card.w - 48
        group_gap = 24
        for group in ("Geometry", "Lights"):
            label = small_font.render(group.upper(), True, layout.TEXT_MUTED)
            surface.blit(label, (x, y))
            y += 25

            group_entries = [entry for entry in ENTRIES if entry.group == group]
            columns = 2 if content_w < 760 else 3
            gap = 12
            item_w = (content_w - gap * (columns - 1)) // columns
            item_h = 82
            for index, entry in enumerate(group_entries):
                col = index % columns
                row = index // columns
                rect = pygame.Rect(
                    x + col * (item_w + gap),
                    y + row * (item_h + gap),
                    item_w,
                    item_h,
                )
                self._draw_card(surface, font, small_font, rect, entry)
                self.cards[entry.action] = rect
            rows = (len(group_entries) + columns - 1) // columns
            y += rows * item_h + max(0, rows - 1) * gap + group_gap

    def _draw_card(self, surface: Any, font: Any, small_font: Any, rect: Any, entry: AddEntry) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hovered = rect.collidepoint(mouse_pos)
        bg = (42, 47, 56, 245) if hovered else (255, 255, 255, 12)
        edge = layout.ACCENT if hovered else layout.FIELD_EDGE
        pygame.draw.rect(surface, bg, rect, border_radius=7)
        pygame.draw.rect(surface, edge, rect, width=1, border_radius=7)

        title = font.render(entry.title, True, layout.TEXT)
        surface.blit(title, (rect.x + 14, rect.y + 13))
        detail = small_font.render(entry.detail, True, layout.TEXT_MUTED)
        surface.blit(detail, (rect.x + 14, rect.y + 43))
