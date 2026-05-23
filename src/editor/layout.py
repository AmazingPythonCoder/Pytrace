from __future__ import annotations

from dataclasses import dataclass


WINDOW_SIZE = (1600, 900)
TOP_BAR_HEIGHT = 44
LEFT_PANEL_WIDTH = 260
RIGHT_PANEL_WIDTH = 340
PANEL_PAD = 12

BG = (18, 20, 24, 235)
PANEL = (28, 31, 37, 238)
PANEL_DARK = (20, 22, 27, 245)
PANEL_EDGE = (64, 70, 82, 255)
TEXT = (232, 236, 242, 255)
TEXT_MUTED = (154, 163, 176, 255)
ACCENT = (96, 165, 250, 255)
ACCENT_SOFT = (37, 99, 235, 120)
FIELD_BG = (15, 17, 21, 255)
FIELD_EDGE = (75, 85, 99, 255)
SUCCESS = (74, 222, 128, 255)
WARN = (251, 191, 36, 255)
DANGER = (248, 113, 113, 255)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.x <= pos[0] < self.right and self.y <= pos[1] < self.bottom


def viewport_rect(size: tuple[int, int]) -> Rect:
    width, height = size
    return Rect(
        LEFT_PANEL_WIDTH,
        TOP_BAR_HEIGHT,
        max(1, width - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH),
        max(1, height - TOP_BAR_HEIGHT),
    )


def left_panel_rect(size: tuple[int, int]) -> Rect:
    return Rect(0, TOP_BAR_HEIGHT, LEFT_PANEL_WIDTH, max(1, size[1] - TOP_BAR_HEIGHT))


def right_panel_rect(size: tuple[int, int]) -> Rect:
    return Rect(
        max(0, size[0] - RIGHT_PANEL_WIDTH),
        TOP_BAR_HEIGHT,
        RIGHT_PANEL_WIDTH,
        max(1, size[1] - TOP_BAR_HEIGHT),
    )


def top_bar_rect(size: tuple[int, int]) -> Rect:
    return Rect(0, 0, size[0], TOP_BAR_HEIGHT)

