from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]
    DOUBLEBUF = OPENGL = RESIZABLE = 0  # type: ignore[assignment]

from src.scene.lights import AreaLight, PointLight
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Plane, SceneObject, Sphere
from src.scene.serializer import load_scene, save_scene
from src.scene.scene import Scene

from . import layout
from .gizmos import TransformController
from .orbit_camera import OrbitCamera
from .outliner import Outliner
from .properties import Properties
from .render_window import RenderWindow
from .selection import pick
from .toolbar import Toolbar
from .viewport import Viewport, draw_overlay_surface


def run(scene: Scene | None = None, use_gpu: bool = True) -> int:
    if pygame is None:
        raise RuntimeError("pygame and PyOpenGL are required for the editor. Run: pip install -r requirements.txt")

    pygame.init()
    pygame.display.set_caption("PyTrace Editor")
    pygame.display.set_mode(layout.WINDOW_SIZE, DOUBLEBUF | OPENGL | RESIZABLE)
    clock = pygame.time.Clock()

    scene = scene or Scene.default()
    orbit = OrbitCamera()
    viewport = Viewport()
    viewport.setup()
    toolbar = Toolbar()
    outliner = Outliner()
    properties = Properties()
    transforms = TransformController()
    render_window: RenderWindow | None = None

    font = pygame.font.SysFont("Segoe UI", 16)
    small_font = pygame.font.SysFont("Segoe UI", 13)
    running = True

    while running:
        clock.tick(60)
        size = pygame.display.get_surface().get_size()
        vrect = layout.viewport_rect(size)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.VIDEORESIZE:
                pygame.display.set_mode((event.w, event.h), DOUBLEBUF | OPENGL | RESIZABLE)
                size = pygame.display.get_surface().get_size()
                vrect = layout.viewport_rect(size)
                continue

            if render_window is not None:
                result = render_window.handle(event)
                if result == "close" and not render_window.is_running():
                    render_window = None
                continue

            if properties.handle(event, scene):
                continue

            if event.type == pygame.KEYDOWN:
                if transforms.handle_key(event, scene):
                    continue
                mods = pygame.key.get_mods()
                if event.key == pygame.K_F12:
                    render_window = RenderWindow(scene, use_gpu=use_gpu)
                    render_window.start()
                    continue
                if event.key == pygame.K_a and mods & pygame.KMOD_SHIFT:
                    toolbar.add_menu_open = True
                    continue
                if event.key == pygame.K_s and mods & pygame.KMOD_CTRL:
                    _save_dialog(scene)
                    continue
                if event.key == pygame.K_o and mods & pygame.KMOD_CTRL:
                    loaded = _load_dialog()
                    if loaded is not None:
                        scene = loaded
                    continue
                if event.key in (pygame.K_DELETE, pygame.K_x):
                    _delete_selected(scene)
                    continue

            if transforms.handle_mouse_button(event):
                continue
            if transforms.handle_motion(event, orbit):
                continue

            action = toolbar.handle(event)
            if action:
                if action == "toggle_gpu":
                    use_gpu = not use_gpu
                elif action == "save":
                    _save_dialog(scene)
                elif action == "load":
                    loaded = _load_dialog()
                    if loaded is not None:
                        scene = loaded
                elif action == "render":
                    render_window = RenderWindow(scene, use_gpu=use_gpu)
                    render_window.start()
                elif action.startswith("add_"):
                    _add_item(scene, action, orbit.target)
                continue

            if outliner.handle(event, scene):
                continue

            if event.type == pygame.MOUSEWHEEL and vrect.contains(pygame.mouse.get_pos()):
                orbit.zoom(event.y)
                continue

            if event.type == pygame.MOUSEMOTION and vrect.contains(event.pos):
                buttons = pygame.mouse.get_pressed(3)
                if buttons[1]:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        orbit.pan(event.rel[0], event.rel[1])
                    else:
                        orbit.orbit(event.rel[0], event.rel[1])
                    continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and vrect.contains(event.pos):
                scene.selected = pick(event.pos[0], event.pos[1], vrect, orbit, scene)
                continue

        viewport.draw(scene, orbit, layout.viewport_rect(pygame.display.get_surface().get_size()), pygame.display.get_surface().get_size())
        overlay = pygame.Surface(pygame.display.get_surface().get_size(), pygame.SRCALPHA)
        toolbar.draw(overlay, font, small_font, use_gpu)
        outliner.draw(overlay, scene, font, small_font)
        properties.draw(overlay, scene, font, small_font)
        _draw_status(overlay, small_font, transforms.status())
        if render_window is not None:
            render_window.draw(overlay, font, small_font)
        draw_overlay_surface(overlay)
        pygame.display.flip()

    pygame.quit()
    return 0


def _add_item(scene: Scene, action: str, target: np.ndarray) -> None:
    position = np.asarray(target, dtype=np.float64).copy()
    if action == "add_sphere":
        position[1] += 0.5
        scene.add(
            Sphere(
                name=f"Sphere {len(scene.objects) + 1}",
                position=position,
                radius=0.5,
                material=DiffuseMaterial(color=np.array([0.7, 0.7, 0.78], dtype=np.float64)),
            )
        )
    elif action == "add_plane":
        scene.add(
            Plane(
                name=f"Plane {len(scene.objects) + 1}",
                position=position,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                material=DiffuseMaterial(color=np.array([0.55, 0.55, 0.55], dtype=np.float64)),
            )
        )
    elif action == "add_point_light":
        position[1] += 3.0
        scene.add(
            PointLight(
                name=f"Point Light {len(scene.lights) + 1}",
                position=position,
                color=np.ones(3, dtype=np.float64),
                intensity=120.0,
            )
        )
    elif action == "add_area_light":
        position[1] += 4.0
        scene.add(
            AreaLight(
                name=f"Area Light {len(scene.lights) + 1}",
                position=position,
                normal=np.array([0.0, -1.0, 0.0], dtype=np.float64),
                radius=1.0,
                color=np.ones(3, dtype=np.float64),
                intensity=80.0,
            )
        )


def _delete_selected(scene: Scene) -> None:
    selected = scene.selected
    if selected is None or selected is scene.camera:
        return
    scene.remove(selected)


def _save_dialog(scene: Scene) -> None:
    path = _file_dialog(save=True)
    if path:
        save_scene(scene, path)


def _load_dialog() -> Scene | None:
    path = _file_dialog(save=False)
    if not path:
        return None
    return load_scene(path)


def _file_dialog(save: bool) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        if save:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("PyTrace Scene", "*.json"), ("JSON", "*.json")],
                initialfile="scene.json",
            )
        else:
            path = filedialog.askopenfilename(
                filetypes=[("PyTrace Scene", "*.json"), ("JSON", "*.json")],
            )
        root.destroy()
        return str(path or "")
    except Exception:
        return ""


def _draw_status(surface: Any, font: Any, text: str) -> None:
    if pygame is None:
        return
    img = font.render(text, True, layout.TEXT_MUTED)
    surface.blit(img, (layout.LEFT_PANEL_WIDTH + 12, surface.get_height() - 24))

