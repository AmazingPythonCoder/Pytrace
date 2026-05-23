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

from src.scene.lights import AreaLight, DirectionalLight, PointLight
from src.scene.materials import DiffuseMaterial
from src.scene.objects import Mesh, Plane, SceneObject, Sphere
from src.scene.serializer import load_scene, save_scene
from src.scene.scene import Scene

from . import layout
from .add_menu import AddMenu
from .gizmos import TransformController
from .orbit_camera import OrbitCamera
from .outliner import Outliner
from .properties import Properties
from .render_window import RenderWindow
from .selection import pick
from .toolbar import Toolbar
from .viewport import Viewport, draw_overlay_surface


QUALITY_ORDER = ("low", "med", "high", "ultra")
QUALITY_PRESETS = {
    "low": (800, 450, 32, 8),
    "med": (1280, 720, 128, 10),
    "high": (1920, 1080, 384, 12),
    "ultra": (3840, 2160, 768, 16),
}


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
    add_menu = AddMenu()
    outliner = Outliner()
    properties = Properties()
    transforms = TransformController()
    render_window: RenderWindow | None = None

    font = pygame.font.SysFont("Segoe UI", 16)
    small_font = pygame.font.SysFont("Segoe UI", 13)
    running = True
    viewport_drag: str | None = None
    quality = _quality_for_scene(scene)

    while running:
        dt = clock.tick(60) / 1000.0
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
            if event.type == pygame.WINDOWLEAVE:
                viewport_drag = None
                continue

            if render_window is not None:
                result = render_window.handle(event)
                if result == "close" and not render_window.is_running():
                    render_window = None
                continue

            if add_menu.open:
                action = add_menu.handle(event)
                if action and action != "handled":
                    _handle_add_action(scene, action, orbit.target)
                continue

            if properties.handle(event, scene):
                continue

            if event.type == pygame.KEYDOWN:
                nav_keys = (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_q, pygame.K_e)
                mouse_buttons = pygame.mouse.get_pressed(3)
                nav_active = viewport_drag == "look" or mouse_buttons[2]
                if event.key in nav_keys and nav_active:
                    continue
                if transforms.handle_key(event, scene):
                    continue
                mods = pygame.key.get_mods()
                if event.key == pygame.K_F12:
                    render_window = RenderWindow(scene, use_gpu=use_gpu)
                    render_window.start()
                    continue
                if event.key == pygame.K_a and mods & pygame.KMOD_SHIFT:
                    add_menu.show()
                    continue
                if event.key == pygame.K_s and mods & pygame.KMOD_CTRL:
                    _save_dialog(scene)
                    continue
                if event.key == pygame.K_o and mods & pygame.KMOD_CTRL:
                    loaded = _load_dialog()
                    if loaded is not None:
                        scene = loaded
                        quality = _quality_for_scene(scene)
                    continue
                if event.key in (pygame.K_DELETE, pygame.K_x):
                    _delete_selected(scene)
                    continue

            if transforms.handle_gizmo_press(event, scene, orbit, vrect):
                continue
            if transforms.handle_mouse_button(event):
                continue
            if transforms.handle_motion(event, orbit):
                continue

            action = toolbar.handle(event)
            if action:
                if action == "add_menu":
                    add_menu.show()
                elif action == "toggle_gpu":
                    use_gpu = not use_gpu
                elif action == "save":
                    _save_dialog(scene)
                elif action == "load":
                    loaded = _load_dialog()
                    if loaded is not None:
                        scene = loaded
                        quality = _quality_for_scene(scene)
                elif action == "cycle_quality":
                    quality = _next_quality(quality)
                    _apply_quality_preset(scene, quality)
                elif action == "render":
                    render_window = RenderWindow(scene, use_gpu=use_gpu)
                    render_window.start()
                elif action == "import_obj" or action.startswith("add_"):
                    _handle_add_action(scene, action, orbit.target)
                continue

            if outliner.handle(event, scene):
                continue

            if event.type == pygame.MOUSEWHEEL and vrect.contains(pygame.mouse.get_pos()):
                orbit.zoom(event.y)
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and vrect.contains(event.pos):
                if event.button == 2:
                    viewport_drag = "pan" if pygame.key.get_mods() & pygame.KMOD_SHIFT else "orbit"
                    continue
                if event.button == 3:
                    viewport_drag = "look"
                    continue

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2 and viewport_drag in {"orbit", "pan"}:
                    viewport_drag = None
                    continue
                if event.button == 3 and viewport_drag == "look":
                    viewport_drag = None
                    continue

            if event.type == pygame.MOUSEMOTION and (viewport_drag is not None or vrect.contains(event.pos)):
                buttons = pygame.mouse.get_pressed(3)
                if viewport_drag == "look" and buttons[2]:
                    orbit.look(event.rel[0], event.rel[1])
                    continue
                if viewport_drag in {"orbit", "pan"} and buttons[1]:
                    if viewport_drag == "pan" or pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        orbit.pan(event.rel[0], event.rel[1])
                    else:
                        orbit.orbit(event.rel[0], event.rel[1])
                    continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and vrect.contains(event.pos):
                scene.selected = pick(event.pos[0], event.pos[1], vrect, orbit, scene)
                continue

        if render_window is None and transforms.mode is None and properties.active_key is None:
            _handle_viewport_keyboard(orbit, dt, pygame.key.get_pressed(), pygame.key.get_mods(), viewport_drag == "look")

        viewport.draw(scene, orbit, layout.viewport_rect(pygame.display.get_surface().get_size()), pygame.display.get_surface().get_size())
        overlay = pygame.Surface(pygame.display.get_surface().get_size(), pygame.SRCALPHA)
        toolbar.draw(overlay, font, small_font, use_gpu, quality)
        outliner.draw(overlay, scene, font, small_font)
        properties.draw(overlay, scene, font, small_font)
        _draw_status(overlay, small_font, _status_text(transforms, viewport_drag))
        add_menu.draw(overlay, font, small_font)
        if render_window is not None:
            render_window.draw(overlay, font, small_font)
        draw_overlay_surface(overlay)
        pygame.display.flip()

    pygame.quit()
    return 0


def _apply_quality_preset(scene: Scene, quality: str) -> None:
    width, height, samples, bounces = QUALITY_PRESETS[quality]
    scene.render.width = width
    scene.render.height = height
    scene.render.samples = samples
    scene.render.max_bounces = bounces


def _next_quality(quality: str) -> str:
    try:
        index = QUALITY_ORDER.index(quality)
    except ValueError:
        index = 0
    return QUALITY_ORDER[(index + 1) % len(QUALITY_ORDER)]


def _quality_for_scene(scene: Scene) -> str:
    render_tuple = (
        scene.render.width,
        scene.render.height,
        scene.render.samples,
        scene.render.max_bounces,
    )
    for quality, preset in QUALITY_PRESETS.items():
        if render_tuple == preset:
            return quality
    return "med"


def _handle_viewport_keyboard(orbit: OrbitCamera, dt: float, keys: Any, mods: int, require_look_mode: bool) -> None:
    if not require_look_mode:
        return
    right = 0.0
    up = 0.0
    forward = 0.0
    if keys[pygame.K_d]:
        right += 1.0
    if keys[pygame.K_a]:
        right -= 1.0
    if keys[pygame.K_e]:
        up += 1.0
    if keys[pygame.K_q]:
        up -= 1.0
    if keys[pygame.K_w]:
        forward += 1.0
    if keys[pygame.K_s]:
        forward -= 1.0
    if right == 0.0 and up == 0.0 and forward == 0.0:
        return
    speed = max(0.5, orbit.distance * 0.65)
    if mods & pygame.KMOD_SHIFT:
        speed *= 3.0
    if mods & pygame.KMOD_CTRL:
        speed *= 0.25
    orbit.move(right, up, forward, speed * max(0.0, dt))


def _status_text(transforms: TransformController, viewport_drag: str | None) -> str:
    if transforms.mode is not None:
        return transforms.status()
    if viewport_drag == "look":
        return "RMB look + WASD/QE move | Shift fast | Ctrl slow"
    if viewport_drag == "orbit":
        return "MMB orbit | Shift+MMB pan | Wheel zoom"
    if viewport_drag == "pan":
        return "MMB pan | Wheel zoom"
    return f"{transforms.status()} | RMB+WASD/QE viewport"


def _handle_add_action(scene: Scene, action: str, target: np.ndarray) -> None:
    if action == "import_obj":
        _import_obj_dialog(scene)
    elif action.startswith("add_"):
        _add_item(scene, action, target)


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
    elif action == "add_cube":
        position[1] += 0.5
        scene.add(
            Mesh.cube(
                name=f"Cube {len(scene.objects) + 1}",
                position=position,
                material=DiffuseMaterial(color=np.array([0.62, 0.68, 0.76], dtype=np.float64)),
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
    elif action == "add_directional_light":
        position[1] += 3.0
        scene.add(
            DirectionalLight(
                name=f"Directional Light {len(scene.lights) + 1}",
                position=position,
                direction=np.array([-1.0, -2.0, -1.0], dtype=np.float64),
                color=np.ones(3, dtype=np.float64),
                intensity=1.2,
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


def _import_obj_dialog(scene: Scene) -> None:
    path = _file_dialog(save=False, obj=True)
    if not path:
        return
    try:
        mesh = Mesh.from_obj(
            path,
            material=DiffuseMaterial(color=np.array([0.72, 0.72, 0.78], dtype=np.float64)),
        )
    except Exception:
        return
    scene.add(mesh)


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


def _file_dialog(save: bool, obj: bool = False) -> str:
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
            filetypes = [("Wavefront OBJ", "*.obj")] if obj else [("PyTrace Scene", "*.json"), ("JSON", "*.json")]
            path = filedialog.askopenfilename(
                filetypes=filetypes,
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
