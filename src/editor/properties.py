from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

from src.scene.camera import Camera
from src.scene.lights import AreaLight, DirectionalLight, Light
from src.scene.materials import DiffuseMaterial, EmissiveMaterial, GlassMaterial, Material, SpecularMaterial
from src.scene.objects import Mesh, Plane, SceneObject, Sphere
from src.scene.scene import Scene

from . import layout
from .selection import display_name


@dataclass
class Field:
    key: str
    rect: Any
    getter: Callable[[], Any]
    setter: Callable[[Any], None]
    kind: str = "float"


class Properties:
    def __init__(self) -> None:
        self.fields: dict[str, Field] = {}
        self.material_buttons: dict[str, Any] = {}
        self.option_buttons: dict[str, tuple[Any, Callable[[], None]]] = {}
        self.active_key: str | None = None
        self.edit_text = ""

    def handle(self, event: Any, scene: Scene) -> bool:
        if pygame is None:
            return False

        if event.type == pygame.KEYDOWN and self.active_key is not None:
            return self._handle_key(event)

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False

        for material_type, rect in self.material_buttons.items():
            if rect.collidepoint(event.pos):
                if isinstance(scene.selected, SceneObject):
                    scene.selected.material = self._convert_material(scene.selected.material, material_type)
                self.active_key = None
                return True

        for _, (rect, action) in self.option_buttons.items():
            if rect.collidepoint(event.pos):
                action()
                self.active_key = None
                return True

        for key, field in self.fields.items():
            if field.rect.collidepoint(event.pos):
                if field.kind == "bool":
                    field.setter(True)
                    self.active_key = None
                    return True
                self.active_key = key
                self.edit_text = self._format_value(field.getter(), field.kind)
                return True

        rect = layout.right_panel_rect(pygame.display.get_surface().get_size())
        if rect.contains(event.pos):
            self._commit_active()
            self.active_key = None
            return True
        return False

    def draw(self, surface: Any, scene: Scene, font: Any, small_font: Any) -> None:
        if pygame is None:
            return
        rect = layout.right_panel_rect(surface.get_size())
        panel = pygame.Rect(rect.x, rect.y, rect.w, rect.h)
        pygame.draw.rect(surface, layout.PANEL, panel)
        pygame.draw.line(surface, layout.PANEL_EDGE, (rect.x, rect.y), (rect.x, rect.bottom))

        self.fields = {}
        self.material_buttons = {}
        self.option_buttons = {}
        x = rect.x + layout.PANEL_PAD
        y = rect.y + layout.PANEL_PAD
        self._draw_text(surface, font, "Properties", (x, y), layout.TEXT)
        y += 34

        selected = scene.selected
        if selected is None:
            self._draw_text(surface, font, "No selection", (x, y), layout.TEXT_MUTED)
            return

        self._draw_text(surface, small_font, display_name(selected), (x, y), layout.ACCENT)
        y += 30
        if isinstance(selected, SceneObject):
            y = self._draw_object(surface, font, small_font, selected, x, y)
        elif isinstance(selected, Light):
            y = self._draw_light(surface, font, small_font, selected, x, y)
        elif isinstance(selected, Camera):
            y = self._draw_camera(surface, font, small_font, scene, x, y)

        if self.active_key is not None and self.active_key not in self.fields:
            self.active_key = None

    def _draw_object(self, surface: Any, font: Any, small_font: Any, obj: SceneObject, x: int, y: int) -> int:
        y = self._section(surface, small_font, "TRANSFORM", x, y)
        y = self._field(surface, font, "name", "Name", lambda: obj.name, lambda v: setattr(obj, "name", str(v)), x, y, "str", width=180)
        y = self._bool_button(surface, font, "Visible", obj.visible, lambda v: setattr(obj, "visible", v), x, y)
        y = self._vector(surface, font, "position", "Location", obj.position, x, y)
        y = self._vector(surface, font, "rotation", "Rotation", obj.rotation, x, y)
        y = self._vector(surface, font, "scale", "Scale", obj.scale, x, y)
        if isinstance(obj, Sphere):
            y = self._field(surface, font, "radius", "Radius", lambda: obj.radius, lambda v: setattr(obj, "radius", max(0.02, float(v))), x, y)
        if isinstance(obj, Plane):
            y = self._vector(surface, font, "normal", "Normal", obj.normal, x, y, normalize=True)
        if isinstance(obj, Mesh):
            y = self._mesh_info(surface, font, obj, x, y)

        y += 10
        y = self._section(surface, small_font, "MATERIAL", x, y)
        y = self._material_type_buttons(surface, font, obj, x, y)
        y = self._draw_material(surface, font, obj.material, x, y)
        return y

    def _draw_light(self, surface: Any, font: Any, small_font: Any, light: Light, x: int, y: int) -> int:
        y = self._section(surface, small_font, "LIGHT", x, y)
        y = self._field(surface, font, "light_name", "Name", lambda: light.name, lambda v: setattr(light, "name", str(v)), x, y, "str", width=180)
        y = self._vector(surface, font, "light_position", "Position", light.position, x, y)
        y = self._vector(surface, font, "light_color", "Color", light.color, x, y, min_value=0.0, max_value=20.0)
        y = self._field(surface, font, "intensity", "Intensity", lambda: light.intensity, lambda v: setattr(light, "intensity", max(0.0, float(v))), x, y)
        if isinstance(light, DirectionalLight):
            y = self._vector(surface, font, "directional_direction", "Direction", light.direction, x, y, normalize=True)
        if isinstance(light, AreaLight):
            y = self._vector(surface, font, "area_normal", "Normal", light.normal, x, y, normalize=True)
            y = self._field(surface, font, "area_radius", "Radius", lambda: light.radius, lambda v: setattr(light, "radius", max(0.02, float(v))), x, y)
        return y

    def _draw_camera(self, surface: Any, font: Any, small_font: Any, scene: Scene, x: int, y: int) -> int:
        cam = scene.camera
        render = scene.render
        y = self._section(surface, small_font, "CAMERA", x, y)
        y = self._vector(surface, font, "camera_position", "Position", cam.position, x, y)
        y = self._vector(surface, font, "camera_target", "Target", cam.target, x, y)
        y = self._field(surface, font, "camera_fov", "FOV", lambda: cam.fov, lambda v: setattr(cam, "fov", max(5.0, min(160.0, float(v)))), x, y)
        y = self._field(surface, font, "camera_aperture", "Aperture", lambda: cam.aperture, lambda v: setattr(cam, "aperture", max(0.0, float(v))), x, y)
        y = self._field(
            surface,
            font,
            "camera_focus",
            "Focus Dist",
            lambda: 0.0 if cam.focus_distance is None else cam.focus_distance,
            lambda v: setattr(cam, "focus_distance", None if float(v) <= 0.0 else float(v)),
            x,
            y,
        )
        y += 10
        y = self._section(surface, small_font, "RENDER", x, y)
        y = self._option_buttons(
            surface,
            font,
            "render_mode",
            "Mode",
            (("direct", "Direct"), ("path", "Path")),
            lambda: render.render_mode,
            lambda v: setattr(render, "render_mode", v),
            x,
            y,
        )
        y = self._option_buttons(
            surface,
            font,
            "background_mode",
            "BG Mode",
            (("solid", "Solid"), ("gradient", "Gradient"), ("environment", "Env")),
            lambda: render.background_mode,
            lambda v: setattr(render, "background_mode", v),
            x,
            y,
        )
        y = self._field(surface, font, "render_width", "Width", lambda: render.width, lambda v: setattr(render, "width", max(1, int(v))), x, y, "int")
        y = self._field(surface, font, "render_height", "Height", lambda: render.height, lambda v: setattr(render, "height", max(1, int(v))), x, y, "int")
        y = self._field(surface, font, "render_samples", "Samples", lambda: render.samples, lambda v: setattr(render, "samples", max(1, int(v))), x, y, "int")
        y = self._field(surface, font, "render_bounces", "Bounces", lambda: render.max_bounces, lambda v: setattr(render, "max_bounces", max(0, int(v))), x, y, "int")
        y = self._field(surface, font, "render_exposure", "Exposure", lambda: render.exposure, lambda v: setattr(render, "exposure", max(0.001, float(v))), x, y)
        y = self._field(surface, font, "render_area_samples", "Area SPP", lambda: render.area_light_samples, lambda v: setattr(render, "area_light_samples", max(1, int(v))), x, y, "int")
        y = self._vector(surface, font, "background", "Background", render.background_color, x, y, min_value=0.0, max_value=20.0)
        y = self._environment_row(surface, font, scene, x, y)
        return y

    def _draw_material(self, surface: Any, font: Any, material: Material, x: int, y: int) -> int:
        if isinstance(material, DiffuseMaterial):
            y = self._vector(surface, font, "mat_color", "Color", material.color, x, y, min_value=0.0, max_value=1.0)
            y = self._field(surface, font, "mat_roughness", "Roughness", lambda: material.roughness, lambda v: setattr(material, "roughness", self._clamp(float(v), 0.0, 1.0)), x, y)
        elif isinstance(material, SpecularMaterial):
            y = self._vector(surface, font, "mat_color", "Color", material.color, x, y, min_value=0.0, max_value=1.0)
            y = self._field(surface, font, "mat_roughness", "Roughness", lambda: material.roughness, lambda v: setattr(material, "roughness", self._clamp(float(v), 0.0, 1.0)), x, y)
            y = self._field(surface, font, "mat_ior", "IOR", lambda: material.ior, lambda v: setattr(material, "ior", max(1.0, float(v))), x, y)
        elif isinstance(material, GlassMaterial):
            y = self._vector(surface, font, "mat_tint", "Tint", material.tint, x, y, min_value=0.0, max_value=1.0)
            y = self._vector(surface, font, "mat_absorption", "Absorb", material.absorption_color, x, y, min_value=0.0, max_value=1.0)
            y = self._field(surface, font, "mat_roughness", "Roughness", lambda: material.roughness, lambda v: setattr(material, "roughness", self._clamp(float(v), 0.0, 1.0)), x, y)
            y = self._field(surface, font, "mat_ior", "IOR", lambda: material.ior, lambda v: setattr(material, "ior", max(1.0, float(v))), x, y)
        elif isinstance(material, EmissiveMaterial):
            y = self._vector(surface, font, "mat_color", "Color", material.color, x, y, min_value=0.0, max_value=1.0)
            y = self._field(surface, font, "mat_strength", "Strength", lambda: material.strength, lambda v: setattr(material, "strength", max(0.0, float(v))), x, y)
        return y

    def _material_type_buttons(self, surface: Any, font: Any, obj: SceneObject, x: int, y: int) -> int:
        labels = [("diffuse", "Diffuse"), ("specular", "Specular"), ("glass", "Glass"), ("emissive", "Emit")]
        current = getattr(obj.material, "type", "diffuse")
        bx = x + 88
        for material_type, label in labels:
            rect = pygame.Rect(bx, y, 56, 24)
            color = layout.ACCENT_SOFT if material_type == current else (255, 255, 255, 12)
            pygame.draw.rect(surface, color, rect, border_radius=5)
            pygame.draw.rect(surface, layout.FIELD_EDGE, rect, width=1, border_radius=5)
            self._draw_text(surface, font, label, (rect.x + 7, rect.y + 3), layout.TEXT)
            self.material_buttons[material_type] = rect
            bx += 60
        self._draw_text(surface, font, "Type", (x, y + 3), layout.TEXT_MUTED)
        return y + 32

    def _mesh_info(self, surface: Any, font: Any, mesh: Mesh, x: int, y: int) -> int:
        vertex_count = mesh.vertices.shape[0]
        triangle_count = mesh.triangles.shape[0]
        normal_count = 0 if mesh.normals is None else mesh.normals.shape[0]
        self._draw_text(surface, font, "Mesh", (x, y + 3), layout.TEXT_MUTED)
        self._draw_text(
            surface,
            font,
            f"{vertex_count}v {triangle_count}t {normal_count}n",
            (x + 116, y + 3),
            layout.TEXT,
        )
        y += 30
        if mesh.source_path:
            source = mesh.source_path
            if len(source) > 28:
                source = "..." + source[-25:]
            self._draw_text(surface, font, "Source", (x, y + 3), layout.TEXT_MUTED)
            self._draw_text(surface, font, source, (x + 116, y + 3), layout.TEXT)
            y += 30
        return y

    def _option_buttons(
        self,
        surface: Any,
        font: Any,
        key: str,
        label: str,
        options: tuple[tuple[str, str], ...],
        getter: Callable[[], str],
        setter: Callable[[str], None],
        x: int,
        y: int,
    ) -> int:
        current = str(getter()).strip().lower()
        self._draw_text(surface, font, label, (x, y + 3), layout.TEXT_MUTED)
        bx = x + 116
        for value, text_label in options:
            width = max(52, min(78, font.size(text_label)[0] + 18))
            rect = pygame.Rect(bx, y, width, 24)
            selected = current == value
            pygame.draw.rect(surface, layout.ACCENT_SOFT if selected else (255, 255, 255, 12), rect, border_radius=5)
            pygame.draw.rect(surface, layout.FIELD_EDGE, rect, width=1, border_radius=5)
            self._draw_text(surface, font, text_label, (rect.centerx - font.size(text_label)[0] // 2, rect.y + 3), layout.TEXT)
            self.option_buttons[f"{key}_{value}"] = (rect, lambda value=value: setter(value))
            bx += width + 6
        return y + 30

    def _environment_row(self, surface: Any, font: Any, scene: Scene, x: int, y: int) -> int:
        render = scene.render
        self._draw_text(surface, font, "Env Map", (x, y + 3), layout.TEXT_MUTED)
        field_rect = pygame.Rect(x + 116, y, 128, 24)
        self._draw_field_box(
            surface,
            font,
            Field(
                "environment_path",
                field_rect,
                lambda: render.environment_path,
                lambda v: setattr(render, "environment_path", str(v)),
                "str",
            ),
        )
        browse_rect = pygame.Rect(field_rect.right + 6, y, 70, 24)
        pygame.draw.rect(surface, (255, 255, 255, 12), browse_rect, border_radius=5)
        pygame.draw.rect(surface, layout.FIELD_EDGE, browse_rect, width=1, border_radius=5)
        label = "Browse"
        self._draw_text(surface, font, label, (browse_rect.centerx - font.size(label)[0] // 2, browse_rect.y + 3), layout.TEXT)
        self.option_buttons["environment_browse"] = (browse_rect, lambda: self._pick_environment(scene))
        return y + 30

    def _convert_material(self, old: Material, material_type: str) -> Material:
        color = np.array([0.8, 0.8, 0.8], dtype=np.float64)
        roughness = float(getattr(old, "roughness", 0.2))
        if isinstance(old, (DiffuseMaterial, SpecularMaterial)):
            color = old.color.copy()
        elif isinstance(old, GlassMaterial):
            color = old.tint.copy()
        if material_type == "diffuse":
            return DiffuseMaterial(color=color, roughness=max(0.0, min(1.0, roughness)))
        if material_type == "specular":
            return SpecularMaterial(color=color, roughness=max(0.0, min(1.0, roughness)), ior=float(getattr(old, "ior", 1.5)))
        if material_type == "glass":
            return GlassMaterial(tint=color, roughness=max(0.0, min(1.0, roughness)), ior=float(getattr(old, "ior", 1.45)))
        if material_type == "emissive":
            return EmissiveMaterial(color=color, strength=5.0)
        return old

    def _section(self, surface: Any, font: Any, label: str, x: int, y: int) -> int:
        self._draw_text(surface, font, label, (x, y), layout.TEXT_MUTED)
        return y + 24

    def _field(
        self,
        surface: Any,
        font: Any,
        key: str,
        label: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
        x: int,
        y: int,
        kind: str = "float",
        width: int = 96,
    ) -> int:
        rect = pygame.Rect(x + 116, y, width, 24)
        self._draw_text(surface, font, label, (x, y + 3), layout.TEXT_MUTED)
        self._draw_field_box(surface, font, Field(key, rect, getter, setter, kind))
        return y + 30

    def _vector(
        self,
        surface: Any,
        font: Any,
        prefix: str,
        label: str,
        arr: np.ndarray,
        x: int,
        y: int,
        normalize: bool = False,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> int:
        self._draw_text(surface, font, label, (x, y + 3), layout.TEXT_MUTED)
        labels = ("x", "y", "z")
        for i, suffix in enumerate(labels):
            rect = pygame.Rect(x + 116 + i * 65, y, 58, 24)

            def getter(index=i, array=arr):
                return float(array[index])

            def setter(value, index=i, array=arr):
                value = float(value)
                if min_value is not None:
                    value = max(min_value, value)
                if max_value is not None:
                    value = min(max_value, value)
                array[index] = value
                if normalize:
                    length = float(np.linalg.norm(array))
                    if length > 1e-12:
                        array[:] = array / length

            self._draw_field_box(surface, font, Field(f"{prefix}_{suffix}", rect, getter, setter, "float"))
        return y + 30

    def _bool_button(self, surface: Any, font: Any, label: str, value: bool, setter: Callable[[bool], None], x: int, y: int) -> int:
        key = f"bool_{label}"
        rect = pygame.Rect(x + 116, y, 72, 24)
        self._draw_text(surface, font, label, (x, y + 3), layout.TEXT_MUTED)
        pygame.draw.rect(surface, layout.ACCENT_SOFT if value else (255, 255, 255, 12), rect, border_radius=5)
        pygame.draw.rect(surface, layout.FIELD_EDGE, rect, width=1, border_radius=5)
        self._draw_text(surface, font, "On" if value else "Off", (rect.x + 18, rect.y + 3), layout.TEXT)
        self.fields[key] = Field(key, rect, lambda: value, lambda _v: setter(not value), "bool")
        return y + 30

    def _draw_field_box(self, surface: Any, font: Any, field: Field) -> None:
        self.fields[field.key] = field
        active = self.active_key == field.key
        pygame.draw.rect(surface, layout.FIELD_BG, field.rect, border_radius=5)
        pygame.draw.rect(surface, layout.ACCENT if active else layout.FIELD_EDGE, field.rect, width=1, border_radius=5)
        value = self.edit_text if active else self._format_value(field.getter(), field.kind)
        text = font.render(value, True, layout.TEXT)
        old_clip = surface.get_clip()
        surface.set_clip(field.rect)
        surface.blit(text, (field.rect.x + 6, field.rect.y + 3))
        surface.set_clip(old_clip)

    def _handle_key(self, event: Any) -> bool:
        if event.key == pygame.K_RETURN:
            self._commit_active()
            self.active_key = None
            return True
        if event.key == pygame.K_ESCAPE:
            self.active_key = None
            return True
        if event.key == pygame.K_BACKSPACE:
            self.edit_text = self.edit_text[:-1]
            return True
        if event.unicode:
            field = self.fields.get(self.active_key or "")
            if field and field.kind == "str":
                self.edit_text += event.unicode
            elif event.unicode in "0123456789.-+eE":
                self.edit_text += event.unicode
            return True
        return True

    def _commit_active(self) -> None:
        if self.active_key is None:
            return
        field = self.fields.get(self.active_key)
        if field is None:
            return
        try:
            if field.kind == "str":
                field.setter(self.edit_text)
            elif field.kind == "int":
                field.setter(int(float(self.edit_text)))
            elif field.kind == "bool":
                field.setter(True)
            else:
                field.setter(float(self.edit_text))
        except (TypeError, ValueError):
            return

    def _format_value(self, value: Any, kind: str) -> str:
        if kind == "str":
            return str(value)
        if kind == "int":
            return str(int(value))
        if kind == "bool":
            return "On" if value else "Off"
        return f"{float(value):.3f}"

    def _draw_text(self, surface: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int, int]) -> None:
        img = font.render(text, True, color)
        surface.blit(img, pos)

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _choice(self, value: str, allowed: tuple[str, ...], fallback: str) -> str:
        normalized = value.strip().lower()
        return normalized if normalized in allowed else fallback

    def _pick_environment(self, scene: Scene) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Environment Images", "*.hdr *.pic *.exr *.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("HDR Images", "*.hdr *.pic *.exr"),
                    ("All Files", "*.*"),
                ]
            )
            root.destroy()
        except Exception:
            path = ""
        if path:
            scene.render.environment_path = str(path)
            scene.render.background_mode = "environment"
