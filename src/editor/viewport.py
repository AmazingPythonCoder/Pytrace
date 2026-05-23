from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - exercised only when editor deps are missing
    pygame = None  # type: ignore[assignment]

try:
    from OpenGL.GL import (
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_FRONT_AND_BACK,
        GL_FILL,
        GL_LINE,
        GL_LINES,
        GL_MODELVIEW,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_PROJECTION,
        GL_QUADS,
        GL_RGBA,
        GL_SRC_ALPHA,
        GL_TEXTURE_2D,
        GL_UNSIGNED_BYTE,
        glBegin,
        glBlendFunc,
        glClear,
        glClearColor,
        glColor3f,
        glColor4f,
        glDisable,
        glDrawPixels,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glPolygonMode,
        glPopMatrix,
        glPushMatrix,
        glRasterPos2i,
        glTranslatef,
        glVertex3f,
        glViewport,
        glWindowPos2i,
    )
    from OpenGL.GLU import gluLookAt, gluPerspective, gluQuadricDrawStyle, gluSphere, gluNewQuadric
except ImportError:  # pragma: no cover
    glBegin = None  # type: ignore[assignment]

from src.raytracer.camera_frame import compute_camera_frame
from src.scene.lights import AreaLight, PointLight
from src.scene.materials import DiffuseMaterial, GlassMaterial, SpecularMaterial
from src.scene.objects import Plane, Sphere
from src.scene.scene import Scene

from .layout import Rect
from .orbit_camera import OrbitCamera


def _require_gl() -> None:
    if glBegin is None:
        raise RuntimeError("PyOpenGL is required for the editor. Install requirements.txt.")


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / length


class Viewport:
    def __init__(self) -> None:
        self._quadric: Any | None = None

    def setup(self) -> None:
        _require_gl()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.06, 0.07, 0.09, 1.0)
        self._quadric = gluNewQuadric()

    def draw(self, scene: Scene, camera: OrbitCamera, rect: Rect, screen_size: tuple[int, int]) -> None:
        _require_gl()
        if self._quadric is None:
            self.setup()

        screen_h = screen_size[1]
        glViewport(rect.x, screen_h - rect.bottom, rect.w, rect.h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(camera.fov, rect.w / rect.h, 0.05, 500.0)

        eye = camera.eye
        target = camera.target
        up = camera.up
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            eye[0],
            eye[1],
            eye[2],
            target[0],
            target[1],
            target[2],
            up[0],
            up[1],
            up[2],
        )

        self._draw_grid()
        for obj in scene.objects:
            if not obj.visible:
                continue
            if isinstance(obj, Sphere):
                self._draw_sphere(obj, selected=scene.selected is obj)
            elif isinstance(obj, Plane):
                self._draw_plane(obj, selected=scene.selected is obj)
        for light in scene.lights:
            self._draw_light(light, selected=scene.selected is light)
        self._draw_camera_frustum(scene, selected=scene.selected is scene.camera)

    def _draw_grid(self, size: int = 20) -> None:
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-size, size + 1):
            alpha = 0.18 if i % 5 else 0.34
            glColor4f(0.52, 0.56, 0.62, alpha)
            glVertex3f(float(i), 0.0, float(-size))
            glVertex3f(float(i), 0.0, float(size))
            glVertex3f(float(-size), 0.0, float(i))
            glVertex3f(float(size), 0.0, float(i))
        glEnd()

        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(0.9, 0.2, 0.2, 0.85)
        glVertex3f(float(-size), 0.004, 0.0)
        glVertex3f(float(size), 0.004, 0.0)
        glColor4f(0.2, 0.75, 0.35, 0.85)
        glVertex3f(0.0, 0.004, float(-size))
        glVertex3f(0.0, 0.004, float(size))
        glEnd()

    def _material_color(self, obj: object) -> tuple[float, float, float]:
        material = getattr(obj, "material", None)
        if isinstance(material, DiffuseMaterial):
            return tuple(float(v) for v in material.color)
        if isinstance(material, SpecularMaterial):
            return tuple(float(v) for v in material.color)
        if isinstance(material, GlassMaterial):
            return tuple(float(v) for v in material.tint)
        return (0.7, 0.72, 0.76)

    def _draw_sphere(self, sphere: Sphere, selected: bool = False) -> None:
        assert self._quadric is not None
        color = self._material_color(sphere)
        radius = float(sphere.radius * max(sphere.scale[0], sphere.scale[1], sphere.scale[2]))
        glPushMatrix()
        glTranslatef(float(sphere.position[0]), float(sphere.position[1]), float(sphere.position[2]))
        glColor3f(color[0], color[1], color[2])
        gluSphere(self._quadric, radius, 28, 14)
        if selected:
            glLineWidth(3.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.38, 0.72, 1.0)
            gluSphere(self._quadric, radius * 1.015, 28, 14)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()

    def _plane_basis(self, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = _normalize(normal)
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, n))) > 0.9:
            helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tangent = _normalize(np.cross(helper, n))
        bitangent = _normalize(np.cross(n, tangent))
        return tangent, bitangent

    def _draw_plane(self, plane: Plane, selected: bool = False) -> None:
        color = self._material_color(plane)
        tangent, bitangent = self._plane_basis(plane.normal)
        half_size = max(1.0, float(max(plane.scale[0], plane.scale[2]))) * 4.0
        corners = [
            plane.position + (-tangent - bitangent) * half_size,
            plane.position + (tangent - bitangent) * half_size,
            plane.position + (tangent + bitangent) * half_size,
            plane.position + (-tangent + bitangent) * half_size,
        ]
        glColor4f(color[0], color[1], color[2], 0.55)
        glBegin(GL_QUADS)
        for c in corners:
            glVertex3f(float(c[0]), float(c[1]), float(c[2]))
        glEnd()

        glLineWidth(3.0 if selected else 1.0)
        glColor4f(0.38, 0.72, 1.0, 1.0) if selected else glColor4f(0.82, 0.86, 0.92, 0.45)
        glBegin(GL_LINES)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            glVertex3f(float(corners[a][0]), float(corners[a][1]), float(corners[a][2]))
            glVertex3f(float(corners[b][0]), float(corners[b][1]), float(corners[b][2]))
        glEnd()

    def _draw_light(self, light: object, selected: bool = False) -> None:
        assert self._quadric is not None
        color = getattr(light, "color", np.ones(3, dtype=np.float64))
        pos = getattr(light, "position", np.zeros(3, dtype=np.float64))
        glPushMatrix()
        glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
        glColor3f(float(color[0]), float(color[1]), float(color[2]))
        gluSphere(self._quadric, 0.18 if isinstance(light, PointLight) else 0.24, 16, 8)
        if selected:
            glLineWidth(3.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.38, 0.72, 1.0)
            gluSphere(self._quadric, 0.32, 16, 8)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()

        if isinstance(light, AreaLight):
            normal = _normalize(light.normal)
            glLineWidth(2.0)
            glColor4f(float(color[0]), float(color[1]), float(color[2]), 0.8)
            glBegin(GL_LINES)
            glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
            end = pos + normal * float(light.radius)
            glVertex3f(float(end[0]), float(end[1]), float(end[2]))
            glEnd()

    def _draw_camera_frustum(self, scene: Scene, selected: bool = False) -> None:
        cam = scene.camera
        frame = compute_camera_frame(cam.position, cam.target, cam.up, cam.fov, 16, 9)
        o = frame.origin
        ll = frame.lower_left
        lr = frame.lower_left + frame.horizontal
        ul = frame.lower_left + frame.vertical
        ur = frame.lower_left + frame.horizontal + frame.vertical
        glLineWidth(3.0 if selected else 1.5)
        glColor3f(0.38, 0.72, 1.0) if selected else glColor4f(0.95, 0.95, 0.7, 0.8)
        glBegin(GL_LINES)
        for p in (ll, lr, ul, ur):
            glVertex3f(float(o[0]), float(o[1]), float(o[2]))
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        for a, b in ((ll, lr), (lr, ur), (ur, ul), (ul, ll)):
            glVertex3f(float(a[0]), float(a[1]), float(a[2]))
            glVertex3f(float(b[0]), float(b[1]), float(b[2]))
        glEnd()


def draw_overlay_surface(surface: Any) -> None:
    _require_gl()
    if pygame is None:
        raise RuntimeError("pygame is required for the editor. Install requirements.txt.")

    width, height = surface.get_size()
    data = pygame.image.tostring(surface, "RGBA", True)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_TEXTURE_2D)
    try:
        glWindowPos2i(0, 0)
    except Exception:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRasterPos2i(-1, -1)
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glEnable(GL_DEPTH_TEST)
