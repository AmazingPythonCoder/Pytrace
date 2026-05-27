from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from OpenGL import GL

from src.raytracer.camera_frame import compute_camera_frame
from src.scene.camera import Camera
from src.scene.lights import AreaLight, DirectionalLight, PointLight
from src.scene.materials import DiffuseMaterial, EmissiveMaterial, GlassMaterial, SpecularMaterial
from src.scene.objects import Mesh, Plane, SceneObject, Sphere
from src.scene.scene import Scene

from .orbit_camera import OrbitCamera


VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
uniform mat4 u_vp;
out vec4 v_color;
void main()
{
    gl_Position = u_vp * vec4(a_pos, 1.0);
    v_color = a_color;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main()
{
    frag_color = v_color;
}
"""


@dataclass
class _MeshData:
    vertices: np.ndarray
    triangles: np.ndarray
    wire_edges: tuple[tuple[int, int], ...]


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / length


def _compile_shader(shader_type: int, source: str) -> int:
    shader_obj = GL.glCreateShader(shader_type)
    if shader_obj is None:
        raise RuntimeError("OpenGL shader creation failed")
    shader = int(cast(Any, shader_obj))
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenGL shader compile failed: {log}")
    return int(shader)


def _create_program() -> int:
    vertex = _compile_shader(GL.GL_VERTEX_SHADER, VERTEX_SHADER)
    fragment = _compile_shader(GL.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
    program_obj = GL.glCreateProgram()
    if program_obj is None:
        raise RuntimeError("OpenGL program creation failed")
    program = int(cast(Any, program_obj))
    GL.glAttachShader(program, vertex)
    GL.glAttachShader(program, fragment)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vertex)
    GL.glDeleteShader(fragment)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenGL shader link failed: {log}")
    return program


def _perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
    return np.array(
        [
            [f / max(aspect, 1e-6), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    view_up = np.cross(right, forward)
    return np.array(
        [
            [right[0], right[1], right[2], -float(np.dot(right, eye))],
            [view_up[0], view_up[1], view_up[2], -float(np.dot(view_up, eye))],
            [-forward[0], -forward[1], -forward[2], float(np.dot(forward, eye))],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rotation_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(euler_degrees, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent = _normalize(np.cross(helper, n))
    bitangent = _normalize(np.cross(n, tangent))
    return tangent, bitangent


def _sphere_mesh(segments: int = 24, rings: int = 12) -> _MeshData:
    vertices: list[list[float]] = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        y = math.cos(phi)
        radius = math.sin(phi)
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            vertices.append([math.cos(theta) * radius, y, math.sin(theta) * radius])

    triangles: list[list[int]] = []
    edges: set[tuple[int, int]] = set()
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + s
            d = (r + 1) * segments + (s + 1) % segments
            if r != 0:
                triangles.append([a, c, b])
            if r != rings - 1:
                triangles.append([b, c, d])
            for e0, e1 in ((a, b), (a, c)):
                edges.add((min(e0, e1), max(e0, e1)))
    return _MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int64),
        wire_edges=tuple(sorted(edges)),
    )


UNIT_SPHERE = _sphere_mesh()


class PreviewRenderer:
    """Modern OpenGL scene preview rendered into a texture for ImGui."""

    def __init__(self) -> None:
        self.program = 0
        self.vao = 0
        self.vbo = 0
        self.fbo = 0
        self.texture = 0
        self.depth = 0
        self.size = (0, 0)
        self._initialized = False
        self.last_view = np.eye(4, dtype=np.float32)
        self.last_projection = np.eye(4, dtype=np.float32)

    def setup(self) -> None:
        if self._initialized:
            return
        self.program = _create_program()
        self.vao = int(GL.glGenVertexArrays(1))
        self.vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        stride = 7 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        self._initialized = True

    def cleanup(self) -> None:
        if self.vbo:
            GL.glDeleteBuffers(1, [self.vbo])
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
        if self.program:
            GL.glDeleteProgram(self.program)
        if self.texture:
            GL.glDeleteTextures([self.texture])
        if self.depth:
            GL.glDeleteRenderbuffers(1, [self.depth])
        if self.fbo:
            GL.glDeleteFramebuffers(1, [self.fbo])
        self.__init__()

    def render(self, scene: Scene, camera: OrbitCamera, width: int, height: int) -> int:
        self.setup()
        width = max(1, int(width))
        height = max(1, int(height))
        self._resize(width, height)

        triangles: list[list[float]] = []
        lines: list[list[float]] = []
        self._build_scene(scene, triangles, lines)

        self.last_view = _look_at(camera.eye, camera.target, camera.up)
        self.last_projection = _perspective(camera.fov, width / height, 0.05, 500.0)
        vp = self.last_projection @ self.last_view

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.055, 0.065, 0.085, 1.0)
        clear_mask = int(cast(Any, GL.GL_COLOR_BUFFER_BIT)) | int(cast(Any, GL.GL_DEPTH_BUFFER_BIT))
        GL.glClear(clear_mask)
        GL.glUseProgram(self.program)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "u_vp"), 1, GL.GL_TRUE, vp)
        GL.glBindVertexArray(self.vao)

        self._draw_buffer(triangles, GL.GL_TRIANGLES)
        GL.glLineWidth(1.0)
        self._draw_buffer(lines, GL.GL_LINES)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return self.texture

    def _resize(self, width: int, height: int) -> None:
        if self.size == (width, height) and self.texture:
            return
        if not self.fbo:
            self.fbo = int(GL.glGenFramebuffers(1))
        if not self.texture:
            self.texture = int(GL.glGenTextures(1))
        if not self.depth:
            self.depth = int(GL.glGenRenderbuffers(1))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture, 0)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depth)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"OpenGL framebuffer incomplete: {status}")
        self.size = (width, height)

    def _draw_buffer(self, vertices: list[list[float]], mode: int) -> None:
        if not vertices:
            return
        data = np.asarray(vertices, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
        GL.glDrawArrays(mode, 0, int(data.shape[0]))

    def _vertex(self, pos: np.ndarray, color: tuple[float, float, float, float]) -> list[float]:
        return [float(pos[0]), float(pos[1]), float(pos[2]), color[0], color[1], color[2], color[3]]

    def _add_line(
        self,
        lines: list[list[float]],
        a: np.ndarray,
        b: np.ndarray,
        color: tuple[float, float, float, float],
    ) -> None:
        lines.append(self._vertex(a, color))
        lines.append(self._vertex(b, color))

    def _add_tri(
        self,
        triangles: list[list[float]],
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        color: tuple[float, float, float, float],
    ) -> None:
        triangles.append(self._vertex(a, color))
        triangles.append(self._vertex(b, color))
        triangles.append(self._vertex(c, color))

    def _build_scene(self, scene: Scene, triangles: list[list[float]], lines: list[list[float]]) -> None:
        self._grid(lines)
        for obj in scene.objects:
            if not obj.visible:
                continue
            selected = scene.selected is obj
            if isinstance(obj, Sphere):
                self._sphere(obj.position, obj.radius * float(max(obj.scale)), self._material_color(obj), selected, triangles, lines)
            elif isinstance(obj, Plane):
                self._plane(obj, selected, triangles, lines)
            elif isinstance(obj, Mesh):
                self._mesh(obj, selected, triangles, lines)
        for light in scene.lights:
            self._light(light, scene.selected is light, triangles, lines)
        self._camera(scene.camera, scene.selected is scene.camera, lines)
        self._transform_gizmo(scene.selected, lines)

    def _grid(self, lines: list[list[float]], size: int = 20) -> None:
        for i in range(-size, size + 1):
            color = (0.45, 0.48, 0.54, 0.25 if i % 5 else 0.42)
            self._add_line(lines, np.array([i, 0.0, -size]), np.array([i, 0.0, size]), color)
            self._add_line(lines, np.array([-size, 0.0, i]), np.array([size, 0.0, i]), color)
        self._add_line(lines, np.array([-size, 0.004, 0.0]), np.array([size, 0.004, 0.0]), (0.95, 0.2, 0.2, 0.9))
        self._add_line(lines, np.array([0.0, 0.004, -size]), np.array([0.0, 0.004, size]), (0.2, 0.82, 0.35, 0.9))

    def _material_color(self, obj: object) -> tuple[float, float, float, float]:
        material = getattr(obj, "material", None)
        if isinstance(material, DiffuseMaterial):
            rgb = material.color
        elif isinstance(material, SpecularMaterial):
            rgb = material.color
        elif isinstance(material, GlassMaterial):
            rgb = material.tint
        elif isinstance(material, EmissiveMaterial):
            rgb = np.clip(material.color * material.strength, 0.0, 1.0)
        else:
            rgb = np.array([0.7, 0.72, 0.76], dtype=np.float64)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.86)

    def _sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: tuple[float, float, float, float],
        selected: bool,
        triangles: list[list[float]],
        lines: list[list[float]],
    ) -> None:
        verts = UNIT_SPHERE.vertices * max(0.01, float(radius)) + center
        for tri in UNIT_SPHERE.triangles:
            self._add_tri(triangles, verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])], color)
        wire_color = (0.36, 0.72, 1.0, 1.0) if selected else (0.82, 0.86, 0.92, 0.16)
        if selected:
            verts = UNIT_SPHERE.vertices * max(0.01, float(radius) * 1.02) + center
        for a, b in UNIT_SPHERE.wire_edges[::4 if not selected else 2]:
            self._add_line(lines, verts[a], verts[b], wire_color)

    def _plane(self, plane: Plane, selected: bool, triangles: list[list[float]], lines: list[list[float]]) -> None:
        tangent, bitangent = _plane_basis(plane.normal)
        half_size = max(1.0, float(max(plane.scale[0], plane.scale[2]))) * 4.0
        corners = [
            plane.position + (-tangent - bitangent) * half_size,
            plane.position + (tangent - bitangent) * half_size,
            plane.position + (tangent + bitangent) * half_size,
            plane.position + (-tangent + bitangent) * half_size,
        ]
        color = self._material_color(plane)
        color = (color[0], color[1], color[2], 0.45)
        self._add_tri(triangles, corners[0], corners[1], corners[2], color)
        self._add_tri(triangles, corners[0], corners[2], corners[3], color)
        line_color = (0.36, 0.72, 1.0, 1.0) if selected else (0.82, 0.86, 0.92, 0.25)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            self._add_line(lines, corners[a], corners[b], line_color)

    def _mesh_world_vertices(self, mesh: Mesh) -> np.ndarray:
        if mesh.vertices.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return (mesh.vertices * mesh.scale) @ _rotation_matrix(mesh.rotation).T + mesh.position

    def _mesh(self, mesh: Mesh, selected: bool, triangles: list[list[float]], lines: list[list[float]]) -> None:
        verts = self._mesh_world_vertices(mesh)
        color = self._material_color(mesh)
        color = (color[0], color[1], color[2], 0.82)
        edge_color = (0.36, 0.72, 1.0, 1.0) if selected else (0.82, 0.86, 0.92, 0.2)
        for tri in mesh.triangles:
            a, b, c = verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])]
            self._add_tri(triangles, a, b, c, color)
            if selected:
                self._add_line(lines, a, b, edge_color)
                self._add_line(lines, b, c, edge_color)
                self._add_line(lines, c, a, edge_color)

    def _light(self, light: object, selected: bool, triangles: list[list[float]], lines: list[list[float]]) -> None:
        color_arr = getattr(light, "color", np.ones(3, dtype=np.float64))
        color = (float(color_arr[0]), float(color_arr[1]), float(color_arr[2]), 0.95)
        pos = getattr(light, "position", np.zeros(3, dtype=np.float64))
        radius = 0.18 if isinstance(light, PointLight) else 0.24
        self._sphere(pos, radius, color, selected, triangles, lines)
        if isinstance(light, AreaLight):
            normal = _normalize(light.normal)
            end = pos + normal * float(light.radius)
            self._add_line(lines, pos, end, color)
        elif isinstance(light, DirectionalLight):
            direction = _normalize(light.direction)
            self._add_line(lines, pos - direction * 0.8, pos + direction * 0.8, color)

    def _camera(self, camera: Camera, selected: bool, lines: list[list[float]]) -> None:
        frame = compute_camera_frame(camera.position, camera.target, camera.up, camera.fov, 16, 9)
        origin = frame.origin
        ll = frame.lower_left
        lr = frame.lower_left + frame.horizontal
        ul = frame.lower_left + frame.vertical
        ur = frame.lower_left + frame.horizontal + frame.vertical
        color = (0.36, 0.72, 1.0, 1.0) if selected else (0.95, 0.92, 0.58, 0.85)
        for p in (ll, lr, ul, ur):
            self._add_line(lines, origin, p, color)
        for a, b in ((ll, lr), (lr, ur), (ur, ul), (ul, ll)):
            self._add_line(lines, a, b, color)

    def _selected_position(self, item: object | None) -> np.ndarray | None:
        if item is None:
            return None
        if hasattr(item, "position"):
            return np.asarray(getattr(item, "position"), dtype=np.float64)
        if isinstance(item, Camera):
            return item.position
        return None

    def _transform_gizmo(self, selected: object | None, lines: list[list[float]]) -> None:
        pos = self._selected_position(selected)
        if pos is None:
            return
        length = 1.25
        self._add_line(lines, pos, pos + np.array([length, 0.0, 0.0]), (0.98, 0.18, 0.18, 1.0))
        self._add_line(lines, pos, pos + np.array([0.0, length, 0.0]), (0.18, 0.9, 0.3, 1.0))
        self._add_line(lines, pos, pos + np.array([0.0, 0.0, length]), (0.25, 0.5, 1.0, 1.0))


def matrices_for_imguizmo(renderer: PreviewRenderer) -> tuple[list[float], list[float]]:
    """Return column-major view/projection matrices for ImGuizmo."""
    return (
        renderer.last_view.T.astype(np.float32).reshape(-1).tolist(),
        renderer.last_projection.T.astype(np.float32).reshape(-1).tolist(),
    )
