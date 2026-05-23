from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .lights import AreaLight, DirectionalLight, Light
from .materials import DiffuseMaterial, GlassMaterial, SpecularMaterial
from .objects import Plane, SceneObject, Sphere


@dataclass
class RenderConfig:
    width: int = 800
    height: int = 450
    samples: int = 32
    max_bounces: int = 10
    exposure: float = 1.0
    area_light_samples: int = 4
    render_mode: str = "direct"
    background_mode: str = "gradient"
    environment_path: str = ""
    background_color: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.08], dtype=np.float64)
    )


@dataclass
class Scene:
    objects: list[SceneObject] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    render: RenderConfig = field(default_factory=RenderConfig)
    selected: object | None = field(default=None, repr=False, compare=False)

    def add(self, item: SceneObject | Light) -> None:
        if isinstance(item, SceneObject):
            self.objects.append(item)
        elif isinstance(item, Light):
            self.lights.append(item)
        else:
            raise TypeError(f"unsupported scene item: {type(item)!r}")
        self.selected = item

    def remove(self, item: object) -> None:
        if isinstance(item, SceneObject) and item in self.objects:
            self.objects.remove(item)
        elif isinstance(item, Light) and item in self.lights:
            self.lights.remove(item)
        else:
            return
        if self.selected is item:
            self.selected = None

    def to_dict(self) -> dict:
        from .serializer import scene_to_dict

        return scene_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Scene:
        from .serializer import scene_from_dict

        return scene_from_dict(data)

    @classmethod
    def default(cls) -> Scene:
        """Classic showcase: diffuse, glass, and mirror spheres on a ground plane."""
        s = cls()
        s.objects = [
            Sphere(
                name="Red Sphere",
                position=np.array([-1.2, 0.7, -0.5], dtype=np.float64),
                radius=0.7,
                material=DiffuseMaterial(
                    color=np.array([0.6, 0.05, 0.05], dtype=np.float64),
                    roughness=0.15,
                ),
            ),
            Sphere(
                name="Glass Sphere",
                position=np.array([-0.1, 0.6, 0.7], dtype=np.float64),
                radius=0.6,
                material=GlassMaterial(
                    ior=1.5,
                    roughness=0.0,
                    absorption_color=np.array([0.82, 0.95, 0.78], dtype=np.float64),
                    tint=np.array([0.92, 0.98, 0.88], dtype=np.float64),
                ),
            ),
            Sphere(
                name="Mirror Sphere",
                position=np.array([1.3, 0.7, 0.2], dtype=np.float64),
                radius=0.7,
                material=SpecularMaterial(
                    color=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                    roughness=0.13,
                    ior=1.5,
                ),
            ),
            Sphere(
                name="Gold Sphere",
                position=np.array([0.8, 0.3, 2.0], dtype=np.float64),
                radius=0.3,
                material=SpecularMaterial(
                    color=np.array([0.9, 0.7, 0.2], dtype=np.float64),
                    roughness=0.4,
                    ior=2.0,
                ),
            ),
            Sphere(
                name="Green Sphere",
                position=np.array([-2.2, 0.4, -1.0], dtype=np.float64),
                radius=0.4,
                material=SpecularMaterial(
                    color=np.array([0.2, 0.8, 0.3], dtype=np.float64),
                    roughness=0.1,
                    ior=1.5,
                ),
            ),
            Sphere(
                name="Tiny White Sphere",
                position=np.array([-0.6, 0.2, 1.8], dtype=np.float64),
                radius=0.2,
                material=DiffuseMaterial(
                    color=np.array([0.9, 0.9, 0.9], dtype=np.float64),
                    roughness=0.8,
                ),
            ),
            Sphere(
                name="Bronze Sphere",
                position=np.array([0.2, 0.5, -1.5], dtype=np.float64),
                radius=0.5,
                material=SpecularMaterial(
                    color=np.array([0.8, 0.5, 0.3], dtype=np.float64),
                    roughness=0.25,
                    ior=2.5,
                ),
            ),
            Plane(
                name="Ground",
                position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.6, 0.6, 0.6], dtype=np.float64),
                    roughness=1.0,
                ),
            ),
            Plane(
                name="Left Wall",
                position=np.array([-4.0 * np.cos(np.pi / 3), 0.0, -4.0 * np.sin(np.pi / 3)], dtype=np.float64),
                normal=np.array([np.cos(np.pi / 3), 0.0, np.sin(np.pi / 3)], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.2, 0.4, 0.6], dtype=np.float64),
                    roughness=1.0,
                ),
            ),
            Plane(
                name="Back Wall",
                position=np.array([5.0 * np.sin(np.pi / 3), 0.0, -5.0 * np.cos(np.pi / 3)], dtype=np.float64),
                normal=np.array([-np.sin(np.pi / 3), 0.0, np.cos(np.pi / 3)], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.6, 0.5, 0.4], dtype=np.float64),
                    roughness=1.0,
                ),
            ),
            Plane(
                name="Right Wall",
                position=np.array([12.0 * np.cos(np.pi / 3), 0.0, 12.0 * np.sin(np.pi / 3)], dtype=np.float64),
                normal=np.array([-np.cos(np.pi / 3), 0.0, -np.sin(np.pi / 3)], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.6, 0.4, 0.2], dtype=np.float64), # Complementary to Left Wall (blue)
                    roughness=1.0,
                ),
            ),
            Plane(
                name="Front Wall",
                position=np.array([-12.0 * np.sin(np.pi / 3), 0.0, 12.0 * np.cos(np.pi / 3)], dtype=np.float64),
                normal=np.array([np.sin(np.pi / 3), 0.0, -np.cos(np.pi / 3)], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.4, 0.5, 0.6], dtype=np.float64), # Complementary to Back Wall (warm)
                    roughness=1.0,
                ),
            ),
            Plane(
                name="Roof",
                position=np.array([0.0, 12.0, 0.0], dtype=np.float64),
                normal=np.array([0.0, -1.0, 0.0], dtype=np.float64),
                material=DiffuseMaterial(
                    color=np.array([0.7, 0.7, 0.7], dtype=np.float64), # Neutral white ceiling
                    roughness=1.0,
                ),
            ),
        ]
        scene_center = np.array([0.0, 0.5, 0.0], dtype=np.float64)
        key_pos = np.array([7.0, 5.0, 6.0], dtype=np.float64)
        key_normal = (scene_center - key_pos) / np.linalg.norm(scene_center - key_pos)
        fill_pos = np.array([-2.5, 4.0, -2.5], dtype=np.float64)
        fill_normal = (scene_center - fill_pos) / np.linalg.norm(scene_center - fill_pos)
        corner_target = np.array([-1.2, 0.6, -2.2], dtype=np.float64)
        corner_pos = np.array([1.5, 5.5, -5.0], dtype=np.float64)
        corner_normal = (corner_target - corner_pos) / np.linalg.norm(corner_target - corner_pos)
        s.lights = [
            # Warm key — disk area light for soft shadows
            AreaLight(
                name="Warm Key",
                position=key_pos,
                normal=key_normal,
                radius=1.5,
                color=np.array([1.0, 0.75, 0.45], dtype=np.float64),
                intensity=240.0,
            ),
            # Cool fill — aimed to reach left wall and back corner
            AreaLight(
                name="Cool Fill",
                position=fill_pos,
                normal=fill_normal,
                radius=1.1,
                color=np.array([0.45, 0.65, 1.0], dtype=np.float64),
                intensity=48.0,
            ),
            # Back-corner rim — lifts dark pocket seen in the mirror
            AreaLight(
                name="Corner Rim",
                position=corner_pos,
                normal=corner_normal,
                radius=1.8,
                color=np.array([0.55, 0.58, 0.65], dtype=np.float64),
                intensity=28.0,
            ),
        ]
        s.camera = Camera(
            position=np.array([0.0, 2.0, 8.0], dtype=np.float64),
            target=np.array([0.0, 0.5, 0.0], dtype=np.float64),
            fov=38.0,
        )
        s.render = RenderConfig(
            width=800,
            height=450,
            samples=32,
            max_bounces=10,
            exposure=1.0,
        )
        return s
