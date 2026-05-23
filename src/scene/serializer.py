from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera import Camera
from .lights import AreaLight, Light, PointLight
from .materials import DiffuseMaterial, GlassMaterial, Material, SpecularMaterial
from .objects import Plane, SceneObject, Sphere
from .scene import RenderConfig, Scene


def _array(value: Any, fallback: list[float]) -> np.ndarray:
    if value is None:
        value = fallback
    return np.asarray(value, dtype=np.float64)


def _list(value: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).tolist()]


def material_to_dict(material: Material) -> dict:
    if isinstance(material, DiffuseMaterial):
        return {
            "type": "diffuse",
            "color": _list(material.color),
            "roughness": float(material.roughness),
        }
    if isinstance(material, SpecularMaterial):
        return {
            "type": "specular",
            "color": _list(material.color),
            "roughness": float(material.roughness),
            "ior": float(material.ior),
        }
    if isinstance(material, GlassMaterial):
        return {
            "type": "glass",
            "ior": float(material.ior),
            "roughness": float(material.roughness),
            "absorption_color": _list(material.absorption_color),
            "tint": _list(material.tint),
        }
    raise TypeError(f"unsupported material type: {type(material)!r}")


def material_from_dict(data: dict) -> Material:
    material_type = data.get("type", "diffuse")
    if material_type == "diffuse":
        return DiffuseMaterial(
            color=_array(data.get("color"), [0.8, 0.8, 0.8]),
            roughness=float(data.get("roughness", 1.0)),
        )
    if material_type == "specular":
        return SpecularMaterial(
            color=_array(data.get("color"), [0.9, 0.9, 0.9]),
            roughness=float(data.get("roughness", 0.0)),
            ior=float(data.get("ior", 1.5)),
        )
    if material_type == "glass":
        return GlassMaterial(
            ior=float(data.get("ior", 1.45)),
            roughness=float(data.get("roughness", 0.0)),
            absorption_color=_array(data.get("absorption_color"), [1.0, 1.0, 1.0]),
            tint=_array(data.get("tint"), [1.0, 1.0, 1.0]),
        )
    raise ValueError(f"unsupported material type: {material_type!r}")


def object_to_dict(obj: SceneObject) -> dict:
    base = {
        "name": obj.name,
        "position": _list(obj.position),
        "rotation": _list(obj.rotation),
        "scale": _list(obj.scale),
        "visible": bool(obj.visible),
        "material": material_to_dict(obj.material),
    }
    if isinstance(obj, Sphere):
        return {"type": "sphere", **base, "radius": float(obj.radius)}
    if isinstance(obj, Plane):
        return {"type": "plane", **base, "normal": _list(obj.normal)}
    raise TypeError(f"unsupported object type: {type(obj)!r}")


def object_from_dict(data: dict) -> SceneObject:
    object_type = data.get("type", "sphere")
    base = {
        "name": str(data.get("name", object_type.title())),
        "position": _array(data.get("position"), [0.0, 0.0, 0.0]),
        "rotation": _array(data.get("rotation"), [0.0, 0.0, 0.0]),
        "scale": _array(data.get("scale"), [1.0, 1.0, 1.0]),
        "visible": bool(data.get("visible", True)),
        "material": material_from_dict(data.get("material", {"type": "diffuse"})),
    }
    if object_type == "sphere":
        return Sphere(radius=float(data.get("radius", 0.5)), **base)
    if object_type == "plane":
        return Plane(normal=_array(data.get("normal"), [0.0, 1.0, 0.0]), **base)
    raise ValueError(f"unsupported object type: {object_type!r}")


def light_to_dict(light: Light) -> dict:
    base = {
        "name": getattr(light, "name", light.type.title()),
        "position": _list(light.position),
        "color": _list(light.color),
        "intensity": float(light.intensity),
    }
    if isinstance(light, PointLight):
        return {"type": "point", **base}
    if isinstance(light, AreaLight):
        return {
            "type": "area",
            **base,
            "normal": _list(light.normal),
            "radius": float(light.radius),
        }
    raise TypeError(f"unsupported light type: {type(light)!r}")


def light_from_dict(data: dict) -> Light:
    light_type = data.get("type", "point")
    base = {
        "name": str(data.get("name", light_type.title())),
        "position": _array(data.get("position"), [0.0, 5.0, 0.0]),
        "color": _array(data.get("color"), [1.0, 1.0, 1.0]),
        "intensity": float(data.get("intensity", 100.0)),
    }
    if light_type == "point":
        return PointLight(**base)
    if light_type == "area":
        return AreaLight(
            normal=_array(data.get("normal"), [0.0, -1.0, 0.0]),
            radius=float(data.get("radius", 1.0)),
            **base,
        )
    raise ValueError(f"unsupported light type: {light_type!r}")


def camera_to_dict(camera: Camera) -> dict:
    return {
        "position": _list(camera.position),
        "target": _list(camera.target),
        "up": _list(camera.up),
        "fov": float(camera.fov),
    }


def camera_from_dict(data: dict) -> Camera:
    return Camera(
        position=_array(data.get("position"), [0.0, 2.0, 8.0]),
        target=_array(data.get("target"), [0.0, 0.0, 0.0]),
        up=_array(data.get("up"), [0.0, 1.0, 0.0]),
        fov=float(data.get("fov", 55.0)),
    )


def render_config_to_dict(render: RenderConfig) -> dict:
    return {
        "width": int(render.width),
        "height": int(render.height),
        "samples": int(render.samples),
        "max_bounces": int(render.max_bounces),
        "exposure": float(render.exposure),
        "area_light_samples": int(render.area_light_samples),
        "background_color": _list(render.background_color),
    }


def render_config_from_dict(data: dict) -> RenderConfig:
    return RenderConfig(
        width=int(data.get("width", 800)),
        height=int(data.get("height", 450)),
        samples=int(data.get("samples", 32)),
        max_bounces=int(data.get("max_bounces", 10)),
        exposure=float(data.get("exposure", 1.0)),
        area_light_samples=int(data.get("area_light_samples", 4)),
        background_color=_array(data.get("background_color"), [0.05, 0.05, 0.08]),
    )


def scene_to_dict(scene: Scene) -> dict:
    return {
        "camera": camera_to_dict(scene.camera),
        "objects": [object_to_dict(obj) for obj in scene.objects],
        "lights": [light_to_dict(light) for light in scene.lights],
        "render": render_config_to_dict(scene.render),
    }


def scene_from_dict(data: dict) -> Scene:
    return Scene(
        objects=[object_from_dict(obj) for obj in data.get("objects", [])],
        lights=[light_from_dict(light) for light in data.get("lights", [])],
        camera=camera_from_dict(data.get("camera", {})),
        render=render_config_from_dict(data.get("render", {})),
    )


def save_scene(scene: Scene, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene_to_dict(scene), indent=2), encoding="utf-8")


def load_scene(path: str | Path) -> Scene:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return scene_from_dict(data)
