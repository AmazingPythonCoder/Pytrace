"""Verify viewport navigation math without opening an editor window."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.editor.app import (
    EditorState,
    _GIZMO,
    _align_camera_to_view,
    _apply_quality_preset,
    _gizmo_linear_for_operation,
    _next_quality,
    _quality_for_scene,
)
from src.editor.add_actions import ENTRIES
from src.editor.gizmos import TransformController
from src.editor.orbit_camera import OrbitCamera
from src.scene.scene import Scene


def main() -> None:
    camera = OrbitCamera(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        distance=10.0,
        yaw=0.0,
        pitch=0.0,
    )
    original_eye = camera.eye.copy()
    original_target = camera.target.copy()

    camera.move(local_right=0.0, local_up=0.0, local_forward=1.0, amount=2.0)
    assert np.allclose(camera.target, original_target + np.array([0.0, 0.0, -2.0]))
    assert np.allclose(camera.eye, original_eye + np.array([0.0, 0.0, -2.0]))
    assert np.isclose(camera.distance, 10.0)

    camera.move(local_right=1.0, local_up=1.0, local_forward=0.0, amount=1.5)
    assert np.allclose(camera.target, np.array([1.5, 1.5, -2.0], dtype=np.float64))

    eye_before_look = camera.eye.copy()
    camera.look(1000.0, 1000.0)
    assert np.allclose(camera.eye, eye_before_look)
    assert camera.pitch <= 85.0
    camera.look(-1000.0, -2000.0)
    assert camera.pitch >= -85.0
    old_distance = camera.distance
    camera.zoom(1.0)
    assert camera.distance < old_distance
    camera.zoom(-1.0)
    assert np.isclose(camera.distance, old_distance)
    old_target = camera.target.copy()
    camera.pan_pixels(40.0, -20.0, viewport_height=800)
    assert not np.allclose(camera.target, old_target)

    scene = Scene.default()
    _apply_quality_preset(scene, "low")
    assert (scene.render.width, scene.render.height, scene.render.samples, scene.render.max_bounces) == (800, 450, 32, 8)
    assert _quality_for_scene(scene) == "low"
    assert _next_quality("low") == "med"
    assert _next_quality("med") == "high"
    assert _next_quality("high") == "ultra"
    assert _next_quality("ultra") == "low"

    state = EditorState(scene=scene)
    state.orbit.target[:] = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    state.orbit.distance = 7.5
    state.orbit.yaw = 35.0
    state.orbit.pitch = -12.0
    state.orbit.fov = 47.0
    _align_camera_to_view(state)
    assert state.scene.selected is state.scene.camera
    assert np.allclose(state.scene.camera.position, state.orbit.eye)
    assert np.allclose(state.scene.camera.target, state.orbit.target)
    assert np.allclose(state.scene.camera.up, state.orbit.up)
    assert np.isclose(state.scene.camera.fov, state.orbit.fov)
    assert np.isclose(state.scene.camera.focus_distance, state.orbit.distance)

    actions = {entry.action for entry in ENTRIES}
    assert {
        "add_sphere",
        "add_plane",
        "add_cube",
        "import_obj",
        "add_point_light",
        "add_directional_light",
        "add_area_light",
    }.issubset(actions)

    transforms = TransformController()
    scene.selected = scene.objects[0]
    assert transforms.handle_command("g", scene, (0, 0))
    assert transforms.mode == "move"
    assert transforms.handle_command("x", scene, (0, 0))
    assert transforms.axis == "X"
    before = scene.selected.position.copy()
    assert transforms.drag((20.0, 0.0), camera, viewport_height=800)
    assert scene.selected.position[0] > before[0]
    moved = scene.selected.position.copy()
    assert transforms.drag((20.0, 0.0), camera, viewport_height=800)
    assert np.allclose(scene.selected.position, moved)
    assert transforms.drag((-20.0, 0.0), camera, viewport_height=800)
    assert scene.selected.position[0] < before[0]
    assert transforms.handle_command("escape", scene, (0, 0))
    assert transforms.mode is None
    assert np.allclose(scene.selected.position, before)

    reverse_view_camera = OrbitCamera(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        distance=10.0,
        yaw=180.0,
        pitch=0.0,
    )
    scene.selected = scene.objects[0]
    before = scene.selected.position.copy()
    assert transforms.handle_command("g", scene, (100, 100))
    assert transforms.handle_command("x", scene, (100, 100))
    assert transforms.drag((60.0, 100.0), reverse_view_camera, viewport_height=800, viewport_width=1200)
    assert scene.selected.position[0] > before[0]
    transforms.cancel()

    assert transforms.handle_command("s", scene, (0, 0))
    before_radius = scene.selected.radius
    assert transforms.drag((40.0, 0.0), camera, viewport_height=800)
    assert scene.selected.radius > before_radius
    scaled_radius = scene.selected.radius
    assert transforms.drag((40.0, 0.0), camera, viewport_height=800)
    assert np.isclose(scene.selected.radius, scaled_radius)
    assert transforms.drag((0.0, 0.0), camera, viewport_height=800)
    assert np.isclose(scene.selected.radius, before_radius)
    assert transforms.drag((-40.0, 0.0), camera, viewport_height=800)
    assert scene.selected.radius < before_radius
    transforms.confirm()

    if _GIZMO is not None:
        scene.selected.rotation[:] = np.array([0.0, 35.0, 0.0], dtype=np.float64)
        scene.selected.scale[:] = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        world_scale_linear = _gizmo_linear_for_operation(scene.selected, _GIZMO.OPERATION.scale, False)
        local_scale_linear = _gizmo_linear_for_operation(scene.selected, _GIZMO.OPERATION.scale, True)
        world_move_linear = _gizmo_linear_for_operation(scene.selected, _GIZMO.OPERATION.translate, False)
        local_move_linear = _gizmo_linear_for_operation(scene.selected, _GIZMO.OPERATION.translate, True)
        assert np.allclose(world_scale_linear, np.diag(scene.selected.scale))
        assert not np.allclose(local_scale_linear, world_scale_linear)
        assert np.allclose(world_move_linear, np.eye(3))
        assert not np.allclose(local_move_linear, world_move_linear)

    print("Editor navigation OK: viewport, quality cycling, add entries, and transform state verified")


if __name__ == "__main__":
    main()
