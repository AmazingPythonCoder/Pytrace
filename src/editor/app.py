from __future__ import annotations

import os
from dataclasses import dataclass

if os.getenv("XDG_SESSION_TYPE") == "wayland" and not os.getenv("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "x11"

import numpy as np
from imgui_bundle import ImVec2, ImVec4, hello_imgui, imgui, immapp

try:  # ImGuizmo is bundled, but keeping this optional makes import smoke tests friendlier.
    from imgui_bundle import imguizmo

    _GIZMO = imguizmo.im_guizmo
except Exception:  # pragma: no cover
    _GIZMO = None

from src.scene.camera import Camera
from src.scene.lights import AreaLight, DirectionalLight, Light, PointLight
from src.scene.materials import DiffuseMaterial, EmissiveMaterial, GlassMaterial, Material, SpecularMaterial
from src.scene.objects import Mesh, Plane, SceneObject, Sphere
from src.scene.serializer import load_scene, save_scene
from src.scene.scene import Scene

from . import layout
from .add_actions import ENTRIES
from .gizmos import TransformController
from .imgui_render_window import ImguiRenderWindow
from .orbit_camera import OrbitCamera
from .preview_gl import PreviewRenderer, matrices_for_imguizmo
from .selection import display_name, pick


QUALITY_ORDER = ("low", "med", "high", "ultra")
QUALITY_PRESETS = {
    "low": (800, 450, 32, 8),
    "med": (1280, 720, 128, 10),
    "high": (1920, 1080, 384, 12),
    "ultra": (3840, 2160, 768, 16),
}


@dataclass
class EditorState:
    scene: Scene
    use_gpu: bool = True
    quality: str = "med"
    orbit: OrbitCamera = None  # type: ignore[assignment]
    preview: PreviewRenderer = None  # type: ignore[assignment]
    transforms: TransformController = None  # type: ignore[assignment]
    render_window: ImguiRenderWindow | None = None
    show_add_popup: bool = False
    viewport_drag: str | None = None
    viewport_hovered: bool = False
    gizmo_operation: object | None = None

    def __post_init__(self) -> None:
        self.quality = _quality_for_scene(self.scene)
        self.orbit = OrbitCamera()
        self.preview = PreviewRenderer()
        self.transforms = TransformController()
        if _GIZMO is not None:
            self.gizmo_operation = _GIZMO.OPERATION.translate


def run(scene: Scene | None = None, use_gpu: bool = True) -> int:
    state = EditorState(scene=scene or Scene.default(), use_gpu=use_gpu)
    runner_params = _runner_params(state)
    immapp.run(runner_params)
    return 0


def _runner_params(state: EditorState) -> hello_imgui.RunnerParams:
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "PyTrace Editor"
    runner_params.app_window_params.window_geometry.size = layout.WINDOW_SIZE
    runner_params.app_window_params.resizable = True
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    runner_params.imgui_window_params.show_menu_bar = True
    runner_params.imgui_window_params.show_menu_app = False
    runner_params.imgui_window_params.show_menu_view = True
    runner_params.imgui_window_params.show_status_bar = True
    runner_params.imgui_window_params.show_status_fps = True
    runner_params.imgui_window_params.menu_app_title = "PyTrace"
    runner_params.imgui_window_params.enable_viewports = True
    runner_params.callbacks.setup_imgui_style = _setup_theme
    runner_params.callbacks.post_init = state.preview.setup
    runner_params.callbacks.before_exit = lambda: _cleanup(state)
    runner_params.callbacks.show_menus = lambda: _draw_menus(state)
    runner_params.callbacks.show_status = lambda: imgui.text(_status_text(state))
    runner_params.callbacks.show_gui = lambda: _draw_popups_and_shortcuts(state)
    runner_params.callbacks.add_edge_toolbar(
        hello_imgui.EdgeToolbarType.top,
        lambda: _draw_toolbar(state),
        hello_imgui.EdgeToolbarOptions(size_em=2.65, window_bg=ImVec4(0.08, 0.09, 0.11, 0.94)),
    )
    runner_params.docking_params = _docking_params(state)
    runner_params.docking_params.layout_condition = hello_imgui.DockingLayoutCondition.application_start
    runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    runner_params.ini_filename = "PyTrace/PyTraceEditor.ini"
    return runner_params


def _docking_params(state: EditorState) -> hello_imgui.DockingParams:
    docking_params = hello_imgui.DockingParams()
    split_left = hello_imgui.DockingSplit()
    split_left.initial_dock = "MainDockSpace"
    split_left.new_dock = "OutlinerDock"
    split_left.direction = imgui.Dir.left
    split_left.ratio = 0.18

    split_right = hello_imgui.DockingSplit()
    split_right.initial_dock = "MainDockSpace"
    split_right.new_dock = "PropertiesDock"
    split_right.direction = imgui.Dir.right
    split_right.ratio = 0.24

    viewport = hello_imgui.DockableWindow()
    viewport.label = "Viewport"
    viewport.dock_space_name = "MainDockSpace"
    viewport.gui_function = lambda: _draw_viewport(state)
    viewport.imgui_window_flags = imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse

    outliner = hello_imgui.DockableWindow()
    outliner.label = "Outliner"
    outliner.dock_space_name = "OutlinerDock"
    outliner.gui_function = lambda: _draw_outliner(state)

    properties = hello_imgui.DockableWindow()
    properties.label = "Properties"
    properties.dock_space_name = "PropertiesDock"
    properties.gui_function = lambda: _draw_properties(state)

    docking_params.docking_splits = [split_left, split_right]
    docking_params.dockable_windows = [outliner, viewport, properties]
    return docking_params


def _setup_theme() -> None:
    hello_imgui.imgui_default_settings.setup_default_imgui_style()
    tweaked = hello_imgui.ImGuiTweakedTheme()
    tweaked.theme = hello_imgui.ImGuiTheme_.darcula_darker
    tweaked.tweaks.rounding = 4.0
    hello_imgui.apply_tweaked_theme(tweaked)
    style = imgui.get_style()
    style.window_padding = ImVec2(10, 8)
    style.frame_padding = ImVec2(8, 4)
    style.item_spacing = ImVec2(8, 6)
    style.set_color_(imgui.Col_.window_bg, (0.09, 0.10, 0.12, 1.0))
    style.set_color_(imgui.Col_.header, (0.16, 0.25, 0.36, 1.0))
    style.set_color_(imgui.Col_.header_hovered, (0.20, 0.34, 0.50, 1.0))
    style.set_color_(imgui.Col_.button, (0.17, 0.20, 0.25, 1.0))
    style.set_color_(imgui.Col_.button_hovered, (0.23, 0.30, 0.39, 1.0))


def _cleanup(state: EditorState) -> None:
    if state.render_window is not None:
        state.render_window.cleanup()
    state.preview.cleanup()


def _draw_toolbar(state: EditorState) -> None:
    if imgui.button("Add"):
        state.show_add_popup = True
    imgui.same_line()
    if imgui.button("Save"):
        _save_dialog(state.scene)
    imgui.same_line()
    if imgui.button("Load"):
        _replace_scene_from_dialog(state)
    imgui.same_line()
    if imgui.button(f"Quality {state.quality.title()}"):
        state.quality = _next_quality(state.quality)
        _apply_quality_preset(state.scene, state.quality)
    imgui.same_line()
    if imgui.button("Render F12"):
        _start_render(state)
    imgui.same_line()
    if imgui.button(f"GPU {'On' if state.use_gpu else 'Off'}"):
        state.use_gpu = not state.use_gpu
    imgui.same_line()
    if _GIZMO is not None:
        if imgui.button("Move"):
            _set_gizmo_operation(state, "translate")
        imgui.same_line()
        if imgui.button("Rotate"):
            _set_gizmo_operation(state, "rotate")
        imgui.same_line()
        if imgui.button("Scale"):
            _set_gizmo_operation(state, "scale")
        imgui.same_line()
    if imgui.button("OBJ"):
        _import_obj_dialog(state.scene)
    imgui.same_line()
    imgui.text_disabled("RMB look + WASD/QE | MMB orbit | Shift+MMB pan")


def _draw_menus(state: EditorState) -> None:
    if imgui.begin_menu("File"):
        clicked, _ = imgui.menu_item("Save Scene", "Ctrl+S", False, True)
        if clicked:
            _save_dialog(state.scene)
        clicked, _ = imgui.menu_item("Load Scene", "Ctrl+O", False, True)
        if clicked:
            _replace_scene_from_dialog(state)
        clicked, _ = imgui.menu_item("Import OBJ", "", False, True)
        if clicked:
            _import_obj_dialog(state.scene)
        imgui.separator()
        clicked, _ = imgui.menu_item("Quit", "", False, True)
        if clicked:
            hello_imgui.get_runner_params().app_shall_exit = True
        imgui.end_menu()

    if imgui.begin_menu("Add"):
        for entry in ENTRIES:
            clicked, _ = imgui.menu_item(entry.title, "", False, True)
            if clicked:
                _handle_add_action(state.scene, entry.action, state.orbit.target)
        imgui.end_menu()

    if imgui.begin_menu("Render"):
        clicked, _ = imgui.menu_item("Start Render", "F12", False, True)
        if clicked:
            _start_render(state)
        clicked, _ = imgui.menu_item("Cycle Quality", "", False, True)
        if clicked:
            state.quality = _next_quality(state.quality)
            _apply_quality_preset(state.scene, state.quality)
        clicked, selected = imgui.menu_item("Use GPU When Possible", "", state.use_gpu, True)
        if clicked:
            state.use_gpu = bool(selected)
        imgui.end_menu()


def _draw_popups_and_shortcuts(state: EditorState) -> None:
    _handle_shortcuts(state)
    if state.show_add_popup:
        imgui.open_popup("Add To Scene")
        state.show_add_popup = False
    _draw_add_popup(state)

    if state.render_window is not None:
        action = state.render_window.draw()
        if action == "close" and not state.render_window.is_running():
            state.render_window.cleanup()
            state.render_window = None


def _draw_viewport(state: EditorState) -> None:
    avail = imgui.get_content_region_avail()
    width = max(1, int(avail.x))
    height = max(1, int(avail.y))
    texture = state.preview.render(state.scene, state.orbit, width, height)
    imgui.image(imgui.ImTextureRef(texture), ImVec2(width, height), ImVec2(0, 1), ImVec2(1, 0))
    hovered = imgui.is_item_hovered()
    rect_min = imgui.get_item_rect_min()
    state.viewport_hovered = hovered
    _draw_imguizmo(state, rect_min, width, height)
    _handle_viewport_input(state, rect_min, width, height, hovered)


def _draw_imguizmo(state: EditorState, rect_min: ImVec2, width: int, height: int) -> None:
    if _GIZMO is None or state.scene.selected is None or not hasattr(state.scene.selected, "position"):
        return
    selected = state.scene.selected
    old_position = np.asarray(getattr(selected, "position"), dtype=np.float64).copy()
    view, projection = matrices_for_imguizmo(state.preview)
    matrix = _object_matrix_for_gizmo(selected)
    _GIZMO.begin_frame()
    _GIZMO.set_drawlist()
    _GIZMO.set_rect(float(rect_min.x), float(rect_min.y), float(width), float(height))
    _GIZMO.manipulate(
        _GIZMO.Matrix16(view),
        _GIZMO.Matrix16(projection),
        state.gizmo_operation or _GIZMO.OPERATION.translate,
        _GIZMO.MODE.world,
        matrix,
    )
    if _GIZMO.is_using():
        components = _GIZMO.decompose_matrix_to_components(matrix)
        new_position = np.asarray(components.translation.values, dtype=np.float64)
        getattr(selected, "position")[:] = new_position
        if isinstance(selected, SceneObject):
            selected.rotation[:] = np.asarray(components.rotation.values, dtype=np.float64)
            selected.scale[:] = np.maximum(0.02, np.asarray(components.scale.values, dtype=np.float64))
        if isinstance(selected, Camera):
            selected.target[:] = selected.target + (new_position - old_position)


def _object_matrix_for_gizmo(item: object) -> object:
    pos = np.asarray(getattr(item, "position"), dtype=np.float64)
    scale = np.asarray(getattr(item, "scale", np.ones(3, dtype=np.float64)), dtype=np.float64)
    rotation = _rotation_matrix(np.asarray(getattr(item, "rotation", np.zeros(3, dtype=np.float64)), dtype=np.float64))
    linear = rotation @ np.diag(scale)
    values = [
        float(linear[0, 0]), float(linear[1, 0]), float(linear[2, 0]), 0.0,
        float(linear[0, 1]), float(linear[1, 1]), float(linear[2, 1]), 0.0,
        float(linear[0, 2]), float(linear[1, 2]), float(linear[2, 2]), 0.0,
        float(pos[0]), float(pos[1]), float(pos[2]), 1.0,
    ]
    return _GIZMO.Matrix16(values)


def _set_gizmo_operation(state: EditorState, operation: str) -> None:
    if _GIZMO is None:
        return
    if operation == "rotate":
        state.gizmo_operation = _GIZMO.OPERATION.rotate
    elif operation == "scale":
        state.gizmo_operation = _GIZMO.OPERATION.scale
    else:
        state.gizmo_operation = _GIZMO.OPERATION.translate


def _rotation_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(euler_degrees, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


def _handle_viewport_input(state: EditorState, rect_min: ImVec2, width: int, height: int, hovered: bool) -> None:
    io = imgui.get_io()
    viewport = layout.Rect(0, 0, width, height)
    mouse = (
        int(io.mouse_pos.x - rect_min.x),
        int(io.mouse_pos.y - rect_min.y),
    )
    in_viewport = hovered or state.viewport_drag is not None or state.transforms.mode is not None
    if not in_viewport:
        return

    if hovered and abs(float(io.mouse_wheel)) > 1e-6:
        state.orbit.zoom(float(io.mouse_wheel))

    if state.transforms.mode is not None:
        if imgui.is_mouse_down(imgui.MouseButton_.left):
            state.transforms.drag(mouse, state.orbit)
        if imgui.is_mouse_released(imgui.MouseButton_.left):
            state.transforms.confirm()
        if imgui.is_mouse_clicked(imgui.MouseButton_.right):
            state.transforms.cancel()
        return

    if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.left):
        axis = state.transforms.pick_axis_at(mouse, state.scene, state.orbit, viewport)
        if axis is not None and state.scene.selected is not None:
            state.transforms.begin("move", state.scene.selected, mouse)
            state.transforms.axis = axis
        elif _GIZMO is None or not _GIZMO.is_over():
            state.scene.selected = pick(mouse[0], mouse[1], viewport, state.orbit, state.scene)

    if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.middle):
        state.viewport_drag = "pan" if io.key_shift else "orbit"
    if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
        state.viewport_drag = "look"

    if state.viewport_drag == "orbit" and imgui.is_mouse_down(imgui.MouseButton_.middle):
        delta = io.mouse_delta
        if io.key_shift:
            state.orbit.pan(float(delta.x), float(delta.y))
        else:
            state.orbit.orbit(float(delta.x), float(delta.y))
    elif state.viewport_drag == "pan" and imgui.is_mouse_down(imgui.MouseButton_.middle):
        delta = io.mouse_delta
        state.orbit.pan(float(delta.x), float(delta.y))
    elif state.viewport_drag == "look" and imgui.is_mouse_down(imgui.MouseButton_.right):
        delta = io.mouse_delta
        state.orbit.look(float(delta.x), float(delta.y))
        _handle_viewport_keyboard(state.orbit, float(io.delta_time), io, True)

    if state.viewport_drag in {"orbit", "pan"} and not imgui.is_mouse_down(imgui.MouseButton_.middle):
        state.viewport_drag = None
    if state.viewport_drag == "look" and not imgui.is_mouse_down(imgui.MouseButton_.right):
        state.viewport_drag = None


def _handle_shortcuts(state: EditorState) -> None:
    if state.render_window is not None:
        return
    io = imgui.get_io()
    if io.want_text_input:
        return
    mouse = (int(io.mouse_pos.x), int(io.mouse_pos.y))

    if imgui.is_key_pressed(imgui.Key.escape) and state.transforms.mode is not None:
        state.transforms.cancel()
        return
    if imgui.is_key_pressed(imgui.Key.enter) and state.transforms.mode is not None:
        state.transforms.confirm()
        return
    for key_name, key in (("x", imgui.Key.x), ("y", imgui.Key.y), ("z", imgui.Key.z)):
        if state.transforms.mode is not None and imgui.is_key_pressed(key):
            state.transforms.handle_command(key_name, state.scene, mouse)
            return

    if imgui.is_key_pressed(imgui.Key.f12):
        _start_render(state)
        return
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.s):
        _save_dialog(state.scene)
        return
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.o):
        _replace_scene_from_dialog(state)
        return
    if io.key_shift and imgui.is_key_pressed(imgui.Key.a):
        state.show_add_popup = True
        return
    if imgui.is_key_pressed(imgui.Key.delete) or imgui.is_key_pressed(imgui.Key.x):
        _delete_selected(state.scene)
        return
    for key_name, key in (("g", imgui.Key.g), ("s", imgui.Key.s), ("r", imgui.Key.r)):
        if imgui.is_key_pressed(key):
            if key_name == "g":
                _set_gizmo_operation(state, "translate")
            elif key_name == "r":
                _set_gizmo_operation(state, "rotate")
            elif key_name == "s":
                _set_gizmo_operation(state, "scale")
            state.transforms.handle_command(key_name, state.scene, mouse)
            return


def _draw_outliner(state: EditorState) -> None:
    imgui.text("Scene")
    imgui.separator()
    if imgui.collapsing_header("Objects", imgui.TreeNodeFlags_.default_open):
        for index, obj in enumerate(state.scene.objects):
            imgui.push_id(f"object_{index}")
            changed, visible = imgui.checkbox("##visible", obj.visible)
            if changed:
                obj.visible = visible
            imgui.same_line()
            clicked, _ = imgui.selectable(f"{_icon(obj)} {obj.name}", state.scene.selected is obj)
            if clicked:
                state.scene.selected = obj
            imgui.pop_id()
    if imgui.collapsing_header("Lights", imgui.TreeNodeFlags_.default_open):
        for index, light in enumerate(state.scene.lights):
            imgui.push_id(f"light_{index}")
            clicked, _ = imgui.selectable(f"{_icon(light)} {getattr(light, 'name', light.type)}", state.scene.selected is light)
            if clicked:
                state.scene.selected = light
            imgui.pop_id()
    if imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open):
        clicked, _ = imgui.selectable("C Camera", state.scene.selected is state.scene.camera)
        if clicked:
            state.scene.selected = state.scene.camera


def _draw_properties(state: EditorState) -> None:
    selected = state.scene.selected
    if selected is None:
        imgui.text_disabled("No selection")
        return

    imgui.text_colored((0.38, 0.72, 1.0, 1.0), display_name(selected))
    imgui.separator()
    if isinstance(selected, SceneObject):
        _draw_object_properties(selected)
    elif isinstance(selected, Light):
        _draw_light_properties(selected)
    elif isinstance(selected, Camera):
        _draw_camera_properties(state.scene)


def _draw_object_properties(obj: SceneObject) -> None:
    _separator("Transform")
    changed, obj.name = imgui.input_text("Name", obj.name)
    changed, obj.visible = imgui.checkbox("Visible", obj.visible)
    _drag_vec3("Location", obj.position, 0.05)
    _drag_vec3("Rotation", obj.rotation, 0.25)
    _drag_vec3("Scale", obj.scale, 0.02, min_value=0.02)
    if isinstance(obj, Sphere):
        changed, value = imgui.drag_float("Radius", float(obj.radius), 0.02, 0.02, 100.0)
        if changed:
            obj.radius = max(0.02, value)
    if isinstance(obj, Plane):
        _drag_vec3("Normal", obj.normal, 0.02, normalize=True)
    if isinstance(obj, Mesh):
        imgui.text_disabled(f"Mesh: {obj.vertices.shape[0]} vertices, {obj.triangles.shape[0]} triangles")
        if obj.source_path:
            imgui.text_wrapped(f"Source: {obj.source_path}")

    _separator("Material")
    material_types = ["diffuse", "specular", "glass", "emissive"]
    labels = ["Diffuse", "Specular", "Glass", "Emissive"]
    current = material_types.index(getattr(obj.material, "type", "diffuse")) if getattr(obj.material, "type", "diffuse") in material_types else 0
    changed, current = imgui.combo("Type", current, labels)
    if changed:
        obj.material = _convert_material(obj.material, material_types[current])
    _draw_material(obj.material)


def _draw_light_properties(light: Light) -> None:
    _separator("Light")
    changed, light.name = imgui.input_text("Name", light.name)
    _drag_vec3("Position", light.position, 0.05)
    _drag_vec3("Color", light.color, 0.02, min_value=0.0)
    changed, value = imgui.drag_float("Intensity", float(light.intensity), 1.0, 0.0, 100000.0)
    if changed:
        light.intensity = max(0.0, value)
    if isinstance(light, DirectionalLight):
        _drag_vec3("Direction", light.direction, 0.02, normalize=True)
    if isinstance(light, AreaLight):
        _drag_vec3("Normal", light.normal, 0.02, normalize=True)
        changed, value = imgui.drag_float("Radius", float(light.radius), 0.02, 0.02, 1000.0)
        if changed:
            light.radius = max(0.02, value)


def _draw_camera_properties(scene: Scene) -> None:
    cam = scene.camera
    render = scene.render
    _separator("Camera")
    _drag_vec3("Position", cam.position, 0.05)
    _drag_vec3("Target", cam.target, 0.05)
    changed, cam.fov = imgui.slider_float("FOV", float(cam.fov), 5.0, 160.0)
    changed, cam.aperture = imgui.drag_float("Aperture", float(cam.aperture), 0.01, 0.0, 100.0)
    focus_value = 0.0 if cam.focus_distance is None else float(cam.focus_distance)
    changed, focus_value = imgui.drag_float("Focus Dist", focus_value, 0.05, 0.0, 10000.0)
    if changed:
        cam.focus_distance = None if focus_value <= 0.0 else focus_value

    _separator("Render")
    _combo_attr(render, "render_mode", "Mode", ("direct", "path"), ("Direct", "Path"))
    _combo_attr(render, "background_mode", "BG Mode", ("solid", "gradient", "environment"), ("Solid", "Gradient", "Environment"))
    _input_int_attr(render, "width", "Width", 1)
    _input_int_attr(render, "height", "Height", 1)
    _input_int_attr(render, "samples", "Samples", 1)
    _input_int_attr(render, "max_bounces", "Bounces", 0)
    _input_int_attr(render, "area_light_samples", "Area SPP", 1)
    changed, render.exposure = imgui.drag_float("Exposure", float(render.exposure), 0.01, 0.001, 1000.0)
    _drag_vec3("Background", render.background_color, 0.01, min_value=0.0)
    changed, render.environment_path = imgui.input_text("Env Map", render.environment_path)
    if imgui.button("Browse Environment"):
        path = _file_dialog(save=False, env=True)
        if path:
            render.environment_path = path
            render.background_mode = "environment"


def _draw_material(material: Material) -> None:
    if isinstance(material, DiffuseMaterial):
        _color3("Color", material.color)
        changed, material.roughness = imgui.slider_float("Roughness", float(material.roughness), 0.0, 1.0)
    elif isinstance(material, SpecularMaterial):
        _color3("Color", material.color)
        changed, material.roughness = imgui.slider_float("Roughness", float(material.roughness), 0.0, 1.0)
        changed, material.ior = imgui.drag_float("IOR", float(material.ior), 0.01, 1.0, 5.0)
    elif isinstance(material, GlassMaterial):
        _color3("Tint", material.tint)
        _color3("Absorption", material.absorption_color)
        changed, material.roughness = imgui.slider_float("Roughness", float(material.roughness), 0.0, 1.0)
        changed, material.ior = imgui.drag_float("IOR", float(material.ior), 0.01, 1.0, 5.0)
    elif isinstance(material, EmissiveMaterial):
        _color3("Color", material.color)
        changed, material.strength = imgui.drag_float("Strength", float(material.strength), 0.05, 0.0, 100000.0)


def _draw_add_popup(state: EditorState) -> None:
    imgui.set_next_window_size(ImVec2(760, 460), imgui.Cond_.appearing)
    visible, _ = imgui.begin_popup_modal("Add To Scene", None, imgui.WindowFlags_.no_saved_settings)
    if not visible:
        return
    imgui.text("Add To Scene")
    imgui.same_line()
    imgui.text_disabled("Esc closes")
    imgui.separator()
    for group in ("Geometry", "Lights"):
        _separator(group)
        columns = 2
        for index, entry in enumerate([entry for entry in ENTRIES if entry.group == group]):
            if index % columns:
                imgui.same_line()
            imgui.push_id(entry.action)
            if imgui.button(entry.title, ImVec2(210, 34)):
                _handle_add_action(state.scene, entry.action, state.orbit.target)
                imgui.close_current_popup()
            if imgui.is_item_hovered():
                imgui.set_tooltip(entry.detail)
            imgui.pop_id()
    imgui.separator()
    if imgui.button("Close"):
        imgui.close_current_popup()
    imgui.end_popup()


def _drag_vec3(
    label: str,
    arr: np.ndarray,
    speed: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    normalize: bool = False,
) -> None:
    v_min = 0.0 if min_value is None else min_value
    v_max = 0.0 if max_value is None else max_value
    changed, values = imgui.drag_float3(label, [float(arr[0]), float(arr[1]), float(arr[2])], speed, v_min, v_max)
    if changed:
        values_arr = np.asarray(values, dtype=np.float64)
        if min_value is not None:
            values_arr = np.maximum(min_value, values_arr)
        if max_value is not None:
            values_arr = np.minimum(max_value, values_arr)
        if normalize:
            length = float(np.linalg.norm(values_arr))
            if length > 1e-12:
                values_arr = values_arr / length
        arr[:] = values_arr


def _color3(label: str, arr: np.ndarray) -> None:
    changed, values = imgui.color_edit3(label, [float(np.clip(arr[0], 0.0, 1.0)), float(np.clip(arr[1], 0.0, 1.0)), float(np.clip(arr[2], 0.0, 1.0))])
    if changed:
        arr[:] = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _combo_attr(obj: object, attr: str, label: str, values: tuple[str, ...], labels: tuple[str, ...]) -> None:
    current_value = str(getattr(obj, attr, values[0])).strip().lower()
    current = values.index(current_value) if current_value in values else 0
    changed, current = imgui.combo(label, current, labels)
    if changed:
        setattr(obj, attr, values[current])


def _input_int_attr(obj: object, attr: str, label: str, min_value: int) -> None:
    changed, value = imgui.input_int(label, int(getattr(obj, attr)))
    if changed:
        setattr(obj, attr, max(min_value, int(value)))


def _separator(label: str) -> None:
    if hasattr(imgui, "separator_text"):
        imgui.separator_text(label)
    else:
        imgui.separator()
        imgui.text_disabled(label)


def _convert_material(old: Material, material_type: str) -> Material:
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


def _handle_viewport_keyboard(orbit: OrbitCamera, dt: float, io: object, require_look_mode: bool) -> None:
    if not require_look_mode:
        return
    right = 0.0
    up = 0.0
    forward = 0.0
    if imgui.is_key_down(imgui.Key.d):
        right += 1.0
    if imgui.is_key_down(imgui.Key.a):
        right -= 1.0
    if imgui.is_key_down(imgui.Key.e):
        up += 1.0
    if imgui.is_key_down(imgui.Key.q):
        up -= 1.0
    if imgui.is_key_down(imgui.Key.w):
        forward += 1.0
    if imgui.is_key_down(imgui.Key.s):
        forward -= 1.0
    if right == 0.0 and up == 0.0 and forward == 0.0:
        return
    speed = max(0.5, orbit.distance * 0.65)
    if getattr(io, "key_shift", False):
        speed *= 3.0
    if getattr(io, "key_ctrl", False):
        speed *= 0.25
    orbit.move(right, up, forward, speed * max(0.0, dt))


def _status_text(state: EditorState) -> str:
    if state.transforms.mode is not None:
        return state.transforms.status()
    if state.viewport_drag == "look":
        return "RMB look + WASD/QE move | Shift fast | Ctrl slow"
    if state.viewport_drag == "orbit":
        return "MMB orbit | Shift+MMB pan | Wheel zoom"
    if state.viewport_drag == "pan":
        return "MMB pan | Wheel zoom"
    selected = display_name(state.scene.selected) if state.scene.selected is not None else "None"
    return f"{state.transforms.status()} | Selected: {selected} | GPU {'on' if state.use_gpu else 'off'} | {state.quality.title()}"


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


def _start_render(state: EditorState) -> None:
    if state.render_window is not None and state.render_window.is_running():
        return
    if state.render_window is not None:
        state.render_window.cleanup()
    state.render_window = ImguiRenderWindow(state.scene, use_gpu=state.use_gpu)
    state.render_window.start()


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


def _replace_scene_from_dialog(state: EditorState) -> None:
    loaded = _load_dialog()
    if loaded is not None:
        state.scene = loaded
        state.quality = _quality_for_scene(state.scene)
        state.transforms.confirm()


def _load_dialog() -> Scene | None:
    path = _file_dialog(save=False)
    if not path:
        return None
    return load_scene(path)


def _file_dialog(save: bool, obj: bool = False, env: bool = False) -> str:
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
            if obj:
                filetypes = [("Wavefront OBJ", "*.obj")]
            elif env:
                filetypes = [
                    ("Environment Images", "*.hdr *.pic *.exr *.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("HDR Images", "*.hdr *.pic *.exr"),
                    ("All Files", "*.*"),
                ]
            else:
                filetypes = [("PyTrace Scene", "*.json"), ("JSON", "*.json")]
            path = filedialog.askopenfilename(filetypes=filetypes)
        root.destroy()
        return str(path or "")
    except Exception:
        return ""


def _icon(item: object) -> str:
    if isinstance(item, Sphere):
        return "S"
    if isinstance(item, Plane):
        return "P"
    if isinstance(item, Mesh):
        return "M"
    if isinstance(item, DirectionalLight):
        return "D"
    if isinstance(item, Light):
        return "L"
    if isinstance(item, Camera):
        return "C"
    return "-"
