# Architecture

PyTrace is split into three independent layers that communicate through a shared scene graph. They never import each other — only the scene layer sits in the middle.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   EDITOR    │────▶│    SCENE     │◀────│  RAYTRACER   │
│  (OpenGL)   │     │   GRAPH      │     │  (CPU/numpy) │
└─────────────┘     └──────────────┘     └──────────────┘
   real-time           shared data          offline render
```

---

## Directory Layout

```
pytrace/
├── src/
│   ├── main.py              # Entry point. Parses args, boots editor or headless render.
│   │
│   ├── scene/               # ── SCENE LAYER ──────────────────────────────────────────
│   │   ├── scene.py         # Scene: holds lists of objects, lights, camera, render cfg
│   │   ├── objects.py       # SceneObject base class. Sphere, Plane, Mesh subclasses.
│   │   ├── materials.py     # Material dataclasses: Diffuse, Specular, Glass, Emissive
│   │   ├── lights.py        # PointLight, DirectionalLight, AreaLight dataclasses
│   │   ├── camera.py        # Camera: position, target, fov, near/far
│   │   └── serializer.py    # scene → JSON, JSON → scene
│   │
│   ├── editor/              # ── EDITOR LAYER ─────────────────────────────────────────
│   │   ├── app.py           # Main editor loop. Owns the pygame window and GL context.
│   │   ├── viewport.py      # OpenGL 3D viewport. Draws grid, objects, lights, camera.
│   │   ├── orbit_camera.py  # Editor camera: orbit, pan, zoom. Separate from scene cam.
│   │   ├── gizmos.py        # Transform gizmos: colored axis handles for G/R/S.
│   │   ├── selection.py     # Ray-cast picking: click → find hit object.
│   │   ├── outliner.py      # Left panel: list of scene objects, click to select.
│   │   ├── properties.py    # Right panel: transform fields, material editor.
│   │   ├── toolbar.py       # Top bar: Add menu, Render button, Save/Load.
│   │   └── render_window.py # Overlay shown during/after render. Progress + image.
│   │
│   ├── raytracer/           # ── RAYTRACER LAYER ──────────────────────────────────────
│   │   ├── renderer.py      # Entry point. Spawns tile workers, assembles image.
│   │   ├── ray.py           # Ray dataclass: origin (vec3), direction (vec3).
│   │   ├── intersect.py     # Ray vs sphere, plane, triangle. Returns HitRecord.
│   │   ├── shading.py       # Given a HitRecord + scene: compute final pixel color.
│   │   ├── bvh.py           # BVH tree: build from scene, fast ray traversal.
│   │   ├── sampling.py      # Random sample generators: hemisphere, cosine, disk.
│   │   └── tonemapping.py   # HDR float → 8-bit sRGB. Reinhard + gamma.
│   │
│   └── math/
│       ├── vec3.py          # Vec3 class. Thin numpy wrapper. dot, cross, normalize.
│       └── transform.py     # 4×4 matrix utils: translate, rotate, scale, look_at.
│
├── scenes/
│   ├── default.json         # Classic 3-sphere showcase scene.
│   └── cornell_box.json     # Cornell box: 5 colored walls + 2 objects + ceiling light.
│
├── output/
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Data Flow

### Editor → Render

When the user hits `F12`:

1. `editor/app.py` grabs the current `Scene` object
2. Passes it directly to `raytracer/renderer.py`
3. Renderer reads objects, lights, camera — no conversion needed
4. Renderer writes pixel data to a numpy array
5. `render_window.py` polls for progress and displays tiles as they finish

### Scene serialization

```
scene.py  ──serialize()──▶  dict  ──json.dump()──▶  file.json
scene.py  ◀──deserialize()──  dict  ◀──json.load()──  file.json
```

All scene objects implement `to_dict()` and `from_dict()`. The serializer walks the scene graph and calls them.

---

## Key Design Decisions

**No shared OpenGL state in the raytracer.**
The raytracer is pure numpy/Python. It never touches pygame or OpenGL. This means it can run headlessly from the command line without a display.

**Editor camera ≠ Scene camera.**
The editor has its own orbit camera for navigating the viewport. The scene has its own camera object (position, FOV, etc.) that the raytracer uses. They are completely separate. The editor draws the scene camera as a visible object (a small frustum wireframe) so you can see and move it.

**Scene objects are dumb dataclasses.**
They hold position, material, geometry parameters — nothing else. No OpenGL buffers, no raytracer state. The editor and raytracer each know how to draw/intersect a `Sphere(center, radius, material)` without the sphere knowing anything about rendering.

**Gizmos live in the editor only.**
The scene never knows about gizmos. When the user drags a gizmo, the editor updates `object.position` (or rotation, scale) directly. The raytracer sees the updated value on next render.

---

## Threading Model

The editor runs on the main thread (required by OpenGL + pygame on most platforms).

When rendering starts:
- A `multiprocessing.Pool` is created with `cpu_count()` workers
- The image is divided into N×N pixel tiles
- Each tile is a separate task: `render_tile(scene_data, tile_bounds) → pixel_array`
- Scene data is serialized to a dict before being sent to workers (avoids pickling issues with numpy arrays inside objects)
- The main thread polls for completed tiles every frame and blits them to the render window

Progress is reported as `tiles_done / tiles_total`.

---

## Extension Points

| What to add | Where it goes |
|---|---|
| New object type (cylinder, torus) | `scene/objects.py` + `raytracer/intersect.py` + `editor/viewport.py` |
| New material type (SSS, hair) | `scene/materials.py` + `raytracer/shading.py` |
| New light type (spot, sky) | `scene/lights.py` + `raytracer/shading.py` + `editor/viewport.py` |
| OBJ importer | `scene/objects.py` (Mesh class) + import dialog in `editor/toolbar.py` |
| Denoiser | Post-process step in `raytracer/renderer.py` after tiles assemble |
| Path tracing | Replace `raytracer/shading.py` with a path tracer — scene layer unchanged |
