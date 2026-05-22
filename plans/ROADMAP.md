# Roadmap

Build order designed so you have something working at the end of every week. Never spend more than a week without a visible result.

---

## Week 1 — Core Raytracer (no editor, no window)

**Goal:** render a hard-coded scene to a PNG file from the command line.

**What to build:**
- `math/vec3.py` — Vec3 class, dot, cross, normalize, reflect
- `scene/` — Sphere, Plane, DiffuseMaterial, PointLight, Camera, Scene (hard-coded, no JSON yet)
- `raytracer/ray.py` — Ray dataclass
- `raytracer/intersect.py` — ray–sphere and ray–plane
- `raytracer/shading.py` — Lambertian diffuse + shadow rays only (no reflections yet)
- `raytracer/renderer.py` — single-threaded render loop, writes PNG via Pillow

**End of week result:**
```bash
python src/main.py --headless
# → output/render.png  (matte spheres, hard shadows, no reflections)
```

**Don't worry about:**
- The editor
- Anti-aliasing
- Reflections or glass
- Performance

---

## Week 2 — Reflections + Glass + Anti-aliasing

**Goal:** render the classic raytracer showcase scene.

**What to build:**
- `raytracer/shading.py` — add specular reflection (recursive `trace()`)
- `raytracer/shading.py` — add glass refraction (Snell's law, Schlick Fresnel)
- `raytracer/renderer.py` — add samples-per-pixel loop with random jitter
- `raytracer/tonemapping.py` — Reinhard tone map + gamma correction
- `scene/materials.py` — SpecularMaterial, GlassMaterial

**End of week result:**
A proper raytraced image: mirror sphere, glass sphere, colored matte sphere, ground plane, two lights. Looks like the classic raytracer demo image.

**Don't worry about:**
- Performance (still single-threaded)
- The editor

---

## Week 3 — Multiprocessing + BVH + Pygame Window

**Goal:** fast enough to iterate on, and a window to display the result in.

**What to build:**
- `raytracer/bvh.py` — AABB and BVH tree. Build once before render, traverse per ray.
- `raytracer/renderer.py` — tile-based multiprocessing with `Pool.imap_unordered`
- `editor/app.py` — minimal Pygame window. No 3D yet. Just display the rendered image.
- `editor/render_window.py` — show tiles as they complete, progress bar.

**End of week result:**
Press Enter → render window opens → image builds up tile by tile at full multicore speed.

---

## Week 4 — Real-Time OpenGL Viewport

**Goal:** a 3D viewport you can orbit around.

**What to build:**
- `editor/viewport.py` — OpenGL setup, draw spheres and planes as preview geometry
- `editor/orbit_camera.py` — orbit, pan, zoom with mouse
- `editor/grid.py` — XZ grid floor
- Light icons (small sphere at light position) and camera frustum wireframe

**End of week result:**
Open the app → see the scene in 3D → orbit around it → press F12 → render window opens.

---

## Week 5 — Selection + Outliner + Properties Panel

**Goal:** click objects and edit them.

**What to build:**
- `editor/selection.py` — ray-cast picking on left click
- `editor/outliner.py` — left panel listing scene objects. Click = select.
- `editor/properties.py` — right panel with transform fields and material editor
- Transform fields write directly to `obj.position`, `obj.rotation`, `obj.scale`
- Material dropdowns update `obj.material` type; fields update material params

**End of week result:**
Click a sphere → it highlights → right panel shows its material → change color → F12 → rendered image uses the new color.

---

## Week 6 — Add/Delete + Gizmos + Save/Load

**Goal:** a fully functional editor workflow.

**What to build:**
- `editor/gizmos.py` — colored axis move handles on selected object
- `Shift + A` add menu — spawns sphere, plane, cube, point light, area light at origin
- `X` to delete selected
- `G`, `R`, `S` keyboard modes with axis constraints
- `scene/serializer.py` — `to_dict()` / `from_dict()` for all scene objects
- `Ctrl + S` / `Ctrl + O` — save/load with `tkinter.filedialog`

**End of week result:**
Full editor workflow: open app → orbit scene → click object → move it with G → change material → add a light → save scene → F12 → render.

---

## Stretch Goals (no order)

These can be added any time after Week 6. Each is self-contained.

### OBJ Importer
Load any `.obj` file as a `Mesh` object. The raytracer already handles triangle meshes via Möller–Trumbore; just need a parser for the file format and a way to add it from the editor.

```python
# In Mesh.from_obj(path):
# Parse "v x y z" lines → vertices
# Parse "f i j k" lines → triangle indices
# Parse "vn x y z" lines → vertex normals
```

Add "Import OBJ" to the File menu.

### Depth of Field
Simulate a real camera lens. The camera has an `aperture` radius and `focus_distance`. Instead of a single pinhole ray per pixel, sample a random point on the lens disk and adjust the ray direction so everything at `focus_distance` is in focus.

```python
# In camera.generate_ray():
if self.aperture > 0:
    lens_sample = random_in_unit_disk() * self.aperture
    offset = u * lens_sample[0] + v * lens_sample[1]
    origin    = self.position + offset
    direction = normalize(focus_point - origin)
```

### Environment Map (HDRI Sky)
Load a `.hdr` equirectangular image. When a ray misses all objects, instead of returning a flat background color, sample the environment map:

```python
def background_color(ray):
    phi   = atan2(ray.direction[2], ray.direction[0])
    theta = acos(ray.direction[1])
    u = phi / (2*pi) + 0.5
    v = theta / pi
    return hdri_image.sample(u, v)
```

### Dear ImGui UI
Replace the hand-rolled pygame panels with [pyimgui](https://pyimgui.readthedocs.io/). Gives you proper sliders, color pickers, dockable windows, drag-and-drop. Significant improvement in editor feel.

```bash
pip install imgui[pygame]
```

### Path Tracing Mode
Replace direct lighting in `shading.py` with a full path tracer. Instead of explicit shadow rays to lights:

```python
# At each hit point: scatter a random ray in the hemisphere
scatter_dir = hit.normal + random_unit_vector()
return material.color * trace(Ray(hit.point, scatter_dir), scene, depth+1)
```

Requires high sample counts (256+) but produces global illumination: color bleeding, ambient occlusion, indirect light bouncing between surfaces. The classic Cornell box renders correctly in this mode.

### GPU Acceleration
Port the inner loop to [Numba](https://numba.readthedocs.io/) with `@cuda.jit`. The tile worker loop is a perfect fit for GPU parallelism. Expect 10–50× speedup on a modern GPU.

---

## What the Finished Repo Looks Like on GitHub

```
pytrace/
├── README.md          ← GIF of the editor + final render side by side
├── ARCHITECTURE.md
├── RAYTRACER.md
├── EDITOR.md
├── SCENE.md
├── ROADMAP.md
├── src/
│   ├── editor/
│   ├── raytracer/
│   ├── scene/
│   ├── math/
│   └── main.py
├── scenes/
│   ├── default.json
│   └── cornell_box.json
└── output/
    └── render.png     ← the final showcase render, committed to the repo
```

The README leads with a screen recording of the editor in action and the rendered output next to it. That combination — interactive editor + photorealistic output — is what makes someone star the repo.
