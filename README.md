# PyTrace

A Python path tracer that renders a built-in showcase scene to PNG. The interactive scene editor is not implemented yet; running the program always performs a headless render.

## Quick start

From the project root:

```bash
pip install numpy Pillow numba
python src/main.py --headless
```

The image is written to `output/render.png`. Use `--level low|med|high|ultra` for resolution and sample presets, or override with `--width`, `--height`, `--samples`, and `--bounces`.

Example:

```bash
python src/main.py --headless --level high --output output/render.png
```

Smoke tests (small renders):

```bash
python scripts/verify_step1.py
python scripts/verify_step2.py
python scripts/verify_step3.py
```

## What works today

### Rendering pipeline

- Pinhole camera with position, look-at target, up vector, and vertical field of view.
- Primary rays with per-pixel jitter for anti-aliasing (configurable samples per pixel).
- Recursive ray tracing up to a configurable bounce limit.
- Tile-based rendering (64×64 tiles) with optional multiprocessing across CPU cores.
- HDR float framebuffer, then exposure, luminance-based Reinhard tone mapping, gamma, and PNG export via Pillow.
- Terminal progress bar with elapsed time and ETA.

### Geometry

- **Spheres** — analytic intersection.
- **Planes** — infinite planes (used for ground and room walls).

Closest-hit selection returns hit distance, position, shading normal (front-face corrected), and material index. Shadow rays use a separate any-hit test.

### Materials

| Type | Behavior |
|------|----------|
| **Diffuse** | Direct lighting only (no reflection continuation). Roughness controls wrap lighting and optional Blinn–Phong specular highlights. |
| **Specular** | Reflective metal-like surfaces with Schlick Fresnel, optional roughness (fuzzed reflection direction), and stored IOR (used in material packing). |
| **Glass** | Refraction and reflection with Schlick probability, index of refraction, tint, and Beer–Lambert-style absorption on transmit. |

### Lighting

- **Disk area lights** — random point sampling on the disk for soft shadows (configurable samples per light).
- **Point lights** — supported in the scene-to-GPU-style flat arrays (the default scene uses area lights only).
- Inverse-square falloff, shadow tests, and distance-limited shadow rays.

Diffuse shading also adds a simple sky-tinted ambient term. Missed rays use a vertical sky gradient.

### Default scene

`Scene.default()` builds a furnished room:

- Seven spheres (diffuse red with procedural bumps/noise, glass, mirror, gold, green, small white, bronze).
- Six planes (ground with procedural checkerboard, four colored walls, ceiling).
- Three warm/cool/rim disk area lights.
- Camera aimed at the cluster from `(0, 2, 8)` with 38° FOV.

Scenes are defined in code (dataclasses); there is no file loader or editor UI yet.

### Output and gallery

- Each headless run saves `output/render.png` and a timestamped copy under `output/history/`.
- `output/history/history.js` is updated with render metadata (resolution, samples, level, timestamp).
- `gallery.html` at the project root is a standalone page that reads that history and displays past renders.

### Performance

Hot paths (`intersect`, `shading`, `vec3`, sampling`) are compiled with **Numba** (`@njit`). The first run pays a JIT warmup cost; later tiles and runs are faster.

Intersection uses a **BVH** over all primitives (spheres and room planes with conservative finite bounds). Build time is once per render in `build_render_context`; traversal is in `src/raytracer/bvh.py`.

**GPU (default):** Renders use **CUDA** via Numba when a GPU is available (`src/raytracer/gpu/`). Requires an NVIDIA GPU, driver, and CUDA toolkit so this succeeds:

```bash
python -c "from numba import cuda; print(cuda.is_available())"
```

Use **`--gpu false`** for the original CPU tile renderer (multiprocessing). The first GPU run compiles CUDA kernels and may take extra time.

```bash
python src/main.py --headless --level low
python src/main.py --headless --level low --gpu false
python scripts/verify_gpu.py
```

Benchmark (CPU, after warmup):

```bash
python scripts/bench_render.py
```

### Project layout (implementation)

```
src/
  main.py              CLI entry (--headless)
  math/vec3.py         3D vector helpers (Numba)
  scene/               Scene, camera, materials, objects, lights
  raytracer/           Camera frame, intersection, shading, renderer, BVH, gpu/
scripts/               verify_step1.py … verify_step3.py
output/                Renders and history/
gallery.html           Browser gallery for render history
```

## CLI options (implemented)

| Flag | Purpose |
|------|---------|
| `--headless` | Render to PNG (implicit; editor falls back to this) |
| `--output PATH` | PNG path (default `output/render.png`) |
| `--level low\|med\|high\|ultra` | Presets: 800×450 / 32 spp, 1280×720 / 128 spp, 1920×1080 / 384 spp, 3840×2160 / 768 spp |
| `--width`, `--height`, `--samples`, `--bounces` | Override preset values |
| `--gpu [BOOL]` | Use CUDA when available (default: true). `--gpu false` for CPU |
| `--workers N` | Process pool size (CPU only) |
| `--no-parallel` | Single-threaded CPU render |
| `--debug-workers` | Print worker PIDs and tile completion hints |

## Not implemented yet

- Interactive scene editor (launching without `--headless` prints a message and renders anyway).
- Loading or saving scenes from disk.
- Meshes, boxes, textures, image-based lighting, or denoising.
- GUI live preview.
