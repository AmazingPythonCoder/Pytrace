# PyTrace

A Python path tracer with an imgui-bundle/GLFW/OpenGL scene editor. Build scenes interactively, save/load JSON, then render with CPU tiles or CUDA when the current scene uses GPU-supported features.

## Quick Start

From the project root:

```bash
pip install -r requirements.txt
python src/main.py
```

Launching without `--headless` opens the editor. Use `--headless` to render directly to `output/render.png`.

```bash
python src/main.py --headless --level low --output output/render.png
python src/main.py --scene scenes/default.json
python src/main.py --headless --scene scenes/cornell_box.json --gpu false
```

## What Works

### Editor

- Docked imgui-bundle editor with a real-time OpenGL viewport, orbit, pan, zoom, RMB-look WASD/QE fly navigation, grid, preview geometry, lights, camera frustum, selection outlines, and ImGuizmo transform handles.
- Outliner and properties panels for objects, lights, camera, materials, render settings, depth of field, background mode, and environment path.
- Add modal/menu for spheres, planes, cubes, point lights, directional lights, area lights, and OBJ import.
- Selection from the viewport or outliner, `G`/`R`/`S` transform modes with optional `X/Y/Z` constraints, `X`/Delete removal, `Ctrl+S`/`Ctrl+O` scene save/load, toolbar quality cycling, and `F12` rendering.
- Render overlay with cancel/save controls and live CPU tile preview.

### Rendering

- Spheres, planes, and triangle meshes with BVH traversal.
- Diffuse, specular, glass, and emissive materials.
- Point, disk area, and directional lights.
- Direct lighting mode plus a simple path-tracing mode.
- Anti-aliasing, recursive reflections/refractions, soft shadows, depth of field, gradient/solid backgrounds, and optional equirectangular LDR/HDR environment images.
- HDR float framebuffer with exposure, Reinhard tone mapping, gamma correction, and PNG export.
- CUDA rendering for supported sphere/plane point/area-light direct scenes; editor-only features such as meshes, emissive materials, directional lights, DOF, environment maps, and path mode fall back to CPU.

### Scenes And Gallery

- `Scene.default()` builds the showcase room used by headless renders.
- `scenes/default.json` and `scenes/cornell_box.json` can be loaded in the editor or headless mode with `--scene`.
- Headless and editor F12 renders save a timestamped PNG under `output/history/` and update `output/history/history.js`.
- `gallery.html` displays render history from `output/history/history.js`.

## Verification

```bash
python scripts/verify_step1.py
python scripts/verify_step2.py
python scripts/verify_step3.py
python scripts/verify_editor_imports.py
python scripts/verify_selection.py
python scripts/verify_serializer.py
python scripts/verify_mesh_editor.py
python scripts/verify_render_callbacks.py
python scripts/verify_environment.py
```

GPU parity check, when CUDA is available:

```bash
python scripts/verify_gpu.py
```

## CLI Options

| Flag | Purpose |
|------|---------|
| `--headless` | Render to PNG instead of opening the editor |
| `--output PATH` | PNG path, default `output/render.png` |
| `--scene PATH` | Load a serialized scene JSON |
| `--level low\|med\|high\|ultra` | Render presets from 800x450 to 3840x2160 |
| `--width`, `--height`, `--samples`, `--bounces` | Override preset values |
| `--gpu [BOOL]` | Use CUDA when available, default true |
| `--workers N` | CPU process pool size, default available logical cores minus 1 |
| `--no-parallel` | Single-threaded CPU render |
| `--debug-workers` | Print worker PIDs and tile completion hints |

## Project Layout

```text
src/
  main.py              CLI/editor entry
  math/vec3.py         3D vector helpers
  scene/               Scene, camera, materials, objects, lights, JSON serializer
  editor/              imgui-bundle editor, docked UI, OpenGL preview renderer
  raytracer/           Camera frame, intersections, shading, renderer, BVH, CUDA backend
scripts/               Verification and benchmark helpers
scenes/                Example scene JSON files
output/                Renders and history
gallery.html           Browser gallery for render history
```

## Not Implemented Yet

- Denoising and full texture/material stacks.
- Production path-tracing features such as MIS, denoising, and material/texture importance sampling.
- GPU parity for the editor-only CPU fallback features.
