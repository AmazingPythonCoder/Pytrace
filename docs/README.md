# PyTrace documentation

PyTrace is a Python ray tracer with a built-in showcase scene. The project is structured toward a future Blender-style editor, but **today the editor is not implemented** — the CLI runs headless renders to PNG.

## Documentation index

| Document | Description |
|----------|-------------|
| [Getting started](getting-started.md) | Install dependencies, run your first render |
| [CLI reference](cli.md) | Command-line flags, quality presets, parallelism |
| [Architecture](architecture.md) | Modules, data flow, and project layout |
| [Scene model](scene-model.md) | Objects, materials, lights, camera, default scene |
| [Raytracer](raytracer.md) | Intersection, shading, sampling, tonemapping, performance |
| [Output & gallery](output-and-gallery.md) | PNG output, render history, `gallery.html` |

## Current capabilities

- **Headless rendering** of `Scene.default()` (Cornell-style room with multiple spheres)
- **Materials**: diffuse (with optional specular highlight), specular/mirror (roughness), glass (IOR, absorption)
- **Geometry**: spheres and infinite planes
- **Lighting**: point lights with inverse-square falloff and shadow rays
- **Effects**: multi-sample anti-aliasing, recursive reflections/refraction, procedural ground checkerboard and red-sphere detail
- **Performance**: Numba JIT for hot paths; optional multiprocessing over 64×64 tiles
- **Output**: PNG via Pillow; optional timestamped copies under `output/history/` and a browser gallery

## Planned (not in codebase yet)

Design notes and a build roadmap live under [`plans/`](../plans/). That folder describes the OpenGL editor, JSON scene I/O, BVH, area lights, and other future work — see [plans/README.md](../plans/README.md).

## Quick render

From the project root:

```bash
pip install -r requirements.txt
pip install numba
python src/main.py
```

Default behavior renders at the **high** preset (~1920×1080, 384 samples/pixel) and writes `output/render.png`. Use `--level low` for a fast preview (~2 seconds on a typical desktop).
