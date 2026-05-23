# PyTrace Documentation

PyTrace is a Python ray tracer with a Pygame/OpenGL scene editor, JSON scene I/O, CPU tile rendering, and optional CUDA rendering for supported scenes.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Getting started](getting-started.md) | Install dependencies, open the editor, run your first render |
| [CLI reference](cli.md) | Command-line flags, quality presets, parallelism |
| [Architecture](architecture.md) | Modules, data flow, and project layout |
| [Scene model](scene-model.md) | Objects, materials, lights, camera, default scene |
| [Raytracer](raytracer.md) | Intersection, shading, sampling, tonemapping, performance |
| [Output & gallery](output-and-gallery.md) | PNG output, render history, `gallery.html` |

## Current Capabilities

- **Editor**: OpenGL viewport with orbit/pan/zoom and RMB-look fly navigation, outliner, properties panel, full-screen Add menu, transform modes, toolbar quality cycling, save/load, and live CPU tile render preview.
- **Geometry**: spheres, planes, triangle meshes, cubes, and OBJ import.
- **Materials**: diffuse, specular/mirror, glass, and emissive.
- **Lighting**: point, disk area, and directional lights.
- **Camera/rendering**: anti-aliasing, recursive reflections/refraction, depth of field, direct/basic path modes, solid/gradient/LDR+HDR environment backgrounds.
- **Performance**: Numba JIT for hot CPU paths, multiprocessing over 64x64 tiles, and CUDA for compatible scenes.
- **Output**: PNG via Pillow, timestamped history copies, and a browser gallery.

## Quick Start

```bash
pip install -r requirements.txt
python src/main.py
```

Use `python src/main.py --headless --level low` for a fast PNG render without opening the editor.
