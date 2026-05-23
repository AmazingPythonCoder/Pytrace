# Getting Started

## Requirements

- Python 3.10+.
- Dependencies listed in `requirements.txt`, including numpy, Pillow/imageio, pygame, PyOpenGL, and the CUDA/Numba stack used by the renderer.

## Install

```bash
cd pytrace
pip install -r requirements.txt
```

## Open The Editor

```bash
python src/main.py
```

The editor opens with the default scene. Use the viewport to orbit/pan/zoom, hold right mouse and use `W/A/S/D` plus `Q/E` to fly around, select from the viewport or outliner, edit properties, open the full-screen Add menu for objects/lights, cycle render quality from the toolbar, save/load JSON, and press `F12` to render.

## First Headless Render

```bash
python src/main.py --headless --level low
```

Headless output:

- Primary image: `output/render.png`
- History copy: `output/history/render_YYYYMMDD_HHMMSS_<level>.png`
- Gallery index: `output/history/history.js`

Editor `F12` renders also save a timestamped history copy and update the same gallery index automatically when the render finishes.

## Load Example Scenes

```bash
python src/main.py --scene scenes/default.json
python src/main.py --scene scenes/cornell_box.json
```

Use `--headless` with either scene path to render from the command line.

## Fast Preview

| Preset | Resolution | Samples/px | Max bounces |
|--------|------------|------------|-------------|
| `low` | 800x450 | 32 | 8 |
| `med` | 1280x720 | 128 | 10 |
| `high` | 1920x1080 | 384 | 12 |
| `ultra` | 3840x2160 | 768 | 16 |

## View Render History

Open `gallery.html` in a browser. It loads `output/history/history.js` and displays past renders with metadata.

## Verification Scripts

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

Run commands from the project root so `src` imports resolve correctly.
