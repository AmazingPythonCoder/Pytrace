# PyTrace

A Blender-style 3D scene editor with a from-scratch raytracer built in Python.

Place objects, lights, and a camera in a real-time OpenGL viewport — then hit `F12` to render everything with a proper raytracer: real reflections, refractions, soft shadows, and anti-aliasing.

![render preview placeholder](docs/render_preview.png)

---

## Features

- Real-time 3D viewport (PyOpenGL) with orbit/pan/zoom
- Blender-style controls: `G` grab, `R` rotate, `S` scale, axis constraints
- Add spheres, planes, cubes, point lights, area lights
- Material editor: diffuse, specular, glass, emissive
- From-scratch raytracer: shadows, reflections, refraction, anti-aliasing
- Multithreaded rendering with live progress
- Scene save/load as JSON
- Outputs `.png` at any resolution

---

## Quickstart

```bash
git clone https://github.com/yourname/pytrace
cd pytrace
pip install -r requirements.txt
python src/main.py
```

To render the default scene headlessly (no editor):

```bash
python src/main.py --render scenes/default.json --output output/render.png
```

---

## Controls

| Input | Action |
|---|---|
| Middle mouse drag | Orbit |
| Shift + MMB drag | Pan |
| Scroll wheel | Zoom |
| Left click | Select object |
| `G` / `R` / `S` | Grab / Rotate / Scale |
| `G X` / `G Y` / `G Z` | Constrain to axis |
| `Shift + A` | Add object menu |
| `X` | Delete selected |
| `F12` | Render |
| `Ctrl + S` | Save scene |
| `Ctrl + O` | Open scene |

---

## Requirements

```
pygame>=2.5.0
PyOpenGL>=3.1.7
numpy>=1.26.0
Pillow>=10.0.0
```

---

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full breakdown of every module.

## Raytracer Details

See [RAYTRACER.md](RAYTRACER.md) for the math behind intersections, shading, and acceleration.

## Editor Details

See [EDITOR.md](EDITOR.md) for the viewport, UI panels, and gizmo system.

## Build Roadmap

See [ROADMAP.md](ROADMAP.md) for the recommended week-by-week build order.
