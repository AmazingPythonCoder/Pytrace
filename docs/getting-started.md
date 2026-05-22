# Getting started

## Requirements

- Python 3.10+ (3.11+ recommended)
- Dependencies listed in `requirements.txt`:
  - **numpy** — scene math and image buffers
  - **Pillow** — PNG export
- **numba** — used for JIT-compiled intersection and shading (`@njit` in `intersect.py`, `shading.py`, `vec3.py`, `sampling.py`). Install explicitly:

```bash
pip install numba
```

The editor stack described in `plans/` (pygame, PyOpenGL) is **not** required for rendering today.

## Install

```bash
cd pytrace
pip install -r requirements.txt
pip install numba
```

## First render

```bash
python src/main.py
```

Because the scene editor is not built yet, `main.py` always performs a **headless** render (it prints that the editor is not implemented, then renders anyway).

Output:

- Primary image: `output/render.png` (path overridable with `--output`)
- History copy: `output/history/render_YYYYMMDD_HHMMSS_<level>.png`
- Gallery index: `output/history/history.js` (updated after each render)

## Fast preview

Use the **low** preset for a quick check (~800×450, 32 samples/pixel):

```bash
python src/main.py --level low
```

Approximate targets (machine-dependent):

| Preset | Resolution | Samples/px | Max bounces | Typical time |
|--------|------------|------------|-------------|--------------|
| `low`  | 800×450    | 32         | 8           | ~2 s         |
| `med`  | 1280×720   | 128        | 10          | ~30 s        |
| `high` | 1920×1080  | 384        | 12          | ~2 min       |

## View render history

Open `gallery.html` in a browser (file URL or local static server). It loads `output/history/history.js` and displays past renders with metadata (timestamp, level, resolution, samples).

## Verification scripts

Incremental sanity checks live under `scripts/`:

```bash
python scripts/verify_step1.py   # vec3 math
python scripts/verify_step2.py   # intersection / rays
python scripts/verify_step3.py   # full render smoke test
```

Run from the project root so `src` imports resolve correctly.

## Project root convention

`src/main.py` adds the repository root to `sys.path`. Always invoke commands from the project root (or set `PYTHONPATH` to that directory).
