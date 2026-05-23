# CLI Reference

Entry point: `python src/main.py`

## Behavior

- `python src/main.py` opens the graphical editor.
- `python src/main.py --headless` renders a scene to PNG without opening a window.
- `--scene PATH` loads a serialized scene JSON for either editor or headless mode.
- Progress is printed as a terminal progress bar in headless mode.
- After headless rendering, the image is tonemapped and saved as PNG; a history PNG and `history.js` gallery entry are also written.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | off | Render to PNG instead of opening the editor |
| `--output PATH` | `output/render.png` | Output PNG path, relative to project root |
| `--scene PATH` | none | Load scene JSON |
| `--width N` | preset | Override image width |
| `--height N` | preset | Override image height |
| `--samples N` | preset | Samples per pixel |
| `--bounces N` | preset | Maximum ray recursion depth |
| `--level {low,med,high,ultra}` | `med` | Quality preset |
| `--gpu [BOOL]` | true | Use CUDA when available |
| `--workers N` | available logical CPUs minus 1 | Worker processes for CPU tile rendering |
| `--no-parallel` | off | Force single-threaded CPU render |
| `--debug-workers` | off | Print main PID and tile-completion hints |

CLI overrides apply after the level preset is applied.

## Quality Presets

| Level | Width | Height | Samples | Max bounces |
|-------|-------|--------|---------|-------------|
| `low` | 800 | 450 | 32 | 8 |
| `med` | 1280 | 720 | 128 | 10 |
| `high` | 1920 | 1080 | 384 | 12 |
| `ultra` | 3840 | 2160 | 768 | 16 |

## Examples

Open the editor:

```bash
python src/main.py
```

Headless preview:

```bash
python src/main.py --headless --level low
```

Load an example scene:

```bash
python src/main.py --scene scenes/cornell_box.json
```

Custom resolution and sample count:

```bash
python src/main.py --headless --level med --width 640 --height 360 --samples 16
```

Single-threaded CPU debug render:

```bash
python src/main.py --headless --level low --gpu false --no-parallel
```

## Parallelism

By default, CPU rendering uses `multiprocessing.Pool` with `default_workers(reserve=1)`, which means available logical CPUs minus one. The image is split into 64x64 pixel tiles; workers return tiles that are stitched into the final buffer and can be streamed to the editor render window.

Scenes using meshes, emissive materials, directional lights, depth of field, solid/environment backgrounds, or path mode automatically fall back to CPU when CUDA is requested. Environment maps can be ordinary LDR images or Radiance HDR files; path mode is intentionally basic and does not include MIS, denoising, or material/texture importance sampling.
