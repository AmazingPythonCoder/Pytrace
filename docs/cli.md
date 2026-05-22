# CLI reference

Entry point: `python src/main.py`

## Behavior

- Without a graphical editor, every invocation performs a **headless render** of `Scene.default()`.
- Progress is printed as a terminal progress bar (tiles in parallel mode, pixel batches in single-threaded mode).
- After rendering, the image is tonemapped and saved as PNG; a history entry may be appended (see [Output & gallery](output-and-gallery.md)).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | *(implicit)* | Acknowledged for future use; rendering is headless today regardless |
| `--output PATH` | `output/render.png` | Output PNG path, relative to project root |
| `--width N` | *(preset)* | Override image width |
| `--height N` | *(preset)* | Override image height |
| `--samples N` | *(preset)* | Samples per pixel (anti-aliasing) |
| `--bounces N` | *(preset)* | Maximum ray recursion depth |
| `--level {low,med,high}` | `high` | Quality preset (see below) |
| `--workers N` | CPU count − 1 | Worker processes for tile rendering |
| `--no-parallel` | off | Force single-threaded render (debugging) |
| `--debug-workers` | off | Print main PID and tile-completion hints |

CLI overrides apply **after** the level preset is applied.

## Quality presets (`--level`)

| Level | Width | Height | Samples | Max bounces |
|-------|-------|--------|---------|-------------|
| `low` | 800 | 450 | 32 | 8 |
| `med` | 1280 | 720 | 128 | 10 |
| `high` | 1920 | 1080 | 384 | 12 |

## Examples

Fast preview:

```bash
python src/main.py --level low
```

Custom resolution and sample count:

```bash
python src/main.py --level med --width 640 --height 360 --samples 16
```

Single-threaded debug render:

```bash
python src/main.py --level low --no-parallel
```

Custom output path:

```bash
python src/main.py --level high --output output/my_render.png
```

Fixed worker count:

```bash
python src/main.py --level med --workers 4
```

## Parallelism

By default, `renderer.render()` uses `multiprocessing.Pool` with `default_workers(reserve=1)` — typically **logical CPUs minus one**. The image is split into **64×64 pixel tiles**; workers return tiles that are stitched into the final buffer.

Use `--no-parallel` to render sequentially (useful when debugging or under restricted environments).

On Windows, the worker entry point must be import-safe; the project uses the standard `if __name__ == "__main__"` guard in `main.py`.
