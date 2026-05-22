# Editor

The real-time viewport and UI. Everything the user interacts with before hitting `F12`.

---

## Application Loop (`editor/app.py`)

```python
def run():
    pygame.init()
    screen = pygame.display.set_mode((1600, 900), DOUBLEBUF | OPENGL | RESIZABLE)
    clock  = pygame.time.Clock()
    scene  = Scene.default()

    viewport    = Viewport(scene)
    outliner    = Outliner(scene)
    properties  = Properties(scene)
    toolbar     = Toolbar(scene)
    render_win  = None   # set when rendering starts

    while True:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()

        for event in events:
            if event.type == QUIT: return
            toolbar.handle(event)
            if render_win:
                render_win.handle(event)
            else:
                outliner.handle(event)
                properties.handle(event)
                viewport.handle(event, scene)

        # Draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        viewport.draw(scene)
        # UI panels drawn on top via pygame surface blitted to GL texture
        outliner.draw(screen, scene)
        properties.draw(screen, scene)
        toolbar.draw(screen, scene)
        if render_win:
            render_win.draw(screen)

        pygame.display.flip()
```

---

## Viewport (`editor/viewport.py`)

Responsible for the 3D OpenGL view. Nothing from the raytracer runs here.

### Setup

```python
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Viewport takes up the center region (leave room for panels)
glViewport(panel_left_width, panel_bottom_height, viewport_w, viewport_h)
```

### Drawing objects (preview mode)

Objects are drawn as **flat-shaded solids** or **wireframes** — fast enough for 60fps.

```
for obj in scene.objects:
    if isinstance(obj, Sphere):
        draw_sphere_preview(obj)    # glut sphere or subdivided icosphere
    elif isinstance(obj, Plane):
        draw_plane_preview(obj)     # large flat quad
    elif isinstance(obj, Mesh):
        draw_mesh_preview(obj)      # draw_arrays with vertex buffer

# Draw selection highlight
if scene.selected:
    draw_wireframe_overlay(scene.selected)   # same shape, thicker lines, accent color

# Draw lights as icons (small emissive sphere or sun icon)
for light in scene.lights:
    draw_light_icon(light)

# Draw camera frustum wireframe
draw_camera_frustum(scene.camera)

# Grid
draw_grid()
```

### Projection and view matrices

```python
def set_projection(fov, aspect, near, far):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, aspect, near, far)

def set_view(orbit_cam):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(*orbit_cam.eye, *orbit_cam.target, *orbit_cam.up)
```

---

## Orbit Camera (`editor/orbit_camera.py`)

The **editor's own camera** for navigating the 3D viewport. Not the scene camera.

```python
@dataclass
class OrbitCamera:
    target: np.ndarray = (0, 0, 0)
    distance: float    = 8.0
    yaw: float         = 45.0    # degrees, horizontal angle
    pitch: float       = 25.0    # degrees, vertical angle

    @property
    def eye(self) -> np.ndarray:
        r = radians
        x = self.target[0] + self.distance * cos(r(self.pitch)) * sin(r(self.yaw))
        y = self.target[1] + self.distance * sin(r(self.pitch))
        z = self.target[2] + self.distance * cos(r(self.pitch)) * cos(r(self.yaw))
        return np.array([x, y, z])
```

### Input handling

```python
def handle_mouse(self, event, prev_mouse):
    dx = event.pos[0] - prev_mouse[0]
    dy = event.pos[1] - prev_mouse[1]
    buttons = pygame.mouse.get_pressed()
    mods    = pygame.key.get_mods()

    if buttons[1]:   # middle mouse
        if mods & KMOD_SHIFT:
            self.pan(dx, dy)
        else:
            self.orbit(dx, dy)

def handle_scroll(self, event):
    self.distance *= 0.9 if event.y > 0 else 1.1
    self.distance  = max(0.5, min(200, self.distance))

def orbit(self, dx, dy):
    self.yaw   += dx * 0.4
    self.pitch -= dy * 0.4
    self.pitch  = max(-89, min(89, self.pitch))

def pan(self, dx, dy):
    # Move target in camera's local XY plane
    right = normalize(cross(self.eye - self.target, self.up))
    up    = normalize(cross(right, self.eye - self.target))
    speed = self.distance * 0.001
    self.target -= right * dx * speed
    self.target += up    * dy * speed
```

---

## Selection (`editor/selection.py`)

Click → find which object was hit.

Cast a ray from the **editor camera** through the clicked pixel and run the same intersection code as the raytracer (but only the geometric part — no shading).

```python
def pick(click_x, click_y, viewport_rect, orbit_cam, scene) -> SceneObject | None:
    # Convert click to normalized device coordinates
    ndx = (click_x - viewport_rect.x) / viewport_rect.w * 2 - 1
    ndy = 1 - (click_y - viewport_rect.y) / viewport_rect.h * 2

    # Unproject to world ray
    ray = unproject_ray(ndx, ndy, orbit_cam, aspect=viewport_rect.w/viewport_rect.h)

    # Find nearest hit (same code as raytracer, no material needed)
    nearest = None
    nearest_t = float('inf')
    for obj in scene.objects:
        hit = intersect(ray, obj, t_min=0.001, t_max=float('inf'))
        if hit and hit.t < nearest_t:
            nearest_t = hit.t
            nearest   = obj

    return nearest
```

---

## Transform Gizmos (`editor/gizmos.py`)

Drawn on top of the selected object as colored axis handles.

### Visual

```
     ↑ Y (green)
     │
     ●──────── X (red)
    ╱
   ╱ Z (blue)
```

Each axis is a thick colored line with a cone at the tip (move gizmo), a torus (rotate), or a cube endpoint (scale).

### Interaction

```python
class MoveGizmo:
    def __init__(self, obj):
        self.obj = obj
        self.active_axis = None   # 'X', 'Y', 'Z', or None

    def handle_mouse_down(self, ray):
        # Check which axis cone was clicked
        for axis, direction in [('X', [1,0,0]), ('Y', [0,1,0]), ('Z', [0,0,1])]:
            cone_pos = self.obj.position + np.array(direction) * GIZMO_LENGTH
            if ray_hits_cone(ray, cone_pos, direction, radius=0.1):
                self.active_axis = axis
                return True
        return False

    def handle_mouse_drag(self, ray, prev_ray):
        if not self.active_axis: return
        # Project mouse delta onto the active axis in world space
        delta = project_delta_to_axis(ray, prev_ray, self.active_axis, self.obj.position)
        self.obj.position += delta
```

### Keyboard shortcuts (alternative to gizmo dragging)

| Key | Mode |
|---|---|
| `G` | Enter grab mode. Mouse movement moves object on XY plane. |
| `G X` | Constrain grab to X axis only. |
| `G Y` | Constrain grab to Y axis only. |
| `G Z` | Constrain grab to Z axis only. |
| `R` | Rotate mode (around Y by default). |
| `S` | Scale mode (uniform). |
| `Enter` / LMB | Confirm. |
| `Escape` | Cancel, restore original transform. |

In grab mode: translate mouse delta to world-space movement along the constrained axis using the inverse of the viewport projection.

---

## Grid (`editor/grid.py`)

An infinite-looking grid on the XZ plane. Two layers: major lines every 1 unit, minor lines every 5 units.

```python
def draw_grid(orbit_cam, size=20, step=1):
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for i in np.arange(-size, size + step, step):
        alpha = 0.15 if i % 5 != 0 else 0.35
        glColor4f(0.5, 0.5, 0.5, alpha)
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0,  size)
        glVertex3f(-size, 0, i)
        glVertex3f( size, 0, i)
    glEnd()

    # Origin axes — always visible
    glLineWidth(2.0)
    glColor3f(0.7, 0.15, 0.15); glVertex3f(-size,0,0); glVertex3f(size,0,0)   # X red
    glColor3f(0.15, 0.6, 0.15); glVertex3f(0,0,-size); glVertex3f(0,0,size)   # Z green
```

---

## Outliner Panel (`editor/outliner.py`)

Left sidebar. Lists every object and light in the scene. Click to select.

```
┌─────────────────┐
│  Scene          │
│  ├ 👁 Sphere    │  ← selected (highlighted)
│  ├ 👁 Ground    │
│  ├ 💡 Point L.  │
│  └ 📷 Camera    │
│                 │
│  [+ Add]        │
└─────────────────┘
```

Implementation: rendered as a pygame surface each frame (or only on dirty). Blitted over the OpenGL viewport using `pygame.display.get_surface()` and a texture blit.

Each row is a rect that checks for mouse click. Click → `scene.selected = obj`. Double click → rename.

---

## Properties Panel (`editor/properties.py`)

Right sidebar. Shows editable fields for the selected object.

### Sections

**Transform**
```
Location   [ 0.00 ] [ 1.50 ] [ 0.00 ]
Rotation   [ 0.00 ] [ 0.00 ] [ 0.00 ]
Scale      [ 1.00 ] [ 1.00 ] [ 1.00 ]
```

Each field is a text input. On enter/unfocus → update `obj.position[0]`, etc.

**Material** (varies by type)
```
Type       [ Diffuse ▼ ]

Color      [■■■■■■■■■■] (color picker popup)
Roughness  ────●──────  0.50
Metallic   ●──────────  0.00
```

For glass:
```
IOR        ────●──────  1.45
Color Abs. [■■■■■■■■■■]
```

**Render Settings** (when camera is selected)
```
Resolution  [ 1920 ] × [ 1080 ]
Samples     [ 64 ]
Max Bounces [ 6 ]
Exposure    ────●──────  1.00
```

---

## Render Window (`editor/render_window.py`)

Shown when `F12` is pressed. Covers the main viewport.

```
┌────────────────────────────────────────┐
│  Rendering... 47 / 120 tiles           │
│  ██████████████░░░░░░░░░░░  39%        │
│                                        │
│  [rendered tiles appear here live]     │
│  [new tiles blit in as they finish]    │
│                                        │
│  [Cancel]                [Save PNG]    │
└────────────────────────────────────────┘
```

```python
class RenderWindow:
    def __init__(self, scene):
        self.surface = pygame.Surface((W, H))
        self.surface.fill((10, 10, 10))
        self.progress = (0, 1)
        self.thread   = None

    def start(self, scene):
        def worker():
            def on_tile(tile, pixels):
                x0, y0, x1, y1 = tile
                tile_surf = pygame.surfarray.make_surface(pixels.swapaxes(0,1))
                self.surface.blit(tile_surf, (x0, y0))
                self.progress = (done, total)
            render(scene, tile_callback=on_tile)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
```

Note: use `threading` (not `multiprocessing`) for the UI thread that polls tile results. The raytracer itself uses `multiprocessing.Pool` internally for CPU parallelism.

---

## Add Object Menu

Triggered by `Shift + A` or the toolbar `+ Add` button. A simple popup list:

```
┌──────────────────┐
│  Mesh            │
│  ├ Sphere        │
│  ├ Plane         │
│  └ Cube          │
│                  │
│  Light           │
│  ├ Point         │
│  ├ Directional   │
│  └ Area          │
└──────────────────┘
```

On click: create the object at the origin (or in front of the editor camera), add it to the scene, select it.

---

## Scene Save / Load

`Ctrl + S` → save dialog → `scene.serialize()` → write JSON.
`Ctrl + O` → open dialog → read JSON → `Scene.deserialize()`.

Use `tkinter.filedialog` for native file dialogs (already in Python stdlib, no extra dependency):

```python
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()   # hide the tk window
path = filedialog.asksaveasfilename(
    defaultextension=".json",
    filetypes=[("PyTrace Scene", "*.json")],
)
```

---

## UI Rendering Strategy

Two options for drawing the panels:

**Option A: Pygame surfaces blitted over OpenGL** (recommended for simplicity)
- Draw OpenGL 3D scene first
- Read framebuffer: `pygame.display.get_surface()`
- Draw panels as pygame rects/text onto that surface
- Flip display

**Option B: Dear ImGui** via `pyimgui`
- Much more powerful widgets (sliders, color pickers, dockable panels)
- Higher quality UI overall
- More setup: `pip install imgui[pygame]`
- Worth it for stretch goal polish

Start with Option A. Migrate to ImGui once core editor features work.
