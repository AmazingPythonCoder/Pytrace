# Scene

The shared data layer. Neither the editor nor the raytracer imports the other — they both read and write this.

---

## `scene/scene.py`

```python
@dataclass
class RenderConfig:
    width: int        = 1920
    height: int       = 1080
    samples: int      = 64
    max_bounces: int  = 6
    exposure: float   = 1.0
    background_color: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.08]))

@dataclass
class Scene:
    objects: list[SceneObject]  = field(default_factory=list)
    lights:  list[Light]        = field(default_factory=list)
    camera:  Camera             = field(default_factory=Camera)
    render:  RenderConfig       = field(default_factory=RenderConfig)
    selected: SceneObject | None = field(default=None, repr=False)

    def add(self, obj):
        self.objects.append(obj)
        self.selected = obj

    def remove(self, obj):
        self.objects.remove(obj)
        if self.selected is obj:
            self.selected = None

    def to_dict(self) -> dict: ...
    def from_dict(cls, d: dict) -> Scene: ...

    @classmethod
    def default(cls) -> Scene:
        """The startup scene: one sphere, one plane, one point light."""
        s = cls()
        s.objects = [
            Sphere(position=[0, 0.5, 0], radius=0.8,
                   material=DiffuseMaterial(color=[0.8, 0.1, 0.1])),
            Plane(position=[0, 0, 0], normal=[0, 1, 0],
                  material=DiffuseMaterial(color=[0.7, 0.7, 0.7])),
        ]
        s.lights = [
            PointLight(position=[4, 6, 4], color=[1,1,1], intensity=100),
        ]
        s.camera = Camera(position=[0, 2, 8], target=[0, 0.5, 0], fov=55)
        return s
```

---

## `scene/objects.py`

```python
@dataclass
class SceneObject:
    name:     str
    position: np.ndarray   = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray   = field(default_factory=lambda: np.zeros(3))  # Euler XYZ degrees
    scale:    np.ndarray   = field(default_factory=lambda: np.ones(3))
    material: Material     = field(default_factory=DiffuseMaterial)
    visible:  bool         = True

    @property
    def bbox(self) -> AABB:
        """Bounding box for BVH. Must be implemented by subclasses."""
        raise NotImplementedError

@dataclass
class Sphere(SceneObject):
    radius: float = 0.5
    name: str = "Sphere"

    @property
    def bbox(self) -> AABB:
        r = self.radius
        return AABB(self.position - r, self.position + r)

@dataclass
class Plane(SceneObject):
    normal: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    name: str = "Plane"

    @property
    def bbox(self) -> AABB:
        # Infinite plane: very large bounding box
        INF = 1e9
        return AABB(
            np.array([-INF, self.position[1] - 0.001, -INF]),
            np.array([ INF, self.position[1] + 0.001,  INF]),
        )

@dataclass
class Mesh(SceneObject):
    vertices: np.ndarray   = None   # shape (N, 3)
    triangles: np.ndarray  = None   # shape (M, 3) indices into vertices
    normals: np.ndarray    = None   # shape (N, 3) vertex normals
    name: str = "Mesh"

    @property
    def bbox(self) -> AABB:
        return AABB(self.vertices.min(axis=0), self.vertices.max(axis=0))

    @classmethod
    def cube(cls, size=1.0) -> Mesh:
        """Unit cube centered at origin."""
        h = size / 2
        verts = np.array([...])    # 8 corners
        tris  = np.array([...])    # 12 triangles (6 faces × 2)
        return cls(vertices=verts, triangles=tris, name="Cube")
```

---

## `scene/materials.py`

```python
@dataclass
class Material:
    type: str   # discriminant for serialization

@dataclass
class DiffuseMaterial(Material):
    type: str          = "diffuse"
    color: np.ndarray  = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))
    roughness: float   = 1.0

@dataclass
class SpecularMaterial(Material):
    type: str          = "specular"
    color: np.ndarray  = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    roughness: float   = 0.0
    ior: float         = 1.5

@dataclass
class GlassMaterial(Material):
    type: str                     = "glass"
    ior: float                    = 1.45
    absorption_color: np.ndarray  = field(default_factory=lambda: np.ones(3))

@dataclass
class EmissiveMaterial(Material):
    type: str          = "emissive"
    color: np.ndarray  = field(default_factory=lambda: np.ones(3))
    strength: float    = 5.0
```

---

## `scene/lights.py`

```python
@dataclass
class Light:
    type:     str
    position: np.ndarray
    color:    np.ndarray  = field(default_factory=lambda: np.ones(3))
    intensity: float      = 100.0

@dataclass
class PointLight(Light):
    type: str = "point"
    position: np.ndarray = field(default_factory=lambda: np.array([3., 6., 3.]))

@dataclass
class DirectionalLight(Light):
    type:      str          = "directional"
    direction: np.ndarray   = field(default_factory=lambda: np.array([-1., -2., -1.]))
    position:  np.ndarray   = field(default_factory=lambda: np.zeros(3))  # unused, kept for consistency

@dataclass
class AreaLight(Light):
    """Rectangular area light. Sampled for soft shadows."""
    type:    str          = "area"
    position: np.ndarray = field(default_factory=lambda: np.array([0., 5., 0.]))
    normal:  np.ndarray  = field(default_factory=lambda: np.array([0., -1., 0.]))
    width:   float       = 2.0
    height:  float       = 2.0
    samples: int         = 4    # shadow samples per intersection (4 = 2×2 grid)
```

---

## `scene/camera.py`

```python
@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0., 2., 8.]))
    target:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    up:       np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    fov:      float      = 55.0       # vertical FOV in degrees

    # Depth of field (set aperture > 0 to enable)
    aperture:      float = 0.0        # lens radius; 0 = pinhole (no DOF)
    focus_distance: float = None      # defaults to distance(position, target)

    def generate_ray(self, px, py, image_w, image_h) -> Ray:
        """Generate a ray for pixel (px, py). Includes DOF if aperture > 0."""
        ...
```

---

## `scene/serializer.py`

All objects implement `to_dict()` and are reconstructed via `from_dict()`. Material type is stored as a string discriminant so deserialization knows which class to instantiate.

```python
MATERIAL_TYPES = {
    "diffuse":   DiffuseMaterial,
    "specular":  SpecularMaterial,
    "glass":     GlassMaterial,
    "emissive":  EmissiveMaterial,
}

OBJECT_TYPES = {
    "sphere": Sphere,
    "plane":  Plane,
    "mesh":   Mesh,
}

LIGHT_TYPES = {
    "point":       PointLight,
    "directional": DirectionalLight,
    "area":        AreaLight,
}

def material_from_dict(d: dict) -> Material:
    cls = MATERIAL_TYPES[d["type"]]
    return cls(**{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()})
```

---

## Default Scene JSON (`scenes/default.json`)

```json
{
  "camera": {
    "position": [0, 2, 8],
    "target": [0, 0.5, 0],
    "fov": 55,
    "aperture": 0.0
  },
  "objects": [
    {
      "type": "sphere",
      "name": "Red Sphere",
      "position": [-1.5, 0.5, 0],
      "rotation": [0, 0, 0],
      "scale": [1, 1, 1],
      "radius": 0.7,
      "material": { "type": "diffuse", "color": [0.8, 0.1, 0.1], "roughness": 0.9 }
    },
    {
      "type": "sphere",
      "name": "Glass Sphere",
      "position": [0.2, 0.5, 0.5],
      "rotation": [0, 0, 0],
      "scale": [1, 1, 1],
      "radius": 0.6,
      "material": { "type": "glass", "ior": 1.5, "absorption_color": [1, 1, 1] }
    },
    {
      "type": "sphere",
      "name": "Mirror Sphere",
      "position": [1.8, 0.5, -0.3],
      "rotation": [0, 0, 0],
      "scale": [1, 1, 1],
      "radius": 0.7,
      "material": { "type": "specular", "color": [0.9, 0.9, 0.9], "roughness": 0.02, "ior": 1.5 }
    },
    {
      "type": "plane",
      "name": "Ground",
      "position": [0, 0, 0],
      "rotation": [0, 0, 0],
      "scale": [1, 1, 1],
      "normal": [0, 1, 0],
      "material": { "type": "diffuse", "color": [0.6, 0.6, 0.6], "roughness": 1.0 }
    }
  ],
  "lights": [
    {
      "type": "point",
      "name": "Key Light",
      "position": [4, 8, 4],
      "color": [1.0, 0.95, 0.9],
      "intensity": 200
    },
    {
      "type": "point",
      "name": "Fill Light",
      "position": [-6, 4, 2],
      "color": [0.8, 0.85, 1.0],
      "intensity": 60
    }
  ],
  "render": {
    "width": 1920,
    "height": 1080,
    "samples": 64,
    "max_bounces": 6,
    "exposure": 1.0,
    "background_color": [0.05, 0.05, 0.08]
  }
}
```
