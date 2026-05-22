# Raytracer

Everything that runs when you press `F12`. Pure Python + numpy. No graphics libraries.

---

## Overview

```
for each pixel (x, y):
    ray = camera.generate_ray(x, y)
    color = trace(ray, scene, depth=0)
    image[y][x] = tonemap(color)
```

For anti-aliasing, repeat N times with sub-pixel jitter and average:

```
for each pixel (x, y):
    color = vec3(0, 0, 0)
    for s in range(samples):
        jx, jy = random() - 0.5, random() - 0.5
        ray = camera.generate_ray(x + jx, y + jy)
        color += trace(ray, scene, depth=0)
    image[y][x] = tonemap(color / samples)
```

---

## Module: `ray.py`

The most fundamental data structure. A ray is an origin point and a direction vector.

```python
@dataclass
class Ray:
    origin: np.ndarray     # shape (3,)
    direction: np.ndarray  # shape (3,), always normalized

    def at(self, t: float) -> np.ndarray:
        """Point along the ray at parameter t."""
        return self.origin + t * self.direction
```

---

## Module: `intersect.py`

### HitRecord

Returned by every intersection function. Contains everything shading needs.

```python
@dataclass
class HitRecord:
    t: float              # Ray parameter at hit point
    point: np.ndarray     # World-space hit position
    normal: np.ndarray    # Surface normal (always points outward)
    front_face: bool      # True if ray hit the outside of the surface
    material: Material    # Reference to the object's material
    uv: tuple             # (u, v) texture coordinates
```

---

### Ray–Sphere Intersection

A sphere of radius `r` centered at `c`. Solve for `t` where `|ray.at(t) - c|² = r²`.

```
oc = ray.origin - center
a  = dot(ray.direction, ray.direction)   # 1.0 if direction is normalized
b  = 2 * dot(oc, ray.direction)
c  = dot(oc, oc) - radius²

discriminant = b² - 4ac

if discriminant < 0:
    return None   # Miss

t1 = (-b - sqrt(discriminant)) / (2a)
t2 = (-b + sqrt(discriminant)) / (2a)

# Take the nearest positive t
t = t1 if t1 > t_min else t2
if t < t_min: return None

point  = ray.at(t)
normal = (point - center) / radius
```

Set `front_face = dot(ray.direction, normal) < 0`. If not front face, flip normal.

---

### Ray–Plane Intersection

An infinite plane defined by a point `p0` and normal `n`.

```
denom = dot(ray.direction, n)
if abs(denom) < 1e-6: return None   # Ray is parallel to plane

t = dot(p0 - ray.origin, n) / denom
if t < t_min: return None

point  = ray.at(t)
normal = n (or -n if not front face)
```

For a **checkerboard material**, compute UV from world X/Z and apply color alternation in the shader.

---

### Ray–Triangle Intersection (Möller–Trumbore)

Used for mesh objects (cubes, imported OBJ files).

```
Given vertices v0, v1, v2:

e1 = v1 - v0
e2 = v2 - v0
h  = cross(ray.direction, e2)
a  = dot(e1, h)

if abs(a) < 1e-8: return None   # Ray is parallel to triangle

f = 1.0 / a
s = ray.origin - v0
u = f * dot(s, h)
if u < 0 or u > 1: return None

q = cross(s, e1)
v = f * dot(ray.direction, q)
if v < 0 or u + v > 1: return None

t = f * dot(e2, q)
if t < t_min: return None

# Hit confirmed
point  = ray.at(t)
normal = normalize(cross(e1, e2))   # or interpolated vertex normals
uv     = (u, v)
```

---

## Module: `shading.py`

The `trace(ray, scene, depth)` function.

```python
def trace(ray, scene, depth):
    if depth > scene.render.max_bounces:
        return scene.render.background_color

    hit = find_closest_hit(ray, scene)   # tests all objects, returns nearest HitRecord

    if hit is None:
        return background_color(ray, scene)

    return shade(hit, ray, scene, depth)
```

### `shade(hit, ray, scene, depth)`

Dispatch on material type:

---

#### Diffuse (Lambertian)

```
emitted = material.emissive_color * material.emissive_strength   # (0 if not emissive)

# Direct lighting
color = vec3(0,0,0)
for light in scene.lights:
    shadow_ray = Ray(hit.point + normal*1e-4, toward_light)
    if not is_occluded(shadow_ray, scene, light_distance):
        attenuation = light.intensity / light_distance²
        color += material.color * light.color * attenuation * max(0, dot(normal, light_dir))

return emitted + color
```

For path tracing mode: replace direct light loop with a random hemisphere bounce.

---

#### Specular (Mirror / Glossy)

```
reflected = reflect(ray.direction, hit.normal)

# Add roughness: perturb the reflection direction
if material.roughness > 0:
    reflected += material.roughness * random_in_unit_sphere()
    reflected = normalize(reflected)

reflected_ray = Ray(hit.point + normal*1e-4, reflected)
reflected_color = trace(reflected_ray, scene, depth + 1)

# Fresnel blend: more reflective at glancing angles
fresnel = schlick(cos_theta, material.ior)
return mix(diffuse_color, reflected_color, fresnel)
```

**Schlick's Fresnel approximation:**
```
r0 = ((1 - ior) / (1 + ior))²
schlick(cos_theta) = r0 + (1 - r0) * (1 - cos_theta)^5
```

**Reflect function:**
```
reflect(v, n) = v - 2 * dot(v, n) * n
```

---

#### Glass (Dielectric)

Uses Snell's Law. A glass ray either refracts (passes through) or reflects (total internal reflection).

```
ior_ratio = 1/ior if front_face else ior   # air→glass or glass→air

cos_theta = min(dot(-ray.direction, normal), 1.0)
sin_theta = sqrt(1 - cos_theta²)

# Total internal reflection check
cannot_refract = ior_ratio * sin_theta > 1.0
fresnel = schlick(cos_theta, ior)

if cannot_refract or random() < fresnel:
    direction = reflect(ray.direction, normal)
else:
    direction = refract(ray.direction, normal, ior_ratio)

return trace(Ray(hit.point, direction), scene, depth + 1)
```

**Refract function (Snell's Law):**
```
cos_theta = dot(-uv, n)
r_perp  = ior_ratio * (uv + cos_theta * n)
r_par   = -sqrt(abs(1 - |r_perp|²)) * n
return r_perp + r_par
```

---

## Module: `bvh.py`

Without acceleration, every ray tests every object: O(objects × pixels). With a BVH it's O(log(objects) × pixels).

### AABB (Axis-Aligned Bounding Box)

```python
@dataclass
class AABB:
    min: np.ndarray   # (3,) lower corner
    max: np.ndarray   # (3,) upper corner

    def hit(self, ray, t_min, t_max) -> bool:
        for axis in range(3):
            inv_d = 1.0 / ray.direction[axis]
            t0 = (self.min[axis] - ray.origin[axis]) * inv_d
            t1 = (self.max[axis] - ray.origin[axis]) * inv_d
            if inv_d < 0: t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max <= t_min: return False
        return True
```

### BVH Node

```python
@dataclass
class BVHNode:
    bbox: AABB
    left: BVHNode | SceneObject
    right: BVHNode | SceneObject
```

### Building the BVH

```
def build(objects):
    if len(objects) == 1:
        return leaf(objects[0])
    if len(objects) == 2:
        return node(leaf(objects[0]), leaf(objects[1]))

    # Sort along the longest axis of the combined bounding box
    axis = longest_axis(combined_bbox(objects))
    objects.sort(key=lambda o: o.bbox.centroid[axis])
    mid = len(objects) // 2

    left  = build(objects[:mid])
    right = build(objects[mid:])
    return BVHNode(bbox=surrounding_box(left.bbox, right.bbox), left=left, right=right)
```

### Traversal

```
def hit_bvh(node, ray, t_min, t_max):
    if not node.bbox.hit(ray, t_min, t_max):
        return None

    if is_leaf(node):
        return intersect(node.object, ray, t_min, t_max)

    left_hit  = hit_bvh(node.left,  ray, t_min, t_max)
    right_hit = hit_bvh(node.right, ray, t_min, left_hit.t if left_hit else t_max)

    if right_hit: return right_hit
    return left_hit
```

---

## Module: `sampling.py`

Random samples needed for anti-aliasing, soft shadows, and path tracing.

```python
def random_in_unit_sphere() -> np.ndarray:
    """Rejection sample: pick random point in unit cube until inside sphere."""
    while True:
        p = np.random.uniform(-1, 1, 3)
        if np.dot(p, p) < 1.0:
            return p

def random_unit_vector() -> np.ndarray:
    return normalize(random_in_unit_sphere())

def random_cosine_direction() -> np.ndarray:
    """Cosine-weighted hemisphere sample. Better than uniform for diffuse."""
    r1, r2 = random(), random()
    phi = 2 * pi * r1
    x = cos(phi) * sqrt(r2)
    y = sin(phi) * sqrt(r2)
    z = sqrt(1 - r2)
    return np.array([x, y, z])

def random_in_unit_disk() -> np.ndarray:
    """For depth of field: sample camera lens."""
    while True:
        p = np.array([uniform(-1,1), uniform(-1,1), 0])
        if np.dot(p, p) < 1.0:
            return p
```

---

## Module: `tonemapping.py`

Raw float HDR values → displayable 8-bit PNG.

```python
def tonemap(color: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """color: float array shape (H, W, 3), values can exceed 1.0"""

    # 1. Exposure
    color = color * exposure

    # 2. Reinhard tone map (maps [0,∞) → [0,1))
    color = color / (1.0 + color)

    # OR: ACES filmic (more cinematic look)
    # a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    # color = np.clip((color*(a*color+b))/(color*(c*color+d)+e), 0, 1)

    # 3. Gamma correction (linear → sRGB)
    color = np.clip(color, 0, 1) ** (1.0 / 2.2)

    # 4. Convert to 8-bit
    return (color * 255).astype(np.uint8)
```

---

## Module: `renderer.py`

The top-level render entry point.

```python
def render(scene: Scene, progress_callback=None) -> np.ndarray:
    W, H = scene.render.width, scene.render.height
    image = np.zeros((H, W, 3), dtype=np.float32)

    # Build BVH once before spawning workers
    bvh = build_bvh(scene.objects)

    # Serialize scene for multiprocessing (avoids pickle issues)
    scene_dict = scene.to_dict()

    # Divide into tiles
    tile_size = 64
    tiles = [
        (x, y, min(x+tile_size, W), min(y+tile_size, H))
        for y in range(0, H, tile_size)
        for x in range(0, W, tile_size)
    ]

    with multiprocessing.Pool() as pool:
        tasks = [(scene_dict, tile) for tile in tiles]
        for i, (tile, result) in enumerate(pool.imap_unordered(render_tile_worker, tasks)):
            x0, y0, x1, y1 = tile
            image[y0:y1, x0:x1] = result
            if progress_callback:
                progress_callback(i + 1, len(tiles))

    return tonemap(image, scene.render.exposure)


def render_tile_worker(args):
    scene_dict, tile = args
    scene = Scene.from_dict(scene_dict)
    x0, y0, x1, y1 = tile
    result = np.zeros((y1-y0, x1-x0, 3), dtype=np.float32)

    for py in range(y0, y1):
        for px in range(x0, x1):
            color = np.zeros(3)
            for _ in range(scene.render.samples):
                jx, jy = random() - 0.5, random() - 0.5
                ray = scene.camera.generate_ray(px + jx, py + jy, scene.render.width, scene.render.height)
                color += trace(ray, scene, depth=0)
            result[py-y0, px-x0] = color / scene.render.samples

    return tile, result
```

---

## Camera Ray Generation

```python
def generate_ray(self, px, py, image_width, image_height) -> Ray:
    # Compute viewport dimensions from FOV
    aspect = image_width / image_height
    half_h = tan(radians(self.fov / 2))
    half_w = aspect * half_h

    # Build camera coordinate frame
    w = normalize(self.position - self.target)   # backward
    u = normalize(cross(self.up, w))             # right
    v = cross(w, u)                              # up

    lower_left = self.position - half_w*u - half_h*v - w
    horizontal = 2 * half_w * u
    vertical   = 2 * half_h * v

    # Map pixel to [0,1] UV with sub-pixel jitter already baked in
    s = px / image_width
    t = 1.0 - (py / image_height)   # flip Y

    direction = normalize(lower_left + s*horizontal + t*vertical - self.position)
    return Ray(origin=self.position, direction=direction)
```

---

## Performance Notes

| Technique | Speedup | When to add |
|---|---|---|
| BVH | 10–100× for large scenes | Week 3+ |
| Multiprocessing | ~N× on N cores | Week 2 |
| numpy vectorize per tile | 2–5× | Optional |
| Early ray termination | Varies | Built into BVH traversal |
| Russian roulette path termination | Reduces wasted bounces | Path tracing mode only |

For a 1920×1080 render at 64 SPP on a modern 8-core machine: expect roughly 2–10 minutes depending on scene complexity and max bounces. At 16 SPP with BVH: under 2 minutes.
