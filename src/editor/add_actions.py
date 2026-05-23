from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AddEntry:
    action: str
    title: str
    detail: str
    group: str


ENTRIES = (
    AddEntry("add_sphere", "Sphere", "Round primitive with radius control", "Geometry"),
    AddEntry("add_plane", "Plane", "Infinite render plane with preview tile", "Geometry"),
    AddEntry("add_cube", "Cube", "Triangle mesh cube for hard-surface blocking", "Geometry"),
    AddEntry("import_obj", "Import OBJ", "Load a Wavefront mesh from disk", "Geometry"),
    AddEntry("add_point_light", "Point Light", "Small positional light", "Lights"),
    AddEntry("add_directional_light", "Directional Light", "Sun-style light with direction", "Lights"),
    AddEntry("add_area_light", "Area Light", "Disk light for soft shadows", "Lights"),
)
