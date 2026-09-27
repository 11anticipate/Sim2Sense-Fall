"""Convert the same CPU scene plan consumed by USD into radio meshes in world metres.

This is the authored initial apartment, including roofs and every furniture part.
Dynamic furniture transforms after a physics trial are not available in this schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim2sense_fall.scenes.geometry import WorldShape, world_shapes
from sim2sense_fall.scenes.planner import plan_scene
from sim2sense_fall.scenes.spec import SceneSpec

from .mesh_import import build_mitsuba_mesh


@dataclass(frozen=True, slots=True)
class RadioGeometry:
    path: str
    material: str
    vertices: np.ndarray
    faces: np.ndarray
    category: str
    movable: bool


def triangulate(shape: WorldShape, *, cylinder_segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(cylinder_segments, bool) or cylinder_segments < 8 or cylinder_segments % 4:
        raise ValueError("cylinder_segments must be a multiple of four >= 8")
    if shape.cylinder:
        n = cylinder_segments
        angle = np.arange(n) * 2 * np.pi / n
        ring = np.column_stack([np.cos(angle), np.sin(angle)]) * shape.size[0] / 2
        points = np.vstack(
            [
                np.column_stack([ring, np.full(n, -shape.size[2] / 2)]),
                np.column_stack([ring, np.full(n, shape.size[2] / 2)]),
                [0, 0, -shape.size[2] / 2],
                [0, 0, shape.size[2] / 2],
            ]
        )
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.extend(
                [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
            )
    else:
        points = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
            )
            * np.asarray(shape.size)
            / 2
        )
        faces = [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ]
    theta = np.radians(shape.yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ rotation.T + shape.center, np.asarray(faces, dtype=np.int64)


def apartment_geometry(
    spec: SceneSpec, *, cylinder_segments: int = 32
) -> tuple[RadioGeometry, ...]:
    plan = plan_scene(spec)
    shapes = world_shapes(plan.prims)
    result = []
    for prim in plan.prims:
        if not prim.is_geometry:
            continue
        if prim.material is None or spec.materials[prim.material].em is None:
            raise ValueError(f"missing EM mapping for {prim.path}")
        vertices, faces = triangulate(shapes[prim.path], cylinder_segments=cylinder_segments)
        vertices.setflags(write=False)
        faces.setflags(write=False)
        result.append(
            RadioGeometry(
                prim.path, prim.material, vertices, faces, prim.category, prim.in_rigid_body
            )
        )
    return tuple(result)


def build_apartment_scene(
    spec: SceneSpec,
    *,
    frequency_hz: float,
    cylinder_segments: int = 32,
    dry_run: bool = False,
) -> tuple[Any, dict[str, Any]]:
    geometry = apartment_geometry(spec, cylinder_segments=cylinder_segments)
    material_values = {
        name: spec.materials[name].em_parameters(frequency_hz)
        for name in sorted({item.material for item in geometry})
    }
    report = {
        "scene_id": spec.scene_id,
        "seed": spec.seed,
        "geometry_count": len(geometry),
        "materials": material_values,
        "coordinate_system": "world_z_up_xyz",
        "units": "m",
        "cylinder_segments": cylinder_segments,
        "dynamic_furniture": "authored_initial_transforms; not updated from PhysX",
        "objects": [
            {
                "path": g.path,
                "material": g.material,
                "category": g.category,
                "movable": g.movable,
                "vertices": len(g.vertices),
                "faces": len(g.faces),
                "bounds_min": g.vertices.min(axis=0).tolist(),
                "bounds_max": g.vertices.max(axis=0).tolist(),
            }
            for g in geometry
        ],
    }
    if dry_run:
        return None, report
    try:
        import sionna.rt as rt
    except ImportError as exc:
        raise RuntimeError(
            "apartment import needs the Sionna runtime; use --dry-run for CPU"
        ) from exc
    scene = rt.load_scene()
    scene.frequency = frequency_hz
    for name, values in material_values.items():
        scene.add(
            rt.RadioMaterial(
                name=name,
                relative_permittivity=values["relative_permittivity"],
                conductivity=values["conductivity_s_per_m"],
                thickness=values["thickness_m"],
                scattering_coefficient=values["scattering_coefficient"],
            )
        )
    objects = []
    for index, item in enumerate(geometry):
        name = f"apartment_{index:04d}"
        mesh = build_mitsuba_mesh(item.vertices, item.faces, name=name)
        objects.append(
            rt.SceneObject(
                mi_mesh=mesh, name=name, radio_material=scene.radio_materials[item.material]
            )
        )
    scene.edit(add=objects)
    if len(scene.objects) != len(geometry):
        raise RuntimeError("apartment object count differs after Scene.edit")
    return scene, report
