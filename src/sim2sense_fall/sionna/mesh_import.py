"""Import a collected human mesh into a Sionna RT scene.

The Sionna RT 2.1 API for this is not the obvious one, and getting it wrong fails
quietly, so the two facts that matter are recorded here:

* ``Scene.add`` accepts only a ``Transmitter``, a ``Receiver`` or a ``RadioMaterialBase``.
  Passing a ``SceneObject`` raises. Geometry goes in through
  ``Scene.edit(add=...)`` (or ``scene_utils.extend_scene_with_mesh`` on the raw
  Mitsuba scene). An earlier attempt built the object and never added it, and the
  path solve happily returned the unchanged channel.
* A ``SceneObject`` can be built from an **in-memory** ``mitsuba.Mesh`` via
  ``SceneObject(mi_mesh=...)``, so a per-frame sequence does not have to round-trip
  through a mesh file on disk. ``mitsuba.Mesh`` itself is filled through
  ``mitsuba.traverse``: ``vertex_positions`` (flattened float) and ``faces``
  (flattened uint32).

Everything in this module that can be checked without the runtime is checked without
it: array validation, the material model and the manifest reading are plain numpy, and
only :func:`build_mitsuba_mesh` and :func:`place_mesh_in_scene` touch mitsuba.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "HumanMaterial",
    "ImportedObjectReport",
    "build_mitsuba_mesh",
    "candidate_mesh_arrays",
    "human_tissue_material",
    "place_mesh_in_scene",
    "scene_radio_material",
]


@dataclass(frozen=True, slots=True)
class HumanMaterial:
    """The radio material assigned to the human surface.

    **This is a modelling assumption, not a measurement.** Sionna's ITU-R P.2040 table
    holds 19 building materials (concrete, brick, plasterboard, glass, wood, the ground
    types, ``vacuum``, ...) and **none of them is human tissue**. The body is therefore
    given an explicit ``RadioMaterial`` built from high-water-content soft-tissue
    constants at the band of interest, and the values travel to the output with the
    ``source`` string attached so a reader can see what was assumed and review it.

    ``vacuum`` would have been the other "no new assumption" choice, and it is the wrong
    one: it makes the body electromagnetically absent, which would silently delete the
    very interaction this stage exists to measure.
    """

    name: str
    relative_permittivity: float
    conductivity_s_per_m: float
    thickness_m: float
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("material name must be non-empty")
        if not np.isfinite(self.relative_permittivity) or self.relative_permittivity < 1.0:
            raise ValueError(
                "relative permittivity must be finite and at least 1 "
                f"(the value for vacuum), got {self.relative_permittivity!r}"
            )
        if not np.isfinite(self.conductivity_s_per_m) or self.conductivity_s_per_m < 0.0:
            raise ValueError(
                f"conductivity must be finite and non-negative, got {self.conductivity_s_per_m!r}"
            )
        if not np.isfinite(self.thickness_m) or self.thickness_m <= 0.0:
            raise ValueError(f"thickness must be finite and positive, got {self.thickness_m!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relative_permittivity": self.relative_permittivity,
            "conductivity_s_per_m": self.conductivity_s_per_m,
            "thickness_m": self.thickness_m,
            "model": "sionna.rt.RadioMaterial (single-layer slab, Fresnel)",
            "provenance": "modelling assumption, not measured",
            "source": self.source,
        }


def human_tissue_material(
    *,
    relative_permittivity: float = 51.0,
    conductivity_s_per_m: float = 2.16,
    thickness_m: float = 0.02,
    source: str = (
        "high-water-content soft tissue at ~3.5 GHz; Sionna's ITU-R P.2040 table has no "
        "body-tissue entry, so these are tissue constants entered by hand and need a "
        "literature citation before they are used for reported results"
    ),
) -> HumanMaterial:
    """The default body material, with its assumption stated rather than implied."""

    return HumanMaterial(
        name="human_tissue",
        relative_permittivity=relative_permittivity,
        conductivity_s_per_m=conductivity_s_per_m,
        thickness_m=thickness_m,
        source=source,
    )


@dataclass(frozen=True, slots=True)
class ImportedObjectReport:
    """What was actually handed to the scene, for the run's provenance."""

    name: str
    vertex_count: int
    face_count: int
    bounding_box_min: tuple[float, float, float]
    bounding_box_max: tuple[float, float, float]
    material: HumanMaterial

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounding_box_min_m": list(self.bounding_box_min),
            "bounding_box_max_m": list(self.bounding_box_max),
            "material": self.material.as_dict(),
        }


def candidate_mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalise a (vertices, faces) pair for the ray tracer.

    Sionna's ray tracer consumes a Mitsuba mesh, which wants ``float32`` positions and
    ``uint32`` face indices, and which intersects triangles by index -- so a face index
    outside the vertex array is not a rendering artefact but an out-of-bounds read. The
    checks are here, in numpy, so they run on a CPU machine with no runtime installed.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {points.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {triangles.shape}")
    if points.shape[0] < 3:
        raise ValueError(f"a mesh needs at least three vertices, got {points.shape[0]}")
    if triangles.shape[0] == 0:
        raise ValueError("a mesh needs at least one face")
    if not np.all(np.isfinite(points)):
        raise ValueError("vertices contain a non-finite value")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError(f"face indices must be integers, got {triangles.dtype}")
    if int(triangles.min()) < 0:
        raise ValueError(f"face indices must be non-negative, min is {int(triangles.min())}")
    if int(triangles.max()) >= points.shape[0]:
        raise ValueError(
            f"face index {int(triangles.max())} is outside the {points.shape[0]} vertices"
        )
    return (
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(triangles, dtype=np.uint32),
    )


def build_mitsuba_mesh(vertices: Any, faces: Any, *, name: str) -> Any:
    """An in-memory ``mitsuba.Mesh`` from raw arrays.

    Filled through ``mitsuba.traverse`` rather than a mesh file, so a per-frame sequence
    does not have to write 145 OBJ files to drive a ray tracer.
    """

    if not name.strip():
        raise ValueError("mesh name must be non-empty")
    points, triangles = candidate_mesh_arrays(vertices, faces)
    mitsuba = _mitsuba()
    mesh = mitsuba.Mesh(
        name,
        vertex_count=int(points.shape[0]),
        face_count=int(triangles.shape[0]),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    params = mitsuba.traverse(mesh)
    params["vertex_positions"] = mitsuba.Float(points.ravel())
    params["faces"] = mitsuba.UInt32(triangles.ravel())
    params.update()
    return mesh


def scene_radio_material(scene: Any, material: HumanMaterial) -> Any:
    """The scene's ``RadioMaterial`` for ``material``, registered at most once.

    A radio material belongs to the *scene*, not to one object, and Sionna keeps it
    registered after the object using it is removed. Re-creating it per frame therefore
    fails on the second frame with "Name 'human_tissue' is already used by another item
    of the scene" -- which is exactly what this script did the first time it ran. Reusing
    the registered material is both correct and cheaper: a per-frame sequence would
    otherwise accumulate identical materials, one per frame.
    """

    radio_material_module = _sionna_rt()
    existing = scene.radio_materials.get(material.name)
    if existing is not None:
        return existing
    radio_material = radio_material_module.RadioMaterial(
        name=material.name,
        relative_permittivity=material.relative_permittivity,
        conductivity=material.conductivity_s_per_m,
        thickness=material.thickness_m,
    )
    scene.add(radio_material)
    return radio_material


def place_mesh_in_scene(
    scene: Any, vertices: Any, faces: Any, *, name: str, material: HumanMaterial
) -> ImportedObjectReport:
    """Add the mesh to a Sionna scene as a radio-material scene object.

    ``Scene.edit(add=...)`` is what puts geometry into a Sionna RT 2.1 scene;
    ``Scene.add`` rejects anything that is not a transmitter, receiver or material.
    The object is read back out of ``scene.objects`` afterwards rather than trusted,
    because a silently-dropped object would leave the path solver measuring an empty
    scene and reporting a plausible channel for the wrong world.
    """

    radio_material_module = _sionna_rt()
    mesh = build_mitsuba_mesh(vertices, faces, name=name)
    obj = radio_material_module.SceneObject(
        mi_mesh=mesh, name=name, radio_material=scene_radio_material(scene, material)
    )
    scene.edit(add=obj)
    if name not in scene.objects:
        raise RuntimeError(
            f"the scene reports no object named {name!r} after edit(); it holds "
            f"{sorted(scene.objects)}. The mesh was not added, so any channel solved "
            "from this scene would describe a world without the body in it."
        )
    points = np.asarray(vertices, dtype=np.float64)
    return ImportedObjectReport(
        name=name,
        vertex_count=int(points.shape[0]),
        face_count=int(np.asarray(faces).shape[0]),
        bounding_box_min=tuple(float(v) for v in points.min(axis=0)),
        bounding_box_max=tuple(float(v) for v in points.max(axis=0)),
        material=material,
    )


def _mitsuba() -> Any:
    try:
        import mitsuba
    except ImportError as exc:  # pragma: no cover - exercised only without the runtime
        raise RuntimeError(
            "the Sionna runtime is not importable from this interpreter. It lives in "
            "its own environment; run with that interpreter, e.g. "
            "/home/gsh/.local/opt/sionna/bin/python, rather than adding the runtime to "
            f"the project's CPU dependencies ({exc})"
        ) from exc
    return mitsuba


def _sionna_rt() -> Any:
    try:
        import sionna.rt
    except ImportError as exc:  # pragma: no cover - exercised only without the runtime
        raise RuntimeError(
            "sionna.rt is not importable from this interpreter; use the Sionna "
            f"environment's interpreter ({exc})"
        ) from exc
    return sionna.rt
