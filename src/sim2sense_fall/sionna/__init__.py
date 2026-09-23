"""Sionna RT integration boundaries.

The heavy runtime (``mitsuba`` / ``sionna.rt``) is imported lazily, so importing this
package on a CPU-only machine -- or in CI -- must not fail. Everything that can be
validated without the runtime is validated in plain numpy.
"""

from __future__ import annotations

from .mesh_import import (
    HumanMaterial,
    ImportedObjectReport,
    build_mitsuba_mesh,
    candidate_mesh_arrays,
    human_tissue_material,
    place_mesh_in_scene,
    scene_radio_material,
)

__all__ = [
    "HumanMaterial",
    "ImportedObjectReport",
    "build_mitsuba_mesh",
    "candidate_mesh_arrays",
    "human_tissue_material",
    "place_mesh_in_scene",
    "scene_radio_material",
]
