"""CPU tests for the Sionna RT import boundary.

The runtime (``mitsuba`` / ``sionna.rt``) is not installed in this environment and must
not be needed to run these tests. That is the point of the split in
``sim2sense_fall.sionna.mesh_import``: everything that can be checked without the ray
tracer is checked without it, so a CPU machine -- and CI -- still covers the array
contract and the material model. Only the two functions that genuinely need Mitsuba are
left untested here, and they are exercised by
``scripts/sionna/import_fall_mesh.py`` in the Sionna environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim2sense_fall.sionna import (
    build_mitsuba_mesh,
    candidate_mesh_arrays,
    human_tissue_material,
    place_mesh_in_scene,
)


def _triangle() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# the array contract, checked without the runtime
# ---------------------------------------------------------------------------


def test_the_package_imports_without_the_sionna_runtime():
    """A CPU machine must be able to import this package.

    The project keeps heavyweight runtimes behind lazy imports on purpose; if this
    module ever grows a module-level ``import mitsuba``, every CPU test run and the CI
    job break, which is how a "small" convenience import becomes a landmine.
    """

    import importlib

    module = importlib.import_module("sim2sense_fall.sionna.mesh_import")
    assert module is not None
    # And the two runtime-touching calls must fail with an actionable message rather
    # than an ImportError from inside the function.
    vertices, faces = _triangle()
    with pytest.raises(RuntimeError, match="Sionna runtime is not importable"):
        build_mitsuba_mesh(vertices, faces, name="probe")


def test_candidate_arrays_are_cast_to_what_the_ray_tracer_wants():
    vertices, faces = _triangle()
    points, triangles = candidate_mesh_arrays(vertices, faces)
    assert points.dtype == np.float32
    assert triangles.dtype == np.uint32
    assert points.flags["C_CONTIGUOUS"] and triangles.flags["C_CONTIGUOUS"]


def test_a_face_index_outside_the_vertex_array_is_rejected():
    """An out-of-range index is an out-of-bounds read in the tracer, not a cosmetic bug."""

    vertices, _ = _triangle()
    with pytest.raises(ValueError, match="outside the 3 vertices"):
        candidate_mesh_arrays(vertices, np.array([[0, 1, 7]]))


def test_a_negative_face_index_is_rejected():
    vertices, _ = _triangle()
    with pytest.raises(ValueError, match="non-negative"):
        candidate_mesh_arrays(vertices, np.array([[0, 1, -1]]))


def test_non_integer_faces_are_rejected():
    vertices, _ = _triangle()
    with pytest.raises(ValueError, match="must be integers"):
        candidate_mesh_arrays(vertices, np.array([[0.0, 1.0, 2.0]]))


def test_non_finite_vertices_are_rejected():
    vertices, faces = _triangle()
    vertices = vertices.copy()
    vertices[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        candidate_mesh_arrays(vertices, faces)


def test_degenerate_shapes_are_rejected():
    vertices, faces = _triangle()
    with pytest.raises(ValueError, match=r"vertices must be \(V, 3\)"):
        candidate_mesh_arrays(vertices[:, :2], faces)
    with pytest.raises(ValueError, match=r"faces must be \(F, 3\)"):
        candidate_mesh_arrays(vertices, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="at least three vertices"):
        candidate_mesh_arrays(np.zeros((2, 3)), faces)
    with pytest.raises(ValueError, match="at least one face"):
        candidate_mesh_arrays(vertices, np.zeros((0, 3), dtype=np.int64))


# ---------------------------------------------------------------------------
# the material model
# ---------------------------------------------------------------------------


def test_the_default_body_material_is_not_vacuum():
    """``vacuum`` would delete the interaction this whole stage exists to measure.

    Sionna's ITU table has no body-tissue entry, so the temptation is to reach for the
    one material that needs no new assumption. That material is ``vacuum`` (eps_r = 1,
    sigma = 0), which makes the human electromagnetically absent and would produce a
    body that changes no channel at all.
    """

    material = human_tissue_material()
    assert material.relative_permittivity > 1.0
    assert material.conductivity_s_per_m > 0.0


def test_the_material_carries_its_provenance():
    """The values are an assumption, and the output has to say so."""

    material = human_tissue_material()
    payload = material.as_dict()
    assert payload["provenance"] == "modelling assumption, not measured"
    assert "no body-tissue entry" in payload["source"]
    assert payload["model"].startswith("sionna.rt.RadioMaterial")


def test_material_parameters_are_validated():
    with pytest.raises(ValueError, match="at least 1"):
        human_tissue_material(relative_permittivity=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        human_tissue_material(conductivity_s_per_m=-0.1)
    with pytest.raises(ValueError, match="positive"):
        human_tissue_material(thickness_m=0.0)
    with pytest.raises(ValueError, match="finite"):
        human_tissue_material(relative_permittivity=float("nan"))


# ---------------------------------------------------------------------------
# the scene boundary, without the runtime
# ---------------------------------------------------------------------------


def test_placing_a_mesh_without_the_runtime_fails_with_a_readable_message():
    """The failure has to name the environment it needs, not just raise ImportError.

    Either runtime module may be the one that is missing, so the assertion is on the
    shared, actionable part of the message rather than on which of the two it names.
    """

    vertices, faces = _triangle()
    with pytest.raises(RuntimeError, match="not importable from this interpreter"):
        place_mesh_in_scene(
            object(), vertices, faces, name="human", material=human_tissue_material()
        )
