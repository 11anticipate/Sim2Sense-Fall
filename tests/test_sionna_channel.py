"""Regressions for phase, fixed time/delay coordinates, and geometry conversion."""

from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.scenes.geometry import WorldShape, world_shapes
from sim2sense_fall.scenes.planner import plan_scene
from sim2sense_fall.scenes.spec import load_scene_spec
from sim2sense_fall.sionna.apartment import apartment_geometry, triangulate
from sim2sense_fall.sionna.channel import (
    complex_amplitudes,
    delay_grid,
    paths_to_cir,
    regular_frame_indices,
)


def test_imaginary_paths_preserved_and_opposite_phases_cancel():
    a = complex_amplitudes((np.array([0.0, 0.0]), np.array([2.0, -2.0])))
    assert np.sum(abs(a) ** 2) == 8
    taps = paths_to_cir(a, np.array([1e-8, 1e-8]), np.ones(2), delay_grid(1e8, 1e-7))
    np.testing.assert_allclose(taps, 0, atol=1e-14)


def test_absolute_delay_does_not_shift_when_first_path_disappears():
    grid = delay_grid(1e8, 1e-7)
    a = np.array([1 + 1j, 2 - 1j])
    tau = np.array([1e-8, 5e-8])
    first = paths_to_cir(a, tau, np.ones(2), grid)
    second = paths_to_cir(a, tau, np.array([0, 1]), grid)
    assert np.argmax(abs(second)) == 5
    assert second[5] == pytest.approx(first[5])
    assert np.iscomplexobj(second)
    with pytest.raises(ValueError, match="outside"):
        paths_to_cir(a, tau * 100, np.ones(2), grid)


def test_uniform_frame_selection_and_degenerate_requests():
    indices = regular_frame_indices(145, 12)
    assert len(indices) <= 12
    assert len(set(np.diff(indices))) == 1
    with pytest.raises(ValueError):
        regular_frame_indices(1, 12)


@pytest.mark.parametrize("cylinder", [False, True])
def test_primitives_have_outward_winding_and_are_closed(cylinder):
    shape = WorldShape("/test", (2, 3, 4), (1, 1, 2), 0.0, cylinder)
    v, f = triangulate(shape)
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    assert np.all(np.sum(normals * (v[f].mean(axis=1) - shape.center), axis=1) > 0)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)


def test_apartment_preserves_each_authored_part_and_world_bounds():
    spec = load_scene_spec(Path("configs/scenes/indoor_apartment.yaml"))
    plan = plan_scene(spec)
    shapes = world_shapes(plan.prims)
    converted = apartment_geometry(spec)
    assert len(converted) == len(shapes)
    assert any(g.category == "ceiling" for g in converted)
    for g in converted:
        shape = shapes[g.path]
        low, high = shape.bounds
        # Cylinder tessellation is inscribed; boxes are exact.
        assert np.all(g.vertices.min(axis=0) >= np.array(low) - 1e-9)
        assert np.all(g.vertices.max(axis=0) <= np.array(high) + 1e-9)
        if not shape.cylinder:
            np.testing.assert_allclose(g.vertices.min(axis=0), low, atol=1e-9)
            np.testing.assert_allclose(g.vertices.max(axis=0), high, atol=1e-9)
