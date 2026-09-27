"""Sanitized AMASS swing arcs: anchor closure, endpoint rest, retargeting, extraction."""

import math
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.contact_gait import (
    AmassSwingShape,
    ContactFootPlanner,
    ContactGaitConfig,
    _quintic,
)
from sim2sense_fall.humans.rig import LinkTransform, plan_human_rig
from sim2sense_fall.humans.rotations import axis_angle_to_matrix

REPO = Path(__file__).resolve().parents[2]


def synthetic_shape(displacement_m: float = 1.0, samples: int = 33) -> AmassSwingShape:
    grid = np.linspace(0., 1., samples)
    return AmassSwingShape(
        grid=grid,
        along_m=grid * displacement_m,
        cross_m=.1 * np.sin(np.pi * grid),
        height_m=.04 * np.sin(np.pi * grid),
        pitch_rad=.6 * np.sin(np.pi * grid),
        displacement_m=displacement_m,
    )


def pose(x: float, y: float, yaw_rad: float = 0., z: float = .07) -> LinkTransform:
    return LinkTransform(axis_angle_to_matrix(np.array([0., 0., yaw_rad])),
                         np.array([x, y, z]))


def test_quintic_has_zero_endpoint_slope_and_unit_span():
    assert _quintic(0.) == 0. and _quintic(1.) == 1.
    h = 1e-6
    for endpoint in (0., 1.):
        slope = (_quintic(endpoint + h) - _quintic(endpoint - h)) / (2 * h)
        assert slope == pytest.approx(0., abs=1e-4)


def test_sample_joins_anchors_exactly_and_rests_at_contacts():
    shape = synthetic_shape()
    start, end = pose(1., 2., yaw_rad=.3), pose(1.4, 2.4, yaw_rad=.65)
    for u, expected in ((0., start), (1., end)):
        sampled = shape.sample(u, start, end)
        np.testing.assert_allclose(sampled.translation, expected.translation, atol=1e-12)
        np.testing.assert_allclose(sampled.rotation, expected.rotation, atol=1e-9)
    h = 1e-4
    for u in (0., 1.):
        direction = 1. if u == 0. else -1.
        velocity = (shape.sample(u + direction * h, start, end).translation
                    - shape.sample(u, start, end).translation) / h
        assert np.linalg.norm(velocity) < 1e-3


def test_sample_retargets_horizontal_stride_and_keeps_vertical_arc():
    shape = synthetic_shape(displacement_m=1.0)
    height_mid = .04  # shape height at u=.5
    pitch_mid = .6
    # Same start, two landings: double the stride, and one turned 90 degrees.
    straight = shape.sample(.5, pose(0., 0.), pose(.8, 0.))
    turned = shape.sample(.5, pose(0., 0., yaw_rad=.5), pose(0., .8, yaw_rad=.5))
    # The vertical arc and foot pitch are per-step properties, unscaled.
    assert straight.translation[2] == pytest.approx(.07 + height_mid)
    assert turned.translation[2] == pytest.approx(.07 + height_mid)
    # Lateral bow scales with the stride and follows the landing heading.
    assert straight.translation[1] == pytest.approx(.1 * .8, abs=1e-9)
    assert turned.translation[0] == pytest.approx(-.1 * .8, abs=1e-12)
    yaw_mid = math.atan2(turned.rotation[1, 0], turned.rotation[0, 0])
    assert yaw_mid == pytest.approx(.5, abs=1e-6)
    # Sagittal pitch is invariant under the heading change: unwind the yaw and
    # the local frame shows the same pitch rotation.
    for sampled, yaw in ((straight, 0.), (turned, .5)):
        local = axis_angle_to_matrix(np.array([0., 0., -yaw])) @ sampled.rotation
        assert local[0, 2] == pytest.approx(math.sin(pitch_mid), rel=1e-6)


def test_shape_rejects_unsanitized_samples():
    grid = np.linspace(0., 1., 17)
    with pytest.raises(ValueError, match="displacement"):
        synthetic_shape().__class__(grid=grid, along_m=grid, cross_m=np.zeros_like(grid),
                                    height_m=np.zeros_like(grid), pitch_rad=np.zeros_like(grid),
                                    displacement_m=0.)
    with pytest.raises(ValueError, match="re-based"):
        AmassSwingShape(grid=grid, along_m=grid, cross_m=np.zeros_like(grid),
                        height_m=np.ones_like(grid) * .04, pitch_rad=np.zeros_like(grid),
                        displacement_m=1.)


@pytest.mark.parametrize("kwargs", [{"swing_shape": "linear"}, {"root_bob": "yes"}])
def test_config_rejects_invalid_shape_settings(kwargs):
    with pytest.raises(ValueError):
        ContactGaitConfig(**kwargs)


def test_load_gait_extracts_shapes_and_clips_root_bob_at_reach_cap():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.teleop import load_gait

    plan = plan_human_rig(load_human_config(REPO / "configs/humans/human_smpl_multiaxis.yaml"))
    spec = {
        "file": REPO / "data/humans/amass_raw/CMU/08/08_04_poses.npz",
        "start_s": 1.9667,
        "duration_s": 1.3083,
        "contact_cycle": {"cycle_distance_m": .6, "stance_fraction": .62,
                          "clearance_m": .03, "swing_shape": "amass", "root_bob": True},
    }
    gait = load_gait(spec, plan, dt_s=1 / 120)
    assert set(gait.swing_shapes) == {"left", "right"}
    for shape in gait.swing_shapes.values():
        assert shape.displacement_m > .3
        assert shape.height_m.max() > .01
        assert np.abs(shape.pitch_rad).max() > .2
    # The root never asks for an unreachable rise; this forward clip rides at
    # the stride cap for most of the cycle, so its bob clips nearly flat --
    # the dip scaling itself is covered by the synthetic bake test below.
    cap = gait.provenance["contact_cycle"]["root_height_m"]
    assert gait.provenance["contact_cycle"]["root_bob"] is True
    assert gait.height_m.max() <= cap + 1e-9
    # The baked cycle stays closed across the seam.
    np.testing.assert_allclose(gait.joints[-1], gait.joints[0], atol=1e-12)


def test_bake_scales_root_bob_deviations_below_reach_cap():
    import copy

    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.contact_gait import bake_contact_cycle
    from sim2sense_fall.humans.teleop import load_gait

    plan = plan_human_rig(load_human_config(REPO / "configs/humans/human_smpl_multiaxis.yaml"))
    spec = {
        "file": REPO / "data/humans/amass_raw/CMU/08/08_04_poses.npz",
        "start_s": 1.9667,
        "duration_s": 1.3083,
        "contact_cycle": {"cycle_distance_m": .6, "stance_fraction": .62,
                          "clearance_m": .03, "root_bob": True},
    }
    reference = bake_contact_cycle(load_gait(dict(spec), plan, dt_s=1 / 120), plan,
                                   ContactGaitConfig(**spec["contact_cycle"]))
    cap = reference.provenance["contact_cycle"]["root_height_m"]
    raw = load_gait({key: value for key, value in spec.items() if key != "contact_cycle"},
                    plan, dt_s=1 / 120)
    count = len(raw.joints)
    dipped = bake_contact_cycle(
        copy.deepcopy(raw), plan, ContactGaitConfig(**spec["contact_cycle"]),
        source_height_m=np.full(count, cap - .02), source_height_scale=.5)
    np.testing.assert_allclose(dipped.height_m, cap - .01, atol=1e-9)
    clipped = bake_contact_cycle(
        copy.deepcopy(raw), plan, ContactGaitConfig(**spec["contact_cycle"]),
        source_height_m=np.full(count, cap + .02), source_height_scale=.5)
    np.testing.assert_allclose(clipped.height_m, cap, atol=1e-9)


def test_bake_produces_seam_closed_com_offset_table():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.teleop import load_gait

    plan = plan_human_rig(load_human_config(REPO / "configs/humans/human_smpl_multiaxis.yaml"))
    spec = {
        "file": REPO / "data/humans/amass_raw/CMU/08/08_04_poses.npz",
        "start_s": 1.9667,
        "duration_s": 1.3083,
        "contact_cycle": {"cycle_distance_m": .6, "stance_fraction": .62,
                          "clearance_m": .03, "swing_shape": "amass", "root_bob": True},
    }
    gait = load_gait(spec, plan, dt_s=1 / 120)
    table = gait.com_offset_m
    assert table.shape == (len(gait.joints), 3)
    assert np.isfinite(table).all()
    # Seam closed and the COM sits inside the body, above the root origin.
    np.testing.assert_allclose(table[-1], table[0], atol=1e-9)
    assert 0.0 < table[:, 2].mean() < 0.5
    assert np.abs(table[:, :2]).max() < 0.3


def test_circular_smooth_preserves_constants_and_periodic_signals():
    from sim2sense_fall.humans.contact_gait import _circular_smooth

    constant = np.full(37, 2.5)
    np.testing.assert_allclose(_circular_smooth(constant, 5), constant, atol=1e-12)
    # A circularly consistent signal passes the moving average nearly unchanged.
    grid = np.linspace(0., 2. * np.pi, 120, endpoint=False)
    wave = np.stack([np.sin(grid), 0.5 * np.cos(2 * grid)], axis=1)
    smoothed = _circular_smooth(wave, 5)
    assert smoothed.shape == wave.shape
    np.testing.assert_allclose(smoothed, wave, atol=0.05)
    # Multi-column input is smoothed per column.
    assert abs(smoothed[:, 1].max() - .5) < .05


def test_planner_swing_goal_tracks_shape_and_keeps_contract():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.rig import forward_kinematics

    plan = plan_human_rig(load_human_config(REPO / "configs/humans/human_smpl_multiaxis.yaml"))
    shape = synthetic_shape(displacement_m=.6)
    planner = ContactFootPlanner(
        plan, ContactGaitConfig(), .4, .8,
        swing_shapes={"forward": {"left": shape, "right": shape}})
    with pytest.raises(ValueError, match="swing_shapes"):
        ContactFootPlanner(plan, ContactGaitConfig(), .4, .8,
                           swing_shapes={"sideways": {"left": shape}})
    q = np.zeros(len(plan.joints))
    root = LinkTransform(np.eye(3), np.array([0., 0., plan.ground_offset_m - .03]))
    actual = forward_kinematics(plan, root_position=(0., 0., plan.ground_offset_m))
    args = {"heading": 0., "turn_rate": 0., "speed": .4, "dt_s": 1 / 120}
    q = planner.correct(q, root, actual, command=1., **args)
    side, (start, end, _elapsed, carried) = next(iter(planner.swings.items()))
    assert carried is shape
    q = planner.correct(q, root, actual, command=1., **args)
    assert planner.last_residual_m < .02
    solved = planner.solvers[side].poses(q, root)[f"{side}_ankle"]
    expected = shape.sample(min(2. / 120 / (.6 / .4 * (1 - .62)), 1.), start, end)
    np.testing.assert_allclose(solved.translation, expected.translation, atol=.02)
