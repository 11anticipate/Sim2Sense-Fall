"""Swing-lift baking on the real forward cycle, plus argument validation.

The bake exists because the retargeted swing foot skims the floor (median lift
14.7 mm right / 32.1 mm left on the shipped cycle) and is dragged wherever it
touches, which dominates the measured slip. These tests pin the three properties
that make the fix safe: it actually raises the lowest swing point, it leaves
mask-stance frames untouched, and it keeps the loop closure closed.
"""

from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.contact_control import capsule_bottom
from sim2sense_fall.humans.rig import forward_kinematics, plan_human_rig
from sim2sense_fall.humans.teleop import (
    SWING_LIFT_MAX_M,
    load_gait,
    load_keyboard_config,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lifted_gait():
    settings = load_keyboard_config(ROOT / "configs/humans/keyboard.yaml", ROOT)
    plan = plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))
    dt = 1.0 / 120.0
    # The shipped spec already carries swing_lift_m; strip it so "raw" is the
    # unlifted reference the bake is compared against.
    raw_spec = {
        key: value
        for key, value in settings["gaits"]["forward"].items()
        if key != "swing_lift_m"
    }
    raw = load_gait(raw_spec, plan, dt_s=dt, max_stance_slip_m_s=0.15)
    lifted = load_gait(
        {**raw_spec, "swing_lift_m": 0.05},
        plan,
        dt_s=dt,
        max_stance_slip_m_s=0.15,
    )
    return plan, raw, lifted


def _swing_heights(plan, gait):
    names = list(plan.dof_names)
    mask = np.asarray(gait.support_mask)
    heights = np.zeros((len(gait.joints), 2))
    for index in range(len(gait.joints)):
        phase = index / (len(gait.joints) - 1)
        q, height = gait.sample(phase)
        poses = forward_kinematics(
            plan,
            dict(zip(names, q, strict=True)),
            root_position=(0.0, 0.0, height),
            root_rotation=gait.tilt(phase),
        )
        for column, side in enumerate(("left", "right")):
            heights[index, column] = capsule_bottom(plan, poses, f"{side}_ankle")
    return mask, heights


def test_bake_raises_the_lowest_swing_points(lifted_gait):
    plan, raw, lifted = lifted_gait
    raw_mask, raw_heights = _swing_heights(plan, raw)
    lifted_mask, lifted_heights = _swing_heights(plan, lifted)
    assert np.array_equal(raw_mask, lifted_mask)
    for column in range(2):
        swing = ~raw_mask[:, column]
        # The edges of a swing run are touchdown/liftoff and legitimately stay on
        # the floor (the sine profile is zero there), so a min/count comparison
        # just re-counts edge frames. The lift shows up in the distribution: both
        # the shallow tail and the typical swing height rise.
        assert np.percentile(lifted_heights[swing, column], 10) > np.percentile(
            raw_heights[swing, column], 10
        )
        assert np.median(lifted_heights[swing, column]) > np.median(
            raw_heights[swing, column]
        )
        assert lifted.provenance["swing_lift_frames"] > 0


def test_bake_leaves_mask_stance_frames_untouched(lifted_gait):
    _plan, raw, lifted = lifted_gait
    mask = np.asarray(raw.support_mask)
    interior = np.zeros(len(mask), dtype=bool)
    interior[1:-1] = mask[1:-1].all(axis=1) & mask[:-2].all(axis=1) & mask[2:].all(axis=1)
    assert interior.any()
    assert np.allclose(lifted.joints[interior], raw.joints[interior])


def test_bake_keeps_loop_closure(lifted_gait):
    _plan, raw, lifted = lifted_gait
    raw_closure = float(np.rad2deg(np.abs(raw.joints[-1] - raw.joints[0])).max())
    lifted_closure = float(np.rad2deg(np.abs(lifted.joints[-1] - lifted.joints[0])).max())
    assert lifted_closure <= raw_closure + 1.0
    assert lifted.provenance["endpoint_joint_difference_deg_after_lift"] <= raw_closure + 1.0


def test_bake_rejects_bad_arguments():
    from sim2sense_fall.humans.teleop import Gait, bake_swing_clearance

    gait = Gait(
        np.zeros((4, 57)),
        np.full(4, 0.9),
        1.0,
        0.5,
        {},
        support_mask=np.zeros((4, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="clearance_m"):
        bake_swing_clearance(gait, object(), clearance_m=SWING_LIFT_MAX_M + 0.01)
    with pytest.raises(ValueError, match="clearance_m"):
        bake_swing_clearance(gait, object(), clearance_m=0.0)
    bare = Gait(np.zeros((4, 57)), np.full(4, 0.9), 1.0, 0.5, {})
    with pytest.raises(ValueError, match="support mask"):
        bake_swing_clearance(bare, object(), clearance_m=0.05)


def test_swing_lift_max_bound_matches_config_validation():
    # The loader rejects anything outside (0, SWING_LIFT_MAX_M]; the bound is
    # re-validated inside bake_swing_clearance, so one constant guards both.
    assert 0 < SWING_LIFT_MAX_M <= 0.2
