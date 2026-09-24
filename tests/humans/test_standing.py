from dataclasses import replace

import numpy as np
import pytest

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.rig import plan_human_rig
from sim2sense_fall.humans.standing import standing_metrics


def test_partial_topple_fails_stationary_gate_even_without_fall_label():
    root = np.array([[0, 0, 1.0], [0.1, 0, 0.72]])
    trunk = np.array([[0, 0, 1.0], [0.7, 0, 0.7]])
    m = standing_metrics(
        root, trunk, reference_root_z=1.0, max_drop_m=0.05, max_tilt_deg=15.0, max_drift_m=0.05
    )
    assert not m["passed"]
    assert m["max_pelvis_drop_m"] == pytest.approx(0.28)


def test_flat_sole_preserves_lowest_surface_and_skeleton():
    c = load_human_config("configs/humans/human_smpl_neutral.yaml")
    a = plan_human_rig(c)
    b = plan_human_rig(replace(c, rig=replace(c.rig, horizontal_foot_capsules=True)))
    assert a.rest_joint_positions == b.rest_joint_positions
    assert a.spawn_root_position == b.spawn_root_position
    for name in ("left_ankle", "right_ankle"):
        old, new = a.link(name).capsule, b.link(name).capsule
        assert new.direction[2] == pytest.approx(0, abs=1e-8)
        assert new.lowest_point_z() == pytest.approx(old.lowest_point_z(), abs=1e-8)
    for old, new in zip(a.joints, b.joints, strict=True):
        assert old.max_force == new.max_force
