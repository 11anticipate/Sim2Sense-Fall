"""Keyboard semantics and bounded targets independent of Isaac or licensed data."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.rig import plan_human_rig
from sim2sense_fall.humans.teleop import (
    Gait,
    KeyboardIntent,
    TeleopController,
    load_keyboard_config,
)
from sim2sense_fall.humans.usd_human import HumanRuntime

ROOT = Path(__file__).resolve().parents[2]


def test_keyboard_repeat_release_brake_and_reset():
    keys = KeyboardIntent()
    for _ in range(20):
        keys.event("W", True)
    assert keys.command() == (1, 0)
    keys.event("S", True)
    assert keys.command() == (0, 0)
    keys.event("S", False)
    keys.event("A", True)
    assert keys.command() == (1, 1)
    keys.event("SPACE", True)
    assert keys.command() == (0, 0)
    keys.clear()
    assert keys.command() == (0, 0)
    keys.event("R", True)
    assert keys.reset_requested
    keys.reset_requested = False
    keys.event("R", True)
    assert not keys.reset_requested
    keys.event("ESCAPE", True)
    assert keys.quit_requested


def test_gait_is_continuous_at_wrap():
    gait = Gait(
        np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.3]]), np.array([0.9, 0.95, 0.92]), 1.0, 0.5, {}
    )
    assert np.allclose(gait.sample(1 - 1e-8)[0], gait.sample(0)[0], atol=1e-6)
    assert gait.sample(1 - 1e-8)[1] == pytest.approx(gait.sample(0)[1], abs=1e-6)


def test_blocked_root_target_does_not_wind_up_and_release_stops():
    settings = load_keyboard_config(ROOT / "configs/humans/keyboard.yaml", ROOT)
    plan = plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))
    n = len(plan.joints)
    q = np.zeros((3, n))
    q[1] = 0.05
    gait = Gait(q, np.array([0.92, 0.93, 0.92]), 1.0, 0.5, {})
    idle = Gait(np.zeros((3, n)), np.full(3, 0.92), 1.0, 0.0, {})
    ctrl = TeleopController(
        settings["controller"], {"forward": gait, "backward": gait}, idle, plan, 0.0
    )
    actual = np.asarray(plan.spawn_root_position)
    previous = ctrl.joints.copy()
    for _ in range(600):
        target = ctrl.advance((1.0, 1.0), 1 / 120, actual, 0.0)
        assert np.linalg.norm(target.position[:2] - actual[:2]) <= 0.100001
        assert abs(ctrl.heading) <= np.deg2rad(15.0001)
        assert np.abs(target.joints - previous).max() <= 4.0 / 120 + 1e-10
        previous = target.joints
    for _ in range(240):
        target = ctrl.advance((0.0, 0.0), 1 / 120, actual, 0.0)
    assert target.mode == "stand"
    assert np.allclose(target.linear_velocity, 0, atol=1e-9)
    assert np.allclose(target.angular_velocity, 0, atol=1e-9)
    with pytest.raises(ValueError):
        ctrl.advance((1.0, 0.0), float("nan"), actual, 0.0)


def test_contact_velocity_uses_rotated_com_and_angular_motion():
    class Array:
        def __init__(self, value):
            self.value = np.asarray([value], dtype=float)

        def numpy(self):
            return self.value

    runtime = object.__new__(HumanRuntime)
    runtime._rigids = {
        "foot": SimpleNamespace(
            get_coms=lambda: (Array([1, 0, 0]), Array([1, 0, 0, 0])),
            get_velocities=lambda: (Array([1, 0, 0]), Array([0, 0, 2])),
        )
    }
    runtime._world_pose_of = lambda _: (
        np.array([5.0, 2.0, 0.0]),
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    )
    # COM world=(5,3,0); the contact offset (0,-1,0) adds +2 m/s in X.
    assert np.allclose(runtime.link_point_velocity("foot", np.array([5.0, 2.0, 0.0])), [3, 0, 0])
    with pytest.raises(ValueError, match="unknown"):
        runtime.link_point_velocity("missing", np.zeros(3))
