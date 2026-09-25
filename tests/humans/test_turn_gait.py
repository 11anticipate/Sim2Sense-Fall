"""Stepped-turn controller mode on the real turn reference.

Holding A/D used to pivot both flat feet on the floor (the body rotated like a
skater). The controller now plays a verified stepping-turn reference
(CMU/83/83_56: 177 deg, six alternating steps) while the heading follows the
commanded rate. These tests pin the mode logic and the fact that the reference
actually lifts feet, on CPU, without Isaac.
"""

from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.contact_control import capsule_bottom
from sim2sense_fall.humans.rig import forward_kinematics, plan_human_rig
from sim2sense_fall.humans.rotations import (
    matrix_to_axis_angle,
    quaternion_to_matrix,
)
from sim2sense_fall.humans.teleop import (
    TeleopController,
    load_gait,
    load_keyboard_config,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def controller():
    settings = load_keyboard_config(ROOT / "configs/humans/keyboard.yaml", ROOT)
    plan = plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))
    dt = 1.0 / 120.0
    gaits = {
        key: load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=0.15)
        for key, spec in settings["gaits"].items()
    }
    idle = load_gait(settings["idle"], plan, dt_s=dt, max_stance_slip_m_s=0.15)
    turn = load_gait(settings["turn"], plan, dt_s=dt, max_stance_slip_m_s=None)
    ctrl = TeleopController(
        settings["controller"], gaits, idle, plan, np.deg2rad(90.0), turn=turn
    )
    return settings, plan, ctrl


def test_turn_reference_contains_both_feet_lifting(controller):
    _settings, plan, ctrl = controller
    turn = ctrl.turn
    assert turn is not None
    mask = np.asarray(turn.support_mask)
    names = list(plan.dof_names)
    lifts = {side: 0 for side in ("left", "right")}
    for index in range(len(turn.joints)):
        phase = index / (len(turn.joints) - 1)
        q, height = turn.sample(phase)
        poses = forward_kinematics(
            plan,
            dict(zip(names, q, strict=True)),
            root_position=(0.0, 0.0, height),
            root_rotation=turn.tilt(phase),
        )
        for column, side in enumerate(("left", "right")):
            if mask[index, column]:
                continue
            if capsule_bottom(plan, poses, f"{side}_ankle") > 0.03:
                lifts[side] += 1
    assert lifts["left"] > 0 and lifts["right"] > 0


def test_holding_turn_enters_turning_mode_and_lifts_a_foot(controller):
    settings, plan, ctrl = controller
    dt = 1.0 / 120.0
    position = ctrl.position.copy()
    modes = set()
    turning_frames = 0
    max_foot_height = 0.0
    names = list(plan.dof_names)
    for _ in range(int(3.0 / dt)):  # 3 s of held A: ramp + stepping
        target = ctrl.advance((0.0, 1.0), dt, position, ctrl.heading)
        position = np.asarray(ctrl.position)
        modes.add(target.mode)
        if target.mode == "turning":
            turning_frames += 1
            poses = forward_kinematics(
                plan,
                dict(zip(names, target.joints, strict=True)),
                root_position=target.position,
                root_rotation=matrix_to_axis_angle(quaternion_to_matrix(target.quaternion)),
            )
            for side in ("left", "right"):
                max_foot_height = max(
                    max_foot_height, capsule_bottom(plan, poses, f"{side}_ankle")
                )
    assert "turning" in modes
    assert turning_frames > 120  # over a second of stepped turning
    assert max_foot_height > 0.05  # a foot genuinely leaves the floor


def test_releasing_turn_returns_to_stand(controller):
    settings, plan, ctrl = controller
    dt = 1.0 / 120.0
    position = ctrl.position.copy()
    for _ in range(int(2.0 / dt)):
        ctrl.advance((0.0, 1.0), dt, position, ctrl.heading)
        position = np.asarray(ctrl.position)
    modes = []
    for _ in range(int(1.5 / dt)):
        target = ctrl.advance((0.0, 0.0), dt, position, ctrl.heading)
        position = np.asarray(ctrl.position)
        modes.append(target.mode)
    # The turn-rate decay (30 deg/s down through the 8 deg/s activation) keeps the
    # stepping mode alive briefly so the in-flight step finishes; afterwards the
    # state must settle at stand and never report a stale locomotion mode.
    assert modes[-1] == "stand"
    assert "forward" not in modes and "backward" not in modes
    assert modes.count("turning") < int(0.5 / dt)


def test_turn_mode_without_turn_gait_stays_stand(controller):
    settings, plan, ctrl = controller
    dt = 1.0 / 120.0
    gaits = {
        key: load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=0.15)
        for key, spec in settings["gaits"].items()
    }
    idle = load_gait(settings["idle"], plan, dt_s=dt, max_stance_slip_m_s=0.15)
    plain = TeleopController(settings["controller"], gaits, idle, plan, np.deg2rad(90.0))
    position = plain.position.copy()
    modes = set()
    for _ in range(int(1.5 / dt)):
        target = plain.advance((0.0, 1.0), dt, position, plain.heading)
        position = np.asarray(plain.position)
        modes.add(target.mode)
    assert modes == {"stand"}
