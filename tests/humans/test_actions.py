"""Action requests preserve physics semantics, interruptibility and reset behaviour."""

import numpy as np
import pytest

from sim2sense_fall.humans.actions import ActionConfig, ActionState, Posture
from sim2sense_fall.humans.teleop import KeyboardIntent, TeleopTarget


def fixture() -> tuple[ActionState, TeleopTarget]:
    state = ActionState(
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50),
        Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {}),
    )
    target = TeleopTarget(
        np.zeros(2),
        np.zeros(2),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        "stand",
    )
    return state, target


def apply(state: ActionState, target: TeleopTarget, speed: float = 0) -> TeleopTarget:
    return state.apply(
        target,
        dt_s=0.1,
        speed_m_s=speed,
        idle_joints=np.zeros(2),
        idle_height_m=1,
        idle_tilt=np.zeros(3),
        heading_rad=0,
    )


def test_action_keys_are_edges_and_focus_loss_cancels_pending_request():
    intent = KeyboardIntent()
    intent.event("C", True)
    assert intent.action_requested == "crouch"
    intent.action_requested = None
    intent.event("C", True)
    assert intent.action_requested is None
    intent.event("V", True)
    assert intent.action_requested == "stand"
    intent.clear()
    assert intent.action_requested is None


def test_crouch_waits_for_braking_then_can_reverse_without_pose_jump():
    state, target = fixture()
    state.request("crouch", time_s=0, heading_rad=0, measured_joints=target.joints)
    assert state.command((1, 0)) == (0, 0)
    assert apply(state, target, speed=0.4).mode == "braking_for_crouch"
    assert state.blend == 0
    for _ in range(4):
        low = apply(state, target)
    state.request("stand", time_s=1, heading_rad=0, measured_joints=low.joints)
    next_target = apply(state, target)
    assert next_target.mode == "standing_up"
    assert np.linalg.norm(next_target.joints - low.joints) < 0.2
    for _ in range(12):
        upright = apply(state, target)
    assert upright.mode == "stand"
    assert state.command((1, 0)) == (1, 0)


def test_fall_request_is_not_a_fall_label_and_stays_latched_until_reset():
    state, target = fixture()
    state.request("fall", time_s=2, heading_rad=np.pi / 2, measured_joints=target.joints)
    assert state.mode == "falling"
    assert state.fall_force(2.1) == pytest.approx([0, -350, 0])
    assert np.allclose(state.fall_force(2.3), 0)
    state.observe(time_s=3, root_height_m=0.3, tilt_deg=80, standing_height_m=1, body_impact=False)
    assert state.mode == "falling"
    state.observe(time_s=3.1, root_height_m=0.3, tilt_deg=80, standing_height_m=1, body_impact=True)
    assert state.mode == "fallen"
    state.request("stand", time_s=4, heading_rad=0, measured_joints=target.joints)
    assert state.command((1, 1)) == (0, 0)
    state.reset()
    assert state.command((1, 1)) == (1, 1)
    assert state.impact_time_s is None
    assert len(state.history) == 1
    assert state.history[0]["fallen_time_s"] == 3.1
    assert state.history[0]["impact_time_s"] == 3.1


def test_fall_report_survives_the_reset_that_ended_the_session():
    state, target = fixture()
    state.request("fall", time_s=2, heading_rad=0, measured_joints=target.joints)
    state.observe(time_s=3, root_height_m=0.3, tilt_deg=80, standing_height_m=1, body_impact=True)
    state.reset()
    report = state.fall_report()
    assert report["requested_time_s"] == 2
    assert report["impact_time_s"] == 3
    assert report["outcome"] == "fallen"
    assert report["falls_requested"] == 1
    # positive control: the live state really is cleared, so reading it directly
    # is what lost the fall from the report before
    assert state.fall_time_s is None
    assert state.impact_time_s is None
    assert report["state_at_exit"] == "locomotion"


def test_fall_report_separates_no_impact_from_impact_that_never_lay_down():
    untouched = ActionState(
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50),
        Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {}),
    )
    assert untouched.fall_report()["outcome"] is None
    assert untouched.fall_report()["events"] == []

    state, target = fixture()
    state.request("fall", time_s=2, heading_rad=0, measured_joints=target.joints)
    state.observe(time_s=3, root_height_m=0.3, tilt_deg=80, standing_height_m=1, body_impact=False)
    assert state.fall_report()["outcome"] == "no_body_floor_impact"

    propped, target = fixture()
    propped.request("fall", time_s=2, heading_rad=0, measured_joints=target.joints)
    propped.observe(time_s=3, root_height_m=0.9, tilt_deg=10, standing_height_m=1, body_impact=True)
    assert propped.fall_report()["outcome"] == "impacted_not_fallen"
    assert propped.fall_report()["fallen_time_s"] is None


def test_invalid_action_settings_fail_before_physics():
    with pytest.raises(ValueError):
        ActionConfig(float("nan"), 0.025, 350, 0.25, 0, 0.6, 50)
    with pytest.raises(ValueError):
        ActionConfig(1, 0.025, 350, 0.25, 2, 0.6, 50)
