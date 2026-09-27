"""Action requests preserve physics semantics, interruptibility and reset behaviour."""

import numpy as np
import pytest

from sim2sense_fall.humans.actions import (
    ActionClip,
    ActionConfig,
    ActionState,
    Posture,
)
from sim2sense_fall.humans.rotations import quaternion_to_matrix
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
    assert apply(state, target, speed=0.4).mode == "braking_for_action"
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
    # Zero damping is now legal (fully passive collapse; the solver-NaN regime
    # is closed by the authored maxJointVelocity clamp, not by damping). Above
    # 1 would amplify the damping. NaN is caught by the generic finiteness
    # check before the range check.
    assert ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50, fall_damping_scale=0.0)
    for bad_scale in (-0.1, 1.5):
        with pytest.raises(ValueError, match="fall damping scale"):
            ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50, fall_damping_scale=bad_scale)
    with pytest.raises(ValueError, match="must be finite"):
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50, fall_damping_scale=float("nan"))


def test_new_action_keys_reach_the_state_machine():
    intent = KeyboardIntent()
    for key, name in {"B": "bend", "N": "sit", "G": "get_up", "C": "crouch", "V": "stand"}.items():
        intent.event(key, True)
        assert intent.action_requested == name
        intent.action_requested = None
        intent.event(key, False)


def test_posture_switch_mid_transition_is_continuous():
    state, target = fixture()
    state.postures = {
        "crouch": Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {}),
        "bend": Posture(np.array([0.2, 0.1]), 0.8, np.zeros(3), {}),
    }
    state.request("crouch", time_s=0, heading_rad=0, measured_joints=target.joints)
    for _ in range(4):
        crouching = apply(state, target)
    state.request("bend", time_s=1, heading_rad=0, measured_joints=crouching.joints)
    switched = apply(state, target)
    # The new transition continues from the pose on the floor, not from standing.
    assert np.linalg.norm(switched.joints - crouching.joints) < 0.2
    assert switched.mode == "bending"
    for _ in range(12):
        settled = apply(state, target)
    assert settled.mode == "bend"
    assert np.allclose(settled.joints, [0.2, 0.1])


def test_v_returns_from_any_posture_and_releases_movement():
    state, target = fixture()
    state.postures = {
        "crouch": Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {}),
        "bend": Posture(np.array([0.2, 0.1]), 0.8, np.zeros(3), {}),
    }
    state.request("bend", time_s=0, heading_rad=0, measured_joints=target.joints)
    for _ in range(12):
        apply(state, target)
    assert state.command((1, 0)) == (0, 0)
    state.request("stand", time_s=2, heading_rad=0, measured_joints=target.joints)
    for _ in range(12):
        upright = apply(state, target)
    assert upright.mode == "stand"
    assert state.command((1, 0)) == (1, 0)


def test_get_up_only_from_fallen_and_plays_out_to_standing():
    clip = ActionClip(
        joints=np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0]]),
        heights_m=np.array([0.2, 0.5, 0.95, 0.95]),
        offsets_xy=np.array([[0.0, 0.0], [0.05, 0.0], [0.2, 0.0], [0.3, 0.0]]),
        tilts=np.zeros((4, 3)),
        frame_dt_s=0.1,
        provenance={},
    )
    state, target = fixture()
    state = ActionState(
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50),
        {"crouch": Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {})},
        playback_clip=clip,
    )
    position = np.array([1.0, 2.0, 0.2])
    # Without the fallen latch the request is a no-op, not an error.
    assert state.request(
        "get_up", time_s=0, heading_rad=0,
        measured_joints=target.joints, measured_position=position,
    ) is False
    state.request("fall", time_s=1, heading_rad=0, measured_joints=target.joints)
    state.observe(time_s=2, root_height_m=0.3, tilt_deg=80, standing_height_m=1, body_impact=True)
    assert state.phase == "fallen"
    from sim2sense_fall.humans.rotations import axis_angle_to_quaternion
    assert state.request(
        "get_up", time_s=3, heading_rad=0,
        measured_joints=target.joints, measured_position=position,
        measured_quaternion=axis_angle_to_quaternion(np.array([0.0, np.deg2rad(85.0), 0.0])),
    ) is True
    assert state.falling is False, "recovery must run with assistance restored"
    assert state.command((1, 0)) == (0, 0)
    assert state.suppresses_stance
    # Replay runs ~0.3 s + 0.5 s blend; step past it, then the return blend runs.
    seen_getting_up = False
    for _ in range(20):
        out = apply(state, target)
        seen_getting_up |= out.mode == "getting_up"
        if out.mode == "getting_up":
            # The root follows the anchored clip path in world coordinates.
            assert out.position[0] >= 1.0 - 1e-9 and out.position[2] <= 0.95 + 1e-9
    assert seen_getting_up
    assert state.phase == "idle"
    for _ in range(12):
        upright = apply(state, target)
    assert upright.mode == "stand"
    assert state.command((1, 0)) == (1, 0)
    assert state.history[0]["fallen_time_s"] == 2, "fall record survives the recovery"


def test_get_up_without_a_configured_clip_is_a_no_op():
    state, target = fixture()
    state.request("fall", time_s=1, heading_rad=0, measured_joints=target.joints)
    state.observe(time_s=2, root_height_m=0.3, tilt_deg=80, standing_height_m=1, body_impact=True)
    assert state.request(
        "get_up", time_s=3, heading_rad=0,
        measured_joints=target.joints, measured_position=np.zeros(3),
        measured_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    ) is False
    assert state.phase == "fallen"


def test_without_yaw_strips_turn_and_keeps_pitch():
    from sim2sense_fall.humans.actions import _without_yaw as strip
    from sim2sense_fall.humans.rotations import axis_angle_to_matrix
    tilt = np.array([0.0, 0.6, np.deg2rad(30.0)])  # 俯仰 + 30° 自转
    stripped = strip(tilt)
    rotation = axis_angle_to_matrix(stripped)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    assert abs(np.degrees(yaw)) <= 1e-6
    # 俯仰保留: 前向分量仍倾斜
    assert rotation[2, 2] < 1.0


def test_getup_replay_does_not_inherit_the_clip_turn():
    """CMU/140 起身时自身转了 ~30°: 旧实现把片段 yaw 带进世界朝向,
    混出瞬间传送 ~91°——观感"起立后转几圈"。片段 yaw 必须被剔除,
    回放全程世界朝向锚定在倒地朝向上。"""
    turn = np.deg2rad(30.0)
    clip = ActionClip(
        joints=np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0]]),
        heights_m=np.array([0.2, 0.5, 0.95, 0.95]),
        offsets_xy=np.zeros((4, 2)),
        tilts=np.array([[turn * f, 0.1, 0.0] for f in np.linspace(0, 1, 4)]),
        frame_dt_s=0.1,
        provenance={},
    )
    state, target = fixture()
    state = ActionState(
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50),
        {},
        playback_clip=clip,
    )
    heading = np.deg2rad(40.0)
    state.phase = "fallen"
    from sim2sense_fall.humans.rotations import axis_angle_to_quaternion
    lying_quaternion = axis_angle_to_quaternion(np.array([0.0, np.deg2rad(85.0), 0.0]))
    assert state.request(
        "get_up", time_s=1, heading_rad=heading,
        measured_joints=target.joints,
        measured_position=np.array([0.0, 0.0, 0.2]),
        measured_quaternion=lying_quaternion,
    ) is True
    saw_replay = False
    max_deviation = 0.0
    for _ in range(24):
        out = apply(state, target)  # dt 0.1 s
        if state.phase == "getting_up" or out.mode == "getting_up":
            saw_replay = True
            rotation = quaternion_to_matrix(out.quaternion)
            world_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            # 片段自带 30° 自然转身: 目标朝向跟随, 但不得超出片段转身量级
            deviation = abs(np.degrees(world_yaw - heading))
            max_deviation = max(max_deviation, deviation)
    assert saw_replay and state.phase == "idle"
    assert max_deviation <= 45.0, (
        f"replay orientation swung {max_deviation:.1f} deg beyond the clip turn"
    )
