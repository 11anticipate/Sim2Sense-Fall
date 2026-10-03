"""Walk-coupled trip (route-A increment): perturb while stepping, measure the outcome.

An F press while moving is a trip, not a fall request: the gait keeps stepping
(the stumble steps are the body's own recovery attempt), the pelvis assist fades
via the "tripping" mode, and a bounded forward shove is the only horizontal
authority. Two exits, both measured: the fall thresholds hand the body to the
drive-release collapse, and a survived window returns control to locomotion
with the stumble on the record. The frozen command after a trip fall must be
the CURRENT stumbling pose, not the pre-trip one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim2sense_fall.humans.actions import ActionConfig, ActionState, Posture
from sim2sense_fall.humans.teleop import TeleopTarget

DT_S = 1.0 / 120.0


def fixture(**config_overrides) -> tuple[ActionState, TeleopTarget]:
    config = ActionConfig(1, 0.025, 60, 0.25, 0, 0.6, 50, **config_overrides)
    state = ActionState(
        config,
        Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {}),
    )
    target = TeleopTarget(
        np.array([0.1, 0.2]),
        np.zeros(2),
        np.array([0.0, 0.0, 0.9]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        "forward",
    )
    return state, target


def apply(state: ActionState, target: TeleopTarget) -> TeleopTarget:
    return state.apply(
        target,
        dt_s=DT_S,
        speed_m_s=0.4,
        idle_joints=np.zeros(2),
        idle_height_m=1,
        idle_tilt=np.zeros(3),
        heading_rad=0.0,
    )


def test_trip_accepts_only_outside_other_phases_and_keeps_walking():
    state, target = fixture()
    assert state.request("trip", time_s=0.0, heading_rad=0.0, measured_joints=target.joints)
    assert state.tripping and not state.falling
    assert not state.suppresses_stance, "the stumble steps need the stance corrections"
    out = apply(state, target)
    assert out.mode == "tripping"
    assert np.allclose(out.joints, target.joints), "the gait must keep stepping"


def test_trip_snags_the_swing_foot_backward_bounded():
    """The perturbation is a caught toe at the ankle, not a COM shove.

    Pushing the centre of mass accelerates the feet along with the body and
    nothing trips (trip_walk_v1..v3: 450 N, peak tilt 9.8 deg); the drag must
    point BACKWARD along the walking heading so it arrests the swing foot
    while the body's momentum continues.
    """

    state, target = fixture()
    state.request(
        "trip", time_s=0.0, heading_rad=0.0, measured_joints=target.joints,
        trip_foot="right_ankle",
    )
    assert state.trip_foot == "right_ankle"
    peak = state.trip_force(0.1)
    assert peak[0] < 0 and float(np.linalg.norm(peak)) == pytest.approx(
        state.config.trip_force_n
    ), "the snag drags the swing foot backward"
    assert float(np.linalg.norm(state.trip_force(0.4))) == 0.0, (
        "the snag is a bounded impulse, not a sustained pull"
    )


def test_trip_falls_into_release_when_balance_is_measured_lost():
    state, target = fixture()
    state.request("trip", time_s=0.0, heading_rad=0.0, measured_joints=target.joints)
    stumbling = None
    for step in range(int(0.6 / DT_S)):
        out = apply(state, target)
        stumbling = out
        state.observe(
            time_s=step * DT_S,
            root_height_m=0.8,
            tilt_deg=20.0,
            standing_height_m=0.9,
            body_impact=False,
        )
    assert state.tripping, "upright and unimpacted must keep stumbling"
    assert stumbling is not None
    state.observe(
        time_s=0.7,
        root_height_m=0.45,
        tilt_deg=60.0,
        standing_height_m=0.9,
        body_impact=True,
    )
    assert state.falling and not state.fall_replay_engaged, (
        "the measured fall must hand the body to the drive-release path"
    )
    assert state.fall_report()["events"][-1]["mechanism"] == "trip_shove_release_fall"
    # The frozen command stream must be the CURRENT stumbling pose: apply() keeps
    # re-capturing it during the trip, so the release does not rewind the body.
    assert np.allclose(state.fall_pose, stumbling.joints)


def test_trip_window_expiry_returns_to_locomotion_as_survived():
    state, target = fixture(trip_window_s=0.5)
    state.request("trip", time_s=0.0, heading_rad=0.0, measured_joints=target.joints)
    for step in range(int(0.6 / DT_S)):
        apply(state, target)
        state.observe(
            time_s=step * DT_S,
            root_height_m=0.88,
            tilt_deg=15.0,
            standing_height_m=0.9,
            body_impact=False,
        )
    assert not state.tripping and not state.falling, "a caught stumble returns control"
    report = state.fall_report()
    assert report["outcome"] == "stumble_survived"
    assert report["events"][-1]["mechanism"] == "trip_shove_walk_recovery"
    assert report["events"][-1]["fallen_time_s"] is None, (
        "a survived stumble must never carry a fall label"
    )


def test_trip_request_rejected_while_not_idle():
    state, target = fixture()
    assert state.request("trip", time_s=0.0, heading_rad=0.0, measured_joints=target.joints)
    assert not state.request("trip", time_s=0.2, heading_rad=0.0, measured_joints=target.joints)
    state.observe(
        time_s=0.3, root_height_m=0.4, tilt_deg=60.0,
        standing_height_m=0.9, body_impact=True,
    )
    assert state.falling
    assert not state.request("trip", time_s=0.4, heading_rad=0.0, measured_joints=target.joints)


def test_trip_knobs_are_validated():
    with pytest.raises(ValueError, match="trip_force_n"):
        ActionConfig(1, 0.025, 60, 0.25, 0, 0.6, 50, trip_force_n=0.0)
    with pytest.raises(ValueError, match="must be finite"):
        ActionConfig(1, 0.025, 60, 0.25, 0, 0.6, 50, trip_window_s=float("nan"))
    with pytest.raises(ValueError, match="trip_window_s"):
        ActionConfig(1, 0.025, 60, 0.25, 0, 0.6, 50, trip_window_s=-1.0)


def test_trip_control_scale_knob_validated():
    with pytest.raises(ValueError, match="trip_control_scale"):
        ActionConfig(1, 0.025, 60, 0.25, 0, 0.6, 50, trip_control_scale=1.5)


def test_consume_trip_release_is_one_shot_and_separate_from_replay():
    state, _ = fixture()
    assert state.consume_trip_release() is False
    state.request(
        "trip", time_s=0.0, heading_rad=0.0, measured_joints=np.zeros(2),
        trip_foot="left_ankle",
    )
    state.observe(
        time_s=0.5, root_height_m=0.4, tilt_deg=60.0,
        standing_height_m=0.9, body_impact=True,
    )
    assert state.falling
    assert state.consume_trip_release() is True
    assert state.consume_trip_release() is False, "the release fires exactly once"
    assert state.consume_fall_replay_release() is False, (
        "a trip fall has no replay authority to release"
    )
