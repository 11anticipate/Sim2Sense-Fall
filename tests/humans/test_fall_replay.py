"""The anchored fall replay must command a real descent, not a released collapse.

User review of the exported render frames (2026-09-27) showed the drive-release
fall settling into a twisted supine -- pelvis/legs rotated against the trunk --
while the per-DOF flexion gates all read normal: hip yaw/roll is invisible to
them, and ``impact_time_s`` only marks first contact (the body was still 0.40 m
up, mid-crouch) with the real settling happening ~2 s later. The fix under test
here commands the fall itself as a grounded clip playback (the get-up clip
reversed) so the rest pose comes from human kinematics, while ``observe()``
keeps owning the impact/fallen labels from measured state only.

Evidence: ``docs/progress.md`` 2026-09-27, ``fall_release_ab/G_damped0005``
mesh rebuild (settled pose xy spread 0.93 x 0.83 m, torso/legs twisted).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.actions import ActionConfig, ActionState, Posture, load_action_clip
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import fit_collision_capsules
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig
from sim2sense_fall.humans.rotations import axis_angle_to_quaternion
from sim2sense_fall.humans.teleop import TeleopTarget, load_keyboard_config

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYBOARD_YAML = REPO_ROOT / "configs/humans/keyboard.yaml"
DT_S = 1.0 / 120.0
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))


@pytest.fixture(scope="module")
def plan():
    from common import DEFAULT_MOTIONS, load_inputs  # noqa: PLC0415

    settings = load_keyboard_config(KEYBOARD_YAML, REPO_ROOT)
    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    built = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    built, _ = fit_collision_capsules(
        built, mesh, **{k: v for k, v in settings["collision_fit"].items() if k != "enabled"}
    )
    return built


@pytest.fixture(scope="module")
def settings():
    return load_keyboard_config(KEYBOARD_YAML, REPO_ROOT)


@pytest.fixture(scope="module")
def fall_clip(plan, settings):
    return load_action_clip(settings["fall_replay"], plan, dt_s=DT_S)


def _state(plan, settings, fall_clip, *, config=None) -> tuple[ActionState, TeleopTarget]:
    """An ActionState asked to fall from the clip's own measured standing pose.

    Using the clip's first frame as the "measured" standing body keeps the 0.5 s
    blend-in physical (standing pose to standing pose), the same fixture rule the
    get-up tests use for their lying start.
    """

    from sim2sense_fall.humans.actions import load_posture

    state = ActionState(
        config if config is not None else ActionConfig(**settings["actions"]),
        {"crouch": load_posture(settings["crouch"], plan)},
        None,
        plan=plan,
        fall_clip=fall_clip,
    )
    standing_height_m = float(fall_clip.heights_m[0])
    target = TeleopTarget(
        joints=fall_clip.joints[0].copy(),
        joint_velocities=np.zeros(len(plan.dof_names)),
        position=np.array([*settings["spawn_xy"], standing_height_m]),
        quaternion=axis_angle_to_quaternion(np.asarray(fall_clip.tilts[0], dtype=float)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        mode="stand",
    )
    accepted = state.request(
        "fall",
        time_s=0.0,
        heading_rad=0.0,
        measured_joints=target.joints,
        measured_position=target.position,
        measured_quaternion=target.quaternion,
    )
    assert accepted, "a configured fall replay must accept the F request"
    return state, target


def test_fall_replay_clip_is_a_real_trip_fall(fall_clip, settings):
    """The shipped fall source must be a trip-style fall, not a reversed get-up.

    The reversed get-up read as a deliberate lowering (user review); its
    replacement is CMU 128_09 cropped to the walk -> lurch -> floor window:
    standing start, floor end, and the horizontal momentum of a real trip.
    """

    provenance = fall_clip.provenance
    assert provenance.get("reverse") is None and provenance.get("speed", 1) == 1, (
        "the trip clip plays forward at its own pace"
    )
    standing, lying = float(fall_clip.heights_m[0]), float(fall_clip.heights_m[-1])
    assert 0.75 <= standing <= 0.98, f"clip must start standing, got {standing:.3f} m"
    assert lying < 0.35, f"clip must end on the floor, got {lying:.3f} m"
    travel = float(np.linalg.norm(fall_clip.offsets_xy[-1] - fall_clip.offsets_xy[0]))
    assert travel > 1.0, (
        f"a trip fall must carry horizontal momentum, travel {travel:.2f} m "
        "(the in-place sinking of the reversed get-up is what read as deliberate)"
    )
    duration = float(provenance["duration_s"])
    assert 0.8 <= duration <= 2.0, f"fall window {duration:.2f} s outside the real-fall band"
    assert float(np.abs(fall_clip.offsets_xy[0]).max()) == 0.0, "clip must be re-anchored"


def test_fall_replay_first_command_is_the_measured_body_pose(plan, settings, fall_clip):
    state, target = _state(plan, settings, fall_clip)
    assert state.fall_replay_engaged, "the replay must own the command stream"
    out = state.apply(
        target,
        dt_s=DT_S,
        speed_m_s=0.0,
        idle_joints=np.zeros(len(plan.dof_names)),
        idle_height_m=0.9,
        idle_tilt=np.zeros(3),
        heading_rad=0.0,
    )
    assert out.mode == "falling"
    error_deg = float(np.rad2deg(np.abs(out.joints - target.joints)).max())
    assert error_deg < 0.5, f"the first replay command moved the pose by {error_deg:.1f} deg"
    # The command stream keeps the drives engaged: no release pulse may fire.
    assert float(np.linalg.norm(state.fall_force(0.1))) == 0.0


def test_fall_replay_runs_to_lying_and_releases_once_measured(plan, settings, fall_clip):
    # The full-replay policy (no height truncation) is the one under test here.
    config = replace(
        ActionConfig(**settings["actions"]), fall_replay_release_tilt_deg=None
    )
    state, target = _state(plan, settings, fall_clip, config=config)
    duration_s = float(fall_clip.provenance["duration_s"])
    steps = int((duration_s + 1.0) / DT_S)
    latched_step = None
    first_command_after_latch = None
    first_command_after_latch_time_s = None
    for step in range(steps):
        out = state.apply(
            target,
            dt_s=DT_S,
            speed_m_s=0.0,
            idle_joints=np.zeros(len(plan.dof_names)),
            idle_height_m=0.9,
            idle_tilt=np.zeros(3),
            heading_rad=0.0,
        )
        target = out
        if state.phase == "fallen" and latched_step is None:
            latched_step = step
        elif latched_step is not None and first_command_after_latch is None:
            first_command_after_latch = out
            first_command_after_latch_time_s = (step + 1) * DT_S
        # The measured body crosses the fallen thresholds MID-DESCENT (GPU run
        # 1: latched 1.18 s after trigger while the clip was still commanding).
        # Feeding the same early latch here must not hijack the command stream.
        state.observe(
            time_s=step * DT_S,
            root_height_m=0.45 if step * DT_S >= 1.2 else 0.9,
            tilt_deg=60.0 if step * DT_S >= 1.2 else 5.0,
            standing_height_m=0.9,
            body_impact=step * DT_S >= 1.2,
        )
    assert latched_step is not None, "the fixture must reproduce the early measured latch"
    assert state.fall_replay_finished, "the playback must keep advancing past the latch"
    source_frame = min(
        int(first_command_after_latch_time_s / fall_clip.frame_dt_s),
        len(fall_clip.joints) - 1,
    )
    error_deg = float(
        np.rad2deg(np.abs(first_command_after_latch.joints - fall_clip.joints[source_frame])).max()
    )
    assert error_deg < 5.0, (
        f"after the latch the commands must still track the clip (deviation "
        f"{error_deg:.1f} deg at frame {source_frame}) -- the frozen pre-fall pose "
        "took over and yanked the body into a plank"
    )
    assert state.consume_fall_replay_release() is True, "the release fires at replay end"
    assert state.consume_fall_replay_release() is False, "and exactly once"
    # After the release the command stream is the clip's final lying pose.
    out = state.apply(
        target,
        dt_s=DT_S,
        speed_m_s=0.0,
        idle_joints=np.zeros(len(plan.dof_names)),
        idle_height_m=0.9,
        idle_tilt=np.zeros(3),
        heading_rad=0.0,
    )
    error_deg = float(np.rad2deg(np.abs(out.joints - fall_clip.joints[-1])).max())
    assert error_deg < 0.5, f"the fallen command stream must be the lying pose, {error_deg:.1f} deg"


def test_fall_replay_records_mechanism_and_labels(plan, settings, fall_clip):
    config = replace(
        ActionConfig(**settings["actions"]), fall_replay_release_tilt_deg=None
    )
    state, target = _state(plan, settings, fall_clip, config=config)
    duration_s = float(fall_clip.provenance["duration_s"])
    for step in range(int((duration_s + 1.0) / DT_S)):
        target = state.apply(
            target,
            dt_s=DT_S,
            speed_m_s=0.0,
            idle_joints=np.zeros(len(plan.dof_names)),
            idle_height_m=0.9,
            idle_tilt=np.zeros(3),
            heading_rad=0.0,
        )
        state.observe(
            time_s=step * DT_S,
            root_height_m=float(target.position[2]),
            tilt_deg=5.0 if step * DT_S < duration_s * 0.5 else 60.0,
            standing_height_m=0.9,
            body_impact=step * DT_S >= duration_s * 0.5,
        )
    report = state.fall_report()
    assert report["mechanism"] == "anchored_fall_clip_replay"
    event = report["events"][-1]
    assert event["mechanism"] == "anchored_fall_clip_replay"
    assert "128_09" in json.dumps(event["clip"], ensure_ascii=False), (
        "the event must name the trip-fall source clip"
    )
    assert report["outcome"] == "fallen"
    assert event["impact_time_s"] is not None and event["fallen_time_s"] is not None


def test_fall_replay_releases_at_the_configured_tilt(plan, settings, fall_clip):
    """A tilt-truncated replay hands the last stretch to physics.

    The reversed clip is a deliberate lowering (vertical speed -0.3..-0.9 m/s
    throughout, a 0.5 m plateau measured on GPU run 2), and a root-height
    trigger cut it at the hands-and-knees phase into a stable kneel (GPU run
    4). The trigger must be the commanded trunk tilt: released at the tipping
    point the remaining topple is gravity, which is what makes the fall read
    as involuntary.
    """

    release_tilt = settings["actions"]["fall_replay_release_tilt_deg"]
    assert release_tilt is not None and 0 < release_tilt < 90, (
        "keyboard.yaml must ship a tilt trigger inside (0, 90)"
    )
    config = ActionConfig(**settings["actions"])
    from sim2sense_fall.humans.actions import load_posture  # noqa: PLC0415

    state = ActionState(
        config,
        {"crouch": load_posture(settings["crouch"], plan)},
        None,
        plan=plan,
        fall_clip=fall_clip,
    )
    target = TeleopTarget(
        joints=fall_clip.joints[0].copy(),
        joint_velocities=np.zeros(len(plan.dof_names)),
        position=np.array([*settings["spawn_xy"], float(fall_clip.heights_m[0])]),
        quaternion=axis_angle_to_quaternion(np.asarray(fall_clip.tilts[0], dtype=float)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        mode="stand",
    )
    assert state.request(
        "fall",
        time_s=0.0,
        heading_rad=0.0,
        measured_joints=target.joints,
        measured_position=target.position,
        measured_quaternion=target.quaternion,
    )
    release_height = None
    for _ in range(int(4.0 / DT_S)):
        target = state.apply(
            target,
            dt_s=DT_S,
            speed_m_s=0.0,
            idle_joints=np.zeros(len(plan.dof_names)),
            idle_height_m=0.9,
            idle_tilt=np.zeros(3),
            heading_rad=0.0,
        )
        if state.consume_fall_replay_release():
            assert state.consume_fall_replay_release() is False, "release fires exactly once"
            release_height = float(target.position[2])
            break
    assert release_height is not None, "the replay must release at the configured tilt"
    assert release_height < 0.80, (
        f"release happened at {release_height:.3f} m root height -- too high means the "
        "trigger fired in the upright kneel phase, not at the tipping point"
    )
    # The released command stream is the truncation pose, not the clip's lying end.
    error_deg = float(np.rad2deg(np.abs(target.joints - fall_clip.joints[-1])).max())
    assert error_deg > 10.0, (
        "a truncating release must NOT command the clip's lying end (that would be "
        "the deliberate-lowering read the truncation exists to remove)"
    )


def test_action_config_release_tilt_validation(settings):
    config = ActionConfig(**settings["actions"])
    assert config.fall_replay_release_tilt_deg is not None
    with pytest.raises(ValueError, match="fall_replay_release_tilt_deg"):
        replace(config, fall_replay_release_tilt_deg=0.0)
    with pytest.raises(ValueError, match="fall_replay_release_tilt_deg"):
        replace(config, fall_replay_release_tilt_deg=95.0)


def test_legacy_fall_keeps_its_release_mechanism_without_a_clip():
    """Clip-less states stay exactly on the legacy drive-release path."""

    state = ActionState(
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50),
        Posture(np.array([1.0, 0.5]), 0.5, np.zeros(3), {}),
    )
    assert not state.fall_replay_engaged
    assert state.consume_fall_replay_release() is False
    accepted = state.request(
        "fall",
        time_s=0.0,
        heading_rad=0.0,
        measured_joints=np.zeros(2),
    )
    assert accepted
    assert not state.fall_replay_engaged, "a clip-less state must stay on the legacy release"
    force = state.fall_force(0.1)
    assert float(np.linalg.norm(force)) > 0.0, "the legacy bias pulse must survive"
    report = state.fall_report()
    assert report["mechanism"] == (
        "root_assist_off_position_drives_scaled_backward_force_pulse"
    ), "the legacy mechanism string is a label contract with exported sessions"


def test_action_config_closure_knob_unchanged(settings):
    config = ActionConfig(**settings["actions"])
    assert replace(config, get_up_contact_closure=True).get_up_contact_closure is True
