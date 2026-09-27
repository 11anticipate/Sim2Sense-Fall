"""The G get-up must command a grounded body, not a body carried on the pelvis wire.

Measured 2026-09-27 (`scripts/humans/diagnose_getup_float.py`, evidence in
`artifacts/humans/getup_float/`): the shipped `CMU/140/140_01` clip is grounded once
on its first frame, and from then on its absolute root-height curve holds the lowest
collision body **87.6 mm (p50) / 142.6 mm (max)** above the floor, still 118 mm high
on the final standing frame. With nothing under the body, PhysX can only follow the
curve by lifting: the recorded session shows the assist pinned at 1.00 body weight
for 637 of 888 frames and a median floor normal force of 0 N.

These tests pin the contact-closure fix (`ActionConfig.get_up_contact_closure`) and
keep the diagnosis itself under test, so a future source clip that really is grounded
cannot silently re-introduce the wire.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.actions import (
    ActionConfig,
    ActionState,
    load_action_clip,
    recovery_command_anchor,
)
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import (
    capsule_bottom,
    fit_collision_capsules,
    has_collision_geometry,
)
from sim2sense_fall.humans.rig import (
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)
from sim2sense_fall.humans.rotations import (
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_to_matrix,
)
from sim2sense_fall.humans.teleop import TeleopTarget, load_keyboard_config

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYBOARD_YAML = REPO_ROOT / "configs/humans/keyboard.yaml"
DT_S = 1.0 / 120.0
# Measured worst frame-to-frame step over the shipped clip: 7.7 mm with closure,
# 9.5 mm without. Anything above this bound means the recovery height stopped being
# a smooth curve and started being a teleport.
MAX_HEIGHT_STEP_M = 0.01
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


def _fallen_state(plan, settings, *, closure: bool) -> tuple[ActionState, TeleopTarget]:
    """An ActionState latched in `fallen`, measured as the clip's own lying first frame.

    Using the clip's first frame as the "measured" fallen body keeps the 0.5 s
    blend-in physical (lying pose to lying pose); an all-zero stand-in would make
    the blend transient a fixture artefact rather than a property of the replay.
    """

    config = ActionConfig(**settings["actions"])
    if not closure:
        config = replace(config, get_up_contact_closure=False)
    clip = load_action_clip(settings["get_up"], plan, dt_s=DT_S)
    from sim2sense_fall.humans.actions import load_posture

    state = ActionState(
        config, {"crouch": load_posture(settings["crouch"], plan)}, clip, plan=plan
    )
    lying_height_m = float(clip.heights_m[0])
    lying_tilt_deg = float(np.linalg.norm(clip.tilts[0]) * 180.0 / np.pi)
    target = TeleopTarget(
        joints=clip.joints[0].copy(),
        joint_velocities=np.zeros(len(plan.dof_names)),
        position=np.array([*settings["spawn_xy"], lying_height_m]),
        quaternion=axis_angle_to_quaternion(np.asarray(clip.tilts[0], dtype=float)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        mode="stand",
    )
    state.request("fall", time_s=0.0, heading_rad=0.0, measured_joints=target.joints)
    state.observe(
        time_s=0.5, root_height_m=lying_height_m, tilt_deg=lying_tilt_deg,
        standing_height_m=0.9, body_impact=True,
    )
    assert state.phase == "fallen", "the fixture must reach the latched fallen state"
    accepted = state.request(
        "get_up",
        time_s=1.0,
        heading_rad=0.0,
        measured_joints=target.joints,
        measured_position=target.position,
        measured_quaternion=target.quaternion,
    )
    assert accepted
    return state, target


def _lowest_bottom_m(plan, joints: np.ndarray, quaternion: np.ndarray, height_m: float) -> float:
    """Height of the lowest collision body for one commanded pose, floor at 0."""

    poses = forward_kinematics(
        plan,
        dict(zip(plan.dof_names, joints, strict=True)),
        root_position=(0.0, 0.0, height_m),
        root_rotation=matrix_to_axis_angle(quaternion_to_matrix(quaternion)),
    )
    return min(
        capsule_bottom(plan, poses, link.name)
        for link in plan.links
        if has_collision_geometry(link)
    )


def _replay(plan, settings, *, closure: bool) -> list[TeleopTarget]:
    state, target = _fallen_state(plan, settings, closure=closure)
    frames = []
    for _ in range(int(12.0 / DT_S)):
        out = state.apply(
            target,
            dt_s=DT_S,
            speed_m_s=0.0,
            idle_joints=np.zeros(len(plan.dof_names)),
            idle_height_m=0.9,
            idle_tilt=np.zeros(3),
            heading_rad=0.0,
        )
        frames.append(out)
        target = out
        if state.phase == "idle" and out.mode != "getting_up":
            break
    assert any(frame.mode == "getting_up" for frame in frames), "replay must have run"
    return frames


def test_clip_curve_alone_floats_and_the_diagnosis_holds(plan, settings):
    """Without closure the commanded body is never on the floor (the reported defect)."""

    frames = _replay(plan, settings, closure=False)
    gaps = np.array([
        _lowest_bottom_m(plan, frame.joints, frame.quaternion, frame.position[2])
        for frame in frames
        if frame.mode == "getting_up"
    ])
    assert np.median(gaps) * 1000 > 50.0, (
        f"expected the un-closed clip curve to float (measured 87.6 mm median); got "
        f"{np.median(gaps) * 1000:.1f} mm -- a new source clip may already be grounded,"
        " re-run scripts/humans/diagnose_getup_float.py before trusting this gate"
    )
    final = [frame for frame in frames if frame.mode == "getting_up"][-1]
    assert final.position[2] > 0.9, "the raw clip ends 118 mm above the grounded height"


def test_closure_grounds_every_getting_up_frame(plan, settings):
    frames = _replay(plan, settings, closure=True)
    getting_up = [frame for frame in frames if frame.mode == "getting_up"]
    # Judged from the first frame: the fixture's fallen body IS the clip's own
    # grounded lying pose, so the 0.5 s blend cannot hide a floating command.
    gaps = np.array([
        _lowest_bottom_m(plan, frame.joints, frame.quaternion, frame.position[2])
        for frame in getting_up
    ])
    worst_mm = float(np.abs(gaps).max() * 1000)
    assert worst_mm <= 2.0, (
        f"closure must keep the commanded pose exactly on the floor, worst {worst_mm:.2f} mm"
    )
    heights = np.array([frame.position[2] for frame in getting_up])
    worst_step = float(np.abs(np.diff(heights)).max())
    assert worst_step <= MAX_HEIGHT_STEP_M, (
        f"the grounded height must stay a smooth curve, worst step {worst_step * 1e3:.1f} mm"
    )
    assert float(np.abs(np.diff(heights)).max()) <= 2 * MAX_HEIGHT_STEP_M, (
        "no frame of the recovery may teleport"
    )
    assert heights[-1] < 0.88, "the recovery must not command a standing pelvis above reach"


def test_closure_is_a_declared_knob(plan, settings):
    config = ActionConfig(**settings["actions"])
    assert config.get_up_contact_closure is True, "keyboard.yaml ships the closure fix"
    with pytest.raises(ValueError, match="get_up_contact_closure"):
        replace(config, get_up_contact_closure="yes")


def test_getup_first_command_is_the_measured_body_pose(plan, settings):
    """The recovery must start from where the body actually is, not from a snapshot.

    This is the invariant `recovery_command_anchor` relies on: the action layer's
    first commanded frame is the measured fallen pose, so re-anchoring the slew
    limiter to the measured joints costs no extra step.
    """

    state, target = _fallen_state(plan, settings, closure=True)
    out = state.apply(
        target,
        dt_s=DT_S,
        speed_m_s=0.0,
        idle_joints=np.zeros(len(plan.dof_names)),
        idle_height_m=0.9,
        idle_tilt=np.zeros(3),
        heading_rad=0.0,
    )
    assert out.mode == "getting_up"
    error_deg = float(np.rad2deg(np.abs(out.joints - target.joints)).max())
    assert error_deg < 0.5, f"the first recovery command moved the pose by {error_deg:.1f} deg"


def test_recovery_anchor_reanchors_only_at_the_getup_handover():
    measured = np.array([2.37, -0.05, 0.4])
    assert recovery_command_anchor("fallen", measured) is None
    assert recovery_command_anchor("stand", measured) is None
    anchored = recovery_command_anchor("getting_up", measured)
    assert anchored is not None
    assert np.array_equal(anchored, measured)
    anchored[0] = 0.0
    assert measured[0] == 2.37, "the anchor must be a copy, never an alias of the readback"
    with pytest.raises(ValueError, match="finite measured joints"):
        recovery_command_anchor("getting_up", np.array([np.nan, 0.0, 0.0]))
