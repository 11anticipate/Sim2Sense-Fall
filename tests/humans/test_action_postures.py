"""Real-source posture targets and the get-up clip, validated against the rig.

The shipped keyboard.yaml names the source files and frames; these tests load
the same assets the simulator will and check the contact projection actually
grounds the declared contacts and that nothing leaves the rig limits. The plan
fixture fits the shipped collision geometry (convex-hull feet, fitted capsules)
because the projection anchors on it: the bare rig's per-link skin vertices
extend far below the joints and would mis-ground the pose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.actions import load_action_clip, load_posture
from sim2sense_fall.humans.contact_control import capsule_bottom, fit_collision_capsules
from sim2sense_fall.humans.rig import fit_rest_skeleton, forward_kinematics, plan_human_rig
from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix
from sim2sense_fall.humans.teleop import TeleopTarget, load_keyboard_config

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYBOARD_YAML = REPO_ROOT / "configs/humans/keyboard.yaml"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))


def _bottoms(plan, posture, links):
    poses = forward_kinematics(
        plan,
        dict(zip(plan.dof_names, posture.joints, strict=True)),
        root_position=(0.0, 0.0, posture.height_m),
        root_rotation=posture.tilt,
    )
    return {name: capsule_bottom(plan, poses, name) for name in links}


@pytest.fixture(scope="module")
def plan():
    from common import DEFAULT_MOTIONS, load_inputs  # noqa: PLC0415

    from sim2sense_fall.humans.assets import select_body  # noqa: PLC0415

    settings = load_keyboard_config(KEYBOARD_YAML, REPO_ROOT)
    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    plan, _ = fit_collision_capsules(
        plan, mesh, **{k: v for k, v in settings["collision_fit"].items() if k != "enabled"}
    )
    return plan


@pytest.fixture(scope="module")
def settings():
    return load_keyboard_config(KEYBOARD_YAML, REPO_ROOT)


def test_keyboard_yaml_declares_the_surveyed_actions(settings):
    assert settings["bend"]["file"].name == "115_01_poses.npz"
    assert settings["sit"]["contacts"] == [
        "left_ankle", "right_ankle", "pelvis", "left_hand", "right_hand",
    ]
    assert settings["get_up"]["file"].name == "140_01_poses.npz"


def test_crouch_posture_projects_both_feet_flat(plan, settings):
    """2026-09-26 换源: 对称深蹲 CMU/01_06 f744，双踝声明接触投影落地。"""

    posture = load_posture(settings["crouch"], plan)
    assert posture.provenance["height_method"] == "declared_contact_projection"
    assert posture.provenance["projection"]["max_residual_m"] == pytest.approx(0.0, abs=1e-4)
    bottoms = _bottoms(plan, posture, ("left_ankle", "right_ankle"))
    for name, value in bottoms.items():
        assert value == pytest.approx(0.0, abs=1e-4), name
    # 深蹲带: 骨盆约为站高的 40-48%（随胶囊拟合略有差异）
    assert 0.35 <= posture.height_m <= 0.45


def test_bend_posture_grounds_both_feet_and_flexes_the_trunk(plan, settings):
    posture = load_posture(settings["bend"], plan)
    assert posture.provenance["projection"]["method"] == "bounded_ik_contact_projection"
    bottoms = _bottoms(plan, posture, ("left_ankle", "right_ankle"))
    for name, value in bottoms.items():
        assert value == pytest.approx(0.0, abs=0.003), name
    poses = forward_kinematics(
        plan,
        dict(zip(plan.dof_names, posture.joints, strict=True)),
        root_position=(0.0, 0.0, posture.height_m),
        root_rotation=posture.tilt,
    )
    axis = poses["neck"].translation - poses["pelvis"].translation
    trunk = float(np.degrees(np.arccos(np.clip(axis[2] / np.linalg.norm(axis), -1, 1))))
    assert trunk > 60.0, "the bend action must exceed the forward-fall trunk threshold"


def test_sit_posture_grounds_every_declared_contact(plan, settings):
    posture = load_posture(settings["sit"], plan)
    contacts = settings["sit"]["contacts"]
    projection = posture.provenance["projection"]
    # The retargeted seated pose has an intrinsic ~48 mm contact-plane spread
    # (measured across the clip's seated frames; joint limits bind before the
    # limbs close the last ~40 mm). Pelvis is the anchor, so the primary
    # support rests exactly on the floor and the limbs hover by at most the
    # spread. Sit-hold acceptance must therefore treat pelvis contact as
    # support; that gate is to be pre-registered before the Isaac run.
    assert projection["max_residual_m"] <= 0.05, projection
    bottoms = _bottoms(plan, posture, contacts)
    for name, value in bottoms.items():
        assert abs(value) <= 0.05, f"{name} floats/penetrates by {value * 1000:.1f} mm"
    assert 0.05 < posture.height_m < 0.45, "a floor-sit root height is well below standing"
    assert posture.provenance["height_method"] == "declared_contact_projection"


def test_get_up_clip_is_grounded_lie_to_stand(plan, settings, request):
    dt_s = 1.0 / 120.0
    clip = load_action_clip(settings["get_up"], plan, dt_s=dt_s)
    assert clip.frame_dt_s == pytest.approx(dt_s)
    assert 5.0 < clip.duration_s < 10.0
    # First frame lies on the floor (low root), final frame stands (near spawn height).
    assert clip.heights_m[0] < 0.45
    assert clip.heights_m[-1] > 0.75
    travel = float(np.linalg.norm(clip.offsets_xy[-1] - clip.offsets_xy[0]))
    assert 0.3 < travel < 1.5
    assert (
        clip.provenance["derivation"]
        == "full_clip_retargeted_grounded_resampled_assisted_playback"
    )


def test_posture_transition_height_is_contact_closure(plan, settings):
    """The commanded root height must be derived from the blended configuration.

    The crouch-hold failure was feet floating for 1.7 s: joints and height were
    blended independently, so gravity-sagged legs lifted the feet off a root
    the assist held at a kinematically unrelated height. With the rig available
    the height IS the contact closure of the commanded pose -- the lowest
    active contact rests exactly on the floor at every frame, the hold height
    reproduces the projected posture height, and mid-hold retargets stay
    continuous because the closure is a pure function of the commanded pose.
    """

    from sim2sense_fall.humans.actions import ActionConfig, ActionState

    state = ActionState(
        ActionConfig(1, 0.025, 350, 0.25, 0, 0.6, 50),
        {"crouch": load_posture(settings["crouch"], plan),
         "sit": load_posture(settings["sit"], plan)},
        plan=plan,
    )
    joints0 = np.zeros(len(plan.joints))
    target = TeleopTarget(
        joints0.copy(), np.zeros_like(joints0), np.array([0.0, 0.0, plan.ground_offset_m]),
        np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), "stand",
    )

    def bottoms_at(applied, links):
        tilt = matrix_to_axis_angle(quaternion_to_matrix(applied.quaternion))
        poses = forward_kinematics(
            plan, dict(zip(plan.dof_names, applied.joints, strict=True)),
            root_position=(0.0, 0.0, applied.position[2]), root_rotation=tilt,
        )
        return {name: capsule_bottom(plan, poses, name) for name in links}

    def step():
        return state.apply(
            target, dt_s=0.1, speed_m_s=0.0, idle_joints=joints0, idle_height_m=1.0,
            idle_tilt=np.zeros(3), heading_rad=0,
        )

    state.request("crouch", time_s=0, heading_rad=0, measured_joints=joints0)
    last = None
    for _ in range(15):
        last = step()
    assert state.mode == "crouch"
    crouch = load_posture(settings["crouch"], plan)
    # Hold height reproduces the projected posture height (feet grounded).
    assert last.position[2] == pytest.approx(crouch.height_m, abs=1e-9)
    # The commanded geometry rests its lowest active contact exactly on the
    # floor -- that is what closure means.
    assert min(bottoms_at(last, state.active_contacts).values()) == pytest.approx(0.0, abs=1e-9)

    hold = (last.joints.copy(), float(last.position[2]))
    state.request("stand", time_s=2, heading_rad=0, measured_joints=last.joints)
    # Continuity is a pure-function property: at a vanishing blend step the
    # closure of the (unchanged) source pose must equal the held height
    # exactly. The deep-squat source moves ~2 cm within the first 10% of a
    # 1 s return blend, so a coarse first step is transition motion, not a
    # discontinuity.
    tiny = state.apply(
        target, dt_s=1e-4, speed_m_s=0.0, idle_joints=joints0, idle_height_m=1.0,
        idle_tilt=np.zeros(3), heading_rad=0,
    )
    assert tiny.position[2] == pytest.approx(hold[1], abs=0.005)
    first = step()
    assert first.position[2] == pytest.approx(hold[1], abs=0.04)
    assert np.linalg.norm(first.joints - hold[0]) < 0.2

    # Sit transition: the pelvis joins the contact set, so the descent ends
    # with the sit posture's own projected height, pelvis as the lowest support.
    state.request("sit", time_s=3, heading_rad=0, measured_joints=last.joints)
    sat = None
    for _ in range(25):
        sat = step()
    assert state.mode == "sit"
    sit = load_posture(settings["sit"], plan)
    assert sat.position[2] == pytest.approx(sit.height_m, abs=1e-9)
    assert min(bottoms_at(sat, state.active_contacts).values()) == pytest.approx(0.0, abs=1e-9)
    assert "pelvis" in state.active_contacts
