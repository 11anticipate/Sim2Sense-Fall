"""Regression checks for physics replay boundaries, not just successful imports."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.amass import crop_amass_clip
from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.motion import load_motion_library
from sim2sense_fall.humans.rig import plan_human_rig
from sim2sense_fall.humans.root_control import RootAssistConfig, root_wrench
from sim2sense_fall.humans.usd_human import ContactSample, HumanRuntime

ROOT = Path(__file__).resolve().parents[2]


def test_head_and_hands_have_real_terminal_collision_shapes():
    config = load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml")
    plan = plan_human_rig(config)
    for name in ("head", "left_hand", "right_hand"):
        link = next(link for link in plan.links if link.name == name)
        assert link.capsule is not None
        assert link.capsule.cylinder_length_m == 0
        assert link.capsule.center == config.rig.segments[name].terminal_center_m
    with pytest.raises(ValueError, match="three coordinates"):
        replace(config.rig.segments["head"], terminal_center_m=(0.0,))


def test_contacts_use_paths_and_keep_nonhuman_contacts_out():
    # The contact point can be outside the post-solver capsule. Identity remains
    # exact even in that case; unrelated furniture-floor contacts must be ignored.
    body = "/World/Human/head/collider"
    floor = "/World/rooms/living_room/floor"
    sample = ContactSample(1, 2, (20.0, 20.0, 0.0), (0.0, 0.0, -1.0),
                           (0.0, 0.0, -2.0), 0.03, floor, body)
    unrelated = replace(sample, collider1_path="/World/chair")
    runtime = object.__new__(HumanRuntime)
    runtime.root_path = "/World/Human"
    runtime._link_collider_paths = {"head": body}
    runtime.capabilities = {}
    runtime.contact_samples = lambda: (unrelated, sample)
    result = runtime.attributed_contact_samples()
    assert len(result) == 1
    contact, segment = result[0]
    assert segment == "head"
    assert contact.collider0_path == body
    assert contact.collider1_path == floor
    assert contact.normal == (0.0, 0.0, 1.0)
    assert contact.impulse_ns == (0.0, 0.0, 2.0)
    assert runtime.capabilities["contact_attribution"] == "usd_collider_path"


def test_root_assistance_is_bounded_and_damps_velocity():
    config = RootAssistConfig.load(ROOT / "configs/humans/root_assist.yaml")
    state = dict(position=np.zeros(3), quaternion=np.array([1.0, 0, 0, 0]),
                 linear_velocity=np.zeros(3), angular_velocity=np.zeros(3),
                 target_position=np.array([100.0, 0, 0]),
                 target_quaternion=np.array([0.0, 1.0, 0, 0]),
                 target_linear_velocity=np.zeros(3), target_angular_velocity=np.zeros(3),
                 mass_kg=72.0, gravity_m_s2=9.81)
    force, torque = root_wrench(config, **state)
    assert np.linalg.norm(force) == pytest.approx(config.max_force_n)
    assert np.linalg.norm(torque) == pytest.approx(config.max_torque_nm)
    state.update(target_position=np.zeros(3), target_quaternion=state["quaternion"],
                 linear_velocity=np.array([1.0, 0, 0]))
    force, torque = root_wrench(config, **state)
    assert force[0] < 0
    assert force[2] > 0
    assert np.allclose(torque, 0)
    with pytest.raises(ValueError, match="finite"):
        root_wrench(config, **{**state, "position": np.array([np.nan, 0, 0])})


def test_pd_velocity_feedforward_is_reordered_and_reset_for_a_hold():
    class Articulation:
        dof_names = ("b", "a")

        def set_dof_position_targets(self, value):
            self.position = value

        def set_dof_velocity_targets(self, value):
            self.velocity = value

    runtime = object.__new__(HumanRuntime)
    runtime.plan = type("Plan", (), {"dof_names": ("a", "b")})()
    runtime._plan_to_runtime = np.array([1, 0])
    runtime.articulation = Articulation()
    runtime.set_joint_targets([0.2, -0.3], [1.0, -2.0])
    np.testing.assert_allclose(runtime.articulation.velocity, [[-2.0, 1.0]])
    runtime.set_joint_targets([0.2, -0.3])
    np.testing.assert_allclose(runtime.articulation.velocity, [[0.0, 0.0]])


def test_motion_interval_preserves_source_frame_provenance():
    clip = load_motion_library(ROOT / "configs/humans/motions.yaml")["stand_neutral"]
    cropped = crop_amass_clip(clip, start_s=0.2, duration_s=0.5)
    first = round(0.2 * clip.fps)
    np.testing.assert_allclose(cropped.root_translation[0], clip.root_translation[first])
    assert cropped.metadata["source_interval"]["first_frame"] == first
    with pytest.raises(ValueError, match="at least two"):
        crop_amass_clip(clip, start_s=1000)
