"""Collision fitting and stance targets with procedural, license-free fixtures."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.contact_control import (
    StanceFootController,
    capsule_bottom,
    measured_support_feet,
    sideways_leg_dofs,
)
from sim2sense_fall.humans.rig import forward_kinematics, plan_human_rig
from sim2sense_fall.humans.teleop import Gait

ROOT = Path(__file__).resolve().parents[2]


def plan():
    return plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))


def lateral_demand(shift_m: float = 0.02, **overrides):
    """Bend a leg target sideways off its touchdown anchors and see what pays for it.

    The anchor is frozen in the world, so sliding the root sideways is exactly the error
    a stance constraint has to absorb. Which axes absorb it is the whole question: the
    axes that move a foot sideways are the cheapest, and they are the knees.
    """

    rig = plan()
    spec = {
        "height_m": 0.025,
        "iterations": 3,
        "max_correction_rad": 0.087,
        "damping": 0.0001,
        "orientation_weight": 0.12,
        "release_residual_m": 0.03,
        "abduction_weight": 1000.0,
    }
    spec.update(overrides)
    controller = StanceFootController(rig, **spec)
    q = np.zeros(len(rig.joints))
    # A slightly bent knee avoids the straight-leg singularity.
    for side in ("left", "right"):
        q[rig.dof_names.index(f"{side}_hip")] = -0.1
        q[rig.dof_names.index(f"{side}_knee")] = 0.2
        q[rig.dof_names.index(f"{side}_ankle")] = -0.1
    position = np.array([0.0, 0.0, rig.ground_offset_m - 0.005])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    controller.correct(q, position, quaternion, standing=True)
    position[1] += shift_m
    corrected = controller.correct(q, position, quaternion, standing=True)
    return rig, controller, q, corrected


def test_box_support_uses_all_rotated_extents():
    rig = plan()
    foot = replace(
        rig.link("left_ankle"), collision_box_bounds=((-0.1, -0.05, -0.02), (0.2, 0.05, 0.08))
    )
    rig = replace(rig, links=tuple(foot if link.name == foot.name else link for link in rig.links))
    poses = forward_kinematics(rig, {"left_ankle": 0.35}, root_position=(0, 0, 1))
    corners = np.array(
        [[x, y, z] for x in (-0.1, 0.2) for y in (-0.05, 0.05) for z in (-0.02, 0.08)]
    )
    expected = min(poses[foot.name].transform_point(point)[2] for point in corners)
    assert capsule_bottom(rig, poses, foot.name) == pytest.approx(expected)


def test_stance_ik_holds_contact_after_root_motion_and_releases_swing():
    rig = plan()
    controller = StanceFootController(
        rig,
        height_m=0.025,
        iterations=8,
        max_correction_rad=0.7,
        damping=0.0001,
        orientation_weight=0.12,
    )
    q = np.zeros(len(rig.joints))
    # A slightly bent knee avoids the straight-leg singularity.
    for side in ("left", "right"):
        q[rig.dof_names.index(side + "_hip")] = -0.1
        q[rig.dof_names.index(side + "_knee")] = 0.2
        q[rig.dof_names.index(side + "_ankle")] = -0.1
    pos = np.array([0.0, 0.0, rig.ground_offset_m - 0.005])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    controller.correct(q, pos, quat, standing=True)
    original = controller.anchors["left_ankle"].translation.copy()
    pos[0] += 0.015
    out = controller.correct(q, pos, quat, standing=True)
    poses = forward_kinematics(rig, dict(zip(rig.dof_names, out, strict=True)), root_position=pos)
    assert np.linalg.norm(poses["left_ankle"].translation - original) < 0.001
    controller.correct(q, pos, quat, standing=False, support_feet={"right_ankle"})
    assert "left_ankle" not in controller.anchors
    assert "right_ankle" in controller.anchors


def test_swing_fraction_wraps_at_cycle_boundary():
    gait = Gait(
        np.zeros((9, 1)),
        np.ones(9),
        1,
        1,
        {},
        support_mask=np.array(
            [
                [False, True],
                [False, True],
                [True, False],
                [True, False],
                [True, False],
                [True, True],
                [False, True],
                [False, True],
                [False, True],
            ]
        ),
    )
    assert gait.swing_fraction(0)["left_ankle"] == pytest.approx(2.5 / 4)
    assert gait.supporting_feet(0) == {"right_ankle"}
    assert 0 < gait.swing_fraction(0.5)["right_ankle"] < 1


@pytest.mark.parametrize("iterations", [0, True, 1.5])
def test_invalid_ik_iteration_count_fails(iterations):
    with pytest.raises(ValueError):
        StanceFootController(
            plan(),
            height_m=0.02,
            iterations=iterations,
            max_correction_rad=0.7,
            damping=0.001,
            orientation_weight=0.1,
        )


def test_sideways_probe_selects_efficient_lateral_axes_not_axis_letters():
    rig = plan()
    probe = set(sideways_leg_dofs(rig))
    assert probe == {"left_hip__dof1", "right_hip__dof1", "left_knee__dof1", "right_knee__dof1"}
    # the same four are today declared on x, but the probe is what decides, so reordering
    # a chain in the config cannot silently point the prior at the wrong rotation
    assert {rig.joints[rig.dof_names.index(name)].axis for name in probe} == {"x"}
    # the twist axes of the same joints do move the ankle a little sideways (~3 cm per
    # 30 deg against ~37 cm for abduction) and are deliberately left free
    twist = {
        name
        for name in rig.dof_names
        if name.endswith("__dof2") and any(part in name for part in ("hip", "knee"))
    }
    assert len(twist) == 4 and not twist & probe


def test_joint_space_prior_takes_the_penalised_axes_out_of_the_solution():
    """Measured where the claim is testable: a budget wide enough not to clip.

    At the shipped 5 deg cap almost every demand saturates, and a clipped solve is no
    longer the weighted solve, so this uses the old wide budget to isolate the prior.
    The live-trajectory effect of the shipped pair is in artifacts/humans/
    stance_ik_prior_sweep, produced by scripts/humans/audit_stance_ik.py.
    """

    wide = {"max_correction_rad": 0.7, "release_residual_m": None}
    rig, prior_solver, before, prior = lateral_demand(
        shift_m=0.12, abduction_weight=1000.0, **wide
    )
    _, unprior, _, no_prior = lateral_demand(shift_m=0.12, abduction_weight=1.0, **wide)

    sideways = [rig.dof_names.index(name) for name in prior_solver.sideways_dofs]

    def sideways_total(values):
        return float(np.rad2deg(np.abs(values - before)[sideways]).sum())

    # 4.96 deg against 18.22: the penalised set stops absorbing the lateral demand, and
    # the residual below grows to pay for it
    assert sideways_total(prior) < 0.4 * sideways_total(no_prior)
    # the demand does not vanish, it is refused or carried elsewhere: that is the trade
    # this knob buys, and it is why the release threshold has to exist alongside it
    assert prior_solver.residual_m > unprior.residual_m
    # positive control on the unweighted solve: hip abduction alone absorbs 8 deg of a
    # 12 cm lateral error, which is the class of spending that splayed the knee on the
    # shipped AMASS cycle
    hip = rig.dof_names.index("left_hip__dof1")
    assert np.rad2deg(abs(no_prior[hip] - before[hip])) > 5.0


def test_shipped_budget_bounds_every_leg_dof_and_releases_what_it_cannot_hold():
    rig, controller, before, corrected = lateral_demand(shift_m=0.20)
    leg = sorted({i for side in ("left", "right") for i in controller.indices[side]})
    assert np.rad2deg(np.abs(corrected[leg] - before[leg])).max() <= np.rad2deg(
        controller.max_correction_rad
    ) + 1e-9
    # 20 cm of lateral anchor error is far outside what 5 deg of leg can give back, so the
    # anchor is let go and the miss is recorded instead of being bought with joint angle
    assert controller.released_anchors >= 1
    assert controller.anchors == {}
    assert controller.residual_m > controller.release_residual_m
    # positive control: without the release the same anchor is kept and re-pulled every
    # step, which is where the old 35 deg of knee abduction came from
    _, kept, _, _ = lateral_demand(shift_m=0.20, release_residual_m=None)
    assert kept.released_anchors == 0
    assert set(kept.anchors) == {"left_ankle", "right_ankle"}
    assert kept.residual_m > 0.03


def test_release_counter_survives_reset_because_it_is_a_session_measure():
    _, controller, _, _ = lateral_demand(shift_m=0.20)
    released = controller.released_anchors
    assert released >= 1
    controller.reset()
    assert controller.anchors == {}
    assert controller.released_anchors == released


def test_plantedness_settings_are_validated_before_physics():
    rig = plan()
    for key, value in (("abduction_weight", 0.0), ("release_residual_m", float("nan"))):
        with pytest.raises(ValueError):
            StanceFootController(
                rig,
                height_m=0.025,
                iterations=3,
                max_correction_rad=0.087,
                damping=0.0001,
                orientation_weight=0.12,
                **{key: value},
            )


class _Sample:
    """Duck-typed contact sample: only the attributes the selector reads."""

    def __init__(self, impulse: float, collider0: str, collider1: str) -> None:
        self.impulse_magnitude_ns = impulse
        self.collider0_path = collider0
        self.collider1_path = collider1


FLOOR = "/World/rooms/living_room/floor"
WALL = "/World/rooms/living_room/walls/south_00"


def test_measured_support_feet_reads_the_simulator_not_the_gait_model():
    """Anchorability comes from reported floor contact, per foot, above the threshold.

    The gait model's travel-derived mask misses 46-56% of the right foot's real floor
    contacts on archived sessions (docs/mask-vs-contact-audit.md), so the anchor gate is
    read from contact instead. This pins the selector's contract: only feet, only the floor,
    only above threshold, and both sides normalised to the ankle segment the plan uses.
    """

    left = "/World/Human/left_ankle/collider"
    left_foot = "/World/Human/left_foot/collider"
    right = "/World/Human/right_ankle/collider"
    hand = "/World/Human/right_hand/collider"

    attributed = [
        (_Sample(1.20, left, FLOOR), "left_ankle"),
        # Same foot reported twice, as a multi-point contact does: still one entry.
        (_Sample(0.40, left, FLOOR), "left_ankle"),
        # The foot segment alias must land on the ankle the plan names.
        (_Sample(0.90, left_foot, FLOOR), "left_foot"),
        # A wall is not a floor, however hard the hand pushes.
        (_Sample(5.00, hand, WALL), "right_hand"),
        # On the floor but under the reporting threshold: not load-bearing.
        (_Sample(0.001, right, FLOOR), "right_ankle"),
    ]
    assert measured_support_feet(attributed, min_impulse_ns=0.01) == {"left_ankle"}


def test_measured_support_feet_is_empty_when_nothing_is_on_the_floor():
    """Positive control: the threshold and the floor test are load-bearing, not decoration.

    With the threshold above every impulse the selector must return nothing rather than the
    last foot it saw, and a wall-only report must not produce a stance either. If either
    guard were dropped from the selector, this fails while the test above still passes.
    """

    left = "/World/Human/left_ankle/collider"
    right = "/World/Human/right_ankle/collider"
    on_floor = [
        (_Sample(1.20, left, FLOOR), "left_ankle"),
        (_Sample(0.90, right, FLOOR), "right_ankle"),
    ]
    assert measured_support_feet(on_floor, min_impulse_ns=0.01) == {"left_ankle", "right_ankle"}
    assert measured_support_feet(on_floor, min_impulse_ns=10.0) == set()
    assert measured_support_feet(
        [(_Sample(9.0, left, WALL), "left_ankle")], min_impulse_ns=0.01
    ) == set()
    assert measured_support_feet([], min_impulse_ns=0.01) == set()

    with pytest.raises(ValueError, match="min_impulse_ns"):
        measured_support_feet(on_floor, min_impulse_ns=float("nan"))
