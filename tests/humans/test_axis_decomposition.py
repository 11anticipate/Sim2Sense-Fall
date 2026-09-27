"""CPU tests for clip-to-DOF axis decomposition and the multi-axis mapper.

A revolute joint has one degree of freedom, so reproducing a recorded 3-DOF joint
rotation needs a *chain* of revolute joints. The shipped rig is single-axis and
could only ever replay axis-aligned references: every AMASS clip failed the
screen, and the failure was reported per joint rather than per DOF, so the reason
("this motion was never the problem; the rig cannot express it") was invisible.

These tests pin the decomposition itself and the mapper contract around it:

* the decomposition is exact for every reachable target, for every axis order;
* a three-axis chain spans ``SO(3)``, so an arbitrary recorded rotation becomes
  expressible, which is what unblocks the AMASS work;
* a two-axis chain reports the geodesic angle it genuinely cannot reach, and
  adding a DOF never makes the miss worse;
* a single-axis rig keeps its historical numbers exactly, so no existing
  configuration or reference motion changes meaning;
* the residual metric itself resolves machine precision.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.amass import load_amass_clip
from sim2sense_fall.humans.config import DriveConfig, JointConfig, load_human_config
from sim2sense_fall.humans.rig import (
    axis_residuals,
    dof_groups,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
)
from sim2sense_fall.humans.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    reconstruct_from_angles,
    rotation_split_residual_rad,
    split_rotation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "humans" / "human_smpl_neutral.yaml"
MULTIAXIS_CONFIG_PATH = REPO_ROOT / "configs" / "humans" / "human_smpl_multiaxis.yaml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _geodesic_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Independent geodesic metric, so the tests do not grade the code with itself."""

    delta = first @ second.T
    skew = 0.5 * math.sqrt(
        (delta[2, 1] - delta[1, 2]) ** 2
        + (delta[0, 2] - delta[2, 0]) ** 2
        + (delta[1, 0] - delta[0, 1]) ** 2
    )
    return math.degrees(math.atan2(skew, (float(np.trace(delta)) - 1.0) / 2.0))


def _random_rotation(generator: np.random.Generator) -> np.ndarray:
    axis = generator.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis_angle_to_matrix(axis * float(generator.uniform(-math.pi, math.pi)))


def _drive() -> DriveConfig:
    return DriveConfig(stiffness=6000.0, damping=150.0, max_force=300.0)


def _with_axes(config, joint: str, axes: tuple[str, ...], limits_deg: tuple | None = None):
    """Re-plan ``config`` with one joint turned into a multi-axis chain."""

    existing = config.rig.joints[joint]
    limits = limits_deg if limits_deg is not None else tuple(
        existing.limits_deg[0] for _ in axes
    )
    upgraded = JointConfig(
        joint=joint,
        dof=existing.dof,
        rotations=axes,
        limits_deg=limits,
        drive=existing.drive,
    )
    return replace(config, rig=replace(config.rig, joints={**config.rig.joints, joint: upgraded}))


def _clip_with_rotations(tmp_path: Path, joint_vectors: dict[int, tuple[float, float, float]]):
    """An AMASS clip whose pose block carries the given rotations, in the source frame.

    Rotations are written on the raw SMPL joint indices, so the test never has to
    reason about the body-frame conversion: assertions read the pipeline-frame
    rotation straight back out of the loaded clip.
    """

    frames = 3
    poses = np.zeros((frames, 156), dtype=np.float64)
    trans = np.zeros((frames, 3), dtype=np.float64)
    for index, vector in joint_vectors.items():
        poses[:, index * 3 : index * 3 + 3] = vector
    source = tmp_path / "Subject1" / "probe.npz"
    source.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        source,
        poses=poses,
        trans=trans,
        mocap_framerate=np.asarray(30.0),
        gender=np.asarray("neutral"),
        betas=np.zeros(16, dtype=np.float64),
    )
    return load_amass_clip(source, root=tmp_path)


def _reconstruct_chain(plan, values: np.ndarray, chain_joint: str) -> np.ndarray:
    """Rebuild one chain's rotation from the flat DOF vector the mapper returned."""

    joints = dof_groups(plan)[chain_joint]
    angles = [float(values[plan.dof_names.index(joint.name)]) for joint in joints]
    return reconstruct_from_angles(tuple(joint.axis for joint in joints), angles)


# ---------------------------------------------------------------------------
# the decomposition itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 3])
def test_split_rotation_round_trips_every_axis_order(length: int) -> None:
    """A reachable target must come back exactly, whatever order the axes are in."""

    generator = np.random.default_rng(20260924)
    for axes in itertools.permutations("xyz", length):
        for _ in range(60):
            angles = tuple(float(value) for value in generator.uniform(-3.0, 3.0, length))
            target = reconstruct_from_angles(axes, angles)
            split = split_rotation(target, axes, axis_angle=matrix_to_axis_angle(target))
            rebuilt = reconstruct_from_angles(axes, split.angles)
            assert _geodesic_deg(target, rebuilt) < 1e-6, axes
            assert split.residual_deg < 1e-6, axes


def test_three_axis_split_spans_so3() -> None:
    """Every rotation is reachable by a three-axis chain, in every axis order."""

    generator = np.random.default_rng(7)
    for axes in itertools.permutations("xyz", 3):
        for _ in range(40):
            target = _random_rotation(generator)
            split = split_rotation(target, axes)
            assert split.residual_deg < 1e-6, (axes, split.residual_deg)
            rebuilt = reconstruct_from_angles(axes, split.angles)
            assert _geodesic_deg(target, rebuilt) < 1e-6, axes


def test_adding_an_axis_never_increases_the_residual() -> None:
    """A chain with more freedom cannot fit worse -- one, two, then three axes."""

    generator = np.random.default_rng(11)
    for _ in range(40):
        target = _random_rotation(generator)
        residuals = [
            split_rotation(target, axes).residual_rad
            for axes in (("x",), ("x", "y"), ("x", "y", "z"))
        ]
        assert residuals[0] >= residuals[1] - 1e-12
        assert residuals[1] >= residuals[2] - 1e-12


def test_residual_metric_resolves_machine_precision() -> None:
    """The residual of an exact reconstruction must be zero, not acos noise.

    An ``acos((trace - 1) / 2)`` metric has a noise floor near 1.4e-8 rad, which
    is larger than any sensible tolerance -- it reported exact decompositions as
    measurable misses and would have made the screen reject them.
    """

    generator = np.random.default_rng(3)
    for _ in range(50):
        target = _random_rotation(generator)
        axes = ("y", "x", "z")
        angles = split_rotation(target, axes).angles
        assert rotation_split_residual_rad(target, axes, angles) < 1e-12


def test_gimbal_lock_is_finite_and_exact() -> None:
    """The singular middle angle must not produce NaN or a wrong branch."""

    singular = reconstruct_from_angles(("x", "y", "z"), (0.4, math.pi / 2.0, -0.7))
    split = split_rotation(singular, ("x", "y", "z"))
    assert all(math.isfinite(value) for value in split.angles)
    assert split.residual_deg < 1e-6
    also = reconstruct_from_angles(("x", "y", "z"), (0.4, -math.pi / 2.0, -0.7))
    assert split_rotation(also, ("x", "y", "z")).residual_deg < 1e-6


def test_split_rotation_rejects_inputs_it_cannot_define() -> None:
    rotation = axis_angle_to_matrix(np.array([0.2, 0.3, 0.4]))
    with pytest.raises(ValueError, match="at least one axis"):
        split_rotation(rotation, ())
    with pytest.raises(ValueError, match="distinct axes"):
        split_rotation(rotation, ("y", "y"))
    # A chain longer than three axes cannot be all-distinct, so it is the
    # duplicate rule that refuses it; the axis-count guard behind it is
    # unreachable by construction and exists only to fail loudly if that changes.
    with pytest.raises(ValueError, match="distinct axes"):
        split_rotation(rotation, ("y", "x", "z", "x"))
    with pytest.raises(ValueError, match="must be one of"):
        split_rotation(rotation, ("w",))
    with pytest.raises(ValueError, match="rotation matrix"):
        split_rotation(np.full((3, 3), 2.0), ("y",))
    with pytest.raises(ValueError, match=r"\(3, 3\) matrix or a \(3,\)"):
        split_rotation(np.zeros(4), ("y",))


# ---------------------------------------------------------------------------
# the single-axis contract that must not change
# ---------------------------------------------------------------------------


def test_single_axis_keeps_the_historical_projection(tmp_path: Path) -> None:
    """One axis still means "the component about it", with the off-axis part reported.

    This is the behaviour every shipped configuration and authored reference was
    written against, so it has to survive byte for byte rather than merely
    approximately.
    """

    config = load_human_config(CONFIG_PATH)
    plan = plan_human_rig(config)
    knee = plan.joint("left_knee")
    assert knee.axis == "y"

    aligned = _clip_with_rotations(tmp_path, {4: (0.6, 0.0, 0.0)})
    vector = np.asarray(aligned.rotation_of(0, "left_knee"), dtype=np.float64)
    assert abs(vector[1]) > 0.5, "the source X component should land on the pipeline Y axis"
    assert np.linalg.norm(np.delete(vector, 1)) < 1e-12

    residuals = axis_residuals(aligned, 0, plan)
    value, residual = residuals["left_knee"]
    assert value == pytest.approx(float(vector[1]))
    assert residual == pytest.approx(0.0, abs=1e-12)

    values, worst = joint_values_from_clip(aligned, 0, plan)
    assert values[plan.dof_names.index("left_knee")] == pytest.approx(value)
    assert worst == pytest.approx(0.0, abs=1e-12)

    # A rotation the single axis cannot express keeps the off-axis norm as the
    # residual and is refused rather than silently projected.
    twisted_source = {4: (0.6, 0.0, 0.5)}
    twisted = _clip_with_rotations(tmp_path / "twist", twisted_source)
    twisted_vector = np.asarray(twisted.rotation_of(0, "left_knee"), dtype=np.float64)
    _, twisted_residual = axis_residuals(twisted, 0, plan)["left_knee"]
    assert twisted_residual == pytest.approx(
        float(np.linalg.norm(np.delete(twisted_vector, 1))), rel=1e-9
    )
    with pytest.raises(ValueError, match="axis the rig cannot express"):
        joint_values_from_clip(twisted, 0, plan)


def test_axis_residuals_covers_every_dof_in_plan_order(tmp_path: Path) -> None:
    """One entry per DOF, in plan order -- the contract the runtime relies on."""

    aligned = _clip_with_rotations(tmp_path, {4: (0.6, 0.0, 0.0)})
    for rotations in (("y",), ("y", "x"), ("y", "x", "z")):
        plan = plan_human_rig(_with_axes(load_human_config(CONFIG_PATH), "left_knee", rotations))
        residuals = axis_residuals(aligned, 0, plan)
        assert tuple(residuals) == plan.dof_names
        values, _ = joint_values_from_clip(aligned, 0, plan)
        assert values.shape == (len(plan.dof_names),)


# ---------------------------------------------------------------------------
# the multi-axis mapper: what actually unblocks the AMASS work
# ---------------------------------------------------------------------------


def test_dof_groups_are_in_chain_order(tmp_path: Path) -> None:
    """Proxies come first, then the real link: the order the chain applies them."""

    plan = plan_human_rig(
        _with_axes(load_human_config(CONFIG_PATH), "left_knee", ("y", "x", "z"))
    )
    groups = dof_groups(plan)
    assert tuple(joint.name for joint in groups["left_knee"]) == (
        "left_knee__dof1",
        "left_knee__dof2",
        "left_knee",
    )
    assert tuple(joint.axis for joint in groups["left_knee"]) == ("y", "x", "z")
    # Single-axis joints are one-element groups, so both paths share one code path.
    assert len(groups["right_knee"]) == 1


def test_three_axis_joint_expresses_an_arbitrary_recorded_rotation(tmp_path: Path) -> None:
    """The headline fix: a full 3-DOF recording becomes replayable.

    Under the single-axis rig this exact motion is refused. With a three-axis
    knee it is reproduced to machine precision, and the reconstruction from the
    returned angles matches the clip -- not just the residual.
    """

    config = load_human_config(CONFIG_PATH)
    clip = _clip_with_rotations(tmp_path, {4: (1.1, -0.7, 0.9)})
    single = plan_human_rig(config)
    with pytest.raises(ValueError, match="axis the rig cannot express"):
        joint_values_from_clip(clip, 0, single)

    plan = plan_human_rig(_with_axes(config, "left_knee", ("y", "x", "z")))
    values, worst = joint_values_from_clip(clip, 0, plan)
    assert worst < 1e-9
    target = axis_angle_to_matrix(np.asarray(clip.rotation_of(0, "left_knee")))
    rebuilt = _reconstruct_chain(plan, values, "left_knee")
    assert _geodesic_deg(target, rebuilt) < 1e-6


def test_two_axis_joint_reports_the_miss_it_cannot_reach(tmp_path: Path) -> None:
    """Two axes are honest about their limit, and better than one axis.

    The residual is the geodesic angle to the nearest reachable rotation, so it
    is a bound a caller can compare against an anatomical tolerance instead of a
    coordinate-wise difference that means nothing.
    """

    config = load_human_config(CONFIG_PATH)
    clip = _clip_with_rotations(tmp_path, {4: (1.1, -0.7, 0.9)})
    target = axis_angle_to_matrix(np.asarray(clip.rotation_of(0, "left_knee")))

    single = plan_human_rig(config)
    _, single_residual = axis_residuals(clip, 0, single)["left_knee"]

    plan = plan_human_rig(_with_axes(config, "left_knee", ("y", "x")))
    residuals = axis_residuals(clip, 0, plan)
    values, worst = joint_values_from_clip(clip, 0, plan, axis_tolerance_rad=math.inf)

    # Every DOF of the chain shares one residual, because they were solved together.
    shared = {
        residuals[name][1] for name in plan.dof_names_for("left_knee")
    }
    assert len(shared) == 1
    assert worst < single_residual
    # ``worst`` is in radians; the independent metric reports degrees.
    assert math.degrees(worst) == pytest.approx(
        _geodesic_deg(target, _reconstruct_chain(plan, values, "left_knee")), abs=1e-6
    )


def test_multi_axis_chain_is_rejected_when_the_motion_needs_more_axes(tmp_path: Path) -> None:
    """A chain that still cannot reach the motion is refused, not approximated.

    The gate is unchanged in spirit: an inexpressible frame raises instead of
    being projected into a different motion, it is only now measured against what
    the chain can actually do.
    """

    config = load_human_config(CONFIG_PATH)
    clip = _clip_with_rotations(tmp_path, {4: (1.1, -0.7, 0.9)})
    plan = plan_human_rig(_with_axes(config, "left_knee", ("y", "x")))
    with pytest.raises(ValueError, match="axis the rig cannot express"):
        joint_values_from_clip(clip, 0, plan)


# ---------------------------------------------------------------------------
# the multi-axis rig must be the same body, not a different one
# ---------------------------------------------------------------------------


def test_multi_axis_rig_places_every_link_where_the_single_axis_rig_does() -> None:
    """A chain must not move the bone it articulates.

    The proxy chain's first link hangs off the joint's own SMPL parent, so it is the
    one link that has to carry the bone offset. Dropping that anchor put every chain
    link at the origin, collapsed the body onto its root and reported a 0.80 m
    standing height for a 1.70 m figure -- invisible in a link count and in every
    "the plan built" check, obvious only in the kinematics.
    """

    single_config = load_human_config(CONFIG_PATH)
    multi_config = load_human_config(MULTIAXIS_CONFIG_PATH)
    # Isolate chain layout from the AMASS config's additional terminal colliders.
    multi_config = replace(multi_config, rig=replace(
        multi_config.rig, segments=single_config.rig.segments
    ))
    single = plan_human_rig(single_config)
    multi = plan_human_rig(multi_config)
    assert multi.stats["dof_count"] > single.stats["dof_count"]
    assert multi.stats["standing_height_m"] == pytest.approx(
        single.stats["standing_height_m"], abs=1e-6
    )
    assert multi.standing_root_height_m == pytest.approx(
        single.standing_root_height_m, abs=1e-6
    )
    reference = forward_kinematics(single, {}, root_position=(0.0, 0.0, 0.0))
    chained = forward_kinematics(multi, {}, root_position=(0.0, 0.0, 0.0))
    for link in single.links:
        assert np.allclose(
            chained[link.name].translation, reference[link.name].translation, atol=1e-9
        ), link.name
    # Every proxy except the first sits on its parent; the first carries the bone.
    for joint in multi.joints:
        if joint.proxy and joint.local_pos0 != (0.0, 0.0, 0.0):
            assert joint.name.endswith("__dof1"), joint.name


def test_shipped_multiaxis_config_is_three_axis_per_moving_joint() -> None:
    """The shipped AMASS rig: every joint the corpus rotates, three axes each."""

    config = load_human_config(MULTIAXIS_CONFIG_PATH)
    plan = plan_human_rig(config)
    groups = dof_groups(plan)
    assert plan.stats["driven_joint_count"] == len(groups)
    for name, joints in groups.items():
        assert len(joints) == 3, name
        assert tuple(j.axis for j in joints) == tuple(dict.fromkeys(j.axis for j in joints))
        assert len({joint.axis for joint in joints}) == 3, name
    # The hands have no AMASS counterpart and the feet measure no rotation.
    for name in ("left_hand", "right_hand", "left_foot", "right_foot"):
        assert name not in groups
    # A three-axis chain must be able to hold any rotation, so every group's declared
    # limits have to include the identity.
    for name, joints in groups.items():
        assert all(j.lower_deg < 0 < j.upper_deg for j in joints), name


def test_neutral_pose_is_validated_against_the_axis_it_is_applied_to() -> None:
    """The pose check must read the limits of the DOF that carries the joint's name.

    ``apply_neutral_pose`` writes a named joint's value into the slot of the axis that
    DOF actually has -- the LAST declared axis, which belongs to the real link. The
    config check read ``limits_deg[0]`` instead, so on a three-axis joint it validated
    against a proxy axis: it rejected valid poses and would have admitted invalid ones.
    """

    from dataclasses import replace

    from sim2sense_fall.humans.config import VisualizationConfig

    base = load_human_config(CONFIG_PATH)
    shoulder = base.rig.joints["left_shoulder"]
    # Primary axis (last) admits +20 deg only; the first axis is wide open.
    tightened = JointConfig(
        joint="left_shoulder",
        dof=shoulder.dof,
        rotations=("y", "z", "x"),
        limits_deg=((-180.0, 180.0), (-180.0, 180.0), (0.0, 30.0)),
        drive=shoulder.drive,
    )
    rig = replace(base.rig, joints={**base.rig.joints, "left_shoulder": tightened})

    inside = replace(
        base,
        rig=rig,
        visualization=VisualizationConfig(default_pose_rad=(("left_shoulder", 0.35),)),
    )
    assert inside.visualization.default_pose_rad == (("left_shoulder", 0.35),)

    # -100 deg sits inside the FIRST axis's range but outside the axis that carries
    # the value; validating against [0] would have accepted it.
    with pytest.raises(ValueError, match="outside its configured limits"):
        replace(
            base,
            rig=rig,
            visualization=VisualizationConfig(
                default_pose_rad=(("left_shoulder", -1.7453292519943295),)
            ),
        )
