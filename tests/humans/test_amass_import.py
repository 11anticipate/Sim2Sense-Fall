"""CPU tests for the local AMASS import and fall-candidate screen."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.amass import (
    annotate_clip,
    load_amass_clip,
    load_amass_clip_by_id,
    load_amass_library,
    normalize_root_motion,
    screen_amass_clip,
)
from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.motion import AMASS_BODY_FRAME
from sim2sense_fall.humans.rig import joint_values_from_clip, plan_human_rig
from sim2sense_fall.humans.rotations import axis_angle_to_matrix, matrix_to_axis_angle

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/humans/human_smpl_neutral.yaml"


def _write_sequence(path: Path, *, falling: bool) -> None:
    frame_count = 61
    poses = np.zeros((frame_count, 156), dtype=np.float64)
    poses[:, :3] = matrix_to_axis_angle(AMASS_BODY_FRAME.basis())
    trans = np.zeros((frame_count, 3), dtype=np.float64)
    if falling:
        # AMASS authors local joint rotations in the body model's own frame, where
        # +X is the body's left axis. The pipeline maps that onto its +Y, which is
        # the axis every shipped rig joint turns about -- so the source X component
        # is what a single-axis rig can express. Writing source Z instead (the body's
        # forward axis) would land on the pipeline X and be correctly reported as
        # inexpressible.
        trans[:, 2] = np.linspace(0.0, -0.30, frame_count)
        for frame, angle in enumerate(np.linspace(0.0, np.deg2rad(150.0), frame_count)):
            for joint in (3, 6, 9):
                poses[frame, joint * 3 + 0] = angle / 3.0
    np.savez(
        path,
        poses=poses,
        trans=trans,
        mocap_framerate=np.asarray(30.0),
        gender=np.asarray("neutral"),
        betas=np.zeros(16, dtype=np.float64),
    )


def test_load_amass_clip_validates_fields_hashes_source_and_converts_axes(tmp_path: Path):
    source = tmp_path / "Subject1" / "walk.npz"
    source.parent.mkdir()
    _write_sequence(source, falling=False)

    clip = load_amass_clip(source, root=tmp_path)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    assert clip.provenance.kind == "amass"
    assert clip.provenance.subject == "Subject1"
    assert clip.provenance.source_sha256 == expected_hash
    assert clip.root_translation.shape == (61, 3)
    assert np.allclose(clip.root_translation, 0.0)
    assert clip.joint_rotations.shape == (61, 24, 3)
    assert clip.metadata["source_body_frame"] == "up=y forward=z left=x"


def test_screen_marks_fall_candidate_and_annotation(tmp_path: Path):
    source = tmp_path / "Subject1" / "fall.npz"
    source.parent.mkdir()
    _write_sequence(source, falling=True)
    clip = load_amass_clip(source, root=tmp_path)
    plan = plan_human_rig(load_human_config(CONFIG_PATH))

    result = screen_amass_clip(clip, plan)
    tagged = annotate_clip(clip, result)

    assert result.accepted
    assert result.peak_trunk_angle_deg > 60.0
    assert result.root_drop_m >= 0.20
    assert "fall_reference" in tagged.tags
    assert tagged.metadata["amass_screen"]["accepted"] is True


def test_normalize_root_motion_anchors_translation_and_global_rotation(tmp_path: Path):
    source = tmp_path / "Subject1" / "sit_stand.npz"
    source.parent.mkdir()
    _write_sequence(source, falling=False)
    with np.load(source) as archive:
        poses = archive["poses"].copy()
        trans = archive["trans"].copy()
    tilt = axis_angle_to_matrix(np.array([0.0, 0.4, 0.0]))
    yaw = axis_angle_to_matrix(np.array([0.0, 0.0, 0.7]))
    poses[:, :3] = matrix_to_axis_angle(yaw @ tilt @ AMASS_BODY_FRAME.basis())
    trans[:, 0] = np.linspace(2.0, 2.3, trans.shape[0])
    trans[:, 2] = np.linspace(1.0, 0.6, trans.shape[0])
    np.savez(
        source,
        poses=poses,
        trans=trans,
        mocap_framerate=np.asarray(30.0),
        gender=np.asarray("neutral"),
    )
    clip = load_amass_clip(source, root=tmp_path)
    anchored = normalize_root_motion(clip)

    assert np.allclose(anchored.root_translation[0], 0.0)
    assert np.allclose(axis_angle_to_matrix(anchored.root_rotation[0]), tilt)
    assert np.allclose(clip.root_translation[0], [2.0, 0.0, 1.0])
    assert np.allclose(anchored.root_translation[-1], yaw.T @ [0.3, 0.0, -0.4])
    assert anchored.metadata["root_motion_normalization"]["preserves_gravity_and_initial_tilt"]


def test_retarget_preserves_source_world_geometry_for_noncommuting_rotations(tmp_path: Path):
    """An independently posed source bone must end up at the same world point."""
    source = tmp_path / "geometry.npz"
    _write_sequence(source, falling=False)
    with np.load(source) as archive:
        poses, trans = archive["poses"].copy(), archive["trans"].copy()
    poses[:, :3] = [1.3, 0.25, -0.45]
    poses[:, 3:6] = [0.3, -0.2, 0.4]
    trans[:] = [2.0, -3.0, 0.8]
    np.savez(source, poses=poses, trans=trans, mocap_framerate=30.0)
    clip = load_amass_clip(source)
    bone = np.array([0.1, -0.4, 0.03])
    source_point = (
        axis_angle_to_matrix(poses[0, :3]) @ axis_angle_to_matrix(poses[0, 3:6]) @ bone
        + trans[0]
    )
    target_point = (
        axis_angle_to_matrix(clip.root_rotation[0])
        @ axis_angle_to_matrix(clip.joint_rotations[0, 1])
        @ (AMASS_BODY_FRAME.basis() @ bone)
        + clip.root_translation[0]
    )
    np.testing.assert_allclose(target_point, source_point, atol=1e-12)
    anchored = normalize_root_motion(clip)
    np.testing.assert_allclose(
        axis_angle_to_matrix(anchored.root_rotation[0])[2],
        axis_angle_to_matrix(clip.root_rotation[0])[2],
        atol=1e-12,
    )


def test_library_is_deterministic_and_rejects_missing_required_fields(tmp_path: Path):
    first = tmp_path / "Subject2" / "b.npz"
    second = tmp_path / "Subject1" / "a.npz"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_sequence(first, falling=False)
    _write_sequence(second, falling=False)

    library = load_amass_library(tmp_path)
    assert list(library.clips) == ["amass__a", "amass__b"]
    assert library.failures == ()
    assert library.scanned == 2

    broken = tmp_path / "Subject3" / "broken.npz"
    broken.parent.mkdir()
    np.savez(broken, poses=np.zeros((2, 156)), trans=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="frame-rate"):
        load_amass_clip(broken, root=tmp_path)
    selected = load_amass_clip_by_id(tmp_path, "amass__a")
    assert selected.clip_id == "amass__a"
    with pytest.raises(FileNotFoundError, match="no AMASS .npz"):
        load_amass_clip_by_id(tmp_path, "amass__missing")


def test_screen_rejects_rotations_outside_physical_joint_limits(tmp_path: Path):
    source = tmp_path / "knee.npz"
    _write_sequence(source, falling=False)
    clip = load_amass_clip(source)
    rotations = clip.joint_rotations.copy()
    rotations[:, 4, 1] = np.deg2rad(170)
    clip = replace(clip, joint_rotations=rotations)
    result = screen_amass_clip(clip, plan_human_rig(load_human_config(CONFIG_PATH)))
    assert not result.rig_expressible
    assert result.joint_limit_excess_deg > 0
    assert result.joint_limit_joint == "left_knee"
    assert "joint limit" in result.reason


def test_screen_reports_the_binding_joint_and_agrees_with_the_rig_mapper(tmp_path: Path):
    """A refused clip must say which DOF the rig is missing, and both paths must agree.

    The screen used to apply its own looser 25-degree bound while the simulator's
    mapper accepted nothing at all, so a clip could be reported usable here and raise
    there. Both now consult :func:`joint_values_from_clip`, and a bare "0 clips passed"
    used to be indistinguishable from "this corpus has no falls" -- which is how the
    whole AMASS library came to be written off.
    """

    source = tmp_path / "Subject1" / "fall.npz"
    source.parent.mkdir()
    _write_sequence(source, falling=True)
    clip = load_amass_clip(source, root=tmp_path)
    plan = plan_human_rig(load_human_config(CONFIG_PATH))

    expressible = screen_amass_clip(clip, plan)
    assert expressible.rig_expressible
    assert expressible.fall_candidate
    assert expressible.accepted
    assert expressible.axis_residual_joint == ""

    # Rotate the left knee about an axis no shipped joint turns about.
    offset = clip.joint_rotations.copy()
    offset[:, 4, 2] += 0.5
    twisted = replace(clip, joint_rotations=offset)

    screened = screen_amass_clip(twisted, plan)
    assert not screened.rig_expressible
    assert not screened.accepted
    assert screened.fall_candidate, "the motion is still a fall; the rig is what fails"
    assert screened.axis_residual_joint == "left_knee"
    with pytest.raises(ValueError, match="axis the rig cannot express"):
        joint_values_from_clip(twisted, 0, plan)


def test_library_skips_and_records_an_unreadable_sequence_instead_of_aborting(tmp_path: Path):
    """One unusable file must not cost a screening job the other two thousand.

    The batch used to raise on the first sequence whose motion was discontinuous,
    so a single bad recording turned a full-library screen into no result at all --
    and a skipped file that is not named makes a candidate count unreadable.
    """

    good = tmp_path / "Subject1" / "a.npz"
    spike = tmp_path / "Subject2" / "b.npz"
    twin = tmp_path / "Subject3" / "a.npz"
    for directory in (good, spike, twin):
        directory.parent.mkdir()
    _write_sequence(good, falling=False)
    _write_sequence(spike, falling=False)
    _write_sequence(twin, falling=False)
    with np.load(spike) as archive:
        poses = np.asarray(archive["poses"]).copy()
        trans = np.asarray(archive["trans"]).copy()
    poses[5, 3:6] += 1.6  # a 92-degree single-frame jump, under the 180-degree bound
    np.savez(
        spike,
        poses=poses,
        trans=trans,
        mocap_framerate=np.asarray(30.0),
        gender=np.asarray("neutral"),
    )

    library = load_amass_library(tmp_path)

    assert list(library.clips) == ["amass__a"]
    assert library.scanned == 3
    assert len(library.failures) == 2
    reasons = [error for _, error in library.failures]
    assert any("continuity" in reason for reason in reasons)
    assert any("duplicate" in reason for reason in reasons)
