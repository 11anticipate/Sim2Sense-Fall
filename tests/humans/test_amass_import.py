"""CPU tests for the local AMASS import and fall-candidate screen."""

from __future__ import annotations

import hashlib
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
from sim2sense_fall.humans.rig import plan_human_rig

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/humans/human_smpl_neutral.yaml"


def _write_sequence(path: Path, *, falling: bool) -> None:
    frame_count = 61
    poses = np.zeros((frame_count, 156), dtype=np.float64)
    trans = np.zeros((frame_count, 3), dtype=np.float64)
    if falling:
        # AMASS is Y-up.  A source Z-axis body rotation maps to the rig's Y axis
        # after Y-up -> Z-up conversion, and source Y translation is vertical.
        trans[:, 1] = np.linspace(0.0, -0.30, frame_count)
        for frame, angle in enumerate(np.linspace(0.0, np.deg2rad(150.0), frame_count)):
            for joint in (3, 6, 9):
                poses[frame, joint * 3 + 2] = angle / 3.0
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
    assert clip.metadata["up_axis_conversion"] == "y->z"


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
    poses[:, 0] = np.linspace(0.4, 0.7, poses.shape[0])
    trans[:, 0] = np.linspace(2.0, 2.3, trans.shape[0])
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
    assert np.allclose(anchored.root_rotation[0], 0.0, atol=1e-10)
    assert np.allclose(clip.root_translation[0], [2.0, 0.0, 0.0])
    assert anchored.metadata["root_motion_normalization"]["anchor"] == "first_frame"


def test_library_is_deterministic_and_rejects_missing_required_fields(tmp_path: Path):
    first = tmp_path / "Subject2" / "b.npz"
    second = tmp_path / "Subject1" / "a.npz"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_sequence(first, falling=False)
    _write_sequence(second, falling=False)

    library = load_amass_library(tmp_path)
    assert list(library) == ["amass__a", "amass__b"]

    broken = tmp_path / "Subject3" / "broken.npz"
    broken.parent.mkdir()
    np.savez(broken, poses=np.zeros((2, 156)), trans=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="frame-rate"):
        load_amass_clip(broken, root=tmp_path)
    selected = load_amass_clip_by_id(tmp_path, "amass__a")
    assert selected.clip_id == "amass__a"
    with pytest.raises(FileNotFoundError, match="no AMASS .npz"):
        load_amass_clip_by_id(tmp_path, "amass__missing")
