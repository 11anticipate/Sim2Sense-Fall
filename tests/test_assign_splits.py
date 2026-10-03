"""`assign_splits.collect` must name a sample exactly like the training loader does.

The two batches share session names (`01_fall_standing_sp040_bedroom` exists in both
train02 and train03), so a group derived from a bare directory name would merge two
different physics runs into one leakage group -- and a sample id derived the same way would
merge two different CIR files into one row. Round 2 trains on a root that merges the
batches, so both identities have to carry every directory level under the root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sionna"))

import assign_splits  # noqa: E402


def _write_sample(root: Path, namespace: str, stem: str, *, source: str,
                  label: str = "stand", failures: list[str] | None = None) -> Path:
    directory = root / namespace if namespace else root
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.import.json"
    path.write_text(json.dumps({
        "source": {"source": source},
        "source_event_label": label,
        "activity": "adl" if label == "stand" else "fall",
        "frames": [0, 1, 2, 3],
        "sample_rate_hz": 120.0,
        "seed": 42,
        "subject_id": "smpl_neutral",
        "failures": failures or [],
    }), encoding="utf-8")
    return path


def test_a_single_batch_root_names_samples_by_session_and_segment(tmp_path: Path):
    root = tmp_path / "sionna"
    _write_sample(root, "session_a", "stand_00", source="/x/session_a_export/manifest.json")
    _write_sample(root, "session_a", "walk_00", source="/x/session_a_export/manifest.json")
    _write_sample(root, "session_b", "stand_00", source="/y/session_b_export/manifest.json")
    rows, refused = assign_splits.collect(tmp_path, root)
    assert refused == []
    assert [row["sample_id"] for row in rows] == [
        "session_a/stand_00", "session_a/walk_00", "session_b/stand_00"]
    assert [row["group"] for row in rows] == ["session_a", "session_a", "session_b"], (
        "segments of one session stay in one leakage group")


def test_two_batches_with_the_same_session_name_stay_two_groups(tmp_path: Path):
    root = tmp_path / "merged"
    for batch in ("train02", "train03"):
        _write_sample(root / batch, "session_01_fall_standing", "stand_00",
                      source=f"/artifacts/batches/{batch}/session_01_fall_standing_export/manifest.json")
    rows, _ = assign_splits.collect(tmp_path, root)
    ids = [row["sample_id"] for row in rows]
    groups = [row["group"] for row in rows]
    assert len(set(ids)) == 2, f"the two batches overwrote each other's identity: {ids}"
    assert len(set(groups)) == 2, f"two different runs collapsed into one group: {groups}"
    assert groups == ["train02/session_01_fall_standing", "train03/session_01_fall_standing"]


def test_root_level_trial_samples_keep_the_path_derived_group(tmp_path: Path):
    root = tmp_path / "sionna"
    _write_sample(root, "", "push_backward__low",
                  source="/artifacts/trials/push_backward__low.trial.json")
    rows, _ = assign_splits.collect(tmp_path, root)
    assert [row["sample_id"] for row in rows] == ["push_backward__low"]
    assert [row["group"] for row in rows] == ["push_backward__low"]


def test_a_failed_sample_is_refused_and_not_grouped(tmp_path: Path):
    root = tmp_path / "sionna"
    _write_sample(root, "session_a", "stand_00", source="/x/session_a_export/manifest.json",
                  failures=["static_body_repeat"])
    _write_sample(root, "session_a", "stand_01", source="/x/session_a_export/manifest.json")
    rows, refused = assign_splits.collect(tmp_path, root)
    assert refused == ["session_a/stand_00"]
    assert [row["sample_id"] for row in rows] == ["session_a/stand_01"]


def test_a_missing_source_or_label_fails_loudly(tmp_path: Path):
    root = tmp_path / "sionna"
    path = _write_sample(root, "session_a", "stand_00", source="/x/manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing source path or event label"):
        assign_splits.collect(tmp_path, root)
