"""Batch plan expansion must work on a fresh output directory (regression).

`batch_generate.py` is the pre-training data orchestration brain: it writes
``run_batch.sh`` for the pending stages. It used to write that file into a batch
root it never created, so the *first* expansion of a new batch died with
``FileNotFoundError`` and only a pre-made directory worked.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sionna"))

import batch_generate  # noqa: E402

# batch_generate refuses output roots outside the repository, so the scratch batch
# lives under the gitignored artifacts/ tree and is removed by the fixture.
SCRATCH = REPO_ROOT / "artifacts" / "_pytest_batch"


@pytest.fixture
def batch_root(tmp_path: Path):
    path = SCRATCH / tmp_path.name
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_plan(path: Path) -> Path:
    path.write_text(
        "description: test batch\n"
        "trials: []\n"
        "sessions:\n"
        "  - {config: configs/humans/keyboard.yaml, name: keyboard}\n"
        "rt_frames: 12\n",
        encoding="utf-8",
    )
    return path


def test_expansion_creates_a_fresh_batch_root(batch_root: Path, monkeypatch):
    plan_path = _write_plan(batch_root / "plan.yaml")
    out = batch_root / "fresh"
    assert not out.exists()
    monkeypatch.setattr(
        sys, "argv", ["batch_generate", "--plan", str(plan_path), "--out", str(out)]
    )
    assert batch_generate.main() == 0
    script = out / "run_batch.sh"
    assert script.is_file(), "the first expansion of a new batch must still be writable"
    body = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in body
    assert "keyboard.py --headless --demo --native-mesh" in body


def test_expansion_is_resumable_and_only_emits_pending_stages(batch_root: Path, monkeypatch):
    plan_path = _write_plan(batch_root / "plan.yaml")
    out = batch_root / "resume"
    monkeypatch.setattr(
        sys, "argv", ["batch_generate", "--plan", str(plan_path), "--out", str(out)]
    )
    assert batch_generate.main() == 0
    first = (out / "run_batch.sh").read_text(encoding="utf-8")
    # Pretend the session stage finished and was admitted: the next expansion must
    # not re-emit it, and must instead talk about the stages that are still missing.
    session = out / "session_keyboard"
    session.mkdir(parents=True, exist_ok=True)
    # A usable session record: it ran, it has no errors, and its measurement block
    # carries the schema the exporter understands (see session_accepted).
    (session / "report.json").write_text(
        '{"runtime_completed": true, "errors": [], "motion_accuracy_accepted": true,'
        ' "motion_quality": {"schema_version": 2, "accepted": true, "modes": {}}}',
        encoding="utf-8",
    )
    assert batch_generate.main() == 0
    second = (out / "run_batch.sh").read_text(encoding="utf-8")
    assert "keyboard.py --headless" not in second
    assert "export_session_mesh.py" in second
    assert len(second.splitlines()) < len(first.splitlines())


def _finish_session(out: Path, name: str, sample_id: str) -> None:
    """Mark a session as run and exported, with one admitted segment."""

    import json

    session = out / f"session_{name}"
    session.mkdir(parents=True, exist_ok=True)
    (session / "report.json").write_text(json.dumps({
        "runtime_completed": True, "errors": [], "motion_accuracy_accepted": True,
    }), encoding="utf-8")
    export = out / f"session_{name}_export"
    export.mkdir(parents=True, exist_ok=True)
    (export / "manifest.json").write_text(json.dumps({
        "session_invariants_ok": True,
        "samples": [{"sample_id": sample_id, "admitted_for_training": True,
                     "label": {"label": "stand"}}],
    }), encoding="utf-8")


def test_identical_segment_ids_from_two_sessions_do_not_collide(batch_root: Path, monkeypatch):
    """Segment ids repeat per session; a shared RT output directory overwrote them.

    The first train01 batch turned 26 admitted segments into 9 CIR files this way.
    Each session must get its own namespace under sionna/.
    """

    plan_path = batch_root / "plan2.yaml"
    plan_path.write_text(
        "description: collision test\ntrials: []\n"
        "sessions:\n"
        "  - {config: configs/humans/keyboard.yaml, name: session_a}\n"
        "  - {config: configs/humans/keyboard.yaml, name: session_b}\n"
        "rt_frames: 12\n",
        encoding="utf-8",
    )
    out = batch_root / "collide"
    _finish_session(out, "session_a", "stand_00")
    _finish_session(out, "session_b", "stand_00")
    monkeypatch.setattr(
        sys, "argv", ["batch_generate", "--plan", str(plan_path), "--out", str(out)]
    )
    assert batch_generate.main() == 0
    body = (out / "run_batch.sh").read_text(encoding="utf-8")
    rt_lines = [line for line in body.splitlines() if "import_fall_mesh.py" in line]
    assert len(rt_lines) == 2, body
    targets = {line.split("--out")[1].split()[0] for line in rt_lines}
    assert len(targets) == 2, f"both sessions write to the same RT directory: {targets}"
    assert any("session_a" in target for target in targets)
    assert any("session_b" in target for target in targets)


def test_a_run_completed_session_is_not_re_run_forever(batch_root: Path, monkeypatch):
    """Per-segment admission means one bad activity must not blacklist the session.

    train01 re-emitted 7 of 8 sessions because the idempotency check required the
    aggregate motion verdict, which any single failing activity drags down.
    """

    import json

    plan_path = batch_root / "plan3.yaml"
    plan_path.write_text(
        "description: rerun test\ntrials: []\n"
        "sessions:\n  - {config: configs/humans/keyboard.yaml, name: keyboard}\n"
        "rt_frames: 12\n",
        encoding="utf-8",
    )
    out = batch_root / "rerun"
    session = out / "session_keyboard"
    session.mkdir(parents=True, exist_ok=True)
    (session / "report.json").write_text(json.dumps({
        "runtime_completed": True, "errors": [],
        "motion_accuracy_accepted": False,
        "motion_quality": {"schema_version": 2, "accepted": False,
                           "modes": {"stand": {"accepted": False}}},
    }), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["batch_generate", "--plan", str(plan_path), "--out", str(out)]
    )
    assert batch_generate.main() == 0
    body = (out / "run_batch.sh").read_text(encoding="utf-8")
    assert "keyboard.py --headless" not in body, "a completed session must not be re-run"
    assert "export_session_mesh.py" in body


def test_plan_can_request_a_channel_rate_instead_of_a_frame_count(batch_root: Path,
                                                                  monkeypatch):
    """`--frames` is a stride maximum, so it silently under-delivers the rate.

    train01 asked for 24 frames per sample and got 5-40 Hz, while the pre-registered
    front-end is 120 Hz. `rt_target_hz` must reach the importer as `--target-hz`.
    """

    plan_path = batch_root / "plan_rate.yaml"
    plan_path.write_text(
        "description: rate plan\ntrials: []\n"
        "sessions:\n  - {config: configs/humans/keyboard.yaml, name: keyboard}\n"
        "rt_frames: 24\nrt_target_hz: 120\n",
        encoding="utf-8",
    )
    out = batch_root / "rate"
    _finish_session(out, "keyboard", "stand_00")
    monkeypatch.setattr(
        sys, "argv", ["batch_generate", "--plan", str(plan_path), "--out", str(out)]
    )
    assert batch_generate.main() == 0
    body = (out / "run_batch.sh").read_text(encoding="utf-8")
    assert "--target-hz 120" in body
    assert "--frames 24" not in body
