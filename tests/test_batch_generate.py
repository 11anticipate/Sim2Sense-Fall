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


def test_per_sample_rt_seeds_are_stable_across_expansions(batch_root: Path, monkeypatch):
    """A resumable batch must not let completion order change a sample's seed."""

    plan_path = batch_root / "plan_seed.yaml"
    plan_path.write_text(
        "description: seed plan\ntrials: []\n"
        "sessions:\n"
        "  - {config: configs/humans/keyboard.yaml, name: session_a}\n"
        "  - {config: configs/humans/keyboard.yaml, name: session_b}\n"
        "rt_frames: 12\nrt_seed_base: 20260927\nrt_target_hz: 120\n",
        encoding="utf-8",
    )
    out = batch_root / "seeds"
    _finish_session(out, "session_a", "stand_00")
    _finish_session(out, "session_b", "stand_00")
    monkeypatch.setattr(
        sys, "argv", ["batch_generate", "--plan", str(plan_path), "--out", str(out)]
    )
    assert batch_generate.main() == 0
    first = [line for line in (out / "run_batch.sh").read_text(encoding="utf-8").splitlines()
             if "import_fall_mesh" in line]
    seeds_a = [line.split("--seed")[1].split()[0] for line in first]
    assert len(set(seeds_a)) == 2, f"both samples got the same seed: {seeds_a}"
    assert "--target-hz 120" in first[0]
    assert "--frames" not in first[0]
    assert batch_generate.main() == 0
    second = [line for line in (out / "run_batch.sh").read_text(encoding="utf-8").splitlines()
              if "import_fall_mesh" in line]
    assert [line.split("--seed")[1].split()[0] for line in second] == seeds_a


def test_a_negative_seed_base_is_rejected(batch_root: Path):
    plan_path = batch_root / "plan_bad_seed.yaml"
    plan_path.write_text(
        "description: bad\ntrials: []\n"
        "sessions:\n  - {config: configs/humans/keyboard.yaml, name: keyboard}\n"
        "rt_seed_base: -1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rt_seed_base"):
        batch_generate.load_plan(plan_path)


def test_session_chunk_covers_every_session_once_and_in_order():
    """Chunking is a disk workaround, so it must never drop or duplicate a session."""

    names = [f"s{i:02d}" for i in range(7)]
    chunks = [batch_generate.session_chunk(names, i, 3) for i in range(3)]
    assert [len(chunk) for chunk in chunks] == [3, 2, 2]
    assert [name for chunk in chunks for name in chunk] == names


def test_session_chunk_fits_when_there_are_fewer_sessions_than_chunks():
    assert batch_generate.session_chunk(["a", "b"], 2, 3) == []
    assert batch_generate.session_chunk(["a", "b"], 0, 3) == ["a"]


def test_session_chunk_rejects_an_out_of_range_index():
    with pytest.raises(ValueError, match="outside"):
        batch_generate.session_chunk(["a"], 1, 1)


def _plan(batch_root: Path, name: str, body: str) -> Path:
    path = batch_root / name
    path.write_text(body, encoding="utf-8")
    return path


_TWO_SESSIONS = ("description: chunk test\ntrials: []\n"
                 "sessions:\n"
                 "  - {config: configs/humans/keyboard.yaml, name: session_a}\n"
                 "  - {config: configs/humans/keyboard.yaml, name: session_b}\n"
                 "rt_frames: 12\nrt_target_hz: 120\n")


def test_a_plan_chunk_only_emits_its_own_sessions(batch_root: Path, monkeypatch):
    plan_path = _plan(batch_root, "plan_chunk.yaml", _TWO_SESSIONS)
    out = batch_root / "chunked"
    monkeypatch.setattr(sys, "argv", [
        "batch_generate", "--plan", str(plan_path), "--out", str(out),
        "--session-chunk-index", "1", "--session-chunk-count", "2",
    ])
    assert batch_generate.main() == 0
    body = (out / "run_batch.sh").read_text(encoding="utf-8")
    assert "session_session_b" in body
    assert "session_session_a" not in body


def _pruned_rt_line(batch_root: Path, monkeypatch, out_name: str, prune: bool) -> str:
    plan_path = _plan(batch_root, f"plan_{out_name}.yaml", _TWO_SESSIONS
                      + f"prune_mesh_after_import: {'true' if prune else 'false'}\n")
    out = batch_root / out_name
    _finish_session(out, "session_a", "stand_00")
    monkeypatch.setattr(sys, "argv", ["batch_generate", "--plan", str(plan_path),
                                      "--out", str(out)])
    assert batch_generate.main() == 0
    lines = [line for line in (out / "run_batch.sh").read_text(encoding="utf-8").splitlines()
             if "import_fall_mesh" in line]
    assert len(lines) == 1, lines
    return lines[0]


def test_a_pruned_mesh_is_deleted_only_after_its_own_trace_succeeds(batch_root: Path,
                                                                   monkeypatch):
    """`cmd && rm || echo` would delete the mesh of a failed trace, and its retry needs it."""

    line = _pruned_rt_line(batch_root, monkeypatch, "prune", True)
    assert line.startswith("if ")
    assert "; then rm -f '" in line
    assert "session_session_a_export/stand_00.mesh.npz';" in line
    assert "else echo 'transient-failure: session_a/stand_00'" in line
    assert "; fi" in line
    # Only this segment's mesh: a wildcard would erase the retries of its siblings.
    assert "*" not in line.split("rm -f")[1].split(";")[0]


def test_without_pruning_the_retry_keeps_the_plain_form(batch_root: Path, monkeypatch):
    line = _pruned_rt_line(batch_root, monkeypatch, "keep", False)
    assert "|| echo 'transient-failure:" in line
    assert "rm -f" not in line


def test_the_disk_guard_runs_after_the_cd_and_uses_the_plan_floor(batch_root: Path,
                                                                 monkeypatch):
    plan_path = _plan(batch_root, "plan_guard.yaml", _TWO_SESSIONS + "min_free_gb: 6\n")
    out = batch_root / "guard"
    monkeypatch.setattr(sys, "argv", ["batch_generate", "--plan", str(plan_path),
                                      "--out", str(out)])
    assert batch_generate.main() == 0
    lines = (out / "run_batch.sh").read_text(encoding="utf-8").splitlines()
    guard = [index for index, line in enumerate(lines) if line.startswith("free_kb=")][0]
    cd = [index for index, line in enumerate(lines) if line.startswith("cd ")][0]
    assert guard > cd, "df must measure the filesystem the batch actually writes to"
    assert f"-lt {6 * 2 ** 20}" in "\n".join(lines)


def test_a_plan_without_a_floor_emits_no_guard(batch_root: Path, monkeypatch):
    plan_path = _plan(batch_root, "plan_noguard.yaml", _TWO_SESSIONS)
    out = batch_root / "noguard"
    monkeypatch.setattr(sys, "argv", ["batch_generate", "--plan", str(plan_path),
                                      "--out", str(out)])
    assert batch_generate.main() == 0
    assert "free_kb=" not in (out / "run_batch.sh").read_text(encoding="utf-8")


def test_generated_header_text_is_never_counted_as_a_pending_stage(batch_root: Path,
                                                                  monkeypatch):
    """run_batch_loop.sh counts pending stages by grepping the script for stage names.

    A comment or guard line that merely mentions a stage script would make every pass look
    non-convergent, so the non-command part of the file has to stay free of those names.
    """

    import re

    plan_path = _plan(batch_root, "plan_header.yaml",
                      _TWO_SESSIONS + "min_free_gb: 6\nprune_mesh_after_import: true\n")
    out = batch_root / "header"
    monkeypatch.setattr(sys, "argv", ["batch_generate", "--plan", str(plan_path),
                                      "--out", str(out)])
    assert batch_generate.main() == 0
    lines = (out / "run_batch.sh").read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"keyboard\.py|export_session_mesh|import_fall_mesh|simulate\.py")
    preamble = [line for line in lines
                if line.startswith(("#", "free_kb=", "if [", "  echo", "  exit",
                                    "fi", "set ", "cd "))
                and "then rm -f" not in line]
    assert preamble
    assert not [line for line in preamble if pattern.search(line)], preamble


def test_a_negative_free_space_floor_is_rejected(batch_root: Path):
    plan_path = _plan(batch_root, "plan_bad_floor.yaml", _TWO_SESSIONS + "min_free_gb: -1\n")
    with pytest.raises(ValueError, match="min_free_gb"):
        batch_generate.load_plan(plan_path)
