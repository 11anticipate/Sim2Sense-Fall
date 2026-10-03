#!/usr/bin/env python3
"""Expand a batch plan into an idempotent, resumable stage script.

The pipeline stages are stable CLIs (Isaac physics trials, keyboard sessions,
session export, Sionna RT). This tool is the orchestration BRAIN, not the
executor: it validates the plan, checks which stages already have accepted
outputs, and writes ``run_batch.sh`` -- an ordered, ``set -euo pipefail``
script containing exactly the pending commands, shell-quoted. Re-invoking it
after an interruption emits only the stages that are still pending, which
makes a batch resumable without any execution logic living here.

1. physics trials   -- simulate.py per (motion, perturbation); a trial counts
   as done when its JSON exists and trials_index.json gates it as usable;
2. keyboard sessions -- keyboard.py --demo --native-mesh per config; done when
   report.json reports runtime_completed with no errors and motion admission;
3. session export    -- export_session_mesh.py (admission is per segment);
4. Sionna RT         -- import_fall_mesh.py for every admitted source; done
   when the .import.json exists with no failures.

After the script has run, ``--summarize`` collects every sample's activity,
label and source chain into ``batch_report.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.humans.quality import SCHEMA_VERSION  # noqa: E402

DEFAULT_ISAAC_PYTHON = Path.home() / "isaacsim" / "python.sh"
DEFAULT_SIONNA_PYTHON = Path.home() / ".local" / "opt" / "sionna" / "bin" / "python"

# Plan-provided names become command arguments; a strict allowlist keeps the
# emitted script free of anything the shell could reinterpret.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
# Distinct seeds available per base, kept well below the range a single sample's
# repeats need to alias.
RT_SEED_SPREAD = 10_000


@dataclass
class BatchPlan:
    description: str = ""
    trials: list[dict[str, str]] = field(default_factory=list)
    sessions: list[dict[str, str]] = field(default_factory=list)
    rt_frames: int = 12
    # When set, every RT stage traces at this channel rate instead of a frame count
    # (`--frames` is a stride maximum, so 96 requested frames of a 492-frame 120 Hz
    # segment silently become 82 frames at 20 Hz).
    rt_target_hz: float | None = None
    render_frames: bool = False
    # Recorded into every RT report. The split itself is assigned afterwards by
    # scripts/sionna/assign_splits.py; this only says what the batch promises.
    split_label: str = "smoke_only_not_train_test"
    # When set, every RT trace gets a per-sample seed derived from this base and the
    # sample's identity, so the batch covers different ray-sampling realisations instead
    # of one fixed channel seed. Derived from a checksum, never a counter: the batch is
    # resumable, so a seed must not depend on the order the stages happen to complete in.
    rt_seed_base: int | None = None
    # A 120 Hz native-mesh session costs ~30 MB of disk per simulated second (recording
    # ~10 MB/s plus segment meshes ~20 MB/s), so an 800 s ADL batch is bigger than a
    # developer disk. When enabled, each segment's ``*.mesh.npz`` is removed only after
    # its own RT command exits 0, which keeps the cumulative footprint at the recordings.
    # Nothing else reads those files (RT takes the mesh, the loaders take ``.cir.npz`` plus
    # the export ``manifest.json``, which stays), and the meshes are re-derivable from
    # ``recording.npz`` by re-running export_session_mesh.py on CPU.
    prune_mesh_after_import: bool = False
    # Refuse to start a pass when the filesystem holding the batch root has less than this
    # much free. A disk that fills mid-run corrupts Isaac's own scratch, not just the batch.
    min_free_gb: float = 0.0


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"{field_name} must match {_IDENTIFIER.pattern}, got {value!r}")
    return value


def _inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"{path} resolves outside the repository")
    return resolved


def load_plan(path: Path) -> BatchPlan:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("batch plan must be a mapping")
    plan = BatchPlan(
        description=str(payload.get("description", "")),
        trials=list(payload.get("trials") or []),
        sessions=list(payload.get("sessions") or []),
        rt_frames=int(payload.get("rt_frames", 12)),
        rt_target_hz=(None if payload.get("rt_target_hz") is None
                      else float(payload["rt_target_hz"])),
        render_frames=bool(payload.get("render_frames", False)),
        split_label=str(payload.get("split_label", "smoke_only_not_train_test")),
        rt_seed_base=(None if payload.get("rt_seed_base") is None
                      else int(payload["rt_seed_base"])),
        prune_mesh_after_import=bool(payload.get("prune_mesh_after_import", False)),
        min_free_gb=float(payload.get("min_free_gb", 0.0)),
    )
    for trial in plan.trials:
        if not {"motion", "perturbation"} <= set(trial):
            raise ValueError("each trial needs motion and perturbation")
        _identifier(trial["motion"], "trial.motion")
        _identifier(trial["perturbation"], "trial.perturbation")
    for session in plan.sessions:
        if "config" not in session:
            raise ValueError("each session needs a config path")
        _identifier(Path(session["config"]).stem, "session.config")
        if "name" in session:
            _identifier(session["name"], "session.name")
    if not plan.trials and not plan.sessions:
        raise ValueError("the plan declares no work")
    if not 2 <= plan.rt_frames <= 512:
        raise ValueError("rt_frames must be within [2, 512]")
    if plan.rt_target_hz is not None and not 1.0 <= plan.rt_target_hz <= 2000.0:
        raise ValueError("rt_target_hz must be within [1, 2000]")
    if plan.rt_seed_base is not None and plan.rt_seed_base < 0:
        raise ValueError("rt_seed_base must be non-negative")
    if plan.min_free_gb < 0:
        raise ValueError("min_free_gb must be non-negative")
    return plan


def usable_trial(trials_dir: Path, motion: str, perturbation: str) -> Path | None:
    index_path = trials_dir / "trials_index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for row in index["trials"]:
        name = Path(row["json"]).name
        if f"__{motion}__{perturbation}." in name and row.get("gates", {}).get("usable", False):
            return trials_dir / name
    return None


def session_accepted(report_path: Path) -> bool:
    """Did the session *run*, not was every activity good.

    Admission is per segment (`export_session_mesh._activity_gate`), and the
    aggregate `motion_accuracy_accepted` flag is dragged down by any single failing
    activity -- so gating the batch's idempotency on it made the driver re-run
    perfectly usable sessions forever (train01 re-emitted 7 of 8 sessions). The
    exporter keeps its own invariants (completed, no errors, current measurement
    schema) as the hard gate, and this mirrors exactly those.
    """

    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(
        report.get("runtime_completed") is True
        and not report.get("errors")
        and (report.get("motion_quality") or {}).get("schema_version") == SCHEMA_VERSION
    )


def admitted_session_sources(export_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("session_invariants_ok") is not True:
        raise ValueError(f"{export_dir}: session measurement invariants failed")
    return [
        {
            "sample_id": row["sample_id"],
            "export_dir": export_dir,
            "label": row["label"]["label"],
        }
        for row in manifest["samples"]
        if row.get("admitted_for_training") is True
        and row["label"]["label"] != "unknown"
    ]


def rt_done(sionna_dir: Path, sample_id: str) -> bool:
    """``sionna_dir`` may be a per-session namespace; see the session loop below."""

    report = sionna_dir / f"{sample_id}.import.json"
    if not report.is_file():
        return False
    return not json.loads(report.read_text(encoding="utf-8")).get("failures")


def session_chunk(names: list[str], index: int, count: int) -> list[str]:
    """Contiguous slice ``index`` of ``count`` equal parts, remainder to the early parts.

    The batch exists to keep GPU time resumable, but every session's 120 Hz mesh is ~30 MB
    per simulated second, so a 23-session plan can outgrow the disk before its RT stage ever
    runs. Splitting the plan into contiguous chunks bounds how many recordings and segment
    meshes coexist; outputs still accumulate in one batch root, so the split assignment and
    the data card see a single batch.
    """

    if count < 1:
        raise ValueError("chunk count must be >= 1")
    if not 0 <= index < count:
        raise ValueError(f"chunk index {index} outside [0, {count})")
    size, remainder = divmod(len(names), count)
    start = index * size + min(index, remainder)
    end = start + size + (1 if index < remainder else 0)
    return names[start:end]


def expand(plan: BatchPlan, args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    """Return the pending shell commands plus the expected sample manifest."""

    def rate_args() -> list[str]:
        return (["--target-hz", str(plan.rt_target_hz)] if plan.rt_target_hz
                else ["--frames", str(plan.rt_frames)])

    def seed_args(sample_id: str) -> list[str]:
        if plan.rt_seed_base is None:
            return []
        # zlib.crc32, not hash(): the built-in string hash is salted per process, so a
        # counter-free but salt-free derivation is what makes the seed reproducible.
        digest = zlib.crc32(sample_id.encode("utf-8")) % RT_SEED_SPREAD
        return ["--seed", str(plan.rt_seed_base + digest)]

    def rt_command(sample_id: str, argv: list[str], *, prune: Path | None = None) -> str:
        # Rare GPU scheduling transients (~2 in 9 runs) can fail the strict static-repeat
        # check. The importer still writes its report, with `failures` non-empty, and
        # `rt_done` treats that as not-done -- so the next re-expansion re-emits exactly this
        # sample. The batch must not abort on it, and the loader/`assign_splits` refuse it.
        failure_log = shlex.quote(str(args.out / "batch_failures.log"))
        if prune is None:
            return (shlex.join(argv) + " || "
                    + shlex.join(["echo", f"transient-failure: {sample_id}"])
                    + f" >> {failure_log}")
        # `if ... then ... else ... fi` rather than `cmd && rm || echo`: with the chained
        # form a failed rm would be logged as an RT transient, and a *successful* rm after
        # a failed import would delete the only input the retry needs.
        return (f"if {shlex.join(argv)}; then rm -f {shlex.quote(str(prune))}; "
                f"else echo 'transient-failure: {sample_id}' >> {failure_log}; fi")

    trials_dir = args.out / "trials"
    sionna_dir = args.out / "sionna"
    commands: list[str] = []
    expected: list[dict[str, str]] = []

    # simulate.py rewrites trials_index.json per invocation (it contains only
    # the trials of that invocation), so ALL planned trials go into ONE call:
    # after it the index covers every planned trial and the idempotency checks
    # stay stable across expansions.
    trial_states = [
        (entry, usable_trial(trials_dir, entry["motion"], entry["perturbation"]))
        for entry in plan.trials
    ]
    if any(existing is None for _, existing in trial_states):
        argv = [str(args.isaac_python), "scripts/humans/simulate.py", "--out", str(trials_dir)]
        for entry in plan.trials:
            argv += ["--trial", f"{entry['motion']}:{entry['perturbation']}"]
        commands.append(shlex.join(argv))
    for entry, existing in trial_states:
        if existing is None:
            # The combined simulate above will produce it; its RT command is
            # emitted by the next expansion once the index covers it.
            expected.append({
                "sample_id": f"{entry['motion']}__{entry['perturbation']}",
                "activity": "fall?", "source": "simulate",
            })
            continue
        trial_json = _inside_repo(existing)
        sample_id = trial_json.stem.removesuffix(".trial")
        expected.append({"sample_id": sample_id, "activity": "fall", "source": str(trial_json)})
        if not rt_done(sionna_dir, sample_id):
            commands.append(rt_command(sample_id, [
                str(args.sionna_python), "scripts/sionna/import_fall_mesh.py",
                *rate_args(), *seed_args(sample_id), "--out", str(sionna_dir),
                "--split", plan.split_label,
                *(["--render-frames"] if plan.render_frames else []),
                "--trial-json", str(trial_json),
            ]))

    selected = set(session_chunk(
        [e.get("name") or Path(e["config"]).stem for e in plan.sessions],
        int(getattr(args, "session_chunk_index", 0)),
        int(getattr(args, "session_chunk_count", 1))))
    for entry in plan.sessions:
        config = _inside_repo(REPO_ROOT / entry["config"])
        name = entry.get("name") or config.stem
        if name not in selected:
            continue
        session_dir = args.out / f"session_{name}"
        export_dir = session_dir.parent / (session_dir.name + "_export")
        if not session_accepted(session_dir / "report.json"):
            commands.append(shlex.join([
                str(args.isaac_python), "scripts/humans/keyboard.py", "--headless",
                "--demo", "--native-mesh", "--config", str(config), "--out", str(session_dir),
            ]))
        if not (export_dir / "manifest.json").is_file():
            commands.append(shlex.join([
                sys.executable, "scripts/humans/export_session_mesh.py",
                "--run", str(session_dir), "--out", str(export_dir),
            ]))
        # Segments are only known after the export; declare the intent and let
        # the summarize pass (or a re-expansion) enumerate the admitted ones.
        expected.append({"sample_id": f"{name}::*", "activity": "adl", "source": str(export_dir)})
        if (export_dir / "manifest.json").is_file():
            # Segment ids are per-session (`stand_00`, `walk_00`, ...), so writing every
            # session's samples into one directory makes them overwrite each other: the
            # first train01 batch produced 9 CIR files from 26 admitted segments. Each
            # session therefore gets its own namespace under sionna/.
            session_out = sionna_dir / name
            for row in admitted_session_sources(export_dir):
                sample_id = row["sample_id"]
                expected.append({
                    "sample_id": f"{name}/{sample_id}", "activity": "adl",
                    "source": str(row["export_dir"] / f"{sample_id}.mesh.npz"),
                })
                if not rt_done(session_out, sample_id):
                    commands.append(rt_command(f"{name}/{sample_id}", [
                        str(args.sionna_python), "scripts/sionna/import_fall_mesh.py",
                        *rate_args(), *seed_args(f"{name}/{sample_id}"),
                        "--out", str(session_out),
                        "--split", plan.split_label,
                        *(["--render-frames"] if plan.render_frames else []),
                        "--dir", str(export_dir), "--sample", sample_id,
                    ], prune=(export_dir / f"{sample_id}.mesh.npz")
                          if plan.prune_mesh_after_import else None))
    return commands, expected


def summarize(sionna_dir: Path) -> dict[str, Any]:
    rows = []
    for report_path in sorted(sionna_dir.glob("**/*.import.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append({
            "sample_id": report.get("channel_sample_id") or report_path.stem,
            "activity": report.get("activity"),
            "source_event_label": report.get("source_event_label"),
            "frames": report.get("frames"),
            "frequency_hz": report.get("frequency_hz"),
            "failures": report.get("failures"),
            "source": (report.get("source") or {}).get("source"),
        })
    return {
        "samples": rows,
        "failures": [row["sample_id"] for row in rows if row["failures"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="batch root directory")
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--sionna-python", type=Path, default=DEFAULT_SIONNA_PYTHON)
    parser.add_argument("--rt-frames", type=int, default=None, help="override the plan")
    parser.add_argument("--render", action="store_true", help="render path figures per sample")
    parser.add_argument("--summarize", action="store_true",
                        help="collect .import.json results into batch_report.json and exit")
    parser.add_argument("--session-chunk-index", type=int, default=0,
                        help="contiguous session chunk to emit (0-based); pair with --count")
    parser.add_argument("--session-chunk-count", type=int, default=1,
                        help="number of session chunks; 1 keeps the whole plan (default)")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    args.rt_frames = args.rt_frames if args.rt_frames is not None else plan.rt_frames
    args.out = _inside_repo(args.out)
    args.plan = _inside_repo(args.plan)
    sionna_dir = args.out / "sionna"
    if args.summarize:
        summary = summarize(sionna_dir)
        summary.update({"plan": str(args.plan), "description": plan.description})
        summary_path = args.out / "batch_report.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"batch summary -> {summary_path} ({len(summary['samples'])} samples, "
              f"{len(summary['failures'])} failures)")
        return 1 if summary["failures"] else 0

    commands, expected = expand(plan, args)
    # The batch root must exist before the script can be written into it: a fresh
    # plan has no stage output to create it, and the first expansion of
    # artifacts/batches/<name> died with FileNotFoundError on run_batch.sh.
    args.out.mkdir(parents=True, exist_ok=True)
    script_path = args.out / "run_batch.sh"
    guard = ""
    if plan.min_free_gb > 0:
        need_kb = int(plan.min_free_gb * 2 ** 20)
        guard = (
            "# Disk guard: 120 Hz mesh costs ~10 MB of recording plus ~20 MB of segment export\n"
            "# per simulated second, and a filesystem that fills mid-run corrupts Isaac's own\n"
            "# scratch, not just this batch. Checked once per pass, before any stage starts.\n"
            "free_kb=$(df -Pk '.' | awk 'NR==2 {print $4}')\n"
            f"if [ \"${{free_kb:-0}}\" -lt {need_kb} ]; then\n"
            f"  echo \"run_batch.sh: ${{free_kb}} KB free, under the {plan.min_free_gb:g} GB "
            "floor; clear space or raise --session-chunk-count\" >&2\n"
            "  exit 1\n"
            "fi\n"
        )
    header = (
        "#!/usr/bin/env bash\n"
        f"# Generated by scripts/sionna/batch_generate.py from {args.plan.name}.\n"
        f"# {plan.description}\n"
        "# Pending stages only; re-run the generator to emit the remainder.\n"
        + ("# Each segment's .mesh.npz is deleted once its own RT trace is accepted; the\n"
           "# meshes come back from the recording directory by re-running the session export.\n"
           if plan.prune_mesh_after_import else "")
        + "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        + guard
    )
    script_path.write_text(header + "\n".join(commands) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    print(f"batch script -> {script_path} ({len(commands)} pending commands, "
          f"{len(expected)} expected samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
