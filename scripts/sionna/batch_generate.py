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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ISAAC_PYTHON = Path.home() / "isaacsim" / "python.sh"
DEFAULT_SIONNA_PYTHON = Path.home() / ".local" / "opt" / "sionna" / "bin" / "python"

# Plan-provided names become command arguments; a strict allowlist keeps the
# emitted script free of anything the shell could reinterpret.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


@dataclass
class BatchPlan:
    description: str = ""
    trials: list[dict[str, str]] = field(default_factory=list)
    sessions: list[dict[str, str]] = field(default_factory=list)
    rt_frames: int = 12
    render_frames: bool = False


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
        render_frames=bool(payload.get("render_frames", False)),
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
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(
        report.get("runtime_completed") is True
        and not report.get("errors")
        and report.get("motion_accuracy_accepted") is True
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
    report = sionna_dir / f"{sample_id}.import.json"
    if not report.is_file():
        return False
    return not json.loads(report.read_text(encoding="utf-8")).get("failures")


def expand(plan: BatchPlan, args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    """Return the pending shell commands plus the expected sample manifest."""

    def rt_command(sample_id: str, argv: list[str]) -> str:
        # Rare GPU scheduling transients (~2 in 9 runs) can fail the strict
        # static-repeat check; a failed sample leaves no import.json, so the
        # next re-expansion re-emits it. The batch must not abort on it.
        return shlex.join(argv) + " || " + shlex.join([
            "echo", f"transient-failure: {sample_id}",
        ]) + f" >> {shlex.quote(str(args.out / 'batch_failures.log'))}"

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
                "--frames", str(args.rt_frames), "--out", str(sionna_dir),
                *(["--render-frames"] if plan.render_frames else []),
                "--trial-json", str(trial_json),
            ]))

    for entry in plan.sessions:
        config = _inside_repo(REPO_ROOT / entry["config"])
        name = entry.get("name") or config.stem
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
            for row in admitted_session_sources(export_dir):
                sample_id = row["sample_id"]
                expected.append({
                    "sample_id": sample_id, "activity": "adl",
                    "source": str(row["export_dir"] / f"{sample_id}.mesh.npz"),
                })
                if not rt_done(sionna_dir, sample_id):
                    commands.append(rt_command(sample_id, [
                        str(args.sionna_python), "scripts/sionna/import_fall_mesh.py",
                        "--frames", str(args.rt_frames), "--out", str(sionna_dir),
                        *(["--render-frames"] if plan.render_frames else []),
                        "--dir", str(export_dir), "--sample", sample_id,
                    ]))
    return commands, expected


def summarize(sionna_dir: Path) -> dict[str, Any]:
    rows = []
    for report_path in sorted(sionna_dir.glob("*.import.json")):
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
    script_path = args.out / "run_batch.sh"
    header = (
        "#!/usr/bin/env bash\n"
        f"# Generated by scripts/sionna/batch_generate.py from {args.plan.name}.\n"
        f"# {plan.description}\n"
        "# Pending stages only; re-run the generator to emit the remainder.\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
    )
    script_path.write_text(header + "\n".join(commands) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    print(f"batch script -> {script_path} ({len(commands)} pending commands, "
          f"{len(expected)} expected samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
