"""Print a one-screen training progress summary for a ProtoMotions run.

Usage (with the ProtoMotions newton venv python, or any python with tensorboard):
    python scripts/protomotions/train_progress.py [results_dir]

Defaults to the running r3 experiment. Reads the newest TensorBoard event file in
``<results_dir>/lightning_logs/*/`` and prints iteration progress (from the log
file when present), learning-curve quintiles, and termination/eval tails.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

DEFAULT_RESULTS = Path.home() / ".local/opt/ProtoMotions/results/smpl_verify_r3"
DEFAULT_LOG = Path("/tmp/protomotions_train_r3.log")


def epoch_from_log(log_path: Path) -> tuple[str, str]:
    """(latest epoch, exit state) parsed from the training log, if readable."""
    epoch, exit_state = "?", "running"
    for candidate in (
        log_path,
        Path("/root/autodl-tmp/train_production.log"),
        Path("/root/train_production.log"),
        Path("/root/autodl-tmp/train_maskedmimic.log"),
    ):
        try:
            text = candidate.read_text(errors="ignore")
        except OSError:
            continue
        epochs = re.findall(r"^Epoch (\d+)", text, re.M)
        if epochs:
            epoch = epochs[-1]
        exits = re.findall(r"^EXIT=(\d+)", text, re.M)
        if exits:
            exit_state = f"FINISHED (EXIT={exits[-1]})"
        return epoch, exit_state
    return epoch, "log not found"


def quintiles(values: list[float], parts: int = 5) -> list[float]:
    n = len(values)
    q = max(1, n // parts)
    return [round(sum(values[i * q : (i + 1) * q]) / q, 2) for i in range(min(parts, n))]


def main() -> int:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    epoch, exit_state = epoch_from_log(DEFAULT_LOG)
    events = sorted(glob.glob(str(results / "lightning_logs/*/events*")))
    if not events:
        print(f"no tensorboard events under {results}")
        return 1
    if epoch != "?":
        print(f"iteration: {epoch} / 10000   [{exit_state}]")

    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(events[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    for tag, label in (
        ("info/episode_length", "episode length (steps, max ~69 = full clip)"),
        ("env/raw_r/gt_rew_mean", "tracking reward (1.0 = perfect follow)"),
        ("info/episode_reward", "total episode reward"),
    ):
        if tag in tags:
            vals = [e.value for e in ea.Scalars(tag)]
            print(f"{label}: {' -> '.join(str(v) for v in quintiles(vals))}")
    interesting = (
        "env/terminate_mean",
        "eval/success_rate",
        "eval/gt_error/failure_rate",
    )
    for tag in sorted(tags):
        if tag.startswith(interesting):
            vals = [e.value for e in ea.Scalars(tag)]
            print(f"{tag}: {vals[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
