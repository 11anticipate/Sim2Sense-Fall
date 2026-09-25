#!/usr/bin/env python3
"""Screen the local AMASS library for walk sequences with addressable arm swing.

Why this exists
---------------
P0-A item 4 needed a fix for the right-arm complaint. The first candidate
(strategy 1: re-select a window inside the shipped clip,
``walkbackwards_stand_poses.npz``) was measured to be impossible: a 0.05 s sweep
of all 137 windows found **zero** that clear the pre-registered gate, because the
clip's clean-antiphase stretches also have elbow right/left ratios of 0.21-0.55
and speeds that collapse as the subject decelerates into a stand. See
``docs/arm-swing-audit.md``.

This script attacks strategy 2 (choose a different source sequence). It walks the
AMASS library and, for every sequence that plausibly contains sustained walking,
cuts fixed-length windows and measures the same quantities as
``arm_fix_window_search.py``. The output is a ranked shortlist; nothing is
retargeted on the GPU and no Isaac Sim import is touched.

Pre-filter (cheap, no retarget): a window is *sustained walking* only if
  * the root travels a minimum distance over the window, and
  * the same-side knee trace has at least ``min_cycles`` swing cycles.
That removes stand/sit/transition sequences before the expensive test, and it
removes them on a criterion that does not involve the arms at all -- so the arm
numbers it later reports cannot be an artefact of the pre-filter.

Because the AMASS ``.npz`` files carry no activity label and the CMU stems are
numeric (``32_01_poses``), walking is detected by **measurement**, not by name:
``--tokens`` remains available for libraries whose filenames do describe the
motion (the bundled Transitions_mocap subset), while ``CMU`` is screened whole.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from arm_fix_window_search import (
    DEFAULT_GATE,
    Gate,
    _antiphase,
    _chain_index,
    _crossing_count,
    _load_plan,
    _ratio,
    _sample_target,
    _wrap_ratio,
    load_gate,
)
from common import REPO_ROOT, write_json  # type: ignore[import-not-found]

from sim2sense_fall.humans.amass import load_amass_clip  # noqa: E402
from sim2sense_fall.humans.teleop import load_gait  # noqa: E402

LOGGER = logging.getLogger("arm_source_screen")

# Substring filters on the AMASS stem, used only where filenames describe the
# motion. Passing an empty list screens every file in the given subdirectories.
WALK_TOKENS = ("walk", "walking", "locomotion")


def parse_list(text: str) -> tuple[str, ...]:
    return tuple(chunk.strip() for chunk in text.split(",") if chunk.strip())


def discover(root: Path, subdirs: Iterable[str], tokens: Iterable[str]) -> list[Path]:
    """Collect candidate ``.npz`` paths, optionally narrowed by stem substring."""

    found: list[Path] = []
    for subdir in subdirs:
        base = root / subdir if subdir else root
        if not base.is_dir():
            raise FileNotFoundError(f"AMASS subdirectory does not exist: {base}")
        for path in sorted(base.rglob("*.npz")):
            if tokens and not any(token in path.stem.lower() for token in tokens):
                continue
            found.append(path)
    if not found:
        raise FileNotFoundError(
            f"no candidate sequences below {root} in {list(subdirs)} with tokens {list(tokens)}"
        )
    return found


def _pedestrian_prefilter(clip: Any, start_s: float, duration_s: float, gate: Gate) -> bool:
    """Cheap 'is this person actually walking' test that never looks at the arms.

    Uses the root's horizontal travel and the knee's swing-cycle count, both of
    which are independent of the arm channels whose behaviour is being screened.
    """

    times = np.asarray(clip.times_s)
    inside = (times >= start_s) & (times <= start_s + duration_s)
    if int(inside.sum()) < 8:
        return False
    translation = np.asarray(clip.root_translation)[inside]
    travel = float(np.linalg.norm(translation[-1, :2] - translation[0, :2]))
    speed = travel / duration_s
    if speed < gate.min_speed_m_s or speed > gate.max_speed_m_s:
        return False
    if travel < 0.35:
        return False
    # Knee flexion channel: a coarse cycle count rejects stand/sit/transition
    # sequences. The knee rotation is read through the clip's own joint index so
    # the check does not depend on the AMASS pose slot ordering.
    try:
        knee = clip.joint_index("left_knee")
    except KeyError:
        return True
    rotations = np.asarray(clip.joint_rotations)[inside]
    angle = np.linalg.norm(rotations[:, knee], axis=1)
    return _crossing_count(angle) >= gate.min_cycles_per_window


def measure_candidate(
    path: Path,
    plan: Any,
    *,
    dt_s: float,
    samples: int,
    start_s: float,
    duration_s: float,
    gate: Gate,
) -> dict[str, Any] | None:
    dof_names = [joint.name for joint in plan.joints]
    indices = {
        "shoulder_left": _chain_index(dof_names, plan, "left_shoulder", "x"),
        "shoulder_right": _chain_index(dof_names, plan, "right_shoulder", "x"),
        "elbow_left": _chain_index(dof_names, plan, "left_elbow", "x"),
        "elbow_right": _chain_index(dof_names, plan, "right_elbow", "x"),
        "knee_left": _chain_index(dof_names, plan, "left_knee", "x"),
        "knee_right": _chain_index(dof_names, plan, "right_knee", "x"),
        "wrist_left": _chain_index(dof_names, plan, "left_wrist", "x"),
        "wrist_right": _chain_index(dof_names, plan, "right_wrist", "x"),
    }
    if any(value is None for value in indices.values()):
        raise RuntimeError(f"rig lacks an x-axis DOF for {indices}")

    try:
        gait = load_gait(
            {"file": path, "start_s": start_s, "duration_s": duration_s}, plan, dt_s=dt_s
        )
    except ValueError:
        return None

    sampled, _ = _sample_target(gait, samples)
    shoulder = _ratio(sampled, indices["shoulder_left"], indices["shoulder_right"])
    elbow = _ratio(sampled, indices["elbow_left"], indices["elbow_right"])
    knee = _ratio(sampled, indices["knee_left"], indices["knee_right"])
    antiphase = _antiphase(sampled, indices["shoulder_left"], indices["shoulder_right"])
    wrap = _wrap_ratio(gait, samples)
    cycles = min(
        _crossing_count(sampled[:, indices["shoulder_left"]]),
        _crossing_count(sampled[:, indices["knee_left"]]),
    )

    failures = []
    if not np.isfinite(shoulder) or shoulder < gate.min_shoulder_ratio:
        failures.append(f"shoulder_ratio {shoulder:.4f} < {gate.min_shoulder_ratio}")
    if not np.isfinite(elbow) or elbow < gate.min_elbow_ratio:
        failures.append(f"elbow_ratio {elbow:.4f} < {gate.min_elbow_ratio}")
    if not np.isfinite(knee) or knee < gate.min_leg_ratio:
        failures.append(f"leg_ratio {knee:.4f} < {gate.min_leg_ratio}")
    if not gate.min_speed_m_s <= gait.speed_m_s <= gate.max_speed_m_s:
        failures.append(
            f"speed {gait.speed_m_s:.3f} outside [{gate.min_speed_m_s}, {gate.max_speed_m_s}]"
        )
    if not np.isfinite(antiphase) or antiphase > gate.max_antiphase_correlation:
        failures.append(f"antiphase {antiphase:.4f} > {gate.max_antiphase_correlation}")
    if cycles < gate.min_cycles_per_window:
        failures.append(f"cycles {cycles} < {gate.min_cycles_per_window}")
    if wrap > gate.max_wrap_over_interior:
        failures.append(f"wrap_ratio {wrap:.4f} > {gate.max_wrap_over_interior}")

    return {
        "file": str(path),
        "clip_id": f"amass__{path.stem}",
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "speed_m_s": float(gait.speed_m_s),
        "shoulder_ratio": float(shoulder),
        "elbow_ratio": float(elbow),
        "knee_ratio": float(knee),
        "wrist_ratio": _ratio(sampled, indices["wrist_left"], indices["wrist_right"]),
        "antiphase_correlation": float(antiphase),
        "cycles": cycles,
        "wrap_over_interior": float(wrap),
        "min_ratio": float(min(shoulder, elbow, knee)),
        "passes": not failures,
        "failures": failures,
    }


def _screen_sequence(task: tuple[str, str, str, float, float, int]) -> tuple[
    list[dict[str, Any]], int, int
]:
    """Worker: screen every candidate window of one sequence.

    Returns ``(survivors, windows_that_passed_the_prefilter, unreadable_flag)``.
    Rebuilding the plan per call is deliberate -- see the note in ``main``.
    """

    path_text, config_text, gate_text, window_s, stride_s, samples = task
    _, config, plan = _load_plan(Path(config_text))
    gate = load_gate(Path(gate_text))
    path = Path(path_text)
    try:
        clip = load_amass_clip(path)
    except (OSError, ValueError):
        return [], 0, 1
    clip_duration = float(clip.times_s[-1])
    if clip_duration < window_s:
        return [], 0, 0
    survivors: list[dict[str, Any]] = []
    screened = 0
    for start in np.arange(0.0, clip_duration - window_s + 1e-9, stride_s):
        if not _pedestrian_prefilter(clip, float(start), window_s, gate):
            continue
        screened += 1
        row = measure_candidate(
            path,
            plan,
            dt_s=config.simulation.physics_dt_s,
            samples=samples,
            start_s=float(start),
            duration_s=window_s,
            gate=gate,
        )
        if row is not None:
            survivors.append(row)
    return survivors, screened, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/humans/amass_raw")
    parser.add_argument(
        "--subdirs",
        type=parse_list,
        default=("CMU",),
        help="comma-separated subdirectories under --root",
    )
    parser.add_argument(
        "--tokens",
        type=parse_list,
        default=(),
        help="comma-separated stem substrings to keep; empty = screen every file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts/humans/arm_source_screen",
    )
    parser.add_argument("--window-s", type=float, default=1.2)
    parser.add_argument("--stride-s", type=float, default=0.4)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0, help="0 = every discovered file")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4)),
        help="process-pool size; the per-window retarget is pure CPU",
    )
    args = parser.parse_args(argv)
    if args.window_s <= 0 or args.stride_s <= 0:
        parser.error("window-s and stride-s must be positive")
    if args.samples < 24:
        parser.error("samples must be at least 24")
    if args.workers < 1:
        parser.error("workers must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings, config, plan = _load_plan(args.config)
    gate = load_gate(args.gate)
    paths = discover(args.root, args.subdirs, args.tokens)
    if args.limit:
        paths = paths[: args.limit]
    LOGGER.info(
        "screening %d sequences, %.2fs windows at %.2fs stride, %d workers",
        len(paths),
        args.window_s,
        args.stride_s,
        args.workers,
    )

    # Loading a clip and retargeting each window is pure CPU and independent per
    # sequence, so the sweep is spread over a process pool. The plan is rebuilt
    # inside each worker rather than pickled: ``HumanRigPlan`` holds numpy arrays
    # and the rebuild is cheap relative to a sequence's retargeting cost.
    tasks = [
        (
            str(path),
            str(args.config),
            str(args.gate),
            args.window_s,
            args.stride_s,
            args.samples,
        )
        for path in paths
    ]
    rows: list[dict[str, Any]] = []
    prefilters = 0
    unreadable = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (survivors, screened, failures) in enumerate(
            pool.map(_screen_sequence, tasks, chunksize=4), start=1
        ):
            prefilters += screened
            unreadable += failures
            rows.extend(survivors)
            if done % 100 == 0:
                LOGGER.info("%d/%d sequences, %d candidates so far", done, len(paths), len(rows))

    if not rows:
        LOGGER.error("no candidate window survived the pre-filter (unreadable %d)", unreadable)
        return 1

    passing = [row for row in rows if row["passes"]]
    passing.sort(key=lambda row: (-row["min_ratio"], -row["speed_m_s"]))
    near = sorted(rows, key=lambda row: -row["min_ratio"])[: args.top]
    report = {
        "root": str(args.root),
        "subdirs": list(args.subdirs),
        "tokens": list(args.tokens),
        "config": str(args.config),
        "gate": asdict(gate),
        "gate_sha256": hashlib.sha256(Path(args.gate).read_bytes()).hexdigest(),
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "samples_per_window": args.samples,
        "sequences_discovered": len(paths),
        "sequences_unreadable": unreadable,
        "candidates_prefiltered": prefilters,
        "candidates_measured": len(rows),
        "passing": len(passing),
        "top_passing": passing[: args.top],
        "top_by_min_ratio": near,
        "selection_rule": (
            "maximise min(shoulder,elbow,knee) right/left ratio, then speed; "
            "a passing row must clear every pre-registered gate"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "arm_source_screen.json", report)
    LOGGER.info(
        "%d/%d measured windows pass; best min-ratio %.4f (%s %.2f-%.2fs)",
        len(passing),
        len(rows),
        near[0]["min_ratio"],
        Path(near[0]["file"]).stem,
        near[0]["start_s"],
        near[0]["start_s"] + near[0]["duration_s"],
    )
    LOGGER.info("wrote %s", args.out / "arm_source_screen.json")
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
