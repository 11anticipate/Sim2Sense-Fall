#!/usr/bin/env python3
"""Per-cycle stride symmetry of recorded walks, judged on the axis the body faces.

The limp complaint ("the left leg strides visibly wider than the right") lives
along the *fore-aft* axis of the body frame. The body's facing direction is the
first column of the root rotation matrix (``travel.py`` established the same
convention when it fixed the arm-capture report that mislabelled world x as
forward). Everything here is measured in that frame, per foot, from forward
kinematics of the recorded physics state -- and, separately, of the recorded
joint targets, so a disagreement between the two attributes the offset to
tracking rather than to the reference.

A walking segment is a maximal run of frames whose mode is forward/backward,
whose gait weight is at least ``weight_floor``, with no reset teleport between
neighbours. Within a segment each gait-phase wrap starts a new cycle; only
cycles of at least ``min_cycle_fraction`` of a full phase are scored. Per cycle
the two feet each get a stride (peak-to-peak fore-aft travel) and the cycle gets
a stagger (mean left fore-aft minus mean right fore-aft -- the constant offset
that reads as one leg permanently forward). A partial cycle is not scored: its
feet have not completed their excursions, so its "stagger" is a phase snapshot,
not a limp. The demo's short S/W segments are exactly the case this refuses to
misread.

Read-only: consumes an existing run's ``control.npz`` and ``keyboard.yaml``. No
Isaac Sim.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from common import (  # type: ignore[import-not-found]
    DEFAULT_MOTIONS,
    REPO_ROOT,
    load_inputs,
    write_json,
)

from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints
from sim2sense_fall.humans.rig import (
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)
from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix
from sim2sense_fall.humans.teleop import load_keyboard_config

LOGGER = logging.getLogger("audit_stride_symmetry")

SIDES = ("left", "right")
DIRECTIONS = ("forward", "backward")
# A root jump larger than any legitimate step (the controller caps target lead at
# 0.10 m; one frame at 0.4 m/s is 3.3 mm) is a reset teleport, not walking.
_RESET_STEP_M = 0.05


def load_gate(path: Path) -> dict[str, float]:
    gate = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "max_stagger_m",
        "min_stride_ratio",
        "max_stride_ratio",
        "weight_floor",
        "min_cycle_fraction",
    ):
        if key not in gate or not np.isfinite(gate[key]):
            raise ValueError(f"gate is missing finite {key}")
    return gate


def build_plan(settings: dict[str, Any]) -> Any:
    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    if settings.get("collision_fit", {}).get("enabled", False):
        from sim2sense_fall.humans.contact_control import fit_collision_capsules

        plan, _ = fit_collision_capsules(
            plan, mesh, margin_m=settings["collision_fit"]["margin_m"]
        )
    return plan


def foot_fore_aft(plan: Any, control: Any, column: str) -> np.ndarray:
    """Per frame, per foot: fore-aft of the ankle in the body frame."""

    roots = np.asarray(control["root"], dtype=np.float64)
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    angles = np.asarray(control[column], dtype=np.float64)
    names = list(plan.dof_names)
    out = np.zeros((angles.shape[0], 2), dtype=np.float64)
    for index in range(angles.shape[0]):
        rotation = quaternion_to_matrix(quats[index])
        poses = forward_kinematics(
            plan,
            dict(zip(names, angles[index], strict=True)),
            root_position=roots[index],
            root_rotation=matrix_to_axis_angle(rotation),
        )
        facing = rotation[:, 0]
        for foot, side in enumerate(SIDES):
            out[index, foot] = float(
                (poses[f"{side}_ankle"].translation - roots[index]) @ facing
            )
    return out


def walking_segments(control: Any, weight_floor: float) -> list[tuple[str, np.ndarray]]:
    """Maximal (direction, frame-index) runs of clean walking."""

    mode = np.asarray(control["mode"]).astype(str)
    weight = np.asarray(control["gait_weight"], dtype=np.float64)
    time_s = np.asarray(control["time_s"], dtype=np.float64)
    roots = np.asarray(control["root"], dtype=np.float64)
    keep = np.isin(mode, DIRECTIONS) & (weight >= weight_floor)
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
    segments: list[tuple[str, np.ndarray]] = []
    current: list[int] = []
    current_direction = ""
    for index in range(len(mode)):
        jump = (
            index > 0
            and np.linalg.norm(roots[index, :2] - roots[index - 1, :2]) > _RESET_STEP_M
        )
        gap = index > 0 and time_s[index] - time_s[index - 1] > 2.0 * max(dt, 1e-9)
        direction = str(mode[index]) if keep[index] else ""
        if not direction or jump or gap or (current and direction != current_direction):
            if current:
                segments.append((current_direction, np.asarray(current)))
            current, current_direction = [], direction
        if direction:
            current.append(index)
            current_direction = direction
    if current:
        segments.append((current_direction, np.asarray(current)))
    return segments


def cycle_stats(
    fore_aft: np.ndarray, frames: np.ndarray, phase: np.ndarray, min_fraction: float
) -> list[dict[str, float]]:
    """Split a segment at phase wraps; score only whole cycles.

    Backward playback *decreases* the recorded phase, so a wrap is a raw jump of
    either sign larger than half a turn -- never merely ``diff < 0`` (that is
    every step of a backward walk). Cycle length comes from the unwrapped phase,
    whose signed steps fold every jump back into [-0.5, 0.5).
    """

    if len(frames) < 2:
        return []
    ph = phase[frames]
    raw = np.diff(ph)
    step = (raw + 0.5) % 1.0 - 0.5
    unwrapped = np.concatenate([[0.0], np.cumsum(step)])
    boundaries = [0, *(np.where(np.abs(raw) > 0.5)[0] + 1).tolist(), len(frames)]
    cycles = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if abs(unwrapped[stop - 1] - unwrapped[start]) < min_fraction:
            continue
        chunk = fore_aft[frames[start:stop]]
        mean = chunk.mean(axis=0)
        cycles.append(
            {
                "stride_left_m": float(np.ptp(chunk[:, 0])),
                "stride_right_m": float(np.ptp(chunk[:, 1])),
                "stagger_m": float(mean[0] - mean[1]),
                "frames": int(stop - start),
            }
        )
    return cycles


def audit_run(
    run: Path, gate: dict[str, float], column_actual: str, column_target: str
) -> dict[str, Any]:
    settings = load_keyboard_config(run / "keyboard.yaml", REPO_ROOT)
    plan = build_plan(settings)
    control = np.load(run / "control.npz", allow_pickle=False)
    phase = np.asarray(control["gait_phase"], dtype=np.float64)
    actual = foot_fore_aft(plan, control, column_actual)
    target = foot_fore_aft(plan, control, column_target)

    directions: dict[str, Any] = {}
    for direction in DIRECTIONS:
        cycles: list[dict[str, float]] = []
        target_staggers: list[float] = []
        for name, frames in walking_segments(control, gate["weight_floor"]):
            if name != direction:
                continue
            cycles.extend(cycle_stats(actual, frames, phase, gate["min_cycle_fraction"]))
            target_staggers.extend(
                c["stagger_m"]
                for c in cycle_stats(target, frames, phase, gate["min_cycle_fraction"])
            )
        if not cycles:
            directions[direction] = {"cycles": 0, "verdict": "unmeasured"}
            continue
        left = np.asarray([c["stride_left_m"] for c in cycles])
        right = np.asarray([c["stride_right_m"] for c in cycles])
        stagger = np.asarray([c["stagger_m"] for c in cycles])
        ratio = float(left.mean() / max(right.mean(), 1e-9))
        checks = {
            "stagger_within_gate": bool(
                np.abs(stagger).max() <= gate["max_stagger_m"]
            ),
            "stride_ratio_within_gate": bool(
                gate["min_stride_ratio"] <= ratio <= gate["max_stride_ratio"]
            ),
        }
        directions[direction] = {
            "cycles": len(cycles),
            "stride_left_m": {"mean": float(left.mean()), "std": float(left.std())},
            "stride_right_m": {"mean": float(right.mean()), "std": float(right.std())},
            "stride_ratio_left_over_right": ratio,
            "stagger_m": {
                "mean": float(stagger.mean()),
                "max_abs": float(np.abs(stagger).max()),
                "per_cycle": stagger.tolist(),
            },
            "target_stagger_mean_m": float(np.mean(target_staggers)) if target_staggers else None,
            "tracking_minus_target_stagger_m": (
                float(stagger.mean() - np.mean(target_staggers))
                if target_staggers
                else None
            ),
            "checks": checks,
            "verdict": "pass" if all(checks.values()) else "fail",
        }
    return {"run": str(run), "directions": directions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, action="append", required=True, help="run dir with control.npz"
    )
    parser.add_argument(
        "--gate", type=Path, default=REPO_ROOT / "configs/humans/stride_symmetry_gate.yaml"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    gate = load_gate(args.gate)
    results = [audit_run(run, gate, "joints", "joint_target") for run in args.run]
    for report in results:
        print(f"\n=== {report['run']} ===")
        for direction, stats in report["directions"].items():
            if stats.get("cycles", 0) == 0:
                print(f"  {direction:9s} no whole cycle recorded -- unmeasured")
                continue
            left = stats["stride_left_m"]
            right = stats["stride_right_m"]
            stagger = stats["stagger_m"]
            print(
                f"  {direction:9s} {stats['cycles']:2d} cycles  stride L "
                f"{left['mean'] * 1000:6.1f}±{left['std'] * 1000:4.1f} mm  R "
                f"{right['mean'] * 1000:6.1f}±{right['std'] * 1000:4.1f} mm  "
                f"ratio {stats['stride_ratio_left_over_right']:.3f}"
            )
            print(
                f"            stagger mean {stagger['mean'] * 1000:+6.1f} mm, "
                f"worst |{stagger['max_abs'] * 1000:.1f}| mm (gate "
                f"{gate['max_stagger_m'] * 1000:.0f} mm)  target "
                f"{(stats['target_stagger_mean_m'] or 0.0) * 1000:+6.1f} mm"
            )
            print(f"            verdict: {stats['verdict']}  {stats['checks']}")
    if args.out is not None:
        write_json(args.out, results)
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
