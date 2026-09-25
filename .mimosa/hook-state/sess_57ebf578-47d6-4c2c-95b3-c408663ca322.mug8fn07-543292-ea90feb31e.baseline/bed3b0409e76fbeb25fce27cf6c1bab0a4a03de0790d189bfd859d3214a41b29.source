#!/usr/bin/env python3
"""Search the shipped gait source clips for a naturally symmetric arm window.

``docs/arm-swing-audit.md`` located the P0-A right-arm complaint in the source
AMASS window itself: the shipped forward window sits at the 11th percentile of
right/left shoulder fore-aft amplitude across the whole sequence (ratio 0.373),
while the pipeline reproduces the reference to better than 0.1 deg.  The fix
option chosen here is therefore *re-selection of the source window*, not a
symmetric-correction patch: if a window exists in the same clip that tracks the
expected human arm swing, using it leaves the retarget/control stack untouched
and keeps the motion a genuine recorded walk.

Selection is driven by a **pre-registered** criterion.  The thresholds live in
``configs/humans/arm_symmetry_gate.yaml``, which is committed before the search
runs, so the window cannot be picked to fit whatever the search returns.

Measured per window, all on the retargeted control target (what the controller
actually sends, so retargeting fidelity is included):

* ``shoulder__x`` right/left peak-to-peak  -- ``min_shoulder_ratio``
* ``elbow__x`` right/left peak-to-peak      -- ``min_elbow_ratio``
* shoulder ``x`` = the fore-aft swing axis, confirmed by ``forward_kinematics``
* leg ``x`` right/left ratio                -- ``leg_ratio_floor`` (a window whose
  legs have also collapsed is not a walk, it is a stand)
* source-clip displacement speed            -- ``min_speed_m_s`` / ``max_speed_m_s``
* left/right shoulder antiphase correlation -- ``max_antiphase_correlation``

Enumerating one window per ``stride_s`` over the whole clip is O(frames) and CPU
only; the Isaac Sim import is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from common import REPO_ROOT, write_json  # type: ignore[import-not-found]

from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints  # noqa: E402
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig  # noqa: E402
from sim2sense_fall.humans.teleop import Gait, load_gait, load_keyboard_config  # noqa: E402

LOGGER = logging.getLogger("arm_fix_window_search")

DEFAULT_GATE = REPO_ROOT / "configs/humans/arm_symmetry_gate.yaml"


@dataclass(frozen=True)
class Gate:
    """Pre-registered acceptance thresholds for a replacement window."""

    min_shoulder_ratio: float
    min_elbow_ratio: float
    min_leg_ratio: float
    min_speed_m_s: float
    max_speed_m_s: float
    max_antiphase_correlation: float
    min_cycles_per_window: int
    max_wrap_over_interior: float

    def __post_init__(self) -> None:
        for name in ("min_shoulder_ratio", "min_elbow_ratio", "min_leg_ratio"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1], got {value}")
        if self.min_speed_m_s <= 0 or self.max_speed_m_s <= self.min_speed_m_s:
            raise ValueError("speed bounds must satisfy 0 < min < max")
        if not -1.0 <= self.max_antiphase_correlation < 0.0:
            raise ValueError("antiphase correlation must be in [-1, 0)")
        if self.min_cycles_per_window < 1:
            raise ValueError("min_cycles_per_window must be >= 1")
        if self.max_wrap_over_interior <= 0:
            raise ValueError("max_wrap_over_interior must be positive")


def load_gate(path: Path) -> Gate:
    """Read the pre-registered gate, rejecting unknown keys."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "gate" not in raw:
        raise ValueError(f"{path} must contain a top-level 'gate' mapping")
    payload = raw["gate"]
    known = set(Gate.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{path} has unknown gate keys: {sorted(unknown)}")
    return Gate(**payload)


def _load_plan(config_path: Path) -> tuple[dict[str, Any], Any, Any]:
    settings = load_keyboard_config(config_path, REPO_ROOT)
    from common import DEFAULT_MOTIONS, load_inputs  # type: ignore[import-not-found]

    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    return settings, config, plan


def _chain_index(dof_names: list[str], plan: Any, joint: str, axis: str) -> int | None:
    """Index of the DOF inside ``joint``'s 3-revolute chain that rotates about ``axis``.

    Chain membership is decided by the ``__`` prefix only; the axes come from the
    plan spec, because the ``__dof1``/``__dof2`` suffixes are *not* axis names.
    """

    chain = [index for index, dof in enumerate(dof_names) if dof.split("__")[0] == joint]
    for index in chain:
        if plan.joints[index].axis == axis:
            return index
    return None


def _sample_target(gait: Gait, samples: int) -> tuple[np.ndarray, np.ndarray]:
    phases = np.linspace(0.0, 1.0, samples, endpoint=False)
    stack = [gait.sample(float(p)) for p in phases]
    return np.stack([row[0] for row in stack]), np.asarray([row[1] for row in stack])


def _ratio(sampled: np.ndarray, left: int, right: int) -> float:
    span_left = float(np.ptp(sampled[:, left]))
    span_right = float(np.ptp(sampled[:, right]))
    if span_left <= 1e-9:
        return float("nan")
    return span_right / span_left


def _antiphase(sampled: np.ndarray, left: int, right: int) -> float:
    a = sampled[:, left] - sampled[:, left].mean()
    b = sampled[:, right] - sampled[:, right].mean()
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / scale)


def _wrap_ratio(gait: Gait, samples: int) -> float:
    sampled, _ = _sample_target(gait, samples)
    steps = np.abs(np.diff(sampled, axis=0)).max(axis=0)
    wrap = np.abs(sampled[0] - sampled[-1])
    ratio = np.divide(wrap, np.maximum(steps, 1e-12), out=np.zeros_like(wrap), where=steps > 1e-12)
    return float(ratio.max())


def _crossing_count(trace: np.ndarray) -> int:
    """Gait cycles estimated from the number of mean crossings of a joint trace."""

    if len(trace) < 3 or not np.isfinite(trace).all():
        return 0
    centred = trace - trace.mean()
    return int(np.count_nonzero(np.diff(np.sign(centred)) != 0)) // 2


def measure_window(
    name: str,
    spec: dict[str, Any],
    plan: Any,
    *,
    dt_s: float,
    samples: int,
    start_s: float,
    duration_s: float,
    gate: Gate,
) -> dict[str, Any] | None:
    """Retarget one candidate window and measure it; ``None`` if unrepresentable."""

    dof_names = [joint.name for joint in plan.joints]
    indices = {
        "shoulder_left": _chain_index(dof_names, plan, "left_shoulder", "x"),
        "shoulder_right": _chain_index(dof_names, plan, "right_shoulder", "x"),
        "elbow_left": _chain_index(dof_names, plan, "left_elbow", "x"),
        "elbow_right": _chain_index(dof_names, plan, "right_elbow", "x"),
        "knee_left": _chain_index(dof_names, plan, "left_knee", "x"),
        "knee_right": _chain_index(dof_names, plan, "right_knee", "x"),
    }
    missing = [key for key, value in indices.items() if value is None]
    if missing:
        raise RuntimeError(f"rig exposes no x-axis DOF for {missing}")

    try:
        gait = load_gait({**spec, "start_s": start_s, "duration_s": duration_s}, plan, dt_s=dt_s)
    except ValueError as error:
        return {"start_s": start_s, "duration_s": duration_s, "rejected": f"retarget: {error}"}

    sampled, _ = _sample_target(gait, samples)
    shoulder = _ratio(sampled, indices["shoulder_left"], indices["shoulder_right"])
    elbow = _ratio(sampled, indices["elbow_left"], indices["elbow_right"])
    knee = _ratio(sampled, indices["knee_left"], indices["knee_right"])
    antiphase = _antiphase(sampled, indices["shoulder_left"], indices["shoulder_right"])
    wrap = _wrap_ratio(gait, samples)
    cycles = _crossing_count(sampled[:, indices["shoulder_left"]])

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

    asymmetric_score = 1.0 - min(value for value in (shoulder, elbow, knee) if np.isfinite(value))
    return {
        "gait": name,
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "end_s": float(start_s + duration_s),
        "frames": int(gait.joints.shape[0]),
        "speed_m_s": float(gait.speed_m_s),
        "shoulder_ratio": float(shoulder),
        "elbow_ratio": float(elbow),
        "knee_ratio": float(knee),
        "antiphase_correlation": float(antiphase),
        "cycles": cycles,
        "wrap_over_interior": float(wrap),
        "limit_margin_min_deg": float(
            min(
                np.rad2deg(
                    min(
                        np.deg2rad(plan.joints[index].upper_deg) - sampled[:, index].max(),
                        sampled[:, index].min() - np.deg2rad(plan.joints[index].lower_deg),
                    )
                )
                for index in indices.values()
            )
        ),
        "asymmetry_score": float(asymmetric_score),
        "passes": not failures,
        "failures": failures,
    }


def _parse_duration(value: str) -> float:
    duration = float(value)
    if duration <= 0:
        raise argparse.ArgumentTypeError("window duration must be positive")
    return duration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/humans/arm_fix_search")
    parser.add_argument(
        "--gait",
        default="forward",
        help="gait key to re-select (forward/backward)",
    )
    parser.add_argument("--window-s", type=_parse_duration, default=None)
    parser.add_argument("--stride-s", type=float, default=0.15)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args(argv)
    if args.stride_s <= 0:
        parser.error("stride-s must be positive")
    if args.samples < 24:
        parser.error("samples must be at least 24")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings, config, plan = _load_plan(args.config)
    gate = load_gate(args.gate)
    if args.gait not in settings["gaits"]:
        parser.error(f"unknown gait {args.gait!r}; have {sorted(settings['gaits'])}")
    base = settings["gaits"][args.gait]
    window_s = args.window_s if args.window_s is not None else float(base["duration_s"])

    from sim2sense_fall.humans.amass import load_amass_clip  # noqa: E402

    clip = load_amass_clip(Path(base["file"]))
    clip_duration = float(clip.times_s[-1])
    if window_s > clip_duration:
        parser.error(f"window {window_s}s exceeds clip duration {clip_duration:.3f}s")

    starts = np.arange(0.0, clip_duration - window_s + 1e-9, args.stride_s)
    LOGGER.info(
        "scanning %s: %.3fs clip, %.3fs window, %d starts, stride %.3fs",
        args.gait,
        clip_duration,
        window_s,
        len(starts),
        args.stride_s,
    )
    rows = []
    for start in starts:
        row = measure_window(
            args.gait,
            base,
            plan,
            dt_s=config.simulation.physics_dt_s,
            samples=args.samples,
            start_s=float(start),
            duration_s=window_s,
            gate=gate,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        LOGGER.error("no candidate window was representable on this rig")
        return 1

    passing = [row for row in rows if row["passes"]]
    passing.sort(key=lambda row: (row["asymmetry_score"], -row["speed_m_s"]))
    rejected = [row for row in rows if not row["passes"]]

    shipped = {
        "start_s": float(base["start_s"]),
        "duration_s": float(base["duration_s"]),
        "end_s": float(base["start_s"] + base["duration_s"]),
    }
    shipped_row = next(
        (row for row in rows if abs(row["start_s"] - shipped["start_s"]) < 1e-6), None
    )

    selected = passing[0] if passing else None
    report = {
        "config": str(args.config),
        "gate_path": str(args.gate),
        "gate": asdict(gate),
        "gate_sha256": hashlib.sha256(Path(args.gate).read_bytes()).hexdigest(),
        "clip": {
            "file": str(base["file"]),
            "duration_s": clip_duration,
            "sha256": hashlib.sha256(Path(base["file"]).read_bytes()).hexdigest(),
        },
        "gait": args.gait,
        "window_s": window_s,
        "stride_s": args.stride_s,
        "samples_per_window": args.samples,
        "candidates": len(rows),
        "passing": len(passing),
        "shipped_window": shipped,
        "shipped_measurement": shipped_row,
        "selected": selected,
        "selection_rule": (
            "among passing windows minimise (1 - min(shoulder,elbow,knee) ratio); "
            "ties broken by higher measured clip speed"
        ),
        "top_candidates": passing[: args.top],
        "rejected_examples": rejected[: min(args.top, len(rejected))],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"{args.gait}_window_search.json", report)
    if shipped_row is not None:
        LOGGER.info(
            "shipped %.3f-%.3fs: shoulder %.4f elbow %.4f knee %.4f passes=%s",
            shipped["start_s"],
            shipped["end_s"],
            shipped_row["shoulder_ratio"],
            shipped_row["elbow_ratio"],
            shipped_row["knee_ratio"],
            shipped_row["passes"],
        )
    if selected is None:
        LOGGER.error("no window passed the pre-registered gate (%d candidates)", len(rows))
        return 2
    LOGGER.info(
        "selected %.3f-%.3fs: shoulder %.4f elbow %.4f knee %.4f speed %.3fs passes=%s",
        selected["start_s"],
        selected["end_s"],
        selected["shoulder_ratio"],
        selected["elbow_ratio"],
        selected["knee_ratio"],
        selected["speed_m_s"],
        selected["passes"],
    )
    LOGGER.info("wrote %s", args.out / f"{args.gait}_window_search.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
