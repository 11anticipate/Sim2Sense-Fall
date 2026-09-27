#!/usr/bin/env python3
"""A/B/C harness for the fall collapse mechanism (user decision 2026-09-26).

Why this exists
---------------
The user judged the drives-off fall "like a statue". The remaining drive torque
during the collapse is the kept viscous term: ``fall_damping_scale`` 0.15
rescales the authored joint damping (60-150 Nm s/rad) to 9-22.5 Nm s/rad, and
with velocity targets forced to zero that term is a pure speed brake
(``damping * (0 - qd)``) of 150-390 Nm at the ballistic joint speeds a fall
reaches. Zeroing it reopens a measured solver-NaN regime (2864 deg/s mid-fall
NaN'd the PhysX solver on floor impact), which this harness closes instead with
a per-joint ``maxJointVelocity`` clamp authored on the rig.

Variants (one Isaac session each, two identical falls per session)
------------------------------------------------------------------
* ``A_damped015`` -- shipped baseline: damping kept at 0.15, no velocity clamp.
* ``B_damped005`` -- damping at the old action-config floor 0.05, no clamp.
* ``C_release0``  -- fully passive limbs (damping 0) + 20 rad/s velocity clamp.

The three configs are generated from ``configs/humans/keyboard.yaml`` so every
other setting is bit-identical; the harness records their hashes. Analysis
(``--analyze``, CPU) reads the session products: fall event timing, per-fall
joint-speed peaks (the NaN regime signature), knee/hip fold angles, collapse
duration, non-finite detection, and -- for C -- that the saved stage really
carries the clamp on every revolute joint (a text check on the .usda, valid
because the physics run itself is the positive control: without the clamp the
zero-damping regime NaNs).

Example
-------
    ~/isaacsim/python.sh scripts/humans/fall_release_ab.py --dry-run
    ~/isaacsim/python.sh scripts/humans/fall_release_ab.py --run A_damped015
    ~/isaacsim/python.sh scripts/humans/fall_release_ab.py --run B_damped005
    ~/isaacsim/python.sh scripts/humans/fall_release_ab.py --run C_release0
    python3 scripts/humans/fall_release_ab.py --analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from common import REPO_ROOT, write_json  # type: ignore[import-not-found]

LOGGER = logging.getLogger("fall_release_ab")

OUT_ROOT = REPO_ROOT / "artifacts/humans/fall_release_ab"
BASE_CONFIG = REPO_ROOT / "configs/humans/keyboard.yaml"

# Orthogonal single-factor variants; every other line comes from the shipped
# keyboard config. damping_scale 0 requires the clamp (actions.py contract).
# Round 2 (user feedback: C reads as "a puddle of water", a body without joints;
# real passive joint viscosity sits between 0 and the statue regime): finer
# sweep with the clamp always on. Free-fall reference: the torso drops 0.9 m
# in ~0.43 s; C(0) collapsed to 60% height in 0.30 s = essentially free fall,
# A/B(0.05-0.15) took 1.5 s = braced; a real human fall is ~0.6-1.0 s.
VARIANTS: dict[str, dict[str, Any]] = {
    "A_damped015": {"fall_damping_scale": 0.15, "joint_velocity_limit_rad_s": None},
    "B_damped005": {"fall_damping_scale": 0.05, "joint_velocity_limit_rad_s": None},
    "C_release0": {"fall_damping_scale": 0.0, "joint_velocity_limit_rad_s": 20.0},
    "D_damped002": {"fall_damping_scale": 0.02, "joint_velocity_limit_rad_s": 20.0},
    "E_damped001": {"fall_damping_scale": 0.01, "joint_velocity_limit_rad_s": 20.0},
    "F_damped004": {"fall_damping_scale": 0.04, "joint_velocity_limit_rad_s": 20.0},
    "G_damped0005": {"fall_damping_scale": 0.005, "joint_velocity_limit_rad_s": 20.0},
}

# Two identical falls per session. The post-trigger evolution is split so the
# --capture screenshots land at ~0.6/1.1/2.1/3.1/5.1/8.1 s after the trigger,
# which is where the collapse character (statue crumple vs passive fold) shows.
FALL_TIMELINE: list[dict[str, Any]] = [
    {"duration_s": 2.0, "keys": []},
    {"duration_s": 0.1, "keys": ["F"]},
    {"duration_s": 0.5, "keys": []},
    {"duration_s": 0.5, "keys": []},
    {"duration_s": 1.0, "keys": []},
    {"duration_s": 1.0, "keys": []},
    {"duration_s": 2.0, "keys": []},
    {"duration_s": 3.0, "keys": []},
    {"duration_s": 0.1, "keys": ["R"]},
    {"duration_s": 2.0, "keys": []},
    {"duration_s": 0.1, "keys": ["F"]},
    {"duration_s": 0.5, "keys": []},
    {"duration_s": 0.5, "keys": []},
    {"duration_s": 1.0, "keys": []},
    {"duration_s": 1.0, "keys": []},
    {"duration_s": 2.0, "keys": []},
    {"duration_s": 3.0, "keys": []},
]

# The joint axes the statue complaint is about; fold = |angle_after - angle_at|
# in plan DOF order, taken at trigger vs 2 s later (settled on the floor).
FOLD_JOINTS = (
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
)


def _variant_config(variant: str) -> tuple[Path, dict[str, Any]]:
    """Write one per-variant keyboard config and return its path and hash."""

    import yaml

    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base keyboard config must be a mapping")
    overrides = VARIANTS[variant]
    actions = payload.setdefault("actions", {})
    actions["fall_damping_scale"] = overrides["fall_damping_scale"]
    if overrides["joint_velocity_limit_rad_s"] is None:
        payload.pop("joint_velocity_limit_rad_s", None)
    else:
        payload["joint_velocity_limit_rad_s"] = overrides["joint_velocity_limit_rad_s"]
    payload["demo"] = FALL_TIMELINE
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = {
        "variant": variant,
        "overrides": overrides,
        "config_path": str(config_path),
        "config_sha256": digest,
        "demo_duration_s": float(sum(s["duration_s"] for s in FALL_TIMELINE)),
    }
    write_json(out_dir / "variant.json", record)
    return config_path, record


def _run_variant(variant: str) -> int:
    """Reuse scripts/humans/keyboard.py's exact entry point for one variant."""

    config_path, record = _variant_config(variant)
    LOGGER.info("variant %s: config %s (sha256 %s)", variant, config_path, record["config_sha256"])
    from keyboard import main as keyboard_main

    session_out = OUT_ROOT / variant
    return keyboard_main(
        [
            "--config",
            str(config_path),
            "--out",
            str(session_out),
            "--demo",
            "--headless",
            "--capture",
        ]
    )


def _stage_clamp_check(session_dir: Path, expected: float | None) -> dict[str, Any]:
    """Text check on the saved .usda: clamp present on every revolute joint."""

    stage_path = session_dir / "human_keyboard.usda"
    if not stage_path.is_file():
        return {"checked": False, "reason": "stage file missing"}
    text = stage_path.read_text(encoding="utf-8", errors="replace")
    values: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "physxJoint:maxJointVelocity" in stripped and "=" in stripped:
            values.append(float(stripped.rsplit("=", 1)[1].strip().rstrip(",")))
    if expected is None:
        return {"checked": True, "expected": None, "authored": len(values), "ok": not values}
    # The USD attribute carries USD's angular unit (deg/s); the config is rad/s.
    expected_deg = float(np.rad2deg(expected))
    ok = len(values) > 0 and bool(np.allclose(values, expected_deg))
    return {
        "checked": True,
        "expected_rad_s": expected,
        "expected_deg_s": expected_deg,
        "authored": len(values),
        "unique_values": sorted(set(values)),
        "ok": ok,
    }


def _fall_segments(mode: np.ndarray, time_s: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous falling+fallen spans, as [start, end] row indices."""

    in_fall = np.isin(mode, ["falling", "fallen"])
    edges = np.diff(in_fall.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if in_fall[0]:
        starts.insert(0, 0)
    if in_fall[-1]:
        ends.append(len(in_fall))
    return [(s, e) for s, e in zip(starts, ends, strict=False) if e > s + 1]


def _analyze_variant(variant: str) -> dict[str, Any]:
    session_dir = OUT_ROOT / variant
    if not all(
        (session_dir / name).is_file()
        for name in ("variant.json", "report.json", "control.npz")
    ):
        return {"variant": variant, "status": "session_not_run"}
    record = json.loads((session_dir / "variant.json").read_text(encoding="utf-8"))
    report = json.loads((session_dir / "report.json").read_text(encoding="utf-8"))
    control = np.load(session_dir / "control.npz", allow_pickle=False)
    dof_names = [str(n) for n in control["dof_names"]]
    mode = control["mode"].astype(str)
    time_s = control["time_s"]
    joints = control["joints"]
    speeds = control["joint_speeds"]
    root = control["root"]
    clamp = record["overrides"]["joint_velocity_limit_rad_s"]

    standing_height = float(np.max(root[:, 2]))
    falls: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(_fall_segments(mode, time_s)):
        settle = slice(max(start - 24, 0), start)
        pre_height = float(np.median(root[settle, 2]))
        trigger_t = float(time_s[start])
        window = slice(start, end)
        segment_speeds = np.abs(speeds[window])
        peak_row = int(np.argmax(segment_speeds.max(axis=1)))
        # Fold at ~2 s after the trigger (settled) against the trigger pose.
        settled = min(start + int(2.0 / (time_s[1] - time_s[0])), end - 1)
        fold = {
            name: float(
                np.rad2deg(
                    abs(
                        joints[settled, dof_names.index(name)]
                        - joints[start, dof_names.index(name)]
                    )
                )
            )
            for name in FOLD_JOINTS
            if name in dof_names
        }
        fell_below_60 = time_s[window][root[window, 2] < 0.60 * pre_height]
        falls.append(
            {
                "index": index,
                "trigger_time_s": trigger_t,
                "pre_height_m": pre_height,
                "collapse_to_60pct_height_s": (
                    None if not len(fell_below_60) else float(fell_below_60[0] - trigger_t)
                ),
                "joint_speed_peak_rad_s": float(segment_speeds.max()),
                "joint_speed_peak_time_s": float(time_s[start + peak_row]),
                "joint_speed_peak_deg_s": float(np.rad2deg(segment_speeds.max())),
                "fold_angles_deg": fold,
                "nonfinite_joint_rows": int(
                    (~np.isfinite(joints[window]).all(axis=1)).sum()
                ),
                "duration_s": float(time_s[end - 1] - trigger_t),
            }
        )
    nonfinite_total = int((~np.isfinite(joints).all(axis=1)).sum())
    return {
        "variant": variant,
        "overrides": record["overrides"],
        "config_sha256": record["config_sha256"],
        "errors": report.get("errors"),
        "fall_events": report.get("fall", {}).get("events"),
        "falls": falls,
        "standing_height_m": standing_height,
        "nonfinite_joint_rows_total": nonfinite_total,
        "stage_clamp_check": _stage_clamp_check(session_dir, clamp),
        "screenshots": sorted(p.name for p in session_dir.glob("*.png")),
    }


def _analyze_all() -> int:
    results = {variant: _analyze_variant(variant) for variant in VARIANTS}
    write_json(OUT_ROOT / "analysis.json", results)
    header = (
        f"{'variant':<14} {'fall':<5} {'peak deg/s':>10} {'collapse s':>10} "
        f"{'knee fold deg':>13} {'nonfinite':>9} {'clamp ok':>8}"
    )
    print(header)
    print("-" * len(header))
    for variant, result in results.items():
        if result.get("status") == "session_not_run":
            print(f"{variant:<14} -- session not run --")
            continue
        clamp_ok = result["stage_clamp_check"].get("ok")
        for fall in result["falls"]:
            fold = fall["fold_angles_deg"]
            knee = max(
                (value for name, value in fold.items() if "knee" in name),
                default=float("nan"),
            )
            collapse = fall["collapse_to_60pct_height_s"]
            collapse_text = f"{collapse:>10.3f}" if collapse is not None else f"{'--':>10}"
            print(
                f"{variant:<14} {fall['index']:<5} {fall['joint_speed_peak_deg_s']:>10.1f} "
                f"{collapse_text} {knee:>13.1f} {fall['nonfinite_joint_rows']:>9d} "
                f"{str(clamp_ok):>8}"
            )
    LOGGER.info("analysis written to %s", OUT_ROOT / "analysis.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=sorted(VARIANTS),
        help="run one variant as an Isaac keyboard session (headless demo)",
    )
    parser.add_argument("--dry-run", action="store_true", help="write configs, no Isaac")
    parser.add_argument("--analyze", action="store_true", help="CPU analysis of all sessions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.run:
        return _run_variant(args.run)
    if args.dry_run:
        records = []
        for variant in VARIANTS:
            _variant_config(variant)
            records.append(
                json.loads((OUT_ROOT / variant / "variant.json").read_text(encoding="utf-8"))
            )
        write_json(OUT_ROOT / "variants.json", records)
        LOGGER.info("wrote %d variant configs under %s", len(records), OUT_ROOT)
        return 0
    if args.analyze:
        return _analyze_all()
    parser.error("one of --run / --dry-run / --analyze is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
