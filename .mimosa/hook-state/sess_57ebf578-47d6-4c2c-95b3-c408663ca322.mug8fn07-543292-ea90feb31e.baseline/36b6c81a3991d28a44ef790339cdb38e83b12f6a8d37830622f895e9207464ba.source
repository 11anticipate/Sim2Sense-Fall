#!/usr/bin/env python3
"""Measure whether a rig configuration can express real recorded motion.

Example::

    python3 scripts/humans/measure_axes.py \
        --root data/humans/amass_raw \
        --config configs/humans/human_smpl_multiaxis.yaml

The rig is only able to replay a recorded motion if each joint's declared chain
of revolute axes can reproduce the rotation that joint performs. A single-axis
rig could not reproduce any real AMASS clip, which is why the whole library was
once written off: the screen reported "0 clips pass" without saying that the
*rig* was the limiting factor rather than the data.

For every driven joint this reports, over a deterministic sample of the local
AMASS library:

* the geodesic miss of the best 1-axis, 2-axis and declared-chain solution;
* the per-axis angle distribution the declared chain actually returns;
* the fraction of sampled frames whose angles would be clipped by the
  configured limits, because a clipped angle replays a different pose.

It is CPU-only and never downloads anything; the AMASS root must already exist.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from common import DEFAULT_CONFIG, DEFAULT_OUTPUT_DIR, Checks, write_json  # noqa: E402

from sim2sense_fall.humans.amass import load_amass_library  # noqa: E402
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.rig import dof_groups, plan_human_rig  # noqa: E402
from sim2sense_fall.humans.rotations import (  # noqa: E402
    rotation_split_residual_rad,
    split_rotation,
)

LOGGER = logging.getLogger("measure_axes")

#: Percentiles reported for every distribution, as (low, centre, high).
_PERCENTILES = (0.5, 50.0, 99.5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="directory holding the local AMASS corpus (searched recursively)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clips", type=int, default=60, help="clips to sample")
    parser.add_argument("--frames-per-clip", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "axis_measurement.json",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    """Percentiles of a distribution, or nulls when no sample applies.

    A single-axis rig has no two-axis measurement to report, and a JSON artefact
    that cannot be written because ``nan`` is not valid JSON is worse than one
    that says "not applicable".
    """

    if not values:
        return {"p_low": None, "p50": None, "p_high": None}
    low, centre, high = (float(np.percentile(values, p)) for p in _PERCENTILES)
    return {"p_low": low, "p50": centre, "p_high": high}


def _degrees(value: float | None) -> float:
    """Convert radians to degrees for display, tolerating "not applicable"."""

    return float("nan") if value is None else math.degrees(value)


def measure(
    root: Path, config_path: Path, *, clips: int, frames_per_clip: int, seed: int
) -> dict[str, Any]:
    config = load_human_config(config_path)
    plan = plan_human_rig(config)
    library = load_amass_library(root)
    identifiers = sorted(library.clips)
    if not identifiers:
        raise SystemExit(f"no AMASS sequences found under {root}")
    generator = np.random.default_rng(seed)
    count = min(clips, len(identifiers))
    picked = generator.choice(len(identifiers), size=count, replace=False)

    groups = dof_groups(plan)
    limits = plan.limits_deg()
    axis_samples: dict[str, list[tuple[float, ...]]] = {name: [] for name in groups}
    residual_one: dict[str, list[float]] = {name: [] for name in groups}
    residual_two: dict[str, list[float]] = {name: [] for name in groups}
    residual_full: dict[str, list[float]] = {name: [] for name in groups}
    ignored: set[str] = set()
    frames_seen = 0
    expressible = 0

    for index in picked:
        clip = library.clips[identifiers[int(index)]]
        # A joint the corpus rotates but the rig does not drive would be silently
        # ignored: the replay would look plausible and be wrong.
        for name in clip.joint_names:
            if name in groups:
                continue
            if np.linalg.norm(np.asarray(clip.rotation_of(0, name))) > 1e-9 or any(
                np.linalg.norm(np.asarray(clip.rotation_of(frame, name))) > 1e-9
                for frame in range(clip.frame_count)
            ):
                ignored.add(name)
        stride = max(1, clip.frame_count // max(1, frames_per_clip))
        for frame in range(0, clip.frame_count, stride):
            frames_seen += 1
            worst = 0.0
            for chain_joint, joints in groups.items():
                if chain_joint not in clip.joint_names:
                    continue
                vector = np.asarray(clip.rotation_of(frame, chain_joint), dtype=np.float64)
                axes = tuple(joint.axis for joint in joints)
                solution = split_rotation(vector, axes, axis_angle=vector)
                axis_samples[chain_joint].append(solution.angles)
                residual_full[chain_joint].append(solution.residual_rad)
                worst = max(worst, solution.residual_rad)
                # The same target against the shorter chains answers "would fewer
                # axes have done?", which is the question the rig design turns on.
                for length, bucket in ((1, residual_one), (2, residual_two)):
                    if len(axes) <= length:
                        continue
                    shorter = split_rotation(vector, axes[:length], axis_angle=vector)
                    bucket[chain_joint].append(
                        rotation_split_residual_rad(vector, axes[:length], shorter.angles)
                    )
            expressible += worst <= 1e-6

    joints: dict[str, Any] = {}
    # Keyed by DOF name, because a limit belongs to a DOF and a chain has several.
    clipped_frames: dict[str, int] = {joint.name: 0 for joint in plan.joints}
    for chain_joint, samples in axis_samples.items():
        if not samples:
            continue
        angles_rad = np.asarray(samples)
        joints_for_chain = groups[chain_joint]
        entry: dict[str, Any] = {
            "dof_count": len(joints_for_chain),
            "samples": len(samples),
            "residual_rad": {
                "one_axis": _percentiles(residual_one[chain_joint]),
                "two_axis": _percentiles(residual_two[chain_joint]),
                "declared_chain": _percentiles(residual_full[chain_joint]),
            },
            "axes": {},
        }
        for position, joint in enumerate(joints_for_chain):
            low, high = limits[joint.name]
            column_deg = np.degrees(angles_rad[:, position])
            entry["axes"][joint.name] = {
                "axis": joint.axis,
                "limits_deg": [low, high],
                **_percentiles([float(value) for value in angles_rad[:, position]]),
            }
            outside = np.count_nonzero((column_deg < low - 1e-6) | (column_deg > high + 1e-6))
            clipped_frames[joint.name] += int(outside)
        joints[chain_joint] = entry

    return {
        "config": str(config_path),
        "library_root": str(root),
        "library_clips": len(identifiers),
        "library_failures": len(library.failures),
        "sampled_clips": count,
        "sampled_frames": frames_seen,
        "dof_count": plan.stats["dof_count"],
        "driven_joint_count": plan.stats["driven_joint_count"],
        "seed": seed,
        "expressible_fraction": (expressible / frames_seen) if frames_seen else 0.0,
        "ignored_joints": sorted(ignored),
        "joints": joints,
        "limits_clipped": {
            name: {
                "clipped_samples": clipped_frames[name],
                "limits_deg": list(limits[name]),
                "samples": next(
                    (entry["samples"] for entry in joints.values() if name in entry["axes"]),
                    0,
                ),
            }
            for name in clipped_frames
        },
    }


def report(payload: dict[str, Any], checks: Checks) -> None:
    total_files = payload["library_clips"] + payload["library_failures"]
    checks.check(
        "unreadable AMASS sequences stay a small minority of the corpus",
        payload["library_failures"] <= 0.05 * max(1, total_files),
        f"{payload['library_failures']} of {total_files}",
    )
    fraction = payload["expressible_fraction"]
    checks.check(
        "the declared chains reproduce the sampled motion",
        fraction >= 0.999,
        f"{fraction:.4%} of {payload['sampled_frames']} sampled frames",
    )
    checks.check(
        "the rig drives every joint the corpus rotates",
        not payload["ignored_joints"],
        f"ignored {payload['ignored_joints']}" if payload["ignored_joints"] else "none ignored",
    )
    total_clipped = sum(entry["clipped_samples"] for entry in payload["limits_clipped"].values())
    checks.check(
        "no judged angle is clipped by its declared limits",
        total_clipped == 0,
        f"{total_clipped} clipped samples",
    )
    checks.info(
        f"rig DOF {payload['dof_count']} over {payload['driven_joint_count']} joints; "
        f"{payload['sampled_clips']} clips, {payload['sampled_frames']} frames"
    )
    for name, entry in payload["joints"].items():
        one = _degrees(entry["residual_rad"]["one_axis"]["p50"])
        two = _degrees(entry["residual_rad"]["two_axis"]["p50"])
        high = _degrees(entry["residual_rad"]["declared_chain"]["p_high"])
        spread = "  ".join(
            f"{values['axis']}:{_degrees(values['p_low']):.0f}/"
            f"{_degrees(values['p50']):.0f}/{_degrees(values['p_high']):.0f}"
            for values in entry["axes"].values()
        )
        checks.info(
            f"{name:<15} dof {entry['dof_count']}  median miss "
            f"1ax {one:6.2f} deg, 2ax {two:6.2f} deg, "
            f"chain p99.5 {high:.2e} deg  |  angles {spread}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    checks = Checks()
    payload = measure(
        args.root,
        args.config,
        clips=args.clips,
        frames_per_clip=args.frames_per_clip,
        seed=args.seed,
    )
    report(payload, checks)
    write_json(args.output, payload)
    checks.info(f"wrote {args.output}")
    return checks.report(banner="axis measurement")


if __name__ == "__main__":
    raise SystemExit(main())
