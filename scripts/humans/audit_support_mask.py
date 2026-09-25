#!/usr/bin/env python3
"""Test whether the stance-support mask actually marks a foot that can be planted.

``task_plan.md`` P0-B rests on one number: "the foot the reference labels as support is
itself sliding at a median 0.134 / 0.108 m/s". The world-frozen anchor, the release rule and
the plan to re-select source clips are all built on that premise, so this measures the mask
that produced it instead of trusting it.

The mask is derived in ``load_gait`` from *horizontal foot travel only*: a foot counts as
support while its x moves opposite the root travel. That test never looks at height, and it
never looks at whether the foot is world-stationary -- the two things an anchor needs. So it
is checked against both, using forward kinematics of the reference with no physics involved.

Three measurements, per gait:

A. **Against foot height.** A foot is on the floor when its capsule bottom is at its own
   lowest point over the cycle. The whole height distribution over masked frames is reported,
   not just the share past one threshold, so the tolerance can be judged rather than trusted.
B. **Against plantability.** Cross-tabulated as the anchor needs it: masked and world-still
   (holdable), masked but moving (unreachable by construction), world-still but not masked
   (a chance the mask throws away).
C. **Is a planted foot reachable at all?** The slower of the two feet in the world, per frame,
   under two root transports. This is the most generous planting test there is: it asks only
   whether the reference ever has *one* foot nearly stationary.

The left/right asymmetry is measured on the pelvis-relative quantity ``gait.speed_m_s`` is
actually estimated from. Reading it off world coordinates, or off a phase window one leg
occupies and the other does not, produces a gap that is a sampling artefact.

Read-only and CPU-only; no Isaac Sim, no new data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from common import (  # type: ignore[import-not-found]
    DEFAULT_MOTIONS,
    REPO_ROOT,
    load_inputs,
    write_json,
)

from sim2sense_fall.humans.amass import (
    crop_amass_clip,
    ground_amass_clip,
    load_amass_clip,
    normalize_root_motion,
)
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import capsule_bottom, fit_collision_capsules
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints
from sim2sense_fall.humans.rig import (
    fit_rest_skeleton,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
)
from sim2sense_fall.humans.teleop import load_gait, load_keyboard_config

LOGGER = logging.getLogger("audit_support_mask")

SIDES = ("left", "right")
ON_FLOOR_TOLERANCE_M = 0.02
AIRBORNE_MARGIN_M = 0.05
PLANTED_SPEED_M_S = 0.05
PHASE_BINS = 12


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
        plan, _audit = fit_collision_capsules(
            plan, mesh, **{k: v for k, v in settings["collision_fit"].items() if k != "enabled"}
        )
    return plan


def body_frame(gait: Any, plan: Any) -> dict[str, Any]:
    """Reproduce ``load_gait``'s own body-frame derivation, bit for bit.

    This is the quantity ``gait.speed_m_s`` is the median of, so comparing it against
    anything else would be comparing two different measurements.
    """

    count = len(gait.joints)
    ankle_x = []
    for phase in np.linspace(0.0, 1.0, count, endpoint=True):
        q, h = gait.sample(float(phase))
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, q, strict=True)),
            root_position=(0.0, 0.0, h),
            root_rotation=gait.tilt(float(phase)),
        )
        ankle_x.append([poses[f"{side}_ankle"].translation[0] for side in SIDES])
    derivative = np.gradient(np.asarray(ankle_x), axis=0)
    direction = float(np.sign(gait.provenance["source_forward_sign"]) or 1.0)
    dt_frame = gait.duration_s / (count - 1)
    return {
        "recede": -derivative * direction / dt_frame,
        "travel": (derivative * direction) < 0,
        "count": count,
    }


def world_frame(plan: Any, clip: Any, root_mode: str) -> dict[str, np.ndarray]:
    """Forward kinematics under one root-transport convention.

    Both arms share the joint angles, so any difference is attributable to the transport.
    ``body_frame`` leaves the root horizontally still; ``source_root`` uses the clip's own
    translation; ``constant_root`` advances at the clip's mean speed.
    """

    dt = 1.0 / clip.fps
    displacement = clip.root_translation[-1, :2] - clip.root_translation[0, :2]
    span = max(float(np.linalg.norm(displacement)), 1e-9)
    speed = span / (clip.times_s[-1] - clip.times_s[0])
    direction = displacement / span
    x = np.zeros((clip.frame_count, 2))
    bottom = np.zeros((clip.frame_count, 2))
    for frame in range(clip.frame_count):
        values, _ = joint_values_from_clip(clip, frame, plan)
        root = np.asarray(plan.spawn_root_position, dtype=np.float64).copy()
        root[2] += clip.root_translation[frame, 2]
        if root_mode == "source_root":
            root[:2] += clip.root_translation[frame, :2]
        elif root_mode == "constant_root":
            root[:2] += direction * speed * (frame * dt)
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, values, strict=True)),
            root_position=root,
            root_rotation=clip.root_rotation[frame],
        )
        for column, side in enumerate(SIDES):
            x[frame, column] = poses[f"{side}_ankle"].translation[0]
            bottom[frame, column] = capsule_bottom(plan, poses, f"{side}_ankle")
    return {
        "ankle_x": x,
        "foot_bottom_z": bottom,
        "root_speed": np.array([speed]),
        "root_direction_x": np.array([direction[0]]),
    }


def cross_tab(mask: np.ndarray, other: np.ndarray, side: str, index: int) -> dict[str, Any]:
    a = mask[:, index]
    b = other[:, index]
    both = int((a & b).sum())
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    return {
        "side": side,
        "masked_and_other": both,
        "masked_only": a_only,
        "other_only": b_only,
        "neither": int((~a & ~b).sum()),
        "share_of_masked_failing_other": float(a_only / max(both + a_only, 1)),
        "share_of_other_missed": float(b_only / max(both + b_only, 1)),
    }


def audit(plan: Any, settings: dict[str, Any], key: str, spec: dict[str, Any]) -> dict[str, Any]:
    dt = 1.0 / 120.0
    planted_tolerance = settings.get("max_stance_slip_m_s")
    gait = load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=planted_tolerance)
    clip = ground_amass_clip(
        normalize_root_motion(
            crop_amass_clip(
                load_amass_clip(spec["file"]),
                start_s=spec["start_s"],
                duration_s=spec["duration_s"],
            )
        ),
        plan,
        support_z_m=0.0,
    ).resample(1.0 / dt, method="slerp")
    frames = clip.frame_count
    frame_dt = 1.0 / clip.fps

    body = body_frame(gait, plan)
    travel = np.asarray(body["travel"])
    shipped = np.asarray(gait.support_mask)
    if travel.shape != (frames, 2) or shipped.shape != (frames, 2):
        raise ValueError(f"mask shape {shipped.shape} does not match {frames} frames")

    source = world_frame(plan, clip, "source_root")
    lowest = source["foot_bottom_z"].min(axis=0)
    on_floor = source["foot_bottom_z"] <= lowest + ON_FLOOR_TOLERANCE_M
    airborne = source["foot_bottom_z"] > lowest + AIRBORNE_MARGIN_M
    world_speed = np.zeros_like(source["ankle_x"])
    for column in range(2):
        world_speed[:, column] = np.abs(np.gradient(source["ankle_x"][:, column]) / frame_dt)
    still = world_speed < PLANTED_SPEED_M_S

    report: dict[str, Any] = {
        "gait": key,
        "file": str(spec["file"]),
        "frames": int(frames),
        "gait_speed_m_s": float(gait.speed_m_s),
        "travel_frames": {side: int(travel[:, i].sum()) for i, side in enumerate(SIDES)},
        "shipped_mask_frames": {side: int(shipped[:, i].sum()) for i, side in enumerate(SIDES)},
        "shipped_gate": planted_tolerance,
        "source_root_speed_m_s": float(source["root_speed"][0]),
    }

    masked_height = np.concatenate(
        [
            (source["foot_bottom_z"][shipped[:, i], i] - lowest[i])
            for i in range(2)
            if shipped[:, i].any()
        ]
    )
    report["foot_height"] = {
        "definition": "masked-frame foot capsule bottom above that foot's own lowest point",
        "foot_lowest_z_m": {side: float(lowest[i]) for i, side in enumerate(SIDES)},
        "tolerance_m": ON_FLOOR_TOLERANCE_M,
        "airborne_margin_m": AIRBORNE_MARGIN_M,
        "p50_m": float(np.median(masked_height)),
        "p90_m": float(np.percentile(masked_height, 90)),
        "max_m": float(masked_height.max()),
        "share_above_10_mm": float(np.mean(masked_height > 0.010)),
        "share_above_20_mm": float(np.mean(masked_height > 0.020)),
        "share_above_50_mm": float(np.mean(masked_height > 0.050)),
    }
    for label, mask in (("travel", travel), ("shipped", shipped)):
        rows = [cross_tab(mask, on_floor, side, i) for i, side in enumerate(SIDES)]
        failures = sum(row["masked_only"] for row in rows)
        total = sum(row["masked_and_other"] + row["masked_only"] for row in rows)
        report[f"mask_{label}_vs_on_floor"] = {
            "sides": rows,
            "share_of_masked_frames_off_floor": float(failures / max(total, 1)),
            "share_of_masked_frames_clearly_airborne": float(
                sum(int((mask[:, i] & airborne[:, i]).sum()) for i in range(2)) / max(total, 1)
            ),
        }
        rows = [cross_tab(mask, still, side, i) for i, side in enumerate(SIDES)]
        holdable = sum(row["masked_and_other"] for row in rows)
        unreachable = sum(row["masked_only"] for row in rows)
        missed = sum(row["other_only"] for row in rows)
        report[f"mask_{label}_vs_planted"] = {
            "sides": rows,
            "masked_and_world_still": holdable,
            "masked_but_world_moving": unreachable,
            "world_still_but_not_masked": missed,
            "share_of_masked_frames_holdable": float(holdable / max(holdable + unreachable, 1)),
            "share_of_plantable_frames_missed": float(missed / max(missed + holdable, 1)),
        }

    # Is the left/right gap real, or does each leg's mask simply cover different phases?
    recede = body["recede"]
    phase = np.linspace(0.0, 1.0, frames, endpoint=False)
    bin_of = np.minimum((phase * PHASE_BINS).astype(int), PHASE_BINS - 1)
    common = [
        value
        for value in range(PHASE_BINS)
        if (travel[bin_of == value, 0].any() and travel[bin_of == value, 1].any())
    ]
    in_common = np.isin(bin_of, common)
    medians_all = {
        side: float(np.median(recede[travel[:, i], i])) for i, side in enumerate(SIDES)
    }
    medians_common = {}
    for index, side in enumerate(SIDES):
        selected = travel[:, index] & in_common
        medians_common[side] = float(np.median(recede[selected, index])) if selected.any() else None
    report["left_right"] = {
        "quantity": "pelvis-relative recede speed, the quantity gait.speed_m_s is the median of",
        "median_all_masked_frames_m_s": medians_all,
        "pooled_median_m_s": float(np.median(recede[travel])),
        "pooled_median_matches_gait_speed": bool(
            abs(float(np.median(recede[travel])) - float(gait.speed_m_s)) < 1e-6
        ),
        "gap_all_masked_frames": float(
            abs(medians_all["left"] - medians_all["right"]) / abs(medians_all["right"])
        ),
        "phase_bins_masked_for_both_legs": len(common),
        "phase_bins_total": PHASE_BINS,
        "median_common_phase_bins_m_s": medians_common,
        "gap_common_phase_bins": None
        if not all(medians_common.values())
        else float(
            abs(medians_common["left"] - medians_common["right"]) / abs(medians_common["right"])
        ),
        "phase_bin_table": [
            {
                "phase_bin": value,
                "travel_left": int(travel[bin_of == value, 0].sum()),
                "travel_right": int(travel[bin_of == value, 1].sum()),
                "recede_left_m_s": float(np.median(recede[bin_of == value, 0])),
                "recede_right_m_s": float(np.median(recede[bin_of == value, 1])),
            }
            for value in range(PHASE_BINS)
        ],
    }

    reachability: dict[str, Any] = {}
    for mode in ("source_root", "constant_root"):
        data = world_frame(plan, clip, mode)
        speed = np.zeros_like(data["ankle_x"])
        for column in range(2):
            speed[:, column] = np.abs(np.gradient(data["ankle_x"][:, column]) / frame_dt)
        slower = speed.min(axis=1)
        reachability[mode] = {
            "slower_foot_p05_m_s": float(np.percentile(slower, 5)),
            "slower_foot_p50_m_s": float(np.median(slower)),
            "share_a_foot_below_0_05_m_s": float(np.mean(slower < 0.05)),
            "share_a_foot_below_0_15_m_s": float(np.mean(slower < 0.15)),
        }
    report["planted_foot_reachability"] = reachability
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_keyboard_config(args.config, REPO_ROOT)
    plan = build_plan(settings)
    results = [audit(plan, settings, key, spec) for key, spec in settings["gaits"].items()]
    for report in results:
        if args.json:
            continue
        height = report["foot_height"]
        print(f"\n=== {report['gait']}  ({Path(report['file']).name}) ===")
        print(
            f"  frames={report['frames']}  gait.speed_m_s={report['gait_speed_m_s']:.3f}  "
            f"travel mask L/R={report['travel_frames']['left']}/"
            f"{report['travel_frames']['right']}  "
            f"shipped gate {report['shipped_gate']} L/R={report['shipped_mask_frames']['left']}/"
            f"{report['shipped_mask_frames']['right']}"
        )
        print(
            f"  A. height of the masked foot above its own lowest point: "
            f"p50={height['p50_m'] * 1000:.1f} p90={height['p90_m'] * 1000:.1f} "
            f"max={height['max_m'] * 1000:.1f} mm; above 10/20/50 mm in "
            f"{height['share_above_10_mm'] * 100:.0f}/{height['share_above_20_mm'] * 100:.0f}/"
            f"{height['share_above_50_mm'] * 100:.0f}% of them"
        )
        for label in ("travel", "shipped"):
            floor = report[f"mask_{label}_vs_on_floor"]
            print(
                f"     {label:8s} mask: {floor['share_of_masked_frames_off_floor'] * 100:.1f}% of "
                f"masked frames have the foot off the floor, "
                f"{floor['share_of_masked_frames_clearly_airborne'] * 100:.1f}% clearly airborne"
            )
        for label in ("travel", "shipped"):
            planted = report[f"mask_{label}_vs_planted"]
            print(
                f"     {label:8s} mask: holdable={planted['masked_and_world_still']:4d} "
                f"unreachable={planted['masked_but_world_moving']:4d} "
                f"missed={planted['world_still_but_not_masked']:4d}  ->  "
                f"{planted['share_of_masked_frames_holdable'] * 100:.1f}% of masked frames are "
                f"holdable, {planted['share_of_plantable_frames_missed'] * 100:.1f}% of plantable "
                f"frames are missed"
            )
        left_right = report["left_right"]
        print("  B. left/right on the body-frame quantity:")
        print(
            f"     pooled median {left_right['pooled_median_m_s']:.3f} m/s "
            f"(equals gait.speed_m_s: {left_right['pooled_median_matches_gait_speed']}); "
            f"per leg over their own masked frames "
            f"{left_right['median_all_masked_frames_m_s']['left']:.3f} vs "
            f"{left_right['median_all_masked_frames_m_s']['right']:.3f} m/s "
            f"(gap {left_right['gap_all_masked_frames'] * 100:.1f}%)"
        )
        common_medians = left_right["median_common_phase_bins_m_s"]
        if left_right["gap_common_phase_bins"] is None:
            print("     no phase bin is masked for both legs")
        else:
            print(
                f"     over the {left_right['phase_bins_masked_for_both_legs']}/"
                f"{left_right['phase_bins_total']} phase bins masked for both legs: "
                f"{common_medians['left']:.3f} vs {common_medians['right']:.3f} m/s "
                f"(gap {left_right['gap_common_phase_bins'] * 100:.1f}%)"
            )
        print("  C. slower-foot world speed (can the reference plant ANY foot?)")
        for mode, row in report["planted_foot_reachability"].items():
            print(
                f"     {mode:14s} p05={row['slower_foot_p05_m_s']:.3f} "
                f"p50={row['slower_foot_p50_m_s']:.3f} m/s; a foot under 0.05 m/s in "
                f"{row['share_a_foot_below_0_05_m_s'] * 100:.1f}% of frames, under 0.15 m/s in "
                f"{row['share_a_foot_below_0_15_m_s'] * 100:.1f}%"
            )
    if args.out is not None:
        write_json(args.out, results)
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
