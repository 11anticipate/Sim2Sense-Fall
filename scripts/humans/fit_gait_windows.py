#!/usr/bin/env python3
"""Fit each gait window to a whole number of strides, and show what it costs not to.

``Gait.sample`` makes the reference periodic by subtracting a linear ramp::

    q(phase) = interpolate(phase) - phase * (joints[-1] - joints[0])

That is harmless only if the window holds a whole number of gait cycles, because then
``joints[-1] - joints[0]`` is ~0 and the ramp does nothing. When the window is *not* a whole
cycle the ramp has to cancel real motion, at a per-DOF rate set by that DOF's endpoint
difference. The two legs sit half a cycle apart, so the same ramp function of phase reaches
them at opposite phases and comes out of the geometry as a **constant fore-aft stagger
between the feet** -- a limp no gain or joint limit can remove, because it is in the reference.

Measured on the shipped windows: ``forward`` (CMU/08/04, 144 frames from 0.0 s) has an endpoint
difference of **16.95 deg** and ``backward`` (CMU/08/11, 144 frames from 0.4 s) has **32.87 deg**,
against a real stride of 160 frames for 08/04.

The search scores a candidate window on exactly the quantity the ramp must cancel, then
rejects half-strides: a candidate whose *two-cycle* score is better than its one-cycle score is
half a stride, not a stride. Every reported number is produced by calling ``load_gait`` with an
overridden window, so it goes through the shipped reference pipeline including the correction,
the cadence estimate and the support mask. Nothing here is a reimplementation.

Read-only: it measures and prints. Applying it is a config edit, made deliberately.
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
)

from sim2sense_fall.humans.amass import load_amass_clip
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.rig import (
    fit_rest_skeleton,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
)
from sim2sense_fall.humans.teleop import load_gait, load_keyboard_config

LOGGER = logging.getLogger("fit_gait_windows")

SIDES = ("left", "right")
MIN_PERIOD_S = 0.70
MAX_PERIOD_S = 2.20
# A candidate period counts as a full stride only if two of them close no better than one does.
DOUBLE_PERIOD_SLACK = 1.05


def joint_track(plan: Any, clip: Any) -> np.ndarray:
    return np.stack(
        [joint_values_from_clip(clip, frame, plan)[0] for frame in range(clip.frame_count)]
    )


def best_window(angles: np.ndarray, *, fps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Find the stride lag globally, then the cut point that loops cleanly for it.

    Two quantities, deliberately measured differently. The **period** comes from the lag that
    minimises the whole track's self-distance: a stride is a property of the sequence, and using
    the whole sequence is more robust than any single start. The **start** then comes from the
    endpoint difference at that lag, because closing the loop is a property of the cut.

    Confirming a period by closing two strides at the same start needs two strides of runway,
    which a short clip cannot give. The global distance has no such limit, and a half-stride is
    still caught: at half a stride, ``lag 2P`` scores better than ``lag P``.
    """

    frames = angles.shape[0]
    low = max(2, int(round(MIN_PERIOD_S * fps)))
    high = min(frames // 2, int(round(MAX_PERIOD_S * fps)))
    if low >= high:
        raise ValueError("clip is too short for the period range")
    degrees = 180.0 / np.pi
    spread = angles.std(axis=0)
    normalized = (angles - angles.mean(axis=0)) / np.where(spread > 1e-9, spread, 1.0)

    def lag_distance(lag: int) -> float:
        """Mean normalised self-distance at this lag, so wide-range DOFs do not dominate."""

        return float(np.abs(normalized[lag:] - normalized[:-lag]).mean(axis=0).mean())

    rows: list[dict[str, Any]] = []
    for period in range(low, high + 1):
        start = int(np.argmin(np.abs(angles[period:] - angles[:-period]).max(axis=1)))
        rows.append(
            {
                "period_frames": period,
                "period_s": period / fps,
                "start_frame": start,
                "start_s": start / fps,
                "start_endpoint_deg": float(
                    np.abs(angles[start + period] - angles[start]).max() * degrees
                ),
                "periodicity": lag_distance(period),
                "periodicity_at_two": lag_distance(2 * period) if 2 * period < frames else None,
            }
        )
    # A half-stride measures better at twice its own length than at its own length.
    kept = [
        row
        for row in rows
        if row["periodicity_at_two"] is None
        or row["periodicity_at_two"] >= row["periodicity"] / DOUBLE_PERIOD_SLACK
    ]
    rejected = [row for row in rows if row not in kept]
    if not kept:
        raise ValueError("no candidate period survives the half-stride test")
    kept.sort(key=lambda row: row["start_endpoint_deg"])
    rejected.sort(key=lambda row: row["periodicity"])
    best = dict(kept[0])
    runway = best["start_frame"] + 2 * best["period_frames"] < frames
    best["two_stride_deg"] = (
        float(
            np.abs(
                angles[best["start_frame"] + 2 * best["period_frames"]]
                - angles[best["start_frame"]]
            ).max()
            * degrees
        )
        if runway
        else None
    )
    return best, rejected[:3]


def window_geometry(plan: Any, spec: dict[str, Any], dt_s: float) -> dict[str, Any]:
    """Measure a candidate window through the shipped reference pipeline."""

    gait = load_gait(spec, plan, dt_s=dt_s, max_stance_slip_m_s=None)
    count = len(gait.joints)
    ankle_x: list[list[float]] = []
    for phase in np.linspace(0.0, 1.0, count, endpoint=True):
        q, height = gait.sample(float(phase))
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, q, strict=True)),
            root_position=(0.0, 0.0, height),
            root_rotation=gait.tilt(float(phase)),
        )
        ankle_x.append([poses[f"{side}_ankle"].translation[0] for side in SIDES])
    values = np.asarray(ankle_x)
    span = values.max(axis=0) - values.min(axis=0)
    mean = values.mean(axis=0)
    return {
        "duration_s": gait.duration_s,
        "frames": count,
        "endpoint_max_deg": float(
            np.rad2deg(np.abs(gait.joints[-1] - gait.joints[0])).max()
        ),
        "gait_speed_m_s": float(gait.speed_m_s),
        "root_advance_per_cycle_m": float(gait.speed_m_s * gait.duration_s),
        "step_length_m": {"left": float(span[0]), "right": float(span[1])},
        "step_length_ratio_left_over_right": float(span[0] / max(span[1], 1e-9)),
        "mean_fore_aft_offset_m": {"left": float(mean[0]), "right": float(mean[1])},
        "fore_aft_stagger_m": float(abs(mean[0] - mean[1])),
        "support_frames": {
            side: int(np.asarray(gait.support_mask)[:, index].sum())
            for index, side in enumerate(SIDES)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_keyboard_config(args.config, REPO_ROOT)
    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    plan = plan_human_rig(config, rest=fit_rest_skeleton(config, body.model.mesh().rest_skeleton()))
    dt_s = config.simulation.physics_dt_s

    payload: list[dict[str, Any]] = []
    for key, spec in settings["gaits"].items():
        clip = load_amass_clip(spec["file"])
        best, rejected = best_window(joint_track(plan, clip), fps=clip.fps)
        fitted_spec = {
            **spec,
            "start_s": float(best["start_s"]),
            "duration_s": float(best["period_s"]),
        }
        before = window_geometry(plan, spec, dt_s)
        after = window_geometry(plan, fitted_spec, dt_s)
        payload.append(
            {"gait": key, "file": str(spec["file"]), "best": best, "rejected": rejected,
             "configured": before, "fitted": after}
        )

        print(f"\n=== {key}  ({Path(str(spec['file'])).name}) ===")
        print(
            f"  configured window: start {spec['start_s']} s for {before['duration_s']:.4f} s "
            f"({before['frames']} frames)"
        )
        print(
            f"  fitted window:     start {best['start_s']:.4f} s for {best['period_s']:.4f} s "
            f"({best['period_frames']} frames)"
        )
        print(
            f"  loop closure (what Gait.sample has to cancel): "
            f"{before['endpoint_max_deg']:6.2f} deg  ->  {after['endpoint_max_deg']:6.2f} deg"
        )
        if best["two_stride_deg"] is None:
            print(
                "  two-stride confirmation: not available (clip has no runway past this start); "
                "the half-stride test was applied to the global periodicity instead"
            )
        else:
            print(
                f"  closes at two strides: {best['two_stride_deg']:.2f} deg vs one stride "
                f"{best['start_endpoint_deg']:.2f} deg -- not better, so this is a stride"
            )
        print(
            f"  global periodicity at this lag: {best['periodicity']:.4f}"
            + (
                ""
                if best["periodicity_at_two"] is None
                else f", at twice the lag: {best['periodicity_at_two']:.4f} (not better)"
            )
        )
        if rejected:
            print("  candidates rejected as half-strides (frames / one-stride / two-stride deg):")
            for row in rejected:
                print(
                    f"     {row['period_frames']:5d} frames  start {row['start_frame']:4d}  "
                    f"periodicity {row['periodicity']:.4f} vs at twice "
                    f"{row['periodicity_at_two']:.4f}"
                )
        print(
            f"  step length: left {before['step_length_m']['left'] * 1000:6.1f} -> "
            f"{after['step_length_m']['left'] * 1000:6.1f} mm   right "
            f"{before['step_length_m']['right'] * 1000:6.1f} -> "
            f"{after['step_length_m']['right'] * 1000:6.1f} mm   ratio L/R "
            f"{before['step_length_ratio_left_over_right']:.3f} -> "
            f"{after['step_length_ratio_left_over_right']:.3f}"
        )
        print(
            f"  fore-aft stagger between the feet: "
            f"{before['fore_aft_stagger_m'] * 1000:6.1f} -> "
            f"{after['fore_aft_stagger_m'] * 1000:6.1f} mm   (left "
            f"{before['mean_fore_aft_offset_m']['left'] * 1000:+.1f} -> "
            f"{after['mean_fore_aft_offset_m']['left'] * 1000:+.1f}, right "
            f"{before['mean_fore_aft_offset_m']['right'] * 1000:+.1f} -> "
            f"{after['mean_fore_aft_offset_m']['right'] * 1000:+.1f})"
        )
        print(
            f"  root advance per cycle: {before['root_advance_per_cycle_m'] * 1000:.1f} -> "
            f"{after['root_advance_per_cycle_m'] * 1000:.1f} mm  "
            f"(gait.speed_m_s {before['gait_speed_m_s']:.4f} -> {after['gait_speed_m_s']:.4f})"
        )
        print(
            f"  suggested config for {key}: "
            f"start_s: {best['start_s']:.4f}, duration_s: {best['period_s']:.4f}"
        )
    if args.out is not None:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
