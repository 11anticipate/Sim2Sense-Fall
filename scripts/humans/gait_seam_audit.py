#!/usr/bin/env python3
"""Loop-seam and phase audit for the P0-A arm-swing checklist.

``task_plan.md`` P0-A asks for the loop seam to be *recorded*, not just the
amplitude: the cyclic control reference is derived from AMASS through a
distributed linear endpoint correction (``Gait.sample``), so the seam is a
place where a wrong correction would show up as a discontinuity or as a
systematic amplitude bias.

What this measures, all on CPU with no Isaac Sim:

1. **Joint trace continuity.** For every DOF, the per-frame step size of the
   *sampled control target* (the actual reference the controller sends) around
   the wrap point ``phase 1 -> 0`` compared against the largest step inside the
   cycle. A seam is clean when the wrap step is not an outlier.
2. **Endpoint correction.** The correction magnitude the distributed loop term
   removes, per DOF, and which DOFs dominate it. This is the quantity that
   creates a seam if it is left in.
3. **Left/right arm amplitude per pipeline axis**, at the *target* level (what
   the controller commands), matching the source-level numbers in
   ``artifacts/humans/arm_swing_audit/arm_swing_audit.json`` so the retarget
   stage can be compared stage by stage.
4. **Phase.** Zero-shift normalised correlation between the left and right arm
   on the dominant (fore-aft) axis, and the same for arm-versus-leg.

The output is a JSON report plus an optional PNG; nothing is asserted as a gate
here, because the P0-A finding is diagnostic (the shipped window is one quiet
right-arm stretch of the source), not a pass/fail threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from common import REPO_ROOT, write_json  # type: ignore[import-not-found]

from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints  # noqa: E402
from sim2sense_fall.humans.rig import (
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)  # noqa: E402
from sim2sense_fall.humans.teleop import load_gait, load_keyboard_config  # noqa: E402

LOGGER = logging.getLogger("gait_seam_audit")

# An axis letter never tells you an anatomical role. The same letter is a knee's
# flexion and a shoulder's twist, because every chain declares its own rotation order
# and the bone's rest orientation differs between a hanging arm and the SMPL template.
# So the role is read out of the geometry this plan actually has -- the direction the
# distal joint travels when the DOF turns -- and cannot drift away from it.
DISTAL_LINK = {
    "hip": "knee",
    "knee": "ankle",
    "ankle": "foot",
    "shoulder": "elbow",
    "elbow": "wrist",
    "wrist": "hand",
}
PROBE_DEG = 30.0


def measured_axis_roles(plan: Any) -> dict[str, str]:
    """``{dof_name: role}`` from the tip displacement each single DOF produces."""

    rest = forward_kinematics(plan)
    link_names = {link.name for link in plan.links}
    roles: dict[str, str] = {}
    for joint in plan.joints:
        side, _, part = joint.chain_joint.partition("_")
        tip = f"{side}_{DISTAL_LINK.get(part, '')}"
        if part not in DISTAL_LINK or tip not in link_names:
            continue
        angles = {name: 0.0 for name in plan.dof_names}
        angles[joint.name] = np.deg2rad(PROBE_DEG)
        moved = forward_kinematics(plan, angles)[tip].translation - rest[tip].translation
        if float(np.linalg.norm(moved)) < 0.01:
            roles[joint.name] = "indeterminate (tip barely moves)"
            continue
        direction = ("fore-aft", "sideways", "vertical")[int(np.argmax(np.abs(moved)))]
        travel_cm = float(np.max(np.abs(moved))) * 100
        roles[joint.name] = f"{direction} {PROBE_DEG:.0f}deg={travel_cm:.1f}cm at {tip}"
    return roles


ARM_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
LEG_JOINTS = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


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


def _normalised_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Zero-shift Pearson correlation, guarding against a zero-variance side."""

    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    if scale <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / scale)


def _best_shift(left: np.ndarray, right: np.ndarray, max_lag: int) -> dict[str, float]:
    """Largest normalised correlation over integer lags, and that lag in frames."""

    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    if scale <= 1e-12:
        return {"correlation": float("nan"), "lag_frames": 0.0}
    best_correlation, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        rolled = np.roll(right, lag)
        value = float(np.dot(left, rolled) / scale)
        if value > best_correlation:
            best_correlation, best_lag = value, lag
    return {"correlation": best_correlation, "lag_frames": float(best_lag)}


def trace_continuity(samples: np.ndarray) -> dict[str, Any]:
    """Wrap-step versus interior-step size for a cyclic control trace ``(T, D)``."""

    steps = np.abs(np.diff(samples, axis=0))
    wrap = np.abs(samples[0] - samples[-1])
    interior = steps.max(axis=0)
    ratio = np.divide(
        wrap, np.maximum(interior, 1e-12), out=np.zeros_like(wrap), where=interior > 1e-12
    )
    return {
        "wrap_step_rad": wrap,
        "interior_max_step_rad": interior,
        "wrap_over_interior": ratio,
        "wrap_is_outlier": ratio > 1.5,
    }


def analyse_gait(name: str, gait: Any, plan: Any, samples: int, seed: int) -> dict[str, Any]:
    dof_names = [joint.name for joint in plan.joints]
    lower = np.deg2rad([joint.lower_deg for joint in plan.joints])
    upper = np.deg2rad([joint.upper_deg for joint in plan.joints])
    reference = gait.joints
    phases = np.linspace(0.0, 1.0, samples, endpoint=False)
    sampled = np.stack([gait.sample(float(p))[0] for p in phases])
    heights = np.asarray([gait.sample(float(p))[1] for p in phases])

    # The loop correction is exactly what sample() subtracts; recover it by
    # comparing a raw interpolation against the corrected one.
    raw = np.stack([_raw_interpolate(reference, float(p)) for p in phases])
    correction = raw - sampled

    continuity = trace_continuity(trace_continuity_input(sampled))
    dominant = {
        dof_names[index]: float(np.rad2deg(np.ptp(sampled[:, index])))
        for index in range(len(dof_names))
    }
    wrap_ratio = {
        dof_names[index]: float(continuity["wrap_over_interior"][index])
        for index in range(len(dof_names))
    }
    outliers = sorted(
        (name_ for name_, ratio in wrap_ratio.items() if ratio > 1.5),
        key=lambda name_: -wrap_ratio[name_],
    )

    axis_span: dict[str, Any] = {}
    roles = measured_axis_roles(plan)
    for joints in (ARM_JOINTS, LEG_JOINTS):
        for joint in joints:
            # dof_groups() names the chain ``<joint>``, ``<joint>__dof1``,
            # ``<joint>__dof2``; the split order is not the rotation-axis order,
            # so read the axes from the plan spec (primary axis last) instead of
            # parsing the DOF name.
            chain = [index for index, dof in enumerate(dof_names) if dof.split("__")[0] == joint]
            spec_axes = [plan.joints[index].axis for index in chain]
            if len(chain) != 3 or len(set(spec_axes)) != 3:
                continue
            for axis_index, axis in enumerate(spec_axes):
                index = chain[axis_index]
                span = float(np.rad2deg(np.ptp(sampled[:, index])))
                axis_span[f"{joint}__{axis}"] = {
                    "role": roles[dof_names[index]],
                    "dof_name": dof_names[index],
                    "is_primary": axis_index == 2,
                    "target_ptp_deg": span,
                    "limit_margin_deg": float(
                        min(
                            upper[index] - sampled[:, index].max(),
                            sampled[:, index].min() - lower[index],
                        )
                        * 180.0
                        / np.pi
                    ),
                }
    ratios = {}
    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        for axis in "xyz":
            left = axis_span.get(f"left_{joint}__{axis}")
            right = axis_span.get(f"right_{joint}__{axis}")
            if left is None or right is None:
                continue
            ratios[f"{joint}__{axis}"] = (
                0.0
                if left["target_ptp_deg"] <= 1e-9
                else right["target_ptp_deg"] / left["target_ptp_deg"]
            )

    shoulder_index = {
        side: [
            index for index, dof in enumerate(dof_names) if dof.startswith(f"{side}_shoulder__")
        ][0]
        for side in ("left", "right")
    }
    hip_index = {
        side: [index for index, dof in enumerate(dof_names) if dof.startswith(f"{side}_hip__")][0]
        for side in ("left", "right")
    }
    knee_index = {
        side: [index for index, dof in enumerate(dof_names) if dof.startswith(f"{side}_knee__")][0]
        for side in ("left", "right")
    }
    max_lag = max(len(sampled) // 4, 1)
    phase = {
        "shoulder_left_vs_right": {
            "zero_shift_correlation": _normalised_correlation(
                sampled[:, shoulder_index["left"]] - sampled[:, shoulder_index["left"]].mean(),
                sampled[:, shoulder_index["right"]] - sampled[:, shoulder_index["right"]].mean(),
            ),
            **_best_shift(
                sampled[:, shoulder_index["left"]] - sampled[:, shoulder_index["left"]].mean(),
                sampled[:, shoulder_index["right"]] - sampled[:, shoulder_index["right"]].mean(),
                max_lag,
            ),
        },
        "shoulder_left_vs_same_side_hip": _normalised_correlation(
            sampled[:, shoulder_index["left"]] - sampled[:, shoulder_index["left"]].mean(),
            sampled[:, hip_index["left"]] - sampled[:, hip_index["left"]].mean(),
        ),
        "shoulder_right_vs_same_side_hip": _normalised_correlation(
            sampled[:, shoulder_index["right"]] - sampled[:, shoulder_index["right"]].mean(),
            sampled[:, hip_index["right"]] - sampled[:, hip_index["right"]].mean(),
        ),
        "knee_left_vs_right": {
            "zero_shift_correlation": _normalised_correlation(
                sampled[:, knee_index["left"]] - sampled[:, knee_index["left"]].mean(),
                sampled[:, knee_index["right"]] - sampled[:, knee_index["right"]].mean(),
            ),
            **_best_shift(
                sampled[:, knee_index["left"]] - sampled[:, knee_index["left"]].mean(),
                sampled[:, knee_index["right"]] - sampled[:, knee_index["right"]].mean(),
                max_lag,
            ),
        },
    }

    return {
        "gait": name,
        "source": gait.provenance["source"],
        "source_metadata": gait.provenance["metadata"],
        "loop_correction": gait.provenance["loop_correction"],
        "endpoint_joint_difference_deg": gait.provenance["endpoint_joint_difference_deg"],
        "frame_count": int(reference.shape[0]),
        "duration_s": float(gait.duration_s),
        "speed_m_s": float(gait.speed_m_s),
        "samples": int(samples),
        "root_height_m": {
            "min": float(heights.min()),
            "max": float(heights.max()),
            "wrap_step_m": float(abs(heights[0] - heights[-1])),
            "interior_max_step_m": float(np.abs(np.diff(heights)).max()),
        },
        "dominant_axis_per_dof_deg": dominant,
        "wrap_over_interior_ratio": wrap_ratio,
        "wrap_outlier_dofs": outliers,
        "axis_span_deg": axis_span,
        "right_over_left_ratio": ratios,
        "loop_correction_rad": {
            dof_names[index]: {
                "correction_ptp_rad": float(np.ptp(correction[:, index])),
                "correction_ptp_deg": float(np.rad2deg(np.ptp(correction[:, index]))),
                "endpoint_rad": float(correction[-1, index] - correction[0, index]),
            }
            for index in range(len(dof_names))
        },
        "phase": phase,
    }


def trace_continuity_input(samples: np.ndarray) -> np.ndarray:
    return samples


def _raw_interpolate(reference: np.ndarray, phase: float) -> np.ndarray:
    """Linear interpolation across the cycle *without* the endpoint correction."""

    point = phase * (len(reference) - 1)
    index = min(int(point), len(reference) - 2)
    fraction = point - index
    return (1 - fraction) * reference[index] + fraction * reference[index + 1]


def render(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaits = [row for row in report["gaits"]]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_amp, ax_ratio, ax_seam, ax_phase) = axes.ravel()

    labels = [row["gait"] for row in gaits]
    positions = np.arange(len(labels))
    arm_axes = ["shoulder__x", "shoulder__y", "shoulder__z"]
    for offset, key in enumerate(arm_axes):
        ax_amp.bar(
            positions + (offset - 1) * 0.25,
            [row["axis_span_deg"][f"left_{key}"]["target_ptp_deg"] for row in gaits],
            0.23,
            label=f"left {key}",
        )
        ax_amp.bar(
            positions + (offset - 1) * 0.25,
            [
                -row["axis_span_deg"].get(f"right_{key}", {}).get("target_ptp_deg", 0.0)
                for row in gaits
            ],
            0.23,
            color="tab:red",
            alpha=0.7,
        )
    ax_amp.axhline(0.0, color="black", linewidth=0.8)
    ax_amp.set_xticks(positions)
    ax_amp.set_xticklabels(labels)
    ax_amp.set_ylabel("peak-to-peak target angle (deg)")
    ax_amp.set_title("shoulder axis span, left (+) vs right (-)")
    ax_amp.legend(fontsize=8)

    ratio_keys = sorted(gaits[0]["right_over_left_ratio"])
    width = 0.8 / max(len(gaits), 1)
    for index, row in enumerate(gaits):
        ax_ratio.bar(
            np.arange(len(ratio_keys)) + index * width,
            [row["right_over_left_ratio"][key] for key in ratio_keys],
            width,
            label=row["gait"],
        )
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax_ratio.set_xticks(np.arange(len(ratio_keys)) + width * (len(gaits) - 1) / 2)
    ax_ratio.set_xticklabels(ratio_keys, rotation=60, ha="right", fontsize=8)
    ax_ratio.set_ylabel("right / left peak-to-peak")
    ax_ratio.set_title("symmetry of the sampled control target")
    ax_ratio.legend(fontsize=8)

    for row in gaits:
        ratios = np.asarray(list(row["wrap_over_interior_ratio"].values()))
        ax_seam.hist(
            ratios,
            bins=np.linspace(0, max(2.5, ratios.max() + 0.1), 30),
            alpha=0.5,
            label=row["gait"],
        )
    ax_seam.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax_seam.axvline(1.5, color="tab:red", linestyle=":", linewidth=1.0)
    ax_seam.set_xlabel("wrap step / largest interior step")
    ax_seam.set_ylabel("DOF count")
    ax_seam.set_title("loop seam continuity (< 1 = seam is not an outlier)")
    ax_seam.legend(fontsize=8)

    for row in gaits:
        phase = row["phase"]["shoulder_left_vs_right"]
        ax_phase.bar(
            row["gait"],
            phase["zero_shift_correlation"],
            color="tab:blue",
            alpha=0.6,
            label="zero shift" if row is gaits[0] else None,
        )
        ax_phase.bar(
            row["gait"],
            phase["correlation"],
            color="tab:orange",
            alpha=0.4,
            label="best lag" if row is gaits[0] else None,
        )
    ax_phase.axhline(-1.0, color="black", linestyle="--", linewidth=0.8, label="antiphase")
    ax_phase.set_ylabel("normalised correlation")
    ax_phase.set_title("left vs right shoulder (antiphase is expected)")
    ax_phase.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/humans/gait_seam_audit")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 10:
        parser.error("samples must be at least 10")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    settings, config, plan = _load_plan(args.config)
    dt = config.simulation.physics_dt_s
    gaits = {
        key: load_gait(
            spec, plan, dt_s=dt, max_stance_slip_m_s=settings.get("max_stance_slip_m_s")
        )
        for key, spec in settings["gaits"].items()
    }
    rows = [
        analyse_gait(key, gait, plan, args.samples, config.export.surface_point_seed)
        for key, gait in gaits.items()
    ]

    report = {
        "rig": str(settings["rig"]),
        "keyboard_config": str(args.config),
        "keyboard_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "physics_dt_s": dt,
        "seed": config.export.surface_point_seed,
        "samples_per_cycle": args.samples,
        "axis_role_method": (
            f"role measured per DOF: {PROBE_DEG:.0f} deg about the declared axis, then the "
            "direction the distal joint actually travels in the body frame (+X forward, "
            "+Y left, +Z up). An axis letter carries no fixed anatomical role: each chain "
            "declares its own rotation order and the bone's rest orientation differs."
        ),
        "gaits": rows,
    }
    write_json(args.out / "gait_seam_audit.json", report)
    if not args.no_plot:
        render(report, args.out / "gait_seam_audit.png")
    for row in rows:
        LOGGER.info(
            "%s: seam outliers %d dof, left/right shoulder-x ratio %.4f, "
            "shoulder antiphase corr %.4f",
            row["gait"],
            len(row["wrap_outlier_dofs"]),
            row["right_over_left_ratio"].get("shoulder__x", float("nan")),
            row["phase"]["shoulder_left_vs_right"]["zero_shift_correlation"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
