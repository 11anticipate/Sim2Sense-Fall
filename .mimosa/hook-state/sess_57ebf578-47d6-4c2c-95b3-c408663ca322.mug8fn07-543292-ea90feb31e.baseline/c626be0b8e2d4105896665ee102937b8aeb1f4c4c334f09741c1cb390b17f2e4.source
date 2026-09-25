#!/usr/bin/env python3
"""P0-A: left/right arm comparison for the keyboard gait.

The user reported that during manual keyboard walking the left arm swings
normally while the right arm looks wrong. The report is about the *visible*
swing, so this tool compares every stage the swing passes through, on one page:

1. the raw AMASS local rotations of each arm joint,
2. the retargeted clip channels the rig reads,
3. the multi-axis DOF reference the controller feeds PhysX, with limit binding,
4. the hand trajectory from rig forward kinematics (amplitude and left/right
   phase, in the direction the eye actually sees), and
5. a side-by-side skeleton plot of both arms at matched gait phases.

It is read-only: nothing here changes the controller or the configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_MOTIONS, REPO_ROOT, load_inputs, write_json  # noqa: E402

from sim2sense_fall.humans.amass import (  # noqa: E402
    crop_amass_clip,
    load_amass_clip,
    normalize_root_motion,
)
from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)
from sim2sense_fall.humans.teleop import (  # noqa: E402
    TeleopController,
    load_gait,
    load_keyboard_config,
)

DEG = 180.0 / np.pi

ARM_PAIRS = (
    ("shoulder", "left_shoulder", "right_shoulder"),
    ("elbow", "left_elbow", "right_elbow"),
    ("wrist", "left_wrist", "right_wrist"),
    ("collar", "left_collar", "right_collar"),
)

#: Pipeline body axes, in the order a rotation vector stores them.
AXIS_NAMES = ("x", "y", "z")
#: What each pipeline axis means for a limb in this rig (see the rig config header).
AXIS_ROLE = {
    "x": "flexion/extension (sagittal, the walking swing)",
    "y": "abduction/adduction (frontal)",
    "z": "axial twist",
}


def _degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.asarray(values, dtype=np.float64))


def mirror_vector(vector: np.ndarray) -> np.ndarray:
    """Mirror a body-frame rotation vector across the sagittal plane (+Y left).

    Conjugating a rotation by the reflection ``diag(1, -1, 1)`` maps the rotation
    vector ``v`` to ``(-v_x, v_y, -v_z)``: components along the mirrored axes
    (x, z) flip, the component about the mirror normal (y) does not.
    """

    return np.array([-vector[0], vector[1], -vector[2]], dtype=np.float64)


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    angle = norm
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(angle)
        + cross * np.sin(angle)
        + np.outer(axis, axis) * (1.0 - np.cos(angle))
    )


def angle_between_vectors(left: np.ndarray, right: np.ndarray) -> float:
    """Geodesic angle in degrees between two axis-angle rotation vectors."""

    delta = _rodrigues(right) @ _rodrigues(left).T
    trace = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _span(block: np.ndarray) -> dict[str, float]:
    """Per-axis peak-to-peak spread of a rotation-vector series, in degrees."""

    spread = _degrees(block).ptp(axis=0)
    return {f"{axis}_ptp_deg": float(spread[index]) for index, axis in enumerate(AXIS_NAMES)}


def raw_rotation_report(clip: Any) -> dict[str, Any]:
    """Mirror consistency of the retargeted local rotations, per arm pair.

    The left and right arms of a real person are not mirror images, so a non-zero
    deviation is not by itself a defect. What it does establish is whether the
    asymmetry the user sees exists in the source motion or was introduced later:
    a source motion whose two shoulders already differ explains a visually
    different swing without any code being wrong.
    """

    report: dict[str, Any] = {}
    for label, left_name, right_name in ARM_PAIRS:
        if left_name not in clip.joint_names or right_name not in clip.joint_names:
            report[label] = {"present": False}
            continue
        left = np.asarray(
            [clip.rotation_of(frame, left_name) for frame in range(clip.frame_count)],
            dtype=np.float64,
        )
        right = np.asarray(
            [clip.rotation_of(frame, right_name) for frame in range(clip.frame_count)],
            dtype=np.float64,
        )
        mirrored = np.stack([mirror_vector(row) for row in left])
        deviations = np.array(
            [angle_between_vectors(m, r) for m, r in zip(mirrored, right, strict=True)]
        )
        left_span, right_span = _span(left), _span(right)
        swing_axis = "x"
        report[label] = {
            "present": True,
            "left": left_span,
            "right": right_span,
            "swing_amplitude_ratio_right_over_left": (
                float(right_span[f"{swing_axis}_ptp_deg"])
                / float(left_span[f"{swing_axis}_ptp_deg"])
                if left_span[f"{swing_axis}_ptp_deg"] > 1e-9
                else None
            ),
            "mirror_deviation_deg_mean": float(deviations.mean()),
            "mirror_deviation_deg_max": float(deviations.max()),
        }
    return report


def applied_dof_report(plan: Any, gait: Any) -> dict[str, Any]:
    """Per-DOF range, endpoint drift and limit binding of the cyclic reference."""

    names = list(plan.dof_names)
    joints_deg = _degrees(gait.joints)
    endpoint_deg = _degrees(gait.joints[-1] - gait.joints[0])
    out: dict[str, Any] = {}
    for index, name in enumerate(names):
        joint = plan.joint(name)
        chain = [n for n in names if n.startswith(joint.chain_joint)]
        out[name] = {
            "chain_joint": joint.chain_joint,
            "axis": joint.axis,
            "axis_role": AXIS_ROLE[joint.axis],
            "is_primary_axis_of_chain": name == joint.chain_joint,
            "limits_deg": [joint.lower_deg, joint.upper_deg],
            "min_deg": float(joints_deg[:, index].min()),
            "max_deg": float(joints_deg[:, index].max()),
            "ptp_deg": float(np.ptp(joints_deg[:, index])),
            "endpoint_drift_deg": float(endpoint_deg[index]),
            "at_lower_limit": bool(joints_deg[:, index].min() <= joint.lower_deg + 1e-6),
            "at_upper_limit": bool(joints_deg[:, index].max() >= joint.upper_deg - 1e-6),
            "chain_axes": [plan.joint(n).axis for n in chain],
        }
    return out


def limb_trajectory(plan: Any, gait: Any) -> dict[str, Any]:
    """Hand path over one gait cycle from rig forward kinematics.

    The root is pinned at the origin with identity rotation, so the numbers
    describe the arm motion alone and are directly comparable between sides.
    """

    names = list(plan.dof_names)
    tracks: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for row in gait.joints:
        angles = dict(zip(names, row, strict=True))
        poses = forward_kinematics(plan, angles)
        for side in tracks:
            link = f"{side}_hand"
            if link in poses:
                tracks[side].append(poses[link].translation.copy())
    report: dict[str, Any] = {}
    series: dict[str, np.ndarray] = {}
    for side, points in tracks.items():
        if len(points) < 2:
            report[side] = {"present": False}
            continue
        array = np.asarray(points, dtype=np.float64)
        series[side] = array
        report[side] = {
            "present": True,
            "fore_aft_x_ptp_m": float(np.ptp(array[:, 0])),
            "lateral_y_ptp_m": float(np.ptp(array[:, 1])),
            "vertical_z_ptp_m": float(np.ptp(array[:, 2])),
            "min_z_m": float(array[:, 2].min()),
        }
    if "left" in series and "right" in series:
        left = series["left"][:, 0] - series["left"][:, 0].mean()
        right = series["right"][:, 0] - series["right"][:, 0].mean()
        scale = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlations = [
            float(np.dot(left, np.roll(right, shift)) / scale) if scale > 1e-12 else 0.0
            for shift in range(len(left))
        ]
        best = int(np.argmax(correlations))
        worst = int(np.argmin(correlations))
        report["phase_offset_fraction_of_cycle"] = round(best / len(left), 4)
        report["anticorrelation_fraction_of_cycle"] = round(worst / len(left), 4)
        report["correlation_at_zero_shift"] = round(correlations[0], 4)
        report["fore_aft_amplitude_ratio_right_over_left"] = float(
            np.ptp(series["right"][:, 0]) / max(np.ptp(series["left"][:, 0]), 1e-12)
        )
    return report


def controller_clip_report(
    controller: TeleopController, plan: Any, dt_s: float, cycles: int = 3
) -> dict[str, Any]:
    """Run the controller on CPU at constant forward command and count clipping."""

    names = list(plan.dof_names)
    lower, upper = controller.lower, controller.upper
    position = controller.position.copy()
    clipped: dict[str, int] = {name: 0 for name in names}
    frames = 0
    total_cycles = cycles * controller.gaits["forward"].duration_s
    while frames * dt_s < total_cycles + 1.0:
        before = controller.weight
        target = controller.advance((1.0, 0.0), dt_s, position, controller.heading)
        position = target.position
        if before > 0.99:
            for index, name in enumerate(names):
                value = target.joints[index]
                if value <= lower[index] + 1e-9 or value >= upper[index] - 1e-9:
                    clipped[name] += 1
            frames += 1
    return {
        "evaluated_frames": frames,
        "clipped_frames_by_dof": {name: count for name, count in clipped.items() if count},
    }


def recorded_run_report(path: Path, plan: Any) -> dict[str, Any]:
    """Per-DOF tracking error of a recorded PhysX run, for the arm DOFs."""

    with np.load(path, allow_pickle=True) as payload:
        joints = payload["joints"]
        targets = payload["joint_target"]
        mode = payload["mode"]
    mask = np.asarray([str(value) == "forward" for value in mode])
    if not mask.any():
        mask = np.ones(len(joints), dtype=bool)
    error_deg = _degrees(np.abs(joints[mask] - targets[mask]))
    names = list(plan.dof_names)
    out: dict[str, Any] = {
        "source": str(path.resolve()),
        "forward_frames": int(mask.sum()),
        "max_error_deg_all_dof": float(error_deg.max()),
    }
    worst = int(np.argmax(error_deg.max(axis=0)))
    out["worst_dof"] = {"name": names[worst], "max_error_deg": float(error_deg[:, worst].max())}
    arms: dict[str, Any] = {}
    for label, left_name, right_name in ARM_PAIRS:
        row: dict[str, Any] = {}
        for side, chain in (("left", left_name), ("right", right_name)):
            indices = [index for index, name in enumerate(names) if name.startswith(chain)]
            if not indices:
                continue
            block = error_deg[:, indices]
            row[side] = {
                "dofs": [names[index] for index in indices],
                "max_error_deg": [float(block[:, offset].max()) for offset in range(len(indices))],
                "mean_error_deg": [
                    float(block[:, offset].mean()) for offset in range(len(indices))
                ],
            }
        if row:
            arms[label] = row
    out["arms"] = arms
    return out


def render_arm_plot(report: dict[str, Any], gait: Any, plan: Any, out_dir: Path) -> str | None:
    """Side-by-side hand trajectory, one row per view direction, saved as PNG."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional evidence
        return None

    names = list(plan.dof_names)
    tracks: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for row in gait.joints:
        poses = forward_kinematics(plan, dict(zip(names, row, strict=True)))
        for side in tracks:
            tracks[side].append(poses[f"{side}_hand"].translation.copy())
    left = np.asarray(tracks["left"])
    right = np.asarray(tracks["right"])
    phase = np.linspace(0.0, 1.0, len(left))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, (index, label) in zip(
        axes,
        (
            (0, "fore-aft x (what the eye reads as swing)"),
            (1, "lateral y (abduction)"),
            (2, "vertical z"),
        ),
        strict=True,
    ):
        axis.plot(phase, left[:, index], lw=2.0, label="left hand")
        axis.plot(phase, right[:, index], lw=2.0, label="right hand")
        axis.set_xlabel("gait phase (cycles)")
        axis.set_ylabel("hand position (m)")
        axis.set_title(label)
        axis.grid(alpha=0.3)
    axes[0].legend(loc="best")
    figure.suptitle(
        "P0-A hand trajectory over one forward gait cycle, rig forward kinematics "
        "(root pinned at origin)"
    )
    figure.tight_layout()
    path = out_dir / "hand_trajectory.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)

    # The swing amplitude per axis, left against right, as a bar chart.
    amplitude = {
        "left": [float(np.ptp(left[:, index])) for index in range(3)],
        "right": [float(np.ptp(right[:, index])) for index in range(3)],
    }
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    positions = np.arange(3)
    axis.bar(positions - 0.19, amplitude["left"], 0.38, label="left")
    axis.bar(positions + 0.19, amplitude["right"], 0.38, label="right")
    for offset, side in ((-0.19, "left"), (0.19, "right")):
        for index, value in enumerate(amplitude[side]):
            axis.text(index + offset, value + 0.006, f"{value:.2f}", ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(["x fore-aft", "y lateral", "z vertical"])
    axis.set_ylabel("hand excursion over one cycle (m)")
    axis.set_title("P0-A hand excursion by axis: left vs right")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    second = out_dir / "hand_excursion.png"
    figure.savefig(second, dpi=130)
    plt.close(figure)
    return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/humans/arm_swing_audit")
    parser.add_argument(
        "--recording",
        type=Path,
        default=None,
        help="control.npz from a keyboard run, for PhysX-vs-target comparison",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip the matplotlib evidence figures"
    )
    args = parser.parse_args()

    settings = load_keyboard_config(args.config, REPO_ROOT)
    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    plan = plan_human_rig(
        config,
        rest=fit_rest_skeleton(config, body.model.mesh().rest_skeleton()),
        spawn_xy=tuple(settings["spawn_xy"]),
    )
    dt = config.simulation.physics_dt_s

    gaits = {key: load_gait(spec, plan, dt_s=dt) for key, spec in settings["gaits"].items()}
    idle = load_gait(settings["idle"], plan, dt_s=dt)
    controller = TeleopController(
        settings["controller"], gaits, idle, plan, np.deg2rad(settings["heading_deg"])
    )

    forward_spec = settings["gaits"]["forward"]
    raw = load_amass_clip(forward_spec["file"])
    clip = normalize_root_motion(
        crop_amass_clip(
            raw,
            start_s=forward_spec["start_s"],
            duration_s=forward_spec["duration_s"],
        )
    )

    report: dict[str, Any] = {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "rig": str(settings["rig"].resolve()),
        "source": str(forward_spec["file"].resolve()),
        "source_interval_s": [forward_spec["start_s"], forward_spec["duration_s"]],
        "source_metadata": dict(clip.metadata),
        "physics_dt_s": dt,
        "gait_frames": int(gaits["forward"].joints.shape[0]),
        "gait_duration_s": float(gaits["forward"].duration_s),
        "axis_role": AXIS_ROLE,
        "source_local_rotations": raw_rotation_report(clip),
        "applied_dof": applied_dof_report(plan, gaits["forward"]),
        "hand_trajectory": limb_trajectory(plan, gaits["forward"]),
        "controller_clipping": controller_clip_report(controller, plan, dt),
        "endpoint_drift_max_deg": float(
            np.abs(_degrees(gaits["forward"].joints[-1] - gaits["forward"].joints[0])).max()
        ),
    }
    if args.recording is not None and args.recording.is_file():
        report["recorded_run"] = recorded_run_report(args.recording, plan)

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        plot = render_arm_plot(report, gaits["forward"], plan, args.out)
        if plot:
            report["figures"] = {"hand_trajectory": plot}
    write_json(args.out / "arm_swing_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
