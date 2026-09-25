#!/usr/bin/env python3
"""Attribute the measured floor slip of stance feet to its three physical sources.

The slip gate (p95 tangent speed of contacting ankle points <= 0.15 m/s) fails on
every shipped session, but "it fails" is not actionable: the recorded slip is the
sum of three independent terms with three different owners.

For every physics step where an ankle is in floor contact (normal impulse above
the slip threshold), the ankle's world velocity is differentiated from forward
kinematics of the recorded state, and split as::

    d(p_act)/dt   =  v_root + d(rel_act)/dt                    (actual, what is graded)
    d(p_tgt)/dt   =  v_root + d(rel_tgt)/dt                    (joints tracked perfectly)
    v_cmd*heading + d(rel_ref)/dt                              (root held at command speed
                                                                and reference tracked exactly)

which yields

- **root_fluctuation** -- the actual root deviating from the commanded velocity
  (root-assist spring lag/overshoot). Owner: root_control.
- **reference_stance_slide** -- the reference's own stance foot motion relative to
  the root, minus the ideal "-v_cmd" that a planted foot must have. Owner: the
  AMASS window / cadence choice, measurable on CPU before any simulation.
- **joint_tracking** -- actual minus target relative foot motion. Owner: PD
  gains, joint speed limits, stance-IK corrections.

The three terms add up to the actual by construction (up to angular-velocity of
the root, ignored here: a straight walk pitches/rolls by fractions of a degree,
and the recorded slip it is compared against uses the same COM+omega definition
only in the shipped report, while this tool grades the ankle point directly).

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
from sim2sense_fall.humans.teleop import load_gait, load_keyboard_config

LOGGER = logging.getLogger("audit_slip_attribution")

SIDES = ("left", "right")
DIRECTIONS = ("forward", "backward")


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


def ankle_world_positions(
    plan: Any, control: Any, column: str, frames: np.ndarray
) -> np.ndarray:
    """(frames, 2, 3) world ankle positions from FK of the given joint column."""

    roots = np.asarray(control["root"], dtype=np.float64)
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    angles = np.asarray(control[column], dtype=np.float64)
    names = list(plan.dof_names)
    out = np.zeros((len(frames), 2, 3), dtype=np.float64)
    for row, index in enumerate(frames):
        rotation = quaternion_to_matrix(quats[index])
        poses = forward_kinematics(
            plan,
            dict(zip(names, angles[index], strict=True)),
            root_position=roots[index],
            root_rotation=matrix_to_axis_angle(rotation),
        )
        for foot, side in enumerate(SIDES):
            out[row, foot] = poses[f"{side}_ankle"].translation
    return out


def contacting_frames(contacts: list[dict[str, Any]], threshold_ns: float) -> np.ndarray:
    """Per frame, per foot: is the ankle contacting the floor above threshold."""

    present = np.zeros((len(contacts), 2), dtype=bool)
    for index, frame in enumerate(contacts):
        for sample in frame.get("samples", ()):
            paths = (sample.get("collider0_path") or "", sample.get("collider1_path") or "")
            body = next((path for path in paths if "/Human/" in path), "")
            part = body.split("/")[3] if len(body.split("/")) > 3 else ""
            if part not in ("left_ankle", "right_ankle"):
                continue
            normal = np.asarray(sample["normal"], dtype=np.float64)
            length = float(np.linalg.norm(normal))
            if length <= 0:
                continue
            impulse = float(np.asarray(sample["impulse_ns"], dtype=np.float64) @ normal)
            if max(0.0, impulse / length) > threshold_ns:
                present[index, 0 if part == "left_ankle" else 1] = True
    return present


def speed_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(values.mean()),
    }


def audit_run(run: Path) -> dict[str, Any]:
    settings = load_keyboard_config(run / "keyboard.yaml", REPO_ROOT)
    plan = build_plan(settings)
    control = np.load(run / "control.npz", allow_pickle=False)
    contacts = json_load(run / "contacts.json")
    dt = float(settings.get("physics_dt_s") or 1.0 / 120.0)
    threshold = float(settings["slip_min_impulse_ns"])
    command_speed = float(settings["speed_m_s"])

    gaits = {
        key: load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=None)
        for key, spec in settings["gaits"].items()
    }
    touching = contacting_frames(contacts, threshold)
    mode = np.asarray(control["mode"]).astype(str)
    weight = np.asarray(control["gait_weight"], dtype=np.float64)
    time_s = np.asarray(control["time_s"], dtype=np.float64)
    root_vel = np.asarray(control["velocity"], dtype=np.float64)

    act = ankle_world_positions(plan, control, "joints", np.arange(len(time_s)))
    tgt = ankle_world_positions(plan, control, "joint_target", np.arange(len(time_s)))
    vel_act = np.gradient(act, dt, axis=0)
    vel_tgt = np.gradient(tgt, dt, axis=0)

    heading_dir = np.zeros((len(time_s), 2))
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    for index in range(len(time_s)):
        rotation = quaternion_to_matrix(quats[index])
        forward = rotation[:, 0][:2]
        norm = float(np.linalg.norm(forward))
        heading_dir[index] = forward / norm if norm > 1e-9 else np.array([0.0, 1.0])

    report: dict[str, Any] = {"run": str(run), "command_speed_m_s": command_speed,
                              "dt_s": dt, "directions": {}}
    for direction in DIRECTIONS:
        gait = gaits[direction]
        sign = 1.0 if direction == "forward" else -1.0
        # Reference stance slide, computed on the gait alone: with the root held at
        # command speed, how fast does the masked-stance foot move in the world?
        ref_slide: dict[str, np.ndarray] = {side: [] for side in SIDES}
        dphi = 1.0 / (len(gait.joints) - 1)
        for row in range(len(gait.joints) - 1):
            if not gait.support_mask[row].any():
                continue
            for foot, side in enumerate(SIDES):
                if not gait.support_mask[row, foot]:
                    continue
                q0, h0 = gait.sample(row * dphi)
                q1, h1 = gait.sample((row + 1) * dphi)
                name = list(plan.dof_names)
                p0 = forward_kinematics(
                    plan, dict(zip(name, q0, strict=True)),
                    root_position=(0, 0, h0), root_rotation=gait.tilt(row * dphi),
                )[f"{side}_ankle"].translation
                p1 = forward_kinematics(
                    plan, dict(zip(name, q1, strict=True)),
                    root_position=(0, 0, h1), root_rotation=gait.tilt((row + 1) * dphi),
                )[f"{side}_ankle"].translation
                rel_speed = (p1[0] - p0[0]) / (gait.duration_s / (len(gait.joints) - 1))
                # world speed if the root advanced at exactly the commanded speed
                ref_slide[side].append(abs(command_speed + sign * rel_speed))
        ref_stats = {
            side: speed_stats(np.asarray(ref_slide[side])) for side in SIDES
        }

        per_term: dict[str, list[float]] = {
            f"{side}_{term}": [] for side in SIDES
            for term in ("actual", "root_fluctuation", "reference_slide", "joint_tracking")
        }
        walking = np.isin(mode, DIRECTIONS) & (weight >= 0.95)
        for index in range(1, len(time_s) - 1):
            if mode[index] != direction or not walking[index]:
                continue
            for foot, side in enumerate(SIDES):
                if not touching[index, foot]:
                    continue
                v_actual = vel_act[index, foot][:2]
                v_target = vel_tgt[index, foot][:2]
                v_command = command_speed * sign * heading_dir[index]
                per_term[f"{side}_actual"].append(float(np.linalg.norm(v_actual)))
                per_term[f"{side}_root_fluctuation"].append(
                    float(np.linalg.norm(root_vel[index, :2] - v_command))
                )
                per_term[f"{side}_reference_slide"].append(
                    float(np.linalg.norm(v_target - v_command))
                )
                per_term[f"{side}_joint_tracking"].append(
                    float(np.linalg.norm(v_actual - v_target))
                )
        directions_report: dict[str, Any] = {
            "contact_frames": {
                side: int((touching[:, foot] & walking).sum())
                for foot, side in enumerate(SIDES)
            },
            "terms": {
                key: speed_stats(np.asarray(values))
                for key, values in per_term.items() if values
            },
            "reference_stance_slide_gait_only": ref_stats,
        }
        report["directions"][direction] = directions_report
    return report


def json_load(path: Path) -> list[dict[str, Any]]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = [audit_run(run) for run in args.run]
    for report in results:
        print(f"\n=== {report['run']} ===")
        for direction, stats in report["directions"].items():
            if not stats["terms"]:
                continue
            print(f"  {direction}: contact frames "
                  f"{stats['contact_frames']}")
            print(f"    {'term':34s} {'p50':>7s} {'p95':>7s} {'mean':>7s}   (m/s)")
            for key, values in stats["terms"].items():
                print(f"    {key:34s} {values['p50']:7.3f} {values['p95']:7.3f} "
                      f"{values['mean']:7.3f}")
            for side, values in stats["reference_stance_slide_gait_only"].items():
                print(f"    gait-only stance slide {side:5s}      "
                      f"{values['p50']:7.3f} {values['p95']:7.3f} {values['mean']:7.3f}")
    if args.out is not None:
        write_json(args.out, results)
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
