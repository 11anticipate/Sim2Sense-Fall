#!/usr/bin/env python3
"""Independent mesh displacement and contact-point target-motion slip diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from common import write_json
from keyboard import prepare

from sim2sense_fall.humans.rig import forward_kinematics
from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix


def diagnose(run: Path, config: Path) -> dict:
    settings, _, plan, mesh, controller = prepare(config)
    data = dict(np.load(run / "control.npz", allow_pickle=False))
    contacts = json.loads((run / "contacts.json").read_text())
    report = json.loads((run / "report.json").read_text())
    recording = np.load(run / "recording.npz", allow_pickle=False)
    if not np.allclose(recording["time_s"], data["time_s"], rtol=0, atol=1e-8):
        raise ValueError("diagnosis requires simultaneous physics-rate actual mesh")
    if list(data["dof_names"]) != list(plan.dof_names):
        raise ValueError("recorded DOF ordering differs from configured rig")
    vertices = recording["mesh_vertices_xyz"]
    owners = np.asarray(mesh.topology.joint_names)[mesh.weights.argmax(axis=1)]
    foot_vertices = {side: np.flatnonzero(np.isin(owners, [f"{side}_ankle", f"{side}_foot"]))
                     for side in ("left", "right")}
    actual, target = [], []
    for i in range(len(data["time_s"])):
        root_rotation = matrix_to_axis_angle(quaternion_to_matrix(data["root_quaternion"][i]))
        for output, key in ((actual, "joints"), (target, "joint_target")):
            poses = forward_kinematics(
                plan, dict(zip(plan.dof_names, data[key][i], strict=True)),
                root_position=data["root"][i], root_rotation=root_rotation)
            output.append({side: poses[f"{side}_ankle"] for side in ("left", "right")})
    observations = []
    for i in range(1, len(contacts)):
        mode = str(data["mode"][i])
        if mode not in {"forward", "backward"} or data["reset_id"][i] != data["reset_id"][i-1]:
            continue
        dt = float(data["time_s"][i] - data["time_s"][i-1])
        if abs(contacts[i]["time_s"] - data["time_s"][i]) > 1e-8:
            raise ValueError("contact/control timestamps differ")
        samples = [s for s in contacts[i]["samples"]
                   if s["collider1_path"].endswith("/floor")
                   and s["collider0_path"].split("/")[-2] in {"left_ankle", "right_ankle"}
                   and s["impulse_magnitude_ns"] >= settings["slip_min_impulse_ns"]]
        slips = contacts[i]["floor_contact_slips_m_s"]
        if len(samples) != len(slips):
            raise ValueError("contact/slip sample alignment differs")
        for sample, slip in zip(samples, slips, strict=True):
            side = sample["collider0_path"].split("/")[-2].split("_")[0]
            point = np.asarray(sample["position_m"])
            pose = actual[i][side]
            local_point = pose.rotation.T @ (point - pose.translation)
            ids = foot_vertices[side]
            vertex = ids[np.argmin(np.linalg.norm(vertices[i, ids] - point, axis=1))]
            # Same material vertex/rigid local point in successive measured states.
            skin_v = (vertices[i, vertex] - vertices[i-1, vertex]) / dt
            actual_v = (pose.transform_point(local_point)
                        - actual[i-1][side].transform_point(local_point)) / dt
            target_v = (target[i][side].transform_point(local_point)
                        - target[i-1][side].transform_point(local_point)) / dt
            gait = controller.gaits[mode]
            model_support = f"{side}_ankle" in gait.supporting_feet(data["gait_phase"][i])
            observations.append({
                "time_s": float(data["time_s"][i]), "mode": mode, "side": side,
                "phase": float(data["gait_phase"][i]), "model_support": model_support,
                "steady": bool(data["gait_weight"][i] >= .95),
                "turning": bool(data["command"][i, 1] != 0),
                "impulse_ns": sample["impulse_magnitude_ns"], "slip": float(slip),
                "skin_speed": float(np.linalg.norm(skin_v[:2])),
                "fk_speed": float(np.linalg.norm(actual_v[:2])),
                "target_speed": float(np.linalg.norm(target_v[:2])),
                "tracking_speed": float(np.linalg.norm((actual_v-target_v)[:2])),
                "foot_min_z": float(recording["foot_min_z_m"][i, int(side == "right")]),
                "point_to_skin": float(np.linalg.norm(vertices[i, vertex] - point)),
            })
    summaries = {}
    for mode in ("forward", "backward"):
        for group in ("all", "steady", "model_stance", "model_swing", "impact"):
            rows = [r for r in observations if r["mode"] == mode and (
                group == "all" or (group == "steady" and r["steady"] and not r["turning"])
                or (group == "model_stance" and r["model_support"])
                or (group == "model_swing" and not r["model_support"])
                or (group == "impact" and r["impulse_ns"] >= 1.))]
            if not rows:
                continue
            summaries[f"{mode}/{group}"] = {
                "samples": len(rows),
                **{key: np.percentile([r[key] for r in rows], [50, 95]).tolist()
                   for key in ("slip", "skin_speed", "fk_speed", "target_speed", "tracking_speed",
                               "foot_min_z", "point_to_skin", "impulse_ns")},
            }
    return {
        "summary": summaries, "observations": observations,
        "source_report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "seed": report["seed"], "scene_split": report["scene_split"],
        "metric": "p50,p95; m/s unless named otherwise; actual mesh is independent of FK proxy",
        "target_scope": "commanded joints at measured root; tracking is a velocity vector residual",
        "support_scope": (
            "model_support is the AMASS reference-cycle mask, not the live contact planner's "
            "independent step schedule; physical support is always from recorded contacts"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, diagnose(args.run, args.config))


if __name__ == "__main__":
    main()
