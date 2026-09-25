#!/usr/bin/env python3
"""Check the reference's support mask against the contact PhysX actually produced.

``load_gait`` decides stance from horizontal ankle travel in forward kinematics -- a model
of contact, evaluated before the simulation runs, with no access to the simulation.
``docs/support-mask-audit.md`` measured that model against geometry and found 44% of its
"support" frames have the foot off the floor. Geometry is still a model. This checks it
against the only thing that settles the question: the contact report from a real run.

Per frame of an existing Isaac session, per foot, three independent statements are read:

- **claimed**  -- ``gait.supporting_feet(gait_phase)``, the mask's own answer.
- **contacted** -- PhysX reported this ankle colliding with the floor above the impulse
  threshold. Simulator ground truth.
- **height**   -- foot capsule bottom from forward kinematics of the *recorded* joint
  states. The recorded angles are physics state, so this is the body the simulator was
  actually integrating, not the reference it was chasing.

``contacted`` is what the anchor needs. ``height`` is there to attribute a disagreement:
claimed-but-not-contacted with the foot low is a marginal contact, while
claimed-but-not-contacted with the foot high means the mask was simply wrong about where
the foot was.

Read-only: consumes an existing run's ``control.npz`` and ``contacts.json``. No Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
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
from sim2sense_fall.humans.contact_control import capsule_bottom, fit_collision_capsules
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints
from sim2sense_fall.humans.rig import (
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)
from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix
from sim2sense_fall.humans.teleop import load_gait, load_keyboard_config

LOGGER = logging.getLogger("audit_mask_vs_contact")

SIDES = ("left", "right")
WALKING_MODES = {"forward": "forward", "backward": "backward"}


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
            plan, mesh, margin_m=settings["collision_fit"]["margin_m"]
        )
    return plan


def contact_presence(
    contacts: list[dict[str, Any]], threshold_ns: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, per foot: is the ankle touching the floor, and with what normal impulse."""

    present = np.zeros((len(contacts), 2), dtype=bool)
    impulse = np.zeros((len(contacts), 2), dtype=np.float64)
    for index, frame in enumerate(contacts):
        for sample in frame.get("samples", ()):
            paths = (sample.get("collider0_path") or "", sample.get("collider1_path") or "")
            if not any("/Human/" in path for path in paths):
                continue
            body = next((path for path in paths if "/Human/" in path), "")
            part = body.split("/")[3] if len(body.split("/")) > 3 else ""
            if part not in ("left_ankle", "right_ankle"):
                continue
            normal = np.asarray(sample["normal"], dtype=np.float64)
            length = float(np.linalg.norm(normal))
            if length <= 0:
                continue
            normal /= length
            value = max(0.0, float(np.asarray(sample["impulse_ns"], dtype=np.float64) @ normal))
            column = 0 if part == "left_ankle" else 1
            impulse[index, column] += value
            if value > threshold_ns:
                present[index, column] = True
    return present, impulse / dt


def actual_foot_height(plan: Any, control: Any) -> np.ndarray:
    """Foot capsule bottom per frame from the recorded physics joint states."""

    roots = np.asarray(control["root"], dtype=np.float64)
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    angles = np.asarray(control["joints"], dtype=np.float64)
    names = list(plan.dof_names)
    height = np.zeros((angles.shape[0], 2), dtype=np.float64)
    for index in range(angles.shape[0]):
        poses = forward_kinematics(
            plan,
            dict(zip(names, angles[index], strict=True)),
            root_position=roots[index],
            root_rotation=matrix_to_axis_angle(quaternion_to_matrix(quats[index])),
        )
        for column, side in enumerate(SIDES):
            height[index, column] = capsule_bottom(plan, poses, f"{side}_ankle")
    return height


def audit(plan: Any, settings: dict[str, Any], run: Path, weight_floor: float) -> dict[str, Any]:
    control = np.load(run / "control.npz", allow_pickle=True)
    contacts = json.loads((run / "contacts.json").read_text(encoding="utf-8"))
    dt = float(settings.get("physics_dt_s") or 1.0 / 120.0)
    threshold = float(settings["slip_min_impulse_ns"])

    gait_plan = {
        key: load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=settings.get("max_stance_slip_m_s"))
        for key, spec in settings["gaits"].items()
    }
    presence, normal_force = contact_presence(contacts, threshold, dt)
    height = actual_foot_height(plan, control)

    mode = np.asarray(control["mode"]).astype(str)
    phase = np.asarray(control["gait_phase"], dtype=np.float64)
    weight = np.asarray(control["gait_weight"], dtype=np.float64)

    claimed = np.zeros((len(contacts), 2), dtype=bool)
    considered = np.zeros(len(contacts), dtype=bool)
    for index in range(len(contacts)):
        key = WALKING_MODES.get(mode[index])
        if key is None or not np.isfinite(weight[index]) or weight[index] < weight_floor:
            continue
        considered[index] = True
        feet = gait_plan[key].supporting_feet(float(phase[index])) or set()
        for column, side in enumerate(SIDES):
            claimed[index, column] = f"{side}_ankle" in feet

    report: dict[str, Any] = {
        "run": str(run),
        "frames_total": int(len(contacts)),
        "frames_considered": int(considered.sum()),
        "weight_floor": weight_floor,
        "contact_threshold_ns": threshold,
        "modes_seen": {
            str(name): int(count)
            for name, count in zip(*np.unique(mode, return_counts=True), strict=True)
        },
    }
    per_foot: dict[str, Any] = {}
    for column, side in enumerate(SIDES):
        keep = considered
        claim = claimed[keep, column]
        touch = presence[keep, column]
        both = int((claim & touch).sum())
        claim_only = int((claim & ~touch).sum())
        touch_only = int((~claim & touch).sum())
        neither = int((~claim & ~touch).sum())
        low_when_claimed_only = (
            float(np.median(height[keep, column][claim & ~touch]))
            if claim_only
            else None
        )
        per_foot[side] = {
            "claimed_frames": int(claim.sum()),
            "contacted_frames": int(touch.sum()),
            "claimed_and_contacted": both,
            "claimed_without_contact": claim_only,
            "contact_without_claim": touch_only,
            "neither": neither,
            "share_of_claims_the_simulator_contradicts": float(
                claim_only / max(both + claim_only, 1)
            ),
            "share_of_real_contacts_the_mask_misses": float(
                touch_only / max(both + touch_only, 1)
            ),
            "median_foot_height_when_claimed_without_contact_m": low_when_claimed_only,
            "median_contact_normal_force_n": float(
                np.median(normal_force[keep, column][touch])
            )
            if touch.any()
            else None,
        }
    report["per_foot"] = per_foot

    # A frame where the mask names a supporting foot and the simulator reports none is the
    # failure the anchor cares about; so is a frame with real load and no claim.
    report["summary"] = {
        "share_of_claimed_support_physx_contradicts": float(
            sum(per_foot[side]["claimed_without_contact"] for side in SIDES)
            / max(sum(per_foot[side]["claimed_frames"] for side in SIDES), 1)
        ),
        "share_of_real_contacts_missed_by_the_mask": float(
            sum(per_foot[side]["contact_without_claim"] for side in SIDES)
            / max(sum(per_foot[side]["contacted_frames"] for side in SIDES), 1)
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--weight-floor", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not 0.0 <= args.weight_floor <= 1.0:
        parser.error("weight floor must be within [0, 1]")

    settings = load_keyboard_config(args.config, REPO_ROOT)
    plan = build_plan(settings)
    results = [audit(plan, settings, run, args.weight_floor) for run in args.run]
    for report in results:
        print(f"\n=== {report['run']} ===")
        print(
            f"  frames considered {report['frames_considered']} of {report['frames_total']} "
            f"(mode in forward/backward, gait_weight >= {report['weight_floor']})"
        )
        print(
            f"  {'':18s} claimed   contacted   both   claim-only   contact-only"
        )
        for side in SIDES:
            row = report["per_foot"][side]
            print(
                f"  {side:18s} {row['claimed_frames']:7d}   {row['contacted_frames']:9d}   "
                f"{row['claimed_and_contacted']:4d}   "
                f"{row['claimed_without_contact']:10d}   {row['contact_without_claim']:12d}"
            )
        summary = report["summary"]
        print(
            f"  PhysX contradicts the mask on "
            f"{summary['share_of_claimed_support_physx_contradicts'] * 100:.1f}% of claimed "
            f"support; the mask misses "
            f"{summary['share_of_real_contacts_missed_by_the_mask'] * 100:.1f}% of real contacts"
        )
        for side in SIDES:
            row = report["per_foot"][side]
            if row["median_foot_height_when_claimed_without_contact_m"] is not None:
                print(
                    f"     {side:5s} when claimed without contact, the foot sat at "
                    f"{row['median_foot_height_when_claimed_without_contact_m'] * 1000:+.1f} mm"
                )
    if args.out is not None:
        write_json(args.out, results)
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
