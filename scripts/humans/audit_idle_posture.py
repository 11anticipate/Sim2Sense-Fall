"""Quantify why the idle (standing) pose has bent legs under the contact planner.

The walking contact cycle caps the root height so the leg still reaches the floor
at the stride's extreme fore-aft offset (0.5 * cycle_distance * stance_fraction).
``fit_contact_idle`` then re-fits the idle legs at exactly that capped height even
though a standing foot sits directly under its hip. Any leg length left over at
that reduced height has to go somewhere, and the IK spends it as knee flexion.

This script measures, on the real configs and clips:
  - the contact-cycle root height cap and its straight-leg (standing) equivalent,
  - the shipped idle pose's knee flexion and foot clearance,
  - what the idle pose looks like when fit at the straight-leg height instead.

CPU only: no Isaac Sim, no Sionna, no simulator state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.contact_control import capsule_bottom
from sim2sense_fall.humans.contact_gait import (
    ContactGaitConfig,
    fit_contact_idle,
)
from sim2sense_fall.humans.rig import forward_kinematics, plan_human_rig
from sim2sense_fall.humans.teleop import load_gait

REPO_ROOT = Path(__file__).resolve().parents[2]


def knee_flexion_deg(plan, q: np.ndarray, side: str) -> float:
    """Angle at the knee between thigh and shin, 180 = perfectly straight."""
    poses = forward_kinematics(plan, dict(zip(plan.dof_names, q, strict=True)))
    hip = poses[f"{side}_hip"].translation
    knee = poses[f"{side}_knee"].translation
    ankle = poses[f"{side}_ankle"].translation
    thigh, shin = knee - hip, ankle - knee
    cos = float(thigh @ shin / (np.linalg.norm(thigh) * np.linalg.norm(shin)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def foot_heights(plan, q: np.ndarray, root_height: float) -> dict[str, float]:
    poses = forward_kinematics(
        plan, dict(zip(plan.dof_names, q, strict=True)),
        root_position=(0.0, 0.0, root_height),
    )
    return {
        side: float(capsule_bottom(plan, poses, f"{side}_ankle"))
        for side in ("left", "right")
    }


def main() -> int:
    plan = plan_human_rig(load_human_config(REPO_ROOT / "configs/humans/human_smpl_multiaxis.yaml"))
    # Minimal inline spec matching configs/humans/keyboard.yaml so the audit does
    # not need the YAML loader's full schema validation.
    contact = {"cycle_distance_m": 0.6, "stance_fraction": 0.62, "clearance_m": 0.03}
    config = ContactGaitConfig(**contact)

    forward = load_gait(
        {
            "contact_cycle": contact,
            "file": REPO_ROOT / "data/humans/amass_raw/CMU/08/08_04_poses.npz",
            "start_s": 1.9667,
            "duration_s": 1.3083,
        },
        plan,
        dt_s=1 / 120.0,
    )

    walk_height = float(forward.provenance["contact_cycle"]["root_height_m"])

    # Straight-leg standing height: the same formula bake_contact_cycle uses for
    # its cap, but with the foot under the hip (no stride offset).
    neutral = forward_kinematics(plan, root_position=(0.0, 0.0, 0.0))
    standing_cap = np.inf
    for side in ("left", "right"):
        ankle = neutral[f"{side}_ankle"]
        hip = neutral[f"{side}_hip"]
        sole = ankle.translation[2] - capsule_bottom(plan, neutral, f"{side}_ankle")
        leg = (
            np.linalg.norm(hip.translation - neutral[f"{side}_knee"].translation)
            + np.linalg.norm(neutral[f"{side}_knee"].translation - ankle.translation)
        )
        horizontal = abs(hip.translation[0])
        standing_cap = min(
            standing_cap,
            sole - hip.translation[2] + np.sqrt((config.reach_fraction * leg) ** 2 - horizontal**2),
        )

    # Shipped behaviour: idle fit at the walking height.
    shipped = load_gait(
        {
            "file": REPO_ROOT / "data/humans/amass_raw/Transitions_mocap/mazen_c3d/"
            "walkbackwards_stand_poses.npz",
            "start_s": 3.4,
            "duration_s": 0.1,
        },
        plan,
        dt_s=1 / 120.0,
    )
    fit_contact_idle(shipped, plan, walk_height, config)
    q_shipped = shipped.sample(0.0)[0]

    # Candidate A: idle fit at the stride-capped standing height (reach_fraction
    # still limits the leg, so this alone leaves ~28 deg of flexion).
    capped = load_gait(
        {
            "file": REPO_ROOT / "data/humans/amass_raw/Transitions_mocap/mazen_c3d/"
            "walkbackwards_stand_poses.npz",
            "start_s": 3.4,
            "duration_s": 0.1,
        },
        plan,
        dt_s=1 / 120.0,
    )
    fit_contact_idle(capped, plan, float(standing_cap), config)
    q_capped = capped.sample(0.0)[0]

    # Candidate B: idle fit at the rig's natural standing height, where the foot
    # sits directly under the hip and the knee is near straight.
    natural = load_gait(
        {
            "file": REPO_ROOT / "data/humans/amass_raw/Transitions_mocap/mazen_c3d/"
            "walkbackwards_stand_poses.npz",
            "start_s": 3.4,
            "duration_s": 0.1,
        },
        plan,
        dt_s=1 / 120.0,
    )
    fit_contact_idle(natural, plan, float(plan.standing_root_height_m), config)
    q_natural = natural.sample(0.0)[0]

    report = {
        "walk_root_height_m": round(walk_height, 4),
        "straight_leg_root_height_m": round(float(standing_cap), 4),
        "height_drop_if_idle_walk_height_m": round(float(standing_cap) - walk_height, 4),
        "plan_standing_root_height_m": round(plan.standing_root_height_m, 4),
        "shipped_idle": {
            "root_height_m": round(float(shipped.height_m[0]), 4),
            "knee_flexion_deg": {
                side: round(knee_flexion_deg(plan, q_shipped, side), 2)
                for side in ("left", "right")
            },
            "foot_bottom_m": {
                side: round(v, 4)
                for side, v in foot_heights(plan, q_shipped, float(shipped.height_m[0])).items()
            },
        },
        "capped_idle": {
            "root_height_m": round(float(capped.height_m[0]), 4),
            "knee_flexion_deg": {
                side: round(knee_flexion_deg(plan, q_capped, side), 2)
                for side in ("left", "right")
            },
            "foot_bottom_m": {
                side: round(v, 4)
                for side, v in foot_heights(plan, q_capped, float(standing_cap)).items()
            },
        },
        "natural_idle": {
            "root_height_m": round(float(natural.height_m[0]), 4),
            "knee_flexion_deg": {
                side: round(knee_flexion_deg(plan, q_natural, side), 2)
                for side in ("left", "right")
            },
            "foot_bottom_m": {
                side: round(v, 4)
                for side, v in foot_heights(
                    plan, q_natural, float(plan.standing_root_height_m)
                ).items()
            },
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
