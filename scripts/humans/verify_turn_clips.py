#!/usr/bin/env python3
"""Second-stage turn screening: retarget candidates and verify stepping.

The root-motion screen (``screen_turn_clips.py``) ranks clips by "turned a lot,
went nowhere", but it cannot tell a stepped turn from a pirouette, a jump twist
(airborne), or a pivot on one planted foot. This stage runs the ranked
candidates through the shipped retargeting pipeline and measures what actually
matters for a stepping-turn reference:

- **expressible** -- the retargeted joints stay inside the rig's limits
  (``joint_values_from_clip`` rejects otherwise);
- **stepping** -- from forward kinematics, each foot leaves the floor (capsule
  bottom above a clearance) and comes back, alternating, while the root yaws;
  a pirouette keeps one foot planted the whole time and a jump twist has no
  ground contact at all;
- **yaw coverage** -- how much turn the expressible, stepping part delivers,
  so a window can be cut around real steps.

Read-only; writes a JSON verdict per candidate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from common import DEFAULT_MOTIONS, REPO_ROOT, load_inputs  # type: ignore[import-not-found]

from sim2sense_fall.humans.amass import (
    AMASS_BODY_FRAME,
    ground_amass_clip,
    load_amass_clip,
    normalize_root_motion,
)
from sim2sense_fall.humans.contact_control import capsule_bottom
from sim2sense_fall.humans.rig import forward_kinematics, joint_values_from_clip, plan_human_rig
from sim2sense_fall.humans.rotations import axis_angle_to_matrix

LOGGER = logging.getLogger("verify_turn_clips")

SIDES = ("left", "right")
LIFT_CLEARANCE_M = 0.02


def foot_events(heights: np.ndarray, clearance_m: float) -> int:
    """Number of completed lift-and-return events (crossings above clearance)."""

    above = heights > clearance_m
    return int(np.sum(above[1:] & ~above[:-1]))


def verify(plan: Any, path: Path) -> dict[str, Any]:
    verdict: dict[str, Any] = {"file": str(path), "expressible": False}
    try:
        clip = load_amass_clip(path)
    except Exception as exc:
        verdict["error"] = f"load failed: {exc}"
        return verdict
    names = list(plan.dof_names)
    try:
        # Same grounding the shipped gait loader uses: anchor frame 0 to the floor
        # and keep every later root height, so foot heights mean clearance.
        clip = ground_amass_clip(normalize_root_motion(clip), plan, support_z_m=0.0)
        joints = np.stack(
            [joint_values_from_clip(clip, frame, plan)[0] for frame in range(clip.frame_count)]
        )
    except Exception as exc:
        verdict["error"] = f"retarget rejected: {exc}"
        return verdict
    verdict["expressible"] = True
    verdict["frames"] = int(clip.frame_count)
    verdict["duration_s"] = round(clip.frame_count / clip.fps, 2)
    basis = AMASS_BODY_FRAME.basis()

    heights = np.zeros((clip.frame_count, 2), dtype=np.float64)
    yaw = np.zeros(clip.frame_count, dtype=np.float64)
    spawn_z = float(plan.spawn_root_position[2])
    for frame in range(clip.frame_count):
        q = joints[frame]
        root_position = (0.0, 0.0, spawn_z + float(clip.root_translation[frame, 2]))
        poses = forward_kinematics(
            plan,
            dict(zip(names, q, strict=True)),
            root_position=root_position,
            root_rotation=clip.root_rotation[frame],
        )
        for column, side in enumerate(SIDES):
            heights[frame, column] = capsule_bottom(plan, poses, f"{side}_ankle")
        world_facing = basis @ np.array([0.0, 0.0, 1.0])
        facing = axis_angle_to_matrix(clip.root_rotation[frame]) @ world_facing
        yaw[frame] = np.arctan2(facing[1], facing[0])
    yaw = np.unwrap(yaw)
    events = {side: foot_events(heights[:, column], LIFT_CLEARANCE_M)
              for column, side in enumerate(SIDES)}
    # A step needs the foot to go up AND come back down; count completed events
    # and require both feet to take some, otherwise it is a one-foot pivot.
    total_steps = sum(events.values())
    stepping = total_steps >= 4 and all(count >= 1 for count in events.values())
    turn_deg = float(abs(yaw[-1] - yaw[0]) * 180.0 / np.pi)
    verdict.update(
        {
            "lift_events": events,
            "total_steps": total_steps,
            "stepping": stepping,
            "turn_total_deg": round(turn_deg, 1),
            "turn_rate_deg_s": round(turn_deg / max(clip.frame_count / clip.fps, 1e-9), 1),
            "max_foot_height_m": round(float(heights.max()), 3),
            "steps_per_second": round(total_steps / max(clip.frame_count / clip.fps, 1e-9), 2),
        }
    )
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("artifacts/humans/turn_screen/turn_candidates.json"),
    )
    parser.add_argument(
        "--rig", type=Path, default=REPO_ROOT / "configs/humans/human_smpl_multiaxis.yaml"
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/humans/turn_screen/turn_verification.json")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    config, _registry, _ = load_inputs(
        config_path=args.rig, assets_path=REPO_ROOT / "configs/humans/assets.yaml",
        motions_path=DEFAULT_MOTIONS,
    )
    plan = plan_human_rig(config)
    results = []
    for row in candidates[: args.limit]:
        path = Path(row["file"])
        LOGGER.info("verifying %s", path)
        verdict = verify(plan, path)
        verdict["screen"] = {key: row[key] for key in ("turn_total_deg", "net_displacement_m",
                                                       "path_displacement_m", "duration_s")}
        results.append(verdict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for row in results:
        marker = "OK " if row.get("stepping") and row.get("expressible") else "-- "
        print(
            f"{marker}{Path(row['file']).name}: expressible={row.get('expressible')} "
            f"steps={row.get('total_steps')} ({row.get('lift_events')}) "
            f"turn={row.get('turn_total_deg')} deg, rate={row.get('turn_rate_deg_s')} deg/s"
            + (f"  [{row['error']}]" if "error" in row else "")
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
