#!/usr/bin/env python3
"""Screen the local AMASS library for turn-in-place reference clips.

Turning with the shipped controller pivots both flat feet on the floor -- the
user sees the body rotate like a skater because nothing ever lifts a foot. A
reference-driven fix needs a clip where a subject actually *steps* through a
turn: root yaw rotates substantially while the root barely translates.

The screen reads only root motion (SMPL root rotation ``poses[:, :3]`` and root
translation ``trans``), so it covers the whole 2198-clip library on CPU. For
every clip it computes the unwrapped yaw trajectory (SMPL is Y-up and faces
+Z, so the facing direction is the rotated +Z axis), the total turn, the net
and path horizontal displacement, and the mean yaw rate. Candidates need a
real turn (>= 90 deg) that is not a walk (net displacement small for the
amount turned). Foot-stepping is NOT checked here -- that needs forward
kinematics and is the second stage, run only on the ranked candidates.

Read-only; writes a ranked JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from sim2sense_fall.humans.rotations import axis_angle_to_matrix

LOGGER = logging.getLogger("screen_turn_clips")

MIN_TURN_DEG = 90.0
MAX_MEAN_SPEED_M_S = 0.45
MIN_DURATION_S = 1.5
MAX_DURATION_S = 12.0


def yaw_trajectory(root_rotations: np.ndarray) -> np.ndarray:
    """Unwrapped yaw of a (N, 3) SMPL root axis-angle track.

    AMASS capture world is Z-up and SMPL faces +Z in model coordinates, so the
    world facing is ``R @ [0,0,1]`` and the yaw is ``atan2(facing_y, facing_x)``
    about the vertical. (An earlier version used ``atan2(x, z)`` -- that mixes a
    horizontal with the vertical component and is not a yaw; it reported 700 deg
    turns for clips whose real facing barely moved.)
    """

    yaw = np.zeros(len(root_rotations), dtype=np.float64)
    for index, vector in enumerate(root_rotations):
        facing = axis_angle_to_matrix(vector) @ np.array([0.0, 0.0, 1.0])
        yaw[index] = np.arctan2(facing[1], facing[0])
    return np.unwrap(yaw)


def screen_clip(path: Path) -> dict[str, float] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "poses" not in data or "trans" not in data or "mocap_framerate" not in data:
                return None
            poses = np.asarray(data["poses"], dtype=np.float64)
            trans = np.asarray(data["trans"], dtype=np.float64)
            fps = float(data["mocap_framerate"])
    except Exception as exc:  # unreadable file: skip, the loader records it
        LOGGER.debug("skipping %s: %s", path.name, exc)
        return None
    if poses.ndim != 2 or poses.shape[0] < 10 or trans.shape != (poses.shape[0], 3):
        return None
    if not np.isfinite(poses[:, :3]).all() or not np.isfinite(trans).all():
        return None
    stride = max(1, int(fps // 30))  # ~30 Hz is plenty for yaw statistics
    yaw = yaw_trajectory(poses[::stride, :3])
    turn_total_deg = float(abs(yaw[-1] - yaw[0]) * 180.0 / np.pi)
    horizontal = trans[:, :2]
    net_m = float(np.linalg.norm(horizontal[-1] - horizontal[0]))
    path_m = float(np.linalg.norm(np.diff(horizontal[::stride], axis=0), axis=1).sum())
    duration_s = float(poses.shape[0] / fps)
    if not MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
        return None
    if turn_total_deg < MIN_TURN_DEG:
        return None
    mean_speed = path_m / duration_s
    if mean_speed > MAX_MEAN_SPEED_M_S:
        return None
    duration_turning = duration_s * turn_total_deg / max(
        float(np.abs(np.diff(yaw)).sum()) * 180.0 / np.pi, 1e-9
    )
    return {
        "file": str(path),
        "duration_s": round(duration_s, 2),
        "turn_total_deg": round(turn_total_deg, 1),
        "turn_rate_deg_s": round(turn_total_deg / duration_s, 1),
        "net_displacement_m": round(net_m, 2),
        "path_displacement_m": round(path_m, 2),
        "mean_speed_m_s": round(mean_speed, 2),
        "turn_purity": round(
            turn_total_deg / 90.0 / max(net_m / max(path_m, 1e-6), 1e-6) / max(mean_speed, 0.05),
            2,
        ),
        "duration_turning_s": round(duration_turning, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/humans/amass_raw"))
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/humans/turn_screen/turn_candidates.json")
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    files = sorted(args.root.rglob("*.npz"))
    LOGGER.info("screening %d clips", len(files))
    candidates = []
    for path in files:
        row = screen_clip(path)
        if row is not None:
            candidates.append(row)
    # Rank: turned a lot, went nowhere. Net displacement per degree of turn is the
    # cheapest proxy for "turning in place rather than along an arc".
    candidates.sort(key=lambda row: row["net_displacement_m"] / max(row["turn_total_deg"], 1.0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidates, indent=1), encoding="utf-8")
    LOGGER.info("%d candidates -> %s", len(candidates), args.out)
    for row in candidates[: args.top]:
        print(
            f"{Path(row['file']).relative_to(args.root)}: turn {row['turn_total_deg']:.0f} deg "
            f"in {row['duration_s']:.1f} s ({row['turn_rate_deg_s']:.0f} deg/s), "
            f"net {row['net_displacement_m']:.2f} m, path {row['path_displacement_m']:.2f} m"
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
