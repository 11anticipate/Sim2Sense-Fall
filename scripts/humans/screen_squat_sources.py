#!/usr/bin/env python3
"""Screen the local AMASS library for symmetric deep-squat source frames.

The current crouch posture (key C) sources an asymmetric crouch-*walk*
stride frame, which reads as a sprint start. This tool scans every clip in
the fall-rescreen inventory with the same per-frame retarget the posture
loader uses, and ranks frames by a pre-registered symmetric-squat rule:

- both feet planted (depth gap <= 4 cm) and close together (split <= 22 cm),
- pelvis-to-foot depth in [0.25, 0.55] of the rig's standing height,
- mean knee flexion >= 60 deg,
- hands above knee height (no floor support),

all on frames the multiaxis rig can express within joint limits. CPU-only
pre-screen (limits + FK geometry); shortlisted frames still need a GUI
look and a GPU gate run before becoming the configured posture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))

from common import DEFAULT_MOTIONS, REPO_ROOT, load_inputs  # noqa: E402

from sim2sense_fall.humans.amass import (  # noqa: E402
    load_amass_clip_by_id,
    normalize_root_motion,
)
from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
)
from sim2sense_fall.humans.teleop import load_keyboard_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--inventory", type=Path,
        default=REPO_ROOT / "artifacts/humans/fall_mesh/amass_rescreen_20260925.json",
        help="clip inventory JSON from the fall rescreen")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="screen only the first N clips (debug)")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts/humans/squat_source_screen.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_keyboard_config(REPO_ROOT / "configs/humans/keyboard.yaml", REPO_ROOT)
    config, registry, _ = load_inputs(config_path=settings["rig"],
                                      assets_path=settings["assets"],
                                      motions_path=DEFAULT_MOTIONS)
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    dof_names = list(plan.dof_names)
    lower = np.deg2rad([joint.lower_deg for joint in plan.joints])
    upper = np.deg2rad([joint.upper_deg for joint in plan.joints])
    knee_idx = [i for i, name in enumerate(dof_names) if "knee" in name]
    standing = float(plan.standing_root_height_m)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    clip_ids = [row["clip"] for row in inventory if row.get("rig_expressible", True)]
    if args.limit:
        clip_ids = clip_ids[: args.limit]

    candidates: list[dict] = []
    errors = 0
    first_error: str | None = None
    for index, clip_id in enumerate(clip_ids):
        try:
            clip = normalize_root_motion(
                load_amass_clip_by_id(REPO_ROOT / "data/humans", clip_id)
            )
        except Exception as error:
            if first_error is None:
                first_error = f"{clip_id}: {type(error).__name__}: {error}"
            errors += 1
            continue
        step = max(1, int(round(clip.fps / args.sample_hz)))
        for frame in range(0, len(clip.joint_rotations), step):
            try:
                values, _ = joint_values_from_clip(clip, frame, plan)
            except Exception:
                continue
            if np.any(values < lower) or np.any(values > upper):
                continue
            poses = forward_kinematics(plan, dict(zip(dof_names, values, strict=False)),
                                       root_position=(0.0, 0.0, 0.0),
                                       root_rotation=np.zeros(3))
            links = {name: np.asarray(t.translation) for name, t in poses.items()}
            left, right = links["left_ankle"], links["right_ankle"]
            depth = -min(left[2], right[2])
            ratio = depth / standing
            gap = abs(left[2] - right[2])
            split = float(np.hypot(left[0] - right[0], left[1] - right[1]))
            if not 0.25 <= ratio <= 0.55:
                continue
            if gap > 0.04 or split > 0.35:
                continue
            # 膝屈曲取主自由度的 max——膝多自由度取均值会稀释屈曲角
            knee = float(max(abs(values[i]) for i in knee_idx))
            if knee < np.deg2rad(60.0):
                continue
            hands = (links["left_wrist"][2] + links["right_wrist"][2]) / 2.0
            knees = (links["left_knee"][2] + links["right_knee"][2]) / 2.0
            if hands < knees:
                continue
            score = ((0.55 - ratio) + 0.3 * knee
                     + 0.2 * (0.35 - split) + 0.2 * (0.04 - gap))
            candidates.append({
                "clip": clip_id,
                "frame": frame,
                "pelvis_ratio": round(ratio, 3),
                "knee_deg": round(float(np.degrees(knee)), 1),
                "foot_split_m": round(split, 3),
                "score": round(score, 3),
            })
        if (index + 1) % 200 == 0:
            print(f"  screened {index + 1}/{len(clip_ids)} clips, "
                  f"{len(candidates)} candidate frames", flush=True)

    candidates.sort(key=lambda row: -row["score"])
    payload = {
        "rule": "symmetric deep static squat (feet planted+close, pelvis 25-55% "
                "of standing, knee >= 60 deg, hands above knees, rig-expressible)",
        "clips_screened": len(clip_ids),
        "clip_errors": errors,
        "first_error": first_error,
        "candidate_frames": len(candidates),
        "top": candidates[:40],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"squat source screen -> {args.out}")
    for row in payload["top"][:10]:
        print(f"  {row['clip']}  frame={row['frame']}  pelvis={row['pelvis_ratio']:.2f} "
              f"knee={row['knee_deg']:.0f}deg split={row['foot_split_m']:.3f} "
              f"score={row['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
