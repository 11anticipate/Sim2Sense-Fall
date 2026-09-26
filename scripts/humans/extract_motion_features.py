#!/usr/bin/env python3
"""One-pass per-frame motion-feature extraction over the AMASS inventory.

Every action-source screen (squat, bend, floor-sit, lying, ...) needs the
same per-frame geometry: pelvis height ratio, trunk pitch, foot contact
symmetry, knee flexion, hand heights. This tool computes those once with
the posture loader's own retarget and writes a compact feature table, so
source queries become instant lookups instead of one heavy pass each.

CPU-only pre-screen on the multiaxis rig (limits-checked, 10 Hz subsample);
shortlisted frames still need visual renders and a GPU gate run before
being configured.
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

from sim2sense_fall.humans.amass import load_amass_clip_by_id, normalize_root_motion  # noqa: E402
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
        default=REPO_ROOT / "artifacts/humans/fall_mesh/amass_rescreen_20260925.json")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts/humans/action_source_features.npz")
    parser.add_argument("--summary", type=Path,
                        default=REPO_ROOT / "artifacts/humans/action_source_features.json")
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

    clip_index_all: list[np.ndarray] = []
    frame_all: list[np.ndarray] = []
    pelvis_all: list[np.ndarray] = []
    trunk_all: list[np.ndarray] = []
    gap_all: list[np.ndarray] = []
    split_all: list[np.ndarray] = []
    knee_all: list[np.ndarray] = []
    hand_all: list[np.ndarray] = []
    speed_all: list[np.ndarray] = []
    kept_clips: list[str] = []
    errors = 0
    first_error = None

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
        rows = []
        step = max(1, int(round(clip.fps / args.sample_hz)))
        root_translation = np.asarray(clip.root_translation)
        root_speed = np.linalg.norm(np.diff(root_translation, axis=0), axis=1)
        root_speed = np.append(root_speed, root_speed[-1]) * clip.fps  # m/s
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
            trunk = links.get("neck") if "neck" in links else links.get("head")
            if trunk is None:
                trunk_pitch = float("nan")
            else:
                vec = trunk - links["pelvis"]
                trunk_pitch = float(np.degrees(np.arccos(
                    np.clip(vec[2] / max(np.linalg.norm(vec), 1e-9), -1.0, 1.0))))
            rows.append((
                frame,
                depth / standing,
                trunk_pitch,
                abs(left[2] - right[2]),
                float(np.hypot(left[0] - right[0], left[1] - right[1])),
                float(np.degrees(max(abs(values[i]) for i in knee_idx))),
                (links["left_wrist"][2] + links["right_wrist"][2]) / 2.0,
                float(root_speed[frame]),
            ))
        if not rows:
            continue
        array = np.asarray(rows, dtype=np.float32)
        clip_index_all.append(np.full(len(rows), len(kept_clips), dtype=np.int32))
        frame_all.append(array[:, 0].astype(np.int32))
        pelvis_all.append(array[:, 1])
        trunk_all.append(array[:, 2])
        gap_all.append(array[:, 3])
        split_all.append(array[:, 4])
        knee_all.append(array[:, 5])
        hand_all.append(array[:, 6])
        speed_all.append(array[:, 7])
        kept_clips.append(clip_id)
        if (index + 1) % 200 == 0:
            print(f"  extracted {index + 1}/{len(clip_ids)} clips, "
                  f"{sum(len(a) for a in pelvis_all)} frames", flush=True)

    np.savez_compressed(
        args.out,
        clip_ids=np.asarray(kept_clips),
        clip_index=np.concatenate(clip_index_all),
        frame=np.concatenate(frame_all),
        pelvis_ratio=np.concatenate(pelvis_all),
        trunk_pitch_deg=np.concatenate(trunk_all),
        foot_gap_m=np.concatenate(gap_all),
        foot_split_m=np.concatenate(split_all),
        knee_max_deg=np.concatenate(knee_all),
        hand_z=np.concatenate(hand_all),
        root_speed_m_s=np.concatenate(speed_all),
        standing_height_m=np.float32(standing),
        sample_hz=np.float32(args.sample_hz),
    )
    args.summary.write_text(json.dumps({
        "clips_requested": len(clip_ids),
        "clips_extracted": len(kept_clips),
        "clip_errors": errors,
        "first_error": first_error,
        "frames": int(sum(len(a) for a in pelvis_all)),
        "sample_hz": args.sample_hz,
    }, indent=2), encoding="utf-8")
    print(f"features -> {args.out}")
    print(f"  clips {len(kept_clips)}/{len(clip_ids)} (errors {errors}), "
          f"frames {sum(len(a) for a in pelvis_all)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
