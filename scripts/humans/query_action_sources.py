#!/usr/bin/env python3
"""Classify extracted motion features into project-relevant action categories.

Reads the per-frame feature table from ``extract_motion_features.py`` and
labels every frame against pre-registered geometric signatures (quiet stand,
walk/run, squat hold, low crouch, floor sit, lying), then reports how many
clips carry each action and lists representative stationary frames per
category. The labels are geometric pre-screen — a candidate still needs a
render and a GPU gate run before being configured as an action source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = {
    # 名字: (骨盆高比区间, 躯干俯角区间 deg, 根速度上限 m/s, 备注)
    "quiet_stand": ((0.80, 1.05), (0, 25), 0.10),
    "walk_run": ((0.55, 1.05), (0, 40), None),       # 速度下限在代码里区分
    "squat_hold": ((0.25, 0.62), (0, 50), 0.10),
    "low_crouch": ((0.12, 0.25), (0, 70), 0.10),
    "floor_sit": ((0.08, 0.22), (0, 75), 0.10),
    "lying": ((0.02, 0.16), (55, 180), 0.10),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path,
                        default=REPO_ROOT / "artifacts/humans/action_source_features.npz")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts/humans/action_source_query.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = np.load(args.features, allow_pickle=False)
    clip_ids = [str(c) for c in data["clip_ids"]]
    clip_index = data["clip_index"]
    standing = float(data["standing_height_m"])

    pelvis = data["pelvis_ratio"]
    trunk = data["trunk_pitch_deg"]
    gap = data["foot_gap_m"]
    split = data["foot_split_m"]
    knee = data["knee_max_deg"]
    hand = data["hand_z"] / standing
    speed = data["root_speed_m_s"]

    labels = np.full(len(pelvis), "", dtype=object)
    for name, (ratio_range, pitch_range, max_speed) in CATEGORIES.items():
        mask = (
            (pelvis >= ratio_range[0]) & (pelvis <= ratio_range[1])
            & (trunk >= pitch_range[0]) & (trunk <= pitch_range[1])
        )
        if name == "walk_run":
            mask &= (speed >= 0.25) & (speed <= 3.5)
            walk = mask & (speed <= 1.5)
            run = mask & (speed > 1.5)
            labels[walk] = "walk"
            labels[run] = "run"
            continue
        if name == "quiet_stand":
            mask &= (gap <= 0.03) & (split <= 0.30) & (speed <= max_speed)
        elif name == "squat_hold":
            mask &= (gap <= 0.05) & (knee >= 60.0) & (hand >= 0.15) & (speed <= max_speed)
        elif name in ("low_crouch", "floor_sit", "lying"):
            mask &= (speed <= max_speed)
        labels[mask & (labels == "")] = name

    report = {}
    for name in CATEGORIES:
        subset = name if name != "walk_run" else "walk"
        if name == "walk_run":
            targets = ["walk", "run"]
        else:
            targets = [name]
        for target in targets:
            mask = labels == target
            rows = np.flatnonzero(mask)
            if not len(rows):
                report[target] = {"clips": 0, "frames": 0}
                continue
            clips = sorted({clip_ids[clip_index[i]] for i in rows})
            # 每条片段的代表性帧: 最静止的那帧
            representatives = []
            for cid in clips:
                cid_rows = [i for i in rows if clip_ids[clip_index[i]] == cid]
                best = min(cid_rows, key=lambda i: speed[i])
                representatives.append({
                    "clip": cid,
                    "frame": int(data["frame"][best]),
                    "pelvis_ratio": round(float(pelvis[best]), 3),
                    "trunk_pitch_deg": round(float(trunk[best]), 1),
                    "root_speed_m_s": round(float(speed[best]), 3),
                })
            representatives.sort(key=lambda r: r["root_speed_m_s"])
            report[target] = {
                "clips": len(clips),
                "frames": int(mask.sum()),
                "representatives": representatives[:8],
            }

    summary = {
        "frames_total": int(len(pelvis)),
        "frames_labeled": int(sum(labels != "")),
        "standing_height_m": standing,
        "categories": {key: value for key, value in report.items()},
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"action query -> {args.out}")
    for target, info in report.items():
        print(f"  {target:12s} {info['clips']:4d} 条片段 / {info['frames']:6d} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
