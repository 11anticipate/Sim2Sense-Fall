#!/usr/bin/env python3
"""Emit the keyboard-session configs that make up a training-data batch.

Every dataset session must run the *shipped* control stack, so each generated config
is `configs/humans/keyboard.yaml` verbatim with only three things changed: the
scripted `demo:` protocol, the `spawn_xy:` / `heading_deg:` placement, and nothing
else. Nothing here tunes control parameters -- if a control default changes,
regenerate.

Spawns are not invented: each one must appear in the clearance audit's picked list
(`scripts/humans/audit_spawn_clearance.py`, which proves the standing or the
activity-sized envelope is free over a real floor slab), so a session can never
start inside furniture. Run the audit first:

    PYTHONPATH=src python3 scripts/humans/audit_spawn_clearance.py \
        --human-radius-m 0.9 --pick 8 --out artifacts/humans/spawn_clearance_activity.json
    PYTHONPATH=src python3 scripts/humans/make_dataset_configs.py
    python3 scripts/sionna/batch_generate.py --plan configs/sionna/batch_train01.yaml \
        --out artifacts/batches/train01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs/humans/keyboard.yaml"
OUT_DIR = REPO_ROOT / "configs/humans/dataset"
CLEARANCE = REPO_ROOT / "artifacts/humans/spawn_clearance_activity.json"

# (spawn, heading_deg, protocol). Headings face the open direction of the room: the
# corridor runs along +x, the living room's clear band runs along +y from the
# activity-clear spot. Protocols are activity episodes so the exporter can judge
# each one separately.
SESSIONS: dict[str, dict[str, Any]] = {
    "fall_standing": {
        "spawn_xy": [11.35, 1.25], "heading_deg": 90.0,
        "demo": [(1.0, []), (0.1, ["F"]), (4.0, []), (0.1, ["R"]), (1.0, [])],
    },
    "fall_walking": {
        "spawn_xy": [5.15, 7.65], "heading_deg": 0.0,
        "demo": [(1.0, []), (3.0, ["W"]), (0.1, ["F"]), (4.5, []), (0.1, ["R"]), (1.0, [])],
    },
    "fall_turning": {
        "spawn_xy": [13.05, 7.55], "heading_deg": 180.0,
        "demo": [(1.0, []), (2.0, ["W", "A"]), (0.1, ["F"]), (4.5, []), (0.1, ["R"]), (1.0, [])],
    },
    "fall_backward": {
        "spawn_xy": [11.35, 1.25], "heading_deg": 90.0,
        "demo": [(1.0, []), (2.0, ["S"]), (0.1, ["F"]), (4.5, []), (0.1, ["R"]), (1.0, [])],
    },
    "adl_locomotion": {
        "spawn_xy": [5.15, 7.65], "heading_deg": 0.0,
        "demo": [
            (1.0, []), (3.0, ["W"]), (1.0, ["SPACE"]), (2.0, ["W", "A"]), (2.0, ["W", "D"]),
            (1.0, ["SPACE"]), (2.0, ["S"]), (1.0, ["SPACE"]), (1.5, []),
        ],
    },
    "adl_postures": {
        "spawn_xy": [11.35, 1.25], "heading_deg": 90.0,
        "demo": [
            (1.0, []), (2.5, ["C"]), (1.5, []), (2.5, ["V"]), (1.0, []),
            (2.5, ["B"]), (1.5, []), (2.5, ["V"]), (1.0, []),
            (2.5, ["N"]), (1.5, []), (2.5, ["V"]), (1.5, []),
        ],
    },
    # Recovery sessions: fall then get up. The get_up episode is judged by the
    # low-posture gates; if it fails, only that segment is refused while the fall
    # and standing segments in the same session still enter the batch.
    "fall_getup_standing": {
        "spawn_xy": [11.35, 1.25], "heading_deg": 90.0,
        "demo": [(1.0, []), (0.1, ["F"]), (3.5, []), (0.1, ["G"]), (9.0, []), (2.0, [])],
    },
    "fall_getup_walking": {
        "spawn_xy": [13.05, 7.55], "heading_deg": 180.0,
        "demo": [
            (1.0, []), (3.0, ["W"]), (0.1, ["F"]), (3.5, []), (0.1, ["G"]), (9.0, []), (2.0, []),
        ],
    },
}


def validate(
    base: dict[str, Any], name: str, spec: dict[str, Any], clear: set[tuple[float, float]]
) -> dict[str, Any]:
    xy = tuple(spec["spawn_xy"])
    if xy not in clear:
        raise ValueError(
            f"session {name} spawn {xy} is not in {CLEARANCE.relative_to(REPO_ROOT)}'s picked "
            "list; run scripts/humans/audit_spawn_clearance.py first (spawns must be measured "
            "clear, not invented)"
        )
    payload = dict(base)
    payload["spawn_xy"] = list(xy)
    payload["heading_deg"] = float(spec["heading_deg"])
    payload["demo"] = [{"duration_s": t, "keys": keys} for t, keys in spec["demo"]]
    total = sum(t for t, _ in spec["demo"])
    if not 5.0 <= total <= 120.0:
        raise ValueError(f"session {name} is {total:.1f} s, outside [5, 120]")
    if not any(keys for _, keys in spec["demo"]):
        raise ValueError(f"session {name} presses no keys")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=BASE_CONFIG)
    parser.add_argument("--clearance", type=Path, default=CLEARANCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--print-plan", action="store_true",
                        help="print the matching batch plan YAML instead of writing configs")
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not args.clearance.is_file():
        raise SystemExit(f"missing {args.clearance}; run scripts/humans/audit_spawn_clearance.py")
    clear = {
        (row["xy"][0], row["xy"][1])
        for row in json.loads(args.clearance.read_text(encoding="utf-8"))["picked_spawns"]
    }
    if args.print_plan:
        plan = {
            "description": "训练前批量包：4 类摔倒触发 + 行走/姿势 ADL + 2 类摔倒后起身",
            "trials": [],
            "sessions": [{"config": f"configs/humans/dataset/{name}.yaml", "name": name}
                         for name in SESSIONS],
            "rt_frames": 24,
            "render_frames": False,
        }
        print(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in SESSIONS.items():
        payload = validate(base, name, spec, clear)
        header = (
            "# Generated by scripts/humans/make_dataset_configs.py -- do not hand-edit.\n"
            f"# Dataset session '{name}': configs/humans/keyboard.yaml verbatim except\n"
            "# spawn_xy / heading_deg (clearance-audited) and the scripted demo protocol.\n"
        )
        path = args.out_dir / f"{name}.yaml"
        path.write_text(header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True,
                                                width=100), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)} "
              f"({sum(t for t, _ in spec['demo']):.1f} s protocol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
