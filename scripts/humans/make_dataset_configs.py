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
from dataclasses import dataclass
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
    if not 5.0 <= total <= 240.0:
        raise ValueError(f"session {name} is {total:.1f} s, outside [5, 240]")
    if not any(keys for _, keys in spec["demo"]) and total < MIN_KEYLESS_SECONDS:
        # A session that presses no key is a quiet-standing exposure -- legitimate,
        # but only if it is long enough to matter for false alarms per hour.
        raise ValueError(
            f"session {name} presses no keys and is only {total:.1f} s; a keyless"
            f" exposure needs >= {MIN_KEYLESS_SECONDS:.0f} s"
        )
    return payload


# Heading is axis-aligned so the walk demand maps exactly onto the audit's run lengths.
AXIS_BY_HEADING: dict[float, tuple[str, str]] = {
    0.0: ("plus_x", "minus_x"), 90.0: ("plus_y", "minus_y"),
    180.0: ("minus_x", "plus_x"), 270.0: ("minus_y", "plus_y"),
}
# Turns, the deceleration ramp and the controller's target lead all overshoot the pure
# ``speed * duration`` figure; the fall sprawl itself is covered by the activity-radius
# check on the spawn point, so this only has to cover the walk.
WALK_SAFETY_M = 0.40
# Shortest keyless (quiet standing) session worth tracing for the FP/h denominator.
MIN_KEYLESS_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class Protocol:
    """One session template: which keys to press and how much clear floor it demands.

    ``walk_forward_s`` / ``walk_backward_s`` are the seconds of held ``W`` / ``S``, turned
    into metres by the session's own speed; ``turning`` discounts forward progress because
    an arcing body advances less than ``speed * duration``.
    """

    kind: str
    walk_forward_s: float
    walk_backward_s: float
    turning: bool
    episodes: tuple[tuple[float, tuple[str, ...]], ...]
    # How many times the longest leg is repeated. A patrol returns along the line it
    # came in on, so the clear run only has to cover ONE leg -- but the controller
    # drifts a little per leg, so the demand grows by a fraction of the safety margin
    # per extra repetition rather than by a whole leg.
    legs: int = 1
    # True when the backward legs walk back along the line the forward legs came in
    # on (a patrol). Then the reverse axis needs no extra clear floor -- demanding a
    # full leg there as well starved the longest protocols down to one usable spawn.
    retraces: bool = False

    def needs(self, speed_m_s: float) -> tuple[float, float]:
        forward = self.walk_forward_s * speed_m_s * (0.6 if self.turning else 1.0)
        drift = WALK_SAFETY_M * 0.5 * max(0, self.legs - 1)
        backward = 0.0 if self.retraces else self.walk_backward_s * speed_m_s
        return (forward + (WALK_SAFETY_M + drift if self.walk_forward_s else 0.0),
                backward + (WALK_SAFETY_M if (self.walk_backward_s and not self.retraces) else 0.0))

    def demo(self) -> list[tuple[float, list[str]]]:
        return [(seconds, list(keys)) for seconds, keys in self.episodes if seconds > 0]


def _fall(walk_forward_s: float, walk_backward_s: float, turning: bool,
          keys: tuple[str, ...]) -> tuple[tuple[float, tuple[str, ...]], ...]:
    """Settle, walk into the trigger context, F, collapse and settle, reset, stand."""

    return ((1.0, ()), (walk_forward_s, keys), (walk_backward_s, ("S",)),
            (0.1, ("F",)), (4.5, ()), (0.1, ("R",)), (1.0, ()))


# Only activities whose measured gates currently pass. Floor recovery (`G`) and the
# posture *transitions* are deliberately absent: they fail their gates, so those segments
# would be refused and the Isaac time spent on them wasted
# (docs/getup-float-2026-09-27.md).
PROTOCOLS: tuple[Protocol, ...] = (
    Protocol("fall_standing", 0.0, 0.0, False, _fall(0.0, 0.0, False, ())),
    Protocol("fall_forward", 3.0, 0.0, False, _fall(3.0, 0.0, False, ("W",))),
    Protocol("fall_backward", 0.0, 2.0, False, _fall(0.0, 2.0, False, ())),
    Protocol("fall_turning", 2.0, 0.0, True, _fall(2.0, 0.0, True, ("W", "A"))),
    Protocol("adl_walk_patrol", 4.0, 2.0, False, (
        (1.0, ()), (4.0, ("W",)), (1.0, ("SPACE",)), (2.0, ("S",)),
        (1.0, ("SPACE",)), (1.5, ()),
    )),
    Protocol("adl_turn_walk", 3.0, 0.0, True, (
        (1.0, ()), (1.5, ("W", "A")), (1.5, ("W", "D")), (1.5, ("W",)), (1.5, ()),
    )),
    Protocol("adl_crouch", 0.0, 0.0, False, (
        (1.0, ()), (2.5, ("C",)), (3.0, ()), (2.5, ("V",)), (2.0, ()),
    )),
    Protocol("adl_sit", 0.0, 0.0, False, (
        (1.0, ()), (2.5, ("N",)), (3.0, ()), (2.5, ("V",)), (2.0, ()),
    )),
    Protocol("adl_bend", 0.0, 0.0, False, (
        (1.0, ()), (2.5, ("B",)), (3.0, ()), (2.5, ("V",)), (2.0, ()),
    )),
    # --- train03: false-alarm exposure. train02 admitted only 210 s of ADL channel
    # material, so 每小时误报 had no usable denominator. Measured acceptance in that
    # batch: walking classes 100% (no-support 0 ms), quiet standing 75% (a 58-75 ms
    # contact gap kills a whole segment), so the long sessions are patrol-based and
    # the standing ones stay short. Chopping a long stream into short segments to
    # inflate admitted hours is not allowed -- the gate is per segment on purpose.
    Protocol("adl_long_patrol", 12.0, 12.0, False, (
        (1.0, ()),
        *((12.0, ("W",)), (1.0, ("SPACE",)), (12.0, ("S",)), (1.0, ("SPACE",))) * 3,
        (2.0, ()),
    ), legs=3, retraces=True),
    Protocol("adl_patrol_mid", 6.0, 6.0, False, (
        (1.0, ()),
        *((6.0, ("W",)), (1.0, ("SPACE",)), (6.0, ("S",)), (1.0, ("SPACE",))) * 3,
        (2.0, ()),
    ), legs=3, retraces=True),
    Protocol("adl_long_stand", 0.0, 0.0, False, ((1.0, ()), (45.0, ()),)),
    Protocol("adl_bend_hold", 0.0, 0.0, False, (
        (1.0, ()), (2.5, ("B",)), (8.0, ()), (2.5, ("V",)), (3.0, ()),
    )),
)


def build_matrix(
    spawns: list[dict[str, Any]], *, protocols: list[Protocol], speeds_m_s: list[float],
    max_sessions: int,
) -> dict[str, dict[str, Any]]:
    """Assign a protocol + heading + speed to every clearance-audited spawn.

    Each spawn is offered its protocols in order, and the heading is the axis-aligned
    direction with the most clear floor for that protocol's walk demand. A pair that the
    measured run lengths cannot support is skipped rather than emitted -- a session whose
    walk segment collides with furniture produces a refused segment and wastes the whole
    Isaac run, so the audit's numbers decide the schedule.
    """

    # Collect every compatible (protocol, spawn, heading) triple first, then interleave
    # the per-protocol queues: taking the cap off the queue order would let the first two
    # protocols fill the batch and leave the rare classes (sit, crouch, bend) out.
    candidates: dict[str, list[dict[str, Any]]] = {protocol.kind: [] for protocol in protocols}
    for speed_m_s in speeds_m_s:
        for protocol in protocols:
            for spawn in spawns:
                needed_forward, needed_backward = protocol.needs(speed_m_s)
                best: tuple[float, float, tuple[str, str]] | None = None
                for heading, (forward_axis, backward_axis) in AXIS_BY_HEADING.items():
                    runs = spawn["runs_m"]
                    if runs[forward_axis] < needed_forward:
                        continue
                    if runs[backward_axis] < needed_backward:
                        continue
                    margin = runs[forward_axis] + runs[backward_axis]
                    if best is None or margin > best[0]:
                        best = (margin, heading, (forward_axis, backward_axis))
                if best is None:
                    continue
                candidates[protocol.kind].append({
                    "spawn_xy": list(spawn["xy"]), "heading_deg": float(best[1]),
                    "speed_m_s": speed_m_s, "demo": protocol.demo(),
                    "room": spawn["room"],
                })
    sessions: dict[str, dict[str, Any]] = {}
    index = 0
    while len(sessions) < max_sessions and any(candidates.values()):
        for kind, queue in candidates.items():
            if not queue:
                continue
            row = queue.pop(0)
            index += 1
            name = (f"{index:02d}_{kind}_sp{int(round(row['speed_m_s'] * 100)):03d}"
                    f"_{row['room'].split('/')[-2]}")
            sessions[name] = row
            if len(sessions) >= max_sessions:
                break
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=BASE_CONFIG)
    parser.add_argument("--clearance", type=Path, default=CLEARANCE)
    parser.add_argument("--batch", default="train02",
                        help="batch name; train01 keeps its legacy flat layout and fixed "
                             "session list, anything else builds the spawn x context matrix")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default configs/humans/dataset/<batch> "
                             "(train01 writes the legacy configs/humans/dataset)")
    parser.add_argument("--max-sessions", type=int, default=30)
    parser.add_argument("--only", default=None,
                        help="comma-separated protocol kinds to use (default: all). An ADL "
                             "exposure batch uses the long patrol/stand kinds only, so the "
                             "cap is not spent on classes the previous batch already has.")
    parser.add_argument("--speeds", default="0.4",
                        help="comma-separated walking speeds (m/s) to matrix over; 0.4 is "
                             "the shipped, gate-validated value and anything else is an "
                             "unvalidated diversity probe")
    parser.add_argument("--rt-seed-base", type=int, default=20260927,
                        help="recorded in the plan; the RT stage derives a deterministic "
                             "per-sample seed from it")
    parser.add_argument("--print-plan", action="store_true",
                        help="print the matching batch plan YAML instead of writing configs")
    args = parser.parse_args()
    if args.max_sessions < 1:
        parser.error("max-sessions must be >= 1")
    protocols = list(PROTOCOLS)
    if args.only:
        wanted = [kind.strip() for kind in args.only.split(",") if kind.strip()]
        known = {protocol.kind for protocol in PROTOCOLS}
        unknown = sorted(set(wanted) - known)
        if unknown:
            parser.error(f"--only names unknown protocols {unknown}; known: {sorted(known)}")
        protocols = [protocol for protocol in PROTOCOLS if protocol.kind in wanted]
    speeds = [float(value) for value in args.speeds.split(",")]
    if any(not 0.1 <= value <= 1.2 for value in speeds):
        parser.error(f"speeds must be within [0.1, 1.2] m/s, got {speeds}")

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not args.clearance.is_file():
        raise SystemExit(f"missing {args.clearance}; run scripts/humans/audit_spawn_clearance.py")
    audit = json.loads(args.clearance.read_text(encoding="utf-8"))
    clear = {(row["xy"][0], row["xy"][1]) for row in audit["picked_spawns"]}
    legacy = args.batch == "train01"
    out_dir = args.out_dir or (OUT_DIR if legacy
                               else REPO_ROOT / "configs/humans/dataset" / args.batch)
    if not out_dir.is_absolute():
        # A relative --out-dir made `path.relative_to(REPO_ROOT)` raise on the very
        # first written config, so the generator died after creating files.
        out_dir = REPO_ROOT / out_dir
    if legacy:
        sessions = SESSIONS
    else:
        if any("runs_m" not in row for row in audit["picked_spawns"]):
            raise SystemExit(
                f"{args.clearance} has no run-length probes; re-run "
                "scripts/humans/audit_spawn_clearance.py -- the matrix needs runs_m to "
                "decide which spawns may be given a walking protocol"
            )
        sessions = build_matrix(audit["picked_spawns"], protocols=protocols,
                                speeds_m_s=speeds, max_sessions=args.max_sessions)
        if not sessions:
            raise SystemExit("the matrix is empty: no audited spawn supports a protocol")
    if args.print_plan:
        plan = {
            "description": (f"{args.batch}: {len(sessions)} 会话，spawn x 触发情境 x 朝向矩阵，"
                            "只含当前过门的活动"),
            "trials": [],
            "sessions": [{"config": str((out_dir / f"{name}.yaml").relative_to(REPO_ROOT)),
                          "name": name} for name in sessions],
            "rt_target_hz": 120,
            "rt_frames": 24,
            "rt_seed_base": args.rt_seed_base,
            "render_frames": False,
            "split_label": f"{args.batch}_grouped_by_session_pending_splits_json",
        }
        print(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in sessions.items():
        payload = validate(base, name, spec, clear)
        header = (
            "# Generated by scripts/humans/make_dataset_configs.py -- do not hand-edit.\n"
            f"# Dataset session '{name}' of batch {args.batch}: keyboard.yaml verbatim\n"
            "# except spawn_xy / heading_deg / speed_m_s (clearance-audited) and the\n"
            "# scripted demo protocol.\n"
        )
        path = out_dir / f"{name}.yaml"
        path.write_text(header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True,
                                                width=100), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)} "
              f"({sum(t for t, _ in spec['demo']):.1f} s protocol, "
              f"speed {spec.get('speed_m_s', base['speed_m_s'])} m/s)")
    print(f"{len(sessions)} session configs under {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
