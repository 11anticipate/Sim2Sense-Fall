"""The train02 session matrix must respect measured floor, not hope.

`build_matrix` decides which spawn gets which protocol from the clearance audit's run
lengths. A walk that collides with furniture yields a refused segment and a wasted Isaac
session, so this pins the scheduling rules: only feasible pairs are emitted, the rare
activity classes still appear under a session cap, and the same audit gives the same
schedule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))

from make_dataset_configs import (  # noqa: E402
    PROTOCOLS,
    Protocol,
    build_matrix,
    validate,
)


def spawn(x: float, y: float, *, plus_x: float, minus_x: float, plus_y: float,
          minus_y: float, room: str = "rooms/hall/floor") -> dict:
    return {"xy": [x, y], "room": room,
            "runs_m": {"plus_x": plus_x, "minus_x": minus_x, "plus_y": plus_y,
                       "minus_y": minus_y}}


NARROW = [spawn(1.0, 1.0, plus_x=0.6, minus_x=0.6, plus_y=0.6, minus_y=0.6)]
# 4 s at 0.4 m/s needs 1.6 m + the 0.4 m safety margin, so only the long axis fits.
LONG_X = [spawn(2.0, 2.0, plus_x=3.9, minus_x=3.9, plus_y=0.6, minus_y=0.6)]
BASE = {"speed_m_s": 0.4}


def test_infeasible_walks_are_dropped_instead_of_colliding():
    sessions = build_matrix(NARROW, protocols=list(PROTOCOLS), speeds_m_s=[0.4],
                            max_sessions=40)
    assert sessions, "a clear point must still host the non-walking protocols"
    assert all("walk_patrol" not in name and "fall_forward" not in name
               for name in sessions), sorted(sessions)


def test_a_walking_protocol_is_placed_on_the_axis_that_fits():
    sessions = build_matrix(LONG_X, protocols=list(PROTOCOLS), speeds_m_s=[0.4],
                            max_sessions=40)
    patrol = [row for name, row in sessions.items() if "walk_patrol" in name]
    assert patrol, sorted(sessions)
    assert patrol[0]["heading_deg"] in (0.0, 180.0), "the x axis is the only one long enough"


def test_the_cap_still_covers_every_protocol_class():
    # Wide-open spawns: the longest patrol leg (12 s x 0.4 m/s + drift allowance)
    # needs ~5.6 m, so a 3.9 m room would legitimately exclude that class.
    spawns = [spawn(float(x), 1.0, plus_x=12.0, minus_x=12.0, plus_y=12.0, minus_y=12.0)
              for x in range(1, 9)]
    sessions = build_matrix(spawns, protocols=list(PROTOCOLS), speeds_m_s=[0.4],
                            max_sessions=len(PROTOCOLS))
    kinds = {protocol.kind for protocol in PROTOCOLS}
    present = {kind for name in sessions for kind in kinds if kind in name}
    assert present == kinds, f"{sorted(kinds - present)} vanished under the cap"


def test_the_schedule_is_deterministic_for_a_given_audit():
    """Same audit file, same schedule -- the batch must be reproducible.

    Order-independence is deliberately *not* claimed: under a session cap the schedule
    follows the audit's own order, which is deterministic because the audit scans a fixed
    grid of a fixed scene.
    """

    spawns = [spawn(float(x), 1.0, plus_x=3.9, minus_x=3.9, plus_y=3.9, minus_y=3.9)
              for x in range(1, 6)]
    first = build_matrix(spawns, protocols=list(PROTOCOLS), speeds_m_s=[0.4], max_sessions=20)
    second = build_matrix(spawns, protocols=list(PROTOCOLS), speeds_m_s=[0.4], max_sessions=20)
    assert list(first) == list(second)
    assert first == second
    assert len(first) == 20, "the cap must be reachable with five open spawns"


def test_an_unaudited_spawn_is_rejected_before_it_reaches_isaac(tmp_path):
    spec = {"spawn_xy": [99.0, 99.0], "heading_deg": 0.0, "demo": [(2.0, ["W"])]}
    with pytest.raises(ValueError, match="not in"):
        validate(BASE, "ghost", spec, clear={(1.0, 1.0)})


def test_a_keyless_session_needs_too_long_to_be_worth_tracing():
    """Quiet standing is legitimate exposure, a 2 s idle clip is not."""

    short = Protocol("idle", 0.0, 0.0, False, ((3.0, ()), (3.0, ())))
    assert short.demo() == [(3.0, []), (3.0, [])]
    with pytest.raises(ValueError, match="presses no keys"):
        validate(BASE, "nothing", {"spawn_xy": [1.0, 1.0], "heading_deg": 0.0,
                                   "demo": short.demo()}, clear={(1.0, 1.0)})
    long_stand = next(p for p in PROTOCOLS if p.kind == "adl_long_stand")
    spec = {"spawn_xy": [1.0, 1.0], "heading_deg": 0.0, "demo": long_stand.demo()}
    payload = validate(BASE, "quiet", spec, clear={(1.0, 1.0)})
    assert sum(step["duration_s"] for step in payload["demo"]) >= 40.0
    assert all(not step["keys"] for step in payload["demo"])
