"""Spawn clearance probes on a synthetic room (no scene file, so it runs anywhere).

The real audit reads ``artifacts/scenes/indoor_apartment.scene.json``, which is
gitignored, so these tests exercise the geometry helpers directly: a 10 x 10 floor with
one wall block, and the run-length probe that decides whether a spawn may be given a
walking protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))

from audit_spawn_clearance import clearance_probe, is_clear  # noqa: E402

from sim2sense_fall.scenes.geometry import WorldShape  # noqa: E402

FLOOR = WorldShape("rooms/a/floor", (5.0, 5.0, -0.06), (10.0, 10.0, 0.12), 0.0, False)
WALL = WorldShape("rooms/a/walls/east", (6.0, 5.0, 1.4), (0.2, 6.0, 2.8), 0.0, False)
FLOORS = {FLOOR.path: FLOOR}
OBSTACLES = [WALL]


def _probe(x: float, y: float, *, radius_m: float = 0.3, margin_m: float = 0.05) -> dict:
    return clearance_probe(
        (x, y), floors=FLOORS, obstacles=OBSTACLES, radius_m=radius_m, height_m=1.72,
        margin_m=margin_m, step_m=0.1, max_run_m=4.0,
    )


def test_a_clear_point_with_floor_underneath_is_accepted():
    ok, supporting = is_clear((2.0, 2.0), floors=FLOORS, obstacles=OBSTACLES,
                              radius_m=0.3, height_m=1.72, margin_m=0.05)
    assert ok and supporting == FLOOR.path


def test_a_point_inside_the_wall_is_rejected_and_names_the_blocker():
    ok, blocker = is_clear((6.0, 5.0), floors=FLOORS, obstacles=OBSTACLES,
                           radius_m=0.3, height_m=1.72, margin_m=0.05)
    assert not ok and blocker == WALL.path


def test_run_length_stops_at_the_wall_and_is_capped_by_max_run():
    runs = _probe(2.0, 2.0)
    # Wall face at x=5.9; envelope radius 0.3 + margin 0.05 -> the last clear centre is
    # x=5.5, i.e. a 3.5 m run from x=2.0.
    assert runs["plus_x"] == pytest.approx(3.5, abs=0.1), runs
    # Floor spans [0, 10] on both axes, so the negative directions stop 0.35 m short of
    # the edge (a 1.6 m run from 2.0) and the positive y run hits the probe cap.
    assert runs["minus_x"] == pytest.approx(1.6, abs=0.1), runs
    assert runs["minus_y"] == pytest.approx(1.6, abs=0.1), runs
    assert runs["plus_y"] == pytest.approx(3.9, abs=0.1), "capped at max_run_m"


def test_a_wider_envelope_reaches_less_far():
    narrow = _probe(2.0, 2.0, radius_m=0.3)
    wide = _probe(2.0, 2.0, radius_m=0.9)
    assert wide["plus_x"] < narrow["plus_x"]
    # The radius difference (0.6 m) is eaten from both the wall side and the floor edge.
    assert narrow["plus_x"] - wide["plus_x"] == pytest.approx(0.6, abs=0.1), (narrow, wide)
    assert narrow["minus_y"] - wide["minus_y"] == pytest.approx(0.6, abs=0.1)


def test_probe_arguments_are_validated():
    with pytest.raises(ValueError, match="step_m"):
        clearance_probe((2.0, 2.0), floors=FLOORS, obstacles=OBSTACLES, radius_m=0.3,
                        height_m=1.72, margin_m=0.0, step_m=0.0, max_run_m=4.0)
    with pytest.raises(ValueError, match="max_run_m"):
        clearance_probe((2.0, 2.0), floors=FLOORS, obstacles=OBSTACLES, radius_m=0.3,
                        height_m=1.72, margin_m=0.0, step_m=0.1, max_run_m=float("nan"))
