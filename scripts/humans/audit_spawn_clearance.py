#!/usr/bin/env python3
"""Find keyboard-spawn points that are actually clear in the built apartment.

P1-B item "检查出生点/复位点的支撑面和人体净空" needs a measured answer, not the
single hand-placed spawn the shipped config carries. This walks a grid over every
room floor of the built scene and keeps the points where a standing human envelope
(radius ``--human-radius-m``, height ``--human-height-m``, floor at z=0) has no
positive-volume intersection with any collidable wall, opening or furniture prim, and
where the floor really extends underneath.

Output: ``spawn_clearance.json`` with the free-area fraction per room, the largest
inscribed clearance found, and ``--pick`` well-separated candidate spawns (greedy
max-min distance) that ``make_dataset_configs.py`` consumes.

CPU only: reads the scene manifest, no Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sim2sense_fall.scenes.geometry import WorldShape, intersects, world_shapes
from sim2sense_fall.scenes.verification import load_scene_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "artifacts/scenes/indoor_apartment.scene.json"
# Anything a standing person must not be inside. Floors are the support surface and
# ceilings/ground slabs sit outside the human envelope's z range anyway, but they are
# excluded by category so a low ceiling cannot be missed silently.
OBSTRUCTION_CATEGORIES = ("wall", "opening", "furniture", "furniture_body", "lighting_fixture")


def human_envelope(x: float, y: float, *, radius_m: float, height_m: float) -> WorldShape:
    return WorldShape(
        path="human", center=(x, y, height_m / 2), size=(2 * radius_m, 2 * radius_m, height_m),
        yaw_deg=0.0, cylinder=False,
    )


def floor_shapes(shapes: dict[str, WorldShape]) -> dict[str, WorldShape]:
    return {path: shape for path, shape in shapes.items() if path.endswith("/floor")}


def is_clear(
    point: tuple[float, float], *, floors: dict[str, WorldShape],
    obstacles: list[WorldShape], radius_m: float, height_m: float, margin_m: float,
) -> tuple[bool, str | None]:
    """Return (clear, floor path) for one candidate spawn point."""

    body = human_envelope(point[0], point[1], radius_m=radius_m + margin_m, height_m=height_m)
    supporting = None
    for path, floor in floors.items():
        low, high = floor.bounds
        if not (low[0] <= point[0] <= high[0] and low[1] <= point[1] <= high[1]):
            continue
        # The whole envelope must sit over one floor slab, not across a wall gap.
        if (low[0] <= body.bounds[0][0] and body.bounds[1][0] <= high[0]
                and low[1] <= body.bounds[0][1] and body.bounds[1][1] <= high[1]):
            supporting = path
            break
    if supporting is None:
        return False, None
    for obstacle in obstacles:
        if intersects(body, obstacle, tolerance=GEOMETRY_TOLERANCE):
            return False, obstacle.path
    return True, supporting


GEOMETRY_TOLERANCE = 1e-6


def candidates(
    scene: Path, *, radius_m: float, height_m: float, margin_m: float, step_m: float,
) -> dict[str, Any]:
    plan = load_scene_manifest(scene)
    shapes = world_shapes(plan.prims)
    floors = floor_shapes(shapes)
    # Everything except the support slabs themselves: walls, openings, furniture
    # (static and dynamic) and hanging fixtures.
    obstacles = [shape for path, shape in shapes.items()
                 if not path.endswith(("/floor", "/ceiling")) and "/foundation" not in path]
    rooms: dict[str, Any] = {}
    free_points: list[tuple[float, float, str]] = []
    for path, floor in sorted(floors.items()):
        low, high = floor.bounds
        xs = np.arange(low[0] + radius_m + margin_m, high[0] - radius_m - margin_m, step_m)
        ys = np.arange(low[1] + radius_m + margin_m, high[1] - radius_m - margin_m, step_m)
        if not len(xs) or not len(ys):
            rooms[path] = {"samples": 0, "clear_fraction": 0.0, "note": "floor too small"}
            continue
        grid = np.meshgrid(xs, ys, indexing="ij")
        total = clear = 0
        for x, y in zip(grid[0].ravel(), grid[1].ravel(), strict=True):
            total += 1
            ok, _ = is_clear((float(x), float(y)), floors=floors, obstacles=obstacles,
                             radius_m=radius_m, height_m=height_m, margin_m=margin_m)
            if ok:
                clear += 1
                free_points.append((round(float(x), 3), round(float(y), 3), path))
        rooms[path] = {"samples": total, "clear_fraction": clear / max(total, 1),
                       "bounds_xy": [list(low[:2]), list(high[:2])]}
    return {"rooms": rooms, "free_points": free_points,
            "grid_step_m": step_m, "human_radius_m": radius_m,
            "human_height_m": height_m, "margin_m": margin_m}


def pick_spawns(free_points: list[tuple[float, float, str]], *, count: int) -> list[dict[str, Any]]:
    """Greedy farthest-point sampling over the clear points, so spawns are separated."""

    if not free_points:
        return []
    points = np.array([[x, y] for x, y, _ in free_points])
    # Start from the point with the largest clearance proxy: distance to the nearest
    # blocked sample, approximated by distance to the room bounds centre.
    chosen = [int(np.argmax(np.hypot(*(points - points.mean(axis=0)).T)))]
    while len(chosen) < min(count, len(points)):
        distances = np.linalg.norm(
            points[:, None, :] - points[chosen][None, :, :], axis=2
        ).min(axis=1)
        chosen.append(int(np.argmax(distances)))
    return [{
        "xy": [round(float(points[i][0]), 3), round(float(points[i][1]), 3)],
        "room": free_points[i][2],
    } for i in chosen]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts/humans/spawn_clearance.json")
    parser.add_argument("--human-radius-m", type=float, default=0.30)
    parser.add_argument("--human-height-m", type=float, default=1.72)
    parser.add_argument("--margin-m", type=float, default=0.05,
                        help="extra clearance demanded around the envelope")
    parser.add_argument("--step-m", type=float, default=0.10)
    parser.add_argument("--pick", type=int, default=8)
    args = parser.parse_args()
    for name, value in (("human_radius_m", args.human_radius_m),
                        ("human_height_m", args.human_height_m), ("step_m", args.step_m)):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")
    if args.margin_m < 0 or args.pick < 1:
        parser.error("margin-m must be >= 0 and pick >= 1")

    result = candidates(
        args.scene, radius_m=args.human_radius_m, height_m=args.human_height_m,
        margin_m=args.margin_m, step_m=args.step_m,
    )
    result["picked_spawns"] = pick_spawns(result["free_points"], count=args.pick)
    del result["free_points"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for path, row in result["rooms"].items():
        print(f"{path:34s} clear {row['clear_fraction'] * 100:5.1f}%  ({row['samples']} samples)")
    print("picked spawns:")
    for row in result["picked_spawns"]:
        print(f"  {row['room']:34s} {row['xy']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
