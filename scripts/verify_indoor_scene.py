#!/usr/bin/env python3
"""Load an exported indoor scene in Isaac Sim and check that it is physically sane.

This is the smoke test for the *authored* scene, not just the plan: the file
opens, colliders and rigid bodies are present, the exported footprint matches the
plan, and the physics scene really drives the bodies.

The dynamics check is deliberately a **positive control**. A scene where nothing
moves and a scene where physics is silently switched off look identical if you
only measure drift, so the test lifts every rigid body off the floor and requires
gravity and contact to put it back. That is what makes "the chair rests on the
floor" a real result rather than a vacuous one.

Isaac Sim's stage-open hooks add ``/Render`` and viewport cameras to the in-memory
layer, so the test runs against a **temporary copy** and hashes the original
before and after. That keeps the reported primitive count honest and proves the
check does not mutate the shipped asset.

    ~/isaacsim/python.sh scripts/verify_indoor_scene.py
    ~/isaacsim/python.sh scripts/verify_indoor_scene.py --drop-height 0.25 --gui

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_USD = REPO_ROOT / "artifacts" / "scenes" / "indoor_apartment.usda"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "scenes" / "indoor_apartment.scene.json"

LOGGER = logging.getLogger("verify_indoor_scene")

#: A settled body may not end up further than this from where it started.
SETTLE_TOLERANCE_M = 0.02
#: A dropped body must actually come back down by about the height it was lifted.
DROP_TOLERANCE_M = 0.02
#: Expected overall footprint of the shipping apartment scene.
EXPECTED_FOOTPRINT_M = (8.4, 7.0)
#: Primitives in the exported file as written, before Isaac Sim opens it.
EXPECTED_FILE_PRIMS = 344


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test an exported indoor scene.")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help="scene to verify")
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="plan manifest to cross-check"
    )
    parser.add_argument(
        "--settle-seconds", type=float, default=3.0, help="simulated seconds per settle phase"
    )
    parser.add_argument(
        "--drop-height", type=float, default=0.25, help="positive-control lift height in metres"
    )
    parser.add_argument("--gui", action="store_true", help="keep the viewport open afterwards")
    return parser.parse_args(argv)


class Checks:
    """Tiny assertion recorder that reports everywhere, including under Kit."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.lines: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        self.lines.append(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not condition:
            self.failures.append(label)

    def report(self) -> None:
        stream = getattr(sys, "__stdout__", None) or sys.stdout
        try:
            stream.write("\n".join(self.lines) + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass


def measure_floor_footprint(stage: object) -> tuple[float, float] | None:
    """Return the ``(x, y)`` extent of the room floor slabs."""

    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    found = False
    for prim in stage.Traverse():
        if prim.GetName() != "floor":
            continue
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = aligned.GetMin(), aligned.GetMax()
        x_min, y_min = min(x_min, low[0]), min(y_min, low[1])
        x_max, y_max = max(x_max, high[0]), max(y_max, high[1])
        found = True
    if not found:
        return None
    return (x_max - x_min, y_max - y_min)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    usd_path = args.usd if args.usd.is_absolute() else (Path.cwd() / args.usd)
    if not usd_path.is_file():
        print(f"scene not found: {usd_path}", file=sys.stderr)
        return 2
    original_hash = _sha256(usd_path)

    from isaacsim.simulation_app import SimulationApp

    sys.argv = [sys.argv[0]]
    app = SimulationApp({"headless": not args.gui, "width": 1600, "height": 900})
    checks = Checks()
    exit_code = 0
    workdir = tempfile.TemporaryDirectory(prefix="sim2sense-verify-")
    try:
        import omni.usd

        from sim2sense_fall.isaac_scene import (
            activate_physics,
            lift_rigid_bodies,
            rigid_body_world_positions,
            stage_summary,
            step_simulation,
        )

        # Work on a copy: Isaac Sim's stage-open hooks add /Render and viewport
        # cameras to the in-memory layer, which would otherwise inflate the
        # primitive count reported here and could be written back on save.
        sandbox = Path(workdir.name) / usd_path.name
        shutil.copy2(usd_path, sandbox)

        summary = stage_summary(sandbox)
        checks.check("scene opens", summary["prim_count"] > 0, f"{summary['prim_count']} prims")
        checks.check(
            "exported file has the expected primitive count",
            summary["prim_count"] == EXPECTED_FILE_PRIMS,
            f"{summary['prim_count']} vs {EXPECTED_FILE_PRIMS}",
        )
        checks.check(
            "Z-up with metres as the unit",
            summary["up_axis"] == "Z" and summary["meters_per_unit"] == 1.0,
            f"up={summary['up_axis']} m/unit={summary['meters_per_unit']}",
        )
        checks.check(
            "colliders cover the geometry",
            summary["collider_count"] == summary["shape_count"] == 231,
            f"{summary['collider_count']} colliders / {summary['shape_count']} shapes",
        )
        checks.check("rigid bodies authored", summary["rigid_body_count"] >= 1)

        if args.manifest.is_file():
            stats = json.loads(args.manifest.read_text(encoding="utf-8"))["stats"]
            checks.check(
                "authored geometry matches the plan",
                summary["shape_count"] == stats["geometry_count"],
                f"{summary['shape_count']} vs {stats['geometry_count']}",
            )
            checks.check(
                "authored rigid bodies match the plan",
                summary["rigid_body_count"] == stats["rigid_body_count"],
                f"{summary['rigid_body_count']} vs {stats['rigid_body_count']}",
            )
        else:
            checks.check("plan manifest available", False, str(args.manifest))

        omni.usd.get_context().open_stage(str(sandbox))
        for _ in range(5):
            app.update()

        footprint = measure_floor_footprint(omni.usd.get_context().get_stage())
        if footprint is None:
            checks.check("floor slabs found", False)
        else:
            checks.check(
                "floor footprint matches the plan",
                abs(footprint[0] - EXPECTED_FOOTPRINT_M[0]) < 0.02
                and abs(footprint[1] - EXPECTED_FOOTPRINT_M[1]) < 0.02,
                f"{footprint[0]:.2f} m x {footprint[1]:.2f} m",
            )

        activation = activate_physics()
        # ``simulating`` is False until the timeline plays, so assert on the scene
        # and engine instead; the drop control below is what proves physics runs.
        checks.check(
            "physics scene activated",
            activation["physics_scenes"] == ["/World/PhysicsScene"]
            and activation["active_engine"] == "physx",
            json.dumps(activation),
        )

        if args.settle_seconds > 0:
            stage = omni.usd.get_context().get_stage()
            authored = rigid_body_world_positions(stage)
            checks.check("dynamic bodies reported", bool(authored), json.dumps(authored))

            # Phase 1 -- stable at rest.
            step_simulation(app, args.settle_seconds)
            settled = rigid_body_world_positions(stage)
            for path, position in settled.items():
                drift = max(abs(position[axis] - authored[path][axis]) for axis in range(3))
                checks.check(
                    f"{path} stays put at rest",
                    drift <= SETTLE_TOLERANCE_M,
                    f"drift {drift:.4f} m",
                )

            # Phase 2 -- positive control: raised bodies must fall back.
            lifted = lift_rigid_bodies(stage, args.drop_height)
            for _ in range(5):
                app.update()
            step_simulation(app, args.settle_seconds)
            dropped = rigid_body_world_positions(stage)
            for path, position in dropped.items():
                fall = lifted[path][2] - position[2]
                residual = max(abs(position[axis] - authored[path][axis]) for axis in range(3))
                checks.check(
                    f"{path} falls back under gravity",
                    fall >= args.drop_height - DROP_TOLERANCE_M,
                    f"fell {fall:.4f} m from {lifted[path][2]:.4f} m to {position[2]:.4f} m",
                )
                checks.check(
                    f"{path} comes to rest where it started",
                    residual <= SETTLE_TOLERANCE_M,
                    f"residual {residual:.4f} m",
                )

        if args.gui:
            print("GUI is open. Close the window or press Ctrl-C to exit.")
            while app.is_running():
                app.update()

        checks.check(
            "verification left the exported scene untouched",
            _sha256(usd_path) == original_hash,
            str(usd_path),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in the check report
        LOGGER.exception("verification failed")
        checks.check("verification completed without error", False, f"{type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        workdir.cleanup()
        checks.report()
        print(f"verification: {'FAILED' if checks.failures else 'PASSED'}")
        if checks.failures:
            print("failed checks: " + ", ".join(checks.failures))
        app.close()
    return exit_code or (1 if checks.failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
