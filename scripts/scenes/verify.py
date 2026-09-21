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

    ~/isaacsim/python.sh scripts/scenes/verify.py
    ~/isaacsim/python.sh scripts/scenes/verify.py --drop-height 0.25 --gui

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    parser.add_argument("--position-tolerance", type=float, default=SETTLE_TOLERANCE_M)
    parser.add_argument("--drop-tolerance", type=float, default=DROP_TOLERANCE_M)
    parser.add_argument("--dry-run", action="store_true", help="validate manifest on CPU only")
    parser.add_argument("--static-only", action="store_true", help="check USD without dynamics")
    args = parser.parse_args(argv)
    for name in ("settle_seconds", "drop_height", "position_tolerance", "drop_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.drop_height <= args.drop_tolerance:
        parser.error("--drop-height must exceed --drop-tolerance for a positive control")
    return args


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    from sim2sense_fall.scenes.verification import load_scene_manifest

    try:
        plan = load_scene_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        LOGGER.error("manifest validation failed: %s", exc)
        return 2
    if args.dry_run:
        print(f"CPU manifest check: PASS ({len(plan.prims)} primitives; USD/physics not checked)")
        return 0
    usd_path = args.usd.resolve()
    if not usd_path.is_file():
        LOGGER.error("scene not found: %s", usd_path)
        return 2
    original_hash = _sha256(usd_path)
    try:
        from isaacsim.simulation_app import SimulationApp

        sys.argv = [sys.argv[0]]
        app = SimulationApp({"headless": not args.gui, "width": 1600, "height": 900})
    except Exception as exc:
        LOGGER.error("Isaac Sim startup failed; use ~/isaacsim/python.sh: %s", exc)
        return 1
    checks = Checks()
    exit_code = 1
    workdir = tempfile.TemporaryDirectory(prefix="sim2sense-verify-")
    try:
        import omni.usd

        from sim2sense_fall.scenes.usd import (
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
        from pxr import Usd

        from sim2sense_fall.scenes.verification import stage_manifest_errors

        errors = stage_manifest_errors(Usd.Stage.Open(str(sandbox)), plan)
        checks.check(
            "USD matches every manifest primitive and material", not errors, "; ".join(errors[:10])
        )
        if errors:
            raise ValueError("USD/manifest mismatch; physics check stopped")

        if not omni.usd.get_context().open_stage(str(sandbox)):
            raise RuntimeError("Isaac Sim could not open the scene")
        for _ in range(5):
            app.update()

        if not args.static_only:
            activation = activate_physics()
            checks.check(
                "physics scene activated",
                activation["physics_scenes"] == ["/World/PhysicsScene"]
                and activation["active_engine"] == "physx",
                json.dumps(activation),
            )
        if not args.static_only:
            stage = omni.usd.get_context().get_stage()
            authored = rigid_body_world_positions(stage)
            checks.check("dynamic bodies reported", bool(authored), json.dumps(authored))

            # Phase 1 -- stable at rest.
            step_simulation(app, args.settle_seconds)
            settled = rigid_body_world_positions(stage)
            checks.check("settled body set preserved", set(settled) == set(authored))
            for path, position in settled.items():
                drift = max(abs(position[axis] - authored[path][axis]) for axis in range(3))
                checks.check(
                    f"{path} stays put at rest",
                    drift <= args.position_tolerance,
                    f"drift {drift:.4f} m",
                )

            # Phase 2 -- positive control: raised bodies must fall back.
            lifted = lift_rigid_bodies(stage, args.drop_height)
            for _ in range(5):
                app.update()
            step_simulation(app, args.settle_seconds)
            dropped = rigid_body_world_positions(stage)
            checks.check("dropped body set preserved", set(dropped) == set(authored))
            for path, position in dropped.items():
                fall = lifted[path][2] - position[2]
                residual = max(abs(position[axis] - authored[path][axis]) for axis in range(3))
                checks.check(
                    f"{path} falls back under gravity",
                    fall >= args.drop_height - args.drop_tolerance,
                    f"fell {fall:.4f} m from {lifted[path][2]:.4f} m to {position[2]:.4f} m",
                )
                checks.check(
                    f"{path} comes to rest where it started",
                    residual <= args.position_tolerance,
                    f"residual {residual:.4f} m",
                )

        if args.static_only:
            checks.lines.append("[SKIP] dynamics (--static-only)")

        if args.gui:
            print("GUI is open. Close the window or press Ctrl-C to exit.")
            while app.is_running():
                app.update()

        checks.check(
            "verification left the exported scene untouched",
            _sha256(usd_path) == original_hash,
            str(usd_path),
        )
        exit_code = 1 if checks.failures else 0
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
        app.close(exit_code=exit_code)
    return exit_code or (1 if checks.failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
