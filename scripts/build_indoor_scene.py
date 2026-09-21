#!/usr/bin/env python3
"""Build the indoor apartment scene and export it as USD.

Three ways to run it.

1. CPU-only dry run -- no Isaac Sim needed, validates the scene file and writes
   the reproducibility manifest so a scene change can be reviewed in CI:

       python3 scripts/build_indoor_scene.py --config configs/scenes/indoor_apartment.yaml

2. Headless export inside Isaac Sim -- writes the ``.usda`` scene plus manifest:

       ~/isaacsim/python.sh scripts/build_indoor_scene.py \\
           --config configs/scenes/indoor_apartment.yaml --out artifacts/scenes --headless

3. Interactive build -- same export, then opens the scene in the Isaac Sim GUI and
   steps physics for a few seconds so dynamic furniture can be watched settling:

       ~/isaacsim/python.sh scripts/build_indoor_scene.py \\
           --config configs/scenes/indoor_apartment.yaml --out artifacts/scenes --gui

The scene is planned with pure Python *before* Isaac Sim boots, so an invalid
scene file fails immediately and never leaves a half-authored stage behind.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sim2sense_fall.scene_planner import ScenePlan, plan_scene, write_manifest  # noqa: E402
from sim2sense_fall.scene_spec import load_scene_spec  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "scenes" / "indoor_apartment.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "scenes"
DEFAULT_NAME = "indoor_apartment"

LOGGER = logging.getLogger("build_indoor_scene")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and export the indoor apartment USD scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="scene YAML file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--name", default=DEFAULT_NAME, help="output file stem")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and write the manifest only; never imports Isaac Sim",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="export without opening a window (default)",
    )
    mode.add_argument(
        "--gui",
        dest="headless",
        action="store_false",
        help="export and then show the scene in the Isaac Sim GUI",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="simulated seconds to step before reporting, GUI mode only",
    )
    parser.add_argument(
        "--exit-after-seconds",
        type=float,
        default=0.0,
        help="close the GUI after this many wall-clock seconds (0 = wait for the user)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


class Reporter:
    """Collect a build report, show it on the terminal and save it to disk.

    Kit replaces ``sys.stdout`` once the Omniverse runtime starts, which silently
    swallows ``print`` in headless mode. Writing to ``sys.__stdout__`` bypasses the
    redirect, and the report file guarantees the numbers survive either way.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)
        stream = getattr(sys, "__stdout__", None) or sys.stdout
        try:
            stream.write(text + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass

    def rule(self) -> None:
        self("=" * 72)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def report_plan(report: Reporter, plan: ScenePlan, manifest_path: Path) -> None:
    stats = plan.stats
    report.rule()
    report(f"scene            : {plan.scene_id}")
    report(f"description      : {plan.description.strip()[:120]}")
    report(f"frequency        : {plan.frequency_hz / 1e9:.3f} GHz")
    report(f"seed             : {plan.seed}")
    report(f"footprint        : {stats['footprint_m'][0]:.2f} m x {stats['footprint_m'][1]:.2f} m")
    report(
        f"floor area       : {stats['total_floor_area_m2']:.2f} m2 over {stats['room_count']} rooms"
    )
    report(f"wall area        : {stats['total_wall_area_m2']:.2f} m2")
    report(f"primitive count  : {stats['prim_count']} ({stats['geometry_count']} geometry)")
    report(f"lights           : {stats['light_count']}")
    report(
        f"rigid bodies     : {stats['rigid_body_count']} "
        f"(total {stats['rigid_body_mass_kg']:.2f} kg)"
    )
    report(f"by category      : {json.dumps(stats['primitives_by_category'])}")
    report(f"by room          : {json.dumps(stats['primitives_by_room'])}")
    report(f"materials used   : {len(plan.materials)}")
    for name, entry in sorted(plan.materials.items()):
        em = entry.get("electromagnetic")
        tag = em["sionna_material"] if em else "no RF mapping"
        friction = f"{entry['static_friction']:.2f}/{entry['dynamic_friction']:.2f}"
        report(
            f"  - {name:<18} friction {friction}"
            f"  density {entry['density_kg_m3']:>6.1f} kg/m3  -> {tag}"
        )
    report(f"manifest         : {manifest_path}")
    report.rule()


def boot_isaac(headless: bool) -> object:
    """Start the Omniverse runtime; must happen before importing ``pxr``.

    The script's own CLI arguments are removed from ``sys.argv`` first: the
    launcher forwards unrecognised arguments to Kit, which would otherwise
    receive ``--config``/``--out`` and complain about them.
    """

    import isaacsim
    from isaacsim.simulation_app import SimulationApp

    sys.argv = [sys.argv[0]]
    LOGGER.info("booting Isaac Sim (headless=%s) from %s", headless, Path(isaacsim.__file__).parent)
    return SimulationApp({"headless": headless, "width": 1600, "height": 900})


def settle_physics(app: object, seconds: float) -> None:
    """Activate PhysX and step the timeline so dynamic bodies reach rest."""

    if seconds <= 0:
        return
    from sim2sense_fall.isaac_scene import activate_physics, step_simulation

    activation = activate_physics()
    LOGGER.info("physics scene %s on %s", activation["physics_scenes"], activation["active_engine"])
    steps = step_simulation(app, seconds)
    LOGGER.info("stepped to %d physics frames", steps)


def run_gui(app: object, exit_after_seconds: float) -> None:
    import time

    deadline = time.monotonic() + exit_after_seconds if exit_after_seconds > 0 else None
    while app.is_running():
        app.update()
        if deadline is not None and time.monotonic() >= deadline:
            break


def keep_window_open(report: Reporter, app: object) -> None:
    report("GUI is open. Close the Isaac Sim window or press Ctrl-C to exit.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    report = Reporter()

    spec = load_scene_spec(args.config)
    plan = plan_scene(spec)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(plan, args.out / f"{args.name}.scene.json")
    report_path = args.out / f"{args.name}.build_report.txt"

    if args.dry_run:
        report_plan(report, plan, manifest_path)
        report("dry run only: pass --headless or --gui to author the USD scene")
        report.save(report_path)
        return 0

    usd_path = args.out / f"{args.name}.usda"
    app = boot_isaac(headless=args.headless)
    exit_code = 0
    try:
        from sim2sense_fall.isaac_scene import (
            build_stage,
            rigid_body_world_positions,
            stage_summary,
        )

        written = build_stage(plan, usd_path)
        summary = stage_summary(written)
        report_plan(report, plan, manifest_path)
        report(f"usd              : {written}")
        report(f"stage up axis    : {summary['up_axis']}  meters/unit {summary['meters_per_unit']}")
        report(f"stage prims      : {summary['prim_count']}  shapes {summary['shape_count']}")
        report(f"colliders        : {summary['collider_count']}")
        report(f"stage bodies     : {summary['rigid_body_count']}")
        report(f"semantic labels  : {len(summary['semantic_counts'])} kinds")

        if not args.headless:
            import omni.usd

            omni.usd.get_context().open_stage(str(written))
            settle_physics(app, args.settle_seconds)
            positions = rigid_body_world_positions(omni.usd.get_context().get_stage())
            report(f"rigid body rest  : {json.dumps(positions)}")
            keep_window_open(report, app)
            run_gui(app, args.exit_after_seconds)
    except Exception as exc:  # noqa: BLE001 - reported to the console and the report file
        LOGGER.exception("scene build failed")
        report(f"ERROR: {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        # The report must be written before the runtime shuts down: Isaac Sim's
        # fast shutdown terminates the interpreter, so anything after close() is
        # lost.
        report.save(report_path)
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
