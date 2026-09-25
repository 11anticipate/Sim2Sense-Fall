#!/usr/bin/env python3
"""Open an exported indoor scene in the Isaac Sim GUI.

This is the convenience wrapper around Isaac Sim's standalone launcher. It does
not rebuild anything -- it opens the ``.usda`` produced by
``scripts/scenes/build.py`` and keeps the viewport alive until the window is
closed.

    ~/isaacsim/python.sh scripts/scenes/view.py
    ~/isaacsim/python.sh scripts/scenes/view.py --usd artifacts/scenes/indoor_apartment.usda

The equivalent manual route, if you prefer to use Isaac Sim's own launcher, is:

    cd ~/isaacsim && ./isaac-sim.sh
    # then File > Open ... and pick the exported .usda

Both routes need a GPU and a display; for a remote/SSH session use Isaac Sim's
streaming launcher instead (``~/isaacsim/isaac-sim.streaming.sh``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_USD = REPO_ROOT / "artifacts" / "scenes" / "indoor_apartment.usda"

LOGGER = logging.getLogger("view_indoor_scene")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an exported scene in the Isaac Sim GUI.")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help="USD scene to open")
    parser.add_argument(
        "--view",
        choices=["top", "roofless", "exterior"],
        default="top",
        help="inspection view: top (default), roofless oblique, or closed exterior",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="simulated seconds to step immediately after opening",
    )
    parser.add_argument(
        "--activate-physics",
        action="store_true",
        help=(
            "attach PhysX to the stage and start on the timeline; without this a "
            "standalone session shows a static scene"
        ),
    )
    parser.add_argument(
        "--exit-after-seconds",
        type=float,
        default=0.0,
        help="close automatically after this many wall-clock seconds (0 = wait for the user)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    usd_path = args.usd if args.usd.is_absolute() else (Path.cwd() / args.usd)
    if not usd_path.is_file():
        print(f"scene not found: {usd_path}", file=sys.stderr)
        print(
            "build it first:\n  ~/isaacsim/python.sh scripts/scenes/build.py --headless",
            file=sys.stderr,
        )
        return 2

    import isaacsim
    from isaacsim.simulation_app import SimulationApp

    LOGGER.info("opening %s with Isaac Sim", usd_path)
    sys.argv = [sys.argv[0]]
    app = SimulationApp(
        {"headless": False, "width": 1600, "height": 900, "open_usd": str(usd_path)}
    )
    exit_code = 1
    try:
        import omni.usd

        from sim2sense_fall.scenes.view import apply_inspection_view

        for _ in range(5):
            app.update()
        apply_inspection_view(
            omni.usd.get_context().get_stage(),
            mode=args.view,
            aspect_ratio=1600 / 900,
            require_viewport=True,
        )
        LOGGER.info("inspection view: %s (temporary session layer)", args.view)
        if args.activate_physics or args.settle_seconds > 0:
            from sim2sense_fall.scenes.usd import activate_physics, step_simulation

            activation = activate_physics()
            LOGGER.info(
                "physics scene %s on %s", activation["physics_scenes"], activation["active_engine"]
            )
            if args.settle_seconds > 0:
                step_simulation(app, args.settle_seconds)
        import time

        print("GUI is open. Close the Isaac Sim window or press Ctrl-C to exit.")
        deadline = (
            time.monotonic() + args.exit_after_seconds if args.exit_after_seconds > 0 else None
        )
        while app.is_running():
            app.update()
            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        exit_code = 130
    else:
        exit_code = 0
    finally:
        app.close(exit_code=exit_code)
    del isaacsim
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
