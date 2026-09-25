"""Measure how much physics time one ``app.update()`` actually advances.

The ``scripts/humans`` entry points assume each ``app.update()`` advances the
physics clock by exactly one configured ``physics_dt`` (``scenes.usd.step_simulation``
states the assumption in its docstring; ``verify.py``'s ``hold`` and the trial loop
in ``simulate.py`` both count ``seconds / physics_dt`` updates). The assumption is
a property of the ``omni.kit.loop-isaac`` loop runner, which is enabled in
``isaacsim.exp.base.kit`` but NOT in ``isaacsim.exp.base.python.kit`` -- the kit file
the Python entry points launch. Without it, the default Kit runner advances real
(render) time per update and PhysX subdivides that into ``physics_dt`` substeps.

This probe measures the actual advance rate through the channels the pipeline reads
(``SimulationManager`` step counter and simulation clock, ``RigidPrim`` pose reads)
and cross-checks the reported clock against free-falling spheres: each measured drop
must match ``1/2 g t^2`` for the physics time the counter reports, so a step count
cannot pass while the clock it reports describes no real motion. Each sphere is
dropped exactly once; tensor-backed prim *reads* are the only body API used, because
``RigidPrim.set_world_poses`` produced silent native deaths on this build.

Run under the Isaac interpreter::

    ~/isaacsim/python.sh scripts/humans/probe_loop_timing.py                 # current launch
    ~/isaacsim/python.sh scripts/humans/probe_loop_timing.py --enable-loop   # with the runner
    ~/isaacsim/python.sh scripts/humans/probe_loop_timing.py --manual-loop   # GUI-kit runLoops
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

faulthandler.enable()

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "humans"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_SCENE,
    Checks,
    activate_physics,
    open_scene,
    set_physics_dt,
)

#: The loop runner whose presence the entry points' timing arithmetic assumes.
LOOP_EXTENSION = "omni.kit.loop-isaac"
#: ``isaacsim.exp.base.kit`` (the GUI kit) sets these, ``isaacsim.exp.base.python.kit``
#: does not. ``manualModeEnabled`` stops the runner from advancing simulation time
#: itself so one ``app.update()`` performs exactly one physics step of ``physics_dt``;
#: without it the runner advances a fixed 1/60 s per update and PhysX subdivides that
#: into ``ceil(1/60 / physics_dt)`` substeps (measured: 2 steps per update).
MANUAL_LOOP_SETTINGS = [
    "--/app/runLoops/main/manualModeEnabled=true",
    "--/app/runLoops/main/rateLimitEnabled=false",
]
#: Free-fall probe starts high enough that even the slow (2x) loop cannot reach
#: furniture or walls inside the measurement window.
DROP_START_Z_M = 10.0
GRAVITY_M_S2 = 9.81
#: Living-room centre in the apartment plan, clear of every wall.
BALL_XY = (13.68, 3.30)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-loop",
        action="store_true",
        help=f"launch with --enable {LOOP_EXTENSION}",
    )
    parser.add_argument(
        "--manual-loop",
        action="store_true",
        help="inject the runLoops manual-mode settings from isaacsim.exp.base.kit "
        "(implies --enable-loop); candidate fix for 1 update = 1 physics step",
    )
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--physics-hz", type=float, default=120.0)
    parser.add_argument(
        "--play-speed",
        type=float,
        default=1.0,
        help="timeline play speed applied before measuring (0.5 halves the clock advance)",
    )
    return parser.parse_args(argv)


def _trace(label: str) -> None:
    """Progress marker on the original stderr: kit replaces both std streams."""

    import sys as _sys

    stream = getattr(_sys, "__stderr__", None) or _sys.stderr
    try:
        stream.write(f"[trace] {label}\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def author_ball(stage: object, path: str) -> None:
    """Author one dynamic sphere at the drop start height."""

    from pxr import Gf, UsdGeom, UsdPhysics

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(0.05)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
        Gf.Vec3d(BALL_XY[0], BALL_XY[1], DROP_START_Z_M)
    )


def measure_block(
    app: object,
    manager: object,
    ball_path: str,
    *,
    updates: int,
    physics_dt: float,
    stepper: str,
) -> dict[str, float]:
    """Drop one fresh ball through one stepping block and report what the clock did.

    The view is created here, after the timeline is already playing, and the ball
    falls exactly once: resetting a rigid body through ``RigidPrim`` writes is not
    survivable on this build (silent native death), so every block gets its own
    freshly authored sphere instead.
    """

    from isaacsim.core.experimental.prims import RigidPrim

    ball = RigidPrim(ball_path)
    z0 = float(np.asarray(ball.get_world_poses()[0]).reshape(-1, 3)[0, 2])
    steps0 = int(manager.get_num_physics_steps())
    time0 = float(manager.get_simulation_time())
    if stepper == "app_update":
        for _ in range(updates):
            app.update()
    else:
        manager.step(steps=updates, update_fabric=False)
    steps1 = int(manager.get_num_physics_steps())
    time1 = float(manager.get_simulation_time())
    z1 = float(np.asarray(ball.get_world_poses()[0]).reshape(-1, 3)[0, 2])
    delta_steps = steps1 - steps0
    delta_time = time1 - time0
    drop = z0 - z1
    expected_drop = 0.5 * GRAVITY_M_S2 * delta_time * delta_time
    return {
        "updates": updates,
        "delta_steps": delta_steps,
        "steps_per_update": delta_steps / updates,
        "delta_time_s": delta_time,
        "time_per_update_s": delta_time / updates,
        "physics_dt_s": physics_dt,
        "drop_m": drop,
        "expected_drop_m": expected_drop,
        "drop_error_m": drop - expected_drop,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = Checks()
    extra_args = []
    if args.manual_loop or args.enable_loop:
        extra_args += ["--enable", LOOP_EXTENSION]
    if args.manual_loop:
        extra_args += MANUAL_LOOP_SETTINGS
    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        checks.check("Isaac Sim importable", False, "run under ~/isaacsim/python.sh")
        return checks.report(banner="loop timing probe")
    argv_backup = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        app = SimulationApp(
            {"headless": True, "width": 640, "height": 480, "extra_args": extra_args}
        )
    finally:
        del argv_backup

    try:
        _trace("opening scene")
        stage = open_scene(app, DEFAULT_SCENE)
        checks.check("apartment scene opened", stage is not None, str(DEFAULT_SCENE))
        _trace("activating physics")
        activate_physics()
        physics_dt, accepted = set_physics_dt(1.0 / args.physics_hz)
        checks.check(
            "physics dt applied", accepted, f"{physics_dt!r} s (requested 1/{args.physics_hz:g})"
        )

        _trace("importing manager/pxr")
        import omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager

        _trace("authoring first ball")
        author_ball(stage, "/World/probe_ball_a")
        # Tensor-backed prim views must be built with the timeline playing: views
        # and writes while paused produced silent native deaths on this build.
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        if args.play_speed != 1.0:
            timeline.set_play_speed(args.play_speed)
            checks.info(f"timeline play speed set to {args.play_speed:g}")
        for _ in range(10):
            app.update()

        if args.manual_loop:
            mode = "with loop-isaac + runLoops manual mode (GUI-kit settings)"
        elif args.enable_loop:
            mode = "with omni.kit.loop-isaac"
        else:
            mode = "current launch (no loop-isaac)"
        checks.info(f"stepping via app.update(), {mode}")
        _trace("measuring app.update block")
        via_update = measure_block(
            app, SimulationManager, "/World/probe_ball_a",
            updates=args.updates, physics_dt=physics_dt, stepper="app_update",
        )
        _trace(f"app.update block done: {via_update}")
        checks.check(
            "one app.update() advances exactly one physics step",
            abs(via_update["steps_per_update"] - 1.0) < 0.01,
            f"{via_update['delta_steps']} steps over {args.updates} updates "
            f"({via_update['steps_per_update']:.3f} steps/update), "
            f"{via_update['time_per_update_s']*1000.0:.3f} ms/update "
            f"vs physics_dt {physics_dt*1000.0:.3f} ms",
        )
        checks.check(
            "app-update clock matches free fall",
            abs(via_update["drop_error_m"]) < max(0.02, 0.10 * via_update["expected_drop_m"]),
            f"ball dropped {via_update['drop_m']:.4f} m, 1/2 g t^2 for the reported "
            f"{via_update['delta_time_s']:.4f} s is {via_update['expected_drop_m']:.4f} m "
            f"(error {via_update['drop_error_m']*100.0:+.2f} cm)",
        )

        _trace("authoring second ball")
        author_ball(stage, "/World/probe_ball_b")
        for _ in range(10):
            app.update()  # let the new rigid body register with the sim view

        checks.info("stepping via SimulationManager.step(steps=n, update_fabric=False)")
        _trace("measuring manager block")
        via_manager = measure_block(
            app, SimulationManager, "/World/probe_ball_b",
            updates=args.updates, physics_dt=physics_dt, stepper="manager",
        )
        _trace(f"manager block done: {via_manager}")
        checks.check(
            "SimulationManager.step advances exactly the requested steps",
            via_manager["delta_steps"] == args.updates
            and abs(via_manager["delta_time_s"] - args.updates * physics_dt) < 1e-6,
            f"{via_manager['delta_steps']} steps, {via_manager['delta_time_s']:.6f} s "
            f"for {args.updates} x {physics_dt:.6f} s",
        )
        checks.check(
            "manager-stepped clock matches free fall",
            abs(via_manager["drop_error_m"]) < max(0.02, 0.10 * via_manager["expected_drop_m"]),
            f"ball dropped {via_manager['drop_m']:.4f} m, expected "
            f"{via_manager['expected_drop_m']:.4f} m "
            f"(error {via_manager['drop_error_m']*100.0:+.2f} cm)",
        )
        # The report must be written BEFORE app.close(): Kit's shutdown closes the
        # original stdout and Checks.report swallows the resulting write error,
        # silently dropping every verdict (measured 2026-09-24 -- the first run of
        # this probe logged its trace dicts but never its PASS/FAIL lines).
        exit_code = checks.report(banner="loop timing probe")
    finally:
        app.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
