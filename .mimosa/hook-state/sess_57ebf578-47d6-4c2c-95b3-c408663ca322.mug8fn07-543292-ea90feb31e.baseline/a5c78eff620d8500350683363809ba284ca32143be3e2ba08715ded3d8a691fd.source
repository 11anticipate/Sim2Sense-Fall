"""Measure whether USD-stage transform reads track physics under the new stepper.

The AMASS physics trial died at frame 2298 with "none of the 1 reported contact
points fell inside a body capsule" (``artifacts/humans/trials_amass_anchored.log``).
``UsdHuman._capsule_volumes`` composes its capsule volumes from
``UsdGeom.Xformable.ComputeLocalToWorldTransform`` -- a **stage** read. The trial
loop was rewired to ``SimulationManager.step(steps=n, update_fabric=False)``, which
by design does not push physics results into fabric/USD, so the stage may still
hold the spawn pose while ``RigidPrim.get_world_poses`` (the tensor/physics read
that ``probe_loop_timing`` proved live) holds the current pose. If the stage is
stale, every capsule volume is frozen at the spawn pose and the first contact away
from it -- the first floor touch of a fall -- is unattributable. Exactly the
observed failure: attribution worked through the whole standing settle, then died
at the first fall contact.

This probe measures both read channels against the same falling sphere, and once
the sphere rests on a probe floor, checks where PhysX's reported contact point
lands relative to each channel's volume:

* stage read frozen + live read current -> the trial failure is explained and
  capsule volumes must be composed from live link poses plus the authored
  capsule-to-link offset;
* the resting point vs the live sphere tells whether the existing
  ``CAPSULE_CONTAINMENT_TOLERANCE_M`` covers the contact-offset shell;
* the static floor's stage read serves as a control: staleness must hit only
  *dynamic* prims, otherwise the reading channel itself is suspect.

Run under the Isaac interpreter::

    ~/isaacsim/python.sh scripts/humans/probe_stage_staleness.py
"""

from __future__ import annotations

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

GRAVITY_M_S2 = 9.81
BALL_RADIUS_M = 0.05
DROP_START_Z_M = 2.0
BALL_XY = (13.68, 3.30)
FALL_STEPS = 30
SETTLE_STEPS = 60
MAX_CONTACT_WAIT_STEPS = 300
CONTACT_TOLERANCE_M = 0.02
BALL_PATH = "/World/probe_stale_ball"
FLOOR_PATH = "/World/probe_stale_floor"


def _trace(label: str) -> None:
    """Progress marker on the original stderr: kit replaces both std streams."""

    import sys as _sys

    stream = getattr(_sys, "__stderr__", None) or _sys.stderr
    try:
        stream.write(f"[trace] {label}\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def _author_prims(stage: object) -> None:
    """One dynamic sphere above one static floor, both contact-report tagged."""

    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    floor = UsdGeom.Cube.Define(stage, FLOOR_PATH)
    floor.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(floor.GetPrim()).SetTranslate(
        Gf.Vec3d(BALL_XY[0], BALL_XY[1], -0.05)
    )
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(floor.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(floor.GetPrim()).CreateThresholdAttr().Set(0.0)

    sphere = UsdGeom.Sphere.Define(stage, BALL_PATH)
    sphere.CreateRadiusAttr(BALL_RADIUS_M)
    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
        Gf.Vec3d(BALL_XY[0], BALL_XY[1], DROP_START_Z_M)
    )
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
    collision.CreateContactOffsetAttr().Set(0.01)
    collision.CreateRestOffsetAttr().Set(0.0)
    PhysxSchema.PhysxContactReportAPI.Apply(sphere.GetPrim()).CreateThresholdAttr().Set(0.0)


def _stage_centre(stage: object, path: str) -> tuple[float, float, float]:
    """Origin of ``path`` in world coordinates, read from the USD stage."""

    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"stage prim missing: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    centre = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    return (float(centre[0]), float(centre[1]), float(centre[2]))


def _live_centre(ball: object) -> tuple[float, float, float]:
    """Ball centre via the tensor/physics read channel."""

    position = np.asarray(ball.get_world_poses()[0].numpy(), dtype=np.float64).reshape(-1, 3)[0]
    return (float(position[0]), float(position[1]), float(position[2]))


def _sphere_surface_distance(
    point: tuple[float, float, float], centre: tuple[float, float, float]
) -> float:
    """Distance from ``point`` to the surface of the probe sphere."""

    return float(np.linalg.norm(np.asarray(point) - np.asarray(centre)) - BALL_RADIUS_M)


def _report_points(interface: object) -> list[tuple[float, float, float]]:
    """Contact points from one poll of the PhysX report."""

    headers, data = interface.get_contact_report()
    points: list[tuple[float, float, float]] = []
    for header in headers:
        offset = int(header.contact_data_offset)
        for record in data[offset : offset + int(header.num_contact_data)]:
            position = record.position
            points.append(
                (float(position[0]), float(position[1]), float(position[2]))
            )
    return points


def main() -> int:
    checks = Checks()
    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        checks.check("Isaac Sim importable", False, "run under ~/isaacsim/python.sh")
        return checks.report(banner="stage staleness probe")
    argv_backup = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        app = SimulationApp({"headless": True, "width": 640, "height": 480})
    finally:
        del argv_backup

    try:
        _trace("opening scene")
        stage = open_scene(app, DEFAULT_SCENE)
        checks.check("apartment scene opened", stage is not None, str(DEFAULT_SCENE))
        _trace("activating physics")
        activate_physics()
        physics_dt, accepted = set_physics_dt(1.0 / 120.0)
        checks.check("physics dt applied", accepted, f"{physics_dt!r} s")

        _trace("authoring probe prims")
        _author_prims(stage)

        import omni.timeline
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(10):
            app.update()

        ball = RigidPrim(BALL_PATH)
        live0 = _live_centre(ball)
        stage0 = _stage_centre(stage, BALL_PATH)
        floor0 = _stage_centre(stage, FLOOR_PATH)
        checks.info(
            f"spawn: live z={live0[2]:.4f} stage z={stage0[2]:.4f} "
            f"floor stage z={floor0[2]:.4f}"
        )

        _trace(f"free-fall block: {FALL_STEPS} steps via SimulationManager.step")
        SimulationManager.step(steps=FALL_STEPS, update_fabric=False)
        live1 = _live_centre(ball)
        stage1 = _stage_centre(stage, BALL_PATH)
        floor1 = _stage_centre(stage, FLOOR_PATH)
        elapsed = FALL_STEPS * physics_dt
        expected_drop = 0.5 * GRAVITY_M_S2 * elapsed * elapsed
        live_drop = live0[2] - live1[2]
        stage_drop = stage0[2] - stage1[2]
        checks.info(
            f"after {elapsed:.4f} s: live drop {live_drop:.4f} m (expected "
            f"{expected_drop:.4f}), stage drop {stage_drop:.6f} m"
        )
        checks.check(
            "live RigidPrim read tracks free fall",
            abs(live_drop - expected_drop) < 0.02,
            f"dropped {live_drop:.4f} m vs {expected_drop:.4f} m expected",
        )
        checks.check(
            "static floor stage read unchanged (control)",
            abs(floor1[2] - floor0[2]) < 1e-6,
            f"floor stage z {floor0[2]:.4f} -> {floor1[2]:.4f}",
        )
        checks.check(
            "dynamic stage read frozen while physics moved",
            abs(stage_drop) < 1e-6,
            f"stage z {stage0[2]:.4f} -> {stage1[2]:.4f} "
            f"(drop {stage_drop:.6f} m vs live {live_drop:.4f} m)",
        )

        _trace("waiting for first floor contact")
        import omni.physx

        interface = omni.physx.get_physx_simulation_interface()
        first_points: list[tuple[float, float, float]] = []
        for _ in range(MAX_CONTACT_WAIT_STEPS):
            SimulationManager.step(steps=1, update_fabric=False)
            first_points = _report_points(interface)
            if first_points:
                break
        if not first_points:
            checks.check("sphere reached floor and reported a contact", False,
                         f"no contact within {MAX_CONTACT_WAIT_STEPS} steps")
            return checks.report(banner="stage staleness probe")

        live_contact = _live_centre(ball)
        stage_contact = _stage_centre(stage, BALL_PATH)
        point = first_points[0]
        live_surface = _sphere_surface_distance(point, live_contact)
        stage_surface = _sphere_surface_distance(point, stage_contact)
        checks.info(
            f"first contact point {point}: live centre {live_contact} "
            f"(surface dist {live_surface:+.4f} m), stage centre {stage_contact} "
            f"(surface dist {stage_surface:+.4f} m)"
        )
        checks.check(
            "contact point inside LIVE sphere (+2 cm tolerance)",
            live_surface <= CONTACT_TOLERANCE_M,
            f"surface distance {live_surface:+.4f} m vs tolerance {CONTACT_TOLERANCE_M} m",
        )
        checks.check(
            "contact point far outside STAGE sphere (failure mode reproduced)",
            stage_surface > 0.5,
            f"surface distance {stage_surface:+.4f} m from the frozen spawn pose",
        )

        _trace("settling for a resting-contact sample")
        rest_points: list[tuple[float, float, float]] = []
        for _ in range(SETTLE_STEPS):
            SimulationManager.step(steps=1, update_fabric=False)
            points = _report_points(interface)
            if points:
                rest_points = points
        if rest_points:
            live_rest = _live_centre(ball)
            stage_rest = _stage_centre(stage, BALL_PATH)
            rest = rest_points[0]
            checks.info(
                f"resting point {rest}: live centre {live_rest} (surface dist "
                f"{_sphere_surface_distance(rest, live_rest):+.4f} m), stage centre "
                f"{stage_rest} (surface dist {_sphere_surface_distance(rest, stage_rest):+.4f} m)"
            )
            checks.check(
                "resting contact inside LIVE sphere (+2 cm tolerance)",
                _sphere_surface_distance(rest, live_rest) <= CONTACT_TOLERANCE_M,
                "attribution stays healthy once volumes track the body",
            )
        # The report must be written BEFORE app.close(): Kit's shutdown closes
        # the original stdout and Checks.report swallows the resulting write
        # error, silently dropping every verdict (measured -- a report written
        # after close() never reached the log).
        exit_code = checks.report(banner="stage staleness probe")
    finally:
        app.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
