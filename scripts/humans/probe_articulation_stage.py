"""Measure articulation-link pose writeback and contact-point/pose agreement.

Two facts are still unmeasured after the sphere probe
(``probe_stage_staleness.log``) proved that a *free rigid body's* USD-stage
transform tracks physics under ``SimulationManager.step(update_fabric=False)``:

1. **Articulation links.** The human is an articulation; if link writeback to
   the stage differs from free-body writeback, every capsule volume read from
   the stage freezes and the first fall contact away from the standing pose is
   unattributable -- the exact frame-2298 signature of the AMASS trial.
2. **Contact point vs pose timing.** The known report-lags-one-step hazard
   means a reported point may describe the *previous* step. During fast limb
   motion a 1-step-old point can sit centimetres outside the current capsule --
   the ``CAPSULE_CONTAINMENT_TOLERANCE_M`` of 2 cm may or may not absorb it.

This probe drops a free-falling two-link articulation (capsule links, the
human's contact/rest offsets, both contact-report tagged), compares the child
link's stage read against its live physics read at every step, and for every
reported contact point measures the distance to the contacting capsules at the
current pose and at the previous step's pose.

Run under the Isaac interpreter::

    ~/isaacsim/python.sh scripts/humans/probe_articulation_stage.py
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

BALL_XY = (13.68, 3.30)
LINK_RADIUS_M = 0.05
LINK_HEIGHT_M = 0.2
CONTACT_TOLERANCE_M = 0.02
TOTAL_STEPS = 120
ART_PATH = "/World/probe_art"
CHILD_PATH = f"{ART_PATH}/child_link"
ROOT_PATH = f"{ART_PATH}/root_link"


def _trace(label: str) -> None:
    """Progress marker on the original stderr: kit replaces both std streams."""

    import sys as _sys

    stream = getattr(_sys, "__stderr__", None) or _sys.stderr
    try:
        stream.write(f"[trace] {label}\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def _author_articulation(stage: object) -> None:
    """Two capsule links joined by a revolute joint, dropped as one articulation."""

    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    art = UsdGeom.Xform.Define(stage, ART_PATH)
    UsdPhysics.ArticulationRootAPI.Apply(art.GetPrim())

    def capsule_link(path: str, centre_z: float) -> None:
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateRadiusAttr(LINK_RADIUS_M)
        capsule.CreateHeightAttr(LINK_HEIGHT_M)
        UsdGeom.XformCommonAPI(capsule.GetPrim()).SetTranslate(
            Gf.Vec3d(BALL_XY[0], BALL_XY[1], centre_z)
        )
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(capsule.GetPrim())
        collision.CreateContactOffsetAttr().Set(0.01)
        collision.CreateRestOffsetAttr().Set(0.0)
        PhysxSchema.PhysxContactReportAPI.Apply(capsule.GetPrim()).CreateThresholdAttr().Set(0.0)

    capsule_link(ROOT_PATH, 1.60)
    capsule_link(CHILD_PATH, 1.30)

    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ART_PATH}/joint")
    joint.CreateBody0Rel([ROOT_PATH])
    joint.CreateBody1Rel([CHILD_PATH])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -0.15))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.15))


def _pose(prim_view: object) -> tuple[np.ndarray, np.ndarray]:
    """Live world pose (position, w-first quaternion) of one prim view."""

    positions, orientations = prim_view.get_world_poses()
    position = np.asarray(positions.numpy(), dtype=np.float64).reshape(-1, 3)[0]
    quaternion = np.asarray(orientations.numpy(), dtype=np.float64).reshape(-1, 4)[0]
    return position, quaternion


def _stage_pose(stage: object, path: str) -> np.ndarray:
    """Origin of ``path`` in world coordinates, read from the USD stage."""

    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    centre = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    return np.array([float(centre[0]), float(centre[1]), float(centre[2])])


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a w-first unit quaternion."""

    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _capsule_surface_distance(
    point: np.ndarray,
    centre: np.ndarray,
    rotation: np.ndarray,
) -> float:
    """Distance from ``point`` to the surface of the probe link's capsule."""

    axis_half = rotation @ np.array([0.0, 0.0, LINK_HEIGHT_M * 0.5])
    axis_sq = float(axis_half @ axis_half)
    offset = point - centre
    if axis_sq > 0.0:
        along = float(offset @ axis_half) / axis_sq
        along = min(1.0, max(-1.0, along))
        offset = offset - along * axis_half
    return float(np.linalg.norm(offset) - LINK_RADIUS_M)


def _report_points(interface: object) -> list[tuple[float, float, float]]:
    """Contact points from one poll of the PhysX report."""

    headers, data = interface.get_contact_report()
    points: list[tuple[float, float, float]] = []
    for header in headers:
        offset = int(header.contact_data_offset)
        for record in data[offset : offset + int(header.num_contact_data)]:
            position = record.position
            points.append((float(position[0]), float(position[1]), float(position[2])))
    return points


def main() -> int:
    checks = Checks()
    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        checks.check("Isaac Sim importable", False, "run under ~/isaacsim/python.sh")
        return checks.report(banner="articulation stage probe")
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

        _trace("authoring two-link articulation")
        _author_articulation(stage)
        _trace("authoring done")

        import omni.physx
        import omni.timeline
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(10):
            app.update()
        _trace("timeline playing, registration updates done")

        root = RigidPrim(ROOT_PATH)
        child = RigidPrim(CHILD_PATH)
        interface = omni.physx.get_physx_simulation_interface()
        _trace("views and report interface bound")

        child_live: list[float] = []
        child_stage: list[float] = []
        child_prev: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        worst_same = 0.0
        worst_prev = 0.0
        worst_contact_now = 0.0
        worst_contact_prev = 0.0
        contact_steps = 0

        _trace(f"stepping {TOTAL_STEPS} steps, sampling every step")
        for step_index in range(TOTAL_STEPS):
            SimulationManager.step(steps=1, update_fabric=False)
            if step_index % 20 == 0:
                _trace(f"step {step_index}")
            position, quaternion = _pose(child)
            stage_position = _stage_pose(stage, CHILD_PATH)
            child_live.append(float(position[2]))
            child_stage.append(float(stage_position[2]))
            if len(child_live) > 1:
                worst_same = max(worst_same, abs(child_live[-1] - child_stage[-1]))
                worst_prev = max(worst_prev, abs(child_live[-2] - child_stage[-1]))

            points = _report_points(interface)
            if points:
                contact_steps += 1
                views = {
                    "root": _pose(root),
                    "child": (position, quaternion),
                }
                for point in points:
                    point_array = np.asarray(point, dtype=np.float64)
                    distances = {
                        name: _capsule_surface_distance(
                            point_array, view[0], _rotation(view[1])
                        )
                        for name, view in views.items()
                    }
                    nearest_now = min(distances.values())
                    worst_contact_now = max(worst_contact_now, nearest_now)
                    if child_prev:
                        previous = {
                            name: _capsule_surface_distance(
                                point_array, old[0], _rotation(old[1])
                            )
                            for name, old in child_prev.items()
                        }
                        worst_contact_prev = max(worst_contact_prev, min(previous.values()))
            child_prev = {"root": _pose(root), "child": (position, quaternion)}

        checks.info(
            f"child link over {TOTAL_STEPS} steps: max |stage - live (same step)| = "
            f"{worst_same:.6f} m, max |stage - live (previous step)| = {worst_prev:.6f} m"
        )
        checks.info(
            f"contacts reported on {contact_steps} steps: worst point-to-capsule "
            f"surface distance {worst_contact_now:+.4f} m against the current pose, "
            f"{worst_contact_prev:+.4f} m against the previous step's pose"
        )
        checks.check(
            "articulation child link stage read tracks live physics",
            worst_same < 1e-4,
            f"max same-step stage-vs-live difference {worst_same:.6f} m",
        )
        checks.check(
            "stage read is current, not a one-step-old copy",
            worst_same <= worst_prev + 1e-9,
            f"same-step error {worst_same:.6f} m vs previous-step error {worst_prev:.6f} m",
        )
        checks.check(
            "every reported contact point lands within tolerance of a live capsule",
            worst_contact_now <= CONTACT_TOLERANCE_M,
            f"worst surface distance {worst_contact_now:+.4f} m vs tolerance "
            f"{CONTACT_TOLERANCE_M} m",
        )
        # The report must be written BEFORE app.close(): Kit's shutdown closes
        # the original stdout and Checks.report swallows the resulting write
        # error, silently dropping every verdict.
        exit_code = checks.report(banner="articulation stage probe")
    finally:
        app.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
