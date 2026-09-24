"""Author a :class:`~sim2sense_fall.humans.rig.HumanRigPlan` as an Isaac Sim articulation.

This is the only module in the human pipeline that needs the USD runtime, and the
USD import is deferred into functions so that ``import
sim2sense_fall.humans.usd_human`` never fails on a machine without Isaac Sim.

What gets authored
------------------

* one root ``Xform`` carrying ``UsdPhysics.ArticulationRootAPI``,
  ``UsdPhysics.RigidBodyAPI`` and ``UsdPhysics.MassAPI`` -- the pelvis, which is a
  free body so the figure can actually topple;
* one child ``Xform`` per link with its own mass, plus a capsule collider carrying
  ``UsdPhysics.CollisionAPI`` and a ``PhysxCollisionAPI`` contact/rest offset pair;
* one ``UsdPhysics.RevoluteJoint`` per degree of freedom, with the rotation axis,
  the limit pair in degrees and a PD ``UsdPhysics.DriveAPI``;
* one ``UsdPhysics.FixedJoint`` per rigid attachment;
* ``PhysxSchema.PhysxArticulationAPI`` so self-collisions and the solver iteration
  counts are explicit rather than left to the default;
* an optional ``UsdPhysics.FixedJoint`` from the world to the pelvis when
  ``root_mode`` is ``anchored``. The anchor is a planning decision, and the export
  provenance records that it was used.

Units at this boundary
----------------------

USD revolute limits and angular drive targets are **degrees**; everything the
pipeline stores is **radians**. The plan carries limits in degrees precisely so the
authoring step needs no conversion, and the runtime wrapper converts back through
Isaac's ``Articulation`` API, which reports radians. That asymmetry is the single
most likely place for a silent units bug, so ``scripts/humans/verify.py`` checks the
limits read back from Physics against the plan's limit table.

Runtime behaviour that was probed, not assumed
----------------------------------------------

* ``Articulation.get_dof_positions`` / ``get_dof_position_targets`` report
  **radians**; ``get_dof_limits`` also reports radians even though USD stores
  degrees.
* ``set_dof_positions`` writes joint angles directly and reads back exactly, which
  is what makes the kinematic replay path a real check of the retargeting.
* ``RigidPrim`` only creates a contact view when ``contact_filter_paths`` and
  ``max_contact_count`` are passed to the constructor; without them,
  ``get_net_contact_forces`` raises. Contact reporting is therefore best-effort and
  its absence is recorded rather than faked.
* On Isaac Sim 6.0.1 that view does not actually work here: constructing it with any
  filter set makes ``omni.physics.tensors`` log ``Pattern '/World/Human' did not match
  any rigid contact for filters`` for every filter path, and the wrapper it hands back
  has no backend, so ``contact_forces`` stays false however it is constructed.
  Requesting it is not merely useless, it is not a quiet failure: the same ``check()``
  runs again from ``RigidPrim._on_physics_ready``, which ``SimulationManager``
  dispatches as an event callback over a weakref proxy, so the ``AttributeError`` it
  raises there (``'NoneType' object has no attribute 'check'``) lands in the app's
  stderr instead of in any ``try`` of ours -- every run that asks for the view paints a
  screenful of plugin errors and a traceback, including at shutdown when the timeline
  stops and the event fires again. The entry points therefore pass
  ``enable_contact_views=False``; the parameter remains as an opt-in for a build where
  the view does work. The working channel is
  PhysX's own polled report --
  ``omni.physx.get_physx_simulation_interface().get_contact_report()`` -- which does
  return per-contact position, normal, impulse and separation. It requires
  ``PhysxSchema.PhysxContactReportAPI`` on **both** sides of a contact pair, which is
  why :func:`author_human` tags the human's colliders and
  :func:`author_contact_reporting` tags the environment's. ``HumanRuntime`` prefers
  that channel and says so in ``capabilities["contact_source"]``.
* A body with no joint to the world free-falls, so a free-root trial really does
  move: the "lifted body falls back" positive control from the scene stage is
  reused here, with the same reason.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .config import PerturbationConfig
from .rig import HumanRigPlan, LinkTransform, pose_surface_points
from .rotations import quaternion_to_matrix

__all__ = [
    "HUMAN_ROOT_PATH",
    "DISPLAY_SKIN_PATH",
    "ContactSourceUnavailable",
    "ContactSample",
    "HumanRuntime",
    "HumanRuntimeUnavailable",
    "TrialRecording",
    "author_contact_reporting",
    "author_human",
    "build_human_stage",
    "environment_contact_paths",
    "human_stage_summary",
    "pxr_modules",
]

LOGGER = logging.getLogger(__name__)

HUMAN_ROOT_PATH = "/World/Human"
#: World-space playback mesh; see :func:`write_display_skin`. It is deliberately not a
#: child of the articulation, so updating it cannot be mistaken for changing the body.
DISPLAY_SKIN_PATH = "/World/DisplaySkin"
JOINTS_SUFFIX = "Joints"

#: Contact reporting is filtered to these environment categories. A fall impact is
#: measured against the supporting surface first, but a body that topples into a wall
#: or a sofa also deserves a contact event, so the walls, furniture and steps are
#: included. Restricting to the floor silently dropped those impacts.
DEFAULT_CONTACT_CATEGORIES: tuple[str, ...] = (
    "floor",
    "ground",
    "wall",
    "furniture",
    "step",
    "obstacle",
)
#: Contacts to track per link before PhysX starts dropping them.
DEFAULT_MAX_CONTACT_COUNT = 24
#: Categories that count as a surface the body can REST on. A wall or a wardrobe is a
#: contact but not support: the body has hit something without its weight being carried
#: by the ground, which is exactly the distinction a fall label turns on.
SUPPORT_CATEGORIES: tuple[str, ...] = ("floor", "ground", "step")
#: Slack when testing whether a reported contact point lies inside a body capsule.
#: PhysX reports the point on the contact-offset shell, which sits outside the
#: collision surface by the offset distance, so a zero-tolerance test misses resting
#: contacts entirely.
CAPSULE_CONTAINMENT_TOLERANCE_M = 0.02
#: A contact normal must point at least this far along +Z to count as resting on a
#: surface rather than glancing off it or pressing sideways into a wall.
_SUPPORT_NORMAL_MIN_Z = 0.5
#: How far above the measured surface a contact may sit and still count as resting on
#: it. Covers the solver's allowed penetration depth in both directions.
_SUPPORT_HEIGHT_TOLERANCE_M = 0.03


class HumanRuntimeUnavailable(RuntimeError):
    """Raised when the USD/Isaac Sim runtime is not importable."""


class ContactSourceUnavailable(RuntimeError):
    """Raised when no contact channel could be established.

    A caller that needs contact evidence must let this surface instead of reading an
    empty result as "nothing was touched".
    """


@dataclass(frozen=True, slots=True)
class ContactSample:
    """One contact point, as reported by PhysX's polled contact report.

    Numeric collider identifiers are decoded with PhysicsSchemaTools.intToSdfPath.
    Both encoded identifiers and decoded paths are retained for traceability.
    """

    collider0: int
    collider1: int
    position_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    impulse_ns: tuple[float, float, float]
    separation_m: float
    collider0_path: str = ""
    collider1_path: str = ""

    @property
    def impulse_magnitude_ns(self) -> float:
        return float(np.linalg.norm(np.asarray(self.impulse_ns, dtype=np.float64)))

    def as_dict(self) -> dict[str, Any]:
        """Plain-data form, with the handles kept under an explicit ``handle`` key.

        ``collider0``/``collider1`` are mirrored for backwards compatibility with
        readers that only ever wanted "which pair", but the ``*_handle`` names are the
        truthful ones and are what new code should use.
        """

        return {
            "collider0": self.collider0,
            "collider1": self.collider1,
            "collider0_handle": self.collider0,
            "collider1_handle": self.collider1,
            "collider0_path": self.collider0_path,
            "collider1_path": self.collider1_path,
            "position_m": list(self.position_m),
            "normal": list(self.normal),
            "impulse_ns": list(self.impulse_ns),
            "impulse_magnitude_ns": round(self.impulse_magnitude_ns, 9),
            "separation_m": self.separation_m,
        }


def pxr_modules() -> SimpleNamespace:
    """Import the USD modules or raise a clear, actionable error."""

    try:
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - depends on the host runtime
        raise HumanRuntimeUnavailable(
            "USD modules are unavailable; run this script through Isaac Sim's bundled "
            "interpreter, for example: ~/isaacsim/python.sh scripts/humans/build.py"
        ) from exc
    return SimpleNamespace(
        Gf=Gf,
        PhysxSchema=PhysxSchema,
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
    )


def _ensure_xform(runtime: SimpleNamespace, stage: Any, path: str) -> Any:
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        return existing
    current = ""
    for part in path.strip("/").split("/"):
        current = f"{current}/{part}"
        if stage.GetPrimAtPath(current).IsValid():
            continue
        runtime.UsdGeom.Xform.Define(stage, current)
    return stage.GetPrimAtPath(path)


def _capsule_surface_distance(
    point: tuple[float, float, float],
    centre: tuple[float, float, float],
    radius: float,
    axis_half: tuple[float, float, float],
) -> float:
    """Signed distance from ``point`` to the capsule surface (negative = inside).

    ``axis_half`` is the half-height vector along the capsule's own Z axis, so it
    carries both the length and the orientation. The diagnostic half of
    :func:`_point_in_capsule`: when attribution fails, the raise must say how far
    outside every capsule the point was and for which limb, because "3 mm outside
    the contact shell" and "a metre away in another room" are different defects
    with different fixes.
    """

    ax, ay, az = axis_half
    axis_length_sq = ax * ax + ay * ay + az * az
    px = point[0] - centre[0]
    py = point[1] - centre[1]
    pz = point[2] - centre[2]
    if axis_length_sq <= 0.0:
        offset = (px, py, pz)
    else:
        along = (px * ax + py * ay + pz * az) / axis_length_sq
        along = min(1.0, max(-1.0, along))
        offset = (px - along * ax, py - along * ay, pz - along * az)
    distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2)
    return distance - radius


def _point_in_capsule(
    point: tuple[float, float, float],
    centre: tuple[float, float, float],
    radius: float,
    axis_half: tuple[float, float, float],
) -> bool:
    """True when ``point`` lies inside the capsule described by the other arguments.

    A small tolerance is added because PhysX reports the contact point on the
    *contact offset* shell, which sits a few millimetres outside the collision
    surface; without it, resting contacts would fall just outside their own
    capsule.
    """

    return (
        _capsule_surface_distance(point, centre, radius, axis_half)
        <= CAPSULE_CONTAINMENT_TOLERANCE_M
    )


def _prim_local_box(prim: Any, runtime: Any) -> Any:
    """Geometry-faithful local box of a unit primitive, or ``None`` if unknown.

    Every collider in this project's scenes is a ``Cube`` or ``Cylinder`` whose
    real extent comes from ``size``/``radius``/``height`` plus the xform ops, so
    the box can be derived from the authored attributes directly.

    ``UsdGeom.BBoxCache.ComputeLocalBound`` is deliberately not used. Measured on
    Isaac Sim 6.0.1 it returns a box that is already scaled but *still offset by
    the prim's own translate*, which makes composing it with the local-to-world
    transform count the translation twice. For the room floor
    (``translate z = -0.06``, ``scale z = 0.12``) it reports ``z in [-0.12, 0.0]``
    instead of the authored ``[-0.06, +0.06]``, so the composed world top came out
    at -0.06 m while the body's own resting contacts -- measured independently --
    sit at exactly 0.00 m. The same call also returns an inverted-infinite sentinel
    for the ``render``/``proxy``/``guide`` purposes, so those filters are not a
    fallback either.
    """

    gf = runtime.Gf
    size = 1.0
    attribute = prim.GetAttribute("size")
    if attribute and attribute.IsValid() and attribute.Get() is not None:
        size = float(attribute.Get())
    half = abs(size) / 2.0
    type_name = str(prim.GetTypeName())
    if type_name == "Cube":
        return gf.Range3d(gf.Vec3d(-half, -half, -half), gf.Vec3d(half, half, half))
    if type_name == "Cylinder":
        radius = half
        attribute = prim.GetAttribute("radius")
        if attribute and attribute.IsValid() and attribute.Get() is not None:
            radius = float(attribute.Get())
        height_half = half
        attribute = prim.GetAttribute("height")
        if attribute and attribute.IsValid() and attribute.Get() is not None:
            height_half = float(attribute.Get()) / 2.0
        radius = abs(radius)
        height_half = abs(height_half)
        return gf.Range3d(
            gf.Vec3d(-radius, -radius, -height_half),
            gf.Vec3d(radius, radius, height_half),
        )
    return None


def _world_box_max_z(local_box: Any, local_to_world: Any, runtime: Any) -> float:
    """Highest world-space ``z`` of an axis-aligned local box under a transform.

    The eight corners are mapped individually and re-reduced. Mapping only the
    local ``max`` corner would be wrong the moment a parent rotation is not
    axis-aligned, and ``Gf.Transform`` is about the only composition verb this
    build offers -- ``BBox3d``/``Range3d`` have no ``TransformBy``.
    """

    lo = local_box.GetMin()
    hi = local_box.GetMax()
    vec3d = runtime.Gf.Vec3d
    return max(
        float(
            local_to_world.Transform(
                vec3d(
                    hi[0] if index & 1 else lo[0],
                    hi[1] if index & 2 else lo[1],
                    hi[2] if index & 4 else lo[2],
                )
            )[2]
        )
        for index in range(8)
    )


def _segment_in_volumes(
    point: tuple[float, float, float],
    volumes: Mapping[
        str, tuple[tuple[float, float, float], float, tuple[float, float, float]]
    ],
) -> str | None:
    """Name of the body segment whose capsule contains ``point``, or ``None``.

    Split out from the runtime so the containment rule can be exercised on plain
    tuples without a stage. ``volumes`` maps segment name to
    ``(centre, radius, axis_half)`` in the same world frame as ``point``.
    """

    for name, (centre, radius, axis_half) in volumes.items():
        if _point_in_capsule(point, centre, radius, axis_half):
            return name
    return None


def _capsule_geometry_in_link_frame(
    runtime: SimpleNamespace,
    capsule_prim: Any,
    link_prim: Any,
) -> tuple[tuple[float, float, float], float, tuple[float, float, float]]:
    """Capsule ``(centre, radius, half-axis)`` expressed in its LINK's frame.

    The capsule prim is authored once as a child of its link with translate and
    orient ops only, so its offset from the link origin is a constant of the rig.
    Reading it as the *relative* transform between the two prims (both sampled
    from the stage at the same instant) keeps it valid no matter when it is
    sampled: the ratio of two equally stale poses is still the authored offset.
    Composing this constant with the link's live physics pose at attribution
    time is what keeps contact volumes on the body -- a USD-stage world read
    freezes the moment stepping bypasses the fabric/USD sync, which is the
    measured cause of the frame-2298 attribution escape in the AMASS trial
    (``scripts/humans/probe_stage_staleness.py``).

    Raises ``ValueError`` when the relative transform is not rigid: a scaled or
    sheared offset would silently deform the capsule and misplace every contact.
    """

    if not capsule_prim or not capsule_prim.IsValid():
        raise ValueError("capsule prim is invalid")
    if not link_prim or not link_prim.IsValid():
        raise ValueError("link prim is invalid")
    capsule = runtime.UsdGeom.Capsule(capsule_prim)
    if not capsule:
        raise ValueError(f"prim {capsule_prim.GetPath()} is not a UsdGeom.Capsule")
    radius = float(capsule.GetRadiusAttr().Get() or 0.0)
    height = float(capsule.GetHeightAttr().Get() or 0.0)
    if radius <= 0.0:
        raise ValueError(f"capsule {capsule_prim.GetPath()} has non-positive radius {radius}")
    time = runtime.Usd.TimeCode.Default()
    link_matrix = runtime.UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(time)
    cap_matrix = runtime.UsdGeom.Xformable(capsule_prim).ComputeLocalToWorldTransform(time)
    to_link = link_matrix.GetInverse()
    world_centre = cap_matrix.Transform(runtime.Gf.Vec3d(0.0, 0.0, 0.0))
    world_axis = cap_matrix.TransformDir(runtime.Gf.Vec3d(0.0, 0.0, 1.0))
    local_centre = to_link.Transform(world_centre)
    local_axis = to_link.TransformDir(world_axis)
    axis_norm = local_axis.GetLength()
    if abs(axis_norm - 1.0) > 1e-6:
        raise ValueError(
            f"capsule-to-link transform for {capsule_prim.GetPath()} is not rigid "
            f"(axis scale {axis_norm:.9f}); a deformed offset would misplace every "
            "attributed contact"
        )
    half = max(0.0, height * 0.5)
    return (
        (float(local_centre[0]), float(local_centre[1]), float(local_centre[2])),
        radius,
        (
            float(local_axis[0]) * half,
            float(local_axis[1]) * half,
            float(local_axis[2]) * half,
        ),
    )


def _capsule_world_volume(
    link_rotation: np.ndarray,
    link_translation: np.ndarray,
    geometry: tuple[tuple[float, float, float], float, tuple[float, float, float]],
) -> tuple[tuple[float, float, float], float, tuple[float, float, float]]:
    """Map an authored link-frame capsule into world space through the link pose.

    ``link_rotation`` (3x3, world-from-link) and ``link_translation`` come from
    the physics read channel, which stays current even when stepping bypasses the
    fabric/USD sync; ``geometry`` is the constant authored offset from
    :func:`_capsule_geometry_in_link_frame`. The radius passes through unchanged:
    rigid composition cannot scale it. Split out from the runtime so the
    composition can be exercised on plain numbers without a stage.
    """

    centre = link_rotation @ np.asarray(geometry[0], dtype=np.float64) + np.asarray(
        link_translation, dtype=np.float64
    )
    axis_half = link_rotation @ np.asarray(geometry[2], dtype=np.float64)
    return (
        (float(centre[0]), float(centre[1]), float(centre[2])),
        float(geometry[1]),
        (float(axis_half[0]), float(axis_half[1]), float(axis_half[2])),
    )


def _author_metadata(runtime: SimpleNamespace, prim: Any, entries: Mapping[str, Any]) -> None:
    types = runtime.Sdf.ValueTypeNames
    for name, value in entries.items():
        if isinstance(value, bool):
            prim.CreateAttribute(name, types.Bool, custom=True).Set(value)
        elif isinstance(value, int):
            prim.CreateAttribute(name, types.Int, custom=True).Set(value)
        elif isinstance(value, float):
            prim.CreateAttribute(name, types.Double, custom=True).Set(value)
        else:
            prim.CreateAttribute(name, types.String, custom=True).Set(str(value))


def _author_capsule(
    runtime: SimpleNamespace, stage: Any, plan: HumanRigPlan, link_name: str
) -> str | None:
    link = plan.link(link_name)
    capsule = link.capsule
    if capsule is None:
        return None
    geometry = runtime.UsdGeom.Capsule.Define(stage, capsule.path)
    geometry.CreateAxisAttr(runtime.UsdGeom.Tokens.z)
    geometry.CreateHeightAttr(float(capsule.cylinder_length_m))
    geometry.CreateRadiusAttr(float(capsule.radius_m))
    xformable = runtime.UsdGeom.Xformable(geometry.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(runtime.Gf.Vec3d(*[float(v) for v in capsule.center]))
    xformable.AddOrientOp().Set(
        runtime.Gf.Quatf(
            float(capsule.orientation_wxyz[0]),
            float(capsule.orientation_wxyz[1]),
            float(capsule.orientation_wxyz[2]),
            float(capsule.orientation_wxyz[3]),
        )
    )
    geometry.CreateDisplayColorAttr().Set([runtime.Gf.Vec3f(0.85, 0.55, 0.45)])
    prim = geometry.GetPrim()
    # Capsules are the physical collision proxy.  Rendering them together with
    # the SMPL surface makes limbs look detached or duplicated, so keep the
    # geometry in the stage for PhysX while hiding it from the viewport.
    runtime.UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
        runtime.UsdGeom.Tokens.invisible
    )
    runtime.UsdPhysics.CollisionAPI.Apply(prim)
    collision = runtime.PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr().Set(float(plan.contact_offset_m))
    collision.CreateRestOffsetAttr().Set(float(plan.rest_offset_m))
    # PhysX reports a contact pair only when BOTH colliders carry this API, so the
    # human side is tagged here and the environment side by author_contact_reporting.
    # A zero threshold keeps resting contacts visible: a body already lying on the
    # floor is still touching it.
    contact_report = runtime.PhysxSchema.PhysxContactReportAPI.Apply(prim)
    contact_report.CreateThresholdAttr().Set(0.0)
    _author_metadata(
        runtime,
        prim,
        {
            "sim2sense:category": "body_segment",
            "sim2sense:semantic": f"human:{link.chain_joint}:capsule",
            "sim2sense:massKg": float(link.mass_kg),
            "sim2sense:physicsMode": "dynamic",
            "sim2sense:movable": True,
            "sim2sense:renderable": False,
            "sim2sense:collisionProxy": True,
        },
    )
    return capsule.path


def _author_joint(
    runtime: SimpleNamespace,
    stage: Any,
    path: str,
    joint: Any,
    *,
    root_path: str,
    limits_deg: tuple[float, float] | None,
    drive: tuple[float, float, float, str] | None,
) -> None:
    """Author either a fixed or a revolute joint from a plan entry."""

    if limits_deg is None:
        definition = runtime.UsdPhysics.FixedJoint.Define(stage, path)
    else:
        definition = runtime.UsdPhysics.RevoluteJoint.Define(stage, path)
        definition.CreateAxisAttr(joint.axis.upper())
        definition.CreateLowerLimitAttr(float(limits_deg[0]))
        definition.CreateUpperLimitAttr(float(limits_deg[1]))
    definition.CreateBody0Rel().SetTargets([f"{root_path}/{joint.parent_link}"])
    definition.CreateBody1Rel().SetTargets([f"{root_path}/{joint.child_link}"])
    definition.CreateLocalPos0Attr().Set(runtime.Gf.Vec3f(*[float(v) for v in joint.local_pos0]))
    definition.CreateLocalPos1Attr().Set(runtime.Gf.Vec3f(*[float(v) for v in joint.local_pos1]))
    definition.CreateLocalRot0Attr().Set(runtime.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    definition.CreateLocalRot1Attr().Set(runtime.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if drive is not None and limits_deg is not None:
        stiffness, damping, max_force, drive_type = drive
        api = runtime.UsdPhysics.DriveAPI.Apply(definition.GetPrim(), "angular")
        api.CreateTypeAttr().Set(drive_type)
        api.CreateStiffnessAttr().Set(float(stiffness))
        api.CreateDampingAttr().Set(float(damping))
        api.CreateMaxForceAttr().Set(float(max_force))
        api.CreateTargetPositionAttr().Set(0.0)


def _apply_damping(
    runtime: SimpleNamespace,
    prim: Any,
    *,
    linear_damping: float,
    angular_damping: float,
    report: dict[str, Any],
) -> None:
    """Author PhysX rigid-body damping.

    Damping is what keeps a passive ragdoll from jittering on contact, so it is
    wanted rather than cosmetic. The schema lives in ``PhysxSchema`` (the attribute
    is ``physxRigidBody:linearDamping``), not in ``UsdPhysics.RigidBodyAPI``, which
    is why it is applied separately. If the schema is unavailable the build continues
    and the report records that damping was not authored, rather than pretending it
    was.
    """

    if not linear_damping and not angular_damping:
        return
    try:
        api = runtime.PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateLinearDampingAttr().Set(float(linear_damping))
        api.CreateAngularDampingAttr().Set(float(angular_damping))
    except AttributeError as exc:
        report["damping_authored"] = False
        report.setdefault("damping_error", f"{type(exc).__name__}: {exc}")
        LOGGER.warning("could not author rigid body damping: %s", exc)
    else:
        report["damping_authored"] = True


def author_human(
    stage: Any,
    plan: HumanRigPlan,
    *,
    root_path: str = HUMAN_ROOT_PATH,
    skin_points: np.ndarray | None = None,
    skin_faces: np.ndarray | None = None,
) -> dict[str, Any]:
    """Author the articulation into an existing stage under ``root_path``.

    ``skin_points`` / ``skin_faces``, when given, are the licensed SMPL surface measured
    RELATIVE TO THE PELVIS LINK, and are written as ``<root>/Skin``. They are authored only
    from a model that actually loaded: a stage whose skin was skipped must keep saying
    ``capsule_proxy_surface``, because the geometry it carries is the proxy.

    Returns a small report so callers can log what was written.
    """

    runtime = pxr_modules()
    if not root_path.startswith("/"):
        raise ValueError(f"root_path must be an absolute prim path, got {root_path!r}")
    _ensure_xform(runtime, stage, "/World")
    report: dict[str, Any] = {"damping_authored": False}

    # --- root link: the pelvis, a free body carrying the articulation ---------
    root = runtime.UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    runtime.UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    runtime.UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass_api = runtime.UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(float(plan.root.mass_kg))
    _apply_damping(
        runtime,
        root_prim,
        linear_damping=plan.linear_damping,
        angular_damping=plan.angular_damping,
        report=report,
    )
    articulation_api = runtime.PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    articulation_api.CreateEnabledSelfCollisionsAttr().Set(bool(plan.self_collisions))
    articulation_api.CreateSolverPositionIterationCountAttr().Set(16)
    articulation_api.CreateSolverVelocityIterationCountAttr().Set(4)
    _author_metadata(
        runtime,
        root_prim,
        {
            "sim2sense:category": "human_root",
            "sim2sense:semantic": f"human:{plan.root_link}",
            "sim2sense:massKg": float(plan.root.mass_kg),
            "sim2sense:physicsMode": "dynamic",
            "sim2sense:movable": True,
            "sim2sense:rootMode": plan.root_mode,
        },
    )

    # --- child links ---------------------------------------------------------
    collider_paths: list[str] = []
    for link in plan.links:
        if link.name == plan.root_link:
            path = root_path
        else:
            path = f"{root_path}/{link.name}"
            xform = runtime.UsdGeom.Xform.Define(stage, path)
            xformable = runtime.UsdGeom.Xformable(xform.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(
                runtime.Gf.Vec3d(*[float(v) for v in link.local_translation])
            )
            prim = xform.GetPrim()
            runtime.UsdPhysics.RigidBodyAPI.Apply(prim)
            runtime.UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(link.mass_kg))
            _apply_damping(
                runtime,
                prim,
                linear_damping=plan.linear_damping,
                angular_damping=plan.angular_damping,
                report=report,
            )
            _author_metadata(
                runtime,
                prim,
                {
                    "sim2sense:category": "body_link",
                    "sim2sense:semantic": f"human:{link.chain_joint}",
                    "sim2sense:massKg": float(link.mass_kg),
                    "sim2sense:physicsMode": "dynamic",
                    "sim2sense:movable": True,
                    "sim2sense:linkRole": link.role,
                },
            )
        capsule_path = _author_capsule(runtime, stage, plan, link.name)
        if capsule_path:
            collider_paths.append(capsule_path)

    # --- joints --------------------------------------------------------------
    joints_root = f"{root_path}/{JOINTS_SUFFIX}"
    _ensure_xform(runtime, stage, joints_root)
    for joint in plan.joints:
        spec = SimpleNamespace(
            axis=joint.axis,
            parent_link=joint.parent_link,
            child_link=joint.child_link,
            local_pos0=joint.local_pos0,
            local_pos1=joint.local_pos1,
        )
        _author_joint(
            runtime,
            stage,
            f"{joints_root}/{joint.name}",
            spec,
            root_path=root_path,
            limits_deg=(joint.lower_deg, joint.upper_deg),
            drive=(joint.stiffness, joint.damping, joint.max_force, joint.drive_type),
        )
    for fixed in plan.fixed_joints:
        spec = SimpleNamespace(
            axis="z",
            parent_link=fixed.parent_link,
            child_link=fixed.child_link,
            local_pos0=fixed.local_pos0,
            local_pos1=(0.0, 0.0, 0.0),
        )
        _author_joint(
            runtime,
            stage,
            f"{joints_root}/{fixed.name}",
            spec,
            root_path=root_path,
            limits_deg=None,
            drive=None,
        )

    # --- optional world anchor ----------------------------------------------
    anchored = plan.root_mode == "anchored"
    if anchored:
        anchor = runtime.UsdPhysics.FixedJoint.Define(stage, f"{root_path}/WorldAnchor")
        anchor.CreateBody0Rel().SetTargets([])
        anchor.CreateBody1Rel().SetTargets([root_path])
        anchor.CreateLocalPos0Attr().Set(runtime.Gf.Vec3f(0.0, 0.0, 0.0))
        anchor.CreateLocalPos1Attr().Set(runtime.Gf.Vec3f(0.0, 0.0, 0.0))

    skin_vertex_count = 0
    if skin_points is not None and skin_faces is not None:
        points = np.asarray(skin_points, dtype=np.float64)
        faces = np.asarray(skin_faces, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"skin_points must be (V, 3), got {points.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"skin_faces must be (F, 3), got {faces.shape}")
        if faces.size and int(faces.max()) >= points.shape[0]:
            raise ValueError(
                f"skin face indices run to {int(faces.max())} but only {points.shape[0]}"
                " vertices were given"
            )
        skin = runtime.UsdGeom.Mesh.Define(stage, f"{root_path}/Skin")
        skin.GetPointsAttr().Set([runtime.Gf.Vec3f(*row) for row in points])
        skin.GetFaceVertexIndicesAttr().Set([int(index) for row in faces for index in row])
        skin.GetFaceVertexCountsAttr().Set([3] * int(faces.shape[0]))
        skin.GetSubdivisionSchemeAttr().Set("none")
        skin.GetPrim().GetReferences()  # touch, so an empty reference list is not implied
        _author_metadata(
            runtime,
            skin.GetPrim(),
            {
                "sim2sense:category": "human_skin",
                "sim2sense:semantic": "smpl_skin_mesh",
                "sim2sense:collision": "none (visual; collision is the capsule set)",
                "sim2sense:follows": "CPU linear blend skinning from the physics link poses",
            },
        )
        skin_vertex_count = int(points.shape[0])

    LOGGER.info(
        "authored %s: %d links, %d joints, %d fixed joints, %d colliders, %d skin verts, anchor=%s",
        plan.human_id,
        len(plan.links),
        len(plan.joints),
        len(plan.fixed_joints),
        len(collider_paths),
        skin_vertex_count,
        anchored,
    )
    return {
        "root_path": root_path,
        "link_count": len(plan.links),
        "joint_count": len(plan.joints),
        "fixed_joint_count": len(plan.fixed_joints),
        "collider_paths": collider_paths,
        "anchored": anchored,
        "damping_authored": bool(report["damping_authored"]),
        "skin_vertex_count": skin_vertex_count,
        "skin_face_count": 0 if skin_faces is None else int(np.asarray(skin_faces).shape[0]),
    }


def build_human_stage(
    plan: HumanRigPlan,
    usd_path: str | Path,
    *,
    base_scene: str | Path | None = None,
    spawn_position: Sequence[float] | None = None,
    root_path: str = HUMAN_ROOT_PATH,
    skin_points: np.ndarray | None = None,
    skin_faces: np.ndarray | None = None,
) -> Path:
    """Author the human, optionally into a copy of an existing scene, and save.

    ``base_scene`` is a scene USD (the fixed indoor apartment). It is *copied*
    first, so exporting a human never rewrites the verified scene artifact.
    """

    runtime = pxr_modules()
    target = Path(usd_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if base_scene is not None:
        source = Path(base_scene)
        if not source.is_file():
            raise FileNotFoundError(f"base scene not found: {source}")
        shutil.copy2(source, target)
        stage = runtime.Usd.Stage.Open(str(target))
        if stage is None:
            raise RuntimeError(f"could not open base scene {source}")
    else:
        stage = runtime.Usd.Stage.CreateNew(str(target))
        runtime.UsdGeom.SetStageUpAxis(stage, runtime.UsdGeom.Tokens.z)
        runtime.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = runtime.UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        physics_scene = runtime.UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(runtime.Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    report = author_human(
        stage,
        plan,
        root_path=root_path,
        skin_points=skin_points,
        skin_faces=skin_faces,
    )
    position = tuple(float(v) for v in (spawn_position or plan.spawn_root_position))
    root_prim = stage.GetPrimAtPath(root_path)
    xformable = runtime.UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(runtime.Gf.Vec3d(*position))
    xformable.AddOrientOp().Set(runtime.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    if not target.is_file() or target.stat().st_size <= len("#usda 1.0\n") + 1:
        raise RuntimeError(f"USD export produced an empty layer at {target}")
    LOGGER.info("wrote human stage %s at %s", target, position)
    report["spawn_position"] = position
    return target


def human_stage_summary(usd_path: str | Path) -> dict[str, Any]:
    """Re-open a saved stage and report what is actually on it."""

    runtime = pxr_modules()
    stage = runtime.Usd.Stage.Open(str(usd_path))
    skins = 0
    skin_vertices = 0
    skin_faces_count = 0
    if stage is None:
        raise FileNotFoundError(f"could not open stage: {usd_path}")
    joints = 0
    fixed = 0
    colliders = 0
    bodies = 0
    capsules = 0
    invisible_capsules = 0
    articulation_roots = 0
    for prim in stage.Traverse():
        if prim.IsA(runtime.UsdPhysics.RevoluteJoint):
            joints += 1
        elif prim.IsA(runtime.UsdPhysics.FixedJoint):
            fixed += 1
        if prim.HasAPI(runtime.UsdPhysics.CollisionAPI):
            colliders += 1
        if prim.HasAPI(runtime.UsdPhysics.RigidBodyAPI):
            bodies += 1
        if prim.HasAPI(runtime.UsdPhysics.ArticulationRootAPI):
            articulation_roots += 1
        if prim.IsA(runtime.UsdGeom.Capsule):
            capsules += 1
            if (
                runtime.UsdGeom.Imageable(prim).ComputeVisibility()
                == runtime.UsdGeom.Tokens.invisible
            ):
                invisible_capsules += 1
        if prim.IsA(runtime.UsdGeom.Mesh):
            skins += 1
            points = prim.GetAttribute("points").Get() or []
            skin_vertices = max(skin_vertices, len(points))
            counts = prim.GetAttribute("faceVertexCounts").Get() or []
            skin_faces_count += len(counts)
    default_prim = stage.GetDefaultPrim()
    return {
        "path": str(usd_path),
        "up_axis": str(runtime.UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(runtime.UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(default_prim.GetPath()) if default_prim else "",
        "articulation_root_count": articulation_roots,
        "revolute_joint_count": joints,
        "fixed_joint_count": fixed,
        "collider_count": colliders,
        "rigid_body_count": bodies,
        "capsule_count": capsules,
        "invisible_capsule_count": invisible_capsules,
        "skin_mesh_count": skins,
        "skin_vertex_count": skin_vertices,
        "skin_face_count": skin_faces_count,
    }


def environment_contact_paths(
    stage: Any,
    *,
    human_root: str = HUMAN_ROOT_PATH,
    categories: Sequence[str] = DEFAULT_CONTACT_CATEGORIES,
) -> tuple[str, ...]:
    """Collider paths the human is allowed to report contacts against.

    Every environment collider carrying a matching ``sim2sense:category`` is
    included. The floor is the surface a fall is measured against, but a body that
    topples into a wall or a sofa has hit something, and dropping those pairs would
    make the contact history quietly incomplete.
    """

    runtime = pxr_modules()
    wanted = {str(category) for category in categories}
    paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(runtime.UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if path == human_root or path.startswith(f"{human_root}/"):
            continue
        attribute = prim.GetAttribute("sim2sense:category")
        category = attribute.Get() if attribute and attribute.IsValid() else None
        if category is None:
            continue
        if str(category) in wanted:
            paths.append(path)
    return tuple(sorted(paths))


def author_contact_reporting(
    stage: Any,
    *,
    human_root: str = HUMAN_ROOT_PATH,
    categories: Sequence[str] = DEFAULT_CONTACT_CATEGORIES,
    threshold_n: float = 0.0,
) -> tuple[str, ...]:
    """Tag both sides of every reportable contact pair with PhysX contact reporting.

    ``PhysxSchema.PhysxContactReportAPI`` only emits a pair when **both** colliders
    carry it, so tagging the human alone yields an empty report. The human's own
    colliders are tagged by :func:`author_human`; this tags the environment side and
    returns the paths it touched.

    ``threshold_n`` is the force above which a pair is reported. Zero reports
    resting contacts as well as impacts, which is what a fall detector needs: a body
    lying still on the floor is still touching it.
    """

    runtime = pxr_modules()
    paths = environment_contact_paths(
        stage, human_root=human_root, categories=categories
    )
    tagged: list[str] = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        PhysxSchema = runtime.PhysxSchema
        PhysxSchema.PhysxContactReportAPI.Apply(prim)
        PhysxSchema.PhysxContactReportAPI(prim).CreateThresholdAttr().Set(float(threshold_n))
        tagged.append(path)
    return tuple(tagged)


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialRecording:
    """Raw per-physics-step state of one trial, before labelling or export."""

    time_s: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_positions_rad: np.ndarray
    joint_velocities_rad_s: np.ndarray
    joint_names: tuple[str, ...]
    link_positions: np.ndarray
    link_names: tuple[str, ...]
    body_points: np.ndarray
    body_point_owners: tuple[str, ...]
    reference_joint_positions_rad: np.ndarray
    contact_force_n: np.ndarray | None
    contact_categories: tuple[str, ...]
    notes: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return int(np.shape(self.time_s)[0])

    @property
    def tracking_error_deg(self) -> float:
        """Largest joint tracking error across the trial, in degrees."""

        return float(
            np.degrees(np.abs(self.joint_positions_rad - self.reference_joint_positions_rad).max())
        )

    @property
    def max_penetration_m(self) -> float:
        """Deepest excursion of any body point below the floor."""

        return float(min(0.0, self.body_points[:, :, 2].min()))


def tracking_target_rad(low: float, high: float, min_amplitude: float = 0.0) -> float:
    """A probe target far enough from rest to falsify a tracking claim.

    ``low`` and ``high`` must already be radians, as :meth:`HumanRuntime.dof_limits_rad`
    reports them. The stage-7 review found a caller converting them a second time, which
    turned a 75 degree knee target into 1.3 degrees -- inside the 15 degree tolerance, so
    the tracking check passed while commanding almost nothing.

    Half of the wider side is used rather than the midpoint of the pair, because
    asymmetric limits can put the midpoint almost on zero: the ankle's ``[-35, 45]``
    midpoints to 5 degrees, which no tolerance can meaningfully discriminate against.
    When that is still inside ``min_amplitude`` -- a three-axis chain has small
    secondary-axis spans a single-axis rig never had -- the target is pushed toward
    the same limit until the amplitude clears the bar, clamped to stay off the bound
    so the drive has room to settle.
    """

    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"limits must be finite and ordered, got [{low!r}, {high!r}]")
    target = 0.5 * (high if abs(high) >= abs(low) else low)
    if abs(target) < min_amplitude:
        sign = 1.0 if target >= 0.0 else -1.0
        target = sign * float(min_amplitude)
    margin = 0.05 * (high - low) + 1e-9
    return float(min(max(target, low + margin), high - margin))


def dof_permutation(plan_names: Sequence[str], runtime_names: Sequence[str]) -> np.ndarray:
    """Index map translating plan-ordered DOF arrays into articulation order.

    ``result[k]`` is the plan index of the DOF that PhysX keeps in slot ``k``, so
    ``plan_values[result]`` is the array to hand to the articulation and
    ``out[result] = runtime_values`` translates a reading back. The two orders
    coincide for a single-axis rig and diverge as soon as a joint carries a proxy
    chain, which is why every crossing of that boundary goes through this map.
    """

    plan_index = {name: index for index, name in enumerate(plan_names)}
    try:
        return np.array([plan_index[name] for name in runtime_names], dtype=np.int64)
    except KeyError as exc:
        raise HumanRuntimeUnavailable(
            f"the articulation exposes DOF {exc.args[0]!r} that the plan does not declare"
        ) from exc


def _gather_runtime_order(values: np.ndarray, permutation: np.ndarray | None) -> np.ndarray:
    """Reorder plan-ordered values into articulation order; no map means pass-through.

    The permutation is optional state rather than a required argument because
    ``set_control_scale`` is also exercised duck-typed against a bare namespace by
    the CPU tests: callers that carry a map get the translation, callers that do
    not get their arrays back untouched, which matches a single-axis rig where the
    two orders coincide anyway.
    """

    if permutation is None:
        return values
    return values[..., permutation]


def _scatter_plan_order(values: np.ndarray, permutation: np.ndarray | None) -> np.ndarray:
    """Reorder articulation-ordered values into plan order; no map means pass-through."""

    if permutation is None:
        return values
    out = np.empty_like(values)
    out[..., permutation] = values
    return out


class HumanRuntime:
    """Thin, defensive wrapper around the articulation and its link rigid bodies.

    The wrapper exists so the physics entry points do not have to re-derive which
    Isaac API returns radians, how a contact view is created, or what happens when
    contact tracking is unavailable. Anything it cannot do is reported through
    :attr:`capabilities` and recorded in the export, never silently omitted.
    """

    def __init__(
        self,
        stage: Any,
        plan: HumanRigPlan,
        *,
        contact_categories: Sequence[str] = DEFAULT_CONTACT_CATEGORIES,
        max_contact_count: int = DEFAULT_MAX_CONTACT_COUNT,
        enable_contact_views: bool = False,
        root_path: str = HUMAN_ROOT_PATH,
        app: Any = None,
    ) -> None:
        try:
            from isaacsim.core.experimental.prims import Articulation, RigidPrim
        except ImportError as exc:  # pragma: no cover - host runtime dependent
            raise HumanRuntimeUnavailable(
                "isaacsim.core.experimental.prims is unavailable; run through ~/isaacsim/python.sh"
            ) from exc
        self._rigid_prim = RigidPrim
        self.plan = plan
        self._stage = stage
        self.root_path = root_path
        # Kept so callers can advance physics; nothing on the read path uses it.
        self._app = app
        self._link_collider_paths: dict[str, str] = {}
        self.articulation = Articulation(root_path)
        self.capabilities: dict[str, Any] = {
            # Kept for callers that predate the polled channel: true only when the
            # tensor contact view (net force per link) actually works.
            "contact_forces": False,
            "contact_filter_paths": [],
            "contact_error": None,
            # Which channel contact samples come from, or "unavailable". Recorded so a
            # consumer can tell a measured contact from a missing channel.
            "contact_source": "unavailable",
            # Set to usd_collider_path after decoding a contact report.
            "contact_attribution": "unresolved",
            "dof_order_matches_plan": False,
        }
        self._contact_interface: Any = None
        runtime_names = tuple(str(name) for name in self.articulation.dof_names)
        if sorted(runtime_names) != sorted(plan.dof_names):
            raise HumanRuntimeUnavailable(
                "the articulation's degrees of freedom do not match the plan: runtime "
                f"{sorted(runtime_names)} vs plan {sorted(plan.dof_names)}. The rig was "
                "not authored as planned, so joint targets would be applied to the wrong "
                "joints."
            )
        self.capabilities["dof_order_matches_plan"] = runtime_names == plan.dof_names
        self.capabilities["dof_order"] = list(runtime_names)
        # PhysX orders an articulation's DOFs by link traversal, not by the plan's
        # joint order. On a single-axis rig the two orders happen to coincide; a
        # proxy chain (a multi-axis joint) does not, and every positional write
        # then lands on the wrong joint -- the verify probes that bent spine2 when
        # asked for left_knee are that failure, measured. Every array crossing
        # this boundary is therefore translated explicitly, in both directions.
        self._plan_to_runtime = dof_permutation(plan.dof_names, runtime_names)
        self.capabilities["dof_plan_permutation"] = bool(
            np.any(self._plan_to_runtime != np.arange(len(self._plan_to_runtime)))
        )
        self._base_gains: tuple[np.ndarray, np.ndarray] | None = None
        self.control_scale = 1.0

        # ``Articulation.get_world_poses`` reports only the articulation's ROOT pose
        # (verified: a two-link articulation returns a single row). Per-link world
        # poses therefore come from a ``RigidPrim`` view over each link prim, which
        # reads fine even for non-root articulation links. The root prim is named
        # after the articulation path ("Human"), not after the skeleton joint it
        # carries ("pelvis"), so paths are built from the plan and the names are kept
        # out of the lookup entirely.
        self._link_paths = {
            link.name: (root_path if link.name == plan.root_link else f"{root_path}/{link.name}")
            for link in plan.links
        }
        # Capsule colliders live one level below their link (``<link>/collider``), and
        # only links with a capsule have one. They are what a contact point can be
        # matched against, so they are resolved once from the stage rather than
        # assumed from the plan -- the plan's capsule set and the authored set can
        # diverge, and containment must use what physics actually has.
        self._link_collider_paths = self._discover_collider_paths()
        # Contact volumes are composed at attribution time from the link's LIVE
        # physics pose plus this constant authored offset. USD-stage world
        # transforms freeze the moment stepping bypasses the fabric/USD sync
        # (``SimulationManager.step(update_fabric=False)`` -- measured in
        # ``scripts/humans/probe_stage_staleness.py``), so a stage-read volume
        # would hold the spawn pose forever and the first contact away from it
        # -- the first floor touch of a fall -- would be unattributable. That
        # was the frame-2298 escape of the AMASS trial, measured 2026-09-24.
        self._capsule_local_geometry = {
            name: _capsule_geometry_in_link_frame(
                pxr_modules(),
                stage.GetPrimAtPath(collider_path),
                stage.GetPrimAtPath(self._link_paths[name]),
            )
            for name, collider_path in self._link_collider_paths.items()
        }
        self.capabilities["contact_volume_poses"] = "live_link_poses"
        self._rigids: dict[str, Any] = {}
        self.capabilities["contact_filter_paths"] = []
        self._build_link_views(
            stage,
            contact_categories,
            max_contact_count,
            enable_contact_views=enable_contact_views,
        )

    # -- setup -------------------------------------------------------------

    def _discover_collider_paths(self) -> dict[str, str]:
        """Segment name -> capsule collider path, resolved from the authored stage.

        Walks the body's own prim tree and keeps links that actually carry a
        ``UsdGeom.Capsule``. Deriving this from the stage instead of the plan means
        containment always tests against the capsules physics knows about, so a
        planned-but-unauthored collider cannot silently swallow a contact.
        """

        runtime = pxr_modules()
        found: dict[str, str] = {}
        for name, link_path in self._link_paths.items():
            # The root body's capsule is under the non-rigid pelvis Xform, while
            # its RigidPrim is the articulation root itself.
            declared = self._stage.GetPrimAtPath(f"{self.root_path}/{name}/collider")
            if declared and declared.IsValid() and runtime.UsdGeom.Capsule(declared):
                found[name] = str(declared.GetPath())
                continue
            prim = self._stage.GetPrimAtPath(link_path)
            if not prim or not prim.IsValid():
                continue
            for child in prim.GetChildren():
                if runtime.UsdGeom.Capsule(child):
                    found[name] = str(child.GetPath())
                    break
        return found

    def _build_link_views(
        self,
        stage: Any,
        categories: Sequence[str],
        max_contact_count: int,
        *,
        enable_contact_views: bool,
    ) -> None:
        """Create a rigid-body view per link and pick a contact source.

        The tensor contact view is treated as a bonus, not the plan: on Isaac Sim
        6.0.1 it fails to construct with any filter set here. The reliable channel is
        PhysX's polled report, opened lazily on first read so that constructing the
        runtime never depends on the physics interface being ready.
        """

        try:
            filter_paths = environment_contact_paths(
                stage, human_root=self.root_path, categories=categories
            )
        except Exception as exc:  # noqa: BLE001 - degraded, and reported
            filter_paths = ()
            self.capabilities["contact_error"] = f"{type(exc).__name__}: {exc}"
        self.capabilities["contact_filter_paths"] = list(filter_paths)

        if filter_paths and enable_contact_views:
            try:
                self._rigids = {
                    name: self._rigid_prim(
                        path,
                        contact_filter_paths=list(filter_paths),
                        max_contact_count=int(max_contact_count),
                    )
                    for name, path in self._link_paths.items()
                }
                # The view raises on read rather than on construction, so probe once.
                next(iter(self._rigids.values())).get_net_contact_forces()
            except Exception as exc:  # noqa: BLE001 - degraded, and reported
                self.capabilities["contact_tensor_view_error"] = f"{type(exc).__name__}: {exc}"
                LOGGER.debug("tensor contact view unavailable: %s", exc)
                self._rigids = {}
            else:
                self.capabilities["contact_tensor_view"] = True
                self.capabilities["contact_forces"] = True

        if not self._rigids:
            # Pose reading must still work without contact reporting.
            self._rigids = {name: self._rigid_prim(path) for name, path in self._link_paths.items()}

        if not filter_paths:
            self.capabilities["contact_error"] = (
                "no environment colliders carried sim2sense:category in "
                f"{list(categories)}, so there is nothing to report contacts against"
            )
        elif not self.capabilities.get("contact_tensor_view"):
            if not enable_contact_views:
                # Recorded as a reason rather than an error: nothing failed in this
                # process, the caller declined a channel this build cannot deliver.
                # Keeping it distinct from ``contact_error`` is what lets a reader
                # tell "we did not ask" from "we asked and it broke".
                self.capabilities["contact_tensor_view_reason"] = (
                    "not requested: the tensors rigid-contact view cannot be built for "
                    "these patterns on this Isaac build, and asking for it raises from "
                    "an async physics-ready callback that cannot be caught; contacts "
                    "come from the polled physx report"
                )
            # The polled report does not need a tensor view, so the channel may still
            # be available. It is probed on first read.
            self.capabilities["contact_source"] = "physx_contact_report"

    def _open_contact_report(self) -> Any:
        """Lazily bind the polled PhysX contact report.

        Split out from construction because the interface is only valid once the
        timeline exists; binding it early is what made the tensor view path fragile.
        """

        if self.capabilities.get("contact_source") != "physx_contact_report":
            return None
        if self._contact_interface is not None:
            return self._contact_interface
        try:
            import omni.physx
        except ImportError as exc:  # pragma: no cover - host runtime dependent
            self.capabilities["contact_error"] = f"ImportError: {exc}"
            self.capabilities["contact_source"] = "unavailable"
            return None
        try:
            self._contact_interface = omni.physx.get_physx_simulation_interface()
        except Exception as exc:  # noqa: BLE001 - degraded, and reported
            self.capabilities["contact_error"] = f"{type(exc).__name__}: {exc}"
            self.capabilities["contact_source"] = "unavailable"
            return None
        return self._contact_interface

    def contact_samples(self) -> tuple[ContactSample, ...]:
        """Every contact point PhysX reported on the last step.

        Raises :class:`ContactSourceUnavailable` when no channel exists, so a caller
        cannot mistake "the channel is missing" for "the body touched nothing". An
        empty tuple from a working channel genuinely means no reported contacts.

        The report is polled, not accumulated: call once per step, immediately after
        stepping, or pairs will be missed or double-counted.
        """

        if not self.capabilities.get("contact_channel_usable", True):
            raise ContactSourceUnavailable(
                str(self.capabilities.get("contact_error") or "no contact channel")
            )
        interface = self._open_contact_report()
        if interface is None:
            raise ContactSourceUnavailable(
                str(self.capabilities.get("contact_error") or "no contact channel")
            )
        try:
            headers, data = interface.get_contact_report()
        except Exception as exc:  # noqa: BLE001 - degraded, and reported
            self.capabilities["contact_source"] = "unavailable"
            self.capabilities["contact_error"] = f"{type(exc).__name__}: {exc}"
            raise ContactSourceUnavailable(f"{type(exc).__name__}: {exc}") from exc

        rows = list(data)
        from pxr import PhysicsSchemaTools

        samples: list[ContactSample] = []
        for header in headers:
            offset = int(header.contact_data_offset)
            collider0 = int(header.collider0)
            collider1 = int(header.collider1)
            paths = tuple(str(PhysicsSchemaTools.intToSdfPath(value))
                          for value in (collider0, collider1))
            if not all(path.startswith("/") for path in paths):
                raise ContactSourceUnavailable(f"contact collider path decoding failed: {paths}")
            for record in rows[offset : offset + int(header.num_contact_data)]:
                samples.append(
                    ContactSample(
                        collider0=collider0,
                        collider1=collider1,
                        position_m=tuple(float(v) for v in record.position),
                        normal=tuple(float(v) for v in record.normal),
                        impulse_ns=tuple(float(v) for v in record.impulse),
                        separation_m=float(record.separation),
                        collider0_path=paths[0],
                        collider1_path=paths[1],
                    )
                )
        return tuple(samples)

    def human_contact_samples(self) -> tuple[ContactSample, ...]:
        """Human contacts with the body's collider first and normals pointing toward it."""

        return tuple(sample for sample, _ in self.attributed_contact_samples())

    def attributed_contact_samples(self) -> tuple[tuple[ContactSample, str], ...]:
        """Contact samples paired with the limb each landed on, from **one** report read.

        Sampling the report twice in a step can straddle a physics update -- the
        report for step *n* is replaced by step *n+1*'s between reads -- which would
        leave the positions and the limb names describing different steps. Callers
        that need both must therefore use this method, never two separate reads.
        """

        samples = self.contact_samples()
        collider_segments = {path: name for name, path in self._link_collider_paths.items()}
        resolved: list[tuple[ContactSample, str]] = []
        for sample in samples:
            for path in (sample.collider0_path, sample.collider1_path):
                if path in collider_segments:
                    if path == sample.collider1_path:
                        sample = replace(
                            sample, collider0=sample.collider1, collider1=sample.collider0,
                            collider0_path=sample.collider1_path,
                            collider1_path=sample.collider0_path,
                            normal=tuple(-v for v in sample.normal),
                            impulse_ns=tuple(-v for v in sample.impulse_ns),
                        )
                    resolved.append((sample, collider_segments[path]))
                    break
                if path == self.root_path or path.startswith(self.root_path + "/"):
                    raise ContactSourceUnavailable(
                        f"unregistered human collider in contact: {path}"
                    )
        self.capabilities["contact_attribution"] = "usd_collider_path"
        return tuple(resolved)

    def contact_segments(self) -> tuple[tuple[ContactSample, str], ...]:
        """Human contacts paired with the limb identified by the decoded collider path.

        Thin alias for :meth:`attributed_contact_samples`; both read the report once
        so the positions and the limb names can never describe different steps.
        """

        return self.attributed_contact_samples()

    def contact_segment_names(self) -> tuple[str, ...]:
        """Body segments that reported a contact on the last step, one per sample.

        This is the limb-level attribution the export records, in the same order as
        :meth:`human_contact_samples`.
        """

        return tuple(segment for _, segment in self.attributed_contact_samples())

    def _capsule_volumes(
        self,
    ) -> dict[str, tuple[tuple[float, float, float], float, tuple[float, float, float]]]:
        """World-space capsule volumes for every collidable link, computed fresh.

        Composed from each link's LIVE physics pose (the ``RigidPrim`` read that
        ``probe_loop_timing`` proved current under
        ``SimulationManager.step(update_fabric=False)``) and the constant authored
        capsule-to-link offset captured once at construction. Recomputed on each
        call rather than cached: the body moves every step, and a stale volume
        would attribute a contact to wherever the limb used to be. USD-stage world
        transforms are deliberately NOT used here: they freeze when stepping
        bypasses the fabric/USD sync, which is the measured cause of the AMASS
        trial's frame-2298 attribution escape (see
        ``scripts/humans/probe_stage_staleness.py``).
        """

        volumes: dict[
            str, tuple[tuple[float, float, float], float, tuple[float, float, float]]
        ] = {}
        for name, geometry in self._capsule_local_geometry.items():
            position, quaternion = self._world_pose_of(name)
            volumes[name] = _capsule_world_volume(
                quaternion_to_matrix(quaternion), position, geometry
            )
        return volumes

    def contact_support_summary(self) -> dict[str, Any]:
        """Contacts on the last step, split by whether they carry the body weight.

        A fall is not "the foot brushed the floor": it is the body's weight arriving
        somewhere. Separating resting contacts from glancing ones gives the labelling
        stage something to threshold on without re-parsing the raw report.

        A contact whose body-directed normal points upward and whose point sits at
        the resting surface is marked as support. Both collider paths remain available.

        The report is read exactly once here, so the counts, the impulses and the
        segment names all describe the same physics step.
        """

        attributed = self.attributed_contact_samples()
        support_z = self.support_surface_height_m()
        on_floor = [
            sample
            for sample, _ in attributed
            if support_z is not None
            and sample.normal[2] > self._SUPPORT_NORMAL_MIN_Z
            and sample.position_m[2] <= support_z + self._SUPPORT_HEIGHT_TOLERANCE_M
        ]
        return {
            "contact_count": len(attributed),
            "floor_contact_count": len(on_floor),
            "other_contact_count": len(attributed) - len(on_floor),
            "impulse_magnitude_ns": float(
                sum(sample.impulse_magnitude_ns for sample, _ in attributed)
            ),
            "max_impulse_ns": float(
                max(
                    (sample.impulse_magnitude_ns for sample, _ in attributed),
                    default=0.0,
                )
            ),
            "lowest_contact_z_m": float(
                min(
                    (sample.position_m[2] for sample, _ in attributed),
                    default=float("nan"),
                )
            ),
            "contact_segments": [segment for _, segment in attributed],
        }

    def support_surface_height_m(self) -> float | None:
        """World z of the highest ground-like surface under the body, or ``None``.

        Read from the surface prims' own bounds rather than assumed to be zero: a
        scene with a step or a raised floor makes ``z == 0`` the wrong reference, and a
        hard-coded zero would quietly mislabel every contact on the upper level.

        The world box is composed by hand -- the prim's own geometry box, then its
        eight corners pushed through the local-to-world transform. Both ``pxr``
        shortcuts are unusable here, each measured against a scene whose floor top
        the body's own resting contacts independently confirm to be ``z = 0.00``:

        * ``UsdGeom.BBoxCache.ComputeWorldBound`` returns a box in the wrong frame,
          carrying neither ``xformOp:scale`` nor the prim translate; the room floor
          reported ``z in [-0.0, -0.0]`` and the ceiling above it reported the same.
        * ``ComputeLocalBound`` returns an already-scaled box that is still offset by
          the prim translate, so composing it double-counts the translate and reported
          the same floor as ``-0.06``. It also yields an inverted-infinite sentinel for
          the ``render``/``proxy``/``guide`` purposes.
        * Neither ``BBox3d`` nor ``Range3d`` has ``TransformBy`` in this build
          (``hasattr`` is ``False`` for both), so the box cannot be transformed in
          place the way the USD documentation suggests.
        """

        runtime = pxr_modules()
        xform = runtime.UsdGeom.XformCache(runtime.Usd.TimeCode.Default())
        heights: list[float] = []
        for path in self._support_paths():
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            local = _prim_local_box(prim, runtime)
            if local is None:
                continue
            heights.append(_world_box_max_z(local, xform.GetLocalToWorldTransform(prim), runtime))
        return max(heights) if heights else None

    def _support_paths(self) -> tuple[str, ...]:
        """Environment collider paths that count as a resting surface."""

        wanted = set(SUPPORT_CATEGORIES)
        paths: list[str] = []
        for path in self.capabilities.get("contact_filter_paths") or ():
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            attribute = prim.GetAttribute("sim2sense:category")
            category = attribute.Get() if attribute and attribute.IsValid() else None
            if category is not None and str(category) in wanted:
                paths.append(str(path))
            elif any(token in str(path) for token in ("floor", "ground")):
                paths.append(str(path))
        return tuple(paths)

    # -- timeline ----------------------------------------------------------

    def play(self) -> None:
        """Start the timeline.

        Isaac's physics tensor views are only valid while the timeline is playing:
        reading a joint position before ``play()`` raises
        "Instance's physics tensor entity is not valid". Every state accessor here
        therefore requires the timeline to be running, and the entry points call
        this immediately after construction.
        """

        import omni.timeline

        omni.timeline.get_timeline_interface().play()

    def pause(self) -> None:
        """Stop the timeline."""

        import omni.timeline

        omni.timeline.get_timeline_interface().pause()

    # -- state -------------------------------------------------------------

    @property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.articulation.dof_names)

    def _to_runtime(self, values: np.ndarray) -> np.ndarray:
        """Reorder a plan-ordered vector into the articulation's DOF order."""

        return _gather_runtime_order(values, self._plan_to_runtime)

    def _to_plan(self, values: np.ndarray) -> np.ndarray:
        """Reorder an articulation-ordered vector into the plan's DOF order."""

        return _scatter_plan_order(values, self._plan_to_runtime)

    def dof_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.articulation.get_dof_limits()
        return (
            self._to_plan(low.numpy()[0].astype(np.float64)),
            self._to_plan(high.numpy()[0].astype(np.float64)),
        )

    def set_joint_positions(self, values_rad: Sequence[float]) -> None:
        """Write joint angles directly (kinematic replay).

        ``values_rad`` is in plan DOF order; the translation to the articulation's
        own order happens here, so callers never see the PhysX permutation.
        """

        self._check_length(values_rad)
        self.articulation.set_dof_positions(self._to_runtime(self._row(values_rad)))

    def set_joint_targets(
        self, values_rad: Sequence[float], velocities_rad_s: Sequence[float] | None = None
    ) -> None:
        """Set PD position and velocity targets in plan order; a hold has zero velocity."""

        self._check_length(values_rad)
        self.articulation.set_dof_position_targets(self._to_runtime(self._row(values_rad)))
        velocities = np.zeros(len(values_rad)) if velocities_rad_s is None else velocities_rad_s
        self._check_length(velocities)
        self.articulation.set_dof_velocity_targets(self._to_runtime(self._row(velocities)))

    def set_control_scale(self, scale: float) -> bool:
        """Scale the active control authority relative to the authored gains.

        The scale multiplies the drive **stiffness** (position feedback) only. The
        authored **damping is always kept**: passive joint damping is a property of
        the joint, not of the controller, and a body that lost active control is a
        damped ragdoll, not a frictionless one. The measured reason this matters: a
        zero-damped 57-DOF multi-axis ragdoll folds ballistically -- 975 deg/s of
        joint speed within the 0.2 s verification hold, 2864 deg/s mid-fall -- and
        the floor impact of that state NaNs the PhysX solver
        (``non-finite world position for link 'pelvis'``). Damping alone cannot
        track a position target, so the drives-off negative control stays valid,
        and it cannot hold a pose, so the gravity-drop positive control stays real.

        Used by the ``control_failure`` perturbation and by the verification negative
        control. Returning ``False`` lets the caller mark the trial as not carrying the
        intended perturbation instead of recording a claim that was never applied.

        The reference gains are captured on the first call and reused, so calling this
        once per frame (or driving the scale to 0 and back to 1) does not compound: an
        earlier version multiplied the CURRENT gains every time, which meant a 0.5 scale
        applied over 180 frames left the body with no drives at all, and a negative
        control could never be undone.
        """

        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"control scale must be within [0, 1], got {scale!r}")
        try:
            # This method is also exercised duck-typed against a bare namespace by
            # the CPU tests, so besides the articulation it may only rely on the
            # two attributes below plus an optional DOF permutation.
            permutation = getattr(self, "_plan_to_runtime", None)
            if self._base_gains is None:
                try:
                    stiffness, damping = self.articulation.get_dof_gains()
                    # Stored in plan order, like every other array this class
                    # exchanges with callers; the write below translates back.
                    self._base_gains = (
                        _scatter_plan_order(
                            np.asarray(stiffness.numpy(), dtype=np.float64), permutation
                        ),
                        _scatter_plan_order(
                            np.asarray(damping.numpy(), dtype=np.float64), permutation
                        ),
                    )
                except Exception:
                    # Isaac's experimental articulation view can expose the setter
                    # while refusing the getter until a tensor view has been fully
                    # initialised.  The USD plan is the authoritative source for the
                    # gains we authored, so use it as a deterministic fallback rather
                    # than silently disabling every controller test.
                    self._base_gains = (
                        np.asarray(
                            [joint.stiffness for joint in self.plan.joints], dtype=np.float64
                        ),
                        np.asarray(
                            [joint.damping for joint in self.plan.joints], dtype=np.float64
                        ),
                    )
            base_stiffness, base_damping = self._base_gains
            self.articulation.set_dof_gains(
                _gather_runtime_order(base_stiffness * float(scale), permutation),
                # Passive damping is not scaled: see the docstring. It is rewritten
                # unchanged so a partially-scaled state cannot linger after the
                # reference gains were re-captured.
                _gather_runtime_order(base_damping, permutation),
            )
            self.control_scale = float(scale)
            return True
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            LOGGER.warning("could not scale joint drives: %s", exc)
            return False

    def joint_positions_rad(self) -> np.ndarray:
        return self._to_plan(
            self.articulation.get_dof_positions().numpy()[0].astype(np.float64)
        )

    def control_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current stiffness and damping, in plan DOF order."""

        try:
            stiffness, damping = self.articulation.get_dof_gains()
            return (
                self._to_plan(np.asarray(stiffness.numpy(), dtype=np.float64).reshape(-1)),
                self._to_plan(np.asarray(damping.numpy(), dtype=np.float64).reshape(-1)),
            )
        except Exception:
            if self._base_gains is None:
                raise RuntimeError(
                    "control gains are not available before set_control_scale()"
                ) from None
            base_stiffness, base_damping = self._base_gains
            # What is actually commanded: scaled stiffness, unscaled passive damping.
            return (
                base_stiffness * self.control_scale,
                base_damping,
            )

    def joint_velocities_rad_s(self) -> np.ndarray:
        return self._to_plan(
            self.articulation.get_dof_velocities().numpy()[0].astype(np.float64)
        )

    def _world_pose_of(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self._rigids[link_name].get_world_poses()
        position = np.asarray(positions.numpy(), dtype=np.float64).reshape(-1, 3)[0]
        quaternion = np.asarray(orientations.numpy(), dtype=np.float64).reshape(-1, 4)[0]
        if not np.all(np.isfinite(position)):
            raise ValueError(f"non-finite world position for link {link_name!r}: {position}")
        if not np.all(np.isfinite(quaternion)):
            raise ValueError(f"non-finite world quaternion for link {link_name!r}: {quaternion}")
        return (
            position,
            quaternion,
        )

    def root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._world_pose_of(self.plan.root_link)

    def link_poses(self) -> dict[str, LinkTransform]:
        poses: dict[str, LinkTransform] = {}
        for link in self.plan.links:
            position, quaternion = self._world_pose_of(link.name)
            poses[link.name] = LinkTransform(
                rotation=quaternion_to_matrix(quaternion), translation=position
            )
        return poses

    def link_positions(self) -> np.ndarray:
        """World translations of every link, in plan order."""

        return np.stack([self._world_pose_of(link.name)[0] for link in self.plan.links], axis=0)

    def set_root_pose(self, position: Sequence[float], quaternion_wxyz: Sequence[float]) -> None:
        """Place the root body (used to reset between trials)."""

        self.articulation.set_world_poses(
            positions=np.asarray(position, dtype=np.float64).reshape(1, 3),
            orientations=np.asarray(quaternion_wxyz, dtype=np.float64).reshape(1, 4),
        )

    def reset_velocities(self) -> None:
        """Clear root and joint velocities after a verification teleport.

        Isaac's pose setters intentionally teleport the articulation while leaving
        its velocity state untouched.  Verification trials reuse one articulation,
        so carrying momentum from the previous target would make a single-DOF PD
        result depend on trial order.  This explicit reset is kept separate from
        :meth:`set_joint_positions` because motion replay may deliberately preserve
        velocities between frames.
        """

        zeros_dof = np.zeros((1, len(self.dof_names)), dtype=np.float64)
        zeros_root = np.zeros((1, 3), dtype=np.float64)
        self.articulation.set_dof_velocities(zeros_dof)
        self.articulation.set_velocities(zeros_root, zeros_root)

    def set_body_collisions_enabled(self, enabled: bool) -> int:
        """Enable or disable the human's authored collision shapes.

        The PD acceptance probe is a controller test and must not depend on a
        shoulder touching furniture or a wall in the fixed apartment.  Collision
        is restored before the gravity and floor checks, which continue to verify
        the physical scene response.
        """

        runtime = pxr_modules()
        changed = 0
        prefix = f"{self.root_path}/"
        for prim in self._stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix):
                continue
            if not prim.HasAPI(runtime.UsdPhysics.CollisionAPI):
                continue
            runtime.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(bool(enabled))
            changed += 1
        return changed

    def contact_force_magnitudes(self) -> np.ndarray | None:
        """Total contact impulse magnitude over the human, or ``None``.

        Prefers the tensor view's net contact force when it exists; otherwise derives
        an equivalent scalar from the polled report's impulses, so the recorded
        ``contact_force_n`` column is populated on this machine instead of being null.
        ``None`` still means no channel at all -- never "no contact".
        """

        if self.capabilities.get("contact_forces"):
            total = 0.0
            for rigid in self._rigids.values():
                forces = rigid.get_net_contact_forces().numpy()
                total += float(np.linalg.norm(np.asarray(forces, dtype=np.float64)))
            return np.array([total])
        try:
            summary = self.contact_support_summary()
        except ContactSourceUnavailable:
            return None
        return np.array([float(summary["impulse_magnitude_ns"])])

    def apply_force(self, body: str, force_n: Sequence[float]) -> bool:
        """Apply an external force at the named link's centre of mass."""

        rigid = self._rigids.get(body)
        if rigid is None:
            LOGGER.warning("cannot apply a force to unknown link %r", body)
            return False
        try:
            rigid.apply_forces_and_torques_at_pos(
                forces=np.asarray(force_n, dtype=np.float64).reshape(1, 3)
            )
            return True
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            LOGGER.warning("could not apply force to link %r: %s", body, exc)
            return False

    def root_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        linear, angular = self._rigids[self.plan.root_link].get_velocities()
        return linear.numpy()[0].astype(np.float64), angular.numpy()[0].astype(np.float64)

    def link_point_velocity(self, body: str, point_world: np.ndarray) -> np.ndarray:
        """World velocity at a point, including rotation about the link's COM.

        PhysX reports COM pose in the actor frame and linear velocity at the COM.
        This is body-side velocity; moving contact partners must be subtracted
        separately before treating it as relative slip.
        """
        if body not in self._rigids:
            raise ValueError(f"unknown body: {body}")
        if np.shape(point_world) != (3,) or not np.isfinite(point_world).all():
            raise ValueError("contact point must be a finite world-space vector")
        rigid = self._rigids[body]
        local_com, _ = rigid.get_coms()
        position, quaternion = self._world_pose_of(body)
        com = position + quaternion_to_matrix(quaternion) @ local_com.numpy()[0]
        linear, angular = rigid.get_velocities()
        return linear.numpy()[0] + np.cross(angular.numpy()[0], point_world - com)

    def apply_root_wrench(self, force_n: np.ndarray, torque_nm: np.ndarray) -> None:
        if any(np.shape(v) != (3,) or not np.isfinite(v).all() for v in (force_n, torque_nm)):
            raise ValueError("root wrench requires finite three-component force and torque")
        self._rigids[self.plan.root_link].apply_forces_and_torques_at_pos(
            forces=force_n.reshape(1, 3), torques=torque_nm.reshape(1, 3), local_frame=False
        )

    def _check_length(self, values: Sequence[float]) -> None:
        if len(values) != len(self.dof_names):
            raise ValueError(f"expected {len(self.dof_names)} joint values, got {len(values)}")

    def _row(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("joint values contain a non-finite entry")
        return array.reshape(1, -1)


def write_display_skin(stage: Any, points: Any, faces: Any) -> int:
    """Create or update a world-space display mesh, outside the articulation subtree.

    The authored ``/World/Human/Skin`` is written once from the rest template and is a
    rigid child of the pelvis, so during a trial it rides the root while the limbs stay
    where the rest pose put them: the viewport shows a T-pose body whose feet pop out of
    the floor as the body tips, and none of that is what physics did. This prim is the
    picture the recording actually supports -- same vertex order, world coordinates,
    sibling of the human rather than a child, and the authored skin is hidden in the
    transient session layer so only one body draws.

    Returns the number of vertices written.
    """

    runtime = pxr_modules()
    rows = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"display skin points must be (V, 3), got {rows.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"display skin faces must be (F, 3), got {triangles.shape}")
    if triangles.size and int(triangles.max()) >= rows.shape[0]:
        raise ValueError(
            f"display skin face index runs to {int(triangles.max())} with only "
            f"{rows.shape[0]} vertices"
        )
    mesh = runtime.UsdGeom.Mesh.Get(stage, DISPLAY_SKIN_PATH)
    if mesh is None or not mesh.GetPrim().IsValid():
        mesh = runtime.UsdGeom.Mesh.Define(stage, DISPLAY_SKIN_PATH)
        mesh.GetSubdivisionSchemeAttr().Set("none")
        authored = runtime.UsdGeom.Mesh.Get(stage, f"{HUMAN_ROOT_PATH}/Skin")
        if authored is not None and authored.GetPrim().IsValid():
            runtime.UsdGeom.Imageable(authored.GetPrim()).CreateVisibilityAttr().Set(
                runtime.UsdGeom.Tokens.invisible
            )
    mesh.GetPointsAttr().Set([runtime.Gf.Vec3f(*row) for row in rows])
    mesh.GetFaceVertexIndicesAttr().Set([int(value) for row in triangles for value in row])
    mesh.GetFaceVertexCountsAttr().Set([3] * int(triangles.shape[0]))
    return int(rows.shape[0])


def sample_body_points(
    plan: HumanRigPlan,
    poses: Mapping[str, LinkTransform],
    *,
    per_segment: int,
    seed: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Body surface proxy for one frame, from the links' **physics** poses."""

    return pose_surface_points(plan, poses, per_segment=per_segment, seed=seed)


def temporary_scene_copy(source: str | Path) -> tuple[Any, tempfile.TemporaryDirectory[str], Path]:
    """Copy a scene to a scratch directory and open it.

    Isaac Sim's stage-open hooks add ``/Render`` and viewport cameras to the
    in-memory layer, so verification and simulation must never run on the shipped
    file itself.
    """

    runtime = pxr_modules()
    workdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="sim2sense-human-"
    )
    source_path = Path(source)
    sandbox = Path(workdir.name) / source_path.name
    shutil.copy2(source_path, sandbox)
    stage = runtime.Usd.Stage.Open(str(sandbox))
    if stage is None:
        workdir.cleanup()
        raise FileNotFoundError(f"could not open stage: {sandbox}")
    return stage, workdir, sandbox


def perturbed_trial_notes(perturbation: PerturbationConfig, *, applied: bool) -> tuple[str, ...]:
    """Human-readable record of whether a perturbation actually took effect."""

    if perturbation.kind == "none":
        return ("unperturbed control trial; no perturbation was applied",)
    if perturbation.kind == "support_loss":
        return (
            f"support_loss perturbation {perturbation.id!r} is declared but NOT implemented "
            "by this runtime: removing support needs live floor-collider or friction "
            "mutation, which has not been validated. The trial carries no disturbance.",
        )
    if applied:
        return (f"{perturbation.kind} perturbation {perturbation.id!r} applied as configured",)
    return (
        f"{perturbation.kind} perturbation {perturbation.id!r} could NOT be applied by the "
        "runtime; the trial does not carry the intended disturbance and must not be "
        "reported as a perturbed trial",
    )
