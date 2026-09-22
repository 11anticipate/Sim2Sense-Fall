"""Plan a human rig: capsule links, PhysX joints, mass allocation, DOF mapping.

This module is pure CPU. It turns a validated :class:`RestSkeleton` plus a
:class:`RigConfig` into an immutable :class:`HumanRigPlan` that the Isaac Sim
author consumes verbatim, and it exposes a matching forward-kinematics function
so the same rig geometry can be checked without a simulator.

Joint mapping
-------------

Every SMPL joint rotation is expressed about the joint's **rest-aligned** axes:
SMPL's skinning chain is ``G_child = G_parent @ [R_child | J_child - J_parent]``,
so the offset to the child is carried by the parent's posed frame while the
rotation is applied about axes that stay aligned with the rest pose. A USD
revolute joint authored with identity local orientations, an axis token matching
the body axis, and ``localPos0`` set to the rest bone vector reproduces exactly
that chain -- there is no frame re-expression step that could hide a sign error.
:func:`forward_kinematics` is an independent CPU implementation of the same
chain, used to cross-check what the simulator reports.

Approximations this plan makes, on purpose
------------------------------------------

* One revolute joint per declared rotation axis. A joint with several axes
  becomes a chain of revolute joints with zero-length proxy links, because a
  USD/PhysX revolute joint has exactly one degree of freedom. The shipped
  configuration uses one axis per joint, so no proxy link is generated; the
  multi-axis path is planned and unit-tested but has **not** been run in Isaac
  Sim, and that is recorded in ``docs/human-simulation.md``.
* Capsules stand in for the body surface: they are a collision proxy, not skin.
  Capsules of neighbouring segments overlap in the rest pose (the torso is a
  stack of short bones with large radii). That is harmless because articulation
  self-collisions are disabled, and the overlap count is reported as a statistic
  rather than treated as an error.
* Mass weights are hand-set relative values, not fitted anthropometric data.
  They are normalised to the configured total mass, and the raw weights are kept
  in the plan so the assumption stays reviewable.
* Inertia is computed here as a solid-cylinder approximation so the plan can be
  sanity-checked, but the author lets PhysX derive inertia from the collision
  shape; the two are compared at verification time instead of being assumed
  equal.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import HumanConfig, RigConfig, SegmentConfig
from .rotations import (
    axis_angle_to_matrix,
    quaternion_to_matrix,
    validate_axis,
)
from .skeleton import RestSkeleton, SkeletonTopology, default_rest_skeleton

__all__ = [
    "CapsuleSpec",
    "HumanRigPlan",
    "JointSpec",
    "LinkTransform",
    "forward_kinematics",
    "plan_human_rig",
    "pose_surface_points",
    "quaternion_from_z",
    "validate_plan_geometry",
]

LOGGER = logging.getLogger(__name__)

#: Smallest chain-link mass the planner will emit, in kilograms.
MIN_LINK_MASS_KG = 1e-4
#: No single link may carry more than this share of the body mass.
MAX_LINK_MASS_FRACTION = 0.5
#: Proxy links carry no collider, so their inertia uses this nominal extent.
PROXY_NOMINAL_RADIUS_M = 0.04
PROXY_NOMINAL_LENGTH_M = 0.08
#: How far a capsule may sit below the floor at the declared spawn before it is an
#: error. Not zero: the spawn z is rounded to 1e-6 m for a readable plan, and a
#: sub-micron dip costs nothing while a real convention error is millimetres.
_SPAWN_CLEARANCE_TOLERANCE_M = 1e-5

_PROXY_MARKER = "__dof"


def _rounded(values: Iterable[float], digits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values)


def quaternion_from_z(direction: Sequence[float]) -> tuple[float, float, float, float]:
    """Unit quaternion ``(w, x, y, z)`` rotating local ``+Z`` onto ``direction``."""

    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("direction must have three components")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("cannot orient a capsule along a zero-length bone")
    target = vector / norm
    dot = float(np.clip(target[2], -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-12:
        # A half turn has a whole plane of valid axes; +X is the deterministic pick.
        return (0.0, 1.0, 0.0, 0.0)
    cross = np.cross(np.array([0.0, 0.0, 1.0]), target)
    quaternion = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    quaternion /= float(np.linalg.norm(quaternion))
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


def _cylinder_inertia(
    mass_kg: float, radius_m: float, length_m: float
) -> tuple[float, float, float]:
    """Principal inertia of a solid cylinder about its centre, axis along local ``+Z``."""

    axial = 0.5 * mass_kg * radius_m**2
    transverse = mass_kg * (3.0 * radius_m**2 + length_m**2) / 12.0
    return (transverse, transverse, axial)


@dataclass(frozen=True, slots=True)
class CapsuleSpec:
    """One capsule collider, positioned in its link's rest-aligned local frame."""

    link: str
    path: str
    center: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    axis: str
    radius_m: float
    cylinder_length_m: float
    bone_from: str
    bone_to: str

    def __post_init__(self) -> None:
        if self.axis != "z":
            raise ValueError(
                f"{self.link}: capsule axis must be 'z' (the author applies the "
                f"orientation), got {self.axis!r}"
            )
        if self.radius_m <= 0:
            raise ValueError(f"{self.link}: capsule radius must be positive")
        if self.cylinder_length_m < 0:
            raise ValueError(f"{self.link}: capsule cylinder length must be non-negative")
        _validate_unit_quaternion(self.orientation_wxyz, f"{self.link}.capsule.orientation")

    @property
    def total_length_m(self) -> float:
        """Full capsule extent along its axis, including the two end caps."""

        return self.cylinder_length_m + 2.0 * self.radius_m

    @property
    def direction(self) -> np.ndarray:
        """Capsule axis as a unit vector in the link frame."""

        return quaternion_to_matrix(self.orientation_wxyz) @ np.array([0.0, 0.0, 1.0])

    def lowest_point_z(self) -> float:
        """Lowest point of the capsule in its link's local frame."""

        _, _, z = self.center
        return z - abs(float(self.direction[2])) * self.cylinder_length_m / 2.0 - self.radius_m

    def highest_point_z(self) -> float:
        """Highest point of the capsule in its link's local frame."""

        _, _, z = self.center
        return z + abs(float(self.direction[2])) * self.cylinder_length_m / 2.0 + self.radius_m

    def lowest_world_z(self, link_rest_position: Sequence[float]) -> float:
        """Lowest point of the capsule in the body rest frame.

        Link frames are rest-aligned and translated, so the link's rest position
        is the only correction needed.
        """

        return float(link_rest_position[2]) + self.lowest_point_z()

    def highest_world_z(self, link_rest_position: Sequence[float]) -> float:
        """Highest point of the capsule in the body rest frame."""

        return float(link_rest_position[2]) + self.highest_point_z()

    def as_dict(self) -> dict[str, Any]:
        return {
            "link": self.link,
            "path": self.path,
            "center": list(self.center),
            "orientation_wxyz": list(self.orientation_wxyz),
            "axis": self.axis,
            "radius_m": self.radius_m,
            "cylinder_length_m": self.cylinder_length_m,
            "total_length_m": round(self.total_length_m, 6),
            "bone_from": self.bone_from,
            "bone_to": self.bone_to,
        }


def _validate_unit_quaternion(values: Sequence[float], name: str) -> None:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"{name} must be a 4-element quaternion, got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{name} is not a unit quaternion (norm {norm:.9f})")


@dataclass(frozen=True, slots=True)
class JointSpec:
    """One revolute joint in the articulation chain.

    ``name`` is both the USD prim name and the PhysX DOF name, so the runtime maps
    degrees of freedom by name rather than trusting an index ordering.
    """

    name: str
    chain_joint: str
    parent_link: str
    child_link: str
    axis: str
    lower_deg: float
    upper_deg: float
    local_pos0: tuple[float, float, float]
    local_pos1: tuple[float, float, float]
    stiffness: float
    damping: float
    max_force: float
    drive_type: str
    proxy: bool
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("joint spec needs a non-empty name")
        if self.parent_link == self.child_link:
            raise ValueError(f"{self.name}: a joint cannot join {self.parent_link!r} to itself")
        validate_axis(self.axis)
        if not self.lower_deg < self.upper_deg:
            raise ValueError(
                f"{self.name}: lower limit {self.lower_deg} is not below upper {self.upper_deg}"
            )
        for label, value in (("stiffness", self.stiffness), ("damping", self.damping)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{self.name}: {label} must be finite and non-negative")
        if not math.isfinite(self.max_force) or self.max_force <= 0:
            raise ValueError(f"{self.name}: max_force must be finite and positive")

    @property
    def range_deg(self) -> float:
        return self.upper_deg - self.lower_deg

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chain_joint": self.chain_joint,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
            "axis": self.axis,
            "lower_deg": self.lower_deg,
            "upper_deg": self.upper_deg,
            "local_pos0": list(self.local_pos0),
            "local_pos1": list(self.local_pos1),
            "stiffness": self.stiffness,
            "damping": self.damping,
            "max_force": self.max_force,
            "drive_type": self.drive_type,
            "proxy": self.proxy,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class FixedJointSpec:
    """A rigid attachment: the child link cannot move relative to its parent."""

    name: str
    chain_joint: str
    parent_link: str
    child_link: str
    local_pos0: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chain_joint": self.chain_joint,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
            "local_pos0": list(self.local_pos0),
        }


@dataclass(frozen=True, slots=True)
class LinkSpec:
    """One rigid body in the articulation."""

    name: str
    chain_joint: str
    role: str
    parent_link: str | None
    parent_joint: str | None
    path: str
    rest_position: tuple[float, float, float]
    local_translation: tuple[float, float, float]
    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    capsule: CapsuleSpec | None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ("root", "segment", "dof_proxy"):
            raise ValueError(f"{self.name}: unsupported link role {self.role!r}")
        if self.parent_link is None and self.role != "root":
            raise ValueError(f"{self.name}: only the root link may lack a parent")
        if self.role == "root" and self.parent_link is not None:
            raise ValueError(f"{self.name}: the root link must not have a parent")
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError(f"{self.name}: link mass must be finite and positive")
        for value in self.inertia_kg_m2:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{self.name}: principal inertia must be finite and positive, got "
                    f"{self.inertia_kg_m2}"
                )
        if self.capsule is not None and self.capsule.link != self.name:
            raise ValueError(f"{self.name}: capsule belongs to {self.capsule.link!r}")

    @property
    def has_collider(self) -> bool:
        return self.capsule is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chain_joint": self.chain_joint,
            "role": self.role,
            "parent_link": self.parent_link,
            "parent_joint": self.parent_joint,
            "path": self.path,
            "rest_position": list(self.rest_position),
            "local_translation": list(self.local_translation),
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "center_of_mass": list(self.center_of_mass),
            "has_collider": self.has_collider,
            "capsule": None if self.capsule is None else self.capsule.as_dict(),
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class HumanRigPlan:
    """The complete, reviewable description of the articulated human."""

    human_id: str
    topology_name: str
    root_link: str
    root_mode: str
    skeleton_source: str
    links: tuple[LinkSpec, ...]
    joints: tuple[JointSpec, ...]
    fixed_joints: tuple[FixedJointSpec, ...]
    total_mass_kg: float
    mass_weight_total: float
    self_collisions: bool
    contact_offset_m: float
    rest_offset_m: float
    linear_damping: float
    angular_damping: float
    ground_offset_m: float
    spawn_root_position: tuple[float, float, float]
    rest_joint_positions: tuple[tuple[float, float, float], ...]
    notes: tuple[str, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    def link(self, name: str) -> LinkSpec:
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(f"plan has no link {name!r}")

    def joint(self, name: str) -> JointSpec:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(f"plan has no joint {name!r}")

    def joint_for(self, chain_joint: str) -> JointSpec | None:
        """The revolute joint driven by a named skeleton joint, if it has one."""

        for joint in self.joints:
            if joint.chain_joint == chain_joint:
                return joint
        return None

    @property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def driven_joints(self) -> tuple[str, ...]:
        seen: list[str] = []
        for joint in self.joints:
            if joint.chain_joint not in seen:
                seen.append(joint.chain_joint)
        return tuple(seen)

    @property
    def colliders(self) -> tuple[CapsuleSpec, ...]:
        return tuple(link.capsule for link in self.links if link.capsule is not None)

    @property
    def standing_root_height_m(self) -> float:
        """Root-link world z at spawn, i.e. the resting pelvis height.

        This is the quantity a fall threshold should be expressed as a fraction of.
        It is deliberately the spawn coordinate rather than a separately stored
        ``ground_offset_m``: keeping one number means the two cannot disagree.
        """

        return float(self.spawn_root_position[2])

    @property
    def root(self) -> LinkSpec:
        return self.link(self.root_link)

    def dof_names_for(self, chain_joint: str) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints if joint.chain_joint == chain_joint)

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        return {joint.name: (joint.lower_deg, joint.upper_deg) for joint in self.joints}

    def as_dict(self, *, include_links: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "human_id": self.human_id,
            "topology": self.topology_name,
            "root_link": self.root_link,
            "root_mode": self.root_mode,
            "skeleton_source": self.skeleton_source,
            "total_mass_kg": self.total_mass_kg,
            "mass_weight_total": self.mass_weight_total,
            "self_collisions": self.self_collisions,
            "contact_offset_m": self.contact_offset_m,
            "rest_offset_m": self.rest_offset_m,
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
            "ground_offset_m": round(self.ground_offset_m, 6),
            "spawn_root_position": list(self.spawn_root_position),
            "dof_names": list(self.dof_names),
            "driven_joints": list(self.driven_joints),
            "joints": [joint.as_dict() for joint in self.joints],
            "fixed_joints": [joint.as_dict() for joint in self.fixed_joints],
            "rest_joint_positions": [list(position) for position in self.rest_joint_positions],
            "notes": list(self.notes),
            "stats": dict(self.stats),
        }
        if include_links:
            payload["links"] = [link.as_dict() for link in self.links]
        return payload

    def to_json(self, *, indent: int = 2, include_links: bool = True) -> str:
        return json.dumps(
            self.as_dict(include_links=include_links),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class LinkTransform:
    """A link's world pose, from forward kinematics or read back from PhysX."""

    rotation: np.ndarray
    translation: np.ndarray

    def transform_point(self, point: Sequence[float]) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=np.float64) + self.translation


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def _chain_layout(rig: RigConfig, topology: SkeletonTopology) -> list[tuple[str, str, str]]:
    """Flatten the skeleton into the articulation chain, in topological order.

    Returns ``(link_name, chain_joint, role)`` triples. A joint with ``n`` rotation
    axes becomes ``n - 1`` proxy links followed by the real link, and the segment's
    mass weight is split evenly across all ``n`` of them.
    """

    layout: list[tuple[str, str, str]] = []
    for name in topology.joint_names:
        if name == topology.root_name:
            layout.append((name, name, "root"))
            continue
        joint = rig.joints.get(name)
        count = 0 if joint is None else joint.dof_count
        for index in range(1, count):
            layout.append((f"{name}{_PROXY_MARKER}{index}", name, "dof_proxy"))
        layout.append((name, name, "segment"))
    return layout


def _split_weights(layout: Sequence[tuple[str, str, str]], rig: RigConfig) -> dict[str, float]:
    """Allocate each segment's relative mass weight across its DOF chain."""

    weights: dict[str, float] = {}
    for link_name, chain_joint, role in layout:
        weight = float(rig.segment(chain_joint).mass_weight)
        if role == "dof_proxy":
            continue
        chain = [entry for entry in layout if entry[1] == chain_joint]
        weights[link_name] = weight / len(chain)
        for proxy_name, _, proxy_role in chain:
            if proxy_role == "dof_proxy":
                weights[proxy_name] = weight / len(chain)
    return weights


def _capsule_for(
    joint: str,
    segment: SegmentConfig,
    topology: SkeletonTopology,
    rest: RestSkeleton,
    *,
    path: str,
) -> CapsuleSpec | None:
    """Build the capsule covering the bone from ``joint`` to its primary child."""

    children = topology.children_of(joint)
    if not children:
        if segment.aim is not None:
            raise ValueError(
                f"segment {joint!r}: aim {segment.aim!r} was given but {joint!r} has no child"
            )
        return None
    if segment.aim is not None:
        if segment.aim not in children:
            raise ValueError(
                f"segment {joint!r}: aim {segment.aim!r} is not a child of {joint!r} "
                f"(children: {list(children)})"
            )
        target = segment.aim
    else:
        # An anatomical bone is the child that continues the limb. Taking the
        # longest bone reproduces that for every joint except the root, which is
        # required to declare its aim explicitly.
        target = max(children, key=lambda child: rest.bone_length(child))
    head = np.asarray(rest.position(joint), dtype=np.float64)
    tail = np.asarray(rest.position(target), dtype=np.float64)
    bone = tail - head
    length = float(np.linalg.norm(bone))
    if length <= 0.0:
        raise ValueError(f"segment {joint!r}: zero-length bone to {target!r}")
    orientation = quaternion_from_z(bone)
    return CapsuleSpec(
        link=joint,
        path=path,
        center=_rounded(bone / 2.0),
        orientation_wxyz=tuple(round(float(value), 9) for value in orientation),  # type: ignore[arg-type]
        axis="z",
        radius_m=round(float(segment.radius_m), 6),
        cylinder_length_m=round(length, 6),
        bone_from=joint,
        bone_to=target,
    )


def _proxy_parent(link_name: str, smpl_parent: str) -> str:
    """Parent link of a ``<joint>__dofN`` proxy.

    The first proxy in a chain hangs off the joint's own SMPL parent; later proxies
    hang off the previous proxy. Returning the joint name for the first proxy (the
    obvious but wrong reading of the name) would make the link its own parent.
    """

    chain_joint, index = link_name.rsplit(_PROXY_MARKER, 1)
    position = int(index)
    if position <= 1:
        return smpl_parent
    return f"{chain_joint}{_PROXY_MARKER}{position - 1}"


def _standing_extent(config: HumanConfig, skeleton: RestSkeleton) -> tuple[float, float]:
    """Lowest and highest capsule point of the rest pose, in the body rest frame."""

    lowest = math.inf
    highest = -math.inf
    for joint, segment in config.rig.segments.items():
        capsule = _capsule_for(joint, segment, config.topology, skeleton, path="")
        if capsule is None:
            continue
        rest_z = float(skeleton.position(joint)[2])
        lowest = min(lowest, rest_z + capsule.lowest_point_z())
        highest = max(highest, rest_z + capsule.highest_point_z())
    if not math.isfinite(lowest) or not math.isfinite(highest):
        raise ValueError("rest pose has no capsules, so it has no measurable extent")
    return lowest, highest


def standing_height_for(config: HumanConfig, skeleton: RestSkeleton) -> float:
    """Floor-to-crown standing height of a rest skeleton under a rig config."""

    lowest, highest = _standing_extent(config, skeleton)
    return highest - lowest


def fit_rest_skeleton(
    config: HumanConfig, base: RestSkeleton, *, max_scale: float = 3.0
) -> RestSkeleton:
    """Rescale any rest skeleton so its **standing height** hits the target.

    One solver serves both the procedural stand-in and an imported SMPL body, so the two
    cannot drift into different scaling conventions. ``skeleton.height_m`` is the
    floor-to-crown stature: scaling the joint span to that number would be wrong, because
    the floor contact sits below the ankle joint and the crown sits above the head joint.
    The standing height is affine in a uniform scale factor (joint offsets scale, capsule
    radii do not), so two evaluations pin the slope and one step solves it exactly.
    """

    target = float(config.skeleton.height_m)
    first, second = 1.0, 1.05
    height_first = standing_height_for(config, base.scaled_by(first))
    height_second = standing_height_for(config, base.scaled_by(second))
    slope = (height_second - height_first) / (second - first)
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(
            f"standing height does not grow with scale (slope {slope!r}); check the "
            "segment radii and the rest skeleton"
        )
    scale = first + (target - height_first) / slope
    if not 0.2 <= scale <= max_scale:
        raise ValueError(
            f"a scale of {scale:.3f} would be needed for a {target:g} m figure, outside "
            f"the supported 0.2 to {max_scale:g} range; check skeleton.height_m"
        )
    fitted = base.scaled_by(scale, name=f"fitted to {target:g} m standing height")
    achieved = standing_height_for(config, fitted)
    if abs(achieved - target) > 1e-6:
        raise ValueError(
            f"height fit did not converge: target {target:g} m, achieved {achieved:.9f} m"
        )
    return fitted


def fit_procedural_skeleton(config: HumanConfig, *, max_scale: float = 3.0) -> RestSkeleton:
    """Fit the built-in proportional skeleton to the configured stature."""

    return fit_rest_skeleton(config, default_rest_skeleton(), max_scale=max_scale)


def plan_human_rig(
    config: HumanConfig,
    *,
    rest: RestSkeleton | None = None,
    spawn_xy: Sequence[float] = (0.0, 0.0),
    link_root: str = "/World/Human",
) -> HumanRigPlan:
    """Turn a validated configuration into an immutable rig plan.

    ``link_root`` is the articulation's root prim path; every link and collider
    path in the plan is relative to it.
    """

    if not isinstance(config, HumanConfig):
        raise ValueError("plan_human_rig needs a HumanConfig")
    rig: RigConfig = config.rig
    topology = config.topology
    skeleton = rest if rest is not None else fit_procedural_skeleton(config)
    if skeleton.topology.name != topology.name:
        raise ValueError(
            f"rest skeleton topology {skeleton.topology.name!r} does not match the "
            f"configured {topology.name!r}"
        )
    if topology.root_name != rig.root_joint:
        raise ValueError(f"rig.root_joint {rig.root_joint!r} is not the topology root")
    if len(spawn_xy) != 2:
        raise ValueError("spawn_xy must have two components")

    total_mass = float(config.skeleton.mass_kg)
    layout = _chain_layout(rig, topology)
    layout_names = [entry[0] for entry in layout]
    if len(set(layout_names)) != len(layout_names):
        raise ValueError(f"chain layout repeats link names: {layout_names}")
    weights = _split_weights(layout, rig)
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("rig mass weights sum to zero; nothing to allocate")

    # --- links -----------------------------------------------------------
    links: list[LinkSpec] = []
    for link_name, chain_joint, role in layout:
        if role == "root":
            parent_link: str | None = None
            parent_joint: str | None = None
            local_translation = (0.0, 0.0, 0.0)
        else:
            smpl_parent = topology.parent_of(chain_joint)
            assert smpl_parent is not None
            if _PROXY_MARKER in link_name:
                parent_link = _proxy_parent(link_name, smpl_parent)
            else:
                # The real link sits at the end of its own DOF chain, so its parent is
                # the last proxy when the joint has more than one axis.
                dof_count = (
                    0 if chain_joint not in rig.joints else rig.joints[chain_joint].dof_count
                )
                proxies = [f"{chain_joint}{_PROXY_MARKER}{index}" for index in range(1, dof_count)]
                parent_link = proxies[-1] if proxies else smpl_parent
            parent_joint = smpl_parent
            parent_segment_joint = parent_link.split(_PROXY_MARKER)[0]
            local_translation = _rounded(
                np.asarray(skeleton.position(chain_joint), dtype=np.float64)
                - np.asarray(skeleton.position(parent_segment_joint), dtype=np.float64)
            )
        segment = rig.segment(chain_joint)
        capsule = (
            _capsule_for(
                chain_joint,
                segment,
                topology,
                skeleton,
                path=f"{link_root}/{link_name}/collider",
            )
            if role != "dof_proxy"
            else None
        )
        mass = total_mass * weights[link_name] / weight_total
        if capsule is not None:
            inertia = _cylinder_inertia(mass, capsule.radius_m, capsule.total_length_m)
            center_of_mass = capsule.center
        else:
            inertia = _cylinder_inertia(mass, PROXY_NOMINAL_RADIUS_M, PROXY_NOMINAL_LENGTH_M)
            center_of_mass = (0.0, 0.0, 0.0)
        links.append(
            LinkSpec(
                name=link_name,
                chain_joint=chain_joint,
                role=role,
                parent_link=parent_link,
                parent_joint=parent_joint,
                path=f"{link_root}/{link_name}",
                rest_position=_rounded(skeleton.position(chain_joint)),
                local_translation=local_translation,
                mass_kg=round(mass, 9),
                inertia_kg_m2=tuple(round(float(value), 9) for value in inertia),  # type: ignore[arg-type]
                center_of_mass=tuple(round(float(value), 9) for value in center_of_mass),  # type: ignore[arg-type]
                capsule=capsule,
                tags=segment.tags,
            )
        )

    # --- joints ----------------------------------------------------------
    joints: list[JointSpec] = []
    fixed_joints: list[FixedJointSpec] = []
    for link_name, chain_joint, role in layout:
        if role == "root":
            continue
        link = next(entry for entry in links if entry.name == link_name)
        assert link.parent_link is not None
        joint_config = rig.joints.get(chain_joint)
        if role == "dof_proxy":
            assert joint_config is not None
            index = int(link_name.rsplit(_PROXY_MARKER, 1)[1])
            axis = joint_config.rotations[index - 1]
            low, high = joint_config.limits_deg[index - 1]
            anchor = (0.0, 0.0, 0.0)
            is_proxy = True
        elif joint_config is not None and joint_config.dof_count:
            axis = joint_config.rotations[-1]
            low, high = joint_config.limits_deg[-1]
            anchor = (
                _rounded(
                    np.asarray(skeleton.position(chain_joint), dtype=np.float64)
                    - np.asarray(skeleton.position(link.parent_link), dtype=np.float64)
                )
                if link.parent_link == topology.parent_of(chain_joint)
                else (0.0, 0.0, 0.0)
            )
            is_proxy = False
        else:
            anchor = _rounded(
                np.asarray(skeleton.position(chain_joint), dtype=np.float64)
                - np.asarray(skeleton.position(link.parent_link), dtype=np.float64)
            )
            fixed_joints.append(
                FixedJointSpec(
                    name=f"{chain_joint}__fixed",
                    chain_joint=chain_joint,
                    parent_link=link.parent_link,
                    child_link=link_name,
                    local_pos0=anchor,
                )
            )
            continue
        assert joint_config is not None
        drive = joint_config.drive
        joints.append(
            JointSpec(
                name=link_name,
                chain_joint=chain_joint,
                parent_link=link.parent_link,
                child_link=link_name,
                axis=axis,
                lower_deg=float(low),
                upper_deg=float(high),
                local_pos0=anchor,
                local_pos1=(0.0, 0.0, 0.0),
                stiffness=drive.stiffness,
                damping=drive.damping,
                max_force=drive.max_force,
                drive_type=drive.drive_type,
                proxy=is_proxy,
                tags=joint_config.tags,
            )
        )

    colliders = [link for link in links if link.capsule is not None]
    # Ground clearance is measured with forward kinematics from the root link at the
    # world origin -- the same chain the USD author emits and the simulator drives --
    # rather than by offsetting each capsule by its link's rest position. The two
    # agree only when the root link's rest z is zero, and SMPL's root (``pelvis``)
    # sits 0.2336 m above the body origin. Offsetting by the rest position ignores
    # that, which used to place the feet 44.6 mm through the floor at spawn.
    rest_poses = _rest_poses(links, joints, fixed_joints, root_name=rig.root_joint)
    ground_offset, standing_height = _measure_clearance(links, rest_poses)
    overlap_pairs = _rest_overlap_pairs(colliders)
    plan = HumanRigPlan(
        human_id=config.human_id,
        topology_name=topology.name,
        root_link=topology.root_name,
        root_mode=rig.root_mode,
        skeleton_source=skeleton.source,
        links=tuple(links),
        joints=tuple(joints),
        fixed_joints=tuple(fixed_joints),
        total_mass_kg=round(total_mass, 6),
        mass_weight_total=round(weight_total, 6),
        self_collisions=rig.self_collisions,
        contact_offset_m=rig.contact_offset_m,
        rest_offset_m=rig.rest_offset_m,
        linear_damping=rig.linear_damping,
        angular_damping=rig.angular_damping,
        ground_offset_m=ground_offset,
        spawn_root_position=(
            round(float(spawn_xy[0]), 6),
            round(float(spawn_xy[1]), 6),
            round(float(ground_offset), 6),
        ),
        rest_joint_positions=tuple(skeleton.joint_positions),
        notes=(
            "one revolute joint per declared rotation axis; a multi-axis joint becomes "
            "a chain of revolute joints with zero-length proxy links",
            "capsules are a collision proxy, not a skin mesh",
            "mass weights are hand-set relative values normalised to the configured "
            f"total of {total_mass:g} kg, not fitted anthropometric data",
            "inertia is a solid-cylinder approximation recorded for review; the author "
            "lets PhysX derive inertia from the collision shape",
        ),
        stats={
            "link_count": len(links),
            "collider_count": len(colliders),
            "dof_count": len(joints),
            "fixed_joint_count": len(fixed_joints),
            "proxy_link_count": sum(link.role == "dof_proxy" for link in links),
            "driven_joint_count": len({joint.chain_joint for joint in joints}),
            "rest_capsule_overlap_pairs": overlap_pairs,
            "standing_height_m": round(standing_height, 6),
            "target_height_m": config.skeleton.height_m,
            "total_capsule_length_m": round(
                sum(link.capsule.total_length_m for link in colliders),
                6,  # type: ignore[union-attr]
            ),
        },
    )
    validate_plan_geometry(plan)
    LOGGER.info(
        "planned rig %s: %d links, %d colliders, %d DOF, ground offset %.4f m",
        plan.human_id,
        plan.stats["link_count"],
        plan.stats["collider_count"],
        plan.stats["dof_count"],
        plan.ground_offset_m,
    )
    return plan


def _rest_poses(
    links: Sequence[LinkSpec],
    joints: Sequence[JointSpec],
    fixed_joints: Sequence[FixedJointSpec],
    *,
    root_name: str,
) -> dict[str, LinkTransform]:
    """Forward kinematics of the planned chain at rest, root link at the world origin.

    ``forward_kinematics`` needs a finished plan, but the plan's ground offset is
    itself measured from the rest pose, so this walks the same chain from the parts
    that already exist. It mirrors :func:`forward_kinematics` exactly and the two are
    asserted to agree in the test suite, so an edit here cannot silently drift from
    the function the simulator is cross-checked against.

    The chain is walked breadth-first from the root rather than in joint declaration
    order: a link's parent is not guaranteed to appear earlier in the joint tuple
    (``left_collar`` attaches to ``spine3``, which is declared after it), so a single
    linear pass would read an unplaced parent.

    The returned translations are therefore **root-link world positions**: the root
    link sits at ``(0, 0, 0)`` and every other link is placed relative to it.
    """

    by_name = {link.name: link for link in links}
    if root_name not in by_name:
        raise ValueError(f"root link {root_name!r} is not part of the chain")
    parent_of: dict[str, str] = {}
    joint_of: dict[str, JointSpec] = {}
    fixed_of: dict[str, FixedJointSpec] = {}
    for joint in joints:
        if joint.child_link in parent_of:
            raise ValueError(f"{joint.child_link}: more than one joint claims this link")
        parent_of[joint.child_link] = joint.parent_link
        joint_of[joint.child_link] = joint
    for joint in fixed_joints:
        if joint.child_link in parent_of:
            raise ValueError(f"{joint.child_link}: more than one joint claims this link")
        parent_of[joint.child_link] = joint.parent_link
        fixed_of[joint.child_link] = joint

    poses: dict[str, LinkTransform] = {
        root_name: LinkTransform(rotation=np.eye(3), translation=np.zeros(3))
    }
    pending = [root_name]
    while pending:
        parent_name = pending.pop(0)
        parent = poses[parent_name]
        for child_name in by_name:
            if parent_of.get(child_name) != parent_name:
                continue
            anchor: np.ndarray
            if child_name in joint_of:
                joint = joint_of[child_name]
                anchor = np.asarray(joint.local_pos0, dtype=np.float64)
                rotation = parent.rotation @ _axis_rotation(joint.axis, 0.0)
            else:
                fixed = fixed_of[child_name]
                anchor = np.asarray(fixed.local_pos0, dtype=np.float64)
                rotation = parent.rotation.copy()
            poses[child_name] = LinkTransform(
                rotation=rotation,
                translation=parent.translation + parent.rotation @ anchor,
            )
            pending.append(child_name)
    missing = [link.name for link in links if link.name not in poses]
    if missing:
        raise ValueError(f"rest pose is unreachable for links: {missing}")
    return poses


def _measure_clearance(
    links: Sequence[LinkSpec],
    poses: Mapping[str, LinkTransform],
) -> tuple[float, float]:
    """Standing root-link height and stature of the rest pose, in metres.

    ``ground_offset`` is the world z a root link needs for the lowest capsule to
    touch ``z = 0``; it is also the standing pelvis height, so a fall threshold
    expressed as a fraction of it means what it claims to mean. ``standing_height``
    is the vertical extent of the collider set in the same pose, reported for
    review -- not the nominal stature, which includes the skin.
    """

    lowest = math.inf
    highest = -math.inf
    for link in links:
        capsule = link.capsule
        if capsule is None:
            continue
        pose = poses[link.name]
        # Capsules are authored in a rest-aligned link frame, so only the link's
        # pose matters: no per-link rotation is applied to the capsule itself.
        base = float(pose.translation[2])
        lowest = min(lowest, base + capsule.lowest_point_z())
        highest = max(highest, base + capsule.highest_point_z())
    if not math.isfinite(lowest):
        raise ValueError("rig has no capsule colliders, so it has no ground offset")
    return -lowest, highest - lowest


def _rest_overlap_pairs(links: Sequence[LinkSpec]) -> int:
    """Count capsule pairs that interpenetrate in the rest pose.

    This is informational, not an error: the torso is a stack of short bones with
    large radii, so neighbouring capsules overlap by design. Whether that matters
    is decided by ``self_collisions``, which is disabled in the shipped rig.
    Capsules are compared in the **body rest frame**, using each link's rest
    position, because a capsule's centre is stored in its own link frame.
    """

    count = 0
    for index, left in enumerate(links):
        for right in links[index + 1 :]:
            assert left.capsule is not None and right.capsule is not None
            if _world_separation(left, right) < -1e-6:
                count += 1
    return count


def _world_separation(left: LinkSpec, right: LinkSpec) -> float:
    """Smallest surface distance between two links' capsules in the rest frame."""

    assert left.capsule is not None and right.capsule is not None
    left_a, left_b = _world_segment(left)
    right_a, right_b = _world_segment(right)
    distance = _segment_distance(left_a, left_b, right_a, right_b)
    return distance - left.capsule.radius_m - right.capsule.radius_m


def _world_segment(link: LinkSpec) -> tuple[np.ndarray, np.ndarray]:
    capsule = link.capsule
    assert capsule is not None
    center = np.asarray(capsule.center, dtype=np.float64) + np.asarray(
        link.rest_position, dtype=np.float64
    )
    axis = capsule.direction
    half = capsule.cylinder_length_m / 2.0
    return center - axis * half, center + axis * half


def validate_plan_geometry(
    plan: HumanRigPlan,
    *,
    min_mass_kg: float = MIN_LINK_MASS_KG,
    max_mass_fraction: float = MAX_LINK_MASS_FRACTION,
) -> None:
    """Reject a plan that could not be simulated sensibly.

    Checks mass allocation, rest-pose ground clearance and the DOF structure. It
    deliberately does **not** reject rest-pose capsule overlap, for the reason
    given in :func:`_rest_overlap_pairs`.
    """

    if not math.isfinite(plan.ground_offset_m):
        raise ValueError("ground offset must be finite")
    if plan.ground_offset_m <= 0.0:
        raise ValueError(
            f"ground offset {plan.ground_offset_m:.6f} m must be positive: it is the root "
            "link height that puts the lowest capsule on the floor"
        )
    if abs(sum(link.mass_kg for link in plan.links) - plan.total_mass_kg) > 1e-6:
        raise ValueError(
            f"link masses sum to {sum(link.mass_kg for link in plan.links):.6f} kg but the "
            f"plan declares {plan.total_mass_kg:.6f} kg"
        )
    for link in plan.links:
        if link.mass_kg < min_mass_kg:
            raise ValueError(f"{link.name}: mass {link.mass_kg} kg is below {min_mass_kg} kg")
        share = link.mass_kg / plan.total_mass_kg
        if share > max_mass_fraction:
            raise ValueError(
                f"{link.name}: carries {share:.1%} of the body mass, above the "
                f"{max_mass_fraction:.0%} bound; check the mass weights"
            )
    # Spawn check: the plan's own spawn root position must put the lowest capsule on
    # the floor. Validating the rest pose instead used to mask a wrong spawn, because
    # the two were computed with different conventions.
    spawned_poses = forward_kinematics(plan, {}, root_position=plan.spawn_root_position)
    for link in plan.links:
        if link.capsule is None:
            continue
        pose = spawned_poses[link.name]
        lowest = float(pose.translation[2]) + link.capsule.lowest_point_z()
        if lowest < -_SPAWN_CLEARANCE_TOLERANCE_M:
            raise ValueError(
                f"{link.name}: capsule dips {abs(lowest):.4f} m below the floor at the "
                f"declared spawn root position {tuple(plan.spawn_root_position)}"
            )
    if abs(plan.spawn_root_position[2] - plan.ground_offset_m) > _SPAWN_CLEARANCE_TOLERANCE_M:
        raise ValueError(
            f"spawn root height {plan.spawn_root_position[2]:.6f} m does not match the "
            f"ground offset {plan.ground_offset_m:.6f} m; both must use the root-link "
            "world-z convention"
        )
    for joint in plan.joints:
        parent = plan.link(joint.parent_link)
        # A DOF proxy deliberately carries no collider: it exists only to give a
        # multi-axis joint a second revolute frame.
        if parent.capsule is None and parent.role != "dof_proxy" and parent.name != plan.root_link:
            raise ValueError(
                f"{joint.name}: parent link {joint.parent_link!r} has no collider; a chain "
                "link without collision geometry is a planning mistake"
            )
        if joint.local_pos0 != (0.0, 0.0, 0.0) and joint.proxy:
            raise ValueError(f"{joint.name}: a proxy joint must sit at its parent's origin")


def forward_kinematics(
    plan: HumanRigPlan,
    joint_angles_rad: Mapping[str, float] | None = None,
    *,
    root_position: Sequence[float] | None = None,
    root_rotation: Sequence[float] | None = None,
) -> dict[str, LinkTransform]:
    """CPU forward kinematics for the planned chain.

    ``joint_angles_rad`` maps a DOF name (see :attr:`HumanRigPlan.dof_names`) to a
    scalar angle in radians; missing DOFs are zero. The result maps every link name
    to its world pose.

    This is an independent implementation of the same chain the USD author emits,
    so agreement between the two is real evidence rather than a restatement of the
    same code.
    """

    angles: dict[str, float] = {}
    for joint in plan.joints:
        value = (joint_angles_rad or {}).get(joint.name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{joint.name}: joint angle must be a single number, got {value!r}; "
                "multi-axis joints take one entry per DOF name"
            )
        if not math.isfinite(value):
            raise ValueError(f"{joint.name}: joint angle must be finite, got {value!r}")
        angles[joint.name] = float(value)
    source = np.zeros(3) if root_position is None else np.asarray(root_position, dtype=np.float64)
    if source.shape != (3,):
        raise ValueError("root_position must have three components")
    rotation = axis_angle_to_matrix(np.zeros(3) if root_rotation is None else root_rotation)

    poses: dict[str, LinkTransform] = {
        plan.root_link: LinkTransform(rotation=rotation, translation=source)
    }
    children: dict[str, list[str]] = {}
    for link in plan.links:
        if link.parent_link is not None:
            children.setdefault(link.parent_link, []).append(link.name)
    joint_by_child = {joint.child_link: joint for joint in plan.joints}

    pending = [plan.root_link]
    while pending:
        parent = pending.pop(0)
        parent_pose = poses[parent]
        for child_name in children.get(parent, []):
            joint = joint_by_child.get(child_name)
            child = plan.link(child_name)
            if joint is None:
                # Fixed attachment: also a USD joint, so the same anchor rule applies.
                offset = np.asarray(child.local_translation, dtype=np.float64)
                poses[child_name] = LinkTransform(
                    rotation=parent_pose.rotation.copy(),
                    translation=parent_pose.translation + parent_pose.rotation @ offset,
                )
            else:
                anchor = np.asarray(joint.local_pos0, dtype=np.float64)
                # A USD joint constrains the child's origin to the parent's joint
                # frame, so the authored local translation of the child link is
                # overridden by the constraint. Only the anchor counts here; adding
                # the authored offset too would double-count the bone vector.
                joint_position = parent_pose.translation + parent_pose.rotation @ anchor
                child_rotation = parent_pose.rotation @ _axis_rotation(
                    joint.axis, angles[joint.name]
                )
                poses[child_name] = LinkTransform(
                    rotation=child_rotation,
                    translation=joint_position,
                )
            pending.append(child_name)
    return poses


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    vector = np.zeros(3)
    vector["xyz".index(validate_axis(axis))] = float(angle_rad)
    return axis_angle_to_matrix(vector)


def pose_surface_points(
    plan: HumanRigPlan,
    poses: Mapping[str, LinkTransform],
    *,
    per_segment: int,
    seed: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Sample a deterministic point cloud on the capsule surfaces of each link.

    The wireless stage needs body geometry per time step. Until the licensed SMPL
    skin is available there is no skinned mesh to export, so this returns the
    physical surface proxy instead: points fixed in each link's local frame and
    carried into world space by the link's **physics** pose, so the cloud follows
    the simulated body rather than the reference motion. The substitution is
    recorded in the export provenance.

    Returns the world-space points and the owning link name for each point.
    """

    if per_segment < 0:
        raise ValueError("per_segment must be non-negative")
    generator = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    owners: list[str] = []
    for link in plan.links:
        capsule = link.capsule
        if capsule is None or per_segment == 0:
            continue
        pose = poses.get(link.name)
        if pose is None:
            raise KeyError(f"missing pose for link {link.name!r}")
        local = capsule_surface_samples(capsule, per_segment, generator)
        chunks.append((pose.rotation @ local.T).T + pose.translation)
        owners.extend([link.name] * len(local))
    if not chunks:
        return np.zeros((0, 3)), ()
    return np.concatenate(chunks, axis=0), tuple(owners)


def capsule_surface_samples(
    capsule: CapsuleSpec, count: int, generator: np.random.Generator
) -> np.ndarray:
    """``count`` points distributed uniformly by area on a capsule's surface.

    Returned in the link's local frame. Cylinder points are chosen by area; cap
    points use the ``z = +- (h/2 + r|cos t|)`` parameterisation, which is exactly
    the hemisphere surface under a uniform inclination cosine.
    """

    if count < 1:
        return np.zeros((0, 3))
    radius = capsule.radius_m
    half = capsule.cylinder_length_m / 2.0
    cylinder_area = 2.0 * math.pi * radius * capsule.cylinder_length_m
    cap_area = 4.0 * math.pi * radius**2
    total = cylinder_area + cap_area
    cylinder_share = cylinder_area / total if total > 0 else 0.0

    phi = generator.random(count) * 2.0 * math.pi
    on_cylinder = generator.random(count) < cylinder_share
    samples = np.empty((count, 3), dtype=np.float64)
    cylinder_count = int(on_cylinder.sum())
    if cylinder_count:
        samples[:cylinder_count, 0] = radius * np.cos(phi[:cylinder_count])
        samples[:cylinder_count, 1] = radius * np.sin(phi[:cylinder_count])
        samples[:cylinder_count, 2] = generator.uniform(-half, half, cylinder_count)
    cap_count = count - cylinder_count
    if cap_count:
        cosine = generator.uniform(0.0, 1.0, cap_count)
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
        sign = generator.choice([-1.0, 1.0], cap_count)
        samples[cylinder_count:, 0] = radius * sine * np.cos(phi[cylinder_count:])
        samples[cylinder_count:, 1] = radius * sine * np.sin(phi[cylinder_count:])
        samples[cylinder_count:, 2] = sign * (half + radius * cosine)
    orientation = quaternion_to_matrix(capsule.orientation_wxyz)
    return samples @ orientation.T + np.asarray(capsule.center, dtype=np.float64)


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Distance between two 3D line segments (Ericson, *Real-Time Collision Detection*)."""

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a <= 1e-18 and e <= 1e-18:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-18:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= 1e-18:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            if denominator > 1e-18:
                s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))
