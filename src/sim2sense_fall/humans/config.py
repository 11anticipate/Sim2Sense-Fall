"""Strict configuration for the human rig, its controller and the export contract.

Everything the physics stage needs -- capsule radii, mass weights, joint axes,
limits, drive gains, contact parameters, solver iterations, perturbation scripts
and the ground-truth sampling rate -- is declared here rather than hard-coded in
the simulator entry points. The document is validated on load, so a rig that
would silently produce a limbless or explosively-driven body fails before Isaac
Sim is ever started.

Units are explicit in every field name: metres, kilograms, newtons, degrees,
seconds. Joint limits are in **degrees** because that is what ``UsdPhysics``
revolute limits and angular drives use; the rest of the pipeline works in radians
and converts at the authoring boundary, which is the one place a units mistake
would otherwise be invisible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..scenes.numbers import finite_number, strict_bool
from .rotations import validate_axis
from .skeleton import SkeletonTopology, smpl_skeleton

__all__ = [
    "CONTROL_MODES",
    "DOF_KINDS",
    "PERTURBATION_KINDS",
    "ROOT_MODES",
    "ControlConfig",
    "DriveConfig",
    "ExportConfig",
    "HumanConfig",
    "JointConfig",
    "PerturbationConfig",
    "RigConfig",
    "SegmentConfig",
    "SimulationConfig",
    "SkeletonConfig",
    "VisualizationConfig",
    "load_human_config",
]

DOF_KINDS: tuple[str, ...] = ("fixed", "revolute")
ROOT_MODES: tuple[str, ...] = ("free", "anchored")
CONTROL_MODES: tuple[str, ...] = ("kinematic", "pd", "torque_free")
PERTURBATION_KINDS: tuple[str, ...] = ("none", "force", "support_loss", "control_failure")

#: A revolute joint may not be limited beyond a full turn in either direction.
MAX_LIMIT_DEG = 180.0
#: Mass weights are relative, so only their ratios matter; they must be positive.
MIN_MASS_WEIGHT = 1e-6


def _as_float(mapping: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    if key not in mapping:
        if default is None:
            raise ValueError(f"missing required numeric field {key!r}")
        return float(default)
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        hint = ""
        if isinstance(value, str):
            hint = (
                " (YAML 1.1 needs a signed exponent for floats, so write 2.4e+9 or "
                "0.008333333333333333 rather than 8.33e-3)"
            )
        raise ValueError(f"{key!r} must be a number, got {value!r}{hint}")
    return finite_number(value, key)


def _as_int(mapping: Mapping[str, Any], key: str, *, default: int) -> int:
    if key not in mapping:
        return int(default)
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer, got {value!r}")
    return value


def _as_bool(mapping: Mapping[str, Any], key: str, *, default: bool) -> bool:
    if key not in mapping:
        return default
    return strict_bool(mapping[key], key)


def _as_str(mapping: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    if key not in mapping:
        if default is None:
            raise ValueError(f"missing required string field {key!r}")
        return default
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be a non-empty string, got {value!r}")
    return value.strip()


def _as_triple(mapping: Mapping[str, Any], key: str) -> tuple[float, float, float]:
    value = mapping.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key!r} must be a list of three numbers, got {value!r}")
    return tuple(  # type: ignore[return-value]
        finite_number(component, f"{key}[{index}]") for index, component in enumerate(value)
    )


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{where}: unsupported keys {unknown}")


def _require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class SkeletonConfig:
    """Which topological skeleton to use and what body to instantiate."""

    topology: str
    model_asset: str | None
    allow_procedural_skeleton: bool
    height_m: float
    mass_kg: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.topology not in ("smpl", "smplh"):
            raise ValueError(
                f"skeleton.topology must be 'smpl' or 'smplh', got {self.topology!r}; "
                "SMPL-X is deliberately not supported (the project decided against it)"
            )
        if self.model_asset is not None and not self.model_asset.strip():
            raise ValueError("skeleton.model_asset must be a non-empty string or omitted")
        strict_bool(self.allow_procedural_skeleton, "skeleton.allow_procedural_skeleton")
        finite_number(self.height_m, "skeleton.height_m")
        finite_number(self.mass_kg, "skeleton.mass_kg")
        if self.height_m <= 0:
            raise ValueError(f"skeleton.height_m must be positive, got {self.height_m!r}")
        if self.mass_kg <= 0:
            raise ValueError(f"skeleton.mass_kg must be positive, got {self.mass_kg!r}")


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Pose and framing values used by the static human inspection view.

    The values are radians and keyed by the planned DOF names.  Keeping this
    in the YAML configuration makes the visual pose reproducible and prevents
    a display-only angle from silently diverging from the PhysX rig.
    """

    default_pose_rad: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.default_pose_rad]
        if len(names) != len(set(names)):
            raise ValueError("visualization.default_pose_rad contains duplicate DOF names")
        for name, value in self.default_pose_rad:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("visualization.default_pose_rad keys must be non-empty strings")
            finite_number(value, f"visualization.default_pose_rad.{name}")

    def pose_dict(self) -> dict[str, float]:
        """Return the configured pose as a mutable mapping for kinematics/runtime APIs."""

        return {name: float(value) for name, value in self.default_pose_rad}


@dataclass(frozen=True, slots=True)
class SegmentConfig:
    """A capsule collider and a relative mass weight for one body link.

    ``aim`` names the child joint the capsule points at, so the capsule covers the
    bone that leaves this joint. It may be omitted when the joint has exactly one
    anatomically dominant child; the root joint must state it, because a hip
    happens to have a longer bone than the spine and picking one silently would
    tilt the pelvis capsule.
    """

    joint: str
    radius_m: float
    mass_weight: float
    aim: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.joint, str) or not self.joint.strip():
            raise ValueError("segment.joint must be a non-empty string")
        finite_number(self.radius_m, f"{self.joint}.radius_m")
        finite_number(self.mass_weight, f"{self.joint}.mass_weight")
        if self.radius_m <= 0:
            raise ValueError(f"{self.joint}: radius_m must be positive, got {self.radius_m!r}")
        if self.mass_weight < MIN_MASS_WEIGHT:
            raise ValueError(
                f"{self.joint}: mass_weight must be at least {MIN_MASS_WEIGHT}, "
                f"got {self.mass_weight!r}"
            )
        if self.aim is not None and (not isinstance(self.aim, str) or not self.aim.strip()):
            raise ValueError(f"{self.joint}: aim must be a non-empty joint name or omitted")
        for tag in self.tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(f"{self.joint}: segment tags must be non-empty strings")


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """Proportional-derivative gains for a joint drive."""

    stiffness: float
    damping: float
    max_force: float
    drive_type: str = "force"

    def __post_init__(self) -> None:
        for name in ("stiffness", "damping", "max_force"):
            finite_number(getattr(self, name), f"drive.{name}")
            if getattr(self, name) < 0:
                raise ValueError(f"drive.{name} must be non-negative, got {getattr(self, name)!r}")
        if self.max_force <= 0:
            raise ValueError("drive.max_force must be positive")
        if self.drive_type not in ("force", "acceleration"):
            raise ValueError(
                f"drive.drive_type must be 'force' or 'acceleration', got {self.drive_type!r}"
            )


@dataclass(frozen=True, slots=True)
class JointConfig:
    """How one skeleton joint becomes PhysX joint structure.

    ``rotations`` lists the body axes the joint is allowed to turn about, in
    order. One entry becomes a single revolute joint; more than one becomes a
    short chain of revolute joints with zero-length intermediate links, because a
    USD/PhysX revolute joint carries exactly one degree of freedom. ``limits_deg``
    must then list a ``(low, high)`` pair per rotation.
    """

    joint: str
    dof: str
    rotations: tuple[str, ...]
    limits_deg: tuple[tuple[float, float], ...]
    drive: DriveConfig
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.joint, str) or not self.joint.strip():
            raise ValueError("joint config needs a non-empty joint name")
        if self.dof not in DOF_KINDS:
            raise ValueError(f"{self.joint}: dof must be one of {DOF_KINDS}, got {self.dof!r}")
        if not isinstance(self.drive, DriveConfig):
            raise ValueError(f"{self.joint}: drive must be a DriveConfig")
        if self.dof == "fixed":
            if self.rotations:
                raise ValueError(
                    f"{self.joint}: a fixed joint cannot declare rotations; remove "
                    f"{list(self.rotations)} or set dof: revolute"
                )
            if self.limits_deg:
                raise ValueError(f"{self.joint}: a fixed joint cannot declare limits")
            return
        if not 1 <= len(self.rotations) <= 3:
            raise ValueError(
                f"{self.joint}: a revolute joint needs 1 to 3 rotations, got {len(self.rotations)}"
            )
        if len(set(self.rotations)) != len(self.rotations):
            raise ValueError(f"{self.joint}: duplicate rotation axes {list(self.rotations)}")
        for axis in self.rotations:
            validate_axis(axis)
        if len(self.limits_deg) != len(self.rotations):
            raise ValueError(
                f"{self.joint}: {len(self.rotations)} rotations need "
                f"{len(self.rotations)} limit pairs, got {len(self.limits_deg)}"
            )
        for index, (low, high) in enumerate(self.limits_deg):
            finite_number(low, f"{self.joint}.limits[{index}].low")
            finite_number(high, f"{self.joint}.limits[{index}].high")
            if low >= high:
                raise ValueError(
                    f"{self.joint}: limit pair {index} must have low < high, got {(low, high)}"
                )
            if low < -MAX_LIMIT_DEG or high > MAX_LIMIT_DEG:
                raise ValueError(
                    f"{self.joint}: limit pair {index} {(low, high)} exceeds the "
                    f"+/-{MAX_LIMIT_DEG:g} degree bound; check for a radians/degrees mix-up"
                )
            if high - low > 2 * MAX_LIMIT_DEG:
                raise ValueError(f"{self.joint}: limit pair {index} spans more than a full turn")

    @property
    def dof_count(self) -> int:
        return 0 if self.dof == "fixed" else len(self.rotations)


@dataclass(frozen=True, slots=True)
class RigConfig:
    """Everything needed to turn a rest skeleton into an articulated rig."""

    root_joint: str
    root_mode: str
    collider: str
    self_collisions: bool
    contact_offset_m: float
    rest_offset_m: float
    linear_damping: float
    angular_damping: float
    segments: Mapping[str, SegmentConfig]
    joints: Mapping[str, JointConfig]

    def __post_init__(self) -> None:
        if not isinstance(self.root_joint, str) or not self.root_joint.strip():
            raise ValueError("rig.root_joint must be a non-empty string")
        if self.root_mode not in ROOT_MODES:
            raise ValueError(f"rig.root_mode must be one of {ROOT_MODES}, got {self.root_mode!r}")
        if self.collider != "capsule":
            raise ValueError(
                f"rig.collider must be 'capsule' (the only collider shape this stage "
                f"authors), got {self.collider!r}"
            )
        strict_bool(self.self_collisions, "rig.self_collisions")
        for name in ("contact_offset_m", "rest_offset_m", "linear_damping", "angular_damping"):
            finite_number(getattr(self, name), f"rig.{name}")
        if self.contact_offset_m < 0 or self.rest_offset_m < 0:
            raise ValueError("contact and rest offsets must be non-negative")
        if self.rest_offset_m >= self.contact_offset_m:
            raise ValueError(
                "rig.rest_offset_m must be smaller than rig.contact_offset_m; equal or "
                "larger values make PhysX reject the pair"
            )
        for name in ("linear_damping", "angular_damping"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"rig.{name} must be within [0, 1]")
        if not self.segments:
            raise ValueError("rig.segments must declare at least one segment")
        for key, segment in self.segments.items():
            if key != segment.joint:
                raise ValueError(f"segment key {key!r} does not match its joint {segment.joint!r}")
        for key, joint in self.joints.items():
            if key != joint.joint:
                raise ValueError(f"joint key {key!r} does not match its joint {joint.joint!r}")
        # The world anchor for 'anchored' mode is authored from ``root_mode`` itself (a
        # fixed joint from the world to the pelvis), so the root joint does not need a
        # revolute entry in ``rig.joints``: it has no parent to rotate against.

    def segment(self, joint: str) -> SegmentConfig:
        if joint not in self.segments:
            raise KeyError(f"rig has no segment for joint {joint!r}")
        return self.segments[joint]

    def joint(self, joint: str) -> JointConfig:
        return self.joints.get(
            joint,
            JointConfig(
                joint=joint,
                dof="fixed",
                rotations=(),
                limits_deg=(),
                drive=DriveConfig(stiffness=0.0, damping=0.0, max_force=1.0),
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """How the reference motion is tracked."""

    mode: str
    warmup_seconds: float
    tracking_tolerance_deg: float
    pose_hold_seconds: float
    max_root_linear_velocity_m_s: float

    def __post_init__(self) -> None:
        if self.mode not in CONTROL_MODES:
            raise ValueError(f"control.mode must be one of {CONTROL_MODES}, got {self.mode!r}")
        for name in (
            "warmup_seconds",
            "tracking_tolerance_deg",
            "pose_hold_seconds",
            "max_root_linear_velocity_m_s",
        ):
            finite_number(getattr(self, name), f"control.{name}")
            if getattr(self, name) < 0:
                raise ValueError(f"control.{name} must be non-negative")
        if self.tracking_tolerance_deg <= 0:
            raise ValueError("control.tracking_tolerance_deg must be positive")
        if self.max_root_linear_velocity_m_s <= 0:
            raise ValueError("control.max_root_linear_velocity_m_s must be positive")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Deterministic physics parameters for the human stage."""

    physics_dt_s: float
    solver_position_iterations: int
    solver_velocity_iterations: int
    settle_seconds: float
    gravity_m_s2: float
    stabilization_threshold_s: float

    def __post_init__(self) -> None:
        finite_number(self.physics_dt_s, "simulation.physics_dt_s")
        finite_number(self.settle_seconds, "simulation.settle_seconds")
        finite_number(self.gravity_m_s2, "simulation.gravity_m_s2")
        finite_number(self.stabilization_threshold_s, "simulation.stabilization_threshold_s")
        if self.physics_dt_s <= 0 or self.physics_dt_s > 1.0 / 30.0:
            raise ValueError(
                f"simulation.physics_dt_s must be in (0, 1/30], got {self.physics_dt_s!r}; "
                "a coarser step cannot resolve human contact"
            )
        if self.settle_seconds < 0:
            raise ValueError("simulation.settle_seconds must be non-negative")
        if self.gravity_m_s2 <= 0:
            raise ValueError("simulation.gravity_m_s2 must be positive")
        for name in ("solver_position_iterations", "solver_velocity_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"simulation.{name} must be an integer >= 1, got {value!r}")

    @property
    def gravity(self) -> tuple[float, float, float]:
        return (0.0, 0.0, -self.gravity_m_s2)


@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """A scripted loss of balance, recorded so the trial can be reproduced.

    ``kind`` selects the mechanism: ``force`` applies a constant external force to
    ``body`` for ``duration_s``, ``support_loss`` disables the floor contacts for
    the human for the same window, and ``control_failure`` scales every joint
    drive by ``control_scale`` from ``start_s`` onwards without applying any
    external load. The applied perturbation is *not* the label: labels come from
    :mod:`sim2sense_fall.humans.events`.
    """

    id: str
    kind: str
    body: str
    start_s: float
    duration_s: float
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnitude_n: float = 0.0
    control_scale: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("perturbation.id must be a non-empty string")
        if self.kind not in PERTURBATION_KINDS:
            raise ValueError(
                f"{self.id}: kind must be one of {PERTURBATION_KINDS}, got {self.kind!r}"
            )
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError(f"{self.id}: body must name a joint")
        for name in ("start_s", "duration_s", "magnitude_n", "control_scale"):
            finite_number(getattr(self, name), f"{self.id}.{name}")
        if self.start_s < 0:
            raise ValueError(f"{self.id}: start_s must be non-negative")
        if self.duration_s <= 0:
            raise ValueError(f"{self.id}: duration_s must be positive")
        if self.magnitude_n < 0:
            raise ValueError(f"{self.id}: magnitude_n must be non-negative")
        if not 0.0 <= self.control_scale <= 1.0:
            raise ValueError(f"{self.id}: control_scale must be within [0, 1]")
        length = math.sqrt(sum(component * component for component in self.direction))
        if self.kind == "none":
            if self.magnitude_n != 0.0 or length > 1e-9:
                raise ValueError(
                    f"{self.id}: kind 'none' must not apply a force; leave magnitude_n and "
                    "direction at zero"
                )
            if self.control_scale != 1.0:
                raise ValueError(
                    f"{self.id}: kind 'none' must leave control_scale at 1.0, got "
                    f"{self.control_scale}"
                )
            return
        if self.kind == "force":
            if self.magnitude_n <= 0:
                raise ValueError(
                    f"{self.id}: a force perturbation needs magnitude_n > 0 so the "
                    "recorded push magnitude is meaningful"
                )
            if abs(length - 1.0) > 1e-6:
                raise ValueError(
                    f"{self.id}: direction must be a unit vector for a force "
                    f"perturbation, got norm {length:.6f}"
                )
        elif length > 1e-9 and abs(length - 1.0) > 1e-6:
            raise ValueError(f"{self.id}: direction must be zero or a unit vector")
        if self.kind == "control_failure" and not 0.0 <= self.control_scale < 1.0:
            raise ValueError(
                f"{self.id}: control_failure needs control_scale < 1 (got "
                f"{self.control_scale}); use kind 'none' for an unperturbed trial"
            )


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """The ground-truth contract handed to the wireless stage.

    ``physics_dt_s`` at the simulation rate and ``channel_sample_hz`` at the
    wireless sampling rate are deliberately separate: resampling is explicit and
    its method recorded, rather than being implied by leaving both at one value.
    """

    channel_sample_hz: float
    resample_method: str
    include_contact_forces: bool
    include_surface_points: bool
    surface_points_per_segment: int
    surface_point_seed: int
    group_by: tuple[str, ...]
    record_wireless_interval: bool

    def __post_init__(self) -> None:
        finite_number(self.channel_sample_hz, "export.channel_sample_hz")
        if self.channel_sample_hz <= 0:
            raise ValueError("export.channel_sample_hz must be positive")
        if self.resample_method not in ("linear", "smoothstep", "slerp"):
            raise ValueError(
                "export.resample_method must be 'linear', 'smoothstep' or 'slerp', got "
                f"{self.resample_method!r}"
            )
        strict_bool(self.include_contact_forces, "export.include_contact_forces")
        strict_bool(self.include_surface_points, "export.include_surface_points")
        strict_bool(self.record_wireless_interval, "export.record_wireless_interval")
        if isinstance(self.surface_points_per_segment, bool) or self.surface_points_per_segment < 0:
            raise ValueError("export.surface_points_per_segment must be a non-negative integer")
        if isinstance(self.surface_point_seed, bool) or self.surface_point_seed < 0:
            raise ValueError("export.surface_point_seed must be a non-negative integer")
        if self.include_surface_points and self.surface_points_per_segment < 1:
            raise ValueError("export.include_surface_points needs surface_points_per_segment >= 1")
        if not self.group_by:
            raise ValueError(
                "export.group_by must name at least one field so that train/test splits "
                "cannot leak across subjects or source sequences"
            )


@dataclass(frozen=True, slots=True)
class EventsConfig:
    """Pre-registered definitions of fall onset, impact and stabilisation.

    These thresholds are project-chosen heuristics, fixed here **before** any trial
    is labelled, and applied identically to every trial. They are not validated
    against labelled real human falls, and no result produced with them may be
    described as clinically or physiologically validated.
    """

    trunk_angle_deg: float
    trunk_onset_fraction: float
    pelvis_height_fraction: float
    min_low_frames: int
    max_transition_s: float
    impact_height_m: float
    impact_speed_m_s: float
    settle_window_s: float
    settle_speed_m_s: float
    controlled_descent_speed_m_s: float
    divergence_limit_m: float
    penetration_limit_m: float
    recovery_height_fraction: float

    def __post_init__(self) -> None:
        for name in (
            "trunk_angle_deg",
            "trunk_onset_fraction",
            "pelvis_height_fraction",
            "max_transition_s",
            "impact_height_m",
            "impact_speed_m_s",
            "settle_window_s",
            "settle_speed_m_s",
            "controlled_descent_speed_m_s",
            "divergence_limit_m",
            "penetration_limit_m",
            "recovery_height_fraction",
        ):
            finite_number(getattr(self, name), f"events.{name}")
        if not 0.0 < self.trunk_angle_deg < 90.0:
            raise ValueError(
                f"events.trunk_angle_deg must be within (0, 90), got {self.trunk_angle_deg}"
            )
        if not 0.0 < self.trunk_onset_fraction < 1.0:
            raise ValueError(
                "events.trunk_onset_fraction must be within (0, 1) so that tipping is "
                f"detected strictly before the fall threshold, got {self.trunk_onset_fraction}"
            )
        if not 0.0 < self.pelvis_height_fraction < 1.0:
            raise ValueError(
                f"events.pelvis_height_fraction must be within (0, 1), got "
                f"{self.pelvis_height_fraction}"
            )
        if not 0.0 < self.recovery_height_fraction < 1.0:
            raise ValueError(
                "events.recovery_height_fraction must be within (0, 1), got "
                f"{self.recovery_height_fraction}"
            )
        for name in ("max_transition_s", "settle_window_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"events.{name} must be positive")
        for name in ("impact_height_m", "impact_speed_m_s", "settle_speed_m_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"events.{name} must be non-negative")
        if isinstance(self.min_low_frames, bool) or self.min_low_frames < 1:
            raise ValueError("events.min_low_frames must be an integer >= 1")
        if self.divergence_limit_m <= 0:
            raise ValueError("events.divergence_limit_m must be positive")


@dataclass(frozen=True, slots=True)
class HumanConfig:
    """The whole human-stage configuration document."""

    human_id: str
    description: str
    skeleton: SkeletonConfig
    rig: RigConfig
    control: ControlConfig
    simulation: SimulationConfig
    perturbations: tuple[PerturbationConfig, ...]
    export: ExportConfig
    events: EventsConfig
    topology: SkeletonTopology = field(default_factory=smpl_skeleton)
    source_path: Path | None = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.human_id, str) or not self.human_id.strip():
            raise ValueError("human_id must be a non-empty string")
        for name in ("skeleton", "rig", "control", "simulation", "export", "events"):
            if getattr(self, name) is None:
                raise ValueError(f"human config is missing the {name!r} section")
        identifiers = [entry.id for entry in self.perturbations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate perturbation ids {sorted(identifiers)}")
        self.validate_against_topology()
        for name, value in self.visualization.default_pose_rad:
            if name not in self.rig.joints or self.rig.joints[name].dof != "revolute":
                raise ValueError(
                    f"visualization pose names unknown or non-revolute DOF {name!r}"
                )
            joint = self.rig.joints[name]
            low_deg, high_deg = joint.limits_deg[0]
            low_rad, high_rad = math.radians(low_deg), math.radians(high_deg)
            if not low_rad <= value <= high_rad:
                raise ValueError(
                    f"visualization pose {name!r}={value} rad is outside "
                    f"its configured limits [{low_rad}, {high_rad}] rad"
                )

    def validate_against_topology(self) -> None:
        """Cross-check the rig against the skeleton it claims to articulate."""

        topology = self.topology
        names = set(topology.joint_names)
        if self.rig.root_joint != topology.root_name:
            raise ValueError(
                f"rig.root_joint is {self.rig.root_joint!r} but the {topology.name} topology "
                f"roots at {topology.root_name!r}"
            )
        missing_segments = sorted(names - set(self.rig.segments))
        if missing_segments:
            raise ValueError(
                f"rig.segments is missing {missing_segments}; every joint needs a capsule "
                "so the body has no un-collided limb"
            )
        extra_segments = sorted(set(self.rig.segments) - names)
        if extra_segments:
            raise ValueError(f"rig.segments names unknown joints {extra_segments}")
        extra_joints = sorted(set(self.rig.joints) - names)
        if extra_joints:
            raise ValueError(f"rig.joints names unknown joints {extra_joints}")
        for joint_name, joint in self.rig.joints.items():
            parent = topology.parent_of(joint_name)
            if parent is None:
                raise ValueError(
                    f"{joint_name}: the root joint has no parent to rotate against; it is "
                    "driven by the free root body instead"
                )
            if self.rig.segment(joint_name).tags and joint.tags:
                overlap = set(self.rig.segment(joint_name).tags) & set(joint.tags)
                if overlap:
                    raise ValueError(f"{joint_name}: tags duplicated across segment and joint")
        for joint_name, segment in self.rig.segments.items():
            if segment.aim is None:
                if topology.parent_of(joint_name) is None:
                    raise ValueError(
                        f"segment {joint_name!r} is the root and must declare 'aim', "
                        "because its longest child bone is a hip rather than the spine"
                    )
                continue
            if segment.aim not in topology.children_of(joint_name):
                raise ValueError(
                    f"segment {joint_name!r}: aim {segment.aim!r} is not a child of "
                    f"{joint_name!r} (children: {list(topology.children_of(joint_name))})"
                )
        for perturbation in self.perturbations:
            if perturbation.body not in names:
                raise ValueError(
                    f"{perturbation.id}: body {perturbation.body!r} is not a joint of the "
                    f"{topology.name} topology"
                )

    def perturbation(self, perturbation_id: str) -> PerturbationConfig:
        for entry in self.perturbations:
            if entry.id == perturbation_id:
                return entry
        raise KeyError(
            f"unknown perturbation {perturbation_id!r}; known: "
            f"{[entry.id for entry in self.perturbations]}"
        )


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_SEGMENT_FIELDS = {"joint", "radius_m", "mass_weight", "aim", "tags"}
_JOINT_FIELDS = {"joint", "dof", "rotations", "limits_deg", "drive", "tags"}
_DRIVE_FIELDS = {"stiffness", "damping", "max_force", "drive_type"}


def _segment_from_mapping(key: str, payload: Any) -> SegmentConfig:
    mapping = _require_mapping(payload, where=f"segment {key}")
    _reject_unknown(mapping, _SEGMENT_FIELDS | {"name"}, where=f"segment {key}")
    joint = str(mapping.get("joint", mapping.get("name", key)))
    tags_payload = mapping.get("tags", [])
    if not isinstance(tags_payload, list):
        raise ValueError(f"segment {key}: 'tags' must be a list")
    aim = mapping.get("aim")
    if aim is not None and (not isinstance(aim, str) or not aim.strip()):
        raise ValueError(f"segment {key}: 'aim' must be a joint name or omitted")
    return SegmentConfig(
        joint=joint,
        radius_m=_as_float(mapping, "radius_m"),
        mass_weight=_as_float(mapping, "mass_weight"),
        aim=None if aim is None else str(aim).strip(),
        tags=tuple(str(tag) for tag in tags_payload),
    )


def _joint_from_mapping(key: str, payload: Any) -> JointConfig:
    mapping = _require_mapping(payload, where=f"joint {key}")
    _reject_unknown(mapping, _JOINT_FIELDS | {"name"}, where=f"joint {key}")
    joint = str(mapping.get("joint", mapping.get("name", key)))
    dof = str(mapping.get("dof", "fixed")).strip().lower()
    rotations_payload = mapping.get("rotations", [])
    if not isinstance(rotations_payload, list):
        raise ValueError(f"joint {key}: 'rotations' must be a list of axis tokens")
    rotations = tuple(validate_axis(str(entry)) for entry in rotations_payload)
    limits_payload = mapping.get("limits_deg", [])
    if not isinstance(limits_payload, list):
        raise ValueError(f"joint {key}: 'limits_deg' must be a list of [low, high] pairs")
    limits: list[tuple[float, float]] = []
    for entry in limits_payload:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"joint {key}: each limit must be a [low, high] pair, got {entry!r}")
        limits.append(
            (
                finite_number(entry[0], f"{key}.limit.low"),
                finite_number(entry[1], f"{key}.limit.high"),
            )
        )
    drive_payload = mapping.get("drive", {})
    drive_mapping = _require_mapping(drive_payload, where=f"joint {key} drive")
    _reject_unknown(drive_mapping, _DRIVE_FIELDS, where=f"joint {key} drive")
    drive = DriveConfig(
        stiffness=_as_float(drive_mapping, "stiffness", default=0.0),
        damping=_as_float(drive_mapping, "damping", default=0.0),
        max_force=_as_float(drive_mapping, "max_force", default=1.0),
        drive_type=_as_str(drive_mapping, "drive_type", default="force"),
    )
    tags_payload = mapping.get("tags", [])
    if not isinstance(tags_payload, list):
        raise ValueError(f"joint {key}: 'tags' must be a list")
    return JointConfig(
        joint=joint,
        dof=dof,
        rotations=rotations,
        limits_deg=tuple(limits),
        drive=drive,
        tags=tuple(str(tag) for tag in tags_payload),
    )


def _perturbation_from_mapping(payload: Any) -> PerturbationConfig:
    mapping = _require_mapping(payload, where="perturbation")
    _reject_unknown(
        mapping,
        {
            "id",
            "kind",
            "body",
            "start_s",
            "duration_s",
            "direction",
            "magnitude_n",
            "control_scale",
            "notes",
        },
        where="perturbation",
    )
    direction = (0.0, 0.0, 0.0)
    if "direction" in mapping:
        direction = _as_triple(mapping, "direction")
    return PerturbationConfig(
        id=_as_str(mapping, "id"),
        kind=_as_str(mapping, "kind"),
        body=_as_str(mapping, "body"),
        start_s=_as_float(mapping, "start_s"),
        duration_s=_as_float(mapping, "duration_s"),
        direction=direction,
        magnitude_n=_as_float(mapping, "magnitude_n", default=0.0),
        control_scale=_as_float(mapping, "control_scale", default=1.0),
        notes=str(mapping.get("notes", "")),
    )


def human_config_from_mapping(
    payload: Mapping[str, Any], *, source_path: Path | None = None
) -> HumanConfig:
    """Build a validated :class:`HumanConfig` from a parsed YAML mapping."""

    if not isinstance(payload, Mapping):
        raise ValueError("human config must be a mapping")
    _reject_unknown(
        payload,
        {
            "human_id",
            "description",
            "skeleton",
            "rig",
            "control",
            "simulation",
            "perturbations",
            "export",
            "events",
            "visualization",
        },
        where="human config",
    )
    skeleton_payload = _require_mapping(payload.get("skeleton"), where="skeleton")
    _reject_unknown(
        skeleton_payload,
        {"topology", "model_asset", "allow_procedural_skeleton", "height_m", "mass_kg", "note"},
        where="skeleton",
    )
    model_asset = skeleton_payload.get("model_asset")
    if model_asset is not None and (not isinstance(model_asset, str) or not model_asset.strip()):
        raise ValueError("skeleton.model_asset must be a non-empty string or omitted")
    skeleton = SkeletonConfig(
        topology=_as_str(skeleton_payload, "topology", default="smpl"),
        model_asset=None if model_asset is None else str(model_asset).strip(),
        allow_procedural_skeleton=_as_bool(
            skeleton_payload, "allow_procedural_skeleton", default=False
        ),
        height_m=_as_float(skeleton_payload, "height_m", default=1.7),
        mass_kg=_as_float(skeleton_payload, "mass_kg", default=72.0),
        note=str(skeleton_payload.get("note", "")),
    )

    rig_payload = _require_mapping(payload.get("rig"), where="rig")
    _reject_unknown(
        rig_payload,
        {
            "root_joint",
            "root_mode",
            "collider",
            "self_collisions",
            "contact_offset_m",
            "rest_offset_m",
            "linear_damping",
            "angular_damping",
            "segments",
            "joints",
        },
        where="rig",
    )
    segments_payload = _require_mapping(rig_payload.get("segments"), where="rig.segments")
    joints_payload = _require_mapping(rig_payload.get("joints", {}), where="rig.joints")
    segments = {
        str(key): _segment_from_mapping(str(key), value) for key, value in segments_payload.items()
    }
    joints = {
        str(key): _joint_from_mapping(str(key), value) for key, value in joints_payload.items()
    }
    rig = RigConfig(
        root_joint=_as_str(rig_payload, "root_joint"),
        root_mode=_as_str(rig_payload, "root_mode", default="free"),
        collider=_as_str(rig_payload, "collider", default="capsule"),
        self_collisions=_as_bool(rig_payload, "self_collisions", default=False),
        contact_offset_m=_as_float(rig_payload, "contact_offset_m", default=0.01),
        rest_offset_m=_as_float(rig_payload, "rest_offset_m", default=0.0),
        linear_damping=_as_float(rig_payload, "linear_damping", default=0.0),
        angular_damping=_as_float(rig_payload, "angular_damping", default=0.0),
        segments=segments,
        joints=joints,
    )

    control_payload = _require_mapping(payload.get("control", {}), where="control")
    _reject_unknown(
        control_payload,
        {
            "mode",
            "warmup_seconds",
            "tracking_tolerance_deg",
            "pose_hold_seconds",
            "max_root_linear_velocity_m_s",
        },
        where="control",
    )
    control = ControlConfig(
        mode=_as_str(control_payload, "mode", default="pd"),
        warmup_seconds=_as_float(control_payload, "warmup_seconds", default=0.3),
        tracking_tolerance_deg=_as_float(control_payload, "tracking_tolerance_deg", default=15.0),
        pose_hold_seconds=_as_float(control_payload, "pose_hold_seconds", default=0.2),
        max_root_linear_velocity_m_s=_as_float(
            control_payload, "max_root_linear_velocity_m_s", default=3.0
        ),
    )

    simulation_payload = _require_mapping(payload.get("simulation", {}), where="simulation")
    _reject_unknown(
        simulation_payload,
        {
            "physics_dt_s",
            "solver_position_iterations",
            "solver_velocity_iterations",
            "settle_seconds",
            "gravity_m_s2",
            "stabilization_threshold_s",
        },
        where="simulation",
    )
    simulation = SimulationConfig(
        physics_dt_s=_as_float(simulation_payload, "physics_dt_s", default=1.0 / 120.0),
        solver_position_iterations=_as_int(
            simulation_payload, "solver_position_iterations", default=16
        ),
        solver_velocity_iterations=_as_int(
            simulation_payload, "solver_velocity_iterations", default=4
        ),
        settle_seconds=_as_float(simulation_payload, "settle_seconds", default=0.5),
        gravity_m_s2=_as_float(simulation_payload, "gravity_m_s2", default=9.81),
        stabilization_threshold_s=_as_float(
            simulation_payload, "stabilization_threshold_s", default=0.001
        ),
    )

    perturbations_payload = payload.get("perturbations", [])
    if not isinstance(perturbations_payload, list):
        raise ValueError("'perturbations' must be a list")
    perturbations = tuple(_perturbation_from_mapping(entry) for entry in perturbations_payload)

    export_payload = _require_mapping(payload.get("export", {}), where="export")
    _reject_unknown(
        export_payload,
        {
            "channel_sample_hz",
            "resample_method",
            "include_contact_forces",
            "include_surface_points",
            "surface_points_per_segment",
            "surface_point_seed",
            "group_by",
            "record_wireless_interval",
        },
        where="export",
    )
    group_by_payload = export_payload.get("group_by", ["subject", "sequence"])
    if not isinstance(group_by_payload, list) or not group_by_payload:
        raise ValueError("export.group_by must be a non-empty list")
    export = ExportConfig(
        channel_sample_hz=_as_float(export_payload, "channel_sample_hz", default=50.0),
        resample_method=_as_str(export_payload, "resample_method", default="slerp"),
        include_contact_forces=_as_bool(export_payload, "include_contact_forces", default=True),
        include_surface_points=_as_bool(export_payload, "include_surface_points", default=True),
        surface_points_per_segment=_as_int(export_payload, "surface_points_per_segment", default=8),
        surface_point_seed=_as_int(export_payload, "surface_point_seed", default=20260922),
        group_by=tuple(str(entry) for entry in group_by_payload),
        record_wireless_interval=_as_bool(export_payload, "record_wireless_interval", default=True),
    )

    events_payload = _require_mapping(payload.get("events", {}), where="events")
    _reject_unknown(
        events_payload,
        {
            "trunk_angle_deg",
            "trunk_onset_fraction",
            "pelvis_height_fraction",
            "min_low_frames",
            "max_transition_s",
            "impact_height_m",
            "impact_speed_m_s",
            "settle_window_s",
            "settle_speed_m_s",
            "controlled_descent_speed_m_s",
            "divergence_limit_m",
            "penetration_limit_m",
            "recovery_height_fraction",
        },
        where="events",
    )
    events = EventsConfig(
        trunk_angle_deg=_as_float(events_payload, "trunk_angle_deg", default=60.0),
        trunk_onset_fraction=_as_float(events_payload, "trunk_onset_fraction", default=0.5),
        pelvis_height_fraction=_as_float(events_payload, "pelvis_height_fraction", default=0.55),
        min_low_frames=_as_int(events_payload, "min_low_frames", default=10),
        max_transition_s=_as_float(events_payload, "max_transition_s", default=1.5),
        impact_height_m=_as_float(events_payload, "impact_height_m", default=0.25),
        impact_speed_m_s=_as_float(events_payload, "impact_speed_m_s", default=1.0),
        settle_window_s=_as_float(events_payload, "settle_window_s", default=0.5),
        settle_speed_m_s=_as_float(events_payload, "settle_speed_m_s", default=0.15),
        controlled_descent_speed_m_s=_as_float(
            events_payload, "controlled_descent_speed_m_s", default=1.2
        ),
        divergence_limit_m=_as_float(events_payload, "divergence_limit_m", default=5.0),
        penetration_limit_m=_as_float(events_payload, "penetration_limit_m", default=-0.05),
        recovery_height_fraction=_as_float(events_payload, "recovery_height_fraction", default=0.6),
    )

    visualization_payload = _require_mapping(
        payload.get("visualization", {}), where="visualization"
    )
    _reject_unknown(visualization_payload, {"default_pose_rad"}, where="visualization")
    pose_payload = visualization_payload.get("default_pose_rad", {})
    if not isinstance(pose_payload, Mapping):
        raise ValueError("visualization.default_pose_rad must be a mapping of DOF to radians")
    visualization = VisualizationConfig(
        default_pose_rad=tuple(
            (str(name), finite_number(value, f"visualization.default_pose_rad.{name}"))
            for name, value in pose_payload.items()
        )
    )

    return HumanConfig(
        human_id=_as_str(payload, "human_id"),
        description=str(payload.get("description", "")),
        skeleton=skeleton,
        rig=rig,
        control=control,
        simulation=simulation,
        perturbations=perturbations,
        export=export,
        events=events,
        visualization=visualization,
        topology=smpl_skeleton(),
        source_path=source_path,
    )


def load_human_config(path: str | Path) -> HumanConfig:
    """Load and strictly validate a human configuration YAML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"human config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{config_path}: top level must be a mapping")
    return human_config_from_mapping(payload, source_path=config_path)


def sequence_to_tuple(values: Sequence[float]) -> tuple[float, ...]:
    """Small helper used by tests to normalise literal vectors."""

    return tuple(finite_number(value, "value") for value in values)
