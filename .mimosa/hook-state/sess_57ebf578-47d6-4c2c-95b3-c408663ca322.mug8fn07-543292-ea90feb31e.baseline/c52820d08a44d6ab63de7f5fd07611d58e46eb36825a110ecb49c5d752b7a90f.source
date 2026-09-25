"""Export one trial as unified-timeline ground truth for the wireless stage.

What a trial artefact contains
------------------------------

* **Two clocks, both recorded.** ``time_physics_s`` samples the simulation step
  and ``time_channel_s`` samples the wireless rate. They are stored side by side
  with the resampling method and both rates, so a downstream consumer never has to
  guess how one was derived from the other.
* **Body state that follows the physics**, not the reference: root pose, joint
  positions and velocities, and a world-space body-point cloud carried by the
  links' simulated poses.
* **Contact evidence**: per-window total contact force magnitude when the runtime
  can report it, and the resolved contact-event times from the labeller.
* **Provenance**: scene, model, motion source, controller, body shape, seed and the
  code paths that produced the file, plus the SHA-256 of the rig plan and of the
  human configuration so a run can be matched back to its inputs.
* **Split keys**: the fields listed in ``export.group_by`` are emitted as one
  namespaced key PER FIELD, so a train/test split groups on all of them at once. A single
  joined ``subject|sequence`` key is not enough: it makes ``person1|clip1`` and
  ``person1|clip2`` two different groups that can land on opposite sides, which is how the
  same human leaks from training into test.
  key and cannot leak a subject or a source sequence across the boundary.

Every array is written with ``allow_pickle=False`` and NaN is never written, so a
corrupt run fails loudly instead of producing a file that silently poisons training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .events import TrialLabel
from .mesh_sequence import MeshSequence, MeshTopology
from .rig import HumanRigPlan
from .rotations import (
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    rotation_matrix_error,
)

__all__ = [
    "CHANNEL_TIME_KEY",
    "PHYSICS_TIME_KEY",
    "GroundTruth",
    "TrialProvenance",
    "resample_series",
    "clip_group_keys",
    "connected_groups",
    "split_violations",
]

LOGGER = logging.getLogger(__name__)

PHYSICS_TIME_KEY = "time_physics_s"
CHANNEL_TIME_KEY = "time_channel_s"

#: Provenance fields that must be present for a trial to be traceable.
_REQUIRED_PROVENANCE_FIELDS = (
    "scene_id",
    "scene_sha256",
    "human_id",
    "rig_plan_sha256",
    "config_sha256",
    "motion_id",
    "motion_kind",
    "controller_mode",
    "root_mode",
    "body_representation",
    "seed",
    "physics_dt_s",
    "channel_sample_hz",
    "resample_method",
    "standing_height_m",
    "total_mass_kg",
    "generated_by",
)


@dataclass(frozen=True, slots=True)
class TrialProvenance:
    """Reproducibility block for one exported trial."""

    scene_id: str
    scene_sha256: str
    human_id: str
    rig_plan_sha256: str
    config_sha256: str
    motion_id: str
    motion_kind: str
    motion_sha256: str | None
    controller_mode: str
    root_mode: str
    root_anchor_used: bool
    body_representation: str
    perturbation_id: str
    perturbation_detail: Mapping[str, Any]
    seed: int
    physics_dt_s: float
    channel_sample_hz: float
    resample_method: str
    standing_height_m: float
    total_mass_kg: float
    generated_by: str
    subject: str
    sequence: str
    model_asset_id: str | None = None
    model_asset_sha256: str | None = None
    model_asset_available: bool = False
    license_notes: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in _REQUIRED_PROVENANCE_FIELDS:
            value = getattr(self, name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"trial provenance is missing {name!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("trial provenance seed must be a non-negative integer")
        for name in ("physics_dt_s", "channel_sample_hz", "standing_height_m", "total_mass_kg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"trial provenance {name} must be finite and positive")
        if self.root_anchor_used and self.root_mode != "anchored":
            raise ValueError(
                "root_anchor_used is set but root_mode is not 'anchored'; a hidden root "
                "constraint must be visible in the provenance"
            )
        if self.body_representation not in (
            "smpl_skin_mesh",
            "capsule_proxy_surface",
        ):
            raise ValueError(
                "body_representation must be 'smpl_skin_mesh' or 'capsule_proxy_surface', "
                f"got {self.body_representation!r}"
            )
        for name in ("rig_plan_sha256", "config_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise ValueError(f"trial provenance {name} must be a 64-character digest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_sha256": self.scene_sha256,
            "human_id": self.human_id,
            "rig_plan_sha256": self.rig_plan_sha256,
            "config_sha256": self.config_sha256,
            "motion_id": self.motion_id,
            "motion_kind": self.motion_kind,
            "motion_sha256": self.motion_sha256,
            "controller_mode": self.controller_mode,
            "root_mode": self.root_mode,
            "root_anchor_used": self.root_anchor_used,
            "body_representation": self.body_representation,
            "perturbation_id": self.perturbation_id,
            "perturbation_detail": dict(self.perturbation_detail),
            "seed": self.seed,
            "physics_dt_s": self.physics_dt_s,
            "channel_sample_hz": self.channel_sample_hz,
            "resample_method": self.resample_method,
            "standing_height_m": self.standing_height_m,
            "total_mass_kg": self.total_mass_kg,
            "generated_by": self.generated_by,
            "subject": self.subject,
            "sequence": self.sequence,
            "model_asset_id": self.model_asset_id,
            "model_asset_sha256": self.model_asset_sha256,
            "model_asset_available": self.model_asset_available,
            "license_notes": self.license_notes,
            "notes": self.notes,
        }


def clip_group_keys(
    fields: Mapping[str, str], *, isolate: Sequence[str], dataset: str | None = None
) -> tuple[str, ...]:
    """Every key a trial must be kept together on, one per isolation field.

    Each key is namespaced with the dataset because two sources can both name a performer
    ``person_01``; without the prefix, one subject would be merged with an unrelated
    identically-named one and the grouping would silently over-merge.
    """

    if not isolate:
        raise ValueError("isolate must name at least one field")
    namespace = str(dataset or fields.get("dataset") or fields.get("source_id") or "unspecified")
    missing = [name for name in isolate if name not in fields]
    if missing:
        raise ValueError(
            f"isolate names unavailable fields {missing}; available {sorted(fields)}"
        )
    return tuple(f"{name}={namespace}:{fields[name]}" for name in isolate)


def connected_groups(keys_by_record: Mapping[str, Sequence[str]]) -> list[frozenset[str]]:
    """Group record ids so that any two sharing ANY isolation key end up together.

    Union-find over the keys rather than over a joined key: a trial that carries both
    ``subject=...`` and ``sequence=...`` joins its whole subject group and its whole clip
    group, which is the only way to isolate several related identifiers at once.
    """

    parent: dict[str, str] = {record: record for record in keys_by_record}

    def find(record: str) -> str:
        while parent[record] != record:
            parent[record] = parent[parent[record]]
            record = parent[record]
        return record

    def union(first: str, second: str) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    owner: dict[str, str] = {}
    for record, keys in keys_by_record.items():
        for key in keys:
            if key in owner:
                union(owner[key], record)
            else:
                owner[key] = record
    groups: dict[str, set[str]] = {}
    for record in keys_by_record:
        groups.setdefault(find(record), set()).add(record)
    return [frozenset(members) for members in groups.values()]


def split_violations(
    groups: Sequence[frozenset[str]], side_of: Mapping[str, str]
) -> list[tuple[frozenset[str], tuple[str, ...]]]:
    """Return the groups whose members were assigned to more than one split side.

    Checking the final sets, rather than trusting the grouping code, is what catches a
    split built from the wrong key: an empty list here is the actual evidence of isolation.
    """

    violations: list[tuple[frozenset[str], tuple[str, ...]]] = []
    for group in groups:
        sides = tuple(sorted({side_of[record] for record in group if record in side_of}))
        if len(sides) > 1:
            violations.append((group, sides))
    return violations


def _slerp_quaternions(
    quaternions: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    output = np.empty((len(lower), 4), dtype=np.float64)
    for row in range(len(lower)):
        first = quaternions[lower[row]]
        second = quaternions[upper[row]]
        dot = float(first @ second)
        if dot < 0.0:
            second = -second
            dot = -dot
        blend = float(weight[row])
        if dot > 1.0 - 1e-9:
            combined = first + blend * (second - first)
        else:
            theta = math.acos(max(-1.0, min(1.0, dot)))
            sine = math.sin(theta)
            combined = (
                math.sin((1.0 - blend) * theta) / sine * first
                + math.sin(blend * theta) / sine * second
            )
        norm = float(np.linalg.norm(combined))
        output[row] = combined / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    return output


def resample_series(
    source_times: np.ndarray,
    target_times: np.ndarray,
    values: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """Resample a per-frame series onto ``target_times``.

    ``quaternion`` and ``slerp`` interpolate on the rotation manifold; ``linear``
    and ``smoothstep`` interpolate components. Interpolation is exact for small
    per-step deltas, and any rate change is recorded in the export metadata rather
    than being silent.
    """

    if method not in ("linear", "smoothstep", "slerp", "quaternion"):
        raise ValueError(
            "resample method must be 'linear', 'smoothstep', 'slerp' or 'quaternion', "
            f"got {method!r}"
        )
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("resample times must be one-dimensional")
    if array.shape[0] != len(source_times):
        raise ValueError(
            f"values first dimension {array.shape[0]} does not match {len(source_times)} times"
        )
    if len(source_times) < 2:
        raise ValueError("resampling needs at least two source samples")
    if np.any(np.diff(source_times) <= 0) or np.any(np.diff(target_times) <= 0):
        raise ValueError("resample times must strictly increase")
    upper = np.clip(
        np.searchsorted(source_times, target_times, side="left"), 1, len(source_times) - 1
    )
    lower = upper - 1
    span = source_times[upper] - source_times[lower]
    weight = np.clip((target_times - source_times[lower]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    if method == "smoothstep":
        weight = weight * weight * (3.0 - 2.0 * weight)
    if method in ("slerp", "quaternion"):
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("quaternion resampling needs a (N, 4) array")
        return _slerp_quaternions(array, lower, upper, weight)
    expanded = weight
    while expanded.ndim < array.ndim:
        expanded = expanded[..., None]
    return array[lower] + (array[upper] - array[lower]) * expanded


@dataclass(frozen=True, slots=True, eq=False)
class GroundTruth:
    """One trial, ready to write: two clocks, body state, events and provenance."""

    provenance: TrialProvenance
    label: TrialLabel
    time_physics_s: np.ndarray
    time_channel_s: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_positions_rad: np.ndarray
    joint_velocities_rad_s: np.ndarray
    joint_names: tuple[str, ...]
    link_positions: np.ndarray
    link_names: tuple[str, ...]
    body_points: np.ndarray
    body_point_owners: tuple[str, ...]
    phase_labels: tuple[str, ...]
    mesh_vertices_xyz: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    mesh_representation: str | None = None
    mesh_vertex_owners: tuple[str, ...] = ()
    reference_joint_positions_rad: np.ndarray | None = None
    contact_force_n: np.ndarray | None = None
    channel_root_position: np.ndarray | None = None
    channel_root_quaternion: np.ndarray | None = None
    channel_joint_positions_rad: np.ndarray | None = None
    channel_body_points: np.ndarray | None = None
    channel_mesh_vertices_xyz: np.ndarray | None = None
    rig_plan: HumanRigPlan | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = int(np.shape(self.time_physics_s)[0])
        for name, array, shape in (
            ("time_physics_s", self.time_physics_s, (frames,)),
            ("root_position", self.root_position, (frames, 3)),
            ("root_quaternion", self.root_quaternion, (frames, 4)),
            (
                "joint_positions_rad",
                self.joint_positions_rad,
                (frames, len(self.joint_names)),
            ),
            (
                "joint_velocities_rad_s",
                self.joint_velocities_rad_s,
                (frames, len(self.joint_names)),
            ),
        ):
            actual = np.shape(array)
            if actual != shape:
                raise ValueError(f"{name} must be {shape}, got {actual}")
        if self.link_positions.ndim != 3 or self.link_positions.shape != (
            frames,
            len(self.link_names),
            3,
        ):
            raise ValueError(
                f"link_positions must be ({frames}, {len(self.link_names)}, 3), got "
                f"{self.link_positions.shape}"
            )
        if self.body_points.ndim != 3 or self.body_points.shape[0] != frames:
            raise ValueError(f"body_points must be (N, P, 3), got {self.body_points.shape}")
        if len(self.body_point_owners) != self.body_points.shape[1]:
            raise ValueError(
                f"body_point_owners has {len(self.body_point_owners)} entries for "
                f"{self.body_points.shape[1]} points"
            )
        if len(self.phase_labels) not in (0, frames):
            raise ValueError(
                f"phase_labels has {len(self.phase_labels)} entries for {frames} frames"
            )
        if (self.mesh_vertices_xyz is None) != (self.mesh_faces is None):
            raise ValueError("mesh_vertices_xyz and mesh_faces must be supplied together")
        if self.mesh_vertices_xyz is not None and self.mesh_faces is not None:
            representation = self.mesh_representation or {
                "capsule_proxy_surface": "capsule_proxy_mesh",
                "smpl_skin_mesh": "smpl_skin_mesh",
            }.get(self.provenance.body_representation, self.provenance.body_representation)
            sequence = MeshSequence(
                self.time_physics_s,
                self.mesh_vertices_xyz,
                MeshTopology(self.mesh_faces),
                representation=representation,
                vertex_owners=self.mesh_vertex_owners,
            )
            object.__setattr__(self, "mesh_vertices_xyz", sequence.vertices_xyz)
            object.__setattr__(self, "mesh_faces", sequence.faces)
            object.__setattr__(self, "mesh_representation", sequence.representation)
            if not self.mesh_vertex_owners:
                object.__setattr__(self, "mesh_vertex_owners", sequence.vertex_owners)
        elif self.mesh_representation is not None:
            raise ValueError("mesh_representation requires mesh vertices and faces")
        if self.contact_force_n is not None and np.shape(self.contact_force_n) != (frames,):
            raise ValueError("contact_force_n must be one value per physics frame")
        if self.reference_joint_positions_rad is not None and np.shape(
            self.reference_joint_positions_rad
        ) != (frames, len(self.joint_names)):
            raise ValueError("reference_joint_positions_rad must match the joint state shape")
        channel_frames = int(np.shape(self.time_channel_s)[0])
        for name, array in (
            ("channel_root_position", self.channel_root_position),
            ("channel_root_quaternion", self.channel_root_quaternion),
            ("channel_joint_positions_rad", self.channel_joint_positions_rad),
            ("channel_body_points", self.channel_body_points),
            ("channel_mesh_vertices_xyz", self.channel_mesh_vertices_xyz),
        ):
            if array is not None and np.shape(array)[0] != channel_frames:
                raise ValueError(
                    f"{name} has {np.shape(array)[0]} frames but time_channel_s has "
                    f"{channel_frames}"
                )
        for name, array in (
            ("time_physics_s", self.time_physics_s),
            ("root_position", self.root_position),
            ("root_quaternion", self.root_quaternion),
            ("joint_positions_rad", self.joint_positions_rad),
            ("joint_velocities_rad_s", self.joint_velocities_rad_s),
            ("link_positions", self.link_positions),
            ("body_points", self.body_points),
            ("mesh_vertices_xyz", self.mesh_vertices_xyz),
            ("mesh_faces", self.mesh_faces),
            ("channel_mesh_vertices_xyz", self.channel_mesh_vertices_xyz),
        ):
            if array is not None and not np.all(np.isfinite(array)):
                raise ValueError(f"ground truth {name} contains a non-finite value before writing")
        if self.mesh_vertices_xyz is not None and self.channel_mesh_vertices_xyz is not None:
            if np.shape(self.channel_mesh_vertices_xyz)[1:] != np.shape(self.mesh_vertices_xyz)[1:]:
                raise ValueError("channel mesh vertices must preserve the physics mesh shape")

    @property
    def frame_count(self) -> int:
        return int(np.shape(self.time_physics_s)[0])

    def quaternion_errors(self) -> np.ndarray:
        """Angle between each stored root quaternion and its nearest rotation matrix."""

        return np.array(
            [rotation_matrix_error(quaternion_to_matrix(row)) for row in self.root_quaternion]
        )

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provenance": self.provenance.as_dict(),
            "label": self.label.as_dict(),
            "frame_count": self.frame_count,
            "channel_frame_count": int(np.shape(self.time_channel_s)[0]),
            "joint_names": list(self.joint_names),
            "link_names": list(self.link_names),
            "body_point_count": int(self.body_points.shape[1]),
            "mesh_representation": self.mesh_representation,
            "mesh_vertex_count": None
            if self.mesh_vertices_xyz is None
            else int(self.mesh_vertices_xyz.shape[1]),
            "mesh_face_count": None if self.mesh_faces is None else int(self.mesh_faces.shape[0]),
            "mesh_topology_hash": None
            if self.mesh_faces is None
            else MeshTopology(self.mesh_faces).sha256,
            "mesh_coordinate_system": (
                "world_z_up_xyz" if self.mesh_vertices_xyz is not None else None
            ),
            "mesh_units": "m" if self.mesh_vertices_xyz is not None else None,
            "mesh_is_fixed_topology": self.mesh_vertices_xyz is not None,
            "has_channel_mesh": self.channel_mesh_vertices_xyz is not None,
            "phase_labels": list(self.phase_labels),
            "has_contact_forces": self.contact_force_n is not None,
            "has_reference_joints": self.reference_joint_positions_rad is not None,
            "max_root_quaternion_error": float(self.quaternion_errors().max())
            if self.frame_count
            else 0.0,
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["root_position"] = self.root_position.tolist()
            payload["joint_positions_rad"] = self.joint_positions_rad.tolist()
        return payload

    def write(self, directory: str | Path, name: str | None = None) -> dict[str, Path]:
        """Write ``<name>.npz`` and ``<name>.trial.json``; return both paths.

        The npz holds the arrays, the json holds everything a human or a script needs
        to decide whether the trial is usable. NaN and infinity are rejected on both
        sides.
        """

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        stem = name or "__".join(
            (
                self.provenance.human_id,
                self.provenance.motion_id,
                self.provenance.perturbation_id,
            )
        )
        arrays: dict[str, np.ndarray] = {
            "root_position": self.root_position,
            "root_quaternion_wxyz": self.root_quaternion,
            "joint_positions_rad": self.joint_positions_rad,
            "joint_velocities_rad_s": self.joint_velocities_rad_s,
            "link_positions": self.link_positions,
            "body_points": self.body_points,
            "phase_label_index": np.array(
                [_phase_index(label) for label in self.phase_labels], dtype=np.int16
            )
            if self.phase_labels
            else np.zeros(0, dtype=np.int16),
            "joint_names": np.array(self.joint_names, dtype="U64"),
            "link_names": np.array(self.link_names, dtype="U64"),
            "body_point_owners": np.array(self.body_point_owners, dtype="U64"),
            "time_physics_s": self.time_physics_s,
            "time_channel_s": self.time_channel_s,
            "channel_root_position": _or_empty(self.channel_root_position),
            "channel_root_quaternion_wxyz": _or_empty(self.channel_root_quaternion),
            "channel_joint_positions_rad": _or_empty(self.channel_joint_positions_rad),
            "channel_body_points": _or_empty(self.channel_body_points),
            "mesh_vertices_xyz": _or_empty(self.mesh_vertices_xyz),
            "mesh_faces": np.zeros((0, 3), dtype=np.int64)
            if self.mesh_faces is None
            else np.asarray(self.mesh_faces, dtype=np.int64),
            "mesh_vertex_owners": np.array(self.mesh_vertex_owners, dtype="U64"),
            "channel_mesh_vertices_xyz": _or_empty(self.channel_mesh_vertices_xyz),
            "contact_force_n": _or_empty(self.contact_force_n),
            "reference_joint_positions_rad": _or_empty(self.reference_joint_positions_rad),
        }
        for key, value in arrays.items():
            if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError(f"refusing to write non-finite values in {key}")
        npz_path = target / f"{stem}.npz"
        np.savez_compressed(npz_path, **arrays, allow_pickle=False)
        json_path = target / f"{stem}.trial.json"
        json_path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        LOGGER.info("wrote trial %s (%d physics frames)", npz_path.name, self.frame_count)
        return {"npz": npz_path, "json": json_path}


def _or_empty(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _phase_index(label: str) -> int:
    from .motion import PHASE_LABELS

    try:
        return PHASE_LABELS.index(label)
    except ValueError as exc:
        raise ValueError(
            f"phase label {label!r} is not in PHASE_LABELS={list(PHASE_LABELS)}"
        ) from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def axis_angle_from_quaternion_series(quaternions: np.ndarray) -> np.ndarray:
    """Convert a ``(N, 4)`` wxyz quaternion series into axis-angle rows."""

    return np.stack(
        [matrix_to_axis_angle(quaternion_to_matrix(row)) for row in quaternions], axis=0
    )


def quaternion_series_from_axis_angle(vectors: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3)`` axis-angle series into ``(N, 4)`` wxyz quaternions."""

    return np.stack([axis_angle_to_quaternion(row) for row in vectors], axis=0)
