"""Load, retarget and screen local AMASS ``.npz`` sequences.

AMASS is a registration-gated dataset, so this module deliberately operates on a
user-provided local directory.  It never downloads or invents a sequence.  The
loader validates the public AMASS fields, records a source hash, retargets the
SMPL-H body block to the project's SMPL topology, and exposes a conservative
kinematic screen for fall candidates.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .motion import (
    AMASS_BODY_FRAME,
    FRAME_SOURCE_AMASS,
    BodyFrame,
    MotionClip,
    MotionProvenance,
    retarget_amass_clip,
)
from .rig import (
    AXIS_PROJECTION_TOLERANCE_RAD,
    HumanRigPlan,
    axis_residuals,
    forward_kinematics,
    joint_values_from_clip,
)
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AmassLibraryLoad",
    "AmassScreenResult",
    "annotate_clip",
    "load_amass_clip",
    "load_amass_clip_by_id",
    "load_amass_library",
    "normalize_root_motion",
    "screen_amass_clip",
    "sha256_file",
]

AMASS_LICENSE = "AMASS non-commercial scientific research"
AMASS_LICENSE_URL = "https://raw.githubusercontent.com/nghorbani/amass/master/LICENSE"


def crop_amass_clip(
    clip: MotionClip, *, start_s: float = 0.0, duration_s: float | None = None
) -> MotionClip:
    """Select a reproducible source interval before anchoring it to the scene."""
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("AMASS start_s must be finite and non-negative")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("AMASS duration_s must be finite and positive")
    first = int(round(start_s * clip.fps))
    stop = clip.frame_count if duration_s is None else min(
        clip.frame_count, first + int(round(duration_s * clip.fps)) + 1
    )
    if stop - first < 2:
        raise ValueError("AMASS interval must contain at least two source frames")
    return replace(
        clip,
        root_translation=clip.root_translation[first:stop],
        root_rotation=clip.root_rotation[first:stop],
        joint_rotations=clip.joint_rotations[first:stop],
        phase_labels=clip.phase_labels[first:stop] if clip.phase_labels else (),
        metadata={**dict(clip.metadata), "source_interval": {
            "first_frame": first, "stop_frame_exclusive": stop,
            "source_fps": clip.fps, "start_s": first / clip.fps,
        }},
    )


def normalize_root_motion(clip: MotionClip) -> MotionClip:
    """Anchor position and heading while preserving gravity and initial body tilt.

    Apply the SAME world yaw H to translations and root orientations. Removing
    full initial rotation would turn a crouched/lying first frame into an upright
    body and rotate the gravity direction. Local joint rotations stay unchanged.
    """

    if clip.provenance.kind != FRAME_SOURCE_AMASS:
        raise ValueError("normalize_root_motion expects an AMASS clip")
    first_rotation = axis_angle_to_matrix(clip.root_rotation[0])
    forward = first_rotation[:, 0]
    if np.linalg.norm(forward[:2]) > 1e-8:
        heading = math.atan2(float(forward[1]), float(forward[0]))
    else:
        left = first_rotation[:, 1]
        heading = math.atan2(float(left[1]), float(left[0])) - math.pi / 2.0
    world_yaw = axis_angle_to_matrix(np.array([0.0, 0.0, -heading]))
    relative_rotation = np.stack(
        [
            matrix_to_axis_angle(world_yaw @ axis_angle_to_matrix(value))
            for value in clip.root_rotation
        ]
    )
    relative_translation = (
        np.asarray(clip.root_translation, dtype=np.float64) - clip.root_translation[0]
    ) @ world_yaw.T
    metadata = {
        **dict(clip.metadata),
        "root_motion_normalization": {
            "anchor": "first_frame_position_and_heading",
            "translation": "H @ (t[k] - t[0])",
            "rotation": "H @ R[k]",
            "world_yaw_rad": -heading,
            "preserves_gravity_and_initial_tilt": True,
            "first_root_translation": np.asarray(clip.root_translation[0]).tolist(),
            "first_root_rotation_axis_angle": np.asarray(clip.root_rotation[0]).tolist(),
        },
    }
    return replace(
        clip,
        root_translation=relative_translation,
        root_rotation=relative_rotation,
        metadata=metadata,
    )


def ground_amass_clip(
    clip: MotionClip, plan: HumanRigPlan, *, support_z_m: float
) -> MotionClip:
    """Align the initial collision geometry once, preserving all subsequent motion."""
    if not math.isfinite(support_z_m):
        raise ValueError("support_z_m must be finite")
    values, _ = joint_values_from_clip(clip, 0, plan)
    poses = forward_kinematics(
        plan, dict(zip(plan.dof_names, values, strict=True)),
        root_position=np.asarray(plan.spawn_root_position) + clip.root_translation[0],
        root_rotation=clip.root_rotation[0],
    )
    lows = []
    for link in plan.links:
        capsule = link.capsule
        if capsule is None:
            continue
        pose = poses[link.name]
        center = pose.translation + pose.rotation @ np.asarray(capsule.center)
        axis = pose.rotation @ capsule.direction
        lows.append(center[2] - abs(axis[2]) * capsule.cylinder_length_m / 2 - capsule.radius_m)
    offset = support_z_m - float(min(lows))
    translation = clip.root_translation.copy()
    translation[:, 2] += offset
    return replace(clip, root_translation=translation, metadata={
        **dict(clip.metadata), "initial_ground_alignment": {
            "z_offset_m": offset, "support_z_m": support_z_m,
            "method": "first_frame_collision_minimum_constant_offset",
        },
    })


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Hash a local source file without loading the complete archive in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"AMASS field {name!r} must contain one scalar value")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="replace")
    text = str(item).strip()
    return text


def _fps(payload: Mapping[str, Any], path: Path) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
        if key in payload:
            value = float(np.asarray(payload[key]).reshape(-1)[0])
            if math.isfinite(value) and value > 0:
                return value
            raise ValueError(f"{path}: {key} must be finite and positive")
    raise ValueError(f"{path}: missing AMASS frame-rate field (mocap_framerate)")


def _sequence_id(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    subject = next((part for part in parts if part.lower().startswith("subject")), "unknown")
    sequence = "/".join(parts)
    return subject, sequence or path.stem


def load_amass_clip(
    path: str | Path,
    *,
    root: str | Path | None = None,
    clip_id: str | None = None,
    source_frame: BodyFrame = AMASS_BODY_FRAME,
) -> MotionClip:
    """Read and retarget one AMASS ``.npz`` file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".npz":
        raise FileNotFoundError(f"AMASS sequence is not a local .npz file: {source}")
    base = Path(root).expanduser().resolve() if root is not None else source.parent
    try:
        subject, sequence = _sequence_id(source, base)
    except ValueError:
        subject, sequence = "unknown", source.stem
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    if "poses" not in payload or "trans" not in payload:
        raise ValueError(f"{source}: AMASS file must contain 'poses' and 'trans' arrays")
    poses = np.asarray(payload["poses"], dtype=np.float64)
    trans = np.asarray(payload["trans"], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"{source}: poses must have shape (N, 156), got {poses.shape}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(
            f"{source}: trans must have shape {(poses.shape[0], 3)}, got {trans.shape}"
        )
    gender = _scalar_text(payload.get("gender", "unknown"), name="gender")
    betas = None
    if "betas" in payload:
        betas = np.asarray(payload["betas"], dtype=np.float64).reshape(-1)[:16]
        if not np.all(np.isfinite(betas)):
            raise ValueError(f"{source}: betas contains a non-finite value")
    source_id = f"amass:{source.parent.name or 'local'}"
    provenance = MotionProvenance(
        kind=FRAME_SOURCE_AMASS,
        source_id=source_id,
        subject=subject,
        sequence=sequence,
        representation="smplh_52",
        license=AMASS_LICENSE,
        license_url=AMASS_LICENSE_URL,
        source_sha256=sha256_file(source),
        notes=(
            f"local source {source}; gender={gender}; "
            f"source_body_frame={source_frame.describe()}"
        ),
    )
    generated_id = clip_id or f"amass__{source.stem}"
    return retarget_amass_clip(
        poses,
        trans,
        clip_id=generated_id,
        fps=_fps(payload, source),
        provenance=provenance,
        betas=betas,
        source_frame=source_frame,
    )


@dataclass(frozen=True, slots=True)
class AmassLibraryLoad:
    """What a batch AMASS load produced, including what it could not read.

    A screening job over thousands of sequences must not abort because one file has a
    motion spike, and it must not quietly shrink either: every skipped file is named
    here with its reason, so a candidate count can be read against the number of files
    that were actually screened.
    """

    clips: dict[str, MotionClip]
    failures: tuple[tuple[str, str], ...]

    @property
    def scanned(self) -> int:
        return len(self.clips) + len(self.failures)


def load_amass_library(
    root: str | Path,
    *,
    limit: int | None = None,
    source_frame: BodyFrame = AMASS_BODY_FRAME,
) -> AmassLibraryLoad:
    """Load every readable AMASS file below ``root`` in deterministic order.

    An unreadable or ambiguously-named sequence is recorded in
    :attr:`AmassLibraryLoad.failures` and skipped; only a root that yields no usable
    sequence at all is an error, because at that point there is nothing to report on.
    """

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"AMASS root does not exist: {directory}. Register at "
            "https://amass.is.tue.mpg.de/download.php and unpack the .npz files locally."
        )
    paths = sorted(directory.rglob("*.npz"))
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"AMASS limit must be positive, got {limit!r}")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no AMASS .npz sequences found below {directory}")
    clips: dict[str, MotionClip] = {}
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            clip = load_amass_clip(path, root=directory, source_frame=source_frame)
        except (OSError, ValueError) as exc:
            LOGGER.warning("skipping unreadable AMASS sequence %s: %s", path, exc)
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        if clip.clip_id in clips:
            reason = f"duplicate AMASS clip id {clip.clip_id!r}"
            LOGGER.warning("skipping %s: %s", path, reason)
            failures.append((str(path), reason))
            continue
        clips[clip.clip_id] = clip
    if not clips:
        first = failures[0][1] if failures else "no files"
        raise ValueError(
            f"none of the {len(paths)} AMASS sequences below {directory} could be loaded; "
            f"first failure: {first}"
        )
    return AmassLibraryLoad(clips=clips, failures=tuple(failures))


def load_amass_clip_by_id(
    root: str | Path,
    clip_id: str,
    *,
    source_frame: BodyFrame = AMASS_BODY_FRAME,
) -> MotionClip:
    """Load one deterministic AMASS clip without scanning every sequence.

    The command-line viewer normally receives an explicit ``amass__<stem>``
    identifier.  Resolving that stem directly keeps startup proportional to the
    selected file instead of parsing an entire downloaded subset.  Duplicate
    stems are rejected because an identifier that depends on directory order is
    not reproducible.
    """

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"AMASS root does not exist: {directory}")
    token = str(clip_id).strip()
    prefix = "amass__"
    if not token.startswith(prefix) or len(token) == len(prefix):
        raise ValueError(f"AMASS clip id must start with {prefix!r}, got {clip_id!r}")
    stem = token[len(prefix) :]
    paths = sorted(directory.rglob(f"{stem}.npz"))
    if not paths:
        raise FileNotFoundError(f"no AMASS .npz named {stem!r} found below {directory}")
    if len(paths) > 1:
        raise ValueError(
            f"AMASS clip id {token!r} is ambiguous; matching files: "
            + ", ".join(str(path) for path in paths)
        )
    return load_amass_clip(
        paths[0],
        root=directory,
        clip_id=token,
        source_frame=source_frame,
    )


class AmassScreenResult:
    """Serializable result of the kinematic fall-candidate screen.

    ``accepted`` is the conjunction the consumers filter on. ``fall_candidate`` and
    ``rig_expressible`` are reported separately so a zero count can be read as either
    "this corpus holds no falls" or "our rig cannot replay them" without guessing.
    """

    __slots__ = (
        "accepted",
        "fall_candidate",
        "peak_trunk_angle_deg",
        "root_drop_m",
        "peak_down_speed_m_s",
        "axis_residual_rad",
        "axis_residual_joint",
        "joint_limit_excess_deg",
        "joint_limit_joint",
        "rig_expressible",
        "reason",
    )

    def __init__(
        self,
        *,
        accepted: bool,
        fall_candidate: bool,
        peak_trunk_angle_deg: float,
        root_drop_m: float,
        peak_down_speed_m_s: float,
        axis_residual_rad: float,
        axis_residual_joint: str = "",
        rig_expressible: bool,
        reason: str,
        joint_limit_excess_deg: float = 0.0,
        joint_limit_joint: str = "",
    ) -> None:
        self.accepted = bool(accepted)
        self.fall_candidate = bool(fall_candidate)
        self.peak_trunk_angle_deg = float(peak_trunk_angle_deg)
        self.root_drop_m = float(root_drop_m)
        self.peak_down_speed_m_s = float(peak_down_speed_m_s)
        self.axis_residual_rad = float(axis_residual_rad)
        self.axis_residual_joint = str(axis_residual_joint)
        self.rig_expressible = bool(rig_expressible)
        self.reason = str(reason)
        self.joint_limit_excess_deg = float(joint_limit_excess_deg)
        self.joint_limit_joint = str(joint_limit_joint)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "fall_candidate": self.fall_candidate,
            "peak_trunk_angle_deg": self.peak_trunk_angle_deg,
            "root_drop_m": self.root_drop_m,
            "peak_down_speed_m_s": self.peak_down_speed_m_s,
            "axis_residual_rad": self.axis_residual_rad,
            "axis_residual_joint": self.axis_residual_joint,
            "rig_expressible": self.rig_expressible,
            "reason": self.reason,
            "joint_limit_excess_deg": self.joint_limit_excess_deg,
            "joint_limit_joint": self.joint_limit_joint,
        }


def screen_amass_clip(
    clip: MotionClip,
    plan: HumanRigPlan,
    *,
    trunk_angle_deg: float = 60.0,
    root_drop_m: float = 0.20,
    down_speed_m_s: float = 1.0,
) -> AmassScreenResult:
    """Screen a retargeted clip for a fall candidate the shipped rig can replay.

    This is a candidate screen, not a clinical label. Two independent questions are
    answered and reported separately, because collapsing them into one boolean is how
    "AMASS contains no falls" gets written down when the true statement is "our rig
    has fourteen axes and a fall needs more":

    * **is it a fall** -- the trunk goes over, and the root either drops far enough or
      drops fast enough;
    * **can this rig express it** -- decided by
      :func:`~sim2sense_fall.humans.rig.joint_values_from_clip`, the same rule the
      simulator applies at run time, so a clip can never be reported usable here and
      then rejected there.

    ``accepted`` is the conjunction, which is what consumers filter on.
    """

    if clip.provenance.kind != FRAME_SOURCE_AMASS:
        raise ValueError("screen_amass_clip expects an AMASS clip")
    residual = 0.0
    residual_frame = 0
    limit_excess = 0.0
    limit_joint = ""
    low = np.array([joint.lower_deg for joint in plan.joints])
    high = np.array([joint.upper_deg for joint in plan.joints])
    link_series: list[np.ndarray] = []
    root_series = np.asarray(plan.spawn_root_position, dtype=np.float64) + clip.root_translation
    for frame in range(clip.frame_count):
        # An infinite tolerance asks the runner's own mapper for its residual without
        # letting it raise; the comparison below then uses the mapper's real bound.
        values, worst = joint_values_from_clip(clip, frame, plan, axis_tolerance_rad=math.inf)
        excess = np.maximum(low - np.degrees(values), np.degrees(values) - high)
        if float(excess.max()) > limit_excess:
            limit_excess = float(excess.max())
            limit_joint = plan.dof_names[int(excess.argmax())]
        if worst >= residual:
            residual, residual_frame = worst, frame
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, values, strict=True)),
            root_position=root_series[frame],
            root_rotation=clip.root_rotation[frame],
        )
        link_series.append(np.stack([poses[link.name].translation for link in plan.links]))
    links = np.stack(link_series)
    pelvis_index = next(index for index, link in enumerate(plan.links) if link.name == "pelvis")
    neck_index = next(index for index, link in enumerate(plan.links) if link.name == "neck")
    axis = links[:, neck_index] - links[:, pelvis_index]
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(axis[:, 2], -1.0, 1.0)))
    drops = -np.diff(root_series[:, 2], prepend=root_series[0, 2]) * clip.fps
    peak_drop = float(np.max(drops, initial=0.0))
    root_drop = float(np.max(root_series[0, 2] - root_series[:, 2], initial=0.0))
    peak_trunk = float(np.max(angles, initial=0.0))
    rig_expressible = residual <= AXIS_PROJECTION_TOLERANCE_RAD and limit_excess <= 1e-6
    fall_candidate = (
        peak_trunk >= trunk_angle_deg and (root_drop >= root_drop_m or peak_drop >= down_speed_m_s)
    )
    accepted = rig_expressible and fall_candidate
    binding_joint = ""
    if limit_excess > 1e-6:
        reason = f"joint limit exceeded by {limit_excess:.3f} deg at {limit_joint}"
    elif not rig_expressible:
        # Naming the joint that binds turns "0 clips passed" into something a reader
        # can act on: it says which DOF the rig is missing, not just that it is missing
        # one. Without it the only way to learn this is to measure the corpus again.
        per_joint = axis_residuals(clip, residual_frame, plan)
        binding_joint = max(per_joint, key=lambda name: per_joint[name][1])
        reason = (
            f"the single-axis rig cannot express this motion: {binding_joint} carries "
            f"{math.degrees(residual):.1f} deg of rotation about an axis it has no DOF "
            f"for, at frame {residual_frame}"
        )
    elif not fall_candidate:
        reason = "upright or slow lowering motion; no fall candidate"
    else:
        reason = "kinematic fall candidate; verify with physics and inspect manually"
    return AmassScreenResult(
        accepted=accepted,
        fall_candidate=fall_candidate,
        axis_residual_joint=binding_joint,
        peak_trunk_angle_deg=peak_trunk,
        root_drop_m=root_drop,
        peak_down_speed_m_s=peak_drop,
        axis_residual_rad=residual,
        rig_expressible=rig_expressible,
        reason=reason,
        joint_limit_excess_deg=limit_excess,
        joint_limit_joint=limit_joint,
    )


def annotate_clip(clip: MotionClip, screen: AmassScreenResult) -> MotionClip:
    """Attach screening metadata and fall tags without changing source arrays."""

    tags = (
        tuple((*clip.tags, "amass", "fall_reference"))
        if screen.accepted
        else tuple((*clip.tags, "amass"))
    )
    return replace(
        clip,
        tags=tags,
        metadata={**dict(clip.metadata), "amass_screen": screen.as_dict()},
    )
