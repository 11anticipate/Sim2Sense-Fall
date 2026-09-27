"""Motion clips: the unit of exchange between motion capture, scripting and physics.

A :class:`MotionClip` is an immutable, fully validated sample of joint rotations
and root motion on a uniform time grid. Three producers create clips:

``scripted``
    Hand-authored keyframes or sine channels from ``configs/humans/motions.yaml``.
    These are the *reference* motions. They are analytic, so they are exactly
    reproducible and cheap to unit-test.
``amass``
    A retargeted AMASS (SMPL-H) sequence. AMASS stores 52 joints x 3 axis-angle
    values per frame: the first triple is the global root orientation, the next
    21 triples are body joints 1..21, and the remaining 30 triples are finger
    joints. Because SMPL shares joints 0..21 with SMPL-H, the body block maps
    across without renaming -- but SMPL's joints 22/23 (``left_hand``,
    ``right_hand``) have *no* AMASS counterpart: SMPL-H's next joints are
    fingers, not hands. Those two joints are therefore left at rest and the fact
    is recorded in the clip's notes rather than being papered over.
``replay``
    A clip captured back from the simulator, so ground truth and reference can be
    compared frame by frame.

Validation is deliberately unforgiving. A clip whose frames are not finite, whose
rotations exceed a plausible joint angle, or whose per-frame change exceeds a
continuity bound is rejected at construction, because a retargeting frame-rate
mistake otherwise shows up much later as an inexplicably violent simulation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .rotations import (
    ANGLE_EPSILON,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_matrix_error,
)
from .skeleton import SkeletonTopology, smpl_skeleton

__all__ = [
    "AMASS_BODY_FRAME",
    "FRAME_SOURCE_AMASS",
    "FRAME_SOURCE_SCRIPTED",
    "MAX_FRAME_DELTA_RAD",
    "MAX_JOINT_ANGLE_RAD",
    "BodyFrame",
    "MotionClip",
    "MotionProvenance",
    "PHASE_LABELS",
    "PIPELINE_FORWARD_AXIS",
    "PIPELINE_LEFT_AXIS",
    "PIPELINE_UP_AXIS",
    "axis_angle_change_rad",
    "body_frame_conversion",
    "compile_scripted_clip",
    "load_motion_library",
    "motion_library_from_mapping",
    "retarget_amass_clip",
    "up_axis_conversion",
]

LOGGER = logging.getLogger(__name__)

FRAME_SOURCE_SCRIPTED = "scripted"
FRAME_SOURCE_AMASS = "amass"

#: Largest rotation any single joint may express, in radians (180 degrees).
MAX_JOINT_ANGLE_RAD = math.pi + 1e-6
#: Largest change of a single joint between adjacent frames, in radians.
#: 60 degrees per frame is already implausible at 60 Hz; anything beyond it is a
#: retargeting or unit mistake rather than a real motion.
MAX_FRAME_DELTA_RAD = math.radians(60.0)

#: Vocabulary of motion phase labels used by the event labeller and the export.
PHASE_LABELS: tuple[str, ...] = (
    "standing",
    "standing_recovery",
    "bending",
    "squatting",
    "sitting",
    "lying_supine",
    "lying_prone",
    "lying_lateral",
    "walking",
    "falling",
    "fallen",
    "unstable",
    "unknown",
)

_INTERPOLATIONS = ("linear", "smoothstep")


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _as_finite_array(value: object, name: str) -> np.ndarray:
    """Coerce to a float64 array of any shape, rejecting non-finite values."""

    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def axis_angle_change_rad(left: object, right: object) -> float:
    """Geodesic distance between two axis-angle rotations, in radians."""

    delta = axis_angle_to_matrix(left).T @ axis_angle_to_matrix(right)
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.acos(cosine)


def up_axis_conversion(source_up: str, target_up: str) -> np.ndarray:
    """Change-of-basis matrix mapping ``source_up`` coordinates into ``target_up``.

    Both frames are right-handed and share the same forward direction (``+X``).
    The returned matrix is a proper rotation, so axis-angle vectors transform by
    a plain matrix product and the rotation angle is preserved -- no
    pseudo-vector sign flipping is needed.

    .. warning::
       This only fixes the **up** axis. It assumes both frames agree on their
       horizontal convention, so it cannot express a yaw. It was previously used
       by the SMPL loader as a whole-body basis, which silently put the template's
       lateral axis (the shoulder span) onto the pipeline's forward axis and made
       every exported mesh lie on its side. When the source frame's forward or
       lateral axis is unknown, use :func:`body_frame_conversion` instead.
    """

    frames = {
        ("y", "z"): np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        ("z", "y"): np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    }
    source = str(source_up).strip().lower()
    target = str(target_up).strip().lower()
    if source == target:
        return np.eye(3)
    if (source, target) not in frames:
        raise ValueError(
            f"unsupported up-axis conversion {source_up!r} -> {target_up!r}; "
            f"supported: {sorted(frames)} and identical axes"
        )
    matrix = frames[(source, target)]
    if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-12:
        raise ValueError("up-axis conversion must be a proper rotation")
    return matrix


#: The pipeline's own body frame, as documented on RestSkeleton: forward, left, up.
PIPELINE_FORWARD_AXIS = "x"
PIPELINE_LEFT_AXIS = "y"
PIPELINE_UP_AXIS = "z"


def body_frame_conversion(
    *,
    up: str,
    forward: str,
    left: str,
    source: str = "body model",
) -> np.ndarray:
    """Change-of-basis matrix for a frame given as an anatomical axis triple.

    ``up`` / ``forward`` / ``left`` name the source frame's own axes -- e.g.
    ``up="y", forward="z", left="x"`` for a template whose ``+Y`` is up, ``+Z`` is
    forward and ``+X`` points left. The returned matrix carries source coordinates
    into the pipeline frame (forward ``+X``, left ``+Y``, up ``+Z``).

    This exists because :func:`up_axis_conversion` cannot express a yaw: it pins the
    up axis and silently assumes the two frames agree horizontally. A body model
    whose lateral axis sits where the pipeline expects forward would keep "up is up"
    and still be exported lying on its side, which is exactly the defect this
    function was added to fix.

    Each token must be one of ``x``/``y``/``z`` with an optional ``-`` prefix, all
    three must be distinct axes, and the triple must be right-handed. A left-handed
    triple describes a mirrored body and is rejected rather than silently flipped,
    because mirroring a skeleton swaps its left and right limbs.
    """

    def parse(token: str, role: str) -> tuple[int, float]:
        text = str(token).strip().lower()
        sign = -1.0 if text.startswith("-") else 1.0
        letter = text[1:] if text[:1] in "+-" else text
        if letter not in "xyz":
            raise ValueError(
                f"{source}: {role} axis must be one of x/y/z (optionally '-'-prefixed), "
                f"got {token!r}"
            )
        return "xyz".index(letter), sign

    up_index, up_sign = parse(up, "up")
    forward_index, forward_sign = parse(forward, "forward")
    left_index, left_sign = parse(left, "left")
    if len({up_index, forward_index, left_index}) != 3:
        raise ValueError(
            f"{source}: the up/forward/left axes must be three distinct axes, got "
            f"up={up!r}, forward={forward!r}, left={left!r}"
        )

    # Rows of B are the pipeline axes expressed in source coordinates: the pipeline
    # forward row reads the source's forward component and so on. Building it this
    # way means B @ v_source = v_pipeline directly.
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix["xyz".index(PIPELINE_FORWARD_AXIS), forward_index] = forward_sign
    matrix["xyz".index(PIPELINE_LEFT_AXIS), left_index] = left_sign
    matrix["xyz".index(PIPELINE_UP_AXIS), up_index] = up_sign

    determinant = float(np.linalg.det(matrix))
    if abs(determinant - 1.0) > 1e-9:
        raise ValueError(
            f"{source}: the frame (up={up!r}, forward={forward!r}, left={left!r}) has "
            f"determinant {determinant:+.3f}; a right-handed body frame needs +1. A -1 "
            "means the triple is left-handed (mirrored), which would swap left and right"
        )
    return matrix


@dataclass(frozen=True, slots=True)
class BodyFrame:
    """The anatomical axes a body file is authored in.

    ``up``/``forward``/``left`` name the *file's own* axes, so the triple can be handed
    straight to :func:`body_frame_conversion`. It is a value, not a preference: either
    the file is authored that way or it is not, and getting it wrong yaws the body while
    leaving every "is up up" check passing.
    """

    up: str
    forward: str
    left: str

    def basis(self, *, source: str = "body frame") -> np.ndarray:
        """Change-of-basis matrix carrying this frame into the pipeline frame."""

        return body_frame_conversion(
            up=self.up, forward=self.forward, left=self.left, source=source
        )

    def describe(self) -> str:
        return f"up={self.up} forward={self.forward} left={self.left}"


#: The frame AMASS pose parameters are authored in.
#:
#: AMASS stores SMPL/SMPL-H *local* joint rotations, and SMPL gives its joints no
#: per-joint canonical offset, so those numbers live in the model's own rest frame --
#: which is not the pipeline frame. Three measurements on the local corpus fix it:
#:
#: * the released template's rest joints put the crown on ``+Y``, the face on ``+Z``
#:   and the shoulder span on ``+X``, which is what ``assets._derive_body_frame``
#:   re-measures from the file on every load;
#: * pose slots 10 and 11 are identically zero in every sampled sequence, the signature
#:   of SMPL's two leaf foot joints, so slot ``k`` names joint ``k`` with no reordering;
#: * with this basis the hip/knee/ankle/spine flexion energy lands on the pipeline's
#:   ``Y``, which is the axis the rig declares. The up-only conversion this replaces
#:   put that flexion on ``X`` instead, so every real sequence read as "rotating about
#:   an axis the rig cannot express" and the whole library screened out.
AMASS_BODY_FRAME = BodyFrame(up="y", forward="z", left="x")


@dataclass(frozen=True, slots=True)
class MotionProvenance:
    """Where a clip came from, in enough detail to reproduce and to cite."""

    kind: str
    source_id: str
    subject: str
    sequence: str
    representation: str
    license: str
    license_url: str
    source_sha256: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "kind",
            "source_id",
            "subject",
            "sequence",
            "representation",
            "license",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"motion provenance field {name!r} must be non-empty")
        if not isinstance(self.license_url, str):
            raise ValueError("motion provenance license_url must be a string (may be empty)")
        if self.kind not in (FRAME_SOURCE_SCRIPTED, FRAME_SOURCE_AMASS):
            raise ValueError(
                f"motion provenance kind must be one of "
                f"{(FRAME_SOURCE_SCRIPTED, FRAME_SOURCE_AMASS)}, got {self.kind!r}"
            )
        if self.source_sha256 is not None and len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a 64-character hex digest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "subject": self.subject,
            "sequence": self.sequence,
            "representation": self.representation,
            "license": self.license,
            "license_url": self.license_url,
            "source_sha256": self.source_sha256,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True, eq=False)
class MotionClip:
    """Joint-space reference motion on a uniform time grid.

    ``joint_rotations[k, j]`` is the axis-angle rotation of joint ``j`` relative
    to its parent at frame ``k``, expressed about the joint's rest-aligned axes.
    ``root_rotation[k]`` is the global orientation of the root joint and
    ``root_translation[k]`` is the world position of the root joint.
    """

    clip_id: str
    fps: float
    joint_names: tuple[str, ...]
    root_translation: np.ndarray
    root_rotation: np.ndarray
    joint_rotations: np.ndarray
    provenance: MotionProvenance
    betas: np.ndarray | None = None
    phase_labels: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise ValueError(f"fps must be a number, got {self.fps!r}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {self.fps!r}")
        if not isinstance(self.provenance, MotionProvenance):
            raise ValueError("provenance must be a MotionProvenance")
        if not self.joint_names:
            raise ValueError("clip must declare its joint names")
        duplicates = sorted({n for n in self.joint_names if self.joint_names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate joint names in clip: {duplicates}")

        object.__setattr__(
            self, "root_translation", _as_finite_array(self.root_translation, "root_translation")
        )
        object.__setattr__(
            self, "root_rotation", _as_finite_array(self.root_rotation, "root_rotation")
        )
        object.__setattr__(
            self, "joint_rotations", _as_finite_array(self.joint_rotations, "joint_rotations")
        )
        self._validate_shapes()
        self._validate_values()
        self._validate_phases()

    # -- structural -------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return int(self.root_translation.shape[0])

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def duration_s(self) -> float:
        return (self.frame_count - 1) / float(self.fps)

    @property
    def times_s(self) -> np.ndarray:
        return np.arange(self.frame_count, dtype=np.float64) / float(self.fps)

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.clip_id}: clip has no joint {name!r}") from exc

    def rotation_of(self, frame: int, joint: str) -> np.ndarray:
        return self.joint_rotations[frame, self.joint_index(joint)]

    def _validate_shapes(self) -> None:
        if self.root_translation.shape != (self.frame_count, 3):
            raise ValueError(f"root_translation must be (N, 3), got {self.root_translation.shape}")
        if self.root_rotation.shape != (self.frame_count, 3):
            raise ValueError(f"root_rotation must be (N, 3), got {self.root_rotation.shape}")
        expected = (self.frame_count, self.joint_count, 3)
        if self.joint_rotations.shape != expected:
            raise ValueError(
                f"joint_rotations must be {expected} for {self.joint_count} joints, "
                f"got {self.joint_rotations.shape}"
            )
        if self.frame_count < 2:
            raise ValueError(
                f"{self.clip_id}: a clip needs at least 2 frames, got {self.frame_count}"
            )
        if self.betas is not None:
            object.__setattr__(
                self, "betas", _finite_array(self.betas, np.shape(self.betas), "betas")
            )
            if self.betas.ndim != 1:
                raise ValueError(f"betas must be a vector, got {self.betas.shape}")

    def _validate_values(self) -> None:
        magnitudes = np.linalg.norm(self.joint_rotations, axis=-1)
        worst = int(np.argmax(magnitudes)) if magnitudes.size else 0
        if float(magnitudes.max(initial=0.0)) > MAX_JOINT_ANGLE_RAD:
            frame, joint = divmod(worst, self.joint_count)
            raise ValueError(
                f"{self.clip_id}: joint {self.joint_names[joint]!r} at frame {frame} rotates "
                f"{math.degrees(float(magnitudes[frame, joint])):.1f} degrees, beyond the "
                f"{math.degrees(MAX_JOINT_ANGLE_RAD):.0f} degree bound; likely a "
                "degrees/radians or representation mix-up"
            )
        deltas = self.joint_rotations[1:] - self.joint_rotations[:-1]
        step = np.linalg.norm(deltas, axis=-1)
        if step.size and float(step.max()) > MAX_FRAME_DELTA_RAD:
            frame, joint = divmod(int(np.argmax(step)), self.joint_count)
            raise ValueError(
                f"{self.clip_id}: joint {self.joint_names[joint]!r} jumps "
                f"{math.degrees(float(step[frame, joint])):.1f} degrees between frames "
                f"{frame} and {frame + 1} at {self.fps:g} Hz, beyond the "
                f"{math.degrees(MAX_FRAME_DELTA_RAD):.0f} degree continuity bound; check the "
                "source frame rate before retargeting"
            )
        root_deltas = np.asarray(
            [
                axis_angle_change_rad(left, right)
                for left, right in zip(self.root_rotation[:-1], self.root_rotation[1:], strict=True)
            ],
            dtype=np.float64,
        )
        if root_deltas.size and float(root_deltas.max()) > MAX_FRAME_DELTA_RAD:
            frame = int(np.argmax(root_deltas))
            raise ValueError(
                f"{self.clip_id}: root rotation jumps "
                f"{math.degrees(float(root_deltas[frame])):.1f} degrees between frames "
                f"{frame} and {frame + 1}"
            )
        self._validate_rotations_as_matrices()

    def _validate_rotations_as_matrices(self) -> None:
        """Re-derive matrices and check they are proper rotations.

        Rodrigues' formula cannot produce a non-rotation, so this is a cheap
        guard against a caller having replaced the arrays with something that was
        not produced by it (for example a raw quaternion stored in the same
        shape).
        """

        for frame in (0, self.frame_count // 2, self.frame_count - 1):
            for index, name in enumerate(self.joint_names):
                error = rotation_matrix_error(
                    axis_angle_to_matrix(self.joint_rotations[frame, index])
                )
                if error > 1e-9:
                    raise ValueError(
                        f"{self.clip_id}: joint {name!r} at frame {frame} is not a "
                        f"proper rotation (error {error:.2e})"
                    )

    def _validate_phases(self) -> None:
        if not self.phase_labels:
            return
        if len(self.phase_labels) != self.frame_count:
            raise ValueError(
                f"phase_labels has {len(self.phase_labels)} entries but the clip has "
                f"{self.frame_count} frames"
            )
        unknown = sorted(set(self.phase_labels) - set(PHASE_LABELS))
        if unknown:
            raise ValueError(f"unknown phase labels {unknown}; allowed: {list(PHASE_LABELS)}")

    # -- derived ----------------------------------------------------------

    def with_phase_labels(self, labels: Sequence[str]) -> MotionClip:
        """Return a copy carrying new per-frame phase labels."""

        return MotionClip(
            clip_id=self.clip_id,
            fps=self.fps,
            joint_names=self.joint_names,
            root_translation=self.root_translation,
            root_rotation=self.root_rotation,
            joint_rotations=self.joint_rotations,
            provenance=self.provenance,
            betas=self.betas,
            phase_labels=tuple(labels),
            tags=self.tags,
            metadata=dict(self.metadata),
        )

    def root_speed_m_s(self) -> np.ndarray:
        """Root translation speed per inter-frame interval, in m/s."""

        deltas = np.linalg.norm(np.diff(self.root_translation, axis=0), axis=-1)
        return deltas * float(self.fps)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "clip_id": self.clip_id,
            "fps": float(self.fps),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 6),
            "joint_count": self.joint_count,
            "joint_names": list(self.joint_names),
            "provenance": self.provenance.as_dict(),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "phase_labels": list(self.phase_labels),
            "max_joint_angle_deg": round(
                math.degrees(float(np.linalg.norm(self.joint_rotations, axis=-1).max())), 4
            ),
            "root_displacement_m": round(
                float(np.linalg.norm(self.root_translation[-1] - self.root_translation[0])), 6
            ),
        }
        if include_arrays:
            payload["root_translation"] = self.root_translation.tolist()
            payload["root_rotation"] = self.root_rotation.tolist()
            payload["joint_rotations"] = self.joint_rotations.tolist()
        return payload

    def resample(self, target_fps: float, *, method: str = "linear") -> MotionClip:
        """Resample onto a new uniform grid.

        ``linear`` interpolates the axis-angle components, which is exact for
        small per-frame deltas and is the cheaper option. ``slerp`` interpolates
        through rotation matrices and stays on the rotation manifold, so it is
        the right choice when upsampling a low-rate capture for physics. The
        chosen method is recorded in the clip metadata.
        """

        if str(method) not in _INTERPOLATIONS and str(method) != "slerp":
            raise ValueError(
                f"resample method must be one of {(*_INTERPOLATIONS, 'slerp')}, got {method!r}"
            )
        target = float(target_fps)
        if not math.isfinite(target) or target <= 0:
            raise ValueError(f"target_fps must be finite and positive, got {target_fps!r}")
        frames = max(2, int(round(self.duration_s * target)) + 1)
        new_times = np.arange(frames, dtype=np.float64) / target
        source_times = self.times_s
        # ``searchsorted`` gives the index of the first sample at or after t.
        upper = np.clip(
            np.searchsorted(source_times, new_times, side="left"), 1, self.frame_count - 1
        )
        lower = upper - 1
        span = source_times[upper] - source_times[lower]
        weight = np.where(
            span > 0, (new_times - source_times[lower]) / np.where(span > 0, span, 1.0), 0.0
        )
        if method == "smoothstep":
            weight = weight * weight * (3.0 - 2.0 * weight)

        root_translation = _lerp(self.root_translation[lower], self.root_translation[upper], weight)
        if method == "slerp":
            root_rotation = _slerp_axis_angle(self.root_rotation, lower, upper, weight)
            joint_rotations = _slerp_axis_angle(self.joint_rotations, lower, upper, weight)
        else:
            root_rotation = _lerp(self.root_rotation[lower], self.root_rotation[upper], weight)
            joint_rotations = _lerp(
                self.joint_rotations[lower], self.joint_rotations[upper], weight[:, None, None]
            )
        return MotionClip(
            clip_id=f"{self.clip_id}@{target:g}Hz",
            fps=target,
            joint_names=self.joint_names,
            root_translation=root_translation,
            root_rotation=root_rotation,
            joint_rotations=joint_rotations,
            provenance=self.provenance,
            betas=None if self.betas is None else self.betas.copy(),
            phase_labels=(),
            tags=self.tags,
            metadata={
                **dict(self.metadata),
                "resampled_from_fps": float(self.fps),
                "resample_method": method,
            },
        )


def _lerp(lower: np.ndarray, upper: np.ndarray, weight: np.ndarray) -> np.ndarray:
    expanded = weight
    while expanded.ndim < lower.ndim:
        expanded = expanded[..., None]
    return lower + (upper - lower) * expanded


def _slerp_axis_angle(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Interpolate axis-angle arrays along the rotation manifold.

    Falls back to a linear blend when a pair of frames is nearly identical, where
    the geodesic is numerically undefined.
    """

    flat = values.reshape(values.shape[0], -1, 3)
    blended = np.empty((len(lower), flat.shape[1], 3), dtype=np.float64)
    for index in range(flat.shape[1]):
        left = flat[lower, index]
        right = flat[upper, index]
        for row in range(len(lower)):
            blended[row, index] = _slerp_pair(left[row], right[row], float(weight[row]))
    return blended.reshape((len(lower), *values.shape[1:]))


def _slerp_pair(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    matrix_left = axis_angle_to_matrix(left)
    matrix_right = axis_angle_to_matrix(right)
    delta = matrix_left.T @ matrix_right
    if rotation_matrix_error(delta) > 1e-9:
        raise ValueError("slerp inputs are not proper rotations")
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < ANGLE_EPSILON:
        return np.asarray(left, dtype=np.float64)
    if abs(math.pi - angle) < 1e-6:
        # Antipodal: pick a deterministic halfway axis rather than dividing by ~0.
        axis = np.array([1.0, 0.0, 0.0])
        return matrix_to_axis_angle(matrix_left @ axis_angle_to_matrix(axis * angle * weight))
    axis = np.array(
        [delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]
    ) / (2.0 * math.sin(angle))
    interpolated = axis_angle_to_matrix(axis * (angle * weight))
    return matrix_to_axis_angle(matrix_left @ interpolated)


def retarget_amass_clip(
    poses: object,
    trans: object,
    *,
    clip_id: str,
    fps: float,
    provenance: MotionProvenance,
    betas: object | None = None,
    source_frame: BodyFrame = AMASS_BODY_FRAME,
    body_joints: int = 22,
    total_joints: int = 52,
    topology: SkeletonTopology | None = None,
) -> MotionClip:
    """Retarget an AMASS (SMPL-H) sequence onto the SMPL joint set.

    ``poses`` is ``(N, 3 * total_joints)`` and ``trans`` is ``(N, 3)``. The first
    triple becomes the global root orientation; the following
    ``3 * (body_joints - 1)`` values become joints 1..21; everything after is
    finger motion and is dropped, with the drop recorded in the clip metadata.
    SMPL's hand joints have no AMASS counterpart and stay at rest.

    ``source_frame`` describes the model's local rest frame. AMASS's capture world
    is Z-up, already matching the pipeline world. Local rotations change basis
    with B R B.T; the root maps model to world, so it becomes R_root B.T.
    World translations must not be transformed by the model-local basis.
    """

    skeleton = topology or smpl_skeleton()
    pose_array = np.asarray(poses, dtype=np.float64)
    trans_array = np.asarray(trans, dtype=np.float64)
    if pose_array.ndim != 2 or pose_array.shape[1] % 3:
        raise ValueError(f"poses must be (N, 3k), got {pose_array.shape}")
    if trans_array.shape != (pose_array.shape[0], 3):
        raise ValueError(f"trans must be (N, 3), got {trans_array.shape}")
    joint_total = pose_array.shape[1] // 3
    if joint_total != total_joints:
        raise ValueError(
            f"expected {total_joints} joints in the AMASS pose block, got {joint_total}; "
            "check the expected pose_parameters in the asset registry"
        )
    if body_joints != skeleton.joint_count - 2:
        raise ValueError(
            f"AMASS body_joints is {body_joints}, but the target topology has "
            f"{skeleton.joint_count} joints; expected body_joints = {skeleton.joint_count - 2} "
            "(SMPL's two hand joints are not present in SMPL-H)"
        )
    if not np.all(np.isfinite(pose_array)) or not np.all(np.isfinite(trans_array)):
        raise ValueError("AMASS sequence contains non-finite values")

    if not isinstance(source_frame, BodyFrame):
        raise ValueError(f"source_frame must be a BodyFrame, got {source_frame!r}")
    basis = source_frame.basis(source=clip_id)
    root_rotation = np.stack([
        matrix_to_axis_angle(axis_angle_to_matrix(value) @ basis.T)
        for value in pose_array[:, :3]
    ])
    joint_rotations = np.zeros((pose_array.shape[0], skeleton.joint_count, 3), dtype=np.float64)
    body_block = pose_array[:, 3 : 3 * body_joints].reshape(pose_array.shape[0], body_joints - 1, 3)
    joint_rotations[:, 1:body_joints] = (basis @ body_block.transpose(0, 2, 1)).transpose(0, 2, 1)
    root_translation = trans_array.copy()

    dropped_joints = joint_total - body_joints
    notes = (
        f"retargeted from {provenance.representation} ({total_joints} joints); "
        f"dropped {dropped_joints} finger joints; "
        "SMPL hand joints 22/23 have no SMPL-H counterpart and stay at rest; "
        f"body frame {source_frame.describe()} converted to the pipeline frame"
    )
    return MotionClip(
        clip_id=clip_id,
        fps=float(fps),
        joint_names=skeleton.joint_names,
        root_translation=root_translation,
        root_rotation=root_rotation,
        joint_rotations=joint_rotations,
        provenance=MotionProvenance(
            kind=provenance.kind,
            source_id=provenance.source_id,
            subject=provenance.subject,
            sequence=provenance.sequence,
            representation=provenance.representation,
            license=provenance.license,
            license_url=provenance.license_url,
            source_sha256=provenance.source_sha256,
            notes=f"{provenance.notes} | {notes}".strip(" |"),
        ),
        betas=None if betas is None else np.asarray(betas, dtype=np.float64),
        metadata={
            "retarget": "amass_smplh_to_smpl",
            "source_joint_count": joint_total,
            "dropped_finger_joints": dropped_joints,
            "source_body_frame": source_frame.describe(),
            "source_world_frame": "z_up",
            "root_transform": "R_source @ B_model.T; t_source unchanged",
            "retarget_version": 2,
        },
    )


def compile_scripted_clip(
    spec: Mapping[str, Any],
    *,
    topology: SkeletonTopology | None = None,
    reference_frame: Sequence[float] | None = None,
) -> MotionClip:
    """Compile a YAML motion program into a clip.

    Two generators are supported. ``keyframes`` interpolates per-joint control
    points, and ``sine`` drives each named channel with a sine of the given
    amplitude, frequency and phase -- enough for a walk-in-place cycle without
    shipping a capture file. Both are analytic, which is what makes the CPU
    acceptance tests meaningful.
    """

    skeleton = topology or smpl_skeleton()
    if not isinstance(spec, Mapping):
        raise ValueError("motion spec must be a mapping")
    allowed = {
        "id",
        "phase",
        "fps",
        "duration_s",
        "tags",
        "notes",
        "generator",
        "root",
        "joint_keyframes",
        "sine",
        "subject",
        "sequence",
    }
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"motion spec: unsupported keys {unknown}")
    clip_id = spec.get("id")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError("every motion spec needs a non-empty 'id'")
    phase = spec.get("phase", "unknown")
    if phase not in PHASE_LABELS:
        raise ValueError(f"{clip_id}: phase must be one of {list(PHASE_LABELS)}, got {phase!r}")
    fps = float(spec.get("fps", 60.0))
    duration = float(spec.get("duration_s", 1.0))
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"{clip_id}: fps must be finite and positive, got {fps!r}")
    if not (math.isfinite(duration) and duration > 0):
        raise ValueError(f"{clip_id}: duration_s must be finite and positive, got {duration!r}")
    generator = str(spec.get("generator", "keyframes")).strip().lower()
    if generator not in ("keyframes", "sine"):
        raise ValueError(f"{clip_id}: generator must be 'keyframes' or 'sine', got {generator!r}")

    frames = max(2, int(round(duration * fps)) + 1)
    times = np.arange(frames, dtype=np.float64) / fps
    root_spec = spec.get("root", {})
    if not isinstance(root_spec, Mapping):
        raise ValueError(f"{clip_id}: 'root' must be a mapping")
    unknown_root = sorted(set(root_spec) - {"translation_keyframes", "rotation_keyframes"})
    if unknown_root:
        raise ValueError(f"{clip_id}: unsupported root keys {unknown_root}")
    origin = (
        (0.0, 0.0, 0.0)
        if reference_frame is None
        else (float(reference_frame[0]), float(reference_frame[1]), float(reference_frame[2]))
    )
    root_translation = _compile_channels(
        root_spec.get("translation_keyframes"),
        times,
        clip_id=f"{clip_id}.root.translation",
        fps=fps,
        default=np.array(origin, dtype=np.float64),
    )
    root_rotation = _compile_channels(
        root_spec.get("rotation_keyframes"),
        times,
        clip_id=f"{clip_id}.root.rotation",
        fps=fps,
        default=np.zeros(3),
    )

    joint_rotations = np.zeros((frames, skeleton.joint_count, 3), dtype=np.float64)
    if generator == "keyframes":
        joint_spec = spec.get("joint_keyframes", {})
        if not isinstance(joint_spec, Mapping):
            raise ValueError(f"{clip_id}: 'joint_keyframes' must be a mapping")
        for joint_name, program in joint_spec.items():
            if joint_name not in skeleton.joint_names:
                raise ValueError(f"{clip_id}: joint_keyframes names unknown joint {joint_name!r}")
            index = skeleton.index(joint_name)
            joint_rotations[:, index] = _compile_channels(
                program,
                times,
                clip_id=f"{clip_id}.{joint_name}",
                fps=fps,
                default=np.zeros(3),
            )
    else:
        joint_rotations = _compile_sine_channels(spec, times, skeleton, clip_id=clip_id)

    provenance = MotionProvenance(
        kind=FRAME_SOURCE_SCRIPTED,
        source_id="scripted",
        subject=str(spec.get("subject", "scripted")),
        sequence=str(spec.get("sequence", clip_id)),
        representation="scripted_axis_angle",
        license="project-internal",
        license_url="",
        notes=str(spec.get("notes", "")),
    )
    return MotionClip(
        clip_id=clip_id,
        fps=fps,
        joint_names=skeleton.joint_names,
        root_translation=root_translation,
        root_rotation=root_rotation,
        joint_rotations=joint_rotations,
        provenance=provenance,
        phase_labels=tuple([phase] * frames),
        tags=tuple(str(tag) for tag in spec.get("tags", [])),
        metadata={"generator": generator, "declared_duration_s": duration},
    )


def _compile_sine_channels(
    spec: Mapping[str, Any],
    times: np.ndarray,
    skeleton: SkeletonTopology,
    *,
    clip_id: str,
) -> np.ndarray:
    block = spec.get("sine")
    if not isinstance(block, Mapping):
        raise ValueError(f"{clip_id}: a sine generator needs a 'sine' mapping")
    unknown = sorted(set(block) - {"channels"})
    if unknown:
        raise ValueError(f"{clip_id}: unsupported sine keys {unknown}")
    channels = block.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{clip_id}: 'sine.channels' must be a non-empty list")
    output = np.zeros((len(times), skeleton.joint_count, 3), dtype=np.float64)
    for channel in channels:
        if not isinstance(channel, Mapping):
            raise ValueError(f"{clip_id}: each sine channel must be a mapping")
        unknown_channel = sorted(
            set(channel)
            - {"joint", "axis", "amplitude_deg", "frequency_hz", "phase_deg", "offset_deg"}
        )
        if unknown_channel:
            raise ValueError(f"{clip_id}: unsupported sine channel keys {unknown_channel}")
        joint = channel.get("joint")
        if joint not in skeleton.joint_names:
            raise ValueError(f"{clip_id}: sine channel names unknown joint {joint!r}")
        axis = str(channel.get("axis", "y")).lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"{clip_id}: sine channel axis must be x, y or z, got {axis!r}")
        amplitude = math.radians(float(channel.get("amplitude_deg", 0.0)))
        frequency = float(channel.get("frequency_hz", 1.0))
        phase = math.radians(float(channel.get("phase_deg", 0.0)))
        offset = math.radians(float(channel.get("offset_deg", 0.0)))
        if not math.isfinite(frequency) or frequency < 0:
            raise ValueError(f"{clip_id}: sine frequency must be finite and non-negative")
        value = offset + amplitude * np.sin(2.0 * math.pi * frequency * times + phase)
        output[:, skeleton.index(str(joint)), "xyz".index(axis)] = value
    return output


def _compile_channels(
    program: object,
    times: np.ndarray,
    *,
    clip_id: str,
    fps: float,
    default: np.ndarray,
) -> np.ndarray:
    """Interpolate a ``{times, values, interpolation}`` program onto ``times``."""

    if program is None:
        return np.tile(default, (len(times), 1))
    if not isinstance(program, Mapping):
        raise ValueError(f"{clip_id}: channel program must be a mapping")
    unknown = sorted(set(program) - {"times_s", "values", "interpolation"})
    if unknown:
        raise ValueError(f"{clip_id}: unsupported channel keys {unknown}")
    key_times = program.get("times_s")
    values = program.get("values")
    if key_times is None:
        # A bare value list means "one value per frame" (or a single held value).
        if values is None:
            raise ValueError(f"{clip_id}: channel needs 'times_s' or 'values'")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{clip_id}: frame-sampled values must be (N, 3), got {array.shape}")
        if array.shape[0] == len(times):
            return array
        if array.shape[0] == 1:
            return np.tile(array[0], (len(times), 1))
        raise ValueError(
            f"{clip_id}: {array.shape[0]} values for {len(times)} frames; give 'times_s' "
            "to interpolate"
        )
    key_times_array = np.asarray(key_times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if key_times_array.ndim != 1 or len(key_times_array) < 2:
        raise ValueError(f"{clip_id}: 'times_s' must list at least two keyframe times")
    if value_array.shape != (len(key_times_array), 3):
        raise ValueError(
            f"{clip_id}: 'values' must be ({len(key_times_array)}, 3), got {value_array.shape}"
        )
    if not np.all(np.isfinite(key_times_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError(f"{clip_id}: keyframes contain a non-finite value")
    if key_times_array[0] > 1e-9 or np.any(np.diff(key_times_array) <= 0):
        raise ValueError(
            f"{clip_id}: keyframe times must start at 0 and strictly increase, got "
            f"{key_times_array.tolist()}"
        )
    method = str(program.get("interpolation", "smoothstep")).lower()
    if method not in _INTERPOLATIONS:
        raise ValueError(f"{clip_id}: interpolation must be one of {_INTERPOLATIONS}")
    upper = np.clip(
        np.searchsorted(key_times_array, times, side="left"), 1, len(key_times_array) - 1
    )
    lower = upper - 1
    span = key_times_array[upper] - key_times_array[lower]
    weight = np.clip((times - key_times_array[lower]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    if method == "smoothstep":
        weight = weight * weight * (3.0 - 2.0 * weight)
    # ``weight`` is clamped, so times past the final keyframe hold the last value.
    return value_array[lower] + (value_array[upper] - value_array[lower]) * weight[:, None]


def motion_library_from_mapping(
    payload: Mapping[str, Any], *, topology: SkeletonTopology | None = None
) -> dict[str, MotionClip]:
    """Compile every motion program in a parsed YAML mapping."""

    if not isinstance(payload, Mapping):
        raise ValueError("motion library must be a mapping")
    unknown = sorted(set(payload) - {"version", "notes", "defaults", "motions"})
    if unknown:
        raise ValueError(f"motion library: unsupported keys {unknown}")
    entries = payload.get("motions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("motion library must declare a non-empty 'motions' list")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("motion library 'defaults' must be a mapping")
    library: dict[str, MotionClip] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each motion entry must be a mapping")
        merged = {**defaults, **entry}
        clip = compile_scripted_clip(merged, topology=topology)
        if clip.clip_id in library:
            raise ValueError(f"duplicate motion id {clip.clip_id!r}")
        library[clip.clip_id] = clip
    return library


def load_motion_library(
    path: str | Path, *, topology: SkeletonTopology | None = None
) -> dict[str, MotionClip]:
    """Load a motion library YAML file."""

    motion_path = Path(path)
    if not motion_path.is_file():
        raise FileNotFoundError(f"motion library not found: {motion_path}")
    with motion_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{motion_path}: top level must be a mapping")
    library = motion_library_from_mapping(payload, topology=topology)
    LOGGER.info("loaded %d motion clips from %s", len(library), motion_path)
    return library
