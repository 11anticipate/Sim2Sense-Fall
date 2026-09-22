"""Load, retarget and screen local AMASS ``.npz`` sequences.

AMASS is a registration-gated dataset, so this module deliberately operates on a
user-provided local directory.  It never downloads or invents a sequence.  The
loader validates the public AMASS fields, records a source hash, retargets the
SMPL-H body block to the project's SMPL topology, and exposes a conservative
kinematic screen for fall candidates.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .motion import FRAME_SOURCE_AMASS, MotionClip, MotionProvenance, retarget_amass_clip
from .rig import HumanRigPlan, forward_kinematics
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle

__all__ = [
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


def normalize_root_motion(clip: MotionClip) -> MotionClip:
    """Anchor an AMASS clip to the project's local preview frame.

    AMASS ``trans`` and the global root rotation are sequence coordinates.  They
    must not be used as an apartment world pose: the first capture sample can be
    metres away from the selected room and its global orientation is often the
    SMPL-H-to-world calibration pose.  The preview therefore keeps the motion
    relative to that first sample while leaving the local joint rotations intact::

        t_rel[k] = t[k] - t[0]
        R_rel[k] = R[k] @ R[0].T

    The returned clip is a new immutable contract value.  This helper is kept
    out of :func:`retarget_amass_clip` so raw import tests and exported source
    provenance retain the original AMASS coordinates.
    """

    if clip.provenance.kind != FRAME_SOURCE_AMASS:
        raise ValueError("normalize_root_motion expects an AMASS clip")
    first_rotation = axis_angle_to_matrix(clip.root_rotation[0])
    relative_rotation = np.stack(
        [
            matrix_to_axis_angle(axis_angle_to_matrix(value) @ first_rotation.T)
            for value in clip.root_rotation
        ]
    )
    relative_translation = (
        np.asarray(clip.root_translation, dtype=np.float64) - clip.root_translation[0]
    )
    metadata = {
        **dict(clip.metadata),
        "root_motion_normalization": {
            "anchor": "first_frame",
            "translation": "t[k] - t[0]",
            "rotation": "R[k] @ R[0].T",
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
    source_up_axis: str = "y",
    target_up_axis: str = "z",
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
        notes=f"local source {source}; gender={gender}; source_up_axis={source_up_axis}",
    )
    generated_id = clip_id or f"amass__{source.stem}"
    return retarget_amass_clip(
        poses,
        trans,
        clip_id=generated_id,
        fps=_fps(payload, source),
        provenance=provenance,
        betas=betas,
        source_up_axis=source_up_axis,
        target_up_axis=target_up_axis,
    )


def load_amass_library(
    root: str | Path,
    *,
    limit: int | None = None,
    source_up_axis: str = "y",
    target_up_axis: str = "z",
) -> dict[str, MotionClip]:
    """Load every valid local AMASS file below ``root`` in deterministic order."""

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
    for path in paths:
        clip = load_amass_clip(
            path,
            root=directory,
            source_up_axis=source_up_axis,
            target_up_axis=target_up_axis,
        )
        if clip.clip_id in clips:
            raise ValueError(f"duplicate AMASS clip id {clip.clip_id!r}")
        clips[clip.clip_id] = clip
    return clips


def load_amass_clip_by_id(
    root: str | Path,
    clip_id: str,
    *,
    source_up_axis: str = "y",
    target_up_axis: str = "z",
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
        source_up_axis=source_up_axis,
        target_up_axis=target_up_axis,
    )


class AmassScreenResult:
    """Serializable result of the kinematic fall-candidate screen."""

    __slots__ = (
        "accepted",
        "peak_trunk_angle_deg",
        "root_drop_m",
        "peak_down_speed_m_s",
        "axis_residual_rad",
        "reason",
    )

    def __init__(
        self,
        *,
        accepted: bool,
        peak_trunk_angle_deg: float,
        root_drop_m: float,
        peak_down_speed_m_s: float,
        axis_residual_rad: float,
        reason: str,
    ) -> None:
        self.accepted = bool(accepted)
        self.peak_trunk_angle_deg = float(peak_trunk_angle_deg)
        self.root_drop_m = float(root_drop_m)
        self.peak_down_speed_m_s = float(peak_down_speed_m_s)
        self.axis_residual_rad = float(axis_residual_rad)
        self.reason = str(reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "peak_trunk_angle_deg": self.peak_trunk_angle_deg,
            "root_drop_m": self.root_drop_m,
            "peak_down_speed_m_s": self.peak_down_speed_m_s,
            "axis_residual_rad": self.axis_residual_rad,
            "reason": self.reason,
        }


def screen_amass_clip(
    clip: MotionClip,
    plan: HumanRigPlan,
    *,
    trunk_angle_deg: float = 60.0,
    root_drop_m: float = 0.20,
    down_speed_m_s: float = 1.0,
    max_axis_residual_deg: float = 25.0,
) -> AmassScreenResult:
    """Screen a retargeted clip for a physically usable fall candidate.

    This is a candidate screen, not a clinical label.  The current Isaac rig is
    single-axis, so clips whose discarded off-axis rotation exceeds the configured
    bound are reported as incompatible instead of silently projected into another
    action.
    """

    if clip.provenance.kind != FRAME_SOURCE_AMASS:
        raise ValueError("screen_amass_clip expects an AMASS clip")
    residual = 0.0
    link_series: list[np.ndarray] = []
    root_series = np.asarray(plan.spawn_root_position, dtype=np.float64) + clip.root_translation
    for frame in range(clip.frame_count):
        values: dict[str, float] = {}
        for joint in plan.joints:
            vector = clip.rotation_of(frame, joint.chain_joint)
            axis_index = "xyz".index(joint.axis)
            values[joint.name] = float(vector[axis_index])
            residual = max(residual, float(np.linalg.norm(np.delete(vector, axis_index))))
        poses = forward_kinematics(
            plan,
            values,
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
    compatible = math.degrees(residual) <= max_axis_residual_deg
    accepted = compatible and float(np.max(angles, initial=0.0)) >= trunk_angle_deg and (
        root_drop >= root_drop_m or peak_drop >= down_speed_m_s
    )
    if not compatible:
        reason = (
            f"off-axis rotation {math.degrees(residual):.1f} deg exceeds "
            f"{max_axis_residual_deg:.1f} deg"
        )
    elif accepted:
        reason = "kinematic fall candidate; verify with physics and inspect manually"
    else:
        reason = "upright or slow lowering motion; no fall candidate"
    return AmassScreenResult(
        accepted=accepted,
        peak_trunk_angle_deg=float(np.max(angles, initial=0.0)),
        root_drop_m=root_drop,
        peak_down_speed_m_s=peak_drop,
        axis_residual_rad=residual,
        reason=reason,
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
