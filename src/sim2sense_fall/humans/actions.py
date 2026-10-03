"""Validated AMASS posture targets, motion playback and keyboard action states."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .amass import (
    crop_amass_clip,
    ground_amass_clip,
    load_amass_clip,
    normalize_root_motion,
    sha256_file,
)
from .contact_control import capsule_bottom, has_collision_geometry, sideways_leg_dofs
from .rig import (
    AXIS_PROJECTION_TOLERANCE_RAD,
    HumanRigPlan,
    forward_kinematics,
    joint_values_from_clip,
)
from .rotations import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_slerp,
    quaternion_to_matrix,
    rotation_about_axis,
)
from .teleop import TeleopTarget, _move_capsule

# Controller modes owned by the action layer. keyboard.py consults this set to
# decide when NOT to reset the stance/contact corrections; new posture actions
# must be added here or the locomotion corrections will fight them.
POSTURE_TRANSITION_MODES = frozenset(
    {
        "crouching",
        "crouch",
        "bending",
        "bend",
        "sitting",
        "sit",
        "standing_up",
        "getting_up",
    }
)
_POSTURE_ING_MODES = {"crouch": "crouching", "bend": "bending", "sit": "sitting"}
_SMOOTHSTEP_EPS = 1e-10
# Blend window from the captured (fallen) pose into the get-up clip. Shorter
# than the posture transition: the fallen body is already on the floor near the
# clip's first pose, so a long blend would just delay the recovery.
_CLIP_BLEND_S = 0.5
# Command-stream mechanism recorded on a legacy (drive-release) fall event, kept
# verbatim for continuity with previously exported sessions.
_LEGACY_FALL_MECHANISM = "root_assist_off_position_drives_scaled_backward_force_pulse"
# A reversed/played fall clip is commanded like a get-up replay: drives stay
# engaged and the recovery root assistance holds the pelvis on the clip's
# grounded curve until the body actually reaches the latched fallen state.
_FALL_REPLAY_MECHANISM = "anchored_fall_clip_replay"

LOGGER = logging.getLogger(__name__)
# Retarget/projection results are pure functions of (source file, spec, rig
# plan, physics dt); caching them to disk removes the two largest startup
# costs -- the per-frame retarget loop (~8 s for an 830-frame get-up clip)
# and the posture contact projection (~1 s each). Bump the version to
# invalidate every cached entry after an algorithm change.
_CLIP_CACHE_VERSION = "v1"


def _clip_cache_dir() -> Path:
    override = os.environ.get("SIM2SENSE_CLIP_CACHE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "artifacts/humans/clip_cache"


def _plan_fingerprint(plan: HumanRigPlan) -> str:
    """Hash everything the retarget/projection results depend on."""

    digest = hashlib.sha256()
    digest.update(";".join(plan.dof_names).encode())
    digest.update(
        np.ascontiguousarray(np.asarray(plan.rest_joint_positions, dtype=np.float64)).tobytes()
    )
    digest.update(np.ascontiguousarray(np.asarray(plan.spawn_root_position)).tobytes())
    for link in plan.links:
        if link.capsule is not None:
            digest.update(
                f"{link.name}:{link.capsule.radius_m:.6f}:{link.capsule.center}".encode()
            )
    return digest.hexdigest()[:20]


def _cache_key(kind: str, source_file: Any, spec_fields: dict[str, Any], plan: HumanRigPlan,
               dt_s: float | None) -> str:
    payload = {
        "v": _CLIP_CACHE_VERSION,
        "kind": kind,
        "source_sha256": sha256_file(Path(source_file)),
        "spec": spec_fields,
        "dt_s": dt_s,
        "plan": _plan_fingerprint(plan),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def _cache_load(name: str) -> dict[str, np.ndarray] | None:
    path = _clip_cache_dir() / f"{name}.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (FileNotFoundError, OSError, ValueError):
        return None


def _cache_save(name: str, arrays: dict[str, np.ndarray]) -> None:
    try:
        directory = _clip_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{name}.npz", **arrays)
    except OSError as exc:
        LOGGER.debug("clip cache write skipped: %s", exc)


@dataclass(frozen=True)
class ActionConfig:
    transition_s: float
    stopped_speed_m_s: float
    fall_force_n: float
    fall_duration_s: float
    fall_control_scale: float
    fallen_height_fraction: float
    fallen_tilt_deg: float
    # Fraction of the authored joint damping kept while falling. The position
    # drives are released (fall_control_scale 0); 0 also releases the viscous
    # term so the limbs fold under gravity alone with no drive torque at all
    # (user decision 2026-09-26: a PD damped collapse reads as a statue).
    # Zero is solver-safe ONLY when the rig was authored with a per-joint
    # maxJointVelocity clamp, which bounds the ballistic joint speeds and thus
    # the floor-impact energy that NaN'd the solver at 2864 deg/s mid-fall.
    # Positive values keep a bounded viscous brake instead (see
    # HumanRuntime.set_control_scale for the measured full-damping crumple).
    fall_damping_scale: float = 0.15
    # Ground the replayed get-up by contact closure instead of trusting the clip's
    # absolute root-height curve. ``load_action_clip`` grounds a clip **once**, on
    # its first frame, so every later frame carries that mocap subject's own pelvis
    # trajectory. Measured on the shipped ``CMU/140/140_01`` source, that curve
    # floats 3-14 cm above the floor for all 830 frames and is still 11.8 cm high on
    # the final standing frame (``scripts/humans/diagnose_getup_float.py``,
    # ``artifacts/humans/getup_float/``). With the feet that far off the ground
    # nothing can push, so the pelvis actuator carries 100% of body weight and the
    # body rises on a wire -- the defect this closes.
    get_up_contact_closure: bool = True
    # Hand the last stretch of the fall replay to physics: once the *commanded*
    # trunk tilt crosses this angle the drives are released and the body simply
    # topples. Two measured defects motivate the trigger and its choice:
    # (a) a root-height trigger at 0.6 m (GPU run 4) cut the clip at its
    #     hands-and-knees phase -- trunk still upright -- and the released body
    #     creaked down into a stable KNEEL over ~2 s (trunk <= 50 deg, outcome
    #     impacted_not_fallen): kneeling down, not falling;
    # (b) no trigger at all (GPU run 2) played the clip to its lying end, whose
    #     vertical velocity never exceeds -1.4 m/s (a 0.5 m plateau included) --
    #     a deliberate lowering, per the user's 2026-09-27 review.
    # The clip's own tilt curve tips through ~42 deg at 0.45 m root height
    # (fall_replay_v2 timeline); released there, the remaining motion is a
    # gravity topple from the tipping point, which is the biomechanical
    # structure of a real fall (controlled descent, then collapse). None
    # replays the clip to its own end (the pre-fix behaviour).
    fall_replay_release_tilt_deg: float | None = None
    # Route-A increment (2026-09-27): an F press WHILE WALKING is a trip, not a
    # replay. The gait keeps stepping (real stumble steps), the pelvis assist
    # fades (keyboard keys the blend off the "tripping" mode), and a bounded
    # forward shove -- a caught foot -- supplies the perturbation. Two knobs:
    # the shove's peak force and how long a stumble may fight for balance
    # before the trip is declared survived and control returns to locomotion.
    trip_force_n: float = 180.0
    trip_window_s: float = 1.6
    # The other half of the balance failure: the leg drives are strong enough
    # to yank a snagged foot back onto its swing trajectory (450 N dragged the
    # ankle and peak tilt still stayed at 10.7 deg, trip_walk_v4), so during
    # the trip window the position drives run at this reduced scale -- muscle
    # strength momentarily gone. A measured fall then releases them fully
    # (fall_control_scale); a survived stumble restores 1.0.
    trip_control_scale: float = 0.35
    # The orientation servo (rotation_stiffness 3000 Nm/rad on BOTH assist
    # profiles, immune to the mode blend) is the system's real balance
    # controller -- it held peak tilt at 9-11 deg through every trip variant
    # (trip_walk_v1..v5). During the trip window the pitch authority scales by
    # this factor, which is what lets the snag + walking momentum actually
    # rotate the body; a measured fall then removes it entirely on release.
    trip_torque_scale: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.get_up_contact_closure, bool):
            raise ValueError("get_up_contact_closure must be a bool")
        if self.fall_replay_release_tilt_deg is not None and (
            not np.isfinite(self.fall_replay_release_tilt_deg)
            or not 0 < self.fall_replay_release_tilt_deg < 90
        ):
            raise ValueError("fall_replay_release_tilt_deg must be finite in (0, 90)")
        if not np.isfinite([v for v in vars(self).values() if v is not None]).all():
            raise ValueError("action settings must be finite")
        if (
            min(self.transition_s, self.stopped_speed_m_s, self.fall_force_n, self.fall_duration_s)
            <= 0
        ):
            raise ValueError("action timings and force must be positive")
        if not 0 <= self.fall_control_scale <= 1:
            raise ValueError("fall control scale must be in [0,1]")
        if not 0 <= self.fall_damping_scale <= 1:
            raise ValueError(
                "fall damping scale must be in [0, 1]: 0 is fully passive and requires the "
                "rig to carry a maxJointVelocity clamp, above 1 would amplify the damping"
            )
        if not 0 < self.fallen_height_fraction < 1 or not 0 < self.fallen_tilt_deg < 90:
            raise ValueError("invalid fallen posture thresholds")
        if not np.isfinite(self.trip_force_n) or self.trip_force_n <= 0:
            raise ValueError("trip_force_n must be finite and positive")
        if not np.isfinite(self.trip_window_s) or self.trip_window_s <= 0:
            raise ValueError("trip_window_s must be finite and positive")
        if not 0 <= self.trip_control_scale <= 1:
            raise ValueError("trip_control_scale must be in [0, 1]")
        if not 0 <= self.trip_torque_scale <= 1:
            raise ValueError("trip_torque_scale must be in [0, 1]")


@dataclass(frozen=True)
class Posture:
    joints: np.ndarray
    height_m: float
    tilt: np.ndarray
    provenance: dict[str, Any]
    # Links whose capsule bottoms define the grounded reference height. The
    # default grounds the feet (the historical crouch rule); declared-contact
    # postures (sit) include their anchor and support limbs.
    contacts: tuple[str, ...] = ("left_ankle", "right_ankle")


def _without_yaw(tilt: np.ndarray) -> np.ndarray:
    """Strip the world-yaw component from a tilt axis-angle.

    The get-up clip turns on its own while rising; replayed against the
    fallen heading that turn shows up as a slow spin followed by a snap
    back to the teleop heading at blend-out. Removing the clip's yaw keeps
    the replay on the fallen heading throughout, so recovery never spins.
    """

    rotation = axis_angle_to_matrix(tilt)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return matrix_to_axis_angle(rotation_about_axis("z", -yaw) @ rotation)


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _contact_chains(side: str, kind: str) -> set[str]:
    return {f"{side}_{name}" for name in ("shoulder", "elbow", "wrist")} if kind == "arm" else {
        f"{side}_{name}" for name in ("hip", "knee", "ankle")
    }


def _project_posture_contacts(
    plan: HumanRigPlan,
    q: np.ndarray,
    tilt: np.ndarray,
    contacts: list[str],
    *,
    max_correction_rad: float = 0.6,
    iterations: int = 80,
    tolerance_m: float = 0.002,
    rounds: int = 4,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Ground a posture by its declared contact set and pull floaters down.

    The AMASS source pose is one frame of a motion performed in the subject's
    own contact geometry; after retargeting, the contacts that should share the
    floor plane can be tens of millimetres apart (measured on the shipped sit
    source: 48 mm spread across feet/hands/pelvis). Shipping such a pose as a
    hold target recreates the crouch-hold failure: the highest contact floats
    with zero impulse while the lowest one, pressed through the floor, carries
    the load alone.

    The root link ("pelvis"), when declared, is the anchor: the root height is
    set so its capsule bottom touches the floor, and every other declared
    contact is pulled down to the floor with the same bounded weighted IK the
    swing-lift bake uses. Limb IK cannot move the root, so the anchor cannot
    float; a contact the limbs cannot reach is reported as a residual instead
    of silently shipping a floating target.
    """

    for name in contacts:
        plan.link(name)  # raises for an unknown link
    anchor = "pelvis" if "pelvis" in contacts else None
    movable = [name for name in contacts if name != anchor]

    def bottoms(values: np.ndarray, height: float) -> dict[str, float]:
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, values, strict=True)),
            root_position=(0.0, 0.0, height),
            root_rotation=tilt,
        )
        return {name: capsule_bottom(plan, poses, name) for name in contacts}

    height = 0.0
    residual = {}
    used_rounds = 0
    for round_index in range(1, rounds + 1):
        used_rounds = round_index
        measured = bottoms(q, height)
        if anchor is None:
            # No root link declared: ground by the lowest declared contact.
            anchor_name = min(measured, key=measured.get)
            height -= measured[anchor_name]
        else:
            height -= measured[anchor]
        measured = bottoms(q, height)
        moved = False
        for name in movable:
            if measured[name] > tolerance_m:
                side, _, kind = name.partition("_")
                chain = _contact_chains(side, kind if kind in {"hand", "ankle"} else "leg")
                penalised = (
                    frozenset(sideways_leg_dofs(plan)) if name.endswith("ankle") else frozenset()
                )
                q = _move_capsule(
                    plan,
                    q,
                    name,
                    chain,
                    target_bottom=0.0,
                    root_height=height,
                    root_tilt=tilt,
                    max_correction_rad=max_correction_rad,
                    iterations=iterations,
                    penalised=penalised,
                )
                moved = True
        residual = bottoms(q, height)
        if not moved and all(abs(value) <= tolerance_m for value in residual.values()):
            break
    # Height was grounded against the pre-IK pose; re-ground once against the
    # final pose so the anchor rests exactly on the floor.
    residual = bottoms(q, height)
    height -= min(residual.values())
    residual = bottoms(q, height)
    return (
        q,
        height,
        {
            "method": "bounded_ik_contact_projection",
            "rounds": used_rounds,
            "tolerance_m": tolerance_m,
            "residual_m": {name: round(value, 5) for name, value in residual.items()},
            "max_residual_m": round(max(abs(v) for v in residual.values()), 5),
        },
    )


def load_posture(spec: dict[str, Any], plan: HumanRigPlan) -> Posture:
    """Use an explicitly named source pose, not a whole out-of-range crouch walk.

    This is a derived stationary posture with a separately generated transition.
    It must not be reported as faithful playback of the entire AMASS sequence.

    With ``contacts`` declared, the pose is additionally projected onto the
    floor through :func:`_project_posture_contacts` and the residuals are
    recorded; without it, the height follows the historical rule (lowest ankle
    capsule bottom grounded, no joint modification).

    Results are cached on disk keyed by (source file, spec, rig plan): the
    bounded-IK contact projection is the third-largest startup cost and a pure
    function of those inputs.
    """
    contacts_spec = spec.get("contacts")
    cache_name = _cache_key(
        "posture",
        spec["file"],
        {
            "start_s": float(spec["start_s"]),
            "duration_s": float(spec["duration_s"]),
            "contacts": list(contacts_spec) if contacts_spec is not None else None,
        },
        plan,
        None,
    )
    cached = _cache_load(cache_name)
    if cached is not None:
        return Posture(
            cached["joints"],
            float(cached["height_m"][0]),
            cached["tilt"],
            json.loads(str(cached["provenance_json"][0])),
            tuple(str(name) for name in cached["contacts"]),
        )
    clip = normalize_root_motion(
        crop_amass_clip(
            load_amass_clip(spec["file"]), start_s=spec["start_s"], duration_s=spec["duration_s"]
        )
    )
    q, _ = joint_values_from_clip(clip, 0, plan)
    lower = np.deg2rad([j.lower_deg for j in plan.joints])
    upper = np.deg2rad([j.upper_deg for j in plan.joints])
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("selected posture exceeds rig limits")
    rotation = axis_angle_to_matrix(clip.root_rotation[0])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    tilt = matrix_to_axis_angle(rotation_about_axis("z", -yaw) @ rotation)
    contacts = spec.get("contacts")
    if contacts is None:
        poses = forward_kinematics(
            plan, dict(zip(plan.dof_names, q, strict=True)), root_rotation=tilt
        )
        height = -min(capsule_bottom(plan, poses, f"{s}_ankle") for s in ("left", "right"))
        derivation = "first_pose_of_authorized_crop_stationary_target_with_smooth_transition"
        projection: dict[str, Any] | None = None
        contact_links: tuple[str, ...] = ("left_ankle", "right_ankle")
    else:
        q, height, projection = _project_posture_contacts(plan, q, tilt, list(contacts))
        if np.any(q < lower) or np.any(q > upper):
            raise ValueError("contact projection drove the posture outside rig limits")
        derivation = "first_pose_of_authorized_crop_projected_onto_declared_floor_contacts"
        contact_links = tuple(contacts)
    provenance = {
        "source": clip.provenance.as_dict(),
        "metadata": dict(clip.metadata),
        "derivation": derivation,
        "height_method": (
            "declared_contact_projection" if contacts else "minimum_foot_collision_support"
        ),
        "height_m": height,
        **({"contacts": list(contacts), "projection": projection} if contacts else {}),
    }
    _cache_save(
        cache_name,
        {
            "joints": q,
            "height_m": np.asarray([height]),
            "tilt": tilt,
            "contacts": np.asarray(contact_links),
            "provenance_json": np.asarray([json.dumps(provenance, ensure_ascii=False)]),
        },
    )
    return Posture(q, height, tilt, provenance, contact_links)


@dataclass(frozen=True)
class ActionClip:
    """A ground-anchored AMASS sequence replayed as one assisted action."""

    joints: np.ndarray  # (T, D) retargeted joint targets
    heights_m: np.ndarray  # (T,) absolute root height, first frame grounded
    offsets_xy: np.ndarray  # (T, 2) horizontal offset from the first frame
    tilts: np.ndarray  # (T, 3) yaw-stripped root tilt
    frame_dt_s: float
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.joints)
        if count < 2 or self.joints.ndim != 2:
            raise ValueError("action clip must have at least two joint frames")
        if self.heights_m.shape != (count,) or self.offsets_xy.shape != (count, 2):
            raise ValueError("action clip height/offset arrays must align with joint frames")
        if self.tilts.shape != (count, 3):
            raise ValueError("action clip tilts must align with joint frames")
        if not np.isfinite(self.joints).all() or not np.isfinite(self.heights_m).all():
            raise ValueError("action clip samples must be finite")
        if not np.isfinite(self.offsets_xy).all() or not np.isfinite(self.tilts).all():
            raise ValueError("action clip samples must be finite")
        if not np.isfinite(self.frame_dt_s) or self.frame_dt_s <= 0:
            raise ValueError("action clip frame period must be finite and positive")

    @property
    def duration_s(self) -> float:
        return (len(self.joints) - 1) * self.frame_dt_s


def load_action_clip(spec: dict[str, Any], plan: HumanRigPlan, *, dt_s: float) -> ActionClip:
    """Load and retarget a full AMASS sequence for assisted playback.

    The clip is yaw-anchored by :func:`normalize_root_motion` (first frame at
    the origin facing +X), grounded once on the first frame's collision
    geometry, and resampled onto the physics grid. Root height is absolute;
    horizontal offsets stay relative so the runtime can anchor the replay at
    the measured fallen position and heading.

    Optional spec keys extend this for the fall replay (``fall_replay``):
    ``start_s``/``duration_s`` crop the source before anchoring, ``speed``
    (integer >= 1) compresses time by resampling at ``speed`` x the physics
    rate and then decimating, and ``reverse`` plays the clip backwards --
    the shipped fall source IS the get-up clip reversed, which the survey
    guarantees to be inside rig limits (axis residual 0, limit excess 0).
    A reversed clip is re-anchored so its own first frame sits at the spawn
    height and the origin: the replay anchors there.
    """
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("action clip physics step must be finite and positive")
    spec_fields = {
        key: (None if spec.get(key) is None else float(spec[key]))
        for key in ("start_s", "duration_s")
    }
    spec_fields["speed"] = int(spec.get("speed", 1))
    spec_fields["reverse"] = bool(spec.get("reverse", False))
    cache_name = _cache_key("clip", spec["file"], spec_fields, plan, dt_s)
    cached = _cache_load(cache_name)
    if cached is not None:
        return ActionClip(
            cached["joints"],
            cached["heights_m"],
            cached["offsets_xy"],
            cached["tilts"],
            float(cached["frame_dt_s"]),
            json.loads(str(cached["provenance_json"][0])),
        )
    clip = load_amass_clip(spec["file"])
    if spec.get("start_s") is not None or spec.get("duration_s") is not None:
        clip = crop_amass_clip(
            clip,
            start_s=float(spec.get("start_s", 0.0)),
            duration_s=None if spec.get("duration_s") is None else float(spec["duration_s"]),
        )
    clip = normalize_root_motion(clip)
    speed = int(spec.get("speed", 1))
    if speed < 1:
        raise ValueError("action clip speed must be >= 1")
    # Time compression: resample the source at (1/dt)/speed, i.e. consecutive
    # frames are ``speed`` source-seconds-apart-of-motion apart, and play each
    # one for one physics step -- wall time becomes source_time / speed. The
    # physics grid still receives exactly one target per step.
    clip = ground_amass_clip(clip, plan, support_z_m=0.0).resample(
        (1.0 / dt_s) / speed, method="slerp"
    )
    joints = []
    lower = np.deg2rad([j.lower_deg for j in plan.joints])
    upper = np.deg2rad([j.upper_deg for j in plan.joints])
    for frame in range(clip.frame_count):
        values, worst = joint_values_from_clip(clip, frame, plan, axis_tolerance_rad=math.inf)
        if worst > AXIS_PROJECTION_TOLERANCE_RAD:
            raise ValueError(
                f"action clip frame {frame} is not expressible by this rig "
                f"(axis residual {worst:.2e} rad)"
            )
        joints.append(values)
    joints = np.stack(joints)
    if np.any(joints < lower - 1e-6) or np.any(joints > upper + 1e-6):
        excess = float(np.maximum(lower - joints, joints - upper).max())
        raise ValueError(f"action clip exceeds rig limits by {np.degrees(excess):.3f} deg")
    heights = (
        np.asarray(plan.spawn_root_position, dtype=np.float64)[2] + clip.root_translation[:, 2]
    )
    offsets = np.asarray(clip.root_translation, dtype=np.float64)[:, :2].copy()
    tilts = np.asarray(clip.root_rotation, dtype=np.float64).copy()
    notes: dict[str, Any] = {}
    if spec.get("reverse"):
        joints, heights, offsets, tilts = (
            values[::-1].copy() for values in (joints, heights, offsets, tilts)
        )
        # Re-anchor the reversed first frame (the source's own last frame) so the
        # runtime can anchor it at the measured trigger pose like any clip start.
        offsets = offsets - offsets[0]
        heights = heights - heights[0] + float(np.asarray(plan.spawn_root_position)[2])
        notes["reverse"] = True
    if speed > 1:
        notes["speed"] = speed
    frame_dt_s = dt_s if speed > 1 else 1.0 / clip.fps
    provenance = {
        "source": clip.provenance.as_dict(),
        "metadata": dict(clip.metadata),
        "derivation": "full_clip_retargeted_grounded_resampled_assisted_playback",
        "duration_s": float((len(joints) - 1) * frame_dt_s),
        "frames": int(len(joints)),
        "root_travel_m": round(float(np.linalg.norm(offsets[-1] - offsets[0])), 4),
        **notes,
    }
    _cache_save(
        cache_name,
        {
            "joints": joints,
            "heights_m": heights,
            "offsets_xy": offsets,
            "tilts": tilts,
            "frame_dt_s": np.asarray(frame_dt_s),
            "provenance_json": np.asarray([json.dumps(provenance, ensure_ascii=False)]),
        },
    )
    return ActionClip(joints, heights, offsets, tilts, frame_dt_s, provenance)


def recovery_command_anchor(mode: str, measured_joints: np.ndarray) -> np.ndarray | None:
    """The pose the command stream must be re-anchored to at a recovery handover.

    While a fall is latched the position drives are scaled to ``fall_control_scale``
    (0 by default) and the commanded pose stays the **frozen pre-fall** one, so the
    body collapses somewhere else: measured on the shipped demo, the right knee sat
    at 136.4 deg against an 18.5 deg command. When ``G`` restores full drives, the
    per-step slew limiter in ``keyboard.py`` still chases that stale command, so the
    first recovery frame is a 114 deg error and the limbs snap straight while the
    pelvis is still on the floor -- the "kicking at the air" half of the defect, on
    top of the floating root height that ``get_up_contact_closure`` fixes.

    Returning the measured joints here re-anchors the limiter to the body's actual
    pose, which is the only state a recovery can legitimately start from.
    """

    if mode != "getting_up":
        return None
    if not np.isfinite(measured_joints).all():
        raise ValueError("recovery anchor requires finite measured joints")
    return np.asarray(measured_joints, dtype=np.float64).copy()


class ActionState:
    """Edge-triggered posture actions, motion playback and a latched physical fall.

    Posture actions (crouch/bend/sit) blend from a captured snapshot of the
    current pose to the target posture, so switching actions mid-transition is
    continuous by construction. With the rig available, the root height along
    the blend is derived by contact closure from the blended configuration
    itself, so the commanded geometry always has its support contacts exactly
    on the floor. ``get_up`` replays a grounded AMASS lying-to-standing clip
    anchored at the measured fallen position; it is only accepted from the
    latched fallen state and restores assistance on entry.
    """

    def __init__(
        self,
        config: ActionConfig,
        postures: dict[str, Posture] | Posture,
        playback_clip: ActionClip | None = None,
        plan: HumanRigPlan | None = None,
        *,
        fall_clip: ActionClip | None = None,
    ) -> None:
        if isinstance(postures, Posture):
            postures = {"crouch": postures}
        unknown = set(postures) - set(_POSTURE_ING_MODES)
        if unknown:
            raise ValueError(f"unknown posture actions {sorted(unknown)}")
        self.config, self.postures, self.playback_clip = config, postures, playback_clip
        # With the rig, posture transition/hold heights are derived from the
        # blended configuration by contact closure instead of blended linearly;
        # without it (state-machine fixtures) the historical linear blend runs.
        self.plan = plan
        # Optional anchored fall replay (F). When configured, the fall is
        # commanded as a grounded clip playback -- the body is *driven* through
        # a real descent instead of being released to gravity, which is what
        # produced the twisted supine rest (docs/progress.md 2026-09-27).
        self.fall_clip = fall_clip
        # Every link carrying collision geometry, for the get-up contact closure. A
        # recovery rests on whatever is underneath -- spine, hip, hand, then feet --
        # so the closure set cannot be a fixed pair of ankles; this is the same rule
        # ``load_action_clip`` applies once, to ground the clip's first frame.
        self.closure_links: tuple[str, ...] = (
            tuple(link.name for link in plan.links if has_collision_geometry(link))
            if plan is not None
            else ()
        )
        self.active_contacts: tuple[str, ...] = ("left_ankle", "right_ankle")
        # Horizontal support compensation: the root xy shifts so the FEET stay
        # planted while the blended configuration moves them in body frame.
        self.source_root_xy: np.ndarray | None = None
        self.source_foot_offsets: np.ndarray | None = None
        self.last_root_xy: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.phase = "idle"  # idle | falling | fallen | getting_up
        self.mode = "locomotion"
        self.requested: str | None = None
        self.blend = 0.0
        self.source: tuple[np.ndarray, float, np.ndarray] | None = None
        self.last_pose: tuple[np.ndarray, float, np.ndarray] | None = None
        self.playback: dict[str, Any] | None = None
        self.active_contacts: tuple[str, ...] = ("left_ankle", "right_ankle")
        self.source_root_xy: np.ndarray | None = None
        self.source_foot_offsets: np.ndarray | None = None
        self.last_root_xy: np.ndarray | None = None
        self.fall_time_s: float | None = None
        self.fall_heading_rad = 0.0
        self.trip_foot: str | None = None
        self.trip_release_pending = False
        self.fall_pose: np.ndarray | None = None
        self.impact_time_s: float | None = None
        self.active_event: dict[str, Any] | None = None

    @property
    def falling(self) -> bool:
        """True while a fall is latched (falling or lying fallen) until reset/get-up."""
        return self.phase in ("falling", "fallen")

    @property
    def tripping(self) -> bool:
        """A walk-coupled trip is in flight: gait stepping, drives fully on.

        The stumble may recover (window expiry returns to locomotion) or tip
        into a measured fall (:meth:`observe` transitions the phase), but while
        it lasts the body is a *walking* body being perturbed, not a released
        one -- stance corrections and the gait keep running.
        """

        return self.phase == "tripping"

    @property
    def fall_replay_engaged(self) -> bool:
        """A fall replay owns the command stream (drives and assistance stay on).

        Covers the whole replay lifetime -- advancing, the finished-but-still-
        descending window after the measured latch, until the keyboard consumes
        the release. Phase alone cannot decide this: ``observe()`` latches
        ``fallen`` mid-descent while the clip is still commanding.
        """

        return self.playback is not None and self.playback.get("kind") == "fall"

    @property
    def fall_replay_finished(self) -> bool:
        """The replay has consumed its clip; the body is held on the lying end."""

        return (
            self.playback is not None
            and self.playback.get("kind") == "fall"
            and bool(self.playback.get("finished", False))
        )

    def consume_trip_release(self) -> bool:
        """One-shot: a tripping body just measured its way into a fall.

        The keyboard releases the drives on ``True`` (from the trip's weakened
        scale down to ``fall_control_scale``) -- the fall latched AFTER the
        request, so the request-time release branch cannot do it. Returns
        ``True`` exactly once per trip.
        """

        if not self.trip_release_pending:
            return False
        self.trip_release_pending = False
        return True

    def consume_fall_replay_release(self) -> bool:
        """One-shot: the replay's authority over the command stream has ended.

        Two release policies: a full replay waits for the *measured* fallen
        latch (``measured_fallen``); a tilt-truncated replay releases as soon
        as the clip is cut at the tipping point (``at_tilt``) so the remaining
        topple is physics -- an involuntary gravity fall instead of a
        deliberate lowering. Returns ``True`` exactly once; :meth:`apply` then
        serves the frozen pose from ``fall_pose``.
        """

        if self.playback is None or self.playback.get("kind") != "fall":
            return False
        if not self.playback.get("finished", False):
            return False
        if self.playback.get("release_policy", "measured_fallen") == "at_tilt":
            self.playback = None
            return True
        if self.phase != "fallen":
            return False
        self.playback = None
        return True

    @property
    def suppresses_stance(self) -> bool:
        """Locomotion foot corrections must not run during fall or recovery.

        A trip in flight is still walking: the stance/planner corrections are
        part of the stumble (they try to keep the feet under the body), so only
        the latched fall/recovery phases suppress them.
        """

        return self.phase not in ("idle", "tripping")

    def _begin_blend(self, target_name: str | None) -> None:
        """Register the contact set for a starting blend.

        The closure height grounds the lowest active contact, so the set must
        cover everything the body may rest on now (previous hold) plus
        everything the target rests on -- during a sit descent the pelvis takes
        over as it becomes the lowest support, during the rise the feet do.
        """

        target_contacts = (
            self.postures[target_name].contacts
            if target_name is not None
            else ("left_ankle", "right_ankle")
        )
        self.active_contacts = tuple(sorted(set(self.active_contacts) | set(target_contacts)))

    def _capture_support_anchor(
        self, q: np.ndarray, tilt: np.ndarray, root_xy: np.ndarray
    ) -> None:
        """Record where the feet stand (body-frame offsets + root xy) at blend start."""

        poses0 = forward_kinematics(
            self.plan,
            dict(zip(self.plan.dof_names, q, strict=True)),
            root_rotation=tilt,
        )
        self.source_foot_offsets = np.array([
            poses0[name].translation[:2]
            for name in self.active_contacts if name.endswith("_ankle")
        ])
        self.source_root_xy = np.asarray(root_xy, dtype=np.float64)[:2].copy()

    def request(
        self,
        name: str,
        *,
        time_s: float,
        heading_rad: float,
        measured_joints: np.ndarray,
        measured_position: np.ndarray | None = None,
        measured_quaternion: np.ndarray | None = None,
        trip_foot: str | None = None,
    ) -> bool:
        if name not in {*_POSTURE_ING_MODES, "stand", "fall", "get_up", "trip"}:
            raise ValueError(f"unknown action {name}")
        if not np.isfinite([time_s, heading_rad]).all() or not np.isfinite(measured_joints).all():
            raise ValueError("action request requires finite measured state")
        if name == "trip":
            return self._request_trip(time_s, heading_rad, measured_joints, trip_foot=trip_foot)
        if name == "get_up":
            return self._request_get_up(
                time_s, heading_rad, measured_joints, measured_position, measured_quaternion
            )
        if self.suppresses_stance:
            # Falling/fallen: only R or get_up leave the state. Getting up:
            # posture requests would fight the replay.
            return False
        if name == "trip":
            return self._request_trip(time_s, heading_rad, measured_joints)
        if name == "fall":
            if self.fall_clip is not None:
                return self._request_fall_replay(
                    time_s, heading_rad, measured_joints, measured_position, measured_quaternion
                )
            self.fall_time_s, self.fall_heading_rad = time_s, heading_rad
            self.fall_pose = measured_joints.copy()
            self.phase = "falling"
            self.mode = "falling"
            self.requested = None
            self.blend = 0.0
            self.source = None
            self.playback = None
            self.active_contacts = ("left_ankle", "right_ankle")
            self.active_event = {
                "requested_time_s": time_s,
                "heading_rad": heading_rad,
                "impact_time_s": None,
                "fallen_time_s": None,
                "root_assistance": False,
                "control_scale": self.config.fall_control_scale,
            }
            self.history.append(self.active_event)
            return True
        if self.blend > 0 and self.last_pose is not None:
            # Mid-transition retarget: continue from the pose currently commanded.
            self.source = (
                self.last_pose[0].copy(),
                self.last_pose[1],
                self.last_pose[2].copy(),
            )
            self.blend = 0.0
            if self.plan is not None and self.last_root_xy is not None:
                self._capture_support_anchor(
                    self.last_pose[0], self.last_pose[2], self.last_root_xy
                )
        self.requested = None if name == "stand" else name
        self._begin_blend(self.requested)
        return True

    def _request_trip(
        self, time_s: float, heading_rad: float, measured_joints: np.ndarray,
        *, trip_foot: str | None = None,
    ) -> bool:
        """Start a walk-coupled trip: snag the swing foot, keep stepping, measure.

        Unlike the fall paths the drives stay fully engaged and the gait keeps
        stepping -- the 1-2 stumble steps are the body's real recovery attempt.
        The perturbation is a backward drag on the swing ankle (a caught toe),
        NOT a COM shove: pushing the centre of mass accelerates the feet along
        with it and nothing trips (GPU runs trip_walk_v1..v3, peak tilt 9.8 deg
        at 450 N). The snag stops the foot while the body's momentum carries,
        which is the physical trip torque. :meth:`observe` owns the two exits.
        """

        if self.phase != "idle":
            return False
        self.fall_time_s, self.fall_heading_rad = time_s, heading_rad
        self.fall_pose = measured_joints.copy()
        self.trip_foot = trip_foot
        self.phase = "tripping"
        self.mode = "tripping"
        self.requested = None
        self.blend = 0.0
        self.source = None
        self.playback = None
        self.active_contacts = ("left_ankle", "right_ankle")
        self.active_event = {
            "kind": "trip",
            "requested_time_s": time_s,
            "heading_rad": heading_rad,
            "impact_time_s": None,
            "fallen_time_s": None,
            "root_assistance": True,
            "control_scale": 1.0,
            "mechanism": "trip_shove_walk_recovery",
        }
        self.history.append(self.active_event)
        return True

    def trip_force(self, time_s: float) -> np.ndarray:
        """The caught-toe drag: backward along the walking heading, bounded.

        Applied at the snagged ankle (``trip_foot``) by the keyboard via
        ``runtime.apply_force`` -- NOT at the root, where it would just
        accelerate the feet along with the body. Added while the pelvis assist
        fades (the "tripping" mode drops the locomotion blend and the keyboard
        removes the horizontal authority), so the trip torque is physical.
        """

        if not self.tripping or self.fall_time_s is None:
            return np.zeros(3)
        if time_s - self.fall_time_s >= self.config.fall_duration_s:
            return np.zeros(3)
        return -self.config.trip_force_n * np.array([
            math.cos(self.fall_heading_rad), math.sin(self.fall_heading_rad), 0.0
        ])

    def _request_fall_replay(
        self,
        time_s: float,
        heading_rad: float,
        measured_joints: np.ndarray,
        measured_position: np.ndarray | None,
        measured_quaternion: np.ndarray | None,
    ) -> bool:
        """Anchor the fall clip at the measured *standing* pose and start the replay.

        The replay drives the body through a real descent (a retargeted AMASS
        clip) instead of releasing the drives, so the rest pose comes from human
        kinematics rather than from unorganized floor contact. Impact/fallen
        labels are still measured by :meth:`observe` -- the clip only organizes
        the commanded trajectory, it never grants the fallen state.
        """

        if measured_position is None or np.shape(measured_position) != (3,):
            raise ValueError("fall replay requires the measured root position")
        if measured_quaternion is None:
            raise ValueError("fall replay requires the measured root orientation")
        if not np.isfinite(measured_position).all() or not np.isfinite(measured_quaternion).all():
            raise ValueError("fall replay requires a finite measured root pose")
        assert self.fall_clip is not None
        # 前向轴对齐（与 G 起身同一规则）：对齐实测身体前向与片段首帧前向的
        # 水平角，片段的倒地方向才与身体实际面朝一致。
        rotation = quaternion_to_matrix(np.asarray(measured_quaternion, dtype=np.float64))
        body_forward = rotation @ np.array([0.0, 0.0, 1.0])
        clip_forward = axis_angle_to_matrix(self.fall_clip.tilts[0]) @ np.array([0.0, 0.0, 1.0])
        if np.hypot(body_forward[0], body_forward[1]) > 0.2 and np.hypot(
            clip_forward[0], clip_forward[1]
        ) > 0.2:
            heading_rad = float(
                np.arctan2(body_forward[1], body_forward[0])
                - np.arctan2(clip_forward[1], clip_forward[0])
            )
        self.fall_time_s, self.fall_heading_rad = time_s, heading_rad
        self.fall_pose = measured_joints.copy()
        self.playback = {
            "kind": "fall",
            "requested_time_s": float(time_s),
            "anchor_xy": np.asarray(measured_position, dtype=np.float64)[:2].copy(),
            "heading_rad": float(heading_rad),
            "elapsed_s": 0.0,
            "source_joints": measured_joints.copy(),
            "source_height_m": float(measured_position[2]),
            "source_quaternion": np.asarray(measured_quaternion, dtype=np.float64)
            / np.linalg.norm(measured_quaternion),
            "finished": False,
        }
        self.phase = "falling"
        self.mode = "falling"
        self.requested = None
        self.blend = 0.0
        self.source = None
        self.active_contacts = ("left_ankle", "right_ankle")
        self.active_event = {
            "requested_time_s": time_s,
            "heading_rad": heading_rad,
            "impact_time_s": None,
            "fallen_time_s": None,
            "root_assistance": True,
            "control_scale": 1.0,
            "mechanism": _FALL_REPLAY_MECHANISM,
            "clip": {
                "source": self.fall_clip.provenance.get("source", {}),
                "duration_s": self.fall_clip.provenance.get("duration_s"),
                "frames": self.fall_clip.provenance.get("frames"),
                "reverse": self.fall_clip.provenance.get("reverse", False),
                "speed": self.fall_clip.provenance.get("speed", 1),
            },
        }
        self.history.append(self.active_event)
        return True

    def _request_get_up(
        self,
        time_s: float,
        heading_rad: float,
        measured_joints: np.ndarray,
        measured_position: np.ndarray | None,
        measured_quaternion: np.ndarray | None,
    ) -> bool:
        if self.playback_clip is None:
            return False
        if self.phase != "fallen":
            return False
        if measured_position is None or np.shape(measured_position) != (3,):
            raise ValueError("get_up requires the measured root position")
        if not np.isfinite(measured_position).all():
            raise ValueError("get_up requires a finite measured root position")
        tilt = np.zeros(3) if self.last_pose is None else self.last_pose[2]
        # 锚定朝向 = 前向轴对齐: 侧躺身体的四元数 yaw 与"面朝"差 ~90°,
        # 用 yaw 锚定会让片段的推地动作落在身体的侧面 = 推空气。对齐
        # 实测前向轴与片段首帧前向轴的水平角, 推地方向才与地板一致。
        measured_forward = axis_angle_to_matrix(
            matrix_to_axis_angle(
                quaternion_to_matrix(np.asarray(measured_quaternion))
            )
        ) if measured_quaternion is not None else None
        clip_forward = axis_angle_to_matrix(self.playback_clip.tilts[0]) @ np.array([0.0, 0.0, 1.0])
        if measured_forward is not None:
            body_forward = measured_forward @ np.array([0.0, 0.0, 1.0])
            horizontal_body = float(np.hypot(body_forward[0], body_forward[1]))
            horizontal_clip = float(np.hypot(clip_forward[0], clip_forward[1]))
            if horizontal_body > 0.2 and horizontal_clip > 0.2:
                heading_rad = float(
                    np.arctan2(body_forward[1], body_forward[0])
                    - np.arctan2(clip_forward[1], clip_forward[0])
                )
        self.playback = {
            "kind": "get_up",
            "requested_time_s": float(time_s),
            "anchor_xy": np.asarray(measured_position, dtype=np.float64)[:2].copy(),
            "heading_rad": float(heading_rad),
            "elapsed_s": 0.0,
            "source_joints": measured_joints.copy(),
            "source_height_m": float(measured_position[2]),
            "source_tilt": np.asarray(tilt, dtype=np.float64).copy(),
            # 实测躺姿四元数: 混入段从身体实际姿态测地过渡到回放轨迹
            "source_quaternion": (
                np.asarray(measured_quaternion, dtype=np.float64)
                / np.linalg.norm(measured_quaternion)
                if measured_quaternion is not None
                else axis_angle_to_quaternion(np.asarray(tilt, dtype=np.float64))
            ),
        }
        self.phase = "getting_up"
        self.mode = "getting_up"
        self.requested = None
        self.blend = 0.0
        self.source = None
        self.active_contacts = ("left_ankle", "right_ankle")
        return True

    def command(self, movement: tuple[float, float]) -> tuple[float, float]:
        if self.tripping:
            # The stumble steps ARE the recovery attempt: keep the gait fed.
            return movement
        blocked = (
            self.phase != "idle"
            or self.requested is not None
            or self.blend > 0
        )
        return (0.0, 0.0) if blocked else movement

    def apply(
        self,
        target: TeleopTarget,
        *,
        dt_s: float,
        speed_m_s: float,
        idle_joints: np.ndarray,
        idle_height_m: float,
        idle_tilt: np.ndarray,
        heading_rad: float,
    ) -> TeleopTarget:
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("action step must be positive and finite")
        if self.phase == "getting_up":
            return self._apply_playback(target, dt_s=dt_s)
        if self.tripping:
            # Gait keeps stepping (command() passes movement through); the
            # commanded pose is continuously re-captured so that a measured
            # fall freezes the body on its CURRENT stumbling pose, not on the
            # pre-trip one. The keyboard adds trip_force() on top of the
            # fading assist and releases the drives when observe() latches.
            self.fall_pose = target.joints.copy()
            return replace(target, mode="tripping")
        if self.playback is not None and self.playback.get("kind") == "fall":
            # Anchored fall replay: the body is *driven* through the clip while
            # the drives stay engaged. Dispatch is by playback kind, NOT by
            # phase: observe() latches `fallen` from the measured state while
            # the clip is still descending (a real fall crosses the height/tilt
            # thresholds mid-descent too), and that latch must not hijack the
            # command stream back to the frozen pre-fall pose -- measured on
            # the first GPU run as a body yanked into a stiff prone plank at
            # 0.39 m. The release happens only via consume_fall_replay_release().
            return self._apply_fall_replay(target, dt_s=dt_s)
        if self.falling:
            assert self.fall_pose is not None
            return replace(
                target,
                joints=self.fall_pose.copy(),
                joint_velocities=np.zeros_like(target.joints),
                mode=self.mode,
            )
        if self.source is None and self.blend <= _SMOOTHSTEP_EPS and self.requested is None:
            self.blend = 0.0
            self.mode = "locomotion"
            return target
        if self.requested is not None and self.blend < 1.0 and abs(speed_m_s) > (
            self.config.stopped_speed_m_s
        ):
            self.mode = "braking_for_action"
            return replace(target, mode=self.mode)
        if self.source is None:
            self.source = (idle_joints.copy(), idle_height_m, idle_tilt.copy())
            self._begin_blend(self.requested)
            if self.plan is not None:
                self._capture_support_anchor(self.source[0], self.source[2], target.position)
        self.blend = min(1.0, self.blend + dt_s / self.config.transition_s)
        if self.requested is None and self.blend >= 1.0:
            # The return blend reached the standing reference: hand control back
            # to the locomotion path (target passed through unchanged, like the
            # pre-refactor completion at blend == 0).
            self.source = None
            self.blend = 0.0
            self.active_contacts = ("left_ankle", "right_ankle")
            self.mode = "locomotion"
            return target
        weight = _smoothstep(self.blend)
        assert self.source is not None
        if self.requested is None:
            dst_q, dst_h, dst_t = idle_joints, idle_height_m, idle_tilt
            self.mode = "standing_up"
        else:
            posture = self.postures[self.requested]
            dst_q, dst_h, dst_t = posture.joints, posture.height_m, posture.tilt
            self.mode = self.requested if self.blend >= 1.0 else _POSTURE_ING_MODES[self.requested]
        q = (1 - weight) * self.source[0] + weight * dst_q
        tilt = (1 - weight) * self.source[2] + weight * dst_t
        position = target.position.copy()
        if self.plan is None:
            # Rig-less state-machine fixtures: historical linear height blend.
            height = (1 - weight) * self.source[1] + weight * dst_h
        else:
            # Contact-consistent height, derived from the blended configuration
            # itself: ground the lowest active contact. The joints/height pair
            # can no longer disagree about the floor -- feet neither float
            # through a hold (the crouch-hold 1.7 s no-support failure) nor
            # sink through a transition, and mid-transition retargets stay
            # continuous because the closure is a pure function of the
            # commanded pose (the previous height WAS its closure).
            poses = forward_kinematics(
                self.plan,
                dict(zip(self.plan.dof_names, q, strict=True)),
                root_rotation=tilt,
            )
            height = -min(
                capsule_bottom(self.plan, poses, name) for name in self.active_contacts
            )
            # Horizontal support compensation: the blend moves the feet in body
            # frame (the posture's feet are not where the previous stance's
            # were), which dragged them across the floor at ~0.5 m/s. Shift the
            # root target by the mean foot offset difference so the feet stay
            # planted; the residual is the feet's relative drift only.
            feet = [name for name in self.active_contacts if name.endswith("_ankle")]
            current_offsets = np.array([poses[name].translation[:2] for name in feet])
            shift_body = self.source_foot_offsets.mean(axis=0) - current_offsets.mean(axis=0)
            cosine, sine = np.cos(heading_rad), np.sin(heading_rad)
            position[:2] = self.source_root_xy + np.array([
                cosine * shift_body[0] - sine * shift_body[1],
                sine * shift_body[0] + cosine * shift_body[1],
            ])
        self.last_pose = (q.copy(), float(height), tilt.copy())
        rotation = rotation_about_axis("z", heading_rad) @ axis_angle_to_matrix(tilt)
        position[2] = height
        self.last_root_xy = position[:2].copy()
        return replace(
            target,
            joints=q,
            position=position,
            quaternion=axis_angle_to_quaternion(matrix_to_axis_angle(rotation)),
            mode=self.mode,
        )

    def _apply_playback(self, target: TeleopTarget, *, dt_s: float) -> TeleopTarget:
        assert self.playback is not None and self.playback_clip is not None
        clip = self.playback_clip
        state = self.playback
        state["elapsed_s"] += dt_s
        elapsed = state["elapsed_s"]
        index = min(int(elapsed / clip.frame_dt_s), len(clip.joints) - 1)
        blend = _smoothstep(elapsed / _CLIP_BLEND_S)
        q = (1 - blend) * state["source_joints"] + blend * clip.joints[index]
        # 朝向 = 从实测躺姿测地过渡到 "锚定 yaw ⊕ 片段倾角"(片段前向已对齐
        # 实测前向, 端点一致故测地线无自旋); 关节/高度仍线性混合。
        # 注: 不用 axis-angle 线性混合倾角——大倾角混合会扫出几十度等效 yaw。
        anchor_rotation = rotation_about_axis("z", state["heading_rad"]) @ axis_angle_to_matrix(
            _without_yaw(clip.tilts[index])
        )
        target_quaternion = axis_angle_to_quaternion(matrix_to_axis_angle(anchor_rotation))
        blend_in = _smoothstep(min(1.0, elapsed / _CLIP_BLEND_S))
        world_quaternion = quaternion_slerp(
            state["source_quaternion"], target_quaternion, blend_in
        )
        tilt = matrix_to_axis_angle(quaternion_to_matrix(world_quaternion))
        # Root height: the clip's absolute curve by default, or -- when contact
        # closure is on -- the height that puts the blended configuration's lowest
        # collision body exactly on the floor. See ActionConfig
        # .get_up_contact_closure for the measurement that forced this.
        clip_height = clip.heights_m[index]
        if self.config.get_up_contact_closure and self.closure_links:
            poses = forward_kinematics(
                self.plan,
                dict(zip(self.plan.dof_names, q, strict=True)),
                root_rotation=tilt,
            )
            clip_height = -min(
                capsule_bottom(self.plan, poses, name) for name in self.closure_links
            )
        height = (1 - blend) * state["source_height_m"] + blend * clip_height
        offset = clip.offsets_xy[index]
        c, s = math.cos(state["heading_rad"]), math.sin(state["heading_rad"])
        anchor = state["anchor_xy"]
        position = np.array(
            [anchor[0] + c * offset[0] - s * offset[1],
             anchor[1] + s * offset[0] + c * offset[1],
             height]
        )
        # 交回时把回放朝向拆成 "heading ⊕ 无 yaw tilt": tilt 里的 yaw 已并入
        # 锚定 heading(由 keyboard 的交回同步采纳), 否则 standing blend 的
        # heading ⊕ tilt 会把这段 yaw 再计一次, 身体被拧一整圈去追重复计数
        # 的朝向。
        tilt_clean = _without_yaw(tilt)
        self.last_pose = (q.copy(), float(height), tilt_clean.copy())
        if index >= len(clip.joints) - 1 and elapsed >= clip.duration_s + _CLIP_BLEND_S:
            # Recovery complete: hand over to the standing blend from the final
            # replay pose, which restores assistance and locomotion modes.
            self.source = self.last_pose
            self.blend = 0.0
            self.requested = None
            self.playback = None
            self.phase = "idle"
            self.mode = "standing_up"
            if self.plan is not None:
                self._capture_support_anchor(
                    self.last_pose[0], self.last_pose[2], target.position
                )
        else:
            self.mode = "getting_up"
        return replace(
            target,
            joints=q,
            position=position,
            quaternion=world_quaternion,
            mode=self.mode,
        )

    def _apply_fall_replay(self, target: TeleopTarget, *, dt_s: float) -> TeleopTarget:
        """Advance the anchored fall replay one physics step.

        Same anchoring family as :meth:`_apply_playback` (measured start pose,
        forward-axis-aligned heading, slerp blend-in, contact-closure height)
        with two differences: the clip keeps its own yaw -- a fall rotates the
        body and there is no standing blend to double-count it later -- and the
        completion does NOT hand control back: the replay freezes on its final
        lying pose until the measured state latches ``fallen`` and the keyboard
        releases the drives.
        """

        assert self.playback is not None and self.fall_clip is not None
        clip = self.fall_clip
        state = self.playback
        if not state["finished"]:
            state["elapsed_s"] += dt_s
        elapsed = state["elapsed_s"]
        index = min(int(elapsed / clip.frame_dt_s), len(clip.joints) - 1)
        blend = _smoothstep(elapsed / _CLIP_BLEND_S)
        q = (1 - blend) * state["source_joints"] + blend * clip.joints[index]
        anchor_rotation = rotation_about_axis("z", state["heading_rad"]) @ axis_angle_to_matrix(
            clip.tilts[index]
        )
        target_quaternion = axis_angle_to_quaternion(matrix_to_axis_angle(anchor_rotation))
        blend_in = _smoothstep(min(1.0, elapsed / _CLIP_BLEND_S))
        world_quaternion = quaternion_slerp(
            state["source_quaternion"], target_quaternion, blend_in
        )
        tilt = matrix_to_axis_angle(quaternion_to_matrix(world_quaternion))
        clip_height = clip.heights_m[index]
        if self.config.get_up_contact_closure and self.closure_links:
            poses = forward_kinematics(
                self.plan,
                dict(zip(self.plan.dof_names, q, strict=True)),
                root_rotation=tilt,
            )
            clip_height = -min(
                capsule_bottom(self.plan, poses, name) for name in self.closure_links
            )
        height = (1 - blend) * state["source_height_m"] + blend * clip_height
        # Horizontal momentum must build from REST: the measured body is
        # stationary at trigger while a walking/running clip subject is not.
        # Commanding the raw offsets yanks the pelvis into an instant glide
        # with the feet planted (GPU run 6 slid 2.84 m, user review: "往前滑
        # 好长一段距离再倒下"). Scaling by the same blend-in keeps the first
        # half-second near the anchor; the remaining travel arrives as the
        # fall's own momentum, with the tilt release handing it to physics.
        offset = blend_in * clip.offsets_xy[index]
        c, s = math.cos(state["heading_rad"]), math.sin(state["heading_rad"])
        anchor = state["anchor_xy"]
        position = np.array(
            [anchor[0] + c * offset[0] - s * offset[1],
             anchor[1] + s * offset[0] + c * offset[1],
             height]
        )
        if not state["finished"] and (
            index >= len(clip.joints) - 1 and elapsed >= clip.duration_s + _CLIP_BLEND_S
        ):
            # Clip exhausted: freeze the command stream on the final lying pose
            # (apply() serves it from fall_pose once the drives are released)
            # and wait for the measured latch. No standing handover here.
            state["finished"] = True
            self.fall_pose = clip.joints[-1].copy()
        release_at = self.config.fall_replay_release_tilt_deg
        if release_at is not None and not state["finished"]:
            w, x, y, z = world_quaternion
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
            if tilt >= release_at:
                # Hand the rest to physics: freeze on the pose commanded right
                # now and let consume_fall_replay_release() free the drives --
                # from the tipping point the drop is gravity, not clip. The
                # measured impact/fallen labels stay with observe().
                state["finished"] = True
                state["release_policy"] = "at_tilt"
                self.fall_pose = q.copy()
        return replace(
            target,
            joints=q,
            position=position,
            quaternion=world_quaternion,
            mode="falling",
        )

    def fall_report(self) -> dict[str, Any]:
        """The last fall as recorded, independent of the live state.

        :meth:`reset` deliberately clears the live fall fields, so a session that
        resets after a fall would otherwise report ``None``/``locomotion`` for a
        fall that physically happened. ``history`` is the durable record, and the
        per-event timestamps are what the label has to be read from.
        """

        last = self.history[-1] if self.history else None
        if last is not None and last.get("survived_time_s") is not None:
            outcome = "stumble_survived"
        else:
            outcome = (
                None
                if last is None
                else "fallen"
                if last["fallen_time_s"] is not None
                else "impacted_not_fallen"
                if last["impact_time_s"] is not None
                else "no_body_floor_impact"
            )
        return {
            "requested_time_s": None if last is None else last["requested_time_s"],
            "impact_time_s": None if last is None else last["impact_time_s"],
            "fallen_time_s": None if last is None else last["fallen_time_s"],
            "outcome": outcome,
            "state_at_exit": self.mode,
            "falls_requested": len(self.history),
            "mechanism": _LEGACY_FALL_MECHANISM if last is None else last.get(
                "mechanism", _LEGACY_FALL_MECHANISM
            ),
            "events": self.history,
        }

    def fall_force(self, time_s: float) -> np.ndarray:
        if self.fall_replay_engaged:
            # The replay commands the descent itself; an external pulse would
            # fight the tracked clip. The legacy release keeps its bias pulse.
            return np.zeros(3)
        if self.fall_time_s is None or time_s - self.fall_time_s >= self.config.fall_duration_s:
            return np.zeros(3)
        # Backward relative to the user's heading, with no upward impulse.
        return -self.config.fall_force_n * np.array(
            [np.cos(self.fall_heading_rad), np.sin(self.fall_heading_rad), 0.0]
        )

    def observe(
        self,
        *,
        time_s: float,
        root_height_m: float,
        tilt_deg: float,
        standing_height_m: float,
        body_impact: bool,
    ) -> None:
        if self.tripping:
            assert self.active_event is not None
            if body_impact and self.impact_time_s is None:
                self.impact_time_s = time_s
                self.active_event["impact_time_s"] = time_s
            # Trip-fall latch: TILT ALONE. The vertical assist keeps holding the
            # pelvis near the gait height while the body pitches (GPU run 6:
            # tilt 118 deg with z still at 0.69 m -- a mid-air flail on a
            # string), so impact and low height never co-occur until AFTER the
            # drives are released; demanding all three deadlocked the state in
            # "tripping" and the assist hauled the body back upright. A real
            # stumble-recovery keeps tilt under ~30 deg, so 50 deg is decisive.
            lost = tilt_deg > self.config.fallen_tilt_deg
            if lost:
                # Balance is measurably gone: mutate the trip event into a fall
                # and hand the body to the drive-release collapse. The keyboard
                # releases the drives on the next step via the falling branch.
                self.active_event["mechanism"] = "trip_shove_release_fall"
                self.phase = "falling"
                self.mode = "falling"
                self.trip_release_pending = True
                return
            if time_s - self.fall_time_s > self.config.trip_window_s:
                # The stumble steps caught the body: back to locomotion, with
                # the survived stumble on the record (a hard negative candidate,
                # deliberately NOT admitted as a fall by the label pipeline).
                self.active_event["survived_time_s"] = time_s
                self.phase = "idle"
                self.mode = "locomotion"
            return
        if not self.falling:
            return
        if body_impact and self.impact_time_s is None:
            self.impact_time_s = time_s
            assert self.active_event is not None
            self.active_event["impact_time_s"] = time_s
        if (
            self.impact_time_s is not None
            and root_height_m < standing_height_m * self.config.fallen_height_fraction
            and tilt_deg > self.config.fallen_tilt_deg
        ):
            self.phase = "fallen"
            self.mode = "fallen"
            assert self.active_event is not None
            if self.active_event["fallen_time_s"] is None:
                self.active_event["fallen_time_s"] = time_s
