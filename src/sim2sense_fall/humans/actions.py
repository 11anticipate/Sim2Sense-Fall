"""Validated AMASS posture targets, motion playback and keyboard action states."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .amass import (
    crop_amass_clip,
    ground_amass_clip,
    load_amass_clip,
    normalize_root_motion,
)
from .contact_control import capsule_bottom, sideways_leg_dofs
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
    # drives are released (fall_control_scale 0), but PhysX drive damping is a
    # property the body keeps: at full damping the collapse is a slow rigid
    # crumple, at zero it folds ballistically and the floor impact NaNs the
    # solver (measured 2864 deg/s mid-fall; see HumanRuntime.set_control_scale).
    # The scale keeps a bounded viscous term: limp on screen, solver-safe.
    fall_damping_scale: float = 0.15

    def __post_init__(self) -> None:
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError("action settings must be finite")
        if (
            min(self.transition_s, self.stopped_speed_m_s, self.fall_force_n, self.fall_duration_s)
            <= 0
        ):
            raise ValueError("action timings and force must be positive")
        if not 0 <= self.fall_control_scale <= 1:
            raise ValueError("fall control scale must be in [0,1]")
        if not 0.05 <= self.fall_damping_scale <= 1:
            raise ValueError(
                "fall damping scale must be in [0.05, 1]: below 0.05 the collapse is the "
                "measured solver-NaN regime, above 1 would amplify the damping"
            )
        if not 0 < self.fallen_height_fraction < 1 or not 0 < self.fallen_tilt_deg < 90:
            raise ValueError("invalid fallen posture thresholds")


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
    """
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
    return Posture(
        q,
        height,
        tilt,
        {
            "source": clip.provenance.as_dict(),
            "metadata": dict(clip.metadata),
            "derivation": derivation,
            "height_method": (
                "declared_contact_projection" if contacts else "minimum_foot_collision_support"
            ),
            "height_m": height,
            **({"contacts": list(contacts), "projection": projection} if contacts else {}),
        },
        contact_links,
    )


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
    """
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("action clip physics step must be finite and positive")
    clip = normalize_root_motion(load_amass_clip(spec["file"]))
    clip = ground_amass_clip(clip, plan, support_z_m=0.0).resample(1.0 / dt_s, method="slerp")
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
    return ActionClip(
        joints,
        heights,
        np.asarray(clip.root_translation, dtype=np.float64)[:, :2].copy(),
        np.asarray(clip.root_rotation, dtype=np.float64).copy(),
        1.0 / clip.fps,
        {
            "source": clip.provenance.as_dict(),
            "metadata": dict(clip.metadata),
            "derivation": "full_clip_retargeted_grounded_resampled_assisted_playback",
            "duration_s": float(clip.times_s[-1]),
            "frames": int(clip.frame_count),
            "root_travel_m": round(float(np.linalg.norm(clip.root_translation[-1, :2])), 4),
        },
    )


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
        self.fall_pose: np.ndarray | None = None
        self.impact_time_s: float | None = None
        self.active_event: dict[str, Any] | None = None

    @property
    def falling(self) -> bool:
        """True while a fall is latched (falling or lying fallen) until reset/get-up."""
        return self.phase in ("falling", "fallen")

    @property
    def suppresses_stance(self) -> bool:
        """Locomotion foot corrections must not run during fall or recovery."""
        return self.phase != "idle"

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
    ) -> bool:
        if name not in {*_POSTURE_ING_MODES, "stand", "fall", "get_up"}:
            raise ValueError(f"unknown action {name}")
        if not np.isfinite([time_s, heading_rad]).all() or not np.isfinite(measured_joints).all():
            raise ValueError("action request requires finite measured state")
        if name == "get_up":
            return self._request_get_up(
                time_s, heading_rad, measured_joints, measured_position, measured_quaternion
            )
        if self.suppresses_stance:
            # Falling/fallen: only R or get_up leave the state. Getting up:
            # posture requests would fight the replay.
            return False
        if name == "fall":
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
        self.playback = {
            "requested_time_s": float(time_s),
            "anchor_xy": np.asarray(measured_position, dtype=np.float64)[:2].copy(),
            "heading_rad": float(heading_rad),
            "elapsed_s": 0.0,
            "source_joints": measured_joints.copy(),
            "source_height_m": float(measured_position[2]),
            "source_tilt": np.asarray(tilt, dtype=np.float64).copy(),
            # 实测躺姿世界朝向: 混入段从它测地过渡到回放朝向
        }
        self.phase = "getting_up"
        self.mode = "getting_up"
        self.requested = None
        self.blend = 0.0
        self.source = None
        self.active_contacts = ("left_ankle", "right_ankle")
        return True

    def command(self, movement: tuple[float, float]) -> tuple[float, float]:
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
        height = (1 - blend) * state["source_height_m"] + blend * clip.heights_m[index]
        # 起身时人自然转身(片段自带 ~80° yaw): 目标朝向跟随片段, 交回控制时
        # keyboard 采纳身体实际朝向——不在朝向上与关节回放对抗(实测对抗会
        # 摆出 350° 级的自旋)。
        tilt = (1 - blend) * state["source_tilt"] + blend * clip.tilts[index]
        offset = clip.offsets_xy[index]
        c, s = math.cos(state["heading_rad"]), math.sin(state["heading_rad"])
        anchor = state["anchor_xy"]
        position = np.array(
            [anchor[0] + c * offset[0] - s * offset[1],
             anchor[1] + s * offset[0] + c * offset[1],
             height]
        )
        # 交回时把回放朝向拆成 "heading ⊕ 无 yaw tilt": tilt 里的 yaw 已并入
        # 锚定 heading, 否则 standing blend 的 heading ⊕ tilt 会把这段 yaw
        # 再计一次, 身体被拧一整圈去追重复计数的朝向。
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
        rotation = rotation_about_axis("z", state["heading_rad"]) @ axis_angle_to_matrix(tilt)
        return replace(
            target,
            joints=q,
            position=position,
            quaternion=axis_angle_to_quaternion(matrix_to_axis_angle(rotation)),
            mode=self.mode,
        )

    def fall_report(self) -> dict[str, Any]:
        """The last fall as recorded, independent of the live state.

        :meth:`reset` deliberately clears the live fall fields, so a session that
        resets after a fall would otherwise report ``None``/``locomotion`` for a
        fall that physically happened. ``history`` is the durable record, and the
        per-event timestamps are what the label has to be read from.
        """

        last = self.history[-1] if self.history else None
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
            "mechanism": "root_assist_off_position_drives_scaled_backward_force_pulse",
            "events": self.history,
        }

    def fall_force(self, time_s: float) -> np.ndarray:
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
