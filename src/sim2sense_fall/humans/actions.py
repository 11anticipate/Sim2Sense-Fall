"""Validated AMASS posture targets and interruptible keyboard action states."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .amass import crop_amass_clip, load_amass_clip, normalize_root_motion
from .contact_control import capsule_bottom
from .rig import HumanRigPlan, forward_kinematics, joint_values_from_clip
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle, rotation_about_axis
from .teleop import TeleopTarget


@dataclass(frozen=True)
class ActionConfig:
    transition_s: float
    stopped_speed_m_s: float
    fall_force_n: float
    fall_duration_s: float
    fall_control_scale: float
    fallen_height_fraction: float
    fallen_tilt_deg: float

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
        if not 0 < self.fallen_height_fraction < 1 or not 0 < self.fallen_tilt_deg < 90:
            raise ValueError("invalid fallen posture thresholds")


@dataclass(frozen=True)
class Posture:
    joints: np.ndarray
    height_m: float
    tilt: np.ndarray
    provenance: dict[str, Any]


def load_posture(spec: dict[str, Any], plan: HumanRigPlan) -> Posture:
    """Use an explicitly named source pose, not a whole out-of-range crouch walk.

    This is a derived stationary posture with a separately generated transition.
    It must not be reported as faithful playback of the entire AMASS sequence.
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
    poses = forward_kinematics(plan, dict(zip(plan.dof_names, q, strict=True)), root_rotation=tilt)
    height = -min(capsule_bottom(plan, poses, f"{s}_ankle") for s in ("left", "right"))
    return Posture(
        q,
        height,
        tilt,
        {
            "source": clip.provenance.as_dict(),
            "metadata": dict(clip.metadata),
            "derivation": "first_pose_of_authorized_crop_stationary_target_with_smooth_transition",
            "height_method": "minimum_foot_collision_support",
            "height_m": height,
        },
    )


class ActionState:
    """Edge-triggered crouch/stand and a latched, physically observed fall."""

    def __init__(self, config: ActionConfig, crouch: Posture) -> None:
        self.config, self.crouch = config, crouch
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.mode = "locomotion"
        self.requested = "stand"
        self.blend = 0.0
        self.fall_time_s: float | None = None
        self.fall_heading_rad = 0.0
        self.fall_pose: np.ndarray | None = None
        self.impact_time_s: float | None = None
        self.active_event: dict[str, Any] | None = None

    @property
    def falling(self) -> bool:
        return self.fall_time_s is not None

    def request(
        self, name: str, *, time_s: float, heading_rad: float, measured_joints: np.ndarray
    ) -> None:
        if name not in {"crouch", "stand", "fall"}:
            raise ValueError(f"unknown action {name}")
        if not np.isfinite([time_s, heading_rad]).all() or not np.isfinite(measured_joints).all():
            raise ValueError("action request requires finite measured state")
        if self.falling:
            return
        if name == "fall":
            self.fall_time_s, self.fall_heading_rad = time_s, heading_rad
            self.fall_pose = measured_joints.copy()
            self.mode = "falling"
            self.active_event = {
                "requested_time_s": time_s,
                "heading_rad": heading_rad,
                "impact_time_s": None,
                "fallen_time_s": None,
                "root_assistance": False,
                "control_scale": self.config.fall_control_scale,
            }
            self.history.append(self.active_event)
        else:
            self.requested = name

    def command(self, movement: tuple[float, float]) -> tuple[float, float]:
        return (
            (0.0, 0.0) if self.falling or self.requested == "crouch" or self.blend > 0 else movement
        )

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
        if self.falling:
            assert self.fall_pose is not None
            return replace(
                target,
                joints=self.fall_pose.copy(),
                joint_velocities=np.zeros_like(target.joints),
                mode=self.mode,
            )
        desired = float(self.requested == "crouch")
        if abs(speed_m_s) > self.config.stopped_speed_m_s and desired > self.blend:
            self.mode = "braking_for_crouch"
            return replace(target, mode=self.mode)
        self.blend += float(
            np.clip(
                desired - self.blend,
                -dt_s / self.config.transition_s,
                dt_s / self.config.transition_s,
            )
        )
        if self.blend < 1e-10:
            self.blend = 0.0
            self.mode = "locomotion"
            return target
        weight = self.blend**2 * (3 - 2 * self.blend)
        q = (1 - weight) * idle_joints + weight * self.crouch.joints
        height = (1 - weight) * idle_height_m + weight * self.crouch.height_m
        tilt = (1 - weight) * idle_tilt + weight * self.crouch.tilt
        from .rotations import axis_angle_to_quaternion

        rotation = rotation_about_axis("z", heading_rad) @ axis_angle_to_matrix(tilt)
        self.mode = (
            "crouch"
            if self.blend >= 1
            else ("crouching" if self.requested == "crouch" else "standing_up")
        )
        position = target.position.copy()
        position[2] = height
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
            self.mode = "fallen"
            assert self.active_event is not None
            if self.active_event["fallen_time_s"] is None:
                self.active_event["fallen_time_s"] = time_s
