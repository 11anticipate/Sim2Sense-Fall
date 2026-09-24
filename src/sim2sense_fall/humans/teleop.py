"""CPU-testable keyboard intent and bounded AMASS target generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .amass import crop_amass_clip, ground_amass_clip, load_amass_clip, normalize_root_motion
from .rig import HumanRigPlan, joint_values_from_clip
from .rotations import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    rotation_about_axis,
)


@dataclass(frozen=True)
class TeleopConfig:
    speed_m_s: float
    acceleration_m_s2: float
    turn_speed_deg_s: float
    turn_acceleration_deg_s2: float
    transition_s: float
    max_joint_speed_rad_s: float
    max_target_lead_m: float
    max_heading_lead_deg: float

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("teleop controller settings must be finite and positive")


def load_keyboard_config(path: Path, project_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("keyboard config must be a mapping")
    payload["controller"] = TeleopConfig(
        **{key: payload[key] for key in TeleopConfig.__dataclass_fields__}
    )
    for key in ("rig", "assets", "scene", "root_assist"):
        payload[key] = (project_root / payload[key]).resolve()
    for spec in (*payload["gaits"].values(), payload["idle"]):
        spec["file"] = (project_root / spec["file"]).resolve()
        if not np.isfinite([spec["start_s"], spec["duration_s"]]).all():
            raise ValueError("gait interval must be finite")
        if spec["start_s"] < 0 or spec["duration_s"] <= 0:
            raise ValueError("gait interval must have positive duration and nonnegative start")
    if set(payload["gaits"]) != {"forward", "backward"}:
        raise ValueError("keyboard mode requires forward and backward gaits")
    if np.shape(payload["spawn_xy"]) != (2,) or not np.isfinite(payload["spawn_xy"]).all():
        raise ValueError("spawn_xy must be a finite pair")
    for key in (
        "render_hz",
        "record_frames",
        "camera_distance_m",
        "slip_min_impulse_ns",
        "slip_speed_tolerance_m_s",
        "skin_penetration_tolerance_m",
    ):
        if not np.isfinite(payload[key]) or payload[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if int(payload["record_frames"]) != payload["record_frames"]:
        raise ValueError("record_frames must be an integer")
    for key in ("heading_deg", "camera_elevation_deg", "camera_azimuth_deg"):
        if not np.isfinite(payload[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < payload["camera_elevation_deg"] < 90:
        raise ValueError("camera elevation must be between 0 and 90 degrees")
    for segment in payload["demo"]:
        if not np.isfinite(segment["duration_s"]) or segment["duration_s"] <= 0:
            raise ValueError("demo durations must be finite and positive")
        if not isinstance(segment["keys"], list) or not set(segment["keys"]) <= KeyboardIntent.KEYS:
            raise ValueError("demo contains unsupported keys")
    return payload


class KeyboardIntent:
    """Held-key state; repeat events never accumulate extra velocity."""

    KEYS = frozenset({"W", "S", "A", "D", "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "R", "ESCAPE"})

    def __init__(self) -> None:
        self.held: set[str] = set()
        self.reset_requested = False
        self.quit_requested = False

    def event(self, key: str, pressed: bool) -> None:
        if key not in self.KEYS:
            return
        if pressed:
            if key not in self.held:
                self.reset_requested |= key == "R"
                self.quit_requested |= key == "ESCAPE"
            self.held.add(key)
        else:
            self.held.discard(key)

    def clear(self) -> None:
        self.held.clear()

    def command(self) -> tuple[float, float]:
        if "SPACE" in self.held:
            return 0.0, 0.0
        forward = int(bool(self.held & {"W", "UP"})) - int(bool(self.held & {"S", "DOWN"}))
        turn = int(bool(self.held & {"A", "LEFT"})) - int(bool(self.held & {"D", "RIGHT"}))
        return float(forward), float(turn)


@dataclass(frozen=True)
class Gait:
    joints: np.ndarray
    height_m: np.ndarray
    duration_s: float
    speed_m_s: float
    provenance: dict[str, Any]
    root_tilt: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.joints.ndim != 2 or len(self.joints) < 2:
            raise ValueError("gait joints must be (T,D), T >= 2")
        if self.height_m.shape != (len(self.joints),):
            raise ValueError("gait heights must align with joint frames")
        if not np.isfinite(self.joints).all() or not np.isfinite(self.height_m).all():
            raise ValueError("gait samples must be finite")
        if not np.isfinite([self.duration_s, self.speed_m_s]).all():
            raise ValueError("gait timing and speed must be finite")
        if self.duration_s <= 0 or self.speed_m_s < 0:
            raise ValueError("gait duration must be positive and speed nonnegative")
        if self.root_tilt is not None and (
            self.root_tilt.shape != (len(self.joints), 3) or not np.isfinite(self.root_tilt).all()
        ):
            raise ValueError("root tilt must align with gait frames")

    def tilt(self, phase: float) -> np.ndarray:
        if self.root_tilt is None:
            return np.zeros(3)
        phase %= 1.0
        point = phase * (len(self.joints) - 1)
        index = min(int(point), len(self.joints) - 2)
        fraction = point - index
        return (
            (1 - fraction) * self.root_tilt[index]
            + fraction * self.root_tilt[index + 1]
            - phase * (self.root_tilt[-1] - self.root_tilt[0])
        )

    def sample(self, phase: float) -> tuple[np.ndarray, float]:
        """A periodic reference with a distributed endpoint correction.

        The correction is recorded, since a cyclic control reference is derived
        from AMASS rather than an unchanged replay of the source sequence.
        """
        if not np.isfinite(phase):
            raise ValueError("gait phase must be finite")
        phase %= 1.0
        point = phase * (len(self.joints) - 1)
        index = min(int(point), len(self.joints) - 2)
        fraction = point - index
        q = (1 - fraction) * self.joints[index] + fraction * self.joints[index + 1]
        q -= phase * (self.joints[-1] - self.joints[0])
        h = (1 - fraction) * self.height_m[index] + fraction * self.height_m[index + 1]
        h -= phase * (self.height_m[-1] - self.height_m[0])
        return q, float(h)


def load_gait(spec: dict[str, Any], plan: HumanRigPlan, *, dt_s: float) -> Gait:
    clip = normalize_root_motion(
        crop_amass_clip(
            load_amass_clip(spec["file"]), start_s=spec["start_s"], duration_s=spec["duration_s"]
        )
    )
    clip = ground_amass_clip(clip, plan, support_z_m=0.0).resample(1 / dt_s, method="slerp")
    joints = np.stack(
        [joint_values_from_clip(clip, frame, plan)[0] for frame in range(clip.frame_count)]
    )
    lower = np.deg2rad([j.lower_deg for j in plan.joints])
    upper = np.deg2rad([j.upper_deg for j in plan.joints])
    if np.any(joints < lower - 1e-6) or np.any(joints > upper + 1e-6):
        raise ValueError(f"gait exceeds rig limits: {spec['file']}")
    duration = float(clip.times_s[-1])
    speed = np.linalg.norm(clip.root_translation[-1, :2] - clip.root_translation[0, :2]) / duration
    tilt = []
    for vector in clip.root_rotation:
        rotation = axis_angle_to_matrix(vector)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        tilt.append(matrix_to_axis_angle(rotation_about_axis("z", -yaw) @ rotation))
    return Gait(
        joints,
        plan.spawn_root_position[2] + clip.root_translation[:, 2],
        duration,
        float(speed),
        {
            "source": clip.provenance.as_dict(),
            "metadata": dict(clip.metadata),
            "loop_correction": "linear_endpoint_drift",
            "endpoint_joint_difference_deg": float(
                np.rad2deg(np.abs(joints[-1] - joints[0])).max()
            ),
        },
        np.asarray(tilt),
    )


@dataclass(frozen=True)
class TeleopTarget:
    joints: np.ndarray
    joint_velocities: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    mode: str


class TeleopController:
    """Smooth keyboard commands; bound positional windup against measured state."""

    def __init__(
        self,
        config: TeleopConfig,
        gaits: dict[str, Gait],
        idle: Gait,
        plan: HumanRigPlan,
        heading_rad: float,
    ) -> None:
        self.config, self.gaits, self.idle, self.plan = config, gaits, idle, plan
        if set(gaits) != {"forward", "backward"} or any(g.speed_m_s <= 0 for g in gaits.values()):
            raise ValueError("moving gaits require nonzero measured speed")
        if any(g.joints.shape[1] != len(plan.joints) for g in (*gaits.values(), idle)):
            raise ValueError("gait DOF count must match rig")
        self.lower = np.deg2rad([j.lower_deg for j in plan.joints])
        self.upper = np.deg2rad([j.upper_deg for j in plan.joints])
        self.reset(np.asarray(plan.spawn_root_position), heading_rad)

    def reset(self, position: np.ndarray, heading_rad: float) -> None:
        if (
            np.shape(position) != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(heading_rad)
        ):
            raise ValueError("reset requires finite position and heading")
        self.position = np.asarray(position, dtype=float).copy()
        self.position[2] = self.idle.height_m[0]
        self.heading = float(heading_rad)
        self.phase = self.speed = self.turn_rate = self.weight = 0.0
        self.mode = "forward"
        self.joints = self.idle.joints[0].copy()
        self.tilt = self.idle.tilt(0)
        self.rotation = rotation_about_axis("z", self.heading) @ axis_angle_to_matrix(self.tilt)

    def advance(
        self,
        command: tuple[float, float],
        dt_s: float,
        measured_position: np.ndarray,
        measured_heading: float,
    ) -> TeleopTarget:
        if (
            not np.isfinite(dt_s)
            or dt_s <= 0
            or np.shape(command) != (2,)
            or not np.isfinite(command).all()
        ):
            raise ValueError("command and positive step must be finite")
        if np.shape(measured_position) != (3,) or not np.isfinite(measured_position).all():
            raise ValueError("measured position must be a finite vector")
        if not np.isfinite(measured_heading) or any(abs(v) > 1 for v in command):
            raise ValueError("invalid heading or command outside [-1,1]")
        c = self.config
        self.speed += float(
            np.clip(
                command[0] * c.speed_m_s - self.speed,
                -c.acceleration_m_s2 * dt_s,
                c.acceleration_m_s2 * dt_s,
            )
        )
        turn_accel = np.deg2rad(c.turn_acceleration_deg_s2) * dt_s
        self.turn_rate += float(
            np.clip(
                command[1] * np.deg2rad(c.turn_speed_deg_s) - self.turn_rate,
                -turn_accel,
                turn_accel,
            )
        )
        self.heading += self.turn_rate * dt_s
        heading_error = (self.heading - measured_heading + np.pi) % (2 * np.pi) - np.pi
        self.heading = measured_heading + float(
            np.clip(
                heading_error,
                -np.deg2rad(c.max_heading_lead_deg),
                np.deg2rad(c.max_heading_lead_deg),
            )
        )
        velocity = self.speed * np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        self.position[:2] += velocity[:2] * dt_s
        offset = self.position[:2] - measured_position[:2]
        self.position[:2] = measured_position[:2] + offset * min(
            1.0, c.max_target_lead_m / max(float(np.linalg.norm(offset)), 1e-12)
        )
        self.mode = (
            "backward" if self.speed < -1e-4 else "forward" if self.speed > 1e-4 else self.mode
        )
        gait = self.gaits[self.mode]
        self.phase = (
            self.phase + abs(self.speed) * dt_s / max(gait.speed_m_s * gait.duration_s, 1e-6)
        ) % 1
        self.weight += float(
            np.clip(
                abs(self.speed) / c.speed_m_s - self.weight,
                -dt_s / c.transition_s,
                dt_s / c.transition_s,
            )
        )
        gait_q, gait_height = gait.sample(self.phase)
        desired = np.clip(
            (1 - self.weight) * self.idle.joints[0] + self.weight * gait_q, self.lower, self.upper
        )
        delta = np.clip(
            desired - self.joints, -c.max_joint_speed_rad_s * dt_s, c.max_joint_speed_rad_s * dt_s
        )
        self.joints += delta
        height = (1 - self.weight) * self.idle.height_m[0] + self.weight * gait_height
        velocity[2] = (height - self.position[2]) / dt_s
        self.position[2] = height
        desired_tilt = (1 - self.weight) * self.idle.tilt(0) + self.weight * gait.tilt(self.phase)
        self.tilt += min(1.0, dt_s / c.transition_s) * (desired_tilt - self.tilt)
        rotation = rotation_about_axis("z", self.heading) @ axis_angle_to_matrix(self.tilt)
        angular_velocity = matrix_to_axis_angle(rotation @ self.rotation.T) / dt_s
        self.rotation = rotation
        quaternion = axis_angle_to_quaternion(matrix_to_axis_angle(rotation))
        return TeleopTarget(
            self.joints.copy(),
            delta / dt_s,
            self.position.copy(),
            quaternion,
            velocity,
            angular_velocity,
            self.mode if self.weight > 0.01 else "stand",
        )
