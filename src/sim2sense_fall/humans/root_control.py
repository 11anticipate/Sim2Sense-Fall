"""Finite external root assistance for physics-based reference tracking.

This is an explicit external actuator, not an unassisted balance controller.
The root is integrated by PhysX; no pose or velocity is overwritten by it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from .rotations import matrix_to_axis_angle, quaternion_to_matrix


@dataclass(frozen=True, slots=True)
class RootAssistConfig:
    position_stiffness_n_m: float
    position_damping_ns_m: float
    rotation_stiffness_nm_rad: float
    rotation_damping_nms_rad: float
    max_force_n: float
    max_torque_nm: float
    gravity_compensation_fraction: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"root assistance {name} must be finite and non-negative")
        if min(self.max_force_n, self.max_torque_nm) <= 0:
            raise ValueError("root assistance force and torque limits must be positive")
        if self.gravity_compensation_fraction > 1:
            raise ValueError("gravity compensation fraction must not exceed 1")

    @classmethod
    def load(cls, path: Path) -> RootAssistConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("root assistance YAML must specify exactly the config fields")
        return cls(**{key: float(value) for key, value in payload.items()})


def root_wrench(
    config: RootAssistConfig,
    *,
    position: np.ndarray,
    quaternion: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    target_linear_velocity: np.ndarray,
    target_angular_velocity: np.ndarray,
    mass_kg: float,
    gravity_m_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a norm-limited world-frame force and torque at the root COM."""
    for vector, size in ((position, 3), (quaternion, 4), (linear_velocity, 3),
                         (angular_velocity, 3), (target_position, 3),
                         (target_quaternion, 4), (target_linear_velocity, 3),
                         (target_angular_velocity, 3)):
        if np.shape(vector) != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"root controller expects a finite ({size},) state vector")
    if not np.isfinite([mass_kg, gravity_m_s2]).all() or min(mass_kg, gravity_m_s2) <= 0:
        raise ValueError("mass and gravity must be finite and positive")
    force = (config.position_stiffness_n_m * (target_position - position)
             + config.position_damping_ns_m * (target_linear_velocity - linear_velocity))
    force[2] += config.gravity_compensation_fraction * mass_kg * gravity_m_s2
    rotation_error = matrix_to_axis_angle(
        quaternion_to_matrix(target_quaternion) @ quaternion_to_matrix(quaternion).T
    )
    torque = (config.rotation_stiffness_nm_rad * rotation_error
              + config.rotation_damping_nms_rad * (target_angular_velocity - angular_velocity))
    force *= min(1.0, config.max_force_n / max(float(np.linalg.norm(force)), 1e-12))
    torque *= min(1.0, config.max_torque_nm / max(float(np.linalg.norm(torque)), 1e-12))
    return force, torque
