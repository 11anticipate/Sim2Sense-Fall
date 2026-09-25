"""Root travel decomposed in the character's own body frame.

The P0-A arm-capture report used to hard-code world ``x`` as "forward". The
shipped ``keyboard.yaml`` spawns the character at ``heading_deg 90``, so its
forward direction is world ``+Y``: a clean 1.1 m walk was reported as
``lateral_drift_max_m = 1.21 m`` and as ``x_max_m = 0.002 m`` of forward
progress. That is a mislabelling that inverts the physical reading of the trial
(a straight walk looked like a sideways stumble).

The fix is to project each inter-step displacement onto the body axes taken from
the recorded root orientation, so the labels mean the same thing at every
heading. Cycle-boundary teleports (the character is re-spawned at each cycle)
are excluded, because a rewind is not motion.

This module is pure NumPy and imports nothing from Isaac Sim, so the rule is
enforceable on CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rotations import quaternion_to_matrix

__all__ = ["TRAVEL_DEFINITION", "root_displacement_in_body_frame"]

TRAVEL_DEFINITION = (
    "root displacement projected on the body forward (X_body) and left (Y_body) "
    "axes, integrated per step and excluding cycle-boundary resets"
)

# A per-step displacement larger than this is a teleport (cycle rewind), not
# walking. The controller's own cap is ``max_target_lead_m`` (0.10 m) plus one
# frame of travel, so 0.05 m is already far above any legitimate step.
_RESET_STEP_M = 0.05


def root_displacement_in_body_frame(
    root: object,
    quaternion: object,
    command: object | None = None,
) -> dict[str, Any]:
    """Decompose a root trajectory into body-forward and body-left travel.

    Parameters
    ----------
    root:
        ``(N, 3)`` world positions in metres.
    quaternion:
        ``(N, 4)`` root orientations as ``(w, x, y, z)``, matching
        :func:`~sim2sense_fall.humans.rotations.quaternion_to_matrix` and the
        ``root_quaternion`` array written by ``scripts/humans/arm_capture.py``.
    command:
        Optional ``(N, 2)`` ``(forward, turn)`` command trace. Unused for the
        arithmetic; accepted so callers can pass the record row directly.

    Returns
    -------
    dict
        ``forward_travel_m`` is the *signed* travel along the body forward axis:
        positive when the character moves the way it faces (W), negative when it
        moves the way it backs (S, or a body facing away from the motion). No sign
        normalisation is applied -- normalising would make W and S indistinguishable
        and hide a reversal, which is exactly the kind of silent misreport this
        module exists to prevent. Gate on ``forward_travel_abs_m`` for magnitude.
    """

    del command  # reserved: the sign is read from the body frame, not the command
    positions = np.asarray(root, dtype=float)
    orientations = np.asarray(quaternion, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"root must be (N, 3), got {positions.shape}")
    if orientations.shape != (positions.shape[0], 4):
        raise ValueError(
            f"quaternion must be ({positions.shape[0]}, 4), got {orientations.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise ValueError("root trajectory and orientation must be finite")

    if positions.shape[0] < 2:
        return {
            "definition": TRAVEL_DEFINITION,
            "sample_count": int(positions.shape[0]),
            "note": "too few samples to decompose",
        }

    forward = np.array([quaternion_to_matrix(q)[:, 0] for q in orientations])[:, :2]
    norm = np.linalg.norm(forward, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    forward /= norm
    left = np.stack([-forward[:, 1], forward[:, 0]], axis=1)

    delta = np.diff(positions[:, :2], axis=0)
    step_length = np.linalg.norm(delta, axis=1)
    real_step = step_length < _RESET_STEP_M

    forward_step = np.einsum("ij,ij->i", delta, forward[1:]) * real_step
    lateral_step = np.einsum("ij,ij->i", delta, left[1:]) * real_step

    return {
        "definition": TRAVEL_DEFINITION,
        "body_axes_source": "per-step root orientation quaternion",
        "forward_axis": "x_body (facing direction; sign follows the body frame)",
        "lateral_axis": "y_body (character's left)",
        "forward_travel_m": float(forward_step.sum()),
        "forward_travel_abs_m": float(np.abs(forward_step).sum()),
        "forward_max_m": float(forward_step.max()),
        "forward_min_m": float(forward_step.min()),
        "lateral_travel_abs_m": float(np.abs(lateral_step).sum()),
        "lateral_net_m": float(lateral_step.sum()),
        "lateral_drift_max_abs_m": float(np.abs(lateral_step).max()),
        "height_final_m": float(positions[-1, 2]),
        "reset_steps": int((~real_step).sum()),
        "sample_count": int(positions.shape[0]),
    }
