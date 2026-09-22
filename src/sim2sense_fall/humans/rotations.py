"""Rotation conversions and validity checks shared by the human pipeline.

Everything here is plain NumPy on CPU so that the motion contract, the retargeter
and the forward-kinematics cross-check can be tested without a simulator.

Conventions
-----------

* Rotations are **axis-angle** vectors (Rodrigues form) or unit quaternions in
  ``(w, x, y, z)`` order, matching ``Gf.Quatf``.
* The body-local frame is right-handed with ``+X`` forward, ``+Y`` left and
  ``+Z`` up, so ``X x Y = Z``.
* Matrices act on column vectors: ``p_world = R @ p_local + t``.

The axis-angle convention matches SMPL: the vector direction is the rotation
axis and its magnitude is the angle in radians, and a zero vector means the
identity. Rotations are expressed about the joint's **rest-aligned** axes, which
is what makes a direct, sign-preserving mapping onto a USD revolute joint
possible (see :mod:`sim2sense_fall.humans.rig`).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "close_rotation",
    "matrix_to_axis_angle",
    "orthonormality_error",
    "quaternion_to_matrix",
    "rotation_about_axis",
    "rotation_matrix_error",
    "skew",
    "validate_axis",
    "validate_axis_angle",
]

#: A rotation matrix must be this close to orthogonal and to determinant +1.
ORTHONORMALITY_TOLERANCE = 1e-9
#: Matrices serialised through float32 (USD) legitimately drift further.
FLOAT32_ORTHONORMALITY_TOLERANCE = 1e-4
#: Axis-angle vectors shorter than this are treated as the identity rotation.
ANGLE_EPSILON = 1e-12

AXES: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def validate_axis(value: str) -> str:
    """Return a canonical single-axis token or raise."""

    token = str(value).strip().lower()
    if token not in AXES:
        raise ValueError(f"joint axis must be one of {sorted(AXES)}, got {value!r}")
    return token


def validate_axis_angle(vector: object, name: str) -> np.ndarray:
    """Coerce to a finite shape-``(3,)`` float array, rejecting strings and NaN."""

    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value: {array.tolist()}")
    return array


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix of a 3-vector."""

    x, y, z = (float(component) for component in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_matrix_error(matrix: object) -> float:
    """Return ``max(|R^T R - I|, |det R - 1|)`` for a rotation matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        return math.inf
    orthogonality = float(np.max(np.abs(array.T @ array - np.eye(3))))
    determinant = float(abs(np.linalg.det(array) - 1.0))
    return max(orthogonality, determinant)


def orthonormality_error(matrix: object) -> float:
    """Alias kept explicit for callers that read as "is this a rotation"."""

    return rotation_matrix_error(matrix)


def close_rotation(matrix: object, *, tolerance: float = ORTHONORMALITY_TOLERANCE) -> bool:
    """True when ``matrix`` is a proper rotation within ``tolerance``."""

    return rotation_matrix_error(matrix) <= tolerance


def axis_angle_to_matrix(vector: object) -> np.ndarray:
    """Rodrigues' formula. A zero-length vector yields the identity."""

    array = validate_axis_angle(vector, "axis-angle")
    angle = float(np.linalg.norm(array))
    if angle < ANGLE_EPSILON:
        return np.eye(3)
    axis = array / angle
    cross = skew(axis)
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def rotation_about_axis(axis: str, angle_rad: float) -> np.ndarray:
    """Rotation of ``angle_rad`` about a canonical body axis."""

    if not isinstance(angle_rad, (int, float)) or isinstance(angle_rad, bool):
        raise ValueError(f"angle must be a number, got {angle_rad!r}")
    if not math.isfinite(angle_rad):
        raise ValueError(f"angle must be finite, got {angle_rad!r}")
    return axis_angle_to_matrix(AXES[validate_axis(axis)] * float(angle_rad))


def axis_angle_to_quaternion(vector: object) -> np.ndarray:
    """Convert an axis-angle vector into a ``(w, x, y, z)`` unit quaternion."""

    array = validate_axis_angle(vector, "axis-angle")
    angle = float(np.linalg.norm(array))
    if angle < ANGLE_EPSILON:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = array / angle
    half = angle / 2.0
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def quaternion_to_matrix(quaternion: object) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion into a rotation matrix."""

    array = np.asarray(quaternion, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("quaternion contains a non-finite value")
    norm = float(np.linalg.norm(array))
    if norm < ANGLE_EPSILON:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = (float(component) / norm for component in array)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_axis_angle(matrix: object, *, tolerance: float = 1e-8) -> np.ndarray:
    """Inverse of :func:`axis_angle_to_matrix`, continuous except at pi.

    Near a half turn the skew part vanishes and the axis is recovered from the
    diagonal instead, which keeps the result stable for flipped-over fall poses.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {array.shape}")
    error = rotation_matrix_error(array)
    if error > tolerance:
        raise ValueError(f"matrix is not a rotation (error {error:.3e})")
    cosine = max(-1.0, min(1.0, (float(np.trace(array)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if abs(math.pi - angle) < 1e-6:
        # Half turn: R + I = 2 * axis axis^T, so the axis is the dominant column.
        diagonal = (np.diag(array) + 1.0) / 2.0
        index = int(np.argmax(diagonal))
        axis = array[:, index] + np.eye(3)[:, index]
        axis = axis / float(np.linalg.norm(axis) or 1.0)
        return axis * angle
    axis = np.array(
        [array[2, 1] - array[1, 2], array[0, 2] - array[2, 0], array[1, 0] - array[0, 1]]
    )
    return axis * (angle / (2.0 * math.sin(angle)))
