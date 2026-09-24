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
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

__all__ = [
    "AngularSplit",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "close_rotation",
    "matrix_to_axis_angle",
    "orthonormality_error",
    "quaternion_to_matrix",
    "rotation_about_axis",
    "rotation_matrix_error",
    "rotation_split_residual_rad",
    "skew",
    "split_rotation",
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


# ---------------------------------------------------------------------------
# axis decomposition
# ---------------------------------------------------------------------------
#
# A revolute joint has one degree of freedom, so a rig that must reproduce a
# full 3-DOF joint rotation carries a chain of revolute joints applied in order:
# ``R = rot(a0, t0) @ rot(a1, t1) @ ...``. Turning a recorded rotation into that
# chain's angles is what follows.
#
# The strategy is one canonical solver plus a relabelling. Conjugating the chain
# by a rotation ``Q`` that sends the declared axes onto ``x, y, z`` turns the
# declared order into the canonical one, whose closed forms are textbook; the
# angles come back through the same relabelling. The closed form is then polished
# by coordinate descent and the result is *verified* by reconstruction, so a bad
# seed can never be reported as a good answer.

#: Round-trip tolerance below which a decomposition is treated as exact, in radians.
SPLIT_TOLERANCE_RAD = 1e-9
#: Sweeps of coordinate descent used to polish a closed-form seed. Each sweep is
#: exact minimisation in one angle at a time, so the residual only ever falls.
SPLIT_REFINEMENT_SWEEPS = 24
#: Canonical axis order; the closed forms are written for this order.
_CANONICAL_AXES: tuple[str, str, str] = ("x", "y", "z")
_AXIS_INDEX: dict[str, int] = {axis: index for index, axis in enumerate(_CANONICAL_AXES)}


class AngularSplit(NamedTuple):
    """The angles of a chained decomposition and how far it misses the target.

    ``angles`` holds one angle per requested axis, in the same order as the
    requested axes. ``residual_rad`` is the **geodesic** angle between the
    reconstruction and the target rotation -- the angle the limb is visibly off
    by -- rather than a coordinate-wise difference. It is zero whenever the chain
    can reproduce the rotation exactly.

    ``limit_violation_deg`` is how far outside the requested limits the returned
    angles sit. A rotation has several equivalent angle sets (each angle is only
    defined modulo a full turn, and a three-axis chain has two branches), and the
    decomposition returns the one that violates the limits least, because a
    clipped angle replays a different pose. The violation is reported so a caller
    can tell "this pose is reachable" from "this pose is reachable but the rig
    cannot hold it".
    """

    angles: tuple[float, ...]
    residual_rad: float
    limit_violation_deg: float = 0.0

    @property
    def residual_deg(self) -> float:
        """The residual in degrees, for reporting alongside the angles."""

        return math.degrees(self.residual_rad)


def _as_rotation(rotation: object) -> np.ndarray:
    """Accept a rotation matrix or an axis-angle vector; reject anything else."""

    array = np.asarray(rotation, dtype=np.float64)
    if array.shape == (3, 3):
        error = rotation_matrix_error(array)
        if error > 1e-6:
            raise ValueError(f"split_rotation needs a rotation matrix (error {error:.3e})")
        return array
    if array.shape == (3,):
        return axis_angle_to_matrix(array)
    raise ValueError(
        f"rotation must be a (3, 3) matrix or a (3,) axis-angle vector, got {array.shape}"
    )


def _geodesic_rad(first: np.ndarray, second: np.ndarray) -> float:
    """Angle of the rotation taking ``second`` onto ``first``, in ``[0, pi]``.

    Written as ``atan2`` of the skew magnitude against the trace rather than as
    ``acos`` of the trace alone. ``acos`` flattens near zero -- its numerical
    noise floor is ``sqrt(eps)``, about 1.4e-8 rad -- which is larger than the
    tolerance an exact decomposition is expected to meet, so an ``acos`` version
    reports every exact answer as a 1e-8 miss. The ``atan2`` form stays accurate
    across the whole range.
    """

    delta = first @ second.T
    skew = 0.5 * math.sqrt(
        (float(delta[2, 1]) - float(delta[1, 2])) ** 2
        + (float(delta[0, 2]) - float(delta[2, 0])) ** 2
        + (float(delta[1, 0]) - float(delta[0, 1])) ** 2
    )
    trace = max(-3.0, min(3.0, float(np.trace(delta))))
    return float(math.atan2(skew, (trace - 1.0) / 2.0))


def reconstruct_from_angles(axes: Sequence[str], angles: Sequence[float]) -> np.ndarray:
    """``rot(axes[0], angles[0]) @ ...`` -- the product the planner's chain applies."""

    matrix = np.eye(3)
    for axis, angle in zip(axes, angles, strict=True):
        matrix = matrix @ rotation_about_axis(axis, angle)
    return matrix


def rotation_split_residual_rad(
    rotation: object, axes: Sequence[str], angles: Sequence[float]
) -> float:
    """Geodesic angle between ``rotation`` and ``rot(axes[0], angles[0]) @ ...``."""

    matrix = _as_rotation(rotation)
    return _geodesic_rad(matrix, reconstruct_from_angles(tuple(axes), tuple(angles)))


def _project_single_axis(matrix: np.ndarray, axis: str) -> float:
    """Signed angle of the rotation ``matrix`` performs about one canonical axis.

    This the exact minimiser of the geodesic distance to the target over a single
    revolute joint about ``axis``: it is the ``atan2`` form of "which angle puts
    ``rot(axis, t)`` closest to ``matrix``", and it is the same projection the
    shipped single-axis rig used.
    """

    axis_index = _AXIS_INDEX[axis]
    first = (axis_index + 1) % 3
    second = (axis_index + 2) % 3
    return math.atan2(
        float(matrix[second, first] - matrix[first, second]),
        float(matrix[first, first] + matrix[second, second]),
    )


def _relabel(axes: Sequence[str]) -> tuple[np.ndarray, float]:
    """Rotation ``Q`` mapping the declared axes onto ``x, y, z``, plus ``det Q``.

    ``Q`` sends the declared axis in slot ``i`` to the canonical axis in slot
    ``i``, so conjugating a rotation by it reads the declared chain as a
    canonical one. Slots beyond the declared chain are filled with whatever axes
    are left, which keeps ``Q`` a true permutation even for a one- or two-axis
    chain. ``det Q`` is ``+1`` for an even axis order and ``-1`` for an odd one;
    a reflection reverses the sense of a rotation, so the caller multiplies the
    solved angles by it.
    """

    used = [_AXIS_INDEX[axis] for axis in axes]
    remaining = [index for index in range(3) if index not in used]
    matrix = np.zeros((3, 3))
    for slot, axis_index in enumerate(used + remaining):
        matrix[slot, axis_index] = 1.0
    return matrix, float(round(float(np.linalg.det(matrix))))


def _canonical_pair(matrix: np.ndarray) -> tuple[float, float]:
    """Closed form for ``Rx(p) Ry(q) == matrix``.

    The third column of ``Rx Ry`` is ``(sin q, -sin p cos q, cos p cos q)``, so
    ``q`` comes from the first entry against the other two and ``p`` from the
    pair that remains.
    """

    q = math.atan2(float(matrix[0, 2]), math.hypot(float(matrix[1, 2]), float(matrix[2, 2])))
    p = math.atan2(-float(matrix[1, 2]), float(matrix[2, 2]))
    return p, q


def _canonical_triple(matrix: np.ndarray) -> list[tuple[float, float, float]]:
    """Both closed-form solutions of ``Rx(r) Ry(p) Rz(y) == matrix``.

    With the columns of ``Rx Ry Rz`` written out, ``R[0, 2] = sin p`` and
    ``hypot(R[1, 2], R[2, 2]) = |cos p|`` pin ``p``; the remaining entries give
    ``r`` and ``y``. Flipping the sign of ``cos p`` -- ``p -> pi - p`` -- is the
    second solution, with both outer angles shifted by ``pi``.
    """

    pitch = math.atan2(float(matrix[0, 2]), math.hypot(float(matrix[1, 2]), float(matrix[2, 2])))
    roll = math.atan2(-float(matrix[1, 2]), float(matrix[2, 2]))
    yaw = math.atan2(-float(matrix[0, 1]), float(matrix[0, 0]))
    mirror = (roll + math.pi, math.pi - pitch, yaw + math.pi)
    return [(roll, pitch, yaw), mirror]


def _wrap_angle(angle: float) -> float:
    """Fold an angle into ``(-pi, pi]``.

    ``rot(axis, t)`` and ``rot(axis, t + 2 pi)`` are the same rotation, so this
    changes nothing about the pose while keeping the returned angles in the range
    a joint limit is expressed in. A three-axis solve happily returns 186 degrees
    for a joint whose limit is 90 when the equivalent -174 is what the branch
    actually meant.
    """

    wrapped = math.remainder(angle, 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


def _limit_violation_deg(
    angles: Sequence[float], limits_deg: Sequence[tuple[float, float]] | None
) -> float:
    """Total degrees by which ``angles`` fall outside ``limits_deg``."""

    if limits_deg is None:
        return 0.0
    return float(
        sum(
            max(0.0, low - math.degrees(angle), math.degrees(angle) - high)
            for angle, (low, high) in zip(angles, limits_deg, strict=True)
        )
    )


def _refine(
    matrix: np.ndarray, axes: Sequence[str], angles: Sequence[float]
) -> tuple[float, ...]:
    """Polish a chained decomposition by exact coordinate descent.

    With ``R = A_0 ... A_{n-1}``, joint ``p``'s own contribution is isolated by
    undoing the earlier joints from the **left** and the later ones from the
    **right**:

        ``A_p = (A_0 ... A_{p-1})^T @ R @ (A_{p+1} ... A_{n-1})^T``

    Reading ``p``'s angle off that product is the exact minimiser over ``p``
    alone, so every step can only reduce the residual. Angles are indexed
    positionally rather than looked up by token, so two joints sharing an axis
    name cannot be confused.
    """

    values = [float(value) for value in angles]
    previous = math.inf
    for _ in range(SPLIT_REFINEMENT_SWEEPS):
        for position, axis in enumerate(axes):
            earlier = np.eye(3)
            for index in range(position):
                earlier = earlier @ rotation_about_axis(axes[index], values[index])
            later = np.eye(3)
            for index in range(position + 1, len(axes)):
                later = later @ rotation_about_axis(axes[index], values[index])
            values[position] = _project_single_axis(earlier.T @ matrix @ later.T, axis)
        current = _geodesic_rad(matrix, reconstruct_from_angles(axes, values))
        if current <= SPLIT_TOLERANCE_RAD or previous - current <= 0.0:
            # Converged, or no further progress is available at double precision.
            break
        previous = current
    return tuple(values)


def _best_of(
    matrix: np.ndarray,
    axes: Sequence[str],
    seeds: Sequence[Sequence[float]],
    limits_deg: Sequence[tuple[float, float]] | None = None,
) -> AngularSplit:
    """Polish every seed, verify by reconstruction, and keep the best usable one.

    A closed-form seed is exact when the chain can reach the target; when it
    cannot, the seed is only a starting point and the refinement reports the true
    nearest reachable rotation. Among the candidates that reproduce the target
    equally well, the one that stays inside the declared limits is preferred,
    because an exactly-representable pose that the rig then clips is replayed
    wrong. Returning a verified candidate means a wrong branch or a stalled seed
    degrades the answer instead of corrupting it.
    """

    candidates: list[tuple[float, ...]] = []
    for seed in seeds:
        candidate = tuple(_wrap_angle(float(value)) for value in seed)
        residual = rotation_split_residual_rad(matrix, axes, candidate)
        if residual > SPLIT_TOLERANCE_RAD:
            candidate = tuple(
                _wrap_angle(float(value)) for value in _refine(matrix, axes, candidate)
            )
            residual = rotation_split_residual_rad(matrix, axes, candidate)
        candidates.append(candidate)
    exact = [
        candidate
        for candidate in candidates
        if rotation_split_residual_rad(matrix, axes, candidate) <= SPLIT_TOLERANCE_RAD
    ]
    pool = exact or candidates
    # Among equally exact representations: respect the limits first, then prefer the
    # least extreme angles. Two Euler branches can both land inside a wide range while
    # one of them needs 134 degrees for a knee to do it, and an angle that extreme is
    # both physically wrong and fragile if anything downstream clamps it.
    best = min(
        pool,
        key=lambda candidate: (
            _limit_violation_deg(candidate, limits_deg),
            max(abs(math.degrees(value)) for value in candidate),
            rotation_split_residual_rad(matrix, axes, candidate),
        ),
    )
    return AngularSplit(
        best,
        rotation_split_residual_rad(matrix, axes, best),
        _limit_violation_deg(best, limits_deg),
    )


def split_rotation(
    rotation: object,
    axes: Sequence[str],
    *,
    axis_angle: object = None,
    limits_deg: Sequence[tuple[float, float]] | None = None,
) -> AngularSplit:
    """Decompose ``rotation`` into a chain of canonical-axis rotations.

    ``axes`` lists the rig's declared rotation axes in the order the planner
    chains them, so ``split_rotation(R, ("x", "y", "z"))`` returns the angles
    ``(tx, ty, tz)`` with ``rot(x, tx) @ rot(y, ty) @ rot(z, tz) == R``.

    A single axis reproduces the projection the shipped single-axis rig already
    used -- the recorded rotation's component about that axis, with the off-axis
    part reported as the residual -- so an unchanged configuration maps
    identically. Two and three axes are solved in closed form and polished.

    ``axis_angle`` optionally supplies the target in its original axis-angle form;
    the single-axis path reads its component directly, which keeps that path
    bit-identical to the old behaviour rather than merely equivalent.
    """

    canonical_axes = tuple(validate_axis(axis) for axis in axes)
    if not canonical_axes:
        raise ValueError("split_rotation needs at least one axis")
    if len(set(canonical_axes)) != len(canonical_axes):
        raise ValueError(
            "split_rotation needs distinct axes (a repeated axis adds no freedom), "
            f"got {list(axes)}"
        )
    matrix = _as_rotation(rotation)
    count = len(canonical_axes)
    if count == 1:
        # Deliberate: the shipped rig's semantics are the axis-angle component,
        # not the geodesic-optimal angle about the axis. They agree for every
        # axis-aligned reference and differ only for a motion this rig cannot
        # express anyway, where the residual is the number that matters.
        vector = (
            validate_axis_angle(axis_angle, "axis-angle")
            if axis_angle is not None
            else matrix_to_axis_angle(matrix)
        )
        component = float(vector[_AXIS_INDEX[canonical_axes[0]]])
        off_axis = float(np.linalg.norm(np.delete(vector, _AXIS_INDEX[canonical_axes[0]])))
        return AngularSplit((component,), off_axis)
    relabel, determinant = _relabel(canonical_axes)
    relabelled = relabel @ matrix @ relabel.T
    canonical_chain = _CANONICAL_AXES[:count]
    if count == 2:
        seeds: list[tuple[float, ...]] = [tuple(_canonical_pair(relabelled))]
    else:
        seeds = [tuple(seed) for seed in _canonical_triple(relabelled)]
    # An odd axis order is a reflection, which reverses the sense of every angle. The
    # solver works in the canonical frame, so a declared limit [lo, hi] reads as
    # [-hi, -lo] there, and the solved angles are negated on the way back. Applying
    # the reflection to the limits as well keeps the "prefer what the rig can hold"
    # choice measured against the declarations the caller actually made.
    canonical_limits = (
        None
        if limits_deg is None
        else [(lo, hi) if determinant > 0 else (-hi, -lo) for lo, hi in limits_deg]
    )
    solved = _best_of(relabelled, canonical_chain, seeds, canonical_limits)
    angles = tuple(determinant * value for value in solved.angles)
    return AngularSplit(
        angles,
        rotation_split_residual_rad(matrix, canonical_axes, angles),
        _limit_violation_deg(angles, limits_deg),
    )
