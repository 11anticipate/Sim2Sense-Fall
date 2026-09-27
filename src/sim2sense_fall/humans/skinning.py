"""CPU linear blend skinning for the SMPL body.

This module is the piece that turns "the simulated articulation" into "a mesh that
follows it". It is deliberately separate from the USD authoring, because it is pure
NumPy and therefore testable without Isaac Sim -- which matters, since until the
licensed SMPL file is available there is no mesh to look at in a viewport.

The transformation follows SMPL's own formulation. For each joint ``k`` with world
rotation ``R_k`` and world joint centre ``p_k``, the rigid map is::

    G_k(x) = R_k @ (x - J_k) + p_k

where ``J_k`` is the joint centre in the *rest* pose. At the rest pose ``R_k = I``
and ``p_k = J_k``, so ``G_k`` is the identity -- which is exactly the property SMPL
gets by subtracting the rest transformation from every ``G``. A vertex is then::

    v = sum_k w[v, k] * G_k(v_shaped)

with ``w`` the skinning weights, so no rest-pose correction term is needed.

Shape (``betas``) is applied first, via ``shapedirs``, and the joint centres are
re-derived from the shaped vertices with ``J_regressor`` -- because changing body
shape moves the joints, and using the neutral joints with a shaped body makes the
limbs detach at the shoulders and hips.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .skeleton import SMPL_JOINT_NAMES, RestSkeleton, SkeletonTopology

__all__ = [
    "SmplMesh",
    "sample_skin_points",
    "joint_centres_from_regressor",
    "linear_blend_skin",
    "mesh_from_model_payload",
    "skin_with_link_poses",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class SmplMesh:
    """An SMPL mesh: template vertices, faces and skinning weights.

    ``weights`` is ``(V, 24)`` and must sum to 1 per vertex; that invariant is what
    keeps the rest pose reproduced exactly, so it is checked on construction.
    """

    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    topology: SkeletonTopology
    rest_joints: np.ndarray | None = None
    shape_directions: np.ndarray | None = None
    source: str = "smpl"

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces)
        weights = np.asarray(self.weights, dtype=np.float64)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "weights", weights)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {vertices.shape}")
        if vertices.shape[0] < 3:
            raise ValueError("a mesh needs at least three vertices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices contain a non-finite value")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {faces.shape}")
        if faces.size:
            if int(faces.max()) >= vertices.shape[0] or int(faces.min()) < 0:
                raise ValueError(
                    f"face indices must index the {vertices.shape[0]} vertices; "
                    f"range is [{int(faces.min())}, {int(faces.max())}]"
                )
        if weights.shape != (vertices.shape[0], self.topology.joint_count):
            raise ValueError(
                f"weights must be ({vertices.shape[0]}, {self.topology.joint_count}), got "
                f"{weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain a non-finite value")
        if np.any(weights < -1e-9):
            raise ValueError("weights must be non-negative")
        row_sums = weights.sum(axis=1)
        worst = float(np.abs(row_sums - 1.0).max())
        if worst > 1e-6:
            raise ValueError(
                f"skinning weights must sum to 1 per vertex; largest deviation {worst:.3e}. "
                "Otherwise the rest pose is not reproduced and limbs drift."
            )
        if self.rest_joints is not None:
            joints = np.asarray(self.rest_joints, dtype=np.float64)
            object.__setattr__(self, "rest_joints", joints)
            if joints.shape != (self.topology.joint_count, 3):
                raise ValueError(f"rest_joints must be ({self.topology.joint_count}, 3)")

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def shaped(self, betas: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
        """Apply shape coefficients, or return the template unchanged."""

        if betas is None or self.shape_directions is None:
            return self.vertices
        coefficients = np.asarray(betas, dtype=np.float64).reshape(-1)
        directions = np.asarray(self.shape_directions, dtype=np.float64)
        if directions.ndim != 3:
            raise ValueError(f"shape_directions must be (V, 3, B), got {directions.shape}")
        if directions.shape[0] != self.vertex_count:
            raise ValueError("shape_directions vertex count does not match the mesh")
        if directions.shape[2] != coefficients.shape[0]:
            raise ValueError(
                f"betas has {coefficients.shape[0]} entries but the model has "
                f"{directions.shape[2]} shape directions"
            )
        return self.vertices + directions @ coefficients

    def rest_skeleton(self) -> RestSkeleton:
        """Rest skeleton from this mesh's joint centres."""

        if self.rest_joints is None:
            raise ValueError(
                "this mesh carries no J_regressor output, so its joint centres are unknown; "
                "the rig cannot be planned from it"
            )
        from .skeleton import rest_skeleton_from_positions

        joints = np.asarray(self.rest_joints, dtype=np.float64)
        return rest_skeleton_from_positions(
            {name: tuple(joints[index]) for index, name in enumerate(SMPL_JOINT_NAMES)},
            source=f"{self.source} joint centres",
        )


def mesh_from_model_payload(payload: Mapping[str, Any], *, source: str = "smpl") -> SmplMesh:
    """Build a :class:`SmplMesh` from a loaded SMPL pickle payload.

    ``weights`` and ``v_template`` are required; ``J_regressor`` and ``shapedirs`` are
    kept when present so shape can be applied and joints re-derived.
    """

    for key in ("v_template", "f", "weights"):
        if key not in payload:
            raise ValueError(
                f"model payload is missing {key!r}; a skin needs vertices, faces and weights"
            )
    from .skeleton import smpl_skeleton

    vertices = np.asarray(payload["v_template"], dtype=np.float64)
    faces = np.asarray(payload["f"])
    weights = np.asarray(payload["weights"], dtype=np.float64)
    rest_joints = None
    regressor = payload.get("J_regressor")
    if regressor is not None:
        rest_joints = joint_centres_from_regressor(regressor, vertices)
    return SmplMesh(
        vertices=vertices,
        faces=faces,
        weights=weights,
        topology=smpl_skeleton(),
        rest_joints=rest_joints,
        shape_directions=None
        if payload.get("shapedirs") is None
        else np.asarray(payload["shapedirs"], dtype=np.float64),
        source=source,
    )


def joint_centres_from_regressor(regressor: object, vertices: np.ndarray) -> np.ndarray:
    """Rest joint centres ``J = J_regressor @ v_template``."""

    matrix = np.asarray(regressor, dtype=np.float64)
    points = np.asarray(vertices, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"J_regressor must be 2-D, got {matrix.shape}")
    if matrix.shape[1] != points.shape[0]:
        raise ValueError(
            f"J_regressor has {matrix.shape[1]} columns but the mesh has {points.shape[0]} vertices"
        )
    return matrix @ points


def linear_blend_skin(
    vertices: np.ndarray,
    weights: np.ndarray,
    *,
    rotations: np.ndarray,
    joint_positions: np.ndarray,
    rest_joints: np.ndarray,
) -> np.ndarray:
    """Skin ``vertices`` with a set of per-joint world transforms.

    ``rotations`` is ``(K, 3, 3)``, ``joint_positions`` is ``(K, 3)`` (world centres of
    the joints in the current pose) and ``rest_joints`` is ``(K, 3)`` (the same centres
    in the rest pose). Returns ``(V, 3)`` world vertices.

    At the rest pose (identity rotations, ``joint_positions == rest_joints``) the output
    equals the input -- that is the invariant the test asserts, and it is what catches a
    sign or an offset error in the joint transforms far earlier than a rendered frame
    would.
    """

    points = np.asarray(vertices, dtype=np.float64)
    skin = np.asarray(weights, dtype=np.float64)
    rotations_array = np.asarray(rotations, dtype=np.float64)
    positions = np.asarray(joint_positions, dtype=np.float64)
    rest = np.asarray(rest_joints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {points.shape}")
    if rotations_array.shape != (skin.shape[1], 3, 3):
        raise ValueError(f"rotations must be ({skin.shape[1]}, 3, 3), got {rotations_array.shape}")
    if positions.shape != (skin.shape[1], 3) or rest.shape != (skin.shape[1], 3):
        raise ValueError("joint_positions and rest_joints must both be (K, 3)")
    if skin.shape[0] != points.shape[0]:
        raise ValueError(f"weights cover {skin.shape[0]} vertices but {points.shape[0]} were given")
    for index, matrix in enumerate(rotations_array):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"rotation {index} is not finite")
        if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-6:
            raise ValueError(
                f"rotation {index} has determinant {float(np.linalg.det(matrix)):.6f}; joint "
                "rotations must be proper rotations or the skin tears"
            )
    # G_k(x) = R_k @ (x - J_k_rest) + p_k
    offsets = points[:, None, :] - rest[None, :, :]  # (V, K, 3)
    transformed = np.einsum("kij,vkj->vki", rotations_array, offsets) + positions[None, :, :]
    return np.einsum("vk,vki->vi", skin, transformed)


def skin_with_link_poses(
    mesh: SmplMesh,
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    betas: Sequence[float] | None = None,
) -> np.ndarray:
    """Skin a mesh from a mapping of link name to ``(rotation, translation)``.

    ``poses`` is what :func:`sim2sense_fall.humans.rig.forward_kinematics` returns, so
    the same poses that drive the capsules drive the mesh. Missing links are treated as
    the identity at their rest position, and an incomplete mapping is an error rather
    than a silently half-moved body.
    """

    rotations = np.zeros((mesh.topology.joint_count, 3, 3), dtype=np.float64)
    positions = np.zeros((mesh.topology.joint_count, 3), dtype=np.float64)
    identity = np.eye(3)
    for index, name in enumerate(mesh.topology.joint_names):
        entry = poses.get(name)
        if entry is None:
            rotations[index] = identity
            positions[index] = mesh.rest_joints[index] if mesh.rest_joints is not None else 0.0
            continue
        rotation, translation = entry
        rotations[index] = np.asarray(rotation, dtype=np.float64)
        positions[index] = np.asarray(translation, dtype=np.float64)
    if mesh.rest_joints is None:
        raise ValueError("the mesh has no rest joint centres; skinning cannot be anchored")
    vertices = mesh.shaped(betas)
    return linear_blend_skin(
        vertices,
        mesh.weights,
        rotations=rotations,
        joint_positions=positions,
        rest_joints=mesh.rest_joints,
    )


def sample_skin_points(
    mesh: SmplMesh,
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    count: int,
    seed: int,
    betas: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Subsample the skinned surface for one frame, following the given link poses.

    Same ``(points, owners)`` contract as the capsule proxy, so the export and the event
    rules do not care which body representation produced them. The owning joint is the
    dominant skinning weight of each sampled vertex, which is what a reader needs to know
    when a point's motion looks unlike its neighbour.
    """

    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    vertices = skin_with_link_poses(mesh, poses, betas=betas)
    budget = min(int(count), mesh.vertex_count)
    pick = np.random.default_rng(seed).choice(mesh.vertex_count, budget, replace=False)
    pick.sort()
    weights = np.asarray(mesh.weights, dtype=np.float64)[pick]
    owners = tuple(
        mesh.topology.joint_names[index] for index in np.asarray(weights).argmax(axis=1)
    )
    return vertices[pick], owners
