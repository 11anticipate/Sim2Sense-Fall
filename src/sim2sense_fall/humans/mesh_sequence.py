"""Fixed-topology world-space mesh sequences for the wireless sensing stage.

The Sionna adapter needs an actual surface mesh at every sample time.  Point
clouds are still useful for event labelling, but they do not carry connectivity
and cannot be passed to a ray tracer without inventing triangles downstream.
This module keeps that boundary explicit:

* ``smpl_skin_mesh`` uses the licensed SMPL vertices, faces and skinning weights.
* ``capsule_proxy_mesh`` is a deterministic, closed triangle mesh generated from
  the physical capsule links when the licensed asset is unavailable.

Both representations use world coordinates in metres, Z-up, ``(x, y, z)`` and a
single face array shared by every frame.  The proxy is an interface smoke test,
not a claim that SMPL was downloaded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .rig import HumanRigPlan, LinkTransform
from .skinning import SmplMesh, skin_with_link_poses

if TYPE_CHECKING:
    from .rig import CapsuleSpec

__all__ = [
    "MeshTopology",
    "MeshSequence",
    "ProxyMeshTemplate",
    "build_capsule_proxy_template",
    "pose_capsule_proxy_mesh",
    "fit_mesh_to_rest_joints",
    "skin_mesh_sequence_frame",
]


@dataclass(frozen=True, slots=True)
class MeshTopology:
    """One fixed triangular connectivity array."""

    faces: np.ndarray

    def __post_init__(self) -> None:
        faces = np.asarray(self.faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"mesh faces must have shape (F, 3), got {faces.shape}")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("mesh face indices must be integers")
        faces = faces.astype(np.int64, copy=True)
        if faces.size and int(faces.min()) < 0:
            raise ValueError("mesh face indices must be non-negative")
        faces.setflags(write=False)
        object.__setattr__(self, "faces", faces)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def validate_vertex_count(self, vertex_count: int) -> None:
        if self.faces.size and int(self.faces.max()) >= int(vertex_count):
            raise ValueError(
                f"mesh face index {int(self.faces.max())} is outside {vertex_count} vertices"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.faces.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MeshSequence:
    """A fixed-topology sequence in the coordinate contract consumed by Sionna."""

    time_s: np.ndarray
    vertices_xyz: np.ndarray
    topology: MeshTopology
    representation: str
    coordinate_system: str = "world_z_up_xyz"
    units: str = "m"
    vertex_owners: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        times = np.asarray(self.time_s, dtype=np.float64)
        vertices = np.asarray(self.vertices_xyz, dtype=np.float64)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("mesh sequence needs at least two one-dimensional time samples")
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("mesh sequence times must be finite and strictly increasing")
        if vertices.ndim != 3 or vertices.shape[0] != len(times) or vertices.shape[2] != 3:
            raise ValueError(
                f"mesh vertices must have shape (N, V, 3) matching time, got {vertices.shape}"
            )
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh vertices contain a non-finite value")
        self.topology.validate_vertex_count(int(vertices.shape[1]))
        if self.representation not in ("smpl_skin_mesh", "capsule_proxy_mesh"):
            raise ValueError(f"unsupported mesh representation {self.representation!r}")
        if self.coordinate_system != "world_z_up_xyz":
            raise ValueError("mesh coordinate_system must be 'world_z_up_xyz'")
        if self.units != "m":
            raise ValueError("mesh units must be metres ('m')")
        if self.vertex_owners and len(self.vertex_owners) != vertices.shape[1]:
            raise ValueError("vertex_owners must have one entry per mesh vertex")
        times.setflags(write=False)
        vertices.setflags(write=False)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "vertices_xyz", vertices)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_xyz.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.vertices_xyz.shape[0])

    @property
    def faces(self) -> np.ndarray:
        return self.topology.faces

    def as_metadata(self) -> dict[str, object]:
        return {
            "mesh_representation": self.representation,
            "mesh_coordinate_system": self.coordinate_system,
            "mesh_units": self.units,
            "mesh_is_fixed_topology": True,
            "mesh_vertex_count": self.vertex_count,
            "mesh_face_count": self.topology.face_count,
            "mesh_topology_hash": self.topology.sha256,
        }


@dataclass(frozen=True, slots=True)
class ProxyMeshTemplate:
    """Rest-frame vertices and faces for all capsule links."""

    vertices_local: np.ndarray
    topology: MeshTopology
    owners: tuple[str, ...]
    vertex_links: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_local, dtype=np.float64)
        links = np.asarray(self.vertex_links, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("proxy vertices_local must be (V, 3)")
        if links.shape != (vertices.shape[0],):
            raise ValueError("proxy vertex_links must have one link index per vertex")
        self.topology.validate_vertex_count(len(vertices))
        if len(self.owners) != len(vertices):
            raise ValueError("proxy owners must have one entry per vertex")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("proxy vertices contain a non-finite value")
        vertices.setflags(write=False)
        links.setflags(write=False)
        object.__setattr__(self, "vertices_local", vertices)
        object.__setattr__(self, "vertex_links", links)


def _capsule_vertices_and_faces(
    capsule: CapsuleSpec, *, segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Make a closed UV-like capsule in the capsule's link-local frame."""

    if segments < 6:
        raise ValueError("capsule mesh needs at least 6 radial segments")
    radius = float(capsule.radius_m)
    half = float(capsule.cylinder_length_m) / 2.0
    # Pole, three lower rings, two cylinder rings, three upper rings, pole.
    ring_specs = (
        (0.0, -half - radius),
        (0.5 * radius, -half - np.sqrt(3.0) * radius / 2.0),
        (np.sqrt(3.0) * radius / 2.0, -half - radius / 2.0),
        (radius, -half),
        (radius, half),
        (np.sqrt(3.0) * radius / 2.0, half + radius / 2.0),
        (0.5 * radius, half + np.sqrt(3.0) * radius / 2.0),
        (0.0, half + radius),
    )
    local_rings: list[np.ndarray] = []
    for radial, z in ring_specs:
        if radial == 0.0:
            local_rings.append(np.array([[0.0, 0.0, z]], dtype=np.float64))
        else:
            angles = np.arange(segments, dtype=np.float64) * 2.0 * np.pi / segments
            local_rings.append(
                np.column_stack(
                    (radial * np.cos(angles), radial * np.sin(angles), np.full(segments, z))
                )
            )
    vertices = np.concatenate(local_rings, axis=0)
    vertices = vertices @ _quaternion_matrix(capsule.orientation_wxyz).T
    vertices += np.asarray(capsule.center, dtype=np.float64)
    offsets = np.cumsum([0, *[len(ring) for ring in local_rings[:-1]]])
    faces: list[tuple[int, int, int]] = []
    for lower, upper in zip(range(len(local_rings) - 1), range(1, len(local_rings)), strict=True):
        a, b = int(offsets[lower]), int(offsets[upper])
        n_a, n_b = len(local_rings[lower]), len(local_rings[upper])
        if n_a == 1:
            faces.extend((a, b + index, b + (index + 1) % n_b) for index in range(n_b))
        elif n_b == 1:
            faces.extend((a + index, b, a + (index + 1) % n_a) for index in range(n_a))
        else:
            faces.extend((a + index, b + index, b + (index + 1) % n_b) for index in range(segments))
            faces.extend(
                (a + index, b + (index + 1) % n_b, a + (index + 1) % n_a)
                for index in range(segments)
            )
    return vertices, np.asarray(faces, dtype=np.int64)


def _quaternion_matrix(values: tuple[float, float, float, float]) -> np.ndarray:
    from .rotations import quaternion_to_matrix

    return quaternion_to_matrix(values)


def build_capsule_proxy_template(plan: HumanRigPlan, *, segments: int = 12) -> ProxyMeshTemplate:
    """Create deterministic fixed topology for all collidable links."""

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    owners: list[str] = []
    link_indices: list[np.ndarray] = []
    vertex_offset = 0
    for link_index, link in enumerate(plan.links):
        capsule = link.capsule
        if capsule is None:
            continue
        local, local_faces = _capsule_vertices_and_faces(capsule, segments=segments)
        vertices.append(local)
        faces.append(local_faces + vertex_offset)
        owners.extend([link.name] * len(local))
        link_indices.append(np.full(len(local), link_index, dtype=np.int64))
        vertex_offset += len(local)
    if not vertices:
        raise ValueError("rig plan has no capsule links; cannot build a proxy mesh")
    return ProxyMeshTemplate(
        vertices_local=np.concatenate(vertices, axis=0),
        topology=MeshTopology(np.concatenate(faces, axis=0)),
        owners=tuple(owners),
        vertex_links=np.concatenate(link_indices),
    )


def pose_capsule_proxy_mesh(
    template: ProxyMeshTemplate,
    plan: HumanRigPlan,
    poses: Mapping[str, LinkTransform],
) -> np.ndarray:
    """Transform proxy vertices with the actual world link poses."""

    output = np.empty_like(template.vertices_local)
    for link_index, link in enumerate(plan.links):
        indices = np.flatnonzero(template.vertex_links == link_index)
        if len(indices) == 0:
            continue
        pose = poses.get(link.name)
        if pose is None:
            raise KeyError(f"missing physics pose for link {link.name!r}")
        output[indices] = (pose.rotation @ template.vertices_local[indices].T).T + pose.translation
    return output


def joint_row_alignment(mesh: SmplMesh, target_joints: np.ndarray) -> np.ndarray:
    """Permutation taking ``mesh.rest_joints`` rows onto the rig's joint rows.

    Both arrays are indexed by the same *names* -- the mesh through
    ``mesh.topology.joint_names``, the rig through
    :attr:`~sim2sense_fall.humans.rig.HumanRigPlan.rest_joint_positions` -- and the names
    are the correspondence that actually carries meaning, so identity is the pairing.
    Geometry is used only to *confirm* it.

    Why the confirmation is worth having. The rig is rescaled to the configured standing
    height while the licensed file keeps its native stature, so a correct pairing leaves
    the raw arrays tens of millimetres apart -- 53 mm at the neutral file's ankle and
    91 mm at the foot. Those gaps are wide enough that a nearest-row search starts
    picking the wrong row: measured at a 1.75 m configuration, a bare ``argmin`` over
    rows returns a *non-injective* map and therefore a body scale of ``1.0794`` where the
    true uniform scale is ``1.0459``. Nothing raises; the exported skin just shrinks away
    from the links it is skinned to. A future rig that reorders its rows would fail the
    same way, silently.

    The confirmation compares **inter-joint distances**, which a uniform scale changes by
    one common factor and which are invariant under the translation between the two sets.
    Absolute positions are unusable for this: scaling about the origin moves the root
    itself, so a pure ``1.08`` scale legitimately puts the spine joints nearer to the
    mesh's pelvis than to their own counterparts. Bone *directions* alone are also
    insufficient, because a spine is collinear -- three joints share one direction and
    a direction test cannot tell them apart. Pairwise distances do distinguish them.

    Returns ``indices`` with ``mesh.rest_joints[indices[k]]`` corresponding to
    ``target_joints[k]``.
    """

    if mesh.rest_joints is None:
        raise ValueError("mesh rest joints are required to align a mesh to the rig")
    target = np.asarray(target_joints, dtype=np.float64)
    source = np.asarray(mesh.rest_joints, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target_joints must be (K, 3), got {target.shape}")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"mesh rest joints must be (K, 3), got {source.shape}")
    if target.shape[0] != source.shape[0]:
        raise ValueError(
            f"the rig has {target.shape[0]} joints but the mesh has {source.shape[0]}; "
            "a per-joint correspondence cannot be built"
        )

    names = tuple(mesh.topology.joint_names)
    if len(names) != source.shape[0]:
        raise ValueError(
            f"the mesh topology names {len(names)} joints but its rest joints are "
            f"{source.shape[0]} rows; a name-to-row correspondence is undefined"
        )
    indices = np.arange(source.shape[0])

    source_distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    target_distances = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    spread = float(source_distances.max())
    if spread <= 1e-12 or float(target_distances.max()) <= 1e-12:
        raise ValueError(
            "joint positions have no spread; a row correspondence cannot be recognised"
        )
    # Pairwise distances are equal up to the single unknown scale, so normalising each
    # set by its own diameter makes them directly comparable and scale-free. The row
    # being tested and the joint being matched are masked out of each comparison,
    # because both self-distances are structurally zero and would otherwise dominate.
    scale = float(target_distances.max())
    keep = ~np.eye(len(names), dtype=bool)
    residual = np.empty((len(names), len(names)), dtype=np.float64)
    for row in range(len(names)):
        for joint in range(len(names)):
            shared = keep[row] & keep[joint]
            residual[row, joint] = float(
                np.abs(
                    source_distances[row, shared] / spread - target_distances[joint, shared] / scale
                ).mean()
            )
    chosen = residual.argmin(axis=0)
    mismatched = np.where(chosen != indices)[0]
    if mismatched.size:
        index = int(mismatched[0])
        row = int(chosen[index])
        raise ValueError(
            f"joint {index} ({names[index]}) corresponds to mesh row {index} by name, but "
            f"mesh row {row} ({names[row]}) reproduces the rig's joint geometry far better "
            f"(mean relative distance residual {residual[row, index]:.2e} against "
            f"{residual[index, index]:.2e}); the mesh and the rig disagree about which "
            "joint is which"
        )
    if len(set(chosen.tolist())) != len(chosen):
        raise ValueError(
            "the mesh and the rig do not correspond one-to-one: two rig joints share the "
            "same best-matching mesh row, so the joint sets are not comparable"
        )
    return indices


def fit_mesh_to_rest_joints(mesh: SmplMesh, target_joints: np.ndarray) -> SmplMesh:
    """Apply the rig's uniform body scale and local translation to a loaded mesh.

    The imported SMPL file is measured in its native stature, while the rig config may
    request another standing height. Skinning an unscaled mesh with scaled link poses
    makes the surface detach from the capsules, so a least-squares scalar fit is applied
    to both vertices and rest joint centres.

    The fit is computed over the name-matched correspondence from
    :func:`joint_row_alignment`, and the residual is then checked: "uniform body scale
    plus local translation" is a *similarity*, so the joint offsets must map onto each
    other by a single scalar. If they do not, the two joint sets disagree about the
    body's orientation and applying the scalar anyway would rotate the exported skin
    away from the links. Measured on the shipped neutral file the residual is 0 um, so
    the check costs nothing and catches a frame mismatch the instant one is introduced.
    """

    indices = joint_row_alignment(mesh, target_joints)
    target = np.asarray(target_joints, dtype=np.float64)
    assert mesh.rest_joints is not None
    source = np.asarray(mesh.rest_joints, dtype=np.float64)[indices]
    source_centered = source - source[0]
    target_centered = target - target[0]
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1e-12:
        raise ValueError("mesh rest joints have zero spread; cannot fit scale")
    scale = float(np.sum(source_centered * target_centered) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"mesh-to-rig scale must be finite and positive, got {scale}")
    residual = target_centered - scale * source_centered
    worst = float(np.linalg.norm(residual, axis=1).max())
    spread = float(np.linalg.norm(target_centered, axis=1).max())
    if spread > 0 and worst / spread > 1e-3:
        raise ValueError(
            "the mesh-to-rig fit is not a uniform scale: worst residual is "
            f"{worst * 1e3:.2f} mm on a body whose joints span {spread:.3f} m "
            f"({worst / spread:.1%} of the span). The mesh and the rig disagree about "
            "the body frame, so scaling the vertices would rotate the skin away from "
            "the links it is skinned to."
        )
    offset = target[0] - scale * source[0]
    return SmplMesh(
        vertices=mesh.vertices * scale + offset,
        faces=mesh.faces,
        weights=mesh.weights,
        topology=mesh.topology,
        rest_joints=target,
        shape_directions=None if mesh.shape_directions is None else mesh.shape_directions * scale,
        source=mesh.source,
    )


def skin_mesh_sequence_frame(
    mesh: SmplMesh, poses: Mapping[str, LinkTransform], *, betas: np.ndarray | None = None
) -> np.ndarray:
    """Skin every SMPL template vertex from one physics frame."""

    return skin_with_link_poses(
        mesh,
        {name: (pose.rotation, pose.translation) for name, pose in poses.items()},
        betas=betas,
    )
