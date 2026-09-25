"""SMPL / SMPL-H joint topology and the articulated rest skeleton.

What is a verified fact here, and what is a stand-in
----------------------------------------------------

The **joint names and the kinematic tree** are the published SMPL skeleton: 24
joints, with joint 0 the pelvis and a topologically sorted parent table. The
body joints 0..21 are shared between SMPL and SMPL-H, which is what makes an
AMASS (SMPL-H) sequence usable on an SMPL rig without renaming anything. The
same table is reproduced by the ``smplx`` reference implementation, the
Meshcapade skeleton documentation and several public SMPL loaders; the loader in
:mod:`sim2sense_fall.humans.assets` re-derives it from the licensed file's
``kintree_table`` whenever that file is present, and
:func:`skeleton_from_kintree_table` fails loudly if the two disagree.

The **rest joint positions** below are *not* SMPL data. They are a hand-set
procedural standing skeleton for a nominal 1.70 m adult, used so the whole
pipeline can be exercised, unit-tested and reviewed before the licence-gated
model file is available. When the real model is present the joint centres come
from ``J_regressor @ v_template`` (plus the shape blend) and these numbers are
replaced. Any artefact produced from the nominal skeleton says so in its
provenance block: it is a *topology-faithful* rig, not a fitted human body.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .rotations import AXES, validate_axis

__all__ = [
    "BODY_JOINT_COUNT",
    "NOMINAL_HEIGHT_M",
    "SMPL_HAND_JOINT_NAMES",
    "SMPL_JOINT_NAMES",
    "SMPL_KINEMATIC_PARENTS",
    "RestSkeleton",
    "SkeletonTopology",
    "default_rest_skeleton",
    "nominal_rest_joint_positions",
    "rest_skeleton_from_positions",
    "skeleton_from_kintree_table",
    "smpl_skeleton",
]

#: Joint names of the released SMPL body model, in index order.
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

#: Parent index per joint; ``-1`` marks the root. ``SMPL``'s ``kintree_table``.
SMPL_KINEMATIC_PARENTS: tuple[int, ...] = (
    -1,  # pelvis
    0,  # left_hip
    0,  # right_hip
    0,  # spine1
    1,  # left_knee
    2,  # right_knee
    3,  # spine2
    4,  # left_ankle
    5,  # right_ankle
    6,  # spine3
    7,  # left_foot
    8,  # right_foot
    9,  # neck
    9,  # left_collar
    9,  # right_collar
    12,  # head
    13,  # left_shoulder
    14,  # right_shoulder
    16,  # left_elbow
    17,  # right_elbow
    18,  # left_wrist
    19,  # right_wrist
    20,  # left_hand
    21,  # right_hand
)

#: Hand joints that SMPL-H adds after the 22 shared body joints.
SMPL_HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)

#: Number of body joints shared by SMPL and SMPL-H (the AMASS pose prefix).
BODY_JOINT_COUNT = 22

#: Nominal standing height of the procedural rest skeleton, in metres.
NOMINAL_HEIGHT_M = 1.70

#: Nominal pelvis height above the floor once the rest skeleton stands.
NOMINAL_PELVIS_HEIGHT_M = 0.975

_NOMINAL_REST_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.000, 0.000, 0.000),  # pelvis
    (0.000, 0.090, -0.045),  # left_hip
    (0.000, -0.090, -0.045),  # right_hip
    (0.000, 0.000, 0.080),  # spine1
    (0.000, 0.090, -0.465),  # left_knee
    (0.000, -0.090, -0.465),  # right_knee
    (0.000, 0.000, 0.190),  # spine2
    (0.000, 0.090, -0.865),  # left_ankle
    (0.000, -0.090, -0.865),  # right_ankle
    (0.000, 0.000, 0.300),  # spine3
    (0.150, 0.090, -0.905),  # left_foot
    (0.150, -0.090, -0.905),  # right_foot
    (0.000, 0.000, 0.460),  # neck
    (0.000, 0.085, 0.400),  # left_collar
    (0.000, -0.085, 0.400),  # right_collar
    (0.000, 0.000, 0.600),  # head
    (0.000, 0.165, 0.400),  # left_shoulder
    (0.000, -0.165, 0.400),  # right_shoulder
    (0.000, 0.175, 0.140),  # left_elbow
    (0.000, -0.175, 0.140),  # right_elbow
    (0.000, 0.180, -0.110),  # left_wrist
    (0.000, -0.180, -0.110),  # right_wrist
    (0.000, 0.180, -0.190),  # left_hand
    (0.000, -0.180, -0.190),  # right_hand
)

#: Joints whose rest bone is so short that a capsule cannot represent them.
MIN_BONE_LENGTH_M = 0.02


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class SkeletonTopology:
    """A validated joint hierarchy: names plus a topologically sorted parent table."""

    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    hand_joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("skeleton name must be a non-empty string")
        if len(self.joint_names) != len(self.parents):
            raise ValueError(
                f"{self.name}: {len(self.joint_names)} joint names but "
                f"{len(self.parents)} parent entries"
            )
        if not self.joint_names:
            raise ValueError(f"{self.name}: skeleton has no joints")
        duplicates = sorted({n for n in self.joint_names if self.joint_names.count(n) > 1})
        if duplicates:
            raise ValueError(f"{self.name}: duplicate joint names {duplicates}")
        roots = [i for i, parent in enumerate(self.parents) if parent < 0]
        if roots != [0]:
            raise ValueError(
                f"{self.name}: expected joint 0 to be the only root, found roots at {roots}"
            )
        for index, parent in enumerate(self.parents):
            if index == 0:
                continue
            if parent >= index:
                raise ValueError(
                    f"{self.name}: parents must be topologically sorted, but joint "
                    f"{index} ({self.joint_names[index]}) points at {parent}"
                )
            if parent < 0:
                raise ValueError(f"{self.name}: joint {index} has a negative parent")

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def root_name(self) -> str:
        return self.joint_names[0]

    def index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.name}: unknown joint {name!r}") from exc

    def parent_of(self, name: str) -> str | None:
        parent = self.parents[self.index(name)]
        return None if parent < 0 else self.joint_names[parent]

    def children_of(self, name: str) -> tuple[str, ...]:
        index = self.index(name)
        return tuple(
            self.joint_names[i] for i, parent in enumerate(self.parents) if parent == index
        )

    def depth(self, name: str) -> int:
        depth = 0
        current = self.index(name)
        while self.parents[current] >= 0:
            current = self.parents[current]
            depth += 1
        return depth

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        current = self.index(descendant)
        target = self.index(ancestor)
        while current >= 0:
            if current == target:
                return True
            current = self.parents[current]
        return False

    def limb_joints(self, side: str) -> tuple[str, ...]:
        """Joints whose name belongs to ``left`` or ``right``, in index order."""

        token = str(side).strip().lower()
        if token not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return tuple(name for name in self.joint_names if name.startswith(f"{token}_"))


def smpl_skeleton() -> SkeletonTopology:
    """The canonical SMPL body topology."""

    return SkeletonTopology(
        name="smpl",
        joint_names=SMPL_JOINT_NAMES,
        parents=SMPL_KINEMATIC_PARENTS,
        hand_joint_names=SMPL_HAND_JOINT_NAMES,
    )


def skeleton_from_kintree_table(
    kintree_table: Iterable[Sequence[int]], *, name: str = "smpl"
) -> SkeletonTopology:
    """Rebuild the topology from a model file's ``kintree_table``.

    The released SMPL layout stores two rows: ``[0]`` the parent's *joint id* and
    ``[1]`` the joint's own *joint id*. Ids are not assumed to equal column
    indices, so the parent row is mapped through the id column exactly as the
    reference loader does.
    """

    rows = [list(row) for row in kintree_table]
    if len(rows) != 2:
        raise ValueError(f"kintree_table must have 2 rows, got {len(rows)}")
    parent_ids, joint_ids = rows
    if len(parent_ids) != len(joint_ids):
        raise ValueError("kintree_table rows disagree on length")
    if len(joint_ids) != len(SMPL_JOINT_NAMES):
        raise ValueError(
            f"expected {len(SMPL_JOINT_NAMES)} joints for {name}, got {len(joint_ids)}"
        )
    id_to_index = {}
    for index, joint_id in enumerate(joint_ids):
        joint_id = int(joint_id)
        if joint_id in id_to_index:
            raise ValueError(f"kintree_table repeats joint id {joint_id}")
        id_to_index[joint_id] = index
    parents: list[int] = []
    for index, parent_id in enumerate(parent_ids):
        # Row 0 is the root. Its parent entry is not a joint id at all -- the released
        # SMPL file stores -1 there (as an unsigned -1 it reads back as 4294967295),
        # so it must be handled before any id lookup.
        if index == 0:
            parents.append(-1)
            continue
        parent_id = int(parent_id)
        if parent_id not in id_to_index:
            raise ValueError(
                f"kintree_table references unknown parent id {parent_id} for joint "
                f"{joint_ids[index]}"
            )
        parents.append(id_to_index[parent_id])
    topology = SkeletonTopology(
        name=name,
        joint_names=SMPL_JOINT_NAMES,
        parents=tuple(parents),
        hand_joint_names=SMPL_HAND_JOINT_NAMES,
    )
    if tuple(parents) != SMPL_KINEMATIC_PARENTS:
        raise ValueError(
            "model kintree_table does not match the SMPL kinematic tree this repo "
            f"assumes; model gives {tuple(parents)}, repo assumes "
            f"{SMPL_KINEMATIC_PARENTS}. Update SMPL_KINEMATIC_PARENTS before "
            "retargeting, and re-check the AMASS mapping."
        )
    return topology


def nominal_rest_joint_positions() -> dict[str, tuple[float, float, float]]:
    """The procedural standing skeleton, keyed by joint name."""

    return dict(zip(SMPL_JOINT_NAMES, _NOMINAL_REST_POSITIONS, strict=True))


@dataclass(frozen=True, slots=True)
class RestSkeleton:
    """Joint centres of a body in its rest pose, in the body-local frame.

    The frame is right-handed with ``+X`` forward, ``+Y`` left and ``+Z`` up.
    ``joint_positions`` is indexed exactly like ``topology.joint_names``.
    """

    topology: SkeletonTopology
    joint_positions: tuple[tuple[float, float, float], ...]
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("rest skeleton source must be a non-empty string")
        if len(self.joint_positions) != self.topology.joint_count:
            raise ValueError(
                f"expected {self.topology.joint_count} joint positions, "
                f"got {len(self.joint_positions)}"
            )
        for index, position in enumerate(self.joint_positions):
            if len(position) != 3:
                raise ValueError(f"joint {index} must have 3 coordinates")
            for axis, value in enumerate(position):
                _as_float(value, f"{self.topology.joint_names[index]}[{axis}]")
        self._check_bones()

    @property
    def height_m(self) -> float:
        """Span of the rest skeleton along the body up axis, in metres."""

        heights = [position[2] for position in self.joint_positions]
        return max(heights) - min(heights)

    def position(self, name: str) -> tuple[float, float, float]:
        return self.joint_positions[self.topology.index(name)]

    def bone_vector(self, name: str) -> tuple[float, float, float]:
        """Rest offset from a joint to its parent, expressed in the body frame.

        Because every link frame is rest-aligned and identity-oriented, this is
        also the value the USD joint's ``localPos0`` needs.
        """

        parent = self.topology.parent_of(name)
        if parent is None:
            return (0.0, 0.0, 0.0)
        child = self.position(name)
        head = self.position(parent)
        return tuple(child[axis] - head[axis] for axis in range(3))

    def bone_length(self, name: str) -> float:
        return math.sqrt(sum(component**2 for component in self.bone_vector(name)))

    def _check_bones(self) -> None:
        """Reject a rest pose that cannot carry a capsule between parent and child."""

        for index, name in enumerate(self.topology.joint_names):
            parent = self.topology.parents[index]
            if parent < 0:
                continue
            length = self.bone_length(name)
            if length < MIN_BONE_LENGTH_M:
                raise ValueError(
                    f"{name}: rest bone to parent {self.topology.joint_names[parent]} is "
                    f"{length:.4f} m, below the {MIN_BONE_LENGTH_M} m minimum; a capsule "
                    "collider cannot represent a zero-length segment"
                )

    def scaled_by(self, factor: float, *, name: str | None = None) -> RestSkeleton:
        """Uniformly rescale every joint position by ``factor``.

        Uniform scaling is **not** a body-shape model: it changes limb lengths but
        not girth or mass distribution. It exists so the tall/short axis of the
        training domain can be exercised before SMPL shape coefficients are
        available, and every artefact records that it was used.
        """

        value = _as_float(factor, "factor")
        if value <= 0:
            raise ValueError(f"scale factor must be positive, got {factor!r}")
        return RestSkeleton(
            topology=self.topology,
            joint_positions=tuple(
                tuple(component * value for component in position)
                for position in self.joint_positions
            ),  # type: ignore[arg-type]
            source=f"{self.source} scaled x{value:.6f} ({name or 'uniform'})",
        )

    def scaled_to_height(self, height_m: float, *, name: str | None = None) -> RestSkeleton:
        """Rescale so the joint span along the body up axis matches ``height_m``.

        Note that the joint span is *not* the standing height: the floor contact and
        the top of the head lie outside the outermost joints. Use
        ``sim2sense_fall.humans.rig.plan_human_rig`` with a target height when the
        intent is a figure of a given stature.
        """

        target = _as_float(height_m, "height_m")
        if target <= 0:
            raise ValueError(f"height_m must be positive, got {height_m!r}")
        current = self.height_m
        if current <= 0:
            raise ValueError("rest skeleton has zero height and cannot be scaled")
        return self.scaled_by(target / current, name=name)

    def with_joint_offset(self, joint: str, offset_m: Sequence[float]) -> RestSkeleton:
        """Return a copy with one joint displaced, for regression tests."""

        index = self.topology.index(joint)
        if len(offset_m) != 3:
            raise ValueError("offset_m must have three components")
        if index == 0:
            raise ValueError("cannot displace the root joint; move the whole body instead")
        delta = tuple(_as_float(value, f"offset[{axis}]") for axis, value in enumerate(offset_m))
        positions = list(self.joint_positions)
        positions[index] = tuple(positions[index][axis] + delta[axis] for axis in range(3))
        return RestSkeleton(
            topology=self.topology,
            joint_positions=tuple(positions),  # type: ignore[arg-type]
            source=f"{self.source} + offset on {joint}",
        )


def rest_skeleton_from_positions(
    positions: Mapping[str, Sequence[float]], *, source: str
) -> RestSkeleton:
    """Build a :class:`RestSkeleton` from a joint-name keyed mapping.

    Raises if a joint is missing or extra, because a partially specified
    skeleton silently changes the articulation's bone lengths.
    """

    topology = smpl_skeleton()
    missing = [name for name in topology.joint_names if name not in positions]
    extra = [name for name in positions if name not in topology.joint_names]
    if missing or extra:
        raise ValueError(f"skeleton positions mismatch: missing {missing}, unexpected {extra}")
    ordered = tuple(
        tuple(_as_float(value, f"{name}[{axis}]") for axis, value in enumerate(positions[name]))
        for name in topology.joint_names
    )
    for index, entry in enumerate(ordered):
        if len(entry) != 3:
            raise ValueError(f"{topology.joint_names[index]} must have 3 coordinates")
    return RestSkeleton(topology=topology, joint_positions=ordered, source=source)


def default_rest_skeleton(*, height_m: float = NOMINAL_HEIGHT_M) -> RestSkeleton:
    """The procedural rest skeleton, optionally rescaled to a target height."""

    topology = smpl_skeleton()
    positions = tuple(_NOMINAL_REST_POSITIONS)
    skeleton = RestSkeleton(
        topology=topology,
        joint_positions=positions,
        source="procedural nominal rest skeleton (not SMPL model data)",
    )
    if abs(height_m - skeleton.height_m) <= 1e-9:
        return skeleton
    return skeleton.scaled_to_height(height_m)


def body_axis_vector(axis: str) -> tuple[float, float, float]:
    """Unit vector for a canonical body axis token."""

    vector = AXES[validate_axis(axis)]
    return (float(vector[0]), float(vector[1]), float(vector[2]))
