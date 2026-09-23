"""Regression tests for the imported body frame.

These exist because an earlier loader derived only the *up* axis from the pelvis-to-head
vector and applied a basis that assumed both frames agreed on their horizontal
convention. The released SMPL template does not: its lateral axis (the shoulder span)
sits where this pipeline expects forward. The result was that "up is up" held, every
extent check passed, the forward kinematics passed against itself, and every exported
mesh still lay on its side -- 1.83 m across and 1.80 m tall.

So these tests deliberately assert **anatomical** properties, not frame arithmetic:
which way the head rises, where the left hand is relative to the left shoulder, and
which way the toes lead. Those are the questions a researcher actually cares about, and
none of them can be satisfied by an internally consistent but wrongly oriented body.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim2sense_fall.humans.assets import load_smpl_model
from sim2sense_fall.humans.motion import body_frame_conversion, up_axis_conversion
from sim2sense_fall.humans.skeleton import SMPL_JOINT_NAMES, default_rest_skeleton

from .test_humans import body_payload

#: The exact change of basis the old loader applied: derive the up axis, then rotate with
#: ``up_axis_conversion``. Reproduced here so the shipped bug is a live control rather
#: than a comment -- if a future change reintroduces this basis, the tests below fail.
_SHIPPED_BUG_FRAME = up_axis_conversion("y", "z")


def _joint_index(name: str) -> int:
    return SMPL_JOINT_NAMES.index(name)


def _fixture_model(tmp_path):
    import pickle

    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(body_payload()))
    return load_smpl_model(path)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


# ---------------------------------------------------------------------------
# the basis itself
# ---------------------------------------------------------------------------


def test_body_frame_conversion_maps_an_anatomical_triple_onto_the_pipeline_frame():
    """A source frame given anatomically must land on forward/left/up."""

    # Source: +Y up, +Z forward, +X left -- the released SMPL template's own frame.
    basis = body_frame_conversion(up="y", forward="z", left="x", source="smpl")

    # The pipeline's forward is +X, and it must read the source's forward axis.
    assert np.allclose(basis @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])
    # Left is +Y and must read the source's lateral axis.
    assert np.allclose(basis @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    # Up is +Z and must read the source's up axis.
    assert np.allclose(basis @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert float(np.linalg.det(basis)) == pytest.approx(1.0)


def test_body_frame_conversion_rejects_a_left_handed_triple():
    """A mirrored body would swap left and right limbs, so it is refused."""

    with pytest.raises(ValueError, match="determinant"):
        body_frame_conversion(up="y", forward="z", left="-x", source="mirrored")


def test_body_frame_conversion_rejects_repeated_axes():
    with pytest.raises(ValueError, match="three distinct axes"):
        body_frame_conversion(up="y", forward="y", left="x", source="degenerate")


def test_up_axis_conversion_alone_cannot_express_the_required_change_of_basis():
    """Documents why the old helper was the wrong tool.

    ``up_axis_conversion("y","z")`` leaves the source's X on the pipeline's X. If the
    source's X is lateral and the pipeline's X is forward, that is a 90 degree yaw --
    and the helper has no way to say so.
    """

    up_only = up_axis_conversion("y", "z")
    correct = body_frame_conversion(up="y", forward="z", left="x", source="smpl")

    assert not np.allclose(up_only, correct)
    # The up-only basis keeps the source lateral axis on the pipeline forward axis.
    assert np.allclose(up_only @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    # The anatomical basis moves it to the lateral axis, where it belongs.
    assert np.allclose(correct @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# the loader, on a coherent fixture
# ---------------------------------------------------------------------------


def test_a_loaded_body_stands_up_in_the_pipeline_frame(tmp_path):
    """The head must rise along +Z and the feet must hang below the pelvis.

    This is the check that would have caught the shipped defect: the delivered body was
    human-sized but lying down, so any stature-only assertion passed.
    """

    model = _fixture_model(tmp_path)
    joints = np.asarray(model.joint_positions, dtype=np.float64)

    pelvis = joints[_joint_index("pelvis")]
    head = joints[_joint_index("head")]
    left_foot = joints[_joint_index("left_foot")]

    assert head[2] > pelvis[2] + 0.3, "the head must rise above the pelvis along +Z"
    assert left_foot[2] < pelvis[2] - 0.3, "the feet must hang below the pelvis along +Z"


def test_the_shoulder_line_is_lateral_not_forward(tmp_path):
    """The shoulder span must lie on Y (left), never on X (forward).

    The shipped defect put the shoulder span on the forward axis while leaving "up"
    correct, which is precisely what this asserts against.
    """

    model = _fixture_model(tmp_path)
    joints = np.asarray(model.joint_positions, dtype=np.float64)

    span = _unit(joints[_joint_index("right_shoulder")] - joints[_joint_index("left_shoulder")])
    assert abs(span[1]) > 0.9, f"shoulder span should be lateral, got {np.round(span, 3).tolist()}"
    assert abs(span[0]) < 0.1, "shoulder span must not run along the forward axis"


def test_left_is_plus_y(tmp_path):
    """`left_*` joints must sit at larger Y than their `right_*` counterparts."""

    model = _fixture_model(tmp_path)
    joints = np.asarray(model.joint_positions, dtype=np.float64)

    for left, right in (("left_hip", "right_hip"), ("left_shoulder", "right_shoulder")):
        assert joints[_joint_index(left)][1] > joints[_joint_index(right)][1], left


def test_the_loaded_body_agrees_with_the_procedural_reference_frame(tmp_path):
    """The imported body and the built-in skeleton must share a frame.

    The fixture is a coarse 10-anchor stand-in, not a real body, so joint positions
    differ from the procedural skeleton's -- what must agree is the *direction* of the
    body's defining axes. The pelvis-to-head rise and the shoulder line are the two
    directions the procedural skeleton pins unambiguously, so those are what is checked.
    """

    model = _fixture_model(tmp_path)
    imported = np.asarray(model.joint_positions, dtype=np.float64)
    reference = np.asarray(default_rest_skeleton().joint_positions, dtype=np.float64)

    # Up: pelvis -> head must be +Z in both frames.
    for label, joints in (("imported", imported), ("procedural", reference)):
        rise = _unit(joints[_joint_index("head")] - joints[_joint_index("pelvis")])
        assert rise[2] > 0.9, f"{label} pelvis->head is not up: {np.round(rise, 3).tolist()}"

    # Lateral: the shoulder line must be ±Y in both frames.
    for label, joints in (("imported", imported), ("procedural", reference)):
        span = _unit(joints[_joint_index("right_shoulder")] - joints[_joint_index("left_shoulder")])
        assert abs(span[1]) > 0.9, f"{label} shoulder span not lateral: {np.round(span, 3)}"
        assert abs(span[0]) < 0.2, f"{label} shoulder span leaks onto the forward axis"


def test_the_loader_records_the_source_frame_it_measured(tmp_path):
    """The manifest needs the convention, not just a stature."""

    model = _fixture_model(tmp_path)
    assert model.source_frame
    assert "up=" in model.source_frame
    assert "forward=" in model.source_frame
    assert "left=" in model.source_frame
    assert model.as_dict()["source_frame"] == model.source_frame


def test_the_shipped_bug_basis_is_reproducibly_wrong(tmp_path):
    """Positive control: the old basis really did misplace the shoulder line.

    Reproducing the defect as a live assertion is what makes the fix mean something. If
    someone reinstates ``up_axis_conversion`` as the whole-body basis, this test still
    passes (it describes the bug), while the anatomical tests above start failing.
    """

    model = _fixture_model(tmp_path)
    joints = np.asarray(model.joint_positions, dtype=np.float64)
    broken = (_SHIPPED_BUG_FRAME @ joints.T).T

    span_loaded = joints[_joint_index("right_shoulder")] - joints[_joint_index("left_shoulder")]
    span_broken = broken[_joint_index("right_shoulder")] - broken[_joint_index("left_shoulder")]

    # What the loader produces: the shoulder line is lateral.
    assert abs(_unit(span_loaded)[1]) > 0.9
    # What the old basis produced: the shoulder line lands on the vertical axis.
    assert abs(_unit(span_broken)[2]) > 0.9
    assert not np.allclose(span_loaded, span_broken)


# ---------------------------------------------------------------------------
# the skin-to-rig correspondence and the exported pose
# ---------------------------------------------------------------------------


def _proportional_joints() -> np.ndarray:
    """A 24-joint body whose joints are not coincident.

    The lightweight ``body_payload`` fixture exists to exercise the *loader*, and it
    collapses most of the 24 SMPL joints onto a handful of anchors so that it stays
    small. That is fine for checking a change of basis, but the joint-to-joint
    correspondence below is a geometric question and coincident joints make it
    ill-posed. This skeleton mirrors a real figure's proportions closely enough that
    every joint is uniquely nearest to its own counterpart, which is the property under
    test.
    """

    from sim2sense_fall.humans.skeleton import SMPL_JOINT_NAMES

    # (x forward, y left, z up) in metres, a 1.75 m figure. Sided joints are mirrored
    # from one definition so left and right stay exactly symmetric.
    anchors = {
        "pelvis": (0.00, 0.00, 0.95),
        "spine1": (0.00, 0.00, 1.06),
        "spine2": (0.00, 0.00, 1.17),
        "spine3": (0.00, 0.00, 1.28),
        "neck": (0.00, 0.00, 1.42),
        "head": (0.00, 0.00, 1.57),
    }
    sided = {
        "hip": (0.00, 0.09, 0.93),
        "knee": (0.00, 0.09, 0.50),
        "ankle": (0.00, 0.09, 0.08),
        "foot": (0.09, 0.09, 0.03),
        "collar": (0.00, 0.09, 1.36),
        "shoulder": (0.00, 0.18, 1.36),
        "elbow": (0.00, 0.44, 1.34),
        "wrist": (0.00, 0.68, 1.32),
        "hand": (0.00, 0.78, 1.31),
    }
    # A T-pose: arms out along +-Y, exactly as the licensed template's rest pose.
    for joint in SMPL_JOINT_NAMES:
        if joint == "pelvis" or joint in anchors:
            continue
        sign = 1.0 if joint.startswith("left_") else -1.0
        base = sided[joint.split("_", 1)[1]]
        anchors[joint] = (base[0], base[1] * sign, base[2])
    return np.array([anchors[name] for name in SMPL_JOINT_NAMES], dtype=np.float64)


def _proportional_mesh(*, scale: float = 1.0):
    """A SmplMesh whose rest joints are the proportional skeleton above."""

    from sim2sense_fall.humans.skeleton import SMPL_JOINT_NAMES, smpl_skeleton
    from sim2sense_fall.humans.skinning import SmplMesh

    count = len(SMPL_JOINT_NAMES)
    joints = _proportional_joints() * scale
    vertices = joints  # one vertex per joint keeps the mesh non-degenerate
    faces = np.array([[i, i + 1, i + 2] for i in range(count - 2)], dtype=np.int64)
    weights = np.zeros((count, count), dtype=np.float64)
    np.fill_diagonal(weights, 1.0)
    return SmplMesh(
        vertices=vertices,
        faces=faces,
        weights=weights,
        topology=smpl_skeleton(),
        rest_joints=joints,
        source="proportional-fixture",
    )


def test_joint_row_alignment_accepts_a_pure_scale():
    """A uniformly rescaled rig is exactly the case the fit exists for."""

    from sim2sense_fall.humans.mesh_sequence import joint_row_alignment

    mesh = _proportional_mesh()
    target = _proportional_joints() * 1.08

    assert mesh.rest_joints is not None
    indices = joint_row_alignment(mesh, target)
    assert np.array_equal(indices, np.arange(mesh.rest_joints.shape[0]))


def test_joint_row_alignment_rejects_a_reordered_rig():
    """If the rig's rows are permuted, the name-matched confirmation must fire.

    The rig rescales joints to the configured standing height while the licensed file
    keeps its native stature, so a correct pairing still leaves the arrays tens of
    millimetres apart -- measured at 53 mm for the ankle and 91 mm for the foot of the
    shipped neutral file. That is wide enough for a positional search to pick the wrong
    row, so the pairing is taken from the shared joint names and merely *confirmed*
    against geometry. This test is what makes the confirmation load-bearing.
    """

    from sim2sense_fall.humans.mesh_sequence import joint_row_alignment

    mesh = _proportional_mesh()
    target = _proportional_joints()
    head, pelvis = _joint_index("head"), _joint_index("pelvis")
    target[[head, pelvis]] = target[[pelvis, head]]

    with pytest.raises(ValueError, match="disagree about which joint is which"):
        joint_row_alignment(mesh, target)


def test_fit_mesh_to_rest_joints_rejects_a_frame_mismatch():
    """A similarity is "uniform scale plus translation"; anything else is an error.

    A rotation disguised as a scale would move the exported skin off the links it is
    skinned to, so the residual is checked rather than assumed.
    """

    from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints

    mesh = _proportional_mesh()
    # A 90 degree yaw of the rig's joints against the mesh: same magnitudes, wrong frame.
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = (yaw @ _proportional_joints().T).T

    with pytest.raises(ValueError, match="not a uniform scale"):
        fit_mesh_to_rest_joints(mesh, target)


def test_fit_mesh_to_rest_joints_preserves_a_pure_scale_fit():
    """The happy path must still work: a scaled rig gives back the same scalar."""

    from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints

    mesh = _proportional_mesh()
    target = _proportional_joints() * 1.08

    fitted = fit_mesh_to_rest_joints(mesh, target)
    assert np.allclose(fitted.vertices, mesh.vertices * 1.08, atol=1e-9)
    assert np.allclose(fitted.rest_joints, target)
