"""CPU tests for the human rig, motion contract, asset audit and event labeller.

Everything here runs without Isaac Sim, Sionna or any licensed asset. The tests that
need the USD runtime live in ``test_human_usd.py`` and skip when ``pxr`` is absent.

The tests are grouped by the stage sub-task they defend:

* 7.1 -- skeleton topology, rest skeleton, asset registry, motion contract, retargeting
* 7.2 -- rig planning, capsule geometry, mass allocation, forward kinematics
* 7.3 -- joint-sign semantics, reference motion representability
* 7.4 -- event labelling boundaries, especially fall versus controlled lowering
* 7.5 -- ground-truth export, two clocks, provenance, split keys
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim2sense_fall.humans.assets import (
    ASSET_ROOT_ENV,
    AssetUnavailable,
    audit_registry,
    candidate_paths,
    default_asset_roots,
    load_asset_registry,
    load_model_payload,
    load_smpl_model,
    registry_from_sequences,
    require_model,
    select_body,
    sha256_file,
)
from sim2sense_fall.humans.config import (
    EventsConfig,
    human_config_from_mapping,
    load_human_config,
)
from sim2sense_fall.humans.events import (
    LABEL_CONTROLLED_LOWERING,
    LABEL_FALL,
    LABEL_INVALID,
    LABEL_NO_FALL,
    Trajectory,
    label_trial,
    phase_labels_from_signal,
    trunk_axis_from_link_positions,
)
from sim2sense_fall.humans.export import (
    GroundTruth,
    TrialProvenance,
    clip_group_keys,
    connected_groups,
    resample_series,
    sha256_text,
    split_violations,
)
from sim2sense_fall.humans.mesh_sequence import (
    MeshSequence,
    MeshTopology,
    build_capsule_proxy_template,
    fit_mesh_to_rest_joints,
    pose_capsule_proxy_mesh,
)
from sim2sense_fall.humans.motion import (
    AMASS_BODY_FRAME,
    FRAME_SOURCE_AMASS,
    MAX_FRAME_DELTA_RAD,
    MotionProvenance,
    compile_scripted_clip,
    load_motion_library,
    motion_library_from_mapping,
    retarget_amass_clip,
    up_axis_conversion,
)
from sim2sense_fall.humans.rig import (
    apply_neutral_pose,
    authors_deviations,
    forward_kinematics,
    plan_human_rig,
    pose_surface_points,
    quaternion_from_z,
    standing_height_for,
)
from sim2sense_fall.humans.rotations import (
    axis_angle_to_matrix,
    close_rotation,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    rotation_matrix_error,
)
from sim2sense_fall.humans.skeleton import (
    BODY_JOINT_COUNT,
    SMPL_HAND_JOINT_NAMES,
    SMPL_JOINT_NAMES,
    SMPL_KINEMATIC_PARENTS,
    default_rest_skeleton,
    rest_skeleton_from_positions,
    skeleton_from_kintree_table,
    smpl_skeleton,
)
from sim2sense_fall.humans.skinning import (
    SmplMesh,
    joint_centres_from_regressor,
    mesh_from_model_payload,
    skin_with_link_poses,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "humans" / "human_smpl_neutral.yaml"
ASSETS_PATH = REPO_ROOT / "configs" / "humans" / "assets.yaml"
MOTIONS_PATH = REPO_ROOT / "configs" / "humans" / "motions.yaml"


@pytest.fixture(scope="module")
def config():
    return load_human_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def plan(config):
    return plan_human_rig(config)


@pytest.fixture(scope="module")
def motions(config):
    return load_motion_library(MOTIONS_PATH, topology=config.topology)


# ---------------------------------------------------------------------------
# 7.1 skeleton topology and the rest skeleton
# ---------------------------------------------------------------------------


def test_smpl_topology_matches_the_published_tree():
    topology = smpl_skeleton()
    assert topology.joint_count == 24
    assert topology.root_name == "pelvis"
    assert topology.parents[0] == -1
    assert all(index == 0 or 0 <= parent < index for index, parent in enumerate(topology.parents))
    assert tuple(topology.parents) == SMPL_KINEMATIC_PARENTS
    assert len(SMPL_HAND_JOINT_NAMES) == 30
    assert topology.joint_names[:BODY_JOINT_COUNT] == SMPL_JOINT_NAMES[:BODY_JOINT_COUNT]


def test_topology_reports_relationships():
    topology = smpl_skeleton()
    assert topology.parent_of("pelvis") is None
    assert topology.parent_of("left_knee") == "left_hip"
    assert set(topology.children_of("pelvis")) == {"left_hip", "right_hip", "spine1"}
    assert topology.depth("head") > topology.depth("neck")
    assert topology.is_ancestor("pelvis", "left_foot")
    assert not topology.is_ancestor("left_hip", "right_knee")
    assert topology.limb_joints("left") == (
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_foot",
        "left_collar",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hand",
    )


def test_topology_rejects_a_malformed_tree():
    from sim2sense_fall.humans.skeleton import SkeletonTopology

    with pytest.raises(ValueError, match="topologically sorted"):
        SkeletonTopology(name="bad", joint_names=("a", "b"), parents=(-1, 5))
    with pytest.raises(ValueError, match="only root"):
        SkeletonTopology(name="bad", joint_names=("a", "b"), parents=(0, -1))
    with pytest.raises(ValueError, match="duplicate joint names"):
        SkeletonTopology(name="bad", joint_names=("a", "a"), parents=(-1, 0))


def test_kintree_table_round_trips_and_rejects_a_different_tree():
    table = [[int(parent) for parent in SMPL_KINEMATIC_PARENTS], list(range(24))]
    assert skeleton_from_kintree_table(table).parents == SMPL_KINEMATIC_PARENTS
    # A still-valid tree that differs from SMPL: the left knee hanging off the right
    # hip. Topologically sorted, but not the SMPL skeleton.
    reordered = list(SMPL_KINEMATIC_PARENTS)
    reordered[4] = 2
    with pytest.raises(ValueError, match="does not match the SMPL kinematic tree"):
        skeleton_from_kintree_table([reordered, list(range(24))])


def test_rest_skeleton_is_finite_and_positively_boned():
    rest = default_rest_skeleton()
    assert rest.topology.joint_count == 24
    assert all(
        math.isfinite(component) for position in rest.joint_positions for component in position
    )
    for index, name in enumerate(rest.topology.joint_names):
        if rest.topology.parents[index] < 0:
            continue
        assert rest.bone_length(name) > 0.02, name
    assert rest.bone_vector("pelvis") == (0.0, 0.0, 0.0)
    knee_bone = rest.bone_vector("left_knee")
    assert knee_bone[0] == 0.0 and knee_bone[1] == 0.0
    assert knee_bone[2] < -0.4


def test_rest_skeleton_scaling_and_offset_are_pure():
    rest = default_rest_skeleton()
    doubled = rest.scaled_by(2.0)
    assert doubled.height_m == pytest.approx(2 * rest.height_m)
    assert rest.height_m < doubled.height_m
    moved = rest.with_joint_offset("left_knee", (0.01, 0.0, 0.0))
    assert moved.position("left_knee")[0] == pytest.approx(rest.position("left_knee")[0] + 0.01)
    assert moved.position("left_ankle") == rest.position("left_ankle")
    with pytest.raises(ValueError, match="cannot displace the root"):
        rest.with_joint_offset("pelvis", (0.1, 0.0, 0.0))
    # Move the knee exactly onto the hip: a zero-length bone, which no capsule can
    # represent. The offset is derived so the test does not depend on the scale.
    coincide = rest.position("left_hip")[2] - rest.position("left_knee")[2]
    with pytest.raises(ValueError, match="zero-length segment"):
        rest.with_joint_offset("left_knee", (0.0, 0.0, coincide))


def test_rest_skeleton_from_positions_rejects_incomplete_input():
    positions = {name: (0.0, 0.0, index * 0.1) for index, name in enumerate(SMPL_JOINT_NAMES)}
    skeleton = rest_skeleton_from_positions(positions, source="test")
    assert skeleton.topology.joint_count == len(SMPL_JOINT_NAMES)
    assert skeleton.source == "test"
    broken = dict(positions)
    del broken["left_knee"]
    with pytest.raises(ValueError, match="missing"):
        rest_skeleton_from_positions(broken, source="test")


# ---------------------------------------------------------------------------
# 7.1 asset registry: the audit must never download, and must fail actionably
# ---------------------------------------------------------------------------


def test_asset_registry_declares_the_licence_and_registration_for_every_model():
    registry = load_asset_registry(ASSETS_PATH, project_root=REPO_ROOT)
    assert set(registry.models) == {
        "smpl_neutral_v1_1_0",
        "smpl_male_v1_1_0",
        "smpl_female_v1_1_0",
        "smpl_male_v1_0_0",
        "smpl_female_v1_0_0",
    }
    assert "amass" in registry.motions
    for model in registry.models.values():
        assert model.requires_registration
        assert model.registration_url.startswith("https://")
        assert "non-commercial" in model.license.lower() or "research" in model.license.lower()
        # `verified` means "this repository opened the file and measured it". Only the
        # neutral v1.1.0 pickle has been, so the others must still say false.
        assert model.verified == (model.id == "smpl_neutral_v1_1_0"), (
            f"{model.id} verification flag does not match what has actually been measured "
            "on this machine"
        )
    amass = registry.motions["amass"]
    assert amass.pose_parameters == 156 == amass.body_joints * 3 + amass.hand_joints * 3
    assert amass.hand_joints == 30
    assert amass.betas == 16
    assert amass.dmpls == 8
    assert not amass.verified
    assert "fall" in amass.notes.lower() or "NOT" in amass.notes


def test_asset_audit_reports_absence_without_creating_anything(tmp_path, monkeypatch):
    monkeypatch.setenv(ASSET_ROOT_ENV, str(tmp_path / "empty"))
    registry = registry_from_sequences(
        [
            {
                "id": "demo",
                "family": "smpl",
                "version": "1.0.0",
                "filename": "basicmodel.pkl",
                "requires_registration": True,
                "registration_url": "https://example.invalid/register",
                "license": "research only",
                "license_url": "https://example.invalid/licence",
                "joint_count": 24,
                "betas": 10,
                "pose_parameters": 72,
                "pickle_layout": "v_template, f, kintree_table, J_regressor",
                "provenance": "test fixture",
                "verified": False,
            }
        ],
        [
            {
                "id": "amass",
                "dataset": "AMASS",
                "representation": "smplh_52",
                "pose_parameters": 156,
                "body_joints": 22,
                "betas": 16,
                "dmpls": 8,
                "subsets": ["CMU"],
                "requires_registration": True,
                "registration_url": "https://example.invalid/amass",
                "license": "research only",
                "license_url": "https://example.invalid/licence",
                "provenance": "test fixture",
                "verified": False,
            }
        ],
        roots=[tmp_path / "empty"],
    )
    audit = audit_registry(registry)
    assert audit["available_model_count"] == 0
    assert not (tmp_path / "empty").exists() or not any((tmp_path / "empty").iterdir())
    with pytest.raises(AssetUnavailable) as excinfo:
        require_model(registry, "demo")
    message = str(excinfo.value)
    assert "https://example.invalid/register" in message
    assert "does not download it" in message
    assert str(tmp_path / "empty" / "basicmodel.pkl") in message


def test_asset_root_env_var_overrides_the_search_path(tmp_path, monkeypatch):
    monkeypatch.setenv(ASSET_ROOT_ENV, str(tmp_path / "a") + ":" + str(tmp_path / "b"))
    roots = default_asset_roots(tmp_path)
    assert roots[0] == tmp_path / "a"
    assert roots[1] == tmp_path / "b"
    assert roots[-1] == Path.home() / ".cache" / "sim2sense-fall" / "humans"


def body_payload(betas: int = 4) -> dict:
    """A small but anatomically coherent stand-in for a released SMPL pickle.

    The loader no longer accepts an arbitrary dict of the right keys. It measures the
    whole body frame -- not just the up axis -- from the joints, cross-checking the
    torso against the thigh, the hip line against the shoulder line, and the forward
    sign against the toes. An all-zeros fixture therefore has to become a body, or the
    very checks that reject a bogus model file would also reject a legitimate one.

    The fixture is authored in the **released SMPL template's own frame** -- ``+Y`` up,
    ``+X`` left, ``+Z`` forward -- deliberately. That is the convention the real files
    use, so these tests exercise the same import path as the licensed model instead of
    a frame the loader happens to find convenient. ``vertex_count`` is 8 so the joints
    have distinct anchors.
    """

    # Named anchors so the regressor below reads as anatomy rather than as indices.
    # The frame is the released SMPL template's own: +Y up, +X left, +Z forward.
    _ANCHORS = {
        "pelvis": 0,
        "head": 1,
        "left_hip": 2,
        "right_hip": 3,
        "left_knee": 4,
        "right_knee": 5,
        "left_shoulder": 6,
        "right_shoulder": 7,
        "left_foot": 8,
        "right_foot": 9,
    }
    count = len(_ANCHORS)
    vertices = np.zeros((count, 3))
    vertices[_ANCHORS["head"], 1] = 0.60  # crown, above the pelvis along +Y
    vertices[_ANCHORS["left_hip"], 0] = 0.09  # +X is left
    vertices[_ANCHORS["right_hip"], 0] = -0.09
    vertices[_ANCHORS["left_knee"], 0] = 0.09
    vertices[_ANCHORS["left_knee"], 1] = -0.45  # knee sits below the hip
    vertices[_ANCHORS["right_knee"], 0] = -0.09
    vertices[_ANCHORS["right_knee"], 1] = -0.45
    vertices[_ANCHORS["left_shoulder"], 0] = 0.18
    vertices[_ANCHORS["left_shoulder"], 1] = 0.45
    vertices[_ANCHORS["right_shoulder"], 0] = -0.18
    vertices[_ANCHORS["right_shoulder"], 1] = 0.45
    vertices[_ANCHORS["left_foot"], 1] = -0.94
    vertices[_ANCHORS["left_foot"], 2] = 0.06  # toes lead forward along +Z
    vertices[_ANCHORS["right_foot"], 1] = -0.94
    vertices[_ANCHORS["right_foot"], 2] = 0.06

    # Joint name -> anchor, for every joint that anchors the body frame plus the
    # endpoints named in the fixture docstring. Unlisted joints fall back to the pelvis.
    _JOINT_ANCHOR = {
        "head": "head",
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_foot": "left_foot",
        "right_foot": "right_foot",
    }
    regressor = np.zeros((24, count))
    regressor[:, _ANCHORS["pelvis"]] = 1.0  # every joint defaults to the pelvis
    for joint, anchor in _JOINT_ANCHOR.items():
        row = SMPL_JOINT_NAMES.index(joint)
        regressor[row, :] = 0.0
        regressor[row, _ANCHORS[anchor]] = 1.0

    return {
        "v_template": vertices,
        "f": np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=int),
        "kintree_table": np.array([list(SMPL_KINEMATIC_PARENTS), list(range(24))], dtype=int),
        "J_regressor": regressor,
        "weights": np.full((count, 24), 1.0 / 24.0),
        "shapedirs": np.zeros((count, 3, betas)),
        "posedirs": np.zeros((count, 3, 207)),
    }


def test_model_loader_rejects_a_file_outside_the_asset_roots(tmp_path):
    import pickle

    payload = body_payload()
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(payload))
    registry = registry_from_sequences(
        [
            {
                "id": "demo",
                "family": "smpl",
                "version": "1.0.0",
                "filename": "model.pkl",
                "requires_registration": True,
                "registration_url": "https://example.invalid",
                "license": "research",
                "license_url": "https://example.invalid",
                "joint_count": 24,
                "betas": 10,
                "pose_parameters": 72,
                "pickle_layout": "v_template, f, kintree_table, J_regressor",
                "provenance": "test fixture",
                "verified": False,
            }
        ],
        [
            {
                "id": "amass",
                "dataset": "AMASS",
                "representation": "smplh_52",
                "pose_parameters": 156,
                "body_joints": 22,
                "betas": 16,
                "dmpls": 8,
                "subsets": ["CMU"],
                "requires_registration": True,
                "registration_url": "https://example.invalid",
                "license": "research",
                "license_url": "https://example.invalid",
                "provenance": "test fixture",
                "verified": False,
            }
        ],
        roots=[tmp_path / "elsewhere"],
    )
    with pytest.raises(AssetUnavailable, match="outside every configured asset root"):
        load_smpl_model(path, registry=registry)
    # Inside the root it loads, and derives the joints from J_regressor.
    inside = tmp_path / "elsewhere"
    inside.mkdir()
    (inside / "model.pkl").write_bytes(path.read_bytes())
    model = load_smpl_model(inside / "model.pkl", registry=registry)
    assert model.topology.joint_count == 24
    assert model.vertex_count == 10
    assert model.face_count == 3
    assert model.joint_names == SMPL_JOINT_NAMES
    assert len(model.source_sha256) == 64


@pytest.fixture
def fake_chumpy(monkeypatch):
    """A picklable stand-in for ``chumpy.ch``, which is what the loader keys on."""

    import sys
    import types

    class ChumpyArray:
        """Mimics a chumpy array: the payload lives in ``.x``, the module name is chumpy."""

        def __init__(self, x: object = None, shape: tuple[int, ...] | None = None) -> None:
            if x is not None:
                self.x = x
            if shape is not None:
                self.shape = shape

    ChumpyArray.__module__ = "chumpy.ch"
    ChumpyArray.__qualname__ = "ChumpyArray"
    package = types.ModuleType("chumpy")
    module = types.ModuleType("chumpy.ch")
    module.ChumpyArray = ChumpyArray
    package.ch = module
    monkeypatch.setitem(sys.modules, "chumpy", package)
    monkeypatch.setitem(sys.modules, "chumpy.ch", module)
    yield ChumpyArray


def test_sha256_helper_is_stable(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"sim2sense")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


# ---------------------------------------------------------------------------
# 7.1 motion contract
# ---------------------------------------------------------------------------


def test_rotation_helpers_round_trip():
    rng = np.random.default_rng(20260922)
    for _ in range(50):
        vector = rng.normal(scale=0.7, size=3)
        matrix = axis_angle_to_matrix(vector)
        assert rotation_matrix_error(matrix) < 1e-12
        assert close_rotation(matrix)
        recovered = matrix_to_axis_angle(matrix)
        assert np.allclose(axis_angle_to_matrix(recovered), matrix, atol=1e-9)
    for angle in (0.0, 0.3, 1.5, math.pi - 1e-9):
        matrix = axis_angle_to_matrix(np.array([0.0, angle, 0.0]))
        assert np.allclose(axis_angle_to_matrix(matrix_to_axis_angle(matrix)), matrix, atol=1e-7)
    with pytest.raises(ValueError, match="not a rotation"):
        matrix_to_axis_angle(np.eye(3) * 2.0)
    assert np.allclose(quaternion_to_matrix((1.0, 0.0, 0.0, 0.0)), np.eye(3))


def test_up_axis_conversion_maps_rotations_and_points():
    basis = up_axis_conversion("y", "z")
    assert abs(float(np.linalg.det(basis)) - 1.0) < 1e-12
    assert np.allclose(up_axis_conversion("z", "z"), np.eye(3))
    rng = np.random.default_rng(7)
    for _ in range(20):
        vector = rng.normal(scale=0.5, size=3)
        point = rng.normal(size=3)
        matrix = axis_angle_to_matrix(vector)
        assert np.allclose(
            axis_angle_to_matrix(basis @ vector) @ (basis @ point),
            basis @ (matrix @ point),
            atol=1e-9,
        )
    # A Y-up rotation about +Y becomes a Z-up rotation about +Z.
    assert np.allclose(basis @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="unsupported up-axis conversion"):
        up_axis_conversion("y", "y_up")


def test_scripted_clip_compiles_and_validates():
    spec = {
        "id": "demo",
        "phase": "standing",
        "fps": 120.0,
        "duration_s": 0.5,
        "joint_keyframes": {
            "left_knee": {
                "times_s": [0.0, 0.25, 0.5],
                "values": [[0, 0, 0], [0, 0.5, 0], [0, 0.0, 0]],
            }
        },
    }
    clip = compile_scripted_clip(spec)
    assert clip.fps == 120.0
    assert clip.frame_count == 61
    assert clip.duration_s == pytest.approx(0.5)
    knee = clip.joint_index("left_knee")
    assert clip.joint_rotations[30, knee, 1] > 0.4
    assert clip.joint_rotations[0, knee, 1] == 0.0
    assert set(clip.phase_labels) == {"standing"}
    assert clip.joint_rotations[:, clip.joint_index("spine1"), :].max() == 0.0


def test_clip_rejects_impossible_inputs():
    base = {"id": "x", "fps": 60.0, "duration_s": 1.0, "phase": "standing"}
    with pytest.raises(ValueError, match="unknown joint"):
        compile_scripted_clip({**base, "joint_keyframes": {"nose": {"values": [[0, 0, 0]]}}})
    with pytest.raises(ValueError, match="phase must be one of"):
        compile_scripted_clip({**base, "phase": "lounging"})
    with pytest.raises(ValueError, match="must start at 0 and strictly increase"):
        compile_scripted_clip(
            {
                **base,
                "joint_keyframes": {
                    "left_knee": {"times_s": [0.5, 1.0], "values": [[0, 0, 0], [0, 1, 0]]}
                },
            }
        )
    with pytest.raises(ValueError, match="unsupported keys"):
        compile_scripted_clip({**base, "nonsense": 1})
    # A 120 degree jump between two frames at 60 Hz is beyond the continuity bound.
    from sim2sense_fall.humans.motion import MotionClip, MotionProvenance

    frames = 6
    rotations = np.zeros((frames, 24, 3))
    rotations[3:, 4, 1] = MAX_FRAME_DELTA_RAD * 2.0
    with pytest.raises(ValueError, match="continuity bound"):
        MotionClip(
            clip_id="jump",
            fps=60.0,
            joint_names=SMPL_JOINT_NAMES,
            root_translation=np.zeros((frames, 3)),
            root_rotation=np.zeros((frames, 3)),
            joint_rotations=rotations,
            provenance=MotionProvenance(
                kind="scripted",
                source_id="scripted",
                subject="s",
                sequence="s",
                representation="scripted_axis_angle",
                license="project-internal",
                license_url="",
            ),
        )
    # And a rotation beyond a half turn is rejected as a representation mix-up.
    too_large = np.zeros((2, 24, 3))
    too_large[:, 4, 1] = 4.0
    with pytest.raises(ValueError, match="beyond the"):
        MotionClip(
            clip_id="huge",
            fps=60.0,
            joint_names=SMPL_JOINT_NAMES,
            root_translation=np.zeros((2, 3)),
            root_rotation=np.zeros((2, 3)),
            joint_rotations=too_large,
            provenance=MotionProvenance(
                kind="scripted",
                source_id="scripted",
                subject="s",
                sequence="s",
                representation="scripted_axis_angle",
                license="project-internal",
                license_url="",
            ),
        )


def test_sine_generator_is_analytic():
    spec = {
        "id": "wave",
        "phase": "walking",
        "fps": 120.0,
        "duration_s": 1.0,
        "generator": "sine",
        "sine": {
            "channels": [
                {"joint": "left_hip", "axis": "y", "amplitude_deg": 30.0, "frequency_hz": 1.0}
            ]
        },
    }
    clip = compile_scripted_clip(spec)
    index = clip.joint_index("left_hip")
    expected = np.radians(30.0) * np.sin(2 * math.pi * 1.0 * clip.times_s)
    assert np.allclose(clip.joint_rotations[:, index, 1], expected, atol=1e-12)
    assert clip.joint_rotations[:, index, 0].max() == 0.0


def test_motion_library_rejects_a_malformed_document():
    with pytest.raises(ValueError, match="non-empty 'motions' list"):
        motion_library_from_mapping({"motions": []})
    with pytest.raises(ValueError, match="unsupported keys"):
        motion_library_from_mapping({"motions": [{"id": "a"}], "extra": 1})
    with pytest.raises(ValueError, match="duplicate motion id"):
        motion_library_from_mapping({"motions": [{"id": "a"}, {"id": "a"}]})


def test_resample_preserves_endpoints_and_changes_rate(motions):
    clip = motions["walk_in_place"]
    upsampled = clip.resample(240.0, method="slerp")
    assert upsampled.fps == 240.0
    assert upsampled.duration_s == pytest.approx(clip.duration_s, abs=1.0 / 240.0)
    assert np.allclose(upsampled.joint_rotations[0], clip.joint_rotations[0], atol=1e-9)
    linear = clip.resample(240.0, method="linear")
    assert linear.metadata["resample_method"] == "linear"
    assert linear.metadata["resampled_from_fps"] == clip.fps
    with pytest.raises(ValueError, match="target_fps must be finite and positive"):
        clip.resample(0.0)


def test_amass_retargeting_maps_the_body_block_and_records_what_it_dropped():
    frames = 40
    rng = np.random.default_rng(11)
    poses = np.zeros((frames, 156))
    poses[:, 0:3] = rng.normal(scale=0.05, size=(frames, 3))
    poses[:, 3:66] = rng.normal(scale=0.1, size=(frames, 63))
    poses[:, 66:] = rng.normal(scale=0.4, size=(frames, 90))
    trans = np.tile(np.array([0.0, 0.95, 0.0]), (frames, 1))
    provenance = MotionProvenance(
        kind=FRAME_SOURCE_AMASS,
        source_id="amass",
        subject="Subject1",
        sequence="walk_001",
        representation="smplh_52",
        license="non-commercial research",
        license_url="https://example.invalid",
    )
    clip = retarget_amass_clip(poses, trans, clip_id="amass_walk", fps=120.0, provenance=provenance)
    assert clip.joint_count == 24
    # The model-local basis differs from the Z-up capture world. Only local
    # joint rotations are conjugated; the root maps between the two frames.
    basis = AMASS_BODY_FRAME.basis()
    assert np.allclose(basis @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(basis @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])
    assert np.allclose(basis @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(
        axis_angle_to_matrix(clip.root_rotation[0]),
        axis_angle_to_matrix(poses[0, :3]) @ basis.T,
        atol=1e-12,
    )
    assert np.allclose(clip.root_translation, trans, atol=1e-12)
    # Joint 0 carries no local rotation: the root orientation is the global one.
    assert np.allclose(clip.joint_rotations[:, 0, :], 0.0)
    # SMPL hand joints have no SMPL-H counterpart and stay at rest.
    assert np.allclose(clip.joint_rotations[:, 22:24, :], 0.0)
    assert clip.metadata["dropped_finger_joints"] == 30
    assert "no SMPL-H counterpart" in clip.provenance.notes
    bad = np.zeros((frames, 156))
    with pytest.raises(ValueError, match="expected 24 joints in the AMASS pose block, got 52"):
        retarget_amass_clip(
            bad, trans, clip_id="x", fps=60.0, provenance=provenance, total_joints=24
        )
    with pytest.raises(ValueError, match="body_joints"):
        retarget_amass_clip(
            np.zeros((frames, 156)),
            trans,
            clip_id="x",
            fps=60.0,
            provenance=provenance,
            body_joints=20,
        )


# ---------------------------------------------------------------------------
# 7.2 rig planning, capsule geometry, forward kinematics
# ---------------------------------------------------------------------------


def test_plan_covers_every_joint_and_allocates_the_mass(plan, config):
    assert len(plan.links) == config.topology.joint_count
    assert plan.root_link == "pelvis"
    assert sum(link.mass_kg for link in plan.links) == pytest.approx(plan.total_mass_kg)
    assert plan.total_mass_kg == pytest.approx(config.skeleton.mass_kg)
    assert plan.stats["link_count"] == 24
    assert plan.stats["collider_count"] == 19
    assert plan.stats["dof_count"] == 14
    assert plan.stats["proxy_link_count"] == 0
    assert plan.stats["fixed_joint_count"] == 9


def test_capsules_cover_anatomical_bones(plan):
    colliders = {capsule.link: capsule for capsule in plan.colliders}
    assert set(colliders) - {"head", "left_foot", "right_foot", "left_hand", "right_hand"}
    assert colliders["left_knee"].bone_from == "left_knee"
    assert colliders["left_knee"].bone_to == "left_ankle"
    assert colliders["pelvis"].bone_to == "spine1"
    assert colliders["spine3"].bone_to == "neck"
    for capsule in plan.colliders:
        assert capsule.radius_m > 0
        assert capsule.cylinder_length_m > 0
        assert np.allclose(np.linalg.norm(capsule.direction), 1.0, atol=1e-9)
        assert capsule.total_length_m == pytest.approx(
            capsule.cylinder_length_m + 2 * capsule.radius_m
        )


def test_quaternion_from_z_handles_the_degenerate_cases():
    assert np.allclose(quaternion_from_z((0.0, 0.0, 5.0)), (1.0, 0.0, 0.0, 0.0))
    half_turn = quaternion_from_z((0.0, 0.0, -5.0))
    assert np.allclose(
        quaternion_to_matrix(half_turn) @ np.array([0.0, 0.0, 1.0]), (0.0, 0.0, -1.0), atol=1e-12
    )
    tilted = quaternion_from_z((1.0, 0.0, 1.0))
    assert np.allclose(
        quaternion_to_matrix(tilted) @ np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="zero-length bone"):
        quaternion_from_z((0.0, 0.0, 0.0))


def test_ground_offset_puts_the_lowest_capsule_on_the_floor(plan):
    """Ground clearance is a property of the spawned pose, not of the rest offsets.

    The rig's root link is not at the body origin (SMPL's ``pelvis`` sits 0.2336 m
    above it), so summing ``lowest_world_z`` over links ignores where the root link
    actually is and reports a clearance the runtime never sees. This asserts the
    quantity the simulator uses: forward kinematics from the declared spawn.
    """

    assert plan.ground_offset_m > 0.0
    assert plan.spawn_root_position[2] == pytest.approx(plan.ground_offset_m, abs=1e-6)

    poses = forward_kinematics(plan, {}, root_position=plan.spawn_root_position)
    lowest = min(
        float(poses[link.name].translation[2]) + link.capsule.lowest_point_z()
        for link in plan.links
        if link.capsule is not None
    )
    # Both feet should be the lowest thing, and neither may be through the floor.
    assert lowest == pytest.approx(0.0, abs=1e-5)

    points, owners = pose_surface_points(plan, poses, per_segment=12, seed=5)
    assert points.shape == (12 * plan.stats["collider_count"], 3)
    assert len(owners) == points.shape[0]
    assert float(points[:, 2].min()) >= -1e-9
    assert float(points[:, 2].max()) == pytest.approx(plan.stats["standing_height_m"], abs=0.05)


def test_spawn_convention_survives_a_root_link_off_the_body_origin(config):
    """A root link above the body origin must not push the feet through the floor.

    This is the regression guard for the convention bug: with the old rest-offset
    sum the shipped SMPL plan spawned its ankles 44.6 mm below z = 0.
    """

    from sim2sense_fall.humans.skeleton import RestSkeleton

    base = default_rest_skeleton()
    shifted = RestSkeleton(
        topology=base.topology,
        joint_positions=tuple((x, y, z + 0.233593) for x, y, z in base.joint_positions),
        source="procedural skeleton shifted to mimic the SMPL rest pelvis",
    )
    plan = plan_human_rig(config, rest=shifted)
    assert plan.root.rest_position[2] == pytest.approx(0.233593, abs=1e-6)

    poses = forward_kinematics(plan, {}, root_position=plan.spawn_root_position)
    lowest = min(
        float(poses[link.name].translation[2]) + link.capsule.lowest_point_z()
        for link in plan.links
        if link.capsule is not None
    )
    assert lowest == pytest.approx(0.0, abs=1e-5)

    # The naive rest-offset sum must disagree with the spawn by exactly the root
    # link's own height; if this ever passes trivially the guard has gone stale.
    # The whole skeleton was raised, so the true spawn is *higher* than the naive sum.
    naive = -min(
        link.capsule.lowest_world_z(link.rest_position)
        for link in plan.links
        if link.capsule is not None
    )
    assert plan.ground_offset_m - naive == pytest.approx(0.233593, abs=1e-5)


def test_rest_pose_helper_agrees_with_forward_kinematics(plan):
    """``_rest_poses`` feeds the clearance measurement, so it must not drift."""

    from sim2sense_fall.humans.rig import _rest_poses

    helper = _rest_poses(plan.links, plan.joints, plan.fixed_joints, root_name=plan.root_link)
    reference = forward_kinematics(plan, {}, root_position=(0.0, 0.0, 0.0))
    assert set(helper) == set(reference)
    for name, pose in helper.items():
        assert np.allclose(pose.translation, reference[name].translation, atol=1e-12)
        assert np.allclose(pose.rotation, reference[name].rotation, atol=1e-12)


def test_standing_height_fit_is_exact_and_monotone(config):
    from dataclasses import replace

    achieved = []
    for target in (1.55, 1.70, 1.90):
        tuned = replace(config, skeleton=replace(config.skeleton, height_m=target))
        # ``standing_height_m`` is floor-to-crown, not the joint span, and it must be
        # hit exactly or the body-shape axis in stage 9 would be a lie.
        fitted = plan_human_rig(tuned)
        assert fitted.stats["standing_height_m"] == pytest.approx(target, abs=1e-6)
        assert fitted.stats["target_height_m"] == target
        achieved.append(fitted.stats["standing_height_m"])
        assert standing_height_for(tuned, default_rest_skeleton()) > 0
    assert achieved == sorted(achieved)
    with pytest.raises(ValueError, match="outside the supported"):
        plan_human_rig(replace(config, skeleton=replace(config.skeleton, height_m=12.0)))


def test_forward_kinematics_joint_sign_semantics(plan, config):
    """The direction each joint moves its descendant, asserted explicitly."""

    origin = plan.spawn_root_position
    rest = forward_kinematics(plan, {}, root_position=origin)

    def delta(joint: str, degrees: float, link: str) -> np.ndarray:
        moved = forward_kinematics(plan, {joint: math.radians(degrees)}, root_position=origin)
        return np.asarray(moved[link].translation) - np.asarray(rest[link].translation)

    cases = (
        ("left_knee", 60.0, "left_ankle", 0, -1.0, "knee flexion sweeps the shank backward"),
        ("left_hip", -60.0, "left_knee", 0, 1.0, "hip flexion swings the thigh forward"),
        ("spine1", 60.0, "neck", 0, 1.0, "spine flexion tips the trunk forward"),
        ("left_shoulder", -60.0, "left_elbow", 1, -1.0, "shoulder adduction sweeps the arm inward"),
        ("left_elbow", -60.0, "left_wrist", 0, 1.0, "elbow flexion brings the hand forward"),
        ("left_ankle", 30.0, "left_foot", 2, -1.0, "ankle plantarflexion drives the foot down"),
    )
    for joint, degrees, link, axis, sign, description in cases:
        change = delta(joint, degrees, link)
        unit = change / np.linalg.norm(change)
        assert abs(unit[axis]) > 0.5, f"{description}: got {unit.tolist()}"
        assert np.sign(unit[axis]) == np.sign(sign), f"{description}: got {unit.tolist()}"

    # The limits must not cap the motion the sign test relies on.
    for joint, degrees, *_ in cases:
        limits = plan.limits_deg()[joint]
        assert limits[0] <= degrees <= limits[1], joint

    left = delta("left_knee", 45.0, "left_ankle")
    right = delta("right_knee", 45.0, "right_ankle")
    assert np.allclose(left, right, atol=1e-9)

    # The arms are a mirror, not a copy: the same signed angle sweeps one inward and
    # the other inward the other way, so the two agree only after the lateral flips.
    left_arm = delta("left_shoulder", -60.0, "left_elbow")
    right_arm = delta("right_shoulder", 60.0, "right_elbow")
    assert np.allclose(left_arm * np.array([1.0, -1.0, 1.0]), right_arm, atol=1e-3)


def test_neutral_pose_offsets_deviations_without_choosing_a_rest_pose(plan, config):
    """A clip authors movement from the body's neutral pose, not from the file's rest.

    The released SMPL template rests in a T-pose and the procedural skeleton rests with
    the arms hanging, so a zero-pose clip means two different bodies unless one declared
    neutral is added. That angle belongs in the human config, once -- not copy-pasted
    into every motion file, where it would go stale the next time the rest changes.
    """

    neutral = config.visualization.pose_dict()
    zero = compile_scripted_clip(
        {
            "id": "zero",
            "fps": 60.0,
            "generator": "keyframes",
            "duration_s": 1.0,
            "phase": "standing",
            "tags": [],
            "notes": "",
        },
        topology=config.topology,
    )
    posed = apply_neutral_pose(zero, plan, neutral)
    for name, value in neutral.items():
        joint = next(item for item in plan.joints if item.name == name)
        slot = "xyz".index(joint.axis)
        index = posed.joint_index(joint.chain_joint)
        assert np.allclose(
            posed.joint_rotations[:, index, slot], value
        ), f"{name}: neutral angle was not added to its own axis slot"
        # Every other slot stays exactly where the clip put it: the offset must not be
        # able to author off-axis rotation the single-axis rig cannot express.
        others = np.delete(posed.joint_rotations[:, index, :], slot, axis=1)
        assert np.allclose(others, 0.0), f"{name}: the offset leaked into another axis"

    # A deviation authored on top of the neutral composes, rather than replacing it.
    lifted = apply_neutral_pose(
        zero, plan, {**neutral, "left_shoulder": neutral["left_shoulder"] + 0.4}
    )
    index = lifted.joint_index("left_shoulder")
    assert np.allclose(lifted.joint_rotations[:, index, 0], neutral["left_shoulder"] + 0.4)

    with pytest.raises(ValueError, match="not a DOF"):
        apply_neutral_pose(zero, plan, {"left_foot": 0.1})
    # +105 deg of left adduction is outside the authored [-120, 75] range: an offset that
    # silently exceeds the joint limit would be a target the controller can never reach.
    with pytest.raises(ValueError, match="outside the authored limits"):
        apply_neutral_pose(zero, plan, {"left_shoulder": math.radians(105.0)})

    # Only deviation-authored clips get the offset. AMASS local rotations are absolute
    # against the model's T-pose rest, so the arm-down adduction is already in the data
    # and adding the neutral again would fold the arms through the body.
    from dataclasses import replace

    assert authors_deviations(zero)
    amass = replace(
        zero,
        provenance=MotionProvenance(
            kind=FRAME_SOURCE_AMASS,
            source_id="amass",
            subject="Subject1",
            sequence="01/01_01",
            representation="smplh_52",
            license="non-commercial research",
            license_url="https://example.invalid",
        ),
    )
    assert not authors_deviations(amass)


def test_forward_kinematics_is_root_relative_and_rigid(plan):
    a = forward_kinematics(plan, {}, root_position=(0.0, 0.0, 0.0))
    b = forward_kinematics(plan, {}, root_position=(10.0, -3.0, 1.0))
    for name in a:
        assert np.allclose(
            np.asarray(b[name].translation) - np.asarray(a[name].translation),
            (10.0, -3.0, 1.0),
            atol=1e-9,
        )
    rotated = forward_kinematics(
        plan, {}, root_position=(0.0, 0.0, 0.0), root_rotation=(0.0, math.pi / 2, 0.0)
    )
    # A +90 degree pitch about +Y sends the body up axis onto +X.
    head = np.asarray(rotated["head"].translation)
    assert head[0] > 0.5 and abs(head[2]) < 0.1


def test_forward_kinematics_rejects_bad_input(plan):
    with pytest.raises(ValueError, match="must be a single number"):
        forward_kinematics(plan, {"left_knee": [0.1, 0.2]})
    with pytest.raises(ValueError, match="must be finite"):
        forward_kinematics(plan, {"left_knee": float("nan")})
    with pytest.raises(ValueError, match="three components"):
        forward_kinematics(plan, {}, root_position=(0.0, 0.0))


def test_multi_axis_joints_become_a_proxy_chain(config):
    from dataclasses import replace

    from sim2sense_fall.humans.config import DriveConfig, JointConfig

    hip = config.rig.joints["left_hip"]
    two_axis = JointConfig(
        joint="left_hip",
        dof="revolute",
        rotations=("y", "x"),
        limits_deg=((-120.0, 30.0), (-45.0, 20.0)),
        drive=DriveConfig(stiffness=400.0, damping=40.0, max_force=400.0),
    )
    rig = replace(config.rig, joints={**config.rig.joints, "left_hip": two_axis})
    tuned = replace(config, rig=rig)
    chained = plan_human_rig(tuned)
    assert chained.stats["proxy_link_count"] == 1
    assert chained.stats["dof_count"] == plan_human_rig(config).stats["dof_count"] + 1
    proxy = chained.link("left_hip__dof1")
    assert proxy.role == "dof_proxy"
    assert proxy.capsule is None
    assert proxy.parent_link == "pelvis"
    assert chained.link("left_hip").parent_link == "left_hip__dof1"
    unnamed = hip.rotations[0]
    assert unnamed in ("x", "y", "z")
    assert chained.joint("left_hip__dof1").axis == "y"
    assert chained.joint("left_hip").axis == "x"
    assert chained.joint("left_hip__dof1").proxy
    # Mass must be conserved across the chain.
    assert sum(link.mass_kg for link in chained.links) == pytest.approx(chained.total_mass_kg)


def test_anchored_root_mode_is_recorded(config):
    from dataclasses import replace

    anchored = plan_human_rig(replace(config, rig=replace(config.rig, root_mode="anchored")))
    assert anchored.root_mode == "anchored"
    free = plan_human_rig(config)
    assert free.root_mode == "free"
    assert anchored.spawn_root_position == free.spawn_root_position


def test_plan_json_has_no_nan(plan):
    payload = json.loads(plan.to_json())
    assert payload["human_id"] == plan.human_id
    assert len(payload["joints"]) == len(plan.joints)
    assert payload["stats"]["collider_count"] == plan.stats["collider_count"]
    assert "capsule" in payload["links"][0]
    assert payload["links"][0]["capsule"]["path"].startswith("/World/Human/")


def test_plan_validation_rejects_a_broken_rig(config):
    from dataclasses import replace

    from sim2sense_fall.humans.config import SegmentConfig

    starved = replace(
        config,
        rig=replace(
            config.rig,
            segments={
                **config.rig.segments,
                "pelvis": SegmentConfig(
                    joint="pelvis", radius_m=0.135, mass_weight=1e-6, aim="spine1"
                ),
            },
        ),
    )
    with pytest.raises(ValueError, match="is below"):
        plan_human_rig(starved)


# ---------------------------------------------------------------------------
# 7.3/7.4 events: fall versus controlled lowering
# ---------------------------------------------------------------------------


def _synthetic_trajectory(
    *,
    frames: int = 240,
    fps: float = 120.0,
    topple_degrees: float = 0.0,
    drop_m: float = 0.0,
    topple_at: float = 0.5,
    topple_duration: float = 0.5,
    standing_height_m: float = 1.7,
) -> Trajectory:
    """A body that tips and drops between two times, with a hollow body point cloud."""

    times = np.arange(frames) / fps
    progress = np.clip((times - topple_at) / topple_duration, 0.0, 1.0)
    angle = np.radians(topple_degrees) * progress
    rotation = np.zeros((frames, 3))
    rotation[:, 1] = angle
    pelvis_z = standing_height_m - drop_m * progress - 0.4
    root = np.stack([np.full(frames, 1.0), np.zeros(frames), pelvis_z], axis=1)
    trunk = np.tile(np.array([0.0, 0.0, 1.0]), (frames, 1))
    trunk[:, 0] = np.sin(angle)
    trunk[:, 2] = np.cos(angle)
    points = np.zeros((frames, 4, 3))
    for frame in range(frames):
        head = pelvis_z[frame] + 0.7 * math.cos(angle[frame])
        foot = pelvis_z[frame] - 0.9 * math.cos(angle[frame])
        points[frame, 0] = (0.0, 0.0, max(0.0, min(head, foot)))
        points[frame, 1] = (0.0, 0.0, pelvis_z[frame])
        points[frame, 2] = (0.0, 0.0, max(0.0, pelvis_z[frame] - 0.9 * progress[frame]))
        points[frame, 3] = (0.0, 0.0, max(0.0, head))
    quaternion = np.stack(
        [np.array([math.cos(row[1] / 2), 0.0, math.sin(row[1] / 2), 0.0]) for row in rotation],
        axis=0,
    )
    return Trajectory(
        times_s=times,
        root_position=root,
        root_quaternion=quaternion,
        joint_positions=np.zeros((frames, 14)),
        joint_names=tuple(f"j{index}" for index in range(14)),
        body_points=points,
        standing_height_m=standing_height_m,
        trunk_axis=trunk,
    )


def test_standing_trial_is_not_a_fall(config):
    label = label_trial(_synthetic_trajectory(), config.events)
    assert label.label == LABEL_NO_FALL
    assert label.valid
    assert label.imbalance_onset_s is None
    assert label.first_impact_s is None
    assert label.final_posture == "upright"
    assert "never disturbed" in label.reasons[0]


def test_slow_lowering_is_not_labelled_a_fall(config):
    """The boundary the stage plan asks for: deliberate lying down is not a fall."""

    label = label_trial(
        _synthetic_trajectory(frames=600, topple_degrees=88.0, drop_m=0.75, topple_duration=3.0),
        config.events,
    )
    assert label.label == LABEL_CONTROLLED_LOWERING
    assert label.final_posture == "lying"
    assert label.imbalance_onset_s is not None
    assert label.first_impact_s is None
    assert any("controlled lowering" in reason for reason in label.reasons)


def test_a_tracked_level_drop_that_never_goes_through_the_floor_is_valid(config):
    label = label_trial(
        _synthetic_trajectory(frames=240, topple_degrees=88.0, drop_m=0.75, topple_duration=0.6),
        config.events,
    )
    assert label.valid
    assert label.label in (LABEL_CONTROLLED_LOWERING, LABEL_FALL)


def test_penetrating_and_diverging_trials_are_invalid_not_labelled(config):
    frames, fps = 240, 120.0
    times = np.arange(frames) / fps
    trajectory = _synthetic_trajectory(frames=frames, topple_degrees=90.0, drop_m=0.9)
    points = trajectory.body_points.copy()
    points[100:, :, 2] -= 0.2
    sunk = Trajectory(
        times_s=times,
        root_position=trajectory.root_position,
        root_quaternion=trajectory.root_quaternion,
        joint_positions=trajectory.joint_positions,
        joint_names=trajectory.joint_names,
        body_points=points,
        standing_height_m=trajectory.standing_height_m,
        trunk_axis=trajectory.trunk_axis,
    )
    label = label_trial(sunk, config.events)
    assert label.label == LABEL_INVALID
    assert not label.valid
    assert "penetration limit" in label.reasons[0]

    # Divergence means the body *travelled*, so the offset must ramp rather than shift
    # the whole trajectory uniformly.
    ramp = np.linspace(0.0, 20.0, frames)
    offset = np.stack([np.zeros(frames), ramp, np.zeros(frames)], axis=1)
    drifted = Trajectory(
        times_s=times,
        root_position=trajectory.root_position + offset,
        root_quaternion=trajectory.root_quaternion,
        joint_positions=trajectory.joint_positions,
        joint_names=trajectory.joint_names,
        body_points=trajectory.body_points + offset[:, None, :],
        standing_height_m=trajectory.standing_height_m,
        trunk_axis=trajectory.trunk_axis,
    )
    diverged = label_trial(drifted, config.events)
    assert diverged.label == LABEL_INVALID
    assert "divergence limit" in diverged.reasons[0]


def test_trunk_axis_sees_a_bend_the_root_cannot(config):
    """A bend at the waist with an upright pelvis must still register as a tilt."""

    frames, fps = 120, 120.0
    times = np.arange(frames) / fps
    root = np.tile(np.array([0.0, 0.0, 0.9]), (frames, 1))
    quaternion = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1))
    points = np.tile(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]]), (frames, 1, 1))
    bend = np.radians(70.0)
    trunk = np.tile(np.array([math.sin(bend), 0.0, math.cos(bend)]), (frames, 1))
    without = Trajectory(
        times_s=times,
        root_position=root,
        root_quaternion=quaternion,
        joint_positions=np.zeros((frames, 1)),
        joint_names=("j",),
        body_points=points,
        standing_height_m=1.7,
    )
    with_axis = Trajectory(
        times_s=times,
        root_position=root,
        root_quaternion=quaternion,
        joint_positions=np.zeros((frames, 1)),
        joint_names=("j",),
        body_points=points,
        standing_height_m=1.7,
        trunk_axis=trunk,
    )
    assert without.trunk_angle_deg.max() == pytest.approx(0.0, abs=1e-9)
    assert with_axis.trunk_angle_deg.max() == pytest.approx(70.0, abs=1e-6)
    assert with_axis.descent_speed_m_s.max() == pytest.approx(0.0, abs=1e-9)
    assert with_axis.speed_m_s.shape == (frames,)


def test_trunk_axis_helper_rejects_impossible_inputs():
    positions = np.zeros((3, 3, 3))
    positions[:, 0, 2] = 1.0
    positions[:, 1, 2] = 2.0
    axis = trunk_axis_from_link_positions(("pelvis", "neck", "head"), positions)
    assert np.allclose(axis, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="no 'pelvis' link"):
        trunk_axis_from_link_positions(("a", "neck"), positions[:, :2])
    with pytest.raises(ValueError, match="coincide"):
        trunk_axis_from_link_positions(("pelvis", "neck"), np.zeros((3, 2, 3)))
    with pytest.raises(ValueError, match="neither 'neck'"):
        trunk_axis_from_link_positions(("pelvis", "tail"), positions[:, :2])


def test_phase_labels_follow_the_detected_events():
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    labels = phase_labels_from_signal(
        times,
        standing_frames=1,
        onset_s=0.2,
        impact_s=0.3,
        stabilisation_s=0.4,
        final_phase="fallen",
    )
    assert labels == ("standing", "standing", "falling", "fallen", "fallen", "fallen")
    recovered = phase_labels_from_signal(
        times,
        standing_frames=1,
        onset_s=0.2,
        impact_s=0.3,
        stabilisation_s=0.4,
        final_phase="standing_recovery",
    )
    assert recovered[-1] == "standing_recovery"
    assert set(
        phase_labels_from_signal(
            times,
            standing_frames=0,
            onset_s=None,
            impact_s=None,
            stabilisation_s=None,
            final_phase="standing",
        )
    ) == {"standing"}
    with pytest.raises(ValueError, match="final_phase must be one of"):
        phase_labels_from_signal(
            times,
            standing_frames=0,
            onset_s=None,
            impact_s=None,
            stabilisation_s=None,
            final_phase="somersault",
        )


def test_events_config_rejects_contradictory_thresholds():
    base = dict(
        trunk_angle_deg=60.0,
        trunk_onset_fraction=0.5,
        pelvis_height_fraction=0.55,
        min_low_frames=10,
        max_transition_s=1.5,
        impact_height_m=0.25,
        impact_speed_m_s=1.0,
        settle_window_s=0.5,
        settle_speed_m_s=0.15,
        controlled_descent_speed_m_s=1.2,
        divergence_limit_m=5.0,
        penetration_limit_m=-0.05,
        recovery_height_fraction=0.6,
    )
    assert EventsConfig(**base).trunk_angle_deg == 60.0
    with pytest.raises(ValueError, match="trunk_onset_fraction must be within"):
        EventsConfig(**{**base, "trunk_onset_fraction": 1.5})
    with pytest.raises(ValueError, match="min_low_frames"):
        EventsConfig(**{**base, "min_low_frames": 0})
    with pytest.raises(ValueError, match="must be within"):
        EventsConfig(**{**base, "recovery_height_fraction": 2.0})


# ---------------------------------------------------------------------------
# 7.5 export contract
# ---------------------------------------------------------------------------


def _provenance() -> TrialProvenance:
    return TrialProvenance(
        scene_id="apartment_cn_two_bedroom",
        scene_sha256="a" * 64,
        human_id="smpl_neutral_standing",
        rig_plan_sha256=sha256_text("plan"),
        config_sha256=sha256_text("config"),
        motion_id="stand_neutral",
        motion_kind="scripted",
        motion_sha256=sha256_text("motion"),
        controller_mode="pd",
        root_mode="free",
        root_anchor_used=False,
        body_representation="capsule_proxy_surface",
        perturbation_id="push_forward",
        perturbation_detail={"kind": "force", "applied": True},
        seed=20260922,
        physics_dt_s=1.0 / 120.0,
        channel_sample_hz=50.0,
        resample_method="slerp",
        standing_height_m=1.7,
        total_mass_kg=72.0,
        generated_by="tests",
        subject="scripted",
        sequence="stand_neutral",
    )


def test_provenance_requires_the_traceable_fields_and_rejects_a_hidden_anchor():
    provenance = _provenance()
    assert provenance.as_dict()["body_representation"] == "capsule_proxy_surface"
    from dataclasses import replace

    with pytest.raises(ValueError, match="root_anchor_used is set"):
        replace(provenance, root_anchor_used=True)
    with pytest.raises(ValueError, match="body_representation"):
        replace(provenance, body_representation="point cloud")
    with pytest.raises(ValueError, match="64-character digest"):
        replace(provenance, config_sha256="short")
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        replace(provenance, seed=-1)


def test_isolation_keys_are_namespaced_and_per_field():
    """R5: a joined subject|sequence key does NOT keep a person on one side.

    The keys are what a splitter groups on, so each isolation field has to appear on its
    own, prefixed with the dataset because two sources can both call a performer
    ``person_01``.
    """

    fields = {"dataset": "AMASS", "subject": "person1", "sequence": "clip1"}
    keys = clip_group_keys(fields, isolate=["subject", "sequence"])
    assert keys == ("subject=AMASS:person1", "sequence=AMASS:clip1")
    other = clip_group_keys({**fields, "sequence": "clip2"}, isolate=["subject", "sequence"])
    assert "subject=AMASS:person1" in other, "the same person must share a group key"
    assert clip_group_keys(fields, isolate=["subject"], dataset="CMU") == (
        "subject=CMU:person1",
    ), "the namespace is the dataset, so two sources cannot collide on a subject name"
    with pytest.raises(ValueError, match="isolate must name"):
        clip_group_keys(fields, isolate=[])
    with pytest.raises(ValueError, match="unavailable fields"):
        clip_group_keys(fields, isolate=["performer"])


def test_connected_groups_keep_a_person_out_of_both_splits():
    """Two clips of one performer join one group, and a split that separates them fails."""

    keys = {
        "trial_a": clip_group_keys(
            {"dataset": "AMASS", "subject": "person1", "sequence": "clip1"},
            isolate=["subject", "sequence"],
        ),
        "trial_b": clip_group_keys(
            {"dataset": "AMASS", "subject": "person1", "sequence": "clip2"},
            isolate=["subject", "sequence"],
        ),
        "trial_c": clip_group_keys(
            {"dataset": "AMASS", "subject": "person2", "sequence": "clip1"},
            isolate=["subject", "sequence"],
        ),
    }
    groups = connected_groups(keys)
    # Isolating on sequence as well joins trial_c in: it reuses the source clip
    # "clip1", so that recording cannot sit on the other side either. This is the point of
    # grouping on every field at once rather than picking one.
    assert groups == [frozenset({"trial_a", "trial_b", "trial_c"})]
    assert split_violations(groups, {"trial_a": "train", "trial_b": "test"}) == [
        (groups[0], ("test", "train"))
    ]
    assert not split_violations(
        groups, {"trial_a": "train", "trial_b": "train", "trial_c": "train"}
    )
    only_subject = connected_groups(
        {name: tuple(k for k in keys[name] if k.startswith("subject")) for name in keys}
    )
    assert only_subject == [frozenset({"trial_a", "trial_b"}), frozenset({"trial_c"})]


def test_a_transitive_chain_joins_one_group():
    """Shared keys chain: person1/clip2 links to someone else through the sequence key."""

    keys = {
        "a": ("subject=D:p1", "sequence=D:s1"),
        "b": ("subject=D:p2", "sequence=D:s1"),
        "c": ("subject=D:p2", "sequence=D:s2"),
    }
    assert connected_groups(keys) == [frozenset({"a", "b", "c"})]


def test_resample_series_interpolates_and_refuses_bad_shapes():
    source = np.linspace(0.0, 1.0, 11)
    values = np.stack([source, source**2], axis=1)
    target = np.linspace(0.0, 1.0, 21)
    linear = resample_series(source, target, values, method="linear")
    assert linear.shape == (21, 2)
    assert np.allclose(linear[0], values[0])
    assert np.allclose(linear[-1], values[-1])
    smooth = resample_series(source, target, values, method="smoothstep")
    assert smooth.shape == linear.shape
    quaternions = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (11, 1))
    assert np.allclose(
        resample_series(source, target, quaternions, method="quaternion"),
        quaternions[0],
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="\\(N, 4\\) array"):
        resample_series(source, target, values, method="slerp")
    with pytest.raises(ValueError, match="does not match"):
        resample_series(source, target, values[:5], method="linear")
    with pytest.raises(ValueError, match="strictly increase"):
        resample_series(source, source[::-1], values, method="linear")


def test_ground_truth_writes_both_clocks_and_refuses_nan(tmp_path, config, plan):
    from dataclasses import replace

    frames = 121
    times = np.arange(frames) / 120.0
    channel_times = np.arange(51) / 50.0
    trajectory = _synthetic_trajectory(frames=frames, topple_degrees=0.0)
    label = label_trial(trajectory, config.events)
    truth = GroundTruth(
        provenance=_provenance(),
        label=label,
        time_physics_s=times,
        time_channel_s=channel_times,
        root_position=trajectory.root_position,
        root_quaternion=trajectory.root_quaternion,
        joint_positions_rad=np.zeros((frames, len(plan.dof_names))),
        joint_velocities_rad_s=np.zeros((frames, len(plan.dof_names))),
        joint_names=plan.dof_names,
        link_positions=np.zeros((frames, len(plan.links), 3)),
        link_names=tuple(link.name for link in plan.links),
        body_points=trajectory.body_points,
        body_point_owners=("pelvis",) * trajectory.body_points.shape[1],
        mesh_vertices_xyz=trajectory.body_points,
        mesh_faces=np.array([[0, 1, 2]], dtype=np.int64),
        mesh_representation="capsule_proxy_mesh",
        mesh_vertex_owners=("pelvis",) * trajectory.body_points.shape[1],
        phase_labels=phase_labels_from_signal(
            times,
            standing_frames=1,
            onset_s=None,
            impact_s=None,
            stabilisation_s=None,
            final_phase="standing",
        ),
        channel_root_position=np.zeros((51, 3)),
        channel_root_quaternion=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (51, 1)),
        channel_joint_positions_rad=np.zeros((51, len(plan.dof_names))),
        channel_body_points=np.zeros((51, trajectory.body_points.shape[1], 3)),
        contact_force_n=np.zeros(frames),
    )
    written = truth.write(tmp_path, name="unit")
    assert written["npz"].is_file() and written["json"].is_file()
    with np.load(written["npz"], allow_pickle=False) as payload:
        assert "time_physics_s" in payload
        assert "time_channel_s" in payload
        assert payload["time_physics_s"].shape == (frames,)
        assert payload["time_channel_s"].shape == (51,)
        assert payload["joint_names"].shape == (len(plan.dof_names),)
        assert payload["phase_label_index"].shape == (frames,)
        assert payload["mesh_vertices_xyz"].shape == trajectory.body_points.shape
        assert payload["mesh_faces"].shape == (1, 3)
    document = json.loads(written["json"].read_text(encoding="utf-8"))
    assert document["provenance"]["channel_sample_hz"] == 50.0
    assert document["frame_count"] == frames
    assert document["channel_frame_count"] == 51
    assert document["has_contact_forces"] is True
    assert document["mesh_representation"] == "capsule_proxy_mesh"
    assert document["mesh_face_count"] == 1
    assert document["max_root_quaternion_error"] < 1e-9

    # NaN is rejected when the record is built, before any file is touched, so a
    # corrupt run can never leave a file that silently poisons training.
    with pytest.raises(ValueError, match="non-finite"):
        replace(truth, joint_velocities_rad_s=np.full((frames, len(plan.dof_names)), np.nan))


def test_ground_truth_rejects_shape_and_label_mismatches(config, plan):
    frames = 20
    times = np.arange(frames) / 120.0
    trajectory = _synthetic_trajectory(frames=frames, topple_degrees=0.0)
    label = label_trial(trajectory, config.events)
    common = dict(
        provenance=_provenance(),
        label=label,
        time_physics_s=times,
        time_channel_s=np.array([0.0, 0.1]),
        root_position=trajectory.root_position,
        root_quaternion=trajectory.root_quaternion,
        joint_positions_rad=np.zeros((frames, len(plan.dof_names))),
        joint_velocities_rad_s=np.zeros((frames, len(plan.dof_names))),
        joint_names=plan.dof_names,
        link_positions=np.zeros((frames, len(plan.links), 3)),
        link_names=tuple(link.name for link in plan.links),
        body_points=trajectory.body_points,
        body_point_owners=("pelvis",) * trajectory.body_points.shape[1],
        phase_labels=(),
    )
    GroundTruth(**common)
    with pytest.raises(ValueError, match="phase_labels has"):
        GroundTruth(**{**common, "phase_labels": ("standing",)})
    with pytest.raises(ValueError, match="body_point_owners has"):
        GroundTruth(**{**common, "body_point_owners": ("pelvis",)})
    with pytest.raises(ValueError, match="contact_force_n must be one value"):
        GroundTruth(**{**common, "contact_force_n": np.zeros(3)})


def test_capsule_proxy_mesh_is_fixed_topology_and_follows_link_pose(plan):
    template = build_capsule_proxy_template(plan, segments=8)
    assert template.vertices_local.ndim == 2
    assert template.vertices_local.shape[1] == 3
    assert template.topology.faces.shape[1] == 3
    template.topology.validate_vertex_count(len(template.vertices_local))
    rest_poses = forward_kinematics(plan, {})
    moved = dict(rest_poses)
    moved[plan.links[1].name] = type(rest_poses[plan.links[1].name])(
        rotation=rest_poses[plan.links[1].name].rotation,
        translation=rest_poses[plan.links[1].name].translation + np.array([0.1, 0.0, 0.0]),
    )
    first = pose_capsule_proxy_mesh(template, plan, rest_poses)
    second = pose_capsule_proxy_mesh(template, plan, moved)
    assert np.all(np.isfinite(first))
    assert np.any(np.abs(first - second) > 1e-12)
    sequence = MeshSequence(
        np.array([0.0, 0.1]),
        np.stack((first, second)),
        template.topology,
        representation="capsule_proxy_mesh",
        vertex_owners=template.owners,
    )
    assert sequence.as_metadata()["mesh_coordinate_system"] == "world_z_up_xyz"
    assert sequence.as_metadata()["mesh_units"] == "m"
    with pytest.raises(ValueError, match="strictly increasing"):
        MeshSequence(
            np.array([0.1, 0.0]),
            np.stack((first, second)),
            template.topology,
            representation="capsule_proxy_mesh",
        )


def test_mesh_topology_rejects_invalid_faces():
    with pytest.raises(ValueError, match="integers"):
        MeshTopology(np.array([[0.0, 1.0, 2.0]]))
    topology = MeshTopology(np.array([[0, 1, 3]], dtype=np.int64))
    with pytest.raises(ValueError, match="outside"):
        topology.validate_vertex_count(3)


def test_smpl_mesh_fit_matches_the_scaled_rig_rest_joints(plan):
    payload = _tiny_mesh_payload()
    mesh = mesh_from_model_payload(payload)
    target = np.asarray(mesh.rest_joints) * 1.25 + np.array([0.01, -0.02, 0.03])
    fitted = fit_mesh_to_rest_joints(mesh, target)
    assert np.allclose(fitted.rest_joints, target)
    assert np.allclose(fitted.vertices, mesh.vertices * 1.25 + np.array([0.01, -0.02, 0.03]))


# ---------------------------------------------------------------------------
# configuration document
# ---------------------------------------------------------------------------


def test_shipped_config_loads_and_is_self_consistent(config):
    assert config.human_id == "smpl_neutral_standing"
    assert config.rig.root_mode == "free"
    assert config.control.mode == "pd"
    assert config.simulation.physics_dt_s == pytest.approx(1.0 / 120.0)
    assert config.export.channel_sample_hz == 50.0
    assert config.export.surface_point_seed == 20260922
    assert config.export.group_by == ("subject", "sequence")
    assert [entry.id for entry in config.perturbations] == [
        "push_forward",
        "push_backward",
        "push_right",
        "push_left",
        "control_loss",
        "none",
    ]
    assert config.perturbation("push_forward").kind == "force"
    assert config.perturbation("none").kind == "none"


def test_display_pose_is_configured_in_radian_limits(config):
    pose = config.visualization.pose_dict()
    # Arms at the sides, matching the standing reference rather than the model's
    # T-pose rest: +pi/2 here used to be a twist about the arm and changed nothing.
    assert pose["left_shoulder"] == pytest.approx(math.radians(-105.0))
    assert pose["right_shoulder"] == pytest.approx(math.radians(105.0))
    assert pose["left_elbow"] == pytest.approx(-0.2)
    assert pose["right_elbow"] == pytest.approx(0.2)
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["visualization"]["default_pose_rad"]["left_shoulder"] = 3.0
    with pytest.raises(ValueError, match="outside"):
        human_config_from_mapping(payload)


def test_config_rejects_a_missing_pelvis_aim():
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    del payload["rig"]["segments"]["pelvis"]["aim"]
    with pytest.raises(ValueError, match="must declare 'aim'"):
        human_config_from_mapping(payload)


def test_config_rejects_an_unknown_perturbation_target():
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["perturbations"][0]["body"] = "nose"
    with pytest.raises(ValueError, match="is not a joint of the smpl topology"):
        human_config_from_mapping(payload)


def test_config_rejects_a_forward_control_failure():
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for entry in payload["perturbations"]:
        if entry["id"] == "control_loss":
            entry["control_scale"] = 1.0
    with pytest.raises(ValueError, match="control_failure needs control_scale < 1"):
        human_config_from_mapping(payload)


def test_config_supports_root_override_and_rejects_bad_limits():
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["rig"]["root_mode"] = "anchored"
    payload["rig"]["joints"]["left_knee"]["limits_deg"] = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="low < high"):
        human_config_from_mapping(payload)
    payload["rig"]["joints"]["left_knee"]["limits_deg"] = [[0.0, 700.0]]
    with pytest.raises(ValueError, match="exceeds the"):
        human_config_from_mapping(payload)
    payload["rig"]["joints"]["left_knee"]["limits_deg"] = [[0.0, 150.0]]
    payload["rig"]["segments"]["pelvis"]["aim"] = "left_hip"
    tuned = human_config_from_mapping(payload)
    assert tuned.rig.root_mode == "anchored"
    assert tuned.rig.segment("pelvis").aim == "left_hip"


def test_reference_library_covers_falls_and_confusables(motions):
    assert len(motions) == 9
    tags = {name: set(clip.tags) for name, clip in motions.items()}
    confusables = [name for name, tag in tags.items() if "confusable" in tag]
    fall_references = [name for name, tag in tags.items() if "fall_reference" in tag]
    assert set(confusables) == {"bend_forward", "squat", "sit_down"}
    assert len(fall_references) >= 3
    for clip in motions.values():
        assert clip.provenance.kind == "scripted"
        assert clip.provenance.source_id == "scripted"
        assert "not SMPL" not in clip.provenance.notes
        assert clip.frame_count >= 2


# ---------------------------------------------------------------------------
# skinning: the missing piece between "an articulation" and "a body with skin"
# ---------------------------------------------------------------------------


def _tiny_mesh_payload(vertex_count: int = 6) -> dict:
    """A minimal, structurally valid SMPL-shaped payload.

    Six vertices on two rings, one face per ring, rigid-ish weights so the rest pose is
    reproduced and a rotation actually moves something.
    """

    vertices = np.zeros((vertex_count, 3))
    for index in range(vertex_count):
        ring = index // (vertex_count // 2)
        vertices[index] = (0.05 * (index % 3), 0.0, 0.5 * ring + 0.05 * (index % 3))
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    weights = np.zeros((vertex_count, 24))
    for index in range(vertex_count):
        ring = index // (vertex_count // 2)
        # Ring 0 hangs off the hips, ring 1 off the knees; each row sums to 1.
        weights[index, 1 if ring == 0 else 4] = 1.0
    regressor = np.zeros((24, vertex_count))
    for joint in range(24):
        regressor[joint, joint % vertex_count] = 1.0
    return {"v_template": vertices, "f": faces, "weights": weights, "J_regressor": regressor}


def test_mesh_from_payload_validates_weights_and_faces():
    mesh = mesh_from_model_payload(_tiny_mesh_payload())
    assert mesh.vertex_count == 6
    assert mesh.face_count == 2
    assert mesh.rest_joints is not None and mesh.rest_joints.shape == (24, 3)
    assert mesh.topology.joint_count == 24

    bad = _tiny_mesh_payload()
    bad["weights"] = bad["weights"] * 0.5
    with pytest.raises(ValueError, match="must sum to 1 per vertex"):
        mesh_from_model_payload(bad)

    bad_faces = _tiny_mesh_payload()
    bad_faces["f"] = np.array([[0, 1, 99]], dtype=int)
    with pytest.raises(ValueError, match="face indices"):
        mesh_from_model_payload(bad_faces)

    incomplete = {"v_template": np.zeros((3, 3)), "f": np.zeros((1, 3), dtype=int)}
    with pytest.raises(ValueError, match="missing 'weights'"):
        mesh_from_model_payload(incomplete)


def test_rest_pose_is_reproduced_exactly():
    """The invariant that catches offset and sign errors in the joint transforms."""

    mesh = mesh_from_model_payload(_tiny_mesh_payload())
    identity = {
        name: (np.eye(3), np.asarray(mesh.rest_joints[index]))
        for index, name in enumerate(mesh.topology.joint_names)
    }
    skinned = skin_with_link_poses(mesh, identity)
    assert np.allclose(skinned, mesh.vertices, atol=1e-12)


def test_skinning_follows_a_joint_rotation():
    mesh = mesh_from_model_payload(_tiny_mesh_payload())
    # Rotate the left hip 90 degrees about +Y: its vertices must swing forward (+X).
    poses = {
        name: (np.eye(3), np.asarray(mesh.rest_joints[index]))
        for index, name in enumerate(mesh.topology.joint_names)
    }
    poses["left_hip"] = (
        axis_angle_to_matrix(np.array([0.0, math.pi / 2, 0.0])),
        np.asarray(mesh.rest_joints[1]),
    )
    skinned = skin_with_link_poses(mesh, poses)
    hip_vertices = skinned[:3]
    knee_vertices = skinned[3:]
    # Only the vertices weighted to the hip may move.
    assert np.allclose(knee_vertices, mesh.vertices[3:], atol=1e-12), "knee vertices must not move"
    movement = float(np.abs(hip_vertices - mesh.vertices[:3]).max())
    assert movement > 0.02, f"hip vertices barely moved ({movement:.4f} m): {hip_vertices.tolist()}"
    # And the skin stayed rigid for a rigid part: distances inside the hip group are held.
    rest_spread = np.linalg.norm(mesh.vertices[0] - mesh.vertices[2])
    posed_spread = np.linalg.norm(hip_vertices[0] - hip_vertices[2])
    assert abs(rest_spread - posed_spread) < 1e-9


def test_skinning_rejects_bad_transforms():
    mesh = mesh_from_model_payload(_tiny_mesh_payload())
    poses = {
        name: (np.eye(3), np.asarray(mesh.rest_joints[index]))
        for index, name in enumerate(mesh.topology.joint_names)
    }
    poses["left_hip"] = (np.eye(3) * 3.0, np.asarray(mesh.rest_joints[1]))
    with pytest.raises(ValueError, match="determinant"):
        skin_with_link_poses(mesh, poses)

    no_joints = SmplMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        weights=mesh.weights,
        topology=mesh.topology,
        rest_joints=None,
    )
    with pytest.raises(ValueError, match="no rest joint centres"):
        skin_with_link_poses(no_joints, poses)
    with pytest.raises(ValueError, match="no J_regressor"):
        no_joints.rest_skeleton()


def test_shape_coefficients_move_vertices_and_are_checked():
    mesh = mesh_from_model_payload(_tiny_mesh_payload())
    directions = np.zeros((mesh.vertex_count, 3, 2))
    directions[:, 2, 0] = 0.1
    shaped_mesh = SmplMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        weights=mesh.weights,
        topology=mesh.topology,
        rest_joints=mesh.rest_joints,
        shape_directions=directions,
    )
    taller = shaped_mesh.shaped([1.0, 0.0])
    assert np.allclose(taller[:, 2], mesh.vertices[:, 2] + 0.1)
    with pytest.raises(ValueError, match="but the model has 2"):
        shaped_mesh.shaped([1.0, 0.0, 0.0])
    # Without shape directions, betas are ignored rather than silently misapplied.
    assert np.allclose(mesh.shaped([9.0]), mesh.vertices)


def test_mesh_rest_skeleton_feeds_the_rig(config):
    """The path a real model would take: payload -> mesh -> skeleton -> rig plan.

    The synthetic payload is too small to carry a plausible joint layout, so the joint
    centres are supplied as a real standing skeleton -- which is exactly what
    ``J_regressor @ v_template`` would produce for a real model file.
    """

    mesh = mesh_from_model_payload(_tiny_mesh_payload())
    real_joints = np.asarray(default_rest_skeleton().joint_positions, dtype=np.float64)
    with_joints = SmplMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        weights=mesh.weights,
        topology=mesh.topology,
        rest_joints=real_joints,
    )
    skeleton = with_joints.rest_skeleton()
    assert skeleton.topology.joint_count == 24
    assert skeleton.source.endswith("joint centres")
    assert np.allclose(np.asarray(skeleton.joint_positions), real_joints, atol=1e-12)
    # And it can drive the planner: the rig is planned from the mesh's skeleton, not
    # from the procedural stand-in.
    from dataclasses import replace

    from sim2sense_fall.humans.rig import plan_human_rig

    planned = plan_human_rig(replace(config, skeleton=replace(config.skeleton)), rest=skeleton)
    assert planned.skeleton_source.endswith("joint centres")
    assert len(planned.links) == 24


def test_joint_centres_from_regressor_is_linear():
    vertices = np.zeros((4, 3))
    vertices[:, 2] = [0.0, 0.5, 1.0, 1.5]
    regressor = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]])
    centres = joint_centres_from_regressor(regressor, vertices)
    assert np.allclose(centres[0], [0.0, 0.0, 0.0])
    assert np.allclose(centres[1], [0.0, 0.0, 0.5])
    assert np.allclose(centres[2], [0.0, 0.0, 0.75])
    with pytest.raises(ValueError, match="columns but the mesh has"):
        joint_centres_from_regressor(np.zeros((24, 9)), vertices)


# ---------------------------------------------------------------------------
# Python 2.7 pickles and chumpy wrappers: what the SMPL download page implies
# ---------------------------------------------------------------------------


def test_registry_matches_the_official_download_page():
    """Locks the registry to what the account's own download page and files show.

    Written after the 2026-09-22 fetch: the page lists 17 offers including the archive
    names, and the unpacked pickle settled the shape-PC count, the sparse J_regressor and
    the chumpy shapedirs. Assertions that used to encode "we have never seen the file" are
    now assertions about what was measured, so a future edit that quietly reverts to
    guesswork fails here.
    """

    registry = load_asset_registry(ASSETS_PATH, project_root=REPO_ROOT)
    versions = {model.version for model in registry.models.values()}
    assert versions == {"1.0.0", "1.1.0"}
    # v1.0.0 has no neutral body: that is the correction this test locks in.
    assert not any(
        "neutral" in model.id and model.version == "1.0.0" for model in registry.models.values()
    )
    v1_0_0 = [model for model in registry.models.values() if model.version == "1.0.0"]
    assert {model.id.split("_")[1] for model in v1_0_0} == {"male", "female"}
    assert {model.betas for model in v1_0_0} == {10}
    neutral = registry.model("smpl_neutral_v1_1_0")
    assert neutral.version == "1.1.0" and neutral.betas == 300
    # Both releases are Python 2.7 pickles, which is what forces the encoding fallback.
    assert {model.pickle_python_major for model in registry.models.values()} == {2}
    # The neutral v1.1.0 pickle has been opened and measured; nothing else here has.
    assert neutral.verified and "6890" in neutral.pickle_layout
    assert not registry.model("smpl_male_v1_1_0").verified
    # The page does publish the archive names now, and the measured pickle name is the
    # lower-case spelling, so the third-party capitalisation survives only as a candidate.
    assert neutral.filename == "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    assert "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl" in neutral.filename_candidates
    assert len(neutral.all_filenames) >= 3
    assert neutral.filename not in neutral.filename_candidates
    # The UV map is declared as an auxiliary asset with the same licence discipline.
    uv = registry.aux("smpl_uv_map_obj")
    assert uv.kind == "uv_map" and uv.requires_registration and uv.verified
    assert uv.filename == "smpl_uv.obj", "measured inside smpl_uv_20200910.zip"
    audit = audit_registry(registry)
    assert len(audit["models_detail"]) == len(registry.models)
    # The audit reports what is on disk. Since the 2026-09-22 fetch the UV file is present
    # in the asset root, so "absent" here would mean the licensed data went missing.
    assert audit["auxiliary"][0]["available"] is True
    assert audit["available_model_count"] >= 1
    assert len(candidate_paths(registry, "smpl_neutral_v1_1_0")) > len(registry.roots)


def test_python2_encoding_fallback_is_used(tmp_path):
    """A Python 2.7 string with a non-ASCII byte must load via latin1, not ASCII."""

    # Hand-built protocol-2 pickle for {"k": "ab<0xe9>"} using SHORT_BINSTRING, which is
    # what Python 2 wrote and what ASCII decoding rejects.
    raw = b"\x80\x02}U\x01kU\x04ab\xe9zs."
    path = tmp_path / "py2.pkl"
    path.write_bytes(raw)
    payload = load_model_payload(path, python_major=2)
    assert payload["k"] == "ab\u00e9z"
    # The same file still loads when the declaration says Python 3: both are tried,
    # because a wrong declaration must not look like a corrupt file.
    assert load_model_payload(path, python_major=3)["k"] == "ab\u00e9z"


def test_pickle_that_is_not_a_mapping_or_is_corrupt_fails_clearly(tmp_path):
    not_a_mapping = tmp_path / "list.pkl"
    import pickle as pickle_module

    not_a_mapping.write_bytes(pickle_module.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="could not be loaded as a pickle"):
        load_model_payload(not_a_mapping, python_major=2)

    truncated = tmp_path / "truncated.pkl"
    # Same pickle with the STOP opcode missing: the real failure mode of an interrupted
    # download, which must not be mistaken for a chumpy problem.
    truncated.write_bytes(b"\x80\x02}U\x01kU\x04ab\xe9")
    with pytest.raises(ValueError, match="could not be loaded as a pickle"):
        load_model_payload(truncated, python_major=3)


def test_chumpy_wrapped_arrays_are_unwrapped(tmp_path, fake_chumpy):
    """A chumpy-style wrapper carrying its array in ``.x`` must load, not error."""

    import pickle as pickle_module

    payload = body_payload()
    payload["shapedirs"] = fake_chumpy(x=payload["shapedirs"])
    payload["posedirs"] = [fake_chumpy(x=np.zeros((10, 3, 1))) for _ in range(2)]
    path = tmp_path / "chumpy.pkl"
    path.write_bytes(pickle_module.dumps(payload))
    loaded = load_model_payload(path, python_major=2)
    model = load_smpl_model(path)
    assert model.vertex_count == 10
    assert model.has_shape_directions
    # The chumpy-backed entries survive as real arrays once unwrapped.
    from sim2sense_fall.humans.assets import _as_float_array

    assert _as_float_array(loaded["shapedirs"], "shapedirs").shape == (10, 3, 4)


def test_chumpy_wrapper_without_a_recoverable_array_says_what_to_do(tmp_path, fake_chumpy):
    import pickle as pickle_module

    payload = {
        "v_template": np.zeros((5, 3)),
        "f": np.zeros((3, 3), dtype=int),
        "kintree_table": np.array([list(SMPL_KINEMATIC_PARENTS), list(range(24))], dtype=int),
        "J_regressor": fake_chumpy(shape=(24, 5)),
    }
    path = tmp_path / "opaque.pkl"
    path.write_bytes(pickle_module.dumps(payload))
    with pytest.raises(ValueError, match="convert the model once"):
        load_smpl_model(path)
    with pytest.raises(ValueError, match="chumpy"):
        load_smpl_model(path)


def test_model_file_found_under_an_alternate_candidate_name(tmp_path):
    """A naming difference must read as "alternate name", not as "asset missing"."""

    import pickle as pickle_module

    registry = registry_from_sequences(
        [
            {
                "id": "demo",
                "family": "smpl",
                "version": "1.1.0",
                "filename": "expected_name.pkl",
                "filename_candidates": ["actual_name.pkl"],
                "requires_registration": True,
                "registration_url": "https://example.invalid",
                "license": "research",
                "license_url": "https://example.invalid",
                "joint_count": 24,
                "betas": 10,
                "pose_parameters": 72,
                "pickle_layout": "v_template, f, kintree_table, J_regressor",
                "provenance": "test fixture",
                "verified": False,
                "pickle_python_major": 2,
            }
        ],
        [
            {
                "id": "amass",
                "dataset": "AMASS",
                "representation": "smplh_52",
                "pose_parameters": 156,
                "body_joints": 22,
                "betas": 16,
                "dmpls": 8,
                "subsets": ["CMU"],
                "requires_registration": True,
                "registration_url": "https://example.invalid",
                "license": "research",
                "license_url": "https://example.invalid",
                "provenance": "test fixture",
                "verified": False,
            }
        ],
        roots=[tmp_path],
    )
    assert audit_registry(registry)["available_model_count"] == 0
    (tmp_path / "smpl").mkdir()
    (tmp_path / "smpl" / "actual_name.pkl").write_bytes(pickle_module.dumps(body_payload()))
    audit = audit_registry(registry, hash_files=True)
    assert audit["available_model_count"] == 0, "the candidate lives in a subdirectory"
    registry = registry_from_sequences(
        [
            {
                "id": "demo",
                "family": "smpl",
                "version": "1.1.0",
                "filename": "expected_name.pkl",
                "filename_candidates": ["actual_name.pkl"],
                "subdirectory": "smpl",
                "requires_registration": True,
                "registration_url": "https://example.invalid",
                "license": "research",
                "license_url": "https://example.invalid",
                "joint_count": 24,
                "betas": 10,
                "pose_parameters": 72,
                "pickle_layout": "v_template, f, kintree_table, J_regressor",
                "provenance": "test fixture",
                "verified": False,
                "pickle_python_major": 2,
            }
        ],
        [
            {
                "id": "amass",
                "dataset": "AMASS",
                "representation": "smplh_52",
                "pose_parameters": 156,
                "body_joints": 22,
                "betas": 16,
                "dmpls": 8,
                "subsets": ["CMU"],
                "requires_registration": True,
                "registration_url": "https://example.invalid",
                "license": "research",
                "license_url": "https://example.invalid",
                "provenance": "test fixture",
                "verified": False,
            }
        ],
        roots=[tmp_path],
    )
    found = require_model(registry, "demo")
    assert found.name == "actual_name.pkl"
    model = load_smpl_model(found, registry=registry)
    assert model.declared_asset_id == "demo"
    assert model.pickle_python_major == 2


# ---------------------------------------------------------------------------
# R1: an asset is only imported when its CONTENT says so
# ---------------------------------------------------------------------------


def _registry_with_model_file(
    tmp_path, content: bytes, *, name: str = "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
):
    """A registry whose one model entry points at a file written with ``content``."""

    root = tmp_path / "humans"
    (root / "smpl").mkdir(parents=True, exist_ok=True)
    (root / "smpl" / name).write_bytes(content)
    entry = {
        "id": "smpl_neutral_v1_1_0",
        "family": "smpl",
        "version": "1.1.0",
        "filename": name,
        "subdirectory": "smpl",
        "requires_registration": True,
        "registration_url": "https://smpl.is.tue.mpg.de/download.php",
        "license": "non-commercial research",
        "license_url": "https://smpl.is.tue.mpg.de/modellicense.html",
        "joint_count": 24,
        "betas": 300,
        "pose_parameters": 72,
        "pickle_layout": "v_template, f, kintree_table, J_regressor, weights, shapedirs",
        "provenance": "test fixture",
        "verified": True,
    }
    return registry_from_sequences([entry], [], roots=[root])


def test_a_registered_path_is_not_an_imported_body(tmp_path):
    """The review's negative case: text masquerading as a SMPL model must not load."""

    registry = _registry_with_model_file(tmp_path, b"not a SMPL model\n")
    selection = select_body(registry, model_id="smpl_neutral_v1_1_0")
    assert selection.representation == "capsule_proxy_surface"
    assert selection.model is None
    assert selection.error and "UnpicklingError" in selection.error
    assert selection.as_dict()["body_representation"] == "capsule_proxy_surface"


def test_degrading_is_refused_when_the_configuration_says_so(tmp_path):
    """allow_procedural_skeleton: false has to be enforced, not merely recorded."""

    registry = _registry_with_model_file(tmp_path, b"not a SMPL model\n")
    strict = select_body(registry, model_id="smpl_neutral_v1_1_0", allow_procedural=False)
    lenient = select_body(registry, model_id="smpl_neutral_v1_1_0", allow_procedural=True)
    assert strict.error.endswith("(degrading is not allowed)")
    assert not lenient.error.endswith("(degrading is not allowed)")


def test_a_missing_model_names_the_registration_instead_of_passing(tmp_path):
    registry = _registry_with_model_file(tmp_path, b"")
    (registry.roots[0] / "smpl" / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl").unlink()
    selection = select_body(registry, model_id="smpl_neutral_v1_1_0", allow_procedural=False)
    assert selection.representation == "capsule_proxy_surface"
    assert "smpl.is.tue.mpg.de" in (selection.error or "")


def test_weights_that_do_not_sum_to_one_are_not_a_body(tmp_path):
    import pickle

    payload = body_payload()
    payload["weights"] = np.zeros((10, 24))
    registry = _registry_with_model_file(tmp_path, pickle.dumps(payload))
    selection = select_body(registry, model_id="smpl_neutral_v1_1_0")
    assert "do not sum to 1" in (selection.error or "")


def test_a_pre_rotated_template_is_refused_rather_than_guessed(tmp_path):
    import pickle

    payload = body_payload()
    diagonal = np.array([[0.8, 0.6, 0.0], [0.0, 0.0, 1.0], [-0.6, 0.8, 0.0]])
    payload["v_template"] = payload["v_template"] @ diagonal.T
    registry = _registry_with_model_file(tmp_path, pickle.dumps(payload))
    selection = select_body(registry, model_id="smpl_neutral_v1_1_0")
    assert (
        "not axis-aligned" in (selection.error or "")
        or "stature" in (selection.error or "").lower()
    )


def test_an_upside_down_template_is_not_silently_flipped(tmp_path):
    """A body whose torso and thighs disagree about which way is up must be refused.

    Flipping the whole template is not enough to make the loader's check fire on its
    own -- a uniformly flipped body is internally consistent and would import as a
    perfectly valid body that happens to be the wrong way up in the file. The fixture
    therefore fixes one limb only, so the torso says one thing and the thigh says
    another, which is the inconsistency the loader refuses to guess about.
    """

    import pickle

    payload = body_payload()
    # Move the foot joints above the pelvis. The torso still points +Y and the thigh
    # still points -Y, but the hip-to-knee check and the foot-height check now disagree
    # about which way is up -- which is exactly the inconsistency the loader refuses.
    payload["v_template"][8, 1] = 0.94
    payload["v_template"][9, 1] = 0.94
    registry = _registry_with_model_file(tmp_path, pickle.dumps(payload))
    selection = select_body(registry, model_id="smpl_neutral_v1_1_0")
    # Rejected on the upright-body check; the exact message is the guard's business, the
    # requirement is that a foot floating above the pelvis is never accepted as a body.
    assert "not upright" in (selection.error or "")
    assert selection.representation == "capsule_proxy_surface"


def test_a_valid_fixture_body_is_reported_as_a_skin_mesh(tmp_path):
    import pickle

    registry = _registry_with_model_file(tmp_path, pickle.dumps(body_payload()))
    selection = select_body(registry, model_id="smpl_neutral_v1_1_0", allow_procedural=False)
    assert selection.representation == "smpl_skin_mesh"
    assert selection.has_skin_mesh and selection.model is not None
    # foot at y=-0.94, crown at y=+0.60 -> 1.54 m along the up axis.
    assert selection.model.stature_m == pytest.approx(1.54)
    assert selection.model.beta_count == 4


def test_representations_match_the_export_schema():
    from sim2sense_fall.humans.assets import REPRESENTATION_PROXY, REPRESENTATION_SKIN_MESH

    assert REPRESENTATION_SKIN_MESH == "smpl_skin_mesh"
    assert REPRESENTATION_PROXY == "capsule_proxy_surface"


# ---------------------------------------------------------------------------
# R3: impact must be one point moving down into the surface
# ---------------------------------------------------------------------------


def _impact_trajectory(body_points: np.ndarray, pelvis_z: np.ndarray, *, fps: float = 120.0):
    """A minimal trajectory whose only geometry is ``body_points``."""

    frames = body_points.shape[0]
    times = np.arange(frames, dtype=np.float64) / fps
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1))
    return Trajectory(
        times_s=times,
        root_position=np.stack([np.zeros(frames), np.zeros(frames), pelvis_z], axis=1),
        root_quaternion=identity,
        joint_positions=np.zeros((frames, 2)),
        joint_names=("a", "b"),
        body_points=body_points,
        standing_height_m=1.7,
        standing_pelvis_height_m=float(pelvis_z[0]),
    )


def test_a_near_floor_creep_with_a_fast_swing_elsewhere_is_not_an_impact(config):
    """The review's counterexample: low point creeping down, high point swinging fast.

    The lowest body point sits at 0.1 m and descends at 0.001 m/s while a different point
    at 0.5 m sweeps sideways at 2 m/s. No point is both near the surface and moving into
    it, so no impact may be reported -- and therefore no fall.
    """

    frames = 240
    times = np.arange(frames, dtype=np.float64) / 120.0
    points = np.zeros((frames, 2, 3))
    points[:, 0, 2] = 0.1 - 0.001 * times  # creeping down, never reaching the floor
    # The swinging point stays high and moves only along X: 2 m/s of horizontal speed.
    points[:, 1, 0] = 2.0 * times
    points[:, 1, 2] = 0.5
    pelvis = np.full(frames, 0.4)
    trajectory = _impact_trajectory(points, pelvis)
    label = label_trial(trajectory, config.events)
    assert label.first_impact_s is None, "a mixed-point speed must not create an impact"
    assert label.label != LABEL_FALL


def test_a_point_that_starts_on_the_floor_produces_no_impact(config):
    """A body already lying down at frame zero has no impact to report.

    The old rule needed only "near the floor", "descending" and "fast somewhere", which a
    settled body satisfies in its first frame if any point twitches.
    """

    frames = 120
    points = np.zeros((frames, 2, 3))
    points[:, 0, 2] = 0.02
    points[:, 1, 2] = 0.05
    # One point dips from 0.05 m to the floor quickly: an approach from above, which
    # must still be excluded because the OTHER point was near the surface from frame zero
    # and the pelvis never falls, so no postural disturbance precedes it.
    points[60:, 1, 2] = 0.001
    trajectory = _impact_trajectory(points, np.full(frames, 0.05))
    label = label_trial(trajectory, config.events)
    assert label.label != LABEL_FALL


def test_a_real_tipping_fall_still_reports_impact_and_its_point(config):
    """The fix must not blunt detection: a body that tips does impact, with evidence."""

    # The configured impact gate is 1.0 m/s within 0.25 m of the surface, so the head
    # falls 1.58 m in one second: a slower fixture would legitimately not be an impact.
    frames = 120
    progress = np.linspace(0.0, 1.0, frames)
    points = np.zeros((frames, 2, 3))
    points[:, 0, 2] = 1.60 - 1.58 * progress  # head, 1.60 m down to the floor
    points[:, 1, 2] = np.maximum(0.05, 1.40 - 1.35 * progress)
    pelvis = 1.20 - 0.70 * progress
    label = label_trial(_impact_trajectory(points, pelvis), config.events)
    assert label.first_impact_s is not None
    metrics = dict(label.metrics)
    assert metrics["impact_body_point_index"] == 0
    assert 0.0 <= metrics["impact_point_height_m"] <= config.events.impact_height_m
    assert metrics["impact_point_down_speed_m_s"] >= config.events.impact_speed_m_s
    assert metrics["contact_evidence"] == "proxy_only", "no force channel was recorded"


# ---------------------------------------------------------------------------
# R1: the exported geometry has to come from the skin that was imported
# ---------------------------------------------------------------------------


def test_sample_skin_points_follows_the_link_poses_and_reports_owners():
    """Skinned points must move with the body, and each point must say which joint owns it."""

    import numpy as np

    from sim2sense_fall.humans.skeleton import smpl_skeleton
    from sim2sense_fall.humans.skinning import SmplMesh, sample_skin_points

    topology = smpl_skeleton()
    vertices = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.5], [0.3, 0.1, 0.9]])
    weights = np.zeros((4, 24))
    weights[0, topology.joint_names.index("head")] = 1.0
    for row in (1, 2, 3):
        weights[row, 0] = 1.0
    mesh = SmplMesh(
        vertices=vertices,
        faces=np.array([[0, 1, 2], [1, 2, 3]]),
        weights=weights,
        topology=topology,
        rest_joints=np.tile(np.array([0.0, 0.0, 0.9]), (24, 1)),
    )
    mesh.rest_joints[15] = (0.0, 0.0, 1.6)
    rest = {
        name: (np.eye(3), mesh.rest_joints[index])
        for index, name in enumerate(topology.joint_names)
    }
    points, owners = sample_skin_points(mesh, rest, count=4, seed=7)
    assert points.shape == (4, 3)
    assert set(owners) <= set(topology.joint_names)
    # Lift the head link by half a metre: only the head-owned point may follow.
    moved = dict(rest)
    moved["head"] = (np.eye(3), mesh.rest_joints[15] + np.array([0.0, 0.0, 0.5]))
    lifted, lifted_owners = sample_skin_points(mesh, moved, count=4, seed=7)
    delta = lifted - points
    head_rows = [i for i, name in enumerate(lifted_owners) if name == "head"]
    assert head_rows, "the fixture has a head-weighted vertex"
    assert np.allclose(delta[head_rows], [0.0, 0.0, 0.5]), "head point must follow the head link"
    other = [i for i in range(4) if i not in head_rows]
    assert np.allclose(delta[other], 0.0), "pelvis-weighted points must not move with the head"
    with pytest.raises(ValueError, match="count must be positive"):
        sample_skin_points(mesh, rest, count=0, seed=1)


# ---------------------------------------------------------------------------
# contact channel
# ---------------------------------------------------------------------------


def test_contact_sample_geometry_and_impulse():
    """A contact sample is a point, not a force: position, normal and impulse."""

    from sim2sense_fall.humans.usd_human import ContactSample

    sample = ContactSample(
        collider0="/World/Human/left_ankle",
        collider1="/World/site/foundation",
        position_m=(1.0, 2.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        impulse_ns=(3.0, 4.0, 0.0),
        separation_m=-1e-4,
    )
    assert sample.impulse_magnitude_ns == pytest.approx(5.0)
    payload = sample.as_dict()
    assert payload["impulse_magnitude_ns"] == pytest.approx(5.0)
    assert payload["position_m"] == [1.0, 2.0, 0.0]


def test_default_contact_categories_include_walls_and_furniture():
    """A topple into a wall is a contact; restricting to the floor dropped it."""

    from sim2sense_fall.humans.usd_human import DEFAULT_CONTACT_CATEGORIES

    for category in ("floor", "wall", "furniture"):
        assert category in DEFAULT_CONTACT_CATEGORIES


def test_contact_rows_names_the_limb_and_flags_support():
    """The pair cannot be split by handle, so the row carries the limb and support flag.

    This build tags contact reporting per ACTOR and exposes only opaque numeric
    collider handles, so ``handle0``/``handle1`` are a fingerprint at best. What the
    export *can* state is which body capsule the point landed in and whether the
    point was resting on a surface -- both derived from geometry, not from the pair.
    """

    import importlib.util
    from pathlib import Path

    from sim2sense_fall.humans.usd_human import ContactSample

    spec = importlib.util.spec_from_file_location(
        "simulate_module",
        Path(__file__).resolve().parents[2] / "scripts" / "humans" / "simulate.py",
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - depends on script import side effects
        pytest.skip(f"simulate.py is not importable in isolation: {exc}")

    on_floor = ContactSample(
        collider0=4141,
        collider1=9001,
        position_m=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        impulse_ns=(0.0, 0.0, 0.0),
        separation_m=0.0,
    )
    # Same pair, but the point sits well above the support surface and its normal is
    # horizontal: a wall touch, not a rest.
    on_wall = ContactSample(
        collider0=9001,
        collider1=4142,
        position_m=(1.0, 0.0, 0.8),
        normal=(1.0, 0.0, 0.0),
        impulse_ns=(0.0, 0.0, 0.0),
        separation_m=0.0,
    )
    rows = module._contact_rows(
        [(), (on_floor,), (on_wall,)],
        segments=[(), ("left_ankle",), ("left_ankle",)],
        support_z=0.0,
    )
    assert list(rows["frame"]) == [1, 2]
    assert rows["position"].shape == (2, 3)
    # Handles survive verbatim, in reported order -- deliberately NOT interpreted.
    assert rows["handle0"][0] == 4141
    assert rows["handle1"][0] == 9001
    assert rows["handle0"][1] == 9001
    assert rows["handle1"][1] == 4142
    # The named limb is what a fall label can be argued from.
    assert list(rows["segment"]) == ["left_ankle", "left_ankle"]
    # Only the upward-normal point at floor height counts as support.
    assert list(rows["is_support"]) == [True, False]


def test_contact_rows_keeps_columns_when_there_are_no_contacts():
    """An empty report must not produce a zero-length column set a reader trips on."""

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "simulate_module_empty",
        Path(__file__).resolve().parents[2] / "scripts" / "humans" / "simulate.py",
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - depends on script import side effects
        pytest.skip(f"simulate.py is not importable in isolation: {exc}")

    for payload in (None, [(), ()]):
        rows = module._contact_rows(payload)
        assert rows["frame"].shape == (0,)
        assert rows["position"].shape == (0, 3)
        assert rows["impulse"].shape == (0, 3)
        assert rows["separation"].shape == (0,)


def test_a_missing_contact_channel_raises_instead_of_reporting_no_contacts():
    """ "The channel is absent" must never masquerade as "nothing was touched"."""

    from sim2sense_fall.humans.usd_human import ContactSourceUnavailable, HumanRuntime

    class Bare(HumanRuntime):
        def __init__(self) -> None:  # noqa: D107 - deliberately bypasses the runtime
            self.capabilities = {"contact_source": "unavailable", "contact_error": None}
            self.root_path = "/World/Human"
            self._contact_interface = None

    runtime = Bare()
    with pytest.raises(ContactSourceUnavailable):
        runtime.contact_samples()


def test_point_in_capsule_respects_the_axis_and_the_tolerance() -> None:
    """Containment is what attributes a contact pair to a limb, so it must be exact.

    A capsule at the origin lying along world +X, radius 0.1, half-height 0.3:
    points inside the swept segment are contained; points off the axis are contained
    only within the radius; points beyond the caps are not.
    """

    from sim2sense_fall.humans.usd_human import (
        CAPSULE_CONTAINMENT_TOLERANCE_M,
        _point_in_capsule,
    )

    centre = (0.0, 0.0, 0.0)
    axis_half = (0.3, 0.0, 0.0)
    radius = 0.1

    # On the axis, mid-segment and at each cap centre.
    assert _point_in_capsule((0.0, 0.0, 0.0), centre, radius, axis_half)
    assert _point_in_capsule((0.3, 0.0, 0.0), centre, radius, axis_half)
    assert _point_in_capsule((-0.3, 0.0, 0.0), centre, radius, axis_half)
    # Inside the radius, just past the segment end (within the spherical cap).
    assert _point_in_capsule((0.35, 0.0, 0.0), centre, radius, axis_half)
    # Perpendicular offset inside the radius.
    assert _point_in_capsule((0.0, 0.09, 0.0), centre, radius, axis_half)

    # Beyond the cap: 0.3 + 0.1 = 0.4 is the surface, so 0.45 is outside even with
    # the containment tolerance.
    assert not _point_in_capsule((0.45, 0.0, 0.0), centre, radius, axis_half)
    # Perpendicular offset well outside the radius.
    assert not _point_in_capsule((0.0, 0.3, 0.0), centre, radius, axis_half)

    # The tolerance is exactly the shell that PhysX reports contact points on.
    shell = 0.4 + CAPSULE_CONTAINMENT_TOLERANCE_M / 2.0
    assert _point_in_capsule((shell, 0.0, 0.0), centre, radius, axis_half)

    # A degenerate capsule (zero half-height) behaves as a sphere of the same radius.
    assert _point_in_capsule((0.0, 0.0, 0.0), centre, radius, (0.0, 0.0, 0.0))
    assert not _point_in_capsule((0.0, 0.0, 0.3), centre, radius, (0.0, 0.0, 0.0))


def test_point_in_capsule_is_orientation_aware() -> None:
    """The same world point must be inside or outside depending on the capsule's tilt."""

    from sim2sense_fall.humans.usd_human import _point_in_capsule

    centre = (0.0, 0.0, 0.0)
    radius = 0.05
    point = (0.0, 0.0, 0.25)

    # Capsule lying along X: the point is far off the axis, so outside.
    assert not _point_in_capsule(point, centre, radius, (0.3, 0.0, 0.0))
    # Capsule standing along Z: the point sits inside the swept segment.
    assert _point_in_capsule(point, centre, radius, (0.0, 0.0, 0.3))


def test_segment_in_volumes_names_the_containing_limb() -> None:
    """Attribution returns the limb whose capsule holds the point, or nothing.

    This is the whole basis for naming a contact's body segment, so the ambiguous
    cases matter as much as the happy path: a point in no capsule must return
    ``None`` rather than the nearest guess, because "unattributed" and "attributed to
    the wrong limb" are different failures for a fall label.
    """

    from sim2sense_fall.humans.usd_human import _segment_in_volumes

    # left_ankle: a short capsule standing at (0, 0, 0.1). right_knee: far away.
    volumes = {
        "left_ankle": ((0.0, 0.0, 0.1), 0.05, (0.0, 0.0, 0.03)),
        "right_knee": ((2.0, 0.0, 0.5), 0.06, (0.0, 0.0, 0.05)),
    }
    assert _segment_in_volumes((0.0, 0.0, 0.1), volumes) == "left_ankle"
    assert _segment_in_volumes((0.0, 0.02, 0.12), volumes) == "left_ankle"
    assert _segment_in_volumes((2.0, 0.0, 0.5), volumes) == "right_knee"
    # Between the two limbs, inside neither: must not be guessed.
    assert _segment_in_volumes((1.0, 0.0, 0.3), volumes) is None
    assert _segment_in_volumes((0.0, 0.0, 5.0), volumes) is None
    # An empty body has no limbs to attribute to.
    assert _segment_in_volumes((0.0, 0.0, 0.0), {}) is None


def test_capsule_world_volume_tracks_the_link_pose() -> None:
    """Composed volumes must move and rotate with the link they hang from.

    This composition is what keeps contact volumes on the body once USD-stage
    world reads go stale (the measured frame-2298 attribution escape), so both
    the placement and the orientation of the offset matter: a convention mixup
    would hang every capsule at a mirrored or swapped position and silently
    misattribute contacts. The last block closes the loop with the rig's own
    containment rule -- containment must be pose-invariant.
    """

    from sim2sense_fall.humans.rotations import quaternion_to_matrix
    from sim2sense_fall.humans.usd_human import (
        _capsule_world_volume,
        _point_in_capsule,
    )

    geometry = ((0.1, -0.2, 0.3), 0.05, (0.0, 0.0, 0.25))

    # Identity pose: the world volume is the authored offset translated as-is.
    centre, radius, axis_half = _capsule_world_volume(
        np.eye(3), np.array([1.0, 2.0, 3.0]), geometry
    )
    assert centre == pytest.approx((1.1, 1.8, 3.3))
    assert radius == pytest.approx(0.05)
    assert axis_half == pytest.approx((0.0, 0.0, 0.25))

    # A 90-degree yaw carries the offset with the link: local +X becomes world +Y.
    yaw_quarter = quaternion_to_matrix(
        (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    )
    centre, _, _ = _capsule_world_volume(
        yaw_quarter, np.zeros(3), ((0.4, 0.0, 0.0), 0.05, (0.0, 0.0, 0.25))
    )
    assert centre == pytest.approx((0.0, 0.4, 0.0))

    # A +90-degree rotation about Y maps local +Z onto world +X: the capsule's
    # half-axis vector must rotate with it, not stay world-upright.
    tilt = quaternion_to_matrix((math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0))
    _, _, axis_half = _capsule_world_volume(
        tilt, np.array([0.0, 0.0, 0.5]), ((0.0, 0.0, 0.0), 0.05, (0.0, 0.0, 0.25))
    )
    assert axis_half[0] == pytest.approx(0.25)
    assert axis_half[2] == pytest.approx(0.0, abs=1e-12)

    # Round trip with an arbitrary unit quaternion pose: a point inside the
    # authored capsule stays inside the composed world volume, and a far point
    # stays outside, under the same containment rule attribution uses.
    arbitrary_rotation = quaternion_to_matrix((0.8, 0.2, -0.4, 0.4))  # already unit
    translation = np.array([12.5, -3.0, 0.4])
    centre, radius, axis_half = _capsule_world_volume(
        arbitrary_rotation, translation, geometry
    )
    inside = tuple(
        arbitrary_rotation @ np.asarray((0.12, -0.21, 0.40), dtype=np.float64) + translation
    )
    assert _point_in_capsule(inside, centre, radius, axis_half)
    outside = tuple(
        arbitrary_rotation @ np.asarray((2.0, -0.2, 0.3), dtype=np.float64) + translation
    )
    assert not _point_in_capsule(outside, centre, radius, axis_half)


def test_world_box_max_z_maps_every_corner_not_just_the_local_maximum() -> None:
    """The world top is a corner maximum, not the transformed local maximum.

    Mapping only the local ``max`` corner is the obvious shortcut and it is wrong
    as soon as the transform carries a sign flip, which a mirrored or rotated
    parent transform may. This runs on fakes so the arithmetic is exercised on the
    CPU rather than only inside the Kit runtime.
    """

    from sim2sense_fall.humans.usd_human import _world_box_max_z

    class _Vec:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z

        def __getitem__(self, index: int) -> float:
            return (self.x, self.y, self.z)[index]

    class _Range:
        def __init__(self, lo: tuple[float, float, float], hi: tuple[float, float, float]) -> None:
            self._lo, self._hi = _Vec(*lo), _Vec(*hi)

        def GetMin(self) -> _Vec:
            return self._lo

        def GetMax(self) -> _Vec:
            return self._hi

    class _Matrix:
        """Only the three ops the helper uses: rotate, translate, apply."""

        def __init__(self, flip_x: bool = False, dz: float = 0.0) -> None:
            self.flip_x, self.dz = flip_x, dz

        def Transform(self, point: _Vec) -> _Vec:
            x = -point.x if self.flip_x else point.x
            return _Vec(x, point.y, point.z + self.dz)

    class _Gf:
        Vec3d = _Vec

    @dataclasses.dataclass(frozen=True, slots=True)
    class _Runtime:
        Gf: object

    runtime = _Runtime(Gf=_Gf)
    box = _Range((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))

    assert _world_box_max_z(box, _Matrix(), runtime) == pytest.approx(3.0)
    # A mirror in x does not move the top here, but it must not be assumed away.
    assert _world_box_max_z(box, _Matrix(flip_x=True), runtime) == pytest.approx(3.0)
    # A translate must shift the reading by exactly that amount.
    assert _world_box_max_z(box, _Matrix(dz=-0.06), runtime) == pytest.approx(2.94)

    # Now an asymmetric box where the local max corner is NOT the world max corner
    # once the transform negates z: the world top must come from the local min.
    flipped_z = _Range((-1.0, -2.0, -3.0), (1.0, 2.0, 1.0))

    class _FlipZ(_Matrix):
        def Transform(self, point: _Vec) -> _Vec:
            return _Vec(point.x, point.y, -point.z)

    assert _world_box_max_z(flipped_z, _FlipZ(), runtime) == pytest.approx(3.0)
    # The local max corner is z=+1, which maps to -1: reading only it would say 1.0.
    assert _world_box_max_z(flipped_z, _FlipZ(), runtime) != pytest.approx(1.0)


def test_prim_local_box_is_centred_for_cube_and_cylinder() -> None:
    """The geometry box is centred; the bbox cache in this build is not.

    ``ComputeLocalBound`` returns an already-scaled box that is still offset by the
    prim translate, which double-counts the translate once it is composed. Deriving
    the box from the authored ``size``/``radius``/``height`` attributes keeps the
    geometry centred, which is what makes the composition above valid. Measured on
    Isaac Sim 6.0.1, the room floor (``translate z = -0.06``, ``scale z = 0.12``)
    reports ``z in [-0.12, 0.0]`` from the cache instead of ``[-0.06, +0.06]``.
    """

    from sim2sense_fall.humans.usd_human import _prim_local_box

    class _Vec:
        def __init__(self, x: float, y: float, z: float) -> None:
            self._v = (x, y, z)

        def __getitem__(self, index: int) -> float:
            return self._v[index]

    class _Range:
        def __init__(self, lo: _Vec, hi: _Vec) -> None:
            self._lo, self._hi = lo, hi

        def GetMin(self) -> _Vec:
            return self._lo

        def GetMax(self) -> _Vec:
            return self._hi

    class _Gf:
        Vec3d = _Vec
        Range3d = _Range

    @dataclasses.dataclass(frozen=True, slots=True)
    class _Runtime:
        Gf: object

    class _Attribute:
        def __init__(self, value: object) -> None:
            self._value = value

        def IsValid(self) -> bool:
            return True

        def Get(self) -> object:
            return self._value

    class _Prim:
        def __init__(self, type_name: str, attributes: dict[str, object]) -> None:
            self._type_name = type_name
            self._attributes = attributes

        def GetTypeName(self) -> str:
            return self._type_name

        def GetAttribute(self, name: str) -> _Attribute:
            return _Attribute(self._attributes.get(name))

    runtime = _Runtime(Gf=_Gf)

    box = _prim_local_box(_Prim("Cube", {"size": 1.0}), runtime)
    assert tuple(box.GetMin())[:] == pytest.approx((-0.5, -0.5, -0.5))
    assert tuple(box.GetMax())[:] == pytest.approx((0.5, 0.5, 0.5))

    # The size attribute scales the geometry; the prim's xform scale is separate.
    box = _prim_local_box(_Prim("Cube", {"size": 2.0}), runtime)
    assert tuple(box.GetMin())[:] == pytest.approx((-1.0, -1.0, -1.0))
    assert tuple(box.GetMax())[:] == pytest.approx((1.0, 1.0, 1.0))

    box = _prim_local_box(_Prim("Cylinder", {"radius": 0.25, "height": 2.0}), runtime)
    assert tuple(box.GetMin())[:] == pytest.approx((-0.25, -0.25, -1.0))
    assert tuple(box.GetMax())[:] == pytest.approx((0.25, 0.25, 1.0))

    # An unknown primitive type has no derivable box and must say so rather than
    # pretend a unit cube.
    assert _prim_local_box(_Prim("Sphere", {}), runtime) is None
