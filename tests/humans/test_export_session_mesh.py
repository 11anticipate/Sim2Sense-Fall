"""Session mesh export on a synthetic physics session, plus label gates.

The exporter is the hand-off between the keyboard data pipeline and the Sionna
channel stage, so the tests pin the contract the importer enforces (uniform
time, finite geometry, trajectory-derived labels) and the fall-coverage check
that refuses a manifest when a session impact has no fall-labelled segment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "humans"))

from export_session_mesh import (  # noqa: E402
    SCHEMA_VERSION,
    export_session,
    label_segment,
    segments_from_control,
    trunk_angle_deg,
    uniform_time,
)


def _session(tmp_path: Path, *, impact_time_s: float | None = 4.35, modes_override=None):
    """A synthetic 6 s session: 3 s walk, 3 s falling/fallen, one body impact."""

    rng = np.random.default_rng(7)
    dt = 1.0 / 30.0
    time_s = np.arange(180) * dt
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 3], [4, 5, 1]])
    base = np.linspace(0.0, 1.0, 6).reshape(6, 1) * np.array([[1.0, 0.0, 0.0]])
    vertices = np.stack(
        [
            base + np.array([0.0, 0.0, 0.9 - 0.4 * (t > 4.0)]) + rng.normal(0, 1e-6, (6, 3))
            for t in time_s
        ]
    )
    modes = modes_override if modes_override is not None else np.array(
        ["forward"] * 90 + ["falling"] * 45 + ["fallen"] * 45
    )
    roots = np.zeros((180, 3))
    roots[:, 2] = np.where(time_s > 4.0, 0.35, 0.95)
    roots[:90, 0] = 0.4 * time_s[:90]
    control_time = np.arange(720) / 120.0
    control_modes = np.repeat(modes, 4)
    control_roots = np.repeat(roots, 4, axis=0)
    control_roots[:, 0] += np.linspace(0, 0.01, 720)
    quaternions = np.tile(np.array([1., 0., 0., 0.]), (720, 1))
    quaternions[control_time > 4.0] = [np.cos(np.pi/4), 0., np.sin(np.pi/4), 0.]
    fall_events = [
        {"requested_time_s": 3.0, "impact_time_s": impact_time_s, "outcome": "fallen"}
    ]
    run = tmp_path / "run"
    run.mkdir()
    np.savez(
        run / "recording.npz",
        time_s=time_s,
        mesh_vertices_xyz=vertices,
        mesh_faces=faces,
    )
    np.savez(
        run / "control.npz",
        time_s=control_time,
        mode=control_modes,
        root=control_roots,
        root_quaternion=quaternions,
    )
    report = {
        "runtime_completed": True, "motion_accuracy_accepted": True, "errors": [],
        "quality_gates": {"complete_recording_window": True},
        "motion_quality": {"schema_version": SCHEMA_VERSION, "accepted": True,
                           "modes": {"forward": {"accepted": True}}},
        "actions": {"fallen_tilt_deg": 50., "fallen_height_fraction": .6},
        "fall": {"events": fall_events},
        "keyboard_sha256": "abc",
        "simulation_time_s": 6.0,
        "seed": 0,
        "scene_sha256": "s",
        "rig_sha256": "r",
    }
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run


def test_segments_split_on_mode(tmp_path):
    run = _session(tmp_path)
    control = dict(np.load(run / "control.npz", allow_pickle=False))
    recording = np.load(run / "recording.npz", allow_pickle=False)
    segments = segments_from_control(control, np.asarray(recording["time_s"]))
    assert [s["mode"] for s in segments] == ["forward", "falling"]


def test_export_labels_segments_from_trajectory(tmp_path):
    run = _session(tmp_path)
    out = tmp_path / "export"
    manifest = export_session(run, out)
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    assert by_id["forward_00"]["label"]["label"] == "walk"
    assert by_id["falling_00"]["label"]["label"] == "fall"
    assert by_id["falling_00"]["label"]["first_impact_s"] == pytest.approx(4.35)
    assert manifest["fidelity"] == "physics_keyboard_session"
    for sample in manifest["samples"]:
        data = np.load(out / f"{sample['sample_id']}.mesh.npz", allow_pickle=False)
        diffs = np.diff(data["time_s"])
        assert (diffs > 0).all()
        assert np.isfinite(data["mesh_vertices_xyz"]).all()
        assert data["mesh_faces"].shape == (4, 3)


def test_export_refuses_impact_without_fall_segment(tmp_path):
    # A session that records a body-floor impact while its mode timeline never
    # leaves 'forward' contradicts itself; the manifest must not be written.
    run = _session(tmp_path, modes_override=np.array(["forward"] * 180))
    with pytest.raises(ValueError, match="contradict each other"):
        export_session(run, tmp_path / "export")


def test_stale_measurement_schema_is_refused(tmp_path):
    """A report from a different quality schema cannot be admitted.

    The batch died on exactly this contract: the exporter pinned schema_version == 1
    while the quality module had moved on, and the fixture repeated the same literal,
    so the drift only showed up on a real session. Both sides now import one constant,
    and this test pins the refusal direction.
    """

    run = _session(tmp_path)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    report["motion_quality"]["schema_version"] = SCHEMA_VERSION - 1
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement_schema"):
        export_session(run, tmp_path / "export")
    assert not (tmp_path / "export" / "manifest.json").exists()


def test_uniform_time_resamples_jitter():
    time_s = np.array([0.0, 0.0333, 0.0667, 0.1167])  # one dropped frame
    vertices = np.zeros((4, 2, 3))
    out_time, out_vertices, resampled = uniform_time(time_s, vertices)
    assert resampled
    assert np.allclose(np.diff(out_time), out_time[1] - out_time[0])
    assert len(out_vertices) == len(out_time)


def test_uniform_time_keeps_clean_axis():
    time_s = np.arange(10) * (1.0 / 30.0)
    vertices = np.zeros((10, 2, 3))
    out_time, _vertices, resampled = uniform_time(time_s, vertices)
    assert not resampled
    assert np.allclose(out_time, time_s)


def test_trunk_angle_upright_vs_lying():
    assert trunk_angle_deg(np.array([1.0, 0, 0, 0])) == pytest.approx(0.0, abs=1e-6)
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    assert trunk_angle_deg(q) == pytest.approx(90.0, abs=1e-6)


def test_trunk_angle_ignores_pure_yaw():
    # The shipped session spawns at heading 90 deg: quaternion (0.707, 0, 0, 0.707).
    # Reading the wrong matrix element (1-2*(y^2+z^2) instead of R22) turns this
    # upright standing pose into a 90 deg lie and poisoned every segment label.
    heading_90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert trunk_angle_deg(heading_90) == pytest.approx(0.0, abs=1e-6)


def test_label_segment_requires_session_evidence():
    vertices = np.zeros((10, 2, 3))
    segment = {"time_s": np.arange(10) * 0.1, "mode": "stand"}
    label = label_segment(segment, vertices, [])
    assert label["label"] == "stand" and label["valid"]


def test_unverified_falling_mode_is_unknown():
    segment = {"time_s": np.arange(10)*.1, "mode": "falling"}
    label = label_segment(segment, np.zeros((10, 2, 3)), [])
    assert label["label"] == "unknown" and not label["valid"]


def test_measurement_machinery_failure_refused_before_output(tmp_path):
    run = _session(tmp_path)
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["runtime_completed"] = False
    path.write_text(json.dumps(report))
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="measurement machinery"):
        export_session(run, out)
    assert not out.exists()


def test_segment_level_admission_isolates_failed_activities(tmp_path):
    """A failed activity must not blacklist the session's healthy segments.

    The pre-segment-admission semantics required the whole session to pass its
    aggregate motion verdict, so one failing activity (crouch-hold skin gap,
    a reversal transient) made every walk/stand segment inadmissible. The
    per-segment gate judges each segment by its own activity's measured entry;
    fall episodes stay gated by their event verification.
    """

    run = _session(tmp_path)
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["motion_accuracy_accepted"] = False
    report["motion_quality"]["accepted"] = False
    report["motion_quality"]["modes"] = {"forward": {"accepted": True}}
    path.write_text(json.dumps(report))
    manifest = export_session(run, tmp_path / "export")
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    assert by_id["forward_00"]["admitted_for_training"] is True
    assert by_id["falling_00"]["admitted_for_training"] is True
    assert manifest["session_invariants_ok"] is True
    assert manifest["admitted_for_training"] is True


def test_failed_activity_gate_blocks_only_its_own_segment(tmp_path):
    run = _session(tmp_path)
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["motion_quality"]["modes"] = {"forward": {"accepted": False}}
    path.write_text(json.dumps(report))
    manifest = export_session(run, tmp_path / "export")
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    assert by_id["forward_00"]["admitted_for_training"] is False
    assert by_id["forward_00"]["admission"]["activity_gates_accepted"] is False
    assert by_id["falling_00"]["admitted_for_training"] is True
    assert manifest["admitted_for_training"] is False
    assert manifest["session_invariants_ok"] is True


def test_impact_without_fallen_posture_refused(tmp_path):
    run = _session(tmp_path)
    control = dict(np.load(run / "control.npz"))
    control["root_quaternion"][:] = [1., 0., 0., 0.]
    np.savez(run / "control.npz", **control)
    with pytest.raises(ValueError, match="unverified"):
        export_session(run, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_truncated_recording_is_not_admitted(tmp_path):
    run = _session(tmp_path)
    path = run / "report.json"
    report = json.loads(path.read_text())
    report["quality_gates"]["complete_recording_window"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="measurement machinery"):
        export_session(run, tmp_path / "export")


def test_reset_between_mesh_frames_splits_same_mode():
    times = np.arange(120)/120
    roots = np.zeros((120, 3))
    roots[61:, 0] = 1.0  # 30Hz mesh samples indices 60 and 64, never reset frame 61.
    control = {"time_s": times, "root": roots, "mode": np.full(120, "stand")}
    segments = segments_from_control(control, times[::4])
    assert len(segments) == 2
    assert segments[0]["stop"] == segments[1]["start"]


def test_in_place_reset_id_is_also_a_boundary():
    times = np.arange(120)/120
    reset = np.zeros(120, dtype=int)
    reset[61:] = 1
    control = {"time_s": times, "root": np.zeros((120, 3)),
               "mode": np.full(120, "stand"), "reset_id": reset}
    assert len(segments_from_control(control, times[::4])) == 2


def test_physics_capture_downsamples_to_50hz_without_extrapolation():
    time = np.arange(120) / 120
    vertices = np.broadcast_to(time[:, None, None], (120, 2, 3)).copy()
    sampled_time, sampled, changed = uniform_time(time, vertices, target_hz=50.)
    assert changed and sampled_time[-1] <= time[-1]
    np.testing.assert_allclose(np.diff(sampled_time), .02)
    np.testing.assert_allclose(sampled[:, 0, 0], sampled_time)


def test_render_capture_cannot_claim_50hz():
    time = np.arange(30) / 30
    with pytest.raises(ValueError, match="capture cadence"):
        uniform_time(time, np.zeros((30, 2, 3)), target_hz=50.)


def test_diagnostic_export_is_never_admitted(tmp_path):
    run = _session(tmp_path)
    manifest = export_session(run, tmp_path / "diagnostic", diagnostic=True)
    assert not manifest["admitted_for_training"]
    assert all(not s["admitted_for_training"] and not s["label"]["valid"]
               for s in manifest["samples"])


def test_rt_boundary_rejects_diagnostic_and_accepts_verified_adl(tmp_path):
    import importlib.util
    from argparse import Namespace

    path = Path(__file__).resolve().parents[2] / "scripts/sionna/import_fall_mesh.py"
    spec = importlib.util.spec_from_file_location("rt_import_boundary", path)
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    run = _session(tmp_path)
    diagnostic = tmp_path / "diagnostic"
    export_session(run, diagnostic, diagnostic=True)
    with pytest.raises(ValueError, match="motion admission"):
        importer.load_geometry(Namespace(trial_json=None, dir=diagnostic, sample="forward_00"))
    admitted = tmp_path / "admitted"
    export_session(run, admitted)
    _, _, _, info = importer.load_geometry(
        Namespace(trial_json=None, dir=admitted, sample="forward_00"))
    assert importer.channel_activity(info["label"]["label"], info["fidelity"]).value == "adl"
    assert importer.channel_activity("stand", "kinematic_replay").value == "unknown"
