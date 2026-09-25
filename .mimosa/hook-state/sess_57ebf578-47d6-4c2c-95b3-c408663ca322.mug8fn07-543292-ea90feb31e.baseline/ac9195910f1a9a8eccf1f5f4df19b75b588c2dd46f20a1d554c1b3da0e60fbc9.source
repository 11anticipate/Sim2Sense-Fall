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
        root_quaternion=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (720, 1)),
    )
    report = {
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
    assert [s["mode"] for s in segments] == ["forward", "falling", "fallen"]


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
