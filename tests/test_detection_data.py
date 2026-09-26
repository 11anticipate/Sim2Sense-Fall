"""Channel-sample loading: labels come from the import payload and its trial."""

import json

import numpy as np
import pytest

from sim2sense_fall.detection_data import (
    ChannelSample,
    MeshMotion,
    iter_channel_samples,
    load_channel_sample,
    load_mesh_motion,
    velocity_labels,
)

TAPS = 201


def _write_sample(directory, name, frames, activity, event_label, trial=None, failures=None):
    time = np.arange(frames) * 0.05
    cir = np.zeros((frames, TAPS), dtype=np.complex128)
    cir[:, 20] = 1.0
    cir_path = directory / f"{name}.cir.npz"
    np.savez(
        cir_path,
        timestamp_s=time,
        cir=cir,
        frame_index=np.arange(frames),
        delay_s=np.arange(TAPS) * 1e-8,
        baseline_cir=cir[0] * 0.5,
    )
    payload = {
        "channel_sample_id": name,
        "activity": activity,
        "source_event_label": event_label,
    }
    if trial is not None:
        trial_path = directory / f"{name}.trial.json"
        trial_path.write_text(json.dumps({"label": trial}), encoding="utf-8")
        payload["source"] = {"source": str(trial_path)}
    if failures:
        payload["failures"] = failures
    (directory / f"{name}.import.json").write_text(json.dumps(payload), encoding="utf-8")
    return cir_path


def test_load_channel_sample_reads_labels_from_the_trial(tmp_path):
    path = _write_sample(
        tmp_path, "fall_a", frames=24, activity="fall", event_label="push_backward",
        trial={"imbalance_onset_s": 0.35, "first_impact_s": 0.98},
    )
    sample = load_channel_sample(path)
    assert sample.sample_id == "fall_a"
    assert sample.activity == "fall"
    assert sample.event_label == "push_backward"
    assert sample.imbalance_onset_s == pytest.approx(0.35)
    assert sample.first_impact_s == pytest.approx(0.98)
    assert sample.baseline_cir is not None and sample.baseline_cir.shape == (TAPS,)
    assert sample.time_s.shape == (24,) and sample.cir.shape == (24, TAPS)


def test_iter_channel_samples_skips_failures_and_orphans(tmp_path):
    _write_sample(tmp_path, "kept", 20, "adl", "walk")
    _write_sample(tmp_path, "failed", 20, "fall", "push_backward", failures=["x"])
    (tmp_path / "orphan.cir.npz").write_bytes(b"not a partner sample")
    samples = iter_channel_samples(tmp_path)
    assert [sample.sample_id for sample in samples] == ["kept"]
    forced = iter_channel_samples(tmp_path, include_failures=True)
    assert [sample.sample_id for sample in forced] == ["failed", "kept"]


def test_malformed_archives_fail_fast(tmp_path):
    path = _write_sample(tmp_path, "bad", 20, "adl", "walk")
    archive = dict(np.load(path))
    archive["timestamp_s"] = archive["timestamp_s"][::-1].copy()
    np.savez(path, **archive)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_channel_sample(path)


def test_import_payload_carries_link_geometry(tmp_path):
    path = _write_sample(tmp_path, "geo", 20, "adl", "walk")
    payload = json.loads((tmp_path / "geo.import.json").read_text(encoding="utf-8"))
    payload["transmitter"] = [9.28, 1.45, 1.4]
    payload["receiver"] = [10.88, 1.45, 1.4]
    payload["frequency_hz"] = 3.5e9
    (tmp_path / "geo.import.json").write_text(json.dumps(payload), encoding="utf-8")
    sample = load_channel_sample(path)
    assert sample.transmitter_xyz == (9.28, 1.45, 1.4)
    assert sample.carrier_hz == pytest.approx(3.5e9)


def _write_bare_cir(path, frames=6):
    time = np.arange(frames) * 0.05
    cir = np.zeros((frames, TAPS), dtype=np.complex128)
    cir[:, 20] = 1.0
    np.savez(path, timestamp_s=time, cir=cir, delay_s=np.arange(TAPS) * 1e-8)


def test_load_mesh_motion_resolves_trial_and_manifest_sources(tmp_path):
    vertices = np.zeros((5, 2, 3))
    # 试验变体: foo.trial.json 的同名 foo.npz 携带原生 mesh
    np.savez(tmp_path / "trial_a.npz", mesh_vertices_xyz=vertices,
             time_physics_s=np.arange(5) / 120.0)
    (tmp_path / "trial_a.trial.json").write_text("{}", encoding="utf-8")
    _write_bare_cir(tmp_path / "trial_a.cir.npz")
    (tmp_path / "trial_a.import.json").write_text(
        json.dumps({"source": {"source": str(tmp_path / "trial_a.trial.json")}}),
        encoding="utf-8")
    sample = load_channel_sample(tmp_path / "trial_a.cir.npz",
                                 tmp_path / "trial_a.import.json")
    motion = load_mesh_motion(sample)
    assert motion is not None and motion.vertices.shape == (5, 2, 3)
    # manifest 变体: <sample_id>.mesh.npz 与 manifest.json 同目录
    np.savez(tmp_path / "walk_x.mesh.npz", mesh_vertices_xyz=vertices,
             time_s=np.arange(5) / 30.0)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    _write_bare_cir(tmp_path / "walk_x.cir.npz")
    # channel_sample_id 带场景前缀, mesh 文件名靠 source_metadata.sample_id 解析
    (tmp_path / "walk_x.import.json").write_text(json.dumps({
        "channel_sample_id": "scene__walk_x",
        "source": {"source": str(tmp_path / "manifest.json"),
                   "source_metadata": {"sample_id": "walk_x"}},
    }), encoding="utf-8")
    sample2 = load_channel_sample(tmp_path / "walk_x.cir.npz",
                                  tmp_path / "walk_x.import.json")
    assert sample2.sample_id == "scene__walk_x"
    assert load_mesh_motion(sample2) is not None
    # 无从解析时返回 None 而不是抛错
    _write_bare_cir(tmp_path / "loose.cir.npz")
    (tmp_path / "loose.import.json").write_text("{}", encoding="utf-8")
    sample3 = load_channel_sample(tmp_path / "loose.cir.npz",
                                  tmp_path / "loose.import.json")
    assert load_mesh_motion(sample3) is None


def test_velocity_labels_match_analytic_monostatic_shift(tmp_path):
    import numpy as np
    wavelength = 299792458.0 / 3.5e9
    frames = 8
    time = np.arange(frames) / 120.0
    cir_time = np.array([0.1, 0.3, 0.5])
    cir = np.zeros((3, TAPS), dtype=np.complex128)
    cir[:, 20] = 1.0
    sample = ChannelSample(
        sample_id="v", cir_path=None, import_path=None, cir=cir,
        delay_s=np.arange(TAPS) * 1e-8, time_s=cir_time, baseline_cir=None,
        activity="fall", event_label=None, imbalance_onset_s=None,
        first_impact_s=None, transmitter_xyz=(0.0, 0.0, 0.0),
        receiver_xyz=(0.0, 0.0, 0.0), carrier_hz=3.5e9,
    )
    vertices = np.zeros((frames, 1, 3))
    vertices[:, 0, 0] = 5.0 - time  # 1 m/s toward the co-located radio
    labels = velocity_labels(sample, MeshMotion(vertices=vertices, time_s=time))
    assert labels == pytest.approx([2.0 / wavelength] * 3, rel=1e-6)
