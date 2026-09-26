"""Channel-sample loading: labels come from the import payload and its trial."""

import json

import numpy as np
import pytest

from sim2sense_fall.detection_data import (
    iter_channel_samples,
    load_channel_sample,
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
