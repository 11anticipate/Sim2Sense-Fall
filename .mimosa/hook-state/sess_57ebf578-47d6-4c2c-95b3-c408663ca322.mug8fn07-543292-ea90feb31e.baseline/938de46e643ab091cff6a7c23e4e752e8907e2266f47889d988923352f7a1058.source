from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim2sense_fall.schema import Activity, ChannelSample
from sim2sense_fall.simulators import check_optional_runtime
from sim2sense_fall.validation import validate_sample
from sim2sense_fall.windowing import WindowConfig, aggregate_alarm, iter_window_starts


def make_sample() -> ChannelSample:
    timestamps = np.arange(5, dtype=float) / 10
    return ChannelSample(
        sample_id="s1",
        scene_id="room-a",
        subject_id="subject-01",
        hardware_profile="wifi-5ghz-link-01",
        activity=Activity.FALL,
        timestamp_s=timestamps,
        channel=np.ones((5, 4), dtype=np.complex64),
        channel_representation="csi",
        sample_rate_hz=10.0,
        simulator_version="dry-run",
    )


def test_valid_sample_passes_contract() -> None:
    validate_sample(make_sample())


def test_non_monotonic_timestamps_fail() -> None:
    sample = make_sample()
    invalid = replace(sample, timestamp_s=np.array([0.0, 0.2, 0.1, 0.3, 0.4]))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_sample(invalid)


def test_window_starts_and_alarm_aggregation() -> None:
    config = WindowConfig(window_length_s=2.0, stride_s=0.5)
    starts = iter_window_starts(3.0, config)
    np.testing.assert_allclose(starts, [0.0, 0.5, 1.0])
    assert aggregate_alarm(np.array([0.2, 0.9, 0.85]), config)
    assert not aggregate_alarm(np.array([0.9, 0.2, 0.9]), config)


def test_dry_run_does_not_require_simulators() -> None:
    assert check_optional_runtime(dry_run=True) == "dry-run"
