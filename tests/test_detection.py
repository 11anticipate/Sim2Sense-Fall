"""Detection baseline: features, step scores and the alarm contract."""

import numpy as np
import pytest

from sim2sense_fall.detection import (
    StepDetectorConfig,
    cir_features,
    detect_fall,
    step_alarm_scores,
    window_scores,
)
from sim2sense_fall.windowing import WindowConfig, aggregate_alarm


def _impulse_cir(times: np.ndarray, delay_s: float, amplitude: float, frames: int):
    taps = np.arange(201) * 1e-8
    cir = np.zeros((frames, len(taps)), dtype=np.complex128)
    tap = int(round(delay_s / 1e-8))
    cir[:, tap] = amplitude
    return cir, taps


def test_features_are_exact_for_a_single_tap():
    cir, taps = _impulse_cir(None, 150e-9, 0.5, 6)
    features = cir_features(cir, taps)
    assert features["power_db"] == pytest.approx(20 * np.log10(0.5), rel=1e-9)
    assert features["mean_delay_s"] == pytest.approx(150e-9, rel=1e-6)
    assert features["rms_delay_spread_s"] == pytest.approx(0.0, abs=1e-12)
    assert features["peak_amplitude"] == pytest.approx(0.5)


def test_features_reject_misaligned_or_degenerate_input():
    cir, taps = _impulse_cir(None, 100e-9, 1.0, 4)
    with pytest.raises(ValueError, match="align"):
        cir_features(cir, taps[:-1])
    zero = cir.copy()
    zero[:, :] = 0
    with pytest.raises(ValueError, match="no power"):
        cir_features(zero, taps)
    with pytest.raises(ValueError, match="strictly increasing"):
        cir_features(cir, taps[::-1].copy())


def test_step_change_alarms_and_stable_stream_does_not():
    config = StepDetectorConfig()
    frames = 40
    time_s = np.arange(frames) * 0.05
    taps = np.arange(201) * 1e-8
    stable = np.zeros((frames, len(taps)), dtype=np.complex128)
    stable[:, 20] = 1.0
    step = stable.copy()
    step[20:, 20] = 10.0  # +20 dB step halfway through
    quiet = {"scores": step_alarm_scores(cir_features(stable, taps), time_s, config)["scores"]}
    assert quiet["scores"].max() == pytest.approx(0.0, abs=1e-9)
    step_result = step_alarm_scores(cir_features(step, taps), time_s, config)
    assert step_result["alarm"].any()
    first = np.flatnonzero(step_result["alarm"])[0]
    assert time_s[first] >= config.warmup_s


def test_window_scores_pool_and_stay_probability_like():
    scores = np.array([0.0, 0.5, 1.0, 1.0, 0.2, 0.0])
    pooled = window_scores(scores, window_frames=2, stride_frames=1)
    assert pooled[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        window_scores(np.array([0.0, 1.5, 0.0]), 2, 1)


def test_detect_fall_end_to_end_contract():
    frames = 40
    time_s = np.arange(frames) * 0.05
    taps = np.arange(201) * 1e-8
    cir = np.zeros((frames, len(taps)), dtype=np.complex128)
    cir[:, 20] = 1.0
    cir[20:, 20] = 10.0
    result = detect_fall(
        cir_features(cir, taps), time_s, StepDetectorConfig(), WindowConfig(
            window_length_s=0.1, stride_s=0.05, threshold=0.8,
            min_consecutive_positive_windows=2,
        )
    )
    assert result["window_alarmed"] is True
    assert result["first_alarm_s"] is not None
    assert result["first_alarm_s"] >= 0.35  # never inside the warmup


def test_aggregate_alarm_still_gates_the_baseline_output():
    assert aggregate_alarm(np.array([1.0, 1.0]), WindowConfig()) is True
    assert aggregate_alarm(np.array([1.0, 0.5]), WindowConfig()) is False
