"""DTW template baseline: template construction, matching and the alarm contract."""

import numpy as np
import pytest

from sim2sense_fall.dtw_baseline import (
    TemplateConfig,
    build_fall_template,
    change_speed_features,
    dba_average,
    detect_fall_template,
    dtw_distance,
    resample_sequence,
    robust_standardize,
    template_windows,
)
from sim2sense_fall.windowing import WindowConfig

TAPS = 201
FRAMES = 60
DT = 0.05


def _collapse_cir(start_frame: int, rise_frames: int = 6) -> np.ndarray:
    """Stand, ramp up over a few frames (the collapse), settle high."""

    amplitudes = np.ones(FRAMES)
    amplitudes[start_frame : start_frame + rise_frames] = np.linspace(1.0, 10.0, rise_frames)
    amplitudes[start_frame + rise_frames :] = 10.0
    cir = np.zeros((FRAMES, TAPS), dtype=np.complex128)
    cir[:, 20] = amplitudes
    return cir


def _axes() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(TAPS) * 1e-8, np.arange(FRAMES) * DT


CONFIG = TemplateConfig(
    resample_points=32, dba_iterations=2, dtw_band_fraction=0.15, window_s=1.0,
)
WINDOW = WindowConfig(
    window_length_s=0.2, stride_s=0.1, threshold=0.8,
    min_consecutive_positive_windows=2,
)


def test_template_matches_a_collapse_and_not_a_flat_stream():
    taps, time = _axes()
    starts = [10, 20, 30]
    template = build_fall_template(
        [_collapse_cir(start) for start in starts],
        [taps] * 3, [time] * 3, [start * DT for start in starts], CONFIG,
    )
    query = _collapse_cir(25)
    centers, distances = template_windows(query, taps, time, template, CONFIG)
    assert len(centers) > 0 and len(centers) == len(distances)
    match = float(distances.min())
    flat = np.ones((FRAMES, TAPS), dtype=np.complex128)
    flat[:, 20] = 1.0
    _, flat_distances = template_windows(flat, taps, time, template, CONFIG)
    assert match < 0.1  # the collapse template reproduces itself near-exactly
    assert float(flat_distances.min()) > 0.4  # a no-change stream sits far away


def test_detector_alarms_on_collapse_and_stays_silent_on_flat():
    taps, time = _axes()
    starts = [10, 20, 30]
    template = build_fall_template(
        [_collapse_cir(start) for start in starts],
        [taps] * 3, [time] * 3, [start * DT for start in starts], CONFIG,
    )
    hit = detect_fall_template(_collapse_cir(25), taps, time, template, CONFIG, WINDOW)
    assert hit["window_alarmed"] is True
    assert hit["first_alarm_s"] is not None and hit["first_alarm_s"] >= 1.0
    flat = np.ones((FRAMES, TAPS), dtype=np.complex128)
    flat[:, 20] = 1.0
    miss = detect_fall_template(flat, taps, time, template, CONFIG, WINDOW)
    assert miss["window_alarmed"] is False
    assert miss["first_alarm_s"] is None


def test_identical_sequences_cost_zero_and_resample_keeps_endpoints():
    values = np.random.default_rng(0).normal(size=(40, 2))
    standardized = robust_standardize(values)
    resampled = resample_sequence(standardized, 64)
    assert resampled.shape == (64, 2)
    assert np.allclose(resampled[0], standardized[0])
    assert np.allclose(resampled[-1], standardized[-1])
    assert dtw_distance(resampled, resampled.copy(), 0.1) == pytest.approx(0.0, abs=1e-12)


def test_dba_average_of_one_sequence_converges_to_itself():
    values = robust_standardize(np.random.default_rng(1).normal(size=(32, 2)))
    averaged = dba_average([values], CONFIG)
    assert np.allclose(averaged, values, atol=1e-9)


def test_too_short_samples_are_skipped_or_rejected():
    taps, time = _axes()
    tiny = np.ones((3, TAPS), dtype=np.complex128)
    normal = _collapse_cir(20)
    template = build_fall_template(
        [tiny, normal], [taps, taps], [time[:3], time], [None, 1.0], CONFIG
    )
    assert template.shape == (CONFIG.resample_points, 2)
    with pytest.raises(ValueError, match="long enough"):
        build_fall_template([tiny], [taps], [time[:3]], [None], CONFIG)


def test_malformed_inputs_and_configs_are_rejected():
    taps, time = _axes()
    cir = _collapse_cir(20)
    with pytest.raises(ValueError, match="same length"):
        build_fall_template([cir], [taps, taps], [time, time], [1.0, 1.0], CONFIG)
    with pytest.raises(ValueError, match="strictly increasing"):
        change_speed_features(cir, taps, time[::-1].copy())
    for bad in (
        {"resample_points": 0},
        {"window_s": 0.0},
        {"dtw_band_fraction": 1.5},
        {"dba_iterations": -1},
    ):
        with pytest.raises(ValueError):
            TemplateConfig(**bad)
