"""Range-time representation: bounded maps, references and doppler rules."""

import numpy as np
import pytest

from sim2sense_fall.range_time import (
    RangeTimeConfig,
    doppler_map,
    range_time_map,
)

TAPS = 201


def _impulse_cir(amplitudes: list[float], tap: int = 20) -> np.ndarray:
    cir = np.zeros((len(amplitudes), TAPS), dtype=np.complex128)
    cir[:, tap] = np.asarray(amplitudes)
    return cir


def _axes(cir: np.ndarray, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    delays = np.arange(TAPS) * 1e-8
    time = np.arange(cir.shape[0]) * dt
    return delays, time


def test_identical_baseline_maps_to_center_of_unit_range():
    cir = _impulse_cir([1.0, 1.0, 1.0, 1.0])
    delays, time = _axes(cir)
    map_ = range_time_map(cir, delays, time, baseline_cir=cir[0])
    assert map_.magnitude.shape == cir.shape
    assert map_.magnitude.dtype == np.float32
    assert np.allclose(map_.magnitude, 0.5)
    assert map_.reference == "baseline_cir"


def test_step_change_saturates_only_at_the_changed_tap():
    cir = _impulse_cir([1.0, 1.0, 10.0, 10.0])
    delays, time = _axes(cir)
    map_ = range_time_map(cir, delays, time, baseline_cir=cir[0])
    assert map_.magnitude[:2, 20] == pytest.approx(0.5)
    assert map_.magnitude[2:, 20] == pytest.approx(0.75)  # +20 dB inside a 40 dB clip
    off_tap = map_.magnitude[:, 100]
    assert np.allclose(off_tap, 0.5)


def test_without_baseline_the_session_median_is_the_reference():
    cir = _impulse_cir([2.0, 3.0, 4.0, 5.0, 6.0])
    delays, time = _axes(cir)
    map_ = range_time_map(cir, delays, time)
    assert map_.reference == "session_median"
    assert map_.magnitude[2, 20] == pytest.approx(0.5)  # frame 2 holds the tap median


def test_malformed_inputs_are_rejected():
    cir = _impulse_cir([1.0, 1.0])
    delays, time = _axes(cir)
    with pytest.raises(ValueError, match="one entry per frame"):
        range_time_map(cir, delays, time[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        range_time_map(cir, delays, time[::-1].copy())
    with pytest.raises(ValueError, match="taps"):
        range_time_map(cir, delays, time, baseline_cir=np.ones(5, dtype=np.complex128))
    with pytest.raises(ValueError, match="clip_db"):
        range_time_map(cir, delays, time, config=RangeTimeConfig(clip_db=0.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        range_time_map(cir, delays[::-1].copy(), time)


def test_doppler_peaks_at_the_modulation_bin_and_stays_bounded():
    frames, dt = 16, 0.05
    time = np.arange(frames) * dt
    cir = np.zeros((frames, TAPS), dtype=np.complex128)
    cir[:, 20] = np.exp(2j * np.pi * 2.5 * time)  # 2.5 Hz -> rfft bin 2
    delays, _ = _axes(cir)
    doppler = doppler_map(cir, time, baseline_cir=np.zeros(TAPS, dtype=np.complex128))
    assert doppler.shape == (TAPS, frames)  # full complex FFT, one bin per frame
    assert doppler.min() >= 0.0 and doppler.max() <= 1.0
    assert int(np.argmax(doppler[20])) == 2


def test_doppler_rejects_non_uniform_or_short_time_axes():
    cir = _impulse_cir([1.0] * 8)
    _, time = _axes(cir)
    time[3] += 1e-3
    with pytest.raises(ValueError, match="uniform"):
        doppler_map(cir, time)
    with pytest.raises(ValueError, match="four frames"):
        doppler_map(cir[:3], np.arange(3) * 0.05)
