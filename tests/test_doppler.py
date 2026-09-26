"""Bistatic Doppler physics and the sampling-budget verdict."""

import numpy as np
import pytest

from sim2sense_fall.doppler import (
    bistatic_doppler_hz,
    doppler_budget,
    frame_peak_doppler,
    geometry_factors,
)

CARRIER = 3.5e9


def test_point_moving_toward_the_bisector_shifts_by_the_full_formula():
    tx, rx = np.array([0.0, 0.0, 1.4]), np.array([1.6, 0.0, 1.4])
    point = np.array([0.8, 0.0, 0.5])  # below the link midpoint
    velocity = np.array([0.0, 0.0, -1.0])  # straight down
    factor = float(geometry_factors(point[None, :], tx, rx)[0])
    shift = float(bistatic_doppler_hz(
        velocity[None, :], point[None, :], tx, rx, CARRIER
    )[0])
    expected = -factor * CARRIER / 299792458.0  # (v·(û_t+û_r))·f_c/c, vertical v
    assert shift == pytest.approx(expected, rel=1e-9)
    assert 0 < factor < 2.0  # bistatic: strictly less than the monostatic 2


def test_monostatic_limit_reduces_to_two_v_over_lambda():
    point = np.array([[5.0, 0.0, 0.0]])  # exactly on the radial axis
    velocity = np.array([[-1.0, 0.0, 0.0]])  # toward the co-located radio
    shift = float(bistatic_doppler_hz(
        velocity, point, np.zeros(3), np.zeros(3), CARRIER
    )[0])
    wavelength = 299792458.0 / CARRIER
    assert shift == pytest.approx(2.0 / wavelength, rel=1e-9)


def test_frame_peak_tracks_the_fastest_point_not_the_root():
    tx, rx = np.array([-2.0, 0.0, 1.4]), np.array([2.0, 0.0, 1.4])
    frames = 5
    time = np.arange(frames) * 0.05
    verts = np.zeros((frames, 2, 3))
    verts[:, 0, :] = [0.0, 0.0, 1.0]  # a still root
    verts[:, 1, 0] = np.linspace(0.0, 1.0, frames)  # a limb sweeping 5 m/s
    doppler, speed = frame_peak_doppler(verts, time, tx, rx, CARRIER)
    assert speed.max() == pytest.approx(5.0, rel=1e-6)
    assert doppler.max() > 0.0
    verts_slow = verts.copy()
    verts_slow[:, 1, 0] = 0.0
    doppler_still, _ = frame_peak_doppler(verts_slow, time, tx, rx, CARRIER)
    assert doppler_still.max() == pytest.approx(0.0, abs=1e-9)


def test_budget_judges_each_candidate_rate_against_nyquist():
    peaks = np.array([10.0, 40.0, 25.0])
    speeds = np.array([0.5, 2.0, 1.2])
    budget = doppler_budget(peaks, speeds, geometry_factor_max=1.8,
                            candidate_rates_hz=(30.0, 50.0, 100.0))
    assert budget.peak_doppler_hz == pytest.approx(40.0)
    assert budget.nyquist_rate_hz == pytest.approx(80.0)
    assert budget.peak_body_speed_m_s == pytest.approx(2.0)
    assert budget.sufficient == ("100",)
    assert budget.insufficient == ("30", "50")
    assert budget.verdict(80.0) == "sufficient"
    assert budget.verdict(79.9) == "insufficient"


def test_malformed_inputs_are_rejected():
    tx, rx = np.array([0.0, 0.0, 1.4]), np.array([1.6, 0.0, 1.4])
    point = np.array([[0.8, 0.0, 0.5]])
    with pytest.raises(ValueError, match="one shape"):
        bistatic_doppler_hz(np.zeros((1, 3, 3)), point, tx, rx, CARRIER)
    with pytest.raises(ValueError, match="monostatic"):
        geometry_factors(point, point, point)
    with pytest.raises(ValueError, match="positive frequency"):
        bistatic_doppler_hz(point, point, tx, rx, 0.0)
    time = np.arange(4) * 0.05
    with pytest.raises(ValueError, match="strictly increasing"):
        frame_peak_doppler(np.zeros((4, 2, 3)), time[::-1].copy(), tx, rx, CARRIER)
    with pytest.raises(ValueError, match="must lie in"):
        doppler_budget(np.array([1.0]), np.array([1.0]), 2.5, (50.0,))
    with pytest.raises(ValueError, match="positive"):
        doppler_budget(np.array([1.0]), np.array([1.0]), 1.5, (-50.0,))
