"""Link-geometry Doppler physics and the sampling-rate budget.

A body scatterer at position ``p`` moving with velocity ``v`` shifts the
carrier by ``f_D = (v · (û_tx + û_rx)) · f_c / c`` where ``û_tx`` / ``û_rx``
point from the scatterer to the transmitter / receiver. For a monostatic
link this reduces to ``2·v_radial/λ``; for our short bistatic links only
the velocity component along the local bisector is observed, so the same
body speed produces a smaller shift — the *geometry factor*
``|û_tx + û_rx| ∈ [0, 2]``.

The budget answers one pre-registered question: given the per-frame peak
Doppler measured on ground-truth mesh kinematics, which slow-time sampling
rates satisfy Nyquist for the fall's fastest body points? It is computed
from simulation truth, before any batch decision, and must not be
re-derived from aliased captures after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT_M_S = 299792458.0


def _validate_positions(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite (..., 3) positions")
    return values


def geometry_factors(points: np.ndarray, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """Per-point ``|û_tx + û_rx|`` of a bistatic link, shape of ``points``."""

    tx = _validate_positions(tx, "tx").reshape(3)
    rx = _validate_positions(rx, "rx").reshape(3)
    if np.allclose(tx, rx):
        raise ValueError("tx and rx coincide; the link is monostatic — pass factor 2.0")
    world = _validate_positions(points, "points")
    to_tx = tx - world
    to_rx = rx - world
    norm_tx = np.linalg.norm(to_tx, axis=-1, keepdims=True)
    norm_rx = np.linalg.norm(to_rx, axis=-1, keepdims=True)
    if float(min(norm_tx.min(), norm_rx.min())) <= 0:
        raise ValueError("a body point sits exactly on the transmitter or receiver")
    unit_tx = to_tx / norm_tx
    unit_rx = to_rx / norm_rx
    return np.linalg.norm(unit_tx + unit_rx, axis=-1)


def bistatic_doppler_hz(
    velocities: np.ndarray,
    points: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    carrier_hz: float,
) -> np.ndarray:
    """Doppler shift per point in Hz; monostatic collapses to ``2v/λ``."""

    if not np.isfinite(carrier_hz) or carrier_hz <= 0:
        raise ValueError("carrier_hz must be a positive frequency")
    vel = _validate_positions(velocities, "velocities")
    world = _validate_positions(points, "points")
    if vel.shape != world.shape:
        raise ValueError("velocities and points must share one shape")
    tx = _validate_positions(tx, "tx").reshape(3)
    rx = _validate_positions(rx, "rx").reshape(3)
    to_tx = tx - world
    to_rx = rx - world
    norm_tx = np.linalg.norm(to_tx, axis=-1, keepdims=True)
    norm_rx = np.linalg.norm(to_rx, axis=-1, keepdims=True)
    if float(min(norm_tx.min(), norm_rx.min())) <= 0:
        raise ValueError("a body point sits exactly on the transmitter or receiver")
    unit_tx = to_tx / norm_tx
    unit_rx = to_rx / norm_rx
    return np.sum(vel * (unit_tx + unit_rx), axis=-1) * carrier_hz / SPEED_OF_LIGHT_M_S


@dataclass(frozen=True, slots=True)
class DopplerBudget:
    """Pre-registered sampling verdict derived from simulation truth."""

    peak_doppler_hz: float
    nyquist_rate_hz: float
    geometry_factor_max: float
    peak_body_speed_m_s: float
    sufficient: tuple[str, ...]
    insufficient: tuple[str, ...]

    def verdict(self, rate_hz: float) -> str:
        return "sufficient" if rate_hz >= self.nyquist_rate_hz else "insufficient"


def frame_peak_doppler(
    vertices: np.ndarray,
    time_s: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    carrier_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval peak |f_D| (Hz) and peak body speed (m/s).

    Velocities are central differences of the mesh vertices; each interval
    reports the maximum over all body points, which is the quantity the
    slow-time sampling must capture — not the root speed.
    """

    verts = _validate_positions(vertices, "vertices")
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError("vertices must be (frames, points, 3)")
    time = np.asarray(time_s, dtype=np.float64)
    if time.shape != (verts.shape[0],) or len(time) < 3:
        raise ValueError("time must be one entry per frame (at least three frames)")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing and finite")
    if not np.isfinite(verts).all():
        raise ValueError("vertices must be finite")
    velocity = np.gradient(verts, time, axis=0)
    doppler = np.abs(bistatic_doppler_hz(velocity, verts, tx, rx, carrier_hz))
    speed = np.linalg.norm(velocity, axis=-1)
    return doppler.max(axis=1), speed.max(axis=1)


def doppler_budget(
    peak_doppler_hz: np.ndarray,
    peak_body_speed_m_s: np.ndarray,
    geometry_factor_max: float,
    candidate_rates_hz: tuple[float, ...],
) -> DopplerBudget:
    """Nyquist verdict per candidate slow-time rate from measured peaks.

    ``peak_doppler_hz`` / ``peak_body_speed_m_s`` are the per-interval maxima
    over all body points (see :func:`frame_peak_doppler`); the verdict uses
    their session maximum, not a mean — the fastest limb defines aliasing.
    """

    peaks = np.asarray(peak_doppler_hz, dtype=np.float64)
    speeds = np.asarray(peak_body_speed_m_s, dtype=np.float64)
    if peaks.ndim != 1 or len(peaks) < 1 or not np.isfinite(peaks).all():
        raise ValueError("peak_doppler_hz must be a finite one-dimensional series")
    if speeds.shape != peaks.shape or not np.isfinite(speeds).all():
        raise ValueError("peak_body_speed_m_s must match peak_doppler_hz in shape")
    if not np.isfinite(geometry_factor_max) or not 0 < geometry_factor_max <= 2.0:
        raise ValueError("geometry_factor_max must lie in (0, 2]")
    if not candidate_rates_hz or any(rate <= 0 for rate in candidate_rates_hz):
        raise ValueError("candidate_rates_hz must be positive")
    peak = float(peaks.max())
    nyquist = 2.0 * peak
    ordered = sorted(candidate_rates_hz)
    return DopplerBudget(
        peak_doppler_hz=peak,
        nyquist_rate_hz=nyquist,
        geometry_factor_max=geometry_factor_max,
        peak_body_speed_m_s=float(speeds.max()),
        sufficient=tuple(f"{rate:g}" for rate in ordered if rate >= nyquist),
        insufficient=tuple(f"{rate:g}" for rate in ordered if rate < nyquist),
    )
