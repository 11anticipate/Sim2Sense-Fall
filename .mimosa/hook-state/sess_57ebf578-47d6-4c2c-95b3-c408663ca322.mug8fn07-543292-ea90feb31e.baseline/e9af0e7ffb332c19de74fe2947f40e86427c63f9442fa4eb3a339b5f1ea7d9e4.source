"""CPU-testable complex channel conversion with an absolute, shared delay grid."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def complex_amplitudes(value: Any) -> np.ndarray:
    """Sionna 2.x returns (real, imaginary), not alternative backends."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("expected real and imaginary amplitude arrays")
        real, imag = (as_numpy(x) for x in value)
        if real.shape != imag.shape:
            raise ValueError("real and imaginary amplitude shapes differ")
        result = real.astype(np.float64) + 1j * imag.astype(np.float64)
    else:
        result = as_numpy(value).astype(np.complex128)
    if not np.isfinite(result).all():
        raise ValueError("non-finite path amplitude")
    return result


def delay_grid(bandwidth_hz: float, max_delay_s: float) -> np.ndarray:
    if not np.isfinite([bandwidth_hz, max_delay_s]).all() or min(bandwidth_hz, max_delay_s) <= 0:
        raise ValueError("bandwidth and max delay must be finite and positive")
    return np.arange(int(np.ceil(max_delay_s * bandwidth_hz)) + 1) / bandwidth_hz


def paths_to_cir(
    amplitude: np.ndarray, delay_s: np.ndarray, valid: np.ndarray, grid_s: np.ndarray
) -> np.ndarray:
    """Bandlimited complex taps: sum a_p sinc(B*(t - tau_p)), t shared by all frames.

    No per-frame delay normalization and no magnitude-only conversion. The output is
    dimensionless channel gain; sum |a_p|^2 is an incoherent gain, not received watts.
    """
    a, tau, mask = amplitude.ravel(), delay_s.ravel(), valid.astype(bool).ravel()
    grid = np.asarray(grid_s, dtype=np.float64)
    if not (a.size == tau.size == mask.size):
        raise ValueError("SISO path arrays must have equal lengths")
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("delay grid needs at least two finite entries")
    dt = float(grid[1] - grid[0])
    if dt <= 0 or grid[0] != 0 or not np.allclose(np.diff(grid), dt, rtol=1e-8, atol=0):
        raise ValueError("delay grid must start at zero and be uniform")
    a, tau = a[mask], tau[mask]
    if not np.isfinite(a).all() or not np.isfinite(tau).all():
        raise ValueError("non-finite valid paths")
    if np.any(tau < 0) or np.any(tau > grid[-1]):
        raise ValueError("valid path delay outside configured CIR grid; increase max_delay_s")
    return np.sum(a[:, None] * np.sinc((grid[None, :] - tau[:, None]) / dt), axis=0)


def regular_frame_indices(total: int, requested: int) -> np.ndarray:
    """Select a uniform stride; requested is a target maximum, not a fake sample rate."""
    if total < 2 or requested < 2:
        raise ValueError("at least two source and requested frames required")
    stride = max(1, int(np.ceil((total - 1) / (requested - 1))))
    return np.arange(0, total, stride, dtype=np.int64)
