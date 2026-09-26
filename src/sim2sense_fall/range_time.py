"""Range-time image representation of a complex CIR sequence.

A fall is easier to separate from daily activities in the *full* tap-time
distribution than in four scalar summaries, so the trained detectors planned
for stage 9 consume a two-dimensional view of the channel: per-tap received
power over time (the "range-time" map), referenced against a no-human
baseline when one exists. This module is a pure-numpy representation layer;
it renders nothing, trains nothing, and claims nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RangeTimeConfig:
    """Clipping and normalisation for dB-domain channel maps.

    Deviations are kept symmetric around the reference in ``[-clip_db,
    +clip_db]`` and rescaled to ``[0, 1]``; anything beyond the clip is
    saturated, so the map is a *bounded* image rather than raw dB.
    """

    clip_db: float = 40.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.clip_db) or self.clip_db <= 0:
            raise ValueError("clip_db must be a positive finite dB value")


@dataclass(frozen=True, slots=True)
class RangeTimeMap:
    """A bounded channel image plus the axes it was built from.

    ``magnitude`` is ``[0, 1]`` shaped ``(frames, taps)``; ``reference``
    records what the dB values were measured against so a map can never be
    read without knowing its zero point.
    """

    magnitude: np.ndarray
    delay_s: np.ndarray
    time_s: np.ndarray
    reference: str
    doppler: np.ndarray | None = None


def _validate_cir(
    cir: np.ndarray,
    delay_s: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(cir)
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError("cir must be a complex (frames, taps) array")
    if not np.isfinite(values).all():
        raise ValueError("cir must be finite")
    if delay_s is not None:
        delays = np.asarray(delay_s, dtype=np.float64)
        if delays.ndim != 1 or len(delays) != values.shape[1] or len(delays) < 2:
            raise ValueError("delay grid must align with the tap axis and have >= 2 entries")
        if not np.isfinite(delays).all() or not np.all(np.diff(delays) > 0):
            raise ValueError("delay grid must be finite and strictly increasing")
    return values


def _validate_baseline(baseline_cir: np.ndarray, taps: int) -> np.ndarray:
    baseline = np.asarray(baseline_cir)
    if baseline.shape != (taps,) or not np.iscomplexobj(baseline):
        raise ValueError("baseline_cir must be a complex (taps,) array")
    if not np.isfinite(baseline).all():
        raise ValueError("baseline_cir must be finite")
    return baseline


def _to_unit(db: np.ndarray, clip_db: float) -> np.ndarray:
    """Clip dB deviations symmetrically around the reference and rescale."""

    clipped = np.clip(db, -clip_db, clip_db)
    return ((clipped + clip_db) / (2.0 * clip_db)).astype(np.float32)


def _power_db(values: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return 10.0 * np.log10(np.abs(values) ** 2 + eps)


def range_time_map(
    cir: np.ndarray,
    delay_s: np.ndarray,
    time_s: np.ndarray,
    baseline_cir: np.ndarray | None = None,
    config: RangeTimeConfig | None = None,
) -> RangeTimeMap:
    """Per-tap dB power over time, referenced and bounded to ``[0, 1]``.

    The reference is the no-human ``baseline_cir`` when provided, otherwise
    the per-tap median across the session itself; the choice is recorded on
    the returned map because the two zero points are not interchangeable.
    """

    cfg = config or RangeTimeConfig()
    values = _validate_cir(cir, delay_s)
    time = np.asarray(time_s, dtype=np.float64)
    if time.shape != (values.shape[0],):
        raise ValueError("time axis must be one entry per frame")
    if len(time) >= 2 and (not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("time axis must be finite and strictly increasing")
    power_db = _power_db(values)
    if baseline_cir is not None:
        reference_db = _power_db(_validate_baseline(baseline_cir, values.shape[1]))
        reference = "baseline_cir"
    else:
        reference_db = np.median(power_db, axis=0)
        reference = "session_median"
    magnitude = _to_unit(power_db - reference_db[None, :], cfg.clip_db)
    return RangeTimeMap(
        magnitude=magnitude,
        delay_s=np.asarray(delay_s, dtype=np.float64),
        time_s=time,
        reference=reference,
    )


def doppler_map(
    cir: np.ndarray,
    time_s: np.ndarray,
    baseline_cir: np.ndarray | None = None,
    config: RangeTimeConfig | None = None,
) -> np.ndarray:
    """Per-tap Doppler spectrum magnitude, bounded to ``[0, 1]``.

    Static paths are removed by subtracting the no-human baseline when
    available, otherwise the time mean; an FFT along the time axis then
    produces the spectrum per tap. Requires a *uniform* time grid — sparse
    RT sub-sampling that violates this raises instead of silently smearing
    the spectrum.
    """

    cfg = config or RangeTimeConfig()
    values = _validate_cir(cir)
    time = np.asarray(time_s, dtype=np.float64)
    if time.shape != (values.shape[0],) or len(time) < 4:
        raise ValueError("doppler needs at least four frames with one time entry each")
    diffs = np.diff(time)
    if not np.isfinite(time).all() or (diffs <= 0).any():
        raise ValueError("time axis must be strictly increasing and finite")
    median_dt = float(np.median(diffs))
    if float(np.abs(diffs - median_dt).max()) > 1e-4 * median_dt:
        raise ValueError("doppler requires a uniform time grid; resample first")
    if baseline_cir is not None:
        signal = values - _validate_baseline(baseline_cir, values.shape[1])[None, :]
    else:
        signal = values - values.mean(axis=0, keepdims=True)
    spectrum = np.fft.fft(signal, axis=0)
    magnitude_db = _power_db(spectrum.T)  # (taps, doppler bins)
    magnitude_db = magnitude_db - magnitude_db.max() + cfg.clip_db
    return np.clip(magnitude_db / cfg.clip_db, 0.0, 1.0).astype(np.float32)
