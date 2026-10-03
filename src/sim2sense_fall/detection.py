"""Physics-informed fall-detection baseline over complex CIR sequences.

A fall is a *step change* in the channel's slow features (received power,
delay spread) relative to the stream's own recent past, while activities of
daily living fluctuate inside a band. With a handful of real samples there is
nothing to train on, so the baseline detector is a robust trailing baseline
with a sustained-deviation rule whose scores feed the existing window alarm
aggregation (:func:`sim2sense_fall.windowing.aggregate_alarm`).

This module exists to exercise the consumer contract end to end and to be
replaced by a trained model later; it must not be reported as detection
performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .windowing import WindowConfig, aggregate_alarm


def cir_features(cir: np.ndarray, delay_s: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame channel features from a complex CIR sequence.

    Parameters
    ----------
    cir:
        Complex array (frames, taps).
    delay_s:
        Uniform, strictly increasing delay grid in seconds, one entry per tap.

    Returns
    -------
    dict with per-frame ``power_db`` (10·log10 Σ|a|²), ``mean_delay_s``,
    ``rms_delay_spread_s`` and ``peak_amplitude``.
    """

    values = np.asarray(cir)
    delays = np.asarray(delay_s, dtype=np.float64)
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError("cir must be a complex (frames, taps) array")
    if delays.ndim != 1 or len(delays) != values.shape[1] or len(delays) < 2:
        raise ValueError("delay grid must align with the tap axis and have >= 2 entries")
    if not np.isfinite(values).all() or not np.isfinite(delays).all():
        raise ValueError("cir and delay grid must be finite")
    if not np.all(np.diff(delays) > 0):
        raise ValueError("delay grid must be strictly increasing")
    power = np.abs(values) ** 2
    total = power.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("a frame carries no power; the CIR is all-zero")
    mean_delay = (power * delays).sum(axis=1) / total
    spread = np.sqrt(
        (power * (delays[None, :] - mean_delay[:, None]) ** 2).sum(axis=1) / total
    )
    return {
        "power_db": 10.0 * np.log10(total),
        "mean_delay_s": mean_delay,
        "rms_delay_spread_s": spread,
        "peak_amplitude": np.abs(values).max(axis=1),
    }


@dataclass(frozen=True, slots=True)
class StepDetectorConfig:
    """Trailing-baseline step-change detector over per-frame features.

    ``power_threshold_db`` and ``delay_spread_threshold_s`` are the deviation
    magnitudes (relative to the trailing baseline) that saturate each feature's
    score; ``baseline_tau_s`` is the exponential time constant of that baseline,
    which must track slow drift (walking) but not steps (a fall). ``warmup_s``
    suppresses alarms until the baseline has history, and ``min_consecutive``
    is the sustain requirement at frame level.
    """

    baseline_tau_s: float = 0.3
    warmup_s: float = 0.35
    power_threshold_db: float = 6.0
    delay_spread_threshold_s: float = 8e-9
    min_consecutive: int = 2

    def __post_init__(self) -> None:
        positive = {
            "baseline_tau_s": self.baseline_tau_s,
            "warmup_s": self.warmup_s,
            "power_threshold_db": self.power_threshold_db,
            "delay_spread_threshold_s": self.delay_spread_threshold_s,
        }
        if not np.isfinite(list(positive.values())).all() or min(positive.values()) <= 0:
            raise ValueError("detector thresholds and time constants must be finite and positive")
        if not np.isfinite(self.min_consecutive) or self.min_consecutive < 1:
            raise ValueError("min_consecutive must be a positive integer-like count")


def step_alarm_scores(
    features: dict[str, np.ndarray],
    time_s: np.ndarray,
    config: StepDetectorConfig,
) -> dict[str, np.ndarray]:
    """Per-frame alarm scores in [0, 1]: deviation from the trailing baseline.

    The baseline is an exponential moving average updated with the *previous*
    frame's value, so a step reaches the current frame before the baseline
    does and shows up as deviation. Power and delay spread each produce a
    ratio of (deviation / threshold); the frame score is the larger ratio,
    clipped to [0, 1] so the values are valid probabilities for
    :func:`~sim2sense_fall.windowing.aggregate_alarm`.
    """

    time = np.asarray(time_s, dtype=np.float64)
    power = np.asarray(features["power_db"], dtype=np.float64)
    spread = np.asarray(features["rms_delay_spread_s"], dtype=np.float64)
    if (time.ndim != 1 or len(time) < 2 or power.shape != time.shape
            or spread.shape != time.shape or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time must be a strictly increasing axis aligned with the features")
    alpha_per_frame = np.exp(-np.diff(time) / config.baseline_tau_s)
    baseline_power = np.empty_like(power)
    baseline_spread = np.empty_like(spread)
    baseline_power[0] = power[0]
    baseline_spread[0] = spread[0]
    for index in range(1, len(time)):
        weight = alpha_per_frame[index - 1]
        baseline_power[index] = weight * baseline_power[index - 1] + (1 - weight) * power[index - 1]
        baseline_spread[index] = (
            weight * baseline_spread[index - 1] + (1 - weight) * spread[index - 1]
        )
    warm = time - time[0] >= config.warmup_s
    power_ratio = np.abs(power - baseline_power) / config.power_threshold_db
    spread_ratio = np.abs(spread - baseline_spread) / config.delay_spread_threshold_s
    scores = np.clip(np.maximum(power_ratio, spread_ratio), 0.0, 1.0)
    scores[~warm] = 0.0
    alarm_frames = np.zeros(len(time), dtype=bool)
    run = 0
    for index, score in enumerate(scores):
        run = run + 1 if score >= 1.0 else 0
        alarm_frames[index] = run >= config.min_consecutive
    return {"scores": scores, "alarm": alarm_frames, "warm": warm}


def window_scores(scores: np.ndarray, window_frames: int, stride_frames: int) -> np.ndarray:
    """Frame-domain mean pooling of per-frame scores into window scores.

    The deployment windowing is seconds-based on a dense stream; the RT smoke
    samples are sparse and irregularly long, so windows are counted in frames
    here. Each pooled value stays in [0, 1] and can be fed to
    :func:`~sim2sense_fall.windowing.aggregate_alarm` with a matching
    ``min_consecutive_positive_windows``.
    """

    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 1 or window_frames < 1 or stride_frames < 1
            or len(values) < window_frames or not np.isfinite(values).all()):
        raise ValueError("scores must be one-dimensional with at least one full window")
    if not 0 <= values.min() <= values.max() <= 1:
        raise ValueError("scores must lie in [0, 1]")
    windows = range(0, len(values) - window_frames + 1, stride_frames)
    return np.array([values[start : start + window_frames].mean() for start in windows])


def detect_fall(
    features: dict[str, np.ndarray],
    time_s: np.ndarray,
    config: StepDetectorConfig,
    window: WindowConfig,
) -> dict[str, Any]:
    """Full baseline decision for one sample: frame scores, alarm and verdict.

    The seconds-based :class:`WindowConfig` is converted to frame counts using
    the sample's median frame period, so sparse RT sampling reuses the same
    aggregation the dense deployment stream will.
    """

    frame_result = step_alarm_scores(features, time_s, config)
    frame_dt = float(np.median(np.diff(time_s)))
    window_length = max(1, int(round(window.window_length_s / frame_dt)))
    stride_length = max(1, int(round(window.stride_s / frame_dt)))
    pooled = window_scores(frame_result["scores"], window_length, stride_length)
    alarmed = aggregate_alarm(pooled, window)
    alarm_frames = np.flatnonzero(frame_result["alarm"])
    return {
        "scores": frame_result["scores"],
        "alarm_frames": alarm_frames,
        # The same decision as a per-frame boolean aligned with ``time_s``, which is what
        # episode counting needs (``alarm_frames`` is only the index list).
        "alarm_flag": frame_result["alarm"],
        "first_alarm_s": float(time_s[alarm_frames[0]]) if len(alarm_frames) else None,
        "window_alarmed": bool(alarmed),
        "window_scores": pooled,
    }
