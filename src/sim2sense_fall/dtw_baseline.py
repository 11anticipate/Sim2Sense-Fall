"""DTW template-matching fall baseline over CIR change-speed features.

The middle rung between the trailing-baseline step detector and a trained
model: build a template of the *shape* a fall traces in channel-change
speed (fast deviation, then settling), then match sliding windows against
it with band-constrained dynamic time warping. Distance — unlike absolute
power — is intrinsically reference-free, which is the property the step
detector lacks when an ADL legitimately steps the channel to a new level.

With a handful of samples the template is in-sample; this module exercises
the matching machinery and must not be reported as detection performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import cir_features, window_scores
from .windowing import WindowConfig, aggregate_alarm


@dataclass(frozen=True, slots=True)
class TemplateConfig:
    """Parameters of template construction and matching.

    ``resample_points`` is the common length every window and the template
    are resampled to; ``dtw_band_fraction`` is the Sakoe-Chiba band as a
    fraction of that length; ``window_s`` is the duration of the sliding
    match window. Scores are calibrated against the template's own
    *null distance* — the cost an all-zero (no-change) window incurs — so
    a flat stream scores 0 and a perfect match scores 1 without a hand-set
    distance scale.
    """

    resample_points: int = 64
    dba_iterations: int = 3
    dtw_band_fraction: float = 0.1
    window_s: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "resample_points": float(self.resample_points),
            "dba_iterations": float(self.dba_iterations),
            "window_s": self.window_s,
        }
        if not np.isfinite(list(positive.values())).all() or min(positive.values()) <= 0:
            raise ValueError("template parameters must be finite and positive")
        if not 0 < self.dtw_band_fraction <= 1:
            raise ValueError("dtw_band_fraction must lie in (0, 1]")


def change_speed_features(cir: np.ndarray, delay_s: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Per-interval channel-change speed: (frames-1, 2) real features.

    Row *k* describes the transition between frame *k* and *k+1* by how fast
    received power (dB/s) and RMS delay spread (1/s) change across it.
    """

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing axis with at least three entries")
    features = cir_features(cir, delay_s)
    dt = np.diff(time)
    rate_power = np.diff(features["power_db"]) / dt
    rate_spread = np.diff(features["rms_delay_spread_s"]) / dt
    return np.column_stack([rate_power, rate_spread])


def robust_standardize(sequence: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Per-column median/MAD standardisation, clipped for outlier safety.

    Removes the sample's own scale so matching is about the *shape* of a
    change, not its absolute size; a constant column (MAD = 0) stays at zero.
    """

    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("sequence must be a finite two-dimensional array")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.where(mad > 0, 1.4826 * mad, 1.0)
    return np.clip((values - median) / scale, -clip, clip)


def resample_sequence(sequence: np.ndarray, points: int) -> np.ndarray:
    """Linear resample of a (n, d) sequence to (points, d) over [0, 1]."""

    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or points < 2 or len(values) < 2:
        raise ValueError("resampling needs a two-dimensional sequence of length >= 2")
    grid = np.linspace(0.0, 1.0, points)
    source = np.linspace(0.0, 1.0, len(values))
    return np.column_stack(
        [np.interp(grid, source, values[:, column]) for column in range(values.shape[1])]
    )


def dtw_path(
    reference: np.ndarray,
    query: np.ndarray,
    band: int,
) -> list[tuple[int, int]]:
    """Band-constrained DTW alignment path between two equal-length sequences."""

    if reference.shape != query.shape or reference.ndim != 2:
        raise ValueError("dtw_path needs two equal-shape two-dimensional sequences")
    length = reference.shape[0]
    if band < 1:
        raise ValueError("band must be >= 1")
    infinity = np.inf
    cost = np.full((length + 1, length + 1), infinity)
    cost[0, 0] = 0.0
    for i in range(1, length + 1):
        low = max(1, i - band)
        high = min(length, i + band)
        for j in range(low, high + 1):
            step = float(np.linalg.norm(reference[i - 1] - query[j - 1]))
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    if not np.isfinite(cost[length, length]):
        raise ValueError("band constraint left the sequence ends unaligned")
    path: list[tuple[int, int]] = []
    i, j = length, length
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        previous = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(previous, key=lambda item: item[0])
    path.reverse()
    return path


def dtw_distance(reference: np.ndarray, query: np.ndarray, band_fraction: float) -> float:
    """Average per-step DTW cost; identical sequences cost exactly zero."""

    path = dtw_path(reference, query, max(1, int(round(band_fraction * len(reference)))))
    total = sum(float(np.linalg.norm(reference[i] - query[j])) for i, j in path)
    return total / len(path)


def dba_average(sequences: list[np.ndarray], config: TemplateConfig) -> np.ndarray:
    """DTW barycenter averaging of equal-shape sequences into one template."""

    if not sequences:
        raise ValueError("dba_average needs at least one sequence")
    shapes = {sequence.shape for sequence in sequences}
    if len(shapes) != 1:
        raise ValueError("dba_average needs sequences of one common shape")
    average = np.mean(np.stack(sequences), axis=0)
    band = max(1, int(round(config.dtw_band_fraction * average.shape[0])))
    for _ in range(config.dba_iterations):
        accumulator = np.zeros_like(average)
        counts = np.zeros(average.shape[0])
        for sequence in sequences:
            for i, j in dtw_path(average, sequence, band):
                accumulator[i] += sequence[j]
                counts[i] += 1.0
        average = accumulator / np.maximum(counts, 1.0)[:, None]
    return average


def _fall_window_speed(
    cir: np.ndarray,
    delay_s: np.ndarray,
    time_s: np.ndarray,
    onset_s: float | None,
    config: TemplateConfig,
) -> np.ndarray | None:
    """Standardised speed features of one fall-sized window.

    The window is centred on the recorded imbalance onset when known,
    otherwise on the window of maximal change energy; returns None when the
    sample cannot host a full window.
    """

    time = np.asarray(time_s, dtype=np.float64)
    if len(time) < 3 or time[-1] - time[0] < config.window_s:
        return None
    speed = change_speed_features(cir, delay_s, time)
    mid_time = 0.5 * (time[:-1] + time[1:])
    start = 0
    if onset_s is not None:
        anchor = float(np.clip(onset_s, time[0], max(time[-1] - config.window_s, time[0])))
        start = int(np.searchsorted(time, anchor - 0.5 * config.window_s, side="left"))
        start = min(start, len(time) - 2)
    else:
        energy = np.abs(speed).sum(axis=1)
        window_points = max(1, int(round(config.window_s / float(np.median(np.diff(time))))))
        starts = range(0, max(1, len(speed) - window_points + 1))
        start = max(starts, key=lambda index: energy[index : index + window_points].sum())
    stop = int(np.searchsorted(time, time[start] + config.window_s, side="right"))
    mask = (mid_time >= time[start]) & (mid_time < time[stop - 1])
    if mask.sum() < 3:
        return None
    return resample_sequence(robust_standardize(speed[mask]), config.resample_points)


def build_fall_template(
    cirs: list[np.ndarray],
    delay_s_list: list[np.ndarray],
    time_s_list: list[np.ndarray],
    onsets_s: list[float | None],
    config: TemplateConfig,
) -> np.ndarray:
    """Template of the channel-change shape shared by the given fall samples.

    Each sample contributes the ``window_s`` window around its imbalance
    onset (or its most-changing window when the onset is unknown), so the
    template lives on the same time scale as the windows matched at runtime.
    Samples too short to host one window are skipped; if none remain, the
    build fails instead of returning a meaningless average.
    """

    if not (len(cirs) == len(delay_s_list) == len(time_s_list) == len(onsets_s)):
        raise ValueError("cir, delay, time and onset lists must have the same length")
    sequences = []
    for cir, delay_s, time_s, onset_s in zip(
        cirs, delay_s_list, time_s_list, onsets_s, strict=True
    ):
        speed = _fall_window_speed(cir, delay_s, time_s, onset_s, config)
        if speed is not None:
            sequences.append(speed)
    if not sequences:
        raise ValueError("no fall sample is long enough to host a template window")
    return dba_average(sequences, config)


def template_windows(
    cir: np.ndarray,
    delay_s: np.ndarray,
    time_s: np.ndarray,
    template: np.ndarray,
    config: TemplateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window DTW distances to the template.

    Returns (center times, distances) for every window of ``config.window_s``
    seconds that fits inside the sample, advanced one frame per step. A
    sample shorter than one window yields empty arrays.
    """

    time = np.asarray(time_s, dtype=np.float64)
    if len(time) < 3:
        return np.empty(0), np.empty(0)
    speed = change_speed_features(cir, delay_s, time)
    mid_time = 0.5 * (time[:-1] + time[1:])
    centers: list[float] = []
    distances: list[float] = []
    start = 0
    while start < len(time) and time[start] + config.window_s <= time[-1] + 1e-12:
        stop = int(np.searchsorted(time, time[start] + config.window_s, side="right"))
        mask = (mid_time >= time[start]) & (mid_time < time[stop - 1])
        if mask.sum() >= 3:
            window = resample_sequence(
                robust_standardize(speed[mask]), config.resample_points
            )
            centers.append(float(0.5 * (time[start] + time[stop - 1])))
            distances.append(dtw_distance(template, window, config.dtw_band_fraction))
        start += 1
    return np.asarray(centers), np.asarray(distances)


def detect_fall_template(
    cir: np.ndarray,
    delay_s: np.ndarray,
    time_s: np.ndarray,
    template: np.ndarray,
    config: TemplateConfig,
    window: WindowConfig,
) -> dict:
    """Full template-baseline decision for one sample.

    Window DTW distances are calibrated against the template's *null
    distance* — the distance an all-zero (no-change) window incurs, equal
    to ``mean|template|`` — giving scores in [0, 1] where a flat stream
    sits at 0 and a perfect match at 1. Scores are interpolated onto the
    frame axis and feed the same frame pooling and alarm aggregation as
    the step baseline, so both detectors share one consumer contract; the
    reported ``first_alarm_s`` is the end of the first sustained positive
    pooling window (window-domain alarm, seconds since sample start).
    """

    centers, distances = template_windows(cir, delay_s, time_s, template, config)
    null_distance = float(np.abs(np.asarray(template)).mean())
    if len(centers) and null_distance > 0:
        window_scores_raw = np.clip(1.0 - distances / null_distance, 0.0, 1.0)
        frame_scores = np.interp(
            time_s, centers, window_scores_raw,
            left=float(window_scores_raw[0]), right=float(window_scores_raw[-1]),
        )
    else:
        frame_scores = np.zeros(len(time_s))
    frame_dt = float(np.median(np.diff(time_s)))
    window_length = max(1, int(round(window.window_length_s / frame_dt)))
    stride_length = max(1, int(round(window.stride_s / frame_dt)))
    if len(frame_scores) >= window_length:
        pooled = window_scores(frame_scores, window_length, stride_length)
    else:
        pooled = np.empty(0)
    alarmed = aggregate_alarm(pooled, window)
    first_alarm_s: float | None = None
    run = 0
    for index, probability in enumerate(pooled):
        run = run + 1 if probability >= window.threshold else 0
        if run >= window.min_consecutive_positive_windows:
            first_alarm_s = float((index * stride_length + window_length) * frame_dt)
            break
    return {
        "frame_scores": frame_scores,
        "window_centers_s": centers,
        "window_distances": distances,
        "window_scores": pooled,
        "window_alarmed": bool(alarmed),
        "first_alarm_s": first_alarm_s,
    }
