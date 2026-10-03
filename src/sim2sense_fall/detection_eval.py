"""Event-level fall-detection metrics: alarm episodes, latency, false alarms per hour.

Window scores say how a classifier behaves; a fall-detection deployment is judged on
*events*. This module turns a per-window alarm flag sequence into the quantities the
project's acceptance rules pre-register:

* an alarm **episode** -- a run of positive windows that starts one event and is not
  re-counted until ``hold_off_s`` has passed, so one long streak is one alarm;
* **detection** -- an episode that starts no later than ``max_detection_latency_s``
  after the labelled imbalance onset. An alarm that eventually fires but never inside
  that budget is a miss, not a detection with bad latency; an episode that was already
  ringing *before* the onset does not count either (``pre_onset_tolerance_s``), because
  that is a false alarm that got lucky;
* **latency** -- reported from both the onset (when it became a fall) and the impact
  (when it became an emergency), because they differ by the whole collapse;
* **false alarms per hour** -- episodes raised on activity-labelled (ADL) streams per
  hour of ADL recording, which is the number a deployment contract is written in.

Aggregation is group-aware: samples from one physics session share a trajectory, so a
session that contributes eight standing segments cannot be allowed to dominate the mean.
Every rate is reported both pooled and as a per-group macro average.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EventEvalConfig:
    hold_off_s: float = 5.0
    max_detection_latency_s: float = 5.0
    # An episode that starts *before* the labelled onset is not a detection of that fall:
    # it is an alarm that happened to be already ringing. The default allows no such
    # credit; raise it only with a stated reason (window centring, onset derivation slack).
    pre_onset_tolerance_s: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("hold_off_s", self.hold_off_s),
                            ("max_detection_latency_s", self.max_detection_latency_s)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not np.isfinite(self.pre_onset_tolerance_s) or self.pre_onset_tolerance_s < 0:
            raise ValueError(
                f"pre_onset_tolerance_s must be finite and >= 0, got {self.pre_onset_tolerance_s!r}"
            )


def alarm_episodes(flag: np.ndarray, time_s: np.ndarray, *, hold_off_s: float) -> np.ndarray:
    """Start times of distinct alarm episodes under a hold-off.

    ``flag`` is the per-window (or per-frame) positive decision in time order. A rising
    edge opens an episode; a later rising edge inside ``hold_off_s`` of the opened one
    does not add a second event.
    """

    flags = np.asarray(flag, dtype=bool)
    times = np.asarray(time_s, dtype=float)
    if flags.shape != times.shape or not flags.ndim == 1:
        raise ValueError("flag and time_s must be one-dimensional and the same length")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be strictly increasing and finite")
    if not np.isfinite(hold_off_s) or hold_off_s <= 0:
        raise ValueError("hold_off_s must be finite and positive")
    edges = np.where(flags & ~np.concatenate(([False], flags[:-1])))[0]
    starts: list[float] = []
    for index in edges:
        if starts and float(times[index]) - starts[-1] < hold_off_s:
            continue
        starts.append(float(times[index]))
    return np.asarray(starts, dtype=float)


def sample_event_metrics(
    time_s: np.ndarray, alarm_flag: np.ndarray, *, activity: str,
    onset_s: float | None, impact_s: float | None, config: EventEvalConfig,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Event metrics for one sample, in the sample's own time origin.

    ``activity`` is ``"fall"``, ``"adl"``, or ``"unknown"`` (an activity the importer
    could not classify: reported, never silently counted as a negative). Every ADL
    episode is a false alarm. A fall sample with no onset cannot be scored as detected or
    missed, so it reports ``scorable: false`` rather than being counted as a miss.

    ``duration_s`` is the sample's *admitted channel exposure*, and it is what the
    false-alarm denominator is built from. Pass it explicitly: a frame-rate baseline scores
    every frame, while a window model only scores window centres, so falling back to
    ``time_s[-1] - time_s[0]`` would divide the model by a span shorter by one window per
    sample and inflate its hourly false-alarm rate (train02: 19.4 s instead of 35.0 s on the
    test split, which turned 412/h into 735/h).
    """

    if activity not in ("fall", "adl", "unknown"):
        raise ValueError(f"unknown activity class {activity!r}")
    episodes = alarm_episodes(alarm_flag, time_s, hold_off_s=config.hold_off_s)
    times = np.asarray(time_s, dtype=float)
    scored_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    exposure_s = scored_s if duration_s is None else float(duration_s)
    if not np.isfinite(exposure_s) or exposure_s < 0.0:
        raise ValueError(f"duration_s must be a non-negative finite span, got {duration_s!r}")
    if exposure_s + 1e-9 < scored_s:
        raise ValueError(
            f"exposure {exposure_s} s cannot be shorter than the scored span {scored_s} s")
    out: dict[str, Any] = {
        "activity": activity,
        "duration_s": exposure_s,
        "scored_span_s": scored_s,
        "episode_count": int(episodes.size),
        "first_episode_s": float(episodes[0]) if episodes.size else None,
        "scorable": True,
        "detected": False,
        "latency_from_onset_s": None,
        "latency_from_impact_s": None,
        "false_alarm_episodes": 0,
    }
    if activity != "fall":
        # Only a classified ADL stream contributes false alarms per hour; an unknown
        # activity is neither a negative nor a positive until it is labelled.
        if activity == "adl":
            out["false_alarm_episodes"] = int(episodes.size)
        return out
    if onset_s is None:
        out["scorable"] = False
        return out
    inside = episodes[(episodes >= onset_s - config.pre_onset_tolerance_s)
                      & (episodes <= onset_s + config.max_detection_latency_s)]
    out["detected"] = bool(inside.size)
    if inside.size:
        out["latency_from_onset_s"] = float(inside[0] - onset_s)
        if impact_s is not None:
            out["latency_from_impact_s"] = float(inside[0] - impact_s)
    return out


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


Z_95 = NormalDist().inv_cdf(0.975)  # 1.959964: the two-sided 95% quantile, no scipy needed


def _alpha(z: float) -> float:
    return 2.0 * (1.0 - NormalDist().cdf(z))


def proportion_ci(successes: int, trials: int, *, z: float = Z_95) -> list[float] | None:
    """Wilson score interval for a detection rate, clipped to [0, 1].

    Reported next to every rate: 4/5 and 5/5 are both "0.8-1.0" to the eye, but the
    intervals say how little 5 events actually pin down. The Wald interval is not used
    because it collapses to zero width at p=0 and goes negative near p=1, which is exactly
    where a small fall set lives.
    """

    if trials <= 0:
        return None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate_ci(events: int, hours: float, *, z: float = Z_95) -> list[float] | None:
    """Garwood-equivalent exact Poisson interval for an events-per-hour rate (Byar).

    Byar's cube-root approximation matches the chi-square interval to ~1% for k >= 1 and
    the k=0 case is exact: the upper bound is -ln(alpha/2) hours of exposure.
    """

    if hours <= 0:
        return None
    if events <= 0:
        return [0.0, round(-math.log(_alpha(z) / 2.0) / hours, 4)]
    low = events * (1.0 - 1.0 / (9.0 * events) - z / (3.0 * math.sqrt(events))) ** 3
    high = (events + 1) * (1.0 - 1.0 / (9.0 * (events + 1))
                           + z / (3.0 * math.sqrt(events + 1))) ** 3
    return [round(low / hours, 4), round(high / hours, 4)]


def event_metrics_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
    config: EventEvalConfig,
) -> list[dict[str, Any]]:
    """Turn per-window model probabilities into per-sample event metrics.

    This is what makes a trained model comparable with the frame-level baseline: both
    are reduced to alarm episodes under the same hold-off and latency budget, so a
    detection rate is always reported next to its false-alarms-per-hour. ``labels`` maps
    a sample id to ``{activity, onset_s, impact_s, duration_s}`` and must come from the same
    loader that produced the predictions -- never from a file name. ``duration_s`` is the
    sample's admitted channel exposure and is what the false-alarm denominator uses, so the
    windowed model and the frame-rate baseline divide by the same seconds.

    Window centres must be strictly increasing per sample: an episode's time is read off
    them, so an unordered or duplicated axis would silently corrupt every latency.

    Every label gets a row. A sample too short to fit one window contributes its exposure to
    the false-alarm denominator with zero episodes (`scored: false, windows: 0`); if it is a
    fall it is reported unscorable rather than as a miss, because the model never saw it.
    """

    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold!r}")
    grouped: dict[str, list[tuple[float, bool]]] = {}
    for row in predictions:
        sample_id = str(row["sample_id"])
        grouped.setdefault(sample_id, []).append(
            (float(row["center_s"]), bool(float(row["fall_probability"]) >= threshold))
        )
    rows: list[dict[str, Any]] = []

    def _identity(sample_id: str, known: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "group": sample_id.rsplit("/", 1)[0] if "/" in sample_id else "ungrouped",
            "activity": str(known["activity"]),
        }

    for sample_id, points in grouped.items():
        points.sort(key=lambda item: item[0])
        centers = np.array([point[0] for point in points], dtype=float)
        if len(centers) > 1 and np.any(np.diff(centers) <= 0):
            raise ValueError(f"{sample_id}: window centres must be strictly increasing")
        known = labels.get(sample_id)
        if known is None:
            raise ValueError(f"{sample_id}: no label entry, so it cannot be scored")
        metrics = sample_event_metrics(
            centers, np.array([point[1] for point in points], dtype=bool),
            activity=str(known["activity"]),
            onset_s=known.get("onset_s"), impact_s=known.get("impact_s"), config=config,
            duration_s=(None if known.get("duration_s") is None
                        else float(known["duration_s"])),
        )
        metrics.update(_identity(sample_id, known))
        metrics["scored"] = True
        metrics["windows"] = len(points)
        rows.append(metrics)
    for sample_id, known in sorted(labels.items()):
        if sample_id in grouped:
            continue
        exposure = float(known.get("duration_s") or 0.0)
        if not np.isfinite(exposure) or exposure < 0.0:
            raise ValueError(f"{sample_id}: an unscored sample needs a finite duration_s")
        # A fall the windower never reached cannot be called detected or missed, but its
        # seconds still count as ADL-free exposure only if it is labelled ADL.
        rows.append({
            **_identity(sample_id, known),
            "duration_s": exposure, "scored_span_s": 0.0, "episode_count": 0,
            "first_episode_s": None, "scorable": False, "detected": False,
            "latency_from_onset_s": None, "latency_from_impact_s": None,
            "false_alarm_episodes": 0, "scored": False, "windows": 0,
            "reason": "no window fitted inside the sample",
        })
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool the per-sample event metrics, and average them per group as well.

    Each row needs ``activity``, ``scorable``, ``detected``, ``duration_s``,
    ``false_alarm_episodes`` and optionally ``group``.
    """

    if not rows:
        return {"samples": 0, "reason": "no samples"}
    falls = [row for row in rows if row["activity"] == "fall"]
    scorable = [row for row in falls if row.get("scorable", True)]
    detected = [row for row in scorable if row.get("detected")]
    # Classified ADL only: an unlabelled stream is neither a negative nor a positive, and
    # putting its seconds in the denominator would lower FP/h without the detector ever
    # having had the chance to be wrong there.
    adl = [row for row in rows if row["activity"] == "adl"]
    adl_hours = sum(float(row["duration_s"]) for row in adl) / 3600.0
    adl_scored_hours = sum(float(row.get("scored_span_s", row["duration_s"]))
                           for row in adl) / 3600.0
    adl_unscored = [row for row in adl if not row.get("scored", True)]
    false_episodes = sum(int(row.get("false_alarm_episodes", 0)) for row in adl)
    latencies = [float(row["latency_from_onset_s"]) for row in detected
                 if row.get("latency_from_onset_s") is not None]
    impact_latencies = [float(row["latency_from_impact_s"]) for row in detected
                        if row.get("latency_from_impact_s") is not None]
    groups: dict[str, dict[str, Any]] = {}
    unknown = [row for row in rows if row["activity"] not in ("fall", "adl")]
    for row in rows:
        group = str(row.get("group") or "all")
        bucket = groups.setdefault(group, {"samples": 0, "falls_scorable": 0, "detected": 0,
                                           "false_alarm_episodes": 0, "adl_hours": 0.0})
        bucket["samples"] += 1
        if row["activity"] == "fall" and row.get("scorable", True):
            bucket["falls_scorable"] += 1
            bucket["detected"] += int(bool(row.get("detected")))
        if row["activity"] == "adl":
            bucket["false_alarm_episodes"] += int(row.get("false_alarm_episodes", 0))
            bucket["adl_hours"] += float(row["duration_s"]) / 3600.0
    per_group_detection = [
        bucket["detected"] / bucket["falls_scorable"]
        for bucket in groups.values() if bucket["falls_scorable"]
    ]
    return {
        "samples": len(rows),
        "unknown_activity_samples": len(unknown),
        "fall_samples": len(falls),
        "fall_samples_scorable": len(scorable),
        "fall_samples_unscorable": len(falls) - len(scorable),
        "detection_rate_pooled": (len(detected) / len(scorable)) if scorable else None,
        "detection_rate_pooled_ci95": proportion_ci(len(detected), len(scorable)),
        "detection_rate_group_macro_median": _median(per_group_detection),
        "groups": len(groups),
        "detection_latency_from_onset_s_median": _median(latencies),
        "detection_latency_from_impact_s_median": _median(impact_latencies),
        "adl_hours": round(adl_hours, 4),
        "adl_scored_hours": round(adl_scored_hours, 4),
        "adl_samples": len(adl),
        "adl_unscored_samples": len(adl_unscored),
        "unknown_adl_hours": round(sum(float(row["duration_s"]) for row in unknown)
                                   / 3600.0, 4),
        "adl_false_alarm_episodes": false_episodes,
        "adl_false_alarms_per_hour": (
            round(false_episodes / adl_hours, 4) if adl_hours else None
        ),
        "adl_false_alarms_per_hour_ci95": rate_ci(false_episodes, adl_hours),
        "per_group": {
            group: {
                "samples": bucket["samples"],
                "falls_scorable": bucket["falls_scorable"],
                "detected": bucket["detected"],
                "false_alarm_episodes": bucket["false_alarm_episodes"],
                "adl_hours": round(float(bucket["adl_hours"]), 4),
            }
            for group, bucket in sorted(groups.items())
        },
    }


def window_roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Mann-Whitney AUC of window scores against window fall labels.

    This is the statistic that survives a small event count: with five scorable falls the
    event-level detection rate can only take the values 0, 0.2, ..., 1.0, while the AUC over
    every scored window still measures how well the representation separates falling from
    activity. Ties count as half a win. ``None`` when one class is absent.
    """

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional and the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    positive = s[y == 1]
    negative = s[y == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    wins = float(np.count_nonzero(positive[:, None] > negative[None, :]))
    ties = float(np.count_nonzero(positive[:, None] == negative[None, :]))
    return float((wins + 0.5 * ties) / (positive.size * negative.size))


def cluster_bootstrap_auc_ci(
    scores: Sequence[float], labels: Sequence[int], clusters: Sequence[str],
    *, resamples: int = 500, seed: int = 0, z: float = Z_95,
) -> list[float] | None:
    """Percentile CI for the window AUC, resampled *by cluster* (physics session).

    Windows inside one session share a trajectory, so resampling windows would understate the
    spread by whatever the session has in common. A cluster with a single class contributes no
    AUC and is skipped, which is why the interval can be wide on a small batch -- that is the
    honest width, not a failure of the estimator.
    """

    if resamples < 10:
        raise ValueError("resamples must be >= 10")
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    keys = np.asarray([str(item) for item in clusters])
    if not (s.shape == y.shape == keys.shape) or s.ndim != 1:
        raise ValueError("scores, labels and clusters must align one-dimensionally")
    groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cluster in np.unique(keys):
        mask = keys == cluster
        groups[str(cluster)] = (s[mask], y[mask])
    names = np.asarray(sorted(groups), dtype=object)
    if names.size < 2:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        picked = rng.choice(names.size, size=names.size, replace=True)
        scores_drawn = np.concatenate([groups[str(names[i])][0] for i in picked])
        labels_drawn = np.concatenate([groups[str(names[i])][1] for i in picked])
        auc = window_roc_auc(scores_drawn, labels_drawn)
        if auc is not None:
            values.append(auc)
    if len(values) < 10:
        return None
    tail = 100.0 * _alpha(z) / 2.0  # z=1.96 -> 2.5th / 97.5th percentile
    lo = float(np.percentile(values, tail))
    hi = float(np.percentile(values, 100.0 - tail))
    return [round(lo, 4), round(hi, 4)]


def paired_episode_test(
    episodes_a: Sequence[int], episodes_b: Sequence[int], clusters: Sequence[str],
    *, resamples: int = 20000, seed: int = 0,
) -> dict[str, Any]:
    """Exact sign-flip permutation test for two detectors scored on the *same* exposure.

    Comparing two hourly false-alarm rates with marginal Poisson intervals throws away the
    pairing: both detectors saw the same samples, so the difference in their episode counts is
    far better estimated than either count alone. Flips are applied per **cluster** (physics
    session), never per segment, because segments from one session share a trajectory -- a
    segment-level test would pretend to have many more independent observations than it does.

    Under the null (both detectors alarm at the same rate on the same material) the detector
    labels are exchangeable within a cluster, so the two-sided p-value is the share of flips
    whose absolute difference is at least as extreme as the observed one.
    """

    a = np.asarray(list(episodes_a), dtype=np.int64)
    b = np.asarray(list(episodes_b), dtype=np.int64)
    keys = np.asarray([str(item) for item in clusters], dtype=object)
    if not (a.shape == b.shape == keys.shape) or a.ndim != 1:
        raise ValueError("episode counts and clusters must align one-dimensionally")
    if resamples < 100:
        raise ValueError("resamples must be >= 100")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("episode counts must be non-negative")
    # Collapse to one row per cluster so a flip is a whole-session swap.
    totals = {key: [0, 0] for key in set(keys.tolist())}
    for index, key in enumerate(keys.tolist()):
        totals[key][0] += int(a[index])
        totals[key][1] += int(b[index])
    session_a = np.asarray([v[0] for v in totals.values()], dtype=np.int64)
    session_b = np.asarray([v[1] for v in totals.values()], dtype=np.int64)
    observed = int(session_a.sum() - session_b.sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(resamples):
        flip = rng.integers(0, 2, size=session_a.size).astype(bool)
        swapped = np.where(flip, session_b, session_a) - np.where(flip, session_a, session_b)
        if abs(int(swapped.sum())) >= abs(observed):
            extreme += 1
    return {
        "test": "cluster sign-flip permutation (flips whole physics sessions)",
        "episodes_a": int(session_a.sum()), "episodes_b": int(session_b.sum()),
        "difference": observed, "clusters": int(session_a.size), "resamples": resamples,
        "p_two_sided": round((extreme + 1) / (resamples + 1), 4),
        "note": ("paired on identical exposure; the null assumes the two detectors alarm at "
                 "the same rate on the same material, not that their episode times coincide"),
    }


def config_dict(config: EventEvalConfig) -> dict[str, float]:
    """The pre-registered knobs, for embedding in a report so numbers are attributable."""

    return asdict(config)


# Pre-registered minima for reading an event-level report as a performance claim. They
# live in the metric module rather than in an analysis note, so a finished report can
# never be re-labelled a result after the fact: below any one of them the run is a
# protocol exercise, and it prints as one. A detection rate over fewer than 20 scorable
# falls, or an hourly false-alarm rate built on less than half an hour of classified ADL
# channel, is dominated by counting noise.
MIN_SCORABLE_FALL_SAMPLES = 20
MIN_ADL_HOURS = 0.5
MIN_EVAL_GROUPS = 8


def data_sufficiency(metrics: dict, protocol: str) -> dict:
    """Is this evaluation big enough to be quoted? Computed, never asserted by hand."""

    scorable = int(metrics.get("fall_samples_scorable") or 0)
    adl_hours = float(metrics.get("adl_hours") or 0.0)
    groups = int(metrics.get("groups") or 0)
    shortfalls = []
    if scorable < MIN_SCORABLE_FALL_SAMPLES:
        shortfalls.append(f"{scorable} scorable fall samples (<{MIN_SCORABLE_FALL_SAMPLES}): "
                          "a detection rate this coarse is one event wide")
    if adl_hours < MIN_ADL_HOURS:
        shortfalls.append(f"{adl_hours:.4f} h of classified ADL exposure "
                          f"(<{MIN_ADL_HOURS} h): false alarms per hour is Poisson noise")
    if groups < MIN_EVAL_GROUPS:
        shortfalls.append(f"{groups} held-out groups (<{MIN_EVAL_GROUPS}): the group-macro "
                          "median cannot separate a lucky session from a good model")
    return {
        "protocol": protocol,
        "claim_supported": not shortfalls,
        "scorable_fall_samples": scorable,
        "adl_hours": adl_hours,
        "adl_unscored_samples": int(metrics.get("adl_unscored_samples") or 0),
        "unknown_activity_samples": int(metrics.get("unknown_activity_samples") or 0),
        "groups": groups,
        "minima": {"scorable_fall_samples": MIN_SCORABLE_FALL_SAMPLES,
                   "adl_hours": MIN_ADL_HOURS, "groups": MIN_EVAL_GROUPS},
        "shortfalls": shortfalls,
    }
