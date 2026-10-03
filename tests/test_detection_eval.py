"""Event-level detection metrics: episode counting, latency budget and FP/hour."""

from __future__ import annotations

import numpy as np
import pytest

from sim2sense_fall.detection_eval import (
    EventEvalConfig,
    aggregate,
    alarm_episodes,
    event_metrics_from_predictions,
    sample_event_metrics,
)

CONFIG = EventEvalConfig(hold_off_s=5.0, max_detection_latency_s=5.0)
# A 60 s stream at 0.1 s resolution: long enough that a 5 s hold-off and a 5 s detection
# budget are actually distinguishable, which a two-second toy stream is not.
TIMES = np.round(np.arange(0.0, 60.0, 0.1), 6)


def flags(*true_indices: int) -> np.ndarray:
    values = np.zeros(len(TIMES), dtype=bool)
    values[list(true_indices)] = True
    return values


def test_a_run_of_positive_windows_is_one_episode():
    episodes = alarm_episodes(flags(3, 4, 5, 6), TIMES, hold_off_s=5.0)
    assert episodes == pytest.approx([0.3])


def test_the_hold_off_merges_nearby_runs_and_splits_distant_ones():
    merged = alarm_episodes(flags(1, 20), TIMES, hold_off_s=5.0)  # 0.1 and 2.0 s, < 5 s
    assert merged == pytest.approx([0.1])
    split = alarm_episodes(flags(1, 20), TIMES, hold_off_s=1.0)
    assert split == pytest.approx([0.1, 2.0])


def test_episode_inputs_are_validated():
    with pytest.raises(ValueError, match="one-dimensional"):
        alarm_episodes(np.ones((3, 2), dtype=bool), TIMES, hold_off_s=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        alarm_episodes(np.array([True, False]), np.array([0.0, 0.0]), hold_off_s=1.0)
    with pytest.raises(ValueError, match="hold_off_s"):
        alarm_episodes(flags(0), TIMES, hold_off_s=0.0)


def test_an_alarm_inside_the_latency_budget_counts_as_detected():
    row = sample_event_metrics(TIMES, flags(10, 11), activity="fall",
                              onset_s=0.9, impact_s=1.2, config=CONFIG)
    assert row["detected"] is True and row["scorable"] is True
    assert row["latency_from_onset_s"] == pytest.approx(0.1)
    # Episode opens at 1.0 s: 0.1 s after the onset, 0.2 s before the impact.
    assert row["latency_from_impact_s"] == pytest.approx(-0.2), "rang before the impact"


def test_an_alarm_outside_the_budget_is_a_miss_not_slow_detection():
    # Onset at 1 s, alarm at 30 s: the budget is 5 s, so this fall was missed even
    # though an alarm eventually fired -- counting it as "detected, slow" would hide it.
    row = sample_event_metrics(TIMES, flags(300), activity="fall", onset_s=1.0,
                              impact_s=1.5, config=CONFIG)
    assert row["detected"] is False and row["episode_count"] == 1
    assert row["latency_from_onset_s"] is None


def test_an_episode_already_ringing_before_the_onset_is_not_credit():
    row = sample_event_metrics(TIMES, flags(20), activity="fall", onset_s=5.0,
                              impact_s=5.5, config=CONFIG)
    assert row["detected"] is False, "a lucky pre-existing alarm is not a detection"
    tolerant = sample_event_metrics(TIMES, flags(20), activity="fall", onset_s=5.0,
                                    impact_s=5.5,
                                    config=EventEvalConfig(pre_onset_tolerance_s=3.5))
    assert tolerant["detected"] is True


def test_a_fall_without_an_onset_is_unscorable_rather_than_a_miss():
    row = sample_event_metrics(TIMES, flags(3), activity="fall", onset_s=None,
                              impact_s=None, config=CONFIG)
    assert row["scorable"] is False and row["detected"] is False


def test_adl_episodes_are_false_alarms_and_never_scored_as_falls():
    row = sample_event_metrics(TIMES, flags(2, 200), activity="adl", onset_s=None,
                              impact_s=None, config=CONFIG)
    assert row["false_alarm_episodes"] == 2 and row["detected"] is False
    assert row["duration_s"] == pytest.approx(TIMES[-1] - TIMES[0])


def test_an_unclassifiable_activity_is_reported_not_counted_as_a_negative():
    row = sample_event_metrics(TIMES, flags(2, 200), activity="unknown", onset_s=None,
                              impact_s=None, config=CONFIG)
    assert row["false_alarm_episodes"] == 0 and row["detected"] is False
    report = aggregate([{**_row(activity="unknown"), "false_alarm_episodes": 0}])
    assert report["unknown_activity_samples"] == 1
    assert report["adl_false_alarm_episodes"] == 0, "unknown must not inflate the ADL rate"


def test_a_bogus_activity_class_is_still_rejected():
    with pytest.raises(ValueError, match="unknown activity"):
        sample_event_metrics(TIMES, flags(1), activity="maybe", onset_s=None,
                             impact_s=None, config=CONFIG)


def _row(**overrides) -> dict:
    base = {"activity": "fall", "scorable": True, "detected": False, "duration_s": 10.0,
            "false_alarm_episodes": 0, "group": "session_a"}
    base.update(overrides)
    return base


def test_aggregation_reports_pooled_and_group_macro_numbers():
    rows = [
        # One group with nine detected falls, one with a single missed fall: the pooled
        # rate is 9/10 while the per-group median is 0.5, and both are reported.
        *[_row(group="big", detected=True, latency_from_onset_s=1.0,
               latency_from_impact_s=1.5) for _ in range(9)],
        _row(group="small", detected=False),
        _row(group="adl_a", activity="adl", duration_s=1800.0, false_alarm_episodes=3),
        _row(group="adl_b", activity="adl", duration_s=1800.0, false_alarm_episodes=1),
    ]
    report = aggregate(rows)
    assert report["fall_samples_scorable"] == 10
    assert report["detection_rate_pooled"] == pytest.approx(0.9)
    assert report["detection_rate_group_macro_median"] == pytest.approx(0.5)
    assert report["adl_hours"] == pytest.approx(1.0)
    assert report["adl_false_alarm_episodes"] == 4
    assert report["adl_false_alarms_per_hour"] == pytest.approx(4.0)
    assert report["detection_latency_from_onset_s_median"] == pytest.approx(1.0)
    assert report["groups"] == 4
    assert report["per_group"]["big"]["detected"] == 9


def test_aggregation_separates_unscorable_falls_and_handles_no_samples():
    rows = [_row(scorable=False), _row(detected=True, latency_from_onset_s=2.0)]
    report = aggregate(rows)
    assert report["fall_samples_unscorable"] == 1
    assert report["detection_rate_pooled"] == pytest.approx(1.0)
    assert aggregate([])["reason"] == "no samples"


def test_event_config_is_validated():
    with pytest.raises(ValueError, match="max_detection_latency_s"):
        EventEvalConfig(hold_off_s=1.0, max_detection_latency_s=float("nan"))


def _pred(sample_id: str, centers: list[float], positives: list[int]) -> list[dict]:
    return [{"sample_id": sample_id, "center_s": center,
             "fall_probability": 0.9 if index in positives else 0.1}
            for index, center in enumerate(centers)]


def test_model_predictions_reach_the_same_event_metrics_as_the_baseline():
    predictions = (
        _pred("session_a/falling_00", [0.0, 0.5, 1.0, 1.5], [2, 3])
        + _pred("session_a/stand_00", [0.0, 0.5, 1.0], [])
        + _pred("session_b/forward_00", [0.0, 0.5, 1.0], [2])
    )
    labels = {
        "session_a/falling_00": {"activity": "fall", "onset_s": 0.8, "impact_s": 1.2},
        "session_a/stand_00": {"activity": "adl", "onset_s": None, "impact_s": None},
        "session_b/forward_00": {"activity": "adl", "onset_s": None, "impact_s": None},
    }
    rows = event_metrics_from_predictions(predictions, labels, threshold=0.5, config=CONFIG)
    by_id = {row["sample_id"]: row for row in rows}
    fall = by_id["session_a/falling_00"]
    assert fall["detected"] is True and fall["group"] == "session_a"
    # Episode opens at centre 1.0 s, onset is 0.8 s -> 0.2 s after the imbalance began.
    assert fall["latency_from_onset_s"] == pytest.approx(0.2)
    assert by_id["session_a/stand_00"]["false_alarm_episodes"] == 0
    assert by_id["session_b/forward_00"]["false_alarm_episodes"] == 1
    report = aggregate(rows)
    assert report["detection_rate_pooled"] == pytest.approx(1.0)
    assert report["adl_false_alarm_episodes"] == 1
    assert report["adl_hours"] > 0


def test_a_higher_threshold_moves_the_decision_not_the_plumbing():
    predictions = _pred("s/falling_00", [0.0, 1.0, 2.0], [1])
    labels = {"s/falling_00": {"activity": "fall", "onset_s": 0.9, "impact_s": 1.4}}
    strict = event_metrics_from_predictions(predictions, labels, threshold=0.95, config=CONFIG)
    loose = event_metrics_from_predictions(predictions, labels, threshold=0.5, config=CONFIG)
    assert strict[0]["detected"] is False and loose[0]["detected"] is True


def test_prediction_inputs_are_validated():
    labels = {"s/x": {"activity": "adl", "onset_s": None, "impact_s": None}}
    with pytest.raises(ValueError, match="threshold must lie"):
        event_metrics_from_predictions([], labels, threshold=0.0, config=CONFIG)
    with pytest.raises(ValueError, match="strictly increasing"):
        event_metrics_from_predictions(
            _pred("s/x", [1.0, 1.0], [0]), labels, threshold=0.5, config=CONFIG)
    with pytest.raises(ValueError, match="no label entry"):
        event_metrics_from_predictions(
            _pred("s/unlabelled", [0.0, 1.0], []), labels, threshold=0.5, config=CONFIG)


def _centers(through_s: float) -> list[float]:
    """Window centres a 0.8 s window can produce on TIMES: the tail one window is unusable."""

    return [float(t) for t in TIMES if t <= TIMES[-1] - through_s]


def test_a_window_model_is_divided_by_the_same_exposure_as_the_frame_baseline():
    """The train02 test-split comparison divided the model by 19.4 s and the baseline by 35 s.

    A frame detector scores every frame, a window model only the centres, so an implicit
    denominator made the model's hourly false-alarm rate 1.8x the baseline's without a single
    extra alarm. Both are now given the sample's admitted exposure explicitly.
    """

    exposure = float(TIMES[-1] - TIMES[0])
    centers = _centers(0.8)
    predictions = [
        {"sample_id": "g0/adl_00", "center_s": center,
         "fall_probability": 0.9 if index in (5, 300) else 0.05}
        for index, center in enumerate(centers)
    ]
    labels = {"g0/adl_00": {"activity": "adl", "onset_s": None, "impact_s": None,
                            "duration_s": exposure}}
    model = aggregate(event_metrics_from_predictions(
        predictions, labels, threshold=0.5, config=CONFIG))
    baseline_row = sample_event_metrics(
        TIMES, flags(5, 300), activity="adl", onset_s=None, impact_s=None,
        config=CONFIG, duration_s=exposure)
    baseline = aggregate([{**baseline_row, "sample_id": "g0/adl_00", "group": "g0"}])
    assert model["adl_false_alarm_episodes"] == baseline["adl_false_alarm_episodes"] == 2
    assert model["adl_hours"] == pytest.approx(baseline["adl_hours"])
    assert model["adl_false_alarms_per_hour"] == pytest.approx(
        baseline["adl_false_alarms_per_hour"])
    # The part the model genuinely could not see is still disclosed, not hidden.
    assert model["adl_scored_hours"] < baseline["adl_scored_hours"]


def test_omitting_the_exposure_would_inflate_the_model_rate():
    exposure = float(TIMES[-1] - TIMES[0])
    centers = _centers(0.8)
    predictions = [
        {"sample_id": "g0/adl_00", "center_s": center,
         "fall_probability": 0.9 if index in (5, 300) else 0.05}
        for index, center in enumerate(centers)
    ]
    given = aggregate(event_metrics_from_predictions(
        predictions, {"g0/adl_00": {"activity": "adl", "onset_s": None, "impact_s": None,
                                    "duration_s": exposure}},
        threshold=0.5, config=CONFIG))
    omitted = aggregate(event_metrics_from_predictions(
        predictions, {"g0/adl_00": {"activity": "adl", "onset_s": None, "impact_s": None}},
        threshold=0.5, config=CONFIG))
    assert omitted["adl_false_alarms_per_hour"] > given["adl_false_alarms_per_hour"]


def test_a_sample_too_short_for_a_window_still_contributes_exposure():
    rows = event_metrics_from_predictions(
        [{"sample_id": "g/long_00", "center_s": 1.0, "fall_probability": 0.9},
         {"sample_id": "g/long_00", "center_s": 2.0, "fall_probability": 0.9}],
        {"g/long_00": {"activity": "adl", "onset_s": None, "impact_s": None,
                       "duration_s": 10.0},
         "g/short_00": {"activity": "adl", "onset_s": None, "impact_s": None,
                        "duration_s": 4.0},
         "g/fall_00": {"activity": "fall", "onset_s": 1.0, "impact_s": 2.0,
                       "duration_s": 6.0}},
        threshold=0.5, config=CONFIG)
    report = aggregate(rows)
    assert [row["sample_id"] for row in rows] == ["g/fall_00", "g/long_00", "g/short_00"]
    unscored = [row for row in rows if row["sample_id"] == "g/short_00"][0]
    assert unscored["scored"] is False and unscored["windows"] == 0
    assert unscored["false_alarm_episodes"] == 0
    assert report["adl_samples"] == 2 and report["adl_unscored_samples"] == 1
    assert report["adl_hours"] == pytest.approx(14.0 / 3600.0, abs=1e-4)
    assert report["adl_false_alarms_per_hour"] == pytest.approx(1 / (14.0 / 3600.0),
                                                                rel=1e-3)
    # An unseen fall is not a miss: it is unscorable, and the detection rate says None.
    assert report["fall_samples"] == 1 and report["fall_samples_scorable"] == 0
    assert report["detection_rate_pooled"] is None


def test_unlabelled_exposure_does_not_dilute_the_false_alarm_rate():
    rows = [_row(activity="adl", duration_s=60.0, false_alarm_episodes=1),
            _row(activity="unknown", duration_s=3600.0)]
    report = aggregate(rows)
    assert report["adl_hours"] == pytest.approx(60.0 / 3600.0, abs=1e-4)
    assert report["unknown_adl_hours"] == pytest.approx(1.0)
    assert report["adl_false_alarms_per_hour"] == pytest.approx(60.0)


def test_exposure_cannot_be_shorter_than_the_scored_span():
    with pytest.raises(ValueError, match="shorter than the scored span"):
        sample_event_metrics(TIMES, flags(1), activity="adl", onset_s=None, impact_s=None,
                            config=CONFIG, duration_s=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        sample_event_metrics(TIMES, flags(1), activity="adl", onset_s=None, impact_s=None,
                            config=CONFIG, duration_s=-1.0)


def test_data_sufficiency_gates_whether_a_report_may_be_quoted():
    from sim2sense_fall.detection_eval import data_sufficiency

    thin = {"fall_samples_scorable": 5, "adl_hours": 0.0054, "groups": 7}
    verdict = data_sufficiency(thin, "formal-round-1")
    assert verdict["claim_supported"] is False
    assert len(verdict["shortfalls"]) == 3, verdict["shortfalls"]
    fat = {"fall_samples_scorable": 24, "adl_hours": 0.92, "groups": 12}
    assert data_sufficiency(fat, "formal-round-2")["claim_supported"] is True


def test_rates_carry_intervals_that_show_what_a_small_batch_cannot_say():
    from sim2sense_fall.detection_eval import proportion_ci, rate_ci

    hours = 34.95 / 3600.0
    # Garwood/exact chi-square reference values, computed with scipy for this test.
    low, high = rate_ci(6, hours)
    assert (low, high) == pytest.approx((226.8, 1345.2), rel=0.01)
    assert rate_ci(0, hours) == pytest.approx((0.0, 380.0), rel=0.01)
    assert rate_ci(4, hours)[1] == pytest.approx(1054.9, rel=0.01)
    # A detection rate of 4/5 does not exclude 0.38; 5/5 does not exclude 1.0.
    assert proportion_ci(4, 5) == pytest.approx([0.3755, 0.9638], abs=1e-3)
    assert proportion_ci(5, 5)[1] == pytest.approx(1.0)
    assert proportion_ci(0, 5)[0] == 0.0
    assert proportion_ci(0, 0) is None and rate_ci(3, 0.0) is None


def test_aggregate_reports_both_intervals_next_to_the_point_estimates():
    report = aggregate([
        _row(activity="fall", detected=True, latency_from_onset_s=1.0, duration_s=4.0),
        _row(activity="fall", detected=False, duration_s=4.0),
        _row(activity="adl", duration_s=1800.0, false_alarm_episodes=3),
    ])
    assert report["detection_rate_pooled"] == pytest.approx(0.5)
    assert report["detection_rate_pooled_ci95"][0] == pytest.approx(0.0945, abs=1e-3)
    assert report["adl_false_alarms_per_hour"] == pytest.approx(6.0)
    assert report["adl_false_alarms_per_hour_ci95"][0] < 6.0 < \
        report["adl_false_alarms_per_hour_ci95"][1]


def test_window_auc_is_the_mann_whitney_statistic_with_ties_at_half():
    from sim2sense_fall.detection_eval import window_roc_auc

    assert window_roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert window_roc_auc([0.1, 0.9, 0.2, 0.8], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert window_roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == pytest.approx(0.5)
    # Two positives vs three negatives: 0.7 beats all three, 0.3 ties 0.3 and beats the
    # other two -> 5 strict wins and 1 tie out of the 6 pairs.
    assert window_roc_auc([0.7, 0.3, 0.3, 0.1, 0.05], [1, 1, 0, 0, 0]) == pytest.approx(
        (5 + 0.5) / 6)
    assert window_roc_auc([0.5, 0.4], [1, 1]) is None, "no negatives -> undefined"
    assert window_roc_auc([0.5, 0.4], [0, 0]) is None
    with pytest.raises(ValueError, match="same length"):
        window_roc_auc([0.5], [1, 0])
    with pytest.raises(ValueError, match="finite"):
        window_roc_auc([0.5, float("nan")], [1, 0])


def test_cluster_bootstrap_auc_resamples_sessions_not_windows():
    from sim2sense_fall.detection_eval import cluster_bootstrap_auc_ci

    scores, labels, clusters = [], [], []
    for session in range(6):
        rng = np.random.default_rng(session)
        pos = rng.normal(0.75, 0.1, 8)
        neg = rng.normal(0.35, 0.1, 8)
        scores.extend(list(pos) + list(neg))
        labels.extend([1] * 8 + [0] * 8)
        clusters.extend([f"s{session}"] * 16)
    first = cluster_bootstrap_auc_ci(scores, labels, clusters, resamples=300, seed=3)
    second = cluster_bootstrap_auc_ci(scores, labels, clusters, resamples=300, seed=3)
    assert first == second, "the bootstrap must be reproducible from its seed"
    assert first is not None and 0.0 <= first[0] < first[1] <= 1.0
    assert first[0] > 0.5, f"a clean signal must not straddle chance: {first}"
    # Windows inside one session share a trajectory: with a single cluster there is nothing
    # to resample, and claiming an interval from that would be a lie.
    assert cluster_bootstrap_auc_ci(scores, labels, ["only"] * len(scores)) is None
    with pytest.raises(ValueError, match="resamples"):
        cluster_bootstrap_auc_ci(scores, labels, clusters, resamples=2)


def test_paired_episode_test_flips_sessions_not_segments():
    from sim2sense_fall.detection_eval import paired_episode_test

    # Identical detectors -> no evidence, whatever the counts.
    same = paired_episode_test([4, 0, 1], [4, 0, 1], ["a", "b", "c"], resamples=400, seed=1)
    assert same["difference"] == 0 and same["p_two_sided"] == pytest.approx(1.0, abs=0.05)
    # One session carries the whole difference: with three clusters the smallest reachable
    # two-sided p is 2/8, so a single-session gap can never look significant. That is the
    # point of clustering -- it refuses to pretend one session is many observations.
    one = paired_episode_test([12, 0, 0], [0, 0, 0], ["a", "b", "c"], resamples=4000, seed=2)
    assert one["episodes_a"] == 12 and one["difference"] == 12
    assert one["p_two_sided"] >= 0.2, one
    # The same difference spread over many sessions does become significant.
    many = paired_episode_test([2] * 12, [0] * 12, [f"s{i}" for i in range(12)],
                               resamples=4000, seed=3)
    assert many["p_two_sided"] < 0.05, many
    # Deterministic in the seed, and it validates its inputs.
    again = paired_episode_test([2] * 12, [0] * 12, [f"s{i}" for i in range(12)],
                                resamples=4000, seed=3)
    assert again == many
    with pytest.raises(ValueError, match="align"):
        paired_episode_test([1], [0, 1], ["a"], resamples=200)
    with pytest.raises(ValueError, match="resamples"):
        paired_episode_test([1], [0], ["a"], resamples=10)
