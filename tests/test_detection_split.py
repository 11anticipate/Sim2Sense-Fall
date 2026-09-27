"""Group-aware split assignment: determinism, leakage and stratification guarantees."""

from __future__ import annotations

import pytest

from sim2sense_fall.detection_split import (
    SplitFractions,
    assign_splits,
    group_of,
    split_report,
)

HALF = SplitFractions(0.5, 0.25, 0.25)


def samples(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"sample_id": f"{group}__{index}", "group": group, "label": label}
            for index, (group, label) in enumerate(pairs)]


def test_group_of_collapses_a_session_export_to_the_session():
    assert group_of("artifacts/batches/t1/session_fall_walking_export") == "session_fall_walking"
    assert group_of(
        "artifacts/batches/t1/session_fall_walking_export/manifest.json"
    ) == "session_fall_walking"
    assert group_of(
        "artifacts/batches/t1/session_fall_walking_export/falling_00.mesh.npz"
    ) == "session_fall_walking"
    # Two exports from two sessions must not collapse onto one group.
    assert group_of(".../session_a_export/manifest.json") != group_of(
        ".../session_b_export/manifest.json"
    )
    # Independent physics trials stay independent groups.
    assert group_of("artifacts/t/trials/stand__push_backward.trial.json") == (
        "stand__push_backward"
    )
    assert group_of("artifacts/t/trials/stand__push_forward.trial.json") == "stand__push_forward"


def test_assignment_is_deterministic_and_order_independent():
    rows = samples(*[
        (f"g{index}", "fall") for index in range(6)
    ] + [(f"a{index}", "walk") for index in range(6)])
    forward = assign_splits(rows, fractions=HALF, seed=7)
    backward = assign_splits(list(reversed(rows)), fractions=HALF, seed=7)
    assert forward == backward
    other = assign_splits(rows, fractions=HALF, seed=8)
    assert other != forward, "the seed must actually change the assignment"


def test_no_group_ever_straddles_a_split():
    rows = samples(
        ("session_a", "fall"), ("session_a", "walk"), ("session_a", "stand"),
        ("session_b", "fall"), ("session_b", "walk"),
        ("session_c", "fall"), ("session_d", "walk"), ("session_e", "fall"),
    )
    report = split_report(rows, assign_splits(rows, fractions=HALF, seed=3))
    assert report["leakage_check"]["groups_spanning_splits"] == []
    # A session carrying several activities is one group, so it cannot be split.
    assert report["group_count"] == 5


def test_every_label_reaches_every_split_when_groups_allow():
    rows = samples(*[
        (f"session_{label}_{index}", label)
        for label in ("fall", "walk", "sit")
        for index in range(4)
    ])
    report = split_report(rows, assign_splits(rows, fractions=SplitFractions(0.5, 0.25, 0.25),
                                             seed=11))
    for split in ("train", "val", "test"):
        assert set(report["samples_by_split_label"][split]) == {"fall", "walk", "sit"}, split
    assert report["insufficient_groups"] == {}


def test_rare_label_is_reported_not_invented():
    rows = samples(("session_a", "fall"), ("session_b", "fall"), ("session_c", "walk"),
                   ("session_d", "walk"), ("session_e", "walk"))
    assignment = assign_splits(rows, fractions=HALF, seed=5)
    report = split_report(rows, assignment)
    assert "fall" in report["insufficient_groups"], "2 groups cannot cover 3 splits"
    assert set(report["samples_by_split_label"]["test"]) <= {"fall", "walk"}


def test_fractions_are_validated():
    with pytest.raises(ValueError, match="sum to 1"):
        SplitFractions(0.6, 0.6, 0.4)
    with pytest.raises(ValueError, match="expected train,val,test"):
        SplitFractions.parse("0.8,0.2")
    assert SplitFractions.parse("0.8,0.1,0.1") == SplitFractions(0.8, 0.1, 0.1)


def test_missing_label_or_group_fails_before_assigning():
    with pytest.raises(ValueError, match="non-empty group and label"):
        assign_splits([{"group": "a", "label": ""}], fractions=HALF, seed=1)
