"""Regression tests for the P0-A item-4 per-segment swing analysis.

Two defects were found while answering "after the fix, re-run forward, backward,
start/stop and turning; confirm no right-arm sticking, jumping or abnormal
twisting", and both are silently plausible:

1. **Whole-run peak-to-peak fabricates a healthy ratio.** ``report.json``'s
   ``joint_span_deg`` takes ``np.ptp`` over the *entire* 10.3 s run. The right
   shoulder's extremum sits at t=0 while the left's sits at t=7.1, i.e. in
   different commanded motions, so their ratio compares two unrelated instants
   and read 1.066 where the walking segment alone reads 0.516.
2. **A window that misses the swing extremes under-reports it.** The source clip
   is one gait cycle in 1.2 s, but the controller plays it at 0.4 m/s against the
   gait's own 1.07 m/s, stretching the cycle to 3.214 s of wall clock. The demo's
   3.0 s W segment therefore starts at the peak and ends at 0.60 of the span,
   measuring 17.42 deg where the whole swing is 21.59 deg.

Every test below has a positive control: a check that the *wrong* rule really
does produce the wrong number, so a future edit that reintroduces it fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WALK_STRIP_DIR = Path(__file__).resolve().parents[2] / "scripts" / "humans"
sys.path.insert(0, str(WALK_STRIP_DIR))

from analyze_motion_swing import (  # noqa: E402
    ARM_CHAINS,
    SEGMENT_LABELS,
    SWING_COMMANDS,
    _font,  # noqa: F401  (imported to prove the module loads without Isaac)
    reset_steps,
    segments,
    swing_coverage,
)

# ----------------------------------------------------------------- small rigs


def _synthetic_run(
    *,
    samples: int = 1025,
    dt_s: float = 0.01,
    cycle_samples: int = 144,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """One shoulder DOF plus filler, walking a fixed number of cycles.

    The signal is a pure sinusoid in *sample* index with period
    ``cycle_samples``, so the true peak-to-peak is exactly 2.0 rad -- and, because
    ``cycle_samples`` is a multiple of 4, the sampled grid hits both extrema
    exactly instead of straddling them. That matters: a grid that misses the peak
    under-reads the whole-run reference by up to one sample step, which is the
    very under-read the production docstring warns about. Callers that need the
    extremum index should ask for it via ``_extrema`` rather than hardcoding it.
    """

    assert cycle_samples % 4 == 0, "the grid must land on both extrema"
    time_s = np.arange(samples, dtype=float) * dt_s
    phase = np.arange(samples, dtype=float) / cycle_samples * 2 * np.pi
    names = ["right_shoulder__dof2", "left_shoulder__dof2"]
    joints = np.zeros((samples, 2), dtype=float)
    joints[:, 0] = np.sin(phase)
    joints[:, 1] = np.sin(phase)
    root = np.zeros((samples, 3), dtype=float)
    # 0.01 m per step keeps motion well under the reset threshold.
    root[:, 0] = np.arange(samples, dtype=float) * 0.01
    return time_s, joints, names, root


def _extrema(joints: np.ndarray, cycle_samples: int = 144) -> tuple[int, int]:
    """First peak and the following trough, in sample index."""

    peak = cycle_samples // 4
    trough = peak + cycle_samples // 2
    # Guard the fixture against an off-by-one in the sinusoid.
    columns = joints[:, 0]
    assert int(np.argmax(columns[: cycle_samples // 2])) == peak
    assert int(np.argmin(columns[peak : peak + cycle_samples])) == trough - peak
    return peak, trough


# The synthetic sinusoid runs from -1 to +1 rad, so its whole interval -- the way
# the production caller passes it -- is a trough of -1 rad and a span of 2 rad.
# Both are converted to degrees here to sit on the same scale ``swing_coverage``
# reports. Passing the peak (+1 rad) as the trough would silently halve every
# fraction, which is why the pair is named after the *interval* and not its ends.
TRUE_TROUGH_DEG = float(np.rad2deg(-1.0))
TRUE_SPAN_DEG = float(np.rad2deg(2.0))


# --------------------------------------------------------------------- coverage


def test_a_whole_period_recovers_the_full_amplitude() -> None:
    time_s, joints, names, _ = _synthetic_run(samples=144 * 4)
    del names
    mask = np.ones(len(time_s), dtype=bool)
    coverage = swing_coverage(
        time_s=time_s,
        joints=joints,
        index=0,
        mask=mask,
        reference_span_deg=TRUE_SPAN_DEG,
        reference_trough_deg=TRUE_TROUGH_DEG,
    )
    # The sinusoid carries a unit amplitude in radians, i.e. 57.30 deg, so a
    # whole period must recover 114.58 deg peak-to-peak.
    assert coverage["ptp_deg"] == pytest.approx(TRUE_SPAN_DEG, abs=1e-6)
    assert coverage["peak_deg"] == pytest.approx(np.rad2deg(1.0), abs=1e-9)
    assert coverage["reaches_both_extremes"] is True
    # The run is four whole periods plus one trailing extra sample, so it opens on
    # the first sample sin(0)=0 and closes a step past sin(0), both of which sit
    # about half way up the [-1, +1] interval. Both fractions are read from the
    # reference trough: if the reference were omitted and the window normalised
    # against its own extremes, the closing fraction would read 0.0 by construction.
    assert coverage["open_fraction_of_span"] == pytest.approx(0.5, abs=0.02)
    assert coverage["close_fraction_of_span"] == pytest.approx(0.478, abs=0.02)
    assert coverage["close_fraction_of_span"] > 0.05


def test_a_half_period_recovers_the_full_amplitude_too() -> None:
    """An arm flexes once per stride: peak-to-trough is a complete swing.

    Requiring a whole *period* would reject this window, but it does contain both
    extremes, so its peak-to-peak value already is the amplitude.
    """

    time_s, joints, names, _ = _synthetic_run(samples=144 * 3)
    del names
    peak, trough = _extrema(joints)
    mask = np.zeros(len(time_s), dtype=bool)
    mask[peak : trough + 1] = True
    coverage = swing_coverage(
        time_s=time_s,
        joints=joints,
        index=0,
        mask=mask,
        reference_span_deg=TRUE_SPAN_DEG,
        reference_trough_deg=TRUE_TROUGH_DEG,
    )
    assert coverage["reaches_both_extremes"] is True
    assert coverage["ptp_deg"] == pytest.approx(TRUE_SPAN_DEG, rel=0.02)
    # Derive both fractions from the fixture rather than hand-computing them. The
    # denominator is the interval top minus its trough, which is exactly the span,
    # because the interval is passed as (trough, trough+span) -- so the fractions
    # are simply (value - trough) / span. The window opens on sample ``peak``
    # (the peak) and closes on sample ``trough``.
    opening_deg = float(np.rad2deg(joints[peak, 0]))
    closing_deg = float(np.rad2deg(joints[trough, 0]))
    assert coverage["open_fraction_of_span"] == pytest.approx(
        (opening_deg - TRUE_TROUGH_DEG) / TRUE_SPAN_DEG, abs=1e-9
    )
    assert coverage["close_fraction_of_span"] == pytest.approx(
        (closing_deg - TRUE_TROUGH_DEG) / TRUE_SPAN_DEG, abs=1e-9
    )
    # Opens exactly at the reference peak and closes exactly on the trough, so the
    # window spans the full interval: one at the top, zero at the bottom.
    assert coverage["open_fraction_of_span"] == pytest.approx(1.0, abs=0.02)
    assert coverage["close_fraction_of_span"] == pytest.approx(0.0, abs=0.02)
    assert coverage["fractions_measured_against"] == "reference"


def test_the_partial_window_is_the_documented_artefact() -> None:
    """The positive control for defect 2: a truncated window really does under-read.

    This reproduces the W-forward case -- open at the peak, close part way down --
    and asserts that the recovered amplitude is materially smaller than the true
    one. If someone later "fixes" the shortfall by relaxing a threshold instead of
    excluding the window, the ratio here stops matching and this test fails.

    It is also the positive control for the *circularity guard*: the fraction is
    read from the reference trough, not from the window's own minimum. Measured
    against the window's own minimum the closing fraction is 0.0 for every
    decreasing window, because the window's last sample **is** its minimum -- this
    assertion is what stops that mistake coming back.
    """

    time_s, joints, names, _ = _synthetic_run(samples=144 * 3)
    del names
    peak, trough = _extrema(joints)
    half = trough - peak
    # Stop 60% of the way along the tick count from the peak to the trough. The
    # signal is a sinusoid, so that is *not* 60% of the travel: the drop is slow
    # near the peak, so 60% of the ticks covers only about 23% of the span.
    stop = peak + int(round(0.60 * half))
    mask = np.zeros(len(time_s), dtype=bool)
    mask[peak : stop + 1] = True
    coverage = swing_coverage(
        time_s=time_s,
        joints=joints,
        index=0,
        mask=mask,
        reference_span_deg=TRUE_SPAN_DEG,
        reference_trough_deg=TRUE_TROUGH_DEG,
    )
    assert coverage["reaches_both_extremes"] is True, "the peak is still reached"
    # Derive the expectation from the fixture rather than hand-computing it: the
    # window closes on sample ``stop``, and the fraction is that value measured
    # from the reference trough over the full travel.
    closing_deg = float(np.rad2deg(joints[stop, 0]))
    expected = (closing_deg - TRUE_TROUGH_DEG) / TRUE_SPAN_DEG
    assert coverage["close_fraction_of_span"] == pytest.approx(expected, abs=1e-9)
    # The critical assertion: the closing fraction is *not* zero. Measured against
    # the window's own minimum it would be, because the last sample is that
    # minimum -- so this is the positive control for the reference-trough fix.
    assert coverage["close_fraction_of_span"] > 0.05
    assert coverage["fractions_measured_against"] == "reference"
    # The window's own trough sits above the reference trough, so the amplitude it
    # recovered is smaller than the whole swing -- which is why a sub-swing window
    # is excluded rather than compared against the gate.
    assert coverage["trough_deg"] > TRUE_TROUGH_DEG
    assert coverage["ptp_deg"] < TRUE_SPAN_DEG


def test_a_flat_window_reaches_no_extremes() -> None:
    time_s, joints, names, _ = _synthetic_run(samples=200)
    del names
    mask = np.zeros(len(time_s), dtype=bool)
    mask[10:20] = True
    joints = joints.copy()
    joints[:, 0] = 5.0  # frozen limb
    coverage = swing_coverage(
        time_s=time_s,
        joints=joints,
        index=0,
        mask=mask,
        reference_span_deg=TRUE_SPAN_DEG,
        reference_trough_deg=TRUE_TROUGH_DEG,
    )
    # A frozen limb has one instant for both extrema, so it did not swing -- and
    # the span must read zero rather than the arbitrary distance from the window's
    # closing sample back to its single extremum.
    assert coverage["ptp_deg"] == pytest.approx(0.0)
    assert coverage["peak_time_s"] == coverage["trough_time_s"]
    assert coverage["reaches_both_extremes"] is False


def test_an_empty_mask_is_not_an_error() -> None:
    time_s, joints, names, _ = _synthetic_run(samples=50)
    del names
    coverage = swing_coverage(
        time_s=time_s,
        joints=joints,
        index=0,
        mask=np.zeros(len(time_s), dtype=bool),
        reference_span_deg=TRUE_SPAN_DEG,
        reference_trough_deg=TRUE_TROUGH_DEG,
    )
    assert coverage["samples"] == 0
    assert coverage["reaches_both_extremes"] is False
    # Every key the caller reads must exist even with no samples, otherwise the
    # "not decidable" downgrade turns into a KeyError.
    for key in ("ptp_deg", "reference_span_deg", "open_fraction_of_span", "close_fraction_of_span"):
        assert coverage[key] == 0.0
    # The extremum times are absent rather than zero: no sample can claim to hold
    # the peak, and reporting 0.0 would name a real instant in the run.
    assert coverage["peak_time_s"] is None
    assert coverage["trough_time_s"] is None
    # No samples means no deviations to measure, so any segment verdict built on
    # these numbers has to be skipped rather than compared against a gate.
    assert coverage["fractions_measured_against"] == "window_min"


# --------------------------------------------------------------------- resets


def test_a_teleport_is_flagged_as_a_reset() -> None:
    root = np.zeros((10, 3))
    root[5:, 0] = 1.0  # 1 m in one step
    mask = reset_steps(root, 0.05)
    assert mask.sum() == 1
    assert bool(mask[4]) is True


def test_ordinary_walking_steps_are_not_resets() -> None:
    """The guard must not eat real motion: measured W steps are ~3.3 mm."""

    root = np.zeros((10, 3))
    root[:, 0] = np.arange(10) * 0.0033
    assert reset_steps(root, 0.05).sum() == 0


# ------------------------------------------------------------------- segments


def _timeline(*pairs: tuple[float, tuple[float, float]]) -> list[dict[str, object]]:
    return [{"time_s": t, "command": list(c)} for t, c in pairs]


def test_segments_are_folded_modulo_the_demo_and_labelled() -> None:
    timeline = _timeline(
        (0.0, (0.0, 0.0)),
        (1.0, (1.0, 0.0)),
        (4.0, (0.0, 0.0)),
        (10.3, (0.0, 0.0)),
        (11.3, (1.0, 0.0)),
        (14.3, (0.0, 0.0)),
    )
    rows = segments(timeline, 10.3)
    # A repeated demo contributes one row, not two: the instants fold together.
    labels = [row["label"] for row in rows]
    assert labels == ["stop / coast", "W forward", "stop / coast"]
    assert rows[1]["start_s"] == pytest.approx(1.0)
    assert rows[1]["stop_s"] == pytest.approx(4.0)
    assert rows[1]["duration_s"] == pytest.approx(3.0)


def test_the_final_segment_ends_at_the_demo_length() -> None:
    rows = segments(_timeline((7.7, (0.0, 0.0))), 10.3)
    assert rows[-1]["stop_s"] == pytest.approx(10.3)


def test_an_unlabelled_command_is_refused() -> None:
    """A new command must not silently inherit someone else's verdict."""

    timeline = _timeline((0.0, (0.5, 0.0)))
    with pytest.raises(KeyError, match="no label"):
        segments(timeline, 10.3)


def test_a_vanishingly_short_segment_is_dropped() -> None:
    timeline = _timeline((0.0, (1.0, 0.0)), (0.02, (0.0, 0.0)))
    rows = segments(timeline, 10.3)
    assert [row["label"] for row in rows] == ["stop / coast"]


# ------------------------------------------------------------------ constants


def test_only_walking_commands_drive_the_swing_check() -> None:
    """Turning in place does not advance the gait, so it has no swing to judge.

    Measured on the real 24 s run, the right shoulder spans 0.10 deg over the
    whole first A-turn segment. Calling that "the right arm is stuck" would be
    wrong, so the swing commands are enumerated rather than inferred from
    "the command is non-zero".
    """

    assert SWING_COMMANDS == {(1.0, 0.0), (-1.0, 0.0)}
    assert (0.0, 1.0) not in SWING_COMMANDS
    assert (0.0, -1.0) not in SWING_COMMANDS
    # Every swing command must still have a human-readable label.
    for command in SWING_COMMANDS:
        assert command in SEGMENT_LABELS


def test_the_arm_chains_are_the_four_the_gate_names() -> None:
    assert set(ARM_CHAINS) == {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
    }
