"""Root-travel decomposition must follow the body frame, not a world axis.

Regression test for the P0-A reporting defect: ``root_displacement_m`` labelled
world ``x`` as "forward" while the shipped config spawns the character at
``heading_deg 90``, i.e. facing world ``+Y``. A clean 1.1 m walk was therefore
written into ``lateral_drift_max_m`` and read as a 1.2 m stumble.

Quaternions here are ``(w, x, y, z)``, matching
:func:`~sim2sense_fall.humans.rotations.quaternion_to_matrix` and the
``root_quaternion`` array the capture script records.

The positive control is ``test_world_axis_labelling_would_fail_the_positive_control``:
it re-runs the *pre-fix* rule on the same fixture and requires it to produce the
wrong answer, so a green suite cannot be explained by the fixture being trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim2sense_fall.humans.travel import root_displacement_in_body_frame

WALK_M = 1.2
STEPS = 240
# Real per-step travel is ~3.3 mm (0.4 m/s at a 1/120 s timestep); a cycle rewind
# teleports ~1.2 m. The fixture must be sized like the real data, because step
# length is what separates walking from a reset. ``STEP_M`` is derived from the
# number of *steps* (N-1 gaps between N samples), not N, so the fixture delivers
# exactly ``WALK_M`` of travel and the assertions can be exact.
STEP_M = WALK_M / (STEPS - 1)


def _yaw_quaternion(heading_deg: float) -> np.ndarray:
    """Quaternion for a pure yaw about +Z, in ``(w, x, y, z)`` order."""

    half = np.deg2rad(heading_deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _straight_walk(heading_deg: float, distance_m: float = WALK_M) -> tuple[np.ndarray, ...]:
    """Move straight along the character's own facing direction, one step at a
    time, holding the orientation fixed -- the geometry of a 'W' walk."""

    heading = np.deg2rad(heading_deg)
    step = np.array([np.cos(heading), np.sin(heading), 0.0]) * (distance_m / (STEPS - 1))
    root = np.array([step * i for i in range(STEPS)])
    root[:, 2] = 0.9
    return root, np.tile(_yaw_quaternion(heading_deg), (STEPS, 1)), np.ones((STEPS, 2))


def _backpedal(heading_deg: float, distance_m: float = WALK_M) -> tuple[np.ndarray, ...]:
    """Facing ``heading_deg`` but travelling the opposite way -- pressing S."""

    heading = np.deg2rad(heading_deg)
    step = np.array([np.cos(heading), np.sin(heading), 0.0]) * (-distance_m / (STEPS - 1))
    root = np.array([step * i for i in range(STEPS)])
    root[:, 2] = 0.9
    return root, np.tile(_yaw_quaternion(heading_deg), (STEPS, 1)), np.ones((STEPS, 2))


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 135.0, 270.0])
def test_forward_travel_is_independent_of_world_axis(heading_deg: float):
    root, quaternion, command = _straight_walk(heading_deg)
    travel = root_displacement_in_body_frame(root, quaternion, command)
    assert travel["forward_travel_m"] == pytest.approx(WALK_M, rel=1e-6)
    assert travel["lateral_travel_abs_m"] == pytest.approx(0.0, abs=1e-9)
    assert travel["lateral_drift_max_abs_m"] == pytest.approx(0.0, abs=1e-9)
    assert travel["reset_steps"] == 0


def test_a_y_facing_walk_is_not_reported_as_lateral_drift():
    """The exact pre-fix failure: at heading 90 forward is world +Y, so the old
    world-x labelling put the whole walk into the lateral bucket."""

    root, quaternion, command = _straight_walk(90.0)
    travel = root_displacement_in_body_frame(root, quaternion, command)
    assert travel["lateral_travel_abs_m"] < 1e-9, "walk along +Y must not be lateral"
    assert travel["forward_travel_m"] == pytest.approx(WALK_M, rel=1e-6)
    assert np.ptp(root[:, 1]) == pytest.approx(WALK_M, rel=1e-6)
    assert np.ptp(root[:, 0]) == pytest.approx(0.0, abs=1e-9)


def _pre_fix_world_axis_projection(root: np.ndarray) -> dict[str, float]:
    """The old ``arm_capture`` rule, kept verbatim: world ``x`` is "forward",
    world ``y`` is "lateral drift"."""

    delta = root - root[0]
    return {
        "x_max_m": float(delta[:, 0].max()),
        "x_min_m": float(delta[:, 0].min()),
        "lateral_drift_max_m": float(np.abs(delta[:, 1]).max()),
    }


def test_world_axis_labelling_would_fail_the_positive_control():
    """Positive control: the pre-fix rule must fail on the very fixture the fixed
    rule passes. If it did not, the suite above would be green regardless of the
    defect and would protect nothing."""

    root, _quaternion, _command = _straight_walk(90.0)
    old = _pre_fix_world_axis_projection(root)
    # The old rule calls a dead-straight walk 1.2 m of lateral drift ...
    assert old["lateral_drift_max_m"] == pytest.approx(WALK_M, rel=1e-6)
    # ... and reports no forward progress at all, while the fixed rule reports the
    # full walk forward and no drift.
    assert abs(old["x_max_m"]) < 1e-9
    fixed = root_displacement_in_body_frame(
        root, np.tile(_yaw_quaternion(90.0), (STEPS, 1)), np.ones((STEPS, 2))
    )
    assert fixed["lateral_drift_max_abs_m"] < 1e-9
    assert fixed["forward_travel_m"] == pytest.approx(WALK_M, rel=1e-6)


def test_cycle_reset_is_not_counted_as_motion():
    """A rewind teleports the character back to spawn; that step is not travel.

    The fixture walks for half the run, then the second half is a fresh spawn 5 m
    away. Exactly one step is the teleport, and it must not contribute to either
    bucket -- otherwise a 1.2 m walk would be reported as a 5 m leap.
    """

    root, quaternion, command = _straight_walk(0.0)
    teleported = root.copy()
    teleported[STEPS // 2 :, 0] += 5.0
    travel = root_displacement_in_body_frame(teleported, quaternion, command)
    assert travel["reset_steps"] == 1
    assert travel["lateral_travel_abs_m"] < 1e-9
    # Only the walking inside each half counts. The teleported step itself is
    # excluded, so the total is the full walk minus exactly that one step.
    assert travel["forward_travel_abs_m"] == pytest.approx(WALK_M - STEP_M, rel=1e-6)
    assert travel["forward_travel_abs_m"] < 5.0
    assert travel["forward_max_m"] < 5.0


def test_backward_walk_reports_negative_forward_travel():
    """Pressing S is negative forward travel: the body still faces +X while the
    motion goes -X, so the projection onto the body forward axis is negative.

    This is the distinction a sign normalisation would erase, and it is the only
    place the sign carries information -- a reversal would otherwise look exactly
    like a forward walk of the same length.
    """

    root, quaternion, command = _backpedal(0.0)
    travel = root_displacement_in_body_frame(root, quaternion, command)
    assert travel["forward_travel_m"] == pytest.approx(-WALK_M, rel=1e-6)
    assert travel["forward_travel_abs_m"] == pytest.approx(WALK_M, rel=1e-6)
    assert travel["lateral_travel_abs_m"] < 1e-9


def test_normalisation_would_hide_a_reversal():
    """Guard against reintroducing a sign normalisation.

    An earlier draft normalised the forward sign to the net direction so that "W
    is always positive". That makes a forward walk and a backpedal report the
    *same* number, so a reversal -- the thing this metric is for -- becomes
    invisible. The two motions must differ in sign.
    """

    ahead, quaternion_ahead, command = _straight_walk(0.0)
    astern, quaternion_astern, _ = _backpedal(0.0)
    forward = root_displacement_in_body_frame(ahead, quaternion_ahead, command)
    backward = root_displacement_in_body_frame(astern, quaternion_astern, command)
    assert forward["forward_travel_m"] > 0
    assert backward["forward_travel_m"] < 0
    assert forward["forward_travel_abs_m"] == pytest.approx(
        backward["forward_travel_abs_m"], rel=1e-6
    )


def test_facing_away_still_counts_as_forward():
    """A character facing -X and walking -X is walking *forward*, just in the other
    direction of the room. The decomposition must say so, otherwise every heading
    beyond 90 deg would read as a backpedal."""

    root, quaternion, command = _straight_walk(180.0)
    travel = root_displacement_in_body_frame(root, quaternion, command)
    assert travel["forward_travel_m"] == pytest.approx(WALK_M, rel=1e-6)
    assert travel["lateral_travel_abs_m"] < 1e-9


def test_pure_lateral_motion_is_not_called_forward():
    """A sidestep must land in the lateral bucket -- the mirror of the defect."""

    heading = np.deg2rad(0.0)
    sideways = np.array([-np.sin(heading), np.cos(heading), 0.0]) * STEP_M
    root = np.array([sideways * i for i in range(STEPS)])
    root[:, 2] = 0.9
    travel = root_displacement_in_body_frame(
        root, np.tile(_yaw_quaternion(0.0), (STEPS, 1)), np.ones((STEPS, 2))
    )
    assert abs(travel["forward_travel_m"]) < 1e-9
    assert travel["lateral_travel_abs_m"] == pytest.approx(WALK_M, rel=1e-6)


def test_rejects_nonfinite_and_wrong_shape():
    """Bad input must fail loudly: a silently empty decomposition would read as
    "no motion" and hide a real stumble."""

    good = np.zeros((4, 3))
    good[:, 2] = 0.9
    quaternion = np.tile(_yaw_quaternion(0.0), (4, 1))
    with pytest.raises(ValueError):
        root_displacement_in_body_frame(np.zeros((4, 2)), quaternion)
    with pytest.raises(ValueError):
        root_displacement_in_body_frame(good, quaternion[:3])
    poisoned = good.copy()
    poisoned[0, 0] = np.nan
    with pytest.raises(ValueError):
        root_displacement_in_body_frame(poisoned, quaternion)


def test_too_few_samples_does_not_raise():
    root = np.zeros((1, 3))
    travel = root_displacement_in_body_frame(root, np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert travel["sample_count"] == 1
    assert "forward_travel_m" not in travel
