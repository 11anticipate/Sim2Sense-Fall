"""CPU regression tests for the R2 findings: joint-drive targets and drive scaling.

Both bugs let a physics check pass without exercising what it claimed to check, so both
are pinned here without Isaac Sim: the target amplitude (a doubly-converted radian value
shrank a 75-degree knee command to 1.3 degrees, inside the 15-degree tolerance) and the
gain scaling (multiplying the *current* gains compounded towards zero).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.rig import plan_human_rig
from sim2sense_fall.humans.usd_human import HumanRuntime, tracking_target_rad

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "humans" / "human_smpl_neutral.yaml"


def _rad(degrees: float) -> float:
    return float(np.radians(degrees))


# ---------------------------------------------------------------------------
# tracking_target_rad
# ---------------------------------------------------------------------------


def test_knee_mid_range_target_is_seventy_five_degrees() -> None:
    """The exact number the stage-7 review expected: 75 deg, not 1.309 deg."""

    target = tracking_target_rad(_rad(0.0), _rad(150.0))
    assert np.degrees(target) == pytest.approx(75.0)


def test_asymmetric_limits_use_the_wider_side_not_the_midpoint() -> None:
    # The ankle's [-35, 45] midpoints to 5 deg, which no tolerance can discriminate.
    assert np.degrees(tracking_target_rad(_rad(-35.0), _rad(45.0))) == pytest.approx(22.5)
    assert np.degrees(tracking_target_rad(_rad(-120.0), _rad(30.0))) == pytest.approx(-60.0)
    assert np.degrees(tracking_target_rad(_rad(-170.0), _rad(40.0))) == pytest.approx(-85.0)


def test_target_stays_inside_the_declared_limits() -> None:
    for low, high in ((-35.0, 45.0), (0.0, 150.0), (-120.0, 30.0), (-25.0, 60.0)):
        target = tracking_target_rad(_rad(low), _rad(high))
        assert _rad(low) <= target <= _rad(high)


@pytest.mark.parametrize(
    ("low", "high"),
    [(np.nan, 1.0), (0.0, np.inf), (_rad(45.0), _rad(-45.0))],
)
def test_rejects_unusable_limit_pairs(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="finite and ordered"):
        tracking_target_rad(low, high)


def test_every_configured_dof_clears_the_tracking_tolerance() -> None:
    """A target below the tolerance makes the acceptance check unfalsifiable.

    This is the regression guard for the double ``np.radians()`` conversion: it is what
    took the largest commanded joint down to 1.309 degrees against a 15 degree gate.
    """

    config = load_human_config(CONFIG)
    plan = plan_human_rig(config)
    tolerance = float(config.control.tracking_tolerance_deg)
    amplitudes = {
        joint.name: abs(
            np.degrees(tracking_target_rad(_rad(joint.lower_deg), _rad(joint.upper_deg)))
        )
        for joint in plan.joints
    }
    assert amplitudes, "the plan exposes no degrees of freedom to drive"
    weakest = min(amplitudes, key=lambda name: amplitudes[name])
    assert amplitudes[weakest] > tolerance, (
        f"{weakest} would be commanded to {amplitudes[weakest]:.2f} deg, which the "
        f"{tolerance:.1f} deg tolerance cannot falsify"
    )


def test_widening_clears_the_tolerance_on_a_small_span_axis() -> None:
    """A three-axis chain has small secondary spans a single-axis rig never had.

    ``spine3``'s secondary axis spans ``[-25, 25]`` degrees, so half the wider side is
    12.5 degrees -- inside the 15 degree tolerance, which made the verification's
    falsifiability gate fail on the multi-axis rig. Widening until the amplitude
    clears ``min_amplitude`` is the fix; the target must still stop short of the
    limit so the drive has room to settle.
    """

    low, high = _rad(-25.0), _rad(25.0)
    target = tracking_target_rad(low, high, min_amplitude=_rad(18.75))
    amplitude = abs(np.degrees(target))
    assert amplitude == pytest.approx(18.75)
    margin = 0.05 * (high - low)
    assert low + margin <= target <= high - margin

    # An already-large target is left where it was.
    assert tracking_target_rad(_rad(0.0), _rad(150.0), min_amplitude=_rad(18.75)) == (
        pytest.approx(_rad(75.0))
    )


# ---------------------------------------------------------------------------
# Drive scaling, against a duck-typed articulation view
# ---------------------------------------------------------------------------


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeArticulation:
    """Records what gains were written, so compounding becomes visible."""

    def __init__(self, stiffness: np.ndarray, damping: np.ndarray) -> None:
        self.stiffness = np.asarray(stiffness, dtype=np.float64)
        self.damping = np.asarray(damping, dtype=np.float64)
        self.written: list[tuple[np.ndarray, np.ndarray]] = []

    def get_dof_gains(self) -> tuple[_Tensor, _Tensor]:
        return _Tensor(self.stiffness), _Tensor(self.damping)

    def set_dof_gains(self, stiffness: Any, damping: Any) -> None:
        self.stiffness = np.asarray(stiffness, dtype=np.float64)
        self.damping = np.asarray(damping, dtype=np.float64)
        self.written.append((self.stiffness.copy(), self.damping.copy()))


def _runtime(stiffness: float = 250.0, damping: float = 50.0, dof: int = 14) -> SimpleNamespace:
    return SimpleNamespace(
        articulation=_FakeArticulation(np.full(dof, stiffness), np.full(dof, damping)),
        _base_gains=None,
        control_scale=1.0,
    )


def test_scaling_is_relative_to_the_original_gains_not_the_current_ones() -> None:
    """Applied every frame, a 0.5 scale must stay 0.5 rather than decay to nothing."""

    runtime = _runtime()
    for _ in range(180):
        assert HumanRuntime.set_control_scale(runtime, 0.5)
    np.testing.assert_allclose(runtime.articulation.stiffness, np.full(14, 125.0))
    np.testing.assert_allclose(runtime.articulation.damping, np.full(14, 50.0))
    assert runtime.control_scale == pytest.approx(0.5)


def test_zeroing_and_restoring_the_scale_returns_the_original_gains() -> None:
    """The R2 negative control needs drives off and then back on, exactly."""

    runtime = _runtime(stiffness=250.0, damping=50.0)
    original = runtime.articulation.stiffness.copy()
    assert HumanRuntime.set_control_scale(runtime, 0.0)
    np.testing.assert_array_equal(runtime.articulation.stiffness, np.zeros(14))
    assert HumanRuntime.set_control_scale(runtime, 1.0)
    np.testing.assert_allclose(runtime.articulation.stiffness, original)
    np.testing.assert_allclose(runtime.articulation.damping, np.full(14, 50.0))


def test_zero_scale_keeps_the_passive_damping() -> None:
    """A body that lost active control is a damped ragdoll, not a frictionless one.

    The verification's gravity control and the drives-off negative control both run
    at scale 0. With damping zeroed too, the 57-DOF multi-axis ragdoll folded
    ballistically (975 deg/s within the 0.2 s hold, 2864 deg/s mid-fall) and its
    floor impact NaN'd the PhysX solver -- the "non-finite world position for link
    'pelvis'" that kept the multi-axis verification red. Damping only opposes
    velocity: it cannot track a target and cannot hold a pose, so both controls
    stay honest with it retained.
    """

    runtime = _runtime(stiffness=250.0, damping=50.0)
    assert HumanRuntime.set_control_scale(runtime, 0.0)
    np.testing.assert_array_equal(runtime.articulation.stiffness, np.zeros(14))
    np.testing.assert_allclose(runtime.articulation.damping, np.full(14, 50.0))
    # Repeated calls must not compound or erode the damping either.
    for _ in range(10):
        assert HumanRuntime.set_control_scale(runtime, 0.0)
    np.testing.assert_allclose(runtime.articulation.damping, np.full(14, 50.0))


@pytest.mark.parametrize("scale", [-0.1, 1.5])
def test_out_of_range_scale_is_refused(scale: float) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        HumanRuntime.set_control_scale(_runtime(), scale)


def test_scaling_reports_failure_instead_of_claiming_it_was_applied() -> None:
    class _Refusing:
        def get_dof_gains(self) -> None:
            raise RuntimeError("no gain API on this build")

    runtime = SimpleNamespace(articulation=_Refusing(), _base_gains=None, control_scale=1.0)
    assert HumanRuntime.set_control_scale(runtime, 0.5) is False


def test_dof_permutation_round_trips_a_shuffled_articulation_order() -> None:
    """PhysX keeps multi-axis chains in traversal order, not plan order.

    The verify probes commanded left_knee and actually bent spine2 -- both probes
    were indexed positionally across that boundary. The permutation is what the
    runtime now applies on every read and write; this pins its round-trip.
    """

    from sim2sense_fall.humans.usd_human import dof_permutation

    plan_names = (
        "left_hip__dof1",
        "left_hip__dof2",
        "left_hip",
        "right_hip__dof1",
        "right_hip__dof2",
        "right_hip",
        "spine1__dof1",
        "spine1__dof2",
        "spine1",
    )
    # PhysX interleaves the chains breadth-first.
    runtime_names = (
        "left_hip__dof1",
        "right_hip__dof1",
        "spine1__dof1",
        "left_hip__dof2",
        "right_hip__dof2",
        "spine1__dof2",
        "left_hip",
        "right_hip",
        "spine1",
    )
    permutation = dof_permutation(plan_names, runtime_names)
    plan_values = np.arange(len(plan_names), dtype=np.float64) * 3.0

    to_runtime = plan_values[permutation]
    assert to_runtime[0] == plan_values[0], "left_hip__dof1 leads both orders"
    assert to_runtime[1] == plan_values[3], "slot 1 is right_hip__dof1, not left_hip__dof2"

    back = np.empty_like(to_runtime)
    back[permutation] = to_runtime
    np.testing.assert_allclose(back, plan_values)

    with pytest.raises(Exception, match="plan does not declare"):
        dof_permutation(plan_names, (*runtime_names[:8], "phantom_dof"))
