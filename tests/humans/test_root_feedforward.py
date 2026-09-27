"""Reference-COM feedforward term of the pelvis actuator."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from sim2sense_fall.humans.root_control import (
    RootAssistConfig,
    blend_assist_configs,
    root_wrench,
)

CONFIG = RootAssistConfig(
    position_stiffness_n_m=0.0,
    position_damping_ns_m=0.0,
    rotation_stiffness_nm_rad=0.0,
    rotation_damping_nms_rad=0.0,
    max_force_n=1e6,
    max_torque_nm=1e6,
    gravity_compensation_fraction=0.6,
    max_vertical_lift_fraction_of_weight=1.0,
)
STATE = dict(
    position=np.zeros(3),
    quaternion=np.array([0., 0., 0., 1.]),
    linear_velocity=np.zeros(3),
    angular_velocity=np.zeros(3),
    target_position=np.zeros(3),
    target_quaternion=np.array([0., 0., 0., 1.]),
    target_linear_velocity=np.zeros(3),
    target_angular_velocity=np.zeros(3),
    mass_kg=76.0,
    gravity_m_s2=9.81,
)


def test_zero_feedforward_keeps_legacy_behaviour():
    force, _ = root_wrench(CONFIG, **STATE)
    np.testing.assert_allclose(force, [0., 0., 0.6 * 76 * 9.81], atol=1e-9)
    force_ff, _ = root_wrench(CONFIG, feedforward_accel_m_s2=np.zeros(3), **STATE)
    np.testing.assert_allclose(force_ff, force, atol=1e-9)


def test_feedforward_follows_reference_acceleration():
    accel = np.array([0.5, -0.2, 1.2])
    force, _ = root_wrench(CONFIG, feedforward_accel_m_s2=accel, **STATE)
    # Horizontal in full; vertical under the declared gravity fraction.
    np.testing.assert_allclose(force[:2], 76.0 * accel[:2], atol=1e-9)
    np.testing.assert_allclose(force[2], 0.6 * 76.0 * (9.81 + accel[2]), atol=1e-9)


def test_feedforward_respects_vertical_lift_cap():
    accel = np.array([0., 0., 30.0])
    force, _ = root_wrench(CONFIG, feedforward_accel_m_s2=accel, **STATE)
    assert force[2] <= CONFIG.max_vertical_lift_fraction_of_weight * 76 * 9.81 + 1e-9


def test_feedforward_rejects_malformed_acceleration():
    with pytest.raises(ValueError, match="feedforward"):
        root_wrench(CONFIG, feedforward_accel_m_s2=np.zeros(2), **STATE)
    with pytest.raises(ValueError, match="feedforward"):
        root_wrench(CONFIG, feedforward_accel_m_s2=np.array([np.nan, 0., 0.]), **STATE)


@dataclass(frozen=True, slots=True)
class OtherProfile:
    """A profile with a different field set, to pin the mismatch guard."""

    gravity_compensation_fraction: float


RECOVERY = RootAssistConfig(
    position_stiffness_n_m=12000.0, position_damping_ns_m=900.0,
    rotation_stiffness_nm_rad=3000.0, rotation_damping_nms_rad=180.0,
    max_force_n=1500.0, max_torque_nm=400.0,
    gravity_compensation_fraction=0.45, max_vertical_lift_fraction_of_weight=0.55,
)
SHIP = replace(CONFIG, position_stiffness_n_m=12000.0, position_damping_ns_m=900.0,
               rotation_stiffness_nm_rad=3000.0, rotation_damping_nms_rad=180.0,
               max_force_n=1500.0, max_torque_nm=400.0)


def test_assist_blend_hits_both_endpoints_exactly():
    assert blend_assist_configs(SHIP, RECOVERY, 0.0) == SHIP
    assert blend_assist_configs(SHIP, RECOVERY, 1.0) == RECOVERY


def test_assist_blend_is_linear_and_keeps_the_lift_cap_invariant():
    middle = blend_assist_configs(SHIP, RECOVERY, 0.5)
    assert middle.gravity_compensation_fraction == pytest.approx(0.525)
    assert middle.max_vertical_lift_fraction_of_weight == pytest.approx(0.775)
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = blend_assist_configs(SHIP, RECOVERY, weight)
        assert blended.max_vertical_lift_fraction_of_weight >= blended.gravity_compensation_fraction


def test_assist_blend_rejects_bad_weights_and_mismatched_profiles():
    with pytest.raises(ValueError, match="blend weight"):
        blend_assist_configs(SHIP, RECOVERY, 1.5)
    with pytest.raises(ValueError, match="blend weight"):
        blend_assist_configs(SHIP, RECOVERY, float("nan"))
    with pytest.raises(ValueError, match="same fields"):
        blend_assist_configs(SHIP, OtherProfile(1.0), 0.5)
