"""Reject suspended feet and activity failures hidden by idle contact samples."""

import numpy as np

from sim2sense_fall.humans.quality import MotionQualityConfig, motion_quality


def measured_session():
    rows = []
    for i in range(100):
        rows.append({
            "time_s": i * .01, "mode": "stand" if i < 90 else "forward",
            "force": np.array([0., 0., 100.]), "joints": np.zeros(2),
            "joint_target": np.zeros(2), "floor_contact_slips_m_s": [.01],
            "contact_detail": [{
                "collider0_path": "/World/Human/left_ankle/collider",
                "collider1_path": "/World/rooms/living_room/floor",
                "normal": [0., 0., 1.], "impulse_ns": [0., 0., 5.],
                "impulse_magnitude_ns": 5.,
            }],
        })
    return rows


def evaluate(rows, height=.003):
    return motion_quality(rows, np.arange(100)*.01, np.full((100, 2), height),
                          config=MotionQualityConfig(), dt_s=.01,
                          mass_kg=72., gravity_m_s2=9.81)


def test_collider_contact_does_not_accept_floating_skin():
    rows = measured_session()
    assert evaluate(rows)["accepted"]
    result = evaluate(rows, height=.025)
    assert not result["accepted"]
    assert not result["modes"]["forward"]["gates"]["contact_skin_grounded"]


def test_standing_cannot_dilute_walking_slip():
    rows = measured_session()
    for row in rows[90:]:
        row["floor_contact_slips_m_s"] = [.4]
    result = evaluate(rows)
    assert result["modes"]["stand"]["accepted"]
    assert not result["modes"]["forward"]["gates"]["floor_slip"]


def test_skin_at_floor_without_support_is_not_physical_contact():
    rows = measured_session()
    for row in rows[90:]:
        row["contact_detail"] = []
    assert not evaluate(rows)["modes"]["forward"]["gates"]["continuous_support"]


def test_short_walking_turn_cannot_be_hidden_by_straight_motion():
    rows = measured_session()
    for row in rows:
        row["mode"] = "forward"
        row["command"] = (1., 0.)
    rows[-1]["command"] = (1., 1.)
    rows[-1]["floor_contact_slips_m_s"] = [.5]
    result = evaluate(rows)
    assert result["modes"]["forward"]["gates"]["floor_slip"]
    assert not result["modes"]["forward_turn_left"]["gates"]["floor_slip"]
    assert not result["accepted"]
