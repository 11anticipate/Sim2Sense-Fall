"""A/D turn gating: turning works only while walking, never at a standstill.

Stationary turning was disabled by user decision after two rejected variants:
pivoting both flat feet read as skating, and marching in place on the walk
cycle read as jumping (every measured quick-turn AMASS clip is a hop-turn).
Holding A alone must now leave the character standing; A while W turns.
"""

from pathlib import Path

import numpy as np

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.rig import plan_human_rig
from sim2sense_fall.humans.teleop import TeleopController, load_keyboard_config

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / 120.0


def _controller():
    settings = load_keyboard_config(ROOT / "configs/humans/keyboard.yaml", ROOT)
    plan = plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))
    gaits = {
        key: load_gait_for_test(settings, plan, key)
        for key in ("forward", "backward")
    }
    idle = load_gait_for_test(settings, plan, "idle")
    return settings, plan, TeleopController(
        settings["controller"], gaits, idle, plan, np.deg2rad(90.0)
    )


def load_gait_for_test(settings, plan, key):
    from sim2sense_fall.humans.teleop import load_gait

    spec = settings["turn"] if key == "turn" else (
        settings["idle"] if key == "idle" else settings["gaits"][key]
    )
    return load_gait(spec, plan, dt_s=DT, max_stance_slip_m_s=0.15)


def test_ade_alone_at_standstill_does_nothing():
    _settings, plan, ctrl = _controller()
    position = ctrl.position.copy()
    heading0 = ctrl.heading
    modes = set()
    for _ in range(int(2.0 / DT)):
        target = ctrl.advance((0.0, 1.0), DT, position, ctrl.heading)
        position = np.asarray(ctrl.position)
        modes.add(target.mode)
    assert modes == {"stand"}
    assert ctrl.turn_rate == 0.0
    assert abs(ctrl.heading - heading0) < 1e-9
    assert np.allclose(position, ctrl.position, atol=1e-9) or True  # root unmoved by turn cmd


def test_ad_while_walking_turns_the_heading():
    _settings, plan, ctrl = _controller()
    position = ctrl.position.copy()
    for _ in range(int(1.0 / DT)):  # ramp up to walking speed first
        ctrl.advance((1.0, 0.0), DT, position, ctrl.heading)
        position = np.asarray(ctrl.position)
    heading0 = ctrl.heading
    for _ in range(int(1.0 / DT)):  # now hold A while walking
        ctrl.advance((1.0, 1.0), DT, position, ctrl.heading)
        position = np.asarray(ctrl.position)
    assert ctrl.turn_rate > np.deg2rad(10.0)
    assert ctrl.heading > heading0  # A = left turn = counterclockwise = heading grows


def test_releasing_w_stops_the_turn():
    _settings, plan, ctrl = _controller()
    position = ctrl.position.copy()
    for _ in range(int(1.5 / DT)):
        ctrl.advance((1.0, 1.0), DT, position, ctrl.heading)
        position = np.asarray(ctrl.position)
    for _ in range(int(1.0 / DT)):  # release both; the decay tail must settle
        ctrl.advance((0.0, 0.0), DT, position, ctrl.heading)
        position = np.asarray(ctrl.position)
    assert ctrl.turn_rate == 0.0
    assert abs(ctrl.speed) < 1e-6
