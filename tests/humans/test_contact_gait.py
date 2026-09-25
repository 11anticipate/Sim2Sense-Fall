"""World contact-velocity continuity and support schedule, independent of mocap."""

from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.contact_gait import ContactGaitConfig, foot_path


def test_contact_endpoints_have_zero_world_horizontal_velocity():
    config = ContactGaitConfig()
    h = 1e-6
    for phase in (0., config.stance_fraction):
        dx = (foot_path(phase+h, config)[0] - foot_path(phase-h, config)[0])/(2*h)
        assert dx + config.cycle_distance_m == pytest.approx(0., abs=1e-6)
        dz = (foot_path(phase+h, config)[1] - foot_path(phase-h, config)[1])/(2*h)
        assert dz == pytest.approx(0., abs=1e-6)


def test_alternating_schedule_has_no_flight_and_lifts_each_swing():
    config = ContactGaitConfig()
    phases = np.linspace(0., 1., 1000)
    paths = [[foot_path(p-offset, config) for p in phases] for offset in (0., .5)]
    assert all(left[2] or right[2] for left, right in zip(*paths, strict=True))
    for path in paths:
        assert max(p[1] for p in path) == pytest.approx(config.clearance_m, abs=1e-5)
        assert all(p[1] >= 0 and (not p[2] or p[1] == 0) for p in path)


@pytest.mark.parametrize("kwargs", [{"stance_fraction": .4}, {"clearance_m": -1},
                                    {"cycle_distance_m": float("nan")}])
def test_invalid_contact_trajectory_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ContactGaitConfig(**kwargs)


def test_stance_goal_stays_in_world_when_root_moves():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.contact_gait import ContactFootPlanner
    from sim2sense_fall.humans.rig import LinkTransform, forward_kinematics, plan_human_rig

    path = Path(__file__).resolve().parents[2] / "configs/humans/human_smpl_multiaxis.yaml"
    plan = plan_human_rig(load_human_config(path))
    planner = ContactFootPlanner(plan, ContactGaitConfig(), .4, .8)
    q = np.zeros(len(plan.joints))
    pos = np.array([0., 0., plan.ground_offset_m-.03])
    actual = forward_kinematics(plan, root_position=(0., 0., plan.ground_offset_m))
    original = None
    for _ in range(10):
        q = planner.correct(q, LinkTransform(np.eye(3), pos), actual,
                            heading=0., turn_rate=0., speed=0., command=0., dt_s=1/120)
        feet = {s: planner.solvers[s].poses(q, LinkTransform(np.eye(3), pos))[f"{s}_ankle"]
                for s in ("left", "right")}
        if original is None:
            original = {s: p.translation.copy() for s, p in feet.items()}
        for side in feet:
            np.testing.assert_allclose(feet[side].translation, original[side], atol=1e-4)
        pos[0] += .001
    assert not planner.swings


def test_early_stop_replans_landing_and_reverse_lifts_rearmost_foot():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.contact_gait import ContactFootPlanner
    from sim2sense_fall.humans.rig import LinkTransform, forward_kinematics, plan_human_rig

    path = Path(__file__).resolve().parents[2] / "configs/humans/human_smpl_multiaxis.yaml"
    plan = plan_human_rig(load_human_config(path))
    planner = ContactFootPlanner(plan, ContactGaitConfig(), .4, .8)
    q = np.zeros(len(plan.joints))
    root = LinkTransform(np.eye(3), np.array([0., 0., plan.ground_offset_m-.03]))
    actual = forward_kinematics(plan, root_position=(0., 0., plan.ground_offset_m))
    args = {"heading": 0., "turn_rate": 0., "speed": .01, "dt_s": 1/120}
    planner.correct(q, root, actual, command=1., **args)
    side = next(iter(planner.swings))
    old_end = planner.swings[side][1].translation.copy()
    planner.correct(q, root, actual, command=0., **args)
    assert planner.swings[side][1].translation[0] < old_end[0] - .1
    planner.swings.clear()
    planner.anchors["left"].translation[0] = .2
    planner.anchors["right"].translation[0] = -.1
    planner.correct(q, root, actual, command=-1., **args)
    assert set(planner.swings) == {"left"}


@pytest.mark.parametrize("change", ["legacy_stance", "missing_cycle", "mismatched_cycle"])
def test_config_rejects_inconsistent_contact_controllers(tmp_path, change):
    import yaml

    from sim2sense_fall.humans.teleop import load_keyboard_config

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/humans/keyboard.yaml").read_text())
    if change == "legacy_stance":
        config["stance"]["enabled"] = True
    elif change == "missing_cycle":
        del config["gaits"]["backward"]["contact_cycle"]
    else:
        config["gaits"]["forward"]["contact_cycle"]["cycle_distance_m"] = .8
    path = tmp_path / "keyboard.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="contact"):
        load_keyboard_config(path, root)


def test_stance_solution_ignores_discontinuous_mocap_leg_seed():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.contact_gait import ContactFootPlanner
    from sim2sense_fall.humans.rig import LinkTransform, forward_kinematics, plan_human_rig

    path = Path(__file__).resolve().parents[2] / "configs/humans/human_smpl_multiaxis.yaml"
    plan = plan_human_rig(load_human_config(path))
    planner = ContactFootPlanner(plan, ContactGaitConfig(), .4, .8)
    root = LinkTransform(np.eye(3), np.array([0., 0., plan.ground_offset_m-.03]))
    actual = forward_kinematics(plan, root_position=(0., 0., plan.ground_offset_m))
    args = {"heading": 0., "turn_rate": 0., "speed": 0., "command": 0., "dt_s": 1/120}
    q = planner.correct(np.zeros(len(plan.joints)), root, actual, **args)
    ref = q.copy()
    for solver in planner.solvers.values():
        ref[solver.indices] += .2
    updated = planner.correct(ref, root, actual, **args)
    np.testing.assert_allclose(updated, q, atol=1e-4)
    assert planner.last_residual_m < 1e-4


def test_turn_reach_guard_lifts_rear_foot_before_clock_phase():
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.contact_gait import ContactFootPlanner
    from sim2sense_fall.humans.rig import LinkTransform, forward_kinematics, plan_human_rig

    path = Path(__file__).resolve().parents[2] / "configs/humans/human_smpl_multiaxis.yaml"
    plan = plan_human_rig(load_human_config(path))
    planner = ContactFootPlanner(plan, ContactGaitConfig(), .4, .8)
    root = LinkTransform(np.eye(3), np.array([0., 0., plan.ground_offset_m-.03]))
    actual = forward_kinematics(plan, root_position=(0., 0., plan.ground_offset_m))
    args = {"heading": 0., "turn_rate": .5, "speed": .4, "dt_s": 1/120}
    q = planner.correct(np.zeros(len(plan.joints)), root, actual, command=0., **args)
    planner.phase = .95  # Neither foot's scheduled swing window is open.
    planner.last_command = 1.
    hip = root.transform_point(planner.neutral["left_hip"].translation)
    anchor = planner.anchors["left"].translation
    vertical = anchor[2]-hip[2]
    distance = planner.config.reach_fraction*planner.leg_lengths["left"]
    anchor[0] = hip[0]-np.sqrt(distance**2-vertical**2)-.001
    planner.correct(q, root, actual, command=1., **args)
    assert set(planner.swings) == {"left"}
    assert planner.phase < .01
