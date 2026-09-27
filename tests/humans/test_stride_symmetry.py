"""Stride-symmetry analysis on synthetic traces, plus a real-rig FK sanity check.

The synthetic traces encode a known limp (a constant fore-aft offset between the
feet) and a known direction, so the verdicts here check the arithmetic rather
than restate it. The backward-direction case is a pinned regression: the phase
of a backward walk *decreases*, and the first version of the splitter treated
every such step as a phase wrap and scored nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

AUDIT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "humans"
sys.path.insert(0, str(AUDIT_DIR))

from audit_stride_symmetry import (  # noqa: E402
    cycle_stats,
    foot_fore_aft,
    load_gate,
    walking_segments,
)

from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.rig import plan_human_rig  # noqa: E402
from sim2sense_fall.humans.rotations import (  # noqa: E402
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    rotation_about_axis,
)

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / 120.0


def synthetic_walk(offset_right_m: float, forward: bool, frames: int, step: float = 0.003):
    """Fore-aft traces of two antiphase legs with a constant right-foot offset."""

    phase_unwrapped = np.arange(frames) * step
    if not forward:
        phase_unwrapped = phase_unwrapped[::-1]
    phase = phase_unwrapped % 1.0
    left = 0.30 * np.sin(2 * np.pi * phase_unwrapped)
    right = -0.30 * np.sin(2 * np.pi * phase_unwrapped) - offset_right_m
    return np.stack([left, right], axis=1), phase


def test_cycle_scores_forward_whole_cycles_and_reads_the_offset():
    fore_aft, phase = synthetic_walk(offset_right_m=0.05, forward=True, frames=720)
    frames = np.arange(720)
    cycles = cycle_stats(fore_aft, frames, phase, 0.98)
    assert len(cycles) == 2
    for cycle in cycles:
        # A cycle runs between wrap boundaries, one playback step short of a full
        # period, so the peak-to-peak of a 0.30 amplitude sinusoid loses
        # 0.30*2*pi*0.003 ~ 5.7 mm. The constant offset is unaffected.
        assert cycle["stride_left_m"] == pytest.approx(0.60, abs=1e-2)
        assert cycle["stride_right_m"] == pytest.approx(0.60, abs=1e-2)
        assert cycle["stagger_m"] == pytest.approx(0.05, abs=1e-4)


def test_cycle_scores_backward_decreasing_phase_without_falsifying_wraps():
    fore_aft, phase = synthetic_walk(offset_right_m=0.05, forward=False, frames=720)
    frames = np.arange(720)
    cycles = cycle_stats(fore_aft, frames, phase, 0.98)
    assert len(cycles) == 2
    assert cycles[0]["stagger_m"] == pytest.approx(0.05, abs=1e-4)
    assert cycles[0]["stride_left_m"] == pytest.approx(0.60, abs=1e-2)


def test_cycle_refuses_partial_cycles():
    fore_aft, phase = synthetic_walk(offset_right_m=0.05, forward=True, frames=300)
    assert cycle_stats(fore_aft, np.arange(300), phase, 0.98) == []
    fore_aft, phase = synthetic_walk(offset_right_m=0.05, forward=True, frames=100)
    assert cycle_stats(fore_aft, np.arange(100), phase, 0.98) == []


def _walking_control(frames: int, direction: str, weight: float, start: float) -> dict:
    time_s = start + np.arange(frames) * DT
    return {
        "mode": np.asarray([direction] * frames),
        "gait_weight": np.full(frames, weight),
        "time_s": time_s,
        "root": np.stack([np.array([0.0, 0.4 * (t - time_s[0]), 0.9]) for t in time_s]),
    }


def test_walking_segments_split_on_reset_teleport_and_direction():
    control = _walking_control(60, "forward", 1.0, 0.0)
    second = _walking_control(60, "forward", 1.0, 60 * DT)
    second["root"] = second["root"] + np.array([1.2, 0.0, 0.0])  # reset teleport
    third = _walking_control(40, "backward", 1.0, 120 * DT)
    merged = {
        key: np.concatenate([control[key], second[key], third[key]])
        for key in ("mode", "gait_weight", "time_s", "root")
    }
    segments = walking_segments(merged, 0.95)
    assert [(name, len(frames)) for name, frames in segments] == [
        ("forward", 60),
        ("forward", 60),
        ("backward", 40),
    ]


def test_walking_segments_drop_low_weight_frames():
    control = _walking_control(90, "forward", 1.0, 0.0)
    control["gait_weight"][20:60] = 0.5  # ramping in/out, not walking yet
    segments = walking_segments(control, 0.95)
    assert [(name, len(frames)) for name, frames in segments] == [("forward", 20), ("forward", 30)]


def test_gate_loader_requires_every_threshold(tmp_path):
    path = tmp_path / "gate.yaml"
    path.write_text("max_stagger_m: 0.05\nmin_stride_ratio: 0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_stride_ratio"):
        load_gate(path)
    good = tmp_path / "good.yaml"
    good.write_text(
        "max_stagger_m: 0.05\nmin_stride_ratio: 0.9\nmax_stride_ratio: 1.1\n"
        "weight_floor: 0.95\nmin_cycle_fraction: 0.98\n",
        encoding="utf-8",
    )
    assert load_gate(good)["max_stagger_m"] == 0.05


def test_foot_fore_aft_is_symmetric_at_rest_and_invariant_to_yaw():
    plan = plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))
    zeros = np.zeros(len(plan.dof_names))
    root = np.asarray(plan.spawn_root_position, dtype=float)
    traces = []
    for heading in (0.0, np.pi / 2):
        rotation = rotation_about_axis("z", heading)
        control = {
            "root": np.asarray([root]),
            "root_quaternion": np.asarray(
                [axis_angle_to_quaternion(matrix_to_axis_angle(rotation))]
            ),
            "joints": np.asarray([zeros]),
        }
        traces.append(foot_fore_aft(plan, control, "joints")[0])
    # The rest rig is left/right symmetric, so neither foot sits ahead of the other.
    for trace in traces:
        assert abs(trace[0] - trace[1]) < 0.005
        assert abs(trace).max() < 0.05
    # Fore-aft is a body-frame quantity: yawing the root must not change it.
    assert np.allclose(traces[0], traces[1], atol=1e-9)
