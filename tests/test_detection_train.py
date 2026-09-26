"""Training skeleton: windowing, labelling rules, model shapes and seeding.

Runs under the Sionna environment (the only place PyTorch exists here);
system-python pytest skips the whole module.
"""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim2sense_fall.detection_data import ChannelSample, MeshMotion  # noqa: E402
from sim2sense_fall.detection_train import (  # noqa: E402
    FallNet,
    TrainConfig,
    WindowSpec,
    _group_split,
    activity_index_map,
    build_window_examples,
    train_model,
    window_slices,
)

TAPS = 201


def _sample(name, activity, event_label, frames=30, onset=None, onset_time=None):
    time = np.arange(frames) * 0.05
    cir = np.zeros((frames, TAPS), dtype=np.complex128)
    cir[:, 20] = 1.0
    if onset_time is not None:
        cir[int(onset_time / 0.05) :, 20] = 10.0
    return ChannelSample(
        sample_id=name,
        cir_path=None,
        import_path=None,
        cir=cir,
        delay_s=np.arange(TAPS) * 1e-8,
        time_s=time,
        baseline_cir=cir[0] * 0.5,
        activity=activity,
        event_label=event_label,
        imbalance_onset_s=onset,
        first_impact_s=None,
    )


SPEC = WindowSpec(window_s=0.5, stride_s=0.15, min_frames=4)


def test_window_slices_cover_the_axis_without_overrunning():
    time = np.arange(30) * 0.05  # 1.45 s
    slices = window_slices(time, SPEC)
    assert all(stop - start >= SPEC.min_frames for start, stop, _ in slices)
    assert all(time[stop - 1] - time[start] <= SPEC.window_s + 1e-9 for start, stop, _ in slices)
    assert slices[0][0] == 0
    centers = [center for _, _, center in slices]
    assert centers == sorted(centers)


def test_fall_windows_before_the_onset_are_excluded_not_labelled_adl():
    sample = _sample("fall_a", "fall", "push_backward", onset=1.0, onset_time=1.1)
    activities = activity_index_map([sample])
    examples = build_window_examples(sample, SPEC, activities)
    assert examples, "a 1.45 s sample must host at least one window"
    early = [item for item in examples if item.center_s < 0.9]
    late = [item for item in examples if item.center_s >= 0.9]
    assert all(not item.usable for item in early)
    assert all(item.usable and item.fall_label == 1 for item in late)
    assert all(item.activity_index == activities["fall"] for item in late)


def test_adl_windows_are_negative_with_their_activity_class():
    sample = _sample("walk_a", "adl", "walk")
    activities = activity_index_map([sample])
    examples = build_window_examples(sample, SPEC, activities)
    assert all(item.usable and item.fall_label == 0 for item in examples)
    assert all(item.activity_index == activities["walk"] for item in examples)


def test_activity_table_is_deterministic_and_sorted():
    samples = [
        _sample("w", "adl", "walk"),
        _sample("s", "adl", "stand"),
        _sample("f", "fall", "push_backward"),
    ]
    assert activity_index_map(samples) == {"fall": 0, "stand": 1, "walk": 2}


def test_group_split_keeps_held_out_samples_whole():
    activities = {}
    examples = []
    for name in ("a", "b", "c"):
        examples.extend(build_window_examples(_sample(name, "adl", "walk"), SPEC, activities))
    train, evaluation = _group_split(examples, {"b"})
    assert {item.sample_id for item in evaluation} == {"b"}
    assert "b" not in {item.sample_id for item in train}
    with pytest.raises(ValueError, match="no usable training windows"):
        _group_split(examples, {"a", "b", "c"})


def test_model_forward_shapes_and_parameter_budget():
    model = FallNet(hidden=8, n_activities=3)
    fall, activity, velocity = model(torch.zeros(2, 1, 6, TAPS))
    assert fall.shape == (2,) and velocity.shape == (2,) and activity.shape == (2, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000


def _tiny_training_set():
    samples = [
        _sample("fall_a", "fall", "push_backward", onset=0.5, onset_time=0.6),
        _sample("fall_b", "fall", "push_forward", onset=0.6, onset_time=0.7),
        _sample("walk_a", "adl", "walk"),
        _sample("walk_b", "adl", "walk"),
        _sample("stand_a", "adl", "stand"),
    ]
    activities = activity_index_map(samples)
    examples = []
    for sample in samples:
        examples.extend(build_window_examples(sample, SPEC, activities))
    return examples, activities


def test_training_runs_reduces_loss_and_is_seed_deterministic():
    examples, activities = _tiny_training_set()
    train, evaluation = _group_split(examples, set())
    config = TrainConfig(epochs=6, batch_size=4, lr=5e-3, seed=0, hidden=8, device="cpu")
    model, history = train_model(train, evaluation, config, len(activities))
    assert len(history) == 6
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    _, second_history = train_model(train, evaluation, config, len(activities))
    assert history[-1]["train_loss"] == pytest.approx(second_history[-1]["train_loss"])
    assert model.fall_head.out_features == 1


def test_velocity_head_trains_when_mesh_labels_exist():
    wavelength = 299792458.0 / 3.5e9
    sample = replace(
        _sample("fall_a", "fall", "push_backward", onset=0.5, onset_time=0.6),
        transmitter_xyz=(0.0, 0.0, 0.0), receiver_xyz=(0.0, 0.0, 0.0),
        carrier_hz=3.5e9,
    )
    time = sample.time_s
    vertices = np.zeros((len(time), 1, 3))
    vertices[:, 0, 0] = 5.0 - time  # 1 m/s radial approach throughout
    motion = MeshMotion(vertices=vertices, time_s=time)
    activities = activity_index_map([sample])
    examples = build_window_examples(sample, SPEC, activities, motion=motion)
    usable = [item for item in examples if item.usable]
    assert usable and all(
        item.velocity == pytest.approx(2.0 / wavelength, rel=1e-3) for item in usable
    )
    train, _ = _group_split(examples, set())
    config = TrainConfig(epochs=4, batch_size=4, lr=5e-3, seed=0, hidden=8,
                         device="cpu", velocity_weight=1.0)
    model, history = train_model(train, train, config, len(activities))
    assert history[-1]["eval_velocity_mae_hz"] is not None
    assert np.isfinite(history[-1]["eval_velocity_mae_hz"])
