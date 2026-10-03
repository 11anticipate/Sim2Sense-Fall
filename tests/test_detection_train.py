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
    ReconstructionDecoder,
    TrainConfig,
    WindowSpec,
    _group_split,
    activity_index_map,
    build_window_examples,
    load_pretrained_encoder,
    mask_windows,
    pretrain_reconstruction,
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
    # Exact, not approx: three builds of the train02 round-1 run (1718 windows, 60 epochs)
    # reported 0.864 / 0.762 / 0.515 held-out accuracy from one seed, because torch folds
    # CPU reductions in a thread-dependent order. `TrainConfig.deterministic` pins that.
    assert [row["train_loss"] for row in history] == [row["train_loss"] for row in second_history]
    assert model.fall_head.out_features == 1


def test_build_model_reproduces_the_network_for_one_seed():
    """The constructor draws from torch's global RNG, so ordering *was* the bug.

    train_baseline built `FallNet(...)` before `train_model` seeded torch, which meant every
    process started from different weights and the same seed rebuilt a different detector.
    """

    from sim2sense_fall.detection_train import build_model

    config = TrainConfig(epochs=2, seed=20260928, hidden=8)
    first = build_model(config, 6)
    second = build_model(config, 6)
    assert torch.equal(first.fall_head.weight, second.fall_head.weight)
    assert torch.equal(first.gru.weight_hh_l0, second.gru.weight_hh_l0)
    other = build_model(replace(config, seed=20260929), 6)
    assert not torch.equal(first.fall_head.weight, other.fall_head.weight)
    # A drifted ambient RNG state must not leak in.
    torch.randn(1000)
    assert torch.equal(build_model(config, 6).fall_head.weight, first.fall_head.weight)


def test_a_caller_supplied_model_is_still_reproducible_end_to_end():
    """Reproduce the exact train_baseline shape: build outside, pass in, run twice."""

    from sim2sense_fall.detection_train import build_model

    examples, activities = _tiny_training_set()
    train, evaluation = _group_split(examples, set())
    config = TrainConfig(epochs=5, batch_size=4, lr=5e-3, seed=7, hidden=8, device="cpu")
    histories = []
    for _ in range(2):
        model = build_model(config, len(activities))
        _, history = train_model(train, evaluation, config, len(activities), model=model)
        histories.append([row["train_loss"] for row in history])
    assert histories[0] == histories[1], (
        "two builds of one seed must give one curve; a difference means something outside "
        "TrainConfig.seed still decides the weights")


def test_average_state_dicts_is_the_equal_weight_mean_and_validates_its_input():
    from sim2sense_fall.detection_train import average_state_dicts

    first = {"w": torch.tensor([1.0, 2.0]), "steps": torch.tensor(3)}
    second = {"w": torch.tensor([3.0, 4.0]), "steps": torch.tensor(3)}
    third = {"w": torch.tensor([5.0, 6.0]), "steps": torch.tensor(3)}
    averaged = average_state_dicts([first, second, third])
    assert averaged["w"] == pytest.approx(torch.tensor([3.0, 4.0]))
    assert int(averaged["steps"]) == 3, "integer buffers are carried, not invented"
    with pytest.raises(ValueError, match="empty"):
        average_state_dicts([])
    with pytest.raises(ValueError, match="different parameter set"):
        average_state_dicts([first, {"w": torch.tensor([1.0, 2.0])}])
    with pytest.raises(ValueError, match="differs between checkpoints"):
        average_state_dicts([first, {"w": torch.tensor([1.0, 2.0]), "steps": torch.tensor(4)}])


def test_averaging_last_epochs_changes_the_weights_not_the_training():
    examples, activities = _tiny_training_set()
    train, evaluation = _group_split(examples, set())
    base = TrainConfig(epochs=6, batch_size=4, lr=5e-3, seed=0, hidden=8, device="cpu")
    plain, plain_history = train_model(train, evaluation, base, len(activities))
    averaged, averaged_history = train_model(
        train, evaluation, replace(base, average_last_epochs=4), len(activities))
    assert [row["train_loss"] for row in averaged_history] == \
        [row["train_loss"] for row in plain_history], "averaging must not perturb optimisation"
    # `_group_split(examples, set())` evaluates on nothing, so compare the weights
    # themselves: the returned model must not be the epoch-6 checkpoint any more.
    before = plain.state_dict()["fall_head.weight"].detach().numpy()
    after = averaged.state_dict()["fall_head.weight"].detach().numpy()
    assert not np.allclose(before, after), "the returned model should be the averaged one"
    with pytest.raises(ValueError, match="average_last_epochs"):
        TrainConfig(epochs=2, average_last_epochs=-1)


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
    # A negative sample is not decoration: with only fall windows `pos_weight` becomes 0
    # and `train_model` now refuses the split instead of silently zeroing the fall head.
    adl = replace(_sample("walk_a", "adl", "walk"), transmitter_xyz=(0.0, 0.0, 0.0),
                  receiver_xyz=(0.0, 0.0, 0.0), carrier_hz=3.5e9)
    adl_vertices = np.zeros((len(adl.time_s), 1, 3))
    adl_vertices[:, 0, 0] = 5.0 - adl.time_s
    activities = activity_index_map([sample, adl])
    examples.extend(item for item in build_window_examples(
        adl, SPEC, activities, motion=MeshMotion(vertices=adl_vertices, time_s=adl.time_s))
        if item.usable)
    train, _ = _group_split(examples, set())
    config = TrainConfig(epochs=4, batch_size=4, lr=5e-3, seed=0, hidden=8,
                         device="cpu", velocity_weight=1.0)
    model, history = train_model(train, train, config, len(activities))
    assert history[-1]["eval_velocity_mae_hz"] is not None
    assert np.isfinite(history[-1]["eval_velocity_mae_hz"])


def test_encoder_and_decoder_shapes_roundtrip():
    model = FallNet(hidden=8, n_activities=3)
    decoder = ReconstructionDecoder()
    window = torch.rand(2, 1, 6, TAPS)
    features = model.encode(window)
    assert features.shape[0] == 2 and features.shape[1] == 16
    assert features.shape[2] == 6  # pooling never touches the time axis
    reconstructed = decoder(features, window)
    assert reconstructed.shape == window.shape


def test_mask_windows_zeroes_one_rectangle_and_reports_mask():
    windows = [np.full((6, TAPS), 0.5, dtype=np.float32) for _ in range(3)]
    generator = np.random.default_rng(0)
    masked, mask = mask_windows(windows, ratio=0.25, generator=generator)
    assert masked.shape == (3, 1, 6, TAPS)
    area = mask.sum(axis=(2, 3))
    assert np.all(area > 0) and np.all(area <= 0.25 * 6 * TAPS + TAPS)  # one rect
    hidden = mask.astype(bool)
    assert np.all(masked.numpy()[hidden] == 0.0)
    outside = ~hidden
    assert np.allclose(masked.numpy()[outside], 0.5)


def test_pretraining_reduces_masked_loss_and_weights_transfer():
    examples, activities = _tiny_training_set()
    config = TrainConfig(epochs=12, batch_size=4, lr=5e-3, seed=0, hidden=8,
                         device="cpu")
    pretrained = pretrain_reconstruction(examples, config, mask_ratio=0.25)
    assert pretrained["final_masked_loss"] < pretrained["first_masked_loss"]
    fresh = FallNet(hidden=8, n_activities=len(activities))
    conv1_before = fresh.conv1.weight.detach().clone()
    load_pretrained_encoder(fresh, pretrained["encoder_state"])
    assert not torch.allclose(fresh.conv1.weight, conv1_before)
    assert torch.allclose(
        fresh.conv1.weight, pretrained["encoder_state"]["conv1.weight"]
    )


def test_history_reports_loss_terms_that_add_up_to_the_objective():
    """Without the split one cannot tell a weak detector from an out-voted one."""

    examples, activities = _tiny_training_set()
    train, evaluation = _group_split(examples, set())
    config = TrainConfig(epochs=3, batch_size=4, lr=5e-3, seed=0, hidden=8, device="cpu",
                         activity_weight=0.5, velocity_weight=0.5)
    _, history = train_model(train, evaluation, config, len(activities))
    for row in history:
        assert row["train_loss_fall"] >= 0.0
        parts = (row["train_loss_fall"] + row["train_loss_activity"]
                 + row["train_loss_velocity"])
        assert parts == pytest.approx(row["train_loss"], rel=1e-4), row
    off = TrainConfig(epochs=2, batch_size=4, lr=5e-3, seed=0, hidden=8, device="cpu",
                      activity_weight=0.0, velocity_weight=0.0)
    _, only_fall = train_model(train, evaluation, off, len(activities))
    assert all(row["train_loss_activity"] == 0.0 for row in only_fall)
    assert all(row["train_loss_velocity"] == 0.0 for row in only_fall)
    # With both auxiliary weights at zero the objective *is* the fall term.
    assert all(row["train_loss"] == pytest.approx(row["train_loss_fall"], rel=1e-4)
               for row in only_fall)


def _velocity_labelled_training_set():
    """One fall and one walking sample, both with a mesh moving 1 m/s radially.

    Both classes are needed on purpose: `train_model` sets ``pos_weight = negatives /
    positives``, so a fixture made only of fall windows gives ``pos_weight = 0`` and a fall
    term that is identically zero -- a test on it would prove nothing about the balance of
    the heads.
    """

    wavelength = 299792458.0 / 3.5e9
    examples = []
    samples = []
    for name, activity, label in (("fall_a", "fall", "push_backward"),
                                  ("walk_a", "adl", "walk")):
        sample = replace(
            _sample(name, activity, label, onset=0.5 if activity == "fall" else None,
                    onset_time=0.6),
            transmitter_xyz=(0.0, 0.0, 0.0), receiver_xyz=(0.0, 0.0, 0.0), carrier_hz=3.5e9,
        )
        vertices = np.zeros((len(sample.time_s), 1, 3))
        vertices[:, 0, 0] = 5.0 - sample.time_s
        samples.append((sample, MeshMotion(vertices=vertices, time_s=sample.time_s)))
    activities = activity_index_map([sample for sample, _ in samples])
    for sample, motion in samples:
        examples.extend(item for item in build_window_examples(sample, SPEC, activities,
                                                               motion=motion)
                        if item.usable)
    assert any(item.velocity is not None for item in examples), (
        "the fixture lost its velocity labels; the scaling test below would be vacuous")
    labelled = [item.velocity for item in examples if item.velocity is not None]
    assert all(value == pytest.approx(2.0 / wavelength, rel=1e-3) for value in labelled), (
        "1 m/s radial at 3.5 GHz is a ~23 Hz Doppler; the fixture's unit scale changed")
    assert 0 < sum(item.fall_label for item in examples) < len(examples), (
        "the fixture needs both classes or pos_weight zeroes the fall term")
    return examples, activities


def test_the_velocity_term_is_scaled_so_the_fall_head_is_not_out_voted():
    """On train02 the Hz-squared MSE was 96.5% of the objective; a 10 Hz reference fixes it.

    Worse than the share: with the unscaled term the shared trunk saturates and the fall BCE
    collapses to ~0, i.e. the head that produces the alarm was not being trained at all.
    """

    from dataclasses import replace as dc_replace

    train, activities = _velocity_labelled_training_set()
    base = TrainConfig(epochs=2, batch_size=4, lr=1e-4, seed=0, hidden=8, device="cpu",
                       activity_weight=0.5, velocity_weight=0.5)
    _, history_un = train_model(train, train, dc_replace(
        base, velocity_loss_scale_hz=1.0), len(activities))
    _, history_sc = train_model(train, train, dc_replace(
        base, velocity_loss_scale_hz=10.0), len(activities))
    # The scale divides the term by exactly its square.
    assert history_sc[0]["train_loss_velocity"] == pytest.approx(
        history_un[0]["train_loss_velocity"] / 100.0, rel=1e-3)
    # Unscaled, the velocity term is the objective and the fall head is a rounding error
    # (train02 measured 96.5%; this fixture reproduces it at 99.6%).
    assert min(row["train_loss_velocity"] / row["train_loss"] for row in history_un) > 0.99
    assert max(row["train_loss_fall"] / row["train_loss"] for row in history_un) < 0.01, (
        f"the fixture no longer reproduces the drowning: {history_un}")
    # Scaled, the fall term is a real part of the objective again.
    assert max(row["train_loss_fall"] / row["train_loss"] for row in history_sc) > 0.1, (
        f"scaling did not give the fall head a voice: {history_sc}")
    assert max(row["train_loss_velocity"] / row["train_loss"] for row in history_sc) < 0.8
    # weight 50 at a 10 Hz reference is the same objective as weight 0.5 in Hz^2 units.
    _, history_eq = train_model(train, train, dc_replace(
        base, velocity_weight=50.0, velocity_loss_scale_hz=10.0), len(activities))
    assert [row["train_loss_velocity"] for row in history_eq] == pytest.approx(
        [row["train_loss_velocity"] for row in history_un], rel=1e-5)
    with pytest.raises(ValueError, match="velocity_loss_scale_hz"):
        TrainConfig(epochs=2, velocity_loss_scale_hz=0.0)


def test_an_all_positive_training_split_fails_instead_of_zeroing_the_fall_head():
    """pos_weight = negatives/positives is 0.0 when nothing is negative."""

    examples, activities = _tiny_training_set()
    only_falls = [item for item in examples if item.usable and item.fall_label == 1]
    assert only_falls and all(item.fall_label == 1 for item in only_falls)
    with pytest.raises(ValueError, match="no negatives"):
        train_model(only_falls, only_falls, TrainConfig(epochs=1, batch_size=4),
                    len(activities))
