"""Training skeleton for the stage-9 CIR fall detector (PyTorch, lazy import).

The representation is the range-time map from :mod:`sim2sense_fall.range_time`;
the backbone is deliberately small (two conv layers over the tap axis, a GRU
over time, < 10k parameters) because fall labels stay scarce even when ADL
does not. Three heads exist from day one — fall (binary), activity (multi
class) and channel-visible radial speed (regression) — and a head only
contributes to the loss when its labels exist. Speed labels come from the
samples' mesh streams via :func:`sim2sense_fall.detection_data.velocity_labels`
(peak Doppler over body points, Hz); windows without labels are masked out.

This module is skeleton + flow validation. Nothing here may be reported as
detection performance; the honest protocol lives in the plan (leave-one-out
by subject/sequence/scene, pre-registered thresholds, event-level metrics).

PyTorch lives in the Sionna environment (``~/.local/opt/sionna/bin/python``);
importing this module without it is fine, using it is not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .detection_data import ChannelSample, MeshMotion, velocity_labels
from .range_time import RangeTimeConfig, range_time_map

try:  # system python has no torch; the Sionna venv does
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on system python
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def require_torch() -> None:
    """Fail with an actionable message instead of an opaque NameError."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for detection training; "
            "run under ~/.local/opt/sionna/bin/python"
        )


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """How a channel sample is cut into training windows."""

    window_s: float = 0.8
    stride_s: float = 0.1
    min_frames: int = 4
    clip_db: float = 40.0

    def __post_init__(self) -> None:
        if self.window_s <= 0 or self.stride_s <= 0 or self.min_frames < 2:
            raise ValueError("window_s and stride_s must be positive; min_frames >= 2")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation hyperparameters; seeds are explicit, never ambient."""

    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    seed: int = 0
    hidden: int = 32
    activity_weight: float = 0.5
    velocity_weight: float = 0.0
    device: str = "cpu"
    # Averaging the state dicts of the last N epochs keeps the returned model near the mean
    # of that trajectory. 0 keeps the historical behaviour (last epoch as-is).
    average_last_epochs: int = 0
    # Three builds of train02 round 1 with the same seed and the same CLI reported held-out
    # window accuracies of 0.864 / 0.762 / 0.515: ATen's multi-thread CPU reductions fold in
    # a run-dependent order, and the accuracy oscillates enough epoch to epoch for that to
    # move the final-epoch number anywhere on the curve. Reproducibility is an AGENTS.md
    # requirement for any recorded result, so it is on by default and costs little: the
    # backbone is under 10k parameters and the runs are I/O bound.
    deterministic: bool = True

    # The Doppler head regresses Hz, so its raw MSE is a *squared frequency*: on train02 it
    # measured 44 of a 45-unit total objective, i.e. the fall head's BCE (~0.75) and the
    # activity head (~0.6) together got 1.7% of the gradient. That is the mechanism behind a
    # detector with a healthy window AUC (0.86) whose fall probabilities nevertheless never
    # leave 0.5-0.78 and whose epoch-to-epoch accuracy oscillates. Dividing the residual by a
    # reference speed makes the term dimensionless and comparable with the other two; the
    # reported MAE stays in Hz.
    velocity_loss_scale_hz: float = 10.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.hidden < 1:
            raise ValueError("epochs/batch_size/hidden must be >= 1 and lr > 0")
        if self.activity_weight < 0 or self.velocity_weight < 0:
            raise ValueError("auxiliary head weights must be non-negative")
        if self.average_last_epochs < 0:
            raise ValueError("average_last_epochs must be >= 0")
        if not np.isfinite(self.velocity_loss_scale_hz) or self.velocity_loss_scale_hz <= 0:
            raise ValueError("velocity_loss_scale_hz must be finite and positive")


@dataclass(frozen=True, slots=True)
class WindowExample:
    """One training window: range-time image plus its labels.

    ``usable=False`` marks windows whose labels would be a guess — e.g. the
    pre-onset part of a fall trial, which is standing, not falling. Unused
    (but still loadable) fields stay ``None`` instead of invented values.
    """

    sample_id: str
    start_s: float
    center_s: float
    magnitude: np.ndarray
    fall_label: int
    activity_index: int | None
    velocity: float | None
    usable: bool


def activity_index_map(samples: list[ChannelSample]) -> dict[str, int]:
    """Deterministic activity class table over event labels and 'fall'."""

    names: set[str] = set()
    for sample in samples:
        if sample.activity == "fall":
            names.add("fall")
        elif sample.event_label:
            names.add(sample.event_label)
    return {name: index for index, name in enumerate(sorted(names))}


def window_slices(time_s: np.ndarray, spec: WindowSpec) -> list[tuple[int, int, float]]:
    """(start, stop, center) for every full window, one stride apart."""

    time = np.asarray(time_s, dtype=np.float64)
    slices: list[tuple[int, int, float]] = []
    start = 0
    while start < len(time) and time[start] + spec.window_s <= time[-1] + 1e-12:
        stop = int(np.searchsorted(time, time[start] + spec.window_s, side="right"))
        if stop - start >= spec.min_frames:
            center = float(0.5 * (time[start] + time[stop - 1]))
            slices.append((start, stop, center))
        start += max(1, int(round(spec.stride_s / float(np.median(np.diff(time))))))
    return slices


def build_window_examples(
    sample: ChannelSample,
    spec: WindowSpec,
    activities: dict[str, int],
    config: RangeTimeConfig | None = None,
    motion: MeshMotion | None = None,
) -> list[WindowExample]:
    """Cut one channel sample into labelled range-time windows.

    Labelling rule (deliberately conservative): a fall sample's windows are
    positive only from just before its recorded imbalance onset onward;
    everything before the onset is *excluded* rather than called ADL, and a
    fall sample without an onset contributes nothing at all. ADL windows are
    negative with their activity class from the event label. When the
    sample's mesh stream is supplied, each window also carries the mean of
    its per-frame peak-Doppler truth (Hz) as the velocity head's regression
    target.
    """

    cfg = config or RangeTimeConfig(clip_db=spec.clip_db)
    map_ = range_time_map(sample.cir, sample.delay_s, sample.time_s, sample.baseline_cir, cfg)
    frame_velocity = velocity_labels(sample, motion) if motion is not None else None
    examples: list[WindowExample] = []
    for start, stop, center in window_slices(sample.time_s, spec):
        fall_label = 0
        activity_index: int | None = None
        usable = True
        if sample.activity == "fall":
            if sample.imbalance_onset_s is None or center < sample.imbalance_onset_s - 0.1:
                usable = False
            else:
                fall_label = 1
                activity_index = activities.get("fall")
        else:
            activity_index = (
                activities.get(sample.event_label) if sample.event_label else None
            )
        velocity = (
            float(np.mean(frame_velocity[start:stop])) if frame_velocity is not None else None
        )
        examples.append(
            WindowExample(
                sample_id=sample.sample_id,
                start_s=float(sample.time_s[start]),
                center_s=center,
                magnitude=map_.magnitude[start:stop],
                fall_label=fall_label,
                activity_index=activity_index,
                velocity=velocity,
                usable=usable,
            )
        )
    return examples


def _pad_batch(magnitudes: list[np.ndarray]) -> torch.Tensor:
    """Stack variable-length windows into one (B, 1, T, D) float tensor."""

    max_t = max(item.shape[0] for item in magnitudes)
    max_d = max(item.shape[1] for item in magnitudes)
    batch = np.zeros((len(magnitudes), 1, max_t, max_d), dtype=np.float32)
    for index, item in enumerate(magnitudes):
        batch[index, 0, : item.shape[0], : item.shape[1]] = item
    return torch.from_numpy(batch)


class FallNet(nn.Module):
    """Small conv+GRU detector: < 10k parameters by design.

    Convolutions run over the tap (delay) axis with padding, the per-timestep
    channel is the delay-mean of the conv features, and a GRU integrates over
    time; heads read the last hidden state. Variable window lengths are fine
    because the GRU is length-agnostic and padding is zero.
    """

    def __init__(self, hidden: int = 32, n_activities: int = 3) -> None:
        require_torch()
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=(1, 2))
        self.gru = nn.GRU(16, hidden, batch_first=True)
        self.fall_head = nn.Linear(hidden, 1)
        self.activity_head = nn.Linear(hidden, n_activities)
        self.velocity_head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared conv feature map (B, 16, T, D/4) before the time head."""

        hidden = torch.relu(self.conv1(x))
        hidden = self.pool(hidden)
        hidden = torch.relu(self.conv2(hidden))
        return self.pool(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(x)
        hidden = hidden.mean(dim=3).transpose(1, 2)  # (B, T, channels)
        sequence, _ = self.gru(hidden)
        last = sequence[:, -1]
        return (
            self.fall_head(last)[:, 0],
            self.activity_head(last),
            self.velocity_head(last)[:, 0],
        )


class ReconstructionDecoder(nn.Module):
    """Masked-reconstruction head over the shared conv encoder.

    The stage-9 plan pretrains the encoder on abundant unlabelled ADL
    windows (masked reconstruction) before fall supervision; this decoder
    is the pretext machine and is discarded after pretraining. Bilinear
    resize back to the input shape keeps variable window sizes supported
    despite the floor in the pooling stages.
    """

    def __init__(self) -> None:
        require_torch()
        super().__init__()
        self.conv = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose2d(8, 1, kernel_size=(1, 4), stride=(1, 4))

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv(features))
        reconstructed = self.upsample(hidden)
        return torch.nn.functional.interpolate(
            reconstructed, size=target.shape[-2:], mode="bilinear",
            align_corners=False,
        )


def mask_windows(
    magnitudes: list[np.ndarray], ratio: float, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Zero one random rectangle per window; return (batch, mask).

    The mask is 1 inside the rectangle and 0 elsewhere, so the pretext loss
    scores only the hidden region. ``ratio`` bounds the *area* fraction.
    """

    if not 0 < ratio < 1:
        raise ValueError("mask ratio must lie in (0, 1)")
    batch = _pad_batch(magnitudes)
    mask = np.zeros_like(batch.numpy())
    for index in range(batch.shape[0]):
        frames, taps = batch.shape[2], batch.shape[3]
        height = max(1, int(round(taps * np.sqrt(ratio))))
        width = max(1, int(round(frames * np.sqrt(ratio))))
        top = int(generator.integers(0, max(taps - height, 1)))
        left = int(generator.integers(0, max(frames - width, 1)))
        mask[index, 0, left : left + width, top : top + height] = 1.0
    return batch * torch.from_numpy(1.0 - mask).float(), mask


def pretrain_reconstruction(
    examples: list[WindowExample],
    config: TrainConfig,
    mask_ratio: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Masked-reconstruction pretraining of the shared conv encoder.

    Runs on *any* usable windows — labels are irrelevant here, which is the
    point: ADL can be generated without the scarce fall labels. Returns the
    trained encoder weights (conv1/conv2) for :func:`load_pretrained_encoder`.
    """

    require_torch()
    if not examples:
        raise ValueError("pretraining needs at least one usable window")
    generator = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    encoder = build_model(config, 2).to(device)
    decoder = ReconstructionDecoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=config.lr
    )
    first_loss = last_loss = None
    for _ in range(config.epochs):
        order = generator.permutation(len(examples))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            chunk = [examples[index] for index in order[start : start + config.batch_size]]
            target = _pad_batch([item.magnitude for item in chunk]).to(device)
            masked, mask = mask_windows(
                [item.magnitude for item in chunk], mask_ratio, generator
            )
            masked = masked.to(device)
            mask_t = torch.from_numpy(mask).float().to(device)
            reconstructed = decoder(encoder.encode(masked), target)
            loss = ((reconstructed - target) ** 2 * mask_t).sum() / mask_t.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        last_loss = epoch_loss / max(batches, 1)
        if first_loss is None:
            first_loss = last_loss
    return {
        "encoder_state": {key: value for key, value in encoder.state_dict().items()
                          if key.startswith("conv")},
        "first_masked_loss": first_loss,
        "final_masked_loss": last_loss,
    }


def load_pretrained_encoder(model: FallNet, encoder_state: dict) -> None:
    """Copy pretrained conv weights into a fresh detector."""

    model.conv1.load_state_dict(
        {key.removeprefix("conv1."): value for key, value in encoder_state.items()
         if key.startswith("conv1.")}
    )
    model.conv2.load_state_dict(
        {key.removeprefix("conv2."): value for key, value in encoder_state.items()
         if key.startswith("conv2.")}
    )


def _group_split(
    examples: list[WindowExample], held_out: set[str]
) -> tuple[list[WindowExample], list[WindowExample]]:
    train = [item for item in examples if item.sample_id not in held_out and item.usable]
    evaluation = [item for item in examples if item.sample_id in held_out and item.usable]
    if not train:
        raise ValueError("group split left no usable training windows")
    return train, evaluation


def build_model(config: TrainConfig, n_activities: int) -> FallNet:
    """Construct the detector under *this run's* seed, so the caller cannot get it wrong.

    ``FallNet.__init__`` draws its initial weights from torch's global RNG at construction
    time. A caller that builds the network before anything seeds torch therefore gets a
    different starting point in every process, and re-running the identical seed/config
    yields a different model -- which is what happened to train02 round 1: two builds of one
    seed reported 0.797 and 0.854 held-out window accuracy. Seeding inside the constructor
    helper is the only place that ordering can be fixed once.
    """

    require_torch()
    torch.manual_seed(config.seed)
    return FallNet(hidden=config.hidden, n_activities=max(1, n_activities))


def train_model(
    train_examples: list[WindowExample],
    eval_examples: list[WindowExample],
    config: TrainConfig,
    n_activities: int,
    model: FallNet | None = None,
) -> tuple[FallNet, list[dict]]:
    """Train the three heads; returns the model and a per-epoch history.

    Class imbalance is handled with ``pos_weight`` (negatives/positives of
    the fall head), not by resampling; the activity head ignores unlabeled
    windows (``-100``) and the velocity head only trains on windows that carry
    a mesh-derived speed label, so ``velocity_weight`` costs nothing on a batch
    whose packets have no mesh stream. With ``average_last_epochs`` the returned
    model holds the mean of that many final epoch checkpoints, not the last one.
    """

    require_torch()
    if config.deterministic:
        # Raises rather than silently continuing if any op lacks a deterministic
        # implementation: a result that cannot be rebuilt is not a result.
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
    generator = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    if model is None:
        # Seeds internally: the initial weights must come from this run's seed, not from
        # whatever the process's RNG happened to be when the caller constructed the model.
        model = build_model(config, n_activities)
    torch.manual_seed(config.seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    positives = sum(item.fall_label for item in train_examples)
    negatives = len(train_examples) - positives
    if negatives == 0:
        # pos_weight would be 0.0, which silently zeroes the fall term: a split with no
        # negatives cannot train the head the whole alarm depends on, and reporting a loss
        # of 0 for it would hide that.
        raise ValueError(
            f"the training split has {positives} fall windows and no negatives; "
            "an event head cannot be fitted (or measured) on that")
    pos_weight_value = (negatives / positives) if positives else 1.0
    fall_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32)
    )
    activity_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    velocity_criterion = nn.MSELoss(reduction="none")

    history: list[dict] = []
    keep_last = min(config.average_last_epochs, config.epochs)
    snapshots: list[dict] = []
    for epoch in range(config.epochs):
        order = generator.permutation(len(train_examples))
        model.train()
        epoch_loss = 0.0
        # Per-term contributions, weighted exactly as they enter the objective: the total
        # loss of the three-head run is dominated by whichever head's target has the largest
        # unit, and a Doppler error in Hz squared is not comparable to a BCE in [0,1]. Without
        # this split one cannot tell "the detector is weak" from "the detector is being
        # out-voted in its own loss" -- and train02 showed a fall head whose probabilities
        # barely leave 0.5 while its window AUC is 0.83-0.93.
        epoch_fall = epoch_activity = epoch_velocity = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            chunk = [train_examples[index] for index in order[start : start + config.batch_size]]
            inputs = _pad_batch([item.magnitude for item in chunk]).to(device)
            fall_target = torch.tensor([item.fall_label for item in chunk], dtype=torch.float32)
            activity_target = torch.tensor(
                [
                    item.activity_index if item.activity_index is not None else -100
                    for item in chunk
                ],
                dtype=torch.long,
            )
            velocity_mask = torch.tensor(
                [item.velocity is not None for item in chunk], dtype=torch.bool
            )
            velocity_target = torch.tensor(
                [item.velocity if item.velocity is not None else 0.0 for item in chunk],
                dtype=torch.float32,
            )
            fall_logits, activity_logits, velocity_pred = model(inputs)
            fall_term = fall_criterion(fall_logits, fall_target.to(device))
            loss = fall_term
            activity_term = loss.new_zeros(())
            velocity_term = loss.new_zeros(())
            if config.activity_weight > 0:
                activity_term = config.activity_weight * activity_criterion(
                    activity_logits, activity_target.to(device)
                )
                loss = loss + activity_term
            if config.velocity_weight > 0 and velocity_mask.any():
                squared = velocity_criterion(velocity_pred, velocity_target.to(device))
                velocity_term = (config.velocity_weight / config.velocity_loss_scale_hz ** 2) \
                    * squared[velocity_mask.to(device)].mean()
                loss = loss + velocity_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_fall += float(fall_term.detach())
            epoch_activity += float(activity_term.detach())
            epoch_velocity += float(velocity_term.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batches, 1),
                "train_loss_fall": epoch_fall / max(batches, 1),
                "train_loss_activity": epoch_activity / max(batches, 1),
                "train_loss_velocity": epoch_velocity / max(batches, 1),
                "eval_fall_accuracy": evaluate_fall_accuracy(model, eval_examples, config),
                "eval_velocity_mae_hz": evaluate_velocity_mae(model, eval_examples, config),
            }
        )
        if keep_last:
            snapshots.append({key: value.detach().clone()
                              for key, value in model.state_dict().items()})
            if len(snapshots) > keep_last:
                snapshots.pop(0)
    if keep_last > 1:
        model.load_state_dict(average_state_dicts(snapshots))
    return model, history


def average_state_dicts(state_dicts: Sequence[Mapping[str, torch.Tensor]]) -> dict:
    """Equal-weight mean of several checkpoints of the same architecture (SWA-style).

    Floating-point tensors are averaged; integer buffers (batch-norm style counters) must
    agree across the checkpoints, because averaging them would invent a value that no
    epoch ever had.
    """

    if not state_dicts:
        raise ValueError("cannot average an empty list of checkpoints")
    first = state_dicts[0]
    for index, state in enumerate(state_dicts[1:], start=1):
        if set(state) != set(first):
            raise ValueError(f"checkpoint {index} has a different parameter set")
    averaged: dict = {}
    for key in first:
        stacked = [state[key] for state in state_dicts]
        if not torch.is_floating_point(stacked[0]):
            reference = stacked[0]
            if not all(torch.equal(reference, other) for other in stacked[1:]):
                raise ValueError(f"non-float buffer {key} differs between checkpoints")
            averaged[key] = reference.clone()
            continue
        averaged[key] = torch.stack([value.to(torch.float64) for value in stacked]).mean(0) \
            .to(stacked[0].dtype)
    return averaged


@torch.no_grad()
def _fall_probabilities(model: FallNet, examples: list[WindowExample], device: str) -> np.ndarray:
    model.eval()
    outputs: list[float] = []
    for start in range(0, len(examples), 16):
        chunk = examples[start : start + 16]
        inputs = _pad_batch([item.magnitude for item in chunk]).to(device)
        fall_logits, _, _ = model(inputs)
        outputs.extend(torch.sigmoid(fall_logits).tolist())
    return np.asarray(outputs, dtype=np.float64)


def evaluate_fall_accuracy(
    model: FallNet, examples: list[WindowExample], config: TrainConfig
) -> float | None:
    """Window-level fall accuracy at the 0.5 logit cut — a flow number only."""

    if not examples:
        return None
    probabilities = _fall_probabilities(model, examples, config.device)
    labels = np.asarray([item.fall_label for item in examples])
    return float(((probabilities >= 0.5) == labels).mean())


@torch.no_grad()
def _velocity_predictions(model: FallNet, examples: list[WindowExample], device: str) -> np.ndarray:
    model.eval()
    outputs: list[float] = []
    for start in range(0, len(examples), 16):
        chunk = examples[start : start + 16]
        inputs = _pad_batch([item.magnitude for item in chunk]).to(device)
        _, _, velocity = model(inputs)
        outputs.extend(velocity.tolist())
    return np.asarray(outputs, dtype=np.float64)


def evaluate_velocity_mae(
    model: FallNet, examples: list[WindowExample], config: TrainConfig
) -> float | None:
    """Mean absolute error (Hz) of the velocity head on labelled windows.

    Only windows that actually carry a mesh-derived label participate; None
    when the evaluation split has none.
    """

    labelled = [item for item in examples if item.velocity is not None]
    if not labelled:
        return None
    predictions = _velocity_predictions(model, labelled, config.device)
    targets = np.asarray([item.velocity for item in labelled])
    return float(np.abs(predictions - targets).mean())
