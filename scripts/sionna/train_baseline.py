#!/usr/bin/env python3
"""Training-flow smoke for the stage-9 CIR fall detector.

Cuts the batch's channel samples into labelled range-time windows, holds out
one sample per activity class, trains the small conv+GRU three-head network
and writes a report. Everything about this run is a *flow* exercise: nine
samples cannot train or rank detectors, the fall class inside one batch is
tiny, and the velocity head trains at weight zero because the current
packets carry no mesh-derived speed labels. Run it under the Sionna
environment (PyTorch lives there):

    ~/.local/opt/sionna/bin/python scripts/sionna/train_baseline.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection_data import (  # noqa: E402
    iter_channel_samples,
    load_mesh_motion,
)
from sim2sense_fall.detection_train import (  # noqa: E402
    FallNet,
    TrainConfig,
    WindowSpec,
    _fall_probabilities,
    _group_split,
    activity_index_map,
    build_window_examples,
    train_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sionna-dir", type=Path, required=True,
                        help="directory with *.cir.npz / *.import.json pairs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <sionna-dir>/detection/train_smoke)")
    parser.add_argument("--window-s", type=float, default=0.8)
    parser.add_argument("--stride-s", type=float, default=0.1)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--clip-db", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--activity-weight", type=float, default=0.5)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--held-out", nargs="*", default=None,
                        help="sample ids to hold out; default: one per activity class")
    parser.add_argument("--splits", type=Path, default=None,
                        help="splits.json from scripts/sionna/assign_splits.py; when given, "
                             "the held-out set is every sample of --eval-split (group-aware: "
                             "a physics session never straddles the split)")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test",
                        help="which split of --splits to score (default test)")
    parser.add_argument("--loo", action="store_true",
                        help="leave-one-sample-out over all samples instead of one split")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="masked-reconstruction pretraining epochs on all usable "
                             "windows before supervised training (0 = off)")
    return parser.parse_args(argv)


def default_holdout(sample_ids: list[tuple[str, str]]) -> list[str]:
    """One held-out sample per activity class, deterministically chosen."""

    held: list[str] = []
    seen: set[str] = set()
    for sample_id, activity in sample_ids:  # sorted by id upstream
        if activity not in seen:
            held.append(sample_id)
            seen.add(activity)
    return held


def train_one_fold(
    examples: list,
    held_out: set[str],
    config: TrainConfig,
    n_activities: int,
) -> dict:
    """Train on everything outside ``held_out`` and score the held sample."""

    from sim2sense_fall.detection_train import _fall_probabilities

    train, evaluation = _group_split(examples, held_out)
    model, history = train_model(train, evaluation, config, n_activities)
    probabilities = _fall_probabilities(model, evaluation, config.device)
    predictions = [
        {
            "sample_id": item.sample_id,
            "center_s": item.center_s,
            "fall_label": item.fall_label,
            "fall_probability": float(probability),
            "velocity_label": item.velocity,
        }
        for item, probability in zip(evaluation, probabilities, strict=True)
    ]
    return {
        "held_out": sorted(held_out),
        "train_windows": len(train),
        "train_fall_windows": sum(item.fall_label for item in train),
        "eval_windows": len(evaluation),
        "eval_fall_accuracy": history[-1]["eval_fall_accuracy"],
        "eval_velocity_mae_hz": history[-1]["eval_velocity_mae_hz"],
        "final_train_loss": history[-1]["train_loss"],
        "predictions": predictions,
    }


def run_loo(examples: list, sample_ids: list[str], config: TrainConfig, n_activities: int) -> dict:
    """Leave-one-sample-out over every sample with usable windows.

    Each window is scored by a model that never saw its sample, so pooled
    numbers are out-of-sample at the *sample* level — still a flow exercise
    at nine samples, but the exact aggregation the batch stage will reuse.
    """

    folds = []
    for sample_id in sample_ids:
        print(f"  fold {sample_id}")
        folds.append(train_one_fold(examples, {sample_id}, config, n_activities))
    pooled = [row for fold in folds for row in fold["predictions"]]
    accuracies = [fold["eval_fall_accuracy"] for fold in folds
                  if fold["eval_fall_accuracy"] is not None]
    maes = [fold["eval_velocity_mae_hz"] for fold in folds
            if fold["eval_velocity_mae_hz"] is not None]
    labels = np.asarray([row["fall_label"] for row in pooled])
    probs = np.asarray([row["fall_probability"] for row in pooled])
    pooled_accuracy = float(((probs >= 0.5) == labels).mean()) if len(pooled) else None
    labelled = [(row["fall_probability"], row["velocity_label"]) for row in pooled
                if row["velocity_label"] is not None]
    pooled_velocity_mae = (
        float(np.mean([abs(p - v) for p, v in labelled])) if labelled else None
    )
    return {
        "mode": "leave_one_sample_out",
        "folds": folds,
        "pooled_windows": len(pooled),
        "pooled_fall_accuracy": pooled_accuracy,
        "pooled_velocity_mae_hz": pooled_velocity_mae,
        "fold_accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
        "fold_velocity_mae_mean_hz": float(np.mean(maes)) if maes else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    samples = iter_channel_samples(args.sionna_dir)
    if not samples:
        print(f"no admitted channel samples under {args.sionna_dir}")
        return 1
    activities = activity_index_map(samples)
    spec = WindowSpec(
        window_s=args.window_s, stride_s=args.stride_s,
        min_frames=args.min_frames, clip_db=args.clip_db,
    )
    use_velocity = args.velocity_weight > 0
    examples = []
    per_sample_counts: dict[str, int] = {}
    label_sources: dict[str, str] = {}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        motion = None
        if use_velocity:
            motion = load_mesh_motion(sample)
            if motion is None:
                print(f"  note: no mesh stream resolved for {sample.sample_id}; "
                      "its windows train without velocity labels")
            else:
                label_sources[sample.sample_id] = "mesh"
        built = build_window_examples(sample, spec, activities, motion=motion)
        per_sample_counts[sample.sample_id] = sum(item.usable for item in built)
        examples.extend(built)
    velocity_labelled = sum(item.velocity is not None for item in examples)
    if args.splits is not None:
        payload = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        held_out = {str(row["sample_id"]) for row in payload["samples"]
                    if row.get("split") == args.eval_split}
        if not held_out:
            raise SystemExit(f"{args.splits}: no samples in split {args.eval_split!r}")
        print(f"splits from {args.splits} (seed {payload.get('seed')}, "
              f"batch {payload.get('batch')}): holding out {len(held_out)} "
              f"{args.eval_split}-split samples")
    else:
        held_out = set(
            args.held_out
            if args.held_out is not None
            else default_holdout(
                [(s.sample_id, s.activity) for s in sorted(samples, key=lambda s: s.sample_id)]
            )
        )
    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        hidden=args.hidden, activity_weight=args.activity_weight,
        velocity_weight=args.velocity_weight, device=args.device,
    )
    pretrain_note = None
    if args.pretrain_epochs > 0 and not args.loo:
        from sim2sense_fall.detection_train import (
            load_pretrained_encoder,
            pretrain_reconstruction,
        )
        usable_all = [item for item in examples if item.usable]
        pretrain_config = replace(config, epochs=args.pretrain_epochs)
        pretrained = pretrain_reconstruction(usable_all, pretrain_config)
        print(f"  pretrain masked loss {pretrained['first_masked_loss']:.4f} -> "
              f"{pretrained['final_masked_loss']:.4f} on {len(usable_all)} windows")
        pretrain_note = {
            "epochs": args.pretrain_epochs,
            "windows": len(usable_all),
            "first_masked_loss": pretrained["first_masked_loss"],
            "final_masked_loss": pretrained["final_masked_loss"],
        }
    if args.loo:
        result = run_loo(
            examples,
            [sid for sid, count in sorted(per_sample_counts.items()) if count > 0],
            config, len(activities),
        )
        out = args.out or (args.sionna_dir / "detection" / "train_smoke")
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "train_loo_report.json"
        report_path.write_text(json.dumps(
            {"flow_smoke_only": True,
             "notes": ["leave-one-sample-out over nine samples is a protocol exercise, "
                        "not a performance claim"],
             "config": {"train": {"epochs": config.epochs, "batch_size": config.batch_size,
                                   "lr": config.lr, "seed": config.seed,
                                   "hidden": config.hidden,
                                   "activity_weight": config.activity_weight,
                                   "velocity_weight": config.velocity_weight,
                                   "device": config.device},
                         "activities": activities},
             **result},
            indent=2), encoding="utf-8")
        print(f"loo report -> {report_path}")
        print(f"  pooled windows {result['pooled_windows']}, "
              f"pooled fall accuracy {result['pooled_fall_accuracy']}, "
              f"pooled velocity MAE {result['pooled_velocity_mae_hz']} Hz")
        return 0
    train, evaluation = _group_split(examples, held_out)
    model = FallNet(hidden=config.hidden, n_activities=len(activities))
    if pretrain_note is not None:
        from sim2sense_fall.detection_train import load_pretrained_encoder
        load_pretrained_encoder(model, pretrained["encoder_state"])
    model, history = train_model(train, evaluation, config, len(activities),
                                 model=model)
    probabilities = _fall_probabilities(model, evaluation, config.device)
    predictions = [
        {
            "sample_id": item.sample_id,
            "center_s": item.center_s,
            "fall_label": item.fall_label,
            "fall_probability": float(probability),
        }
        for item, probability in zip(evaluation, probabilities, strict=True)
    ]
    positives = sum(item.fall_label for item in train)
    report = {
        "flow_smoke_only": True,
        "notes": [
            "nine samples cannot train or rank detectors",
            "velocity head trained at weight zero"
            if config.velocity_weight == 0
            else f"velocity head active on {velocity_labelled} labelled windows",
        ],
        "config": {
            "window": {"window_s": spec.window_s, "stride_s": spec.stride_s,
                       "min_frames": spec.min_frames, "clip_db": spec.clip_db},
            "train": {"epochs": config.epochs, "batch_size": config.batch_size,
                      "lr": config.lr, "seed": config.seed, "hidden": config.hidden,
                      "activity_weight": config.activity_weight,
                      "velocity_weight": config.velocity_weight,
                      "device": config.device},
            "activities": activities,
            "held_out": sorted(held_out),
        },
        "usable_windows_per_sample": per_sample_counts,
        "pretrain": pretrain_note,
        "velocity_labelled_windows": velocity_labelled,
        "velocity_label_sources": label_sources,
        "train_windows": len(train),
        "train_fall_windows": positives,
        "eval_windows": len(evaluation),
        "history": history,
        "eval_predictions": predictions,
    }
    out = args.out or (args.sionna_dir / "detection" / "train_smoke")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"train report -> {report_path}")
    print(f"  windows train/eval {len(train)}/{len(evaluation)} "
          f"(fall windows in train: {positives}), held out: {sorted(held_out)}")
    print(f"  final loss {history[-1]['train_loss']:.4f}, "
          f"eval window accuracy {history[-1]['eval_fall_accuracy']}, "
          f"velocity MAE {history[-1]['eval_velocity_mae_hz']} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
