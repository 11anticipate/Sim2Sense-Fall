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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection_data import (  # noqa: E402
    iter_channel_samples,
    load_mesh_motion,
)
from sim2sense_fall.detection_train import (  # noqa: E402
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
    held_out = set(
        args.held_out
        if args.held_out is not None
        else default_holdout(
            [(s.sample_id, s.activity) for s in sorted(samples, key=lambda s: s.sample_id)]
        )
    )
    train, evaluation = _group_split(examples, held_out)
    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        hidden=args.hidden, activity_weight=args.activity_weight,
        velocity_weight=args.velocity_weight, device=args.device,
    )
    model, history = train_model(train, evaluation, config, len(activities))
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
