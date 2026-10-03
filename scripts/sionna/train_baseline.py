#!/usr/bin/env python3
"""Train and evaluate the stage-9 CIR fall detector on an admitted batch.

Cuts the batch's channel samples into labelled range-time windows, trains the small
conv+GRU three-head network (activity / fall / Doppler) on one split and scores the
held-out split, reducing window scores to alarm episodes through
`sim2sense_fall.detection_eval` so the result is comparable with the frame-level
baseline. Whether the numbers may be quoted is *computed* (`data_sufficiency`) from the
pre-registered minima -- scorable fall samples, classified ADL exposure in hours and
held-out groups -- and printed as a shortfall list when they may not. A single seed and a
final-epoch weight dump are not a performance claim; the report records the runtime
(thread counts, torch version) because CPU reduction order moves the outcome.

    ~/.local/opt/sionna/bin/python scripts/sionna/train_baseline.py \
        --sionna-dir artifacts/batches/train02/sionna \
        --splits artifacts/batches/train02/splits.json --eval-split val \
        --event-threshold-sweep 0.3,0.5,0.7,0.9 --out .../model_val
"""

from __future__ import annotations

import argparse
import json
import os
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
from sim2sense_fall.detection_eval import (  # noqa: E402
    EventEvalConfig,
    aggregate,
    cluster_bootstrap_auc_ci,
    config_dict,
    data_sufficiency,
    event_metrics_from_predictions,
    window_roc_auc,
)
from sim2sense_fall.detection_train import (  # noqa: E402
    TrainConfig,
    WindowSpec,
    _fall_probabilities,
    _group_split,
    activity_index_map,
    build_model,
    build_window_examples,
    evaluate_fall_accuracy,
    evaluate_velocity_mae,
    train_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sionna-dir", type=Path, required=True,
                        help="directory with *.cir.npz / *.import.json pairs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <sionna-dir>/detection/train_eval)")
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
    parser.add_argument("--event-threshold", type=float, default=0.5,
                        help="window probability at or above which a window is positive "
                             "when the run is reduced to alarm episodes")
    parser.add_argument("--event-hold-off-s", type=float, default=5.0)
    parser.add_argument("--max-latency-s", type=float, default=5.0,
                        help="an episode starting later than this after the imbalance onset "
                             "is a miss, not a slow detection")
    parser.add_argument("--event-threshold-sweep", default=None,
                        help="comma-separated thresholds to report on the evaluation split "
                             "only. Use it on the validation run; sweeping on test and then "
                             "reporting test is selection on the test set.")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="masked-reconstruction pretraining epochs on all usable "
                             "windows before supervised training (0 = off)")
    parser.add_argument("--velocity-loss-scale-hz", type=float, default=10.0,
                        help="divide the Doppler residual by this reference speed so the "
                             "regression term is dimensionless. At the old value (1.0) the "
                             "Hz-squared MSE was 96% of the whole objective on train02 and "
                             "the fall head was optimised by accident")
    parser.add_argument("--average-last-epochs", type=int, default=0,
                        help="average the state dicts of the last N epoch checkpoints and "
                             "score that model instead of the final epoch. The held-out "
                             "accuracy oscillates epoch to epoch, so a last-epoch number is "
                             "a coin flip; 0 keeps the old behaviour")
    parser.add_argument("--nondeterministic", action="store_true",
                        help="allow torch's multi-thread CPU reductions. Off by default "
                             "because three builds of one seed reported 0.864 / 0.762 / "
                             "0.515 held-out accuracy on train02")
    parser.add_argument("--protocol", default="unspecified",
                        help="label recorded in the report, e.g. 'formal-round-2-val-only-"
                             "selection'. Whether the numbers can be quoted is decided by "
                             "data_sufficiency, not by this string.")
    return parser.parse_args(argv)


def runtime_fingerprint(device: str) -> dict[str, object]:
    """What the run actually executed under -- recorded because it changes the answer.

    Rebuilding train02 round 1 with the same seed and the same CLI produced a held-out test
    window accuracy of 0.515 instead of 0.864, i.e. the pipeline was not reproducible from
    the recorded configuration alone. Recording thread counts and library versions is the
    minimum that lets a later reader tell "not reproducible" from "reproducible elsewhere".
    """

    import torch

    return {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "device": device,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "env_threads": {key: os.environ.get(key)
                        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cpu_count": os.cpu_count(),
    }


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
        average_last_epochs=args.average_last_epochs,
        velocity_loss_scale_hz=args.velocity_loss_scale_hz,
        deterministic=not args.nondeterministic,
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
        out = args.out or (args.sionna_dir / "detection" / "train_eval")
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "train_loo_report.json"
        report_path.write_text(json.dumps(
            {"protocol": args.protocol,
             "mode": "leave_one_sample_out",
             "notes": ["leave-one-sample-out over a handful of samples is a protocol "
                       "exercise, not a performance claim"],
             "data_sufficiency": {
                 "protocol": args.protocol, "claim_supported": False,
                 "reason": "leave-one-sample-out pools folds; the event-level minima "
                           "(scorable falls, ADL hours, held-out groups) are not defined here"},
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
    # Built through build_model, not FallNet: the constructor draws from torch's global
    # RNG, and building it here -- before train_model seeds -- made every process start
    # from different weights, which is what broke train02 round 1's reproducibility.
    model = build_model(config, len(activities))
    if pretrain_note is not None:
        from sim2sense_fall.detection_train import load_pretrained_encoder
        load_pretrained_encoder(model, pretrained["encoder_state"])
    model, history = train_model(train, evaluation, config, len(activities),
                                 model=model)
    probabilities = _fall_probabilities(model, evaluation, config.device)
    # Scored on the model that is actually returned: with checkpoint averaging the last
    # history row describes a different set of weights than the one being reported.
    final_eval = {
        "eval_fall_accuracy": evaluate_fall_accuracy(model, evaluation, config),
        "eval_velocity_mae_hz": evaluate_velocity_mae(model, evaluation, config),
        "weights": ("mean of the last "
                    f"{min(config.average_last_epochs, config.epochs)} epoch checkpoints"
                    if config.average_last_epochs > 1 else "final epoch checkpoint"),
    }
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
    notes = [
        "velocity head trained at weight zero"
        if config.velocity_weight == 0
        else f"velocity head active on {velocity_labelled} labelled windows",
    ]
    report = {
        "protocol": args.protocol,
        "runtime": runtime_fingerprint(config.device),
        "notes": notes,
        "config": {
            "window": {"window_s": spec.window_s, "stride_s": spec.stride_s,
                       "min_frames": spec.min_frames, "clip_db": spec.clip_db},
            "train": {"epochs": config.epochs, "batch_size": config.batch_size,
                      "lr": config.lr, "seed": config.seed, "hidden": config.hidden,
                      "activity_weight": config.activity_weight,
                      "velocity_weight": config.velocity_weight,
                      "average_last_epochs": config.average_last_epochs,
                      "velocity_loss_scale_hz": config.velocity_loss_scale_hz,
                      "deterministic": config.deterministic,
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
        "final_model_eval": final_eval,
        "eval_predictions": predictions,
    }
    out = args.out or (args.sionna_dir / "detection" / "train_eval")
    out.mkdir(parents=True, exist_ok=True)
    event_config = EventEvalConfig(hold_off_s=args.event_hold_off_s,
                                   max_detection_latency_s=args.max_latency_s)
    # Labels cover the evaluated samples only. `samples` is the whole batch, and an
    # event-metric row is produced for every label entry -- including the ones too short to
    # yield a window -- so feeding it the full batch would add the training split's exposure
    # to the held-out false-alarm denominator.
    evaluated_ids = set(held_out) | {item.sample_id for item in evaluation}
    labels = {
        sample.sample_id: {"activity": sample.activity,
                           "onset_s": sample.imbalance_onset_s,
                           "impact_s": sample.first_impact_s,
                           # The false-alarm denominator is the sample's admitted channel
                           # exposure, identical to what the frame-rate baseline divides by;
                           # window centres alone would shorten it by one window per sample.
                           "duration_s": float(sample.duration_s)}
        for sample in samples if sample.sample_id in evaluated_ids
    }
    event_rows = event_metrics_from_predictions(
        predictions, labels, threshold=args.event_threshold, config=event_config)
    # Window-level separability is the statistic that survives a five-fall batch: the
    # event-level detection rate can only take six values here, while the AUC uses every
    # scored window. The interval resamples by physics session, never by window.
    report["window_auc"] = {
        "value": window_roc_auc([row["fall_probability"] for row in predictions],
                                [row["fall_label"] for row in predictions]),
        "ci95_cluster_bootstrap": cluster_bootstrap_auc_ci(
            [row["fall_probability"] for row in predictions],
            [row["fall_label"] for row in predictions],
            [row["sample_id"] for row in predictions],
            seed=config.seed),
        "windows": len(predictions),
        "positive_windows": int(sum(row["fall_label"] for row in predictions)),
        "clusters": len({row["sample_id"] for row in predictions}),
    }
    report["event_metrics"] = aggregate(event_rows)
    report["data_sufficiency"] = data_sufficiency(report["event_metrics"], args.protocol)
    notes.extend(report["data_sufficiency"]["shortfalls"])
    unscored = int(report["event_metrics"].get("adl_unscored_samples") or 0)
    if unscored:
        notes.append(f"{unscored} classified-ADL samples were too short for one window; "
                     "their exposure is in the false-alarm denominator with zero episodes")
    report["event_config"] = {**config_dict(event_config),
                              "threshold": float(args.event_threshold)}
    if args.event_threshold_sweep:
        report["event_threshold_sweep"] = {
            str(value): aggregate(event_metrics_from_predictions(
                predictions, labels, threshold=value, config=event_config))
            for value in (float(piece) for piece in args.event_threshold_sweep.split(","))
        }
    report_path = out / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"train report -> {report_path}")
    print(f"  windows train/eval {len(train)}/{len(evaluation)} "
          f"(fall windows in train: {positives}), held out: {sorted(held_out)}")
    print(f"  final loss {history[-1]['train_loss']:.4f}, "
          f"eval window accuracy {final_eval['eval_fall_accuracy']} "
          f"[{final_eval['weights']}], "
          f"window AUC {report['window_auc']['value']} "
          f"{report['window_auc']['ci95_cluster_bootstrap']} (by session), "
          f"velocity MAE {final_eval['eval_velocity_mae_hz']} Hz")
    sufficiency = report["data_sufficiency"]
    if sufficiency["claim_supported"]:
        print(f"  data sufficiency: supports a claim "
              f"({sufficiency['scorable_fall_samples']} scorable falls, "
              f"{sufficiency['adl_hours']:.3f} h ADL, {sufficiency['groups']} groups)")
    else:
        print("  data sufficiency: NOT a performance claim -- "
              + "; ".join(sufficiency["shortfalls"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
