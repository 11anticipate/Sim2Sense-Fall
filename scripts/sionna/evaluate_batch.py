#!/usr/bin/env python3
"""Run the detection baseline over a batch of Sionna CIR samples.

Consumes the ``*.cir.npz`` / ``*.import.json`` pairs written by
``import_fall_mesh.py`` (see batch_generate.py), extracts per-frame channel
features, runs the trailing-baseline step detector and the window alarm
aggregation, and reports -- per sample and in aggregate -- whether an alarm
was raised, how fast it followed the label's imbalance onset (fall samples)
and how many alarm frames the ADL samples produced.

This is the consumer-contract exercise for the batch, not a performance
claim: nine samples cannot train or rank detectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection import (  # noqa: E402
    StepDetectorConfig,
    cir_features,
    detect_fall,
)
from sim2sense_fall.detection_data import iter_channel_samples  # noqa: E402
from sim2sense_fall.detection_eval import (  # noqa: E402
    EventEvalConfig,
    aggregate,
    config_dict,
    sample_event_metrics,
)
from sim2sense_fall.windowing import WindowConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sionna-dir", type=Path, required=True,
                        help="directory with *.cir.npz / *.import.json pairs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <sionna-dir>/detection)")
    parser.add_argument("--power-threshold-db", type=float, default=6.0)
    parser.add_argument("--delay-spread-threshold-ns", type=float, default=8.0)
    parser.add_argument("--baseline-tau-s", type=float, default=0.3)
    parser.add_argument("--window-length-s", type=float, default=0.2)
    parser.add_argument("--stride-s", type=float, default=0.1)
    parser.add_argument("--splits", type=Path, default=None,
                        help="splits.json from assign_splits.py; with --only-split it "
                             "restricts the evaluation to one split so the baseline and a "
                             "trained model can be compared on the same held-out sessions")
    parser.add_argument("--only-split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--hold-off-s", type=float, default=5.0,
                        help="alarm episodes closer together than this count as one event")
    parser.add_argument("--max-latency-s", type=float, default=5.0,
                        help="an episode starting later than this after the imbalance onset "
                             "is a miss, not a slow detection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or (args.sionna_dir / "detection")
    out.mkdir(parents=True, exist_ok=True)
    detector = StepDetectorConfig(
        baseline_tau_s=args.baseline_tau_s,
        power_threshold_db=args.power_threshold_db,
        delay_spread_threshold_s=args.delay_spread_threshold_ns * 1e-9,
    )
    event_config = EventEvalConfig(hold_off_s=args.hold_off_s,
                                   max_detection_latency_s=args.max_latency_s)
    window = WindowConfig(
        window_length_s=args.window_length_s, stride_s=args.stride_s,
        threshold=0.8, min_consecutive_positive_windows=2,
    )
    rows: list[dict] = []
    # One loader for identity, label resolution and failure admission: it globs the
    # per-session namespaces the batch writes and derives the imbalance onset from a
    # keyboard session's measured root trajectory (a trial carries its own).
    split_of: dict[str, str] = {}
    if args.splits is not None:
        payload = json.loads(args.splits.read_text(encoding="utf-8"))
        split_of = {str(row["sample_id"]): str(row["split"]) for row in payload["samples"]}
        if args.only_split is None:
            print("note: --splits given without --only-split; evaluating every sample")
    skipped = 0
    for sample in iter_channel_samples(args.sionna_dir):
        if args.only_split is not None and split_of.get(sample.sample_id) != args.only_split:
            skipped += 1
            continue
        result = detect_fall(
            cir_features(sample.cir, sample.delay_s), sample.time_s, detector, window,
        )
        onset_s, impact_s = sample.imbalance_onset_s, sample.first_impact_s
        first_alarm = result["first_alarm_s"]
        latency = (
            first_alarm - onset_s
            if (onset_s is not None and first_alarm is not None) else None
        )
        event = sample_event_metrics(
            sample.time_s, result["alarm_flag"], activity=sample.activity,
            onset_s=onset_s, impact_s=impact_s, config=event_config,
            # Same exposure definition the window model is divided by: the admitted channel
            # duration of the sample, not whatever span this detector happened to score.
            duration_s=float(sample.duration_s),
        )
        rows.append({
            "sample_id": sample.sample_id,
            # Episode and false-alarm rates are aggregated per physics session, because
            # segments cut from one session are not independent measurements.
            "group": sample.sample_id.rsplit("/", 1)[0] or "ungrouped",
            "cir_file": str(sample.cir_path),
            "activity": sample.activity,
            "source_event_label": sample.event_label,
            "imbalance_onset_s": onset_s,
            "onset_source": sample.onset_source,
            "first_impact_s": impact_s,
            "first_alarm_s": first_alarm,
            "detection_latency_s": latency,
            "window_alarmed": result["window_alarmed"],
            "alarm_frame_count": int(result["alarm_frames"].size),
            "duration_s": float(sample.duration_s),
            "event": event,
        })

    falls = [row for row in rows if row["activity"] == "fall"]
    adls = [row for row in rows if row["activity"] == "adl"]
    detected = [row for row in falls if row["window_alarmed"]]
    latencies = [row["detection_latency_s"] for row in detected
                 if row["detection_latency_s"] is not None]
    adl_alarm_seconds = 0.0
    for row in adls:
        stamps = np.load(row["cir_file"])["timestamp_s"]
        adl_alarm_seconds += row["alarm_frame_count"] * float(np.median(np.diff(stamps)))
    adl_hours = sum(row["duration_s"] for row in adls) / 3600.0
    summary = {
        "evaluation_scope": {
            "splits_file": str(args.splits) if args.splits else None,
            "only_split": args.only_split,
            "samples_in_dir": len(rows) + skipped,
            "samples_evaluated": len(rows),
        },
        "detector": {
            "kind": "trailing_baseline_step_change",
            **{key: getattr(detector, key) for key in
               ("baseline_tau_s", "power_threshold_db", "delay_spread_threshold_s")},
            "window": {"window_length_s": window.window_length_s,
                       "stride_s": window.stride_s, "threshold": window.threshold,
                       "min_consecutive_positive_windows":
                           window.min_consecutive_positive_windows},
        },
        "fall_detection_rate": (len(detected) / len(falls)) if falls else None,
        "detection_latency_s_median": (
            float(np.median(latencies)) if latencies else None
        ),
        "adl_false_alarm_seconds": round(adl_alarm_seconds, 3),
        "adl_false_alarm_per_hour": (
            round(adl_alarm_seconds / adl_hours, 3) if adl_hours else None
        ),
        # Event-level contract: episodes (not seconds), a latency budget that turns a
        # late alarm into a miss, and false alarms per hour aggregated per session.
        "event_metrics": aggregate([row["event"] | {"group": row["group"]} for row in rows]),
        "event_config": config_dict(event_config),
        "samples": rows,
    }
    report_path = out / "detection_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"detection report -> {report_path}")
    print(f"  fall detection {len(detected)}/{len(falls)}, "
          f"latency median {summary['detection_latency_s_median']}, "
          f"ADL false alarms {summary['adl_false_alarm_seconds']} s "
          f"({summary['adl_false_alarm_per_hour']}/h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
