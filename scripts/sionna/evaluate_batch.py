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
    return parser.parse_args(argv)


def event_times(import_payload: dict) -> tuple[float | None, float | None]:
    """Resolve (imbalance onset, impact) from the referenced physics trial."""

    trial_path = (import_payload.get("source") or {}).get("source")
    if not trial_path or not Path(trial_path).exists():
        return None, None
    trial = json.loads(Path(trial_path).read_text(encoding="utf-8"))
    label = trial.get("label") or {}
    return label.get("imbalance_onset_s"), label.get("first_impact_s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or (args.sionna_dir / "detection")
    out.mkdir(parents=True, exist_ok=True)
    detector = StepDetectorConfig(
        baseline_tau_s=args.baseline_tau_s,
        power_threshold_db=args.power_threshold_db,
        delay_spread_threshold_s=args.delay_spread_threshold_ns * 1e-9,
    )
    window = WindowConfig(
        window_length_s=args.window_length_s, stride_s=args.stride_s,
        threshold=0.8, min_consecutive_positive_windows=2,
    )
    rows: list[dict] = []
    for cir_path in sorted(args.sionna_dir.glob("*.cir.npz")):
        import_path = cir_path.with_suffix("").with_suffix(".import.json")
        if not import_path.exists():
            continue
        payload = json.loads(import_path.read_text(encoding="utf-8"))
        if payload.get("failures"):
            continue
        archive = np.load(cir_path)
        onset_s, impact_s = event_times(payload)
        result = detect_fall(
            cir_features(archive["cir"], archive["delay_s"]),
            archive["timestamp_s"].astype(np.float64),
            detector, window,
        )
        first_alarm = result["first_alarm_s"]
        latency = (
            first_alarm - onset_s
            if (onset_s is not None and first_alarm is not None) else None
        )
        rows.append({
            "sample_id": payload.get("channel_sample_id") or cir_path.stem,
            "cir_file": str(cir_path),
            "activity": payload.get("activity"),
            "source_event_label": payload.get("source_event_label"),
            "imbalance_onset_s": onset_s,
            "first_impact_s": impact_s,
            "first_alarm_s": first_alarm,
            "detection_latency_s": latency,
            "window_alarmed": result["window_alarmed"],
            "alarm_frame_count": int(result["alarm_frames"].size),
            "duration_s": float(archive["timestamp_s"][-1] - archive["timestamp_s"][0]),
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
