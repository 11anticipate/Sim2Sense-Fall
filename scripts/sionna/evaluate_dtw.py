#!/usr/bin/env python3
"""Evaluate the DTW template baseline over a batch of Sionna CIR samples.

Builds a fall template from the batch's fall samples and scores every sample
with :func:`sim2sense_fall.dtw_baseline.detect_fall_template`, reporting the
same event-level numbers as the step detector (detection rate, latency,
ADL false-alarm seconds per hour). Fall samples are scored *leave-one-out* —
each is matched against a template built from the other falls — so the fall
detection rate is not in-sample; ADL samples are scored against the full
template, which is the honest deployment direction (template from falls,
stream from everything else).

Like every evaluation on this batch, the numbers are a consumer-contract
exercise over nine samples, not a performance claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection_data import ChannelSample, iter_channel_samples  # noqa: E402
from sim2sense_fall.dtw_baseline import (  # noqa: E402
    TemplateConfig,
    build_fall_template,
    detect_fall_template,
)
from sim2sense_fall.windowing import WindowConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sionna-dir", type=Path, required=True,
                        help="directory with *.cir.npz / *.import.json pairs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output file (default: <sionna-dir>/detection/dtw_report.json)")
    parser.add_argument("--window-s", type=float, default=0.8,
                        help="template/match window; must fit inside the samples")
    parser.add_argument("--resample-points", type=int, default=64)
    parser.add_argument("--band-fraction", type=float, default=0.15)
    parser.add_argument("--dba-iterations", type=int, default=3)
    parser.add_argument("--alarm-window-s", type=float, default=0.2)
    parser.add_argument("--alarm-stride-s", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-consecutive", type=int, default=2)
    return parser.parse_args(argv)


def score_sample(sample: ChannelSample, template: np.ndarray,
                 config: TemplateConfig, window: WindowConfig) -> dict:
    result = detect_fall_template(
        sample.cir, sample.delay_s, sample.time_s, template, config, window
    )
    first_alarm = result["first_alarm_s"]
    latency = (
        first_alarm - sample.imbalance_onset_s
        if sample.imbalance_onset_s is not None and first_alarm is not None else None
    )
    return {
        "sample_id": sample.sample_id,
        "activity": sample.activity,
        "event_label": sample.event_label,
        "imbalance_onset_s": sample.imbalance_onset_s,
        "first_alarm_s": first_alarm,
        "detection_latency_s": latency,
        "window_alarmed": result["window_alarmed"],
        "peak_score": float(np.max(result["frame_scores"])) if len(sample.time_s) else 0.0,
        "matched_windows": int(len(result["window_centers_s"])),
        "duration_s": float(sample.time_s[-1] - sample.time_s[0]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    samples = iter_channel_samples(args.sionna_dir)
    falls = [sample for sample in samples if sample.activity == "fall"]
    adls = [sample for sample in samples if sample.activity != "fall"]
    if not falls:
        print("no fall samples: a template cannot be built")
        return 1
    config = TemplateConfig(
        resample_points=args.resample_points,
        dba_iterations=args.dba_iterations,
        dtw_band_fraction=args.band_fraction,
        window_s=args.window_s,
    )
    window = WindowConfig(
        window_length_s=args.alarm_window_s, stride_s=args.alarm_stride_s,
        threshold=args.threshold, min_consecutive_positive_windows=args.min_consecutive,
    )
    notes = []

    def template_from(falls_subset: list[ChannelSample]) -> np.ndarray | None:
        try:
            return build_fall_template(
                [sample.cir for sample in falls_subset],
                [sample.delay_s for sample in falls_subset],
                [sample.time_s for sample in falls_subset],
                [sample.imbalance_onset_s for sample in falls_subset],
                config,
            )
        except ValueError as error:
            notes.append(f"template build failed: {error}")
            return None

    rows: list[dict] = []
    if len(falls) >= 2:
        for held_out in falls:
            template = template_from([sample for sample in falls if sample is not held_out])
            if template is None:
                continue
            rows.append(score_sample(held_out, template, config, window))
        notes.append("fall samples scored leave-one-fall-out")
    else:
        template = template_from(falls)
        notes.append("single fall sample: template is in-sample for it")
        if template is not None:
            rows.append(score_sample(falls[0], template, config, window))
    full_template = template_from(falls)
    if full_template is None:
        print("template construction failed for the batch; see notes")
        return 1
    for adl in adls:
        rows.append(score_sample(adl, full_template, config, window))

    fall_rows = [row for row in rows if row["activity"] == "fall"]
    detected = [row for row in fall_rows if row["window_alarmed"]]
    latencies = [row["detection_latency_s"] for row in detected
                 if row["detection_latency_s"] is not None]
    # ADL false-alarm duration: seconds with frame score at/above the alarm threshold
    adl_alarm_seconds = 0.0
    for adl in adls:
        result = detect_fall_template(
            adl.cir, adl.delay_s, adl.time_s, full_template, config, window
        )
        above = result["frame_scores"] >= window.threshold
        adl_alarm_seconds += float(above.sum()) * float(np.median(np.diff(adl.time_s)))
    adl_hours = sum(row["duration_s"] for row in rows if row["activity"] != "fall") / 3600.0
    summary = {
        "detector": {
            "kind": "dtw_template_matching",
            **{
                "resample_points": config.resample_points,
                "dba_iterations": config.dba_iterations,
                "dtw_band_fraction": config.dtw_band_fraction,
                "window_s": config.window_s,
            },
            "null_distance": float(np.abs(full_template).mean()),
            "alarm_window": {
                "window_length_s": window.window_length_s,
                "stride_s": window.stride_s,
                "threshold": window.threshold,
                "min_consecutive_positive_windows": window.min_consecutive_positive_windows,
            },
        },
        "notes": notes,
        "fall_detection_rate": (len(detected) / len(fall_rows)) if fall_rows else None,
        "detection_latency_s_median": float(np.median(latencies)) if latencies else None,
        "adl_false_alarm_seconds": round(adl_alarm_seconds, 3),
        "adl_false_alarm_per_hour": round(adl_alarm_seconds / adl_hours, 3) if adl_hours else None,
        "samples": rows,
    }
    out = args.out or (args.sionna_dir / "detection" / "dtw_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"dtw report -> {out}")
    print(f"  fall detection {len(detected)}/{len(fall_rows)}, "
          f"latency median {summary['detection_latency_s_median']}, "
          f"ADL false alarms {summary['adl_false_alarm_seconds']} s "
          f"({summary['adl_false_alarm_per_hour']}/h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
