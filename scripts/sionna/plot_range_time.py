#!/usr/bin/env python3
"""Render range-time channel images for a batch of Sionna CIR samples.

Converts every admitted ``*.cir.npz`` into its range-time map (per-tap dB
power over time, referenced against the no-human baseline when available)
and saves one PNG per sample, with the recorded imbalance onset and impact
overlaid for fall samples. The doppler panel is added on request and only
when the sample's time grid is uniform enough for an FFT.

This is the input representation the stage-9 detector will consume, made
inspectable; the images carry no performance claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection_data import iter_channel_samples  # noqa: E402
from sim2sense_fall.range_time import (  # noqa: E402
    RangeTimeConfig,
    doppler_map,
    range_time_map,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sionna-dir", type=Path, required=True,
                        help="directory with *.cir.npz / *.import.json pairs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <sionna-dir>/detection/range_time)")
    parser.add_argument("--clip-db", type=float, default=40.0)
    parser.add_argument("--with-doppler", action="store_true",
                        help="add a doppler panel for samples with a uniform time grid")
    return parser.parse_args(argv)


def plot_sample(sample, config: RangeTimeConfig, out_path: Path, with_doppler: bool) -> str:
    map_ = range_time_map(
        sample.cir, sample.delay_s, sample.time_s, sample.baseline_cir, config
    )
    delay_ns = map_.delay_s * 1e9
    panels = 2 if with_doppler else 1
    fig, axes = plt.subplots(panels, 1, figsize=(9, 3.2 * panels), squeeze=False)
    axis = axes[0][0]
    image = axis.imshow(
        map_.magnitude.T, aspect="auto", origin="lower", cmap="magma",
        extent=[map_.time_s[0], map_.time_s[-1], delay_ns[0], delay_ns[-1]],
        vmin=0.0, vmax=1.0,
    )
    if sample.activity == "fall" and sample.imbalance_onset_s is not None:
        axis.axvline(sample.imbalance_onset_s, color="cyan", ls="--", lw=1, label="onset")
        if sample.first_impact_s is not None:
            axis.axvline(sample.first_impact_s, color="red", ls="--", lw=1, label="impact")
        axis.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=axis, label="dB vs baseline (clipped, normalised)")
    axis.set_title(f"{sample.sample_id} [{sample.activity}/{sample.event_label}] "
                   f"ref={map_.reference}")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("delay (ns)")
    note = "range-time"
    if with_doppler:
        try:
            doppler = doppler_map(
                sample.cir, sample.time_s, sample.baseline_cir, config
            )
        except ValueError as error:
            axis.text(0.02, 0.02, f"doppler skipped: {error}", transform=axis.transAxes,
                      fontsize=7, color="white")
        else:
            axis = axes[1][0]
            sample_rate = 1.0 / float(np.median(np.diff(map_.time_s)))
            image = axis.imshow(
                doppler, aspect="auto", origin="lower", cmap="viridis",
                extent=[0.0, sample_rate, delay_ns[0], delay_ns[-1]],
                vmin=0.0, vmax=1.0,
            )
            axis.set_title("doppler-range map (per tap, full-window FFT)")
            axis.set_xlabel("frequency (Hz)")
            axis.set_ylabel("delay (ns)")
            note = "range-time+doppler"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return note


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or (args.sionna_dir / "detection" / "range_time")
    out.mkdir(parents=True, exist_ok=True)
    config = RangeTimeConfig(clip_db=args.clip_db)
    samples = iter_channel_samples(args.sionna_dir)
    if not samples:
        print(f"no admitted channel samples under {args.sionna_dir}")
        return 1
    for sample in samples:
        note = plot_sample(sample, config, out / f"{sample.sample_id}.png", args.with_doppler)
        print(f"{sample.sample_id}: {note} -> {out / (sample.sample_id + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
