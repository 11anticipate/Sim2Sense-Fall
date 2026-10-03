#!/usr/bin/env python3
"""Quantify foot-contact chatter in a trial npz (the visual-jitter metric).

The visible "limb jitter" of a physics tracker lives almost entirely in the
lowest vertices (feet): penetration depth, bounce peak-to-peak within a
1-second window, and the zero-crossing rate of the lowest-vertex height.
Whole-body averages (NJ, high_jerk) dilute this localized signal and must not
be used as the acceptance metric. Measured 2026-10-02, Trial_77 standing:
reference ~0.3 cm p-p / 0.7 Hz; raw hard-gain export ~10.9 cm p-p / 21.6 Hz
(zero-crossings overcount: the bounce band itself is 5-8 Hz riding on 1-3 Hz
weight shifts -- see the min-z spectrum in docs/progress.md).

Example:
    python scripts/protomotions/measure_contact_chatter.py \
        artifacts/.../tracker_Trial_77.npz [more.npz ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("measure_contact_chatter")


def chatter_metrics(vertices: np.ndarray, time_s: float) -> dict[str, float]:
    """Metrics from (T, Nv, 3) vertices: penetration, 1 s p-p, bounce rate."""
    lowest = vertices[:, :, 2].min(axis=1)
    dt = float(np.median(np.diff(time_s)))
    w = lowest - lowest.mean()
    crossings = int(((w[:-1] * w[1:]) < 0).sum())
    duration = float(time_s[-1] - time_s[0])
    window = max(int(round(1.0 / dt)), 2)
    starts = range(0, max(len(lowest) - window, 1), max(window // 2, 1))
    peak_to_peak = max(
        float(lowest[i : i + window].max() - lowest[i : i + window].min())
        for i in starts
    )
    return {
        "penetration_cm": float(lowest.min()) * 100.0,
        "window_peak_to_peak_cm": peak_to_peak * 100.0,
        "bounce_hz": crossings / duration,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", nargs="+", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print(f"{'npz':50s} {'穿透cm':>8s} {'1s峰峰cm':>10s} {'过零Hz':>8s}")
    for path in args.npz:
        data = np.load(path, allow_pickle=True)
        metrics = chatter_metrics(
            data["mesh_vertices_xyz"].astype(np.float64),
            data["time_physics_s"].astype(float),
        )
        print(
            f"{str(path):50s} {metrics['penetration_cm']:8.1f} "
            f"{metrics['window_peak_to_peak_cm']:10.1f} {metrics['bounce_hz']:8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
