#!/usr/bin/env python3
"""Zero-phase low-pass filter a trial npz's mesh trajectory in place.

This is the CSI-facing conditioning stage for ``physics_tracker`` exports: the
closed-loop tracker carries a foot-texture band that the Sionna/Detection
chain must not see (solver/PD texture, not human motion). Measured on the
r4 hard-gain export (2026-10-02): the lowest-vertex height carries a 5-8 Hz
bounce plus 12-20 Hz solver ripple, while real motion (strides, falls, weight
shifts) lives below ~3 Hz. An 8 Hz cutoff therefore does nothing visible;
the default 4 Hz cutoff removes bounce and ripple while keeping stride and
fall dynamics. Verification is via ``measure_contact_chatter``.

Example:
    python scripts/protomotions/filter_trial_mesh.py \
        --src artifacts/.../export_tracker_r4/tracker_Trial_77.npz \
        --dst artifacts/.../export_r4_filt4/tracker_Trial_77.npz --cutoff-hz 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

LOGGER = logging.getLogger("filter_trial_mesh")


def filter_vertices(verts: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass filter (T, Nv, 3) vertex trajectories along time."""
    nyquist = fps / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff {cutoff_hz} Hz outside (0, Nyquist {nyquist})")
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    t, nv, xyz = verts.shape
    flat = verts.reshape(t, nv * xyz)
    return filtfilt(b, a, flat, axis=0).reshape(t, nv, xyz)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--cutoff-hz", type=float, default=8.0)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = np.load(args.src, allow_pickle=True)
    time_s = data["time_physics_s"].astype(float)
    verts = data["mesh_vertices_xyz"].astype(np.float64)
    fps = 1.0 / float(np.median(np.diff(time_s)))
    filtered = filter_vertices(verts, fps, args.cutoff_hz, args.order)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dst,
        time_physics_s=time_s,
        mesh_vertices_xyz=filtered,
        mesh_faces=data["mesh_faces"],
    )
    delta = float(np.abs(filtered - verts).max())
    LOGGER.info(
        "wrote %s (%.0f Hz source, cutoff %.1f Hz order %d, max |delta| %.3f m)",
        args.dst, fps, args.cutoff_hz, args.order, delta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
