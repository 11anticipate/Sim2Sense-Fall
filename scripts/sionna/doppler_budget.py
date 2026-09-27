#!/usr/bin/env python3
"""Pre-registered Doppler sampling budget from simulation ground truth.

For every (mesh sequence, link geometry) pair given on the command line this
computes, per frame, the maximum Doppler shift over *all* body points —
feet included, since the fastest limb defines aliasing, not the root —
using the bistatic formula ``f_D = (v·(û_tx + û_rx))·f_c/c`` with the link's
recorded TX/RX positions and 3.5 GHz carrier. The session peak doubles into
a Nyquist requirement that each candidate slow-time rate either meets or
misses.

The output is a pre-registered verdict (JSON + stdout): it decides whether
the planned 50 Hz batch is dense enough for unaliased fall Doppler *before*
any batch is produced. Recomputing this from aliased captures afterwards is
exactly what this script exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.doppler import (  # noqa: E402
    doppler_budget,
    frame_peak_doppler,
    geometry_factors,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", metavar=("MESH_NPZ", "IMPORT_JSON"),
                        required=True,
                        help="mesh sequence (trial npz or export mesh.npz) plus the "
                             "CIR import.json carrying the link geometry and carrier")
    parser.add_argument("--rates-hz", type=float, nargs="*",
                        default=[30.0, 50.0, 100.0, 120.0],
                        help="candidate slow-time sampling rates to judge")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON (default: alongside the first mesh)")
    return parser.parse_args(argv)


def mesh_time(archive) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and their time axis from either a trial or an export npz."""

    if "mesh_vertices_xyz" not in archive:
        raise ValueError("archive has no mesh_vertices_xyz")
    vertices = np.asarray(archive["mesh_vertices_xyz"])
    if "time_physics_s" in archive:
        return vertices, np.asarray(archive["time_physics_s"], dtype=np.float64)
    if "time_s" in archive:
        return vertices, np.asarray(archive["time_s"], dtype=np.float64)
    raise ValueError("archive has no time axis (time_physics_s or time_s)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for mesh_path, import_path in args.pair:
        payload = json.loads(Path(import_path).read_text(encoding="utf-8"))
        tx = np.asarray(payload["transmitter"], dtype=np.float64)
        rx = np.asarray(payload["receiver"], dtype=np.float64)
        carrier = float(payload["frequency_hz"])
        archive = np.load(mesh_path)
        vertices, time = mesh_time(archive)
        doppler, speed = frame_peak_doppler(vertices, time, tx, rx, carrier)
        factor = float(geometry_factors(vertices[0], tx, rx).max())
        budget = doppler_budget(
            doppler, speed, factor, tuple(args.rates_hz),
        )
        rows.append({
            "mesh": str(mesh_path),
            "import": str(import_path),
            "frames": int(len(time)),
            "duration_s": float(time[-1] - time[0]),
            "sample_rate_hz": float(1.0 / np.median(np.diff(time))),
            "carrier_hz": carrier,
            "tx_xyz": tx.tolist(),
            "rx_xyz": rx.tolist(),
            "geometry_factor_max": budget.geometry_factor_max,
            "peak_body_speed_m_s": round(budget.peak_body_speed_m_s, 3),
            "peak_doppler_hz": round(budget.peak_doppler_hz, 2),
            "nyquist_rate_hz": round(budget.nyquist_rate_hz, 2),
            "sufficient_rates_hz": list(budget.sufficient),
            "insufficient_rates_hz": list(budget.insufficient),
        })
        print(f"{Path(mesh_path).name}: peak |f_D| {budget.peak_doppler_hz:.1f} Hz "
              f"(body speed {budget.peak_body_speed_m_s:.2f} m/s, "
              f"geometry factor {budget.geometry_factor_max:.2f}) "
              f"-> Nyquist {budget.nyquist_rate_hz:.1f} Hz; "
              f"sufficient {budget.sufficient or 'none'}, "
              f"insufficient {budget.insufficient or 'none'}")
    out = args.out or (Path(args.pair[0][0]).parent / "doppler_budget.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rates_hz": args.rates_hz, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"budget -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
