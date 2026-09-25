#!/usr/bin/env python3
"""Plot the fall signature from an import_fall_mesh CIR archive.

Post-processing only: reads the ``.cir.npz`` written by the Sionna RT import
and renders, without any simulator dependency beyond matplotlib,

1. a CIR waterfall -- |amplitude| in dB over (delay, frame), the multipath
   structure evolving as the body falls;
2. a channel-metric timeline -- per-frame received power and RMS delay spread,
   which is the "what changed, when" view for labelling;
3. the empty-scene baseline for reference.

Base directory for the skill examples is irrelevant here; the script is
invoked with the Sionna venv's python (matplotlib lives there).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cir_npz", type=Path, help="the .cir.npz written by import_fall_mesh.py")
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default: beside the npz)"
    )
    return parser.parse_args(argv)


def received_power_db(cir: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.sum(np.abs(cir) ** 2, axis=-1))


def rms_delay_spread_s(cir: np.ndarray, delay_s: np.ndarray) -> np.ndarray:
    power = np.abs(cir) ** 2
    total = np.sum(power, axis=-1, keepdims=True)
    mean = np.sum(power * delay_s, axis=-1) / total[..., 0]
    spread = np.sqrt(np.sum(power * (delay_s - mean[..., None]) ** 2, axis=-1) / total[..., 0])
    return spread


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or args.cir_npz.parent
    archive = np.load(args.cir_npz)
    cir = archive["cir"].astype(np.complex128)
    delay_s = archive["delay_s"].astype(np.float64)
    times = archive["timestamp_s"].astype(np.float64)
    baseline = archive["baseline_cir"].astype(np.complex128)
    import_json = args.cir_npz.with_suffix("").with_suffix(".import.json")
    label = ""
    impact_s: float | None = None
    if import_json.exists():
        meta = json.loads(import_json.read_text(encoding="utf-8"))
        label = str(meta.get("source_event_label", ""))
        # The trial JSON records the label's impact instant; the import JSON
        # only references its path.
        trial_path = (meta.get("source") or {}).get("source")
        if trial_path and Path(trial_path).exists():
            trial = json.loads(Path(trial_path).read_text(encoding="utf-8"))
            impact_s = (trial.get("label") or {}).get("first_impact_s")

    waterfall_db = 20.0 * np.log10(np.maximum(np.abs(cir), 1e-12))
    power_db = received_power_db(cir)
    spread_s = rms_delay_spread_s(cir, delay_s)
    baseline_power_db = float(received_power_db(baseline[None, :])[0])

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]}, sharex=False
    )
    axis = axes[0]
    image = axis.imshow(
        waterfall_db,
        aspect="auto",
        origin="lower",
        extent=[delay_s[0] * 1e9, delay_s[-1] * 1e9, times[0], times[-1]],
        cmap="magma",
        vmin=waterfall_db.max() - 60.0,
        vmax=waterfall_db.max(),
    )
    if impact_s is not None:
        axis.axhline(
            impact_s, color="cyan", linestyle="--", linewidth=1.0, label=f"impact {impact_s:.2f} s"
        )
        axis.legend(loc="upper right", fontsize=8)
    axis.set_xlabel("delay (ns)")
    axis.set_ylabel("trial time (s)")
    axis.set_title(f"CIR waterfall -- {label or 'trial'} (|amplitude| dB, 60 dB span)")
    fig.colorbar(image, ax=axis, label="dB")

    axis = axes[1]
    axis.plot(times, power_db - baseline_power_db, marker="o", markersize=3,
              label="received power − empty scene (dB)")
    axis.plot(times, spread_s * 1e9, marker="s", markersize=3, label="RMS delay spread (ns)")
    if impact_s is not None:
        axis.axvline(impact_s, color="cyan", linestyle="--", linewidth=1.0)
    axis.set_xlabel("trial time (s)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    axis.set_title("channel metrics over the motion")

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    destination = out / (args.cir_npz.stem.replace(".cir", "") + "_cir_signature.png")
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
