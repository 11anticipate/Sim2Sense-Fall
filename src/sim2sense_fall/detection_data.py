"""Loading Sionna channel samples (*.cir.npz + *.import.json) with labels.

One channel sample is the unit every detection consumer works on: a complex
CIR sequence over a delay grid and frame times, the no-human baseline when
the importer saved one, and the labels derived from the source physics trial
or keyboard session — never from file names. Import-level failures are kept
out unless explicitly requested, mirroring the batch driver's admission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ChannelSample:
    """One imported CIR sample plus the labels it carries."""

    sample_id: str
    cir_path: Path
    import_path: Path
    cir: np.ndarray
    delay_s: np.ndarray
    time_s: np.ndarray
    baseline_cir: np.ndarray | None
    activity: str
    event_label: str | None
    imbalance_onset_s: float | None
    first_impact_s: float | None


def _event_times(import_payload: dict, import_path: Path) -> tuple[float | None, float | None]:
    """Resolve (imbalance onset, impact) from the referenced physics trial."""

    source_path = (import_payload.get("source") or {}).get("source")
    if not source_path or not Path(source_path).exists():
        return None, None
    trial = json.loads(Path(source_path).read_text(encoding="utf-8"))
    label = trial.get("label") or {}
    onset = label.get("imbalance_onset_s")
    impact = label.get("first_impact_s")
    return (
        float(onset) if onset is not None else None,
        float(impact) if impact is not None else None,
    )


def load_channel_sample(cir_path: Path, import_path: Path | None = None) -> ChannelSample:
    """Load one CIR sample; validates shape, finiteness and time monotonicity."""

    cir_path = Path(cir_path)
    import_path = (
        Path(import_path) if import_path
        else cir_path.with_suffix("").with_suffix(".import.json")
    )
    payload = json.loads(import_path.read_text(encoding="utf-8"))
    archive = np.load(cir_path)
    cir = np.asarray(archive["cir"])
    delay_s = np.asarray(archive["delay_s"], dtype=np.float64)
    time_s = np.asarray(archive["timestamp_s"], dtype=np.float64)
    if cir.ndim != 2 or not np.iscomplexobj(cir) or not np.isfinite(cir).all():
        raise ValueError(f"{cir_path.name}: cir must be a finite complex (frames, taps) array")
    if delay_s.shape != (cir.shape[1],) or not np.all(np.diff(delay_s) > 0):
        raise ValueError(f"{cir_path.name}: delay grid must align and be strictly increasing")
    if (time_s.shape != (cir.shape[0],) or not np.isfinite(time_s).all()
            or np.any(np.diff(time_s) <= 0)):
        raise ValueError(f"{cir_path.name}: time axis must be per-frame and strictly increasing")
    baseline = archive["baseline_cir"] if "baseline_cir" in archive else None
    onset_s, impact_s = _event_times(payload, import_path)
    return ChannelSample(
        sample_id=str(payload.get("channel_sample_id") or cir_path.name.removesuffix(".cir.npz")),
        cir_path=cir_path,
        import_path=import_path,
        cir=cir,
        delay_s=delay_s,
        time_s=time_s,
        baseline_cir=np.asarray(baseline) if baseline is not None else None,
        activity=str(payload.get("activity") or "unknown"),
        event_label=(
            str(payload["source_event_label"]) if payload.get("source_event_label") else None
        ),
        imbalance_onset_s=onset_s,
        first_impact_s=impact_s,
    )


def iter_channel_samples(sionna_dir: Path, include_failures: bool = False) -> list[ChannelSample]:
    """All admitted channel samples of a Sionna output directory, sorted by id.

    Samples whose importer reported failures are skipped unless
    ``include_failures`` is set, matching the batch driver's admission rule.
    """

    directory = Path(sionna_dir)
    samples: list[ChannelSample] = []
    for cir_path in sorted(directory.glob("*.cir.npz")):
        import_path = cir_path.with_suffix("").with_suffix(".import.json")
        if not import_path.exists():
            continue
        payload = json.loads(import_path.read_text(encoding="utf-8"))
        if payload.get("failures") and not include_failures:
            continue
        samples.append(load_channel_sample(cir_path, import_path))
    return samples
