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

from .doppler import frame_peak_doppler


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
    # How ``imbalance_onset_s`` was obtained: a trial's own annotation, or a value
    # derived from the measured root tilt. Never present a derived onset as annotated.
    onset_source: str | None = None
    transmitter_xyz: tuple[float, ...] | None = None
    receiver_xyz: tuple[float, ...] | None = None
    carrier_hz: float | None = None

    @property
    def duration_s(self) -> float:
        """Span of the traced stream in seconds; false alarms per hour need a denominator."""

        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0


@dataclass(frozen=True, slots=True)
class MeshMotion:
    """The mesh stream a CIR sample was rendered from, with its own time axis.

    A CIR sample's ``frame_index`` indexes the *native* source — trial
    physics frames (120 Hz) for push trials, the keyboard recording for
    session exports — while the per-sample export mesh may be downsampled
    (30 Hz). Velocity labels are therefore computed on whatever stream is
    available and interpolated onto the CIR frame times.
    """

    vertices: np.ndarray
    time_s: np.ndarray


# Pre-registered 2026-09-27 for deriving a keyboard session's imbalance onset from the
# measured body state. The project rule forbids using the F key-press time as the
# imbalance start, so the onset is taken from the root trajectory: the last moment
# before the recorded impact at which the pelvis was still within
# FALL_ONSET_PLATEAU_TOL_M of its pre-fall height -- i.e. where the drop begins.
# Height-based rather than velocity-based on purpose: a contact bounce must not be able
# to erase the label. The tolerance (5 cm) is above the measured walking root bob
# (2 cm), so a stride cannot be mistaken for the start of a fall, and the descent must
# still exceed FALL_ONSET_MIN_DESCENT_M for the label to exist at all.
# Measured on batch train01 (6 fall samples): the derived onset lands 0.15 - 0.43 s
# after the F key press and 0.17 - 0.52 s before the recorded impact, over a
# 0.37 - 0.44 m drop. So the key press is *not* the imbalance start -- they differ by
# up to 0.43 s -- and that gap is now measured rather than assumed.
FALL_ONSET_MIN_DESCENT_M = 0.15
FALL_ONSET_PLATEAU_TOL_M = 0.05
ONSET_SOURCE_SESSION = "derived:root_descent_to_impact"
ONSET_SOURCE_TRIAL = "trial:imbalance_onset_s"


def session_event_times(
    manifest_path: Path,
    sample_stem: str,
    *,
    min_descent_m: float = FALL_ONSET_MIN_DESCENT_M,
    plateau_tol_m: float = FALL_ONSET_PLATEAU_TOL_M,
) -> tuple[float | None, float | None, str | None]:
    """(onset, impact, source) in the sample's own time origin, for a keyboard session.

    A scripted physics trial carries an explicit ``imbalance_onset_s``. A keyboard
    session does not: its fall record has the key-press request, the measured impact and
    the moment the body was judged lying. The onset is therefore derived from the
    measured root height as the last pre-impact moment that still stood
    within ``plateau_tol_m`` of its pre-fall height (see the constants above), and the returned
    source names itself as derived so no caller can present it as an annotated
    imbalance start.

    Fails closed: no impact, no control record, or a total drop smaller than
    ``min_descent_m`` yields a ``None`` onset, and the sample contributes no positive
    windows -- exactly as an unlabeled trial does. The impact time itself is still
    returned so the caller can see *why* the label is missing.
    """

    for name, value in (("min_descent_m", min_descent_m), ("plateau_tol_m", plateau_tol_m)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    row = next((item for item in manifest.get("samples", [])
                if str(item.get("sample_id")) == sample_stem), None)
    if row is None:
        return None, None, None
    impact_abs = (row.get("label") or {}).get("first_impact_s")
    span = row.get("session_time_span_s") or [None]
    start_abs, run = span[0], row.get("source_run")
    if impact_abs is None or start_abs is None or not run:
        return None, None, None
    control_path = Path(run) / "control.npz"
    if not control_path.is_file():
        return None, None, None
    with np.load(control_path) as data:
        times = np.asarray(data["time_s"], dtype=float)
        heights = np.asarray(data["root"], dtype=float)[:, 2]
    before = np.where((times <= float(impact_abs)) & (times >= float(start_abs)))[0]
    if not len(before):
        return None, None, None
    window = heights[before]
    impact_height = float(window[-1])
    peak = float(window.max())
    impact = float(impact_abs) - float(start_abs)
    if peak - impact_height < min_descent_m:
        return None, float(impact), None
    on_plateau = np.where(window >= peak - plateau_tol_m)[0]
    onset = float(times[before[int(on_plateau[-1])]]) - float(start_abs)
    if not 0.0 <= onset <= max(impact, 0.0):
        return None, float(impact), None
    return onset, float(impact), ONSET_SOURCE_SESSION


def _event_times(
    import_payload: dict, import_path: Path, sample_stem: str | None = None
) -> tuple[float | None, float | None, str | None]:
    """Resolve (imbalance onset, impact, label source) from the referenced run."""

    source = (import_payload.get("source") or {}).get("source")
    if not source or not Path(source).exists():
        return None, None, None
    path = Path(source)
    if path.name == "manifest.json":
        return session_event_times(path, sample_stem or path.parent.name)
    trial = json.loads(path.read_text(encoding="utf-8"))
    label = trial.get("label") or {}
    onset = label.get("imbalance_onset_s")
    impact = label.get("first_impact_s")
    return (
        float(onset) if onset is not None else None,
        float(impact) if impact is not None else None,
        ONSET_SOURCE_TRIAL if onset is not None else None,
    )


def load_channel_sample(
    cir_path: Path, import_path: Path | None = None, *, sample_id: str | None = None
) -> ChannelSample:
    """Load one CIR sample; validates shape, finiteness and time monotonicity.

    ``sample_id`` overrides the id recorded in the import report. Batched session
    exports repeat segment ids across sessions (every session has a ``stand_00``) and
    the importer prefixes only the scene name, so the directory namespace has to become
    part of the identity -- otherwise two different measurements train as one sample.
    """

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
    onset_s, impact_s, onset_source = _event_times(
        payload, import_path, cir_path.name.removesuffix(".cir.npz"))
    transmitter = payload.get("transmitter")
    receiver = payload.get("receiver")
    return ChannelSample(
        sample_id=str(sample_id or payload.get("channel_sample_id")
                      or cir_path.name.removesuffix(".cir.npz")),
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
        onset_source=onset_source,
        first_impact_s=impact_s,
        transmitter_xyz=tuple(float(v) for v in transmitter) if transmitter else None,
        receiver_xyz=tuple(float(v) for v in receiver) if receiver else None,
        carrier_hz=(float(payload["frequency_hz"]) if payload.get("frequency_hz") else None),
    )


def iter_channel_samples(sionna_dir: Path, include_failures: bool = False) -> list[ChannelSample]:
    """All admitted channel samples of a Sionna output directory, sorted by id.

    Samples whose importer reported failures are skipped unless
    ``include_failures`` is set, matching the batch driver's admission rule.
    """

    directory = Path(sionna_dir)
    samples: list[ChannelSample] = []
    for cir_path in sorted(directory.glob("**/*.cir.npz")):
        import_path = cir_path.with_suffix("").with_suffix(".import.json")
        if not import_path.exists():
            continue
        payload = json.loads(import_path.read_text(encoding="utf-8"))
        if payload.get("failures") and not include_failures:
            continue
        namespace = cir_path.parent.relative_to(directory).as_posix()
        stem = cir_path.name.removesuffix(".cir.npz")
        ident = f"{namespace}/{stem}" if namespace != "." else stem
        samples.append(load_channel_sample(cir_path, import_path, sample_id=ident))
    return samples


def load_mesh_motion(sample: ChannelSample) -> MeshMotion | None:
    """Resolve the mesh stream the sample was rendered from, or None.

    Push trials reference a ``*.trial.json`` whose sibling ``*.npz`` holds
    the native 120 Hz mesh; session exports reference a ``manifest.json``
    next to the per-sample ``<sample_id>.mesh.npz``. Anything else (or a
    missing file) returns None — the caller decides whether that is fatal.
    """

    source = json.loads(sample.import_path.read_text(encoding="utf-8"))
    source_block = source.get("source") or {}
    source_path = source_block.get("source")
    if not source_path:
        return None
    path = Path(source_path)
    if path.suffix == ".json" and path.stem.endswith(".trial"):
        npz_path = path.with_suffix("").with_suffix(".npz")
        time_key = "time_physics_s"
    elif path.name == "manifest.json":
        # the manifest dir names meshes by the *source* sample id, which may
        # be shorter than the channel_sample_id (no scene prefix)
        mesh_id = (source_block.get("source_metadata") or {}).get("sample_id") or sample.sample_id
        npz_path = path.parent / f"{mesh_id}.mesh.npz"
        time_key = "time_s"
    else:
        return None
    if not npz_path.exists():
        return None
    archive = np.load(npz_path)
    if "mesh_vertices_xyz" not in archive or time_key not in archive:
        return None
    return MeshMotion(
        vertices=np.asarray(archive["mesh_vertices_xyz"], dtype=np.float64),
        time_s=np.asarray(archive[time_key], dtype=np.float64),
    )


def velocity_labels(
    sample: ChannelSample, motion: MeshMotion, component: str = "peak_doppler_hz"
) -> np.ndarray:
    """Per-CIR-frame motion truth interpolated from the mesh stream.

    ``component="peak_doppler_hz"`` is the per-frame maximum Doppler shift
    over all body points — the fastest limb, i.e. the quantity slow-time
    sampling must capture and the closest mesh-side analogue of what the
    channel actually sees. Values are computed on the motion stream's own
    time axis (central differences) and linearly interpolated onto the CIR
    frame times; CIR frames outside the motion span clamp to the ends.
    """

    if component != "peak_doppler_hz":
        raise ValueError("only 'peak_doppler_hz' is implemented")
    if sample.transmitter_xyz is None or sample.receiver_xyz is None or sample.carrier_hz is None:
        raise ValueError(f"{sample.sample_id}: import payload lacks link geometry or carrier")
    peaks, _ = frame_peak_doppler(
        motion.vertices,
        motion.time_s,
        np.asarray(sample.transmitter_xyz, dtype=np.float64),
        np.asarray(sample.receiver_xyz, dtype=np.float64),
        float(sample.carrier_hz),
    )
    # np.gradient covers every frame (one-sided at the edges), so the peak
    # series aligns 1:1 with the motion time axis
    return np.interp(sample.time_s, motion.time_s, peaks)
