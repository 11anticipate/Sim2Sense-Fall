#!/usr/bin/env python3
"""Export a keyboard session's physics recording as Sionna-ready mesh samples.

The fall-detection pipeline needs *physical* ground truth: real contact, real
gravity and the control loop participating -- not the kinematic reference
replays that ``collect_fall_mesh.py`` writes (its manifest honestly records
``fidelity: kinematic_replay`` and warns against mixing). A keyboard session
records world coordinates at render rate or, when enabled, at physics rate:

- ``recording.npz`` -- ``mesh_vertices_xyz`` (T, 6890, 3), ``mesh_faces``, ``time_s``;
- ``control.npz``   -- per-step mode, root pose, command, contact impulse;
- ``report.json``   -- ``fall.events`` (requested/impact/outcome) and hashes.

This tool splits the recording into maximal segments of constant controller
mode between reset teleports, writes each segment as
``{name}.mesh.npz`` (``time_s``, ``mesh_vertices_xyz``, ``mesh_faces`` -- the
exact arrays the Sionna importer consumes) plus a ``manifest.json`` whose
``label`` is derived from the *trajectory and session events*, never from
filenames: a segment is a fall only when the session recorded a fall request
with a body-floor impact inside that segment's time range and measured low,
tilted posture; other verified modes are labelled with their activity (walk, crouch,
stand_up, turn, stand). Falling and fallen form one event episode;
per-frame controller states distinguish motion from the subsequent lying state.
``fidelity`` is ``physics_keyboard_session``.

Built-in verification (refuses to write a manifest on failure): topology
constant, vertices finite, time uniform and strictly increasing, and every
fall event covered by exactly one fall-labelled segment.

Read-only on the session; no Isaac Sim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("export_session_mesh")

SEGMENT_LABELS = {
    "forward": "walk",
    "backward": "walk",
    "turning": "turn",
    "stand": "stand",
    "crouching": "crouch",
    "crouch": "crouch",
    "bending": "bend",
    "bend": "bend",
    "sitting": "sit",
    "sit": "sit",
    "standing_up": "stand_up",
    "getting_up": "get_up",
}
FALL_MODES = {"falling", "fallen"}
MIN_SEGMENT_FRAMES = 5
_RESET_STEP_M = 0.05


def trunk_angle_deg(quaternion: np.ndarray) -> float:
    """Tilt of the root's up-axis away from world +Z, in degrees.

    ``R[2][2] = 1 - 2*(x^2 + y^2)`` for a (w, x, y, z) quaternion. A pure yaw
    (the standing heading, e.g. 90 deg in (0.707, 0, 0, 0.707)) must read 0 deg
    here; using the ``1 - 2*(y^2 + z^2)`` element instead reads the yaw itself
    as a 90 deg lie.
    """

    w, x, y, z = quaternion
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))


def segments_from_control(
    control: dict[str, np.ndarray], recording_time: np.ndarray
) -> list[dict[str, Any]]:
    """Split the recording timeline into (start, stop, mode) segments.

    Boundaries: a controller-mode change or a root teleport (reset). Recording
    frames are mapped to control frames by time (control runs at 120 Hz, the
    skin at 30 Hz, so each recording frame sits between two control steps).
    """

    mode = np.asarray(control["mode"]).astype(str)
    time_c = np.asarray(control["time_s"], dtype=np.float64)
    roots = np.asarray(control["root"], dtype=np.float64)
    if (time_c.ndim != 1 or len(time_c) < 2 or len(time_c) != len(mode)
            or roots.shape != (len(mode), 3) or not np.isfinite(roots).all()
            or not np.isfinite(time_c).all() or np.any(np.diff(time_c) <= 0)
            or not np.isfinite(recording_time).all() or np.any(np.diff(recording_time) <= 0)):
        raise ValueError("invalid control/recording timeline")
    # Boundaries persist as an episode ID, even if 30Hz never samples the reset step.
    jumps = np.zeros(len(mode), dtype=bool)
    jumps[1:] = np.linalg.norm(np.diff(roots[:, :2], axis=0), axis=1) > _RESET_STEP_M
    if "reset_id" in control:
        reset_id = np.asarray(control["reset_id"])
        if reset_id.shape != time_c.shape or np.any(np.diff(reset_id) < 0):
            raise ValueError("invalid reset_id timeline")
        jumps[1:] = np.diff(reset_id) != 0
    episode_mode = np.where(np.isin(mode, list(FALL_MODES)), "falling", mode)
    boundaries = jumps.copy()
    boundaries[0] = True
    boundaries[1:] |= episode_mode[1:] != episode_mode[:-1]
    episode_id = np.cumsum(boundaries)
    control_index = np.clip(np.searchsorted(time_c, recording_time, side="right") - 1,
                            0, len(mode)-1)
    frame_id = episode_id[control_index]
    segments: list[dict[str, Any]] = []
    for value in np.unique(frame_id):
        frames = np.flatnonzero(frame_id == value)
        if len(frames) < MIN_SEGMENT_FRAMES:
            continue
        controls = np.flatnonzero(episode_id == value)
        end = controls[-1] + 1
        segments.append({
            "start": int(frames[0]), "stop": int(frames[-1] + 1),
            "mode": str(episode_mode[controls[0]]),
            "time_s": recording_time[frames],
            "interval_start_s": float(time_c[controls[0]]),
            "interval_end_s": float(time_c[end]) if end < len(time_c) else
            float(max(time_c[-1], recording_time[-1]) + np.median(np.diff(time_c))),
        })
    return segments


def label_segment(
    segment: dict[str, Any], vertices: np.ndarray, fall_events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Trajectory-derived label: fall only with session-recorded evidence."""

    time_s = segment["time_s"]
    if "start" in segment and "stop" in segment:
        seg_vertices = vertices[segment["start"] : segment["stop"]]
    else:
        seg_vertices = vertices
    covers_event = [
        event
        for event in fall_events
        if event.get("impact_time_s") is not None
        and segment.get("interval_start_s", time_s[0]) <= float(event["impact_time_s"])
        < segment.get("interval_end_s", time_s[-1] + np.median(np.diff(time_s)))
    ]
    mode = segment["mode"]
    if covers_event:
        impact = covers_event[0]["impact_time_s"] if covers_event else None
        label = "fall"
    else:
        impact = None
        label = SEGMENT_LABELS.get(mode, "unknown")
    metrics = {
        "min_skin_z_m": round(float(seg_vertices[:, :, 2].min()), 4),
        "max_skin_z_m": round(float(seg_vertices[:, :, 2].max()), 4),
        "frames": int(len(time_s)),
    }
    return {
        "label": label,
        "valid": label not in ("unknown", "invalid"),
        "activity_mode": mode,
        "first_impact_s": impact,
        "fall_event_count": len(covers_event),
        "metrics": metrics,
    }


def uniform_time(
    time_s: np.ndarray, vertices: np.ndarray, *, target_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (time, vertices) on a uniform grid; resample only when needed.

    The Sionna importer rejects non-uniform time axes, and render pacing can
    quantise the 30 Hz capture onto 120 Hz physics steps with occasional 1-frame
    jitter. Resampling is a linear interpolation of world-space positions --
    recorded in the manifest when it happened.
    """

    diffs = np.diff(time_s)
    if (len(diffs) == 0 or not np.isfinite(time_s).all() or np.any(diffs <= 0)
            or len(vertices) != len(time_s)):
        raise ValueError("resampling requires finite increasing samples")
    if target_hz is not None and (not np.isfinite(target_hz) or target_hz <= 0):
        raise ValueError("target_hz must be finite and positive")
    if target_hz is not None and diffs.max() > 1 / target_hz + 1e-6:
        raise ValueError(
            "capture cadence is too low for requested sample rate; record physics mesh")
    if target_hz is None and float(diffs.max() - diffs.min()) < 1e-3:
        return time_s, vertices, False
    dt = float(np.median(diffs)) if target_hz is None else 1 / target_hz
    uniform = np.arange(time_s[0], time_s[-1] + 1e-9, dt)
    resampled = np.empty((len(uniform),) + vertices.shape[1:], dtype=np.float64)
    for vertex_index in range(vertices.shape[1]):
        resampled[:, vertex_index] = np.array(
            [np.interp(uniform, time_s, vertices[:, vertex_index, axis])
             for axis in range(3)]
        ).T
    return uniform, resampled, True


def topology_sha256(faces: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest()


def export_session(
    run: Path, out: Path, *, diagnostic: bool = False, target_hz: float | None = None,
) -> dict[str, Any]:
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    admitted = (report.get("runtime_completed") is True and not report.get("errors")
                and report.get("motion_accuracy_accepted") is True
                and report.get("quality_gates", {}).get("complete_recording_window") is True
                and report.get("motion_quality", {}).get("schema_version") == 1
                and report.get("motion_quality", {}).get("accepted") is True)
    if not diagnostic and not admitted:
        raise ValueError("session did not pass runtime and measured motion quality admission")
    recording = np.load(run / "recording.npz", allow_pickle=False)
    control = dict(np.load(run / "control.npz", allow_pickle=False))
    vertices = np.asarray(recording["mesh_vertices_xyz"], dtype=np.float64)
    faces = np.asarray(recording["mesh_faces"], dtype=np.int64)
    time_s = np.asarray(recording["time_s"], dtype=np.float64)
    if (vertices.ndim != 3 or vertices.shape[-1] != 3 or len(vertices) != len(time_s)
            or len(time_s) < MIN_SEGMENT_FRAMES or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ValueError("recording contains invalid mesh topology or non-finite vertices")

    # Trunk evidence per recording frame for the fall check.
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    control_time = np.asarray(control["time_s"], dtype=np.float64)
    if (quats.shape != (len(control_time), 4) or not np.isfinite(quats).all()
            or not np.allclose(np.linalg.norm(quats, axis=1), 1., atol=1e-4)):
        raise ValueError("invalid measured root quaternions")
    control_index = np.clip(
        np.searchsorted(control_time, time_s, side="right") - 1, 0, len(quats) - 1
    )

    fall_events = list((report.get("fall") or {}).get("events") or [])
    segments = segments_from_control(control, time_s)
    if not segments:
        raise ValueError("no usable segments in the session recording")

    samples: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    pending: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for segment in segments:
        mode = segment["mode"]
        counters[mode] = counters.get(mode, -1) + 1
        name = f"{mode}_{counters[mode]:02d}"
        seg_vertices = vertices[segment["start"] : segment["stop"]]
        seg_time_raw = time_s[segment["start"] : segment["stop"]] - time_s[segment["start"]]
        seg_time, seg_vertices, resampled = uniform_time(
            seg_time_raw, seg_vertices, target_hz=target_hz)
        if len(seg_time) < MIN_SEGMENT_FRAMES:
            raise ValueError("requested rate leaves too few frames in an activity episode")
        label = label_segment(segment, vertices, fall_events)
        frame_indices = control_index[segment["start"]:segment["stop"]]
        angles = np.array([trunk_angle_deg(q) for q in quats[frame_indices]])
        label["final_trunk_angle_deg"] = round(float(angles[-1]), 2)
        if label["label"] == "fall":
            action = report.get("actions", {})
            if not {"fallen_tilt_deg", "fallen_height_fraction"} <= action.keys():
                raise ValueError("missing preregistered fall thresholds")
            requested = [e for e in fall_events if e.get("impact_time_s") is not None
                         and segment["interval_start_s"] <= e["impact_time_s"]
                         < segment["interval_end_s"]]
            request = min(e["requested_time_s"] for e in requested)
            before = max(0, int(np.searchsorted(control_time, request, side="left")) - 1)
            height = control["root"][frame_indices, 2]
            low = height < control["root"][before, 2] * action["fallen_height_fraction"]
            fallen = low & (angles >= action["fallen_tilt_deg"])
            if not fallen.any():
                label.update(label="unknown", valid=False,
                             rejection="impact without measured low and tilted posture")
        if label["label"] == "unknown" and not diagnostic:
            raise ValueError("unverified motion episode cannot enter training data")
        if diagnostic:
            label["valid"] = False
        raw_indices = np.clip(np.searchsorted(seg_time_raw, seg_time, side="right") - 1,
                              0, len(frame_indices)-1)
        frame_modes = control["mode"][frame_indices][raw_indices]
        pending.append((name, seg_time, seg_vertices, frame_modes))
        diffs = np.diff(seg_time)
        samples.append(
            {
                "sample_id": name,
                "subject": "smpl_neutral",
                "fidelity": "physics_keyboard_session",
                "coordinate_system": "world_z_up_xyz",
                "axis_order": "xyz",
                "controller_mode": mode,
                "time_span_s": [round(float(seg_time[0]), 3), round(float(seg_time[-1]), 3)],
                "frame_count": int(len(seg_time)),
                "uniform_dt_s": round(float(diffs[0]), 6) if len(diffs) else None,
                "resampled_to_uniform": resampled,
                "resample_method": "linear_world_vertices" if resampled else "none",
                "source_mesh_sampling": report.get("mesh_sampling", "legacy_render_update"),
                "label": label,
                "vertex_count": int(seg_vertices.shape[1]),
                "face_count": int(faces.shape[0]),
                "mesh_topology_sha256": topology_sha256(faces),
                "session_report_sha256": hashlib.sha256(
                    (run / "report.json").read_bytes()
                ).hexdigest(),
                "session_keyboard_sha256": report.get("keyboard_sha256"),
                "source_run": str(run.resolve()),
                "admitted_for_training": bool(admitted and not diagnostic),
                "session_time_span_s": [float(segment["time_s"][0]),
                                        float(segment["time_s"][-1])],
            }
        )
        LOGGER.info(
            "prepared %s (%s): %d frames, label=%s%s",
            name,
            mode,
            len(seg_time),
            label["label"],
            ", resampled" if resampled else "",
        )

    # Verification before the manifest is written.
    impacts = [e for e in fall_events if e.get("impact_time_s") is not None]
    contradictions = [
        s for s in samples
        if s["label"]["label"] == "fall" and s["controller_mode"] not in FALL_MODES
    ]
    if contradictions:
        raise ValueError(
            "fall event(s) landed inside non-fall segments "
            f"({[s['sample_id'] for s in contradictions]}); the session's mode "
            "timeline and fall record contradict each other"
        )
    covered = sum(
        1
        for s in samples
        if s["label"]["label"] == "fall" and s["label"]["first_impact_s"] is not None
    )
    if len(impacts) != covered:
        raise ValueError(
            f"fall-event coverage mismatch: {len(impacts)} session impacts but "
            f"{covered} fall-labelled segments carry impact evidence"
        )

    if not diagnostic and any(not sample["label"]["valid"] for sample in samples):
        raise ValueError("invalid sample in training export")
    out.mkdir(parents=True, exist_ok=True)
    for name, seg_time, seg_vertices, frame_modes in pending:
        np.savez(out / f"{name}.mesh.npz", time_s=seg_time,
                 mesh_vertices_xyz=seg_vertices, mesh_faces=faces, activity_state=frame_modes)
    manifest = {
        "description": (
            "Physical keyboard-session segments as world-space mesh sequences "
            "with trajectory-derived labels."
        ),
        "coordinate_system": "world_z_up_xyz",
        "axis_order": "xyz",
        "fidelity": "physics_keyboard_session",
        "fidelity_note": (
            "Segments of a PhysX keyboard session: gravity, contact and the control "
            "loop participate. Labels derive from the session's fall events and mode "
            "timeline, never from filenames. Not to be mixed with kinematic_replay "
            "samples without relabelling."
        ),
        "schema_version": 2,
        "admitted_for_training": bool(admitted and not diagnostic),
        "sample_count": len(samples),
        "samples": samples,
        "session": {
            "simulation_time_s": report.get("simulation_time_s"),
            "seed": report.get("seed"),
            "scene_sha256": report.get("scene_sha256"),
            "rig_sha256": report.get("rig_sha256"),
            "fall_summary": report.get("fall"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("manifest -> %s (%d samples)", out / "manifest.json", len(samples))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="keyboard session dir")
    parser.add_argument(
        "--out", type=Path, default=None, help="export dir (default <run>/mesh_export)"
    )
    parser.add_argument("--diagnostic", action="store_true",
                        help="export rejected diagnostic samples; never eligible for RT/training")
    parser.add_argument("--sample-hz", type=float,
                        help="downsample recorded physics mesh, e.g. 50; refuses upsampling")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = args.out or (args.run / "mesh_export")
    manifest = export_session(args.run, out, diagnostic=args.diagnostic, target_hz=args.sample_hz)
    for sample in manifest["samples"]:
        print(f"{sample['sample_id']}: {sample['label']['label']} ({sample['frame_count']} frames)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
