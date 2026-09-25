#!/usr/bin/env python3
"""Export a keyboard session's physics recording as Sionna-ready mesh samples.

The fall-detection pipeline needs *physical* ground truth: real contact, real
gravity and the control loop participating -- not the kinematic reference
replays that ``collect_fall_mesh.py`` writes (its manifest honestly records
``fidelity: kinematic_replay`` and warns against mixing). A keyboard session
already records everything needed, at 30 Hz, in world coordinates:

- ``recording.npz`` -- ``mesh_vertices_xyz`` (T, 6890, 3), ``mesh_faces``, ``time_s``;
- ``control.npz``   -- per-step mode, root pose, command, contact impulse;
- ``report.json``   -- ``fall.events`` (requested/impact/outcome) and hashes.

This tool splits the recording into maximal segments of constant controller
mode between reset teleports, writes each segment as
``{name}.mesh.npz`` (``time_s``, ``mesh_vertices_xyz``, ``mesh_faces`` -- the
exact arrays the Sionna importer consumes) plus a ``manifest.json`` whose
``label`` is derived from the *trajectory and session events*, never from
filenames: a segment is a fall only when the session recorded a fall request
with a body-floor impact inside that segment's time range, or when the trunk
ends lying; everything else is labelled with its activity (walk, crouch,
stand_up, turn, stand). ``fidelity`` is ``physics_keyboard_session``.

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
    "standing_up": "stand_up",
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
    jumps = np.zeros(len(mode), dtype=bool)
    jumps[1:] = np.linalg.norm(np.diff(roots[:, :2], axis=0), axis=1) > _RESET_STEP_M
    segment_mode = mode.copy()
    segment_mode[jumps] = "__reset__"

    control_index = np.searchsorted(time_c, recording_time, side="right") - 1
    control_index = np.clip(control_index, 0, len(mode) - 1)
    frame_mode = segment_mode[control_index]

    segments: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(recording_time) + 1):
        if index == len(recording_time) or frame_mode[index] != frame_mode[start]:
            if index - start >= MIN_SEGMENT_FRAMES and frame_mode[start] != "__reset__":
                segments.append(
                    {
                        "start": start,
                        "stop": index,
                        "mode": str(frame_mode[start]),
                        "time_s": recording_time[start:index],
                    }
                )
            start = index
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
        and time_s[0] - 0.02 <= float(event["impact_time_s"]) <= time_s[-1] + 0.02
    ]
    mode = segment["mode"]
    if mode in FALL_MODES or covers_event:
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
    time_s: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (time, vertices) on a uniform grid; resample only when needed.

    The Sionna importer rejects non-uniform time axes, and render pacing can
    quantise the 30 Hz capture onto 120 Hz physics steps with occasional 1-frame
    jitter. Resampling is a linear interpolation of world-space positions --
    recorded in the manifest when it happened.
    """

    diffs = np.diff(time_s)
    if len(diffs) and float(diffs.max() - diffs.min()) < 1e-3:
        return time_s, vertices, False
    dt = float(np.median(diffs))
    uniform = np.arange(time_s[0], time_s[-1] + 0.5 * dt, dt)
    resampled = np.empty((len(uniform),) + vertices.shape[1:], dtype=np.float64)
    for vertex_index in range(vertices.shape[1]):
        resampled[:, vertex_index] = np.array(
            [np.interp(uniform, time_s, vertices[:, vertex_index, axis])
             for axis in range(3)]
        ).T
    return uniform, resampled, True


def topology_sha256(faces: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest()


def export_session(run: Path, out: Path) -> dict[str, Any]:
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    recording = np.load(run / "recording.npz", allow_pickle=False)
    control = dict(np.load(run / "control.npz", allow_pickle=False))
    vertices = np.asarray(recording["mesh_vertices_xyz"], dtype=np.float64)
    faces = np.asarray(recording["mesh_faces"], dtype=np.int64)
    time_s = np.asarray(recording["time_s"], dtype=np.float64)
    if not np.isfinite(vertices).all():
        raise ValueError("recording contains non-finite vertices")

    # Trunk evidence per recording frame for the fall check.
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    control_time = np.asarray(control["time_s"], dtype=np.float64)
    control_index = np.clip(
        np.searchsorted(control_time, time_s, side="right") - 1, 0, len(quats) - 1
    )

    fall_events = list((report.get("fall") or {}).get("events") or [])
    segments = segments_from_control(control, time_s)
    if not segments:
        raise ValueError("no usable segments in the session recording")

    samples: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for segment in segments:
        mode = segment["mode"]
        counters[mode] = counters.get(mode, -1) + 1
        name = f"{mode}_{counters[mode]:02d}"
        seg_vertices = vertices[segment["start"] : segment["stop"]]
        seg_time_raw = time_s[segment["start"] : segment["stop"]] - time_s[segment["start"]]
        seg_time, seg_vertices, resampled = uniform_time(seg_time_raw, seg_vertices)
        label = label_segment(segment, vertices, fall_events)
        if label["label"] == "fall" and label["first_impact_s"] is None:
            # A fall segment without a session-recorded impact is still a fall
            # request being executed; keep the label but flag the evidence gap.
            label["impact_evidence"] = "session recorded no body-floor impact for this segment"
        label["final_trunk_angle_deg"] = round(
            trunk_angle_deg(quats[control_index[segment["stop"] - 1]]), 2
        )
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / f"{name}.mesh.npz",
            time_s=seg_time,
            mesh_vertices_xyz=seg_vertices,
            mesh_faces=faces,
        )
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
                "label": label,
                "vertex_count": int(seg_vertices.shape[1]),
                "face_count": int(faces.shape[0]),
                "mesh_topology_sha256": topology_sha256(faces),
                "session_report_sha256": hashlib.sha256(
                    (run / "report.json").read_bytes()
                ).hexdigest(),
                "session_keyboard_sha256": report.get("keyboard_sha256"),
                "source_run": str(run.resolve()),
            }
        )
        LOGGER.info(
            "wrote %s (%s): %d frames, label=%s%s",
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = args.out or (args.run / "mesh_export")
    manifest = export_session(args.run, out)
    for sample in manifest["samples"]:
        print(f"{sample['sample_id']}: {sample['label']['label']} ({sample['frame_count']} frames)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
