#!/usr/bin/env python3
"""Validate apartment/human geometry and produce reproducible complex CIR samples.

Run --dry-run with system Python before invoking this with the Sionna interpreter.
The default consumes the project's fixed apartment; --scene floor_wall retains the
small diagnostic scene. Geometry labels come from the trajectory, never filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.scenes.spec import load_scene_spec  # noqa: E402
from sim2sense_fall.schema import Activity, ChannelSample  # noqa: E402
from sim2sense_fall.sionna import (  # noqa: E402
    candidate_mesh_arrays,
    human_tissue_material,
    place_mesh_in_scene,
)
from sim2sense_fall.sionna.apartment import build_apartment_scene  # noqa: E402
from sim2sense_fall.sionna.channel import (  # noqa: E402
    as_numpy,
    complex_amplitudes,
    delay_grid,
    paths_to_cir,
    regular_frame_indices,
)
from sim2sense_fall.validation import validate_sample  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=REPO_ROOT / "artifacts/humans/fall_mesh")
    p.add_argument("--sample", default="fall_forward_reference")
    p.add_argument("--trial-json", type=Path, help="physics_trial JSON with its matching NPZ")
    p.add_argument("--scene", choices=("apartment", "floor_wall"), default="apartment")
    p.add_argument(
        "--scene-config", type=Path, default=REPO_ROOT / "configs/scenes/indoor_apartment.yaml"
    )
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--frequency-hz", type=float, default=3.5e9)
    p.add_argument("--bandwidth-hz", type=float, default=100e6)
    p.add_argument("--max-delay-s", type=float, default=2e-6)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--samples-per-src", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standoff-m", type=float, default=0.8)
    p.add_argument("--height-m", type=float, default=1.4)
    p.add_argument("--tx", type=float, nargs=3)
    p.add_argument("--rx", type=float, nargs=3)
    p.add_argument("--cylinder-segments", type=int, default=32)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/sionna/apartment_import")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    for key in ("frequency_hz", "bandwidth_hz", "max_delay_s", "standoff_m", "height_m"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be finite and positive")
    if args.frames < 2 or args.samples_per_src < 1 or args.max_depth < 0 or args.seed < 0:
        p.error("frames >= 2, samples > 0, depth and seed >= 0 required")
    if args.frequency_hz != 3.5e9:
        p.error("the cited tissue constants are valid at 3.5 GHz; supply a validated model first")
    if (args.tx is None) != (args.rx is None):
        p.error("--tx and --rx must be supplied together")
    if args.tx is not None and not np.isfinite([args.tx, args.rx]).all():
        p.error("tx/rx coordinates must be finite")
    return args


def load_geometry(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if args.trial_json:
        path = args.trial_json
        manifest = json.loads(path.read_text())
        index_path = path.parent / "trials_index.json"
        if not index_path.is_file():
            raise ValueError("physics source requires the completed trials_index.json acceptance")
        index = json.loads(index_path.read_text())
        accepted = next(
            (r for r in index["trials"] if Path(r["json"]).name == path.name), None
        )
        if accepted is None or not accepted.get("gates", {}).get("usable", False):
            raise ValueError("physics source did not pass its trial acceptance gates")
        manifest["acceptance_gates"] = accepted["gates"]
        npz = path.with_name(path.name.removesuffix(".trial.json") + ".npz")
        with np.load(npz) as data:
            vertices = data["mesh_vertices_xyz"]
            faces = data["mesh_faces"]
            times = data["time_physics_s"]
        info = {
            "source": str(path.resolve()),
            "source_manifest_sha256": sha256(path),
            "source_mesh_sha256": sha256(npz),
            "source_metadata": manifest,
            "label": manifest["label"],
            "fidelity": "physics_trial",
            "subject": manifest["provenance"].get("subject") or "smpl_neutral",
            "sample_id": npz.stem,
        }
    else:
        path = args.dir / "manifest.json"
        manifest = json.loads(path.read_text())
        row = next((r for r in manifest["samples"] if r["sample_id"] == args.sample), None)
        if row is None:
            raise ValueError(f"unknown sample {args.sample}")
        npz = args.dir / f"{args.sample}.mesh.npz"
        with np.load(npz) as data:
            vertices, faces, times = data["mesh_vertices_xyz"], data["mesh_faces"], data["time_s"]
        info = {
            "source": str(path.resolve()),
            "source_manifest_sha256": sha256(path),
            "source_mesh_sha256": sha256(npz),
            "source_metadata": row,
            "label": row["label"],
            "fidelity": row["fidelity"],
            "subject": row["subject"],
            "sample_id": args.sample,
        }
    if not info["label"]["valid"] or info["label"]["label"] == "invalid":
        raise ValueError("invalid source trajectory is excluded from channel generation")
    if vertices.ndim != 3 or vertices.shape[2] != 3 or len(vertices) != len(times):
        raise ValueError("mesh sequence and timestamps have inconsistent shapes")
    if not np.isfinite(vertices).all() or not np.isfinite(times).all() or len(times) < 2:
        raise ValueError("geometry and time must have at least two finite frames")
    if not np.all(np.diff(times) > 0) or not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("source time axis must be uniform and increasing")
    for frame in vertices:
        candidate_mesh_arrays(frame, faces)
    return vertices, faces, times, info


def solve(
    scene: Any, rt: Any, args: argparse.Namespace, grid: np.ndarray
) -> tuple[np.ndarray, dict]:
    paths = rt.PathSolver()(
        scene,
        max_depth=args.max_depth,
        samples_per_src=args.samples_per_src,
        seed=args.seed,
        diffuse_reflection=False,
    )
    a = complex_amplitudes(paths.a)
    valid = as_numpy(paths.valid).astype(bool)
    tau = as_numpy(paths.tau)
    cir = paths_to_cir(a, tau, valid, grid)
    gain = float(np.sum(np.abs(a.ravel()[valid.ravel()]) ** 2))
    return cir, {
        "path_count": int(valid.sum()),
        "incoherent_path_gain": gain,
        "path_gain_db": 10 * float(np.log10(max(gain, 1e-300))),
        "valid_delay_s": tau.ravel()[valid.ravel()].tolist(),
        "amplitude_real": a.ravel()[valid.ravel()].real.tolist(),
        "amplitude_imag": a.ravel()[valid.ravel()].imag.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vertices, faces, times, info = load_geometry(args)
    frames = regular_frame_indices(len(times), args.frames)
    grid = delay_grid(args.bandwidth_hz, args.max_delay_s)
    origin = vertices[0].mean(axis=0)[:2]
    offset = np.zeros(3) if args.scene == "apartment" else np.r_[origin, 0.0]
    vertices = vertices - offset
    origin = origin - offset[:2]
    tx = args.tx or [float(origin[0] - args.standoff_m), float(origin[1]), args.height_m]
    rx = args.rx or [float(origin[0] + args.standoff_m), float(origin[1]), args.height_m]
    scene_info: dict = {"scene_id": "sionna_builtin_floor_wall"}
    scene = None
    if args.scene == "apartment":
        spec = load_scene_spec(args.scene_config)
        source_scene = info["source_metadata"].get("provenance", {}).get("scene_id")
        if source_scene is not None and source_scene != spec.scene_id:
            raise ValueError("physics source and radio apartment scene IDs differ")
        scene, scene_info = build_apartment_scene(
            spec,
            frequency_hz=args.frequency_hz,
            cylinder_segments=args.cylinder_segments,
            dry_run=args.dry_run,
        )
        scene_info["config_sha256"] = sha256(args.scene_config)
    material = human_tissue_material()
    report = {
        "schema_version": 2,
        "source": info,
        "scene": scene_info,
        "material": material.as_dict(),
        "seed": args.seed,
        "split": "smoke_only_not_train_test",
        "hardware_profile": "iso_1x1_3.5GHz_V",
        "subject_id": info["subject"],
        "repeat_tolerance": {"absolute": 1e-9, "relative": 1e-5},
        "transmitter": tx,
        "receiver": rx,
        "body_recentring_offset_xyz": offset.tolist(),
        "frequency_hz": args.frequency_hz,
        "bandwidth_hz": args.bandwidth_hz,
        "max_depth": args.max_depth,
        "samples_per_src": args.samples_per_src,
        "frames": frames.tolist(),
        "delay_grid_s": grid.tolist(),
        "metric": "sum of squared complex path gains, dimensionless, incoherent",
        "channel_model": "complex sinc taps at absolute delays; no delay normalization",
        "dry_run": args.dry_run,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_name = info["sample_id"]
    if args.dry_run:
        (args.out / f"{out_name}.dry-run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"CPU dry-run PASSED: {len(frames)} frames; scene {scene_info['scene_id']}")
        return 0
    import mitsuba as mi
    import sionna.rt as rt

    if scene is None:
        scene = rt.load_scene(rt.scene.floor_wall)
    scene.frequency = args.frequency_hz
    scene.bandwidth = args.bandwidth_hz
    scene.add(rt.Transmitter("tx", position=tx, power_dbm=30.0))
    scene.add(rt.Receiver("rx", position=rx))
    array = dict(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = rt.PlanarArray(**array)
    scene.rx_array = rt.PlanarArray(**array)
    baseline, baseline_info = solve(scene, rt, args, grid)
    baseline_repeat, _ = solve(scene, rt, args, grid)
    channels, records = [], []
    static_error = 0.0
    object_count = len(scene.objects)
    for k, frame in enumerate(frames):
        place_mesh_in_scene(scene, vertices[frame], faces, name="human", material=material)
        if len(scene.objects) != object_count + 1:
            raise RuntimeError("scene does not contain exactly one additional human")
        cir, record = solve(scene, rt, args, grid)
        if k == 0:
            repeat, _ = solve(scene, rt, args, grid)
            static_error = float(np.max(np.abs(repeat - cir)))
        record.update(
            frame=int(frame),
            time_s=float(times[frame]),
            delta_vs_no_body_db=record["path_gain_db"] - baseline_info["path_gain_db"],
        )
        channels.append(cir)
        records.append(record)
        scene.edit(remove="human")
        if len(scene.objects) != object_count:
            raise RuntimeError("human removal changed environment object count")
        print(
            f"frame {frame}: paths={record['path_count']}, "
            f"delta={record['delta_vs_no_body_db']:+.3f} dB"
        )
    channel = np.stack(channels)
    label = info["label"]["label"]
    activity = Activity.FALL if label == "fall" else Activity.UNKNOWN
    # The reference's intended action is not evidence that a lowering was intentional ADL.
    sample = ChannelSample(
        sample_id=f"{scene_info['scene_id']}__{out_name}",
        scene_id=scene_info["scene_id"],
        subject_id=info["subject"],
        hardware_profile="iso_1x1_3.5GHz_V",
        activity=activity,
        timestamp_s=times[frames],
        channel=channel,
        channel_representation="cir",
        sample_rate_hz=1 / float(np.diff(times[frames])[0]),
        simulator_version=f"sionna-rt {importlib.metadata.version('sionna-rt')}",
        metadata=report,
    )
    validate_sample(sample)
    repeat_error = float(np.max(np.abs(baseline_repeat - baseline)))
    body_change = float(np.max(np.abs(channel - baseline)))
    motion_change = float(np.max(np.abs(channel - channel[0])))
    checks = {
        "baseline_has_paths": baseline_info["path_count"] > 0,
        "static_body_repeat": static_error < max(1e-9, 1e-5 * float(np.max(np.abs(channel[0])))),
        "baseline_repeat": repeat_error < max(1e-9, 1e-5 * float(np.max(np.abs(baseline)))),
        "body_changes_complex_channel": body_change > 1e-9,
        "motion_changes_complex_channel": motion_change > 1e-9,
        "finite_complex_cir": bool(np.isfinite(channel).all() and np.iscomplexobj(channel)),
    }
    report.update(
        checks=checks,
        baseline=baseline_info,
        per_frame=records,
        static_repeat_max_abs=static_error,
        baseline_repeat_max_abs=repeat_error,
        body_change_max_abs=body_change,
        motion_change_max_abs=motion_change,
        activity=activity.value,
        source_event_label=label,
        sample_rate_hz=sample.sample_rate_hz,
        channel_sample_id=sample.sample_id,
        sionna_version=sample.simulator_version,
        mitsuba_variant=mi.variant(),
        failures=[k for k, v in checks.items() if not v],
    )
    np.savez_compressed(
        args.out / f"{out_name}.cir.npz",
        timestamp_s=sample.timestamp_s,
        cir=channel,
        frame_index=frames,
        delay_s=grid,
        baseline_cir=baseline,
    )
    (args.out / f"{out_name}.import.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'PASSED' if all(checks.values()) else 'FAILED'}: {checks}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
