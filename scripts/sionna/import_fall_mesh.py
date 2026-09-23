#!/usr/bin/env python3
"""Import a collected fall mesh sequence into Sionna RT and measure the channel.

This is the stage-8 smoke test for the human side of the hand-off. It answers one
question and nothing else: *can the exported body geometry be put into a Sionna RT
scene, move through the scene frame by frame, and change the channel?*

It runs in the Sionna environment, not the project's CPU environment::

    /home/gsh/.local/opt/sionna/bin/python scripts/sionna/import_fall_mesh.py \
        --sample fall_forward_reference --frames 12

The scene is Sionna's built-in ``floor_wall``, not the project's apartment. Converting
the apartment to a Sionna scene is a separate piece of work, and mixing the two would
make a failure here ambiguous between "the body will not import" and "the scene
conversion is wrong". So the body is re-expressed relative to its own spawn point,
which is stated in the output rather than left implicit.

The measurement that matters is a **positive control**: the same scene is solved with
and without the body, at the same transmitter and receiver. An import that quietly
failed would leave the channel unchanged, and a run that only reported "solved N paths"
would not notice.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sim2sense_fall.schema import Activity, ChannelSample  # noqa: E402
from sim2sense_fall.sionna import human_tissue_material, place_mesh_in_scene  # noqa: E402
from sim2sense_fall.validation import validate_sample  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "artifacts" / "humans" / "fall_mesh"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "sionna" / "fall_import"


@dataclass(frozen=True, slots=True)
class SolveResult:
    """One path-solve result, reduced to what the report needs."""

    path_count: int
    power_w: float
    delay_s: np.ndarray
    amplitude: np.ndarray
    valid: np.ndarray

    @property
    def power_db(self) -> float:
        return 10.0 * float(np.log10(max(self.power_w, 1e-300)))

    @property
    def delay_spread_s(self) -> float:
        """Delay spread over the valid paths, or zero when there is only one."""

        delays = self.delay_s.ravel()[self.valid.ravel().astype(bool)]
        return float(np.ptp(delays)) if delays.size > 1 else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--sample", default="fall_forward_reference")
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
        help="how many frames of the clip to solve, evenly spaced",
    )
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--standoff-m",
        type=float,
        default=1.5,
        help="transmitter and receiver distance from the body's own origin, along x",
    )
    parser.add_argument("--height-m", type=float, default=1.4)
    parser.add_argument("--samples-per-src", type=int, default=300000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems: list[str] = []

    def record(label: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    manifest_path = args.dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"no collection manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next((entry for entry in manifest["samples"] if entry["sample_id"] == args.sample), None)
    if row is None:
        known = [entry["sample_id"] for entry in manifest["samples"]]
        print(f"unknown sample {args.sample!r}; known: {known}", file=sys.stderr)
        return 2
    record(
        "the sample is a licensed skin mesh",
        row["mesh"]["representation"] == "smpl_skin_mesh",
        f"{row['mesh']['representation']}, {row['mesh']['vertex_count']} vertices",
    )

    with np.load(args.dir / f"{args.sample}.mesh.npz") as data:
        all_vertices = np.asarray(data["mesh_vertices_xyz"], dtype=np.float64)
        faces = np.asarray(data["mesh_faces"])
        times = np.asarray(data["time_s"], dtype=np.float64)
    frame_count = all_vertices.shape[0]
    record(
        "the mesh sequence matches its manifest",
        all_vertices.shape == (frame_count, row["mesh"]["vertex_count"], 3)
        and faces.shape == (row["mesh"]["face_count"], 3)
        and times.shape == (frame_count,),
        f"{all_vertices.shape}, {faces.shape}, {len(times)} frames",
    )

    # Re-express the body around its own spawn: Sionna's built-in scene is centred on the
    # origin, while the collected mesh is world-referenced inside the apartment. Without
    # this the body sits metres outside the test scene and every path solve misses it.
    spawn_xy = all_vertices[0].mean(axis=0)[:2]
    vertices = all_vertices - np.array([spawn_xy[0], spawn_xy[1], 0.0])
    record(
        "the body was re-centred on its own spawn",
        bool(np.all(np.isfinite(vertices))) and float(np.abs(vertices[:, :, :2]).max()) < 2.0,
        f"collected spawn ({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f}) becomes the origin; "
        f"largest |xy| is {float(np.abs(vertices[:, :, :2]).max()):.3f} m",
    )

    frames = _evenly_spaced(frame_count, max(2, args.frames))

    import sionna.rt as rt

    material = human_tissue_material()
    scene = _build_scene(rt, args)

    baseline = solve(scene, rt, args)
    record(
        "the scene solves without the body",
        baseline.path_count > 0,
        f"{baseline.path_count} paths, {baseline.power_db:+.2f} dB",
    )

    cirs: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    imported: dict[str, Any] | None = None
    for frame in frames:
        name = f"human_f{frame:04d}"
        report = place_mesh_in_scene(scene, vertices[frame], faces, name=name, material=material)
        if imported is None:
            imported = report.as_dict()
        solved = solve(scene, rt, args)
        cir = channel_impulse_response(solved, args)
        cirs.append(cir)
        per_frame.append(
            {
                "frame": int(frame),
                "time_s": float(times[frame]),
                "path_count": solved.path_count,
                "power_db": solved.power_db,
                "delta_vs_no_body_db": solved.power_db - baseline.power_db,
                "cir_bins": int(cir.size),
            }
        )
        # Moving the body means replacing it, otherwise the previous frame's copy stays in
        # the scene and every frame after the first measures two overlapping bodies.
        scene.edit(remove=name)

    record(
        "the body was admitted to the scene",
        imported is not None and imported["vertex_count"] == row["mesh"]["vertex_count"],
        f"{imported['name'] if imported else 'none'} with "
        f"{imported['vertex_count'] if imported else 0} vertices",
    )
    deltas = [entry["delta_vs_no_body_db"] for entry in per_frame]
    largest = max(deltas, key=abs)
    record(
        "the body changes the channel (positive control)",
        any(abs(delta) > 0.05 for delta in deltas),
        f"largest change {largest:+.2f} dB across {len(frames)} frames; "
        f"range {min(deltas):+.2f}..{max(deltas):+.2f} dB",
    )
    record(
        "the channel changes as the body falls",
        float(np.ptp(deltas)) > 0.05,
        f"spread across the fall {np.ptp(deltas):.2f} dB (a static body would give zero)",
    )

    channel = np.stack(cirs)
    record(
        "every frame produced a finite CIR of one shape",
        bool(np.all(np.isfinite(channel))) and channel.shape[0] == len(frames),
        f"{channel.shape}, max |h| {float(np.abs(channel).max()):.3e}",
    )

    sample = ChannelSample(
        sample_id=f"sionna_floor_wall__{args.sample}",
        scene_id="sionna_builtin_floor_wall",
        subject_id=row["subject"],
        hardware_profile=f"planar_1x1_{args.frequency_hz / 1e9:g}GHz",
        activity=Activity.FALL if "fall" in args.sample else Activity.UNKNOWN,
        timestamp_s=np.asarray(times[frames], dtype=np.float64),
        channel=channel,
        channel_representation="cir",
        sample_rate_hz=1.0 / float(np.mean(np.diff(times[frames]))),
        simulator_version=_sionna_version(),
        metadata={
            "source_collection": str(manifest_path.relative_to(REPO_ROOT)),
            "source_sample": args.sample,
            "frames_used": [int(f) for f in frames],
            "fidelity_note": (
                "the body geometry is a kinematic_replay export; the propagation is real "
                "Sionna RT, so the channel is a function of a non-physical trajectory"
            ),
            "scene_note": (
                "Sionna's built-in floor_wall scene, not the project apartment; the body "
                "was re-centred on its own spawn to sit in it"
            ),
            "material": material.as_dict(),
            "baseline_no_body_power_db": baseline.power_db,
            "per_frame": per_frame,
        },
    )
    try:
        validate_sample(sample)
        record("the ChannelSample satisfies the data contract", True, sample.sample_id)
    except ValueError as exc:
        record("the ChannelSample satisfies the data contract", False, str(exc))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"{args.sample}.cir.npz",
        timestamp_s=sample.timestamp_s,
        cir=channel,
        frame_index=np.asarray(frames, dtype=np.int64),
    )
    report = {
        "script": "scripts/sionna/import_fall_mesh.py",
        "sionna_version": _sionna_version(),
        "headless_scene": "sionna.rt.scene.floor_wall",
        "frequency_hz": args.frequency_hz,
        "bandwidth_hz": args.bandwidth_hz,
        "max_depth": args.max_depth,
        "transmitter": [-(args.standoff_m), 0.0, args.height_m],
        "receiver": [args.standoff_m, 0.0, args.height_m],
        "body_recentring_offset_xy": [float(spawn_xy[0]), float(spawn_xy[1])],
        "imported_object": imported,
        "baseline": {
            "path_count": baseline.path_count,
            "power_db": baseline.power_db,
        },
        "per_frame": per_frame,
        "channel_sample_id": sample.sample_id,
        "failures": problems,
    }
    (args.out / f"{args.sample}.import.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print()
    print(f"wrote {args.out}")
    if problems:
        print(f"FAILED {len(problems)} check(s): {problems}")
        return 1
    print(f"the imported body changes the channel; {len(frames)} frames solved")
    return 0


def _build_scene(rt: Any, args: argparse.Namespace) -> Any:
    """Sionna's built-in two-plane scene with one transmitter and one receiver.

    Chosen over a project scene for this test on purpose: a failure has to be readable as
    "the body will not import", not "the apartment conversion is wrong".
    """

    scene = rt.load_scene(rt.scene.floor_wall)
    scene.add(rt.Transmitter("tx", position=[-args.standoff_m, 0.0, args.height_m], power_dbm=30.0))
    scene.add(rt.Receiver("rx", position=[args.standoff_m, 0.0, args.height_m]))
    scene.frequency = args.frequency_hz
    scene.bandwidth = args.bandwidth_hz
    single = dict(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = rt.PlanarArray(**single)
    scene.rx_array = rt.PlanarArray(**single)
    return scene


def solve(scene: Any, rt: Any, args: argparse.Namespace) -> SolveResult:
    paths = rt.PathSolver()(scene, max_depth=args.max_depth, samples_per_src=args.samples_per_src)
    amplitudes = _to_numpy(paths.a)
    valid = _to_numpy(paths.valid)
    tau = _to_numpy(paths.tau)
    power = float((np.abs(amplitudes) ** 2 * valid).sum())
    return SolveResult(
        path_count=int(valid.sum()),
        power_w=power,
        delay_s=tau,
        amplitude=amplitudes,
        valid=valid,
    )


def channel_impulse_response(solved: SolveResult, args: argparse.Namespace) -> np.ndarray:
    """A CIR magnitude per tap over the frame's valid paths.

    Binned by the solver's own delay spread onto a fixed tap grid rather than read from
    ``Paths.cir``, because the ``ChannelSample`` needs one array of identical length per
    frame and the path count varies as the body moves. Each tap holds the coherent sum of
    the path amplitudes that land in it, which is what makes the tap values meaningful
    rather than a count of arrivals.
    """

    delays = solved.delay_s.ravel()
    amplitudes = solved.amplitude.ravel()
    valid = solved.valid.ravel().astype(bool)
    if delays.size != amplitudes.size or delays.size != valid.size:
        raise ValueError(
            f"the solver returned mismatched path arrays: delays {delays.shape}, "
            f"amplitudes {amplitudes.shape}, valid {valid.shape}"
        )
    # Taps resolve the channel bandwidth: a delay spread resolving to `bins` taps is the
    # delay resolution 1/bandwidth, floored so a single-arrival frame still yields a vector.
    spread = solved.delay_spread_s
    bins = max(8, int(np.ceil(spread * args.bandwidth_hz)) + 1) if spread > 0 else 8
    taps = np.zeros(bins, dtype=np.complex128)
    if spread > 0:
        index = np.clip(
            ((delays - delays[valid].min()) / spread * (bins - 1)).round().astype(int),
            0,
            bins - 1,
        )
        np.add.at(taps, index[valid], amplitudes[valid])
    elif valid.any():
        taps[0] = amplitudes[valid].sum()
    return np.abs(taps)


def _to_numpy(value: Any) -> np.ndarray:
    value = value[0] if isinstance(value, tuple) else value
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sionna_version() -> str:
    import importlib.metadata as metadata

    try:
        return f"sionna-rt {metadata.version('sionna-rt')}"
    except Exception:  # noqa: BLE001 - version reporting must never break a run
        return "sionna-rt unknown"


def _evenly_spaced(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    step = total / count
    return [min(int(round(index * step)), total - 1) for index in range(count)]


if __name__ == "__main__":
    raise SystemExit(main())
