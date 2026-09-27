#!/usr/bin/env python3
"""Independently verify a collected fall-mesh directory against the Sionna contract.

This is a **second implementation** of the acceptance checks, written on purpose to
disagree with the collector if either is wrong. ``collect_fall_mesh.py --verify-only``
re-checks a collection using the same helpers that produced it; that is a useful
regression net but it cannot catch a shared misunderstanding. This script recomputes
everything from the ``.npz`` files with its own arithmetic and imports nothing from the
collector, so agreement between the two is real evidence rather than a tautology.

The anatomy section is the part that matters most. Every other check here is invariant
under a rotation about the vertical, and the defect this project actually shipped was a
90-degree yaw of the skin relative to the skeleton -- correct links, correct trajectory,
correct mesh shape, wrong facing. A rotation-invariant suite cannot see that. So the
checks that name anatomical directions are the load-bearing ones:

* the head is above the feet in the standing first frame;
* the body is wider left-right than front-to-back, and the mesh and the links agree
  about which axis that is.

Usage::

    python3 scripts/humans/verify_fall_collection.py [--dir artifacts/humans/fall_mesh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "artifacts" / "humans" / "fall_mesh"

#: The pipeline body frame the export claims, from ``RestSkeleton``.
EXPECTED_LATERAL_AXIS = "y"
EXPECTED_UP_AXIS = "z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)

    base: Path = args.dir
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    print("=== A. the contract the manifest declares ===")
    check("coordinate system is z-up xyz", manifest["coordinate_system"] == "world_z_up_xyz")
    check("units are metres", manifest["units"] == "m")
    check("axis order is xyz", manifest["axis_order"] == "xyz")
    check(
        "fidelity is a kinematic replay, not a physics trial",
        manifest["fidelity"] == "kinematic_replay",
        manifest["fidelity"],
    )
    check("a random seed is recorded", isinstance(manifest["seed"], int), f"{manifest['seed']}")
    check(
        "the scene is identified by hash",
        bool(manifest.get("scene", {}).get("sha256")),
        str(manifest.get("scene", {}).get("scene_id")),
    )
    check(
        "the config is identified by hash",
        bool(manifest.get("config", {}).get("sha256")),
        str(manifest.get("config", {}).get("path")),
    )
    body_model = manifest.get("body_model") or {}
    check(
        "the licensed body is identified by file hash",
        bool(body_model.get("sha256")),
        f"{body_model.get('declared_asset_id')} sha256 {str(body_model.get('sha256'))[:16]}...",
    )
    check(
        "the measured source frame is recorded",
        bool(body_model.get("source_frame")),
        str(body_model.get("source_frame")),
    )
    check(
        "the skin representation matches the body model",
        body_model.get("representation") == "smpl_skin_mesh"
        and manifest["samples"][0]["metric_definitions"]["body_representation"] == "smpl_skin_mesh",
        f"{body_model.get('representation')}, {body_model.get('vertex_count')} vertices",
    )
    check(
        "metric definitions are published per sample",
        all(isinstance(row.get("metric_definitions"), dict) for row in manifest["samples"]),
        f"{len(manifest['samples'])} samples",
    )
    check(
        "every sample names its motion source and rig hash",
        all(
            len(row["hashes"].get("motion_sha256") or "") == 64
            and len(row["hashes"].get("rig_plan_sha256") or "") == 64
            for row in manifest["samples"]
        ),
    )

    print("\n=== B. arrays, re-read off disk ===")
    topology_hashes: set[str] = set()
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        mesh_npz = _resolve(base, row["mesh"]["npz"])
        points_npz = _resolve(base, row["points"]["npz"])
        if not mesh_npz.is_file() or not points_npz.is_file():
            check(f"{sample_id}: both files exist", False, f"{mesh_npz}, {points_npz}")
            continue
        with np.load(mesh_npz) as data:
            times = data["time_s"]
            vertices = data["mesh_vertices_xyz"]
            faces = data["mesh_faces"]
        with np.load(points_npz) as data:
            point_times = data["time_s"]
            root = data["root_position_xyz"]
            links = data["link_positions_xyz"]
            surface = data["surface_points_xyz"]

        frames = len(times)
        check(
            f"{sample_id}: mesh is (N, V, 3) with N = len(time)",
            vertices.ndim == 3 and vertices.shape[0] == frames and vertices.shape[2] == 3,
            f"{vertices.shape}",
        )
        check(
            f"{sample_id}: the two files share one time axis",
            np.array_equal(times, point_times),
            f"{frames} frames",
        )
        check(
            f"{sample_id}: time starts at zero and strictly increases",
            float(times[0]) == 0.0 and bool(np.all(np.diff(times) > 0)),
            f"{float(times[0]):.4f}..{float(times[-1]):.4f} s",
        )
        check(
            f"{sample_id}: every coordinate is finite",
            bool(np.isfinite(vertices).all())
            and bool(np.isfinite(root).all())
            and bool(np.isfinite(links).all())
            and bool(np.isfinite(surface).all()),
        )
        check(
            f"{sample_id}: face indices stay inside the vertex array",
            int(faces.max()) < vertices.shape[1],
            f"max {int(faces.max())} < {vertices.shape[1]}",
        )
        check(
            f"{sample_id}: the (x, y, z) sequences have one row per frame",
            root.shape == (frames, 3) and links.shape == (frames, 24, 3),
            f"root {root.shape}, links {links.shape}",
        )
        check(
            f"{sample_id}: the mesh is not a constant sequence",
            float(np.linalg.norm(vertices[-1] - vertices[0], axis=1).max()) > 1e-6,
            f"{float(np.linalg.norm(vertices[-1] - vertices[0], axis=1).max()):.6f} m travelled",
        )

        topology_hashes.add(
            hashlib.sha256(np.ascontiguousarray(faces, dtype=np.int64).tobytes()).hexdigest()
        )

        # --- the anatomy ----------------------------------------------------
        names = list(row["points"].get("link_names", ()))
        if not names:
            check(f"{sample_id}: the manifest publishes the link ordering", False)
            continue
        lookup = {name: index for index, name in enumerate(names)}
        needed = ["head", "left_ankle", "right_ankle", "pelvis"]
        if any(name not in lookup for name in needed):
            check(f"{sample_id}: the rig exposes the landmark links", False, f"{names}")
            continue

        standing = float(links[0, lookup["head"], 2])
        ankles = min(
            float(links[0, lookup["left_ankle"], 2]), float(links[0, lookup["right_ankle"], 2])
        )
        check(
            f"{sample_id}: the clip starts standing, head above feet",
            standing > ankles,
            f"head link z {standing:+.4f} vs ankle link z {ankles:+.4f}",
        )
        descent = standing - float(links[-1, lookup["head"], 2])
        check(
            f"{sample_id}: the head actually descends over the clip",
            descent > 0.1,
            f"{descent:.3f} m",
        )

        # Facing: a standing human spans more left-right than front-back. The mesh and
        # the links must agree about which horizontal axis that is, because the defect
        # was exactly the two disagreeing.
        mesh_extent = vertices[0].max(axis=0) - vertices[0].min(axis=0)
        link_extent = links[0].max(axis=0) - links[0].min(axis=0)
        mesh_wide = "y" if mesh_extent[1] > mesh_extent[0] else "x"
        link_wide = "y" if link_extent[1] > link_extent[0] else "x"
        check(
            f"{sample_id}: the mesh and the links agree on the body's facing",
            mesh_wide == link_wide == EXPECTED_LATERAL_AXIS,
            f"mesh x {mesh_extent[0]:.4f} y {mesh_extent[1]:.4f} -> {mesh_wide!r}; "
            f"links x {link_extent[0]:.4f} y {link_extent[1]:.4f} -> {link_wide!r}; "
            f"expected lateral {EXPECTED_LATERAL_AXIS!r}",
        )
        # Pitch, from the mesh on its own: an upright body is clearly taller than it is
        # deep, whereas a figure lying prone or supine has those two extents swapped.
        # Measured on the shipped export the margin is 1.796 m against 0.304 m.
        check(
            f"{sample_id}: the first-frame mesh is upright, not lying down",
            float(mesh_extent[2]) > 2.0 * float(mesh_extent[0]),
            f"z extent {mesh_extent[2]:.4f} m vs x extent {mesh_extent[0]:.4f} m "
            f"(ratio {mesh_extent[2] / max(mesh_extent[0], 1e-9):.1f}); "
            f"expected up axis {EXPECTED_UP_AXIS!r}",
        )

    print("\n=== C. one topology across every sample ===")
    check("all samples share one face array", len(topology_hashes) == 1, f"{len(topology_hashes)}")
    check(
        "the recomputed topology hash matches the manifest",
        topology_hashes == {row["hashes"]["mesh_topology_sha256"] for row in manifest["samples"]},
    )

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for label in failures:
            print(f"  - {label}")
        return 1
    print(f"ALL CHECKS PASS ({len(manifest['samples'])} samples)")
    return 0


def _resolve(base: Path, recorded: str) -> Path:
    """Manifest paths are either repo-relative or absolute; both must work."""

    path = Path(recorded)
    if path.is_absolute():
        return path
    return base / path if (base / path).is_file() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
