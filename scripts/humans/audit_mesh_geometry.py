#!/usr/bin/env python3
"""Audit collected triangle topology and area without inferring self-intersection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(path: Path) -> dict:
    with np.load(path) as data:
        v, f = data["mesh_vertices_xyz"], data["mesh_faces"]
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    unique, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    areas, volumes, min_z = [], [], []
    for frame in v:
        a, b, c = frame[f[:, 0]], frame[f[:, 1]], frame[f[:, 2]]
        areas.append(float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2))
        volumes.append(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6))
        min_z.append(float(frame[:, 2].min()))
    return {
        "file": str(path),
        "vertices": v.shape[1],
        "faces": len(f),
        "edges": len(unique),
        "euler": v.shape[1] - len(unique) + len(f),
        "boundary_edges": int(sum(counts == 1)),
        "nonmanifold_edges": int(sum(counts > 2)),
        "inconsistent_edge_winding": int(sum(balance != 0)),
        "area_m2_min_max": [min(areas), max(areas)],
        "signed_volume_m3_min_max": [min(volumes), max(volumes)],
        "minimum_vertex_z_m": min(min_z),
        "self_intersection": "not tested; topology and signed volume do not establish it",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.dir.glob("*.mesh.npz"))
    if not paths:
        parser.error("no mesh NPZ files found")
    report = [audit(path) for path in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
