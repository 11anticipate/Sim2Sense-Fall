#!/usr/bin/env python3
"""Render a collected fall-mesh directory so the motion can be judged by eye.

A figure is not a substitute for a check, but it is the one confirmation that requires no
trust in the checker: if the body faces sideways, or the fall goes the wrong way, or the
skin has slid off the skeleton, it is obvious here immediately. That matters because the
defect this project shipped was a rotation -- and a rotation is exactly what every
rotation-invariant assertion in the pipeline cannot see.

The body is drawn as a point cloud of skin vertices rather than shaded triangles.
matplotlib's 3D depth sorting is unreliable for alpha-blended surfaces and silently drops
them, whereas a scatter always draws, and for judging pose and facing a silhouette is
just as readable.

Usage::

    python3 scripts/humans/preview_fall_meshes.py [--dir artifacts/humans/fall_mesh]
                                                  [--clips a b c] [--frames 0 24 48]
                                                  [--out preview.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "artifacts" / "humans" / "fall_mesh"

SKIN_COLOUR = "#4C6EF5"
LINK_COLOUR = "#F03E3E"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="clip ids to draw; defaults to every fall clip in the manifest",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        default=None,
        help="frame indices to draw; defaults to seven evenly spaced samples",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=4, help="draw every Nth skin vertex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((args.dir / "manifest.json").read_text(encoding="utf-8"))
    available = [row["sample_id"] for row in manifest["samples"]]
    # Only the topples: a preview of every clip in a mixed collection is unreadable, and
    # the point of this figure is to see the fall happen.
    clips = args.clips or [name for name in available if name.startswith("fall_")]
    if not clips:
        print(f"no fall clips in {args.dir}; pass --clips explicitly")
        return 2
    unknown = [name for name in clips if name not in available]
    if unknown:
        print(f"unknown clip(s) {unknown}; available: {available}")
        return 2

    out = args.out or (args.dir / "fall_preview.png")
    figure = plt.figure(figsize=(15.5, 3.5 * len(clips)))
    for row, clip_id in enumerate(clips):
        vertices = np.load(args.dir / f"{clip_id}.mesh.npz")["mesh_vertices_xyz"]
        links = np.load(args.dir / f"{clip_id}.points.npz")["link_positions_xyz"]
        frames = args.frames or _evenly_spaced(vertices.shape[0], 7)
        bounds = _bounds_over(vertices)
        for column, frame in enumerate(frames):
            index = min(int(frame), vertices.shape[0] - 1)
            axis = figure.add_subplot(
                len(clips), len(frames), row * len(frames) + column + 1, projection="3d"
            )
            skin = vertices[index][:: max(args.stride, 1)]
            axis.scatter(
                skin[:, 0],
                skin[:, 1],
                skin[:, 2],
                s=0.35,
                c=SKIN_COLOUR,
                alpha=0.30,
                linewidths=0,
            )
            axis.scatter(
                links[index][:, 0],
                links[index][:, 1],
                links[index][:, 2],
                s=7,
                c=LINK_COLOUR,
                depthshade=False,
            )
            _frame_axes(axis, bounds)
            axis.set_title(f"frame {index}", fontsize=8)
            if column == 0:
                axis.text2D(
                    -0.20,
                    0.5,
                    clip_id.replace("_reference", ""),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )
    figure.suptitle(
        "Exported fall meshes  |  blue dots = SMPL skin vertices, red dots = rig links  |  "
        "equal axes, world frame (z up)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.02, 0, 1, 0.96))
    figure.savefig(out, dpi=115, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


def _bounds_over(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """One box covering the whole clip, so every panel is comparable by eye.

    Deliberately not per-frame autoscaling: that would rescale each panel to its own
    contents and make a topple look like a translation. Every sample spawns at the same
    place, so a single box per clip is both honest and readable.
    """

    flat = vertices.reshape(-1, 3)
    centre_x = float((flat[:, 0].min() + flat[:, 0].max()) / 2)
    centre_y = float((flat[:, 1].min() + flat[:, 1].max()) / 2)
    half = float(max(flat[:, 0].ptp(), flat[:, 1].ptp()) / 2) * 1.15
    top = float(flat[:, 2].max()) * 1.08
    return centre_x - half, centre_x + half, centre_y - half, centre_y + half, top


def _frame_axes(axis: object, bounds: tuple[float, float, float, float, float]) -> None:
    x_lo, x_hi, y_lo, y_hi, top = bounds
    axis.set_xlim(x_lo, x_hi)
    axis.set_ylim(y_lo, y_hi)
    axis.set_zlim(0.0, top)
    axis.set_box_aspect((x_hi - x_lo, y_hi - y_lo, top))
    axis.view_init(elev=10, azim=-60)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])


def _evenly_spaced(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    step = total / count
    return [min(int(round(index * step)), total - 1) for index in range(count)]


if __name__ == "__main__":
    raise SystemExit(main())
