#!/usr/bin/env python3
"""Render trial ``npz`` mesh trajectories into a side-by-side comparison MP4.

Each input npz is the export contract written by ``export_tracker_trials.py``
(``time_physics_s`` / ``mesh_vertices_xyz`` / ``mesh_faces``). Clips are shown
as shaded silhouette panels in a follow camera that rotates the motion's
dominant travel direction to face right, so limb chatter and tracking deviation
are directly visible. The renderer is deliberately dependency-light
(numpy + PIL + ffmpeg) so acceptance renders reproduce anywhere.

Example:
    python scripts/protomotions/render_trial_mesh.py \
        --panel ref=artifacts/.../ref_Trial_77.npz \
        --panel r4=artifacts/.../tracker_Trial_77.npz \
        --out artifacts/.../compare_Trial_77.mp4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LOGGER = logging.getLogger("render_trial_mesh")

PANEL_W = 550
PANEL_H = 620
BASE_COLORS = {
    "ref": (110, 165, 110),
    "r4": (120, 155, 205),
    "fix": (205, 140, 90),
}


@dataclass(frozen=True)
class Clip:
    """A resampled, centred mesh trajectory with per-clip vertical framing."""

    name: str
    frames: np.ndarray  # (T, Nv, 3) world-aligned vertices
    faces: np.ndarray  # (F, 3)
    color: tuple[int, int, int]
    v_center: float  # vertical framing centre (m, clip-aligned frame)
    v_span: float  # vertical framing span (m)
    floor_z: float  # clip-aligned floor level for the ground line


def load_and_align(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one npz and centre it on its first-frame vertex centroid."""
    data = np.load(npz_path, allow_pickle=True)
    verts = data["mesh_vertices_xyz"].astype(np.float64)
    faces = data["mesh_faces"].astype(np.int64)
    verts = verts - verts[0].mean(axis=0, keepdims=True)
    return verts, faces


def resample_to_grid(verts: np.ndarray, time_s: np.ndarray, fps_out: float) -> np.ndarray:
    """Nearest-frame resample onto a uniform ``fps_out`` time grid."""
    t0, t1 = float(time_s[0]), float(time_s[-1])
    grid = np.arange(t0, t1, 1.0 / fps_out)
    idx = np.clip(np.searchsorted(time_s, grid), 0, len(time_s) - 1)
    return verts[idx]


def rotate_to_travel(verts: np.ndarray) -> np.ndarray:
    """Rotate xy so the clip's dominant horizontal travel faces +x."""
    centroid = verts.mean(axis=1)  # (T, 3)
    delta = centroid[-1] - centroid[0]
    if np.hypot(delta[0], delta[1]) < 0.3:  # ~stationary clip: keep world axes
        return verts
    angle = np.arctan2(delta[1], delta[0])
    cos, sin = np.cos(-angle), np.sin(-angle)
    out = verts.copy()
    out[:, :, 0] = cos * verts[:, :, 0] - sin * verts[:, :, 1]
    out[:, :, 1] = sin * verts[:, :, 0] + cos * verts[:, :, 1]
    return out


def build_clip(name: str, npz_path: Path, fps_out: float) -> Clip:
    """Load, align, resample and frame one clip."""
    verts, faces = load_and_align(npz_path)
    time_s = np.load(npz_path, allow_pickle=True)["time_physics_s"].astype(float)
    frames = resample_to_grid(rotate_to_travel(verts), time_s, fps_out)
    zlo, zhi = float(frames[:, :, 2].min()), float(frames[:, :, 2].max())
    v_span = max((zhi - zlo) * 1.22, 1.2)
    return Clip(
        name=name,
        frames=frames,
        faces=faces,
        color=BASE_COLORS.get(name, (150, 150, 150)),
        v_center=0.5 * (zlo + zhi),
        v_span=v_span,
        floor_z=zlo,
    )


def face_normals(verts_frame: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = verts_frame[faces[:, 0]], verts_frame[faces[:, 1]], verts_frame[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norm < 1e-12, 1.0, norm)


def render_panel(clip: Clip, frame_idx: int, u_center: float, label: str) -> Image.Image:
    """One silhouette panel: orthographic side view, painter-sorted flat shade."""
    verts_frame = clip.frames[frame_idx]
    faces = clip.faces
    u, v, depth = verts_frame[:, 0], verts_frame[:, 2], verts_frame[:, 1]
    scale = PANEL_H / clip.v_span

    def to_px(u_arr: np.ndarray, v_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = (u_arr - u_center) * scale + PANEL_W / 2
        py = (clip.v_center + clip.v_span / 2 - v_arr) * scale
        return px, py

    normals = face_normals(verts_frame, faces)
    light = np.array([0.2, 0.5, 0.85])
    light = light / np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(normals @ light)  # two-sided lambert

    px, py = to_px(u[faces], v[faces])
    order = np.argsort(depth[faces].mean(axis=1))  # far first

    img = Image.new("RGB", (PANEL_W, PANEL_H), "white")
    draw = ImageDraw.Draw(img)
    ground_y = (clip.v_center + clip.v_span / 2 - clip.floor_z) * scale
    draw.line([(0, ground_y), (PANEL_W, ground_y)], fill=(190, 190, 190), width=1)
    rgb = np.asarray(clip.color, dtype=float)
    for i in order:
        s = shade[i]
        fill = tuple(int(min(255, c * s + 90 * (1 - s))) for c in rgb)
        draw.polygon(list(zip(px[i], py[i], strict=True)), fill=fill)
    draw.rectangle([0, 0, 150, 24], fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 0))
    return img


def smooth_center(verts: np.ndarray, half_window: int = 7) -> np.ndarray:
    """Per-frame horizontal camera target (box-smoothed centroid)."""
    centroid = verts.mean(axis=1)
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    return np.stack(
        [np.convolve(centroid[:, k], kernel, mode="same") for k in range(3)], axis=1
    )


def render_compare(clips: list[Clip], out_path: Path, fps: int) -> Path:
    """Render panels per frame side by side and encode with ffmpeg."""
    n_frames = min(clip.frames.shape[0] for clip in clips)
    tmp_dir = out_path.parent / f".{out_path.stem}_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    centers = [smooth_center(clip.frames) for clip in clips]
    for i in range(n_frames):
        panels = [
            render_panel(clip, i, float(center[i, 0]), f"{clip.name} t={i / fps:.2f}s")
            for clip, center in zip(clips, centers, strict=True)
        ]
        combined = Image.new("RGB", (PANEL_W * len(panels), PANEL_H), "white")
        for j, panel in enumerate(panels):
            combined.paste(panel, (j * PANEL_W, 0))
        combined.save(tmp_dir / f"{i:05d}.png")

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-framerate", str(fps), "-i", str(tmp_dir / "%05d.png"),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    for frame_path in sorted(tmp_dir.glob("*.png")):
        frame_path.unlink()
    tmp_dir.rmdir()
    LOGGER.info("wrote %s (%d frames @ %d fps)", out_path, n_frames, fps)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--panel", action="append", required=True, metavar="NAME=NPZ",
        help="clip panel, e.g. --panel ref=foo.npz (repeat up to 3; names "
        "ref/r4/fix pick default colors)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    if len(args.panel) > 3:
        parser.error("at most 3 panels")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for spec in args.panel:
        if not spec.partition("=")[2]:
            parser.error(f"--panel expects NAME=NPZ, got {spec!r}")
    clips = [build_clip(name, Path(npz), args.fps) for name, _, npz in
             (spec.partition("=") for spec in args.panel)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_compare(clips, args.out, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
