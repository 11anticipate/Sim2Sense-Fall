#!/usr/bin/env python3
"""Render recorded world-space skin in Isaac, without advancing physics.

The input is a simulate.py trial or CPU reference mesh archive. Rendering is
post-simulation evidence: it does not execute or validate a controller.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import DEFAULT_SCENE, boot_isaac, open_scene, write_json  # noqa: E402

from sim2sense_fall.humans.usd_human import write_display_skin  # noqa: E402
from sim2sense_fall.scenes.view import apply_inspection_view  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--camera-from", type=Path, default=None,
                        help="reuse captures.json camera for a direct reference comparison")
    parser.add_argument("--azimuth-deg", type=float, default=-55.0)
    parser.add_argument("--elevation-deg", type=float, default=65.0)
    args = parser.parse_args()
    if args.count < 2 or not all(np.isfinite(v) for v in (
        args.azimuth_deg, args.elevation_deg
    )):
        parser.error("count must be >= 2 and camera angles must be finite")
    with np.load(args.record, allow_pickle=False) as data:
        points = data["mesh_vertices_xyz"]
        faces = data["mesh_faces"]
        times = data["time_physics_s"] if "time_physics_s" in data else data["time_s"]
    if points.ndim != 3 or points.shape[-1] != 3 or len(points) != len(times):
        raise ValueError("record must contain a time-aligned (T,V,3) skin")
    if not np.isfinite(points).all():
        raise ValueError("record contains non-finite mesh points")
    frames = np.unique(np.linspace(0, len(times) - 1, args.count).round().astype(int))
    args.out.mkdir(parents=True, exist_ok=True)
    app = boot_isaac(True, width=1280, height=960)
    if app is None:
        raise RuntimeError("run this script with Isaac Sim python.sh")
    exit_code = 1
    try:
        import omni.timeline
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            get_active_viewport,
            next_viewport_frame_async,
        )
        from pxr import Gf, UsdGeom

        stage = open_scene(app, args.scene)
        omni.timeline.get_timeline_interface().stop()
        camera_path = apply_inspection_view(stage, mode="roofless", aspect_ratio=4 / 3)
        stage.SetEditTarget(stage.GetSessionLayer())
        # One fixed camera for all frames makes root drift and floor contact visible.
        low, high = points[frames].min(axis=(0, 1)), points[frames].max(axis=(0, 1))
        low[2] = min(0.0, low[2])
        target = (low + high) / 2
        span = max(float(np.max(high - low)), 1.0)
        az, el = np.deg2rad([args.azimuth_deg, args.elevation_deg])
        eye = target + span * 1.8 * np.array([
            np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
        ])
        if args.camera_from is not None:
            camera_record = json.loads(args.camera_from.read_text(encoding="utf-8"))
            eye = np.asarray(camera_record["camera_eye"], dtype=float)
            target = np.asarray(camera_record["camera_target"], dtype=float)
            if eye.shape != (3,) or target.shape != (3,) or not np.isfinite([eye, target]).all():
                raise ValueError("camera manifest must contain finite eye and target coordinates")
        transform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
        transform.ClearXformOpOrder()
        transform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)
        ).GetInverse())
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        camera.GetFocalLengthAttr().Set(32.0)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport for screenshot capture")

        async def capture() -> list[dict]:
            rows = []
            for frame in frames:
                write_display_skin(stage, points[frame], faces)
                await next_viewport_frame_async(viewport, 3)
                path = args.out / f"frame_{int(frame):05d}.png"
                pending = capture_viewport_to_file(viewport, file_path=str(path))
                await pending.wait_for_result()
                deadline = time.monotonic() + 10.0
                while not path.is_file() or path.stat().st_size == 0:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"capture did not flush: {path}")
                    await next_viewport_frame_async(viewport, 1)
                rows.append({"frame": int(frame), "time_s": float(times[frame]),
                             "image": str(path.resolve())})
            await next_viewport_frame_async(viewport, 5)
            return rows

        task = asyncio.ensure_future(capture())
        while app.is_running() and not task.done():
            app.update()
        rows = task.result()
        for row in rows:
            if not Path(row["image"]).is_file() or Path(row["image"]).stat().st_size == 0:
                raise RuntimeError(f"capture did not flush: {row['image']}")
        write_json(args.out / "captures.json", {
            "record": str(args.record.resolve()),
            "record_sha256": hashlib.sha256(args.record.read_bytes()).hexdigest(),
            "render_mode": "recorded_skin_physics_stopped",
            "camera_eye": eye.tolist(), "camera_target": target.tolist(), "frames": rows,
        })
        exit_code = 0
    except Exception:
        logging.exception("recorded mesh rendering failed")
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
