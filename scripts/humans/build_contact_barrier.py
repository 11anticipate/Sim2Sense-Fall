#!/usr/bin/env python3
"""Create a separate scene with a static barrier for a collision negative control."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from common import DEFAULT_SCENE, boot_isaac, open_scene, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--size", type=float, nargs=3, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (*args.center, *args.size)) or min(args.size) <= 0:
        parser.error("barrier coordinates must be finite and sizes positive")
    if args.scene.resolve() == args.out.resolve():
        parser.error("barrier must be written to a separate scene")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"scene": str(args.scene.resolve()), "center_m": args.center,
                "size_m": args.size, "path": "/World/ContactBarrier", "dynamic": False}
    write_json(args.out.with_suffix(".json"), manifest)
    if args.dry_run:
        return 0
    app = boot_isaac(True)
    if app is None:
        raise RuntimeError("run through Isaac Sim python.sh or use --dry-run")
    exit_code = 1
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        stage = open_scene(app, args.scene)
        cube = UsdGeom.Cube.Define(stage, manifest["path"])
        cube.CreateSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*args.center))
        cube.AddScaleOp().Set(Gf.Vec3f(*args.size))
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.15, 0.1)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().CreateAttribute("sim2sense:category", Sdf.ValueTypeNames.String).Set("wall")
        stage.Export(str(args.out))
        exit_code = 0
    except Exception:
        logging.exception("barrier scene creation failed")
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
