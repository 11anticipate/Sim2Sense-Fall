"""Non-destructive inspection cameras and roof visibility in the USD session layer."""

from __future__ import annotations

import math
from typing import Any

from .usd import pxr_modules

VIEW_MODES = ("top", "roofless", "exterior")
CAMERA_PATH = "/InspectionCamera"


def configure_inspection_view(
    stage: Any, *, mode: str = "top", aspect_ratio: float = 16 / 9
) -> str:
    """Frame the floors and hide roofs for inspection, without changing the asset.

    Only visibility and a camera are authored, in the transient session layer.
    Collision geometry and the on-disk simulation scene remain intact.
    """
    if mode not in VIEW_MODES:
        raise ValueError(f"mode must be one of {VIEW_MODES}")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be finite and positive")
    if stage is None:
        raise ValueError("an open USD stage is required")
    rt = pxr_modules()
    cache = rt.UsdGeom.BBoxCache(rt.Usd.TimeCode.Default(), [rt.UsdGeom.Tokens.default_])
    floors = [
        prim for prim in stage.Traverse()
        if prim.GetAttribute("sim2sense:category").Get() == "floor"
    ]
    if not floors:
        raise ValueError("scene has no floor geometry to frame")
    ranges = [cache.ComputeWorldBound(prim).ComputeAlignedRange() for prim in floors]
    low = [min(box.GetMin()[i] for box in ranges) for i in range(3)]
    high = [max(box.GetMax()[i] for box in ranges) for i in range(3)]
    span = max(high[0] - low[0], high[1] - low[1])
    if not math.isfinite(span) or span <= 0:
        raise ValueError("floor bounds must have a finite positive extent")
    target = rt.Gf.Vec3d((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2])
    with rt.Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.GetAttribute("sim2sense:category").Get() in {"ceiling", "lighting_fixture"}:
                imageable = rt.UsdGeom.Imageable(prim)
                imageable.CreateVisibilityAttr().Set(
                    rt.UsdGeom.Tokens.inherited if mode == "exterior"
                    else rt.UsdGeom.Tokens.invisible
                )
        camera = rt.UsdGeom.Camera.Define(stage, CAMERA_PATH)
        transform = rt.UsdGeom.Xformable(camera)
        transform.ClearXformOpOrder()
        camera.CreateClippingRangeAttr().Set(rt.Gf.Vec2f(0.01, span * 20))
        if mode == "top":
            # USD cameras look along -Z, with +Y as image up.
            transform.AddTranslateOp().Set(target + rt.Gf.Vec3d(0, 0, span * 2))
            camera.CreateProjectionAttr().Set(rt.UsdGeom.Tokens.orthographic)
            height = max(high[1] - low[1], (high[0] - low[0]) / aspect_ratio) * 1.15
            # Apertures are expressed in tenths of a scene unit.
            camera.CreateVerticalApertureAttr().Set(height * 10)
            camera.CreateHorizontalApertureAttr().Set(height * aspect_ratio * 10)
        else:
            eye = target + rt.Gf.Vec3d(span * 0.95, -span * 1.15, span * 1.6)
            matrix = rt.Gf.Matrix4d().SetLookAt(eye, target, rt.Gf.Vec3d(0, 0, 1))
            transform.AddTransformOp().Set(matrix.GetInverse())
            camera.CreateProjectionAttr().Set(rt.UsdGeom.Tokens.perspective)
            camera.CreateFocalLengthAttr().Set(24)
            camera.CreateHorizontalApertureAttr().Set(36)
            camera.CreateVerticalApertureAttr().Set(36 / aspect_ratio)
    return CAMERA_PATH
