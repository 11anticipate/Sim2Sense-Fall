"""Inspection must never remove collision geometry or modify the source layer."""

from __future__ import annotations

import pytest

from sim2sense_fall.scenes.view import configure_inspection_view


def test_view_options_fail_before_loading_usd() -> None:
    with pytest.raises(ValueError, match="mode"):
        configure_inspection_view(None, mode="unknown")
    for aspect in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="aspect_ratio"):
            configure_inspection_view(None, aspect_ratio=aspect)
    with pytest.raises(ValueError, match="open USD stage"):
        configure_inspection_view(None)


def test_inspection_is_transient_and_preserves_colliders() -> None:
    pytest.importorskip("pxr.Usd", reason="requires the bundled OpenUSD runtime")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name, center, size in (
        ("floor", (2, 1.5, -0.06), (4, 3, 0.12)),
        ("ceiling", (2, 1.5, 2.775), (4, 3, 0.15)),
    ):
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}")
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.GetPrim().CreateAttribute("sim2sense:category", Sdf.ValueTypeNames.String).Set(name)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    source_text = stage.GetRootLayer().ExportToString()
    roof = stage.GetPrimAtPath("/World/ceiling")
    for mode in ("top", "roofless", "exterior", "top"):
        camera_path = configure_inspection_view(stage, mode=mode)
        assert stage.GetRootLayer().ExportToString() == source_text
        assert roof.HasAPI(UsdPhysics.CollisionAPI)
        expected = "inherited" if mode == "exterior" else "invisible"
        assert UsdGeom.Imageable(roof).ComputeVisibility() == expected
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        assert camera
        if mode == "top":
            frustum = camera.GetCamera(Usd.TimeCode.Default()).frustum
            # All four corners of the apartment must be inside the inspection camera.
            assert all(frustum.Intersects(Gf.Vec3d(x, y, 0)) for x in (0, 4) for y in (0, 3))
    stage.GetSessionLayer().Clear()
    assert UsdGeom.Imageable(roof).ComputeVisibility() == "inherited"
    assert not stage.GetPrimAtPath(camera_path)
