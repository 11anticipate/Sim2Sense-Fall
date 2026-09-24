"""Inspection must never remove collision geometry or modify the source layer."""

from __future__ import annotations

import pytest

from sim2sense_fall.scenes.view import apply_inspection_view, configure_inspection_view


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
    human = UsdGeom.Cube.Define(stage, "/World/Human/Skin")
    human.CreateSizeAttr(1)
    human.AddTranslateOp().Set(Gf.Vec3d(2, 1.5, 1.0))
    human.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 2.0))
    human.GetPrim().CreateAttribute("sim2sense:category", Sdf.ValueTypeNames.String).Set(
        "human_skin"
    )
    source_text = stage.GetRootLayer().ExportToString()
    roof = stage.GetPrimAtPath("/World/ceiling")
    for mode in ("human", "top", "roofless", "exterior", "top"):
        camera_path = configure_inspection_view(stage, mode=mode)
        assert stage.GetRootLayer().ExportToString() == source_text
        assert roof.HasAPI(UsdPhysics.CollisionAPI)
        expected = "inherited" if mode == "exterior" else "invisible"
        assert UsdGeom.Imageable(roof).ComputeVisibility() == expected
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        assert camera
        if mode == "human":
            assert camera.GetProjectionAttr().Get() == UsdGeom.Tokens.perspective
            frustum = camera.GetCamera(Usd.TimeCode.Default()).frustum
            assert all(frustum.Intersects(Gf.Vec3d(2, 1.5, z)) for z in (0, 1, 2))
        if mode == "top":
            frustum = camera.GetCamera(Usd.TimeCode.Default()).frustum
            # All four corners of the apartment must be inside the inspection camera.
            assert all(frustum.Intersects(Gf.Vec3d(x, y, 0)) for x in (0, 4) for y in (0, 3))
    stage.GetSessionLayer().Clear()
    assert UsdGeom.Imageable(roof).ComputeVisibility() == "inherited"
    assert not stage.GetPrimAtPath(camera_path)


def _closed_room():
    """A one-room apartment with a roof and a body inside it."""

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
    human = UsdGeom.Cube.Define(stage, "/World/Human/Skin")
    human.CreateSizeAttr(1)
    human.AddTranslateOp().Set(Gf.Vec3d(2, 1.5, 1.0))
    human.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 2.0))
    return stage


def test_applying_a_view_selects_the_camera_it_authored() -> None:
    """Authoring a camera is only half of it, and the roof must go with it.

    Every GUI entry point used to call ``configure_inspection_view`` and drop the
    returned path, so the viewport stayed on Isaac's default camera -- a closed room
    seen through the roof -- while the trial reported success. The same call also has
    to hide the ceiling, or the body is simply not visible.
    """

    pytest.importorskip("pxr.Usd", reason="requires the bundled OpenUSD runtime")
    from pxr import Usd, UsdGeom

    stage = _closed_room()
    roof = stage.GetPrimAtPath("/World/ceiling")

    # No Isaac Sim in this interpreter: the camera is still authored and returned, and
    # the roof is still hidden, because that part does not need a viewport.
    assert apply_inspection_view(stage, mode="human") == "/InspectionCamera"
    assert UsdGeom.Imageable(roof).ComputeVisibility() == "invisible"
    assert Usd.Stage.Get(stage, "/InspectionCamera")

    # A caller whose whole purpose is a person watching a window must hear about it
    # when there is no window, rather than silently keep the default camera. This is
    # the no-Kit case: under Isaac Sim headless a viewport object does exist, which is
    # why the GUI entry points only ask for this when ``--gui`` was actually passed.
    with pytest.raises(RuntimeError, match="viewport"):
        apply_inspection_view(stage, mode="top", require_viewport=True)
