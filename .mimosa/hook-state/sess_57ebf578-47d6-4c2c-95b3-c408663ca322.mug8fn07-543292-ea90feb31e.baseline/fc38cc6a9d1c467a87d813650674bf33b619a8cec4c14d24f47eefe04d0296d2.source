"""Optional real OpenUSD tests; runnable without a GPU or SimulationApp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sim2sense_fall.scenes.planner import ScenePlan, plan_scene, write_manifest
from sim2sense_fall.scenes.spec import load_scene_spec
from sim2sense_fall.scenes.verification import load_scene_manifest, stage_manifest_errors

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def authored_scene(tmp_path: Path) -> tuple[Any, ScenePlan]:
    pytest.importorskip("pxr.Usd", reason="requires the bundled OpenUSD runtime")
    from pxr import Usd

    from sim2sense_fall.scenes.usd import build_stage

    plan = plan_scene(load_scene_spec(ROOT / "configs/scenes/indoor_apartment.yaml"))
    manifest = write_manifest(plan, tmp_path / "scene.json")
    path = build_stage(plan, tmp_path / "scene.usda")
    return Usd.Stage.Open(str(path)), load_scene_manifest(manifest)


def test_real_export_matches_manifest_and_floors_have_separate_support(
    authored_scene: tuple[Any, ScenePlan],
) -> None:
    from pxr import Usd, UsdGeom

    stage, plan = authored_scene
    assert stage_manifest_errors(stage, plan) == []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    base = cache.ComputeWorldBound(
        stage.GetPrimAtPath("/World/site/foundation")
    ).ComputeAlignedRange()
    for p in plan.prims:
        if p.category != "floor":
            continue
        floor = cache.ComputeWorldBound(
            stage.GetPrimAtPath(f"/World/{p.path}")
        ).ComputeAlignedRange()
        assert base.GetMax()[2] == pytest.approx(floor.GetMin()[2], abs=1e-6)
        assert base.GetMax()[2] < floor.GetMax()[2] - 0.1


@pytest.mark.parametrize(
    "mutation", ["translate", "size", "material", "friction", "em", "collision"]
)
def test_same_count_wrong_asset_is_rejected(
    authored_scene: tuple[Any, ScenePlan],
    mutation: str,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    stage, plan = authored_scene
    floor = stage.GetPrimAtPath("/World/rooms/bedroom/floor")
    original_count = len(list(stage.Traverse()))
    if mutation == "translate":
        floor.GetAttribute("xformOp:translate").Set(Gf.Vec3d(99, 99, 0))
    elif mutation == "size":
        UsdGeom.Cube(floor).GetSizeAttr().Set(2)
    elif mutation == "material":
        UsdShade.MaterialBindingAPI(floor).Bind(
            UsdShade.Material(stage.GetPrimAtPath("/World/Materials/tile_floor"))
        )
    elif mutation == "friction":
        material, _ = UsdShade.MaterialBindingAPI(floor).ComputeBoundMaterial("physics")
        UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Set(0.01)
    elif mutation == "em":
        stage.GetPrimAtPath("/World/Materials/wood_floor").GetAttribute(
            "sim2sense:em_conductivity_s_per_m"
        ).Set(123)
    else:
        UsdPhysics.CollisionAPI(floor).GetCollisionEnabledAttr().Set(False)
    assert len(list(stage.Traverse())) == original_count
    assert stage_manifest_errors(stage, plan), mutation
