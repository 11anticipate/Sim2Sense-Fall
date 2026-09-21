"""Regression cases from the indoor asset acceptance review (R1–R6)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from sim2sense_fall.scenes.furniture import furniture_parts
from sim2sense_fall.scenes.geometry import intersects, world_shapes
from sim2sense_fall.scenes.planner import plan_scene, write_manifest
from sim2sense_fall.scenes.spec import load_scene_spec, scene_spec_from_mapping
from sim2sense_fall.scenes.verification import load_scene_manifest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/scenes/indoor_apartment.yaml"


@pytest.fixture
def payload() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "1.0"])
@pytest.mark.parametrize("field", ["position", "size", "rotation_z_deg", "density_kg_m3"])
def test_invalid_furniture_numbers(payload: dict[str, Any], field: str, invalid: Any) -> None:
    item = payload["rooms"][0]["furniture"][0]
    item[field] = (
        [invalid, 1] if field == "position" else ([invalid, 1, 1] if field == "size" else invalid)
    )
    with pytest.raises(ValueError, match="number"):
        scene_spec_from_mapping(payload)


@pytest.mark.parametrize("field", ["frequency_hz", "foundation_margin_m", "foundation_thickness_m"])
def test_invalid_scene_numbers(payload: dict[str, Any], field: str) -> None:
    payload[field] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        scene_spec_from_mapping(payload)


def test_direct_dataclasses_and_derived_parts_reject_invalid_geometry() -> None:
    spec = load_scene_spec(CONFIG)
    with pytest.raises(ValueError, match="finite"):
        replace(spec.rooms[0], wall_height=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        replace(spec.rooms[0].openings[0], height=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        replace(spec.materials["tile_floor"], thickness_m=float("nan"))
    with pytest.raises(ValueError, match="recipe.*positive"):
        furniture_parts("bed", (0.05, 0.05, 0.05), "wood_furniture")
    with pytest.raises(ValueError, match="positive"):
        replace(plan_scene(spec).prims[0], size=(-1, 1, 1))


@pytest.mark.parametrize("target", ["room", "furniture"])
def test_normalized_identifier_collision(payload: dict[str, Any], target: str) -> None:
    if target == "room":
        payload["rooms"][0]["id"] = "same-name"
        payload["rooms"][1]["id"] = "same_name"
    else:
        item = copy.deepcopy(payload["rooms"][0]["furniture"][0])
        item["id"] = "same-name"
        payload["rooms"][0]["furniture"].append(item)
        item = copy.deepcopy(item)
        item["id"] = "same_name"
        payload["rooms"][0]["furniture"].append(item)
    with pytest.raises(ValueError, match="duplicate normalized prim path"):
        plan_scene(scene_spec_from_mapping(payload))


@pytest.mark.parametrize("room_index", [0, 1, 2])
def test_original_oversized_rugs_are_rejected(payload: dict[str, Any], room_index: int) -> None:
    rug = next(p for p in payload["rooms"][room_index]["furniture"] if p["id"] == "rug")
    rug.pop("size")
    with pytest.raises(ValueError, match="outside room|intersects wall"):
        plan_scene(scene_spec_from_mapping(payload))


def test_original_fridge_orientation_is_rejected(payload: dict[str, Any]) -> None:
    fridge = next(p for p in payload["rooms"][4]["furniture"] if p["id"] == "refrigerator")
    fridge["rotation_z_deg"] = 0
    with pytest.raises(ValueError, match="refrigerator.*intersects wall"):
        plan_scene(scene_spec_from_mapping(payload))


def test_varied_floor_thickness_has_continuous_nonoverlapping_support(
    payload: dict[str, Any],
) -> None:
    payload["rooms"][0]["floor_thickness"] = 0.20
    payload["foundation_thickness_m"] = 0.40
    plan = plan_scene(scene_spec_from_mapping(payload))
    shapes = world_shapes(plan.prims)
    base = shapes["site/foundation"]
    assert base.bounds[0][2] == pytest.approx(-0.60)
    assert base.bounds[1][2] == pytest.approx(-0.20)
    for floor in (p for p in plan.prims if p.category == "floor"):
        shape = shapes[floor.path]
        assert shape.bounds[1][2] == pytest.approx(0)
        assert not intersects(shape, base, tolerance=1e-6)
        support = shapes.get(floor.path.replace("/floor", "/subfloor"), base)
        assert support.bounds[1][2] == pytest.approx(shape.bounds[0][2])
        if support is not base:
            assert support.bounds[0][2] == pytest.approx(base.bounds[1][2])


def test_rotated_boxes_do_not_fail_on_aabb_false_positive() -> None:
    from sim2sense_fall.scenes.geometry import WorldShape

    left = WorldShape("a", (0, 0, 1), (2, 0.1, 1), 45, False)
    right = WorldShape("b", (-0.2, 0.2, 1), (2, 0.1, 1), 45, False)
    assert not intersects(left, right, tolerance=1e-6)
    assert intersects(left, replace(right, center=(-0.02, 0.02, 1)), tolerance=1e-6)


def test_compound_body_transform_is_used_for_layout(payload: dict[str, Any]) -> None:
    chair = next(p for p in payload["rooms"][5]["furniture"] if p["id"] == "chair")
    chair["position"] = [0.01, 1.5]
    with pytest.raises(ValueError, match="chair.*outside room"):
        plan_scene(scene_spec_from_mapping(payload))


def test_manifest_roundtrip_and_corruption(tmp_path: Path) -> None:
    plan = plan_scene(load_scene_spec(CONFIG))
    path = write_manifest(plan, tmp_path / "scene.json")
    restored = load_scene_manifest(path)
    assert restored.as_manifest() == plan.as_manifest()
    payload = json.loads(path.read_text())
    payload["prims"][0]["center"][0] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-finite"):
        load_scene_manifest(path)
    with pytest.raises(ValueError, match="JSON compliant"):
        write_manifest(replace(plan, stats={"bad": float("nan")}), tmp_path / "bad.json")


@pytest.mark.parametrize("script", ["build", "verify"])
@pytest.mark.parametrize("failure", ["startup", "runtime"])
def test_cli_failure_survives_fast_shutdown(
    tmp_path: Path,
    script: str,
    failure: str,
) -> None:
    """A separate interpreter emulates Kit's immediate exit, not just a mocked return."""
    manifest = write_manifest(plan_scene(load_scene_spec(CONFIG)), tmp_path / "scene.json")
    usd = tmp_path / "scene.usda"
    usd.write_text("#usda 1.0\n")
    harness = """
import importlib.util, os, sys, types
from pathlib import Path
root, script, failure, usd, manifest, out = sys.argv[1:]
sys.path.insert(0, str(Path(root) / "src"))
def fail(*args, **kwargs):
    raise RuntimeError("injected runtime failure")
class App:
    def __init__(self, *args, **kwargs):
        if failure == "startup":
            fail()
    def close(self, *, exit_code=0):
        os._exit(exit_code)
isaac = types.ModuleType("isaacsim")
isaac.__file__ = "stub"
simulation = types.ModuleType("isaacsim.simulation_app")
simulation.SimulationApp = App
sys.modules.update({"isaacsim": isaac, "isaacsim.simulation_app": simulation,
                   "omni": types.ModuleType("omni"), "omni.usd": types.ModuleType("omni.usd")})
import sim2sense_fall.scenes.usd as scene
scene.build_stage = fail
scene.stage_summary = fail
spec = importlib.util.spec_from_file_location(script, Path(root)/"scripts"/"scenes"/(script+".py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = ["--out", out] if script.startswith("build") else ["--usd", usd, "--manifest", manifest]
raise SystemExit(module.main(args))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            str(ROOT),
            script,
            failure,
            str(usd),
            str(manifest),
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--settle-seconds", "0"],
        ["--drop-height", "nan"],
        ["--drop-height", "0.01"],
        ["--drop-tolerance", "0"],
    ],
)
def test_invalid_physics_control_cannot_pass(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/scenes/verify.py"), *args],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
