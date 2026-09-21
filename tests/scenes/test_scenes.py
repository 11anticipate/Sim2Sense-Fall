"""CPU tests for the indoor scene pipeline.

Everything here runs without Isaac Sim: the scene schema, the material library and
the geometry planner are all pure Python. The USD authoring step is exercised
separately by ``scripts/scenes/build.py --headless``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sim2sense_fall.scenes.furniture import furniture_parts, known_furniture_kinds
from sim2sense_fall.scenes.materials import (
    DEFAULT_MATERIALS,
    MaterialSpec,
    material_library_from_config,
)
from sim2sense_fall.scenes.planner import plan_scene, write_manifest
from sim2sense_fall.scenes.spec import SceneSpec, load_scene_spec, scene_spec_from_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_CONFIG = REPO_ROOT / "configs" / "scenes" / "indoor_apartment.yaml"


@pytest.fixture(scope="module")
def scene_spec() -> SceneSpec:
    return load_scene_spec(SCENE_CONFIG)


# --------------------------------------------------------------------------------------
# Material library
# --------------------------------------------------------------------------------------


def test_default_materials_are_valid() -> None:
    assert len(DEFAULT_MATERIALS) >= 12
    for name, spec in DEFAULT_MATERIALS.items():
        assert spec.name == name
        assert spec.dynamic_friction <= spec.static_friction
        assert 0.0 <= spec.roughness <= 1.0
        assert spec.density_kg_m3 > 0


def test_dynamic_friction_above_static_is_rejected() -> None:
    with pytest.raises(ValueError, match="dynamic friction"):
        MaterialSpec(
            name="bad",
            base_color=(0.5, 0.5, 0.5),
            roughness=0.5,
            metallic=0.0,
            static_friction=0.2,
            dynamic_friction=0.9,
            restitution=0.0,
            thickness_m=0.01,
        )


def test_itu_power_law_scales_with_frequency() -> None:
    plasterboard = DEFAULT_MATERIALS["painted_drywall"].em
    assert plasterboard is not None
    low = plasterboard.conductivity(2.4e9)
    high = plasterboard.conductivity(5.0e9)
    assert 0 < low < high
    assert plasterboard.relative_permittivity(2.4e9) == pytest.approx(2.73)


def test_unknown_material_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown material override"):
        material_library_from_config({"unicorn_hide": {"roughness": 0.1}})


def test_material_override_only_patches_listed_fields() -> None:
    library = material_library_from_config({"tile_floor": {"dynamic_friction": 0.30}})
    assert library["tile_floor"].dynamic_friction == 0.30
    assert library["tile_floor"].static_friction == DEFAULT_MATERIALS["tile_floor"].static_friction


# --------------------------------------------------------------------------------------
# Scene schema
# --------------------------------------------------------------------------------------


def test_shipping_scene_loads(scene_spec: SceneSpec) -> None:
    assert scene_spec.scene_id == "apartment_cn_two_bedroom"
    assert {room.id for room in scene_spec.rooms} == {
        "bedroom",
        "living_room",
        "corridor",
        "bathroom",
        "kitchen",
        "bedroom_2",
    }
    assert scene_spec.frequency_hz == pytest.approx(2.4e9)
    assert scene_spec.footprint == (0.0, 0.0, 8.4, 7.0)


def test_shared_walls_are_not_built_twice(scene_spec: SceneSpec) -> None:
    """A boundary between two rooms must be authored by exactly one of them.

    Walls are inset and span the full room dimension, so boxes from the *same*
    room legitimately overlap in a corner. Boxes from *different* rooms must never
    overlap, which is what catches a duplicated shared wall.
    """
    plan = plan_scene(scene_spec)
    walls = [prim for prim in plan.prims if prim.category == "wall"]
    for index, left in enumerate(walls):
        for right in walls[index + 1 :]:
            if left.room_id == right.room_id:
                continue
            assert not _boxes_overlap(left, right), (
                f"{left.path} and {right.path} from different rooms overlap"
            )


def _boxes_overlap(left, right, epsilon: float = 1e-6) -> bool:
    for axis in range(3):
        low_a = left.center[axis] - left.size[axis] / 2
        high_a = left.center[axis] + left.size[axis] / 2
        low_b = right.center[axis] - right.size[axis] / 2
        high_b = right.center[axis] + right.size[axis] / 2
        if min(high_a, high_b) - max(low_a, low_b) <= epsilon:
            return False
    return True


def test_overlapping_rooms_are_rejected() -> None:
    payload = {
        "scene_id": "overlap",
        "rooms": [
            {
                "id": "a",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [3.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
            },
            {
                "id": "b",
                "room_type": "bedroom",
                "origin": [2.0, 1.0],
                "size": [3.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
            },
        ],
    }
    with pytest.raises(ValueError, match="overlap"):
        scene_spec_from_mapping(payload)


def test_unknown_material_in_room_is_rejected() -> None:
    payload = {
        "scene_id": "bad-material",
        "rooms": [
            {
                "id": "a",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [3.0, 3.0],
                "wall_material": "unobtainium",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown material"):
        scene_spec_from_mapping(payload)


def test_opening_outside_wall_is_rejected() -> None:
    payload = {
        "scene_id": "bad-opening",
        "rooms": [
            {
                "id": "a",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [3.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
                "openings": [
                    {"wall": "south", "kind": "door", "width": 1.5, "height": 2.0, "offset": 0.4}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="starts before the wall"):
        scene_spec_from_mapping(payload)


def test_opening_on_unbuilt_wall_is_rejected() -> None:
    payload = {
        "scene_id": "opening-on-foreign-wall",
        "rooms": [
            {
                "id": "a",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [3.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
                "walls": ["south"],
                "openings": [
                    {"wall": "north", "kind": "door", "width": 0.8, "height": 2.0, "offset": 1.0}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="that wall is not built"):
        scene_spec_from_mapping(payload)


def test_yaml_float_exponent_hint() -> None:
    payload = {
        "scene_id": "s",
        "frequency_hz": "2.4e9",
        "rooms": [
            {
                "id": "r",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [3.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
            }
        ],
    }
    with pytest.raises(ValueError, match="signed exponent"):
        scene_spec_from_mapping(payload)


# --------------------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------------------


def test_shipping_scene_plan(scene_spec: SceneSpec) -> None:
    plan = plan_scene(scene_spec)
    stats = plan.stats
    assert stats["room_count"] == 6
    assert stats["prim_count"] == stats["geometry_count"] + stats["rigid_body_count"]
    assert stats["total_floor_area_m2"] == pytest.approx(58.8)
    assert stats["footprint_m"] == [8.4, 7.0]
    assert stats["rigid_body_count"] == 1
    assert stats["rigid_body_mass_kg"] == pytest.approx(13.56, abs=0.5)
    assert "wall" in stats["primitives_by_category"]


def test_walls_leave_openings_clear(scene_spec: SceneSpec) -> None:
    """No wall slab may intrude into the void it was cut for."""

    plan = plan_scene(scene_spec)
    for room in scene_spec.rooms:
        for opening in room.openings:
            along_x = opening.wall in ("south", "north")
            axis = 0 if along_x else 1
            start = room.origin[axis]
            center = start + opening.offset
            u_lo, u_hi = center - opening.width / 2, center + opening.width / 2
            z_lo = opening.sill_height
            z_hi = opening.sill_height + opening.height
            prefix = f"rooms/{room.id}/walls/{opening.wall}_"
            for prim in plan.prims:
                if not prim.path.startswith(prefix):
                    continue
                prim_u_lo = prim.center[axis] - prim.size[axis] / 2
                prim_u_hi = prim.center[axis] + prim.size[axis] / 2
                prim_z_lo = prim.center[2] - prim.size[2] / 2
                prim_z_hi = prim.center[2] + prim.size[2] / 2
                overlaps_u = prim_u_lo < u_hi - 1e-9 and u_lo < prim_u_hi - 1e-9
                overlaps_z = prim_z_lo < z_hi - 1e-9 and z_lo < prim_z_hi - 1e-9
                assert not (overlaps_u and overlaps_z), (
                    f"{prim.path} intrudes into the {opening.kind} void of {room.id}"
                )


def test_wall_segmentation_around_a_door() -> None:
    """A 0.9 m door in a 2.7 m wall leaves two piers plus one lintel."""

    payload = {
        "scene_id": "door-split",
        "rooms": [
            {
                "id": "r",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [4.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
                "walls": ["south"],
                "openings": [
                    {"wall": "south", "kind": "door", "width": 0.9, "height": 2.05, "offset": 1.0}
                ],
            }
        ],
    }
    plan = plan_scene(scene_spec_from_mapping(payload))
    segments = [prim for prim in plan.prims if prim.path.startswith("rooms/r/walls/south_")]
    assert len(segments) == 3
    lintel = [prim for prim in segments if prim.center[2] > 2.0]
    assert len(lintel) == 1
    assert lintel[0].size[0] == pytest.approx(0.9)


def test_wall_segmentation_around_a_window() -> None:
    """A sill-level window adds the spandrel below and the lintel above."""

    payload = {
        "scene_id": "window-split",
        "rooms": [
            {
                "id": "r",
                "room_type": "bedroom",
                "origin": [0.0, 0.0],
                "size": [4.0, 3.0],
                "wall_material": "painted_drywall",
                "floor_material": "wood_floor",
                "ceiling_material": "gypsum_ceiling",
                "walls": ["south"],
                "openings": [
                    {
                        "wall": "south",
                        "kind": "window",
                        "width": 1.6,
                        "height": 1.3,
                        "offset": 2.0,
                        "sill_height": 0.9,
                    }
                ],
            }
        ],
    }
    plan = plan_scene(scene_spec_from_mapping(payload))
    segments = [prim for prim in plan.prims if prim.path.startswith("rooms/r/walls/south_")]
    assert len(segments) == 4
    z_ranges = sorted(
        (round(prim.center[2] - prim.size[2] / 2, 6), round(prim.center[2] + prim.size[2] / 2, 6))
        for prim in segments
    )
    assert (0.0, 0.9) in z_ranges
    assert (2.2, 2.7) in z_ranges


def test_dynamic_furniture_is_one_compound_body(scene_spec: SceneSpec) -> None:
    plan = plan_scene(scene_spec)
    bodies = plan.bodies
    assert len(bodies) == 1
    body = bodies[0]
    assert body.path == "rooms/bedroom_2/furniture/chair"
    assert body.mass_kg is not None and body.mass_kg > 5.0
    members = [prim for prim in plan.prims if prim.body_path == body.path and prim.is_geometry]
    assert len(members) >= 5, "a chair recipe should contribute several colliders"
    for member in members:
        assert member.relative_to_body
        assert member.rotation_z_deg == 0.0, "member rotation is carried by the body"
        assert member.collision
    # Static furniture must stay in world space and outside any rigid body.
    assert not any(prim.body_path for prim in plan.prims if prim.category == "wall")


def test_material_override_recolours_the_carcass_only() -> None:
    base = furniture_parts("wardrobe", (1.2, 0.6, 2.0), "wood_furniture")
    dominant = max(base, key=lambda part: part.size[0] * part.size[1] * part.size[2])
    assert dominant.material == "wood_furniture"
    assert any(part.material == "painted_mdf" for part in base)


def test_unknown_furniture_kind_falls_back_to_a_box() -> None:
    parts = furniture_parts("holographic_teleporter", (0.5, 0.5, 1.0), "metal_fixture")
    assert len(parts) == 1
    assert parts[0].shape == "box"
    assert parts[0].size == (0.5, 0.5, 1.0)


def test_furniture_kinds_are_all_covered() -> None:
    kinds = known_furniture_kinds()
    assert "bed" in kinds and "toilet" in kinds and "sofa" in kinds
    for kind in kinds:
        parts = furniture_parts(kind, (0.5, 0.5, 0.5), "wood_furniture")
        assert parts, f"{kind} produced no parts"


def test_manifest_is_json_serialisable(scene_spec: SceneSpec, tmp_path: Path) -> None:
    plan = plan_scene(scene_spec)
    target = write_manifest(plan, tmp_path / "manifest.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["scene_id"] == scene_spec.scene_id
    assert len(payload["prims"]) == plan.stats["prim_count"]
    assert payload["stats"]["room_count"] == 6
    # Radio parameters must travel with the material record, not be recomputed later.
    assert payload["materials"]["painted_drywall"]["electromagnetic"]["sionna_material"] == (
        "itu_plasterboard"
    )


def test_every_planned_prim_has_a_unique_path(scene_spec: SceneSpec) -> None:
    plan = plan_scene(scene_spec)
    paths = [prim.path for prim in plan.prims]
    assert len(paths) == len(set(paths))


def test_config_file_is_ascii_safe_yaml() -> None:
    payload = yaml.safe_load(SCENE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["frequency_hz"] == pytest.approx(2.4e9)
