"""Turn a declarative scene into concrete primitives, without Isaac Sim.

Planning is deliberately separated from authoring. :func:`plan_scene` is pure
Python and runs on any CPU, so wall segmentation, opening placement, furniture
assembly and the reproducibility manifest can all be unit-tested and reviewed in
a normal CI environment. Only :mod:`sim2sense_fall.isaac_scene` needs the USD
runtime, and it consumes this plan verbatim.

Geometry conventions
--------------------
* Z is up and the unit is one metre.
* Walls are **inset**: a wall occupies the outermost ``wall_thickness`` of its
  room rectangle rather than straddling the boundary, so the building envelope is
  exactly the union of the room rectangles.
* Running walls span the full room dimension and are *not* trimmed at corners, so
  a room that owns only a subset of its walls never leaves a gap where it meets a
  neighbour that owns the rest. Corner boxes then overlap by one wall thickness,
  which is harmless for rendering and for static collision.
* An opening ``offset`` is measured along the wall axis from the room's
  lower-coordinate edge (``origin_x`` for south/north, ``origin_y`` for west/east).

Body model
----------
A body of furniture is never a single box, so a dynamic item cannot simply become
a rigid body per part: it would disintegrate on the first contact. Instead the
plan distinguishes three roles:

* ``is_body`` -- an ``xform`` prim that carries ``RigidBodyAPI`` and one mass for
  the whole item;
* ``body_path`` set, ``relative_to_body`` -- a collider authored in the body's
  local frame, so the parts stay rigidly attached to each other;
* neither -- a world-space static collider, which is how the whole apartment and
  most furniture is authored.

Material overrides
------------------
When a furniture item sets ``material``, it replaces the material of the item's
**largest part** (its carcass) and of every other part sharing that material.
Accent parts such as upholstery, glass and metal are left untouched, so an
override recolours the furniture without destroying its construction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .scene_furniture import Part, furniture_default_size, furniture_parts
from .scene_spec import FurnitureSpec, OpeningSpec, RoomSpec, SceneSpec

__all__ = [
    "LightSpec",
    "ScenePlan",
    "ScenePrim",
    "plan_scene",
    "write_manifest",
]

DOOR_LEAF_THICKNESS_M = 0.04
GLASS_PANE_THICKNESS_M = 0.006
WINDOW_FRAME_DEPTH_M = 0.06
WINDOW_FRAME_WIDTH_M = 0.05

BODY_SHAPE = "xform"


def _safe(name: str) -> str:
    """Make an identifier safe to use as a USD prim name."""

    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _volume(shape: str, size: tuple[float, float, float]) -> float:
    if shape == "cylinder":
        radius = size[0] / 2
        return math.pi * radius * radius * size[2]
    return size[0] * size[1] * size[2]


@dataclass(frozen=True, slots=True)
class ScenePrim:
    """One geometry prim, a static collider, or a rigid-body root."""

    path: str
    name: str
    category: str
    shape: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation_z_deg: float
    material: str | None
    room_id: str
    semantic: str
    collision: bool
    body_path: str | None = None
    is_body: bool = False
    relative_to_body: bool = False
    density_kg_m3: float = 0.0
    mass_kg: float | None = None
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.0
    movable: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.shape not in {"box", "cylinder", BODY_SHAPE}:
            raise ValueError(f"{self.path}: unsupported shape {self.shape!r}")
        if self.is_body and self.collision:
            raise ValueError(f"{self.path}: a rigid-body root must not carry its own collider")
        if self.relative_to_body and self.body_path is None:
            raise ValueError(f"{self.path}: relative_to_body requires body_path")
        if self.is_body and self.body_path != self.path:
            raise ValueError(f"{self.path}: a rigid-body root must set body_path to itself")

    @property
    def in_rigid_body(self) -> bool:
        return self.is_body or self.body_path is not None

    @property
    def is_cylinder(self) -> bool:
        return self.shape == "cylinder"

    @property
    def is_geometry(self) -> bool:
        return self.shape != BODY_SHAPE

    @property
    def physics_mode(self) -> str:
        return "dynamic" if self.in_rigid_body else "static"

    def computed_mass_kg(self) -> float:
        if self.mass_kg is not None:
            return self.mass_kg
        return _volume(self.shape, self.size) * self.density_kg_m3

    def as_manifest_entry(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "name": self.name,
            "category": self.category,
            "shape": self.shape,
            "center": [round(value, 6) for value in self.center],
            "rotation_z_deg": self.rotation_z_deg,
            "room_id": self.room_id,
            "semantic": self.semantic,
            "physics": self.physics_mode,
            "movable": self.movable,
            "tags": list(self.tags),
        }
        if self.is_geometry:
            payload["size"] = [round(value, 6) for value in self.size]
            payload["material"] = self.material
            payload["collision"] = self.collision
            payload["density_kg_m3"] = round(self.density_kg_m3, 3)
            payload["mass_kg"] = round(self.computed_mass_kg(), 4)
            payload["static_friction"] = self.static_friction
            payload["dynamic_friction"] = self.dynamic_friction
            payload["restitution"] = self.restitution
            payload["relative_to_body"] = self.relative_to_body
        if self.body_path is not None:
            payload["body_path"] = self.body_path
        if self.is_body:
            payload["mass_kg"] = round(self.computed_mass_kg(), 4)
        return payload


@dataclass(frozen=True, slots=True)
class LightSpec:
    """A light prim that makes the scene legible in the GUI."""

    path: str
    kind: str
    position: tuple[float, float, float]
    intensity: float
    color: tuple[float, float, float]
    radius_m: float = 0.25
    rotation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    room_id: str | None = None

    def as_manifest_entry(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "position": [round(value, 6) for value in self.position],
            "intensity": self.intensity,
            "color": list(self.color),
            "radius_m": self.radius_m,
            "rotation_euler_deg": list(self.rotation_euler_deg),
            "room_id": self.room_id,
        }


@dataclass(frozen=True, slots=True)
class ScenePlan:
    """A fully resolved scene: primitives, lights and reproducibility metadata."""

    scene_id: str
    frequency_hz: float
    seed: int
    description: str
    prims: tuple[ScenePrim, ...]
    lights: tuple[LightSpec, ...]
    materials: Mapping[str, Mapping[str, Any]]
    stats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def bodies(self) -> tuple[ScenePrim, ...]:
        return tuple(prim for prim in self.prims if prim.is_body)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "description": self.description,
            "seed": self.seed,
            "frequency_hz": self.frequency_hz,
            "generator": "sim2sense_fall.scene_planner",
            "materials": {name: dict(entry) for name, entry in self.materials.items()},
            "stats": dict(self.stats),
            "lights": [light.as_manifest_entry() for light in self.lights],
            "prims": [prim.as_manifest_entry() for prim in self.prims],
        }


def write_manifest(plan: ScenePlan, path: str | Path) -> Path:
    """Write the plan manifest as pretty JSON and return the written path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(plan.as_manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


# --------------------------------------------------------------------------------------
# Walls and openings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _WallSlab:
    """A solid piece of wall in wall-local coordinates: ``u`` along, ``z`` up."""

    u_center: float
    u_size: float
    z_center: float
    z_size: float


def _wall_axis(room: RoomSpec, face: str) -> tuple[float, float]:
    """Return the ``(start, end)`` of the wall's running axis in world coordinates."""

    origin_x, origin_y = room.origin
    width, depth = room.size
    if face in ("south", "north"):
        return origin_x, origin_x + width
    return origin_y, origin_y + depth


def _wall_v_center(room: RoomSpec, face: str) -> float:
    """Cross-axis centre of the wall slab, i.e. the line it is built on."""

    origin_x, origin_y = room.origin
    width, depth = room.size
    thickness = room.wall_thickness
    if face == "south":
        return origin_y + thickness / 2
    if face == "north":
        return origin_y + depth - thickness / 2
    if face == "west":
        return origin_x + thickness / 2
    return origin_x + width - thickness / 2


def _opening_center_u(room: RoomSpec, opening: OpeningSpec) -> float:
    start, _ = _wall_axis(room, opening.wall)
    return start + opening.offset


def _wall_slabs(room: RoomSpec, face: str) -> list[_WallSlab]:
    """Split a wall into solid slabs around its door and window openings."""

    start, end = _wall_axis(room, face)
    height = room.wall_height
    openings = sorted(
        (opening for opening in room.openings if opening.wall == face),
        key=lambda opening: opening.offset,
    )
    slabs: list[_WallSlab] = []
    cursor = start

    def solid(u0: float, u1: float, z0: float, z1: float) -> None:
        if u1 - u0 <= 1e-6 or z1 - z0 <= 1e-6:
            return
        slabs.append(
            _WallSlab(
                u_center=(u0 + u1) / 2,
                u_size=u1 - u0,
                z_center=(z0 + z1) / 2,
                z_size=z1 - z0,
            )
        )

    for opening in openings:
        center = _opening_center_u(room, opening)
        left = center - opening.width / 2
        right = center + opening.width / 2
        if left < start - 1e-6 or right > end + 1e-6:
            raise ValueError(
                f"{room.id}.{face}: opening at offset {opening.offset} does not fit "
                f"inside the wall span [{start}, {end}]"
            )
        solid(cursor, left, 0.0, height)
        solid(left, right, 0.0, opening.sill_height)
        solid(left, right, opening.sill_height + opening.height, height)
        cursor = max(cursor, right)
    solid(cursor, end, 0.0, height)
    return slabs


def _static_prim(
    *,
    path: str,
    name: str,
    category: str,
    shape: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    rotation_z_deg: float,
    material: str,
    room_id: str,
    semantic: str,
    physics_source: Mapping[str, float],
    tags: tuple[str, ...],
) -> ScenePrim:
    return ScenePrim(
        path=path,
        name=name,
        category=category,
        shape=shape,
        center=center,
        size=size,
        rotation_z_deg=rotation_z_deg,
        material=material,
        room_id=room_id,
        semantic=semantic,
        collision=True,
        density_kg_m3=physics_source["density_kg_m3"],
        static_friction=physics_source["static_friction"],
        dynamic_friction=physics_source["dynamic_friction"],
        restitution=physics_source["restitution"],
        movable=False,
        tags=tags,
    )


def _wall_prim(
    room: RoomSpec, face: str, index: int, slab: _WallSlab, physics_source: Mapping[str, float]
) -> ScenePrim:
    thickness = room.wall_thickness
    v_center = _wall_v_center(room, face)
    if face in ("south", "north"):
        center = (slab.u_center, v_center, slab.z_center)
        size = (slab.u_size, thickness, slab.z_size)
    else:
        center = (v_center, slab.u_center, slab.z_center)
        size = (thickness, slab.u_size, slab.z_size)
    return _static_prim(
        path=f"rooms/{_safe(room.id)}/walls/{face}_{index:02d}",
        name=f"{room.id} {face} wall segment {index}",
        category="wall",
        shape="box",
        center=center,
        size=size,
        rotation_z_deg=0.0,
        material=room.wall_material,
        room_id=room.id,
        semantic="wall",
        physics_source=physics_source,
        tags=(room.room_type, "envelope"),
    )


def _opening_prims(
    room: RoomSpec,
    opening: OpeningSpec,
    index: int,
    materials: Mapping[str, Any],
    wall_physics: Mapping[str, float],
) -> list[ScenePrim]:
    """Closed door leaves, glazing panes and window frames for one opening."""

    prims: list[ScenePrim] = []
    v_center = _wall_v_center(room, opening.wall)
    center_u = _opening_center_u(room, opening)
    along_x = opening.wall in ("south", "north")
    prefix = f"rooms/{_safe(room.id)}/openings/{opening.wall}"

    def make(
        suffix: str,
        u: float,
        v_thickness: float,
        z: float,
        u_size: float,
        z_size: float,
        material: str,
        semantic: str,
    ) -> ScenePrim:
        if along_x:
            center = (u, v_center, z)
            size = (u_size, v_thickness, z_size)
        else:
            center = (v_center, u, z)
            size = (v_thickness, u_size, z_size)
        material_entry = materials.get(material)
        source = dict(wall_physics)
        if material_entry is not None:
            source["density_kg_m3"] = material_entry.density_kg_m3
            source["static_friction"] = material_entry.static_friction
            source["dynamic_friction"] = material_entry.dynamic_friction
            source["restitution"] = material_entry.restitution
        return _static_prim(
            path=f"{prefix}_{suffix}_{index:02d}",
            name=f"{room.id} {opening.wall} {suffix} {index}",
            category="opening",
            shape="box",
            center=center,
            size=size,
            rotation_z_deg=0.0,
            material=material,
            room_id=room.id,
            semantic=semantic,
            physics_source=source,
            tags=(room.room_type, semantic),
        )

    if opening.is_door:
        if not opening.leaf:
            return prims
        prims.append(
            make(
                "door_leaf",
                center_u,
                DOOR_LEAF_THICKNESS_M,
                opening.height / 2,
                opening.width,
                opening.height,
                "wood_door",
                "door",
            )
        )
        return prims

    prims.append(
        make(
            "window_pane",
            center_u,
            GLASS_PANE_THICKNESS_M,
            opening.sill_height + opening.height / 2,
            opening.width,
            opening.height,
            "glass_window",
            "window",
        )
    )
    frame_half = WINDOW_FRAME_WIDTH_M / 2
    prims.append(
        make(
            "frame_sill",
            center_u,
            WINDOW_FRAME_DEPTH_M,
            opening.sill_height + frame_half,
            opening.width,
            WINDOW_FRAME_WIDTH_M,
            "painted_mdf",
            "window_frame",
        )
    )
    prims.append(
        make(
            "frame_head",
            center_u,
            WINDOW_FRAME_DEPTH_M,
            opening.sill_height + opening.height - frame_half,
            opening.width,
            WINDOW_FRAME_WIDTH_M,
            "painted_mdf",
            "window_frame",
        )
    )
    for side, sign in (("jamb_l", -1), ("jamb_r", 1)):
        prims.append(
            make(
                side,
                center_u + sign * (opening.width / 2 - frame_half),
                WINDOW_FRAME_DEPTH_M,
                opening.sill_height + opening.height / 2,
                WINDOW_FRAME_WIDTH_M,
                opening.height,
                "painted_mdf",
                "window_frame",
            )
        )
    return prims


# --------------------------------------------------------------------------------------
# Room shell and furniture
# --------------------------------------------------------------------------------------


def _slab_prim(
    room: RoomSpec,
    suffix: str,
    category: str,
    z_center: float,
    thickness: float,
    material: str,
    physics_source: Mapping[str, float],
    *,
    semantic: str,
) -> ScenePrim:
    origin_x, origin_y = room.origin
    width, depth = room.size
    return _static_prim(
        path=f"rooms/{_safe(room.id)}/{suffix}",
        name=f"{room.id} {category}",
        category=category,
        shape="box",
        center=(origin_x + width / 2, origin_y + depth / 2, z_center),
        size=(width, depth, thickness),
        rotation_z_deg=0.0,
        material=material,
        room_id=room.id,
        semantic=semantic,
        physics_source=physics_source,
        tags=(room.room_type, "shell"),
    )


def _resolve_parts(item: FurnitureSpec, materials: Mapping[str, Any]) -> list[Part]:
    size = item.size or furniture_default_size(item.kind)
    if size is None:
        raise ValueError(
            f"{item.id}: kind {item.kind!r} has no built-in recipe and no explicit 'size'"
        )
    parts = furniture_parts(item.kind, size, item.material or "wood_furniture")
    if item.material:
        dominant = max(parts, key=lambda part: _volume(part.shape, part.size)).material
        parts = [
            replace(part, material=item.material) if part.material == dominant else part
            for part in parts
        ]
    return parts


def _furniture_prims(
    room: RoomSpec, item: FurnitureSpec, materials: Mapping[str, Any]
) -> list[ScenePrim]:
    parts = _resolve_parts(item, materials)
    theta = math.radians(item.rotation_z_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    anchor_x = room.origin[0] + item.position[0]
    anchor_y = room.origin[1] + item.position[1]

    dynamic = item.physics == "dynamic"
    item_path = f"rooms/{_safe(room.id)}/furniture/{_safe(item.id)}"
    prims: list[ScenePrim] = []

    if dynamic:
        total_mass = 0.0
        total_volume = 0.0
        for part in parts:
            spec = materials.get(part.material) or materials["wood_furniture"]
            density = item.density_kg_m3 or spec.density_kg_m3
            volume = _volume(part.shape, part.size)
            total_mass += volume * density
            total_volume += volume
        prims.append(
            ScenePrim(
                path=item_path,
                name=f"{room.id} {item.id} body",
                category="furniture_body",
                shape=BODY_SHAPE,
                center=(anchor_x, anchor_y, 0.0),
                size=(0.0, 0.0, 0.0),
                rotation_z_deg=item.rotation_z_deg,
                material=None,
                room_id=room.id,
                semantic=item.kind,
                collision=False,
                body_path=item_path,
                is_body=True,
                density_kg_m3=(round(total_mass / total_volume, 3) if total_volume else 0.0),
                mass_kg=round(total_mass, 6),
                static_friction=item.static_friction if item.static_friction is not None else 0.5,
                dynamic_friction=(
                    item.dynamic_friction if item.dynamic_friction is not None else 0.4
                ),
                restitution=item.restitution if item.restitution is not None else 0.0,
                movable=item.movable,
                tags=item.tags,
            )
        )

    for index, part in enumerate(parts):
        spec = materials.get(part.material) or materials["wood_furniture"]
        density = item.density_kg_m3 or spec.density_kg_m3
        local_x, local_y, local_z = part.center
        local_z += item.elevation_m
        if dynamic:
            center = (local_x, local_y, local_z)
        else:
            center = (
                anchor_x + local_x * cos_t - local_y * sin_t,
                anchor_y + local_x * sin_t + local_y * cos_t,
                local_z,
            )
        prims.append(
            ScenePrim(
                path=f"{item_path}/p{index:02d}_{_safe(part.suffix)}",
                name=f"{room.id} {item.id} {part.suffix}",
                category="furniture" if item.kind != "ceiling_lamp" else "lighting_fixture",
                shape=part.shape,
                center=center,
                size=part.size,
                rotation_z_deg=0.0 if dynamic else item.rotation_z_deg,
                material=part.material,
                room_id=room.id,
                semantic=item.kind,
                collision=True,
                body_path=item_path if dynamic else None,
                relative_to_body=dynamic,
                density_kg_m3=density,
                static_friction=(
                    item.static_friction
                    if item.static_friction is not None
                    else spec.static_friction
                ),
                dynamic_friction=(
                    item.dynamic_friction
                    if item.dynamic_friction is not None
                    else spec.dynamic_friction
                ),
                restitution=(
                    item.restitution if item.restitution is not None else spec.restitution
                ),
                movable=item.movable,
                tags=item.tags,
            )
        )
    return prims


def _foundation_prim(spec: SceneSpec) -> ScenePrim:
    x_min, y_min, x_max, y_max = spec.footprint
    margin = spec.foundation_margin_m
    thickness = 0.30
    material_name = "concrete_slab" if "concrete_slab" in spec.materials else "wood_floor"
    material = spec.materials[material_name]
    return _static_prim(
        path="site/foundation",
        name="foundation slab",
        category="ground",
        shape="box",
        center=((x_min + x_max) / 2, (y_min + y_max) / 2, -thickness / 2),
        size=(x_max - x_min + 2 * margin, y_max - y_min + 2 * margin, thickness),
        rotation_z_deg=0.0,
        material=material_name,
        room_id="site",
        semantic="ground",
        physics_source={
            "density_kg_m3": material.density_kg_m3,
            "static_friction": material.static_friction,
            "dynamic_friction": material.dynamic_friction,
            "restitution": material.restitution,
        },
        tags=("site",),
    )


def _physics_of(material: Any) -> dict[str, float]:
    return {
        "density_kg_m3": material.density_kg_m3,
        "static_friction": material.static_friction,
        "dynamic_friction": material.dynamic_friction,
        "restitution": material.restitution,
    }


def _room_lights(room: RoomSpec) -> list[LightSpec]:
    origin_x, origin_y = room.origin
    width, depth = room.size
    return [
        LightSpec(
            path=f"lighting/rooms/{_safe(room.id)}",
            kind="sphere",
            position=(origin_x + width / 2, origin_y + depth / 2, room.wall_height - 0.16),
            intensity=6000.0,
            color=(1.0, 0.96, 0.90),
            radius_m=0.30,
            room_id=room.id,
        )
    ]


def plan_scene(spec: SceneSpec) -> ScenePlan:
    """Resolve ``spec`` into primitives, lights and statistics."""

    prims: list[ScenePrim] = []
    lights: list[LightSpec] = []

    if spec.foundation:
        prims.append(_foundation_prim(spec))

    for room in spec.rooms:
        prims.append(
            _slab_prim(
                room,
                "floor",
                "floor",
                -room.floor_thickness / 2,
                room.floor_thickness,
                room.floor_material,
                _physics_of(spec.materials[room.floor_material]),
                semantic="floor",
            )
        )
        if room.ceiling_enabled:
            prims.append(
                _slab_prim(
                    room,
                    "ceiling",
                    "ceiling",
                    room.wall_height + room.ceiling_thickness / 2,
                    room.ceiling_thickness,
                    room.ceiling_material,
                    _physics_of(spec.materials[room.ceiling_material]),
                    semantic="ceiling",
                )
            )
        wall_physics = _physics_of(spec.materials[room.wall_material])
        for face in room.walls:
            for index, slab in enumerate(_wall_slabs(room, face)):
                prims.append(_wall_prim(room, face, index, slab, wall_physics))
        for index, opening in enumerate(room.openings):
            prims.extend(_opening_prims(room, opening, index, spec.materials, wall_physics))
        for item in room.furniture:
            prims.extend(_furniture_prims(room, item, spec.materials))
        lights.extend(_room_lights(room))

    _add_global_lights(lights, spec)
    used = sorted({prim.material for prim in prims if prim.material is not None})
    return ScenePlan(
        scene_id=spec.scene_id,
        frequency_hz=spec.frequency_hz,
        seed=spec.seed,
        description=spec.description,
        prims=tuple(prims),
        lights=tuple(lights),
        materials={
            name: dict(spec.materials[name].as_manifest_entry(spec.frequency_hz)) for name in used
        },
        stats=_accumulate_stats(prims, lights, spec),
    )


def _add_global_lights(lights: list[LightSpec], spec: SceneSpec) -> None:
    x_min, y_min, x_max, y_max = spec.footprint
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    lights.insert(
        0,
        LightSpec(
            path="lighting/DomeLight",
            kind="dome",
            position=(mid_x, mid_y, max(room.wall_height for room in spec.rooms) + 2.0),
            intensity=1000.0,
            color=(0.82, 0.86, 0.95),
        ),
    )
    lights.insert(
        1,
        LightSpec(
            path="lighting/SunLight",
            kind="distant",
            position=(mid_x, mid_y, 12.0),
            intensity=1500.0,
            color=(1.0, 0.97, 0.90),
            rotation_euler_deg=(45.0, 0.0, 135.0),
        ),
    )


def _accumulate_stats(
    prims: list[ScenePrim], lights: list[LightSpec], spec: SceneSpec
) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    by_room: dict[str, int] = {}
    wall_area_by_room: dict[str, float] = {}
    material_usage: dict[str, int] = {}
    body_count = 0
    body_mass = 0.0

    for prim in prims:
        by_category[prim.category] = by_category.get(prim.category, 0) + 1
        by_room[prim.room_id] = by_room.get(prim.room_id, 0) + 1
        if prim.material:
            material_usage[prim.material] = material_usage.get(prim.material, 0) + 1
        if prim.category == "wall":
            wall_area_by_room[prim.room_id] = wall_area_by_room.get(prim.room_id, 0.0) + (
                prim.size[0] * prim.size[2]
            )
        if prim.is_body:
            body_count += 1
            body_mass += prim.computed_mass_kg()

    floor_area_by_room = {room.id: room.size[0] * room.size[1] for room in spec.rooms}
    x_min, y_min, x_max, y_max = spec.footprint
    return {
        "prim_count": len(prims),
        "geometry_count": sum(1 for prim in prims if prim.is_geometry),
        "light_count": len(lights),
        "room_count": len(spec.rooms),
        "primitives_by_category": dict(sorted(by_category.items())),
        "primitives_by_room": dict(sorted(by_room.items())),
        "material_usage": dict(sorted(material_usage.items())),
        "floor_area_m2_by_room": {
            key: round(value, 4) for key, value in sorted(floor_area_by_room.items())
        },
        "total_floor_area_m2": round(sum(floor_area_by_room.values()), 4),
        "wall_area_m2_by_room": {
            key: round(value, 4) for key, value in sorted(wall_area_by_room.items())
        },
        "total_wall_area_m2": round(sum(wall_area_by_room.values()), 4),
        "footprint_m": [round(x_max - x_min, 4), round(y_max - y_min, 4)],
        "rigid_body_count": body_count,
        "rigid_body_mass_kg": round(body_mass, 3),
    }
