"""Declarative schema for indoor scenes.

A scene is described in metres on a right-handed Z-up frame. Rooms are axis
aligned rectangles in the plan view and are placed by their lower-left corner, so
the same numbers can be read directly off an architectural drawing.

The plan view uses ``x`` to the right and ``y`` upwards::

        north  (y = origin_y + depth)
          +------------------------------+
          |                              |
    west  |            room              |  east
          |                              |  (x = origin_x + width)
          +------------------------------+
        south  (y = origin_y)

Wall faces are named after the compass direction they occupy, and an opening
``offset`` is measured along the wall axis from its lower-coordinate end (from
``origin_x`` for the south/north walls, from ``origin_y`` for the west/east
walls).

Loading is deliberately strict: unknown keys, out-of-range numbers, overlapping
rooms and openings that do not fit inside their wall all raise immediately, so a
broken scene file never reaches the simulator as a silently wrong apartment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .scene_materials import MaterialSpec, material_library_from_config

__all__ = [
    "FurnitureSpec",
    "OpeningSpec",
    "RoomSpec",
    "SceneSpec",
    "load_scene_spec",
    "scene_spec_from_mapping",
]

WALL_FACES: tuple[str, ...] = ("south", "north", "west", "east")
OPENING_KINDS: tuple[str, ...] = ("door", "window")
PHYSICS_MODES: tuple[str, ...] = ("static", "dynamic")


def _as_float(mapping: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    if key not in mapping:
        if default is None:
            raise ValueError(f"missing required numeric field {key!r}")
        return default
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        hint = ""
        if isinstance(value, str):
            hint = (
                " (YAML 1.1 requires a signed exponent for floats, so write 2.4e+9 "
                "rather than 2.4e9)"
            )
        raise ValueError(f"{key!r} must be a number, got {value!r}{hint}")
    return float(value)


def _as_pair(mapping: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = mapping.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key!r} must be a list of two numbers, got {value!r}")
    return (float(value[0]), float(value[1]))


def _as_triple(mapping: Mapping[str, Any], key: str) -> tuple[float, float, float]:
    value = mapping.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key!r} must be a list of three numbers, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{where}: unsupported keys {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class OpeningSpec:
    """A door or window cut into one wall face."""

    wall: str
    kind: str
    width: float
    height: float
    offset: float
    sill_height: float = 0.0
    leaf: bool = True

    def __post_init__(self) -> None:
        if self.wall not in WALL_FACES:
            raise ValueError(f"opening wall must be one of {WALL_FACES}, got {self.wall!r}")
        if self.kind not in OPENING_KINDS:
            raise ValueError(f"opening kind must be one of {OPENING_KINDS}, got {self.kind!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("opening width and height must be positive")
        if self.offset < 0:
            raise ValueError("opening offset must be non-negative")
        if self.sill_height < 0:
            raise ValueError("opening sill_height must be non-negative")
        if self.kind == "door" and self.sill_height != 0.0:
            raise ValueError("doors must start at floor level (sill_height = 0)")

    @property
    def is_door(self) -> bool:
        return self.kind == "door"


@dataclass(frozen=True, slots=True)
class FurnitureSpec:
    """One furnishing or fixture, placed in room-local coordinates."""

    id: str
    kind: str
    position: tuple[float, float]
    rotation_z_deg: float = 0.0
    size: tuple[float, float, float] | None = None
    material: str | None = None
    physics: str = "static"
    density_kg_m3: float | None = None
    elevation_m: float = 0.0
    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None
    movable: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("furniture id must be non-empty")
        if not self.kind.strip():
            raise ValueError(f"{self.id}: furniture kind must be non-empty")
        if self.physics not in PHYSICS_MODES:
            raise ValueError(f"{self.id}: physics must be one of {PHYSICS_MODES}")
        if self.size is not None and any(extent <= 0 for extent in self.size):
            raise ValueError(f"{self.id}: size extents must be positive")
        if self.elevation_m < 0:
            raise ValueError(f"{self.id}: elevation_m must be non-negative")
        if self.density_kg_m3 is not None and self.density_kg_m3 <= 0:
            raise ValueError(f"{self.id}: density_kg_m3 must be positive")
        for label, value in (
            ("static_friction", self.static_friction),
            ("dynamic_friction", self.dynamic_friction),
            ("restitution", self.restitution),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.id}: {label} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class RoomSpec:
    """An axis-aligned room footprint with its surfaces and contents."""

    id: str
    room_type: str
    origin: tuple[float, float]
    size: tuple[float, float]
    wall_material: str
    floor_material: str
    ceiling_material: str
    wall_height: float
    wall_thickness: float
    floor_thickness: float
    ceiling_thickness: float
    walls: tuple[str, ...] = WALL_FACES
    openings: tuple[OpeningSpec, ...] = ()
    furniture: tuple[FurnitureSpec, ...] = ()
    ceiling_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("room id must be non-empty")
        if any(extent <= 0 for extent in self.size):
            raise ValueError(f"{self.id}: room size must be positive, got {self.size!r}")
        if self.wall_height <= 0 or self.wall_thickness <= 0:
            raise ValueError(f"{self.id}: wall height and thickness must be positive")
        if self.floor_thickness <= 0 or self.ceiling_thickness <= 0:
            raise ValueError(f"{self.id}: floor and ceiling thickness must be positive")
        unknown_faces = set(self.walls) - set(WALL_FACES)
        if unknown_faces:
            raise ValueError(f"{self.id}: unsupported wall faces {sorted(unknown_faces)}")
        if len(set(self.walls)) != len(self.walls):
            raise ValueError(f"{self.id}: duplicate wall faces in {self.walls!r}")
        for opening in self.openings:
            if opening.wall not in self.walls:
                raise ValueError(
                    f"{self.id}: opening on {opening.wall!r} but that wall is not built; "
                    "add it to 'walls' or remove the opening"
                )
            if opening.sill_height + opening.height > self.wall_height:
                raise ValueError(
                    f"{self.id}: opening on {opening.wall!r} exceeds wall height "
                    f"({opening.sill_height + opening.height} > {self.wall_height})"
                )
            length = self.wall_length(opening.wall)
            if opening.offset - opening.width / 2 < -1e-9:
                raise ValueError(
                    f"{self.id}: opening on {opening.wall!r} starts before the wall "
                    f"(offset {opening.offset}, width {opening.width})"
                )
            if opening.offset + opening.width / 2 > length + 1e-9:
                raise ValueError(
                    f"{self.id}: opening on {opening.wall!r} does not fit inside the "
                    f"{length} m wall (offset {opening.offset}, width {opening.width})"
                )
        furniture_ids = [item.id for item in self.furniture]
        if len(set(furniture_ids)) != len(furniture_ids):
            raise ValueError(f"{self.id}: duplicate furniture ids in {sorted(furniture_ids)}")

    def wall_length(self, face: str) -> float:
        """Length of a wall face along its own axis."""

        if face not in WALL_FACES:
            raise ValueError(f"unknown wall face {face!r}")
        return self.size[0] if face in ("south", "north") else self.size[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(x_min, y_min, x_max, y_max)`` in world metres."""

        return (
            self.origin[0],
            self.origin[1],
            self.origin[0] + self.size[0],
            self.origin[1] + self.size[1],
        )


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """A complete indoor scene ready to be planned and exported."""

    scene_id: str
    rooms: tuple[RoomSpec, ...]
    materials: Mapping[str, MaterialSpec]
    frequency_hz: float
    seed: int
    description: str = ""
    foundation: bool = True
    foundation_margin_m: float = 0.5
    source_path: Path | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not self.rooms:
            raise ValueError("scene must contain at least one room")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        room_ids = [room.id for room in self.rooms]
        if len(set(room_ids)) != len(room_ids):
            raise ValueError(f"duplicate room ids in {sorted(room_ids)}")
        known_materials = set(self.materials)
        for room in self.rooms:
            for label, name in (
                ("wall_material", room.wall_material),
                ("floor_material", room.floor_material),
                ("ceiling_material", room.ceiling_material),
            ):
                if name not in known_materials:
                    raise ValueError(f"{room.id}.{label}: unknown material {name!r}")
            for item in room.furniture:
                if item.material is not None and item.material not in known_materials:
                    raise ValueError(f"{room.id}.{item.id}: unknown material {item.material!r}")
        self._check_no_overlap()

    def _check_no_overlap(self) -> None:
        bounds = [(room.id, room.bounds) for room in self.rooms]
        for index, (left_id, left) in enumerate(bounds):
            for right_id, right in bounds[index + 1 :]:
                if self._overlaps(left, right):
                    raise ValueError(
                        f"rooms {left_id!r} and {right_id!r} overlap; {left} vs {right}"
                    )

    @staticmethod
    def _overlaps(left: Sequence[float], right: Sequence[float]) -> bool:
        # Touching edges are allowed: neighbouring rooms legitimately share a wall.
        return (
            left[0] < right[2] - 1e-9
            and right[0] < left[2] - 1e-9
            and left[1] < right[3] - 1e-9
            and right[1] < left[3] - 1e-9
        )

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Overall bounds ``(x_min, y_min, x_max, y_max)`` of every room."""

        xs_min = min(room.bounds[0] for room in self.rooms)
        ys_min = min(room.bounds[1] for room in self.rooms)
        xs_max = max(room.bounds[2] for room in self.rooms)
        ys_max = max(room.bounds[3] for room in self.rooms)
        return (xs_min, ys_min, xs_max, ys_max)

    def room(self, room_id: str) -> RoomSpec:
        for room in self.rooms:
            if room.id == room_id:
                return room
        raise KeyError(f"unknown room {room_id!r}")


def scene_spec_from_mapping(
    payload: Mapping[str, Any], *, source_path: Path | None = None
) -> SceneSpec:
    """Build a :class:`SceneSpec` from a parsed YAML mapping."""

    _reject_unknown(
        payload,
        {
            "scene_id",
            "description",
            "frequency_hz",
            "seed",
            "foundation",
            "foundation_margin_m",
            "materials",
            "rooms",
        },
        where="scene",
    )
    materials = material_library_from_config(payload.get("materials"))
    rooms_payload = payload.get("rooms")
    if not isinstance(rooms_payload, list) or not rooms_payload:
        raise ValueError("scene must define a non-empty 'rooms' list")
    rooms = tuple(_room_from_mapping(entry, materials) for entry in rooms_payload)
    scene_id = payload.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a non-empty string")
    return SceneSpec(
        scene_id=scene_id,
        description=str(payload.get("description", "")),
        rooms=rooms,
        materials=materials,
        frequency_hz=_as_float(payload, "frequency_hz", default=2.4e9),
        seed=int(_as_float(payload, "seed", default=0.0)),
        foundation=bool(payload.get("foundation", True)),
        foundation_margin_m=_as_float(payload, "foundation_margin_m", default=0.5),
        source_path=source_path,
    )


def _room_from_mapping(
    payload: Mapping[str, Any], materials: Mapping[str, MaterialSpec]
) -> RoomSpec:
    _reject_unknown(
        payload,
        {
            "id",
            "type",
            "room_type",
            "origin",
            "size",
            "wall_material",
            "floor_material",
            "ceiling_material",
            "wall_height",
            "wall_thickness",
            "floor_thickness",
            "ceiling_thickness",
            "walls",
            "openings",
            "furniture",
            "ceiling",
        },
        where="room",
    )
    room_id = payload.get("id")
    if not isinstance(room_id, str) or not room_id.strip():
        raise ValueError("every room needs a non-empty 'id'")
    room_type = payload.get("room_type", payload.get("type"))
    if not isinstance(room_type, str) or not room_type.strip():
        raise ValueError(f"{room_id}: 'room_type' must be a non-empty string")
    walls_payload = payload.get("walls", list(WALL_FACES))
    if not isinstance(walls_payload, list):
        raise ValueError(f"{room_id}: 'walls' must be a list of wall faces")
    walls = tuple(str(face) for face in walls_payload)
    openings_payload = payload.get("openings", [])
    if not isinstance(openings_payload, list):
        raise ValueError(f"{room_id}: 'openings' must be a list")
    furniture_payload = payload.get("furniture", [])
    if not isinstance(furniture_payload, list):
        raise ValueError(f"{room_id}: 'furniture' must be a list")
    return RoomSpec(
        id=room_id,
        room_type=room_type,
        origin=_as_pair(payload, "origin"),
        size=_as_pair(payload, "size"),
        wall_material=_material_name(payload, "wall_material", materials, room_id),
        floor_material=_material_name(payload, "floor_material", materials, room_id),
        ceiling_material=_material_name(payload, "ceiling_material", materials, room_id),
        wall_height=_as_float(payload, "wall_height", default=2.7),
        wall_thickness=_as_float(payload, "wall_thickness", default=0.12),
        floor_thickness=_as_float(payload, "floor_thickness", default=0.12),
        ceiling_thickness=_as_float(payload, "ceiling_thickness", default=0.15),
        walls=walls,
        openings=tuple(_opening_from_mapping(entry, room_id) for entry in openings_payload),
        furniture=tuple(_furniture_from_mapping(entry, room_id) for entry in furniture_payload),
        ceiling_enabled=bool(payload.get("ceiling", True)),
    )


def _material_name(
    payload: Mapping[str, Any], key: str, materials: Mapping[str, MaterialSpec], room_id: str
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{room_id}: '{key}' must name a material")
    if value not in materials:
        known = ", ".join(sorted(materials))
        raise ValueError(f"{room_id}.{key}: unknown material {value!r}; known: {known}")
    return value


def _opening_from_mapping(payload: Mapping[str, Any], room_id: str) -> OpeningSpec:
    _reject_unknown(
        payload,
        {"wall", "kind", "type", "width", "height", "offset", "sill_height", "leaf"},
        where=f"room {room_id} opening",
    )
    kind = payload.get("kind", payload.get("type"))
    if not isinstance(kind, str):
        raise ValueError(f"{room_id}: opening needs a 'kind' of {OPENING_KINDS}")
    wall = payload.get("wall")
    if not isinstance(wall, str):
        raise ValueError(f"{room_id}: opening needs a 'wall' face")
    return OpeningSpec(
        wall=wall,
        kind=kind,
        width=_as_float(payload, "width"),
        height=_as_float(payload, "height"),
        offset=_as_float(payload, "offset"),
        sill_height=_as_float(payload, "sill_height", default=0.0),
        leaf=bool(payload.get("leaf", True)),
    )


def _furniture_from_mapping(payload: Mapping[str, Any], room_id: str) -> FurnitureSpec:
    _reject_unknown(
        payload,
        {
            "id",
            "kind",
            "type",
            "position",
            "rotation_z_deg",
            "size",
            "material",
            "physics",
            "density_kg_m3",
            "elevation_m",
            "static_friction",
            "dynamic_friction",
            "restitution",
            "movable",
            "tags",
        },
        where=f"room {room_id} furniture",
    )
    item_id = payload.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError(f"{room_id}: every furniture item needs a non-empty 'id'")
    kind = payload.get("kind", payload.get("type"))
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{room_id}.{item_id}: furniture needs a non-empty 'kind'")
    material = payload.get("material")
    if material is not None and not isinstance(material, str):
        raise ValueError(f"{room_id}.{item_id}: 'material' must be a string or omitted")
    size = _as_triple(payload, "size") if "size" in payload else None
    tags_payload = payload.get("tags", [])
    if not isinstance(tags_payload, list):
        raise ValueError(f"{room_id}.{item_id}: 'tags' must be a list")
    physics = payload.get("physics", "static")
    if not isinstance(physics, str):
        raise ValueError(f"{room_id}.{item_id}: 'physics' must be a string")
    density = payload.get("density_kg_m3")
    return FurnitureSpec(
        id=item_id,
        kind=kind,
        position=_as_pair(payload, "position"),
        rotation_z_deg=_as_float(payload, "rotation_z_deg", default=0.0),
        size=size,
        material=material,
        physics=physics,
        density_kg_m3=None if density is None else float(density),
        elevation_m=_as_float(payload, "elevation_m", default=0.0),
        static_friction=None
        if "static_friction" not in payload
        else _as_float(payload, "static_friction"),
        dynamic_friction=(
            None if "dynamic_friction" not in payload else _as_float(payload, "dynamic_friction")
        ),
        restitution=None if "restitution" not in payload else _as_float(payload, "restitution"),
        movable=bool(payload.get("movable", physics == "dynamic")),
        tags=tuple(str(tag) for tag in tags_payload),
    )


def load_scene_spec(path: str | Path) -> SceneSpec:
    """Load and validate a scene YAML file."""

    scene_path = Path(path)
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene config not found: {scene_path}")
    with scene_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{scene_path}: top level must be a mapping")
    return scene_spec_from_mapping(payload, source_path=scene_path)
