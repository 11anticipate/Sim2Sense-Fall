"""CPU geometry checks for yawed boxes and vertical circular cylinders.

Checks use the resolved parts, including compound-body transforms and protruding
handles. Touching surfaces are allowed; positive-volume furniture/wall overlap
is not. This is a construction check, not a robot navigation or clearance test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .numbers import finite_number

if TYPE_CHECKING:
    from .planner import ScenePlan, ScenePrim
    from .spec import SceneSpec

GEOMETRY_TOLERANCE_M = 1e-6


@dataclass(frozen=True, slots=True)
class WorldShape:
    path: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_deg: float
    cylinder: bool

    @property
    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        theta = math.radians(self.yaw_deg)
        c, s = math.cos(theta), math.sin(theta)
        return ((c, s), (-s, c))

    def radius_on(self, axis: tuple[float, float]) -> float:
        if self.cylinder:
            return self.size[0] / 2
        return sum(
            abs(axis[0] * v[0] + axis[1] * v[1]) * self.size[i] / 2 for i, v in enumerate(self.axes)
        )

    @property
    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        radii = (self.radius_on((1, 0)), self.radius_on((0, 1)), self.size[2] / 2)
        return (
            tuple(c - r for c, r in zip(self.center, radii, strict=True)),
            tuple(c + r for c, r in zip(self.center, radii, strict=True)),
        )


def world_shapes(prims: tuple[ScenePrim, ...]) -> dict[str, WorldShape]:
    """Resolve each primitive into world space, including dynamic furniture."""
    bodies = {prim.path: prim for prim in prims if prim.is_body}
    shapes: dict[str, WorldShape] = {}
    for prim in prims:
        if not prim.is_geometry:
            continue
        center, yaw = prim.center, prim.rotation_z_deg
        if prim.relative_to_body:
            if prim.body_path not in bodies:
                raise ValueError(f"{prim.path}: missing body {prim.body_path}")
            body = bodies[prim.body_path]
            theta = math.radians(body.rotation_z_deg)
            c, s = math.cos(theta), math.sin(theta)
            x, y, z = center
            center = (
                body.center[0] + c * x - s * y,
                body.center[1] + s * x + c * y,
                body.center[2] + z,
            )
            yaw += body.rotation_z_deg
        shapes[prim.path] = WorldShape(prim.path, center, prim.size, yaw, prim.is_cylinder)
    return shapes


def intersects(left: WorldShape, right: WorldShape, *, tolerance: float) -> bool:
    """Test positive-volume intersection, rather than overlapping AABBs."""
    if (left.size[2] + right.size[2]) / 2 - abs(left.center[2] - right.center[2]) <= tolerance:
        return False
    dx, dy = right.center[0] - left.center[0], right.center[1] - left.center[1]
    if left.cylinder and right.cylinder:
        return math.hypot(dx, dy) < (left.size[0] + right.size[0]) / 2 - tolerance
    if left.cylinder or right.cylinder:
        circle, box = (left, right) if left.cylinder else (right, left)
        dx, dy = circle.center[0] - box.center[0], circle.center[1] - box.center[1]
        distance = [
            max(abs(dx * v[0] + dy * v[1]) - box.size[i] / 2, 0) for i, v in enumerate(box.axes)
        ]
        return math.hypot(*distance) < circle.size[0] / 2 - tolerance
    for axis in (*left.axes, *right.axes):
        separation = abs(dx * axis[0] + dy * axis[1])
        if left.radius_on(axis) + right.radius_on(axis) - separation <= tolerance:
            return False
    return True


def validate_layout(
    plan: ScenePlan, spec: SceneSpec, *, tolerance: float = GEOMETRY_TOLERANCE_M
) -> None:
    """Reject furniture outside its room, wall penetrations and overlapping slabs."""
    if finite_number(tolerance, "geometry tolerance") < 0:
        raise ValueError("geometry tolerance must be non-negative")
    shapes = world_shapes(plan.prims)
    walls = [shapes[p.path] for p in plan.prims if p.category == "wall"]
    slabs = [shapes[p.path] for p in plan.prims if p.category in {"floor", "ground"}]
    for i, left in enumerate(slabs):
        for right in slabs[i + 1 :]:
            if intersects(left, right, tolerance=tolerance):
                raise ValueError(f"overlapping floor/support slabs: {left.path} and {right.path}")
    for prim in plan.prims:
        if prim.category not in {"furniture", "lighting_fixture"}:
            continue
        shape = shapes[prim.path]
        low, high = shape.bounds
        room = spec.room(prim.room_id)
        x0, y0, x1, y1 = room.bounds
        if (
            low[0] < x0 - tolerance
            or low[1] < y0 - tolerance
            or low[2] < -tolerance
            or high[0] > x1 + tolerance
            or high[1] > y1 + tolerance
            or high[2] > room.wall_height + tolerance
        ):
            raise ValueError(f"{prim.path}: furniture outside room {room.id}: {low} to {high}")
        for wall in walls:
            if intersects(shape, wall, tolerance=tolerance):
                raise ValueError(f"{prim.path}: furniture intersects wall {wall.path}")
