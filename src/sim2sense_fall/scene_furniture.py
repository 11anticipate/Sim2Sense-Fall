"""Parametric furniture built from primitives.

The project deliberately avoids downloading vendor prop libraries: a scene that
depends on a network catalogue is neither reproducible nor reviewable, and the
radio-propagation stage only cares about coarse geometry, thickness and material.
Every piece of furniture is therefore assembled from boxes and cylinders with
dimensions in metres, which keeps the scene fully offline and diff-friendly.

Recipes are expressed in a **furniture-local frame**:

* ``x`` runs along the item's width, ``y`` along its depth, ``z`` upwards from the
  item's own base;
* the footprint is centred on the origin, so it spans
  ``[-w/2, w/2] x [-d/2, d/2]``;
* ``+y`` is the "front" of an item (the side a person faces or opens).

A recipe receives the resolved overall bounding size ``(w, d, h)`` and returns the
list of parts that make it up. Rotating and placing the item is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_FURNITURE_SIZE",
    "Part",
    "furniture_default_size",
    "furniture_parts",
    "known_furniture_kinds",
]


@dataclass(frozen=True, slots=True)
class Part:
    """One primitive of a furniture item, in furniture-local coordinates.

    ``size`` is the full extent for a box and ``(diameter, diameter, height)`` for
    a cylinder, matching what the USD authoring step needs.
    """

    suffix: str
    shape: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str


def _box(
    suffix: str,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    material: str,
) -> Part:
    return Part(suffix, "box", (cx, cy, cz), (sx, sy, sz), material)


def _cyl(
    suffix: str,
    cx: float,
    cy: float,
    cz: float,
    diameter: float,
    height: float,
    material: str,
) -> Part:
    return Part(suffix, "cylinder", (cx, cy, cz), (diameter, diameter, height), material)


#: Overall bounding size ``(width, depth, height)`` used when a scene omits ``size``.
DEFAULT_FURNITURE_SIZE: dict[str, tuple[float, float, float]] = {
    "bed": (1.60, 2.05, 0.90),
    "bunk_bed": (1.00, 2.00, 1.60),
    "nightstand": (0.50, 0.42, 0.55),
    "wardrobe": (1.20, 0.60, 2.00),
    "dresser": (1.00, 0.45, 0.85),
    "rug": (2.00, 3.00, 0.012),
    "sofa": (2.10, 0.90, 0.85),
    "armchair": (0.85, 0.85, 0.80),
    "coffee_table": (1.10, 0.60, 0.45),
    "tv": (1.20, 0.08, 0.70),
    "tv_stand": (1.60, 0.40, 0.50),
    "bookshelf": (0.90, 0.35, 1.80),
    "dining_table": (1.40, 0.90, 0.75),
    "chair": (0.46, 0.48, 0.90),
    "toilet": (0.40, 0.72, 0.78),
    "vanity_sink": (0.90, 0.50, 0.88),
    "bathtub": (1.70, 0.75, 0.58),
    "shower": (1.00, 1.00, 2.05),
    "mirror": (0.70, 0.03, 0.80),
    "towel_rail": (0.60, 0.10, 0.06),
    "shoe_cabinet": (0.90, 0.35, 1.00),
    "coat_rack": (0.40, 0.40, 1.75),
    "kitchen_counter": (2.40, 0.60, 0.92),
    "kitchen_sink": (0.60, 0.46, 0.08),
    "stove": (0.60, 0.60, 0.07),
    "refrigerator": (0.70, 0.70, 1.85),
    "ceiling_lamp": (0.42, 0.42, 0.16),
}


def known_furniture_kinds() -> tuple[str, ...]:
    """Every furniture kind with a built-in recipe."""

    return tuple(sorted(_RECIPES))


def furniture_default_size(kind: str) -> tuple[float, float, float] | None:
    """Default bounding size for ``kind``, or ``None`` when unknown."""

    return DEFAULT_FURNITURE_SIZE.get(kind)


def furniture_parts(kind: str, size: tuple[float, float, float], material: str) -> list[Part]:
    """Return the parts of ``kind`` at the given overall ``size``.

    ``material`` is the fallback material for parts that do not override it. An
    unknown kind degrades to a single solid box rather than failing, so a new
    item can be added to a scene file before its recipe exists.
    """

    recipe = _RECIPES.get(kind)
    if recipe is None:
        return [_box("body", 0.0, 0.0, size[2] / 2, size[0], size[1], size[2], material)]
    return recipe(size[0], size[1], size[2], material)


# --------------------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------------------


def _bed(w: float, d: float, h: float, mat: str) -> list[Part]:
    frame_h = 0.30
    mattress_t = 0.22
    top = frame_h + mattress_t
    return [
        _box("frame", 0.0, 0.0, frame_h / 2, w, d, frame_h, "wood_furniture"),
        _box("headboard", 0.0, -(d / 2) + 0.04, h / 2, w, 0.08, h, "wood_furniture"),
        _box(
            "mattress",
            0.0,
            0.03,
            frame_h + mattress_t / 2,
            w - 0.08,
            d - 0.16,
            mattress_t,
            "upholstery",
        ),
        _box(
            "pillow_l",
            -(w / 4),
            -(d / 2) + 0.36,
            top + 0.05,
            w / 2 - 0.14,
            0.30,
            0.10,
            "upholstery",
        ),
        _box(
            "pillow_r", w / 4, -(d / 2) + 0.36, top + 0.05, w / 2 - 0.14, 0.30, 0.10, "upholstery"
        ),
        _box("duvet", 0.0, d * 0.12, top + 0.03, w - 0.06, d * 0.62, 0.06, "upholstery"),
    ]


def _bunk_bed(w: float, d: float, h: float, mat: str) -> list[Part]:
    post = 0.07
    parts: list[Part] = [
        _box(
            "post_sw",
            -(w / 2) + post / 2,
            -(d / 2) + post / 2,
            h / 2,
            post,
            post,
            h,
            "wood_furniture",
        ),
        _box(
            "post_se", w / 2 - post / 2, -(d / 2) + post / 2, h / 2, post, post, h, "wood_furniture"
        ),
        _box(
            "post_nw", -(w / 2) + post / 2, d / 2 - post / 2, h / 2, post, post, h, "wood_furniture"
        ),
        _box("post_ne", w / 2 - post / 2, d / 2 - post / 2, h / 2, post, post, h, "wood_furniture"),
    ]
    for index, level in enumerate((0.35, h - 0.25)):
        parts.append(
            _box(f"slat_{index}", 0.0, 0.0, level, w - post, d - post, 0.08, "wood_furniture")
        )
        parts.append(
            _box(
                f"mattress_{index}", 0.0, 0.0, level + 0.12, w - 0.12, d - 0.12, 0.16, "upholstery"
            )
        )
    parts.append(
        _box("ladder", w / 2 - post / 2, 0.0, h / 2, post * 2, post * 2, h * 0.9, "wood_furniture")
    )
    return parts


def _nightstand(w: float, d: float, h: float, mat: str) -> list[Part]:
    return [
        _box("body", 0.0, 0.0, h / 2, w, d, h, "wood_furniture"),
        _box("drawer", 0.0, d / 2 + 0.008, h * 0.68, w - 0.08, 0.018, h * 0.24, "painted_mdf"),
        _box("handle", 0.0, d / 2 + 0.03, h * 0.68, w * 0.35, 0.025, 0.025, "metal_fixture"),
    ]


def _wardrobe(w: float, d: float, h: float, mat: str) -> list[Part]:
    door_t = 0.02
    return [
        _box("carcass", 0.0, 0.0, h / 2, w, d, h, "wood_furniture"),
        _box(
            "door_l",
            -w / 4,
            d / 2 + door_t / 2,
            h / 2,
            w / 2 - 0.02,
            door_t,
            h - 0.06,
            "painted_mdf",
        ),
        _box(
            "door_r",
            w / 4,
            d / 2 + door_t / 2,
            h / 2,
            w / 2 - 0.02,
            door_t,
            h - 0.06,
            "painted_mdf",
        ),
        _box("handle_l", -0.03, d / 2 + door_t + 0.03, h * 0.48, 0.03, 0.03, 0.16, "metal_fixture"),
        _box("handle_r", 0.03, d / 2 + door_t + 0.03, h * 0.48, 0.03, 0.03, 0.16, "metal_fixture"),
    ]


def _dresser(w: float, d: float, h: float, mat: str) -> list[Part]:
    parts = [_box("body", 0.0, 0.0, h / 2, w, d, h, "wood_furniture")]
    for index in range(3):
        z = h * (0.20 + 0.27 * index)
        parts.append(
            _box(f"drawer_{index}", 0.0, d / 2 + 0.008, z, w - 0.08, 0.018, h * 0.19, "painted_mdf")
        )
        parts.append(
            _box(f"handle_{index}", 0.0, d / 2 + 0.03, z, w * 0.35, 0.025, 0.022, "metal_fixture")
        )
    return parts


def _rug(w: float, d: float, h: float, mat: str) -> list[Part]:
    return [_box("pile", 0.0, 0.0, h / 2, w, d, h, "carpet")]


def _sofa(w: float, d: float, h: float, mat: str) -> list[Part]:
    seat_h = 0.36
    back_t = 0.22
    arm_w = 0.18
    arm_h = 0.60
    inner_d = d - back_t
    return [
        _box("seat", 0.0, -back_t / 2, seat_h / 2, w, inner_d, seat_h, "upholstery"),
        _box("back", 0.0, d / 2 - back_t / 2, h / 2, w, back_t, h, "upholstery"),
        _box(
            "arm_l",
            -(w / 2) + arm_w / 2,
            -back_t / 2,
            arm_h / 2,
            arm_w,
            inner_d,
            arm_h,
            "upholstery",
        ),
        _box(
            "arm_r", w / 2 - arm_w / 2, -back_t / 2, arm_h / 2, arm_w, inner_d, arm_h, "upholstery"
        ),
        _box(
            "cushion_l",
            -(w / 4),
            -back_t / 2,
            seat_h + 0.06,
            w / 2 - arm_w - 0.06,
            inner_d - 0.08,
            0.12,
            "upholstery",
        ),
        _box(
            "cushion_r",
            w / 4,
            -back_t / 2,
            seat_h + 0.06,
            w / 2 - arm_w - 0.06,
            inner_d - 0.08,
            0.12,
            "upholstery",
        ),
        _box("feet", 0.0, 0.0, 0.02, w - 0.20, d - 0.20, 0.04, "wood_furniture"),
    ]


def _armchair(w: float, d: float, h: float, mat: str) -> list[Part]:
    seat_h = 0.36
    back_t = 0.20
    arm_w = 0.16
    inner_d = d - back_t
    return [
        _box("seat", 0.0, -back_t / 2, seat_h / 2, w, inner_d, seat_h, "upholstery"),
        _box("back", 0.0, d / 2 - back_t / 2, h / 2, w, back_t, h, "upholstery"),
        _box(
            "arm_l", -(w / 2) + arm_w / 2, -back_t / 2, 0.55 / 2, arm_w, inner_d, 0.55, "upholstery"
        ),
        _box("arm_r", w / 2 - arm_w / 2, -back_t / 2, 0.55 / 2, arm_w, inner_d, 0.55, "upholstery"),
    ]


def _coffee_table(w: float, d: float, h: float, mat: str) -> list[Part]:
    top_t = 0.05
    leg = 0.06
    leg_h = h - top_t
    parts = [_box("top", 0.0, 0.0, h - top_t / 2, w, d, top_t, "wood_furniture")]
    for sx, sy, tag in ((1, 1, "ne"), (1, -1, "se"), (-1, 1, "nw"), (-1, -1, "sw")):
        parts.append(
            _cyl(
                f"leg_{tag}",
                sx * (w / 2 - leg),
                sy * (d / 2 - leg),
                leg_h / 2,
                leg,
                leg_h,
                "wood_furniture",
            )
        )
    parts.append(_box("shelf", 0.0, 0.0, leg_h * 0.35, w - 0.20, d - 0.20, 0.03, "wood_furniture"))
    return parts


def _tv(w: float, d: float, h: float, mat: str) -> list[Part]:
    return [
        _box("panel", 0.0, 0.0, h / 2, w, d, h, "metal_fixture"),
        _box("bezel", 0.0, -d / 2 - 0.005, h / 2, w * 0.96, 0.01, h * 0.92, "painted_mdf"),
        _box("foot", 0.0, 0.02, 0.02, w * 0.35, d + 0.14, 0.04, "metal_fixture"),
    ]


def _tv_stand(w: float, d: float, h: float, mat: str) -> list[Part]:
    door_t = 0.018
    return [
        _box("body", 0.0, 0.0, h / 2, w, d, h, "painted_mdf"),
        _box(
            "door_l",
            -w / 4,
            d / 2 + door_t / 2,
            h / 2,
            w / 2 - 0.02,
            door_t,
            h - 0.06,
            "wood_furniture",
        ),
        _box(
            "door_r",
            w / 4,
            d / 2 + door_t / 2,
            h / 2,
            w / 2 - 0.02,
            door_t,
            h - 0.06,
            "wood_furniture",
        ),
        _box("plinth", 0.0, 0.0, 0.03, w - 0.06, d - 0.06, 0.06, "painted_mdf"),
    ]


def _bookshelf(w: float, d: float, h: float, mat: str) -> list[Part]:
    back_t = 0.015
    side_t = 0.02
    board_t = 0.025
    shelves = 4
    parts = [
        _box("back", 0.0, -d / 2 + back_t / 2, h / 2, w, back_t, h, "painted_mdf"),
        _box("side_l", -(w / 2) + side_t / 2, 0.0, h / 2, side_t, d, h, "painted_mdf"),
        _box("side_r", w / 2 - side_t / 2, 0.0, h / 2, side_t, d, h, "painted_mdf"),
        _box("top", 0.0, 0.0, h - board_t / 2, w, d, board_t, "painted_mdf"),
        _box("bottom", 0.0, 0.0, board_t / 2, w, d, board_t, "painted_mdf"),
    ]
    for index in range(1, shelves + 1):
        parts.append(
            _box(
                f"shelf_{index}",
                0.0,
                0.0,
                h * index / (shelves + 1),
                w - 2 * side_t,
                d,
                board_t,
                "painted_mdf",
            )
        )
    return parts


def _desk(w: float, d: float, h: float, mat: str) -> list[Part]:
    top_t = 0.04
    leg = 0.05
    leg_h = h - top_t
    parts = [
        _box("top", 0.0, 0.0, h - top_t / 2, w, d, top_t, "wood_furniture"),
        _box("modesty", 0.0, -d / 2 + 0.03, leg_h / 2, w - 0.20, 0.03, leg_h - 0.06, "painted_mdf"),
    ]
    for sx, sy, tag in ((1, 1, "ne"), (1, -1, "se"), (-1, 1, "nw"), (-1, -1, "sw")):
        parts.append(
            _box(
                f"leg_{tag}",
                sx * (w / 2 - leg / 2 - 0.01),
                sy * (d / 2 - leg / 2 - 0.01),
                leg_h / 2,
                leg,
                leg,
                leg_h,
                "metal_fixture",
            )
        )
    return parts


def _dining_table(w: float, d: float, h: float, mat: str) -> list[Part]:
    top_t = 0.05
    leg = 0.08
    leg_h = h - top_t
    parts = [_box("top", 0.0, 0.0, h - top_t / 2, w, d, top_t, "wood_furniture")]
    for sx, sy, tag in ((1, 1, "ne"), (1, -1, "se"), (-1, 1, "nw"), (-1, -1, "sw")):
        parts.append(
            _box(
                f"leg_{tag}",
                sx * (w / 2 - leg),
                sy * (d / 2 - leg),
                leg_h / 2,
                leg,
                leg,
                leg_h,
                "wood_furniture",
            )
        )
    return parts


def _chair(w: float, d: float, h: float, mat: str) -> list[Part]:
    seat_h = 0.45
    seat_t = 0.05
    leg = 0.04
    back_t = 0.05
    parts = [_box("seat", 0.0, 0.0, seat_h - seat_t / 2, w, d, seat_t, "wood_furniture")]
    for sx, sy, tag in ((1, 1, "ne"), (1, -1, "se"), (-1, 1, "nw"), (-1, -1, "sw")):
        parts.append(
            _box(
                f"leg_{tag}",
                sx * (w / 2 - leg / 2 - 0.01),
                sy * (d / 2 - leg / 2 - 0.01),
                (seat_h - seat_t) / 2,
                leg,
                leg,
                seat_h - seat_t,
                "wood_furniture",
            )
        )
    parts.append(
        _box(
            "back",
            0.0,
            d / 2 - back_t / 2,
            (h + seat_h) / 2,
            w - 0.06,
            back_t,
            h - seat_h,
            "wood_furniture",
        )
    )
    return parts


def _toilet(w: float, d: float, h: float, mat: str) -> list[Part]:
    tank_d = 0.18
    return [
        _cyl("bowl", 0.0, d * 0.14, 0.20, w * 0.92, 0.40, "ceramic_sanitary"),
        _box("seat", 0.0, d * 0.14, 0.42, w * 0.92, d * 0.52, 0.04, "ceramic_sanitary"),
        _box(
            "tank",
            0.0,
            -d / 2 + tank_d / 2,
            h * 0.68,
            w * 0.95,
            tank_d,
            h * 0.52,
            "ceramic_sanitary",
        ),
        _box("flush", 0.0, -d / 2 + tank_d + 0.02, h * 0.90, 0.06, 0.03, 0.03, "metal_fixture"),
    ]


def _vanity_sink(w: float, d: float, h: float, mat: str) -> list[Part]:
    cabinet_h = h - 0.08
    counter_t = 0.05
    bowl = min(w, d) * 0.62
    return [
        _box("cabinet", 0.0, 0.0, cabinet_h / 2, w, d, cabinet_h, "painted_mdf"),
        _box(
            "door",
            0.0,
            d / 2 + 0.01,
            cabinet_h / 2,
            w - 0.10,
            0.02,
            cabinet_h - 0.10,
            "painted_mdf",
        ),
        _box(
            "counter",
            0.0,
            0.0,
            cabinet_h + counter_t / 2,
            w + 0.02,
            d + 0.02,
            counter_t,
            "ceramic_sanitary",
        ),
        _cyl("basin", 0.0, 0.0, cabinet_h + counter_t + 0.03, bowl, 0.10, "ceramic_sanitary"),
        _box(
            "tap",
            0.0,
            -d / 2 + 0.07,
            cabinet_h + counter_t + 0.11,
            0.045,
            0.045,
            0.18,
            "metal_fixture",
        ),
    ]


def _bathtub(w: float, d: float, h: float, mat: str) -> list[Part]:
    wall_t = 0.07
    base_t = 0.06
    return [
        _box("base", 0.0, 0.0, base_t / 2, w, d, base_t, "ceramic_sanitary"),
        _box("side_w", -(w / 2) + wall_t / 2, 0.0, h / 2, wall_t, d, h, "ceramic_sanitary"),
        _box("side_e", w / 2 - wall_t / 2, 0.0, h / 2, wall_t, d, h, "ceramic_sanitary"),
        _box(
            "side_s",
            0.0,
            -(d / 2) + wall_t / 2,
            h / 2,
            w - 2 * wall_t,
            wall_t,
            h,
            "ceramic_sanitary",
        ),
        _box(
            "side_n", 0.0, d / 2 - wall_t / 2, h / 2, w - 2 * wall_t, wall_t, h, "ceramic_sanitary"
        ),
        _cyl("drain", w * 0.25, 0.0, base_t + 0.01, 0.06, 0.02, "metal_fixture"),
        _box("tap", -(w / 2 - wall_t / 2), 0.0, h + 0.10, 0.05, 0.05, 0.20, "metal_fixture"),
    ]


def _shower(w: float, d: float, h: float, mat: str) -> list[Part]:
    tray_h = 0.10
    glass_t = 0.012
    panel_h = h - tray_h
    return [
        _box("tray", 0.0, 0.0, tray_h / 2, w, d, tray_h, "ceramic_sanitary"),
        _box(
            "glass_s",
            0.0,
            -(d / 2) + glass_t / 2,
            tray_h + panel_h / 2,
            w,
            glass_t,
            panel_h,
            "glass_window",
        ),
        _box(
            "glass_w",
            -(w / 2) + glass_t / 2,
            0.0,
            tray_h + panel_h / 2,
            glass_t,
            d,
            panel_h,
            "glass_window",
        ),
        _box(
            "riser",
            w / 2 - 0.06,
            d / 2 - 0.06,
            tray_h + (h - tray_h) / 2,
            0.04,
            0.04,
            h - tray_h,
            "metal_fixture",
        ),
        _box("head", w / 2 - 0.20, d / 2 - 0.20, h - 0.03, 0.10, 0.20, 0.05, "metal_fixture"),
        _cyl("drain", 0.0, 0.0, tray_h - 0.01, 0.07, 0.02, "metal_fixture"),
    ]


def _mirror(w: float, d: float, h: float, mat: str) -> list[Part]:
    return [
        _box("glass", 0.0, 0.0, h / 2, w, d, h, "mirror_glass"),
        _box("frame", 0.0, 0.0, 0.02, w + 0.04, d + 0.02, 0.04, "painted_mdf"),
    ]


def _towel_rail(w: float, d: float, h: float, mat: str) -> list[Part]:
    bar_t = 0.028
    return [
        _box("bar", 0.0, 0.0, 0.0, w, bar_t, bar_t, "metal_fixture"),
        _box("bracket_l", -(w / 2) + 0.02, -0.03, 0.0, 0.03, 0.06, bar_t, "metal_fixture"),
        _box("bracket_r", w / 2 - 0.02, -0.03, 0.0, 0.03, 0.06, bar_t, "metal_fixture"),
    ]


def _shoe_cabinet(w: float, d: float, h: float, mat: str) -> list[Part]:
    door_t = 0.018
    return [
        _box("body", 0.0, 0.0, h / 2, w, d, h, "painted_mdf"),
        _box(
            "door_l",
            -w / 4,
            d / 2 + door_t / 2,
            h / 2,
            w / 2 - 0.02,
            door_t,
            h - 0.06,
            "wood_furniture",
        ),
        _box(
            "door_r",
            w / 4,
            d / 2 + door_t / 2,
            h / 2,
            w / 2 - 0.02,
            door_t,
            h - 0.06,
            "wood_furniture",
        ),
        _box("top", 0.0, 0.0, h + 0.015, w + 0.03, d + 0.03, 0.03, "wood_furniture"),
    ]


def _coat_rack(w: float, d: float, h: float, mat: str) -> list[Part]:
    base = max(w, d) * 0.6
    return [
        _cyl("base", 0.0, 0.0, 0.015, base, 0.03, "metal_fixture"),
        _cyl("post", 0.0, 0.0, h / 2, 0.05, h, "metal_fixture"),
        _box("hook_l", -w * 0.30, 0.0, h - 0.06, w * 0.55, 0.03, 0.03, "metal_fixture"),
        _box("hook_r", w * 0.30, 0.0, h - 0.06, w * 0.55, 0.03, 0.03, "metal_fixture"),
    ]


def _kitchen_counter(w: float, d: float, h: float, mat: str) -> list[Part]:
    top_t = 0.04
    cabinet_h = h - top_t
    door_t = 0.018
    units = 4
    parts = [
        _box("cabinet", 0.0, 0.0, cabinet_h / 2, w, d - 0.02, cabinet_h, "painted_mdf"),
        _box("worktop", 0.0, 0.0, h - top_t / 2, w + 0.02, d + 0.02, top_t, "ceramic_sanitary"),
        _box("upstand", 0.0, d / 2, h - top_t - 0.01, w + 0.02, 0.02, 0.10, "ceramic_sanitary"),
        _box("plinth", 0.0, -0.02, 0.05, w - 0.10, d - 0.10, 0.10, "painted_mdf"),
    ]
    step = w / units
    for index in range(units):
        cx = -w / 2 + step * (index + 0.5)
        parts.append(
            _box(
                f"door_{index}",
                cx,
                d / 2 - 0.01 + door_t / 2,
                cabinet_h / 2,
                step - 0.04,
                door_t,
                cabinet_h - 0.14,
                "painted_mdf",
            )
        )
        parts.append(
            _box(
                f"handle_{index}",
                cx,
                d / 2 + 0.005,
                cabinet_h * 0.82,
                step * 0.5,
                0.02,
                0.02,
                "metal_fixture",
            )
        )
    return parts


def _kitchen_sink(w: float, d: float, h: float, mat: str) -> list[Part]:
    return [
        _box("basin", 0.0, 0.0, h / 2, w, d, h, "metal_fixture"),
        _cyl("bowl", -w * 0.18, 0.0, h + 0.005, min(w, d) * 0.5, 0.02, "metal_fixture"),
        _cyl("bowl2", w * 0.18, 0.0, h + 0.005, min(w, d) * 0.5, 0.02, "metal_fixture"),
        _cyl("tap", 0.0, -d / 2 + 0.05, h + 0.13, 0.045, 0.26, "metal_fixture"),
    ]


def _stove(w: float, d: float, h: float, mat: str) -> list[Part]:
    parts = [
        _box("hob", 0.0, 0.0, h / 2, w, d, h, "metal_fixture"),
        _box("panel", 0.0, -d / 2 + 0.02, h / 2, w, 0.03, h, "painted_mdf"),
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(
                _cyl(
                    f"burner_{'e' if sx > 0 else 'w'}{'n' if sy > 0 else 's'}",
                    sx * w * 0.24,
                    sy * d * 0.22,
                    h + 0.006,
                    min(w, d) * 0.34,
                    0.012,
                    "metal_fixture",
                )
            )
    return parts


def _refrigerator(w: float, d: float, h: float, mat: str) -> list[Part]:
    door_t = 0.04
    return [
        _box("body", 0.0, 0.0, h / 2, w, d, h, "metal_fixture"),
        _box(
            "door_freezer",
            0.0,
            d / 2 + door_t / 2,
            h * 0.76,
            w - 0.02,
            door_t,
            h * 0.44,
            "painted_mdf",
        ),
        _box(
            "door_fridge",
            0.0,
            d / 2 + door_t / 2,
            h * 0.27,
            w - 0.02,
            door_t,
            h * 0.50,
            "painted_mdf",
        ),
        _box(
            "handle_freezer",
            w * 0.36,
            d / 2 + door_t + 0.025,
            h * 0.76,
            0.03,
            0.03,
            h * 0.34,
            "metal_fixture",
        ),
        _box(
            "handle_fridge",
            w * 0.36,
            d / 2 + door_t + 0.025,
            h * 0.27,
            0.03,
            0.03,
            h * 0.36,
            "metal_fixture",
        ),
    ]


def _ceiling_lamp(w: float, d: float, h: float, mat: str) -> list[Part]:
    diameter = min(w, d)
    return [
        _cyl("rose", 0.0, 0.0, h - 0.02, diameter * 0.5, 0.04, "painted_mdf"),
        _cyl("stem", 0.0, 0.0, h * 0.5, 0.02, h, "metal_fixture"),
        _cyl("shade", 0.0, 0.0, 0.06, diameter, 0.12, "painted_mdf"),
    ]


_RECIPES = {
    "bed": _bed,
    "bunk_bed": _bunk_bed,
    "nightstand": _nightstand,
    "wardrobe": _wardrobe,
    "dresser": _dresser,
    "rug": _rug,
    "sofa": _sofa,
    "armchair": _armchair,
    "coffee_table": _coffee_table,
    "tv": _tv,
    "tv_stand": _tv_stand,
    "bookshelf": _bookshelf,
    "desk": _desk,
    "dining_table": _dining_table,
    "chair": _chair,
    "toilet": _toilet,
    "vanity_sink": _vanity_sink,
    "bathtub": _bathtub,
    "shower": _shower,
    "mirror": _mirror,
    "towel_rail": _towel_rail,
    "shoe_cabinet": _shoe_cabinet,
    "coat_rack": _coat_rack,
    "kitchen_counter": _kitchen_counter,
    "kitchen_sink": _kitchen_sink,
    "stove": _stove,
    "refrigerator": _refrigerator,
    "ceiling_lamp": _ceiling_lamp,
}
