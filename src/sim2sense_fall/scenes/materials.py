"""Indoor material library shared by the USD scene and the radio-propagation stage.

Every surface in the apartment scene is described by a single :class:`MaterialSpec`
that carries three views of the same physical material:

* **visual** -- PBR inputs authored into ``UsdPreviewSurface`` so the Isaac Sim GUI
  is readable and material identity is obvious to a human reviewer;
* **mechanical** -- static/dynamic friction and restitution used by the PhysX
  material bound to the collider, which drives how a falling body behaves at
  contact;
* **electromagnetic** -- ITU-R P.2040-3 parameters that the Sionna RT stage needs
  in order to turn the same geometry into a propagation environment.

Keeping the three views in one record prevents the classic failure mode where the
rendered scene and the radio scene silently drift apart and describe two
different apartments.

The electromagnetic numbers are *power-law coefficients*, not fixed values, so a
scene can be evaluated at 2.4 GHz, 5 GHz or any other band without silently
reusing a number that was only valid at one frequency. The authoritative values
should still be cross-checked against the ``itu_*`` materials shipped with the
installed Sionna version; the ``sionna_name`` field records which built-in
material each entry intends to match.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .numbers import finite_number, finite_vector

__all__ = [
    "DEFAULT_MATERIALS",
    "ITUMaterial",
    "MaterialSpec",
    "material_library_from_config",
]


def _require_unit_interval(name: str, value: float) -> None:
    finite_number(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class ITUMaterial:
    """ITU-R P.2040-3 power-law model for a building material.

    The recommendation models the real part of the relative permittivity and the
    effective conductivity as ``a * f^b`` and ``c * f^d`` with ``f`` in GHz.
    """

    sionna_name: str
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d"):
            finite_number(getattr(self, name), f"ITU.{name}")
        if self.a <= 0 or self.c < 0:
            raise ValueError("ITU a must be positive and c non-negative")

    def relative_permittivity(self, frequency_hz: float) -> float:
        """Real part of the relative permittivity at ``frequency_hz``."""

        return self.a * self._ghz(frequency_hz) ** self.b

    def conductivity(self, frequency_hz: float) -> float:
        """Effective conductivity in S/m at ``frequency_hz``."""

        return self.c * self._ghz(frequency_hz) ** self.d

    @staticmethod
    def _ghz(frequency_hz: float) -> float:
        finite_number(frequency_hz, "frequency_hz")
        if frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz!r}")
        return frequency_hz / 1e9


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """Visual, mechanical and electromagnetic description of one surface."""

    name: str
    base_color: tuple[float, float, float]
    roughness: float
    metallic: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    thickness_m: float
    density_kg_m3: float = 700.0
    em: ITUMaterial | None = None
    scattering_coefficient: float = 0.1
    note: str = ""

    def __post_init__(self) -> None:
        finite_number(self.thickness_m, f"{self.name}.thickness_m")
        finite_number(self.density_kg_m3, f"{self.name}.density_kg_m3")
        if not self.name.strip():
            raise ValueError("material name must be non-empty")
        if len(self.base_color) != 3:
            raise ValueError(f"{self.name}: base_color must have three components")
        for channel in self.base_color:
            _require_unit_interval(f"{self.name}.base_color", channel)
        _require_unit_interval(f"{self.name}.roughness", self.roughness)
        _require_unit_interval(f"{self.name}.metallic", self.metallic)
        _require_unit_interval(f"{self.name}.static_friction", self.static_friction)
        _require_unit_interval(f"{self.name}.dynamic_friction", self.dynamic_friction)
        _require_unit_interval(f"{self.name}.restitution", self.restitution)
        _require_unit_interval(f"{self.name}.scattering_coefficient", self.scattering_coefficient)
        if self.dynamic_friction > self.static_friction:
            raise ValueError(f"{self.name}: dynamic friction cannot exceed static friction")
        if self.thickness_m <= 0:
            raise ValueError(f"{self.name}: thickness_m must be positive")
        if self.density_kg_m3 <= 0:
            raise ValueError(f"{self.name}: density_kg_m3 must be positive")

    def em_parameters(self, frequency_hz: float) -> dict[str, float | str] | None:
        """Return Sionna-ready electromagnetic parameters, or ``None`` if unmapped."""

        if self.em is None:
            return None
        return {
            "sionna_material": self.em.sionna_name,
            "relative_permittivity": round(self.em.relative_permittivity(frequency_hz), 6),
            "conductivity_s_per_m": round(self.em.conductivity(frequency_hz), 6),
            "scattering_coefficient": self.scattering_coefficient,
            "thickness_m": self.thickness_m,
            "frequency_hz": frequency_hz,
        }

    def as_manifest_entry(self, frequency_hz: float) -> dict[str, Any]:
        """Serialisable record written to the scene manifest sidecar."""

        return {
            "name": self.name,
            "base_color": list(self.base_color),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "thickness_m": self.thickness_m,
            "density_kg_m3": self.density_kg_m3,
            "note": self.note,
            "electromagnetic": self.em_parameters(frequency_hz),
        }


def _itu(name: str, a: float, b: float, c: float, d: float) -> ITUMaterial:
    return ITUMaterial(sionna_name=name, a=a, b=b, c=c, d=d)


# ITU-R P.2040-3 Table 3 coefficients. ``em`` is intentionally left as ``None``
# only for surfaces where no defensible ITU entry exists.
DEFAULT_MATERIALS: dict[str, MaterialSpec] = {
    "painted_drywall": MaterialSpec(
        name="painted_drywall",
        base_color=(0.88, 0.87, 0.84),
        roughness=0.90,
        metallic=0.0,
        static_friction=0.60,
        dynamic_friction=0.50,
        restitution=0.02,
        thickness_m=0.12,
        density_kg_m3=700.0,
        em=_itu("itu_plasterboard", 2.73, 0.0, 0.0085, 0.9395),
        note="Interior painted partition wall.",
    ),
    "concrete_slab": MaterialSpec(
        name="concrete_slab",
        base_color=(0.55, 0.55, 0.55),
        roughness=0.85,
        metallic=0.0,
        static_friction=0.75,
        dynamic_friction=0.65,
        restitution=0.02,
        thickness_m=0.30,
        density_kg_m3=2300.0,
        em=_itu("itu_concrete", 5.24, 0.0, 0.0462, 0.7822),
        note="Structural slab under the floor and above the ceiling.",
    ),
    "gypsum_ceiling": MaterialSpec(
        name="gypsum_ceiling",
        base_color=(0.92, 0.92, 0.90),
        roughness=0.95,
        metallic=0.0,
        static_friction=0.70,
        dynamic_friction=0.60,
        restitution=0.01,
        thickness_m=0.15,
        density_kg_m3=700.0,
        em=_itu("itu_ceiling_board", 1.5, 0.0, 0.0038, 1.1529),
        note="Suspended plasterboard ceiling.",
    ),
    "wood_floor": MaterialSpec(
        name="wood_floor",
        base_color=(0.52, 0.34, 0.20),
        roughness=0.55,
        metallic=0.0,
        static_friction=0.65,
        dynamic_friction=0.55,
        restitution=0.05,
        thickness_m=0.02,
        density_kg_m3=700.0,
        em=_itu("itu_floorboard", 3.66, 0.0, 0.0044, 1.3515),
        note="Laminated timber floor in the dry rooms.",
    ),
    "tile_floor": MaterialSpec(
        name="tile_floor",
        base_color=(0.80, 0.81, 0.80),
        roughness=0.22,
        metallic=0.0,
        static_friction=0.45,
        dynamic_friction=0.38,
        restitution=0.03,
        thickness_m=0.01,
        density_kg_m3=2000.0,
        em=_itu("itu_marble", 5.86, 0.0, 0.0104, 1.0996),
        note=(
            "Glazed ceramic tile. ITU-R P.2040 has no ceramic entry, so the marble "
            "model is used as a smooth, mirror-like proxy; verify before use."
        ),
    ),
    "carpet": MaterialSpec(
        name="carpet",
        base_color=(0.34, 0.30, 0.28),
        roughness=0.98,
        metallic=0.0,
        static_friction=0.85,
        dynamic_friction=0.80,
        restitution=0.0,
        thickness_m=0.01,
        density_kg_m3=200.0,
        em=_itu("itu_floorboard", 3.66, 0.0, 0.0044, 1.3515),
        scattering_coefficient=0.50,
        note="Fibrous floor covering; the floorboard model is used as a substrate proxy.",
    ),
    "wood_furniture": MaterialSpec(
        name="wood_furniture",
        base_color=(0.45, 0.29, 0.16),
        roughness=0.45,
        metallic=0.0,
        static_friction=0.60,
        dynamic_friction=0.50,
        restitution=0.05,
        thickness_m=0.02,
        density_kg_m3=600.0,
        em=_itu("itu_wood", 1.99, 0.0, 0.0047, 1.0718),
        note="Solid timber frames: bed, table, desk, cupboards.",
    ),
    "painted_mdf": MaterialSpec(
        name="painted_mdf",
        base_color=(0.78, 0.78, 0.76),
        roughness=0.55,
        metallic=0.0,
        static_friction=0.55,
        dynamic_friction=0.45,
        restitution=0.05,
        thickness_m=0.02,
        density_kg_m3=700.0,
        em=_itu("itu_chipboard", 2.73, 0.0, 0.0085, 0.9395),
        note="Lacquered MDF panels and drawer fronts.",
    ),
    "upholstery": MaterialSpec(
        name="upholstery",
        base_color=(0.26, 0.31, 0.40),
        roughness=0.95,
        metallic=0.0,
        static_friction=0.90,
        dynamic_friction=0.85,
        restitution=0.0,
        thickness_m=0.03,
        density_kg_m3=100.0,
        em=_itu("itu_ceiling_board", 1.5, 0.0, 0.0038, 1.1529),
        scattering_coefficient=0.50,
        note="Sofa and mattress fabric; a soft fibrous proxy is used for radio purposes.",
    ),
    "ceramic_sanitary": MaterialSpec(
        name="ceramic_sanitary",
        base_color=(0.95, 0.95, 0.96),
        roughness=0.12,
        metallic=0.0,
        static_friction=0.50,
        dynamic_friction=0.42,
        restitution=0.05,
        thickness_m=0.02,
        density_kg_m3=2300.0,
        em=_itu("itu_marble", 5.86, 0.0, 0.0104, 1.0996),
        note="Vitreous china: toilet, basin, bathtub.",
    ),
    "metal_fixture": MaterialSpec(
        name="metal_fixture",
        base_color=(0.62, 0.63, 0.65),
        roughness=0.30,
        metallic=0.90,
        static_friction=0.35,
        dynamic_friction=0.30,
        restitution=0.10,
        thickness_m=0.005,
        density_kg_m3=7800.0,
        em=_itu("itu_metal", 1.0, 0.0, 1e7, 0.0),
        note="Taps, appliance shells and shelf brackets.",
    ),
    "glass_window": MaterialSpec(
        name="glass_window",
        base_color=(0.75, 0.85, 0.88),
        roughness=0.05,
        metallic=0.0,
        static_friction=0.20,
        dynamic_friction=0.18,
        restitution=0.20,
        thickness_m=0.006,
        density_kg_m3=2500.0,
        em=_itu("itu_glass", 6.31, 0.0, 0.0036, 1.3394),
        note="Panes in window and shower enclosures.",
    ),
    "wood_door": MaterialSpec(
        name="wood_door",
        base_color=(0.42, 0.28, 0.17),
        roughness=0.50,
        metallic=0.0,
        static_friction=0.55,
        dynamic_friction=0.45,
        restitution=0.05,
        thickness_m=0.04,
        density_kg_m3=600.0,
        em=_itu("itu_wood", 1.99, 0.0, 0.0047, 1.0718),
        note="Interior door leaf and frame.",
    ),
    "mirror_glass": MaterialSpec(
        name="mirror_glass",
        base_color=(0.90, 0.92, 0.94),
        roughness=0.03,
        metallic=0.85,
        static_friction=0.20,
        dynamic_friction=0.18,
        restitution=0.20,
        thickness_m=0.004,
        density_kg_m3=2500.0,
        em=_itu("itu_glass", 6.31, 0.0, 0.0036, 1.3394),
        note="Silvered mirror; treated as glass for radio purposes.",
    ),
}


def material_library_from_config(
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, MaterialSpec]:
    """Merge user-supplied material overrides onto :data:`DEFAULT_MATERIALS`.

    Unknown material names are rejected so a typo in a scene file fails loudly
    instead of silently producing an untextured surface.
    """

    library = dict(DEFAULT_MATERIALS)
    if not overrides:
        return library
    for name, patch in overrides.items():
        if name not in library:
            known = ", ".join(sorted(library))
            raise ValueError(f"unknown material override {name!r}; known materials: {known}")
        library[name] = _apply_patch(library[name], patch)
    return library


def _apply_patch(spec: MaterialSpec, patch: dict[str, Any]) -> MaterialSpec:
    allowed = {field.name for field in dataclasses.fields(MaterialSpec)}
    unknown = set(patch) - allowed
    if unknown:
        raise ValueError(f"{spec.name}: unsupported material keys {sorted(unknown)}")
    values: dict[str, Any] = {}
    for key, value in patch.items():
        if key == "base_color":
            finite_vector(value, 3, f"{spec.name}.base_color")
            value = tuple(float(channel) for channel in value)
        values[key] = value
    return dataclasses.replace(spec, **values)
