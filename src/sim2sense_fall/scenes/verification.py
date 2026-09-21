"""Validate a manifest on CPU and compare its individual primitives with USD."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .numbers import finite_number
from .planner import LightSpec, ScenePlan, ScenePrim
from .usd import pxr_modules


def load_scene_manifest(path: Path) -> ScenePlan:
    """Read the complete expected geometry; missing fields and NaN fail early."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"manifest contains non-finite number {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    try:
        prims = []
        for p in payload["prims"]:
            body = p["shape"] == "xform"
            prim = ScenePrim(
                path=p["path"],
                name=p["name"],
                category=p["category"],
                shape=p["shape"],
                center=tuple(p["center"]),
                size=(0, 0, 0) if body else tuple(p["size"]),
                rotation_z_deg=p["rotation_z_deg"],
                material=None if body else p["material"],
                room_id=p["room_id"],
                semantic=p["semantic"],
                collision=False if body else p["collision"],
                is_body=body,
                body_path=p.get("body_path"),
                relative_to_body=p.get("relative_to_body", False),
                density_kg_m3=p.get("density_kg_m3", 0),
                mass_kg=p["mass_kg"],
                static_friction=p.get("static_friction", 0.5),
                dynamic_friction=p.get("dynamic_friction", 0.4),
                restitution=p.get("restitution", 0),
                movable=p["movable"],
                tags=tuple(p["tags"]),
            )
            if p["physics"] != prim.physics_mode:
                raise ValueError(f"{prim.path}: inconsistent physics mode")
            prims.append(prim)
        lights = tuple(LightSpec(**entry) for entry in payload["lights"])
        plan = ScenePlan(
            scene_id=payload["scene_id"],
            frequency_hz=payload["frequency_hz"],
            seed=payload["seed"],
            description=payload["description"],
            prims=tuple(prims),
            lights=lights,
            materials=payload["materials"],
            stats=payload["stats"],
        )
        if not plan.prims:
            raise ValueError("manifest has no primitives")
        for key, actual in (
            ("geometry_count", sum(p.is_geometry for p in plan.prims)),
            ("rigid_body_count", len(plan.bodies)),
            ("prim_count", len(plan.prims)),
        ):
            if plan.stats[key] != actual:
                raise ValueError(f"manifest {key} does not match its primitive records")
        if not all(p.material in plan.materials for p in plan.prims if p.is_geometry):
            raise ValueError("manifest references an unknown material")
        # Enforce finite numbers even in material records and statistics.
        json.dumps(payload, allow_nan=False)
        return plan
    except (KeyError, TypeError) as exc:
        raise ValueError(f"incomplete or malformed scene manifest: {exc}") from exc


def stage_manifest_errors(stage: Any, plan: ScenePlan, *, tolerance: float = 1e-5) -> list[str]:
    """Check paths, transforms, dimensions, labels, materials and collision APIs.

    Run before stepping physics, because a simulation legitimately changes the
    dynamic body transforms. Errors name the individual mismatching primitive.
    """
    if finite_number(tolerance, "comparison tolerance") <= 0:
        raise ValueError("comparison tolerance must be positive")
    rt = pxr_modules()
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def close(actual: Any, expected: float) -> bool:
        return (
            actual is not None
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance)
        )

    check(rt.UsdGeom.GetStageUpAxis(stage) == "Z", "stage must be Z-up")
    check(close(rt.UsdGeom.GetStageMetersPerUnit(stage), 1), "stage must use metres")
    expected_shapes = {f"/World/{p.path}" for p in plan.prims if p.is_geometry}
    actual_shapes = {str(p.GetPath()) for p in stage.Traverse() if p.IsA(rt.UsdGeom.Gprim)}
    check(expected_shapes == actual_shapes, "geometry paths differ from manifest")
    expected_bodies = {f"/World/{p.path}" for p in plan.bodies}
    actual_bodies = {
        str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(rt.UsdPhysics.RigidBodyAPI)
    }
    check(expected_bodies == actual_bodies, "rigid body paths differ from manifest")
    expected_colliders = {f"/World/{p.path}" for p in plan.prims if p.collision}
    actual_colliders = {
        str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(rt.UsdPhysics.CollisionAPI)
    }
    check(expected_colliders == actual_colliders, "collider paths differ from manifest")
    expected_matrices: dict[str, Any] = {}
    for p in plan.prims:
        scale = p.size if p.is_geometry and not p.is_cylinder else (1, 1, 1)
        matrix = rt.Gf.Matrix4d().SetScale(rt.Gf.Vec3d(*scale))
        matrix *= rt.Gf.Matrix4d().SetRotate(rt.Gf.Rotation(rt.Gf.Vec3d(0, 0, 1), p.rotation_z_deg))
        matrix *= rt.Gf.Matrix4d().SetTranslate(rt.Gf.Vec3d(*p.center))
        expected_matrices[p.path] = matrix
    cache = rt.UsdGeom.XformCache()
    for p in plan.prims:
        path = f"/World/{p.path}"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            errors.append(f"missing primitive {path}")
            continue
        expected_type = "Xform" if p.is_body else "Cylinder" if p.is_cylinder else "Cube"
        check(prim.GetTypeName() == expected_type, f"{path}: shape type")
        matrix = expected_matrices[p.path]
        if p.relative_to_body:
            matrix = matrix * expected_matrices[p.body_path]
        actual = cache.GetLocalToWorldTransform(prim)
        check(
            all(close(actual[i][j], matrix[i][j]) for i in range(4) for j in range(4)),
            f"{path}: world transform/dimensions",
        )
        for key, expected in (
            ("roomId", p.room_id),
            ("semantic", p.semantic),
            ("category", p.category),
            ("physicsMode", p.physics_mode),
            ("materialName", p.material or ""),
            ("movable", p.movable),
        ):
            check(prim.GetAttribute(f"sim2sense:{key}").Get() == expected, f"{path}: {key}")
        if p.is_body:
            check(
                close(rt.UsdPhysics.MassAPI(prim).GetMassAttr().Get(), p.mass_kg), f"{path}: mass"
            )
            continue
        if p.is_cylinder:
            geom = rt.UsdGeom.Cylinder(prim)
            check(
                close(geom.GetRadiusAttr().Get(), p.size[0] / 2)
                and close(geom.GetHeightAttr().Get(), p.size[2])
                and geom.GetAxisAttr().Get() == "Z",
                f"{path}: cylinder dimensions",
            )
        else:
            check(close(rt.UsdGeom.Cube(prim).GetSizeAttr().Get(), 1), f"{path}: cube size")
        if p.collision:
            check(
                rt.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                f"{path}: collision disabled",
            )
        binding = rt.UsdShade.MaterialBindingAPI(prim)
        material, _ = binding.ComputeBoundMaterial()
        check(
            bool(material) and str(material.GetPath()) == f"/World/Materials/{p.material}",
            f"{path}: visual material binding",
        )
        physics, _ = binding.ComputeBoundMaterial("physics")
        if not physics:
            errors.append(f"{path}: missing physics material")
        else:
            api = rt.UsdPhysics.MaterialAPI(physics.GetPrim())
            for actual_value, expected in (
                (api.GetStaticFrictionAttr().Get(), p.static_friction),
                (api.GetDynamicFrictionAttr().Get(), p.dynamic_friction),
                (api.GetRestitutionAttr().Get(), p.restitution),
            ):
                check(close(actual_value, expected), f"{path}: contact material coefficients")
    for name, entry in plan.materials.items():
        path = f"/World/Materials/{name}"
        shader = rt.UsdShade.Shader(stage.GetPrimAtPath(f"{path}/PreviewSurface"))
        if not shader:
            errors.append(f"{path}: missing preview shader")
            continue
        actual_color = shader.GetInput("diffuseColor").Get()
        check(
            actual_color is not None
            and all(close(a, b) for a, b in zip(actual_color, entry["base_color"], strict=True)),
            f"{path}: diffuse color",
        )
        for key in ("roughness", "metallic"):
            check(close(shader.GetInput(key).Get(), entry[key]), f"{path}: {key}")
        material = stage.GetPrimAtPath(path)
        for key, expected in (entry.get("electromagnetic") or {}).items():
            if expected is None:
                continue
            actual = material.GetAttribute(f"sim2sense:em_{key}").Get()
            check(
                actual == expected
                if isinstance(expected, (str, bool))
                else close(actual, expected),
                f"{path}: EM {key}",
            )
    return errors
