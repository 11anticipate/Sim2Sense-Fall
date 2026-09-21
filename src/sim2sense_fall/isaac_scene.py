"""Author a :class:`~sim2sense_fall.scene_planner.ScenePlan` as a USD stage.

This is the only module in the scene pipeline that needs a USD runtime, and the
USD import is deferred into a function so that ``import
sim2sense_fall.isaac_scene`` never fails on a machine without Isaac Sim.

What gets authored
------------------
* one ``UsdGeom.Cube``/``Cylinder`` per planned primitive, in metres, Z-up;
* ``UsdPhysics.CollisionAPI`` on every solid, so walls, floors and furniture take
  part in contact;
* ``UsdPhysics.RigidBodyAPI`` + ``UsdPhysics.MassAPI`` only for primitives the
  scene explicitly marks ``dynamic``, keeping the apartment itself immovable;
* one visual ``UsdShade`` material per surface type, plus a deduplicated physics
  material per friction/restitution triple bound with ``materialPurpose`` set to
  ``physics``;
* ``sim2sense:*`` custom attributes carrying room id, semantic label, physics mode
  and mass, so a later stage can query the scene without re-parsing the YAML;
* a dome light, a sun and per-room ceiling lights so the GUI is readable as soon
  as the file is opened.

Box geometry uses a unit ``UsdGeom.Cube`` plus a non-uniform ``xformOp:scale``.
That is the standard procedural-scene idiom and is what lets one prim carry the
exact dimensions from the plan.

Usage (inside Isaac Sim's bundled Python)::

    from sim2sense_fall.isaac_scene import build_stage, stage_summary
    build_stage(plan, "artifacts/scenes/indoor_apartment.usda")
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .scene_planner import ScenePlan, ScenePrim

__all__ = [
    "IsaacRuntimeUnavailable",
    "activate_physics",
    "build_stage",
    "lift_rigid_bodies",
    "pxr_modules",
    "rigid_body_world_positions",
    "stage_summary",
    "step_simulation",
]

LOGGER = logging.getLogger(__name__)

WORLD_PATH = "/World"
MATERIALS_PATH = f"{WORLD_PATH}/Materials"
PHYSICS_MATERIALS_PATH = f"{WORLD_PATH}/PhysicsMaterials"

#: Preview-surface opacity used for glazing so window panes read as glass.
GLASS_OPACITY = 0.35


class IsaacRuntimeUnavailable(RuntimeError):
    """Raised when the USD/Isaac Sim modules are not importable."""


def pxr_modules() -> SimpleNamespace:
    """Import the USD modules or raise a clear, actionable error."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - depends on the host runtime
        raise IsaacRuntimeUnavailable(
            "USD modules are unavailable; run this script through Isaac Sim's bundled "
            "interpreter, for example: ~/isaacsim/python.sh scripts/build_indoor_scene.py"
        ) from exc
    return SimpleNamespace(
        Gf=Gf,
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdLux=UsdLux,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
    )


def _parent_path(path: str) -> str:
    head, _, _ = path.rpartition("/")
    return head or "/"


def validate_prim_paths(runtime: SimpleNamespace, plan: ScenePlan) -> None:
    """Fail early, and legibly, if any planned path is not a legal USD path.

    USD rejects prim names that begin with a digit, so a plan that numbers its
    children directly (``.../00_frame``) blows up deep inside ``DefinePrim`` with
    the opaque message "Path must be an absolute path". Checking up front turns
    that into an error that names the offending prim.
    """

    offenders: list[str] = []
    for prim in plan.prims:
        for segment in prim.path.split("/"):
            if not segment or not runtime.Sdf.Path.IsValidIdentifier(segment):
                offenders.append(f"{prim.path!r} has invalid segment {segment!r}")
                break
    if offenders:
        preview = "; ".join(offenders[:5])
        more = f" (and {len(offenders) - 5} more)" if len(offenders) > 5 else ""
        raise ValueError(f"planned prim paths are not valid USD paths: {preview}{more}")


def _ensure_xform(runtime: SimpleNamespace, stage: Any, path: str) -> Any:
    """Define ``path`` as an Xform, creating any missing ancestors."""

    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        return existing
    current = ""
    for part in path.strip("/").split("/"):
        current = f"{current}/{part}"
        if stage.GetPrimAtPath(current).IsValid():
            continue
        runtime.UsdGeom.Xform.Define(stage, current)
    return stage.GetPrimAtPath(path)


def _author_visual_materials(
    runtime: SimpleNamespace, stage: Any, plan: ScenePlan
) -> dict[str, Any]:
    materials: dict[str, Any] = {}
    for name, entry in plan.materials.items():
        path = f"{MATERIALS_PATH}/{name}"
        material = runtime.UsdShade.Material.Define(stage, path)
        shader = runtime.UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        color = entry["base_color"]
        shader.CreateInput("diffuseColor", runtime.Sdf.ValueTypeNames.Color3f).Set(
            runtime.Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )
        shader.CreateInput("roughness", runtime.Sdf.ValueTypeNames.Float).Set(
            float(entry["roughness"])
        )
        shader.CreateInput("metallic", runtime.Sdf.ValueTypeNames.Float).Set(
            float(entry["metallic"])
        )
        shader.CreateInput("opacity", runtime.Sdf.ValueTypeNames.Float).Set(
            GLASS_OPACITY if name == "glass_window" else 1.0
        )
        shader.CreateInput("ior", runtime.Sdf.ValueTypeNames.Float).Set(1.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        _author_em_metadata(runtime, material.GetPrim(), entry.get("electromagnetic"))
        materials[name] = material
    return materials


def _author_em_metadata(
    runtime: SimpleNamespace, material_prim: Any, em: dict[str, Any] | None
) -> None:
    if not em:
        return
    for key, value in em.items():
        if value is None:
            continue
        attr_name = f"sim2sense:em_{key}"
        if isinstance(value, str):
            material_prim.CreateAttribute(
                attr_name, runtime.Sdf.ValueTypeNames.String, custom=True
            ).Set(value)
        elif isinstance(value, bool):
            material_prim.CreateAttribute(
                attr_name, runtime.Sdf.ValueTypeNames.Bool, custom=True
            ).Set(value)
        else:
            material_prim.CreateAttribute(
                attr_name, runtime.Sdf.ValueTypeNames.Double, custom=True
            ).Set(float(value))


def _physics_material_key(prim: ScenePrim) -> str:
    token = f"{prim.static_friction:.3f}_{prim.dynamic_friction:.3f}_{prim.restitution:.3f}"
    return "contact_" + token.replace(".", "p")


def _author_physics_materials(
    runtime: SimpleNamespace, stage: Any, plan: ScenePlan
) -> dict[str, Any]:
    materials: dict[str, Any] = {}
    for prim in plan.prims:
        if not prim.is_geometry:
            continue
        key = _physics_material_key(prim)
        if key in materials:
            continue
        material = runtime.UsdShade.Material.Define(stage, f"{PHYSICS_MATERIALS_PATH}/{key}")
        api = runtime.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(prim.static_friction))
        api.CreateDynamicFrictionAttr().Set(float(prim.dynamic_friction))
        api.CreateRestitutionAttr().Set(float(prim.restitution))
        materials[key] = material
    return materials


def _material_colors(plan: ScenePlan) -> dict[str, tuple[float, float, float]]:
    return {
        name: tuple(float(channel) for channel in entry["base_color"])  # type: ignore[misc]
        for name, entry in plan.materials.items()
    }


def _apply_transform(runtime: SimpleNamespace, prim: Any, xformable: Any) -> None:
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(runtime.Gf.Vec3d(*[float(value) for value in prim.center]))
    if abs(prim.rotation_z_deg) > 1e-9:
        xformable.AddRotateZOp().Set(float(prim.rotation_z_deg))
    if prim.is_geometry and not prim.is_cylinder:
        xformable.AddScaleOp().Set(runtime.Gf.Vec3f(*[float(value) for value in prim.size]))


def _author_body(
    runtime: SimpleNamespace,
    stage: Any,
    prim: ScenePrim,
    physics_materials: dict[str, Any],
    colors: dict[str, tuple[float, float, float]],
) -> None:
    """Author a rigid-body root: one Xform carrying the whole item's mass."""

    path = f"{WORLD_PATH}/{prim.path}"
    _ensure_xform(runtime, stage, _parent_path(path))
    xform = runtime.UsdGeom.Xform.Define(stage, path)
    _apply_transform(runtime, prim, runtime.UsdGeom.Xformable(xform.GetPrim()))

    usd_prim = xform.GetPrim()
    usd_prim.CreateAttribute(
        "sim2sense:bodyMassKg", runtime.Sdf.ValueTypeNames.Double, custom=True
    ).Set(round(prim.computed_mass_kg(), 6))
    runtime.UsdPhysics.RigidBodyAPI.Apply(usd_prim)
    mass_api = runtime.UsdPhysics.MassAPI.Apply(usd_prim)
    mass_api.CreateMassAttr().Set(float(prim.computed_mass_kg()))
    if prim.density_kg_m3 > 0:
        mass_api.CreateDensityAttr().Set(float(prim.density_kg_m3))
    _author_metadata(runtime, usd_prim, prim)


def _author_prim(
    runtime: SimpleNamespace,
    stage: Any,
    prim: ScenePrim,
    visual_materials: dict[str, Any],
    physics_materials: dict[str, Any],
    colors: dict[str, tuple[float, float, float]],
) -> None:
    if prim.is_body:
        _author_body(runtime, stage, prim, physics_materials, colors)
        return

    path = f"{WORLD_PATH}/{prim.path}"
    _ensure_xform(runtime, stage, _parent_path(path))

    if prim.is_cylinder:
        geom = runtime.UsdGeom.Cylinder.Define(stage, path)
        geom.CreateRadiusAttr(float(prim.size[0] / 2))
        geom.CreateHeightAttr(float(prim.size[2]))
        geom.CreateAxisAttr(runtime.UsdGeom.Tokens.z)
    else:
        geom = runtime.UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)

    _apply_transform(runtime, prim, runtime.UsdGeom.Xformable(geom.GetPrim()))

    display = colors.get(prim.material or "", (0.7, 0.7, 0.7))
    geom.CreateDisplayColorAttr().Set([runtime.Gf.Vec3f(*display)])

    usd_prim = geom.GetPrim()
    if prim.collision:
        runtime.UsdPhysics.CollisionAPI.Apply(usd_prim)
    if prim.in_rigid_body and not prim.relative_to_body:
        runtime.UsdPhysics.RigidBodyAPI.Apply(usd_prim)

    binding = runtime.UsdShade.MaterialBindingAPI.Apply(usd_prim)
    visual = visual_materials.get(prim.material or "")
    if visual is not None:
        binding.Bind(visual)
    physics = physics_materials.get(_physics_material_key(prim))
    if physics is not None:
        binding.Bind(physics, runtime.UsdShade.Tokens.weakerThanDescendants, "physics")

    _author_metadata(runtime, usd_prim, prim)


def _author_metadata(runtime: SimpleNamespace, usd_prim: Any, prim: ScenePrim) -> None:
    value_types = runtime.Sdf.ValueTypeNames
    entries = (
        ("sim2sense:roomId", value_types.String, prim.room_id),
        ("sim2sense:category", value_types.String, prim.category),
        ("sim2sense:semantic", value_types.String, prim.semantic),
        ("sim2sense:materialName", value_types.String, prim.material or ""),
        ("sim2sense:physicsMode", value_types.String, prim.physics_mode),
        ("sim2sense:massKg", value_types.Double, round(prim.computed_mass_kg(), 6)),
        ("sim2sense:movable", value_types.Bool, prim.movable),
    )
    for name, value_type, value in entries:
        usd_prim.CreateAttribute(name, value_type, custom=True).Set(value)
    if prim.tags:
        usd_prim.CreateAttribute("sim2sense:tags", value_types.StringArray, custom=True).Set(
            list(prim.tags)
        )


def _author_lights(runtime: SimpleNamespace, stage: Any, plan: ScenePlan) -> None:
    for light in plan.lights:
        path = f"{WORLD_PATH}/{light.path}"
        _ensure_xform(runtime, stage, _parent_path(path))
        color = runtime.Gf.Vec3f(*[float(value) for value in light.color])
        if light.kind == "dome":
            prim = runtime.UsdLux.DomeLight.Define(stage, path)
            prim.CreateIntensityAttr(float(light.intensity))
            prim.CreateColorAttr(color)
            continue
        if light.kind == "distant":
            prim = runtime.UsdLux.DistantLight.Define(stage, path)
            prim.CreateIntensityAttr(float(light.intensity))
            prim.CreateColorAttr(color)
            prim.CreateAngleAttr(0.53)
            xformable = runtime.UsdGeom.Xformable(prim.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(runtime.Gf.Vec3d(*[float(v) for v in light.position]))
            xformable.AddRotateXYZOp().Set(
                runtime.Gf.Vec3f(*[float(v) for v in light.rotation_euler_deg])
            )
            continue
        prim = runtime.UsdLux.SphereLight.Define(stage, path)
        prim.CreateIntensityAttr(float(light.intensity))
        prim.CreateColorAttr(color)
        prim.CreateRadiusAttr(float(light.radius_m))
        xformable = runtime.UsdGeom.Xformable(prim.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(runtime.Gf.Vec3d(*[float(v) for v in light.position]))


def _author_physics_scene(runtime: SimpleNamespace, stage: Any) -> None:
    scene = runtime.UsdPhysics.Scene.Define(stage, f"{WORLD_PATH}/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(runtime.Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)


def build_stage(
    plan: ScenePlan,
    usd_path: str | Path,
    *,
    stage: Any | None = None,
    meters_per_unit: float = 1.0,
    up_axis: str = "Z",
) -> Path:
    """Author ``plan`` under ``/World`` and export the result to ``usd_path``.

    Pass ``stage`` to author into an existing stage -- for example the live
    Omniverse context stage, so the GUI viewport shows the result immediately and
    the physics engine can step it. When ``stage`` is ``None`` a standalone stage
    is created, which is the reproducible headless export path.
    """

    runtime = pxr_modules()
    validate_prim_paths(runtime, plan)
    target = Path(usd_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    created_here = stage is None
    if created_here:
        if target.exists():
            target.unlink()
        stage = runtime.Usd.Stage.CreateNew(str(target))

    runtime.UsdGeom.SetStageUpAxis(
        stage, runtime.UsdGeom.Tokens.z if up_axis.upper() == "Z" else runtime.UsdGeom.Tokens.y
    )
    runtime.UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    world = runtime.UsdGeom.Xform.Define(stage, WORLD_PATH)
    stage.SetDefaultPrim(world.GetPrim())
    runtime.Usd.ModelAPI(world.GetPrim()).SetKind("assembly")

    visual_materials = _author_visual_materials(runtime, stage, plan)
    physics_materials = _author_physics_materials(runtime, stage, plan)
    _author_physics_scene(runtime, stage)

    colors = _material_colors(plan)
    for prim in plan.prims:
        _author_prim(runtime, stage, prim, visual_materials, physics_materials, colors)
    _author_lights(runtime, stage, plan)

    # ``Sdf.Layer.Export`` to the layer's own identifier is a silent no-op and
    # leaves an empty file behind, so a stage we created from a path must be
    # saved rather than exported. An inherited (anonymous) stage still needs
    # Export, which is how the GUI path writes its copy.
    root_layer = stage.GetRootLayer()
    if created_here:
        root_layer.Save()
    else:
        root_layer.Export(str(target))

    if not target.is_file() or target.stat().st_size <= len("#usda 1.0\n") + 1:
        raise RuntimeError(
            f"USD export produced an empty layer at {target}; the scene was not written"
        )

    LOGGER.info(
        "authored %d prims, %d visual materials, %d contact materials, %d lights into %s",
        len(plan.prims),
        len(visual_materials),
        len(physics_materials),
        len(plan.lights),
        target,
    )
    return target


def activate_physics() -> dict[str, Any]:
    """Attach PhysX to the current stage and switch on result write-back.

    A standalone Isaac Sim session does **not** wire PhysX to the USD stage by
    itself. The timeline advances and ``get_num_physics_steps()`` climbs, but no
    actor is created and no transform is ever written back -- the scene looks
    frozen and every "settled" assertion passes vacuously. Enabling the default
    simulation-manager callbacks and running the simulation setup hooks attaches
    the stage, after which gravity, contacts and the ``physics:*`` materials on
    the scene actually take effect.

    Returns a small report so the caller can log which scene was activated.
    """

    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except ImportError as exc:  # pragma: no cover - depends on the host runtime
        raise IsaacRuntimeUnavailable(
            "isaacsim.core.simulation_manager is unavailable; physics cannot be activated"
        ) from exc

    SimulationManager.enable_all_default_callbacks()
    SimulationManager.setup_simulation()
    return {
        "physics_scenes": [str(scene.path) for scene in SimulationManager.get_physics_scenes()],
        "active_engine": SimulationManager.get_active_physics_engine(),
        "physics_dt": SimulationManager.get_physics_dt(),
        "simulating": SimulationManager.is_simulating(),
    }


def step_simulation(app: Any, seconds: float) -> int:
    """Play the timeline and advance the app for ``seconds``, returning step count.

    ``app.update()`` must drive the loop: with ``omni.kit.loop-isaac`` active each
    update advances the physics clock by one ``physics_dt``.
    """

    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager

    dt = SimulationManager.get_physics_dt() or (1.0 / 60.0)
    frames = max(1, int(round(seconds / dt)))
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(frames):
        app.update()
    timeline.pause()
    return SimulationManager.get_num_physics_steps()


def lift_rigid_bodies(source: Any, delta_z: float) -> dict[str, list[float]]:
    """Raise every rigid body by ``delta_z`` metres and report the new positions.

    Used as a positive control: a body raised off the floor must fall back to it,
    which is the only way to distinguish "the scene is stable" from "physics is
    not running at all".
    """

    runtime = pxr_modules()
    if isinstance(source, runtime.Usd.Stage):
        stage = source
    else:
        stage = runtime.Usd.Stage.Open(str(source))
    if stage is None:
        raise FileNotFoundError(f"could not open stage: {source}")

    edited: dict[str, list[float]] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(runtime.UsdPhysics.RigidBodyAPI):
            continue
        xformable = runtime.UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        current = translate_op.Get() or runtime.Gf.Vec3d(0.0, 0.0, 0.0)
        lifted = runtime.Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]) + delta_z)
        translate_op.Set(lifted)
        edited[str(prim.GetPath())] = [round(float(v), 4) for v in lifted]
    return edited


def rigid_body_world_positions(source: Any) -> dict[str, list[float]]:
    """World-space translation of every rigid body, keyed by prim path.

    Used to check that dynamic furniture settles instead of drifting, sinking or
    exploding during the first seconds of simulation.
    """

    runtime = pxr_modules()
    if isinstance(source, runtime.Usd.Stage):
        stage = source
    else:
        stage = runtime.Usd.Stage.Open(str(source))
    if stage is None:
        raise FileNotFoundError(f"could not open stage: {source}")
    cache = runtime.UsdGeom.XformCache(runtime.Usd.TimeCode.Default())
    positions: dict[str, list[float]] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(runtime.UsdPhysics.RigidBodyAPI):
            continue
        translation = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        positions[str(prim.GetPath())] = [round(float(value), 4) for value in translation]
    return positions


def stage_summary(usd_path: str | Path) -> dict[str, Any]:
    """Re-open a saved stage and report what is actually on it."""

    runtime = pxr_modules()
    stage = runtime.Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"could not open stage: {usd_path}")

    type_counts: dict[str, int] = {}
    colliders = 0
    rigid_bodies = 0
    shapes = 0
    semantic_counts: dict[str, int] = {}
    for prim in stage.Traverse():
        type_name = prim.GetTypeName() or "unknown"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        if prim.HasAPI(runtime.UsdPhysics.CollisionAPI):
            colliders += 1
        if prim.HasAPI(runtime.UsdPhysics.RigidBodyAPI):
            rigid_bodies += 1
        if prim.IsA(runtime.UsdGeom.Cube) or prim.IsA(runtime.UsdGeom.Cylinder):
            shapes += 1
        if prim.HasAttribute("sim2sense:semantic"):
            label = prim.GetAttribute("sim2sense:semantic").Get()
            if label:
                semantic_counts[str(label)] = semantic_counts.get(str(label), 0) + 1

    default_prim = stage.GetDefaultPrim()
    return {
        "path": str(usd_path),
        "up_axis": str(runtime.UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(runtime.UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(default_prim.GetPath()) if default_prim else "",
        "type_counts": dict(sorted(type_counts.items())),
        "prim_count": sum(type_counts.values()),
        "shape_count": shapes,
        "collider_count": colliders,
        "rigid_body_count": rigid_bodies,
        "semantic_counts": dict(sorted(semantic_counts.items())),
    }
