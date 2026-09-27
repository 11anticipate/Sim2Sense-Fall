#!/usr/bin/env python3
"""Author the human rig into the fixed indoor scene and export it as USD.

The rig is planned on CPU *before* Isaac Sim boots, so an incoherent rig fails
immediately and never leaves a half-authored stage behind. The base scene is copied
first, so exporting a human never rewrites the verified apartment artifact.

    python3 scripts/humans/build.py --dry-run
    ~/isaacsim/python.sh scripts/humans/build.py --headless
    ~/isaacsim/python.sh scripts/humans/build.py --headless --root-mode anchored
    ~/isaacsim/python.sh scripts/humans/build.py --gui --settle-seconds 3

With ``--gui`` the viewport stays open so the articulated body can be inspected;
``--settle-seconds`` steps physics before handing over.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _directory in (SRC_DIR, SCRIPTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from dataclasses import replace  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_ASSETS,
    DEFAULT_CONFIG,
    DEFAULT_MOTIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE,
    DEFAULT_SCENE_CONFIG,
    Checks,
    activate_physics,
    boot_isaac,
    load_inputs,
    open_scene,
    resolve_spawn_point,
    step_simulation,
    write_json,
)

from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)

LOGGER = logging.getLogger("build_human")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Author and export the articulated human.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--motions", type=Path, default=DEFAULT_MOTIONS)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="base scene USD")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="human_indoor_apartment", help="output file stem")
    parser.add_argument("--height", type=float, default=None, help="override standing height, m")
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=DEFAULT_SCENE_CONFIG,
        help="scene YAML used to derive a spawn point on a floor slab",
    )
    parser.add_argument(
        "--spawn-x",
        type=float,
        default=None,
        help="root spawn x in the scene, metres; omit to derive one inside the largest room",
    )
    parser.add_argument("--spawn-y", type=float, default=None, help="root spawn y in the scene")
    parser.add_argument(
        "--root-mode",
        choices=("free", "anchored"),
        default=None,
        help="override rig.root_mode; 'anchored' fixes the pelvis to the world",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only; never starts Isaac Sim")
    parser.add_argument(
        "--view",
        choices=("human", "top", "roofless", "exterior"),
        default="human",
        help="GUI inspection view; human frames the body and hides the ceiling",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", dest="headless", action="store_true", default=True)
    mode.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument(
        "--settle-seconds", type=float, default=0.0, help="physics time to run before saving"
    )
    args = parser.parse_args(argv)
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    checks = Checks()
    try:
        config, registry, _motions = load_inputs(
            config_path=args.config,
            assets_path=args.assets,
            motions_path=args.motions,
            height_m=args.height,
        )
    except (OSError, ValueError) as exc:
        checks.check("inputs load and validate", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="human build")

    if args.root_mode is not None:
        config = replace(config, rig=replace(config.rig, root_mode=args.root_mode))

    spawn = resolve_spawn_point(
        args.scene_config, args.spawn_x, args.spawn_y,
        standing_height_m=config.skeleton.height_m,
    )
    body = select_body(
        registry,
        model_id=config.skeleton.model_asset,
        allow_procedural=bool(config.skeleton.allow_procedural_skeleton),
    )
    rest = (
        fit_rest_skeleton(config, body.model.mesh().rest_skeleton())
        if body.has_skin_mesh
        else None
    )
    plan = plan_human_rig(config, rest=rest, spawn_xy=spawn)
    mesh = body.model.mesh() if body.has_skin_mesh else None
    if mesh is not None:
        from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints

        mesh = fit_mesh_to_rest_joints(mesh, plan.rest_joint_positions)
        display_values = config.visualization.pose_dict()
        display_poses = forward_kinematics(
            plan,
            display_values,
            root_position=(0.0, 0.0, 0.0),
        )
        from sim2sense_fall.humans.mesh_sequence import skin_mesh_sequence_frame

        posed_vertices = skin_mesh_sequence_frame(mesh, display_poses)
    else:
        display_values = config.visualization.pose_dict()
        posed_vertices = None
    checks.check(
        "rig planned on CPU before starting the simulator",
        plan.stats["link_count"] == config.topology.joint_count,
        f"{plan.stats['link_count']} links, {plan.stats['dof_count']} DOF, "
        f"{plan.stats['collider_count']} capsules, standing {plan.stats['standing_height_m']} m",
    )
    checks.check(
        "root mode is explicit",
        plan.root_mode in ("free", "anchored"),
        f"{plan.root_mode}; anchored mode adds a fixed joint from the world to the pelvis",
    )
    if args.dry_run:
        checks.skip("USD authoring", "--dry-run")
        checks.skip("physics settle", "--dry-run")
        path = write_json(args.out / f"{args.name}.plan.json", plan.as_dict())
        checks.info(f"wrote {path}")
        return checks.report(banner="human build (dry run)")

    scene = args.scene.resolve()
    if not scene.is_file():
        checks.check("base scene exists", False, str(scene))
        return checks.report(banner="human build")
    scene_digest = _sha256(scene)

    app = boot_isaac(args.headless)
    if app is None:
        checks.check(
            "Isaac Sim started",
            False,
            "isaacsim is not importable; run through ~/isaacsim/python.sh",
        )
        return checks.report(banner="human build")
    exit_code = 1
    try:
        from sim2sense_fall.humans.usd_human import build_human_stage, human_stage_summary

        target = args.out / f"{args.name}.usda"
        build_human_stage(
            plan,
            target,
            base_scene=scene,
            spawn_position=plan.spawn_root_position,
            skin_points=posed_vertices,
            skin_faces=None if mesh is None else mesh.faces,
        )
        summary = human_stage_summary(target)
        checks.check("human stage written and reopens", summary["capsule_count"] > 0, str(target))
        checks.check(
            "articulation has exactly one root",
            summary["articulation_root_count"] == 1,
            f"{summary['articulation_root_count']} articulation roots",
        )
        checks.check(
            "revolute joints match the plan",
            summary["revolute_joint_count"] == len(plan.joints),
            f"{summary['revolute_joint_count']} revolute joints for {len(plan.joints)} plan DOF",
        )
        checks.check(
            "fixed joints match the plan",
            summary["fixed_joint_count"] >= len(plan.fixed_joints),
            f"{summary['fixed_joint_count']} fixed joints (plan {len(plan.fixed_joints)}"
            + (", plus the world anchor" if plan.root_mode == "anchored" else "")
            + ")",
        )
        checks.check(
            "capsule colliders match the plan",
            summary["capsule_count"] == plan.stats["collider_count"],
            f"{summary['capsule_count']} capsules for {plan.stats['collider_count']} plan links",
        )
        checks.check(
            "capsule proxies are hidden but retained for collision",
            summary["invisible_capsule_count"] == summary["capsule_count"],
            f"{summary['invisible_capsule_count']} of {summary['capsule_count']} capsules hidden",
        )
        checks.check(
            "stage is metres, Z-up",
            summary["meters_per_unit"] == 1.0 and summary["up_axis"] == "Z",
            f"{summary['meters_per_unit']} m/unit, {summary['up_axis']}-up",
        )

        stage = open_scene(app, target)
        activation = activate_physics()
        checks.check(
            "physics scene activated",
            activation["physics_scenes"] == ["/World/PhysicsScene"]
            and activation["active_engine"] == "physx",
            str(activation),
        )
        if not args.headless:
            # The posed SMPL surface and collision links must begin from the same
            # DOF values.  Targets are also set so a zero-settle inspection does
            # not immediately return to the horizontal SMPL rest pose.
            from sim2sense_fall.humans.usd_human import HumanRuntime

            runtime = HumanRuntime(stage, plan, enable_contact_views=False)
            runtime.play()
            # ``play()`` only starts the timeline. Isaac's tensor-backed
            # articulation view becomes valid after Kit has processed a few
            # updates; writing DOFs before that raises "physics tensor entity
            # is not valid" on GUI startup.
            for _ in range(4):
                app.update()
            ordered_values = [display_values.get(name, 0.0) for name in plan.dof_names]
            runtime.set_joint_positions(ordered_values)
            runtime.set_joint_targets(ordered_values)
            checks.info("display pose applied to PhysX DOFs")
        if args.settle_seconds > 0:
            steps = step_simulation(app, args.settle_seconds)
            checks.info(f"settled {args.settle_seconds:g} s of physics ({steps} steps)")
        if not args.headless:
            from sim2sense_fall.scenes.view import apply_inspection_view

            apply_inspection_view(
                stage, mode=args.view, aspect_ratio=1600 / 900, require_viewport=True
            )
            checks.info(f"inspection view: {args.view} (temporary session layer)")
            print("GUI is open. Close the window or press Ctrl-C to exit.")
            while app.is_running():
                app.update()
        del stage

        checks.check(
            "the verified base scene was not modified",
            _sha256(scene) == scene_digest,
            str(scene),
        )
        path = write_json(
            args.out / f"{args.name}.build.json",
            {
                "human_id": plan.human_id,
                "root_mode": plan.root_mode,
                "spawn_root_position": list(plan.spawn_root_position),
                "base_scene": str(scene),
                "base_scene_sha256": scene_digest,
                "output_usd": str(target),
                "output_usd_sha256": _sha256(target),
                "stage_summary": summary,
                "plan_stats": dict(plan.stats),
            },
        )
        checks.info(f"wrote {path}")
        exit_code = checks.report(banner="human build")
    except Exception as exc:  # noqa: BLE001 - surfaced in the report
        LOGGER.exception("human build failed")
        checks.check("build completed without error", False, f"{type(exc).__name__}: {exc}")
        exit_code = checks.report(banner="human build")
    finally:
        app.close(exit_code=exit_code)
    return exit_code


def _sha256(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
