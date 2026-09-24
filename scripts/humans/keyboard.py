#!/usr/bin/env python3
"""Drive an assisted SMPL human with W/S, A/D, Space, R and Escape."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    DEFAULT_MOTIONS,
    REPO_ROOT,
    activate_physics,
    boot_isaac,
    load_inputs,
    open_scene,
    set_physics_dt,
    write_json,
)

from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints, skin_mesh_sequence_frame
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig
from sim2sense_fall.humans.root_control import RootAssistConfig, root_wrench
from sim2sense_fall.humans.rotations import quaternion_to_matrix
from sim2sense_fall.humans.teleop import (
    KeyboardIntent,
    TeleopController,
    load_gait,
    load_keyboard_config,
)
from sim2sense_fall.humans.usd_human import (
    DISPLAY_SKIN_PATH,
    HumanRuntime,
    author_contact_reporting,
    build_human_stage,
    write_display_skin,
)
from sim2sense_fall.scenes.view import apply_inspection_view

LOGGER = logging.getLogger("human_keyboard")


def prepare(path: Path) -> tuple[dict[str, Any], Any, Any, Any, TeleopController]:
    settings = load_keyboard_config(path, REPO_ROOT)
    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    settings["model"] = {"id": body.model_id, "sha256": body.model.source_sha256}
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    dt = config.simulation.physics_dt_s
    gaits = {key: load_gait(spec, plan, dt_s=dt) for key, spec in settings["gaits"].items()}
    idle = load_gait(settings["idle"], plan, dt_s=dt)
    controller = TeleopController(
        settings["controller"], gaits, idle, plan, np.deg2rad(settings["heading_deg"])
    )
    return settings, config, plan, mesh, controller


def demo_keys(settings: dict[str, Any], elapsed_s: float) -> set[str]:
    end = 0.0
    for segment in settings["demo"]:
        end += segment["duration_s"]
        if elapsed_s < end:
            return set(segment["keys"])
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/humans/keyboard")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--demo", action="store_true", help="deterministic keyboard event rehearsal"
    )
    parser.add_argument("--seconds", type=float, default=0.0, help="0 keeps GUI open until Escape")
    parser.add_argument(
        "--capture", action="store_true", help="save live viewport at demo boundaries"
    )
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    if args.headless and not args.demo and args.seconds == 0 and not args.dry_run:
        parser.error("headless mode needs --demo or --seconds")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings, config, plan, mesh, controller = prepare(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    assistance = RootAssistConfig.load(settings["root_assist"])
    scene = args.scene or settings["scene"]
    provenance = {
        "mode": "keyboard_bounded_external_root_assistance",
        "unassisted": False,
        "seed": config.export.surface_point_seed,
        "scene_split": "single_fixed_apartment_no_train_test_split",
        "rig_sha256": hashlib.sha256(settings["rig"].read_bytes()).hexdigest(),
        "keyboard_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "root_assist_sha256": hashlib.sha256(settings["root_assist"].read_bytes()).hexdigest(),
        "gaits": {key: gait.provenance for key, gait in controller.gaits.items()},
        "model": settings["model"],
        "idle": controller.idle.provenance,
        "root_assistance": asdict(assistance),
        "physics_dt_s": config.simulation.physics_dt_s,
        "tracking_tolerance_deg": config.control.tracking_tolerance_deg,
        "skin_penetration_tolerance_m": settings["skin_penetration_tolerance_m"],
        "ordinary_motion_teleports": 0,
        "reset_count": 0,
    }
    if args.dry_run:
        intent = KeyboardIntent()
        position = controller.position.copy()
        for frame in range(1200):
            keys = demo_keys(settings, frame * config.simulation.physics_dt_s)
            for key in tuple(intent.held - keys):
                intent.event(key, False)
            for key in keys - intent.held:
                intent.event(key, True)
            target = controller.advance(
                intent.command(), config.simulation.physics_dt_s, position, controller.heading
            )
            position = target.position
            if not np.isfinite(target.joints).all():
                raise RuntimeError("nonfinite dry-run target")
        write_json(
            args.out / "dry_run.json",
            {
                **provenance,
                "physics_run": False,
                "final_target": position.tolist(),
                "status": "cpu_targets_only",
            },
        )
        LOGGER.info("CPU keyboard target rehearsal passed")
        return 0
    app = boot_isaac(args.headless, width=1280, height=960)
    if app is None:
        raise RuntimeError("Isaac Sim is unavailable; use ~/isaacsim/python.sh or --dry-run")
    callback_id = subscription = input_interface = keyboard = None
    records: deque[dict[str, Any]] = deque(maxlen=int(settings["record_frames"]))
    skin_capacity = (
        int(
            np.ceil(
                settings["record_frames"] * config.simulation.physics_dt_s * settings["render_hz"]
            )
        )
        + 1
    )
    skins: deque[np.ndarray] = deque(maxlen=skin_capacity)
    skin_times: deque[float] = deque(maxlen=skin_capacity)
    errors: list[str] = []
    captures: list[dict[str, Any]] = []
    intent = KeyboardIntent()
    clock_s = 0.0
    exit_code = 1
    runtime = None
    try:
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, UsdGeom, Vt

        build_human_stage(
            plan,
            args.out / "human_keyboard.usda",
            base_scene=scene,
            skin_points=mesh.vertices,
            skin_faces=mesh.faces,
        )
        stage = open_scene(app, args.out / "human_keyboard.usda")
        # Create all visual structure before the tensor views. Only points change later.
        write_display_skin(stage, mesh.vertices, mesh.faces)
        camera_path = apply_inspection_view(stage, mode="roofless", aspect_ratio=4 / 3)
        stage.SetEditTarget(stage.GetSessionLayer())
        camera = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
        camera.ClearXformOpOrder()
        camera_op = camera.AddTransformOp()
        UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetFocalLengthAttr().Set(32.0)
        points_attr = UsdGeom.Mesh.Get(stage, DISPLAY_SKIN_PATH).GetPointsAttr()
        author_contact_reporting(stage)
        activate_physics()
        measured_dt, _ = set_physics_dt(config.simulation.physics_dt_s)
        if abs(measured_dt - config.simulation.physics_dt_s) > 1e-9:
            raise RuntimeError("configured physics timestep was not applied")
        runtime = HumanRuntime(stage, plan, enable_contact_views=False, app=app)
        runtime.play()
        for _ in range(4):
            app.update()

        def reset() -> None:
            controller.reset(
                np.asarray(plan.spawn_root_position), np.deg2rad(settings["heading_deg"])
            )
            target = controller.advance(
                (0.0, 0.0), measured_dt, controller.position, controller.heading
            )
            runtime.set_root_pose(target.position, target.quaternion)
            runtime.set_joint_positions(target.joints)
            runtime.set_joint_targets(target.joints)
            runtime.reset_velocities()
            intent.clear()
            intent.reset_requested = False

        reset()
        status_window = ui.Window("Human", width=280, height=140)
        with status_window.frame:
            with ui.VStack(spacing=6):
                ui.Label("External root assistance: ON")
                state_label = ui.Label("State: stand")
                speed_label = ui.Label("Speed: 0.00 m/s")
                clock_label = ui.Label("Physics: 0.00 s")

        def on_key(event: Any, *_args: Any) -> bool:
            key = event.input.name
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                intent.event(key, True)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                intent.event(key, False)
            return key not in KeyboardIntent.KEYS

        window = omni.appwindow.get_default_app_window()
        keyboard = window.get_keyboard()
        input_interface = carb.input.acquire_input_interface()
        subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)
        input_provider = carb.input.acquire_input_provider()
        buffered_keys: set[str] = set()

        def before_step(dt: float, _context: Any) -> None:
            nonlocal clock_s
            if errors:
                return
            try:
                position, quaternion = runtime.root_pose()
                rotation = quaternion_to_matrix(quaternion)
                target = controller.advance(
                    intent.command(),
                    dt,
                    position,
                    float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                )
                linear, angular = runtime.root_velocities()
                force, torque = root_wrench(
                    assistance,
                    position=position,
                    quaternion=quaternion,
                    linear_velocity=linear,
                    angular_velocity=angular,
                    target_position=target.position,
                    target_quaternion=target.quaternion,
                    target_linear_velocity=target.linear_velocity,
                    target_angular_velocity=target.angular_velocity,
                    mass_kg=plan.total_mass_kg,
                    gravity_m_s2=config.simulation.gravity_m_s2,
                )
                runtime.set_joint_targets(target.joints, target.joint_velocities)
                runtime.apply_root_wrench(force, torque)
                attributed = runtime.attributed_contact_samples()
                contacts = [sample for sample, _ in attributed]
                slips = []
                for sample, limb in attributed:
                    if (
                        limb in {"left_ankle", "right_ankle", "left_foot", "right_foot"}
                        and sample.collider1_path.endswith("/floor")
                        and sample.impulse_magnitude_ns >= settings["slip_min_impulse_ns"]
                    ):
                        normal = np.asarray(sample.normal)
                        normal /= max(float(np.linalg.norm(normal)), 1e-12)
                        velocity = runtime.link_point_velocity(limb, np.asarray(sample.position_m))
                        tangent = velocity - normal * np.dot(normal, velocity)
                        slips.append(float(np.linalg.norm(tangent)))
                records.append(
                    {
                        "time_s": clock_s,
                        "root": position.copy(),
                        "root_quaternion": quaternion.copy(),
                        "target": target.position,
                        "joints": runtime.joint_positions_rad(),
                        "joint_target": target.joints,
                        "force": force,
                        "torque": torque,
                        "velocity": linear,
                        "command": intent.command(),
                        "mode": target.mode,
                        "contacts": [s.collider1_path for s in contacts],
                        "contact_detail": [s.as_dict() for s in contacts],
                        "floor_contact_slips_m_s": slips,
                        "impulse_ns": sum(float(np.linalg.norm(s.impulse_ns)) for s in contacts),
                    }
                )
                clock_s += dt
            except Exception as exc:
                LOGGER.exception("physics keyboard callback failed")
                errors.append(f"{type(exc).__name__}: {exc}")

        callback_id = SimulationManager.register_callback(
            before_step, SimulationEvent.PHYSICS_PRE_STEP
        )
        limit_s = args.seconds or (
            sum(s["duration_s"] for s in settings["demo"]) if args.demo else float("inf")
        )
        capture_times = (
            list(np.cumsum([s["duration_s"] for s in settings["demo"]])) if args.capture else []
        )
        next_render = 0.0
        pending: list[tuple[Any, Path, float]] = []
        start_steps = SimulationManager.get_num_physics_steps()
        wall_start = time.monotonic()
        last_progress_wall = wall_start
        while app.is_running() and clock_s < limit_s and not intent.quit_requested and not errors:
            if args.demo:
                keys = demo_keys(settings, clock_s)
                for key in buffered_keys - keys:
                    input_provider.buffer_keyboard_key_event(
                        keyboard,
                        carb.input.KeyboardEventType.KEY_RELEASE,
                        getattr(carb.input.KeyboardInput, key),
                        0,
                    )
                for key in keys - buffered_keys:
                    input_provider.buffer_keyboard_key_event(
                        keyboard,
                        carb.input.KeyboardEventType.KEY_PRESS,
                        getattr(carb.input.KeyboardInput, key),
                        0,
                    )
                buffered_keys = keys
            elif not window.is_focused():
                intent.clear()
            if SimulationManager.is_paused():
                intent.clear()
            if intent.reset_requested:
                reset()
                provenance["reset_count"] += 1
            previous_time = clock_s
            app.update()
            if clock_s > previous_time or SimulationManager.is_paused():
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 15:
                raise RuntimeError("physics callback did not advance for 15 wall-clock seconds")
            if clock_s >= next_render:
                vertices = skin_mesh_sequence_frame(mesh, runtime.link_poses())
                points_attr.Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                skins.append(vertices.astype(np.float32))
                skin_times.append(clock_s)
                position, _ = runtime.root_pose()
                target_view = np.array([position[0], position[1], 0.8])
                el, az = np.deg2rad(
                    [settings["camera_elevation_deg"], settings["camera_azimuth_deg"]]
                )
                eye = target_view + settings["camera_distance_m"] * np.array(
                    [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
                )
                camera_op.Set(
                    Gf.Matrix4d()
                    .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target_view), Gf.Vec3d(0, 0, 1))
                    .GetInverse()
                )
                next_render = clock_s + 1 / settings["render_hz"]
                state_label.text = f"State: {records[-1]['mode']}" if records else "State: stand"
                speed_label.text = f"Speed: {abs(controller.speed):.2f} m/s (target)"
                clock_label.text = f"Physics: {clock_s:.2f} s"
            if capture_times and clock_s >= capture_times[0]:
                capture_times.pop(0)
                output = args.out / f"live_{clock_s:06.2f}.png"
                capture = capture_viewport_to_file(get_active_viewport(), file_path=str(output))
                pending.append((asyncio.ensure_future(capture.wait_for_result()), output, clock_s))
            if not args.headless:
                time.sleep(max(0.0, min(0.02, clock_s - (time.monotonic() - wall_start))))
        runtime.pause()
        deadline = time.monotonic() + 15
        while pending and time.monotonic() < deadline:
            app.update()
            for item in tuple(pending):
                task, path, capture_s = item
                if task.done():
                    task.result()
                    if path.is_file() and path.stat().st_size:
                        captures.append({"path": str(path.resolve()), "request_time_s": capture_s})
                        pending.remove(item)
        if pending:
            errors.append("live screenshot did not flush")
        provenance["measured_physics_steps"] = int(
            SimulationManager.get_num_physics_steps() - start_steps
        )
        provenance["callback_steps"] = int(round(clock_s / measured_dt))
        provenance["retained_control_frames"] = len(records)
        provenance["keyboard_event_route"] = (
            "carb_input_provider" if args.demo else "window_keyboard"
        )
        provenance["simulation_time_s"] = clock_s
        provenance["wall_time_s"] = time.monotonic() - wall_start
        if not records or not skins:
            errors.append("no physics or display records")
        if provenance["measured_physics_steps"] != provenance["callback_steps"]:
            errors.append("physics step count and control callback count disagree")
        exit_code = int(bool(errors))
    except Exception as exc:
        LOGGER.exception("keyboard simulation failed")
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if callback_id is not None:
            SimulationManager.deregister_callback(callback_id)
        if subscription is not None:
            input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
        if records:
            arrays = {
                key: np.asarray([row[key] for row in records])
                for key in (
                    "time_s",
                    "root",
                    "root_quaternion",
                    "target",
                    "joints",
                    "joint_target",
                    "force",
                    "torque",
                    "velocity",
                    "command",
                    "mode",
                    "impulse_ns",
                )
            }
            np.savez_compressed(args.out / "control.npz", **arrays)
            provenance["metrics_scope"] = "retained_record_window"
            provenance["retained_time_interval_s"] = [
                float(arrays["time_s"][0]),
                float(arrays["time_s"][-1]),
            ]
            provenance["joint_error_max_deg"] = float(
                np.rad2deg(np.abs(arrays["joints"] - arrays["joint_target"])).max()
            )
            provenance["force_peak_n"] = float(np.linalg.norm(arrays["force"], axis=1).max())
            provenance["contact_paths"] = sorted({p for row in records for p in row["contacts"]})
            provenance["contact_samples"] = sum(len(row["contacts"]) for row in records)
            slips = [v for row in records for v in row["floor_contact_slips_m_s"]]
            provenance["floor_contact_slip"] = {
                "definition": "body COM linear + omega cross lever arm, tangent to static floor",
                "minimum_impulse_ns": settings["slip_min_impulse_ns"],
                "sample_count": len(slips),
                "p95_m_s": None if not slips else float(np.percentile(slips, 95)),
                "max_m_s": None if not slips else float(max(slips)),
                "tolerance_m_s": settings["slip_speed_tolerance_m_s"],
                "fraction_above_tolerance": None
                if not slips
                else float(np.mean(np.asarray(slips) > settings["slip_speed_tolerance_m_s"])),
            }
            write_json(
                args.out / "contacts.json",
                [
                    {
                        "time_s": row["time_s"],
                        "environment_paths": row["contacts"],
                        "samples": row["contact_detail"],
                        "floor_contact_slips_m_s": row["floor_contact_slips_m_s"],
                        "total_impulse_ns": row["impulse_ns"],
                    }
                    for row in records
                ],
            )
        if skins:
            while records and skin_times[0] < records[0]["time_s"] and len(skins) > 1:
                skins.popleft()
                skin_times.popleft()
            np.savez_compressed(
                args.out / "recording.npz",
                time_s=np.asarray(skin_times),
                mesh_vertices_xyz=np.stack(skins),
                mesh_faces=mesh.faces,
            )
            provenance["skin_min_z_m"] = float(min(skin[:, 2].min() for skin in skins))
        acceptance = {
            "joint_tracking": bool(records)
            and provenance["joint_error_max_deg"] <= config.control.tracking_tolerance_deg,
            "skin_ground_clearance": bool(skins)
            and provenance["skin_min_z_m"] >= -settings["skin_penetration_tolerance_m"],
            "floor_slip": bool(records)
            and bool(provenance["floor_contact_slip"]["sample_count"])
            and provenance["floor_contact_slip"]["p95_m_s"] <= settings["slip_speed_tolerance_m_s"],
        }
        write_json(
            args.out / "report.json",
            {
                **provenance,
                "errors": errors,
                "captures": captures,
                "runtime_completed": exit_code == 0,
                "quality_gates": acceptance,
                "motion_accuracy_accepted": exit_code == 0 and all(acceptance.values()),
                "note": "runtime completion alone does not establish motion or contact accuracy",
            },
        )
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
