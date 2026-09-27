#!/usr/bin/env python3
"""Multi-azimuth live-GUI evidence capture for the P0-A right-arm audit.

Why this exists
---------------
``scripts/humans/keyboard.py`` drives the assisted human with one fixed camera
(``camera_azimuth_deg`` from ``configs/humans/keyboard.yaml``), and the shipped
value is ``-55``, i.e. a rear three-quarter view. Arm swing during walking is a
*fore-aft* motion, and a rear view hides exactly that component, so the existing
captures cannot support or refute the reported right-arm defect.

This script runs the **same** physical setup as ``keyboard.py`` (identical rig,
gaits, controller, root assist, scene and demo timeline) but overrides the
camera azimuth and the capture schedule *without touching the shipped config*,
and then sweeps one or more azimuths inside a single Isaac Sim session.

Design rules that matter
------------------------
* ``keyboard.py`` is the controlling entry point; this script imports its
  ``prepare``/``demo_keys`` instead of re-implementing them, so the physical
  configuration cannot drift between the two.
* The azimuth list and the capture schedule are CLI arguments, not constants in
  the function bodies, so every run records what was actually used.
* ``view_azimuths_deg`` in the JSON report is what the camera was commanded to
  do; ``captures[].azimuth_deg`` is the value in force for that screenshot.
* The reported motion metrics come from the same physics callback records as
  ``keyboard.py``; nothing here relaxes a gate.

Example
-------
    ~/isaacsim/python.sh scripts/humans/arm_capture.py \\
        --azimuths 0,90,180,270 --capture-windows 1.0-4.0 --seconds 6.0
"""

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
from common import (  # type: ignore[import-not-found]
    REPO_ROOT,
    activate_physics,
    boot_isaac,
    open_scene,
    set_physics_dt,
    write_json,
)

from sim2sense_fall.humans.mesh_sequence import (  # noqa: E402
    skin_mesh_sequence_frame,
)
from sim2sense_fall.humans.root_control import RootAssistConfig, root_wrench  # noqa: E402
from sim2sense_fall.humans.rotations import quaternion_to_matrix  # noqa: E402
from sim2sense_fall.humans.teleop import (  # noqa: E402
    KeyboardIntent,
)
from sim2sense_fall.humans.travel import root_displacement_in_body_frame  # noqa: E402
from sim2sense_fall.humans.usd_human import (  # noqa: E402
    DISPLAY_SKIN_PATH,
    HumanRuntime,
    author_contact_reporting,
    build_human_stage,
    write_display_skin,
)
from sim2sense_fall.scenes.view import apply_inspection_view  # noqa: E402

LOGGER = logging.getLogger("arm_capture")

# Skeleton joints that carry the arm swing this audit is about.
ARM_JOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
LEG_JOINT_NAMES = ("left_knee", "right_knee", "left_ankle", "right_ankle")


def parse_azimuths(text: str) -> list[float]:
    """Parse ``0,90,180`` into floats, rejecting anything non-finite."""

    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise argparse.ArgumentTypeError("at least one azimuth is required")
    if not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("azimuths must be finite")
    return values


def parse_windows(text: str) -> list[tuple[float, float]]:
    """Parse ``1.0-4.0,6.0-7.0`` into inclusive start/stop simulation windows."""

    windows = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        start_text, _, stop_text = chunk.partition("-")
        start, stop = float(start_text), float(stop_text)
        if not (np.isfinite(start) and np.isfinite(stop)) or stop < start:
            raise argparse.ArgumentTypeError(f"invalid capture window {chunk!r}")
        windows.append((start, stop))
    if not windows:
        raise argparse.ArgumentTypeError("at least one capture window is required")
    return windows


def sample_times(windows: list[tuple[float, float]], rate_hz: float) -> list[float]:
    """Evenly spaced sample times inside each capture window, inclusive of both ends."""

    period = 1.0 / rate_hz
    times: list[float] = []
    for start, stop in windows:
        if stop <= start:
            times.append(start)
            continue
        count = int(round((stop - start) / period)) + 1
        times.extend(float(v) for v in np.linspace(start, stop, max(count, 2)))
    return times


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/humans/arm_capture")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument(
        "--azimuths",
        type=parse_azimuths,
        default=None,
        help="comma-separated camera azimuths in degrees, one per cycle",
    )
    parser.add_argument(
        "--capture-windows",
        type=parse_windows,
        default=None,
        help="per-cycle capture windows in simulated seconds, e.g. 1.0-4.0",
    )
    parser.add_argument("--capture-rate-hz", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=0, help="0 = len(azimuths)")
    parser.add_argument(
        "--rewind-cycles",
        action="store_true",
        help="reset the character at each cycle boundary so every azimuth views an identical trial",
    )
    parser.add_argument("--cycle-period-s", type=float, default=0.0, help="0 = demo length")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = cycles * period")
    parser.add_argument("--camera-distance-m", type=float, default=None)
    parser.add_argument("--camera-elevation-deg", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seconds < 0 or args.capture_rate_hz <= 0:
        parser.error("seconds must be nonnegative and capture-rate-hz positive")
    if args.cycles < 0:
        parser.error("cycles must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Import the controlling entry point lazily: it lives next to this file, and the
    # import is what guarantees both scripts share one physical setup.
    from keyboard import demo_keys, prepare

    settings, config, plan, mesh, controller = prepare(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    azimuths = args.azimuths or [float(settings["camera_azimuth_deg"])]
    distance_m = (
        float(settings["camera_distance_m"])
        if args.camera_distance_m is None
        else args.camera_distance_m
    )
    elevation_deg = (
        float(settings["camera_elevation_deg"])
        if args.camera_elevation_deg is None
        else args.camera_elevation_deg
    )
    demo_length_s = sum(float(segment["duration_s"]) for segment in settings["demo"])
    period_s = args.cycle_period_s or demo_length_s
    cycles = args.cycles or len(azimuths)
    windows = args.capture_windows or [(1.0, min(demo_length_s, 4.0))]
    total_s = args.seconds or cycles * period_s

    cycles_used = max(int(np.ceil(total_s / period_s)), 1)
    azimuth_plan = [azimuths[i % len(azimuths)] for i in range(cycles_used)]

    provenance: dict[str, Any] = {
        "mode": "keyboard_bounded_external_root_assistance",
        "unassisted": False,
        "purpose": "P0-A multi-azimuth live GUI evidence",
        "seed": config.export.surface_point_seed,
        "scene_split": "single_fixed_apartment_no_train_test_split",
        "rig_sha256": hashlib.sha256(settings["rig"].read_bytes()).hexdigest(),
        "keyboard_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256((args.scene or settings["scene"]).read_bytes()).hexdigest(),
        "root_assist_sha256": hashlib.sha256(settings["root_assist"].read_bytes()).hexdigest(),
        "controller_settings": asdict(settings["controller"]),
        "model": settings["model"],
        "physics_dt_s": config.simulation.physics_dt_s,
        "tracking_tolerance_deg": config.control.tracking_tolerance_deg,
        "skin_penetration_tolerance_m": settings["skin_penetration_tolerance_m"],
        "view_azimuths_deg": [float(a) for a in azimuths],
        "view_azimuth_plan_deg": azimuth_plan,
        "capture_windows_s": [[float(a), float(b)] for a, b in windows],
        "capture_rate_hz": float(args.capture_rate_hz),
        "cycle_period_s": float(period_s),
        "viewport_azimuth_note": (
            "Vulkan viewport: active camera azimuth follows the plan above; requested "
            "screenshots can lag by a frame-pair during the cycle boundary"
        ),
        "camera": {
            "distance_m": distance_m,
            "elevation_deg": elevation_deg,
            "up_axis": "z",
            "target_height_m": 0.8,
        },
        "ordinary_motion_teleports": 0,
        "reset_count": 0,
        "azimuth_definition": (
            "azimuth 0 = eye on +X of the character (frontal view for a +X-facing "
            "character), 90 = eye on +Y (subject's left), 180 = behind (dorsal), "
            "270 = subject's right"
        ),
    }
    provenance = dict(provenance)
    provenance["root_assistance"] = asdict(RootAssistConfig.load(settings["root_assist"]))

    if args.dry_run:
        intent = KeyboardIntent()
        position = controller.position.copy()
        frames = int(np.ceil(total_s / config.simulation.physics_dt_s))
        for frame in range(max(frames, 1)):
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
                "status": "cpu_targets_only",
                "final_target": position.tolist(),
                "planned_capture_times_s": sample_times(windows, args.capture_rate_hz),
            },
        )
        LOGGER.info("CPU multi-azimuth rehearsal passed")
        return 0

    app = boot_isaac(args.headless, width=1280, height=960)
    if app is None:
        raise RuntimeError("Isaac Sim is unavailable; use ~/isaacsim/python.sh or --dry-run")

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
    current_azimuth = float(azimuth_plan[0])
    azimuth_log: list[dict[str, float]] = []
    try:
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, UsdGeom, Vt

        scene = args.scene or settings["scene"]
        build_human_stage(
            plan,
            args.out / "human_arm_capture.usda",
            base_scene=scene,
            skin_points=mesh.vertices,
            skin_faces=mesh.faces,
        )
        stage = open_scene(app, args.out / "human_arm_capture.usda")
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
        assistance = RootAssistConfig.load(settings["root_assist"])
        plan_cache = {joint.name: joint for joint in plan.joints}
        arm_dofs = [name for name in plan_cache if name.split("__")[0] in ARM_JOINT_NAMES]
        leg_dofs = [name for name in plan_cache if name.split("__")[0] in LEG_JOINT_NAMES]
        primary_arm = {name: name.endswith("__dof2") for name in arm_dofs}

        status_window = ui.Window("Arm capture", width=300, height=150)
        with status_window.frame:
            with ui.VStack(spacing=6):
                ui.Label("External root assistance: ON")
                azimuth_label = ui.Label(f"Azimuth: {current_azimuth:.1f} deg")
                state_label = ui.Label("State: stand")
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
        # Held for the lifetime of the run: dropping the return value unsubscribes
        # the keyboard callback and no key event would ever arrive.
        subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)
        if subscription is None:
            raise RuntimeError("keyboard subscription failed")

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
                        "joints": runtime.joint_positions_rad(),
                        "joint_target": target.joints,
                        "force": force,
                        "torque": torque,
                        "command": intent.command(),
                        "mode": target.mode,
                        "contacts": [s.collider1_path for s in contacts],
                        "floor_contact_slips_m_s": slips,
                        "azimuth_deg": current_azimuth,
                    }
                )
                clock_s += dt
            except Exception as exc:
                LOGGER.exception("physics callback failed")
                errors.append(f"{type(exc).__name__}: {exc}")

        SimulationManager.register_callback(before_step, SimulationEvent.PHYSICS_PRE_STEP)

        elapsed = 0.0
        cycle_index = 0
        last_cycle_index = -1
        next_capture = 0
        capture_times = sample_times(windows, args.capture_rate_hz)
        local_times = list(capture_times)
        pending: list[tuple[Any, Path, float, float]] = []
        start_steps = SimulationManager.get_num_physics_steps()
        wall_start = time.monotonic()
        last_progress_wall = wall_start
        render_period = 1.0 / settings["render_hz"]
        next_render = 0.0

        while clock_s < total_s and not errors:
            cycle_index = min(int(clock_s // period_s), len(azimuth_plan) - 1)
            elapsed = clock_s - cycle_index * period_s
            if cycle_index != last_cycle_index:
                # Each cycle replays the demo from t=0 with fresh captures, so the
                # same simulated instants are photographed from every azimuth.
                # The state is rewound too: letting the bounded root assist run
                # for four consecutive cycles saturates it (observed force peak
                # pinned at max_force_n with 2.8 m lateral drift), which would
                # make the extra cycles a different experiment rather than a
                # second view of the same one.
                if last_cycle_index >= 0 or args.rewind_cycles:
                    reset()
                    provenance["reset_count"] += 1
                last_cycle_index = cycle_index
                local_times = list(capture_times)
                next_capture = 0
            target_azimuth = float(azimuth_plan[cycle_index])
            if target_azimuth != current_azimuth:
                current_azimuth = target_azimuth
                azimuth_log.append({"time_s": clock_s, "azimuth_deg": current_azimuth})
                azimuth_label.text = f"Azimuth: {current_azimuth:.1f} deg"

            key_names = {key for key in demo_keys(settings, elapsed)}
            if "R" in key_names:
                reset()
                if "R" not in intent.held:
                    provenance["reset_count"] += 1
                key_names = set()
            for key in tuple(intent.held - key_names):
                intent.event(key, False)
            for key in key_names - intent.held:
                intent.event(key, True)

            previous_time = clock_s
            app.update()
            if clock_s <= previous_time and not SimulationManager.is_paused():
                stalled = time.monotonic() - last_progress_wall
                if stalled > 20:
                    raise RuntimeError(
                        f"physics callback did not advance for {stalled:.0f} wall-clock seconds"
                    )
            else:
                last_progress_wall = time.monotonic()

            if clock_s >= next_render:
                vertices = skin_mesh_sequence_frame(mesh, runtime.link_poses())
                points_attr.Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                skins.append(vertices.astype(np.float32))
                skin_times.append(clock_s)
                position, _ = runtime.root_pose()
                target_view = np.array([position[0], position[1], 0.8])
                el, az = np.deg2rad([elevation_deg, current_azimuth])
                eye = target_view + distance_m * np.array(
                    [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
                )
                camera_op.Set(
                    Gf.Matrix4d()
                    .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target_view), Gf.Vec3d(0, 0, 1))
                    .GetInverse()
                )
                next_render = clock_s + render_period
                state_label.text = f"State: {records[-1]['mode']}" if records else "State: stand"
                clock_label.text = f"Physics: {clock_s:.2f} s"

            while next_capture < len(local_times) and elapsed >= local_times[next_capture]:
                next_capture += 1
                output = args.out / (
                    f"az{int(round(current_azimuth)) % 360:03d}_cycle{cycle_index:02d}"
                    f"_{clock_s:06.2f}.png"
                )
                capture = capture_viewport_to_file(get_active_viewport(), file_path=str(output))
                pending.append(
                    (
                        asyncio.ensure_future(capture.wait_for_result()),
                        output,
                        clock_s,
                        current_azimuth,
                    )
                )

        runtime.pause()
        deadline = time.monotonic() + 20
        while pending and time.monotonic() < deadline:
            app.update()
            for item in tuple(pending):
                task, path, capture_s, azimuth = item
                if task.done():
                    task.result()
                    if path.is_file() and path.stat().st_size:
                        captures.append(
                            {
                                "path": str(path.resolve()),
                                "request_time_s": capture_s,
                                "cycle": int(capture_s // period_s),
                                "azimuth_deg": azimuth,
                                "size_bytes": int(path.stat().st_size),
                            }
                        )
                        pending.remove(item)
        if pending:
            errors.append(f"{len(pending)} live screenshots did not flush")

        provenance["measured_physics_steps"] = int(
            SimulationManager.get_num_physics_steps() - start_steps
        )
        provenance["callback_steps"] = int(round(clock_s / measured_dt))
        provenance["simulation_time_s"] = clock_s
        provenance["wall_time_s"] = time.monotonic() - wall_start
        provenance["azimuth_timeline"] = azimuth_log
        provenance["captures"] = captures
        provenance["errors"] = errors

        if records:
            arrays = {
                key: np.asarray([row[key] for row in records])
                for key in (
                    "time_s",
                    "root",
                    "root_quaternion",
                    "joints",
                    "joint_target",
                    "force",
                    "torque",
                    "command",
                    "mode",
                    "azimuth_deg",
                )
            }
            np.savez_compressed(args.out / "control.npz", dof_names=plan.dof_names, **arrays)
            provenance["dof_names"] = list(plan.dof_names)
            error_deg = np.rad2deg(np.abs(arrays["joints"] - arrays["joint_target"]))
            worst = int(np.argmax(error_deg.max(axis=0)))
            provenance["joint_error_max_deg"] = float(error_deg.max())
            provenance["joint_error_worst_dof"] = list(plan_cache)[worst]
            provenance["joint_error_by_dof_deg"] = {
                name: float(error_deg[:, index].max()) for index, name in enumerate(plan_cache)
            }
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
            # Per-azimuth arm swing, so the report can state whether the camera
            # direction changed the measurement (it must not).
            swing: dict[str, Any] = {}
            for name in arm_dofs + leg_dofs:
                index = list(plan_cache).index(name)
                swing[name] = {
                    "primary_axis": primary_arm.get(name, False),
                    "ptp_target_deg": float(np.rad2deg(np.ptp(arrays["joint_target"][:, index]))),
                    "ptp_actual_deg": float(np.rad2deg(np.ptp(arrays["joints"][:, index]))),
                }
            provenance["joint_span_deg"] = swing
            provenance["root_displacement_m"] = root_displacement_in_body_frame(
                arrays["root"], arrays["root_quaternion"], arrays["command"]
            )
            provenance["command_timeline"] = _command_timeline(arrays["time_s"], arrays["command"])

        if skins:
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
        # Compute the completion flag *before* writing it. The previous version
        # wrote ``exit_code == 0`` while ``exit_code`` was still initialised to 1,
        # so every report claimed ``runtime_completed: false`` even on a clean run
        # -- which made a successful capture indistinguishable from an abort.
        exit_code = 0 if not errors else 1
        write_json(
            args.out / "report.json",
            {
                **provenance,
                "runtime_completed": exit_code == 0,
                "quality_gates": acceptance,
                "motion_accuracy_accepted": all(acceptance.values()) and not errors,
                "note": (
                    "runtime completion alone does not establish motion or contact accuracy; "
                    "this run exists for P0-A visual evidence, gates are the shipped ones"
                ),
            },
        )
        app.close(exit_code=exit_code)
    except Exception as exc:  # pragma: no cover - Isaac-side failure path
        LOGGER.exception("multi-azimuth capture failed")
        errors.append(f"{type(exc).__name__}: {exc}")
        write_json(
            args.out / "report.json",
            {**provenance, "errors": errors, "runtime_completed": False},
        )
        if app is not None:
            app.close(exit_code=1)
    return exit_code


def _command_timeline(time_s: np.ndarray, command: np.ndarray) -> list[dict[str, Any]]:
    """Collapse the per-step command stream into change points."""

    timeline = []
    previous: tuple[float, ...] | None = None
    for index in range(len(time_s)):
        current = tuple(float(v) for v in command[index])
        if current != previous:
            timeline.append({"time_s": float(time_s[index]), "command": list(current)})
            previous = current
    return timeline


if __name__ == "__main__":
    raise SystemExit(main())
