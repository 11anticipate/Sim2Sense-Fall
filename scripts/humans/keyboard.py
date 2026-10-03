#!/usr/bin/env python3
"""Drive an assisted SMPL human with W/S, A/D, Space, R and Escape."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import asdict, replace
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
    repo_relative,
    set_physics_dt,
    write_json,
)

from sim2sense_fall.humans.actions import (
    POSTURE_TRANSITION_MODES,
    ActionConfig,
    ActionState,
    load_action_clip,
    load_posture,
    recovery_command_anchor,
)
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import (
    StanceFootController,
    fit_collision_capsules,
    measured_support_feet,
)
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints, skin_mesh_sequence_frame
from sim2sense_fall.humans.quality import MotionQualityConfig, motion_quality
from sim2sense_fall.humans.recording import ChunkRecorder
from sim2sense_fall.humans.rig import LinkTransform, fit_rest_skeleton, plan_human_rig
from sim2sense_fall.humans.root_control import RootAssistConfig, blend_assist_configs, root_wrench
from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix
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

# Bound on the numerically differentiated reference COM acceleration. The
# second difference is exact for the smooth baked tables, but its input changes
# rate whenever the phase clock does (reversals), and the raw spikes there were
# measured to kick the body hard enough to ring the collar drives ±20 deg. The
# true signal (bob and load-profile accelerations) stays below ~1 m/s^2, so the
# clip is tight and an EMA removes the remaining rate-change transients.
REFERENCE_ACCEL_MAX_M_S2 = 2.5
REFERENCE_ACCEL_EMA_TAU_S = 0.08


def _yaw_of(quaternion: np.ndarray) -> float:
    """Yaw of a (w, x, y, z) quaternion — the runtime's storage order."""
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def cycle_sample(table: np.ndarray, phase: float) -> np.ndarray:
    """Circular linear interpolation into a seam-closed per-cycle table."""
    count = len(table)
    point = (phase % 1.0) * (count - 1)
    index = min(int(point), count - 2)
    fraction = point - index
    return (1 - fraction) * table[index] + fraction * table[(index + 1) % count]


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
    settings["collision_fit_audit"] = []
    if settings.get("collision_fit", {}).get("enabled", False):
        plan, settings["collision_fit_audit"] = fit_collision_capsules(
            plan, mesh, **{k: v for k, v in settings["collision_fit"].items() if k != "enabled"}
        )
    dt = config.simulation.physics_dt_s
    planted = settings.get("max_stance_slip_m_s")
    gaits = {
        key: load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=planted)
        for key, spec in settings["gaits"].items()
    }
    idle = load_gait(settings["idle"], plan, dt_s=dt, max_stance_slip_m_s=planted)
    if settings.get("contact_planner"):
        from sim2sense_fall.humans.contact_gait import (
            ContactFootPlanner,
            ContactGaitConfig,
            fit_contact_idle,
        )

        planner_config = ContactGaitConfig(**settings["contact_planner"])
        # The idle pose stands with each foot directly under its hip, so unlike the
        # gait it needs no stride-reach allowance. Pinning it to the gait's capped
        # height left ~4.7 cm of slack leg that the IK spent as 37-43 deg of standing
        # knee flexion (scripts/humans/audit_idle_posture.py). The full standing
        # height itself overcorrects to a 3 deg near-singular column, which bounces
        # on the floor contact and skids the feet (stand slip p95 0.797 m/s vs the
        # bent baseline's 0.010); a relaxed stand keeps a few degrees of knee
        # flexion for axial compliance. Starting to walk still descends smoothly:
        # TeleopController.advance blends idle->gait height.
        idle_height = float(plan.standing_root_height_m) * settings["idle_stand_height_fraction"]
        fit_contact_idle(idle, plan, idle_height, planner_config)
        swing_shapes = {
            name: gait.swing_shapes for name, gait in gaits.items() if gait.swing_shapes is not None
        }
        settings["contact_foot_planner"] = ContactFootPlanner(
            plan, planner_config, settings["speed_m_s"], settings["acceleration_m_s2"],
            swing_shapes=swing_shapes or None)
    controller = TeleopController(
        settings["controller"], gaits, idle, plan, np.deg2rad(settings["heading_deg"])
    )
    if "actions" in settings:
        postures = {"crouch": load_posture(settings["crouch"], plan)}
        for key in ("bend", "sit"):
            if key in settings:
                postures[key] = load_posture(settings[key], plan)
        playback_clip = (
            load_action_clip(settings["get_up"], plan, dt_s=config.simulation.physics_dt_s)
            if "get_up" in settings
            else None
        )
        fall_clip = (
            load_action_clip(settings["fall_replay"], plan, dt_s=config.simulation.physics_dt_s)
            if "fall_replay" in settings
            else None
        )
        settings["action_state"] = ActionState(
            ActionConfig(**settings["actions"]), postures, playback_clip, plan=plan,
            fall_clip=fall_clip,
        )
        settings["posture_provenance"] = {name: p.provenance for name, p in postures.items()}
        settings["get_up_provenance"] = None if playback_clip is None else playback_clip.provenance
        settings["fall_replay_provenance"] = None if fall_clip is None else fall_clip.provenance
        settings["crouch_provenance"] = postures["crouch"].provenance
    if settings.get("stance", {}).get("enabled", False):
        settings["stance_controller"] = StanceFootController(
            plan, **{key: value for key, value in settings["stance"].items() if key != "enabled"}
        )
    if settings["reference_feedforward_scale"] > 0:
        tables = {name: gait.com_offset_m for name, gait in gaits.items()}
        tables["idle"] = idle.com_offset_m
        missing = sorted(name for name, table in tables.items() if table is None)
        if missing:
            raise ValueError(
                f"reference_feedforward_scale needs COM offset tables on {missing}; "
                "they are produced by the contact cycle bake and the contact idle fit"
            )
    return settings, config, plan, mesh, controller


def demo_keys(settings: dict[str, Any], elapsed_s: float) -> set[str]:
    end = 0.0
    for segment in settings["demo"]:
        end += segment["duration_s"]
        if elapsed_s < end:
            return set(segment["keys"])
    return set()


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--native-mesh",
        action="store_true",
        help=(
            "sample the display skin every physics step (overrides "
            "record_mesh_at_physics_hz); needed for an admissible complete "
            "120 Hz recording when exporting sessions for the Sionna importer"
        ),
    )
    # argv is injectable so experiment harnesses can reuse this exact entry
    # point (identical physics setup) with generated per-trial configs.
    args = parser.parse_args(argv)
    if not args.out.is_absolute():
        # A relative --out is repo-relative, not cwd-relative: the GUI may be
        # launched from any directory and sessions must land under artifacts/.
        args.out = REPO_ROOT / args.out
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    if args.headless and not args.demo and args.seconds == 0 and not args.dry_run:
        parser.error("headless mode needs --demo or --seconds")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings, config, plan, mesh, controller = prepare(args.config)
    actions = settings.get("action_state")
    stance = settings.get("stance_controller")
    contact_planner = settings.get("contact_foot_planner")
    args.out.mkdir(parents=True, exist_ok=True)
    assistance = RootAssistConfig.load(settings["root_assist"])
    locomotion_assistance = RootAssistConfig.load(
        settings.get("locomotion_root_assist", settings["root_assist"])
    )
    recovery_path = settings.get("recovery_root_assist")
    recovery_assistance = None if recovery_path is None else RootAssistConfig.load(recovery_path)
    recovery_modes = set(settings.get("recovery_modes", ["getting_up", "standing_up"]))
    assist_blend = 1.0
    recovery_blend = 0.0
    feedforward_scale = settings["reference_feedforward_scale"]
    ff_com_prev = ff_com_prev2 = None
    ff_accel = None
    ff_mode = None
    quality_config = MotionQualityConfig(**settings.get("quality", {}))
    native_mesh = settings.get("record_mesh_at_physics_hz", False)
    if not isinstance(native_mesh, bool):
        raise ValueError("record_mesh_at_physics_hz must be boolean")
    if args.native_mesh:
        # Data-capture override: skin sampled every physics step, which the
        # mesh exporter and the Sionna importer need for a complete 120 Hz
        # recording window. Heavier than the interactive render-rate default.
        native_mesh = True
    scene = args.scene or settings["scene"]
    provenance = {
        "controller_sources_sha256": {
            repo_relative(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),
                         *sorted((REPO_ROOT / "src/sim2sense_fall/humans").glob("*.py"))]
        },
        "mode": "keyboard_bounded_external_root_assistance",
        "unassisted": settings["root_assist_scale"] == 0,
        "root_assist_scale": settings["root_assist_scale"],
        "reference_feedforward_scale": feedforward_scale,
        "seed": config.export.surface_point_seed,
        "scene_split": "single_fixed_apartment_no_train_test_split",
        "rig_sha256": hashlib.sha256(settings["rig"].read_bytes()).hexdigest(),
        "keyboard_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "root_assist_sha256": hashlib.sha256(settings["root_assist"].read_bytes()).hexdigest(),
        "locomotion_root_assist_sha256": hashlib.sha256(
            settings.get("locomotion_root_assist", settings["root_assist"]).read_bytes()
        ).hexdigest(),
        "gaits": {key: gait.provenance for key, gait in controller.gaits.items()},
        "model": settings["model"],
        "idle": controller.idle.provenance,
        "root_assistance": asdict(assistance),
        "locomotion_root_assistance": asdict(locomotion_assistance),
        "assistance_transition_s": controller.config.transition_s,
        "physics_dt_s": config.simulation.physics_dt_s,
        "mesh_sampling": "physics_pre_step" if native_mesh else "render_update",
        "tracking_tolerance_deg": config.control.tracking_tolerance_deg,
        "skin_penetration_tolerance_m": settings["skin_penetration_tolerance_m"],
        "ordinary_motion_teleports": 0,
        "reset_count": 0,
        "dof_names": list(plan.dof_names),
        "collision_fit": settings["collision_fit_audit"],
        "crouch": settings.get("crouch_provenance"),
        "postures": settings.get("posture_provenance"),
        "get_up": settings.get("get_up_provenance"),
        "actions": settings.get("actions"),
        "stance": settings.get("stance"),
        "contact_planner": settings.get("contact_planner"),
        "demo": settings["demo"] if args.demo or args.dry_run else None,
        "controller_settings": asdict(controller.config),
    }
    if args.dry_run:
        intent = KeyboardIntent()
        position = controller.position.copy()
        previous_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        duration_s = args.seconds or sum(s["duration_s"] for s in settings["demo"])
        frames = int(np.ceil(duration_s / config.simulation.physics_dt_s))
        modes: set[str] = set()
        for frame in range(frames):
            keys = demo_keys(settings, frame * config.simulation.physics_dt_s)
            for key in tuple(intent.held - keys):
                intent.event(key, False)
            for key in keys - intent.held:
                intent.event(key, True)
            if intent.reset_requested:
                controller.reset(
                    np.asarray(plan.spawn_root_position), np.deg2rad(settings["heading_deg"])
                )
                position = controller.position.copy()
                if actions:
                    actions.reset()
                if stance:
                    stance.reset()
                intent.reset_requested = False
                provenance["reset_count"] += 1
            if actions and intent.action_requested:
                # Same walk-coupled routing as the physics loop: moving F = trip.
                requested_action = intent.action_requested
                if requested_action == "fall" and controller.speed > (
                    ActionConfig(**settings["actions"]).stopped_speed_m_s
                ):
                    requested_action = "trip"
                actions.request(
                    requested_action,
                    time_s=frame * config.simulation.physics_dt_s,
                    heading_rad=controller.heading,
                    measured_joints=controller.joints,
                    measured_position=position,
                    measured_quaternion=previous_quaternion,
                )
                intent.action_requested = None
            target = controller.advance(
                actions.command(intent.command()) if actions else intent.command(),
                config.simulation.physics_dt_s,
                position,
                controller.heading,
            )
            if actions:
                was_getting_up = actions.mode == "getting_up"
                target = actions.apply(
                    target,
                    dt_s=config.simulation.physics_dt_s,
                    speed_m_s=controller.speed,
                    idle_joints=controller.idle.joints[0],
                    idle_height_m=controller.idle.height_m[0],
                    idle_tilt=controller.idle.tilt(0),
                    heading_rad=controller.heading,
                )
                if was_getting_up and actions.mode != "getting_up":
                    # Recovery handed control back: adopt the orientation the
                    # body actually got up facing instead of torquing it back
                    # to the pre-fall heading (that tug was the post-get-up
                    # spin).
                    controller.heading = _yaw_of(target.quaternion)
            position = target.position
            previous_quaternion = target.quaternion
            modes.add(target.mode)
            if not np.isfinite(target.joints).all():
                raise RuntimeError("nonfinite dry-run target")
        write_json(
            args.out / "dry_run.json",
            {
                **provenance,
                "physics_run": False,
                "final_target": position.tolist(),
                "status": "cpu_targets_only",
                "rehearsed_frames": frames,
                "rehearsed_duration_s": duration_s,
                "modes": sorted(modes),
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
                settings["record_frames"] * (
                    1 if native_mesh else config.simulation.physics_dt_s * settings["render_hz"]
                )
            )
        )
        + 1
    )
    skins: deque[np.ndarray] = deque(maxlen=skin_capacity)
    skin_times: deque[float] = deque(maxlen=skin_capacity)
    foot_heights: deque[np.ndarray] = deque(maxlen=skin_capacity)
    # Whole-body lowest skinned vertex per mesh frame: the low-posture gate needs
    # "is any part of the body on the floor", which the two foot heights cannot say.
    body_heights: deque[float] = deque(maxlen=skin_capacity)
    owners = np.asarray(mesh.topology.joint_names)[mesh.weights.argmax(axis=1)]
    foot_masks = [np.isin(owners, [f"{side}_ankle", f"{side}_foot"])
                  for side in ("left", "right")]
    errors: list[str] = []
    captures: list[dict[str, Any]] = []
    intent = KeyboardIntent()
    clock_s = 0.0
    exit_code = 1
    runtime = None
    recorder = ChunkRecorder(args.out / "chunks", plan.dof_names, int(settings["record_frames"]))
    input_events: list[dict[str, Any]] = []
    render_count = 0
    try:
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, UsdGeom, Vt

        stage_report: dict[str, Any] = {}
        build_human_stage(
            plan,
            args.out / "human_keyboard.usda",
            base_scene=scene,
            skin_points=mesh.vertices,
            skin_faces=mesh.faces,
            friction=settings.get("human_friction"),
            joint_velocity_limit_rad_s=settings.get("joint_velocity_limit_rad_s"),
            report_out=stage_report,
        )
        provenance["self_collisions"] = stage_report.get("self_collisions")
        provenance["self_collision_filtered_pairs"] = stage_report.get(
            "self_collision_filtered_pairs", []
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
            nonlocal assist_blend, recovery_blend
            nonlocal ff_com_prev, ff_com_prev2, ff_accel, ff_mode
            assist_blend = 1.0
            recovery_blend = 0.0
            ff_com_prev = ff_com_prev2 = None
            ff_accel = None
            ff_mode = None
            if actions:
                actions.reset()
            if stance:
                stance.reset()
            if contact_planner:
                contact_planner.reset()
            if not runtime.set_control_scale(1.0):
                raise RuntimeError("could not restore joint drives on reset")
            controller.reset(
                np.asarray(plan.spawn_root_position), np.deg2rad(settings["heading_deg"])
            )
            target = controller.advance(
                (0.0, 0.0), measured_dt, controller.position, controller.heading
            )
            # The negative offset is a force-control target, not a spawn teleport.
            # Applying it to the reset pose starts the hull/skin inside the floor.
            spawn_position = target.position.copy()
            spawn_position[2] -= controller.config.root_z_offset_m
            runtime.set_root_pose(spawn_position, target.quaternion)
            runtime.set_joint_positions(target.joints)
            runtime.set_joint_targets(target.joints)
            runtime.reset_velocities()
            intent.clear()
            intent.reset_requested = False

        reset()
        previous_target = controller.advance(
            (0.0, 0.0), measured_dt, controller.position, controller.heading
        )
        previous_stance_mode = "stand"
        status_window = ui.Window("Human", width=280, height=140)
        with status_window.frame:
            with ui.VStack(spacing=6):
                assist_label = ui.Label("External root assistance: ON")
                state_label = ui.Label("State: stand")
                speed_label = ui.Label("Speed: 0.00 m/s")
                clock_label = ui.Label("Physics: 0.00 s")
                ui.Label("C: crouch | B: bend | N: sit | V: stand | F: fall | G: get up")

        def on_key(event: Any, *_args: Any) -> bool:
            key = event.input.name
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                intent.event(key, True)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                intent.event(key, False)
            if key in KeyboardIntent.KEYS:
                input_events.append({"time_s": clock_s, "key": key, "event": str(event.type)})
            return key not in KeyboardIntent.KEYS

        window = omni.appwindow.get_default_app_window()
        keyboard = window.get_keyboard()
        input_interface = carb.input.acquire_input_interface()
        subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)
        input_provider = carb.input.acquire_input_provider()
        buffered_keys: set[str] = set()

        def before_step(dt: float, _context: Any) -> None:
            nonlocal clock_s, previous_target, previous_stance_mode, assist_blend
            nonlocal recovery_blend
            nonlocal ff_com_prev, ff_com_prev2, ff_accel, ff_mode
            if errors:
                return
            try:
                position, quaternion = runtime.root_pose()
                rotation = quaternion_to_matrix(quaternion)
                # One read per callback, at the top, so the stance gate below sees the same
                # report the rest of this step records. Polling twice would double-count.
                attributed = runtime.attributed_contact_samples()
                contacts = [sample for sample, _ in attributed]
                anchor_source = settings["anchor_source"]
                measured_feet = (
                    measured_support_feet(
                        attributed, min_impulse_ns=settings["slip_min_impulse_ns"]
                    )
                    if anchor_source != "model"
                    else None
                )
                if actions and intent.action_requested:
                    # get_up anchors the replay at the *measured* fallen heading:
                    # anchoring at the teleop heading left the recovery spinning
                    # from the collapsed orientation back to the pre-fall one.
                    measured_heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                    request_heading = (
                        measured_heading
                        if intent.action_requested == "get_up"
                        else controller.heading
                    )
                    # Route-A increment: F WHILE WALKING is a trip, not a fall
                    # request -- the gait keeps stepping and the measured
                    # outcome decides. Standing F keeps the anchored replay.
                    # The snag lands on the swing foot (the toe that catches).
                    requested_action = intent.action_requested
                    trip_foot = None
                    if requested_action == "fall" and controller.speed > (
                        actions.config.stopped_speed_m_s
                    ):
                        requested_action = "trip"
                        request_heading = measured_heading
                        supporting = controller.gaits[controller.mode].supporting_feet(
                            controller.phase
                        )
                        candidates = sorted({"left_ankle", "right_ankle"} - (supporting or set()))
                        trip_foot = candidates[0] if candidates else "left_ankle"
                    accepted = actions.request(
                        requested_action,
                        time_s=clock_s,
                        heading_rad=request_heading,
                        measured_joints=runtime.joint_positions_rad(),
                        measured_position=position,
                        measured_quaternion=quaternion,
                        trip_foot=trip_foot,
                    )
                    intent.action_requested = None
                    if actions.tripping:
                        # Route-A trip in flight: drives weakened (the muscle
                        # failure that makes the snag stick), stance keeps
                        # running. The release happens when observe() latches
                        # the measured fall; the restore happens on survival.
                        if not runtime.set_control_scale(
                            actions.config.trip_control_scale, 1.0
                        ):
                            raise RuntimeError("could not weaken drives for trip")
                    elif actions.falling and not actions.fall_replay_engaged:
                        if not runtime.set_control_scale(
                            actions.config.fall_control_scale,
                            actions.config.fall_damping_scale,
                        ):
                            raise RuntimeError("could not release position drives for fall")
                        if stance:
                            stance.reset()
                    # A fall replay keeps the drives engaged (it commands the
                    # descent itself); the release happens when the replay's
                    # lying end is actually measured fallen -- see the
                    # consume_fall_replay_release() call after actions.apply().
                    elif accepted and actions.mode == "getting_up":
                        # The fall released the position drives; recovery plays
                        # under full drives plus the standard root assistance.
                        if not runtime.set_control_scale(1.0):
                            raise RuntimeError("could not restore position drives for get_up")
                        if stance:
                            stance.reset()
                        # Re-anchor the per-step slew limiter at the body's real pose.
                        # The fallen command stream is the frozen pre-fall pose (measured
                        # 18.5 deg knee against a 136.4 deg body), and chasing it yanks
                        # the collapsed limbs straight while the pelvis is still down.
                        anchor = recovery_command_anchor(
                            actions.mode, runtime.joint_positions_rad()
                        )
                        if anchor is not None:
                            previous_target = replace(previous_target, joints=anchor)
                target = controller.advance(
                    actions.command(intent.command()) if actions else intent.command(),
                    dt,
                    position,
                    float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                )
                if actions:
                    was_getting_up = actions.mode == "getting_up"
                    was_tripping = actions.tripping
                    target = actions.apply(
                        target,
                        dt_s=dt,
                        speed_m_s=controller.speed,
                        idle_joints=controller.idle.joints[0],
                        idle_height_m=controller.idle.height_m[0],
                        idle_tilt=controller.idle.tilt(0),
                        heading_rad=controller.heading,
                    )
                    if was_getting_up and actions.mode != "getting_up":
                        # Recovery handed control back: adopt the orientation
                        # the body actually got up facing instead of torquing
                        # it back to the pre-fall heading (that tug was the
                        # post-get-up spin).
                        controller.heading = _yaw_of(target.quaternion)
                    if actions.consume_trip_release():
                        # The stumble crossed the measured fall thresholds:
                        # release the drives fully (the trip had them weakened).
                        if not runtime.set_control_scale(
                            actions.config.fall_control_scale,
                            actions.config.fall_damping_scale,
                        ):
                            raise RuntimeError("could not release drives after trip fall")
                        if stance:
                            stance.reset()
                    elif was_tripping and not actions.tripping and not actions.falling:
                        # The stumble steps caught the body: restore full drives
                        # and locomotion. The survived stumble stays on record.
                        if not runtime.set_control_scale(1.0):
                            raise RuntimeError("could not restore drives after survived trip")
                    if actions.consume_fall_replay_release():
                        # The replay commanded its lying end and observe() has
                        # measured the real fallen state: release the drives
                        # exactly like the legacy fall so the body settles.
                        if not runtime.set_control_scale(
                            actions.config.fall_control_scale,
                            actions.config.fall_damping_scale,
                        ):
                            raise RuntimeError(
                                "could not release position drives after fall replay"
                            )
                        if stance:
                            stance.reset()
                if stance and not (actions and actions.suppresses_stance):
                    stance_mode = (
                        "transition"
                        if target.mode in {"forward", "backward"} and controller.weight < 0.95
                        else target.mode
                    )
                    if stance_mode != previous_stance_mode and (
                        target.mode not in POSTURE_TRANSITION_MODES
                    ):
                        stance.reset()
                    previous_stance_mode = stance_mode
                    if stance_mode == "transition" or abs(controller.turn_rate) > 0.01:
                        # Turning while walking re-plants feet at new headings; the
                        # travel-derived anchors would fight that, so no anchors.
                        stance.reset()
                    reference_feet = controller.gaits[controller.mode].supporting_feet(
                        controller.phase
                    )
                    if measured_feet is None or reference_feet is None:
                        # model mode, or a gait without a support mask: as before.
                        support = measured_feet if measured_feet is not None else reference_feet
                    elif anchor_source == "contact":
                        support = measured_feet
                    else:
                        # contact_and_model: anchor only where both agree. The intersection
                        # can only remove anchors, never add them, so it tightens the gate
                        # without inventing stance the reference does not claim.
                        support = reference_feet & measured_feet
                    corrected = stance.correct(
                        target.joints,
                        target.position,
                        target.quaternion,
                        standing=abs(controller.speed) < 0.01,
                        support_feet=support,
                        swing_fraction=controller.gaits[controller.mode].swing_fraction(
                            controller.phase
                        ),
                        measured_position=position,
                        measured_quaternion=quaternion,
                        measured_joints=runtime.joint_positions_rad(),
                    )
                    target = replace(target, joints=corrected)
                if contact_planner and target.mode in {"forward", "backward", "stand"}:
                    root_position = target.position.copy()
                    root_position[:2] = position[:2]
                    # Cancel horizontal root tracking lag. Keep target height and
                    # rotation: feeding back full orientation amplified contact
                    # chatter in the recorded orientation_matrix comparison.
                    corrected = contact_planner.correct(
                        target.joints, LinkTransform(quaternion_to_matrix(target.quaternion),
                                                    root_position),
                        {f"{s}_ankle": runtime.link_pose(f"{s}_ankle") for s in ("left", "right")},
                        heading=controller.heading, turn_rate=controller.turn_rate,
                        speed=controller.speed, command=intent.command()[0], dt_s=dt)
                    target = replace(target, joints=corrected)
                elif contact_planner:
                    contact_planner.reset()
                delta = np.clip(
                    target.joints - previous_target.joints,
                    -controller.config.max_joint_speed_rad_s * dt,
                    controller.config.max_joint_speed_rad_s * dt,
                )
                target = replace(
                    target,
                    joints=previous_target.joints + delta,
                    joint_velocities=delta / dt,
                    linear_velocity=(target.position - previous_target.position) / dt,
                    angular_velocity=matrix_to_axis_angle(
                        quaternion_to_matrix(target.quaternion)
                        @ quaternion_to_matrix(previous_target.quaternion).T
                    )
                    / dt,
                )
                previous_target = target
                # Reference-COM feedforward: Newton-Euler on the blended reference
                # configuration, applied as m*a before the assist's caps. The
                # history resets on any mode change so the second difference never
                # differentiates across a table switch.
                feedforward_accel = None
                if feedforward_scale > 0 and assist_blend > 0 and target.mode in {
                    "forward", "backward", "stand"
                }:
                    if ff_mode != target.mode:
                        ff_mode = target.mode
                        ff_com_prev = ff_com_prev2 = None
                        ff_accel = None
                    gait_table = controller.gaits[controller.mode]
                    offset = (
                        (1 - controller.weight) * controller.idle.com_offset_m[0]
                        + controller.weight * cycle_sample(
                            gait_table.com_offset_m, controller.phase
                        )
                    )
                    cosine, sine = np.cos(controller.heading), np.sin(controller.heading)
                    com_reference = target.position + np.array([
                        cosine * offset[0] - sine * offset[1],
                        sine * offset[0] + cosine * offset[1],
                        offset[2],
                    ])
                    if ff_com_prev is not None and ff_com_prev2 is not None:
                        accel = (com_reference - 2 * ff_com_prev + ff_com_prev2) / (dt * dt)
                        alpha = dt / (dt + REFERENCE_ACCEL_EMA_TAU_S)
                        ff_accel = (
                            accel * alpha
                            if ff_accel is None
                            else ff_accel * (1 - alpha) + accel * alpha
                        )
                        feedforward_accel = feedforward_scale * assist_blend * np.clip(
                            ff_accel, -REFERENCE_ACCEL_MAX_M_S2, REFERENCE_ACCEL_MAX_M_S2
                        )
                    ff_com_prev2 = ff_com_prev
                    ff_com_prev = com_reference
                linear, angular = runtime.root_velocities()
                ordinary = target.mode in {"forward", "backward", "stand"}
                assist_blend = float(np.clip(
                    assist_blend + (1 if ordinary else -1)*dt/controller.config.transition_s,
                    0., 1.,
                ))
                if assist_blend == 1:
                    active_assistance = locomotion_assistance
                elif assist_blend == 0:
                    active_assistance = assistance
                else:
                    active_assistance = RootAssistConfig(**{
                        key: value*(1-assist_blend)
                        + getattr(locomotion_assistance, key)*assist_blend
                        for key, value in asdict(assistance).items()
                    })
                if recovery_assistance is not None:
                    recovery_blend = float(np.clip(
                        recovery_blend
                        + (1 if target.mode in recovery_modes else -1) * dt
                        / controller.config.transition_s,
                        0.0, 1.0,
                    ))
                    if recovery_blend > 0.0:
                        active_assistance = blend_assist_configs(
                            active_assistance, recovery_assistance, recovery_blend
                        )
                if actions and (actions.fall_replay_engaged or actions.tripping):
                    # No horizontal tow-cable during a fall replay OR a trip.
                    # The 12 kN/m pelvis spring absorbs any survivable shove
                    # (first trip run: 180 N deflected the body 1.5 cm, peak
                    # tilt 9 deg) -- the balance failure must remove the
                    # spring, not fight it. The clip/trip still owns the joint
                    # collapse, the vertical support curve and the tilt;
                    # horizontal momentum is the legs' own floor contact.
                    target = replace(
                        target,
                        position=np.array([*position[:2], target.position[2]]),
                        linear_velocity=np.array([*linear[:2], target.linear_velocity[2]]),
                    )
                force, torque = root_wrench(
                    active_assistance,
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
                    feedforward_accel_m_s2=feedforward_accel,
                )
                force *= settings["root_assist_scale"]
                torque *= settings["root_assist_scale"]
                if actions and actions.tripping:
                    # The orientation servo is the system's de-facto balance
                    # controller (3000 Nm/rad, immune to the mode blend): while
                    # it stays full the body cannot pitch past ~10 deg no matter
                    # what the snag does (trip_walk_v1..v5). Scaling the pitch
                    # authority is the measured difference between catching and
                    # toppling; the release removes it entirely.
                    torque *= actions.config.trip_torque_scale
                if actions and actions.tripping and actions.trip_foot:
                    # The caught-toe drag, applied AT THE SNAGGED ANKLE (a root
                    # shove accelerates the feet along with the body and nothing
                    # trips -- trip_walk_v1..v3, peak tilt 9.8 deg at 450 N).
                    # Rides on top of the fading assist; the horizontal pelvis
                    # authority is already removed (tripping branch above), so
                    # the trip torque comes from real foot arrest vs momentum.
                    runtime.apply_force(actions.trip_foot, actions.trip_force(clock_s))
                if actions and actions.falling and not actions.fall_replay_engaged:
                    force, torque = actions.fall_force(clock_s), np.zeros(3)
                    target = replace(target, joint_velocities=np.zeros_like(target.joints))
                runtime.set_joint_targets(target.joints, target.joint_velocities)
                runtime.apply_root_wrench(force, torque)
                if actions:
                    actions.observe(
                        time_s=clock_s,
                        root_height_m=float(position[2]),
                        tilt_deg=float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
                        standing_height_m=float(controller.idle.height_m[0]),
                        body_impact=any(
                            limb not in {"left_ankle", "right_ankle", "left_foot", "right_foot"}
                            and sample.collider1_path.endswith("/floor")
                            and sample.impulse_magnitude_ns > settings["slip_min_impulse_ns"]
                            for sample, limb in attributed
                        ),
                    )
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
                        "joint_speeds": runtime.joint_velocities_rad_s(),
                        "joint_target": target.joints,
                        "force": force,
                        "torque": torque,
                        "velocity": linear,
                        "command": intent.command(),
                        "mode": target.mode,
                        "gait_phase": controller.phase,
                        "gait_weight": controller.weight,
                        "reset_id": provenance["reset_count"],
                        "stance_residual_m": 0.0 if stance is None else stance.residual_m,
                        "foot_planner_residual_m": 0. if contact_planner is None
                        else contact_planner.last_residual_m,
                        "locomotion_assist_blend": assist_blend,
                        "contacts": [s.collider1_path for s in contacts],
                        "contact_detail": [s.as_dict() for s in contacts],
                        "floor_contact_slips_m_s": slips,
                        "impulse_ns": sum(float(np.linalg.norm(s.impulse_ns)) for s in contacts),
                    }
                )
                recorder.append(records[-1])
                if native_mesh:
                    vertices = skin_mesh_sequence_frame(mesh, runtime.link_poses())
                    skins.append(vertices.astype(np.float32))
                    skin_times.append(clock_s)
                    foot_heights.append(np.array([vertices[mask, 2].min() for mask in foot_masks]))
                    body_heights.append(float(vertices[:, 2].min()))
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
                previous_target = controller.advance(
                    (0.0, 0.0), measured_dt, controller.position, controller.heading
                )
                provenance["reset_count"] += 1
            previous_time = clock_s
            app.update()
            recorder.flush()
            if clock_s > previous_time or SimulationManager.is_paused():
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 15:
                raise RuntimeError("physics callback did not advance for 15 wall-clock seconds")
            if clock_s >= next_render:
                render_count += 1
                vertices = skin_mesh_sequence_frame(mesh, runtime.link_poses())
                points_attr.Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                if not native_mesh:
                    skins.append(vertices.astype(np.float32))
                    skin_times.append(clock_s)
                    foot_heights.append(np.array([vertices[mask, 2].min() for mask in foot_masks]))
                    body_heights.append(float(vertices[:, 2].min()))
                if settings.get("camera_follow", True):
                    # camera_follow: false leaves the viewport camera to the user
                    # (orbit/fly in the GUI); we never overwrite it again.
                    position, _ = runtime.root_pose()
                    target_view = np.array([
                        position[0], position[1], settings.get("camera_target_height_m", 0.8)
                    ])
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
                assist_label.text = "External root assistance: " + (
                    "OFF (fall)"
                    if actions and actions.falling
                    else f"{settings['root_assist_scale']:.0%}"
                )
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
        provenance["performance"] = {
            "physics_realtime_factor": clock_s / provenance["wall_time_s"],
            "skin_updates": render_count,
            "skin_updates_per_wall_second": render_count / provenance["wall_time_s"],
            "render_fps": None,
            "note": "skin update rate is not renderer FPS; input timestamps are physics time",
        }
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
        recorder.flush(final=True)
        write_json(args.out / "input_events.json", input_events)
        provenance["complete_control_record"] = "chunks/manifest.json"
        if records:
            arrays = {
                key: np.asarray([row[key] for row in records])
                for key in (
                    "time_s",
                    "root",
                    "root_quaternion",
                    "target",
                    "joints",
                    "joint_speeds",
                    "joint_target",
                    "force",
                    "torque",
                    "velocity",
                    "command",
                    "mode",
                    "gait_phase",
                    "gait_weight",
                    "reset_id",
                    "stance_residual_m",
                    "foot_planner_residual_m",
                    "locomotion_assist_blend",
                    "impulse_ns",
                )
            }
            np.savez_compressed(args.out / "control.npz", dof_names=plan.dof_names, **arrays)
            provenance["metrics_scope"] = "retained_record_window"
            provenance["retained_time_interval_s"] = [
                float(arrays["time_s"][0]),
                float(arrays["time_s"][-1]),
            ]
            # The session-level standing-tracking gate covers the frames it is a
            # statement about. Fall frames are released control; floor-recovery
            # frames are contact-constrained against the ground, which the official
            # rule already excludes: "跌倒后的关节轨迹不强制满足站姿跟踪门槛"
            # (task_plan 验收规则). They stay judged by their own per-activity gate
            # (MotionQualityConfig.low_posture_joint_error_deg), so nothing is hidden.
            controlled = ~np.isin(arrays["mode"], ["falling", "fallen", "getting_up"])
            provenance["joint_error_max_deg"] = (
                None
                if not controlled.any()
                else float(
                    np.rad2deg(
                        np.abs(arrays["joints"][controlled] - arrays["joint_target"][controlled])
                    ).max()
                )
            )
            provenance["joint_tracking_scope"] = (
                "controlled_frames_excluding_fall_and_floor_recovery"
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
            provenance["stance_control"] = (
                None
                if stance is None
                else {
                    "anchor_releases": int(stance.released_anchors),
                    "max_residual_m": float(np.asarray(arrays["stance_residual_m"]).max()),
                    "release_residual_m": stance.release_residual_m,
                    "abduction_weight": stance.abduction_weight,
                    "penalised_sideways_dofs": list(stance.sideways_dofs),
                    "max_correction_rad": stance.max_correction_rad,
                }
            )
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
                foot_heights.popleft()
                body_heights.popleft()
            np.savez_compressed(
                args.out / "recording.npz",
                time_s=np.asarray(skin_times),
                mesh_vertices_xyz=np.stack(skins),
                mesh_faces=mesh.faces,
                foot_min_z_m=np.stack(foot_heights),
                body_min_z_m=np.asarray(body_heights),
            )
            provenance["skin_min_z_m"] = float(min(skin[:, 2].min() for skin in skins))
        provenance["fall"] = None if actions is None else actions.fall_report()
        provenance["motion_quality"] = motion_quality(
            list(records), np.asarray(skin_times), np.asarray(foot_heights),
            config=quality_config, dt_s=config.simulation.physics_dt_s,
            mass_kg=plan.total_mass_kg, gravity_m_s2=config.simulation.gravity_m_s2,
            skin_min_z_m=np.asarray(body_heights),
        )
        acceptance = {
            "complete_recording_window": bool(records)
            and len(records) == provenance.get("callback_steps")
            and bool(skin_times) and skin_times[0] <= 2 * config.simulation.physics_dt_s,
            "per_activity_motion_quality": provenance["motion_quality"]["accepted"],
            "joint_tracking": bool(records)
            and provenance["joint_error_max_deg"] is not None
            and provenance["joint_error_max_deg"] <= config.control.tracking_tolerance_deg,
            "skin_ground_clearance": bool(skins)
            and provenance["skin_min_z_m"] >= -settings["skin_penetration_tolerance_m"],
            "floor_slip": bool(records)
            and bool(provenance["floor_contact_slip"]["sample_count"])
            and provenance["floor_contact_slip"]["p95_m_s"] <= settings["slip_speed_tolerance_m_s"],
        }
        # Decide the completion flag before writing it: ``exit_code`` starts at 1,
        # so reading it here without reassigning first would report every clean run
        # as ``runtime_completed: false``.
        exit_code = 0 if not errors else 1
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
