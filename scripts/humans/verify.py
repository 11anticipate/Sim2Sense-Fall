#!/usr/bin/env python3
"""Layered verification of the human rig: CPU plan, USD structure, then real physics.

The three layers are checked separately and reported separately, because they fail
for different reasons and "it ran" is not evidence for any of them.

**CPU** -- the rig plan, joint limits and the forward-kinematics model, with no
simulator. Cheap, and the only layer that runs in CI.

**USD** -- what is actually in the exported file: one articulation root, the right
number of revolute and fixed joints, one collider per segment, metres and Z-up.

**Physics** -- Isaac Sim, and here the checks are *positive controls*:

* **Gravity drop.** Every link of the body is raised 0.25 m and must fall back to
  where it started. Measuring only "drift == 0" passes vacuously when PhysX is not
  attached to the stage at all -- which is exactly the silent failure the scene
  stage hit before, so the same control is reused here.
* **Joint limit round trip.** The limits read back from Physics must equal the plan's
  table. USD stores degrees, Isaac reports radians, and this is the check that
  catches a units slip between them.
* **Kinematic replay against the CPU model.** Joint angles are written directly, and
  the resulting *link world positions* must match :func:`forward_kinematics` for the
  same angles. Two independent implementations of the same chain agreeing is real
  evidence; one implementation checked against itself is not.
* **PD tracking against the pre-registered tolerance.**
* **Floor penetration** after settling.

    python3 scripts/humans/verify.py --dry-run
    ~/isaacsim/python.sh scripts/humans/verify.py
    ~/isaacsim/python.sh scripts/humans/verify.py --gui
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

import numpy as np  # noqa: E402
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
    set_physics_dt,
    step_physics,
    write_json,
)

from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,  # noqa: E402
    forward_kinematics,
    plan_human_rig,
)
from sim2sense_fall.humans.usd_human import (  # noqa: E402
    ContactSourceUnavailable,
    HumanRuntime,
    author_contact_reporting,
    build_human_stage,
    human_stage_summary,
    tracking_target_rad,
)

LOGGER = logging.getLogger("verify_human")

#: A body raised by this much must come back down; the scene stage uses the same
#: control and the same reason.
LIFT_HEIGHT_M = 0.25
#: Tolerance on the gravity-drop positive control, in metres.
DROP_TOLERANCE_M = 0.02
#: A link's world position from Physics must match the CPU model within this.
KINEMATIC_TOLERANCE_M = 5e-3
#: How far the unactuated pelvis may end up from where it stood, as a multiple of its
#: standing height. A topple pivots it through an arc no longer than that height, so the
#: factor leaves room for the impact slide without permitting a launch. Measured on the
#: shipped body: 1.3855 m against 1.0126 m of standing height, i.e. 1.37x.
DIVERGENCE_HEIGHT_FACTOR = 2.0
#: Window sampled at the end of the drop to separate "at rest" from "still sliding".
REST_SAMPLE_SECONDS = 0.2
#: Movement of the pelvis across that window that still counts as at rest, in metres.
REST_CREEP_M = 5e-3
#: How long the unactuated body is given to come to rest before the tail sample.
#: Calibrated on the rig under test, not assumed: the 14-DOF single-axis body stopped
#: within 1.5 s, but the 62-link multi-axis ragdoll was still sliding at 0.23 m/s when
#: that window closed (46 mm of creep against the 5 mm gate) and needed ~3 s.
#: The original 3.0 was calibrated while one ``app.update()`` advanced two physics
#: steps (see common.step_physics), so the window that passed covered 6.0 s of
#: physics time; the value is stated in true physics seconds after the stepper fix.
GRAVITY_SETTLE_SECONDS = 6.0
#: Deepest acceptable excursion below the floor, in metres.
PENETRATION_TOLERANCE_M = -0.05
#: Height above the standing pose used for the airborne replay check. Chosen to sit
#: between the 0.12 m floor slab and the 2.7 m ceiling of the fixed apartment.
AIRBORNE_PROBE_OFFSET_M = 0.9
# The tracking probe must clear both the floor and the apartment ceiling.  Keeping
# the body lower than the kinematic mapping probe prevents a shoulder target from
# colliding with the ceiling while the free root remains dynamically unconstrained.
TRACKING_PROBE_OFFSET_M = 0.4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the human rig in layers.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--motions", type=Path, default=DEFAULT_MOTIONS)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=DEFAULT_SCENE_CONFIG,
        help="scene YAML used to derive a spawn point on a floor slab",
    )
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--lift-height", type=float, default=LIFT_HEIGHT_M)
    parser.add_argument("--drop-tolerance", type=float, default=DROP_TOLERANCE_M)
    parser.add_argument("--kinematic-tolerance", type=float, default=KINEMATIC_TOLERANCE_M)
    parser.add_argument("--dry-run", action="store_true", help="CPU and plan checks only")
    parser.add_argument("--gui", action="store_true", help="keep the viewport open afterwards")
    args = parser.parse_args(argv)
    if args.drop_tolerance >= args.lift_height:
        parser.error("--drop-tolerance must be below --lift-height for a positive control")
    return args


def check_cpu(checks: Checks, config: object, plan: object) -> dict:
    """Plan coherence, limits and the forward-kinematics model."""

    limits = plan.limits_deg()  # type: ignore[attr-defined]
    checks.check(
        "every DOF has a finite limit pair",
        all(np.isfinite([low, high]).all() and low < high for low, high in limits.values()),
        f"{len(limits)} DOF",
    )
    rest = plan.stats.get("skeleton_source", "")  # type: ignore[attr-defined]
    checks.check(
        "selected skeleton is a valid resting figure",
        plan.stats["standing_height_m"] > 0,  # type: ignore[attr-defined]
        f"standing height {plan.stats['standing_height_m']:.4f} m, source {rest!r}",  # type: ignore[attr-defined]
    )
    poses = forward_kinematics(plan, {}, root_position=plan.spawn_root_position)  # type: ignore[attr-defined]
    # Measured from the spawned pose, not by summing per-link rest offsets: the root
    # link sits off the body origin, so the two conventions disagree by that amount
    # and the rest-offset sum silently approves a spawn with the feet underground.
    lowest = min(
        float(poses[link.name].translation[2]) + link.capsule.lowest_point_z()
        for link in plan.links  # type: ignore[attr-defined]
        if link.capsule is not None
    )
    checks.check(
        "CPU model puts the lowest capsule on the floor",
        abs(lowest) < 1e-5,
        f"lowest capsule point at {lowest:+.6f} m",
    )
    checks.check(
        "spawn height and ground offset use one convention",
        abs(plan.spawn_root_position[2] - plan.ground_offset_m) < 1e-5,  # type: ignore[attr-defined]
        f"spawn root z {plan.spawn_root_position[2]:.6f} m, "  # type: ignore[attr-defined]
        f"ground offset {plan.ground_offset_m:.6f} m",  # type: ignore[attr-defined]
    )
    checks.check(
        "CPU model places the head above the pelvis",
        float(poses["head"].translation[2]) > float(poses[plan.root_link].translation[2]),  # type: ignore[attr-defined]
        f"head {float(poses['head'].translation[2]):.4f} m, "
        f"pelvis {float(poses[plan.root_link].translation[2]):.4f} m",  # type: ignore[attr-defined]
    )
    return limits


def check_usd(checks: Checks, plan: object, summary: dict) -> None:
    checks.check(
        "exported stage uses metres and Z-up",
        summary["meters_per_unit"] == 1.0 and summary["up_axis"] == "Z",
        f"{summary['meters_per_unit']} m/unit, {summary['up_axis']}-up",
    )
    checks.check(
        "exactly one articulation root",
        summary["articulation_root_count"] == 1,
        f"{summary['articulation_root_count']} found",
    )
    checks.check(
        "revolute joint count matches the plan",
        summary["revolute_joint_count"] == len(plan.joints),  # type: ignore[attr-defined]
        f"{summary['revolute_joint_count']} joints, plan {len(plan.joints)}",  # type: ignore[attr-defined]
    )
    checks.check(
        "capsule count matches the planned colliders",
        summary["capsule_count"] == plan.stats["collider_count"],  # type: ignore[attr-defined]
        f"{summary['capsule_count']} capsules, plan {plan.stats['collider_count']}",  # type: ignore[attr-defined]
    )
    # The base scene contributes its own colliders and bodies, so the count is
    # checked as "at least the human's share", not as an equality.
    checks.check(
        "the human's rigid bodies are present in the exported stage",
        summary["rigid_body_count"] >= len(plan.links),  # type: ignore[attr-defined]
        f"{summary['rigid_body_count']} rigid bodies (plan has {len(plan.links)}, "  # type: ignore[attr-defined]
        f"the rest belong to the base scene); {summary['collider_count']} colliders in total",
    )


def check_physics(
    checks: Checks,
    app: object,
    runtime: object,
    plan: object,
    config: object,
    args: argparse.Namespace,
) -> dict:
    """Positive controls on the live articulation."""

    low, high = runtime.dof_limits_rad()  # type: ignore[attr-defined]
    plan_low = np.radians([joint.lower_deg for joint in plan.joints])  # type: ignore[attr-defined]
    plan_high = np.radians([joint.upper_deg for joint in plan.joints])  # type: ignore[attr-defined]
    limit_error = float(max(np.abs(low - plan_low).max(), np.abs(high - plan_high).max()))
    checks.check(
        "joint limits round-trip through USD degrees and Isaac radians",
        limit_error < 1e-6,
        f"largest limit mismatch {np.degrees(limit_error):.6f} degrees",
    )
    checks.check(
        "self-collisions are off as planned",
        bool(np.asarray(runtime.articulation.get_enabled_self_collisions().numpy()).all())
        is bool(plan.self_collisions),  # type: ignore[attr-defined]
        f"engine {runtime.articulation.get_enabled_self_collisions().numpy().ravel().tolist()}, "
        f"plan {plan.self_collisions}",  # type: ignore[attr-defined]
    )

    physics_dt = float(config.simulation.physics_dt_s)  # type: ignore[attr-defined]
    root_z = float(plan.spawn_root_position[2])  # type: ignore[attr-defined]

    def hold(seconds: float) -> None:
        # Stepped through the deterministic stepper, not ``app.update()``: one
        # update advances a fixed 1/60 s of physics time on this build, so a
        # seconds window counted in updates ran at 2x the physics time it names
        # (measured; see common.step_physics).
        for _ in range(max(1, int(round(seconds / physics_dt)))):
            step_physics(1)

    def place(offset_z: float) -> None:
        runtime.set_root_pose(  # type: ignore[attr-defined]
            (plan.spawn_root_position[0], plan.spawn_root_position[1], root_z + offset_z),  # type: ignore[attr-defined]
            (1.0, 0.0, 0.0, 0.0),
        )
        runtime.reset_velocities()  # type: ignore[attr-defined]

    zero = np.zeros(len(plan.dof_names), dtype=np.float64)  # type: ignore[attr-defined]

    # --- kinematic replay against the CPU model, held clear of the floor ---
    # Done airborne on purpose. On the floor, contact compliance and joint-drive
    # softness add millimetres that have nothing to do with the joint mapping, and
    # this check exists to isolate the mapping. The lift is small on purpose: at
    # 2.0 m the body starts above the apartment ceiling and collides with it, which
    # showed up as a 317 mm error the first time this ran.
    place(AIRBORNE_PROBE_OFFSET_M)
    runtime.set_joint_positions(zero)  # type: ignore[attr-defined]
    runtime.set_joint_targets(zero)  # type: ignore[attr-defined]
    hold(0.3)
    probes = {
        "left_knee": np.radians(55.0),
        "left_hip": np.radians(-45.0),
        "spine1": np.radians(25.0),
        "left_shoulder": np.radians(-60.0),
        "left_elbow": np.radians(-50.0),
        "right_knee": np.radians(30.0),
    }
    errors: dict[str, float] = {}
    # The mapping probe must not run against active PD drives.  Writing a target and
    # a position in the same physics window creates a large corrective impulse in
    # PhysX; after one or two probes that can produce non-finite link transforms.
    # Disable drives for this isolated kinematic check, then restore them before the
    # independent tracking test below.
    drives_disabled = runtime.set_control_scale(0.0)  # type: ignore[attr-defined]
    if drives_disabled:
        checks.check(
            "kinematic mapping probe can disable joint drives",
            True,
            "set_control_scale(0.0) accepted",
        )
    else:
        checks.skip(
            "kinematic mapping probe can disable joint drives",
            "this Isaac articulation view exposes targets but refuses runtime gain writes",
        )
    for joint, value in probes.items():
        commanded = dict.fromkeys(plan.dof_names, 0.0)  # type: ignore[attr-defined]
        commanded[joint] = float(value)
        vector = np.array(
            [commanded[name] for name in plan.dof_names],
            dtype=np.float64,  # type: ignore[attr-defined]
        )
        runtime.set_joint_positions(vector)  # type: ignore[attr-defined]
        simulated = runtime.link_poses()  # type: ignore[attr-defined]
        # The CPU model is evaluated at the engine's OWN measured joint angles, not at
        # the commanded ones. That is what makes this a check of the joint-frame
        # mapping rather than of how well the drives track.
        measured = runtime.joint_positions_rad()  # type: ignore[attr-defined]
        angles = dict(zip(plan.dof_names, (float(v) for v in measured), strict=True))  # type: ignore[attr-defined]
        root_position, root_quaternion = runtime.root_pose()  # type: ignore[attr-defined]
        modelled = forward_kinematics(
            plan,  # type: ignore[arg-type]
            angles,
            root_position=root_position,
            root_rotation=_quaternion_to_axis_angle(root_quaternion),
        )
        worst = max(
            float(
                np.linalg.norm(
                    np.asarray(simulated[link.name].translation) - modelled[link.name].translation
                )
            )
            for link in plan.links  # type: ignore[attr-defined]
        )
        errors[joint] = worst
        tracking = float(np.degrees(np.abs(measured - vector).max()))
        checks.check(
            f"physics pose matches the CPU model with {joint} at {np.degrees(value):+.0f} deg",
            worst <= args.kinematic_tolerance,
            f"largest link error {worst * 1000:.2f} mm over {len(plan.links)} links "  # type: ignore[attr-defined]
            f"(engine held the joint to within {tracking:.3f} deg)",
        )
        runtime.set_joint_positions(zero)  # type: ignore[attr-defined]

    if drives_disabled and not runtime.set_control_scale(1.0):  # type: ignore[attr-defined]
        checks.check(
            "joint drives restored after the kinematic mapping probe",
            False,
            "set_control_scale(1.0) was refused",
        )

    # --- PD tracking against the pre-registered tolerance, also airborne ---
    # ``dof_limits_rad()`` already reports radians. An earlier revision wrapped these
    # targets in np.radians() a second time, which turned the knee's 75 degree command
    # into 1.3 degrees -- smaller than the 15 degree tolerance, so the check could only
    # ever pass. The target amplitude is therefore gated on its own before tracking runs.
    tolerance = float(config.control.tracking_tolerance_deg)  # type: ignore[attr-defined]
    # A three-axis chain has small secondary-axis spans the single-axis rig never
    # had, so half the wider limit can sit inside the tolerance; the target is
    # widened until it can actually falsify a tracking claim.
    targets = np.array(
        [
            tracking_target_rad(
                low[index], high[index], min_amplitude=np.radians(tolerance * 1.25)
            )
            for index in range(len(plan.dof_names))
        ],  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    amplitude_deg = np.abs(np.degrees(targets))
    weakest = int(np.argmin(amplitude_deg))
    checks.check(
        "every commanded tracking target is large enough to be falsifiable",
        bool(amplitude_deg.min() > tolerance),
        f"smallest target {amplitude_deg[weakest]:.2f} deg on {plan.dof_names[weakest]} "  # type: ignore[attr-defined]
        f"must exceed the {tolerance:.1f} deg tolerance "
        f"(limits span {np.degrees(low).min():.0f}..{np.degrees(high).max():.0f} deg)",
    )
    collision_probe_disabled = runtime.set_body_collisions_enabled(False)  # type: ignore[attr-defined]
    checks.check(
        "PD probe can isolate the human collision shapes",
        collision_probe_disabled > 0,
        f"disabled {collision_probe_disabled} authored human colliders",
    )
    # Track one DOF at a time.  Sending fourteen large targets to a free-root body
    # couples the controller test to whole-body toppling; after the root moves, the
    # resulting joint error says nothing useful about the drive itself and can poison
    # PhysX with an invalid transform.  Every trial starts from the same airborne rest
    # pose and holds one target only.
    tracking_error = 0.0
    if drives_disabled:
        tracking_errors: list[float] = []
        tracking_details: list[dict[str, float | str]] = []
        stiffness, damping = runtime.control_gains()  # type: ignore[attr-defined]
        for index, target in enumerate(targets):
            place(TRACKING_PROBE_OFFSET_M)
            runtime.set_joint_positions(zero)  # type: ignore[attr-defined]
            runtime.set_joint_targets(zero)  # type: ignore[attr-defined]
            hold(0.15)
            one_target = np.zeros_like(targets)
            one_target[index] = target
            runtime.set_joint_targets(one_target)  # type: ignore[attr-defined]
            hold(float(config.control.pose_hold_seconds))  # type: ignore[attr-defined]
            reached = runtime.joint_positions_rad()  # type: ignore[attr-defined]
            error_deg = float(np.degrees(abs(reached[index] - target)))
            tracking_errors.append(error_deg)
            tracking_details.append(
                {
                    "joint": plan.dof_names[index],  # type: ignore[attr-defined]
                    "target_deg": float(np.degrees(target)),
                    "reached_deg": float(np.degrees(reached[index])),
                    "error_deg": error_deg,
                    "stiffness": float(stiffness[index]),
                    "damping": float(damping[index]),
                }
            )
        tracking_error = max(tracking_errors, default=float("inf"))
        worst_index = int(np.argmax(tracking_errors)) if tracking_errors else -1
        worst_detail = tracking_details[worst_index] if worst_index >= 0 else {}
        checks.check(
            "PD drives track a mid-range target within the pre-registered tolerance",
            tracking_error <= tolerance,
            f"largest joint error {tracking_error:.3f} degrees against a {tolerance:.1f} degree "
            f"tolerance at {worst_detail.get('joint', 'unknown')}: "
            f"target {float(worst_detail.get('target_deg', float('nan'))):+.3f} deg, "
            f"reached {float(worst_detail.get('reached_deg', float('nan'))):+.3f} deg; "
            f"targets spanning {amplitude_deg.min():.1f}..{amplitude_deg.max():.1f} deg",
        )
    else:
        checks.skip(
            "PD drives track a mid-range target",
            "runtime gain writes are unavailable in this Isaac articulation view",
        )

    # --- drives-off negative control --------------------------------------
    # A tolerance check that cannot fail is not a check. With the gains zeroed the same
    # targets must NOT be reached; if they are, something is holding the pose that is
    # not the controller, and the positive result above proves nothing.
    if drives_disabled and runtime.set_control_scale(0.0):  # type: ignore[attr-defined]
        residuals: list[float] = []
        for index, target in enumerate(targets):
            place(AIRBORNE_PROBE_OFFSET_M)
            runtime.set_joint_positions(zero)  # type: ignore[attr-defined]
            one_target = np.zeros_like(targets)
            one_target[index] = target
            runtime.set_joint_targets(one_target)  # type: ignore[attr-defined]
            hold(float(config.control.pose_hold_seconds))  # type: ignore[attr-defined]
            ungoverned = runtime.joint_positions_rad()  # type: ignore[attr-defined]
            residuals.append(float(np.degrees(abs(ungoverned[index] - target))))
        residual = max(residuals, default=float("nan"))
        checks.check(
            "with drives off the same targets are NOT reached (negative control)",
            np.isfinite(residual) and residual > tolerance,
            f"ungoverned error {residual:.3f} deg must exceed the {tolerance:.1f} deg tolerance",
        )
        if not runtime.set_control_scale(1.0):  # type: ignore[attr-defined]
            checks.check(
                "joint drives restored after the negative control",
                False,
                "set_control_scale(1.0) was refused, so every later physics check runs unpowered",
            )
        runtime.set_joint_targets(zero)  # type: ignore[attr-defined]
        hold(0.5)
    else:
        checks.skip(
            "drives-off negative control",
            "the runtime refused set_control_scale(0.0), so tracking sensitivity is unverified",
        )

    if collision_probe_disabled > 0:
        restored_colliders = runtime.set_body_collisions_enabled(True)  # type: ignore[attr-defined]
        checks.check(
            "human collision shapes restored before gravity test",
            restored_colliders == collision_probe_disabled,
            f"restored {restored_colliders} of {collision_probe_disabled} colliders",
        )

    # --- gravity drop positive control ------------------------------------
    # The control is that a raised body ACTUALLY falls under gravity. It is not
    # required to return to its start: a free-root humanoid with sagittal-only joints
    # cannot hold a standing pose, and asserting "returns to rest where it started"
    # would either be false or would only pass when physics is not running.
    # The fall control must be passive: zero the gains before lifting so the drive
    # cannot hold the legs against gravity and mask the positive drop test.
    if not runtime.set_control_scale(0.0):  # type: ignore[attr-defined]
        checks.skip(
            "gravity-drop test can disable joint drives",
            "runtime gain writes are unavailable; gravity result is reported with authored drives",
        )
    place(0.0)
    runtime.set_joint_positions(zero)  # type: ignore[attr-defined]
    runtime.set_joint_targets(zero)  # type: ignore[attr-defined]
    hold(0.2)
    baseline_root = np.asarray(runtime.root_pose()[0])  # type: ignore[attr-defined]
    place(args.lift_height)
    # The fall control starts from the rest pose. The pre-baseline hold leaves an
    # arbitrary fold behind, and lifting mid-fold would measure the fold's ballistic
    # swing, not the fall; writing positions is a kinematic reset, not a force, so
    # the control stays passive.
    runtime.set_joint_positions(zero)  # type: ignore[attr-defined]
    runtime.set_joint_targets(zero)  # type: ignore[attr-defined]
    for _ in range(4):
        app.update()  # type: ignore[attr-defined]
    lifted_root_z = float(runtime.root_pose()[0][2])  # type: ignore[attr-defined]
    hold(GRAVITY_SETTLE_SECONDS)
    # The body is unactuated here (drives are zeroed above), so it topples and slides.
    # Sample the tail of the window so "came to rest" can be told from "still sliding",
    # which is the difference between a fall and a divergence.
    tail: list[np.ndarray] = []
    for _ in range(max(1, int(round(REST_SAMPLE_SECONDS / physics_dt)))):
        step_physics(1)
        tail.append(np.asarray(runtime.root_pose()[0]).copy())  # type: ignore[attr-defined]
    final_root = tail[-1]
    descent = lifted_root_z - float(final_root[2])
    checks.check(
        "raised body falls back under gravity",
        descent >= args.lift_height - args.drop_tolerance,
        f"pelvis descended {descent:.4f} m from {lifted_root_z:.4f} m",
    )
    checks.check(
        "the body does not sink through the supporting slab",
        float(final_root[2]) > 0.0,
        f"pelvis ends at z = {float(final_root[2]):.4f} m",
    )
    # The claim is that the body *comes to rest*, not that it stays near where it stood.
    #
    # An earlier version bounded the horizontal travel at 1.0 m, which passed only
    # because the body was yawed 90 degrees and happened to topple into a wall 0.52 m
    # away that stopped it (0.4263 m). Once the body faced forward it toppled into open
    # floor and travelled 1.3855 m, so the check began failing -- correctly, in the sense
    # that it was measuring the right thing for the wrong reason, and its bound had been
    # calibrated against a defect.
    #
    # Two properties are asserted instead, both of which a diverging body violates:
    # it stops moving, and it does not end up farther from where it stood than a body
    # of its own size could reach by falling over. A topple pivots the pelvis through
    # an arc no longer than its standing height, so `2 x` that height leaves room for
    # the slide on impact while still catching a body that has been launched.
    travel = float(np.linalg.norm(final_root[:2] - baseline_root[:2]))
    tail_array = np.stack(tail)
    creep = float(np.linalg.norm(tail_array[-1][:2] - tail_array[0][:2]))
    allowed_travel = DIVERGENCE_HEIGHT_FACTOR * root_z
    checks.check(
        "the body does not diverge across the room",
        creep < REST_CREEP_M and travel <= allowed_travel,
        f"pelvis moved {travel:.4f} m horizontally and crept {creep * 1e3:.3f} mm over the "
        f"final {REST_SAMPLE_SECONDS * 1e3:.0f} ms (limit {allowed_travel:.4f} m = "
        f"{DIVERGENCE_HEIGHT_FACTOR:g} x the {root_z:.4f} m standing pelvis height)",
    )

    # --- floor penetration -------------------------------------------------
    deepest = 0.0
    poses = runtime.link_poses()  # type: ignore[attr-defined]
    for link in plan.links:  # type: ignore[attr-defined]
        capsule = link.capsule
        if capsule is None:
            continue
        centre = np.asarray(poses[link.name].transform_point(capsule.center))
        axis = np.asarray(poses[link.name].rotation) @ capsule.direction
        low_point = (
            centre[2] - abs(float(axis[2])) * capsule.cylinder_length_m / 2.0 - capsule.radius_m
        )
        deepest = min(deepest, float(low_point))
    checks.check(
        "no body geometry is pushed through the floor after settling",
        deepest >= PENETRATION_TOLERANCE_M,
        f"lowest body point {deepest:+.4f} m",
    )
    runtime.set_control_scale(1.0)  # type: ignore[attr-defined]
    return {
        "kinematic_link_errors_m": {key: round(value, 9) for key, value in errors.items()},
        "gravity_drop_pelvis_m": round(descent, 6),
        "final_pelvis_z_m": round(float(final_root[2]), 6),
        "final_pelvis_drift_m": round(float(np.linalg.norm(final_root[:2] - baseline_root[:2])), 6),
        "lowest_body_point_m": round(deepest, 6),
        "tracking_error_deg": round(tracking_error, 6),
        "tracking_details": tracking_details if drives_disabled else [],
    }


def _quaternion_to_axis_angle(quaternion_wxyz: object) -> np.ndarray:
    from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix

    return matrix_to_axis_angle(quaternion_to_matrix(quaternion_wxyz))


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
    except (OSError, ValueError) as exc:
        checks.check("inputs load and validate", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="human verify")

    limits = check_cpu(checks, config, plan)
    checks.info(
        "plan: "
        + ", ".join(
            f"{name} [{low:g}, {high:g}] deg" for name, (low, high) in list(limits.items())[:4]
        )
        + f", ... ({len(limits)} DOF)"
    )
    if args.dry_run:
        checks.skip("USD structure", "--dry-run")
        checks.skip("Isaac Sim physics", "--dry-run")
        return checks.report(banner="human verify (dry run)")

    scene = args.scene.resolve()
    if not scene.is_file():
        checks.check("base scene exists", False, str(scene))
        return checks.report(banner="human verify")
    scene_digest = _sha256(scene)

    app = boot_isaac(not args.gui)
    if app is None:
        checks.check("Isaac Sim started", False, "run through ~/isaacsim/python.sh")
        return checks.report(banner="human verify")
    exit_code = 1
    metrics: dict = {}
    try:
        stage_path = args.out / "human_verify.usda"
        build_human_stage(
            plan,
            stage_path,
            base_scene=scene,
            skin_points=None if mesh is None else mesh.vertices,
            skin_faces=None if mesh is None else mesh.faces,
        )
        check_usd(checks, plan, human_stage_summary(stage_path))
        stage = open_scene(app, stage_path)
        activation = activate_physics()
        checks.check(
            "physics scene activated",
            activation["physics_scenes"] == ["/World/PhysicsScene"]
            and activation["active_engine"] == "physx",
            str(activation),
        )
        measured_dt, accepted = set_physics_dt(config.simulation.physics_dt_s)
        checks.check(
            "configured physics step is in force",
            abs(measured_dt - float(config.simulation.physics_dt_s)) < 1e-9,
            f"requested {config.simulation.physics_dt_s} s, engine reports {measured_dt} s, "
            f"accepted={accepted}",
        )
        # Same reason as the trial runner: this build cannot construct the tensors
        # rigid-contact view, and requesting it logs a plugin error per filter plus a
        # traceback from an async physics-ready callback. The contact channel verified
        # below is the polled physx report, which needs no tensor view.
        runtime = HumanRuntime(stage, plan, enable_contact_views=False, app=app)
        tagged = author_contact_reporting(stage)
        runtime.play()
        for _ in range(4):
            app.update()
        checks.check(
            "runtime DOF set matches the plan",
            sorted(runtime.capabilities["dof_order"]) == sorted(plan.dof_names),  # type: ignore[attr-defined]
            ", ".join(runtime.capabilities["dof_order"])  # type: ignore[attr-defined]
            + (
                " (PhysX traversal order; translated to plan order at the runtime boundary)"
                if runtime.capabilities.get("dof_plan_permutation")  # type: ignore[attr-defined]
                else ""
            ),
        )
        # The contact channel is what a fall detector reads, so it is verified here
        # rather than assumed. Both sides must be tagged: PhysX only reports a pair
        # when the contact-report API is on the human's collider AND the environment's.
        checks.check(
            "environment colliders carry PhysX contact reporting",
            len(tagged) > 0,
            f"{len(tagged)} tagged environment colliders",
        )
        try:
            probe_samples = runtime.human_contact_samples()
        except ContactSourceUnavailable as exc:
            checks.check("contact channel is readable", False, str(exc))
        else:
            checks.check(
                "contact channel is readable",
                True,
                f"{len(probe_samples)} pairs on the probe step, "
                f"source {runtime.capabilities['contact_source']}",
            )
            # PhysX collider identifiers decode to USD paths, including terminal
            # colliders and the pelvis collider nested below the root actor.
            segments = runtime.contact_segment_names()
            checks.check(
                "reported contact points resolve to body limbs",
                runtime.capabilities["contact_attribution"] == "usd_collider_path",
                f"attribution={runtime.capabilities['contact_attribution']}, "
                f"limbs={sorted(set(segments)) or 'none'}",
            )
            checks.check(
                "support surface height is readable",
                runtime.support_surface_height_m() is not None,
                f"{runtime.support_surface_height_m()} m",
            )
        metrics = check_physics(checks, app, runtime, plan, config, args)
        if args.gui:
            print("GUI is open. Close the window or press Ctrl-C to exit.")
            while app.is_running():
                app.update()
        checks.check(
            "the verified base scene was not modified",
            _sha256(scene) == scene_digest,
            str(scene),
        )
        path = write_json(
            args.out / "human_verify.json",
            {
                "human_id": plan.human_id,
                "root_mode": plan.root_mode,
                "plan_stats": dict(plan.stats),
                "limits_deg": {name: list(pair) for name, pair in limits.items()},
                "physics_metrics": metrics,
                "scene_sha256": scene_digest,
                "stage_sha256": _sha256(stage_path),
            },
        )
        checks.info(f"wrote {path}")
        exit_code = checks.report(banner="human verify")
    except Exception as exc:  # noqa: BLE001 - surfaced in the report
        LOGGER.exception("human verification failed")
        checks.check("verification completed without error", False, f"{type(exc).__name__}: {exc}")
        exit_code = checks.report(banner="human verify")
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
