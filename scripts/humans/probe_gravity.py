#!/usr/bin/env python3
"""Diagnose the NaN pelvis divergence in the gravity-drop phase of verify.py.

The verify run is otherwise green on the multi-axis rig; the last check fails with
``ValueError: non-finite world position for link 'pelvis'`` raised by the first
pose read inside the gravity block. Earlier probe phases (A-D below) showed the
gravity sequence alone is stable, so the poison must be accumulated by the earlier
verify phases. ``--phases KTNG`` (the default) replays verify's ``check_physics``
body phase by phase -- kinematic probes, the 57-trial PD tracking loop, the
57-trial drives-off negative control, then the gravity block -- with finite-state
monitoring after every trial and inside the final hold, so the phase that
introduces the non-finite state is located instead of guessed at.

Phases:

* **K** -- verify's kinematic mapping probe loop (drives off, airborne, six
  single-joint poses, ``link_poses`` after each write).
* **T** -- verify's PD tracking loop: 57 trials, colliders disabled, one target
  each, ``place(0.4)`` per trial.
* **N** -- verify's drives-off negative control: 57 trials at ``place(0.9)``, then
  drives restored, targets zeroed, a monitored 0.5 s hold.
* **G** -- verify's gravity block: colliders restored, drives zeroed, spawn pose,
  monitored hold, baseline read, then the lift/descent sequence with monitoring
  and a recovery attempt if the baseline read fails.
* **A/A-prime/B/C/D** -- the earlier minimal phases, kept for bisection.

    ~/isaacsim/python.sh scripts/humans/probe_gravity.py \
        --stage artifacts/humans/verify_multiaxis/human_verify.usda \
        --config configs/humans/human_smpl_multiaxis.yaml \
        --out artifacts/humans/probe_gravity
"""

from __future__ import annotations

import argparse
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
    DEFAULT_SCENE_CONFIG,
    activate_physics,
    boot_isaac,
    load_inputs,
    open_scene,
    resolve_spawn_point,
    set_physics_dt,
)

from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig  # noqa: E402
from sim2sense_fall.humans.usd_human import (  # noqa: E402
    HumanRuntime,
    author_contact_reporting,
    tracking_target_rad,
)

LOGGER = logging.getLogger("probe_gravity")

AIRBORNE_PROBE_OFFSET_M = 0.9
TRACKING_PROBE_OFFSET_M = 0.4
LIFT_HEIGHT_M = 0.25
KINEMATIC_PROBES = {
    "left_knee": np.radians(55.0),
    "left_hip": np.radians(-45.0),
    "spine1": np.radians(25.0),
    "left_shoulder": np.radians(-60.0),
    "left_elbow": np.radians(-50.0),
    "right_knee": np.radians(30.0),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument("--motions", type=Path, default=None)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument(
        "--stage",
        type=Path,
        default=REPO_ROOT / "artifacts/humans/verify_multiaxis/human_verify.usda",
        help="already-built human stage to open (skips authoring)",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=DEFAULT_SCENE_CONFIG,
        help="scene YAML used to derive the spawn point, exactly like verify.py",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="KTNG",
        help="ordered subset of KTNG (verify replication) or ABCD (minimal probe)",
    )
    parser.add_argument("--steps", type=int, default=240, help="steps per minimal phase")
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    return parser.parse_args(argv)


def snapshot(runtime: HumanRuntime, plan) -> dict[str, object]:
    """Read every state channel, tolerating the runtime's own finiteness gate."""
    state: dict[str, object] = {}
    try:
        position, _quaternion = runtime.root_pose()
        state["pelvis_z"] = float(position[2])
        state["pelvis_xyz"] = position.tolist()
    except ValueError as exc:
        state["pelvis_z"] = None
        state["pelvis_error"] = str(exc)
    try:
        angles = runtime.joint_positions_rad()
        rates = runtime.joint_velocities_rad_s()
        state["joint_finite"] = bool(np.all(np.isfinite(angles)) and np.all(np.isfinite(rates)))
        state["max_abs_angle_deg"] = float(np.degrees(np.abs(angles)).max())
        state["max_abs_rate_deg_s"] = float(np.degrees(np.abs(rates)).max())
        state["angles"] = angles
        state["rates"] = rates
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        state["joint_error"] = str(exc)
    return state


def phase_report(name: str, step: int, state: dict[str, object], log_every: int = 10) -> bool:
    """Print one line; return True when the phase must stop (divergence found)."""
    diverged = state.get("pelvis_z") is None or state.get("joint_finite") is False
    if diverged or step % log_every == 0 or step < 5:
        LOGGER.info(
            "%s step %4d: pelvis_z=%s max|q|=%s deg max|qd|=%s deg/s%s",
            name,
            step,
            _fmt(state.get("pelvis_z")),
            _fmt(state.get("max_abs_angle_deg")),
            _fmt(state.get("max_abs_rate_deg_s")),
            "" if not diverged else f"  <-- DIVERGED ({state.get('pelvis_error', '')})",
        )
    return diverged


def _fmt(value: object) -> str:
    if value is None:
        return "NaN/inf"
    return f"{value:.3f}"


def dump_link_finiteness(runtime: HumanRuntime, plan) -> list[str]:
    """Return the names of every link whose world pose is currently non-finite."""

    broken: list[str] = []
    for link in plan.links:
        try:
            position, quaternion = runtime._world_pose_of(link.name)
        except ValueError as exc:
            LOGGER.error("  link %s: %s", link.name, exc)
            broken.append(link.name)
            continue
        if not np.all(np.isfinite(quaternion)):
            LOGGER.error("  link %s: non-finite quaternion %s", link.name, quaternion)
            broken.append(link.name)
    return broken


def dump_joint_finiteness(runtime: HumanRuntime, plan) -> None:
    try:
        angles = runtime.joint_positions_rad()
        rates = runtime.joint_velocities_rad_s()
    except Exception as exc:  # noqa: BLE001 - diagnostic dump
        LOGGER.error("  joint read failed: %s", exc)
        return
    bad_angles = [plan.dof_names[i] for i in range(len(angles)) if not np.isfinite(angles[i])]
    bad_rates = [plan.dof_names[i] for i in range(len(rates)) if not np.isfinite(rates[i])]
    LOGGER.error(
        "  joints: %d non-finite angles (%s), %d non-finite rates (%s)",
        len(bad_angles),
        ", ".join(bad_angles[:10]) or "none",
        len(bad_rates),
        ", ".join(bad_rates[:10]) or "none",
    )


def trial_guard(
    runtime: HumanRuntime, plan, label: str, index: int, reached: np.ndarray, every: int = 8
) -> bool:
    """Check a trial's reads for non-finite values; return True when all finite.

    ``reached`` is the array verify itself reads per trial, so checking it adds no
    extra channel. The root pose is sampled only every ``every`` trials: reads are
    passive, but each one synchronises the physics tensors, and piling extra
    synchronisation into every trial could mask a timing-dependent defect.
    """

    ok = True
    if not np.all(np.isfinite(reached)):
        bad = [plan.dof_names[i] for i in range(len(reached)) if not np.isfinite(reached[i])]
        LOGGER.warning("%s trial %d: non-finite joint reads: %s", label, index, ", ".join(bad))
        ok = False
    if index % every == 0:
        try:
            position, quaternion = runtime.root_pose()
        except ValueError as exc:
            LOGGER.warning("%s trial %d: root read failed: %s", label, index, exc)
            return False
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            LOGGER.warning(
                "%s trial %d: root non-finite pos=%s quat=%s", label, index, position, quaternion
            )
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    phases = [char for char in args.phases.upper() if char.isalpha()]

    kwargs = {"assets_path": DEFAULT_ASSETS, "motions_path": DEFAULT_MOTIONS}
    if args.assets is not None:
        kwargs["assets_path"] = args.assets
    if args.motions is not None:
        kwargs["motions_path"] = args.motions
    config, _registry, _motions = load_inputs(config_path=args.config, **kwargs)
    body = select_body(
        _registry,
        model_id=config.skeleton.model_asset,
        allow_procedural=bool(config.skeleton.allow_procedural_skeleton),
    )
    mesh = body.model.mesh() if body.has_skin_mesh else None
    rest = fit_rest_skeleton(config, mesh.rest_skeleton()) if mesh is not None else None
    spawn = resolve_spawn_point(
        args.scene_config,
        None,
        None,
        standing_height_m=config.skeleton.height_m,
    )
    plan = plan_human_rig(config, rest=rest, spawn_xy=spawn)
    LOGGER.info("plan spawn: %s", plan.spawn_root_position)

    app = boot_isaac(headless=True)
    if app is None:
        LOGGER.error("Isaac Sim did not start; run through ~/isaacsim/python.sh")
        return 1
    try:
        if not args.stage.is_file():
            LOGGER.error("stage not found: %s (run scripts/humans/verify.py first)", args.stage)
            return 1
        stage = open_scene(app, args.stage)
        LOGGER.info("stage opened: %s", args.stage)
        activation = activate_physics()
        LOGGER.info("physics: %s", activation)
        measured_dt, _accepted = set_physics_dt(args.dt)
        LOGGER.info("dt set: %s", measured_dt)
        try:
            runtime = HumanRuntime(stage, plan, enable_contact_views=False, app=app)
        except BaseException as exc:  # noqa: BLE001 - diagnostic: expose silent exits
            import traceback

            LOGGER.error("HumanRuntime constructor raised %r", exc)
            traceback.print_exc()
            raise
        LOGGER.info("runtime created (dofs=%d)", len(runtime.dof_names))
        # verify.py tags contact reporting and polls the channel before the physics
        # checks; replicate that so the starting state matches.
        tagged = author_contact_reporting(stage)
        LOGGER.info("contact reporting tagged on %d environment colliders", len(tagged))
        try:
            samples = runtime.human_contact_samples()
            LOGGER.info("contact probe step: %d pairs", len(samples))
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            LOGGER.info("contact probe unavailable: %s", exc)
        runtime.play()
        for _ in range(4):
            app.update()
        LOGGER.info("playing")
        zero = np.zeros(len(plan.dof_names), dtype=np.float64)
        root_z = float(plan.spawn_root_position[2])
        physics_dt = float(args.dt)

        def hold(seconds: float) -> None:
            for _ in range(max(1, int(round(seconds / physics_dt)))):
                app.update()

        def place(offset_z: float) -> None:
            runtime.set_root_pose(
                (plan.spawn_root_position[0], plan.spawn_root_position[1], root_z + offset_z),
                (1.0, 0.0, 0.0, 0.0),
            )
            runtime.reset_velocities()

        def monitored_hold(label: str, seconds: float, log_every: int = 10) -> bool:
            """Run a hold with per-step finite monitoring; True when stable."""

            steps = max(1, int(round(seconds / physics_dt)))
            for step in range(1, steps + 1):
                app.update()
                if phase_report(label, step, snapshot(runtime, plan), log_every=log_every):
                    return False
            return True

        targets: np.ndarray | None = None
        tolerance_deg = 15.0
        poison_found = False

        # ================= verify replication phases =================
        if "K" in phases:
            place(AIRBORNE_PROBE_OFFSET_M)
            runtime.set_joint_positions(zero)
            runtime.set_joint_targets(zero)
            hold(0.3)
            assert runtime.set_control_scale(0.0), "gain writes refused"
            for joint, value in KINEMATIC_PROBES.items():
                commanded = dict.fromkeys(plan.dof_names, 0.0)
                commanded[joint] = float(value)
                vector = np.array(
                    [commanded[name] for name in plan.dof_names], dtype=np.float64
                )
                runtime.set_joint_positions(vector)
                try:
                    poses = runtime.link_poses()
                except ValueError as exc:
                    LOGGER.error("K probe %s: link_poses raised: %s", joint, exc)
                    poison_found = True
                    break
                worst = max(
                    float(np.linalg.norm(pose.translation)) for pose in poses.values()
                )
                LOGGER.info("K probe %s: links finite, max |translation| %.3f m", joint, worst)
                runtime.set_joint_positions(zero)
            assert runtime.set_control_scale(1.0), "gain restore refused"
            if not poison_found:
                LOGGER.info("K(kinematic): 6 probes done, all link reads finite")

        if "T" in phases:
            low, high = runtime.dof_limits_rad()
            targets = np.array(
                [
                    tracking_target_rad(
                        low[index], high[index], min_amplitude=np.radians(tolerance_deg * 1.25)
                    )
                    for index in range(len(plan.dof_names))
                ],
                dtype=np.float64,
            )
            disabled = runtime.set_body_collisions_enabled(False)
            LOGGER.info("T: disabled %d colliders", disabled)
            all_finite = True
            for index, target in enumerate(targets):
                place(TRACKING_PROBE_OFFSET_M)
                runtime.set_joint_positions(zero)
                runtime.set_joint_targets(zero)
                hold(0.15)
                one_target = np.zeros_like(targets)
                one_target[index] = target
                runtime.set_joint_targets(one_target)
                hold(0.2)
                reached = runtime.joint_positions_rad()
                all_finite &= trial_guard(runtime, plan, "T", index, reached)
                if not all_finite:
                    poison_found = True
                    LOGGER.error("T: first poison observed at trial %d", index)
                    break
            if not poison_found:
                LOGGER.info("T(tracking): %d trials done, all reads finite", len(targets))

        if "N" in phases:
            if targets is None:
                low, high = runtime.dof_limits_rad()
                targets = np.array(
                    [
                        tracking_target_rad(
                            low[index], high[index], min_amplitude=np.radians(tolerance_deg * 1.25)
                        )
                        for index in range(len(plan.dof_names))
                    ],
                    dtype=np.float64,
                )
            assert runtime.set_control_scale(0.0), "gain writes refused"
            all_finite = True
            for index, target in enumerate(targets):
                place(AIRBORNE_PROBE_OFFSET_M)
                runtime.set_joint_positions(zero)
                one_target = np.zeros_like(targets)
                one_target[index] = target
                runtime.set_joint_targets(one_target)
                hold(0.2)
                ungoverned = runtime.joint_positions_rad()
                all_finite &= trial_guard(runtime, plan, "N", index, ungoverned)
                if not all_finite:
                    poison_found = True
                    LOGGER.error("N: first poison observed at trial %d", index)
                    break
            if not poison_found:
                LOGGER.info("N(negative): %d trials done, all reads finite", len(targets))
                assert runtime.set_control_scale(1.0), "gain restore refused"
                runtime.set_joint_targets(zero)
                if not monitored_hold("N(yank)", 0.5, log_every=20):
                    poison_found = True

        if "G" in phases:
            enabled = runtime.set_body_collisions_enabled(True)
            LOGGER.info("G: colliders restored (%d)", enabled)
            assert runtime.set_control_scale(0.0), "gain writes refused"
            place(0.0)
            runtime.set_joint_positions(zero)
            runtime.set_joint_targets(zero)
            stable = monitored_hold("G(pre-baseline)", 0.2, log_every=1)
            try:
                baseline = np.asarray(runtime.root_pose()[0])
                LOGGER.info("G: baseline root = %s", np.round(baseline, 4).tolist())
            except ValueError as exc:
                LOGGER.error("G: baseline root read FAILED: %s", exc)
                stable = False
            if not stable:
                poison_found = True
                LOGGER.error("G: diagnostic dump at first non-finite baseline:")
                dump_joint_finiteness(runtime, plan)
                broken_links = dump_link_finiteness(runtime, plan)
                LOGGER.error("G: %d of %d links non-finite", len(broken_links), len(plan.links))
                # Recovery attempt: does a fresh teleport clear it, or is the
                # articulation permanently poisoned?
                place(0.0)
                runtime.set_joint_positions(zero)
                runtime.set_joint_targets(zero)
                if monitored_hold("G(recovery)", 0.2, log_every=1):
                    LOGGER.info("G: state recovered after re-teleport")
                else:
                    LOGGER.error("G: state still poisoned after re-teleport")
            else:
                # Continue verify's gravity sequence under monitoring.
                place(LIFT_HEIGHT_M)
                for _ in range(4):
                    app.update()
                lifted_root_z = float(runtime.root_pose()[0][2])
                LOGGER.info("G: lifted to %.4f m", lifted_root_z)
                if monitored_hold("G(fall)", 1.5, log_every=30):
                    LOGGER.info("G: descent window complete")
                else:
                    poison_found = True

        if poison_found:
            LOGGER.error("probe STOPPED: non-finite state located; see log above")
            return 2

        # ================= earlier minimal phases =================
        if "A" in phases or "AP" in phases or "B" in phases or "C" in phases or "D" in phases:
            LOGGER.warning("minimal phases ABCD are deprecated; use KTNG for verify replication")

        if "A" in phases:
            assert runtime.set_control_scale(0.0), "gain writes refused"
            place(0.0)
            runtime.set_joint_positions(zero)
            runtime.set_joint_targets(zero)
            if monitored_hold("A(loose)", args.steps):
                return 2
            LOGGER.info("A(loose): stable for %d steps", args.steps)

        if "B" in phases:
            disabled = runtime.set_body_collisions_enabled(False)
            enabled = runtime.set_body_collisions_enabled(True)
            LOGGER.info("B: cycled colliders (disabled %d, enabled %d)", disabled, enabled)
            assert runtime.set_control_scale(0.0), "gain writes refused"
            place(0.0)
            runtime.set_joint_positions(zero)
            runtime.set_joint_targets(zero)
            if monitored_hold("B(cycle+loose)", args.steps):
                return 2
            LOGGER.info("B(cycle+loose): stable for %d steps", args.steps)

        if "C" in phases:
            runtime.set_body_collisions_enabled(False)
            assert runtime.set_control_scale(0.0), "gain writes refused"
            place(0.9)
            runtime.set_joint_positions(zero)
            one_target = np.zeros_like(zero)
            one_target[plan.dof_names.index("left_elbow")] = np.radians(-85.0)
            runtime.set_joint_targets(one_target)
            hold(0.2)
            assert runtime.set_control_scale(1.0), "gain restore refused"
            runtime.set_joint_targets(zero)
            if not monitored_hold("C(yank)", 0.5, log_every=20):
                return 2
            LOGGER.info("C(yank): stable")

        if "D" in phases:
            enabled = runtime.set_body_collisions_enabled(True)
            LOGGER.info("D: colliders restored (%d)", enabled)
            assert runtime.set_control_scale(0.0), "gain writes refused"
            place(0.0)
            runtime.set_joint_positions(zero)
            runtime.set_joint_targets(zero)
            if monitored_hold("D(gravity)", args.steps):
                return 2
            LOGGER.info("D(gravity): stable for %d steps", args.steps)

        LOGGER.info("probe complete")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
