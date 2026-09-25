#!/usr/bin/env python3
"""Replay the live keyboard pre-step chain on CPU and audit what the stance IK injects.

``task_plan.md`` P0-B asks for a support-phase foot constraint. That constraint rewrites
leg joint targets every physics step, so its quality is invisible in the shipped metrics:
tracking error, slip and clearance all describe the *result*, not how much of it the IK
overwrote. A constraint that buys a planted foot with frontal-plane knee rotation looks
like a walk until you look at the knee.

What this measures, holding one key for a fixed window, with no Isaac Sim:

1. **Injection.** ``|stance.correct(target) - target|`` per DOF (max, mean, p95), the
   share of steps pinned at ``max_correction_rad``, and the worst value among the DOFs
   whose *measured* tip motion is sideways -- the plane that splays a knee.
2. **Geometry.** Inter-knee lateral span in the body frame, from forward kinematics of
   the teleop target and of the corrected target, so the number is what a viewer sees
   rather than a joint angle.
3. **Holding.** Anchor residual, how many foot-steps kept an anchor, and how many the
   controller released. An anchor created and released on the next step means the
   reference has no planted foot to hold: that is a source finding, not a tuning one.

``--variant NAME=KEY=VALUE`` rebuilds the stance controller with one override, so a
before/after pair runs the same teleop trajectory and the same gait windows.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    DEFAULT_MOTIONS,
    REPO_ROOT,
    load_inputs,
    write_json,  # type: ignore[import-not-found]
)

from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import (
    StanceFootController,
    fit_collision_capsules,
    sideways_leg_dofs,
)
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints
from sim2sense_fall.humans.rig import fit_rest_skeleton, forward_kinematics, plan_human_rig
from sim2sense_fall.humans.rotations import quaternion_to_matrix
from sim2sense_fall.humans.teleop import TeleopController, load_gait, load_keyboard_config

LOGGER = logging.getLogger("audit_stance_ik")


def build(settings: dict[str, Any], stance_overrides: dict[str, Any]):
    """The same plan, gaits, teleop controller and stance IK the keyboard entry builds."""

    config, registry, _ = load_inputs(
        config_path=settings["rig"], assets_path=settings["assets"], motions_path=DEFAULT_MOTIONS
    )
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    if settings.get("collision_fit", {}).get("enabled", False):
        plan, _audit = fit_collision_capsules(
            plan, mesh, margin_m=settings["collision_fit"]["margin_m"]
        )
    dt = config.simulation.physics_dt_s
    planted = settings.get("max_stance_slip_m_s")
    gaits = {
        key: load_gait(spec, plan, dt_s=dt, max_stance_slip_m_s=planted)
        for key, spec in settings["gaits"].items()
    }
    idle = load_gait(settings["idle"], plan, dt_s=dt, max_stance_slip_m_s=planted)
    controller = TeleopController(
        settings["controller"], gaits, idle, plan, np.deg2rad(settings["heading_deg"])
    )
    spec = {key: value for key, value in settings["stance"].items() if key != "enabled"}
    spec.update(stance_overrides)
    return config, plan, controller, StanceFootController(plan, **spec)


def knee_lateral_spans(plan, target: np.ndarray, corrected: np.ndarray, position, quaternion):
    """Body-frame distance between the knees, before and after the IK touched the target."""

    rotation = quaternion_to_matrix(quaternion)
    spans = []
    for joints in (target, corrected):
        poses = forward_kinematics(
            plan, dict(zip(plan.dof_names, joints, strict=True)), root_position=(0, 0, 0)
        )
        knees = [rotation.T @ poses[f"{side}_knee"].translation for side in ("left", "right")]
        spans.append(float(abs(knees[0][1] - knees[1][1])))
    return spans


def replay(config, plan, controller, stance, command: tuple[float, float], seconds: float):
    """Mirror of ``keyboard.py``'s PHYSICS_PRE_STEP target generation, without PhysX."""

    dt = config.simulation.physics_dt_s
    controller.reset(np.asarray(plan.spawn_root_position), 0.0)
    position = controller.position.copy()
    columns = len(plan.dof_names)
    injection = np.zeros((int(round(seconds / dt)), columns))
    spans = np.zeros((len(injection), 2))
    residuals = np.zeros(len(injection))
    foot_steps = 0
    previous_mode: str | None = None
    for step in range(len(injection)):
        target = controller.advance(command, dt, position, 0.0)
        if stance is None:
            corrected = target.joints
        else:
            mode = (
                "transition"
                if target.mode in {"forward", "backward"} and controller.weight < 0.95
                else target.mode
            )
            if mode != previous_mode and target.mode not in {
                "crouching",
                "crouch",
                "standing_up",
            }:
                stance.reset()
            previous_mode = mode
            if mode == "transition" or abs(controller.turn_rate) > 0.01:
                stance.reset()
            gait = controller.gaits[controller.mode]
            corrected = stance.correct(
                target.joints,
                target.position,
                target.quaternion,
                standing=abs(controller.speed) < 0.01,
                support_feet=gait.supporting_feet(controller.phase),
                swing_fraction=gait.swing_fraction(controller.phase),
                measured_position=position,
                measured_quaternion=target.quaternion,
            )
            foot_steps += len(stance.anchors)
            residuals[step] = stance.residual_m
        injection[step] = np.rad2deg(np.abs(corrected - target.joints))
        spans[step] = knee_lateral_spans(
            plan, target.joints, corrected, target.position, target.quaternion
        )
        position = target.position
    return injection, spans, residuals, foot_steps


def summarise(plan, injection, spans, residuals, stance, foot_steps, released):
    worst = injection.max(axis=0)
    order = np.argsort(-worst)
    names = list(plan.dof_names)
    sideways = list(stance.sideways_dofs) if stance is not None else sideways_leg_dofs(plan)
    per_step = injection.max(axis=1)
    budget = np.rad2deg(stance.max_correction_rad) if stance is not None else 0.0
    return {
        "frames": int(injection.shape[0]),
        "injection_max_deg": float(worst.max()),
        "injection_by_dof_deg": {
            names[index]: {
                "max": float(worst[index]),
                "mean": float(injection[:, index].mean()),
                "p95": float(np.percentile(injection[:, index], 95)),
            }
            for index in order[:12]
        },
        "sideways_dofs": sideways,
        "sideways_injection_max_deg": float(
            max((worst[names.index(name)] for name in sideways), default=float("nan"))
        ),
        "saturation_rate": float(np.mean(per_step > budget - 0.05)) if stance else 0.0,
        "over_15_deg_rate": float(np.mean(per_step > 15.0)),
        "knee_span_target_range_m": [float(spans[:, 0].min()), float(spans[:, 0].max())],
        "knee_span_corrected_range_m": [float(spans[:, 1].min()), float(spans[:, 1].max())],
        "anchor_residual_max_m": float(residuals.max()),
        "anchor_foot_steps": int(foot_steps),
        "anchor_releases": int(released),
    }


def parse_variant(text: str) -> tuple[str, str, Any]:
    """One ``NAME=KEY=VALUE`` override; repeat a NAME to build a multi-knob variant.

    A single knob cannot describe the configuration being compared against, because the
    splay needed both the wide budget and the absent prior at once.
    """

    name, _, remainder = text.partition("=")
    key, _, raw = remainder.partition("=")
    if not name or not key or not raw or name == "shipped":
        raise argparse.ArgumentTypeError(f"--variant needs NAME=KEY=VALUE, got {text!r}")
    value: Any = {"true": True, "false": False, "none": None, "null": None}.get(raw.lower(), raw)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return name, key, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "artifacts/humans/stance_ik_audit"
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--command", default="1,0", help="forward,turn held for the window")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="NAME=KEY=VALUE stance override, repeatable; pairs against the shipped config",
    )
    parser.add_argument(
        "--no-stance", action="store_true", help="audit the teleop target with the IK switched off"
    )
    parser.add_argument(
        "--no-plantedness-gate",
        action="store_true",
        help="keep every travel-based support frame, i.e. the behaviour before P0-B item 2",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error("seconds must be finite and positive")
    settings = load_keyboard_config(args.config, REPO_ROOT)
    if args.no_plantedness_gate:
        settings["max_stance_slip_m_s"] = None
    forward, turn = (float(value) for value in args.command.split(","))
    if max(abs(forward), abs(turn)) > 1:
        parser.error("keyboard commands are bounded by [-1, 1]")
    variants: dict[str, dict[str, Any]] = {"shipped": {}}
    for text in args.variant:
        name, key, value = parse_variant(text)
        variants.setdefault(name, {})[key] = value
    report: dict[str, Any] = {
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "command": [forward, turn],
        "seconds": args.seconds,
        "physics_dt_s": None,
        "max_stance_slip_m_s": settings.get("max_stance_slip_m_s"),
        "gait_support_counts": {},
        "variants": {},
    }
    for name, overrides in variants.items():
        config, plan, controller, stance = build(settings, overrides)
        if args.no_stance:
            stance = None
        report["physics_dt_s"] = config.simulation.physics_dt_s
        report["gait_support_counts"] = {
            key: {
                "support_frames": int(gait.support_mask.sum()),
                "support_frames_from_travel": gait.provenance.get("support_frames_from_travel"),
                "support_slip_p50_m_s": gait.provenance.get("travel_support_slip_m_s_p50"),
            }
            for key, gait in controller.gaits.items()
        }
        injection, spans, residuals, foot_steps = replay(
            config, plan, controller, stance, (forward, turn), args.seconds
        )
        released = 0 if stance is None else stance.released_anchors
        report["variants"][name] = {
            "stance_overrides": overrides,
            **summarise(plan, injection, spans, residuals, stance, foot_steps, released),
        }
        row = report["variants"][name]
        LOGGER.info(
            "%s: injection max %.2f deg (sideways %.2f), saturation %.1f%%, "
            "knee span %.3f-%.3f m, releases %d",
            name,
            row["injection_max_deg"],
            row["sideways_injection_max_deg"],
            100 * row["saturation_rate"],
            row["knee_span_corrected_range_m"][0],
            row["knee_span_corrected_range_m"][1],
            row["anchor_releases"],
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "stance_ik_audit.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
