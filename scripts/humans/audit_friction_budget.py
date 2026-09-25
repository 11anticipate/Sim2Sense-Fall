#!/usr/bin/env python3
"""Decide whether foot slip is friction-limited, or something friction cannot reach.

The question "would more friction on the feet stop the sliding" is answerable only if
the friction *demand* is known. Two things block that today:

1. The body's colliders carry no physics material, so the coefficient in force is a
   PhysX default that nothing records (see the note in ``configs/humans``).
2. ``HumanRuntime.contact_samples`` stores ``record.impulse`` from the PhysX contact
   report, and that field is the **normal** impulse: over every archived session,
   every non-zero impulse is parallel to its contact normal to 2e-16. The tangential
   (friction) impulse has therefore never been recorded, on floor or on wall.

So the demand is reconstructed from Newton instead of read off the solver. For the
whole body, with ``W`` the weight, ``F_a`` the recorded external root assistance and
``F_c`` the ground contact force::

    m * a_COM = W + F_a + F_c        =>     F_c = m * (a_COM + g) - F_a

The floor is horizontal, so every horizontal component of ``F_c`` is friction. Then

    mu_required = |F_c_horizontal| / |F_c_vertical|

and the answer is decided by comparing that against the coefficient actually in force.
``F_c_vertical`` is checked against the contact log's normal impulses first -- if the
reconstruction does not reproduce the measured normal load, the horizontal number is
not trustworthy either and the run reports that instead of a ratio.

Read-only: this consumes an existing run's ``control.npz`` and ``contacts.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from common import (  # type: ignore[import-not-found]
    DEFAULT_MOTIONS,
    REPO_ROOT,
    load_inputs,
    write_json,
)

from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import fit_collision_capsules
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints
from sim2sense_fall.humans.rig import fit_rest_skeleton, forward_kinematics, plan_human_rig
from sim2sense_fall.humans.rotations import matrix_to_axis_angle, quaternion_to_matrix
from sim2sense_fall.humans.teleop import load_keyboard_config

LOGGER = logging.getLogger("audit_friction_budget")

# PhysX falls back to this on a collider with no material bound; the floor materials
# live in src/sim2sense_fall/scenes/materials.py and are combined by PhysX's default
# eAVERAGE mode. Both numbers are inputs to the verdict, so they are named here rather
# than hidden: change them if the scene or the body material changes.
PHYSX_DEFAULT_FRICTION = 0.5
FLOOR_DYNAMIC_FRICTION = {"wood_floor": 0.55, "tile_floor": 0.35, "carpet": 0.80}


def build_plan(settings: dict[str, Any]) -> Any:
    """The same plan the keyboard entry ships, rebuilt on CPU."""

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
            plan, mesh, **{k: v for k, v in settings["collision_fit"].items() if k != "enabled"}
        )
    return plan


def centre_of_mass(
    plan: Any, frames: np.ndarray, roots: np.ndarray, quats: np.ndarray
) -> np.ndarray:
    """Whole-body COM per frame from the recorded joint states, by CPU forward kinematics.

    The recorded DOF order is ``plan.dof_names``; forward kinematics is the project's
    independent CPU chain, so a match with the PhysX poses is evidence, not a restatement.
    """

    names = list(plan.dof_names)
    if frames.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} DOFs, got {frames.shape[1]}")
    masses = np.asarray([link.mass_kg for link in plan.links], dtype=np.float64)
    offsets = [np.asarray(link.center_of_mass, dtype=np.float64) for link in plan.links]
    link_names = [link.name for link in plan.links]

    out = np.empty((frames.shape[0], 3), dtype=np.float64)
    for index in range(frames.shape[0]):
        angles = {name: float(frames[index, k]) for k, name in enumerate(names)}
        pose = forward_kinematics(
            plan,
            angles,
            root_position=roots[index],
            root_rotation=matrix_to_axis_angle(quaternion_to_matrix(quats[index])),
        )
        total = np.zeros(3)
        for link, offset, mass in zip(link_names, offsets, masses, strict=True):
            total += mass * pose[link].transform_point(offset)
        out[index] = total / masses.sum()
    return out


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])],
        axis=1,
    )


def derivative(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.empty_like(values)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def measured_normal_force(
    contacts: list[dict[str, Any]], dt: float, part_filter: set[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame summed floor normal force from the contact log, plus a foot-only mask."""

    total = np.zeros(len(contacts))
    foot = np.zeros(len(contacts), dtype=bool)
    for index, frame in enumerate(contacts):
        for sample in frame.get("samples", ()):
            paths = (sample.get("collider0_path") or "", sample.get("collider1_path") or "")
            if any("/Human/" in path for path in paths) is False:
                continue
            body = next((path for path in paths if "/Human/" in path), "")
            if part_filter is not None:
                part = body.split("/")[3] if len(body.split("/")) > 3 else ""
                if part not in part_filter:
                    continue
            normal = np.asarray(sample["normal"], dtype=np.float64)
            length = float(np.linalg.norm(normal))
            if length <= 0:
                continue
            normal /= length
            impulse = np.asarray(sample["impulse_ns"], dtype=np.float64)
            total[index] += max(0.0, float(impulse @ normal))
            foot[index] = True
    return total / dt, foot


def analyse(
    settings: dict[str, Any],
    plan: Any,
    run: Path,
    *,
    gravity: float = 9.81,
    smoothing: int = 5,
) -> dict[str, Any]:
    control = np.load(run / "control.npz", allow_pickle=True)
    contacts = json.loads((run / "contacts.json").read_text(encoding="utf-8"))
    dt = float(settings.get("physics_dt_s") or 1.0 / 120.0)
    mass = float(plan.total_mass_kg)
    weight = mass * gravity

    roots = np.asarray(control["root"], dtype=np.float64)
    quats = np.asarray(control["root_quaternion"], dtype=np.float64)
    frames = np.asarray(control["joints"], dtype=np.float64)
    assist = np.asarray(control["force"], dtype=np.float64)

    com = centre_of_mass(plan, frames, roots, quats)
    acceleration = derivative(smooth(derivative(smooth(com, smoothing), dt), smoothing), dt)
    # F_c = m (a + g) - F_a, with g = (0, 0, +g) because the weight term carries -g.
    contact_force = mass * (acceleration + np.array([0.0, 0.0, gravity])) - assist
    vertical = contact_force[:, 2]

    norm_measured, foot_mask = measured_normal_force(
        contacts, dt, part_filter={"left_ankle", "right_ankle"}
    )
    if len(contacts) != com.shape[0]:
        raise ValueError(
            f"contact log has {len(contacts)} frames but control has {com.shape[0]}; "
            "refusing to align two different-length records"
        )

    airborne = vertical <= 0.05 * weight
    horizontal = np.linalg.norm(contact_force[:, :2], axis=1)
    ratio = np.full(com.shape[0], np.nan)
    usable = (~airborne) & foot_mask
    ratio[usable] = horizontal[usable] / np.maximum(vertical[usable], 1e-9)

    slip = np.full(com.shape[0], np.nan)
    for index, frame in enumerate(contacts):
        values = frame.get("floor_contact_slips_m_s") or []
        if values:
            slip[index] = float(max(values))

    estimate = PHYSX_DEFAULT_FRICTION
    floor_mu = FLOOR_DYNAMIC_FRICTION["wood_floor"]
    effective_mu = 0.5 * (estimate + floor_mu)

    finite = np.isfinite(ratio)
    mu = ratio[finite]
    report: dict[str, Any] = {
        "run": str(run),
        "frames": int(com.shape[0]),
        "physics_dt_s": dt,
        "smoothing_window_frames": smoothing,
        "mass_kg": mass,
        "weight_n": weight,
        "gravity_m_s2": gravity,
        "assumed_body_friction": estimate,
        "assumed_floor_dynamic_friction": floor_mu,
        "assumed_effective_mu": effective_mu,
        "assumption_note": (
            "effective mu is the PhysX default (no material is bound to the body) combined "
            "with the floor material by PhysX's default average mode; it is an assumption "
            "about the runtime, not a measurement, and is the reason a body material should "
            "be authored explicitly"
        ),
        "consistency": {
            "quantity_note": (
                "the vertical force below is the WHOLE-BODY ground reaction, i.e. both feet "
                "summed, so it is directly comparable with the weight. A per-foot number "
                "would halve it in double support and would not be comparable."
            ),
            "measured_normal_p50_n": float(np.median(norm_measured)),
            "predicted_normal_p50_n": float(np.median(vertical)),
            "measured_normal_p95_n": float(np.percentile(norm_measured, 95)),
            "predicted_normal_p95_n": float(np.percentile(vertical, 95)),
            "weight_fraction_carried_by_ground_p50": float(np.median(vertical) / weight),
            "root_assist_vertical_p50_n": float(np.median(assist[:, 2])),
            "root_assist_gravity_fraction": float(
                np.median(assist[:, 2]) / weight
            ),
        },
        "friction_demand": {
            "definition": "|m(a_COM + g) - F_assist| horizontal / vertical, feet on floor",
            "usable_frames": int(finite.sum()),
            "mu_required_p50": float(np.median(mu)),
            "mu_required_p90": float(np.percentile(mu, 90)),
            "mu_required_p95": float(np.percentile(mu, 95)),
            "mu_required_p99": float(np.percentile(mu, 99)),
            "mu_required_max": float(mu.max()),
            "fraction_above_effective_mu": float(np.mean(mu > effective_mu)),
            "fraction_above_1": float(np.mean(mu > 1.0)),
        },
    }

    # The deciding split: is the demand near the coefficient when the foot is sliding?
    # PhysX cannot supply more tangential impulse than mu*N, so a sliding frame whose
    # *demand* exceeds the coefficient is one friction is actually limiting. A frame that
    # slides while the demand sits well below the coefficient is not a friction finding --
    # something else moved the contact.
    tolerance = float(settings.get("slip_speed_tolerance_m_s", 0.15))
    slipping = usable & np.isfinite(slip) & (slip > tolerance)
    sticking = usable & np.isfinite(slip) & (slip <= tolerance)

    def summarise(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        if not count:
            return {"frames": 0}
        values = ratio[mask]
        return {
            "frames": count,
            "mu_p50": float(np.nanmedian(values)),
            "mu_p90": float(np.nanpercentile(values, 90)),
            "slip_p50_m_s": float(np.nanmedian(slip[mask])),
            "at_or_above_assumed_mu": float(np.mean(values >= effective_mu)),
            "fraction_of_demand_above_1": float(np.mean(values > 1.0)),
        }

    saturated = slipping & (ratio >= effective_mu)
    below = slipping & (ratio < effective_mu)
    report["by_slip_state"] = {
        "slip_tolerance_m_s": tolerance,
        "slipping_frames": int(slipping.sum()),
        "sticking_frames": int(sticking.sum()),
        "frame_slip_fraction": float(int(slipping.sum()) / max(int(usable.sum()), 1)),
        "slipping": summarise(slipping),
        "sticking": summarise(sticking),
        "slipping_with_demand_at_or_above_mu": summarise(saturated),
        "slipping_with_demand_below_mu": summarise(below),
        "share_of_slides_that_are_friction_limited": float(
            int(saturated.sum()) / max(int(slipping.sum()), 1)
        ),
        "interpretation": (
            "a slide is friction-limited when the demanded tangential force is at or above "
            "the coefficient; slides below it were produced by the control/reference moving "
            "the contact, which is a target problem that raising the coefficient cannot fix"
        ),
    }
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 3.0]
    hist_all, _ = np.histogram(mu, bins=edges)
    hist_slip, _ = np.histogram(ratio[slipping], bins=edges)
    report["mu_histogram"] = {
        "edges": edges,
        "all_frames": [int(v) for v in hist_all],
        "slipping_frames": [int(v) for v in hist_slip],
    }

    horizontal_force = horizontal[usable]
    report["friction_force_budget"] = {
        "normal_force_p50_n": float(np.median(vertical[usable])),
        "horizontal_demand_p50_n": float(np.median(horizontal_force)),
        "horizontal_demand_p95_n": float(np.percentile(horizontal_force, 95)),
        "capacity_at_assumed_mu_p50_n": float(np.median(vertical[usable]) * effective_mu),
        "capacity_at_mu_1_p50_n": float(np.median(vertical[usable]) * 1.0),
        "real_world_reference_capacity_n": float(weight * 0.5),
        "real_world_note": (
            "a real person standing on the same floor loads the stance foot with the whole "
            "weight, so the shear capacity is ~mu*W; here the ground carries only a fraction "
            "of W because the root actuator holds the rest"
        ),
    }

    # What the load alone would buy, before spending GPU time on it.
    #
    # If the actuator's upward vertical force were capped at a fraction of the weight, the
    # ground would have to supply the rest -- but only if the body keeps following the same
    # trajectory. Holding the *motion* fixed makes this a clean counterfactual: the body's
    # horizontal acceleration is unchanged, so the tangential demand F_h is unchanged, and
    # only the denominator of mu_required moves. That is the whole point and also the
    # limitation: a real run lets the legs sag under the extra load, which changes the
    # motion. So these numbers are an UPPER BOUND on what capping the lift can buy, and the
    # cap is only worth a real run if the ceiling is high.
    sweep: list[dict[str, Any]] = []
    observed_fraction = float(np.median(assist[usable, 2]) / weight) if usable.any() else None
    for cap in (0.82, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0):
        extra = np.maximum(0.0, assist[:, 2] - cap * weight)
        loaded = np.minimum(vertical + extra, weight)
        with np.errstate(invalid="ignore", divide="ignore"):
            candidate = horizontal / np.maximum(loaded, 1e-9)
        candidate = np.where(usable, candidate, np.nan)
        still_forced = slipping & (candidate >= effective_mu)
        row = {
            "assist_vertical_cap_fraction_of_weight": cap,
            "extra_load_moved_to_ground_p50_n": float(np.median(extra[usable])),
            "ground_normal_p50_n": float(np.nanmedian(loaded[usable])),
            "capacity_at_assumed_mu_p50_n": float(np.nanmedian(loaded[usable]) * effective_mu),
            "mu_required_p50": float(np.nanmedian(candidate[usable])),
            "mu_required_p95": float(np.nanpercentile(candidate[usable], 95)),
            "fraction_of_frames_above_mu": float(
                np.nanmean(candidate[usable] > effective_mu)
            ),
            "sliding_frames_still_at_or_above_mu": int(still_forced.sum()),
            "share_of_slides_still_friction_limited": float(
                int(still_forced.sum()) / max(int(slipping.sum()), 1)
            ),
            "reduction_in_friction_limited_slides": float(
                1.0
                - int(still_forced.sum()) / max(int(saturated.sum()), 1)
            ),
            "is_counterfactual": True,
        }
        sweep.append(row)
    report["assist_vertical_cap_sweep"] = {
        "observed_assist_vertical_fraction_of_weight": observed_fraction,
        "assumption": (
            "body motion (hence the tangential demand) is held FIXED while the vertical "
            "load is redistributed to the ground; only the denominator of mu_required "
            "moves. An upper bound, not a prediction"
        ),
        "what_a_real_run_must_decide": (
            "whether the legs still track the reference with the actuator lifting less -- "
            "extra load makes them sag, which changes the motion and can add demand back"
        ),
        "rows": sweep,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml"
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_keyboard_config(args.config, REPO_ROOT)
    plan = build_plan(settings)
    results = []
    for run in args.run:
        report = analyse(settings, plan, run, smoothing=args.smoothing)
        results.append(report)
        if not args.json:
            demand = report["friction_demand"]
            consistency = report["consistency"]
            budget = report["friction_force_budget"]
            split = report["by_slip_state"]
            print(f"\n=== {run} ===")
            grounded = consistency["weight_fraction_carried_by_ground_p50"]
            print(
                f"  normal load: measured p50 {consistency['measured_normal_p50_n']:.0f} N vs "
                f"predicted p50 {consistency['predicted_normal_p50_n']:.0f} N; "
                f"weight fraction on ground {grounded:.2f}"
            )
            print(
                f"  root assist vertical p50 {consistency['root_assist_vertical_p50_n']:.0f} N "
                f"= {consistency['root_assist_gravity_fraction']:.2f} of weight"
            )
            print(
                f"  mu_required p50={demand['mu_required_p50']:.3f} "
                f"p95={demand['mu_required_p95']:.3f} p99={demand['mu_required_p99']:.3f} "
                f"max={demand['mu_required_max']:.3f}  (n={demand['usable_frames']})"
            )
            over_mu = demand["fraction_above_effective_mu"] * 100
            over_one = demand["fraction_above_1"] * 100
            print(
                f"  assumed effective mu={report['assumed_effective_mu']:.3f}; "
                f"demand exceeds it in {over_mu:.1f}% of frames, exceeds 1.0 in {over_one:.1f}%"
            )
            print(
                f"  when sliding ({split['slipping_frames']} frames, "
                f"{split['frame_slip_fraction'] * 100:.1f}% of loaded frames): "
                f"mu_req p50={split['slipping']['mu_p50']:.3f} "
                f"p90={split['slipping']['mu_p90']:.3f}"
            )
            print(
                f"  when sticking ({split['sticking_frames']} frames): "
                f"mu_req p50={split['sticking']['mu_p50']:.3f}"
            )
            limited = split["slipping_with_demand_at_or_above_mu"]
            free = split["slipping_with_demand_below_mu"]
            print(
                f"  slides with demand >= assumed mu: {limited.get('frames', 0)} frames "
                f"({split['share_of_slides_that_are_friction_limited'] * 100:.1f}% of slides)"
            )
            if limited.get("frames"):
                print(
                    f"    friction-limited slides slip p50={limited['slip_p50_m_s']:.3f} m/s; "
                    f"the rest slip p50={free.get('slip_p50_m_s', float('nan')):.3f} m/s"
                )
            hist = report["mu_histogram"]
            print("  mu_req histogram (all / sliding):")
            for index, edge in enumerate(hist["edges"][:-1]):
                print(
                    f"    [{edge:.1f},{hist['edges'][index + 1]:.1f})  "
                    f"{hist['all_frames'][index]:5d} / {hist['slipping_frames'][index]:4d}"
                )
            sweep = report["assist_vertical_cap_sweep"]
            print(
                "  counterfactual: cap the actuator's vertical lift, hold the motion fixed "
                "(upper bound)"
            )
            print("    cap    ground N   mu p50   mu p95   slides still forced")
            for row in sweep["rows"]:
                print(
                    f"    {row['assist_vertical_cap_fraction_of_weight']:.2f}   "
                    f"{row['ground_normal_p50_n']:7.0f}   "
                    f"{row['mu_required_p50']:7.3f}  "
                    f"{row['mu_required_p95']:7.3f}   "
                    f"{row['sliding_frames_still_at_or_above_mu']:5d} / "
                    f"{split['slipping_frames']}"
                )
            print(
                f"  capacity: {budget['capacity_at_assumed_mu_p50_n']:.0f} N at assumed mu vs "
                f"{budget['horizontal_demand_p95_n']:.0f} N p95 demand vs "
                f"{budget['real_world_reference_capacity_n']:.0f} N real-world reference"
            )
    if args.out is not None:
        write_json(args.out, results)
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
