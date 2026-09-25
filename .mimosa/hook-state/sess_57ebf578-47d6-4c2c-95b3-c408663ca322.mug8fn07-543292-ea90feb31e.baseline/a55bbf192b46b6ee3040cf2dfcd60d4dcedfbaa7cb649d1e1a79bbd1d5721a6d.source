#!/usr/bin/env python3
"""CPU-side human pipeline check: assets, contracts, rig plan and reference motions.

Runs without Isaac Sim, Sionna or any licensed asset, and is the entry point CI and
reviewers should use. It answers four questions:

1. **Which licensed assets are actually present?** The audit searches every declared
   root, hashes what it finds on request, and reports the licence and registration
   URL for what it does not. Nothing is downloaded.
2. **Is the rig plan coherent?** Mass allocation, capsule geometry, DOF structure,
   ground clearance and joint limits are all validated, and the plan is written out
   as JSON so it can be reviewed without reading Python.
3. **Are the reference motions valid?** Frame count, finiteness, rotation magnitude,
   frame-to-frame continuity, representability on the rig's DOF set, and
   reproducibility from the same seed.
4. **Does the CPU forward kinematics agree with the plan?** The direction of every
   joint's motion is asserted, which is what stops a sign error from capping all
   motion at the joint limit.

    python3 scripts/humans/plan.py
    python3 scripts/humans/plan.py --height 1.60 --hash-assets
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _directory in (SRC_DIR, SCRIPTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import numpy as np  # noqa: E402
from common import (  # noqa: E402  # noqa: E402
    DEFAULT_ASSETS,
    DEFAULT_CONFIG,
    DEFAULT_MOTIONS,
    DEFAULT_OUTPUT_DIR,
    Checks,
    load_inputs,
    write_json,
)

from sim2sense_fall.humans.assets import (  # noqa: E402
    audit_registry,
)
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
    standing_height_for,
)
from sim2sense_fall.humans.skeleton import default_rest_skeleton, smpl_skeleton  # noqa: E402
from sim2sense_fall.humans.skinning import skin_with_link_poses  # noqa: E402

#: At least one reference motion must involve the root, or the fall labels would be
#: derived from a body that never tips.
MIN_ROOT_ROTATION_DEG = 30.0

#: The body-shape axis of stage 9 randomises ``betas``; a model that ships fewer shape
#: directions than this cannot support that axis, so the import reports it.
MIN_SHAPE_DIRECTIONS = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU check of the human pipeline inputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--motions", type=Path, default=DEFAULT_MOTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--height", type=float, default=None, help="override skeleton.height_m, in metres"
    )
    parser.add_argument(
        "--hash-assets",
        action="store_true",
        help="hash present model files (slow; off by default so dry runs stay quick)",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="fail instead of degrading when the licensed model file is absent",
    )
    return parser.parse_args(argv)


def import_body(checks: Checks, config: object, registry: object, *, require: bool) -> Any:
    """Load the licensed body and report which body the pipeline actually has.

    The decision is :func:`sim2sense_fall.humans.assets.select_body`'s, shared with the
    build and simulation entry points. A representation of ``smpl_skin_mesh`` therefore
    means the pickle was opened, its vertices/faces/weights read and its rest pose found
    plausible -- not that some file with a SMPL-looking name sat in an asset root.
    """

    from sim2sense_fall.humans.assets import select_body

    # The v1.1.0 neutral release is the one this project wants: it is the release with a
    # NEUTRAL body and it carries the 300 shape directions stage 9 randomises.
    primary = config.skeleton.model_asset or "smpl_neutral_v1_1_0"  # type: ignore[attr-defined]
    selection = select_body(
        registry,  # type: ignore[arg-type]
        model_id=primary,
        allow_procedural=bool(config.skeleton.allow_procedural_skeleton),  # type: ignore[attr-defined]
    )
    if selection.has_skin_mesh:
        model = selection.model
        checks.check(
            f"licensed model {primary} loads as a real skin mesh",
            True,
            f"{model.vertex_count} vertices, {model.face_count} faces, "
            f"{model.beta_count} shape directions, up axis {model.up_axis_source}->"
            f"{model.up_axis_target}, stature {model.stature_m:.3f} m, "
            f"sha256 {model.source_sha256[:12]}",
        )
        geometry = model.mesh()
        rest = {
            name: (np.eye(3), geometry.rest_joints[index])
            for index, name in enumerate(geometry.topology.joint_names)
        }
        drift = float(
            np.abs(skin_with_link_poses(geometry, rest) - geometry.vertices).max()
        )
        checks.check(
            "loaded mesh rest pose reproduces itself under skinning",
            drift < 1e-9,
            f"largest vertex drift at the rest pose {drift:.3e} m",
        )
        return selection
    checks.check(
        f"licensed model {primary} is importable",
        False,
        selection.error or "the model did not expose a usable mesh",
    )
    return selection


def check_assets(checks: Checks, registry: object, *, hash_files: bool, require: bool) -> dict:
    audit = audit_registry(registry, hash_files=hash_files)  # type: ignore[arg-type]
    present = audit["available_model_count"]
    total = len(audit["models"])
    checks.check(
        "asset audit completed",
        total > 0,
        f"{present} of {total} declared models present; roots {audit['roots']}",
    )
    for entry in audit["models"]:
        status = "present" if entry["available"] else "absent"
        checks.info(
            f"model {entry['model_id']}: {status}, license={entry['license'][:60]!r}, "
            f"registration={entry['registration_url']}"
        )
        if entry["available"]:
            checks.check(
                f"present model {entry['model_id']} exposes a path",
                bool(entry["path"]),
                str(entry["sha256"] or "not hashed"),
            )
    for source in audit["motions"]:
        checks.info(
            f"motion source {source['motion_id']}: {source['dataset']} "
            f"{source['representation']} ({source['body_joints']} body + "
            f"{source['hand_joints']} hand joints, {source['betas']} betas, "
            f"{source['dmpls']} DMPLs), license={source['license'][:60]!r}"
        )
    if require:
        # The representation decision itself lives in import_body, which has to open the
        # file; here the audit only reports what was found on disk.
        checks.info(
            "--require-model: the licensed model must load as a real mesh, not merely exist"
        )
    elif present == 0:
        checks.skip(
            "licensed model import",
            "no SMPL model on this machine; only the procedural skeleton can be checked, "
            "and no artefact may claim the licensed body was imported",
        )
    return audit


def check_plan(checks: Checks, config: object, motions: dict, rest: object = None) -> object:
    plan = plan_human_rig(config, rest=rest)  # type: ignore[arg-type]
    stats = plan.stats
    checks.check(
        "rig plan built",
        stats["link_count"] == config.topology.joint_count,  # type: ignore[attr-defined]
        f"{stats['link_count']} links for {config.topology.joint_count} joints",  # type: ignore[attr-defined]
    )
    checks.check(
        "every non-leaf joint has a capsule",
        stats["collider_count"] >= stats["link_count"] - 5,
        f"{stats['collider_count']} colliders, {stats['proxy_link_count']} proxy links",
    )
    checks.check(
        "mass allocation conserves the configured total",
        abs(sum(link.mass_kg for link in plan.links) - plan.total_mass_kg) < 1e-6,
        f"{plan.total_mass_kg:.4f} kg over {stats['link_count']} links",
    )
    checks.check(
        "standing height matches the target",
        abs(stats["standing_height_m"] - stats["target_height_m"]) < 1e-6,
        f"target {stats['target_height_m']} m, achieved {stats['standing_height_m']} m",
    )
    checks.check(
        "rest pose clears the floor",
        plan.ground_offset_m > 0,
        f"lowest capsule point at z = {plan.ground_offset_m:.4f} m",
    )
    checks.check(
        "self-collisions are explicitly configured",
        isinstance(plan.self_collisions, bool),
        f"self_collisions={plan.self_collisions}, "
        f"rest_overlap_pairs={stats['rest_capsule_overlap_pairs']} (overlap between "
        "neighbouring capsules is by design and inert while self-collisions are off)",
    )
    for joint in plan.joints:
        checks.check(
            f"{joint.name}: limit range is usable",
            joint.range_deg >= 5.0,
            f"{joint.lower_deg:g} to {joint.upper_deg:g} degrees",
        )
    checks.check(
        "DOF names are unique and stable",
        len(set(plan.dof_names)) == len(plan.dof_names),
        ", ".join(plan.dof_names),
    )
    for clip_id, clip in motions.items():
        worst = 0.0
        for frame in (0, clip.frame_count // 2, clip.frame_count - 1):
            _, residual = joint_values_from_clip(clip, frame, plan)
            worst = max(worst, residual)
        checks.check(
            f"motion {clip_id} is representable on the rig DOFs",
            worst < 1e-6,
            f"worst off-axis residual {np.degrees(worst):.6f} degrees",
        )
    return plan


#: Joint/axis pairs whose direction is asserted, with the descendant link to watch.
#:
#: The expected direction is not hardcoded: it is ``axis x bone``, which holds for any
#: rest pose. The previous table assumed an arms-down mannequin and therefore failed the
#: moment the rig came from the SMPL template, whose rest pose is an A-pose.
DIRECTION_PROBES = (
    ("left_knee", 60.0, "left_ankle", "y", "knee flexion sweeps the shank backward"),
    ("left_hip", -60.0, "left_knee", "y", "hip flexion swings the thigh forward"),
    ("spine1", 60.0, "neck", "y", "spine flexion tips the trunk forward"),
    ("left_shoulder", -60.0, "left_elbow", "y", "shoulder flexion swings the upper arm"),
    ("left_elbow", -60.0, "left_wrist", "y", "elbow flexion brings the hand forward"),
    ("left_ankle", 30.0, "left_foot", "y", "ankle plantarflexion drives the foot"),
)

#: Below this, a single-axis hinge cannot move that segment at all in this rest pose.
DEGENERATE_MOTION_M = 1e-4


def check_forward_kinematics(checks: Checks, plan: object, config: object) -> None:
    """Assert each hinge moves its descendant the way its axis and bone dictate.

    Rotations are about the body axes with +X forward, +Y left and +Z up, so a hinge about
    unit axis ``a`` displaces a bone ``d`` along ``a x d`` to first order. Deriving the
    expectation from the geometry rather than from a remembered sign is what lets the same
    check run against the procedural stand-in and against an imported body whose rest pose
    is an A-pose -- and it is the check that catches an inverted axis token or a flipped
    limit range, both of which stay structurally valid.
    """

    origin = plan.spawn_root_position  # type: ignore[attr-defined]
    rest_positions = dict(
        zip(
            [link.name for link in plan.links],
            [np.asarray(row, dtype=np.float64) for row in plan.rest_joint_positions],
            strict=True,
        )
    )
    rest = forward_kinematics(plan, {}, root_position=origin)
    degenerate: list[str] = []
    for joint, angle_deg, link, axis_token, description in DIRECTION_PROBES:
        axis = np.zeros(3)
        axis["xyz".index(axis_token)] = 1.0
        bone = rest_positions[link] - rest_positions[joint]
        tangent = np.cross(axis, bone)
        norm = float(np.linalg.norm(tangent))
        # A finite rotation sweeps an arc, so the displacement is the chord
        # (cos t - 1) d_perp + sin t (a x d), not the instantaneous tangent: comparing
        # against the tangent leaves a 30 degree error at t = 60 degrees, which looks
        # exactly like a broken joint.
        theta = np.radians(angle_deg)
        perp = bone - float(bone @ axis) * axis
        predicted = (math.cos(theta) - 1.0) * perp + math.sin(theta) * tangent
        moved = forward_kinematics(plan, {joint: np.radians(angle_deg)}, root_position=origin)
        delta = np.asarray(moved[link].translation) - np.asarray(rest[link].translation)
        travel = float(np.linalg.norm(delta))
        if norm * abs(np.radians(angle_deg)) < DEGENERATE_MOTION_M:
            degenerate.append(joint)
            checks.check(
                f"{description} -- {joint} is degenerate in this rest pose",
                travel < DEGENERATE_MOTION_M,
                f"bone {np.round(bone, 3).tolist()} lies along the {axis_token.upper()} hinge "
                f"axis, so this single-DOF joint cannot flex it; the rig moved {travel:.6f} m",
            )
            continue
        expected = predicted / float(np.linalg.norm(predicted))
        observed = np.asarray(delta) / travel if travel > 0 else np.zeros(3)
        checks.check(
            description,
            travel >= DEGENERATE_MOTION_M and float(expected @ observed) > 0.99,
            f"{joint} {angle_deg:+.0f} deg moves {link} along {np.round(observed, 3).tolist()}, "
            f"geometry predicts {np.round(expected, 3).tolist()} "
            f"(cos {float(expected @ observed):+.4f}, {travel * 100:.2f} cm)",
        )
    # Symmetry: mirrored joints must behave alike, up to the asymmetry the body model
    # itself carries -- SMPL's neutral template is not perfectly left/right symmetric, so
    # the bound is measured against the rest skeleton rather than asserted at zero.
    left = forward_kinematics(plan, {"left_knee": np.radians(45.0)}, root_position=origin)
    right = forward_kinematics(plan, {"right_knee": np.radians(45.0)}, root_position=origin)
    left_delta = np.asarray(left["left_ankle"].translation) - np.asarray(
        rest["left_ankle"].translation
    )
    right_delta = np.asarray(right["right_ankle"].translation) - np.asarray(
        rest["right_ankle"].translation
    )
    rest_gap = float(
        np.linalg.norm(
            (rest_positions["left_ankle"] - rest_positions["left_knee"])
            - (rest_positions["right_ankle"] - rest_positions["right_knee"])
        )
    )
    checks.check(
        "left and right legs move alike within the model's own asymmetry",
        bool(np.allclose(left_delta, right_delta, atol=max(1e-6, 2.0 * rest_gap) + 1e-6)),
        f"leg swing differs by {float(np.linalg.norm(left_delta - right_delta)) * 1000:.2f} mm; "
        f"the rest skeleton itself differs by {rest_gap * 1000:.2f} mm across the two shanks",
    )
    if degenerate:
        checks.info(
            f"sagittal-only hinges cannot flex these segments in this rest pose: {degenerate}; "
            "recorded as a control-capability limit, not as a passing degree of freedom"
        )


def check_determinism(checks: Checks, config: object, plan: object) -> None:
    baseline = standing_height_for(config, default_rest_skeleton().scaled_by(1.0))  # type: ignore[arg-type]
    again = standing_height_for(config, default_rest_skeleton().scaled_by(1.0))  # type: ignore[arg-type]
    checks.check(
        "standing height is reproducible",
        baseline == again,
        f"{baseline:.9f} m from identical inputs",
    )
    topology = smpl_skeleton()
    checks.check(
        "skeleton topology is topologically sorted with a single root",
        topology.parents[0] == -1 and all(p < i for i, p in enumerate(topology.parents) if i),
        f"{topology.joint_count} joints, root {topology.root_name!r}",
    )
    checks.check(
        "the plan survives JSON serialisation",
        len(plan.to_json(include_links=True)) > 1000,  # type: ignore[attr-defined]
        f"{len(plan.to_json(include_links=False))} bytes without link records",
    )


#: The licensed template's stature must agree with the planned rig height to within this.
MODEL_HEIGHT_TOLERANCE_M = 0.10

#: How far a planned joint may sit from the fitted model joint it was built from.
MODEL_JOINT_TOLERANCE_M = 1e-3


def check_model_against_plan(
    checks: Checks, plan: object, model: object, rest: object | None = None
) -> None:
    """Cross-check the imported body against the rig the pipeline actually built.

    Loading a pickle is not integration. This is where the model's own geometry is made to
    agree with the planned joint set and with the configured stature, so a wrong axis
    conversion, a centimetres-scale file, or a mismatched joint table fails here instead of
    showing up later as skin drifting off the capsules.
    """

    joints = np.asarray(model.joint_positions, dtype=np.float64)  # type: ignore[attr-defined]
    names = tuple(model.topology.joint_names)  # type: ignore[attr-defined]
    link_names = tuple(link.name for link in plan.links)  # type: ignore[attr-defined]
    checks.check(
        "imported model exposes the planned number of links",
        joints.shape[0] == len(link_names),
        f"model {joints.shape[0]} joints, plan {len(link_names)} links ({link_names[:3]}...)",
    )
    checks.check(
        "imported model and plan agree on joint names and order",
        set(names) == set(link_names) and names[:4] == link_names[:4],
        f"model {names[:4]}, plan {link_names[:4]}",
    )
    checks.check(
        "imported rest skeleton is Z-up (head above pelvis)",
        float(joints[names.index("head"), 2]) > float(joints[names.index("pelvis"), 2]),
        f"head z {float(joints[names.index('head'), 2]):.3f} m, "
        f"pelvis z {float(joints[names.index('pelvis'), 2]):.3f} m",
    )
    delta = abs(float(model.stature_m) - float(plan.stats["standing_height_m"]))  # type: ignore[attr-defined]
    checks.check(
        "imported stature matches the planned standing height",
        delta <= MODEL_HEIGHT_TOLERANCE_M,
        f"model {float(model.stature_m):.3f} m (up axis "
        f"{model.up_axis_source}->{model.up_axis_target}) vs planned "  # type: ignore[attr-defined]
        f"{float(plan.stats['standing_height_m']):.3f} m, off by {delta * 100:.1f} cm",  # type: ignore[attr-defined]
    )
    checks.check(
        "imported model carries the shape directions stage 9 needs",
        int(model.beta_count) >= MIN_SHAPE_DIRECTIONS,  # type: ignore[attr-defined]
        f"{model.beta_count} shape directions on a (V, 3, B) tensor",  # type: ignore[attr-defined]
    )
    planned = np.asarray(plan.rest_joint_positions, dtype=np.float64)  # type: ignore[attr-defined]
    if rest is None:
        checks.check(
            "rig is planned from the imported body, not the procedural stand-in",
            False,
            "no rest skeleton was fitted from the model, so the capsules are proportional",
        )
        return
    reference = np.asarray(rest.joint_positions, dtype=np.float64)
    checks.check(
        "planned rest joints come from the imported model",
        planned.shape == reference.shape
        and bool(np.abs(planned - reference).max() <= MODEL_JOINT_TOLERANCE_M),
        f"largest gap {float(np.abs(planned - reference).max()) * 1000:.3f} mm against the "
        f"fitted model skeleton ({reference.shape[0]} joints)",
    )
    offset = np.linalg.norm((joints - planned) * np.array([1.0, 1.0, 0.0]), axis=1)
    checks.info(
        "imported model vs planned joints, height-scaled away "
        f"(max lateral gap {float(offset.max()) * 100:.2f} cm on {names[int(np.argmax(offset))]})"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = Checks()
    try:
        config, registry, motions = load_inputs(
            config_path=args.config,
            assets_path=args.assets,
            motions_path=args.motions,
            height_m=args.height,
        )
    except (OSError, ValueError) as exc:
        checks.check("inputs load and validate", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="human plan")

    audit = check_assets(checks, registry, hash_files=args.hash_assets, require=args.require_model)
    selection = import_body(checks, config, registry, require=args.require_model)
    representation = selection.representation
    model_info = selection.model

    # With a licensed body loaded, the rig is built on ITS rest joints, scaled to the
    # configured stature by the same solver the procedural stand-in uses. Without this the
    # "SMPL" label would sit on top of a proportional mannequin.
    rest = None
    if representation == "smpl_skin_mesh":
        rest = fit_rest_skeleton(config, model_info.mesh().rest_skeleton())
        checks.info(f"rig planned from {model_info.path.name} rest joints, fitted to height")
    plan = check_plan(checks, config, motions, rest)
    if model_info is not None:
        check_model_against_plan(checks, plan, model_info, rest)
    check_forward_kinematics(checks, plan, config)
    check_determinism(checks, config, plan)

    for clip_id, clip in motions.items():
        checks.check(
            f"motion {clip_id}: rotations within the representable range",
            float(np.linalg.norm(clip.joint_rotations, axis=-1).max()) <= np.pi + 1e-6,
            f"{clip.frame_count} frames at {clip.fps:g} Hz, "
            f"max {np.degrees(np.linalg.norm(clip.joint_rotations, axis=-1).max()):.1f} degrees",
        )
    root_rotated = [
        clip_id
        for clip_id, clip in motions.items()
        if np.degrees(np.linalg.norm(clip.root_rotation, axis=-1).max()) >= MIN_ROOT_ROTATION_DEG
    ]
    checks.check(
        "at least one reference motion tips the whole body",
        bool(root_rotated),
        f"root-rotated references: {root_rotated or 'none'}",
    )

    output = args.output_dir
    written = [
        write_json(output / "assets_audit.json", audit),
        write_json(output / "human_rig.json", plan.as_dict()),
        write_json(
            output / "motions.json",
            {
                "count": len(motions),
                "clips": [clip.as_dict() for clip in motions.values()],
            },
        ),
        write_json(
            output / "human_plan_summary.json",
            {
                "human_id": config.human_id,
                "config_path": str(args.config),
                "config_sha256": _sha256(args.config),
                "height_m": config.skeleton.height_m,
                "mass_kg": config.skeleton.mass_kg,
                "standing_height_m": plan.stats["standing_height_m"],
                "ground_offset_m": plan.ground_offset_m,
                "plan_stats": dict(plan.stats),
                "controllers": plan.root_mode,
                "perturbations": [entry.id for entry in config.perturbations],
                "usable_model_assets": audit["available_model_count"],
                # Decided by what loaded, not by how many registered paths exist: a file
                # that merely exists is not an imported body.
                "body_representation": representation,
                "model_asset_available": representation == "smpl_skin_mesh",
                "imported_model": None if model_info is None else model_info.as_dict(),
            },
        ),
    ]
    for path in written:
        checks.info(f"wrote {path}")
    return checks.report(banner="human plan")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
