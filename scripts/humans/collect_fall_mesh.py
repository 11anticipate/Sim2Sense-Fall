#!/usr/bin/env python3
"""Collect fall *motion* as a per-frame 3D mesh plus (x, y, z) coordinate sequences.

This is the Sionna RT hand-off. It answers exactly one question -- *where is the
body in world space, frame by frame* -- and it answers it in the coordinate
contract the propagation stage consumes:

- ``world_z_up_xyz``, metres, strictly increasing time in seconds;
- one **fixed** triangle topology for the whole sequence (faces never change);
- a (x, y, z) sequence for the skin surface and for the skeleton joints.

What this script is **not**: a physics trial. The trajectory is a
forward-kinematics replay of the reference clip, so no contact, no gravity and no
control loop participates. That distinction is recorded honestly in every
sample's ``fidelity`` field rather than left for the reader to guess. Real fall
*dynamics* come from ``scripts/humans/simulate.py``, which needs Isaac Sim.

    # the scripted fall references (CPU, no external assets)
    python3 scripts/humans/collect_fall_mesh.py --fall-only

    # real captured falls, once the AMASS subset is downloaded and unpacked
    python3 scripts/humans/collect_fall_mesh.py --fall-only \
        --amass-root /path/to/AMASS --amass-source artifacts/humans/amass_raw

    # verify what was written, without re-deriving any of it
    python3 scripts/humans/collect_fall_mesh.py --fall-only --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    DEFAULT_SCENE_CONFIG,
    Checks,
    resolve_spawn_point,
    write_json,
)
from common import (  # noqa: E402
    REPO_ROOT as _REPO_ROOT,
)

from sim2sense_fall.humans.amass import (  # noqa: E402
    annotate_clip,
    screen_amass_clip,
)
from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.events import (  # noqa: E402
    Trajectory,
    label_trial,
    trunk_axis_from_link_positions,
)
from sim2sense_fall.humans.export import sha256_text  # noqa: E402
from sim2sense_fall.humans.mesh_sequence import (  # noqa: E402
    MeshSequence,
    MeshTopology,
    build_capsule_proxy_template,
    fit_mesh_to_rest_joints,
    pose_capsule_proxy_mesh,
    skin_mesh_sequence_frame,
)
from sim2sense_fall.humans.motion import load_motion_library  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
    pose_surface_points,
)
from sim2sense_fall.humans.usd_human import joint_values_from_clip  # noqa: E402

LOGGER = logging.getLogger("collect_fall_mesh")

DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "fall_mesh"
DEFAULT_SELECTED_ROOT = DEFAULT_OUTPUT_DIR / "amass_raw"

#: The screen this project already uses to call a clip a fall candidate. Named here
#: so the collection manifest can state which screen admitted each sample instead of
#: leaving the reader to assume one.
SCREEN_SOURCE = "sim2sense_fall.humans.amass.screen_amass_clip"

#: Every sample is a forward-kinematics replay. There is deliberately no other value:
#: a physics-produced trajectory is written by ``simulate.py`` with ``fidelity``
#: ``physics_trial``, and the two must never be presented as interchangeable.
FIDELITY_KINEMATIC = "kinematic_replay"
FIDELITY_PHYSICS = "physics_trial"

#: Axis order inside every ``*_xyz`` array, asserted rather than assumed.
AXIS_ORDER = "xyz"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect fall motions as per-frame 3D meshes plus (x, y, z) coordinate "
            "sequences for the Sionna RT hand-off."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--motions", type=Path, default=DEFAULT_MOTIONS)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--motion",
        action="append",
        default=[],
        metavar="MOTION",
        help="collect only this clip id; repeatable. Overrides --fall-only's set.",
    )
    parser.add_argument(
        "--fall-only",
        action="store_true",
        help="keep only clips this project's screen calls fall candidates",
    )
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=None,
        help=(
            "directory of unpacked AMASS .npz files to screen and collect. Nothing "
            "is downloaded; a missing directory is an error, never a silent skip."
        ),
    )
    parser.add_argument(
        "--amass-source",
        type=Path,
        default=None,
        help=(
            "id-unique slice to use as the AMASS root. The library naming scheme "
            "amass__<file stem> collides across CMU subdirectories, so a raw "
            "download root cannot be loaded as one library. Pass the directory "
            "written by --write-slice."
        ),
    )
    parser.add_argument(
        "--write-slice",
        type=Path,
        default=None,
        help=(
            "materialise an id-unique slice of --amass-root into this directory "
            "(relative symlinks, deterministic, never touches the source tree) and "
            "stop. Feed the result back through --amass-source."
        ),
    )
    parser.add_argument(
        "--amass-limit",
        type=int,
        default=None,
        help="load at most this many AMASS sequences, in sorted path order",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-check an existing collection instead of producing one",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-write samples that already exist on disk",
    )
    return parser.parse_args(argv)


def load_inputs(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any], Any, Any, dict[str, Any]]:
    """Resolve config, registry, motion library, rig plan, licensed skin mesh and body.

    Deliberately mirrors ``scripts/humans/simulate.py``: the rig the mesh is skinned
    to has to be the same rig the physics trials use, otherwise the exported
    geometry and the simulated geometry describe two different bodies.
    """

    config = load_human_config(args.config)
    if args.height is not None:
        if args.height <= 0:
            raise ValueError(f"--height must be positive, got {args.height!r}")
        config = replace(config, skeleton=replace(config.skeleton, height_m=float(args.height)))
    if args.seed is not None:
        config = replace(config, export=replace(config.export, surface_point_seed=int(args.seed)))

    from sim2sense_fall.humans.assets import load_asset_registry

    registry = load_asset_registry(args.assets, project_root=REPO_ROOT)
    motions = load_motion_library(args.motions, topology=config.topology)
    motions = {clip_id: annotate_scripted(clip) for clip_id, clip in motions.items()}

    amass_root = args.amass_source if args.amass_source is not None else args.amass_root
    if amass_root is not None:
        imported = load_amass_library_slice(amass_root, limit=args.amass_limit)
        overlap = sorted(set(motions).intersection(imported))
        if overlap:
            raise ValueError(f"AMASS motion ids collide with scripted motions: {overlap}")
        motions.update(imported)

    body = select_body(
        registry,
        model_id=config.skeleton.model_asset,
        allow_procedural=bool(config.skeleton.allow_procedural_skeleton),
    )
    mesh = body.model.mesh() if body.has_skin_mesh else None
    rest = fit_rest_skeleton(config, mesh.rest_skeleton()) if mesh is not None else None
    spawn = resolve_spawn_point(
        args.scene_config, args.spawn_x, args.spawn_y,
        standing_height_m=config.skeleton.height_m,
    )
    plan = plan_human_rig(config, rest=rest, spawn_xy=spawn)
    if mesh is not None:
        mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    return config, registry, motions, plan, mesh, _body_model_payload(body, mesh)


def _body_model_payload(body: Any, mesh: Any) -> dict[str, Any]:
    """Identify the licensed body that produced the skin, and the frame it was read in.

    ``AGENTS.md`` requires every experiment to record its model version, and this is
    that record. The measured source frame belongs here rather than only in a log line:
    it is the single fact that was wrong when the export was yawed, so a consumer has to
    be able to read it back and check it. ``source_frame`` states the convention the file
    was interpreted under, which is what makes the import reproducible from the file
    alone.
    """

    model = getattr(body, "model", None)
    payload: dict[str, Any] = {
        "asset_id": getattr(body, "model_id", None),
        "declared_asset_id": getattr(model, "declared_asset_id", None),
        "path": str(getattr(model, "path", "")),
        "has_skin_mesh": bool(getattr(body, "has_skin_mesh", False)),
        "representation": "smpl_skin_mesh" if mesh is not None else "capsule_proxy_surface",
        "source_frame": getattr(model, "source_frame", "") or None,
        "stature_m": round(float(getattr(model, "stature_m", 0.0)), 6),
        "vertex_count": int(mesh.vertex_count) if mesh is not None else 0,
        "face_count": int(np.asarray(mesh.faces).shape[0]) if mesh is not None else 0,
        "sha256": None,
    }
    path = getattr(model, "path", None)
    if path is not None and Path(path).is_file():
        payload["sha256"] = _sha256_file(Path(path))
    return payload


def annotate_scripted(clip: Any) -> Any:
    """Tag a scripted clip with the same audit fields the AMASS path carries.

    The collection manifest prefers a uniform schema -- every sample states whether
    the kinematic screen ran and what it decided -- over a heterogeneous one where
    the reader has to know which branch produced a record. A scripted reference was
    never subject to ``screen_amass_clip`` (it raises on non-AMASS provenance), so
    the honest value is "not applicable with a reason", not True and not a silent
    absence.
    """

    if dict(clip.metadata).get("collection_screen") is not None:
        return clip
    return replace(
        clip,
        metadata={
            **dict(clip.metadata),
            "collection_screen": {
                "applied": False,
                "accepted": None,
                "screen": SCREEN_SOURCE,
                "reason": (
                    "scripted analytic reference, not a captured sequence: "
                    "screen_amass_clip rejects non-AMASS provenance, so no candidate "
                    "screen was run and none is claimed"
                ),
            },
        },
    )


def load_amass_library_slice(root: Path, *, limit: int | None) -> dict[str, Any]:
    """Load an id-unique AMASS slice, refusing roots that would silently lose clips.

    ``load_amass_library`` raises on duplicate ``amass__<stem>`` ids, which a raw
    AMASS download triggers immediately: CMU ships ``18_01_poses.npz`` under both
    ``CMU/18_19_rory`` and ``CMU/20_21_rory1``. Rather than let that surface as a
    dataclass traceback, this checks first and names the colliding files, because
    the fix (pick one subject's directory, or run ``--write-slice``) depends on
    knowing which subtree they came from.
    """

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"AMASS root does not exist: {directory}. Register at "
            "https://amass.is.tue.mpg.de/download.php, unpack the .npz files locally, "
            "and pass the directory here or as --amass-source."
        )
    paths = sorted(directory.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no AMASS .npz sequences found below {directory}")
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--amass-limit must be positive, got {limit!r}")
        paths = paths[:limit]

    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        by_stem.setdefault(path.stem, []).append(path)
    duplicates = {stem: found for stem, found in by_stem.items() if len(found) > 1}
    if duplicates:
        sample = next(iter(sorted(duplicates)))
        listing = ", ".join(str(p.relative_to(directory)) for p in duplicates[sample])
        raise ValueError(
            f"{len(duplicates)} AMASS file stems are duplicated under {directory}, so "
            f"the amass__<stem> ids they generate are ambiguous. Example stem "
            f"{sample!r}: {listing}. Run --write-slice to build an id-unique slice, or "
            "point --amass-source at one subject directory."
        )

    from sim2sense_fall.humans.amass import load_amass_library

    return load_amass_library(directory, limit=limit, target_up_axis="z")


def write_amass_slice(source: Path, destination: Path) -> int:
    """Materialise an id-unique slice of an AMASS tree as relative symlinks.

    Only stems that are unique *within the slice* can become library ids, so the
    slice is the largest subset that keeps every stem distinct. Selection is
    deterministic (sorted path order, first occurrence wins) and the source tree is
    only read -- the AMASS download is licence-gated and must stay untouched.
    """

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AMASS source tree does not exist: {root}")
    target = Path(destination).expanduser().resolve()
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no AMASS .npz sequences found below {root}")

    chosen: dict[str, Path] = {}
    dropped: list[tuple[Path, Path]] = []
    for path in paths:
        holder = chosen.get(path.stem)
        if holder is None:
            chosen[path.stem] = path
        else:
            dropped.append((path, holder))

    target.mkdir(parents=True, exist_ok=True)
    for stem, path in sorted(chosen.items()):
        link = target / f"{stem}.npz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path)
    LOGGER.info(
        "wrote %d unique-stem links to %s, dropped %d shadowed duplicates",
        len(chosen),
        target,
        len(dropped),
    )
    if dropped:
        LOGGER.info(
            "example dropped duplicate: %s (kept %s)",
            dropped[0][0],
            dropped[0][1],
        )
    return len(chosen)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def select_motions(args: argparse.Namespace, motions: dict[str, Any], plan: Any) -> dict[str, Any]:
    """Resolve which clips to collect, screening the captured ones for fall content.

    ``--fall-only`` means "collect falls", not "run the screen": this project's screen
    is defined for captured AMASS sequences and rejects scripted provenance outright.
    So an explicit ``--motion`` list is taken as given, and the fall tag alone decides
    membership. The screen runs afterwards purely to record evidence, and a captured
    clip it refuses is dropped rather than collected under a fall label it did not earn.
    """

    if args.motion:
        unknown = [clip_id for clip_id in args.motion if clip_id not in motions]
        if unknown:
            raise ValueError(f"unknown motion(s) {unknown}; known: {sorted(motions)}")
        selection = {clip_id: motions[clip_id] for clip_id in args.motion}
    elif args.fall_only:
        selection = {
            clip_id: clip
            for clip_id, clip in motions.items()
            if "fall_reference" in getattr(clip, "tags", ())
        }
        if not selection:
            raise ValueError(
                "no fall clips found in the motion library, so --fall-only would write "
                "an empty collection. Add tagged fall references, or pass --motion."
            )
    else:
        selection = dict(motions)

    screened: dict[str, Any] = {}
    refused: list[tuple[str, str]] = []
    for clip_id, clip in selection.items():
        if clip.provenance.kind != "amass":
            screened[clip_id] = annotate_scripted(clip)
            continue
        result = screen_amass_clip(clip, plan)
        if result.accepted:
            screened[clip_id] = annotate_clip(clip, result)
        else:
            refused.append((clip_id, result.reason))

    if refused:
        LOGGER.warning(
            "the fall screen refused %d captured clip(s), so they are excluded from the "
            "collection. Example: %s -- %s",
            len(refused),
            refused[0][0],
            refused[0][1],
        )
    if not screened:
        # A fall collection with nothing in it is the interesting failure: it means the
        # threshold and the available library disagree, and reporting "0 samples
        # written" without saying why hides that.
        raise ValueError(
            f"--fall-only selected {len(selection)} clip(s) and {len(refused)} were "
            f"refused by the screen, leaving nothing to collect. Refusal reasons: "
            f"{sorted({reason for _, reason in refused})}"
        )
    return screened


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def collect_one(
    clip_id: str,
    clip: Any,
    plan: Any,
    config: Any,
    mesh: Any,
    *,
    physics_dt_s: float,
) -> dict[str, Any]:
    """Replay one clip on the CPU and return the arrays for both export files."""

    reference = clip.resample(1.0 / physics_dt_s, method="slerp")
    frames = int(reference.frame_count)
    dofs = len(plan.dof_names)
    if frames < 2:
        raise ValueError(
            f"{clip_id}: resampling to {1.0 / physics_dt_s:g} Hz left {frames} frame(s); "
            "a mesh sequence needs at least two strictly increasing time samples"
        )

    joint_values = np.zeros((frames, dofs), dtype=np.float64)
    root_positions = np.zeros((frames, 3), dtype=np.float64)
    root_rotations = np.zeros((frames, 3), dtype=np.float64)
    link_positions = np.zeros((frames, len(plan.links), 3), dtype=np.float64)
    points: list[np.ndarray] = []
    mesh_vertices: list[np.ndarray] = []
    template = build_capsule_proxy_template(plan) if mesh is None else None
    base = np.asarray(plan.spawn_root_position, dtype=np.float64)

    for frame in range(frames):
        values, _ = joint_values_from_clip(reference, frame, plan)
        joint_values[frame] = values
        root_rotations[frame] = reference.root_rotation[frame]
        root_positions[frame] = base + reference.root_translation[frame]
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, values, strict=True)),
            root_position=root_positions[frame],
            root_rotation=root_rotations[frame],
        )
        for index, link in enumerate(plan.links):
            link_positions[frame, index] = poses[link.name].translation
        cloud, _ = pose_surface_points(
            plan,
            poses,
            per_segment=config.export.surface_points_per_segment,
            seed=config.export.surface_point_seed,
        )
        points.append(cloud)
        if mesh is None:
            mesh_vertices.append(pose_capsule_proxy_mesh(template, plan, poses))
        else:
            mesh_vertices.append(skin_mesh_sequence_frame(mesh, poses))

    vertices = np.stack(mesh_vertices, axis=0)
    sequence = MeshSequence(
        np.asarray(reference.times_s, dtype=np.float64),
        vertices,
        template.topology if mesh is None else MeshTopology(mesh.faces),
        representation="capsule_proxy_mesh" if mesh is None else "smpl_skin_mesh",
        vertex_owners=template.owners if mesh is None else (),
    )
    body_points = np.stack(points, axis=0)
    trajectory = Trajectory(
        times_s=sequence.time_s,
        root_position=root_positions,
        root_quaternion=_quaternions(root_rotations),
        joint_positions=joint_values,
        joint_names=plan.dof_names,
        body_points=body_points,
        standing_height_m=plan.stats["standing_height_m"],
        standing_pelvis_height_m=float(plan.spawn_root_position[2]),
        trunk_axis=trunk_axis_from_link_positions(
            tuple(link.name for link in plan.links), link_positions
        ),
    )
    return {
        "reference": reference,
        "mesh_sequence": sequence,
        "root_positions": root_positions,
        "root_rotations": root_rotations,
        "joint_values": joint_values,
        "link_positions": link_positions,
        "body_points": body_points,
        "trajectory": trajectory,
        "label": label_trial(trajectory, config.events),
    }


def _quaternions(axis_angle: np.ndarray) -> np.ndarray:
    from sim2sense_fall.humans.rotations import axis_angle_to_quaternion

    return np.asarray([axis_angle_to_quaternion(row) for row in axis_angle], dtype=np.float64)


def write_sample(
    out_dir: Path,
    clip_id: str,
    collected: dict[str, Any],
    *,
    config: Any,
    plan: Any,
    mesh: Any,
    body_model: dict[str, Any],
    config_digest: str,
    scene_digest: str | None,
    scene_id: str,
) -> dict[str, Any]:
    """Write the mesh file, the coordinate-sequence file and return the manifest row.

    Two files on purpose. Consumers that only need geometry (a ray tracer building
    its scene) load the smaller one; consumers that need the skeleton as well (a
    later labelling or tracking stage) load both and can rely on the shared
    ``time_s`` axis. Splitting them costs one extra read and saves shipping 6890
    vertices to code that will never look at them.
    """

    sequence: MeshSequence = collected["mesh_sequence"]
    label = collected["label"]
    clip = collected["reference"]
    provenance = clip.provenance

    mesh_path = out_dir / f"{clip_id}.mesh.npz"
    points_path = out_dir / f"{clip_id}.points.npz"

    np.savez_compressed(
        mesh_path,
        time_s=sequence.time_s,
        mesh_vertices_xyz=sequence.vertices_xyz,
        mesh_faces=sequence.faces,
        mesh_vertex_owners=np.asarray(sequence.vertex_owners, dtype="U64"),
        allow_pickle=False,
    )
    np.savez_compressed(
        points_path,
        time_s=sequence.time_s,
        root_position_xyz=collected["root_positions"],
        link_positions_xyz=collected["link_positions"],
        joint_positions_rad=collected["joint_values"],
        surface_points_xyz=collected["body_points"],
        link_names=np.asarray([link.name for link in plan.links], dtype="U64"),
        joint_names=np.asarray(plan.dof_names, dtype="U64"),
        allow_pickle=False,
    )

    representation = sequence.representation
    screen = dict(clip.metadata)["collection_screen"]
    row = {
        "sample_id": clip_id,
        "motion_id": clip_id,
        "motion_kind": provenance.kind,
        "subject": provenance.subject,
        "sequence": provenance.sequence,
        "split_key": f"subject={provenance.subject}|sequence={provenance.sequence}",
        "fidelity": FIDELITY_KINEMATIC,
        "controller_mode": "kinematic_reference_only",
        "provenance": provenance.as_dict(),
        "label": label.as_dict(),
        "policy": {
            "contact_force_n": (
                "not recorded: a forward-kinematics replay has no solver and therefore "
                "no contact force. An empty array here means 'not simulated', never "
                "'measured zero' -- use scripts/humans/simulate.py for contacts."
            ),
        },
        "screen": screen,
        "scene": {"scene_id": scene_id, "sha256": scene_digest},
        "hashes": {
            "config_sha256": config_digest,
            "rig_plan_sha256": sha256_text(json.dumps(_plan_digest_payload(plan), sort_keys=True)),
            "motion_sha256": sha256_text(
                json.dumps(
                    {
                        "clip_id": clip.clip_id,
                        "fps": float(clip.fps),
                        "frames": int(clip.frame_count),
                        "root_translation": np.asarray(clip.root_translation).round(9).tolist(),
                        "root_rotation": np.asarray(clip.root_rotation).round(9).tolist(),
                        "joint_rotations": np.asarray(clip.joint_rotations).round(9).tolist(),
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
            "mesh_topology_sha256": sequence.topology.sha256,
            # Two different provenances, previously conflated under one key named
            # "mesh_source_sha256" that was in fact fed from the *motion*. For a
            # scripted clip that is always null, which read as "the skin is
            # unidentifiable" while actually meaning "there is no motion file".
            "motion_source_sha256": provenance.source_sha256,
            "body_model_sha256": body_model.get("sha256"),
        },
        "body_model": body_model,
        "seed": int(config.export.surface_point_seed),
        "coordinate_system": sequence.coordinate_system,
        "units": sequence.units,
        "axis_order": AXIS_ORDER,
        "physics_dt_s": float(config.simulation.physics_dt_s),
        "standing_height_m": float(plan.stats["standing_height_m"]),
        "total_mass_kg": float(config.skeleton.mass_kg),
        "mesh": {
            "representation": representation,
            "is_fixed_topology": True,
            "vertex_count": int(sequence.vertex_count),
            "face_count": int(sequence.topology.face_count),
            "frame_count": int(sequence.frame_count),
            "duration_s": round(float(sequence.time_s[-1]), 9),
            "npz": str(mesh_path.relative_to(_REPO_ROOT))
            if mesh_path.is_relative_to(_REPO_ROOT)
            else str(mesh_path),
        },
        "points": {
            "root_position_xyz": "root joint world position, (N, 3), metres",
            "link_positions_xyz": (
                f"world position of the {len(plan.links)} rigid links, "
                f"(N, {len(plan.links)}, 3), metres"
            ),
            "link_names": [link.name for link in plan.links],
            "joint_positions_rad": (
                f"the {len(plan.dof_names)} actuated hinge angles, "
                f"(N, {len(plan.dof_names)}), radians"
            ),
            "joint_names": list(plan.dof_names),
            "surface_points_xyz": (
                f"{collected['body_points'].shape[1]} capsule-surface sample points, "
                f"(N, {collected['body_points'].shape[1]}, 3), metres"
            ),
            "npz": str(points_path.relative_to(_REPO_ROOT))
            if points_path.is_relative_to(_REPO_ROOT)
            else str(points_path),
        },
        "metric_definitions": {
            "body_representation": (
                "capsule_proxy_surface"
                if representation == "capsule_proxy_mesh"
                else "smpl_skin_mesh"
            ),
            "mesh_topology_hash": "sha256 of the little-endian int64 face array bytes",
            "trunk_axis": "unit vector from the pelvis link origin to the neck link origin",
            "standing_height_m": "top of the head above the floor in the rest pose",
            "frame_time_s": "time_s[k], first frame 0.0, uniform 1/physics_dt_s",
        },
    }
    return row


def _plan_digest_payload(plan: Any) -> dict[str, Any]:
    """Stable, JSON-able summary of the rig a sample was skinned to."""

    return {
        "human_id": plan.human_id,
        "topology_name": plan.topology_name,
        "root_link": plan.root_link,
        "root_mode": plan.root_mode,
        "skeleton_source": plan.skeleton_source,
        "spawn_root_position": [round(float(v), 9) for v in plan.spawn_root_position],
        "ground_offset_m": round(float(plan.ground_offset_m), 9),
        "total_mass_kg": round(float(plan.total_mass_kg), 9),
        "dof_names": list(plan.dof_names),
        "links": [
            {
                "name": link.name,
                "parent_link": link.parent_link,
                "rest_position": [round(float(v), 9) for v in link.rest_position],
                "mass_kg": round(float(link.mass_kg), 9),
                "has_collider": link.has_collider,
            }
            for link in plan.links
        ],
        "capsule_count": len(plan.colliders),
    }


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def _nearest_vertex(cloud: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The surface point closest to a landmark, used to tie mesh and skeleton together.

    A module-level helper rather than a closure so the caller's loop variable cannot be
    captured late: this comparison is the one check that detects the mesh and the links
    disagreeing about the body frame, and it must compare the frame it is looking at.
    """

    return cloud[np.linalg.norm(cloud - target, axis=1).argmin()]


def verify_collection(out_dir: Path, checks: Checks) -> None:
    """Re-derive nothing; assert the written files satisfy the Sionna contract.

    Every check below reads the exported arrays back off disk. None of them reuse an
    in-memory object from the writing pass, because a value that is only correct
    while the writer is still alive is not a deliverable.
    """

    manifest_path = out_dir / "manifest.json"
    checks.check("collection manifest exists", manifest_path.is_file(), str(manifest_path))
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    checks.check("manifest lists at least one sample", len(samples) > 0, f"{len(samples)}")

    topology_hashes: set[str] = set()
    face_arrays: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for row in samples:
        sample_id = row["sample_id"]
        mesh_path = _resolve(row["mesh"]["npz"])
        points_path = _resolve(row["points"]["npz"])
        if not mesh_path.is_file() or not points_path.is_file():
            checks.check(f"{sample_id}: both arrays present", False, f"{mesh_path}, {points_path}")
            continue
        with np.load(mesh_path) as data:
            times = data["time_s"]
            vertices = data["mesh_vertices_xyz"]
            faces = data["mesh_faces"]
        with np.load(points_path) as data:
            point_times = data["time_s"]
            root = data["root_position_xyz"]
            links = data["link_positions_xyz"]
            joints = data["joint_positions_rad"]
            surface = data["surface_points_xyz"]

        ok_finite = bool(np.isfinite(vertices).all()) and bool(np.isfinite(surface).all())
        checks.check(f"{sample_id}: every coordinate is finite", ok_finite)
        checks.check(
            f"{sample_id}: time is strictly increasing",
            bool(np.all(np.diff(times) > 0)) and float(times[0]) == 0.0,
            f"{len(times)} frames, {times[0]:.6f}..{times[-1]:.6f} s",
        )
        checks.check(
            f"{sample_id}: mesh is (N, V, 3) with N = len(time)",
            vertices.ndim == 3 and vertices.shape[0] == len(times) and vertices.shape[2] == 3,
            str(vertices.shape),
        )
        checks.check(
            f"{sample_id}: both files share one time axis",
            point_times.shape == times.shape and bool(np.array_equal(point_times, times)),
            f"{len(times)} frames",
        )
        checks.check(
            f"{sample_id}: face indices stay inside the vertex array",
            int(faces.max()) < vertices.shape[1] and int(faces.min()) >= 0,
            f"max index {int(faces.max())} of {vertices.shape[1]} vertices",
        )
        checks.check(
            f"{sample_id}: face count and topology hash match the manifest",
            int(faces.shape[0]) == row["mesh"]["face_count"]
            and MeshTopology(faces).sha256 == row["hashes"]["mesh_topology_sha256"],
            f"{int(faces.shape[0])} faces",
        )
        checks.check(
            f"{sample_id}: (x, y, z) sequences have one row per frame",
            root.shape == (len(times), 3)
            and links.ndim == 3
            and links.shape[0] == len(times)
            and links.shape[2] == 3
            and joints.shape[0] == len(times)
            and surface.shape[0] == len(times)
            and surface.shape[2] == 3,
            f"root {root.shape}, links {links.shape}, joints {joints.shape}, "
            f"surface {surface.shape}",
        )

        # Cross-check the mesh against the skeleton. The skin is skinned *to* the link
        # poses, so a grossly wrong mapping (a swapped axis, a missing translation)
        # shows up as a mesh whose centroid sits outside every link's neighbourhood.
        # This is a coarse envelope test on purpose: it must not depend on SMPL's
        # particular rest shape, which would make it a restatement of the input.
        lo = links.min(axis=1)
        hi = links.max(axis=1)
        centre = vertices.mean(axis=1)
        margin = 0.75
        inside = np.all(centre >= lo - margin, axis=1) & np.all(centre <= hi + margin, axis=1)
        checks.check(
            f"{sample_id}: mesh centroid tracks the skeleton frame by frame",
            bool(inside.all()),
            f"{int(inside.sum())} of {len(times)} frames within {margin} m of the link hull",
        )

        # Positive control, as a *consistency* check rather than a bare "must move".
        #
        # A constant sequence would pass every shape and finiteness check above while
        # containing no motion at all, and a "fall" that never moves is exactly the
        # failure this pipeline exists to avoid. But "the mesh moved" is the wrong
        # assertion on its own: a declared-static reference such as ``stand_neutral``
        # has a stationary skeleton, and demanding motion from it would fail a correct
        # sample. The failure worth catching is the two disagreeing -- a walk that the
        # skeleton performs while the exported skin stays frozen, which is precisely
        # what a mis-scaled or mis-framed skin looks like from the outside.
        #
        # So the check is: whenever the skeleton moves, the mesh must move too, and by
        # a comparable amount. The surface can cover more ground than the joints (it
        # extends past them) but never less.
        travel = float(np.linalg.norm(vertices[-1] - vertices[0], axis=1).max())
        spread = float(np.linalg.norm(vertices.reshape(len(times), -1).std(axis=0)).max())
        skeleton_travel = float(np.linalg.norm(links[-1] - links[0], axis=1).max())
        moving_skeleton = skeleton_travel > 1e-3
        mesh_followed = travel > 1e-3 and travel >= 0.5 * skeleton_travel
        checks.check(
            f"{sample_id}: the mesh follows the skeleton frame by frame",
            mesh_followed if moving_skeleton else travel < 1e-6,
            (
                f"skeleton travelled {skeleton_travel:.6f} m, mesh travelled {travel:.6f} m "
                f"(max per-vertex spread {spread:.6f} m)"
                + ("" if moving_skeleton else "; both static, which is this clip's intent")
            ),
        )

        # The anatomical checks, and the only ones here that a *rotation* cannot satisfy.
        #
        # Every other check in this function is invariant under a rotation: shape,
        # finiteness, monotone time, fixed topology, centroid-in-hull, mesh-follows-
        # skeleton. An earlier defect yawed the skin 90 degrees about the vertical and
        # passed all of them, exporting a body that faced sideways.
        #
        # Both checks below compare the **mesh against the links**, never links against
        # links. That distinction is the whole point: in the defect the skeleton was
        # correct and only the skin was rotated relative to it, so a check that reads
        # only `links` passes on the broken export -- which is exactly how the defect
        # survived. Two anatomical facts are asserted, and together they pin the frame:
        #
        #   * pitch -- the head is above the feet (a body pitched flat is wrong);
        #   * yaw   -- the mesh and the links agree about which horizontal axis the body
        #              is wide along, i.e. which way it faces (a body turned sideways).
        #
        # A yaw about the vertical leaves "head above feet" true, so the second check is
        # the one that matters for the defect that actually shipped.
        link_names = list(row["points"].get("link_names", ()))
        first_frame = vertices[0]
        if not link_names:
            checks.check(
                f"{sample_id}: manifest publishes the link ordering",
                False,
                "link_names missing, so anatomical landmarks cannot be located",
            )
        else:
            lookup = {name: index for index, name in enumerate(link_names)}
            needed = ("head", "left_ankle", "right_ankle", "pelvis")
            missing = [name for name in needed if name not in lookup]
            if missing:
                checks.check(
                    f"{sample_id}: the rig exposes head and ankle links",
                    False,
                    f"missing {missing} from {link_names}",
                )
            else:
                head_vertex = _nearest_vertex(first_frame, links[0, lookup["head"]])
                ankle_vertices = [
                    _nearest_vertex(first_frame, links[0, lookup["left_ankle"]]),
                    _nearest_vertex(first_frame, links[0, lookup["right_ankle"]]),
                ]
                feet_z = float(min(vertex[2] for vertex in ankle_vertices))
                checks.check(
                    f"{sample_id}: the body stands up in the first frame",
                    float(head_vertex[2]) > feet_z,
                    f"head surface vertex z {head_vertex[2]:+.4f} above ankle surface vertex "
                    f"z {feet_z:+.4f}; first-frame mesh z range "
                    f"{first_frame[:, 2].min():+.3f}..{first_frame[:, 2].max():+.3f}",
                )

                # A standing human is wider left-right than front-back: the shoulder line
                # (or the arms, in the template's T-pose) spans more than the chest is
                # deep. So whichever horizontal axis carries the larger extent is the
                # body's lateral axis, and the mesh must agree with the links about which
                # one that is. Measured on the shipped export the two differ by a factor
                # of six (1.825 m against 0.304 m), and a 90 degree yaw swaps them, so
                # the comparison is decisive rather than marginal.
                mesh_extent = first_frame.max(axis=0) - first_frame.min(axis=0)
                link_extent = links[0].max(axis=0) - links[0].min(axis=0)
                mesh_lateral = "y" if mesh_extent[1] > mesh_extent[0] else "x"
                link_lateral = "y" if link_extent[1] > link_extent[0] else "x"
                checks.check(
                    f"{sample_id}: the mesh and the links agree on the body's facing",
                    mesh_lateral == link_lateral == "y",
                    f"mesh lateral extent x {mesh_extent[0]:.4f} m vs y {mesh_extent[1]:.4f} m "
                    f"-> lateral axis {mesh_lateral!r}; links x {link_extent[0]:.4f} m vs "
                    f"y {link_extent[1]:.4f} m -> lateral axis {link_lateral!r}; the pipeline "
                    "frame requires lateral = y",
                )

        topology_hashes.add(row["hashes"]["mesh_topology_sha256"])
        face_arrays.append(faces)
        offsets.append(vertices[0].mean(axis=0))

    if face_arrays:
        # All samples share one topology only if they were built from one body. Two
        # different topologies (skin vs. proxy) must be reported, not averaged away.
        same = all(np.array_equal(face_arrays[0], other) for other in face_arrays[1:])
        if same:
            checks.check(
                "every sample shares one fixed topology",
                len(topology_hashes) == 1,
                f"sha256 {sorted(topology_hashes)[0][:16]}…, {len(face_arrays)} samples",
            )
        else:
            checks.info(
                f"collection mixes {len(topology_hashes)} mesh topologies "
                "(skin mesh and capsule proxy); each sample declares its own hash"
            )

        # Not a correctness property: a shared spawn point is expected and useful to
        # state, because a Sionna scene places one transmitter for the whole set.
        origins = np.stack(offsets)
        spread_xy = float(np.ptp(origins[:, :2], axis=0).max())
        checks.info(f"samples spawn within {spread_xy:.4f} m of each other in the floor plane")


def _resolve(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else _REPO_ROOT / path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    checks = Checks()

    if args.write_slice is not None:
        if args.amass_root is None:
            checks.check(
                "--write-slice needs --amass-root", False, "pass the downloaded AMASS tree"
            )
            return checks.report(banner="collect fall mesh")
        try:
            written = write_amass_slice(args.amass_root, args.write_slice)
        except (OSError, ValueError) as exc:
            checks.check("AMASS slice written", False, f"{type(exc).__name__}: {exc}")
            return checks.report(banner="collect fall mesh")
        checks.check("AMASS slice written", True, f"{written} id-unique sequences")
        checks.info(f"now pass --amass-source {args.write_slice}")
        return checks.report(banner="collect fall mesh")

    if args.verify_only:
        verify_collection(args.out, checks)
        return checks.report(banner="collect fall mesh (verify)")

    try:
        config, _registry, motions, plan, mesh, body_model = load_inputs(args)
    except (OSError, ValueError) as exc:
        checks.check("inputs load and validate", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="collect fall mesh")

    checks.check(
        "a licensed skin mesh is in use",
        mesh is not None,
        "smpl_skin_mesh" if mesh is not None else "capsule_proxy_mesh (SMPL unavailable)",
    )
    try:
        selected = select_motions(args, motions, plan)
    except (OSError, ValueError) as exc:
        checks.check("motions selected", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="collect fall mesh")

    scene_digest, scene_id = scene_fingerprint(args.scene_config)
    config_digest = _sha256_file(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    rows: list[dict[str, Any]] = []
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [row for row in existing.get("samples", []) if row["sample_id"] not in selected]
        if rows:
            checks.info(f"keeping {len(rows)} sample(s) already in the manifest")

    for clip_id, clip in selected.items():
        try:
            collected = collect_one(
                clip_id, clip, plan, config, mesh, physics_dt_s=config.simulation.physics_dt_s
            )
        except (OSError, ValueError) as exc:
            # A single unusable clip must not silently disappear from the manifest.
            checks.check(f"{clip_id}: replayed and exported", False, f"{type(exc).__name__}: {exc}")
            continue
        row = write_sample(
            out_dir,
            clip_id,
            collected,
            config=config,
            plan=plan,
            mesh=mesh,
            body_model=body_model,
            config_digest=config_digest,
            scene_digest=scene_digest,
            scene_id=scene_id,
        )
        rows.append(row)
        checks.check(
            f"{clip_id}: replayed and exported",
            True,
            f"{row['mesh']['frame_count']} frames, {row['mesh']['vertex_count']} vertices, "
            f"{row['mesh']['representation']}, label {row['label']['label']}",
        )

    if not rows:
        checks.check("collection is non-empty", False, "no sample was written")
        return checks.report(banner="collect fall mesh")

    rows.sort(key=lambda row: row["sample_id"])
    payload = {
        "schema_version": 1,
        "description": (
            "Fall motions as per-frame 3D meshes plus (x, y, z) coordinate sequences, "
            "ready for the Sionna RT scene."
        ),
        "fidelity": FIDELITY_KINEMATIC,
        "fidelity_note": (
            "Forward-kinematics replay of a reference clip. Gravity, contact and the "
            "control loop do not participate. Physics-produced trajectories are written "
            "by scripts/humans/simulate.py with fidelity 'physics_trial' and must not be "
            "mixed into this manifest without relabelling."
        ),
        "coordinate_system": "world_z_up_xyz",
        "units": "m",
        "axis_order": AXIS_ORDER,
        "screen": SCREEN_SOURCE,
        "scene": {"scene_id": scene_id, "sha256": scene_digest},
        "config": {
            "path": str(Path(args.config).relative_to(_REPO_ROOT))
            if Path(args.config).is_absolute() and Path(args.config).is_relative_to(_REPO_ROOT)
            else str(args.config),
            "sha256": config_digest,
        },
        "seed": int(config.export.surface_point_seed),
        # The licensed body, identified by file hash, together with the source frame its
        # geometry was measured to use. `AGENTS.md` asks every experiment to record its
        # model version; the frame is part of that, because reading this file under a
        # different convention produces a different body.
        "body_model": body_model,
        "sample_count": len(rows),
        "samples": rows,
    }
    write_json(manifest_path, payload)
    checks.info(f"wrote {manifest_path}")

    verify_collection(out_dir, checks)
    return checks.report(banner="collect fall mesh")


def scene_fingerprint(scene_config: Path) -> tuple[str | None, str]:
    """Scene id plus the hash of its configuration, or ``None`` when unreadable.

    The Sionna comparison is only meaningful inside one room, so a sample that cannot
    name its scene is not usable for training. Returning ``None`` keeps that visible
    in the manifest instead of inventing an identifier.
    """

    path = Path(scene_config)
    if not path.is_file():
        return None, path.stem
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, ValueError):
        return None, path.stem
    scene_id = path.stem
    if isinstance(payload, dict):
        for key in ("id", "scene_id", "name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                scene_id = value.strip()
                break
    return _sha256_file(path), scene_id


if __name__ == "__main__":
    sys.exit(main())
