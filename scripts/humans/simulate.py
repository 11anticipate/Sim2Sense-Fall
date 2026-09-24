#!/usr/bin/env python3
"""Run human trials in Isaac Sim and export synchronised ground truth.

Each trial is one reference motion crossed with one scripted perturbation. The
export carries two clocks (the physics step and the wireless sampling rate), the
body state **as simulated**, the reference joint trace, contact force when the
runtime can report it, the event labels derived from the trajectory, and a
provenance block naming every version and hash involved.

    # CPU only: build the reference trajectories and label them without physics
    python3 scripts/humans/simulate.py --dry-run

    # real trials in Isaac Sim
    ~/isaacsim/python.sh scripts/humans/simulate.py --headless \
        --trial stand_neutral:none --trial walk_in_place:push_forward

    # every clip crossed with every perturbation in the config
    ~/isaacsim/python.sh scripts/humans/simulate.py --headless --all

``--dry-run`` is not a substitute for a simulation run: it labels a forward-kinematics
replay of the reference motion and says so in the provenance
(``controller_mode: kinematic_reference_only``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, replace
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

from sim2sense_fall.humans.amass import (  # noqa: E402
    annotate_clip,
    crop_amass_clip,
    ground_amass_clip,
    load_amass_clip,
    normalize_root_motion,
    screen_amass_clip,
)
from sim2sense_fall.humans.assets import (  # noqa: E402
    REPRESENTATION_PROXY,
    REPRESENTATION_SKIN_MESH,
    select_body,
)
from sim2sense_fall.humans.events import (  # noqa: E402
    LABEL_FALL,
    LABEL_INVALID,
    LABEL_RECOVERED,
    Trajectory,
    label_trial,
    phase_labels_from_signal,
    trunk_axis_from_link_positions,
)
from sim2sense_fall.humans.export import (  # noqa: E402
    GroundTruth,
    TrialProvenance,
    clip_group_keys,
    resample_series,
    sha256_text,
)
from sim2sense_fall.humans.mesh_sequence import (  # noqa: E402
    MeshSequence,
    MeshTopology,
    build_capsule_proxy_template,
    fit_mesh_to_rest_joints,
    pose_capsule_proxy_mesh,
    skin_mesh_sequence_frame,
)
from sim2sense_fall.humans.rig import (  # noqa: E402
    apply_neutral_pose,
    authors_deviations,
    fit_rest_skeleton,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
    pose_surface_points,
)
from sim2sense_fall.humans.root_control import RootAssistConfig, root_wrench  # noqa: E402
from sim2sense_fall.humans.rotations import (  # noqa: E402
    matrix_to_axis_angle,
    quaternion_to_matrix,
)
from sim2sense_fall.humans.skinning import sample_skin_points  # noqa: E402
from sim2sense_fall.humans.standing import standing_metrics  # noqa: E402
from sim2sense_fall.humans.usd_human import (  # noqa: E402
    _SUPPORT_HEIGHT_TOLERANCE_M,
    _SUPPORT_NORMAL_MIN_Z,
    ContactSample,
    ContactSourceUnavailable,
    HumanRuntime,
    author_contact_reporting,
    build_human_stage,
    perturbed_trial_notes,
    write_display_skin,
)
from sim2sense_fall.scenes.view import VIEW_MODES, apply_inspection_view  # noqa: E402

LOGGER = logging.getLogger("simulate_human")

#: Physics-rate trajectories are the inner loop; the reference is resampled to the
#: physics rate so one frame index maps to one physics step.
MIN_REFERENCE_FRAMES = 8

#: Perturbations whose entire purpose is that the drives stop following the reference, so
#: reference-tracking acceptance is not a usable gate for them: they get their own
#: protocol (a collapse is expected to lose tracking, and calling that "unusable" would
#: delete the only fall evidence the set has).
NON_TRACKING_PERTURBATIONS = frozenset({"control_failure", "support_loss"})

#: Motion tags naming a passive collapse rather than a performed daily action.
PASSIVE_MOTION_TAGS = frozenset({"fall_reference", "passive"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run human trials and export ground truth.")
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR / "trials")
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--trial",
        action="append",
        default=[],
        metavar="MOTION:PERTURBATION",
        help="repeatable; which clip and which perturbation to combine",
    )
    parser.add_argument("--all", action="store_true", help="every clip x every perturbation")
    parser.add_argument(
        "--root-mode", choices=("free", "anchored"), default=None, help="override rig.root_mode"
    )
    parser.add_argument("--seed", type=int, default=None, help="override the surface point seed")
    parser.add_argument(
        "--pin-root",
        action="store_true",
        help=(
            "hold the pelvis on the reference root trajectory for the whole trial. A "
            "recorded assistance: it is what lets a daily posture (bend, squat, sit, lie "
            "down) be captured without the unsupported body collapsing, and it is written "
            "into the trial provenance."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="run without a window (default)",
    )
    mode.add_argument(
        "--gui", dest="headless", action="store_false", help="show the viewport while trials run"
    )
    parser.add_argument(
        "--view",
        choices=VIEW_MODES,
        default="human",
        help=(
            "viewport framing with --gui: 'human' (default) frames the body, 'top' looks "
            "down over the whole apartment, 'roofless' is an oblique outside view -- all "
            "three hide the roof, because the apartment is closed and the body is "
            "otherwise invisible from outside. 'exterior' keeps the roof on. Headless: ignored."
        ),
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help=(
            "with --gui, keep the window open this long after the last trial so the run "
            "can be watched rather than only read; -1 waits until the window is closed. "
            "Without this the app shuts down the moment the trials end."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="CPU only; no Isaac Sim")
    parser.add_argument("--dump-raw", action="store_true", help="also write the raw npz arrays")
    parser.add_argument(
        "--amass-root", type=Path, default=None, help="local AMASS .npz root; never downloads data"
    )
    parser.add_argument("--amass-limit", type=int, default=None)
    parser.add_argument("--amass-file", type=Path, default=None)
    parser.add_argument("--amass-start", type=float, default=0.0)
    parser.add_argument("--amass-duration", type=float, default=None)
    parser.add_argument("--root-position-tolerance-m", type=float, default=0.15)
    parser.add_argument("--root-angle-tolerance-deg", type=float, default=20.0)
    parser.add_argument("--root-assist", type=Path, default=None,
                        help="finite external pelvis actuator YAML; recorded as assisted physics")
    parser.add_argument(
        "--amass-fall-only",
        action="store_true",
        help="after screening local AMASS clips, expose only fall candidates",
    )
    args = parser.parse_args(argv)
    if args.amass_file is not None and args.amass_root is not None:
        parser.error("choose --amass-file or --amass-root, not both")
    if args.root_assist is not None and args.pin_root:
        parser.error("finite root assistance cannot be combined with --pin-root")
    if not all(np.isfinite(v) and v > 0 for v in (
        args.root_position_tolerance_m, args.root_angle_tolerance_deg
    )):
        parser.error("root tracking tolerances must be finite and positive")
    return args


def selection(args: argparse.Namespace, motions: dict, config: object) -> list[tuple[str, str]]:
    """Resolve the requested trials into ``(clip_id, perturbation_id)`` pairs."""

    if args.all:
        return [
            (clip_id, perturbation.id)
            for clip_id in motions
            for perturbation in config.perturbations  # type: ignore[attr-defined]
        ]
    pairs: list[tuple[str, str]] = []
    for entry in args.trial:
        if ":" not in entry:
            raise ValueError(f"--trial must be MOTION:PERTURBATION, got {entry!r}")
        clip_id, perturbation_id = entry.split(":", 1)
        if clip_id not in motions:
            raise ValueError(f"unknown motion {clip_id!r}; known: {sorted(motions)}")
        config.perturbation(perturbation_id)
        pairs.append((clip_id.strip(), perturbation_id.strip()))
    if not pairs:
        if getattr(args, "amass_file", None) is not None:
            pairs = [(name, "none") for name, clip in motions.items()
                     if clip.provenance.kind == "amass"]
            if not pairs:
                raise ValueError("the selected AMASS file did not pass the requested filter")
            return pairs
        fall_ids = [
            clip_id
            for clip_id, clip in motions.items()
            if getattr(args, "amass_root", None) is not None
            and getattr(clip.provenance, "kind", None) == "amass"
            and "fall_reference" in getattr(clip, "tags", ())
        ]
        pairs = [(clip_id, "none") for clip_id in fall_ids]
        if not pairs:
            pairs = [("stand_neutral", "none"), ("walk_in_place", "none")]
    return pairs


def build_reference(clip: object, plan: object, physics_dt_s: float, neutral_rad: dict) -> object:
    """Offset a deviation-only clip by the rig's neutral pose, then resample it.

    The offset goes on before the resample so the interpolation moves between the poses
    the controller will actually hold, rather than between deviations that only mean
    something once the neutral is added back afterwards.
    """

    posed = (
        apply_neutral_pose(clip, plan, neutral_rad)  # type: ignore[arg-type]
        if authors_deviations(clip)
        else clip
    )
    reference = posed.resample(1.0 / physics_dt_s, method="slerp")  # type: ignore[attr-defined]
    if reference.frame_count < MIN_REFERENCE_FRAMES:
        raise ValueError(
            f"clip {reference.clip_id} resamples to {reference.frame_count} frames at "
            f"{1.0 / physics_dt_s:g} Hz, below the {MIN_REFERENCE_FRAMES} frame minimum"
        )
    return reference


def reference_trajectory(
    plan: object,
    reference: object,
    *,
    per_segment: int,
    seed: int,
    mesh: object = None,
    proxy_template: object = None,
) -> tuple[np.ndarray, ...]:
    """CPU forward-kinematics replay of a reference clip.

    Returns joint angles, root positions, root rotations, link positions, body points,
    mesh vertices, faces and vertex owners. Used by ``--dry-run`` and as the cross-check
    target for the simulator.
    """

    frames = reference.frame_count  # type: ignore[attr-defined]
    dofs = len(plan.dof_names)  # type: ignore[attr-defined]
    joint_values = np.zeros((frames, dofs), dtype=np.float64)
    root_positions = np.zeros((frames, 3), dtype=np.float64)
    root_rotations = np.zeros((frames, 3), dtype=np.float64)
    link_positions = np.zeros((frames, len(plan.links), 3), dtype=np.float64)
    points: list[np.ndarray] = []
    mesh_vertices: list[np.ndarray] = []
    template = proxy_template or build_capsule_proxy_template(plan)  # type: ignore[arg-type]
    base = np.asarray(plan.spawn_root_position)  # type: ignore[attr-defined]
    for frame in range(frames):
        values, _ = joint_values_from_clip(reference, frame, plan)
        joint_values[frame] = values
        root_rotations[frame] = reference.root_rotation[frame]
        root_positions[frame] = base + reference.root_translation[frame]
        poses = forward_kinematics(
            plan,  # type: ignore[arg-type]
            dict(zip(plan.dof_names, values, strict=True)),  # type: ignore[attr-defined]
            root_position=root_positions[frame],
            root_rotation=root_rotations[frame],
        )
        for index, link in enumerate(plan.links):  # type: ignore[attr-defined]
            link_positions[frame, index] = poses[link.name].translation
        cloud, _ = pose_surface_points(plan, poses, per_segment=per_segment, seed=seed)  # type: ignore[arg-type]
        points.append(cloud)
        if mesh is None:
            mesh_vertices.append(pose_capsule_proxy_mesh(template, plan, poses))  # type: ignore[arg-type]
        else:
            mesh_vertices.append(skin_mesh_sequence_frame(mesh, poses))  # type: ignore[arg-type]
    vertices = np.stack(mesh_vertices, axis=0)
    mesh_sequence = MeshSequence(
        reference.times_s,
        vertices,
        template.topology if mesh is None else MeshTopology(mesh.faces),  # type: ignore[union-attr]
        representation="capsule_proxy_mesh" if mesh is None else "smpl_skin_mesh",
        vertex_owners=template.owners if mesh is None else (),  # type: ignore[union-attr]
    )
    return (
        joint_values,
        root_positions,
        root_rotations,
        link_positions,
        np.stack(points, axis=0),
        mesh_sequence.vertices_xyz,
        mesh_sequence.faces,
        mesh_sequence.vertex_owners,
    )


def dry_run(
    args: argparse.Namespace,
    config: object,
    motions: dict,
    plan: object,
    *,
    mesh: object = None,
) -> int:
    """Label CPU replays of the reference motions; no physics, no claims about it."""

    checks = Checks()
    args.out.mkdir(parents=True, exist_ok=True)
    checks.skip(
        "Isaac Sim physics",
        "--dry-run: trajectories come from forward kinematics of the reference motion, "
        "so no fall or tracking result is produced",
    )
    physics_dt = config.simulation.physics_dt_s  # type: ignore[attr-defined]
    pairs = selection(args, motions, config)
    index: list[dict] = []
    for clip_id, perturbation_id in pairs:
        clip = motions[clip_id]
        reference = build_reference(clip, plan, physics_dt, config.visualization.pose_dict())
        (
            values,
            root_positions,
            root_rotations,
            link_positions,
            points,
            mesh_vertices,
            mesh_faces,
            mesh_owners,
        ) = reference_trajectory(
            plan,
            reference,
            per_segment=config.export.surface_points_per_segment,  # type: ignore[attr-defined]
            seed=config.export.surface_point_seed,  # type: ignore[attr-defined]
            mesh=mesh,
        )
        times = reference.times_s
        trajectory = Trajectory(
            times_s=times,
            root_position=root_positions,
            root_quaternion=_quaternions(root_rotations),
            joint_positions=values,
            joint_names=plan.dof_names,  # type: ignore[attr-defined]
            body_points=points,
            standing_height_m=plan.stats["standing_height_m"],  # type: ignore[attr-defined]
            standing_pelvis_height_m=float(plan.spawn_root_position[2]),  # type: ignore[attr-defined]
            trunk_axis=trunk_axis_from_link_positions(
                tuple(link.name for link in plan.links),  # type: ignore[attr-defined]
                link_positions,
            ),
        )
        label = label_trial(trajectory, config.events)  # type: ignore[attr-defined]
        mesh_path = args.out / f"{clip_id}.mesh.npz"
        np.savez_compressed(
            mesh_path,
            time_s=times,
            mesh_vertices_xyz=mesh_vertices,
            mesh_faces=mesh_faces,
            mesh_vertex_owners=np.asarray(mesh_owners, dtype="U64"),
            allow_pickle=False,
        )
        index.append(
            {
                "motion_id": clip_id,
                "perturbation_id": perturbation_id,
                "frames": int(reference.frame_count),
                "duration_s": round(float(times[-1]), 6),
                "motion_source": clip.provenance.as_dict(),
                "motion_metadata": dict(clip.metadata),
                "seed": config.export.surface_point_seed,
                "rig_plan_sha256": sha256_text(plan.to_json()),
                "label": label.as_dict(),
                "mesh": {
                    "representation": (
                        "smpl_skin_mesh" if mesh is not None else "capsule_proxy_mesh"
                    ),
                    "vertex_count": int(mesh_vertices.shape[1]),
                    "face_count": int(mesh_faces.shape[0]),
                    "topology_hash": MeshTopology(mesh_faces).sha256,
                    "coordinate_system": "world_z_up_xyz",
                    "units": "m",
                    "npz": str(mesh_path),
                },
                "trajectory": trajectory.as_dict(),
                "note": "forward-kinematics reference replay, not a physics trial",
            }
        )
        checks.info(
            f"{clip_id} x {perturbation_id}: {label.label} "
            f"(onset={label.imbalance_onset_s}, impact={label.first_impact_s})"
        )
    path = write_json(args.out / "reference_labels.json", {"trials": index})
    checks.info(f"wrote {path}")
    checks.check(
        "every reference replay labelled",
        len(index) == len(pairs),
        f"{len(index)} of {len(pairs)}",
    )
    return checks.report(banner="human simulate (dry run)")


def _quaternions(axis_angle: np.ndarray) -> np.ndarray:
    from sim2sense_fall.humans.rotations import axis_angle_to_quaternion

    return np.stack([axis_angle_to_quaternion(row) for row in axis_angle], axis=0)


def _contact_rows(
    samples: object,
    *,
    segments: object = None,
    support_z: float | None = None,
) -> dict[str, np.ndarray]:
    """Flatten per-frame contact samples into parallel columns.

    ``samples`` is a per-frame sequence of :class:`ContactSample` tuples, or ``None``
    when no channel was available. ``segments`` is the matching per-frame sequence of
    limb names, one per sample, or ``None`` when attribution was unavailable. Every
    column keeps its declared dtype even when empty, so a reader can index it without
    a special case.

    Both decoded collider paths are retained alongside the encoded identifiers.
    The body segment comes from collider identity; support remains a geometric flag.
    """

    frames: list[int] = []
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    impulses: list[tuple[float, float, float]] = []
    separations: list[float] = []
    handle0: list[int] = []
    handle1: list[int] = []
    path0: list[str] = []
    path1: list[str] = []
    limbs: list[str] = []
    is_support: list[bool] = []
    for frame, per_frame in enumerate(samples or ()):  # type: ignore[union-attr]
        per_frame_segments = (
            list(segments[frame]) if segments is not None and frame < len(segments) else []
        )
        for index, sample in enumerate(per_frame):
            frames.append(frame)
            positions.append(tuple(sample.position_m))
            normals.append(tuple(sample.normal))
            impulses.append(tuple(sample.impulse_ns))
            separations.append(float(sample.separation_m))
            handle0.append(int(sample.collider0))
            handle1.append(int(sample.collider1))
            path0.append(sample.collider0_path)
            path1.append(sample.collider1_path)
            limbs.append(per_frame_segments[index] if index < len(per_frame_segments) else "")
            is_support.append(
                support_z is not None
                and sample.normal[2] > _SUPPORT_NORMAL_MIN_Z
                and sample.position_m[2] <= support_z + _SUPPORT_HEIGHT_TOLERANCE_M
            )
    return {
        "frame": np.asarray(frames, dtype=np.int64),
        "position": np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        "normal": np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        "impulse": np.asarray(impulses, dtype=np.float64).reshape(-1, 3),
        "separation": np.asarray(separations, dtype=np.float64),
        "handle0": np.asarray(handle0, dtype=np.int64),
        "handle1": np.asarray(handle1, dtype=np.int64),
        "path0": np.asarray(path0, dtype=str),
        "path1": np.asarray(path1, dtype=str),
        # Body segment the contact point fell inside, or "" when unattributed.
        "segment": np.asarray(limbs, dtype="U64"),
        "is_support": np.asarray(is_support, dtype=bool),
    }


def hold_viewport(app: object, *, seconds: float) -> None:
    """Keep the viewport open after the report so a fall can be watched at human speed.

    ``-1`` waits until the window is closed. The trials themselves run at the physics
    rate, which is faster than the eye and over before the report is printed, so the run
    that produced the numbers was never the run anyone could see.
    """

    if seconds < 0:
        while app.is_running():  # type: ignore[attr-defined]
            app.update()  # type: ignore[attr-defined]
        return
    deadline = time.monotonic() + seconds
    while app.is_running() and time.monotonic() < deadline:  # type: ignore[attr-defined]
        app.update()  # type: ignore[attr-defined]


#: Frames per second a finished trial is replayed to the viewport at. Physics runs at
#: 120 Hz; nobody can read a fall at that speed and the renderer cannot keep up either.
PLAYBACK_FPS = 20.0


def play_back_trial(app: object, stage: object, record: dict, *, fps: float) -> int:
    """Replay a completed trial's recorded skin into the viewport.

    Deliberately after the physics has stopped. Writing 6890 mesh points *between*
    physics steps was tried first: in a GUI session it occasionally invalidated PhysX's
    tensor views one trial later, so a run failed over a picture, and the failure had
    nothing to do with the motion being recorded. Playing back from the arrays also means
    the viewport cannot disagree with the evidence -- it is showing the same numbers.
    """

    times = np.asarray(record["time_s"], dtype=np.float64)
    vertices = record.get("mesh_vertices_xyz")
    faces = record.get("mesh_faces")
    if vertices is None or faces is None or len(vertices) < 2 or times[-1] <= times[0]:
        return 0
    stride = max(1, int(round((len(times) - 1) / max(1.0, (times[-1] - times[0]) * fps))))
    played = 0
    # Pace the replay by the recorded clock. Writing every frame as fast as the renderer
    # takes one plays a 5 s fall in a quarter second, which is a data dump with a
    # viewport attached rather than something a person can watch and judge.
    started = time.monotonic()
    for index in range(0, len(times), stride):
        write_display_skin(stage, vertices[index], faces)
        app.update()  # type: ignore[attr-defined]
        played += 1
        due = started + float(times[index] - times[0])
        remaining = due - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    return played


def run_trials(
    args: argparse.Namespace,
    config: object,
    motions: dict,
    plan: object,
    config_digest: str,
    *,
    body: object = None,
    mesh: object = None,
) -> int:
    checks = Checks()
    scene = args.scene.resolve()
    if not scene.is_file():
        checks.check("base scene exists", False, str(scene))
        return checks.report(banner="human simulate")
    scene_digest = _sha256(scene)

    root_assist = RootAssistConfig.load(args.root_assist) if args.root_assist else None
    app = boot_isaac(args.headless)
    if app is None:
        checks.check("Isaac Sim started", False, "run through ~/isaacsim/python.sh")
        return checks.report(banner="human simulate")
    exit_code = 1
    # The last trial that produced arrays; replayed to the viewport once physics stops.
    playback: dict | None = None
    try:
        if plan.root_mode == "anchored":  # type: ignore[attr-defined]
            raise NotImplementedError(
                "rig.root_mode 'anchored' authors a world fixed joint, but the runtime path "
                "is not verified: PhysX returned a non-finite root quaternion for it. The "
                "anchor is authored and the plan records it, but trials must not be run "
                "with it. Use --pin-root for the recorded-assistance route instead. See "
                "docs/human-simulation.md."
            )
        stage_path = args.out / "human_trial.usda"
        build_human_stage(
            plan,
            stage_path,
            base_scene=scene,
            skin_points=None if mesh is None else mesh.vertices,
            skin_faces=None if mesh is None else mesh.faces,
        )  # type: ignore[arg-type]
        stage = open_scene(app, stage_path)
        activation = activate_physics()
        checks.check(
            "physics activated",
            activation["active_engine"] == "physx",
            str(activation["physics_scenes"]),
        )
        measured_dt, accepted = set_physics_dt(config.simulation.physics_dt_s)  # type: ignore[attr-defined]
        checks.check(
            "configured physics step is in force",
            abs(measured_dt - float(config.simulation.physics_dt_s)) < 1e-9,  # type: ignore[attr-defined]
            f"requested {config.simulation.physics_dt_s} s, engine reports {measured_dt} s, "  # type: ignore[attr-defined]
            f"accepted={accepted}",
        )
        # Contact views are NOT requested. This build cannot construct the tensors
        # rigid-contact view for these patterns, and asking for it is not a quiet
        # failure: omni.physics.tensors logs an error per filter path, and the same
        # check re-raises from RigidPrim._on_physics_ready, which SimulationManager
        # dispatches as an event callback -- so a screenful of plugin errors and a
        # traceback land in stderr on every run, including at shutdown, where no
        # try/except in this script can reach them. The contact history comes from
        # the polled physx report, which needs no tensor view. See
        # docs/physics-interaction-audit.md (RC-1).
        runtime = HumanRuntime(stage, plan, enable_contact_views=False, app=app)  # type: ignore[arg-type]
        proxy_template = build_capsule_proxy_template(plan) if mesh is None else None
        # Tag the environment side of every reportable pair. PhysX needs the contact
        # report API on BOTH colliders, and the human side is tagged during authoring;
        # without this the report is empty even though the body is resting on the floor.
        tagged = author_contact_reporting(stage)
        runtime.play()
        for _ in range(4):
            app.update()
        if not args.headless:
            # The apartment is authored closed, so without a de-roofed inspection camera
            # the viewport shows a walled box with the body hidden inside it -- and the
            # trial still reports success, which is the part that makes this easy to miss.
            apply_inspection_view(
                stage, mode=args.view, aspect_ratio=1600 / 900, require_viewport=True
            )
            checks.info(f"inspection view: {args.view} (temporary session layer)")
        checks.check(
            "runtime DOF set matches the plan",
            sorted(runtime.capabilities["dof_order"]) == sorted(plan.dof_names),
            ", ".join(runtime.capabilities["dof_order"])
            + (
                " (PhysX traversal order; translated to plan order at the runtime boundary)"
                if runtime.capabilities.get("dof_plan_permutation")
                else ""
            ),
        )
        try:
            probed = runtime.contact_samples()
        except ContactSourceUnavailable as exc:
            checks.check("contact channel reports contact pairs", False, str(exc))
            probed = None
        else:
            checks.check(
                "contact channel reports contact pairs",
                True,
                f"{len(probed)} pairs on the probe step, "
                f"{len(tagged)} tagged environment colliders, "
                f"source {runtime.capabilities['contact_source']}",
            )
        if probed is not None:
            # A readable channel is not the same as a usable one. The report is a bag
            # of points with no recoverable side identity, so the only way to tell the
            # body from the environment is whether the point falls inside a body
            # capsule. If that test starts failing the accessor raises instead of
            # returning nothing, and that distinction has to surface here rather than
            # silently downgrade every later frame to "no contact".
            try:
                segments = runtime.contact_segment_names()
            except ContactSourceUnavailable as exc:
                checks.check("contact points can be attributed to limbs", False, str(exc))
            else:
                checks.check(
                    "contact points can be attributed to limbs",
                    True,
                    f"{len(segments)} points on the probe step, "
                    f"limbs {sorted(set(segments)) or 'none'}",
                )
        support_z = runtime.support_surface_height_m()
        checks.check(
            "support surface height is readable and below the body",
            support_z is not None,
            f"{support_z} m",
        )
        if runtime.capabilities.get("contact_forces"):
            checks.check(
                "tensor contact force view is usable",
                True,
                f"{len(runtime.capabilities['contact_filter_paths'])} filter paths",
            )
        else:
            # Three distinct facts, three distinct readings: the caller declined the
            # channel, the channel was asked for and failed, or there was nothing to
            # ask against. Collapsing them into one message was how the dead channel
            # stayed invisible.
            checks.skip(
                "tensor contact force view",
                str(
                    runtime.capabilities.get("contact_tensor_view_reason")
                    or runtime.capabilities.get("contact_tensor_view_error")
                    or runtime.capabilities.get("contact_error")
                ),
            )

        # Two different quantities, two different names. This line used to compute the
        # RATE and then store it under ``physics_dt_s`` in the batch index, so downstream
        # consumers read "120 seconds per step" where the step is 1/120 s.
        physics_dt_s = float(config.simulation.physics_dt_s)  # type: ignore[attr-defined]
        physics_hz = 1.0 / physics_dt_s
        tracking_tolerance = float(config.control.tracking_tolerance_deg)  # type: ignore[attr-defined]
        index: list[dict] = []
        for clip_id, perturbation_id in selection(args, motions, config):
            clip = motions[clip_id]
            perturbation = config.perturbation(perturbation_id)  # type: ignore[attr-defined]
            reference = build_reference(
                clip, plan, physics_dt_s, config.visualization.pose_dict()  # type: ignore[attr-defined]
            )
            record = execute_trial(
                runtime,
                app,
                plan,
                reference,
                perturbation,
                config,
                pin_root=args.pin_root,
                mesh=mesh,
                proxy_template=proxy_template,
                root_assist=root_assist,
            )
            playback = record
            if record["pairs"] == 0:
                checks.skip(f"{clip_id} x {perturbation_id}", "trial did not run")
                continue
            ground_truth = assemble_ground_truth(
                plan=plan,
                config=config,
                clip=clip,
                reference=reference,
                record=record,
                perturbation=perturbation,
                config_digest=config_digest,
                scene_digest=scene_digest,
                body=body,
            )
            standing_applies = (
                clip_id == "stand_neutral" and perturbation.kind == "none"
                and plan.root_mode == "free" and not args.pin_root
            )
            standing = None
            if standing_applies:
                link_names = [link.name for link in plan.links]
                standing = standing_metrics(
                    record["root_position"],
                    record["link_positions"][:, link_names.index("neck")]
                    - record["link_positions"][:, link_names.index("pelvis")],
                    reference_root_z=float(plan.spawn_root_position[2]),
                    max_drop_m=config.control.standing_max_drop_m,
                    max_tilt_deg=config.control.standing_max_tilt_deg,
                    max_drift_m=config.control.standing_max_drift_m,
                )
                ground_truth.metadata["standing_acceptance"] = standing
                checks.check("free stationary standing", bool(standing["passed"]), str(standing))
            written = ground_truth.write(args.out)
            # Contact points are variable-length per step, so they get their own file
            # rather than being squeezed into the fixed-shape raw dump: one row per
            # contact, with the frame index as the first column. The empty case still
            # writes the file, so a reader never confuses "absent" with "no contacts".
            contact_rows = _contact_rows(
                record.get("contact_samples"),
                segments=record.get("contact_segments"),
                support_z=record["support_surface_height_m"],
            )
            np.savez_compressed(
                args.out / f"{written['npz'].stem}.contacts.npz",
                contact_frame=contact_rows["frame"],
                contact_position_m=contact_rows["position"],
                contact_normal=contact_rows["normal"],
                contact_impulse_ns=contact_rows["impulse"],
                contact_separation_m=contact_rows["separation"],
                # Both the encoded identifiers and the decoded collider paths are retained.
                contact_handle0=contact_rows["handle0"],
                contact_handle1=contact_rows["handle1"],
                contact_collider0_path=contact_rows["path0"],
                contact_collider1_path=contact_rows["path1"],
                # The body segment whose capsule contained the contact point. This is
                # the limb-level attribution a fall label can be argued from.
                contact_segment=contact_rows["segment"],
                contact_is_support=contact_rows["is_support"],
                frames_with_contact=np.array(
                    sorted({int(f) for f in contact_rows["frame"]}), dtype=np.int64
                ),
                contact_source=np.array(record["contact_source"]),
                contact_attribution=np.array(record["contact_attribution"]),
                support_surface_height_m=np.array(
                    float("nan")
                    if record["support_surface_height_m"] is None
                    else float(record["support_surface_height_m"])
                ),
                allow_pickle=False,
            )
            if args.dump_raw:
                np.savez_compressed(
                    args.out / f"{written['npz'].stem}.raw.npz",
                    **{
                        key: value for key, value in record.items() if isinstance(value, np.ndarray)
                    },
                    allow_pickle=False,
                )
            np.savez_compressed(
                args.out / f"{written['npz'].stem}.control.npz",
                time_s=record["time_s"],
                reference_root_position=record["reference_root_position"],
                reference_root_quaternion_wxyz=record["reference_root_quaternion"],
                root_assist_force_n=record["root_assist_force_n"],
                root_assist_torque_nm=record["root_assist_torque_nm"],
                assisted=np.asarray(root_assist is not None),
            )
            # A controlled daily action is only reproduced if the drives followed the
            # reference; a passive collapse or a deliberate control failure is NOT meant
            # to, so it is judged by a different protocol. Conflating the two let trials
            # with 71.9, 69.1 and 35.4 degree tracking errors count as usable.
            tracking_applies = (
                perturbation.kind not in NON_TRACKING_PERTURBATIONS
                and (clip.provenance.kind == "amass" or not set(clip.tags) & PASSIVE_MOTION_TAGS)
            )
            physically_valid = ground_truth.label.label != LABEL_INVALID
            tracking_gate_error = float(record["tracking_error_deg"])
            tracking_window = "whole_trial"
            if perturbation.kind == "force":
                before_force = record["time_s"] < perturbation.start_s
                if not before_force.any():
                    raise ValueError("force trial requires an unperturbed initial tracking window")
                tracking_gate_error = float(np.degrees(np.abs(
                    record["joint_positions_rad"][before_force]
                    - record["reference_joint_positions_rad"][before_force]
                ).max()))
                tracking_window = "before_external_force"
            tracking_within_tolerance = tracking_gate_error <= tracking_tolerance
            root_tracking = None
            if clip.provenance.kind == "amass":
                desired_roots = np.asarray(plan.spawn_root_position) + reference.root_translation
                root_errors = np.linalg.norm(record["root_position"] - desired_roots, axis=1)
                desired_quaternions = _quaternions(reference.root_rotation)
                dots = np.abs(np.sum(record["root_quaternion"] * desired_quaternions, axis=1))
                angles = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
                root_tracking = {
                    "position_max_m": float(root_errors.max()),
                    "position_rmse_m": float(np.sqrt(np.mean(root_errors ** 2))),
                    "angle_max_deg": float(angles.max()),
                    "position_tolerance_m": args.root_position_tolerance_m,
                    "angle_tolerance_deg": args.root_angle_tolerance_deg,
                    "passed": bool(root_errors.max() <= args.root_position_tolerance_m
                                   and angles.max() <= args.root_angle_tolerance_deg),
                    "root_teleported": bool(record["root_pinned_during_trial"]),
                }
            entry = {
                "motion_id": clip_id,
                "perturbation_id": perturbation_id,
                "npz": str(written["npz"]),
                "json": str(written["json"]),
                "label": ground_truth.label.as_dict(),
                "tracking_error_deg": round(record["tracking_error_deg"], 6),
                "max_penetration_m": round(record["max_penetration_m"], 6),
                "contact_forces_recorded": ground_truth.contact_force_n is not None,
                "contact_source": record["contact_source"],
                "contact_frames": int(
                    sum(1 for samples in (record["contact_samples"] or ()) if samples)
                ),
                "step_s": round(float(record["step_s"]), 12),
                "measured_step_s": round(float(record["measured_step_s"]), 12),
                "gates": {
                    "physically_valid": physically_valid,
                    "standing_acceptance": standing,
                    "tracking_gate_applies": tracking_applies,
                    "tracking_window": tracking_window,
                    "tracking_gate_error_deg": tracking_gate_error,
                    "tracking_within_tolerance": tracking_within_tolerance,
                    "root_tracking": root_tracking,
                    "unassisted_action_reproduced": bool(
                        physically_valid and tracking_within_tolerance
                        and root_tracking is not None and root_tracking["passed"]
                        and not record["root_pinned_during_trial"]
                        and record["root_assistance"] is None
                    ),
                    "label_credible": bool(
                        physically_valid
                        and (
                            ground_truth.label.label != LABEL_FALL
                            or ground_truth.label.first_impact_s is not None
                        )
                    ),
                    "usable": bool(
                        physically_valid and (not tracking_applies or tracking_within_tolerance)
                        and (standing is None or standing["passed"])
                        and (root_tracking is None or root_tracking["passed"])
                    ),
                },
            }
            index.append(entry)
            checks.info(
                f"{clip_id} x {perturbation_id}: {ground_truth.label.label}, "
                f"tracking error {entry['tracking_error_deg']:.3f} deg, "
                f"deepest body point {entry['max_penetration_m']:.4f} m"
            )
            if not tracking_applies and physically_valid:
                checks.info(
                    f"{clip_id} x {perturbation_id}: passive protocol "
                    f"(tracking error {record['tracking_error_deg']:.3f} deg is not gated)"
                )
            elif not tracking_within_tolerance and physically_valid:
                checks.check(
                    f"{clip_id} x {perturbation_id}: reference tracking within tolerance",
                    False,
                    f"{tracking_gate_error:.3f} deg ({tracking_window}) exceeds the pre-registered "
                    f"{tracking_tolerance:.1f} deg, so the clip is not usable as that action",
                )
            if ground_truth.label.label == LABEL_INVALID:
                # An invalid trial is correctly EXCLUDED from the validation set, not a
                # failure of the run. Exclusion is the gate doing its job.
                checks.skip(
                    f"{clip_id} x {perturbation_id}: excluded from the validation set",
                    "; ".join(ground_truth.label.reasons[:2]),
                )
            else:
                checks.check(
                    f"{clip_id} x {perturbation_id}: usable",
                    entry["gates"]["usable"],
                    ground_truth.label.reasons[0],
                )
                # Invariant: a fall label must carry a detected impact. If this ever
                # breaks, the labeller has started inferring falls from the perturbation
                # schedule rather than from the trajectory.
                if ground_truth.label.label == LABEL_FALL:
                    checks.check(
                        f"{clip_id} x {perturbation_id}: a fall label has an impact",
                        ground_truth.label.first_impact_s is not None,
                        f"impact at {ground_truth.label.first_impact_s} s",
                    )
        valid = [e for e in index if e["gates"]["physically_valid"]]
        usable = [e for e in index if e["gates"]["usable"]]
        tracked = [e for e in index if e["gates"]["tracking_gate_applies"]]
        checks.check(
            "the batch index step agrees with every recorded time axis",
            all(
                abs(e["step_s"] - physics_dt_s) <= 1e-12
                and abs(e["measured_step_s"] - physics_dt_s) <= 1e-9
                for e in index
            ),
            f"config {physics_dt_s} s = {physics_hz:g} Hz; largest disagreement "
            f"{max((abs(e['measured_step_s'] - physics_dt_s) for e in index), default=0.0):.2e} s",
        )
        checks.check(
            "trials were produced",
            bool(index),
            f"{len(index)} trials; numerically valid {len(valid)}, tracking-gated "
            f"{len(tracked)} (of which "
            f"{sum(1 for e in tracked if e['gates']['tracking_within_tolerance'])} within "
            f"{tracking_tolerance:g} deg), usable {len(usable)}",
        )
        checks.check(
            "at least one usable trial",
            bool(usable),
            "counts by label over the numerically valid trials: "
            + ", ".join(
                f"{name}={sum(1 for e in valid if e['label']['label'] == name)}"
                for name in sorted({e["label"]["label"] for e in valid})
            ),
        )
        checks.check(
            "the verified base scene was not modified",
            _sha256(scene) == scene_digest,
            str(scene),
        )
        path = write_json(
            args.out / "trials_index.json",
            {
                "config_sha256": config_digest,
                "scene_sha256": scene_digest,
                "human_id": plan.human_id,  # type: ignore[attr-defined]
                # Seconds and hertz are both recorded, under the names that say which.
                "physics_dt_s": physics_dt_s,
                "physics_hz": physics_hz,
                "channel_sample_hz": config.export.channel_sample_hz,  # type: ignore[attr-defined]
                "contact_categories": ["floor", "ground"],
                "runtime_capabilities": {
                    key: value
                    for key, value in runtime.capabilities.items()
                    if key != "contact_filter_paths"
                },
                "root_assistance": None if root_assist is None else asdict(root_assist),
                "trials": index,
            },
        )
        checks.info(f"wrote {path}")
        exit_code = checks.report(banner="human simulate")
    except Exception as exc:  # noqa: BLE001 - surfaced in the report
        LOGGER.exception("trial run failed")
        checks.check("trials completed without error", False, f"{type(exc).__name__}: {exc}")
        exit_code = checks.report(banner="human simulate")
    finally:
        if playback is not None:
            # Rendering must not advance dynamics or invalidate live tensor views.
            runtime.pause()
        if not args.headless and playback is not None:
            try:
                played = play_back_trial(app, stage, playback, fps=PLAYBACK_FPS)
                LOGGER.info("replayed %d recorded frames into the viewport", played)
            except Exception as exc:  # noqa: BLE001 - the report is already decided
                LOGGER.warning("viewport playback failed: %s: %s", type(exc).__name__, exc)
        if not args.headless and args.hold_seconds:
            hold_viewport(app, seconds=args.hold_seconds)
        app.close(exit_code=exit_code)
    return exit_code


def execute_trial(
    runtime: object,
    app: object,
    plan: object,
    reference: object,
    perturbation: object,
    config: object,
    *,
    pin_root: bool = False,
    mesh: object = None,
    proxy_template: object = None,
    root_assist: RootAssistConfig | None = None,
) -> dict:
    """Drive one trial at the physics rate and record every step.

    ``mesh`` selects the geometry that is exported: with the licensed skin loaded, the
    per-frame body points are skinned vertices following the PHYSICS pose of the
    articulation; without it they are the capsule surface proxy. Which one happened is
    recorded rather than assumed, because a wireless sample cannot be traced back to a
    body surface that was never there.


    The reference is resampled to the physics rate first, so one reference frame
    corresponds to one physics step and the recorded reference trace is exactly what
    the controller was asked to track.
    """

    # The STEP, not the rate. Reading the rate here made the recorded time axis run
    # 120x too fast (a 1 s trial exported as 14400 s) and collapsed the settle window
    # to one step, which is why every trial in the first run started already on the
    # floor. See the duration assertion below, which now catches this class of bug.
    step_s = float(config.simulation.physics_dt_s)  # type: ignore[attr-defined]
    frames = reference.frame_count  # type: ignore[attr-defined]
    # Reset once, then allow the solver to settle without free-root pose writes.
    # Explicit --pin-root remains a labelled kinematic assistance for both windows.
    target_joints = np.stack([
        joint_values_from_clip(reference, frame, plan)[0] for frame in range(frames)
    ])
    target_velocities = np.gradient(target_joints, step_s, axis=0)
    first_values = target_joints[0]
    pinned_root = np.asarray(plan.spawn_root_position) + reference.root_translation[0]  # type: ignore[attr-defined]
    pinned_quaternion = _quaternions(reference.root_rotation[:1])[0]
    runtime.set_root_pose(pinned_root, pinned_quaternion)  # type: ignore[attr-defined]
    runtime.set_joint_positions(first_values)  # type: ignore[attr-defined]
    runtime.set_joint_targets(first_values)  # type: ignore[attr-defined]
    runtime.reset_velocities()  # clear momentum from the previous trial
    if not runtime.set_control_scale(0.0 if config.control.mode == "torque_free" else 1.0):
        raise RuntimeError("failed to restore configured drive scale before trial")
    pinned = False
    pin_during_trial = bool(pin_root)
    settle_seconds = float(config.simulation.settle_seconds)  # type: ignore[attr-defined]
    settle_steps = int(round(settle_seconds / step_s))
    support_kind = "none"
    released_at_s = 0.0

    def assist(
        target_position: np.ndarray, target_quaternion: np.ndarray,
        target_linear: np.ndarray, target_angular: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if root_assist is None:
            return np.zeros(3), np.zeros(3)
        position, quaternion = runtime.root_pose()
        linear, angular = runtime.root_velocities()
        force, torque = root_wrench(
            root_assist, position=position, quaternion=quaternion,
            linear_velocity=linear, angular_velocity=angular,
            target_position=target_position, target_quaternion=target_quaternion,
            target_linear_velocity=target_linear, target_angular_velocity=target_angular,
            mass_kg=plan.total_mass_kg, gravity_m_s2=config.simulation.gravity_m_s2,
        )
        runtime.apply_root_wrench(force, torque)
        return force, torque

    for _ in range(settle_steps):
        if pin_during_trial:
            runtime.set_root_pose(pinned_root, pinned_quaternion)
            pinned = True
            support_kind = "kinematic_teleport"
        if root_assist is not None:
            assist(pinned_root, pinned_quaternion, np.zeros(3), np.zeros(3))
            support_kind = "bounded_external_wrench"
        # Deterministic stepper (see common.step_physics): one app.update()
        # advances two physics steps on this build, which ran every
        # seconds-derived window at twice the physics time it names.
        step_physics(1)

    # The reference height a fall is measured against must be where the body actually
    # stands once the solver has settled, not the analytic spawn height. Contact
    # offset, capsule stiffness and the PD drives all move the resting pelvis by a few
    # centimetres; a threshold quoted against the analytic height is then offset by the
    # same amount. Measured and recorded, so a trial can be re-thresholded later.
    settled_root, _ = runtime.root_pose()  # type: ignore[attr-defined]
    settled_pelvis_height = float(settled_root[2])
    if not np.isfinite(settled_pelvis_height) or settled_pelvis_height <= 0.0:
        raise RuntimeError(
            f"settled pelvis height read back as {settled_pelvis_height!r}; the settle "
            "window did not produce a standing body, so no fall threshold can be derived"
        )

    # This standing-start protocol requires an actual contact after settling.
    support_height = runtime.support_surface_height_m()  # type: ignore[attr-defined]
    if support_height is None:
        raise RuntimeError(
            "the support surface height could not be read, so no floor height exists "
            "to test contacts against and a resting point cannot be told from an "
            "impact. Refusing to record a trial whose contacts cannot be classified."
        )
    settle_segments: tuple[str, ...] = ()
    try:
        settle_segments = runtime.contact_segment_names()  # type: ignore[attr-defined]
    except ContactSourceUnavailable as exc:
        # A decoding failure must never be recorded as an empty contact set.
        raise RuntimeError(
            "the contact channel is readable but no reported point could be matched "
            f"to a body capsule ({exc}); the contact history would be recorded empty "
            "even where contacts exist. Refusing to record an unattributable trial."
        ) from exc
    if not settle_segments:
        raise RuntimeError(
            "the body was touching nothing during settle, so no fall threshold can be "
            "derived from its resting contacts. Refusing to record a trial that never "
            "reached the surface; check the spawn height and the settle window."
        )

    control_scale = 1.0
    scale_applied = True
    force_applied_count = 0
    force_failed_count = 0
    supports_removed = False

    times = np.zeros(frames)
    root_positions = np.zeros((frames, 3))
    root_quaternions = np.zeros((frames, 4))
    joint_positions = np.zeros((frames, len(plan.dof_names)))  # type: ignore[attr-defined]
    joint_velocities = np.zeros_like(joint_positions)
    link_positions = np.zeros((frames, len(plan.links), 3))  # type: ignore[attr-defined]
    reference_joints = np.zeros_like(joint_positions)
    reference_roots = np.asarray(plan.spawn_root_position) + reference.root_translation
    reference_quaternions = _quaternions(reference.root_rotation)
    reference_linear = np.gradient(reference_roots, step_s, axis=0)
    reference_angular = np.zeros((frames, 3))
    for index in range(frames - 1):
        reference_angular[index] = matrix_to_axis_angle(
            quaternion_to_matrix(reference_quaternions[index + 1])
            @ quaternion_to_matrix(reference_quaternions[index]).T
        ) / step_s
    reference_angular[-1] = reference_angular[-2]
    assist_forces = np.zeros((frames, 3))
    assist_torques = np.zeros((frames, 3))
    body_points: list[np.ndarray] = []
    mesh_vertices: list[np.ndarray] = []
    resolved_proxy_template = (
        proxy_template or build_capsule_proxy_template(plan)  # type: ignore[arg-type]
        if mesh is None
        else None
    )
    contacts = np.zeros(frames) if runtime.capabilities["contact_forces"] else None  # type: ignore[attr-defined]
    contact_samples: list[tuple[ContactSample, ...]] | None = (
        [() for _ in range(frames)]
        if runtime.capabilities.get("contact_source") == "physx_contact_report"
        else None
    )
    # Per-frame limb names, parallel to ``contact_samples``: frame -> one entry per
    # surviving sample, in the same order. Kept per-frame (not flat) so a reader can
    # join it against ``contact_position_m`` without re-deriving the frame boundaries.
    step_segments: list[tuple[str, ...]] = [() for _ in range(frames)]

    for frame in range(frames):
        now = frame * step_s
        if root_assist is not None:
            assist_forces[frame], assist_torques[frame] = assist(
                reference_roots[frame], reference_quaternions[frame],
                reference_linear[frame], reference_angular[frame],
            )
        values = target_joints[frame]
        using_perturbation = (
            perturbation.kind != "none"
            and perturbation.start_s <= now < perturbation.start_s + perturbation.duration_s
        )
        if using_perturbation and perturbation.kind == "force":
            force = np.asarray(perturbation.direction) * float(perturbation.magnitude_n)
            if runtime.apply_force(perturbation.body, force):  # type: ignore[attr-defined]
                force_applied_count += 1
            else:
                force_failed_count += 1
        if perturbation.kind == "control_failure" and now >= perturbation.start_s:
            if scale_applied:
                scale_applied = runtime.set_control_scale(float(perturbation.control_scale))  # type: ignore[attr-defined]
                control_scale = float(perturbation.control_scale)
        if perturbation.kind == "support_loss":
            supports_removed = True
            values = runtime.joint_positions_rad().copy()  # type: ignore[attr-defined]

        if pin_during_trial:
            # Assistance, not physics: the pelvis is placed on the reference root
            # trajectory instead of being integrated. Recorded, never silent.
            runtime.set_root_pose(  # type: ignore[attr-defined]
                np.asarray(plan.spawn_root_position) + reference.root_translation[frame],  # type: ignore[attr-defined]
                _quaternions(reference.root_rotation[frame : frame + 1])[0],
            )
        if config.control.mode == "kinematic":  # type: ignore[attr-defined]
            runtime.set_joint_positions(values)  # type: ignore[attr-defined]
        else:
            velocity_scale = 0.0 if config.control.mode == "torque_free" else control_scale
            runtime.set_joint_targets(values, target_velocities[frame] * velocity_scale)
        # One reference frame = one physics step (the reference was resampled to
        # the physics rate), so the loop steps the clock by exactly one step;
        # app.update() would advance two and play the reference at half rate.
        step_physics(1)  # type: ignore[attr-defined]

        times[frame] = now
        positions, orientations = runtime.root_pose()  # type: ignore[attr-defined]
        root_positions[frame] = positions
        root_quaternions[frame] = orientations
        joint_positions[frame] = runtime.joint_positions_rad()  # type: ignore[attr-defined]
        joint_velocities[frame] = runtime.joint_velocities_rad_s()  # type: ignore[attr-defined]
        reference_joints[frame] = values
        poses = runtime.link_poses()  # type: ignore[attr-defined]
        for index, link in enumerate(plan.links):  # type: ignore[attr-defined]
            link_positions[frame, index] = poses[link.name].translation
        if mesh is not None:
            cloud, owners = sample_skin_points(
                mesh,  # type: ignore[arg-type]
                {
                    name: (transform.rotation, transform.translation)
                    for name, transform in poses.items()
                },
                count=mesh.vertex_count,
                seed=config.export.surface_point_seed,  # type: ignore[attr-defined]
            )
        else:
            cloud, owners = pose_surface_points(
                plan,  # type: ignore[arg-type]
                poses,
                per_segment=config.export.surface_points_per_segment,  # type: ignore[attr-defined]
                seed=config.export.surface_point_seed,  # type: ignore[attr-defined]
            )
        body_points.append(cloud)
        if mesh is None:
            mesh_vertices.append(
                pose_capsule_proxy_mesh(  # type: ignore[arg-type]
                    resolved_proxy_template, plan, poses
                )
            )
        else:
            mesh_vertices.append(skin_mesh_sequence_frame(mesh, poses))  # type: ignore[arg-type]
        if contacts is not None:
            magnitude = runtime.contact_force_magnitudes()  # type: ignore[attr-defined]
            contacts[frame] = 0.0 if magnitude is None else float(magnitude[0])
        # Contact points are polled per step: the report describes the step just
        # taken, so it is read here and never re-read. A missing channel leaves
        # ``contact_source`` saying so rather than recording a fabricated zero.
        if contact_samples is not None:
            try:
                # One report read yields both the surviving points and their limb
                # names: two separate reads can straddle a physics update, which
                # would leave the positions and the names describing different steps.
                attributed = runtime.attributed_contact_samples()  # type: ignore[attr-defined]
            except ContactSourceUnavailable as exc:
                # The channel opened, answered for an earlier step, and has now failed
                # mid-trial. That is not "no contact" -- it is "no longer measured" --
                # and the trial has to say which. Returning here keeps the samples
                # already gathered and lets the caller mark the trial unusable, rather
                # than discarding the history and reporting an empty result.
                raise RuntimeError(
                    f"the contact channel stopped answering at frame {frame} of "
                    f"{frames}: {exc}. Contact history cannot be completed, so the "
                    "trial would export a partially measured contact sequence."
                ) from exc
            else:
                contact_samples[frame] = tuple(sample for sample, _ in attributed)
                step_segments[frame] = tuple(segment for _, segment in attributed)

    recorded_points = np.stack(body_points, axis=0)
    recorded_mesh = np.stack(mesh_vertices, axis=0)
    mesh_faces = (
        resolved_proxy_template.topology.faces  # type: ignore[union-attr]
        if mesh is None
        else np.asarray(mesh.faces, dtype=np.int64)
    )
    mesh_owners = (
        resolved_proxy_template.owners  # type: ignore[union-attr]
        if mesh is None
        else ()
    )
    MeshSequence(
        times,
        recorded_mesh,
        MeshTopology(mesh_faces),
        representation="capsule_proxy_mesh" if mesh is None else "smpl_skin_mesh",
        vertex_owners=mesh_owners,
    )
    expected_duration = (frames - 1) * step_s
    if abs(float(times[-1]) - expected_duration) > 1e-9:
        raise ValueError(
            f"recorded time axis ends at {float(times[-1])} s but {frames} frames at "
            f"{step_s} s/step span {expected_duration} s"
        )
    applied = (
        force_applied_count > 0
        if perturbation.kind == "force"
        else (scale_applied if perturbation.kind == "control_failure" else supports_removed)
    )
    return {
        "frames": frames,
        "pairs": np.array([frames]),
        "time_s": times,
        "root_position": root_positions,
        "root_quaternion": root_quaternions,
        "joint_positions_rad": joint_positions,
        "joint_velocities_rad_s": joint_velocities,
        "reference_joint_positions_rad": reference_joints,
        "reference_joint_velocities_rad_s": target_velocities,
        "reference_root_position": reference_roots,
        "reference_root_quaternion": reference_quaternions,
        "root_assist_force_n": assist_forces,
        "root_assist_torque_nm": assist_torques,
        "root_assistance": None if root_assist is None else asdict(root_assist),
        "link_positions": link_positions,
        "body_points": recorded_points,
        "body_point_owners": owners,
        "mesh_vertices_xyz": recorded_mesh,
        "mesh_faces": mesh_faces,
        "mesh_vertex_owners": mesh_owners,
        "mesh_representation": "capsule_proxy_mesh" if mesh is None else "smpl_skin_mesh",
        "link_names": tuple(link.name for link in plan.links),  # type: ignore[attr-defined]
        "contact_force_n": contacts,
        "contact_samples": contact_samples,
        "contact_source": str(runtime.capabilities.get("contact_source", "unavailable")),  # type: ignore[attr-defined]
        # The decoded collider path identifies the limb, including pelvis and leaves.
        "contact_attribution": str(  # type: ignore[attr-defined]
            runtime.capabilities.get("contact_attribution", "unresolved")
        ),
        # Height of the highest support surface (floor / step / raised platform) that
        # the body may rest on. Never assumed to be 0 so a raised support is handled.
        "support_surface_height_m": support_height,
        "contact_segments": step_segments,
        "settled_pelvis_height_m": settled_pelvis_height,
        "analytic_pelvis_height_m": float(plan.spawn_root_position[2]),  # type: ignore[attr-defined]
        "tracking_error_deg": float(np.degrees(np.abs(joint_positions - reference_joints).max())),
        "max_penetration_m": float(min(0.0, recorded_points[:, :, 2].min())),
        "perturbation_applied": bool(applied),
        "force_applied_frames": force_applied_count,
        "force_failed_frames": force_failed_count,
        "control_scale": control_scale,
        "supports_removed": supports_removed,
        "settle_used_root_support": pinned or root_assist is not None,
        "settle_seconds": settle_steps * step_s,
        # Pinning teleports the root; bounded assistance applies a recorded wrench.
        "settle_support_kind": support_kind,
        "settle_released_at_s": released_at_s,
        "settle_max_correction_m": 0.0,
        "step_s": step_s,
        "measured_step_s": float(np.median(np.diff(times))) if frames > 1 else step_s,
        # Joint PD alone does not track the root. Keep external assistance and
        # teleport pinning explicit so downstream consumers can distinguish them.
        "root_reference_tracked": pin_during_trial or root_assist is not None,
        "root_pinned_during_trial": pin_during_trial,
        "control_mode": str(config.control.mode),  # type: ignore[attr-defined]
        "body_geometry": REPRESENTATION_SKIN_MESH if mesh is not None else REPRESENTATION_PROXY,
    }


def assemble_ground_truth(
    *,
    plan: object,
    config: object,
    clip: object,
    reference: object,
    record: dict,
    perturbation: object,
    config_digest: str,
    scene_digest: str,
    body: object = None,
) -> GroundTruth:
    """Build the export object: label, resample onto the channel rate, provenance."""

    trajectory = Trajectory(
        times_s=record["time_s"],
        root_position=record["root_position"],
        root_quaternion=record["root_quaternion"],
        joint_positions=record["joint_positions_rad"],
        joint_names=tuple(plan.dof_names),  # type: ignore[attr-defined]
        body_points=record["body_points"],
        standing_height_m=float(plan.stats["standing_height_m"]),  # type: ignore[attr-defined]
        standing_pelvis_height_m=float(record["settled_pelvis_height_m"]),
        trunk_axis=trunk_axis_from_link_positions(record["link_names"], record["link_positions"]),
        contact_force_n=record["contact_force_n"],
    )
    label = label_trial(trajectory, config.events)  # type: ignore[attr-defined]
    phases = phase_labels_from_signal(
        record["time_s"],
        standing_frames=1,
        onset_s=label.imbalance_onset_s,
        impact_s=label.first_impact_s,
        stabilisation_s=label.stabilisation_s,
        final_phase=(
            "standing_recovery"
            if label.label == LABEL_RECOVERED
            else "fallen"
            if label.final_posture == "lying"
            else "standing"
        ),
    )

    physics_dt = float(config.simulation.physics_dt_s)  # type: ignore[attr-defined]
    channel_hz = float(config.export.channel_sample_hz)  # type: ignore[attr-defined]
    channel_times = _channel_times(record["time_s"], channel_hz)
    method = str(config.export.resample_method)  # type: ignore[attr-defined]
    # ``slerp`` in the config means "interpolate rotations on the manifold". Joint
    # angle series are scalars, so they use the smoothstep fallback while the root
    # quaternion uses proper quaternion interpolation.
    scalar_method = "smoothstep" if method == "slerp" else method
    channel_root_position = resample_series(
        record["time_s"], channel_times, record["root_position"], method="linear"
    )
    channel_root_quaternion = resample_series(
        record["time_s"], channel_times, record["root_quaternion"], method="quaternion"
    )
    channel_joints = resample_series(
        record["time_s"], channel_times, record["joint_positions_rad"], method=scalar_method
    )
    flat_points = record["body_points"].reshape(record["body_points"].shape[0], -1)
    channel_points = resample_series(
        record["time_s"], channel_times, flat_points, method="linear"
    ).reshape(len(channel_times), *record["body_points"].shape[1:])
    flat_mesh = record["mesh_vertices_xyz"].reshape(record["mesh_vertices_xyz"].shape[0], -1)
    channel_mesh = resample_series(
        record["time_s"], channel_times, flat_mesh, method="linear"
    ).reshape(len(channel_times), *record["mesh_vertices_xyz"].shape[1:])

    provenance = TrialProvenance(
        scene_id="apartment_cn_two_bedroom",
        scene_sha256=scene_digest,
        human_id=plan.human_id,  # type: ignore[attr-defined]
        rig_plan_sha256=sha256_text(plan.to_json()),  # type: ignore[attr-defined]
        config_sha256=config_digest,
        motion_id=clip.clip_id,  # type: ignore[attr-defined]
        motion_kind=clip.provenance.kind,  # type: ignore[attr-defined]
        motion_sha256=sha256_text(json.dumps(clip.as_dict(), sort_keys=True)),
        controller_mode=str(config.control.mode),  # type: ignore[attr-defined]
        root_mode=plan.root_mode,  # type: ignore[attr-defined]
        root_anchor_used=plan.root_mode == "anchored",  # type: ignore[attr-defined]
        body_representation=str(record["body_geometry"]),
        model_asset_id=None if body is None else body.model_id,
        model_asset_sha256=None if body is None or body.model is None else body.model.source_sha256,
        model_asset_available=bool(record["body_geometry"] == REPRESENTATION_SKIN_MESH),
        perturbation_id=perturbation.id,
        perturbation_detail={
            "kind": perturbation.kind,
            "body": perturbation.body,
            "start_s": perturbation.start_s,
            "duration_s": perturbation.duration_s,
            "direction": list(perturbation.direction),
            "magnitude_n": perturbation.magnitude_n,
            "control_scale": record["control_scale"],
            "applied": record["perturbation_applied"],
            "force_applied_frames": record["force_applied_frames"],
            "force_failed_frames": record["force_failed_frames"],
            "supports_removed": record["supports_removed"],
            "settle_seconds": record["settle_seconds"],
            "settle_used_root_support": record["settle_used_root_support"],
            # Persist assistance semantics, not only an internal implementation flag.
            "settle_support_kind": record["settle_support_kind"],
            "settle_released_at_s": record["settle_released_at_s"],
            "settle_max_correction_m": record["settle_max_correction_m"],
            "root_reference_tracked": record["root_reference_tracked"],
            "root_pinned_during_trial": record["root_pinned_during_trial"],
            "root_assistance": record["root_assistance"],
            "root_assist_force_peak_n": float(np.linalg.norm(
                record["root_assist_force_n"], axis=1).max()),
            "root_assist_torque_peak_nm": float(np.linalg.norm(
                record["root_assist_torque_nm"], axis=1).max()),
            "motion_metadata": dict(clip.metadata),
            "motion_source": clip.provenance.as_dict(),
            "control_mode": record["control_mode"],
            # The fall threshold is a fraction of ``settled_pelvis_height_m``: the
            # height the solver actually rests at. Both numbers are kept so a trial
            # can be re-thresholded against either convention without re-running.
            "settled_pelvis_height_m": round(record["settled_pelvis_height_m"], 6),
            "analytic_pelvis_height_m": round(record["analytic_pelvis_height_m"], 6),
            "pelvis_height_fraction": float(config.events.pelvis_height_fraction),  # type: ignore[attr-defined]
            "pelvis_height_threshold_m": round(
                record["settled_pelvis_height_m"] * float(config.events.pelvis_height_fraction),  # type: ignore[attr-defined]
                6,
            ),
            # Contact provenance travels with the perturbation detail because it is
            # per-trial: the channel's availability and the attribution method are
            # established once per trial, at settle.
            "contact_source": record["contact_source"],
            "contact_attribution": record["contact_attribution"],
            "support_surface_height_m": round(record["support_surface_height_m"], 6),
            "contact_segments_seen": sorted(
                {name for frame in record["contact_segments"] for name in frame}
            ),
        },
        seed=int(config.export.surface_point_seed),  # type: ignore[attr-defined]
        physics_dt_s=physics_dt,
        channel_sample_hz=channel_hz,
        resample_method=method,
        standing_height_m=float(plan.stats["standing_height_m"]),  # type: ignore[attr-defined]
        total_mass_kg=float(plan.total_mass_kg),  # type: ignore[attr-defined]
        generated_by="scripts/humans/simulate.py",
        subject=clip.provenance.subject,  # type: ignore[attr-defined]
        sequence=clip.provenance.sequence,  # type: ignore[attr-defined]
        license_notes=clip.provenance.license,  # type: ignore[attr-defined]
        notes=" | ".join(
            perturbed_trial_notes(perturbation, applied=record["perturbation_applied"])
        ),
    )
    return GroundTruth(
        provenance=provenance,
        label=label,
        time_physics_s=record["time_s"],
        time_channel_s=channel_times,
        root_position=record["root_position"],
        root_quaternion=record["root_quaternion"],
        joint_positions_rad=record["joint_positions_rad"],
        joint_velocities_rad_s=record["joint_velocities_rad_s"],
        joint_names=tuple(plan.dof_names),  # type: ignore[attr-defined]
        link_positions=record["link_positions"],
        link_names=tuple(link.name for link in plan.links),  # type: ignore[attr-defined]
        body_points=record["body_points"],
        body_point_owners=record["body_point_owners"],
        phase_labels=phases,
        reference_joint_positions_rad=record["reference_joint_positions_rad"],
        contact_force_n=record["contact_force_n"],
        channel_root_position=channel_root_position,
        channel_root_quaternion=channel_root_quaternion,
        channel_joint_positions_rad=channel_joints,
        channel_body_points=channel_points,
        mesh_vertices_xyz=record["mesh_vertices_xyz"],
        mesh_faces=record["mesh_faces"],
        mesh_representation=record["mesh_representation"],
        mesh_vertex_owners=record["mesh_vertex_owners"],
        channel_mesh_vertices_xyz=channel_mesh,
        rig_plan=plan,  # type: ignore[arg-type]
        metadata={
            "tracking_error_deg": record["tracking_error_deg"],
            "tracking_tolerance_deg": float(config.control.tracking_tolerance_deg),  # type: ignore[attr-defined]
            "tracking_within_tolerance": record["tracking_error_deg"]
            <= float(config.control.tracking_tolerance_deg),  # type: ignore[attr-defined]
            # One key per isolation field, dataset-namespaced. A single joined
            # "subject|sequence" key groups person1/clip1 apart from person1/clip2, which
            # lets the same human land on both sides of a train/test split.
            "isolation_keys": list(
                clip_group_keys(
                    {
                        "dataset": clip.provenance.source_id,  # type: ignore[attr-defined]
                        "subject": clip.provenance.subject,  # type: ignore[attr-defined]
                        "sequence": clip.provenance.sequence,  # type: ignore[attr-defined]
                    },
                    isolate=config.export.group_by,  # type: ignore[attr-defined]
                )
            ),
            "physics_to_channel_rate_ratio": round(1.0 / (physics_dt * channel_hz), 6),
            "contact_categories": ["floor", "ground"],
            "mesh_coordinate_system": "world_z_up_xyz",
            "mesh_units": "m",
            "mesh_topology_hash": MeshTopology(record["mesh_faces"]).sha256,
        },
    )


def _channel_times(physics_times: np.ndarray, channel_hz: float) -> np.ndarray:
    """Uniform channel-rate time base covering the same interval as the trial."""

    span = float(physics_times[-1])
    count = max(2, int(round(span * channel_hz)) + 1)
    times = np.arange(count, dtype=np.float64) / channel_hz
    if times[-1] > span:
        times = times[times <= span]
        if len(times) < 2:
            times = np.array([0.0, span])
    return times


def _sha256(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    checks = Checks()
    try:
        config, registry, motions = load_inputs(
            config_path=args.config,
            assets_path=args.assets,
            motions_path=args.motions,
            height_m=args.height,
            amass_root=args.amass_root,
            amass_limit=args.amass_limit,
        )
        if args.root_mode is not None:
            config = replace(config, rig=replace(config.rig, root_mode=args.root_mode))
        if args.seed is not None:
            config = replace(
                config, export=replace(config.export, surface_point_seed=int(args.seed))
            )
        # Build on the licensed body when it imports: the same fitted skeleton the plan
        # stage uses, so the simulated capsules and the skin come from one anatomy.
        body = select_body(
            registry,
            model_id=config.skeleton.model_asset,  # type: ignore[attr-defined]
            allow_procedural=bool(config.skeleton.allow_procedural_skeleton),  # type: ignore[attr-defined]
        )
        mesh = body.model.mesh() if body.has_skin_mesh else None
        rest = fit_rest_skeleton(config, mesh.rest_skeleton()) if mesh is not None else None
        spawn = resolve_spawn_point(
            args.scene_config,
            args.spawn_x,
            args.spawn_y,
            standing_height_m=config.skeleton.height_m,
        )
        plan = plan_human_rig(config, rest=rest, spawn_xy=spawn)
        if mesh is not None:
            mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
        if args.amass_file is not None:
            imported = load_amass_clip(args.amass_file)
            motions[imported.clip_id] = imported
        if args.amass_root is not None or args.amass_file is not None:
            screened: dict[str, object] = {}
            for clip_id, clip in motions.items():
                if clip.provenance.kind != "amass":
                    screened[clip_id] = clip
                    continue
                # AMASS root poses are sequence-local: the raw capture-volume
                # translation is not an apartment world pose (the first sample
                # can sit outside every room, so a pinned body would fall
                # forever).  Anchor to the first frame before screening so the
                # candidate test and the physics trials share one coordinate
                # frame, matching view_amass.py.
                clip = normalize_root_motion(crop_amass_clip(
                    clip, start_s=args.amass_start, duration_s=args.amass_duration
                ))
                clip = ground_amass_clip(
                    clip, plan, support_z_m=plan.spawn_root_position[2] - plan.ground_offset_m
                )
                result = screen_amass_clip(clip, plan)
                if args.amass_file is not None and not result.rig_expressible:
                    raise ValueError(f"selected AMASS cannot be tracked: {result.reason}")
                if not args.amass_fall_only or result.accepted:
                    screened[clip_id] = annotate_clip(clip, result)
            motions = screened
    except (OSError, ValueError) as exc:
        checks.check("inputs load and validate", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="human simulate")

    if args.dry_run:
        return dry_run(args, config, motions, plan, mesh=mesh)

    args.out.mkdir(parents=True, exist_ok=True)
    config_digest = _sha256(args.config)
    return run_trials(args, config, motions, plan, config_digest, body=body, mesh=mesh)


if __name__ == "__main__":
    sys.exit(main())
