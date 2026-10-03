"""Export ProtoMotions tracker/reference sessions into the physics-trial contract.

The ProtoMotions route (2026-09-30, user decision) replaces the self-built SMPL rig
as the main data production line. This script bridges its outputs into the SAME
trial contract ``scripts/humans/simulate.py`` writes, so ``import_fall_mesh.py``
and ``detection_data`` consume tracker sessions unchanged:

- ``<sample_id>.npz``           -- ``time_physics_s`` / ``mesh_vertices_xyz`` / ``mesh_faces``
- ``<sample_id>.trial.json``    -- trajectory label (from ``label_trial``, never from
  the file name), provenance, mesh description
- ``trials_index.json``         -- per-trial acceptance gates (``usable`` required by
  the Sionna importer's trial path)

Two capture modes:

``--mode reference`` (CPU): skin the MotionLib's reference body states at their
native cadence (30 fps for AMASS-sourced libs). ``fidelity=kinematic_reference``:
no physics, the same honest class as the 09-23 reference captures. Validates the
whole contract path without a GPU.

``--mode tracker`` (GPU, newton venv): run the trained tracker policy and record
the MEASURED body states at every physics substep. The Newton ``sim.fps`` is 120,
so the recorded mesh truth is native 120 Hz -- no interpolation anywhere.
``fidelity=physics_tracker``. CUDA graphs are disabled for the export run because
the per-substep snapshot hook needs the explicit physics loop; physics behaviour
is unchanged (same solver, same dt, same control targets).

Labels come from the measured trajectory through the pre-registered ``label_trial``
rules (``events`` block of the human config), fail-closed; the motion file name is
recorded as provenance but never used as a label.

Run with the ProtoMotions venv interpreter, e.g.::

    ~/.local/opt/ProtoMotions/.venv_newton/bin/python \\
        scripts/protomotions/export_tracker_trials.py \\
        --mode reference \\
        --motion-file artifacts/protomotions_bridge/amass_subset/verify_motionlib.pt \\
        --out artifacts/protomotions_bridge/export_reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOMOTIONS_ROOT = Path.home() / ".local/opt/ProtoMotions"

sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.humans.assets import load_asset_registry, select_body  # noqa: E402
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.events import (  # noqa: E402
    LABEL_FALL,
    Trajectory,
    label_trial,
    trunk_axis_from_link_positions,
)

LOGGER = logging.getLogger("export_tracker_trials")

#: ProtoMotions SMPL body names (the MJCF body order the motion lib and
#: ``RobotState`` use) mapped to this repo's SMPL topology names. Verified
#: empirically against the packaged motion lib: body 0 is the pelvis at standing
#: height, the toe bodies sit at z~0 and the head tops the standing frame.
PM_TO_OUR_JOINT: tuple[tuple[str, str], ...] = (
    ("Pelvis", "pelvis"),
    ("L_Hip", "left_hip"),
    ("L_Knee", "left_knee"),
    ("L_Ankle", "left_ankle"),
    ("L_Toe", "left_foot"),
    ("R_Hip", "right_hip"),
    ("R_Knee", "right_knee"),
    ("R_Ankle", "right_ankle"),
    ("R_Toe", "right_foot"),
    ("Torso", "spine1"),
    ("Spine", "spine2"),
    ("Chest", "spine3"),
    ("Neck", "neck"),
    ("Head", "head"),
    ("L_Thorax", "left_collar"),
    ("L_Shoulder", "left_shoulder"),
    ("L_Elbow", "left_elbow"),
    ("L_Wrist", "left_wrist"),
    ("L_Hand", "left_hand"),
    ("R_Thorax", "right_collar"),
    ("R_Shoulder", "right_shoulder"),
    ("R_Elbow", "right_elbow"),
    ("R_Wrist", "right_wrist"),
    ("R_Hand", "right_hand"),
)
PM_JOINT_NAMES = tuple(pm for pm, _ in PM_TO_OUR_JOINT)
OUR_JOINT_NAMES = tuple(our for _, our in PM_TO_OUR_JOINT)

#: SMPL neutral v1.1.0 stature, from the model payload this repo logs at load time.
SMPL_NEUTRAL_STANDING_HEIGHT_M = 1.717

#: Activity mapping for the channel contract. ``fall`` keeps its label; every other
#: valid outcome (no_fall / recovered / controlled_lowering) is a measured non-fall
#: trajectory and feeds the negative class; ``invalid`` stays invalid and is
#: excluded downstream by the importer's ``label.valid`` check.
ACTIVITY_BY_LABEL = {
    LABEL_FALL: "fall",
    "no_fall": "adl",
    "recovered": "adl",
    "controlled_lowering": "adl",
}

#: Termination threshold fallback when the env config does not carry one.
DEFAULT_TRACKING_TOLERANCE_M = 0.5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_smpl_mesh():
    registry = load_asset_registry(
        REPO_ROOT / "configs/humans/assets.yaml", project_root=REPO_ROOT
    )
    body = select_body(registry, model_id="smpl_neutral_v1_1_0", allow_procedural=False)
    return body.model.mesh()


def skin_frames(mesh, body_pos: np.ndarray, body_rot_xyzw: np.ndarray) -> np.ndarray:
    """Skin ``(T, 24, 3)`` world positions + ``(T, 24, 4)`` xyzw quaternions.

    ProtoMotions SMPL bodies ARE the SMPL joints in world space (z-up, feet at
    z~0), and :func:`sim2sense_fall.humans.skinning.skin_with_link_poses` takes
    exactly world (rotation, translation) per joint. Verified against reference
    frames: standing extent ~1.68 m, lying extent ~0.63 m.
    """
    from scipy.spatial.transform import Rotation as sRot

    from sim2sense_fall.humans.skinning import skin_with_link_poses

    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_rot_xyzw = np.asarray(body_rot_xyzw, dtype=np.float64)
    if body_pos.shape != body_rot_xyzw.shape[:1] + body_pos.shape[1:] or (
        body_pos.ndim != 3 or body_pos.shape[1] != len(OUR_JOINT_NAMES)
        or body_rot_xyzw.shape[-1] != 4
    ):
        raise ValueError(
            "expected body_pos (T, 24, 3) and body_rot (T, 24, 4), got "
            f"{body_pos.shape} / {body_rot_xyzw.shape}"
        )
    if not np.isfinite(body_pos).all() or not np.isfinite(body_rot_xyzw).all():
        raise ValueError("body states contain non-finite values")

    frames = body_pos.shape[0]
    vertices = np.empty((frames, mesh.vertex_count, 3), dtype=np.float64)
    for frame in range(frames):
        poses = {
            our: (
                sRot.from_quat(body_rot_xyzw[frame, index]).as_matrix(),
                body_pos[frame, index],
            )
            for index, our in enumerate(OUR_JOINT_NAMES)
        }
        vertices[frame] = skin_with_link_poses(mesh, poses)
    return vertices


def build_trajectory(
    times_s: np.ndarray,
    body_pos: np.ndarray,
    body_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    dof_names: tuple[str, ...],
    mesh,
) -> tuple[Trajectory, np.ndarray]:
    """Assemble the labelling trajectory and the skinned mesh from body states."""

    vertices = skin_frames(mesh, body_pos, body_rot_xyzw)
    trajectory = Trajectory(
        times_s=times_s,
        root_position=body_pos[:, 0, :],
        # RobotState / Newton quaternions are xyzw (w_last); recorded as such.
        root_quaternion=body_rot_xyzw[:, 0, :],
        joint_positions=dof_pos,
        joint_names=dof_names,
        body_points=vertices,
        standing_height_m=SMPL_NEUTRAL_STANDING_HEIGHT_M,
        # Per-motion peak pelvis height: the standing reference for the 0.55
        # low-posture fraction. For a fall clip the first frame stands; for a
        # get-up clip the final frame stands. Recorded in provenance.
        standing_pelvis_height_m=float(body_pos[:, 0, 2].max()),
        trunk_axis=trunk_axis_from_link_positions(OUR_JOINT_NAMES, body_pos),
    )
    return trajectory, vertices


def write_trial(
    out_dir: Path,
    sample_id: str,
    times_s: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    label: Any,
    entry_extra: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write ``<id>.npz`` + ``<id>.trial.json`` and return the index entry.

    The importer resolves the npz as ``json_path.with_suffix("") + ".npz"`` from
    the trial json's stem, so the pair must share the bare stem.
    """

    npz_path = out_dir / f"{sample_id}.npz"
    np.savez_compressed(
        npz_path,
        time_physics_s=times_s,
        mesh_vertices_xyz=vertices,
        mesh_faces=faces,
    )
    activity = ACTIVITY_BY_LABEL.get(label.label, "invalid")
    trial = {
        "sample_id": sample_id,
        "subject": "smpl_neutral",
        "fidelity": provenance["fidelity"],
        "activity": activity,
        "seed": provenance["seed"],
        "label": label.as_dict(),
        "provenance": provenance,
        "mesh": {
            "representation": "smpl_skin_mesh",
            "vertex_count": int(vertices.shape[1]),
            "face_count": int(faces.shape[0]),
            "topology_hash": hashlib.sha256(
                np.ascontiguousarray(faces).tobytes()
            ).hexdigest(),
            "coordinate_system": "world_z_up_xyz",
            "units": "m",
            "sample_rate_hz": round(float(1.0 / np.median(np.diff(times_s))), 6),
            "native": provenance["native_cadence"],
            "npz": str(npz_path),
        },
    }
    json_path = out_dir / f"{sample_id}.trial.json"
    json_path.write_text(
        json.dumps(trial, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "motion_id": sample_id,
        "perturbation_id": provenance["fidelity"],
        "npz": str(npz_path),
        "json": str(json_path),
        "frames": int(len(times_s)),
        "duration_s": round(float(times_s[-1] - times_s[0]), 6),
        "activity": activity,
        "sample_rate_hz": trial["mesh"]["sample_rate_hz"],
        "label": label.as_dict(),
        **entry_extra,
    }


def run_reference_mode(args: argparse.Namespace, mesh, events) -> list[dict[str, Any]]:
    """Skin the MotionLib reference states; no physics, no GPU."""

    import torch

    motion_file = Path(args.motion_file)
    lib = torch.load(motion_file, weights_only=False)
    files = list(lib["motion_files"])
    starts = lib["length_starts"].tolist()
    counts = lib["motion_num_frames"].tolist()
    dts = lib["motion_dt"].tolist()
    if not (len(files) == len(starts) == len(counts) == len(dts)):
        raise ValueError("motion lib index arrays disagree on length")

    dof_names = tuple(f"dof_{i}" for i in range(lib["dps"].shape[1]))
    entries: list[dict[str, Any]] = []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _motion_id, (path, start, count, dt) in enumerate(
        zip(files, starts, counts, dts, strict=True)
    ):
        sample_id = f"ref_{Path(path).stem.removesuffix('_poses')}"
        gts = lib["gts"][start : start + count].numpy().astype(np.float64)
        grs = lib["grs"][start : start + count].numpy().astype(np.float64)
        dps = lib["dps"][start : start + count].numpy().astype(np.float64)
        times = np.arange(count, dtype=np.float64) * float(dt)
        trajectory, vertices = build_trajectory(
            times, gts, grs, dps, dof_names, mesh
        )
        label = label_trial(trajectory, events)
        provenance = {
            "fidelity": "kinematic_reference",
            "native_cadence": True,
            "seed": int(args.seed),
            "motion_file": str(path),
            "motion_sha256": sha256_file(path) if Path(path).is_file() else None,
            "motion_lib_sha256": sha256_file(motion_file),
            "standing_pelvis_height_source": "per_motion_peak_pelvis_height",
            "note": (
                "reference FK states skinned directly; no physics, no tracker, "
                "no interpolation -- the lib's own 30 fps cadence"
            ),
        }
        entry = write_trial(
            out_dir,
            sample_id,
            times,
            vertices,
            mesh.faces,
            label,
            {
                "motion_source": provenance["motion_file"],
                "gates": {
                    "physically_valid": True,  # finite FK states, validated above
                    "tracking_within_tolerance": None,  # no tracker involved
                    "label_credible": label.label != "invalid",
                    "usable": label.label != "invalid",
                },
            },
            provenance,
        )
        entries.append(entry)
        LOGGER.info(
            "%s: %s (%s), %d frames @ %.1f Hz",
            sample_id,
            label.label,
            entry["activity"],
            len(times),
            1.0 / float(dt),
        )
    return entries


def run_tracker_mode(args: argparse.Namespace, mesh, events) -> list[dict[str, Any]]:
    """Run the tracker policy and record measured states at every physics substep."""

    pm_root = Path(args.protomotions_root).resolve()
    sys.path.insert(0, str(pm_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from protomotions.utils.simulator_imports import import_simulator_before_torch

    import_simulator_before_torch("newton")

    import torch
    import warp as wp
    from lightning.fabric import Fabric
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )
    from protomotions.simulator.newton.simulator import ControlType, NewtonSimulator
    from protomotions.utils.component_builder import build_all_components
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class

    class ExportNewtonSimulator(NewtonSimulator):
        """NewtonSimulator with a per-substep snapshot hook (no CUDA graph)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # The captured-graph fast path would skip the explicit loop this
            # exporter needs; disabling it changes nothing about the physics.
            self.use_cuda_graph = False
            self.recording = False
            self.recorded_body: list[Any] = []
            self.recorded_dof: list[Any] = []

        def _snapshot(self) -> None:
            transforms = (
                wp.to_torch(self.robot_view.get_link_transforms(self.state_0))
                .squeeze(1)
                .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
            )
            dofs = (
                wp.to_torch(self.robot_view.get_dof_positions(self.state_0))
                .squeeze(1)
                .view(self.num_envs, -1)
            )
            self.recorded_body.append(transforms[0].clone())
            self.recorded_dof.append(dofs[0].clone())

        def _simulate(self) -> None:
            for _ in range(self.decimation):
                self.state_0.clear_forces()
                if self._max_action_latency_ms > 0.0:
                    if self.control_type == ControlType.BUILT_IN_PD:
                        self._apply_control()
                    else:
                        actions = self._get_actions_for_physics_step()
                        self._apply_newton_control(actions)
                        self._advance_action_latency()
                if self.control_type == ControlType.PROPORTIONAL:
                    self._apply_pd_kernel(self.state_0)
                elif self.control_type == ControlType.TORQUE:
                    self._apply_torques_kernel_method()
                if self.viewer:
                    self.viewer.apply_forces(self.state_0)
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )
                self.state_0, self.state_1 = self.state_1, self.state_0
                if self.recording:
                    self._snapshot()

    checkpoint = Path(args.checkpoint)
    resolved = torch.load(
        checkpoint.parent / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved["terrain"]
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]
    if args.pd_damping_scale is not None:
        for control_info in robot_config.control.control_info.values():
            if control_info.damping is not None:
                control_info.damping *= float(args.pd_damping_scale)
        LOGGER.info("PD damping scale = %s", args.pd_damping_scale)
    if args.action_ema is not None:
        # Deployment-style low-pass on the policy output: the trained MLP fires
        # per-step action jumps that read as leg jitter in the playback. This
        # mirrors MimicEvaluator's eval_action_ema_alpha (0.5-0.8 recommended;
        # smaller = smoother). Recorded in provenance so no export can pass a
        # smoothed replay off as the raw policy.
        agent_config.evaluator.eval_action_ema_alpha = float(args.action_ema)
        LOGGER.info("evaluator action EMA alpha = %s", args.action_ema)

    # Keep the original _target_: both convert_friction_for_simulator and
    # build_all_components parse the full "protomotions.simulator.<name>...."
    # form. The recording subclass is swapped in AFTER construction via __class__.
    simulator_config.num_envs = int(args.num_envs)
    simulator_config.headless = True
    motion_lib_config.motion_file = str(Path(args.motion_file).resolve())
    motion_lib_config.motion_file_shard_indices = None

    fabric = Fabric(
        **FabricConfig(
            accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=str(checkpoint.parent),
    )
    env = get_class(env_config._target_)(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )
    agent = get_class(agent_config._target_)(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False, load_training_state=False)
    agent.eval()

    sim = env.simulator
    # Swap in the recording subclass after construction: same layout, identical
    # physics, plus the per-substep snapshot hook. Disabling the captured CUDA
    # graph routes _physics_step through the explicit loop the hook needs.
    sim.__class__ = ExportNewtonSimulator
    sim.use_cuda_graph = False
    sim.recording = False
    sim.recorded_body = []
    sim.recorded_dof = []
    sim_dt = float(sim.sim_dt)
    if abs(sim_dt - 1.0 / 120.0) > 1e-6:
        LOGGER.warning("sim dt is %s s; expected 1/120 for the 120 Hz contract", sim_dt)
    env_dt = float(env.dt)
    tolerance_m = DEFAULT_TRACKING_TOLERANCE_M
    try:
        tolerance_m = float(
            env_config.terminations.configs["tracking"].config.max_body_pos_error
        )
    except (AttributeError, KeyError, TypeError):
        LOGGER.info("no tracking threshold in env config; using %s m", tolerance_m)

    lib = torch.load(Path(args.motion_file), weights_only=False)
    files = list(lib["motion_files"])
    starts = lib["length_starts"].tolist()
    counts = lib["motion_num_frames"].tolist()
    dts = lib["motion_dt"].tolist()
    ref_gts = lib["gts"]

    device = fabric.device
    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    dof_names = tuple(
        f"dof_{i}" for i in range(robot_config.kinematic_info.num_dofs)
    )

    entries: list[dict[str, Any]] = []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for motion_id, (path, start, count, ref_dt) in enumerate(
        zip(files, starts, counts, dts, strict=True)
    ):
        sample_id = f"tracker_{Path(path).stem.removesuffix('_poses')}"
        motion_length_s = count * float(ref_dt)
        max_steps = int(np.ceil(motion_length_s / env_dt))

        env.motion_manager.motion_ids[env_ids] = torch.full(
            (1,), int(motion_id), dtype=torch.long, device=device
        )
        env.motion_manager.motion_times[env_ids] = 0.0
        sim.recorded_body = []
        sim.recorded_dof = []
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        sim.recording = True  # reset placed the robot at reference frame 0
        sim._snapshot()  # frame 0 = the reset state itself
        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)

        terminated_early = False
        nan_aborted = False
        previous_actions = None
        agent.pre_collect_step(0)
        for step_idx in range(max_steps):
            model_outs = agent.model(obs_td)
            actions = (
                model_outs["mean_action"]
                if "mean_action" in model_outs
                else model_outs["action"]
            )
            if args.action_ema is not None:
                if previous_actions is None:
                    previous_actions = actions.clone()
                actions = (
                    args.action_ema * actions
                    + (1.0 - args.action_ema) * previous_actions
                )
                previous_actions = actions.clone()
            try:
                obs, _rewards, dones, terminated, _extras = env.step(actions)
            except AssertionError as e:
                # A soft-gain policy can blow a startup to NaN; record the
                # motion as failed and continue with the next one instead of
                # killing the whole export.
                LOGGER.warning("trial %s aborted by non-finite sim state: %s", sample_id, e)
                nan_aborted = True
                break
            agent.pre_collect_step(step_idx + 1)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            # `terminated` is the failure path (tracking divergence); a bare
            # `done` is the reference clip exhausting (bootstrap) -- a normal end.
            if bool(terminated[0]):
                terminated_early = True
                break
            if bool(dones[0]):
                break
        sim.recording = False
        if nan_aborted:
            LOGGER.error("%s: excluded (non-finite rollout)", sample_id)
            continue

        body = (
            torch.stack(sim.recorded_body).cpu().numpy().astype(np.float64)
        )
        dof = torch.stack(sim.recorded_dof).cpu().numpy().astype(np.float64)
        body_pos = body[:, :, :3]
        body_rot = body[:, :, 3:]
        times = np.arange(len(body), dtype=np.float64) * sim_dt
        if len(times) < 2:
            LOGGER.error("%s: only %d recorded frames; skipping", sample_id, len(times))
            continue

        trajectory, vertices = build_trajectory(
            times, body_pos, body_rot, dof, dof_names, mesh
        )
        label = label_trial(trajectory, events)

        # Tracking gate: measured vs reference (nearest reference frame). The
        # reference is 30 Hz against 120 Hz physics; nearest-neighbour matching
        # is a coarse gate, recorded as such.
        ref_pos = ref_gts[start : start + count].numpy().astype(np.float64)
        ref_idx = np.clip(
            np.round(times / float(ref_dt)).astype(int), 0, count - 1
        )
        # Newton tiles environments across the arena, so the simulated body sits
        # far from the reference's origin even when tracking perfectly. Align by
        # the constant spawn offset (frame-0 roots) -- a real teleport would then
        # still show up as growing error.
        spawn_offset = body_pos[0, 0, :] - ref_pos[0, 0, :]
        ref_matched = ref_pos[ref_idx]
        body_error = np.linalg.norm(
            (body_pos - spawn_offset) - ref_matched, axis=-1
        )
        tracking_max = float(body_error.max())
        tracking_mean = float(body_error.mean())
        if tracking_max > 1.0:
            frame_idx, body_idx = np.unravel_index(
                np.argmax(body_error), body_error.shape
            )
            LOGGER.warning(
                "%s: max error %.3f m at t=%.2f s body=%s (ref frame %d, "
                "measured z %.2f vs ref z %.2f)",
                sample_id,
                tracking_max,
                times[frame_idx],
                OUR_JOINT_NAMES[body_idx] if body_idx < len(OUR_JOINT_NAMES) else body_idx,
                ref_idx[frame_idx],
                body_pos[frame_idx, body_idx, 2],
                ref_pos[ref_idx[frame_idx], body_idx, 2],
            )
        physically_valid = bool(np.isfinite(body).all() and np.isfinite(dof).all())
        usable = bool(
            physically_valid
            and not terminated_early
            and tracking_max <= tolerance_m
            and label.label != "invalid"
        )

        provenance = {
            "fidelity": "physics_tracker",
            "native_cadence": True,
            "seed": int(args.seed),
            "motion_file": str(path),
            "motion_sha256": sha256_file(path) if Path(path).is_file() else None,
            "motion_lib_sha256": sha256_file(Path(args.motion_file)),
            "tracker_checkpoint": str(checkpoint),
            "tracker_checkpoint_sha256": sha256_file(checkpoint),
            "standing_pelvis_height_source": "per_motion_peak_pelvis_height",
            "simulator": "newton",
            "sim_dt_s": sim_dt,
            "cuda_graph_disabled": True,
            "action_ema_alpha": args.action_ema,
            "pd_damping_scale": args.pd_damping_scale,
            "note": (
                "measured states at every physics substep (native 120 Hz); frame 0 "
                "is the reset state, subsequent frames one physics step apart"
            ),
        }
        entry = write_trial(
            out_dir,
            sample_id,
            times,
            vertices,
            mesh.faces,
            label,
            {
                "motion_source": provenance["motion_file"],
                "episode_terminated_early": terminated_early,
                "tracking_error_max_m": round(tracking_max, 6),
                "tracking_error_mean_m": round(tracking_mean, 6),
                "tracking_tolerance_m": tolerance_m,
                "tracking_reference": "nearest reference frame (30 Hz vs 120 Hz)",
                "gates": {
                    "physically_valid": physically_valid,
                    "tracking_within_tolerance": tracking_max <= tolerance_m,
                    "episode_completed": not terminated_early,
                    "label_credible": label.label != "invalid",
                    "usable": usable,
                },
            },
            provenance,
        )
        entries.append(entry)
        LOGGER.info(
            "%s: %s (%s), %d frames @ %.1f Hz, tracking max %.3f m%s",
            sample_id,
            label.label,
            entry["activity"],
            len(times),
            1.0 / sim_dt,
            tracking_max,
            ", TERMINATED EARLY" if terminated_early else "",
        )
    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()
    return entries


def write_index(out_dir: Path, entries: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    usable = [e for e in entries if e["gates"]["usable"]]
    index = {
        "export_mode": args.mode,
        "motion_file": str(Path(args.motion_file).resolve()),
        "seed": int(args.seed),
        "events_config": "configs/humans/human_smpl_multiaxis.yaml::events (pre-registered)",
        "trials": entries,
        "summary": {
            "total": len(entries),
            "usable": len(usable),
            "labels": {
                name: sum(1 for e in entries if e["label"]["label"] == name)
                for name in sorted({e["label"]["label"] for e in entries})
            },
        },
    }
    path = out_dir / "trials_index.json"
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("reference", "tracker"), required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="tracker mode: last.ckpt (dir must hold resolved_configs_inference.pt)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-ema", type=float, default=None,
        help="tracker mode: EMA low-pass on policy actions (0.35-0.8; smaller "
        "= smoother legs). Recorded in provenance; None = raw policy.",
    )
    parser.add_argument(
        "--pd-damping-scale", type=float, default=None,
        help="tracker mode: multiply configured PD damping at runtime; recorded in provenance",
    )
    parser.add_argument("--protomotions-root", type=Path, default=DEFAULT_PROTOMOTIONS_ROOT)
    parser.add_argument(
        "--events-config", type=Path,
        default=REPO_ROOT / "configs/humans/human_smpl_multiaxis.yaml",
        help="human config YAML whose pre-registered events block labels the trials",
    )
    args = parser.parse_args(argv)
    if args.mode == "tracker" and args.checkpoint is None:
        parser.error("tracker mode needs --checkpoint")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.action_ema is not None and not 0.0 <= args.action_ema <= 1.0:
        parser.error("--action-ema must be in [0, 1]")
    if args.pd_damping_scale is not None and args.pd_damping_scale <= 0.0:
        parser.error("--pd-damping-scale must be > 0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    human_config = load_human_config(args.events_config)
    mesh = load_smpl_mesh()
    if args.mode == "reference":
        entries = run_reference_mode(args, mesh, human_config.events)
    else:
        entries = run_tracker_mode(args, mesh, human_config.events)
    index_path = write_index(args.out, entries, args)
    usable = sum(1 for e in entries if e["gates"]["usable"])
    LOGGER.info("wrote %s: %d trials, %d usable", index_path, len(entries), usable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
