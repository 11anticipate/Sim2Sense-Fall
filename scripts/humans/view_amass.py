#!/usr/bin/env python3
"""Preview a reference motion on the SMPL body in Isaac Sim.

Two sources, one playback loop:

* **AMASS** -- pass ``--amass-root`` pointing at already-authorized local ``.npz``
  files. The command never downloads or bypasses the AMASS registration gate.
* **The built-in scripted library** -- omit ``--amass-root`` and it reads
  ``configs/humans/motions.yaml`` instead. This is the source that actually works on a
  machine without AMASS, and it is how the fall references can be watched today.

Both produce the same ``MotionClip``, so the playback below is written once.

The preview is a **kinematic replay**: each frame writes the joint positions and the root
pose directly and zeroes the velocities, so the action is visible even though the capsule
collision proxies stay hidden. That also means it is *not* evidence of collision response
-- PhysX never gets a chance to react, because the next frame overwrites the pose. Use
``scripts/humans/verify.py`` or a future physics replay mode for that.

    # a built-in fall reference, no AMASS needed
    ~/isaacsim/python.sh scripts/humans/view_amass.py --motion fall_forward_reference

    # every fall reference in the library, back to back
    ~/isaacsim/python.sh scripts/humans/view_amass.py --fall-only

    # an authorized AMASS sequence
    ~/isaacsim/python.sh scripts/humans/view_amass.py \
        --amass-root /path/to/AMASS --fall-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import (  # noqa: E402
    DEFAULT_ASSETS,
    DEFAULT_CONFIG,
    DEFAULT_MOTIONS,
    DEFAULT_SCENE,
    DEFAULT_SCENE_CONFIG,
    Checks,
    activate_physics,
    boot_isaac,
    open_scene,
    resolve_spawn_point,
    set_physics_dt,
    write_json,
)

from sim2sense_fall.humans.amass import (  # noqa: E402
    annotate_clip,
    load_amass_clip_by_id,
    load_amass_library,
    normalize_root_motion,
    screen_amass_clip,
)
from sim2sense_fall.humans.assets import load_asset_registry, select_body  # noqa: E402
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.mesh_sequence import (  # noqa: E402
    fit_mesh_to_rest_joints,
    skin_mesh_sequence_frame,
)
from sim2sense_fall.humans.motion import MotionClip, load_motion_library  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    HumanRigPlan,
    fit_rest_skeleton,
    forward_kinematics,
    plan_human_rig,
)
from sim2sense_fall.humans.rotations import axis_angle_to_quaternion  # noqa: E402
from sim2sense_fall.humans.usd_human import (  # noqa: E402
    HumanRuntime,
    build_human_stage,
    pxr_modules,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=None,
        help="local AMASS .npz root; omit to preview the built-in scripted library instead",
    )
    parser.add_argument(
        "--motion",
        default=None,
        help="clip id to preview; an amass__ id needs --amass-root, any other id does not",
    )
    parser.add_argument(
        "--fall-only",
        action="store_true",
        help=(
            "restrict to fall candidates: the AMASS screen when --amass-root is given, "
            "the fall_reference tag otherwise"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--motions", type=Path, default=DEFAULT_MOTIONS)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="headless playback duration; 0 means one clip",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="write viewport screenshots here instead of playing, then exit",
    )
    parser.add_argument(
        "--capture-frames",
        type=int,
        default=6,
        help="how many evenly spaced frames to screenshot with --capture-dir",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", dest="headless", action="store_false", default=False)
    mode.add_argument("--headless", dest="headless", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    if args.capture_frames < 2:
        parser.error("--capture-frames must be at least 2")
    if args.amass_root is None and (args.motion or "").startswith("amass__"):
        # Named an AMASS clip without a root to read it from. Saying so here beats
        # failing later inside the loader with a path error that names neither.
        parser.error(f"--motion {args.motion} is an AMASS id, which needs --amass-root")
    return args


def _screened_library(args: argparse.Namespace, plan: HumanRigPlan) -> dict[str, MotionClip]:
    """Resolve the clips to preview, from whichever source the arguments name.

    The two sources differ only in how a clip is *loaded*; everything after this point
    works on a ``MotionClip`` and does not care where it came from. AMASS clips need
    their root motion anchored to the apartment frame (their root poses are
    sequence-local), while the scripted library is already authored as local offsets.
    """

    if args.amass_root is not None:
        return _amass_library(args, plan)
    return _scripted_library(args)


def _amass_library(args: argparse.Namespace, plan: HumanRigPlan) -> dict[str, MotionClip]:
    if args.motion and args.motion.startswith("amass__") and args.limit is None:
        # An explicit motion id is resolved directly.  This avoids parsing all
        # 110 Transitions files before Isaac Sim can even start.
        clips = {
            args.motion: load_amass_clip_by_id(args.amass_root, args.motion, target_up_axis="z")
        }
    else:
        clips = load_amass_library(args.amass_root, limit=args.limit, target_up_axis="z")
    screened: dict[str, MotionClip] = {}
    for clip_id, source_clip in clips.items():
        # AMASS root poses are sequence-local.  Anchor them before screening so
        # the candidate test and the GUI playback use the same coordinate frame.
        clip = normalize_root_motion(source_clip)
        result = screen_amass_clip(clip, plan)
        if args.fall_only and not result.accepted:
            continue
        screened[clip_id] = annotate_clip(clip, result)
    if not screened:
        raise ValueError(
            "no AMASS clips remain after screening; omit --fall-only to preview any clip"
        )
    return screened


def _apply_frame(
    runtime: object,
    skin_mesh: object,
    mesh: object,
    plan: HumanRigPlan,
    reference: MotionClip,
    usd: object,
    frame: int,
) -> None:
    """Write one replay frame to the skin, the joints and the root together.

    Kept in one place so the live loop and the capture path cannot drift: a screenshot
    taken with the skin at frame *n* and the articulation at frame *n-1* would show the
    surface detached from the capsules, which looks exactly like a real skinning bug.
    """

    values = _joint_values(reference, frame, plan)
    poses = forward_kinematics(
        plan,
        values,
        root_position=(0.0, 0.0, 0.0),
        root_rotation=reference.root_rotation[frame],
    )
    points = skin_mesh_sequence_frame(mesh, poses)
    skin_mesh.GetPointsAttr().Set(  # type: ignore[attr-defined]
        [usd.Gf.Vec3f(*[float(value) for value in point]) for point in points]  # type: ignore[attr-defined]
    )
    ordered = [values.get(name, 0.0) for name in plan.dof_names]
    runtime.set_joint_positions(ordered)  # type: ignore[attr-defined]
    runtime.set_joint_targets(ordered)  # type: ignore[attr-defined]
    runtime.set_root_pose(  # type: ignore[attr-defined]
        np.asarray(plan.spawn_root_position) + reference.root_translation[frame],
        axis_angle_to_quaternion(reference.root_rotation[frame]),
    )
    runtime.reset_velocities()  # type: ignore[attr-defined]


def _capture_frames(
    *,
    app: object,
    runtime: object,
    skin_mesh: object,
    mesh: object,
    plan: HumanRigPlan,
    reference: MotionClip,
    usd: object,
    count: int,
    out_dir: Path,
    clip_id: str,
) -> list[Path]:
    """Step evenly through the clip and screenshot the viewport at each step.

    Deterministic rather than time-driven: the live loop picks the frame from the wall
    clock, which is right for watching and wrong for a figure, because two runs would
    capture different poses. Here every frame is written, the viewport is allowed to
    render it, and only then is the image taken.
    """

    import asyncio

    from omni.kit.viewport.utility import (
        capture_viewport_to_file,
        get_active_viewport,
        next_viewport_frame_async,
    )

    frames = _evenly_spaced(reference.frame_count, max(2, count))
    out_dir.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim did not create an active viewport to capture")

    async def capture() -> list[Path]:
        written: list[Path] = []
        for frame in frames:
            _apply_frame(runtime, skin_mesh, mesh, plan, reference, usd, frame)
            # The replay writes USD and PhysX state; the viewport needs a rendered frame
            # after that write, or the capture shows the previous pose.
            await next_viewport_frame_async(viewport, 2)
            path = out_dir / f"{clip_id}_frame{frame:04d}.png"
            # ``capture_viewport_to_file`` returns a capture *delegate*, not an
            # awaitable, despite the "future-like object" wording in its docstring:
            # awaiting it raises "object MultiAOVFileCapture can't be used in 'await'
            # expression". It is the delegate's ``wait_for_result`` that blocks until the
            # image has actually been written.
            pending = capture_viewport_to_file(viewport, file_path=str(path))
            await pending.wait_for_result()
            written.append(path)
        # The final capture resolves its future before the PNG is flushed, so a run that
        # exits straight after the loop leaves the last frame as a 0-byte ``.cap-*``
        # temporary. Pump a few more frames so the file lands before the app closes.
        await next_viewport_frame_async(viewport, 4)
        return written

    task = asyncio.ensure_future(capture())
    while app.is_running() and not task.done():  # type: ignore[attr-defined]
        app.update()  # type: ignore[attr-defined]
    return task.result()


def _evenly_spaced(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    step = total / count
    return [min(int(round(index * step)), total - 1) for index in range(count)]


def _banner(args: argparse.Namespace) -> str:
    """Name the report after the source that was actually read.

    A check report that says "AMASS preview" while playing a scripted clip is the kind
    of small untruth that makes provenance untrustworthy, and this project keeps its
    replay sources apart everywhere else for the same reason.
    """

    return "AMASS preview" if args.amass_root is not None else "scripted motion preview"


def _scripted_library(args: argparse.Namespace) -> dict[str, MotionClip]:
    """The built-in reference motions, which need no downloaded asset at all.

    ``--fall-only`` selects on the ``fall_reference`` tag rather than on the AMASS
    screen: ``screen_amass_clip`` raises on non-AMASS provenance by design, so running
    it here would be asking the wrong question of the wrong thing. The tag is the
    library's own declaration of which clips are fall references, which is the same
    criterion ``collect_fall_mesh.py --fall-only`` uses.
    """

    library = load_motion_library(args.motions, topology=load_human_config(args.config).topology)
    if args.motion is not None:
        if args.motion not in library:
            raise ValueError(
                f"unknown motion {args.motion!r}; known: {sorted(library)}. "
                "An AMASS clip needs --amass-root."
            )
        return {args.motion: library[args.motion]}
    selection = {
        clip_id: clip
        for clip_id, clip in library.items()
        if not args.fall_only or "fall_reference" in clip.tags
    }
    if not selection:
        raise ValueError(
            f"no scripted clip matches (--fall-only={args.fall_only}) in {args.motions}"
        )
    return dict(sorted(selection.items()))


def _joint_values(clip: MotionClip, frame: int, plan: HumanRigPlan) -> dict[str, float]:
    return {
        joint.name: float(clip.rotation_of(frame, joint.chain_joint)["xyz".index(joint.axis)])
        for joint in plan.joints
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = Checks()
    config = load_human_config(args.config)
    registry = load_asset_registry(args.assets, project_root=REPO_ROOT)
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=True)
    mesh = body.model.mesh() if body.has_skin_mesh else None
    rest = fit_rest_skeleton(config, mesh.rest_skeleton()) if mesh is not None else None
    spawn = resolve_spawn_point(
        args.scene_config, None, None,
        standing_height_m=config.skeleton.height_m,
    )
    plan = plan_human_rig(config, rest=rest, spawn_xy=spawn)
    if mesh is None:
        raise RuntimeError(
            "the motion preview needs the licensed SMPL surface asset; it skins the "
            "same body the rig drives, so without it there is nothing to show"
        )
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    clips = _screened_library(args, plan)
    motion_id = args.motion or next(iter(clips))
    if motion_id not in clips:
        raise ValueError(f"unknown AMASS motion {motion_id!r}; known: {sorted(clips)}")
    clip = clips[motion_id]
    reference = clip.resample(1.0 / config.simulation.physics_dt_s, method="slerp")
    display_values = _joint_values(clip, 0, plan)
    initial_poses = forward_kinematics(
        plan,
        display_values,
        root_position=(0.0, 0.0, 0.0),
        root_rotation=reference.root_rotation[0],
    )
    initial_vertices = skin_mesh_sequence_frame(mesh, initial_poses)
    app = boot_isaac(args.headless)
    if app is None:
        checks.check("Isaac Sim started", False, "run through ~/isaacsim/python.sh")
        return checks.report(banner=_banner(args))
    exit_code = 1
    try:
        target = (
            REPO_ROOT
            / "artifacts"
            / "humans"
            / ("amass_preview.usda" if args.amass_root is not None else "motion_preview.usda")
        )
        build_human_stage(
            plan,
            target,
            base_scene=args.scene,
            spawn_position=plan.spawn_root_position,
            skin_points=initial_vertices,
            skin_faces=mesh.faces,
        )
        stage = open_scene(app, target)
        activation = activate_physics()
        set_physics_dt(config.simulation.physics_dt_s)
        runtime = HumanRuntime(stage, plan, enable_contact_views=False)
        runtime.play()
        for _ in range(4):
            app.update()
        skin_prim = stage.GetPrimAtPath("/World/Human/Skin")
        usd = pxr_modules()
        checks.check(
            "reference clip loaded",
            reference.frame_count >= 2,
            f"{motion_id}: {reference.frame_count} frames",
        )
        checks.check(
            "root motion anchored",
            np.allclose(reference.root_translation[0], 0.0)
            and np.allclose(reference.root_rotation[0], 0.0),
            "first frame is the local apartment anchor",
        )
        checks.check("physics activated", activation["active_engine"] == "physx", str(activation))
        from sim2sense_fall.scenes.view import configure_inspection_view

        # ``configure_inspection_view`` authors a camera and *returns its path*; the
        # caller has to point the viewport at it. This call used to discard the return
        # value, so the viewport kept whatever camera it had and every screenshot showed
        # a corner of the room with no body in it -- a capture that "succeeded" while
        # proving nothing about what was on screen.
        camera_path = configure_inspection_view(stage, mode="human", aspect_ratio=1600 / 900)
        from omni.kit.viewport.utility import get_active_viewport

        active_viewport = get_active_viewport()
        if active_viewport is not None:
            active_viewport.camera_path = camera_path
            checks.info(f"viewport camera set to {camera_path}")
        skin_mesh = usd.UsdGeom.Mesh(skin_prim)
        if args.capture_dir is not None:
            captured = _capture_frames(
                app=app,
                runtime=runtime,
                skin_mesh=skin_mesh,
                mesh=mesh,
                plan=plan,
                reference=reference,
                usd=usd,
                count=args.capture_frames,
                out_dir=args.capture_dir,
                clip_id=motion_id,
            )
            checks.check(
                "frames were captured from the viewport",
                len(captured) >= 2 and all(path.is_file() for path in captured),
                f"{len(captured)} PNGs in {args.capture_dir}; expected {args.capture_frames} "
                f"evenly spaced frames",
            )
            written = sum(1 for path in captured if path.is_file())
            missing = [path.name for path in captured if not path.is_file()]
            checks.check(
                "every captured frame reached the disk",
                not missing,
                f"{written} of {len(captured)} written"
                + (f"; missing {missing}" if missing else ""),
            )
            for path in captured:
                print(f"  {path}")
        else:
            started = time.monotonic()
            frame = -1
            duration = args.seconds if args.seconds > 0 else float(reference.times_s[-1])
            while app.is_running():
                elapsed = time.monotonic() - started
                current = int(elapsed * float(reference.fps))
                if args.headless and elapsed >= duration:
                    break
                current %= reference.frame_count
                if current != frame:
                    _apply_frame(runtime, skin_mesh, mesh, plan, reference, usd, current)
                    frame = current
                app.update()
        write_json(
            target.with_suffix(".json"),
            {
                "motion": clip.as_dict(),
                "reference": {
                    "fps": float(reference.fps),
                    "frame_count": reference.frame_count,
                    "duration_s": reference.duration_s,
                    "resample_method": "slerp",
                },
                "physics": activation,
                "scene": str(args.scene),
                "usd": str(target),
            },
        )
        checks.info(f"previewed {motion_id}: {clip.provenance.subject}/{clip.provenance.sequence}")
        exit_code = checks.report(banner=_banner(args))
    except Exception as exc:  # noqa: BLE001 - surfaced in the check report
        checks.check(f"{_banner(args)} completed", False, f"{type(exc).__name__}: {exc}")
        exit_code = checks.report(banner=_banner(args))
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
