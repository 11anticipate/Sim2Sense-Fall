"""Interactive Isaac Sim viewer for exported ProtoMotions trial meshes.

Keyboard-controlled playback of the 120 Hz physics-truth exports
(``export_tracker_trials.py``) inside the apartment scene, for visual
acceptance. This is a *player*, not a controller: the motion is the recorded
simulation, replayed as a display mesh. Interactive WASD steering of the
physical body is the MaskedMimic line, not this viewer.

    ~/isaacsim/python.sh scripts/protomotions/view_trials_isaac.py \
        --trials "artifacts/protomotions_bridge/export_tracker_r3/tracker_*.npz"

Keys: SPACE pause/resume, LEFT/RIGHT scrub (data frames), 1..9 select clip,
R restart clip, ESC quit. The viewport camera orbits with the mouse.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from common import REPO_ROOT as COMMON_REPO_ROOT  # noqa: E402
from common import boot_isaac, open_scene  # noqa: E402

PLAYBACK_FPS = 30.0
DATA_FPS = 120.0
FRAME_STEP = int(round(DATA_FPS / PLAYBACK_FPS))
SPAWN_XY = (10.255, 1.455)  # apartment living-room spawn (keyboard.yaml)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trials", required=True, help="glob of exported trial .npz mesh files"
    )
    parser.add_argument(
        "--scene", type=Path, default=COMMON_REPO_ROOT / "artifacts/scenes/indoor_apartment.usda"
    )
    parser.add_argument("--start", type=int, default=0, help="clip index to start on")
    parser.add_argument(
        "--seconds", type=float, default=0.0,
        help="auto-quit after this many wall-clock seconds (0 = run until ESC)",
    )
    return parser.parse_args()


def load_clips(pattern: str) -> list[dict[str, np.ndarray]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no trial npz matched {pattern!r}")
    clips = []
    for path_str in paths:
        path = Path(path_str)
        data = np.load(path)
        vertices = np.asarray(data["mesh_vertices_xyz"], dtype=np.float32)
        if vertices.ndim != 3:
            raise SystemExit(f"{path}: mesh_vertices_xyz must be (T, V, 3)")
        # The sibling trial.json carries the measured label -- use it to order
        # and name clips (fall first) instead of relying on file names.
        label, activity = "adl", "adl"
        trial_json = path.with_name(path.stem + ".trial.json")
        if trial_json.is_file():
            payload = json.loads(trial_json.read_text(encoding="utf-8"))
            label = str(payload.get("label", {}).get("label", label))
            activity = str(payload.get("activity", activity))
        # Re-center to the apartment spawn: the Newton arena tiles environments
        # far from the origin, so recorded clips sit metres away from the rooms.
        centroid_xy = vertices[0].mean(axis=0)[:2]
        vertices[:, :, 0] += SPAWN_XY[0] - float(centroid_xy[0])
        vertices[:, :, 1] += SPAWN_XY[1] - float(centroid_xy[1])
        clips.append(
            {
                "name": Path(path).stem,
                "label": label,
                "activity": activity,
                "vertices": vertices.tolist(),
            }
        )
    # Fall clips first so the letter aliases map meaningfully; then by name.
    clips.sort(key=lambda c: (c["label"] != "fall", c["name"]))
    return clips


def main() -> int:
    args = parse_args()
    clips = load_clips(args.trials)
    faces = np.load(sorted(glob.glob(args.trials))[0])["mesh_faces"]

    app = boot_isaac(headless=False)
    stage = open_scene(app, args.scene)
    from pxr import Gf, UsdGeom, Vt

    skin = UsdGeom.Mesh.Define(stage, "/World/TrialPlayback/Skin")
    skin.GetFaceVertexIndicesAttr().Set([int(i) for f in faces for i in f])
    skin.GetFaceVertexCountsAttr().Set([3] * len(faces))
    skin.GetSubdivisionSchemeAttr().Set("none")
    points_attr = skin.GetPointsAttr()

    from sim2sense_fall.scenes.view import apply_inspection_view

    apply_inspection_view(stage, mode="human", aspect_ratio=16 / 9)

    import carb.input  # noqa: F401  (Kit provides carb after boot)
    import omni.appwindow

    keys: set[str] = set()
    window = omni.appwindow.get_default_app_window()
    keyboard = window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()

    def on_key(event, *_args) -> bool:
        key = getattr(event.input, "name", None)
        if not isinstance(key, str):
            return True  # not a keyboard event this handler understands
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Digit key names are not verified in this repo's input path; log
            # every press so any naming mismatch is visible in the console.
            print(f"key: {key!r}")
            keys.add(key)
            command_queue.append(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            keys.discard(key)
        return key not in CONTROL_KEYS

    CONTROL_KEYS = {
        "SPACE", "LEFT", "RIGHT", "R", "ESCAPE", "Z", "X", "C",
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5",
        "KEY_6", "KEY_7", "KEY_8", "KEY_9",
    }
    command_queue: list[str] = []
    subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)

    clip_index = max(0, min(args.start, len(clips) - 1))
    frame = 0
    paused = False
    print(
        "controls: SPACE pause | LEFT/RIGHT scrub | 1-9 (or Z/X/C) clip | "
        "R restart | ESC quit"
    )
    for i, clip in enumerate(clips):
        print(f"  [{i + 1}] {clip['name']}  label={clip['label']}")

    def apply_clip(clip: dict[str, np.ndarray]) -> None:
        nonlocal frame, paused
        frame = 0
        paused = False
        points_attr.Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in clip["vertices"][0]])
        )

    def show(clip: dict[str, np.ndarray], fi: int) -> None:
        points_attr.Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in clip["vertices"][fi]])
        )

    apply_clip(clips[clip_index])
    last_play_step = time.monotonic()
    started = time.monotonic()
    try:
        while app.is_running():
            if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                break
            while command_queue:
                key = command_queue.pop(0)
                if key == "ESCAPE":
                    raise KeyboardInterrupt
                elif key == "SPACE":
                    paused = not paused
                elif key in {"LEFT", "RIGHT"}:
                    direction = -FRAME_STEP if key == "LEFT" else FRAME_STEP
                    frame = int(
                        np.clip(
                            frame + direction,
                            0,
                            len(clips[clip_index]["vertices"]) - 1,
                        )
                    )
                    show(clips[clip_index], frame)
                elif key == "R":
                    apply_clip(clips[clip_index])
                elif key in {"Z", "X", "C"} or any(ch.isdigit() for ch in key):
                    # 1/2/3 (any spelling carrying a digit) or Z/X/C fallback
                    letter_pick = {"Z": 0, "X": 1, "C": 2}.get(key)
                    digits = [ch for ch in key if ch.isdigit()]
                    pick = letter_pick if letter_pick is not None else (
                        int(digits[-1]) - 1 if digits else -1
                    )
                    if 0 <= pick < len(clips):
                        clip_index = pick
                        apply_clip(clips[clip_index])
                        print(f"clip: {clips[clip_index]['name']}")
            if "LEFT" in keys or "RIGHT" in keys:
                direction = -FRAME_STEP if "LEFT" in keys else FRAME_STEP
                frame = int(np.clip(frame + direction, 0, len(clips[clip_index]["vertices"]) - 1))
                show(clips[clip_index], frame)
                last_play_step = time.monotonic()
            elif not paused and time.monotonic() - last_play_step >= 1.0 / PLAYBACK_FPS:
                frame = (frame + FRAME_STEP) % len(clips[clip_index]["vertices"])
                show(clips[clip_index], frame)
                last_play_step = time.monotonic()
            app.update()
    except KeyboardInterrupt:
        pass
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
        close = getattr(app, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
