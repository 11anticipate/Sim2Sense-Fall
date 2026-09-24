"""Shared bootstrap for the ``scripts/humans`` entry points.

Kept deliberately small: path resolution, a check recorder that still reports under
Kit (which replaces ``sys.stdout``), and JSON writing that refuses NaN.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "humans" / "human_smpl_neutral.yaml"
DEFAULT_ASSETS = REPO_ROOT / "configs" / "humans" / "assets.yaml"
DEFAULT_MOTIONS = REPO_ROOT / "configs" / "humans" / "motions.yaml"
DEFAULT_SCENE = REPO_ROOT / "artifacts" / "scenes" / "indoor_apartment.usda"
DEFAULT_SCENE_CONFIG = REPO_ROOT / "configs" / "scenes" / "indoor_apartment.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "humans"

LOGGER = logging.getLogger("humans_common")

__all__ = [
    "DEFAULT_ASSETS",
    "DEFAULT_CONFIG",
    "DEFAULT_MOTIONS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SCENE",
    "DEFAULT_SCENE_CONFIG",
    "REPO_ROOT",
    "human_spawn_clearance_m",
    "resolve_spawn_point",
    "scene_spawn_point",
    "set_physics_dt",
    "Checks",
    "load_inputs",
    "print_report",
    "write_json",
]


class Checks:
    """Tiny assertion recorder that reports through ``sys.__stdout__``.

    Isaac Sim replaces ``sys.stdout``, so a report written only to ``print``
    disappears in headless mode. Every line goes to the original stream, and the
    failures are also returned so the entry point can set an exit code.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        status = "PASS" if condition else "FAIL"
        self.lines.append(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not condition:
            self.failures.append(label)
        return bool(condition)

    def skip(self, label: str, reason: str) -> None:
        self.lines.append(f"[SKIP] {label} -- {reason}")

    def info(self, message: str) -> None:
        self.lines.append(f"[INFO] {message}")

    def report(self, *, banner: str) -> int:
        stream = getattr(sys, "__stdout__", None) or sys.stdout
        try:
            stream.write("\n".join(self.lines) + "\n")
            stream.write(f"{banner}: {'FAILED' if self.failures else 'PASSED'}\n")
            if self.failures:
                stream.write("failed checks: " + ", ".join(self.failures) + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass
        return 1 if self.failures else 0


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return path


def print_report(lines: list[str]) -> None:
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    try:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def load_inputs(
    *,
    config_path: Path,
    assets_path: Path,
    motions_path: Path,
    height_m: float | None = None,
    amass_root: Path | None = None,
    amass_limit: int | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load and validate the human configuration, asset registry and motion library."""

    from sim2sense_fall.humans.assets import load_asset_registry
    from sim2sense_fall.humans.config import load_human_config
    from sim2sense_fall.humans.motion import load_motion_library

    config = load_human_config(config_path)
    if height_m is not None:
        if height_m <= 0:
            raise ValueError(f"--height must be positive, got {height_m!r}")
        config = replace(config, skeleton=replace(config.skeleton, height_m=float(height_m)))
    registry = load_asset_registry(assets_path, project_root=REPO_ROOT)
    motions = load_motion_library(motions_path, topology=config.topology)
    if amass_root is not None:
        from sim2sense_fall.humans.amass import load_amass_library

        loaded = load_amass_library(amass_root, limit=amass_limit)
        if loaded.failures:
            LOGGER.warning(
                "skipped %d of %d AMASS sequences that could not be read; first failure: %s",
                len(loaded.failures),
                loaded.scanned,
                loaded.failures[0][1],
            )
        imported = loaded.clips
        overlap = sorted(set(motions).intersection(imported))
        if overlap:
            raise ValueError(f"AMASS motion ids collide with scripted motions: {overlap}")
        motions.update(imported)
    return config, registry, motions


# ---------------------------------------------------------------------------
# Isaac Sim bootstrap
# ---------------------------------------------------------------------------


def boot_isaac(headless: bool, *, width: int = 1600, height: int = 900) -> Any:
    """Start Isaac Sim and return the app handle.

    ``sys.argv`` is cleared immediately before constructing ``SimulationApp``
    because the launcher forwards unrecognised arguments to Kit, which then fails
    with an unrelated error. Returns ``None`` when Isaac Sim is not importable, so
    the caller can decide whether that is fatal.
    """

    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        return None
    argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        return SimulationApp({"headless": headless, "width": width, "height": height})
    finally:
        # argparse already consumed the real arguments; keep the launcher isolated.
        del argv


def open_scene(app: Any, path: Path, *, updates: int = 5) -> Any:
    """Open ``path`` in the live Isaac Sim context and return the stage."""

    import omni.usd

    if not omni.usd.get_context().open_stage(str(path)):
        raise RuntimeError(f"Isaac Sim could not open {path}")
    for _ in range(updates):
        app.update()
    return omni.usd.get_context().get_stage()


def activate_physics() -> dict[str, Any]:
    """Attach PhysX to the live stage and switch on result write-back."""

    from sim2sense_fall.scenes.usd import activate_physics as activate

    return activate()


def step_simulation(app: Any, seconds: float) -> int:
    """Advance the timeline by ``seconds`` and return the physics step count."""

    from sim2sense_fall.scenes.usd import step_simulation as step

    return step(app, seconds)


def set_physics_dt(dt_s: float) -> tuple[float, bool]:
    """Apply the configured physics step to the running simulation.

    The scene's ``UsdPhysics.Scene`` does not set ``timeStepsPerSecond``, so PhysX
    otherwise runs at the application default (1/60 s) and the configured
    ``simulation.physics_dt_s`` would be quietly ignored. Returns the step actually
    in force and whether the request was accepted, so a mismatch is reported rather
    than assumed away.
    """

    from isaacsim.core.simulation_manager import SimulationManager

    if dt_s <= 0:
        raise ValueError(f"physics dt must be positive, got {dt_s!r}")
    try:
        SimulationManager.set_physics_dt(float(dt_s))
        accepted = True
    except Exception:  # noqa: BLE001 - reported through the return value
        accepted = False
    return float(SimulationManager.get_physics_dt()), accepted


def step_physics(steps: int) -> None:
    """Advance the physics clock by exactly ``steps`` steps of the configured dt.

    ``app.update()`` is NOT a unit of physics time on this build: each call advances a
    fixed 1/60 s of physics time regardless of the configured step. Measured at
    dt = 1/120 s: two steps per update, byte-identical with ``omni.kit.loop-isaac``
    enabled and with the GUI kit's ``runLoops`` manual-mode settings injected
    (``artifacts/humans/probe_loop_*.log``). Every window this pipeline quotes in
    seconds must therefore be stepped here, not by counting updates. The engine's
    step counter is read back afterwards so a silently failing stepper cannot pass
    as a completed window -- the same discipline as the raised-then-dropped gravity
    control, because "the loop ran" and "the clock moved" are different claims.

    Fabric is deliberately left stale (``update_fabric=False``): every physics read
    in this pipeline goes through tensor views, the headless trials render nothing,
    and the viewport replay paths still call ``app.update()`` themselves.
    """

    from isaacsim.core.simulation_manager import SimulationManager

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}")
    before = int(SimulationManager.get_num_physics_steps())
    SimulationManager.step(steps=int(steps), update_fabric=False)
    advanced = int(SimulationManager.get_num_physics_steps()) - before
    if advanced != int(steps):
        raise RuntimeError(
            f"requested {steps} physics steps but the engine advanced {advanced}; "
            "the clock cannot be trusted, so a seconds-based window would be wrong"
        )


#: Human footprint half-width used when choosing a spawn point, in metres.
SPAWN_CLEARANCE_M = 0.40
#: The rest pose is a T-pose, whose arm span slightly exceeds the stature, so the body's
#: sideways half-extent is this fraction of its standing height. It is multiplied into
#: the wall clearance so a spawn chosen for furniture clearance cannot also put the hands
#: through a wall: measured at 0.39 m of arm outside the living room at the shipped
#: 0.40 m clearance and a 1.70 m figure, whose arms span 1.83 m.
BODY_HALF_SPAN_FRACTION = 0.55


def human_spawn_clearance_m(standing_height_m: float) -> float:
    """Wall clearance a standing human needs so its rest pose stays inside the room.

    ``scene_spawn_point`` picks the spot furthest from *furniture*, which is a different
    question from "is the whole body inside the walls". At the shipped clearance the
    answer was no: the figure spawned 0.52 m from two walls and its arms reached 0.39 m
    past the wall face, which is visible in any screenshot of the preview.
    """

    height = float(standing_height_m)
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError(f"standing_height_m must be finite and positive, got {height!r}")
    return SPAWN_CLEARANCE_M + BODY_HALF_SPAN_FRACTION * height


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def scene_spawn_point(
    scene_config: Path, *, clearance_m: float = SPAWN_CLEARANCE_M
) -> tuple[float, float]:
    """Pick a floor position for the human inside the apartment.

    The apartment has one floor slab per room, so a spawn point outside the rooms
    drops the body straight through the world -- which is exactly what a hard-coded
    ``(0, 0)`` did. This samples a grid inside the largest room, keeps points that
    clear the wall thickness, and returns the point furthest from any furniture,
    measured against the plan's own furniture footprints.
    """

    from sim2sense_fall.scenes.spec import load_scene_spec

    spec = load_scene_spec(scene_config)
    room = max(spec.rooms, key=lambda entry: entry.size[0] * entry.size[1])
    x0, y0, x1, y1 = room.bounds
    inset = room.wall_thickness + clearance_m
    xs = _linspace(x0 + inset, x1 - inset, 21)
    ys = _linspace(y0 + inset, y1 - inset, 21)
    obstacles = [
        (
            room.origin[0] + item.position[0],
            room.origin[1] + item.position[1],
            max(
                0.5 * (item.size[0] if item.size else 0.4),
                0.5 * (item.size[1] if item.size else 0.4),
                0.4,
            ),
        )
        for item in room.furniture
    ]
    best = (0.0, 0.0)
    best_clearance = -1.0
    for x in xs:
        for y in ys:
            clearance = min(
                (math.hypot(x - ox, y - oy) - radius for ox, oy, radius in obstacles),
                default=float("inf"),
            )
            if clearance > best_clearance:
                best_clearance = clearance
                best = (float(x), float(y))
    if best_clearance <= 0.0:
        raise ValueError(
            f"room {room.id!r} has no position with {clearance_m:g} m of clearance from its "
            f"furniture (best {best_clearance:.3f} m); move furniture or reduce clearance"
        )
    return best


def resolve_spawn_point(
    scene_config: Path,
    spawn_x: float | None,
    spawn_y: float | None,
    *,
    standing_height_m: float = 0.0,
) -> tuple[float, float]:
    """Use the explicit spawn if both coordinates were given, otherwise derive one.

    ``standing_height_m`` widens the wall clearance by the body's own half-extent. Every
    caller that knows the figure's stature should pass it: without it the derived spawn
    only guarantees furniture clearance, and the T-pose arms reach outside the room.
    Callers that do not pass it keep the old, furniture-only behaviour.
    """

    if (spawn_x is None) != (spawn_y is None):
        raise ValueError("give both --spawn-x and --spawn-y, or neither")
    if spawn_x is None:
        return scene_spawn_point(
            scene_config,
            clearance_m=(
                human_spawn_clearance_m(standing_height_m)
                if standing_height_m > 0.0
                else SPAWN_CLEARANCE_M
            ),
        )
    return (float(spawn_x), float(spawn_y))
