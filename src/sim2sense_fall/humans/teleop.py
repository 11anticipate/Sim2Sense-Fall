"""CPU-testable keyboard intent and bounded AMASS target generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from .amass import (
    AMASS_BODY_FRAME,
    crop_amass_clip,
    ground_amass_clip,
    load_amass_clip,
    normalize_root_motion,
)
from .contact_control import capsule_bottom, sideways_leg_dofs
from .rig import HumanRigPlan, forward_kinematics, joint_values_from_clip
from .rotations import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    rotation_about_axis,
)

if TYPE_CHECKING:
    from .contact_gait import AmassSwingShape


@dataclass(frozen=True)
class TeleopConfig:
    speed_m_s: float
    acceleration_m_s2: float
    turn_speed_deg_s: float
    turn_acceleration_deg_s2: float
    transition_s: float
    max_joint_speed_rad_s: float
    max_target_lead_m: float
    max_heading_lead_deg: float
    # Optional force-target height offset; never used as spawn/reset teleport.
    # Contact-retargeted locomotion uses zero (height follows reachable feet).
    root_z_offset_m: float = 0.0

    def __post_init__(self) -> None:
        positive = all(
            np.isfinite(v) and v > 0
            for key, v in vars(self).items()
            if key != "root_z_offset_m"
        )
        if not positive:
            raise ValueError("teleop controller settings must be finite and positive")
        if not np.isfinite(self.root_z_offset_m) or self.root_z_offset_m > 0:
            raise ValueError("root_z_offset_m must be finite and <= 0")


ANCHOR_SOURCES = frozenset({"model", "contact", "contact_and_model"})


def load_keyboard_config(path: Path, project_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("keyboard config must be a mapping")
    payload.setdefault("root_assist_scale", 1.0)
    if not np.isfinite(payload["root_assist_scale"]) or not 0 <= payload["root_assist_scale"] <= 1:
        raise ValueError("root_assist_scale must be finite in [0, 1]")
    # Feedforward from the reference COM's derived acceleration. 0 keeps the
    # pelvis actuator purely reactive (spring + constant gravity fraction);
    # 1 applies the reference motion's own inertial force in full.
    payload.setdefault("reference_feedforward_scale", 0.0)
    if not np.isfinite(payload["reference_feedforward_scale"]) or not (
            0 <= payload["reference_feedforward_scale"] <= 1):
        raise ValueError("reference_feedforward_scale must be finite in [0, 1]")
    # What decides whether a foot may be anchored:
    #   model             -- the gait's travel-derived support mask (shipped default)
    #   contact           -- the simulator's own contact report, replacing the mask
    #   contact_and_model -- the intersection: anchor only where BOTH agree
    # The mask misses 46-56% of the right foot's real floor contacts while claiming 16-28%
    # that are not touching (docs/mask-vs-contact-audit.md). Replacement anchors feet the
    # reference says are swinging, which fights the swing target; the intersection can only
    # remove anchors, never add them, so it tightens without inventing stance.
    payload.setdefault("anchor_source", "model")
    if payload["anchor_source"] not in ANCHOR_SOURCES:
        raise ValueError(
            f"anchor_source must be one of {sorted(ANCHOR_SOURCES)}, "
            f"got {payload['anchor_source']!r}"
        )
    # Standing height fraction for the contact-retargeted idle pose. 1.0 would be
    # a near-singular straight leg that bounces on its floor contact; anything
    # below ~0.97 reintroduces the crouch the standing fix removed.
    payload.setdefault("idle_stand_height_fraction", 0.988)
    fraction = payload["idle_stand_height_fraction"]
    if not np.isfinite(fraction) or not 0.9 < fraction <= 1.0:
        raise ValueError("idle_stand_height_fraction must be finite in (0.9, 1.0]")
    # Per-joint PhysX velocity clamp authored on every revolute joint. Normal
    # actions peak around 5-9 rad/s; the clamp only bounds the ballistic speeds
    # a drives-off fall reaches, which is what lets fall_damping_scale go to 0
    # (fully passive limbs) without entering the measured solver-NaN regime.
    payload.setdefault("joint_velocity_limit_rad_s", None)
    limit = payload["joint_velocity_limit_rad_s"]
    if limit is not None and (not np.isfinite(limit) or limit <= 0):
        raise ValueError("joint_velocity_limit_rad_s must be finite and positive when given")
    payload["controller"] = TeleopConfig(
        **{
            key: payload[key]
            for key in TeleopConfig.__dataclass_fields__
            if key in payload
        }
    )
    for key in ("rig", "assets", "scene", "root_assist"):
        payload[key] = (project_root / payload[key]).resolve()
    if "locomotion_root_assist" in payload:
        if not payload.get("contact_planner"):
            raise ValueError("locomotion_root_assist requires contact_planner")
        payload["locomotion_root_assist"] = (
            project_root / payload["locomotion_root_assist"]
        ).resolve()
    if "recovery_root_assist" in payload:
        payload["recovery_root_assist"] = (
            project_root / payload["recovery_root_assist"]
        ).resolve()
    if "recovery_modes" in payload:
        modes = payload["recovery_modes"]
        if not isinstance(modes, list) or not all(isinstance(mode, str) and mode for mode in modes):
            raise ValueError("recovery_modes must be a list of non-empty mode names")
        payload["recovery_modes"] = [str(mode) for mode in modes]
    specs = [*payload["gaits"].values(), payload["idle"]]
    if "crouch" in payload:
        specs.append(payload["crouch"])
    for key in ("bend", "sit"):
        if key in payload:
            specs.append(payload[key])
    for spec in specs:
        spec["file"] = (project_root / spec["file"]).resolve()
        if not np.isfinite([spec["start_s"], spec["duration_s"]]).all():
            raise ValueError("gait interval must be finite")
        if spec["start_s"] < 0 or spec["duration_s"] <= 0:
            raise ValueError("gait interval must have positive duration and nonnegative start")
        contacts = spec.get("contacts")
        if contacts is not None:
            if (
                not isinstance(contacts, list)
                or not contacts
                or not all(isinstance(name, str) and name for name in contacts)
            ):
                raise ValueError("posture contacts must be a nonempty list of link names")
            if len(set(contacts)) != len(contacts):
                raise ValueError("posture contacts must be unique link names")
        lift = spec.get("swing_lift_m")
        if lift is not None and (
            not np.isfinite(lift) or not 0 < lift <= SWING_LIFT_MAX_M
        ):
            raise ValueError(f"swing_lift_m must be finite in (0, {SWING_LIFT_MAX_M}]")
        plant = spec.get("stance_plant_speed_m_s")
        if plant is not None and (not np.isfinite(plant) or plant <= 0):
            raise ValueError("stance_plant_speed_m_s must be finite and positive")
        if plant is not None and abs(float(plant) - float(payload["speed_m_s"])) > 1e-9:
            raise ValueError(
                "stance_plant_speed_m_s must equal speed_m_s: planting holds the "
                "foot world-static against the root at exactly the commanded speed"
            )
    if set(payload["gaits"]) != {"forward", "backward"}:
        raise ValueError("keyboard mode requires forward and backward gaits")
    if "get_up" in payload:
        get_up = payload["get_up"]
        if not isinstance(get_up, dict) or not get_up.get("file"):
            raise ValueError("get_up must be a mapping with a file")
        get_up["file"] = (project_root / get_up["file"]).resolve()
    if "contact_planner" in payload:
        from .contact_gait import ContactGaitConfig

        planner = payload["contact_planner"]
        if not isinstance(planner, dict) or not planner:
            raise ValueError("contact_planner must be a nonempty settings mapping")
        planner_config = ContactGaitConfig(**planner)
        if payload.get("stance", {}).get("enabled", False):
            raise ValueError("contact_planner and legacy stance controller are mutually exclusive")
        for spec in payload["gaits"].values():
            cycle = spec.get("contact_cycle")
            if not isinstance(cycle, dict) or not cycle:
                raise ValueError("contact_planner requires contact_cycle on both gaits")
            if ContactGaitConfig(**cycle) != planner_config:
                raise ValueError("contact_cycle settings must match contact_planner")
    if np.shape(payload["spawn_xy"]) != (2,) or not np.isfinite(payload["spawn_xy"]).all():
        raise ValueError("spawn_xy must be a finite pair")
    for key in (
        "render_hz",
        "record_frames",
        "camera_distance_m",
        "slip_min_impulse_ns",
        "slip_speed_tolerance_m_s",
        "skin_penetration_tolerance_m",
    ):
        if not np.isfinite(payload[key]) or payload[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if int(payload["record_frames"]) != payload["record_frames"]:
        raise ValueError("record_frames must be an integer")
    friction = payload.get("human_friction")
    if friction is not None:
        if not np.isfinite(friction).all() or len(friction) != 2 or min(friction) < 0:
            raise ValueError("human_friction must be a finite (static, dynamic) pair >= 0")
        payload["human_friction"] = [float(friction[0]), float(friction[1])]
    planted = payload.get("max_stance_slip_m_s")
    if planted is not None and (not np.isfinite(planted) or planted <= 0):
        raise ValueError("max_stance_slip_m_s must be finite and positive")
    for key in ("heading_deg", "camera_elevation_deg", "camera_azimuth_deg"):
        if not np.isfinite(payload[key]):
            raise ValueError(f"{key} must be finite")
    if not np.isfinite(payload.get("camera_target_height_m", 0.8)):
        raise ValueError("camera_target_height_m must be finite")
    if not 0 < payload["camera_elevation_deg"] < 90:
        raise ValueError("camera elevation must be between 0 and 90 degrees")
    for segment in payload["demo"]:
        if not np.isfinite(segment["duration_s"]) or segment["duration_s"] <= 0:
            raise ValueError("demo durations must be finite and positive")
        if not isinstance(segment["keys"], list) or not set(segment["keys"]) <= KeyboardIntent.KEYS:
            raise ValueError("demo contains unsupported keys")
    return payload


class KeyboardIntent:
    """Held-key state; repeat events never accumulate extra velocity."""

    KEYS = frozenset(
        {
            "W", "S", "A", "D", "UP", "DOWN", "LEFT", "RIGHT", "SPACE",
            "R", "C", "V", "F", "B", "N", "G", "ESCAPE",
        }
    )
    ACTION_KEYS = {
        "C": "crouch",
        "V": "stand",
        "F": "fall",
        "B": "bend",
        "N": "sit",
        "G": "get_up",
    }

    def __init__(self) -> None:
        self.held: set[str] = set()
        self.reset_requested = False
        self.quit_requested = False
        self.action_requested: str | None = None

    def event(self, key: str, pressed: bool) -> None:
        if key not in self.KEYS:
            return
        if pressed:
            if key not in self.held:
                self.reset_requested |= key == "R"
                self.quit_requested |= key == "ESCAPE"
                if key in self.ACTION_KEYS:
                    self.action_requested = self.ACTION_KEYS[key]
            self.held.add(key)
        else:
            self.held.discard(key)

    def clear(self) -> None:
        self.held.clear()
        self.action_requested = None

    def command(self) -> tuple[float, float]:
        if "SPACE" in self.held:
            return 0.0, 0.0
        forward = int(bool(self.held & {"W", "UP"})) - int(bool(self.held & {"S", "DOWN"}))
        turn = int(bool(self.held & {"A", "LEFT"})) - int(bool(self.held & {"D", "RIGHT"}))
        return float(forward), float(turn)


@dataclass(frozen=True)
class Gait:
    joints: np.ndarray
    height_m: np.ndarray
    duration_s: float
    speed_m_s: float
    provenance: dict[str, Any]
    root_tilt: np.ndarray | None = None
    support_mask: np.ndarray | None = None
    # Per-side sanitized mocap swing arcs, set when the gait's contact cycle
    # requests swing_shape "amass"; consumed by the runtime contact planner.
    swing_shapes: dict[str, AmassSwingShape] | None = None
    # COM offset relative to the root origin per frame (contact-cycle gaits and
    # contact-fitted idle only); feeds the reference-COM feedforward.
    com_offset_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.joints.ndim != 2 or len(self.joints) < 2:
            raise ValueError("gait joints must be (T,D), T >= 2")
        if self.height_m.shape != (len(self.joints),):
            raise ValueError("gait heights must align with joint frames")
        if not np.isfinite(self.joints).all() or not np.isfinite(self.height_m).all():
            raise ValueError("gait samples must be finite")
        if not np.isfinite([self.duration_s, self.speed_m_s]).all():
            raise ValueError("gait timing and speed must be finite")
        if self.duration_s <= 0 or self.speed_m_s < 0:
            raise ValueError("gait duration must be positive and speed nonnegative")
        if self.root_tilt is not None and (
            self.root_tilt.shape != (len(self.joints), 3) or not np.isfinite(self.root_tilt).all()
        ):
            raise ValueError("root tilt must align with gait frames")
        if self.support_mask is not None and self.support_mask.shape != (len(self.joints), 2):
            raise ValueError("support mask must align with frames and left/right feet")

    def supporting_feet(self, phase: float) -> set[str] | None:
        if self.support_mask is None:
            return None
        row = min(int((phase % 1) * (len(self.joints) - 1)), len(self.joints) - 2)
        return {
            f"{side}_ankle" for i, side in enumerate(("left", "right")) if self.support_mask[row, i]
        }

    def swing_fraction(self, phase: float) -> dict[str, float]:
        """Progress within a cyclic swing run, independent of playback direction."""
        if self.support_mask is None:
            return {}
        count = len(self.joints) - 1
        index = min(int((phase % 1) * count), count - 1)
        result = {}
        for col, side in enumerate(("left", "right")):
            if self.support_mask[index, col]:
                continue
            before = after = 0
            while before < count and not self.support_mask[(index - before - 1) % count, col]:
                before += 1
            while after < count and not self.support_mask[(index + after + 1) % count, col]:
                after += 1
            result[f"{side}_ankle"] = (before + 0.5) / (before + after + 1)
        return result

    def tilt(self, phase: float) -> np.ndarray:
        if self.root_tilt is None:
            return np.zeros(3)
        phase %= 1.0
        point = phase * (len(self.joints) - 1)
        index = min(int(point), len(self.joints) - 2)
        fraction = point - index
        return (
            (1 - fraction) * self.root_tilt[index]
            + fraction * self.root_tilt[index + 1]
            - phase * (self.root_tilt[-1] - self.root_tilt[0])
        )

    def sample(self, phase: float) -> tuple[np.ndarray, float]:
        """A periodic reference with a distributed endpoint correction.

        The correction is recorded, since a cyclic control reference is derived
        from AMASS rather than an unchanged replay of the source sequence.
        """
        if not np.isfinite(phase):
            raise ValueError("gait phase must be finite")
        phase %= 1.0
        point = phase * (len(self.joints) - 1)
        index = min(int(point), len(self.joints) - 2)
        fraction = point - index
        q = (1 - fraction) * self.joints[index] + fraction * self.joints[index + 1]
        q -= phase * (self.joints[-1] - self.joints[0])
        h = (1 - fraction) * self.height_m[index] + fraction * self.height_m[index + 1]
        h -= phase * (self.height_m[-1] - self.height_m[0])
        return q, float(h)


def planted_support_mask(
    travel_support: np.ndarray, slip_m_s: np.ndarray, max_stance_slip_m_s: float
) -> np.ndarray:
    """Support frames whose foot the reference itself keeps still enough to pin.

    Root-travel-opposing ankle motion (the mask :func:`load_gait` derives) is not the
    same as a planted foot: a retargeted cycle can mark a foot as support while it is
    still sliding. A world-frozen anchor on such a foot is unreachable by construction,
    and a leg IK asked to hold it pays with joint angles the anatomy cannot carry.
    """

    if not np.isfinite(max_stance_slip_m_s) or max_stance_slip_m_s <= 0:
        raise ValueError("max_stance_slip_m_s must be finite and positive")
    if travel_support.shape != slip_m_s.shape:
        raise ValueError("slip and support arrays must align")
    return travel_support & (slip_m_s <= max_stance_slip_m_s)


SWING_LIFT_MAX_M = 0.15
SWING_LIFT_ENDPOINT_SLACK_DEG = 1.0
# Below this capsule-bottom height a grounded reference foot counts as stance
# when the support mask is derived from foot height (turn-in-place clips).
FOOT_HEIGHT_STANCE_M = 0.005
# Rise over the first 30% of a swing run, plateau, then drop in the last 20%.
# A plain sine peaks its descent speed exactly at touchdown (a 0.22 m/s slam
# kicked the whole body: backward sessions hit the 1500 N assist cap and threw
# 37 deg of error into a collar), and a sin^2 arc is too low through the middle,
# where the foot is dragged if it touches. Smoothstep has zero slope at both
# ends. The late drop also shortens the landing skid: while the reference foot
# is still airborne it moves forward relative to the root at ~v_cmd, so every
# airborne millimetre before touchdown is a millimetre of landing at speed --
# get the foot down and the reference itself pins it (stance-relative speed
# cancels the root).
SWING_RISE_FRACTION = 0.30
SWING_FALL_START = 0.80


def _swing_lift_profile(fraction: float) -> float:
    def smoothstep(x: float) -> float:
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)

    rise = smoothstep(fraction / SWING_RISE_FRACTION)
    fall = smoothstep((fraction - SWING_FALL_START) / (1.0 - SWING_FALL_START))
    return rise * (1.0 - fall)
SIDE_COLUMNS = (("left", 0), ("right", 1))


def bake_swing_clearance(
    gait: Gait,
    plan: HumanRigPlan,
    *,
    clearance_m: float,
    max_correction_rad: float = 0.5,
    iterations: int = 60,
    stance_speed_m_s: float | None = None,
    stance_direction: float = 1.0,
    geometry_mask: np.ndarray | None = None,
) -> Gait:
    """Raise swing feet that skim the floor, inside the reference itself.

    Measured on the shipped forward cycle, the retargeted swing foot's capsule
    bottom lifts to a median of only 15-32 mm and spends up to 25 of 158 phase
    frames below 10 mm -- while its forward speed in playback time is ~0.4 m/s
    relative to the root, i.e. ~0.8 m/s over the world. Any floor contact in
    those frames is a forced scrape: the contact frames' target-side foot speed
    has p50 0.40 m/s and p95 0.97 m/s, which dominates the measured slip
    (artifacts/humans/limp_ab/slip_attribution.json), because root fluctuation
    (p50 0.02) and joint tracking (p50 0.03) are an order of magnitude smaller.
    The runtime stance controller cannot compensate: its IK re-derives from the
    fresh reference every step, so its per-step correction budget (0.087 rad)
    never accumulates into the ~20 deg a real 60 mm lift needs.

    So lift the foot in the reference, where the whole chain (targets, ramp
    limiter, PD) sees one consistent motion. Per swing frame the target bottom
    height follows :func:`_swing_lift_profile`: zero height and zero slope at
    touchdown/liftoff (soft landing), full clearance through the middle (no
    scrape window). Loop closure is re-measured and any worsening beyond
    ``SWING_LIFT_ENDPOINT_SLACK_DEG`` is an error. The solve
    is the weighted joint-space least-norm step on that leg's DOFs, bounded per
    frame so one low frame cannot yank a leg; sideways-efficient axes pay the
    same abduction penalty the stance controller uses, chosen by FK probe.
    """

    if not np.isfinite(clearance_m) or not 0 <= clearance_m <= SWING_LIFT_MAX_M:
        # 0 is legitimate: plant the stance feet without adding any swing lift.
        raise ValueError(f"clearance_m must be finite in [0, {SWING_LIFT_MAX_M}]")
    # The run structure and planting follow the *geometry* mask (full stance/swing
    # alternation from foot travel); the runtime anchor mask (planted subset) is a
    # runtime concern and may fragment runs with gaps.
    mask = geometry_mask if geometry_mask is not None else gait.support_mask
    if mask is None:
        raise ValueError("bake_swing_clearance needs a support mask (swing = not support)")
    names = list(plan.dof_names)
    count = len(gait.joints)
    # Fraction of the way through each foot's current swing run, circular so the
    # profile is continuous across the cycle seam, plus the run length (needed to
    # find each frame's descent-start landmark).
    fraction = np.zeros((count, 2), dtype=np.float64)
    run_length = np.zeros((count, 2), dtype=np.int64)
    for column in range(2):
        swing = ~mask[:, column]
        for index in range(count):
            if not swing[index]:
                continue
            before = 0
            while swing[(index - before - 1) % count]:
                before += 1
            after = 0
            while swing[(index + after + 1) % count]:
                after += 1
            run_length[index, column] = before + after + 1
            fraction[index, column] = (before + 0.5) / run_length[index, column]
    # Horizontal foot planting, chained around the cycle. Per foot, per stance
    # run: the foot's body-frame x moves backwards at exactly the commanded root
    # speed (world-static foot -- the source clips' stance feet slide, and the
    # support A/B showed that slide is transmitted as drag once the feet are
    # pressed onto the floor). Per swing run: a minimum-jerk return from the
    # stance-end position to the next touchdown, so the landing velocity is zero
    # in both axes (a foot arriving at up to double the walking speed skids for
    # ~0.3 s every step -- the measured steady-slip tail).
    planted_x = np.full((count, 2), np.nan)
    swing_x = np.full((count, 2), np.nan)
    if stance_speed_m_s is not None and stance_speed_m_s > 0:
        cycle_wall_s = gait.speed_m_s * gait.duration_s / stance_speed_m_s
        wall_per_frame = cycle_wall_s / max(count - 1, 1)
        for column, (side, _c) in enumerate(SIDE_COLUMNS):
            stance = mask[:, column]

            def x_at(frame: int, side: str = side) -> float:
                q0, h0 = gait.sample((frame % count) / (count - 1))
                return _foot_fore_aft(
                    plan, names, q0, h0, gait.tilt((frame % count) / (count - 1)), side
                )

            # Walk the circular run structure; each stance run is planted from its
            # touchdown x, and the swing that follows returns minimum-jerk to the
            # SAME touchdown x (periodic gait: the foot lands at the same
            # body-frame x every cycle while the root advances underneath).
            start = next(
                (i for i in range(count) if stance[i] and not stance[(i - 1) % count]), None
            )
            if start is not None:
                x_land = x_at(start)
                index = 0
                while index < count:
                    # --- stance run: plant world-static ---
                    stance_len = 0
                    while (
                        index + stance_len < count
                        and stance[(start + index + stance_len) % count]
                    ):
                        stance_len += 1
                    for step in range(stance_len):
                        planted_x[(start + index + step) % count, column] = x_land - (
                            stance_direction
                            * stance_speed_m_s
                            * (index + step)
                            * wall_per_frame
                        )
                    index += stance_len
                    # --- swing run: minimum-jerk return to the same landing x ---
                    swing_len = 0
                    while (
                        index + swing_len < count
                        and not stance[(start + index + swing_len) % count]
                    ):
                        swing_len += 1
                    x_start_swing = x_land - (
                        stance_direction * stance_speed_m_s * stance_len * wall_per_frame
                    )
                    delta_x = stance_direction * stance_speed_m_s * stance_len * wall_per_frame
                    for step in range(swing_len):
                        u = (step + 1) / max(swing_len, 1)
                        smooth = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                        swing_x[(start + index + step) % count, column] = (
                            x_start_swing + delta_x * smooth
                        )
                    index += swing_len

    endpoints_before = float(np.rad2deg(np.abs(gait.joints[-1] - gait.joints[0])).max())
    joints = gait.joints.copy()
    lifted = 0
    un_penetrated = 0
    flight_grounded = 0
    planted = 0
    grounded_this_frame = ""
    for index in range(count):
        phase = index / (count - 1)
        q, height = gait.sample(phase)
        q_reference = q.copy()
        tilt = gait.tilt(phase)
        bottoms = {side: _foot_bottom(plan, names, q, height, tilt, side)
                   for side, _column in SIDE_COLUMNS}
        if not mask[index].any() and all(
            bottom > FOOT_HEIGHT_STANCE_M for bottom in bottoms.values()
        ):
            # Only intervene when the phase schedule has no support foot. A planned
            # stance foot is grounded below; pinning the lower SWING foot first
            # would skip its lift and turn it into an unintended sliding contact.
            # Both feet airborne is a hop, not a step (quick turn clips are full of
            # them: CMU/83_56's 180 deg turn flies 0.5 s). A held A/D turn must
            # read as stepping, so pin the *lower* foot back to the floor and let
            # the higher one carry the swing.
            lower = min(bottoms, key=lambda side: bottoms[side])
            grounded_this_frame = lower
            if bottoms[lower] > 0.01:
                q = _move_foot(
                    plan,
                    q,
                    lower,
                    target_bottom=0.0,
                    root_height=height,
                    root_tilt=tilt,
                    max_correction_rad=max_correction_rad,
                    iterations=iterations,
                )
                flight_grounded += 1
                bottoms[lower] = 0.0
        else:
            grounded_this_frame = ""
        for side, column in SIDE_COLUMNS:
            if mask[index, column]:
                bottom = _foot_bottom(plan, names, q, height, tilt, side)
                plant_x = planted_x[index, column]
                if np.isfinite(plant_x):
                    # Stance planting: hold the foot's fore-aft position at the
                    # world-static trajectory and on the floor. Without this the
                    # source's own stance slide is transmitted as drag the moment
                    # the feet are pressed onto the floor (support A/B).
                    current_x = _foot_fore_aft(plan, names, q, height, tilt, side)
                    if abs(current_x - float(plant_x)) > 1e-3 or bottom < 0.0:
                        q = _move_foot(
                            plan,
                            q,
                            side,
                            target_bottom=min(0.0, bottom),
                            root_height=height,
                            root_tilt=tilt,
                            max_correction_rad=max_correction_rad,
                            iterations=iterations,
                            target_fore_aft=float(plant_x),
                        )
                        planted += 1
                    continue
                # Stance frames may reference *below* the floor (retargeting roots
                # the whole body off one point; the other leg's geometry then dips
                # a foot tens of millimetres under). The floor will refuse that
                # anyway, and the PD spending its error budget against the floor is
                # what shakes the whole body -- remove the penetration here.
                if abs(bottom) > 1e-3:
                    # Plant the stance foot on the floor: pull DOWN the frames the
                    # reference holds above it (+8 to +41 mm -- the source's own
                    # foot rolling never reaches the ground after grounding, which
                    # anchored the whole body off frame 0 only) as well as pushing
                    # up the frames below it. Without this the foot floats at its
                    # tracking error and the visible foot never touches.
                    q = _move_foot(
                        plan,
                        q,
                        side,
                        target_bottom=0.0,
                        root_height=height,
                        root_tilt=tilt,
                        max_correction_rad=max_correction_rad,
                        iterations=iterations,
                    )
                    un_penetrated += 1
                continue
            if side == grounded_this_frame:
                # Just pinned by the no-flight rule; the swing lift must not undo it.
                continue
            f = fraction[index, column]
            target = clearance_m * _swing_lift_profile(f)
            # NOTE (measured, rejected): *abruptly* freezing the foot's fore-aft
            # position over the descent doubled the forward session's joint error
            # (3.91 -> 8.21 deg). What the steady-slip tail needed instead is the
            # smooth minimum-jerk return (swing_x below): the foot still reaches
            # zero world velocity at touchdown, but through a smooth lag instead
            # of a per-step fight with the reference.
            target_x = None
            if np.isfinite(swing_x[index, column]):
                target_x = float(swing_x[index, column])
            bottom = _foot_bottom(plan, names, q, height, tilt, side)
            if bottom >= target - 1e-4 and target_x is None:
                continue
            q = _move_foot(
                plan,
                q,
                side,
                target_bottom=max(target, bottom) if target_x is None else target,
                root_height=height,
                root_tilt=tilt,
                max_correction_rad=max_correction_rad,
                iterations=iterations,
                target_fore_aft=target_x,
            )
            lifted += 1
        # One write-back per frame covering BOTH feet' corrections: add the total
        # correction delta to the stored row. Writing the sampled pose back would
        # bake the endpoint ramp into the array on top of the ramp Gait.sample
        # applies again at run time; writing per-side dropped the other foot's
        # correction and double-support frames lost every stance fix.
        joints[index] = gait.joints[index] + (q - q_reference)
    if lifted:
        # Frames 0 and -1 are the same instant of the periodic reference. If the
        # window cut lands mid-swing, both ends of the same circular swing run get
        # lifted from poses that differ by the source's endpoint residual, and
        # unequal deltas would grow the endpoint difference -- feeding the very
        # ramp distortion that caused the original limp. One shared delta keeps
        # the endpoint difference exactly unchanged and the seam continuous.
        delta_first = joints[0] - gait.joints[0]
        delta_last = joints[-1] - gait.joints[-1]
        if (delta_first - delta_last).any():
            shared = (delta_first + delta_last) / 2.0
            joints[0] = gait.joints[0] + shared
            joints[-1] = gait.joints[-1] + shared
    endpoints_after = float(np.rad2deg(np.abs(joints[-1] - joints[0])).max())
    if endpoints_after > endpoints_before + SWING_LIFT_ENDPOINT_SLACK_DEG:
        raise ValueError(
            f"swing lift broke loop closure: endpoint difference "
            f"{endpoints_before:.2f} -> {endpoints_after:.2f} deg"
        )
    object.__setattr__(gait, "joints", joints)
    gait.provenance["swing_lift_m"] = float(clearance_m)
    gait.provenance["swing_lift_frames"] = int(lifted)
    gait.provenance["flight_frames_grounded"] = int(flight_grounded)
    gait.provenance["stance_plant_frames"] = int(planted)
    gait.provenance["stance_penetration_frames"] = int(un_penetrated)
    gait.provenance["endpoint_joint_difference_deg_after_lift"] = round(endpoints_after, 3)
    return gait


def _foot_bottom(
    plan: HumanRigPlan,
    names: list[str],
    q: np.ndarray,
    root_height: float,
    root_tilt: np.ndarray,
    side: str,
) -> float:
    poses = forward_kinematics(
        plan,
        dict(zip(names, q, strict=True)),
        root_position=(0.0, 0.0, root_height),
        root_rotation=root_tilt,
    )
    return capsule_bottom(plan, poses, f"{side}_ankle")


def _foot_fore_aft(
    plan: HumanRigPlan,
    names: list[str],
    q: np.ndarray,
    root_height: float,
    root_tilt: np.ndarray,
    side: str,
) -> float:
    poses = forward_kinematics(
        plan,
        dict(zip(names, q, strict=True)),
        root_position=(0.0, 0.0, root_height),
        root_rotation=root_tilt,
    )
    return float(poses[f"{side}_ankle"].translation[0])


def _move_capsule(
    plan: HumanRigPlan,
    joints: np.ndarray,
    link: str,
    chain_links: set[str],
    *,
    target_bottom: float,
    root_height: float,
    root_tilt: np.ndarray,
    max_correction_rad: float,
    iterations: int,
    target_fore_aft: float | None = None,
    penalised: frozenset[str] = frozenset(),
) -> np.ndarray:
    """Bounded weighted joint-space least-norm IK moving one capsule's bottom.

    Tasks: bring the named link's capsule bottom to ``target_bottom`` and, when
    ``target_fore_aft`` is given, hold the link's body-frame fore-aft coordinate
    there. The solve runs on the DOFs whose ``chain_joint`` is in
    ``chain_links`` (e.g. the hip/knee/ankle chain of one leg, or the
    shoulder/elbow/wrist chain of one arm), clipped to the rig limits and the
    per-call correction budget. ``penalised`` names DOFs that pay extra per
    radian so the correction routes through flexion instead of splay.
    """

    indices = [i for i, joint in enumerate(plan.joints) if joint.chain_joint in chain_links]
    if not indices:
        raise ValueError(f"no DOFs found for chain {sorted(chain_links)}")
    original = joints.copy()
    lower = np.deg2rad([j.lower_deg for j in plan.joints])
    upper = np.deg2rad([j.upper_deg for j in plan.joints])
    weights = np.where(
        [plan.joints[i].name in penalised for i in indices], 1000.0, 1.0
    )

    def task_of(values: np.ndarray) -> np.ndarray:
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, values, strict=True)),
            root_position=(0.0, 0.0, root_height),
            root_rotation=root_tilt,
        )
        bottom = capsule_bottom(plan, poses, link)
        if target_fore_aft is None:
            return np.array([bottom])
        # Root sits at (0, 0, h) with no yaw, so the link's world x *is* the
        # body-frame fore-aft coordinate.
        return np.array([bottom, float(poses[link].translation[0])])

    q = joints.copy()
    for _ in range(iterations):
        task = task_of(q)
        error = np.array([target_bottom - task[0]])
        if target_fore_aft is not None:
            error = np.array([target_bottom - task[0], target_fore_aft - task[1]])
        if np.all(np.abs(error) <= 1e-4):
            break
        step = 0.02
        jacobian = np.zeros((len(error), len(indices)))
        for column, index in enumerate(indices):
            probe = q.copy()
            probe[index] += step
            jacobian[:, column] = (task_of(probe) - task) / step
        # Least-norm with W = diag(weights): paying w times more per radian routes
        # the correction through flexion, not through the sideways splay that caused
        # the knee-abduction defect (docs/progress.md 2026-09-24).
        jw = jacobian / weights
        delta = jw.T @ np.linalg.solve(jw @ jw.T + 1e-9 * np.eye(len(error)), error)
        q[indices] = np.clip(
            q[indices] + delta,
            np.maximum(lower[indices], original[indices] - max_correction_rad),
            np.minimum(upper[indices], original[indices] + max_correction_rad),
        )
    return q


def _move_foot(
    plan: HumanRigPlan,
    joints: np.ndarray,
    side: str,
    *,
    target_bottom: float,
    root_height: float,
    root_tilt: np.ndarray,
    max_correction_rad: float,
    iterations: int,
    target_fore_aft: float | None = None,
) -> np.ndarray:
    """Bounded weighted joint-space least-norm IK moving one foot's pose."""

    return _move_capsule(
        plan,
        joints,
        f"{side}_ankle",
        {f"{side}_hip", f"{side}_knee", f"{side}_ankle"},
        target_bottom=target_bottom,
        root_height=root_height,
        root_tilt=root_tilt,
        max_correction_rad=max_correction_rad,
        iterations=iterations,
        target_fore_aft=target_fore_aft,
        penalised=frozenset(sideways_leg_dofs(plan)),
    )


def load_gait(
    spec: dict[str, Any],
    plan: HumanRigPlan,
    *,
    dt_s: float,
    max_stance_slip_m_s: float | None = None,
) -> Gait:
    clip = normalize_root_motion(
        crop_amass_clip(
            load_amass_clip(spec["file"]), start_s=spec["start_s"], duration_s=spec["duration_s"]
        )
    )
    clip = ground_amass_clip(clip, plan, support_z_m=0.0).resample(1 / dt_s, method="slerp")
    joints = np.stack(
        [joint_values_from_clip(clip, frame, plan)[0] for frame in range(clip.frame_count)]
    )
    lower = np.deg2rad([j.lower_deg for j in plan.joints])
    upper = np.deg2rad([j.upper_deg for j in plan.joints])
    if np.any(joints < lower - 1e-6) or np.any(joints > upper + 1e-6):
        raise ValueError(f"gait exceeds rig limits: {spec['file']}")
    duration = float(clip.times_s[-1])
    speed = np.linalg.norm(clip.root_translation[-1, :2] - clip.root_translation[0, :2]) / duration
    tilt = []
    for vector in clip.root_rotation:
        rotation = axis_angle_to_matrix(vector)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        tilt.append(matrix_to_axis_angle(rotation_about_axis("z", -yaw) @ rotation))
    gait = Gait(
        joints,
        plan.spawn_root_position[2] + clip.root_translation[:, 2],
        duration,
        float(speed),
        {
            "source": clip.provenance.as_dict(),
            "metadata": dict(clip.metadata),
            "loop_correction": "linear_endpoint_drift",
            "endpoint_joint_difference_deg": float(
                np.rad2deg(np.abs(joints[-1] - joints[0])).max()
            ),
        },
        np.asarray(tilt),
    )
    # Total yaw the reference sweeps (used by the controller to pace stepped
    # turns and by audits to compare a clip's facing against its displacement).
    basis = AMASS_BODY_FRAME.basis()

    def _facing_yaw(vector: np.ndarray) -> float:
        facing = axis_angle_to_matrix(vector) @ basis @ np.array([0.0, 0.0, 1.0])
        return float(np.arctan2(facing[1], facing[0]))

    yaw = [_facing_yaw(vector) for vector in clip.root_rotation]
    unwrapped = np.unwrap(np.asarray(yaw))
    gait.provenance["source_yaw_total_deg"] = round(
        float(abs(unwrapped[-1] - unwrapped[0]) * 180.0 / np.pi), 2
    )
    gait.provenance["source_yaw_sign"] = float(np.sign(unwrapped[-1] - unwrapped[0]) or 1.0)
    stance_speed = None
    if spec.get("support_mask_from") == "foot_height":
        # A turn-in-place clip barely translates its root, so "foot moves opposite
        # to root travel" is meaningless; stance is simply where the grounded
        # reference keeps a foot on the floor.
        bottoms = []
        for phase in np.linspace(0, 1, len(joints), endpoint=True):
            q, h = gait.sample(float(phase))
            poses = forward_kinematics(
                plan,
                dict(zip(plan.dof_names, q, strict=True)),
                root_position=(0, 0, h),
                root_rotation=gait.tilt(float(phase)),
            )
            bottoms.append(
                [capsule_bottom(plan, poses, f"{side}_ankle") for side in ("left", "right")]
            )
        travel_support = np.asarray(bottoms) < FOOT_HEIGHT_STANCE_M
        object.__setattr__(gait, "support_mask", travel_support)
        gait.provenance["support_phase_method"] = "reference_foot_bottom_below_5mm"
    else:
        # Infer stance from the derived cycle's foot travel, not absolute height:
        # retargeting can put a swinging foot below the floor. Relative motion must
        # oppose root travel in stance. The mask releases at every swing boundary.
        ankle_x = []
        for phase in np.linspace(0, 1, len(joints), endpoint=True):
            q, h = gait.sample(float(phase))
            poses = forward_kinematics(
                plan,
                dict(zip(plan.dof_names, q, strict=True)),
                root_position=(0, 0, h),
                root_rotation=gait.tilt(float(phase)),
            )
            ankle_x.append([poses[f"{side}_ankle"].translation[0] for side in ("left", "right")])
        direction = np.sign(clip.root_translation[-1, 0] - clip.root_translation[0, 0])
        derivative = np.gradient(np.asarray(ankle_x), axis=0)
        travel_support = derivative * direction < 0
        object.__setattr__(gait, "support_mask", travel_support)
        gait.provenance["support_phase_method"] = "derived_cycle_ankle_travel_opposes_root"
        gait.provenance["source_forward_sign"] = float(direction)
        stance_speed = -derivative * direction / (duration / (len(joints) - 1))
    if stance_speed is not None:
        if spec.get("cadence_from_retargeted_feet", False) and speed > 0:
            estimate = float(np.median(stance_speed[travel_support]))
            if not np.isfinite(estimate) or estimate <= 0:
                raise ValueError("could not estimate retargeted stance cadence")
            gait.provenance["raw_source_speed_m_s"] = float(speed)
            gait.provenance["cadence_speed_m_s"] = estimate
            object.__setattr__(gait, "speed_m_s", estimate)
        # An anchor may only be set on a foot the reference itself keeps still.
        if max_stance_slip_m_s is not None:
            slip_m_s = np.abs(stance_speed - gait.speed_m_s)
            planted = planted_support_mask(travel_support, slip_m_s, max_stance_slip_m_s)
            object.__setattr__(gait, "support_mask", planted)
            gait.provenance["stance_slip_tolerance_m_s"] = float(max_stance_slip_m_s)
            gait.provenance["support_frames_from_travel"] = int(travel_support.sum())
            gait.provenance["support_frames"] = int(planted.sum())
            gait.provenance["support_frames_without_anchors"] = bool(not planted.any())
            gait.provenance["travel_support_slip_m_s_p50"] = (
                None if not travel_support.any() else float(np.median(slip_m_s[travel_support]))
            )
    # The runtime stance controller treats everything the planted mask does not
    # claim as swing, so scrape-proof exactly that set: a swing foot that skims
    # the floor is dragged at ~0.8 m/s wherever it touches.
    lift = spec.get("swing_lift_m")
    plant_speed = spec.get("stance_plant_speed_m_s")
    if (lift or plant_speed) and not spec.get("contact_cycle"):
        gait = bake_swing_clearance(
            gait,
            plan,
            clearance_m=float(lift) if lift else 0.0,
            stance_speed_m_s=float(plant_speed) if plant_speed else None,
            stance_direction=float(gait.provenance.get("source_forward_sign", 1.0)),
            geometry_mask=travel_support,
        )
    if spec.get("contact_cycle"):
        from .contact_gait import ContactGaitConfig, bake_contact_cycle

        cycle_config = ContactGaitConfig(**spec["contact_cycle"])
        if cycle_config.swing_shape == "amass":
            from .contact_gait import extract_swing_shapes

            # Extraction must see the raw pre-bake joints and derive the same
            # cycle schedule bake_contact_cycle uses, hence the placement here.
            object.__setattr__(
                gait, "swing_shapes", extract_swing_shapes(clip, gait, plan, cycle_config)
            )
        source_height_m = None
        source_height_scale = 1.0
        if cycle_config.root_bob:
            # Deviations from the reachable cap scale with the commanded/source
            # stride ratio, the same proportionality the swing arcs follow.
            source_travel = gait.speed_m_s * gait.duration_s
            if not np.isfinite(source_travel) or source_travel < 1e-6:
                raise ValueError("root bob needs a translating source clip")
            source_height_m = plan.spawn_root_position[2] + clip.root_translation[:, 2]
            source_height_scale = cycle_config.cycle_distance_m / source_travel
        gait = bake_contact_cycle(gait, plan, cycle_config, source_height_m=source_height_m,
                                  source_height_scale=source_height_scale)
    scale = spec.get("turn_playback_scale")
    if scale is not None:
        if not np.isfinite(scale) or not 0 < scale <= 2.0:
            raise ValueError("turn_playback_scale must be finite in (0, 2]")
        gait.provenance["turn_playback_scale"] = float(scale)
    return gait


@dataclass(frozen=True)
class TeleopTarget:
    joints: np.ndarray
    joint_velocities: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    mode: str


class TeleopController:
    """Smooth keyboard commands; bound positional windup against measured state."""

    def __init__(
        self,
        config: TeleopConfig,
        gaits: dict[str, Gait],
        idle: Gait,
        plan: HumanRigPlan,
        heading_rad: float,
    ) -> None:
        self.config, self.gaits, self.idle, self.plan = config, gaits, idle, plan
        if set(gaits) != {"forward", "backward"} or any(g.speed_m_s <= 0 for g in gaits.values()):
            raise ValueError("moving gaits require nonzero measured speed")
        if any(g.joints.shape[1] != len(plan.joints) for g in (*gaits.values(), idle)):
            raise ValueError("gait DOF count must match rig")
        self.lower = np.deg2rad([j.lower_deg for j in plan.joints])
        self.upper = np.deg2rad([j.upper_deg for j in plan.joints])
        self.reset(np.asarray(plan.spawn_root_position), heading_rad)

    def reset(self, position: np.ndarray, heading_rad: float) -> None:
        if (
            np.shape(position) != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(heading_rad)
        ):
            raise ValueError("reset requires finite position and heading")
        self.position = np.asarray(position, dtype=float).copy()
        self.position[2] = self.idle.height_m[0] + self.config.root_z_offset_m
        self.heading = float(heading_rad)
        self.phase = self.speed = self.turn_rate = self.weight = 0.0
        self.mode = "forward"
        self.joints = self.idle.joints[0].copy()
        self.tilt = self.idle.tilt(0)
        self.rotation = rotation_about_axis("z", self.heading) @ axis_angle_to_matrix(self.tilt)

    def advance(
        self,
        command: tuple[float, float],
        dt_s: float,
        measured_position: np.ndarray,
        measured_heading: float,
    ) -> TeleopTarget:
        if (
            not np.isfinite(dt_s)
            or dt_s <= 0
            or np.shape(command) != (2,)
            or not np.isfinite(command).all()
        ):
            raise ValueError("command and positive step must be finite")
        if np.shape(measured_position) != (3,) or not np.isfinite(measured_position).all():
            raise ValueError("measured position must be a finite vector")
        if not np.isfinite(measured_heading) or any(abs(v) > 1 for v in command):
            raise ValueError("invalid heading or command outside [-1,1]")
        c = self.config
        self.speed += float(
            np.clip(
                command[0] * c.speed_m_s - self.speed,
                -c.acceleration_m_s2 * dt_s,
                c.acceleration_m_s2 * dt_s,
            )
        )
        # A/D turns only while walking (W/S held). At a standstill the turn command
        # is ignored: the user explicitly disabled stationary turning after the
        # pivoting and marching variants both failed acceptance.
        turn_command = command[1] if abs(command[0]) > 0 else 0.0
        turn_accel = np.deg2rad(c.turn_acceleration_deg_s2) * dt_s
        self.turn_rate += float(
            np.clip(
                turn_command * np.deg2rad(c.turn_speed_deg_s) - self.turn_rate,
                -turn_accel,
                turn_accel,
            )
        )
        self.heading += self.turn_rate * dt_s
        heading_error = (self.heading - measured_heading + np.pi) % (2 * np.pi) - np.pi
        self.heading = measured_heading + float(
            np.clip(
                heading_error,
                -np.deg2rad(c.max_heading_lead_deg),
                np.deg2rad(c.max_heading_lead_deg),
            )
        )
        velocity = self.speed * np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        self.position[:2] += velocity[:2] * dt_s
        offset = self.position[:2] - measured_position[:2]
        self.position[:2] = measured_position[:2] + offset * min(
            1.0, c.max_target_lead_m / max(float(np.linalg.norm(offset)), 1e-12)
        )
        self.mode = (
            "backward" if self.speed < -1e-4 else "forward" if self.speed > 1e-4 else self.mode
        )
        gait = self.gaits[self.mode]
        source_sign = gait.provenance.get(
            "source_forward_sign", 1 if self.mode == "forward" else -1
        )
        playback_sign = (1 if self.mode == "forward" else -1) * source_sign
        self.phase = (
            self.phase
            + playback_sign * abs(self.speed) * dt_s / max(gait.speed_m_s * gait.duration_s, 1e-6)
        ) % 1
        # Exponential, not rate-clipped-linear, approach. A linear ramp ends by
        # stepping its rate to zero in one frame; stopping from a walk then hands
        # the body the blend's full upward velocity (~0.10 m/s) as momentum, which
        # the standing leg column cannot absorb -- measured as a 66 ms support
        # loss and a 16 deg hip error at the idle stand height. The exponential
        # tail decays the blend velocity smoothly instead.
        self.weight += (
            abs(self.speed) / c.speed_m_s - self.weight
        ) * min(1.0, dt_s / c.transition_s)
        gait_q, gait_height = gait.sample(self.phase)
        desired = np.clip(
            (1 - self.weight) * self.idle.joints[0] + self.weight * gait_q, self.lower, self.upper
        )
        delta = np.clip(
            desired - self.joints, -c.max_joint_speed_rad_s * dt_s, c.max_joint_speed_rad_s * dt_s
        )
        self.joints += delta
        height = (1 - self.weight) * self.idle.height_m[0] + self.weight * gait_height
        height += self.config.root_z_offset_m
        velocity[2] = (height - self.position[2]) / dt_s
        self.position[2] = height
        desired_tilt = (1 - self.weight) * self.idle.tilt(0) + self.weight * gait.tilt(self.phase)
        self.tilt += min(1.0, dt_s / c.transition_s) * (desired_tilt - self.tilt)
        rotation = rotation_about_axis("z", self.heading) @ axis_angle_to_matrix(self.tilt)
        angular_velocity = matrix_to_axis_angle(rotation @ self.rotation.T) / dt_s
        self.rotation = rotation
        quaternion = axis_angle_to_quaternion(matrix_to_axis_angle(rotation))
        return TeleopTarget(
            self.joints.copy(),
            delta / dt_s,
            self.position.copy(),
            quaternion,
            velocity,
            angular_velocity,
            # Report stand as soon as the root has stopped, even while the joint
            # blend is still settling -- otherwise a released turn reads as the
            # stale locomotion mode it never actually walked.
            self.mode if abs(self.speed) > 1e-4 else "stand",
        )
