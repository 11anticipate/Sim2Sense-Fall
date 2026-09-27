"""Contact-consistent periodic leg targets derived from an AMASS upper-body cycle.

This is trajectory retargeting, not a physical contact constraint. Actual contact,
slip and joint tracking must still pass the simulator gates independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .contact_control import capsule_bottom
from .rig import HumanRigPlan, LinkTransform, configuration_com, forward_kinematics
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle

if TYPE_CHECKING:
    from .motion import MotionClip
    from .teleop import Gait


@dataclass(frozen=True)
class ContactGaitConfig:
    cycle_distance_m: float = 0.6
    stance_fraction: float = 0.62
    clearance_m: float = 0.06
    lift_ramp_fraction: float = 0.3
    reach_fraction: float = 0.97
    orientation_weight_m: float = 0.15
    damping: float = 1e-6
    iterations: int = 60
    max_step_rad: float = 0.15
    # "quintic" keeps the analytic swing arc; "amass" retargets the source clip's
    # own swing arc (vertical profile, lateral bow, sagittal foot pitch) onto the
    # commanded stride. clearance_m and lift_ramp_fraction are unused by "amass".
    swing_shape: str = "quintic"
    # Follow the source clip's root height trajectory (bounded by leg reach)
    # instead of pinning the root at the constant stride-limited cap.
    root_bob: bool = False

    def __post_init__(self) -> None:
        numeric = [
            value for value in vars(self).values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not np.isfinite(numeric).all():
            raise ValueError("contact gait settings must be finite")
        if self.swing_shape not in {"quintic", "amass"}:
            raise ValueError("swing_shape must be 'quintic' or 'amass'")
        if not isinstance(self.root_bob, bool):
            raise ValueError("root_bob must be boolean")
        if not .5 < self.stance_fraction < .8 or not .8 < self.reach_fraction < 1:
            raise ValueError("invalid stance or leg reach fraction")
        if not 0 < self.lift_ramp_fraction <= .5:
            raise ValueError("lift ramp must be in (0, .5]")
        if (min(self.cycle_distance_m, self.clearance_m, self.orientation_weight_m,
                self.damping, self.max_step_rad) <= 0 or isinstance(self.iterations, bool)
                or not isinstance(self.iterations, int) or self.iterations < 1):
            raise ValueError("contact gait lengths and solver settings must be positive")


def foot_path(phase: float, config: ContactGaitConfig) -> tuple[float, float, bool]:
    """Body-frame x and ground clearance with zero WORLD velocity at both contacts.

    A quintic world-space swing has zero endpoint velocity/acceleration. Subtracting
    root travel gives body-frame endpoint slope -cycle_distance, matching stance.
    A body-frame minimum-jerk curve alone incorrectly lands at root speed.
    """
    p = phase % 1.
    distance, support = config.cycle_distance_m, config.stance_fraction
    front = .5 * distance * support
    if p <= support:
        return front - distance * p, 0., True
    u = (p - support) / (1 - support)
    smooth = u**3 * (10 - 15*u + 6*u*u)
    x = -front + distance * smooth - distance * (p-support)
    # Lift before most horizontal travel and land after it. A bell-shaped arc
    # returns close to the floor while horizontal speed is still high, so finite
    # tracking lag turns its descent into a scrape.
    ramp = min(1., u/config.lift_ramp_fraction, (1-u)/config.lift_ramp_fraction)
    lift = config.clearance_m * ramp**3 * (10 - 15*ramp + 6*ramp*ramp)
    return x, lift, False


def _quintic(u: float) -> float:
    return u * u * u * (10. - 15. * u + 6. * u * u)


@dataclass(frozen=True)
class AmassSwingShape:
    """One sanitized swing arc in the source clip's own start-anchored frame.

    ``along_m``/``cross_m``/``height_m``/``pitch_rad`` are sampled over ``grid``
    (source progress within the swing run). The arc keeps the mocap's vertical
    profile, lateral bow and sagittal foot pitch; endpoints are re-based to zero
    height and pitch because stance anchors are grounded and flat. Retargeting to
    a commanded stride happens in :meth:`sample` -- horizontal displacement is
    scaled and yawed onto the actual (start, end) pair, vertical arc is kept
    as-is, and a raised-cosine time warp makes world velocity and angular
    velocity exactly zero at both contacts (the source itself lands at up to
    1.2 m/s measured, which is the landing skid the analytic arc removes).
    """

    grid: np.ndarray
    along_m: np.ndarray
    cross_m: np.ndarray
    height_m: np.ndarray
    pitch_rad: np.ndarray
    displacement_m: float

    def __post_init__(self) -> None:
        arrays = [self.grid, self.along_m, self.cross_m, self.height_m, self.pitch_rad]
        if any(a.ndim != 1 or len(a) < 4 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("swing shape arrays must be aligned 1-D samples, at least 4")
        if not np.isfinite(np.concatenate(arrays)).all():
            raise ValueError("swing shape samples must be finite")
        if not np.isfinite(self.displacement_m) or self.displacement_m <= 0:
            raise ValueError("swing displacement must be finite and positive")
        if self.grid[0] != 0. or self.grid[-1] != 1. or np.any(np.diff(self.grid) <= 0):
            raise ValueError("swing shape grid must increase from 0 to 1")
        if not abs(self.along_m[0]) < 1e-6 or abs(self.along_m[-1] - self.displacement_m) > 1e-6:
            raise ValueError("swing shape must start at along=0 and end at displacement")
        for name, values in (("cross", self.cross_m), ("height", self.height_m),
                             ("pitch", self.pitch_rad)):
            if abs(values[0]) > 1e-6 or abs(values[-1]) > 1e-6:
                raise ValueError(f"swing shape {name} must be re-based to zero at both contacts")

    def sample(self, u: float, start: LinkTransform, end: LinkTransform) -> LinkTransform:
        """The arc retargeted onto the (start, end) anchors at progress ``u``."""
        if not np.isfinite(u) or not 0. <= u <= 1.:
            raise ValueError("swing progress u must be finite in [0, 1]")
        warp = (1. - math.cos(math.pi * u)) / 2.
        along = float(np.interp(warp, self.grid, self.along_m))
        cross = float(np.interp(warp, self.grid, self.cross_m))
        height = float(np.interp(warp, self.grid, self.height_m))
        pitch = float(np.interp(warp, self.grid, self.pitch_rad))
        blend = _quintic(u)
        delta = end.translation[:2] - start.translation[:2]
        span = float(np.linalg.norm(delta))
        xy = start.translation[:2].copy()
        if span > 1e-9:
            cosine, sine = math.cos(yaw := math.atan2(delta[1], delta[0])), math.sin(yaw)
            scale = span / self.displacement_m
            xy += np.array([cosine * along - sine * cross,
                            sine * along + cosine * cross]) * scale
        # Vertical arc is not scaled: its few centimetres are a per-step property,
        # not part of the stride the retargeting rescales.
        z = start.translation[2] + (end.translation[2] - start.translation[2]) * blend + height
        rotation = start.rotation @ axis_angle_to_matrix(
            blend * matrix_to_axis_angle(start.rotation.T @ end.rotation))
        rotation = rotation @ axis_angle_to_matrix(np.array([0., pitch, 0.]))
        return LinkTransform(rotation.astype(float), np.array([xy[0], xy[1], z]))


#: A foot counts as swinging while its world-frame forward speed exceeds this
#: fraction of the source clip's own speed. Foot roll keeps a planted ankle
#: moving at up to ~0.25 of walking speed (measured), so the threshold must sit
#: above that; swing peaks at 2-3x it, so the separation is wide.
SWING_SPEED_FRACTION = 0.3
#: Threshold flicker near touchdown can shred a few frames into a spurious run;
#: a real swing spans well over a tenth of the cycle (measured 38-58%).
SWING_RUN_MIN_FRACTION = 0.1
#: Soft cap on the swing foot's sagittal pitch excursion. The source clips
#: articulate up to 120 deg within one swing; compressed into the runtime swing
#: window that demands more ankle speed than the joint rate limiter can honor
#: (measured: ankle targets saturate at 8 rad/s, ankle tracking error 34 deg,
#: and every touchdown lands with a stubbed foot). tanh keeps the profile smooth
#: and the small-amplitude shape intact while bounding the demand.
SWING_PITCH_MAX_RAD = 0.45
#: Hard cap on the retargeted arc's ground clearance. The source lifts its ankle
#: 14.5 cm at its own brisk cadence; even stride-scaled that demands more knee
#: flexion speed than the drives track (measured knee lag 16 deg p95 and
#: touchdown drag). Slow human walking clears the floor by ~1-2 cm with the
#: sole, so 4.5 cm keeps a visible natural arc inside the drive budget.
SWING_LIFT_CAP_M = 0.045
#: Hard bound on how far below the reach cap the root bob may dip. Deeper dips
#: are kinematically fine but at a step reversal the bob's vertical *velocity*
#: flips sign instantly, and the position spring's damping term turns that
#: into a measured ~200 N vertical jolt that rings the collar drives ±20 deg
#: (long-protocol backward reversals, 52.7 s). 2 cm keeps visible vertical
#: life inside the reversal budget.
ROOT_BOB_MAX_BELOW_CAP_M = 0.02
#: Circular smoothing window (frames) for the extracted arc profiles. The source
#: path carries mocap + retargeting acceleration spikes (toe-off push, landing
#: roll) that the joint drives cannot follow: tracked with ~1 deg error on the
#: analytic arc, the same drives lag 16 deg on the raw arc, and the lag slides
#: both the touching-down foot and, through the assist spring's reaction, the
#: planted one. A 40 ms circular mean removes the spikes, keeps the endpoints
#: (rebased afterwards) and preserves the arc's shape.
SWING_SMOOTH_FRAMES = 5


def _circular_smooth(values: np.ndarray, kernel: int) -> np.ndarray:
    """Circular moving average over axis 0; the cycle tables wrap by construction."""
    if kernel <= 1:
        return values

    def smooth1d(column: np.ndarray) -> np.ndarray:
        padded = np.concatenate([column[-kernel:], column, column[:kernel]])
        return np.convolve(padded, np.ones(kernel) / kernel, mode="same")[kernel:-kernel]

    if values.ndim == 1:
        return smooth1d(values)
    return np.stack(
        [smooth1d(np.asarray(values[:, i])) for i in range(values.shape[1])], axis=1
    )


def _single_circular_run(flag: np.ndarray, *, name: str) -> list[int]:
    """Frame indices of the one dominant circular True run, in time order.

    Runs shorter than :data:`SWING_RUN_MIN_FRACTION` of the cycle are threshold
    noise (e.g. the landing-deceleration blip) and are dropped; more than one
    surviving run is a configuration error, not something to silently merge.
    """
    if flag.all() or not flag.any():
        raise ValueError(f"{name}: expected exactly one contiguous run, got none or all")
    count = len(flag)
    starts = [i for i in range(count) if flag[i] and not flag[(i - 1) % count]]
    runs = []
    for start in starts:
        run = []
        index = start
        while flag[index % count]:
            run.append(index % count)
            index += 1
        runs.append(run)
    surviving = [run for run in runs if len(run) >= SWING_RUN_MIN_FRACTION * count]
    if len(surviving) != 1:
        raise ValueError(
            f"{name}: expected exactly one dominant run, found {len(surviving)} "
            f"(lengths {[len(r) for r in runs]} of {count})"
        )
    return surviving[0]


def extract_swing_shapes(
    clip: MotionClip,
    gait: Gait,
    plan: HumanRigPlan,
    config: ContactGaitConfig,
) -> dict[str, AmassSwingShape]:
    """Lift each foot's swing arc out of the retargeted cycle, sanitized for contact.

    Swing windows come from each foot's own world-frame motion, not from a
    shared half-cycle assumption or the travel support mask (which classifies
    stance-phase foot roll as swing on one side -- the documented right-foot
    mask defect -- and whose runs need not align with the true swing). The world
    path is rebuilt from the seam-corrected cycle samples, so the arc is
    continuous across the window seam, and the traversal frame is unwrapped for
    runs that cross it.

    The arc keeps the mocap's shape (lateral bow, sagittal foot pitch) scaled to
    the commanded stride: the commanded/source travel ratio scales the vertical
    profile too, with ``clearance_m`` as a floor so a slow command cannot pull
    the arc into tracking-lag scrape territory. Endpoints are re-based to zero
    height and pitch because stance anchors are flat and grounded.
    """

    count = len(gait.joints)
    direction = float(gait.provenance.get("source_forward_sign", 1.))
    source_travel = gait.speed_m_s * gait.duration_s
    if not np.isfinite(source_travel) or source_travel < 1e-6:
        raise ValueError("swing extraction needs a translating source clip")
    stride_scale = config.cycle_distance_m / source_travel
    travel_vector = np.array([*(clip.root_translation[-1, :2] - clip.root_translation[0, :2]), 0.])
    wall_per_frame = gait.duration_s / (count - 1)

    paths: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    rotations: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for index in range(count):
        phase = index / (count - 1)
        q, height = gait.sample(phase)
        tilt_vector = gait.tilt(phase)
        tilt = axis_angle_to_matrix(tilt_vector)
        poses = forward_kinematics(
            plan, dict(zip(plan.dof_names, q, strict=True)),
            root_position=(0., 0., float(height)), root_rotation=tilt_vector)
        full = axis_angle_to_matrix(clip.root_rotation[index])
        root_xy = clip.root_translation[index, :2]
        # The corrected sample height replaces the raw root z: both carry the
        # same gravity-consistent motion, but only the corrected one is
        # continuous across the window seam.
        root_z = float(height) - float(plan.spawn_root_position[2])
        for side in ("left", "right"):
            ankle = poses[f"{side}_ankle"]
            offset = tilt.T @ (ankle.translation - np.array([0., 0., float(height)]))
            paths[side].append(full @ offset + np.array([root_xy[0], root_xy[1], root_z]))
            rotations[side].append(full @ (tilt.T @ ankle.rotation))

    shapes: dict[str, AmassSwingShape] = {}
    for side in ("left", "right"):
        raw_path = np.asarray(paths[side])
        # Run detection uses the raw kinematics; the smoothing below is for
        # trajectory quality and must not perturb the segmentation thresholds.
        velocity = np.gradient(raw_path[:, 0], wall_per_frame)
        swing = velocity * direction > SWING_SPEED_FRACTION * gait.speed_m_s
        run = _single_circular_run(swing, name=f"{side} swing run")
        path = _circular_smooth(raw_path, SWING_SMOOTH_FRAMES)
        positions, run_rotations = [], []
        wraps, previous = 0, run[0]
        for index in run:
            if index < previous:
                wraps += 1
            previous = index
            positions.append(path[index] + wraps * travel_vector)
            run_rotations.append(rotations[side][index])
        start, end = positions[0], positions[-1]
        delta = end[:2] - start[:2]
        displacement = float(np.linalg.norm(delta))
        if displacement < 0.05:
            raise ValueError(f"{side} swing displacement {displacement:.3f} m is degenerate")
        yaw = math.atan2(delta[1], delta[0])
        frame = np.array([[math.cos(-yaw), -math.sin(-yaw)],
                          [math.sin(-yaw), math.cos(-yaw)]])
        rel = np.asarray([p[:2] - start[:2] for p in positions]) @ frame.T
        along = rel[:, 0].copy()
        cross = rel[:, 1].copy()
        along[-1] = displacement
        cross[0] = cross[-1] = 0.
        ground = min(float(start[2]), float(end[2]))
        rebase = _quintic(np.linspace(0., 1., len(run)))
        height = np.asarray([float(p[2]) for p in positions]) - ground
        height -= (1. - rebase) * height[0] + rebase * height[-1]
        ramp = np.minimum(1., np.minimum(
            np.linspace(0., 1., len(run)) / config.lift_ramp_fraction,
            (1. - np.linspace(0., 1., len(run))) / config.lift_ramp_fraction))
        bell = ramp**3 * (10. - 15. * ramp + 6. * ramp * ramp)
        height = np.minimum(
            np.maximum(height * stride_scale, config.clearance_m * bell),
            SWING_LIFT_CAP_M * bell + (1. - bell) * height,
        )
        unwound = axis_angle_to_matrix(np.array([0., 0., -yaw]))
        pitch = np.asarray([math.atan2(q[0, 2], q[2, 2]) for q in
                            (unwound @ rotation for rotation in run_rotations)])
        pitch = _circular_smooth(pitch, SWING_SMOOTH_FRAMES)
        pitch -= (1. - rebase) * pitch[0] + rebase * pitch[-1]
        pitch = SWING_PITCH_MAX_RAD * np.tanh(pitch / SWING_PITCH_MAX_RAD)
        shapes[side] = AmassSwingShape(
            grid=np.linspace(0., 1., len(run)),
            along_m=along, cross_m=cross, height_m=height, pitch_rad=pitch,
            displacement_m=displacement,
        )
    return shapes


class LegPoseSolver:
    def __init__(self, plan: HumanRigPlan, side: str, config: ContactGaitConfig) -> None:
        self.plan, self.side, self.config = plan, side, config
        by_child = {j.child_link: (i, j) for i, j in enumerate(plan.joints)}
        names, current = [], f"{side}_ankle"
        while current != plan.root_link:
            names.append(current)
            current = plan.link(current).parent_link
        self.chain = [(name, by_child[name]) for name in reversed(names)]
        self.indices = np.array([i for _, (i, _) in self.chain])
        self.lower = np.deg2rad([plan.joints[i].lower_deg for i in self.indices])
        self.upper = np.deg2rad([plan.joints[i].upper_deg for i in self.indices])

    def poses(self, q: np.ndarray, root: LinkTransform) -> dict[str, LinkTransform]:
        poses = {self.plan.root_link: root}
        for name, (i, joint) in self.chain:
            parent = poses[joint.parent_link]
            axis = "xyz".index(joint.axis)
            vector = np.zeros(3)
            vector[axis] = q[i]
            poses[name] = LinkTransform(parent.rotation @ axis_angle_to_matrix(vector),
                                        parent.transform_point(joint.local_pos0))
        return poses

    def solve(self, reference: np.ndarray, root: LinkTransform, goal: LinkTransform) -> np.ndarray:
        q = reference.copy()
        # Seed on the flexed-knee branch. Near a straight mocap pose the positional
        # Jacobian is singular and can converge to the hyperextended limit instead.
        flexion = {j.chain_joint: i for _, (i, j) in self.chain if j.axis == "y"}
        knee = flexion[f"{self.side}_knee"]
        bend = max(0., .5 - q[knee])
        q[knee] += bend
        q[flexion[f"{self.side}_hip"]] -= .5*bend
        q[flexion[f"{self.side}_ankle"]] -= .5*bend
        weight = self.config.orientation_weight_m
        for _ in range(self.config.iterations):
            poses = self.poses(q, root)
            ankle = poses[f"{self.side}_ankle"]
            error = np.r_[goal.translation - ankle.translation,
                          weight * matrix_to_axis_angle(goal.rotation @ ankle.rotation.T)]
            if np.linalg.norm(error) < 1e-5:
                break
            jacobian = np.empty((6, len(self.indices)))
            for col, (_, (_i, joint)) in enumerate(self.chain):
                parent = poses[joint.parent_link]
                axis = parent.rotation[:, "xyz".index(joint.axis)]
                jacobian[:3, col] = np.cross(
                    axis, ankle.translation-parent.transform_point(joint.local_pos0))
                jacobian[3:, col] = weight*axis
            # Keep non-flexion knee motion close to the mocap pose while allowing
            # hips and ankles to meet the six-dimensional contact-pose task.
            costs = np.array([1000. if j.chain_joint.endswith("knee") and j.axis != "y"
                              else 1. for _, (_, j) in self.chain])
            inv_cost = 1 / costs
            delta = (jacobian * inv_cost).T @ np.linalg.solve(
                (jacobian * inv_cost) @ jacobian.T + self.config.damping*np.eye(6), error)
            q[self.indices] = np.clip(
                q[self.indices] + np.clip(delta, -self.config.max_step_rad,
                                          self.config.max_step_rad), self.lower, self.upper)
        return q


def bake_contact_cycle(
    gait: Gait,
    plan: HumanRigPlan,
    config: ContactGaitConfig,
    *,
    source_height_m: np.ndarray | None = None,
    source_height_scale: float = 1.0,
) -> Gait:
    """Fit flat support feet and smooth swing while retaining mocap arm/trunk poses.

    With ``source_height_m`` the root follows the source clip's own vertical
    trajectory, so the walk keeps the mocap's vertical life instead of gliding
    at a constant height. Deviations below the stride-limited reachable cap are
    scaled by ``source_height_scale`` (the commanded/source stride ratio) and
    rises above the cap are clipped: a slower command shrinks the bob the same
    way it shrinks the swing arcs, and rising above the cap would hand the leg
    IK an unreachable goal. Dipping below it only adds knee flexion.
    """
    count = len(gait.joints)
    original = [gait.sample(i/(count-1))[0] for i in range(count)]
    neutral = forward_kinematics(plan, root_position=(0., 0., 0.))
    direction = float(gait.provenance.get("source_forward_sign", 1.))
    raw_poses = [forward_kinematics(plan, dict(zip(plan.dof_names, q, strict=True)))
                 for q in original]
    start = int(np.argmax([p["left_ankle"].translation[0]*direction for p in raw_poses[:-1]]))
    phase_origin = start/(count-1)
    sole_height, lateral, solvers = {}, {}, {}
    height_cap = float(np.median(gait.height_m))
    for side in ("left", "right"):
        ankle, hip, knee = (neutral[f"{side}_{part}"] for part in ("ankle", "hip", "knee"))
        sole_height[side] = ankle.translation[2] - capsule_bottom(plan, neutral, f"{side}_ankle")
        lateral[side] = ankle.translation[1]
        reach = (np.linalg.norm(hip.translation-knee.translation)
                 + np.linalg.norm(knee.translation-ankle.translation)) * config.reach_fraction
        horizontal = .5 * config.cycle_distance_m*config.stance_fraction + abs(hip.translation[0])
        if horizontal >= reach:
            raise ValueError("requested contact stride exceeds leg reach")
        height_cap = min(height_cap, sole_height[side] - hip.translation[2]
                         + np.sqrt(reach**2-horizontal**2))
        solvers[side] = LegPoseSolver(plan, side, config)
    if source_height_m is None:
        heights = np.full(count, height_cap)
    else:
        if np.shape(source_height_m) != (count,) or not np.isfinite(source_height_m).all():
            raise ValueError("source root heights must align with cycle frames and be finite")
        if not np.isfinite(source_height_scale) or source_height_scale <= 0:
            raise ValueError("source height scale must be finite and positive")
        heights = np.minimum(
            height_cap + source_height_scale * (np.asarray(source_height_m, dtype=np.float64)
                                                - height_cap),
            height_cap,
        )
        heights = np.maximum(heights, height_cap - ROOT_BOB_MAX_BELOW_CAP_M)
    joints, support = [], []
    max_residual = 0.
    worst = ""
    for i, q_ref in enumerate(original):
        phase = i/(count-1)
        root = LinkTransform(axis_angle_to_matrix(gait.tilt(phase)), np.array([0., 0., heights[i]]))
        q, support_row = q_ref.copy(), []
        for col, side in enumerate(("left", "right")):
            x, lift, stance = foot_path(phase-phase_origin-col*.5, config)
            goal = LinkTransform(np.eye(3), np.array([direction*x, lateral[side],
                                                      sole_height[side]+lift]))
            q = solvers[side].solve(q, root, goal)
            actual = solvers[side].poses(q, root)[f"{side}_ankle"]
            residual = float(np.linalg.norm(actual.translation-goal.translation))
            if residual > max_residual:
                max_residual = residual
                worst = (f"phase={phase:.3f}, side={side}, height={heights[i]:.3f}, "
                         f"tilt={gait.tilt(phase)}")
            support_row.append(stance)
        joints.append(q)
        support.append(support_row)
    # Store an explicitly closed cycle; Gait.sample must not add a second drift ramp.
    joints[-1] = joints[0].copy()
    support[-1] = support[0].copy()
    # COM offset relative to the root origin, per frame (root at the origin, no
    # height baked in): seam-closed by construction and, at runtime, added to the
    # live root target to reconstruct the reference COM for feedforward.
    com_offsets = []
    for i, q in enumerate(joints):
        poses = forward_kinematics(
            plan, dict(zip(plan.dof_names, q, strict=True)),
            root_rotation=gait.tilt(i/(count-1)))
        com_offsets.append(configuration_com(plan, poses))
    object.__setattr__(gait, "joints", np.array(joints))
    object.__setattr__(gait, "height_m", heights)
    object.__setattr__(gait, "support_mask", np.array(support))
    object.__setattr__(gait, "com_offset_m", np.asarray(com_offsets))
    object.__setattr__(gait, "speed_m_s", config.cycle_distance_m/gait.duration_s)
    gait.provenance["contact_cycle"] = {**vars(config), "ik_max_position_residual_m": max_residual,
                                        "root_height_m": float(np.median(heights)),
                                        "root_height_range_m": float(heights.max()-heights.min()),
                                        "phase_origin": phase_origin,
                                        "derivation": "AMASS upper body; contact-retargeted legs"}
    if max_residual > .005:
        raise ValueError(f"contact gait IK residual exceeds 5 mm: {max_residual:.6f}; {worst}")
    return gait


class ContactFootPlanner:
    """World-fixed stance poses with separately completed swing steps.

    Goals are solved as joint targets only. No body teleport, fixed constraint or
    contact-force injection is used to make a foot appear planted.

    ``swing_shapes`` optionally maps "forward"/"backward" to a per-side dict of
    :class:`AmassSwingShape` ({"left": ..., "right": ...}); a swing then follows
    the retargeted mocap arc instead of the analytic quintic. Stance anchors and
    the landing prediction are unchanged either way.
    """

    def __init__(self, plan: HumanRigPlan, config: ContactGaitConfig,
                 nominal_speed_m_s: float, acceleration_m_s2: float,
                 swing_shapes: dict[str, dict[str, AmassSwingShape]] | None = None) -> None:
        if (not np.isfinite([nominal_speed_m_s, acceleration_m_s2]).all()
                or min(nominal_speed_m_s, acceleration_m_s2) <= 0):
            raise ValueError("step speed and acceleration must be positive")
        if swing_shapes is not None:
            unknown = set(swing_shapes) - {"forward", "backward"}
            bad_sides = {
                direction: set(sides) - {"left", "right"}
                for direction, sides in swing_shapes.items()
            }
            if unknown or any(bad_sides.values()):
                raise ValueError(
                    f"swing_shapes must map forward/backward to left/right shapes; "
                    f"unknown directions {sorted(unknown)}, bad sides {bad_sides}"
                )
        self.plan, self.config = plan, config
        self.speed, self.acceleration = nominal_speed_m_s, acceleration_m_s2
        self.swing_shapes = {
            direction: dict(sides) for direction, sides in (swing_shapes or {}).items()
        }
        self.solvers = {s: LegPoseSolver(plan, s, config) for s in ("left", "right")}
        self.neutral = forward_kinematics(plan, root_position=(0., 0., 0.))
        self.leg_lengths = {
            s: sum(np.linalg.norm(self.neutral[f"{s}_{a}"].translation
                                  - self.neutral[f"{s}_{b}"].translation)
                   for a, b in (("hip", "knee"), ("knee", "ankle")))
            for s in self.solvers
        }
        self.reset()

    def reset(self) -> None:
        self.phase = 0.
        self.anchors: dict[str, LinkTransform] = {}
        self.swings: dict[str, tuple[LinkTransform, LinkTransform, float]] = {}
        self.last_residual_m = 0.
        self.last_command = 0.
        self.last_joints: np.ndarray | None = None

    def landing(self, side: str, root: LinkTransform, heading: float, turn_rate: float,
                speed: float, command: float, remaining_s: float) -> LinkTransform:
        c = self.config
        duration = c.cycle_distance_m/self.speed
        desired_speed = command*self.speed
        ramp_time = min(remaining_s, abs(desired_speed-speed)/self.acceleration)
        acceleration = np.sign(desired_speed-speed)*self.acceleration
        travel = (speed*ramp_time + .5*acceleration*ramp_time**2
                  + desired_speed*(remaining_s-ramp_time))
        yaw = heading + turn_rate*remaining_s
        rotation = axis_angle_to_matrix(np.array([0., 0., yaw]))
        ankle = self.neutral[f"{side}_ankle"]
        local = np.array([desired_speed*duration*c.stance_fraction/2,
                          ankle.translation[1], 0.])
        position = root.translation + rotation[:, 0]*travel + rotation@local
        position[2] = ankle.translation[2] - capsule_bottom(
            self.plan, self.neutral, f"{side}_ankle")
        return LinkTransform(rotation, position)

    def correct(self, reference: np.ndarray, root: LinkTransform, actual: dict[str, LinkTransform],
                *, heading: float, turn_rate: float, speed: float, command: float,
                dt_s: float) -> np.ndarray:
        if (not np.isfinite([heading, turn_rate, speed, command, dt_s]).all()
                or dt_s <= 0 or abs(command) > 1 or reference.shape != (len(self.plan.joints),)
                or not np.isfinite(reference).all()):
            raise ValueError("invalid contact planner state or timestep")
        c = self.config
        duration = c.cycle_distance_m/self.speed
        swing_time = duration*(1-c.stance_fraction)
        if not self.anchors:
            for side in self.solvers:
                name = f"{side}_ankle"
                pose = actual[name]
                position = pose.translation.copy()
                position[2] -= capsule_bottom(self.plan, actual, name)
                self.anchors[side] = LinkTransform(pose.rotation.copy(), position)
        moving = abs(command) > 0
        changed_command = command != self.last_command
        if moving and not self.swings:
            direction = command*np.array([np.cos(heading), np.sin(heading), 0.])
            first = min(self.anchors, key=lambda s: float(
                (self.anchors[s].translation-root.translation)@direction))
            hip = root.transform_point(self.neutral[f"{first}_hip"].translation)
            reach = np.linalg.norm(self.anchors[first].translation-hip)
            # Turning changes the hip-to-foot distance even at constant forward
            # speed. End stance before the leg is straight; a clock-only stance
            # waited until 0.76 m on a 0.74 m leg in the action-matrix regression.
            if changed_command or reach >= c.reach_fraction*self.leg_lengths[first]:
                self.phase = 0. if first == "left" else .5
        if moving or self.swings:
            self.phase = (self.phase + dt_s/duration) % 1
        q = reference.copy()
        if self.last_joints is not None:
            # Continue the previous leg solution, not a fresh mocap null-space
            # pose. Re-seeding every frame can jump IK branches during a turn
            # even though the world-space foot goal moves continuously.
            for solver in self.solvers.values():
                q[solver.indices] = self.last_joints[solver.indices]
        residuals = []
        for col, side in enumerate(("left", "right")):
            phase = (self.phase-col*.5) % 1
            if moving and phase < 1-c.stance_fraction and not self.swings:
                end = self.landing(side, root, heading, turn_rate, speed, command, swing_time)
                direction = "forward" if command >= 0. else "backward"
                shape = self.swing_shapes.get(direction, {}).get(side)
                self.swings[side] = (self.anchors[side], end, 0., shape)
            if side in self.swings:
                start, end, elapsed, shape = self.swings[side]
                # A stop early in swing needs a braking landing, not the old
                # continuing-walk prediction. Late swing keeps its nearby landing.
                if changed_command and elapsed < .5*swing_time:
                    end = self.landing(side, root, heading, turn_rate, speed, command,
                                       swing_time-elapsed)
                elapsed += dt_s
                u = min(1., elapsed/swing_time)
                if shape is None:
                    blend = u**3*(10-15*u+6*u*u)
                    position = start.translation*(1-blend) + end.translation*blend
                    ramp = min(1., u/c.lift_ramp_fraction, (1-u)/c.lift_ramp_fraction)
                    position[2] += c.clearance_m*ramp**3*(10-15*ramp+6*ramp*ramp)
                    rotation = start.rotation @ axis_angle_to_matrix(
                        blend*matrix_to_axis_angle(start.rotation.T@end.rotation))
                    goal = LinkTransform(rotation, position)
                else:
                    goal = shape.sample(u, start, end)
                if u >= 1:
                    self.anchors[side] = end
                    del self.swings[side]
                else:
                    self.swings[side] = (start, end, elapsed, shape)
            else:
                goal = self.anchors[side]
            q = self.solvers[side].solve(q, root, goal)
            solved = self.solvers[side].poses(q, root)[f"{side}_ankle"]
            residuals.append(float(np.linalg.norm(solved.translation-goal.translation)))
        self.last_residual_m = max(residuals)
        self.last_command = command
        self.last_joints = q.copy()
        return q


def fit_contact_idle(idle: Gait, plan: HumanRigPlan, height_m: float,
                     config: ContactGaitConfig) -> None:
    """Initialize the idle legs at the same reachable height as the contact gait."""
    neutral = forward_kinematics(plan, root_position=(0., 0., 0.))
    q = idle.sample(0.)[0]
    root = LinkTransform(axis_angle_to_matrix(idle.tilt(0.)), np.array([0., 0., height_m]))
    for side in ("left", "right"):
        name = f"{side}_ankle"
        position = neutral[name].translation.copy()
        position[2] -= capsule_bottom(plan, neutral, name)
        q = LegPoseSolver(plan, side, config).solve(q, root, LinkTransform(np.eye(3), position))
    object.__setattr__(idle, "joints", np.tile(q, (len(idle.joints), 1)))
    object.__setattr__(idle, "height_m", np.full(len(idle.height_m), height_m))
    poses = forward_kinematics(plan, dict(zip(plan.dof_names, q, strict=True)))
    object.__setattr__(
        idle, "com_offset_m",
        np.tile(configuration_com(plan, poses), (len(idle.joints), 1)),
    )
