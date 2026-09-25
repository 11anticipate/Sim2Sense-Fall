"""Contact-consistent periodic leg targets derived from an AMASS upper-body cycle.

This is trajectory retargeting, not a physical contact constraint. Actual contact,
slip and joint tracking must still pass the simulator gates independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .contact_control import capsule_bottom
from .rig import HumanRigPlan, LinkTransform, forward_kinematics
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle

if TYPE_CHECKING:
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

    def __post_init__(self) -> None:
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError("contact gait settings must be finite")
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


def bake_contact_cycle(gait: Gait, plan: HumanRigPlan, config: ContactGaitConfig) -> Gait:
    """Fit flat support feet and smooth swing while retaining mocap arm/trunk poses."""
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
    joints, support = [], []
    max_residual = 0.
    worst = ""
    for i, q_ref in enumerate(original):
        phase = i/(count-1)
        root = LinkTransform(axis_angle_to_matrix(gait.tilt(phase)), np.array([0., 0., height_cap]))
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
                worst = (f"phase={phase:.3f}, side={side}, height={height_cap:.3f}, "
                         f"tilt={gait.tilt(phase)}")
            support_row.append(stance)
        joints.append(q)
        support.append(support_row)
    # Store an explicitly closed cycle; Gait.sample must not add a second drift ramp.
    joints[-1] = joints[0].copy()
    support[-1] = support[0].copy()
    object.__setattr__(gait, "joints", np.array(joints))
    object.__setattr__(gait, "height_m", np.full(count, height_cap))
    object.__setattr__(gait, "support_mask", np.array(support))
    object.__setattr__(gait, "speed_m_s", config.cycle_distance_m/gait.duration_s)
    gait.provenance["contact_cycle"] = {**vars(config), "ik_max_position_residual_m": max_residual,
                                        "root_height_m": height_cap, "phase_origin": phase_origin,
                                        "derivation": "AMASS upper body; contact-retargeted legs"}
    if max_residual > .005:
        raise ValueError(f"contact gait IK residual exceeds 5 mm: {max_residual:.6f}; {worst}")
    return gait


class ContactFootPlanner:
    """World-fixed stance poses with separately completed swing steps.

    Goals are solved as joint targets only. No body teleport, fixed constraint or
    contact-force injection is used to make a foot appear planted.
    """

    def __init__(self, plan: HumanRigPlan, config: ContactGaitConfig,
                 nominal_speed_m_s: float, acceleration_m_s2: float) -> None:
        if (not np.isfinite([nominal_speed_m_s, acceleration_m_s2]).all()
                or min(nominal_speed_m_s, acceleration_m_s2) <= 0):
            raise ValueError("step speed and acceleration must be positive")
        self.plan, self.config = plan, config
        self.speed, self.acceleration = nominal_speed_m_s, acceleration_m_s2
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
                self.swings[side] = (self.anchors[side], end, 0.)
            if side in self.swings:
                start, end, elapsed = self.swings[side]
                # A stop early in swing needs a braking landing, not the old
                # continuing-walk prediction. Late swing keeps its nearby landing.
                if changed_command and elapsed < .5*swing_time:
                    end = self.landing(side, root, heading, turn_rate, speed, command,
                                       swing_time-elapsed)
                elapsed += dt_s
                u = min(1., elapsed/swing_time)
                blend = u**3*(10-15*u+6*u*u)
                position = start.translation*(1-blend) + end.translation*blend
                ramp = min(1., u/c.lift_ramp_fraction, (1-u)/c.lift_ramp_fraction)
                position[2] += c.clearance_m*ramp**3*(10-15*ramp+6*ramp*ramp)
                rotation = start.rotation @ axis_angle_to_matrix(
                    blend*matrix_to_axis_angle(start.rotation.T@end.rotation))
                goal = LinkTransform(rotation, position)
                if u >= 1:
                    self.anchors[side] = end
                    del self.swings[side]
                else:
                    self.swings[side] = (start, end, elapsed)
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
