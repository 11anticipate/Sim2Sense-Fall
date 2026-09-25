"""CPU geometry fitting and stance-foot inverse kinematics for keyboard trials.

These are explicit collision/control approximations, not a learned balance policy.
The IK changes joint targets only; PhysX remains responsible for all actual motion.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .rig import HumanRigPlan, LinkTransform, forward_kinematics, quaternion_from_z
from .rotations import matrix_to_axis_angle, quaternion_to_matrix
from .skinning import SmplMesh

#: Thickness of the sole band the foot collision box is fitted to.
FOOT_SOLE_BAND_M = 0.04


def fit_collision_capsules(
    plan: HumanRigPlan, mesh: SmplMesh, *, margin_m: float,
    foot_shape: str = "box", sole_band_m: float = FOOT_SOLE_BAND_M,
    hand_shape: str = "capsule",
) -> tuple[HumanRigPlan, list[dict[str, Any]]]:
    """Enclose dominant-weight skin regions in deterministic principal-axis capsules.

    Feet are fixed to ankles and share their collider. Dominant ownership is only
    a rest-pose fit; blended posed skin still requires independent penetration QA.
    """
    if not np.isfinite(margin_m) or not 0 <= margin_m <= 0.02:
        raise ValueError("collision margin must be finite in [0, 0.02] m")
    if foot_shape not in {"box", "convex_hull"}:
        raise ValueError("foot_shape must be box or convex_hull")
    if hand_shape not in {"capsule", "convex_hull"}:
        raise ValueError("hand_shape must be capsule or convex_hull")
    if not np.isfinite(sole_band_m) or sole_band_m <= 0:
        raise ValueError("sole_band_m must be finite and positive")
    owners = np.asarray(mesh.topology.joint_names)[mesh.weights.argmax(axis=1)]
    owners = owners.copy()
    for side in ("left", "right"):
        owners[owners == f"{side}_foot"] = f"{side}_ankle"
    links, audit = [], []
    for link in plan.links:
        cap = link.capsule
        points = mesh.vertices[owners == link.name] - np.asarray(link.rest_position)
        if cap is None or len(points) < 4:
            links.append(link)
            continue
        if link.name in {"left_ankle", "right_ankle"}:
            # Fit the foot box to the SOLE BAND only (the bottom 4 cm of the foot
            # region). A full-foot box pitches with the ankle and then rests on a
            # virtual corner that sits tens of millimetres below the actual sole
            # edge -- the visible foot floats above the floor while the collider
            # "touches" (measured 38 mm skin vs 6 mm collider). A thin sole slab
            # corners down exactly where the sole edge is, so the skin tracks the
            # contact at any ankle pitch. The instep loses its collider: flat-floor
            # contact happens at the sole.
            sole_top = points[:, 2].min() + sole_band_m
            band = points[points[:, 2] <= sole_top]
            low, high = band.min(axis=0) - margin_m, band.max(axis=0) + margin_m
            center = (low + high) / 2
            size = high - low
            inertia = link.mass_kg / 12 * (np.sum(size**2) - size**2)
            vertices, faces = (), ()
            if foot_shape == "convex_hull":
                # Use the actual surface patch, not fictitious bounding-box corners.
                # PhysX cooks its convex hull; the source patch need not be watertight.
                selected = mesh.faces[np.all(owners[mesh.faces] == link.name, axis=1)]
                indices, inverse = np.unique(selected, return_inverse=True)
                vertices = tuple(tuple(map(float, p))
                                 for p in mesh.vertices[indices] - link.rest_position)
                faces = tuple(tuple(map(int, face)) for face in inverse.reshape(-1, 3))
            links.append(
                replace(
                    link,
                    center_of_mass=tuple(center),
                    inertia_kg_m2=tuple(inertia),
                    collision_box_bounds=(tuple(low), tuple(high)) if foot_shape == "box" else None,
                    collision_mesh_vertices=vertices,
                    collision_mesh_faces=faces,
                )
            )
            audit.append(
                {
                    "link": link.name,
                    "vertices": len(points),
                    "margin_m": margin_m,
                    "method": f"dominant_skin_weight_foot_{foot_shape}",
                    "inertia_model": "sole_band_box_fixed_across_shape_ablation",
                    "bounds_m": [low.tolist(), high.tolist()],
                }
            )
            continue
        mean = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - mean, full_matrices=False)
        axis = axes[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        projection = (points - mean) @ axis
        radial = points - mean - projection[:, None] * axis
        radius = float(np.linalg.norm(radial, axis=1).max()) + margin_m
        low, high = float(projection.min()), float(projection.max())
        center = mean + 0.5 * (low + high) * axis
        # A full projection interval encloses end-cap vertices without inflating
        # every radius to fit points beyond a prematurely shortened cylinder.
        length = max(0.0, high - low - radius)
        local = points - center
        clamped = np.clip(local @ axis, -length / 2, length / 2)
        radius = float(np.linalg.norm(local - clamped[:, None] * axis, axis=1).max())
        radius += margin_m
        fitted = replace(
            cap,
            center=tuple(float(v) for v in center),
            orientation_wxyz=quaternion_from_z(axis),
            radius_m=radius,
            cylinder_length_m=length,
        )
        axial = 0.5 * link.mass_kg * radius**2
        transverse = link.mass_kg * (3 * radius**2 + length**2) / 12
        fitted_link = replace(
            link, capsule=fitted, center_of_mass=fitted.center,
            inertia_kg_m2=(transverse, transverse, axial))
        is_hand_hull = hand_shape == "convex_hull" and link.name in {"left_hand", "right_hand"}
        if is_hand_hull:
            # Keep the capsule-derived inertial model fixed during shape comparison.
            selected = mesh.faces[np.all(owners[mesh.faces] == link.name, axis=1)]
            indices, inverse = np.unique(selected, return_inverse=True)
            fitted_link = replace(
                fitted_link,
                collision_mesh_vertices=tuple(tuple(map(float, p))
                                             for p in mesh.vertices[indices] - link.rest_position),
                collision_mesh_faces=tuple(tuple(map(int, f)) for f in inverse.reshape(-1, 3)))
        links.append(fitted_link)
        audit.append(
            {
                "link": link.name,
                "vertices": len(points),
                "radius_m": radius,
                "length_m": length,
                "margin_m": margin_m,
                "method": "dominant_skin_weight_hand_convex_hull" if is_hand_hull
                else "dominant_skin_weight_principal_axis_enclosure",
                "inertia_model": "principal_axis_capsule_fixed_across_shape_ablation",
            }
        )
    fitted_plan = replace(plan, links=tuple(links))
    poses = forward_kinematics(fitted_plan)
    minimum = min(capsule_bottom(fitted_plan, poses, cap.link) for cap in fitted_plan.colliders)
    fitted_plan = replace(
        fitted_plan,
        ground_offset_m=-minimum,
        spawn_root_position=(*plan.spawn_root_position[:2], -minimum),
        notes=(*plan.notes, "collision capsules fitted to dominant-weight SMPL rest regions"),
    )
    return fitted_plan, audit


def sideways_leg_dofs(
    plan: HumanRigPlan, probe_deg: float = 30.0, min_lever_m_per_rad: float = 0.10
) -> list[str]:
    """Leg DOFs that are *efficient* lateral movers of the ankle, decided by FK.

    An axis letter carries no anatomical promise here: each chain declares its own
    rotation order, ``x`` is a knee's frontal-plane abduction but the same letter is an
    elevation elsewhere in the body, and the rest orientation the letter acts on depends
    on which skeleton the plan was fitted to. So ask the geometry instead of the name,
    and let a config reorder fail loudly as a changed set rather than a silent mislabel.
    """

    rest = forward_kinematics(plan)
    links = {link.name for link in plan.links}
    probe_rad = float(np.deg2rad(probe_deg))
    sideways: list[str] = []
    for joint in plan.joints:
        side, _, part = joint.chain_joint.partition("_")
        ankle = f"{side}_ankle"
        if part not in {"hip", "knee"} or ankle not in links:
            continue
        angles = {name: 0.0 for name in plan.dof_names}
        angles[joint.name] = probe_rad
        moved = forward_kinematics(plan, angles)[ankle].translation - rest[ankle].translation
        if abs(moved[1]) / probe_rad >= min_lever_m_per_rad and abs(moved[1]) > max(
            abs(moved[0]), abs(moved[2])
        ):
            sideways.append(joint.name)
    return sideways


def capsule_bottom(plan: HumanRigPlan, poses: dict[str, LinkTransform], name: str) -> float:
    vertices = plan.link(name).collision_mesh_vertices
    if vertices:
        pose = poses[name]
        return float((np.asarray(vertices) @ pose.rotation[2] + pose.translation[2]).min())
    bounds = plan.link(name).collision_box_bounds
    if bounds is not None:
        low, high = np.asarray(bounds)
        pose = poses[name]
        center = pose.transform_point((low + high) / 2)
        return float(center[2] - np.dot(np.abs(pose.rotation[2]), (high - low) / 2))
    cap = plan.link(name).capsule
    if cap is None:
        raise ValueError(f"{name} has no collision capsule")
    pose = poses[name]
    center = pose.transform_point(cap.center)
    axis = pose.rotation @ cap.direction
    return float(center[2] - abs(axis[2]) * cap.cylinder_length_m / 2 - cap.radius_m)


class StanceFootController:
    """Keep a contacting reference foot at its touchdown pose using joint targets.

    Anchors are released in swing and are never constraints on the physical body.
    An obstructed foot may still fail to track; residuals and actual slip are logged.
    """

    def __init__(
        self,
        plan: HumanRigPlan,
        *,
        height_m: float,
        iterations: int,
        max_correction_rad: float,
        damping: float,
        orientation_weight: float,
        swing_clearance_m: float = 0.06,
        measured_root_feedback: bool = False,
        abduction_weight: float = 1.0,
        release_residual_m: float | None = None,
        support_gain: float = 0.0,
        support_max_m: float = 0.08,
        measured_anchor: bool = False,
        sideways_correction_rad: float | None = None,
    ) -> None:
        values = [height_m, max_correction_rad, damping, orientation_weight]
        if not np.isfinite(swing_clearance_m) or swing_clearance_m < 0:
            raise ValueError("swing clearance must be finite and nonnegative")
        if (
            not np.isfinite(values).all()
            or min(values) <= 0
            or isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations < 1
        ):
            raise ValueError("stance IK settings must be finite and positive")
        if not np.isfinite(abduction_weight) or abduction_weight <= 0:
            raise ValueError("abduction_weight must be finite and positive")
        if release_residual_m is not None and (
            not np.isfinite(release_residual_m) or release_residual_m <= 0
        ):
            raise ValueError("release_residual_m must be finite and positive")
        self.plan = plan
        self.height_m = height_m
        self.iterations = iterations
        self.max_correction_rad = max_correction_rad
        self.damping = damping
        self.orientation_weight = orientation_weight
        self.swing_clearance_m = swing_clearance_m
        if not isinstance(measured_root_feedback, bool):
            raise ValueError("measured_root_feedback must be boolean")
        self.measured_root_feedback = measured_root_feedback
        if (not isinstance(measured_anchor, bool)
                or (measured_anchor and not measured_root_feedback)):
            raise ValueError("measured_anchor requires measured_root_feedback")
        if sideways_correction_rad is not None and (
            not np.isfinite(sideways_correction_rad) or sideways_correction_rad <= 0
        ):
            raise ValueError("sideways correction bound must be finite and positive")
        self.measured_anchor = measured_anchor
        self.abduction_weight = float(abduction_weight)
        self.release_residual_m = release_residual_m
        self.lower = np.deg2rad([j.lower_deg for j in plan.joints])
        self.upper = np.deg2rad([j.upper_deg for j in plan.joints])
        self.indices = {
            side: [
                i
                for i, j in enumerate(plan.joints)
                if j.chain_joint in {f"{side}_hip", f"{side}_knee", f"{side}_ankle"}
            ]
            for side in ("left", "right")
        }
        # A task-space pseudoinverse alone spreads a foot-position error over every leg
        # DOF, and the sideways axes are the cheapest way to move a foot -- which is how
        # a knee ends up splaying toward a split. Weighting the solve in joint space buys
        # the same foot position out of the axes that can hold it anatomically. Which DOFs
        # those are comes from the geometry, not from the axis letter.
        sideways = set(sideways_leg_dofs(plan))
        self.sideways_dofs = sorted(sideways)
        self.correction_bounds = np.array([
            min(max_correction_rad, sideways_correction_rad)
            if sideways_correction_rad is not None and joint.name in sideways
            else max_correction_rad for joint in plan.joints
        ])
        self.weights = {
            side: np.where(
                [plan.joints[i].name in sideways for i in indices], float(abduction_weight), 1.0
            )
            for side, indices in self.indices.items()
        }
        # Cumulative for the session: reset() clears the anchors, not the count of how
        # often the reference asked for a foot it could not keep still.
        self.released_anchors = 0
        self.anchors: dict[str, LinkTransform] = {}
        self.residual_m = 0.0
        # Stance-leg weight support: an integral joint-space extension per leg.
        # The root assist's constant gravity fraction carries most of the weight
        # (measured: actuator 0.82 body weight, ground 0.18), and the per-step
        # anchor budget (0.087 rad, re-derived from the fresh reference each step)
        # can never accumulate the ~0.3 rad a real 4 cm sink needs -- the same
        # structural flaw the swing lift had. The offset grows while the body sits
        # below its target height and decays when it does not, so the stance leg
        # actively pushes the body up and the ground carries its share.
        self.support_gain = float(support_gain)
        self.support_max_m = float(support_max_m)
        # Per-foot anchor lowering (metres). Grows while the body sits below its
        # target height, decays when it does not; the anchor IK then presses the
        # stance foot into the floor and the ground reaction carries the share the
        # constant gravity fraction no longer does.
        self.support_m: dict[str, float] = {}
        if not np.isfinite([support_gain, support_max_m]).all() or support_max_m <= 0:
            raise ValueError("support settings must be finite with a positive cap")
        # Reuse only the two leg chains. Rebuilding all 62 body transforms for
        # each IK iteration made input processing substantially slower than physics.
        by_child = {joint.child_link: (index, joint) for index, joint in enumerate(plan.joints)}
        self.chains = {}
        for side in ("left", "right"):
            names = []
            current = f"{side}_ankle"
            while current != plan.root_link:
                names.append(current)
                current = plan.link(current).parent_link
            self.chains[side] = [(name, by_child[name]) for name in reversed(names)]

    def reset(self) -> None:
        self.anchors.clear()
        self.support_m.clear()
        self.residual_m = 0.0

    def correct(
        self,
        joints: np.ndarray,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        standing: bool,
        support_feet: set[str] | None = None,
        swing_fraction: dict[str, float] | None = None,
        measured_position: np.ndarray | None = None,
        measured_quaternion: np.ndarray | None = None,
        measured_joints: np.ndarray | None = None,
    ) -> np.ndarray:
        q = joints.copy()
        if self.measured_anchor and (
            measured_joints is None or measured_position is None or measured_quaternion is None
            or measured_joints.shape != joints.shape or not np.isfinite(measured_joints).all()
        ):
            raise ValueError("measured anchors require finite measured root and joint states")
        # Body sink below the target height: the support scalar's input. Read
        # before the measured-feedback gate, which only governs the anchor IK's
        # root convention, not the support.
        sink = (
            float(position[2] - measured_position[2])
            if measured_position is not None
            else 0.0
        )
        if not self.measured_root_feedback:
            measured_position = measured_quaternion = None

        def poses_for(
            values: np.ndarray, side: str, *, measured: bool = False
        ) -> dict[str, LinkTransform]:
            root_pos = measured_position if measured and measured_position is not None else position
            root_quat = (
                measured_quaternion if measured and measured_quaternion is not None else quaternion
            )
            poses = {self.plan.root_link: LinkTransform(quaternion_to_matrix(root_quat), root_pos)}
            for name, (index, joint) in self.chains[side]:
                parent = poses[joint.parent_link]
                axis = "xyz".index(joint.axis)
                a, b = (axis + 1) % 3, (axis + 2) % 3
                rotation = np.eye(3)
                cosine, sine = np.cos(values[index]), np.sin(values[index])
                rotation[a, a] = rotation[b, b] = cosine
                rotation[a, b], rotation[b, a] = -sine, sine
                poses[name] = LinkTransform(
                    parent.rotation @ rotation, parent.transform_point(joint.local_pos0)
                )
            return poses

        residuals = []
        for side, indices in self.indices.items():
            foot = f"{side}_ankle"
            poses = poses_for(q, side)
            bottom = capsule_bottom(self.plan, poses, foot)
            swinging = not standing and (
                foot not in support_feet if support_feet is not None else bottom > self.height_m
            )
            if swinging:
                self.anchors.pop(foot, None)
                self.support_m.pop(foot, None)
                fraction = (swing_fraction or {}).get(foot, 0.5)
                clearance = self.swing_clearance_m * np.sin(np.pi * fraction)
                translation = poses[foot].translation.copy()
                translation[2] += max(0.0, clearance - bottom)
                anchor = LinkTransform(poses[foot].rotation.copy(), translation)
            elif foot not in self.anchors:
                self.support_m[foot] = 0.0
                touchdown = poses_for(measured_joints, side, measured=True) if (
                    self.measured_anchor) else poses
                translation = touchdown[foot].translation.copy()
                translation[2] -= capsule_bottom(self.plan, touchdown, foot)
                self.anchors[foot] = LinkTransform(touchdown[foot].rotation.copy(), translation)
                anchor = self.anchors[foot]
            else:
                if self.support_gain > 0:
                    grown = self.support_m.get(foot, 0.0)
                    if sink > 0:
                        grown = min(grown + self.support_gain * sink, self.support_max_m)
                    else:
                        grown = max(0.0, grown * 0.97)
                    self.support_m[foot] = grown
                anchor_translation = self.anchors[foot].translation.copy()
                anchor_translation[2] -= self.support_m.get(foot, 0.0)
                anchor = LinkTransform(self.anchors[foot].rotation.copy(), anchor_translation)
            for _ in range(self.iterations):
                poses = poses_for(q, side, measured=True)
                actual = poses[foot]
                error = np.r_[
                    anchor.translation - actual.translation,
                    self.orientation_weight
                    * matrix_to_axis_angle(anchor.rotation @ actual.rotation.T),
                ]
                jacobian = np.zeros((6, len(indices)))
                for col, index in enumerate(indices):
                    joint = self.plan.joints[index]
                    parent = poses[joint.parent_link]
                    axis = parent.rotation[:, "xyz".index(joint.axis)]
                    origin = parent.transform_point(joint.local_pos0)
                    jacobian[:3, col] = np.cross(axis, actual.translation - origin)
                    jacobian[3:, col] = self.orientation_weight * axis
                # Weighted least norms: delta = W^-1 J^T (J W^-1 J^T + lambda I)^-1 e,
                # with W = diag(weights). A weight above 1 makes that DOF pay more for
                # the same task-space error, so the solve routes around it.
                scale = 1.0 / np.sqrt(self.weights[side])
                weighted = jacobian * scale
                delta = (
                    weighted.T
                    @ np.linalg.solve(
                        weighted @ weighted.T + self.damping * np.eye(6),
                        error,
                    )
                ) * scale
                q[indices] = np.clip(
                    q[indices] + delta,
                    np.maximum(self.lower[indices],
                               joints[indices] - self.correction_bounds[indices]),
                    np.minimum(self.upper[indices],
                               joints[indices] + self.correction_bounds[indices]),
                )
            residual = float(
                np.linalg.norm(
                    anchor.translation - poses_for(q, side, measured=True)[foot].translation
                )
            )
            residuals.append(residual)
            # An anchor the leg cannot reach without dislocating a joint is a reference
            # defect, not a tracking target. Letting go reports it as slip instead of
            # spending joint angle on it.
            if (
                self.release_residual_m is not None
                and residual > self.release_residual_m
                and foot in self.anchors
            ):
                del self.anchors[foot]
                self.released_anchors += 1
        self.residual_m = max(residuals, default=0.0)
        return q


FOOT_SEGMENTS = ("left_ankle", "right_ankle", "left_foot", "right_foot")


def measured_support_feet(
    attributed: Any,
    *,
    floor_suffix: str = "/floor",
    min_impulse_ns: float,
) -> set[str]:
    """The feet the simulator reports standing on the floor, from one contact report read.

    ``load_gait`` decides which foot may be anchored from horizontal ankle travel in forward
    kinematics -- a guess about contact, made before the simulation runs. Measured against
    PhysX on four archived sessions that guess is wrong in both directions: it misses 46-56%
    of the right foot's real floor contacts and 10-31% of the left's, while 16-28% of the
    feet it calls support are not touching (see docs/mask-vs-contact-audit.md). Contact is
    the thing an anchor needs, and the simulator already reports it, so use it.

    ``attributed`` is ``HumanRuntime.attributed_contact_samples()``: pairs of contact sample
    and the limb the contact landed on. Duck-typed on attributes so this stays CPU-testable
    without importing the runtime.

    Returns the set of foot segments in floor contact above ``min_impulse_ns``. An empty set
    is a real answer -- nothing is standing on the floor -- and the caller should honour it
    rather than falling back, or the fallback is what gets anchored.
    """

    if not np.isfinite(min_impulse_ns) or min_impulse_ns < 0:
        raise ValueError("min_impulse_ns must be finite and non-negative")
    contact: set[str] = set()
    for sample, limb in attributed:
        if limb not in FOOT_SEGMENTS:
            continue
        paths = (getattr(sample, "collider0_path", ""), getattr(sample, "collider1_path", ""))
        if not any(path.endswith(floor_suffix) for path in paths):
            continue
        if float(sample.impulse_magnitude_ns) < min_impulse_ns:
            continue
        contact.add("left_ankle" if limb.startswith("left") else "right_ankle")
    return contact
