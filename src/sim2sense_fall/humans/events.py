"""Label fall events from a recorded trajectory, not from the perturbation schedule.

The rule this module exists to enforce is stated in the stage plan: *applying a
perturbation does not make a fall*. A trial where the figure is pushed and recovers
is not a fall; a trial where the figure collapses with no perturbation at all is
one. So labels are derived only from what the body actually did.

Definitions, applied identically to every trial, with the thresholds taken from
``events`` in the human configuration:

``imbalance_onset``
    The first time the body's upright criterion is lost: the trunk tilts past
    ``trunk_angle_deg``, or the pelvis drops below ``pelvis_height_fraction`` of
    its standing height. This is *not yet* a fall -- recovery is still possible,
    and a recovery is reported as such.
``first_impact``
    The first time, **after the onset**, that one and the same body point is within
    ``impact_height_m`` of the supporting surface, moving down into it faster than
    ``impact_speed_m_s``, and was clear of that surface earlier in the trial. The last
    clause is what keeps a body that starts lying down from reporting an impact in the
    first frame; searching from the onset is what keeps a walking foot, which touches
    down faster than the threshold on every step, from doing the same. Height and speed
    are read from the same point, because the lowest point and the fastest point are
    normally different points moving in different directions. A controlled lie-down
    reaches the floor slowly and produces no impact, and a trial with no onset has no
    impact at all.
``stabilisation``
    The first time after the impact that body-point speed stays below
    ``settle_speed_m_s`` for a full ``settle_window_s`` window.

A trial is labelled ``fall`` when a low posture is reached quickly enough
(``max_transition_s``), it persists for ``min_low_frames``, and an impact was seen.
It is labelled ``lying_controlled`` when the low posture is reached slowly, with no
impact and a descent speed under ``controlled_descent_speed_m_s``. Anything else is
``no_fall``. Trials that diverged or drove geometry through the floor are marked
invalid and excluded, rather than being labelled.

These are heuristics. They are not validated against labelled real human falls and
must not be described as clinically validated.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EventsConfig
from .motion import PHASE_LABELS
from .rotations import quaternion_to_matrix

__all__ = [
    "LABEL_FALL",
    "LABEL_INVALID",
    "LABEL_CONTROLLED_LOWERING",
    "LABEL_NO_FALL",
    "LABEL_RECOVERED",
    "Trajectory",
    "TrialLabel",
    "label_trial",
]

LOGGER = logging.getLogger(__name__)

LABEL_FALL = "fall"
LABEL_NO_FALL = "no_fall"
LABEL_RECOVERED = "recovered"
LABEL_CONTROLLED_LOWERING = "controlled_lowering"
LABEL_INVALID = "invalid"

_UP_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, slots=True, eq=False)
class Trajectory:
    """A recorded trial: root motion, joint state and a body-point cloud per frame.

    Everything an event decision uses comes from here. ``body_points`` is the
    world-space proxy surface sampled in the export, so "lowest body point" means
    the lowest physical geometry rather than the lowest joint centre.

    ``trunk_axis`` is the unit pelvis-to-neck direction per frame. It is optional,
    but without it the trunk tilt falls back to the **root** orientation, which
    cannot see a forward bend: a person bent at the waist with an upright pelvis has
    a zero root tilt, so the fall/non-fall boundary would call that pose upright.
    Callers that have link poses should always supply it.
    """

    times_s: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_positions: np.ndarray
    joint_names: tuple[str, ...]
    body_points: np.ndarray
    standing_height_m: float
    standing_pelvis_height_m: float | None = None
    trunk_axis: np.ndarray | None = None
    contact_force_n: np.ndarray | None = None

    def __post_init__(self) -> None:
        frame_count = int(np.shape(self.times_s)[0])
        for name, array, shape in (
            ("times_s", self.times_s, (frame_count,)),
            ("root_position", self.root_position, (frame_count, 3)),
            ("root_quaternion", self.root_quaternion, (frame_count, 4)),
            ("joint_positions", self.joint_positions, (frame_count, len(self.joint_names))),
        ):
            actual = np.shape(array)
            if actual != shape:
                raise ValueError(f"trajectory {name} must be {shape}, got {actual}")
        if self.body_points.ndim != 3 or self.body_points.shape[0] != frame_count:
            raise ValueError(
                f"trajectory body_points must be (N, P, 3), got {self.body_points.shape}"
            )
        if self.body_points.shape[2] != 3:
            raise ValueError("trajectory body_points must have three coordinates per point")
        if self.trunk_axis is not None:
            if np.shape(self.trunk_axis) != (frame_count, 3):
                raise ValueError(
                    f"trajectory trunk_axis must be ({frame_count}, 3), got "
                    f"{np.shape(self.trunk_axis)}"
                )
            if not np.all(np.isfinite(self.trunk_axis)):
                raise ValueError("trajectory trunk_axis contains a non-finite value")
            if np.any(np.linalg.norm(np.asarray(self.trunk_axis), axis=1) < 1e-9):
                raise ValueError(
                    "trajectory trunk_axis contains a zero-length axis; the pelvis and neck "
                    "links cannot coincide"
                )
        if self.contact_force_n is not None and np.shape(self.contact_force_n) != (frame_count,):
            raise ValueError("trajectory contact_force_n must be (N,)")
        if frame_count < 2:
            raise ValueError("a trajectory needs at least two frames")
        for name, array in (
            ("times_s", self.times_s),
            ("root_position", self.root_position),
            ("root_quaternion", self.root_quaternion),
            ("joint_positions", self.joint_positions),
            ("body_points", self.body_points),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"trajectory {name} contains a non-finite value")
        if np.any(np.diff(self.times_s) <= 0):
            raise ValueError("trajectory times must strictly increase")
        if self.standing_height_m <= 0:
            raise ValueError("trajectory standing_height_m must be positive")
        if self.standing_pelvis_height_m is not None and self.standing_pelvis_height_m <= 0:
            raise ValueError("trajectory standing_pelvis_height_m must be positive when given")
        if self.contact_force_n is not None and np.any(np.asarray(self.contact_force_n) < 0):
            raise ValueError("trajectory contact forces must be non-negative")

    @property
    def frame_count(self) -> int:
        return int(self.times_s.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / float(np.mean(np.diff(self.times_s)))

    @property
    def pelvis_height_m(self) -> np.ndarray:
        return self.root_position[:, 2]

    @property
    def reference_pelvis_height_m(self) -> float:
        """Pelvis height of the upright reference pose.

        Falls back to the first frame of the trial only when the caller did not state
        it, which is a weaker reference: a trial that starts already perturbed would
        then measure its own drop from a lowered pose.
        """

        if self.standing_pelvis_height_m is not None:
            return float(self.standing_pelvis_height_m)
        return float(self.root_position[0, 2])

    @property
    def lowest_point_z(self) -> np.ndarray:
        return self.body_points[:, :, 2].min(axis=1)

    @property
    def highest_point_z(self) -> np.ndarray:
        return self.body_points[:, :, 2].max(axis=1)

    @property
    def trunk_angle_deg(self) -> np.ndarray:
        """Angle between the trunk axis and world up, in degrees.

        Uses the recorded pelvis-to-neck axis when available, so a bend at the waist
        is visible even though the pelvis orientation has not changed.
        """

        if self.trunk_axis is not None:
            axis = np.asarray(self.trunk_axis, dtype=np.float64)
            norms = np.linalg.norm(axis, axis=1, keepdims=True)
            axis = axis / np.where(norms > 0, norms, 1.0)
        else:
            rotations = np.stack(
                [quaternion_to_matrix(row) for row in self.root_quaternion], axis=0
            )
            axis = rotations @ _UP_AXIS
        cosine = np.clip(axis @ _UP_AXIS, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    @property
    def speed_m_s(self) -> np.ndarray:
        """Body-point speed magnitude, in m/s, one entry per frame."""

        deltas = np.linalg.norm(np.diff(self.body_points, axis=0), axis=-1).max(axis=1)
        steps = np.diff(self.times_s)
        speed = np.zeros(self.frame_count)
        speed[1:] = deltas / np.where(steps > 0, steps, 1.0)
        speed[0] = speed[1]
        return speed

    @property
    def root_speed_m_s(self) -> np.ndarray:
        steps = np.diff(self.times_s)
        speed = np.zeros(self.frame_count)
        speed[1:] = np.linalg.norm(np.diff(self.root_position, axis=0), axis=-1) / np.where(
            steps > 0, steps, 1.0
        )
        speed[0] = speed[1]
        return speed

    @property
    def descent_speed_m_s(self) -> np.ndarray:
        """Downward root speed, positive when the pelvis is dropping."""

        steps = np.diff(self.times_s)
        drop = np.zeros(self.frame_count)
        drop[1:] = -np.diff(self.root_position[:, 2]) / np.where(steps > 0, steps, 1.0)
        drop[0] = drop[1]
        return np.clip(drop, 0.0, None)

    @property
    def point_descent_speed_m_s(self) -> np.ndarray:
        """Downward speed of EVERY body point, shaped ``(frames, points)``.

        Positive means that point is falling. Impact detection needs this per point rather
        than one whole-body number, because "the lowest point near the floor" and "the
        fastest point anywhere" are usually different points moving in different
        directions -- the stage-7 review built its counterexample out of that mixing.
        """

        steps = np.diff(self.times_s)[:, None]
        drop = np.zeros((self.frame_count, self.body_points.shape[1]))
        drop[1:] = -np.diff(self.body_points[:, :, 2], axis=0) / np.where(steps > 0, steps, 1.0)
        drop[0] = drop[1]
        return np.clip(drop, 0.0, None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "fps": round(self.fps, 6),
            "duration_s": round(float(self.times_s[-1] - self.times_s[0]), 6),
            "joint_count": len(self.joint_names),
            "body_point_count": int(self.body_points.shape[1]),
            "standing_height_m": self.standing_height_m,
            "peak_trunk_angle_deg": round(float(self.trunk_angle_deg.max()), 4),
            "min_pelvis_height_m": round(float(self.pelvis_height_m.min()), 6),
            "min_body_point_z": round(float(self.lowest_point_z.min()), 6),
            "peak_speed_m_s": round(float(self.speed_m_s.max()), 6),
            "peak_descent_speed_m_s": round(float(self.descent_speed_m_s.max()), 6),
            "max_travel_m": round(
                float(np.linalg.norm(self.root_position[-1, :2] - self.root_position[0, :2])), 6
            ),
        }


@dataclass(frozen=True, slots=True)
class TrialLabel:
    """The outcome of one trial, with every quantity the decision used."""

    label: str
    valid: bool
    final_posture: str
    imbalance_onset_s: float | None
    first_impact_s: float | None
    stabilisation_s: float | None
    time_to_low_posture_s: float | None
    peak_trunk_angle_deg: float
    final_trunk_angle_deg: float
    peak_descent_speed_m_s: float
    min_pelvis_height_m: float
    min_body_point_z_m: float
    standing_height_m: float
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.final_posture not in ("upright", "lying"):
            raise ValueError(
                f"final_posture must be 'upright' or 'lying', got {self.final_posture!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "valid": self.valid,
            "final_posture": self.final_posture,
            "imbalance_onset_s": self.imbalance_onset_s,
            "first_impact_s": self.first_impact_s,
            "stabilisation_s": self.stabilisation_s,
            "time_to_low_posture_s": self.time_to_low_posture_s,
            "peak_trunk_angle_deg": self.peak_trunk_angle_deg,
            "final_trunk_angle_deg": self.final_trunk_angle_deg,
            "peak_descent_speed_m_s": self.peak_descent_speed_m_s,
            "min_pelvis_height_m": self.min_pelvis_height_m,
            "min_body_point_z_m": self.min_body_point_z_m,
            "standing_height_m": self.standing_height_m,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def trunk_axis_from_link_positions(
    link_names: Sequence[str],
    link_positions: np.ndarray,
    *,
    pelvis: str = "pelvis",
    neck: str = "neck",
    fallback: str = "spine3",
) -> np.ndarray:
    """Pelvis-to-neck unit vectors per frame, from recorded link positions.

    Falls back to ``spine3`` when the topology has no ``neck``, and raises rather
    than silently returning a constant axis when neither exists.
    """

    names = list(link_names)
    if np.shape(link_positions)[1] != len(names):
        raise ValueError(
            f"link_positions has {np.shape(link_positions)[1]} links but {len(names)} names"
        )
    if pelvis not in names:
        raise ValueError(f"link positions have no {pelvis!r} link")
    head = neck if neck in names else (fallback if fallback in names else None)
    if head is None:
        raise ValueError(
            f"link positions have neither {neck!r} nor {fallback!r}; a trunk axis cannot be derived"
        )
    axis = (
        np.asarray(link_positions)[:, names.index(head), :]
        - np.asarray(link_positions)[:, names.index(pelvis), :]
    )
    norms = np.linalg.norm(axis, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        raise ValueError(f"{pelvis!r} and {head!r} coincide in at least one frame")
    return axis / norms


def _first_index(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def _persistent_index(mask: np.ndarray, run_length: int) -> int | None:
    """First index where ``mask`` is true for ``run_length`` consecutive frames."""

    if run_length <= 1:
        return _first_index(mask)
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= run_length:
            return index - run_length + 1
    return None


def label_trial(
    trajectory: Trajectory, criteria: EventsConfig, *, floor_z: float = 0.0
) -> TrialLabel:
    """Apply the pre-registered rules to one recorded trajectory.

    ``floor_z`` is the world height of the supporting surface, so the same rules
    work for a scene whose floor is not at the origin.
    """

    if not isinstance(criteria, EventsConfig):
        raise ValueError("label_trial needs an EventsConfig")
    heights = trajectory.standing_height_m
    # The fraction applies to the standing PELVIS height, not to the stature. Against
    # stature the 0.55 threshold lands at 0.94 of the upright pelvis height, so an
    # ordinary 6 cm sag under gravity counted as "posture lost" and an unperturbed
    # standing trial came back labelled as a recovery.
    pelvis_low_threshold = criteria.pelvis_height_fraction * trajectory.reference_pelvis_height_m
    trunk_angle = trajectory.trunk_angle_deg
    # Two separate trunk thresholds, so "started to tip" and "was falling" are not
    # the same event. The transition time is measured between them.
    tipping = trunk_angle >= criteria.trunk_onset_fraction * criteria.trunk_angle_deg
    trunk_long = trunk_angle >= criteria.trunk_angle_deg
    pelvis_low = trajectory.pelvis_height_m <= pelvis_low_threshold
    disturbed = tipping | pelvis_low
    low_posture = trunk_long | pelvis_low

    onset_index = _first_index(disturbed)
    onset = None if onset_index is None else float(trajectory.times_s[onset_index])
    confirmed_index = _persistent_index(low_posture, criteria.min_low_frames)

    # Impact: the first time, after the onset, that ONE AND THE SAME body point is both
    # near the supporting surface and moving down into it fast. The previous version took
    # the lowest point's height from one point and the speed maximum from another, so a
    # slow 0.001 m/s descent at 0.1 m plus an unrelated 2 m/s horizontal swing at 0.5 m
    # produced a "fall" with an impact in the first frame. Searching from the onset still
    # matters, because a walking foot touches down faster than the impact threshold on
    # every step.
    # ``point_heights`` rather than ``heights``: the latter is the scalar standing height
    # used by the labelling rules below, and shadowing it here would quietly turn every
    # later height comparison into an array broadcast.
    point_heights = trajectory.body_points[:, :, 2] - floor_z
    point_down = trajectory.point_descent_speed_m_s
    impact_height = float(criteria.impact_height_m)
    near_surface = point_heights <= impact_height
    approaching = np.zeros_like(near_surface)
    # logical_or.accumulate rather than cummax: the latter is absent from the NumPy this
    # project still supports, and a prefix "was ever above" is all the test needs.
    approaching[1:] = np.logical_or.accumulate(point_heights[:-1] > impact_height, axis=0)
    impact_point = near_surface & approaching & (point_down >= criteria.impact_speed_m_s)
    if onset_index is not None:
        impact_point[:onset_index] = False
    impact_mask = impact_point.any(axis=1)
    impact_index = _first_index(impact_mask) if onset_index is not None else None
    impact_time = None if impact_index is None else float(trajectory.times_s[impact_index])
    impact_point_index = (
        None
        if impact_index is None
        else int(np.argmax(impact_point[impact_index] * point_down[impact_index]))
    )

    # Stabilisation: speed stays low for a full window, measured after the impact if
    # there was one, otherwise over the whole trial.
    search_from = impact_index if impact_index is not None else 0
    settle_index = None
    window_frames = max(1, int(round(criteria.settle_window_s * trajectory.fps)))
    settled = trajectory.speed_m_s <= criteria.settle_speed_m_s
    for index in range(search_from, max(search_from, trajectory.frame_count - window_frames)):
        if bool(settled[index : index + window_frames].all()):
            settle_index = index
            break
    stabilisation = None if settle_index is None else float(trajectory.times_s[settle_index])

    transition = None
    if confirmed_index is not None and onset is not None:
        transition = float(trajectory.times_s[confirmed_index] - trajectory.times_s[onset_index])

    peak_trunk = float(trunk_angle.max())
    final_trunk = float(trunk_angle[-1])
    peak_descent = float(trajectory.descent_speed_m_s.max())
    min_pelvis = float(trajectory.pelvis_height_m.min())
    min_point = float((trajectory.lowest_point_z - floor_z).min())
    final_height = float(trajectory.highest_point_z[-1] - floor_z)
    travel = float(
        np.linalg.norm(trajectory.root_position[-1, :2] - trajectory.root_position[0, :2])
    )

    reasons: list[str] = []
    invalid = False
    if not math.isfinite(peak_trunk) or not math.isfinite(min_point):
        reasons.append("non-finite state in the trajectory")
        invalid = True
    if travel > criteria.divergence_limit_m:
        reasons.append(
            f"root travelled {travel:.2f} m, beyond the {criteria.divergence_limit_m:.2f} m "
            "divergence limit"
        )
        invalid = True
    if min_point < criteria.penetration_limit_m:
        reasons.append(
            f"body geometry reached {min_point:.3f} m below the floor, beyond the "
            f"{criteria.penetration_limit_m:.3f} m penetration limit"
        )
        invalid = True

    metrics: dict[str, Any] = {
        "trunk_angle_threshold_deg": criteria.trunk_angle_deg,
        "trunk_onset_threshold_deg": round(
            criteria.trunk_onset_fraction * criteria.trunk_angle_deg, 6
        ),
        "pelvis_height_threshold_m": round(pelvis_low_threshold, 6),
        "reference_pelvis_height_m": round(trajectory.reference_pelvis_height_m, 6),
        "low_posture_frame_count": int(low_posture.sum()),
        "impact_frame_count": int(impact_mask.sum()),
        "impact_body_point_index": impact_point_index,
        "impact_point_height_m": None
        if impact_index is None or impact_point_index is None
        else round(float(point_heights[impact_index, impact_point_index]), 6),
        "impact_point_down_speed_m_s": None
        if impact_index is None or impact_point_index is None
        else round(float(point_down[impact_index, impact_point_index]), 6),
        # A proxy event is geometry-based inference; a measured contact is a force the
        # solver reported. Only the second may be called contact ground truth.
        "contact_evidence": "none"
        if impact_index is None
        else (
            "measured_contact"
            if trajectory.contact_force_n is not None
            and bool(np.any(np.asarray(trajectory.contact_force_n)[impact_index:] > 0.0))
            else "proxy_only"
        ),
        "settle_window_frames": window_frames,
        "final_height_m": round(final_height, 6),
        "final_trunk_angle_deg": round(final_trunk, 6),
        "used_trunk_axis_from_links": trajectory.trunk_axis is not None,
        "root_travel_m": round(travel, 6),
        "used_contact_forces": trajectory.contact_force_n is not None,
    }

    def build(label: str, valid: bool) -> TrialLabel:
        return TrialLabel(
            label=label,
            valid=valid,
            final_posture=("lying" if final_trunk >= criteria.trunk_angle_deg else "upright"),
            imbalance_onset_s=onset,
            first_impact_s=impact_time,
            stabilisation_s=stabilisation,
            time_to_low_posture_s=transition,
            peak_trunk_angle_deg=peak_trunk,
            final_trunk_angle_deg=final_trunk,
            peak_descent_speed_m_s=peak_descent,
            min_pelvis_height_m=min_pelvis,
            min_body_point_z_m=min_point,
            standing_height_m=heights,
            reasons=tuple(reasons),
            metrics=metrics,
        )

    if invalid:
        return build(LABEL_INVALID, False)
    if onset is None:
        reasons.append(
            "the trunk never passed "
            f"{criteria.trunk_onset_fraction * criteria.trunk_angle_deg:.1f} degrees and the "
            f"pelvis never dropped below {pelvis_low_threshold:.2f} m: upright posture was "
            "never disturbed"
        )
        return build(LABEL_NO_FALL, True)
    if confirmed_index is None:
        if final_trunk < criteria.trunk_onset_fraction * criteria.trunk_angle_deg:
            reasons.append(
                "the body tipped past the onset threshold and then returned to upright: the "
                "disturbance was recovered from before any low posture was held"
            )
            return build(LABEL_RECOVERED, True)
        reasons.append(
            "the body tipped past the onset threshold and stayed tipped, but never held a low "
            f"posture for {criteria.min_low_frames} frames: a partial topple, not a fall"
        )
        return build(LABEL_NO_FALL, True)
    if impact_time is None:
        if peak_descent <= criteria.controlled_descent_speed_m_s:
            reasons.append(
                "a low posture was held, but with a peak descent of only "
                f"{peak_descent:.2f} m/s and no body point within "
                f"{criteria.impact_height_m:g} m of the floor moving faster than "
                f"{criteria.impact_speed_m_s:g} m/s: a controlled lowering"
            )
            return build(LABEL_CONTROLLED_LOWERING, True)
        reasons.append(
            "a low posture was held and the descent exceeded "
            f"{criteria.controlled_descent_speed_m_s:g} m/s, but no body point came within "
            f"{criteria.impact_height_m:g} m of the floor while descending faster than "
            f"{criteria.impact_speed_m_s:g} m/s"
        )
        return build(LABEL_NO_FALL, True)
    if transition is not None and transition > criteria.max_transition_s:
        reasons.append(
            f"the low posture took {transition:.2f} s to reach from the first tip, beyond "
            f"the {criteria.max_transition_s:g} s bound for a fall"
        )
        return build(LABEL_CONTROLLED_LOWERING, True)
    if final_height > criteria.recovery_height_fraction * heights:
        reasons.append(
            f"the body ended {final_height:.2f} m tall, above "
            f"{criteria.recovery_height_fraction:.0%} of standing height: the disturbance "
            "was recovered from"
        )
        return build(LABEL_RECOVERED, True)
    reasons.append(
        f"posture disturbed at {onset:.2f} s, impact at {impact_time:.2f} s after "
        f"{transition if transition is not None else float('nan'):.2f} s, and the body "
        f"finished {final_height:.2f} m tall"
    )
    LOGGER.info(
        "trial labelled %s: onset=%s impact=%s stabilisation=%s",
        LABEL_FALL,
        onset,
        impact_time,
        stabilisation,
    )
    return build(LABEL_FALL, True)


def phase_labels_from_signal(
    times_s: Sequence[float],
    *,
    standing_frames: int,
    onset_s: float | None,
    impact_s: float | None,
    stabilisation_s: float | None,
    final_phase: str,
) -> tuple[str, ...]:
    """Per-frame phase labels derived from the detected events.

    ``standing`` before onset, ``falling`` from onset to impact, ``fallen`` from the
    impact until stabilisation, then ``final_phase`` for the rest of the trial -- one
    of ``standing``, ``standing_recovery`` or ``fallen``. A trial with no onset is
    ``standing`` throughout. The labels summarise the detected events; they are not
    an independent annotation.
    """

    if standing_frames < 0:
        raise ValueError("standing_frames must be non-negative")
    if final_phase not in PHASE_LABELS:
        raise ValueError(f"final_phase must be one of {list(PHASE_LABELS)}, got {final_phase!r}")
    labels: list[str] = []
    for index, time in enumerate(times_s):
        if index < standing_frames or onset_s is None or time < onset_s:
            labels.append("standing")
        elif impact_s is None or time < impact_s:
            labels.append("falling")
        elif stabilisation_s is None or time < stabilisation_s:
            labels.append("fallen")
        else:
            labels.append(final_phase)
    if not labels:
        raise ValueError("phase labelling needs at least one frame")
    return tuple(labels)
