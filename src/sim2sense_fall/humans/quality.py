"""Measured motion gates, separated by activity so idle frames cannot hide failures.

Two support classes are measured, and both sets of numbers are always reported so
a class change can never hide a failure:

* **foot class** (walk/stand/crouch/bend and their transitions): the feet are the
  only legitimate support, so ``left/right ankle|foot`` contacts define support and
  the 5 mm skin gap and 0.15 m/s slip gates apply unchanged.
* **low-posture class** (floor recovery ``getting_up``, floor sit ``sit``/``sitting``):
  the body rests on hips, hands, elbows or knees *by design*. Counting only foot
  contacts for these measures "did the feet happen to touch", which passes a body
  carried on the pelvis actuator and fails a body kneeling on the floor. This class
  gates on a primary support link, a whole-body grounding envelope, and a load share
  the floor actually carries.

The class definitions and the thresholds below were pre-registered on 2026-09-27 from
the measured failure, before any re-run: the shipped get-up had 2.8% / 0.1% left/right
foot contact, a median floor normal force of **0 N**, and the root actuator pinned at
its 1.0-body-weight ceiling for 637 of 888 frames (`artifacts/diag_getup_slerp`,
reproduced by `scripts/humans/diagnose_getup_float.py`). See docs/human-simulation.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Bumped when the report's meaning changes, not when it gains a field: the exporter
# refuses a session whose report was produced by a different schema, so a runner and
# an exporter that drifted apart cannot silently admit data. 1 -> 2 added the
# low-posture support class (primary contacts, ground load, whole-body grounding).
SCHEMA_VERSION = 2

HUMAN_COLLIDER_PREFIX = "/World/Human/"
# Link-name fragments that may legitimately carry load in the low-posture class.
DEFAULT_PRIMARY_SUPPORT_LINKS = (
    "ankle", "foot", "knee", "hip", "hand", "wrist", "elbow", "pelvis", "spine",
)
DEFAULT_LOW_POSTURE_MODES = ("getting_up", "sitting", "sit")
_NUMERIC_FIELDS = (
    "support_skin_gap_m", "max_no_support_s", "slip_speed_m_s", "skin_penetration_m",
    "joint_error_deg", "min_contact_impulse_ns", "floor_z_m", "low_posture_skin_gap_m",
    "low_posture_slip_speed_m_s", "low_posture_joint_error_deg", "min_ground_load_fraction",
)
_STRICTLY_POSITIVE = tuple(name for name in _NUMERIC_FIELDS if name != "floor_z_m")


@dataclass(frozen=True, slots=True)
class MotionQualityConfig:
    support_skin_gap_m: float = 0.005
    max_no_support_s: float = 0.05
    slip_speed_m_s: float = 0.15
    skin_penetration_m: float = 0.005
    joint_error_deg: float = 15.0
    min_contact_impulse_ns: float = 0.01
    floor_z_m: float = 0.0
    # --- low-posture support class, pre-registered 2026-09-27 (see module docstring) ---
    low_posture_modes: tuple[str, ...] = DEFAULT_LOW_POSTURE_MODES
    primary_support_links: tuple[str, ...] = DEFAULT_PRIMARY_SUPPORT_LINKS
    # Whole-body lowest skinned point. A recovery lifts and replants hands, so a 2 cm
    # envelope replaces the 5 mm foot rule for this class only.
    low_posture_skin_gap_m: float = 0.02
    # Bare hands and knees pivot on the floor without a shoe's friction budget. The
    # foot gate stays at 0.15 m/s and this one is declared separately, not merged in.
    low_posture_slip_speed_m_s: float = 0.25
    # A contact-constrained limb cannot track a free reference exactly: recovery is
    # allowed 30 degrees where standing keeps 15.
    low_posture_joint_error_deg: float = 30.0
    # The floor must carry at least this fraction of body weight in every activity.
    # This is the "is it standing on something" gate; the actuator's own lift
    # fraction is reported beside it so the two cannot be confused.
    min_ground_load_fraction: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "low_posture_modes",
                           tuple(str(mode) for mode in self.low_posture_modes))
        object.__setattr__(self, "primary_support_links",
                           tuple(str(link) for link in self.primary_support_links))
        for name in _NUMERIC_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"invalid motion quality setting {name}")
            object.__setattr__(self, name, value)
        for name in _STRICTLY_POSITIVE:
            if getattr(self, name) <= 0:
                raise ValueError(f"invalid motion quality setting {name}")
        if not self.primary_support_links:
            raise ValueError("primary_support_links must not be empty")
        if any(not mode for mode in self.low_posture_modes):
            raise ValueError("low_posture_modes must not contain empty names")
        if not 0 < self.min_ground_load_fraction <= 1:
            raise ValueError("min_ground_load_fraction must be in (0, 1]")


def human_link_from_path(path: str) -> str | None:
    """Return the body link a human collider path belongs to, or None."""

    if not path.startswith(HUMAN_COLLIDER_PREFIX):
        return None
    return path[len(HUMAN_COLLIDER_PREFIX):].split("/")[0] or None


def is_primary_support(link: str, support_links: tuple[str, ...]) -> bool:
    """True when ``link`` may carry load (``left_knee`` matches the ``knee`` fragment)."""

    return any(fragment in link for fragment in support_links)


def longest_interval(mask: np.ndarray, time_s: np.ndarray, dt_s: float) -> float:
    """Longest contiguous true interval, breaking on missing frames."""
    longest = current = 0.0
    previous = -np.inf
    for flag, t in zip(mask, time_s, strict=True):
        if not flag:
            current = 0.0
        else:
            current = current + dt_s if t - previous <= 1.5 * dt_s else dt_s
            longest = max(longest, current)
        previous = t
    return float(longest)


def motion_quality(
    rows: list[dict[str, Any]], mesh_time_s: np.ndarray, foot_min_z_m: np.ndarray,
    *, config: MotionQualityConfig, dt_s: float, mass_kg: float, gravity_m_s2: float,
    skin_min_z_m: np.ndarray | None = None,
) -> dict[str, Any]:
    """Consume actual contact reports and actual skinned geometry, never reference FK.

    Each skin timestamp is associated with the preceding pre-step control row.
    Contact membership consequently has at most one physics step of timing offset.
    Falling/fallen do not require stance contact or reference tracking.
    ``skin_min_z_m`` (whole-body lowest skinned vertex per mesh frame) enables the
    low-posture grounding gate; without it that gate is reported as unavailable and
    fails, rather than being silently passed.
    """
    if not rows or not len(mesh_time_s):
        return {"schema_version": SCHEMA_VERSION, "accepted": False, "modes": {},
                "reason": "no samples"}
    if (foot_min_z_m.shape != (len(mesh_time_s), 2)
            or not np.isfinite(foot_min_z_m).all() or not np.isfinite(mesh_time_s).all()
            or np.any(np.diff(mesh_time_s) <= 0)
            or not np.isfinite([dt_s, mass_kg, gravity_m_s2]).all()
            or min(dt_s, mass_kg, gravity_m_s2) <= 0):
        raise ValueError("invalid measured motion samples")
    if skin_min_z_m is not None and (np.shape(skin_min_z_m) != np.shape(mesh_time_s)
                                     or not np.isfinite(skin_min_z_m).all()):
        raise ValueError("skin_min_z_m must hold one finite height per mesh frame")
    times = np.array([r["time_s"] for r in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("control timestamps must increase")
    modes = np.array([r["mode"] for r in rows])
    weight_n = mass_kg * gravity_m_s2
    contact = np.zeros((len(rows), 2), dtype=bool)
    supported = np.zeros(len(rows), dtype=bool)
    normal_n = np.zeros(len(rows))
    support_links_seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        for sample in row["contact_detail"]:
            paths = (sample["collider0_path"], sample["collider1_path"])
            if not any(p.endswith("/floor") for p in paths):
                continue
            normal_n[i] += max(0., float(np.dot(sample["impulse_ns"], sample["normal"]))) / dt_s
            if sample["impulse_magnitude_ns"] < config.min_contact_impulse_ns:
                continue
            for j, side in enumerate(("left", "right")):
                contact[i, j] |= any(f"/{side}_ankle/" in p or f"/{side}_foot/" in p for p in paths)
            for path in paths:
                link = human_link_from_path(path)
                if link and is_primary_support(link, config.primary_support_links):
                    supported[i] = True
                    support_links_seen[link] = support_links_seen.get(link, 0) + 1
    indices = np.clip(np.searchsorted(times, mesh_time_s, side="right") - 1, 0, len(times)-1)
    summary: dict[str, Any] = {}
    selections = {mode: modes == mode for mode in sorted(set(modes) - {"falling", "fallen"})}
    commands = np.array([r.get("command", (0., 0.)) for r in rows])
    for mode in ("forward", "backward"):
        for direction, sign in (("left", 1), ("right", -1)):
            selected = (modes == mode) & (commands[:, 0] != 0) & (commands[:, 1] * sign > 0)
            if selected.any():
                selections[f"{mode}_turn_{direction}"] = selected
    for mode, selected in selections.items():
        skin_selected = selected[indices]
        if not skin_selected.any():
            summary[mode] = {"accepted": False, "reason": "no actual skin for activity"}
            continue
        low_posture = mode in config.low_posture_modes
        gaps = foot_min_z_m[skin_selected] - config.floor_z_m
        touching = contact[indices[skin_selected]]
        supported_gaps = gaps[touching]
        slips = [v for i, row in enumerate(rows) if selected[i]
                 for v in row["floor_contact_slips_m_s"]]
        support_mask = supported if low_posture else contact.any(axis=1)
        no_support = longest_interval(selected & ~support_mask, times, dt_s)
        joint_error = max(float(np.rad2deg(np.abs(r["joints"] - r["joint_target"])).max())
                          for i, r in enumerate(rows) if selected[i])
        nearest_p95 = float(np.percentile(gaps.min(axis=1), 95))
        support_p95 = float(np.percentile(supported_gaps, 95)) if len(supported_gaps) else None
        slip_p95 = float(np.percentile(slips, 95)) if slips else None
        slip_gate = config.low_posture_slip_speed_m_s if low_posture else config.slip_speed_m_s
        joint_gate = (config.low_posture_joint_error_deg if low_posture
                      else config.joint_error_deg)
        ground_load = float(np.median(normal_n[selected]) / weight_n)
        whole_body_p95 = (float(np.percentile(np.asarray(skin_min_z_m)[skin_selected]
                                              - config.floor_z_m, 95))
                          if skin_min_z_m is not None else None)
        gates = {
            "skin_penetration": bool(gaps.min() >= -config.skin_penetration_m),
            "continuous_support": no_support <= config.max_no_support_s,
            "floor_slip": slip_p95 is not None and slip_p95 <= slip_gate,
            "joint_tracking": joint_error <= joint_gate,
            "ground_load": ground_load >= config.min_ground_load_fraction,
        }
        if low_posture:
            gates["whole_body_grounded"] = bool(whole_body_p95 is not None
                                                and whole_body_p95
                                                <= config.low_posture_skin_gap_m)
        else:
            gates["nearest_skin_grounded"] = nearest_p95 <= config.support_skin_gap_m
            gates["contact_skin_grounded"] = bool(support_p95 is not None
                                                  and support_p95
                                                  <= config.support_skin_gap_m)
        summary[mode] = {
            "accepted": all(gates.values()), "gates": gates,
            "support_class": "low_posture" if low_posture else "foot",
            "control_frames": int(selected.sum()), "skin_frames": int(skin_selected.sum()),
            "nearest_skin_z_p95_m": nearest_p95, "contact_skin_z_p95_m": support_p95,
            "whole_body_skin_z_p95_m": whole_body_p95,
            "foot_min_z_p05_p50_p95_m": np.percentile(gaps, [5, 50, 95], axis=0).tolist(),
            "foot_contact_fraction": contact[selected].mean(axis=0).tolist(),
            "primary_contact_fraction": float(support_mask[selected].mean()),
            "support_link_contact_frames": (
                dict(sorted(support_links_seen.items(), key=lambda item: -item[1]))
                if low_posture else {}
            ),
            "no_support_max_s": no_support, "slip_p95_m_s": slip_p95,
            "slip_gate_m_s": slip_gate, "joint_gate_deg": joint_gate,
            "joint_error_max_deg": joint_error,
            "ground_normal_p50_n": float(np.median(normal_n[selected])),
            "ground_load_fraction_p50": ground_load,
            "root_up_fraction_p50": float(np.median([r["force"][2] for i, r in enumerate(rows)
                                                       if selected[i]]) / weight_n),
        }
    return {"schema_version": SCHEMA_VERSION, "config": asdict(config), "modes": summary,
            "accepted": bool(summary) and all(m["accepted"] for m in summary.values()),
            "scope": "retained_actual_control_and_mesh_window; per activity including transitions"}
