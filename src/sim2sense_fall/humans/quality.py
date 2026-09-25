"""Measured motion gates, separated by activity so idle frames cannot hide failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MotionQualityConfig:
    support_skin_gap_m: float = 0.005
    max_no_support_s: float = 0.05
    slip_speed_m_s: float = 0.15
    skin_penetration_m: float = 0.005
    joint_error_deg: float = 15.0
    min_contact_impulse_ns: float = 0.01
    floor_z_m: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or (name != "floor_z_m" and value <= 0):
                raise ValueError(f"invalid motion quality setting {name}")


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
) -> dict[str, Any]:
    """Consume actual contact reports and actual skinned feet, never reference FK.

    Each skin timestamp is associated with the preceding pre-step control row.
    Contact membership consequently has at most one physics step of timing offset.
    Falling/fallen do not require stance contact or reference tracking.
    """
    if not rows or not len(mesh_time_s):
        return {"schema_version": 1, "accepted": False, "modes": {}, "reason": "no samples"}
    if (foot_min_z_m.shape != (len(mesh_time_s), 2)
            or not np.isfinite(foot_min_z_m).all() or not np.isfinite(mesh_time_s).all()
            or np.any(np.diff(mesh_time_s) <= 0)
            or not np.isfinite([dt_s, mass_kg, gravity_m_s2]).all()
            or min(dt_s, mass_kg, gravity_m_s2) <= 0):
        raise ValueError("invalid measured motion samples")
    times = np.array([r["time_s"] for r in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("control timestamps must increase")
    modes = np.array([r["mode"] for r in rows])
    contact = np.zeros((len(rows), 2), dtype=bool)
    normal_n = np.zeros(len(rows))
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
    indices = np.clip(np.searchsorted(times, mesh_time_s, side="right") - 1, 0, len(times)-1)
    summary = {}
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
        gaps = foot_min_z_m[skin_selected] - config.floor_z_m
        touching = contact[indices[skin_selected]]
        supported_gaps = gaps[touching]
        slips = [v for i, row in enumerate(rows) if selected[i]
                 for v in row["floor_contact_slips_m_s"]]
        no_support = longest_interval(selected & ~contact.any(axis=1), times, dt_s)
        joint_error = max(float(np.rad2deg(np.abs(r["joints"] - r["joint_target"])).max())
                          for i, r in enumerate(rows) if selected[i])
        nearest_p95 = float(np.percentile(gaps.min(axis=1), 95))
        support_p95 = float(np.percentile(supported_gaps, 95)) if len(supported_gaps) else None
        slip_p95 = float(np.percentile(slips, 95)) if slips else None
        gates = {
            "skin_penetration": bool(gaps.min() >= -config.skin_penetration_m),
            "nearest_skin_grounded": nearest_p95 <= config.support_skin_gap_m,
            "contact_skin_grounded": support_p95 is not None
            and support_p95 <= config.support_skin_gap_m,
            "continuous_support": no_support <= config.max_no_support_s,
            "floor_slip": slip_p95 is not None and slip_p95 <= config.slip_speed_m_s,
            "joint_tracking": joint_error <= config.joint_error_deg,
        }
        summary[mode] = {
            "accepted": all(gates.values()), "gates": gates,
            "control_frames": int(selected.sum()), "skin_frames": int(skin_selected.sum()),
            "nearest_skin_z_p95_m": nearest_p95, "contact_skin_z_p95_m": support_p95,
            "foot_min_z_p05_p50_p95_m": np.percentile(gaps, [5, 50, 95], axis=0).tolist(),
            "foot_contact_fraction": contact[selected].mean(axis=0).tolist(),
            "no_support_max_s": no_support, "slip_p95_m_s": slip_p95,
            "joint_error_max_deg": joint_error,
            "ground_normal_p50_n": float(np.median(normal_n[selected])),
            "root_up_fraction_p50": float(np.median([r["force"][2] for i, r in enumerate(rows)
                                                       if selected[i]]) / (mass_kg*gravity_m_s2)),
        }
    return {"schema_version": 1, "config": asdict(config), "modes": summary,
            "accepted": bool(summary) and all(m["accepted"] for m in summary.values()),
            "scope": "retained_actual_control_and_mesh_window; per activity including transitions"}
