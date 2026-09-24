"""Independent stationary-standing acceptance; a no_fall label is insufficient."""

from __future__ import annotations

import numpy as np


def standing_metrics(
    root_xyz: np.ndarray,
    trunk_xyz: np.ndarray,
    *,
    reference_root_z: float,
    max_drop_m: float,
    max_tilt_deg: float,
    max_drift_m: float,
) -> dict[str, float | bool]:
    limits = np.array([max_drop_m, max_tilt_deg, max_drift_m, reference_root_z])
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("standing limits and reference height must be finite and positive")
    root, trunk = np.asarray(root_xyz), np.asarray(trunk_xyz)
    if root.ndim != 2 or root.shape[1] != 3 or root.shape != trunk.shape or len(root) < 2:
        raise ValueError("root and trunk must have matching (T, 3) arrays, T >= 2")
    if not np.isfinite(root).all() or not np.isfinite(trunk).all():
        raise ValueError("standing trajectory must be finite")
    norm = np.linalg.norm(trunk, axis=1)
    if np.any(norm == 0):
        raise ValueError("trunk axis must be nonzero")
    tilt = float(np.degrees(np.arccos(np.clip(trunk[:, 2] / norm, -1, 1))).max())
    drop = max(0.0, float(reference_root_z - root[:, 2].min()))
    drift = float(np.linalg.norm(root[:, :2] - root[0, :2], axis=1).max())
    return {
        "max_pelvis_drop_m": drop,
        "peak_trunk_tilt_deg": tilt,
        "max_horizontal_drift_m": drift,
        "max_drop_m": max_drop_m,
        "max_tilt_deg": max_tilt_deg,
        "max_drift_m": max_drift_m,
        "passed": drop <= max_drop_m and tilt <= max_tilt_deg and drift <= max_drift_m,
    }
