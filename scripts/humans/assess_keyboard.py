#!/usr/bin/env python3
"""Recompute full-session quality and complete-cycle arms from keyboard chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from common import REPO_ROOT, write_json


def load_control(run: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    manifest = run / "chunks/manifest.json"
    if manifest.exists():
        chunks = json.loads(manifest.read_text())["chunks"]
        pieces = [dict(np.load(manifest.parent / row["control"])) for row in chunks]
        contacts = [
            item
            for row in chunks
            for item in json.loads((manifest.parent / row["contacts"]).read_text())
        ]
        if not pieces:
            raise ValueError("recording has no chunks")
        names = pieces[0]["dof_names"]
        if any(not np.array_equal(p["dof_names"], names) for p in pieces):
            raise ValueError("DOF order changed between chunks")
        arrays = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0] if k != "dof_names"}
        arrays["dof_names"] = names
    else:
        arrays = dict(np.load(run / "control.npz"))
        contacts = json.loads((run / "contacts.json").read_text())
    if len(arrays["time_s"]) != len(contacts):
        raise ValueError("control/contact counts disagree")
    return arrays, contacts


def assess(run: Path, gate_path: Path) -> dict[str, Any]:
    report = json.loads((run / "report.json").read_text())
    a, contacts = load_control(run)
    names = [str(n) for n in a["dof_names"]]
    if len(set(names)) != len(names) or len(names) != a["joints"].shape[1]:
        raise ValueError("explicit DOF order is invalid")
    gate = yaml.safe_load(gate_path.read_text())["gate"]
    by_mode = {}
    for mode in np.unique(a["mode"]):
        mask = a["mode"] == mode
        slip = [
            v
            for row, keep in zip(contacts, mask, strict=True)
            if keep
            for v in row["floor_contact_slips_m_s"]
        ]
        by_mode[str(mode)] = {
            "frames": int(mask.sum()),
            "joint_error_max_deg": float(
                np.rad2deg(abs(a["joints"][mask] - a["joint_target"][mask])).max()
            ),
            "slip_samples": len(slip),
            "slip_p95_m_s": float(np.percentile(slip, 95)) if slip else None,
            "root_height_range_m": [
                float(v) for v in [a["root"][mask, 2].min(), a["root"][mask, 2].max()]
            ],
        }
    arm_segments = []
    for mode in ("forward", "backward"):
        mask = (a["mode"] == mode) & (a["gait_weight"] >= 0.99)
        indices = np.flatnonzero(mask)
        for contiguous in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(contiguous) < 2:
                continue
            phases = np.unwrap(a["gait_phase"][contiguous] * 2 * np.pi) / (2 * np.pi)
            cycles = abs(float(phases[-1] - phases[0]))
            row = {
                "mode": mode,
                "start_s": float(a["time_s"][contiguous[0]]),
                "end_s": float(a["time_s"][contiguous[-1]]),
                "cycles": cycles,
                "complete_cycle": cycles >= 1,
                "dofs": {},
                "ratios": {},
            }
            for chain in ("shoulder", "elbow", "wrist"):
                for suffix in ("__dof1", "__dof2", ""):
                    spans = []
                    for side in ("left", "right"):
                        name = f"{side}_{chain}{suffix}"
                        col = names.index(name)
                        actual = a["joints"][contiguous, col]
                        target = a["joint_target"][contiguous, col]
                        spans.append(float(np.rad2deg(np.ptp(actual))))
                        row["dofs"][name] = {
                            "actual_span_deg": spans[-1],
                            "target_span_deg": float(np.rad2deg(np.ptp(target))),
                            "max_error_deg": float(np.rad2deg(abs(actual - target)).max()),
                        }
                    row["ratios"][chain + suffix] = spans[1] / spans[0] if spans[0] > 1e-6 else None
            row["historical_ratio_gate_passed"] = (
                cycles >= 1
                and row["ratios"]["shoulder__dof2"] >= gate["min_shoulder_ratio"]
                and row["ratios"]["elbow__dof2"] >= gate["min_elbow_ratio"]
            )
            arm_segments.append(row)
    return {
        "source": str(run.resolve()),
        "full_control_frames": len(a["time_s"]),
        "by_mode": by_mode,
        "arm_segments": arm_segments,
        "fall_events": report.get("fall", {}).get("events", []),
        "note": (
            "Complete-cycle amplitudes; explicit DOF order; mode-specific quality. "
            "Ratios alone do not establish visual correctness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--gate", type=Path, default=REPO_ROOT / "configs/humans/arm_symmetry_gate.yaml"
    )
    args = parser.parse_args()
    write_json(args.run / "assessment.json", assess(args.run, args.gate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
