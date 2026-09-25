#!/usr/bin/env python3
"""Recompute measured motion gates from retained physics, contact and mesh records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from common import REPO_ROOT, write_json

from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.quality import MotionQualityConfig, motion_quality


def audit(run: Path, rig: Path) -> dict:
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    if hashlib.sha256(rig.read_bytes()).hexdigest() != report["rig_sha256"]:
        raise ValueError("rig differs from the recorded session")
    config = load_human_config(rig)
    with np.load(run / "control.npz", allow_pickle=False) as data:
        control = dict(data)
    contacts = json.loads((run / "contacts.json").read_text())
    times = control["time_s"]
    if len(contacts) != len(times) or not np.allclose(
        [c["time_s"] for c in contacts], times, atol=1e-8, rtol=0
    ):
        raise ValueError("contacts and control timestamps do not match")
    rows = [{**{key: control[key][i] for key in (
        "time_s", "mode", "joints", "joint_target", "force", "command")},
        "contact_detail": contacts[i]["samples"],
        "floor_contact_slips_m_s": contacts[i]["floor_contact_slips_m_s"]}
        for i in range(len(times))]
    with np.load(run / "recording.npz", allow_pickle=False) as data:
        quality = motion_quality(
            rows, data["time_s"], data["foot_min_z_m"],
            config=MotionQualityConfig(**report["motion_quality"]["config"]),
            dt_s=report["physics_dt_s"], mass_kg=config.skeleton.mass_kg,
            gravity_m_s2=config.simulation.gravity_m_s2)
    return {
        "motion_quality": quality,
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "quality_source_sha256": hashlib.sha256(
            (REPO_ROOT / "src/sim2sense_fall/humans/quality.py").read_bytes()).hexdigest(),
        "source_run": str(run.resolve()), "seed": report["seed"],
        "scene_split": report["scene_split"],
        "note": "Offline audit only; original session admission is not changed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.rig)
    write_json(args.out, result)
    return 0 if result["motion_quality"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
