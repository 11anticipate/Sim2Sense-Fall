#!/usr/bin/env python3
"""Migrate pre-fix human trial indexes to the explicit time-step schema.

The first batch writer stored the physics *rate* (120 Hz) in ``physics_dt_s``
instead of the physics step in seconds.  Trial JSON files already contain the
correct value, so this migration only repairs the batch index and adds the
explicit ``physics_hz`` field used by current readers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INDEXES = (
    Path("artifacts/humans/trials/trials_index.json"),
    Path("artifacts/humans/trials_confusable/trials_index.json"),
)


def migrate_index(path: Path, *, write: bool) -> bool:
    """Repair one index, returning whether its contents changed."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    raw_dt = payload.get("physics_dt_s")
    if not isinstance(raw_dt, (int, float)) or raw_dt <= 0:
        raise ValueError(f"{path}: physics_dt_s must be a positive number")
    raw_hz = payload.get("physics_hz")
    if raw_hz is None:
        # The old malformed value is identifiable from the trial records and the
        # configured 120 Hz physics rate.  Refuse other ambiguous indexes.
        trial_dts: set[float] = set()
        for trial in payload.get("trials", []):
            if not isinstance(trial, dict):
                continue
            trial_path = trial.get("json")
            if not isinstance(trial_path, str):
                continue
            candidate = path.parents[3] / trial_path
            if not candidate.is_file():
                raise ValueError(f"{path}: referenced trial JSON is missing: {candidate}")
            with candidate.open(encoding="utf-8") as handle:
                trial_payload = json.load(handle)
            provenance = trial_payload.get("provenance", {})
            if isinstance(provenance, dict) and "physics_dt_s" in provenance:
                trial_dts.add(float(provenance["physics_dt_s"]))
        if trial_dts and len(trial_dts) != 1:
            raise ValueError(f"{path}: trial records disagree on physics_dt_s: {trial_dts}")
        inferred_dt = next(iter(trial_dts), None)
        if inferred_dt is None or abs(float(raw_dt) - 1.0 / inferred_dt) > 1e-9:
            raise ValueError(f"{path}: cannot identify a safe rate-to-step migration")
        payload["physics_dt_s"] = inferred_dt
        payload["physics_hz"] = 1.0 / inferred_dt
    else:
        hz = float(raw_hz)
        dt = float(raw_dt)
        if hz <= 0 or dt <= 0 or abs(dt - 1.0 / hz) > 1e-9:
            raise ValueError(f"{path}: physics_dt_s and physics_hz are inconsistent")
        return False
    if write:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_INDEXES))
    parser.add_argument("--write", action="store_true", help="write migrated indexes in place")
    args = parser.parse_args()
    changed = 0
    for path in args.paths:
        if migrate_index(path, write=args.write):
            changed += 1
            print(f"{'migrated' if args.write else 'would migrate'} {path}")
        else:
            print(f"already current {path}")
    print(f"checked {len(args.paths)} indexes; changed {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
