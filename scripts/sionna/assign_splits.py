#!/usr/bin/env python3
"""Assign group-aware train/val/test splits to a finished data batch.

Reads every ``*.import.json`` the Sionna stage produced for a batch, drops the ones
that failed their own measurement checks, groups the rest by the physics run they came
from (one keyboard session = one group, so its segments can never straddle a split),
and writes ``splits.json`` with the assignment plus the leakage and coverage report the
data card quotes.

    PYTHONPATH=src python3 scripts/sionna/assign_splits.py \
        --batch artifacts/batches/train01 --seed 20260927 --fractions 0.6,0.2,0.2

CPU only. Re-running is idempotent: the assignment is a hash of (seed, label, group).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection_split import (  # noqa: E402
    SplitFractions,
    assign_splits,
    group_of,
    split_report,
)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)


def collect(batch: Path, sionna_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (usable rows, refused sample ids) for one batch directory."""

    root = sionna_dir or (batch / "sionna")
    if not root.is_dir():
        raise FileNotFoundError(f"no Sionna output at {root}; run the batch's RT stage first")
    rows: list[dict[str, Any]] = []
    refused: list[str] = []
    for path in sorted(root.glob("**/*.import.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        # Segment ids repeat across sessions (every session has a stand_00), so the
        # sample identity is namespaced by the directory the RT stage wrote it into.
        stem = path.name.removesuffix(".import.json")
        # Namespaced by every directory level between the root and the sample, exactly like
        # `detection_data.iter_channel_samples`, so the two agree on one identity: segment
        # ids repeat per session (every session has a stand_00), and under a root that
        # merges batches the batch name has to be part of the id too.
        relative = path.parent.relative_to(root).as_posix()
        namespace = "" if relative == "." else relative
        sample_id = f"{namespace}/{stem}" if namespace else stem
        if report.get("failures"):
            refused.append(sample_id)
            continue
        source = str((report.get("source") or {}).get("source") or "")
        label = str(report.get("source_event_label") or report.get("activity") or "")
        if not source or not label:
            raise ValueError(f"{path.name}: missing source path or event label")
        # The leakage group is the run that produced the sample: the session directory under
        # the RT root. train02 and train03 both contain a session named
        # `01_fall_standing_sp040_bedroom`, so a name-only group would merge two different
        # physics runs into one group and quietly move samples across the split. Root-level
        # samples (single-trial traces) keep the path-derived group.
        group = namespace if namespace else group_of(source)
        rows.append({
            "sample_id": sample_id,
            "group": group,
            "label": label,
            "activity": report.get("activity"),
            "rt_frames": len(report.get("frames") or []),
            "sample_rate_hz": report.get("sample_rate_hz"),
            "seed": report.get("seed"),
            "split_declared": report.get("split"),
            "subject": report.get("subject_id"),
            "scene_sha256": None,
            "import_json": _relative(path),
        })
    return rows, refused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--sionna-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, required=True,
                        help="split seed; recorded in splits.json so the split is reproducible")
    parser.add_argument("--fractions", type=SplitFractions.parse, default="0.6,0.2,0.2",
                        help="train,val,test group fractions (default 0.6,0.2,0.2)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default <batch>/splits.json")
    args = parser.parse_args()
    batch = args.batch if args.batch.is_absolute() else REPO_ROOT / args.batch
    sionna_dir = None if args.sionna_dir is None else (
        args.sionna_dir if args.sionna_dir.is_absolute() else REPO_ROOT / args.sionna_dir)
    rows, refused = collect(batch, sionna_dir)
    if not rows:
        raise SystemExit("every sample in the batch failed its measurement checks")
    assignment = assign_splits(rows, fractions=args.fractions, seed=args.seed)
    for row in rows:
        row["split"] = assignment[row["group"]]
    report = split_report(rows, assignment)
    payload = {
        "schema_version": 1,
        "batch": str(batch.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "fractions": {"train": args.fractions.train, "val": args.fractions.val,
                      "test": args.fractions.test},
        "assignment_unit": "physics run (the sample's session directory under the "
                         "RT root, so batch-prefixed when the root merges batches), "
                         "never the individual segment",
        "samples": sorted(rows, key=lambda row: (row["split"], row["label"], row["sample_id"])),
        "report": report,
        "refused_samples": refused,
        "caveats": [
            "single apartment scene, single SMPL subject: no cross-scene or cross-subject "
            "generalisation is measured by this split",
            "labels come from the session's fall events and controller mode timeline, "
            "never from file names",
        ],
    }
    out = args.out or batch / "splits.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"splits -> {out}")
    for split in ("train", "val", "test"):
        labels = report["samples_by_split_label"][split]
        print(f"  {split:5s} {report['samples_per_split'][split]:3d} samples  "
              f"{', '.join(f'{k}={v}' for k, v in sorted(labels.items()))}")
    if report["insufficient_groups"]:
        print("  labels with too few groups to cover all splits: "
              + ", ".join(f"{k}({len(v)} groups)"
                          for k, v in report["insufficient_groups"].items()))
    if refused:
        print(f"  refused (failed RT checks): {', '.join(refused)}")
    return 1 if report["leakage_check"]["groups_spanning_splits"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
