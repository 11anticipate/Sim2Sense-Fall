"""Merge packaged ProtoMotions MotionLib .pt files into one production library.

Per-motion dt is native to MotionLib (frame blending indexes ``motion_dt[motion_ids]``),
so libraries at different fps (KIT 50 Hz, BMLmovi/CMU 30 Hz) merge without resampling:
frame-wise tensors concatenate along the frame axis, per-motion index arrays along the
motion axis with ``length_starts`` offset by the cumulative frame count.

    .venv_mujoco/bin/python scripts/protomotions/merge_motionlibs.py \
        --inputs a.pt b.pt c.pt --output production_motionlib.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")
MOTION_KEYS = (
    "length_starts", "motion_lengths", "motion_dt", "motion_num_frames", "motion_weights",
)


def merge(
    libs: list[dict],
    output: Path,
    *,
    exclude: list[str] | None = None,
    only: list[str] | None = None,
) -> list[str]:
    for lib in libs:
        for key in FRAME_KEYS + MOTION_KEYS:
            if key not in lib:
                raise ValueError(f"{key} missing from an input library")
    first = libs[0]
    for lib in libs[1:]:
        if sorted(lib.keys()) != sorted(first.keys()):
            raise ValueError("input libraries have different key sets")

    selected: list[dict] = []
    kept_names: list[str] = []
    for lib in libs:
        for name in lib["motion_files"]:
            tail = "/".join(Path(name).parts[-2:])
            if only is not None and tail not in only:
                continue
            if exclude is not None and tail in exclude:
                continue
            selected.append(lib)
            kept_names.append(name)
    if not selected:
        raise ValueError("selection left zero motions")

    # group frames per (lib, motion) via length_starts/num_frames to concat only
    # the selected motions' frame slices in order.
    frame_rows: dict[str, list[torch.Tensor]] = {key: [] for key in FRAME_KEYS}
    motion_rows: dict[str, list[torch.Tensor]] = {key: [] for key in MOTION_KEYS}
    kept = 0
    for lib in libs:
        starts = lib["length_starts"].tolist()
        counts = lib["motion_num_frames"].tolist()
        for name, count in zip(lib["motion_files"], counts, strict=True):
            # match on "<subject-dir>/<file>" tail: bare stems repeat across
            # subject directories (KIT has kick06_poses under every subject)
            tail = "/".join(Path(name).parts[-2:])
            if only is not None and tail not in only:
                continue
            if exclude is not None and tail in exclude:
                continue
            index = lib["motion_files"].index(name)
            local_start = starts[index]
            end = local_start + count
            for key in FRAME_KEYS:
                frame_rows[key].append(lib[key][local_start:end])
            for key in MOTION_KEYS:
                motion_rows[key].append(lib[key][index])
            kept += 1

    merged: dict = {key: torch.cat(frame_rows[key], dim=0) for key in FRAME_KEYS}
    merged["motion_files"] = kept_names
    for key in MOTION_KEYS:
        merged[key] = torch.stack(motion_rows[key], dim=0)
    # length_starts are offsets into the merged frame axis; rebuild cumulatively.
    cumulative = torch.cumsum(merged["motion_num_frames"], dim=0)
    merged["length_starts"] = (
        cumulative - merged["motion_num_frames"]
    )

    total_frames = int(merged["motion_num_frames"].sum())
    for key in FRAME_KEYS:
        if len(merged[key]) != total_frames:
            raise ValueError(f"{key}: {len(merged[key])} frames != declared {total_frames}")
    total_time = float(merged["motion_lengths"].sum())

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(
        f"wrote {output}: {kept} motions, "
        f"{total_frames} frames, {total_time:.1f} s total"
    )
    return kept_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--holdout-file", type=Path, default=None,
        help="text file: one motion-file path substring per line; matching motions "
        "are EXCLUDED from --output and (with --holdout-output) written separately "
        "for generalization evaluation",
    )
    parser.add_argument("--holdout-output", type=Path, default=None)
    args = parser.parse_args()
    libs = [torch.load(p, weights_only=False) for p in args.inputs]
    if args.holdout_file is not None:
        substrings = [
            line.strip()
            for line in args.holdout_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        keep = merge(libs, args.output, exclude=substrings)
        if args.holdout_output is not None:
            merge(libs, args.holdout_output, only=substrings)
        else:
            print(f"{len(keep)} motions excluded as hold-out (no --holdout-output given)")
    else:
        merge(libs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
