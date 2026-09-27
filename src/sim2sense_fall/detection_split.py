"""Group-aware train/val/test assignment for fall-detection channel samples.

The unit of assignment is never the sample. Segments cut from one physics keyboard
session share a continuous body trajectory, contact history and controller state, and
the RT imports of one export are the same measurement seen from different frames -- so
putting two of them on opposite sides of the split leaks. ``group`` is therefore the
session (or trial) a sample came from, and every member of a group lands in the same
split.

Assignment is stratified by activity label: each label gets its own seeded permutation
of its groups and is cut at the requested fractions, so a rare label cannot end up
entirely inside train. With fewer groups than splits for some label the remainder is
reported honestly as ``insufficient_groups`` rather than being silently copied into
every split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if min(self.train, self.val, self.test) < 0 or not 0.999 <= total <= 1.001:
            raise ValueError(f"split fractions must be non-negative and sum to 1, got {total:.4f}")

    @classmethod
    def parse(cls, text: str) -> SplitFractions:
        parts = [float(piece) for piece in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected train,val,test, got {text!r}")
        return cls(*parts)


def group_of(source: str) -> str:
    """The leakage group for a sample source path or id.

    Session exports live in ``.../session_<name>_export`` and physics trials in
    ``.../trials/<motion>__<perturbation>.trial.json``; both collapse to the run that
    produced them, so nothing from one continuous simulation reaches two splits.
    """

    parts = [part for part in str(source).replace("\\", "/").split("/") if part]
    if not parts:
        return str(source)
    for index, part in enumerate(parts):
        if part.startswith("session_"):
            # .../session_<name>_export/manifest.json -> session_<name>
            return part.removesuffix("_export")
        if part == "trials" and index + 1 < len(parts):
            # Each physics trial is its own independent run, so it is its own group.
            return parts[index + 1].removesuffix(".trial.json").removesuffix(".json")
    return parts[-1].removesuffix(".json").removesuffix("_export")


def _digest(seed: int, label: str, group: str) -> bytes:
    return hashlib.sha256(f"{seed}|{label}|{group}".encode()).digest()


def assign_splits(
    samples: Sequence[Mapping[str, Any]], *, fractions: SplitFractions, seed: int
) -> dict[str, str]:
    """Return ``group -> split`` for every sample's group, deterministically.

    ``samples`` are mappings with at least ``group`` and ``label``. Ordering is
    irrelevant: the permutation is a hash of (seed, label, group), so re-running on a
    re-listed directory produces the same assignment.
    """

    if not isinstance(fractions, SplitFractions):
        raise ValueError("fractions must be a SplitFractions")
    if not isinstance(seed, int):
        raise ValueError("seed must be an int")
    groups: dict[str, str] = {}
    for row in samples:
        group, label = str(row["group"]), str(row["label"])
        if not group or not label:
            raise ValueError(f"sample {row} needs a non-empty group and label")
        known = groups.setdefault(group, label)
        if known != label and known != "mixed":
            # A group carrying several activities is normal (a session has fall and
            # walk segments); it is stratified under its own mixed key.
            groups[group] = "mixed"
        else:
            groups[group] = known
    by_label: dict[str, list[str]] = {}
    for group, label in groups.items():
        by_label.setdefault(label, []).append(group)
    assignment: dict[str, str] = {}
    for label, members in sorted(by_label.items()):
        ordered = sorted(members, key=lambda group: _digest(seed, label, group))
        count = len(ordered)
        n_train = int(round(count * fractions.train))
        n_val = int(round(count * fractions.val))
        if count >= 3:
            # Guarantee every split is non-empty for a label that can support it,
            # so a 2-group label cannot silently vanish from test.
            n_train = max(1, min(n_train, count - 2))
            n_val = max(1, min(n_val, count - n_train))
        else:
            n_train, n_val = count, 0
        bounds = {"train": n_train, "val": n_train + n_val, "test": count}
        for index, group in enumerate(ordered):
            split = ("train" if index < bounds["train"]
                     else "val" if index < bounds["val"] else "test")
            assignment[group] = split
    return assignment


def split_report(
    samples: Iterable[Mapping[str, Any]], assignment: Mapping[str, str]
) -> dict[str, Any]:
    """Per-split counts by label, plus the groups too small to be represented."""

    rows = list(samples)
    table: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    groups_by_label: dict[str, set[str]] = {}
    all_groups: set[str] = set()
    for row in rows:
        group, label = str(row["group"]), str(row["label"])
        split = assignment.get(group)
        if split is None:
            raise ValueError(f"group {group} has no split assignment")
        table[split][label] = table[split].get(label, 0) + 1
        groups_by_label.setdefault(label, set()).add(group)
        all_groups.add(group)
    counts = {split: sum(table[split].values()) for split in SPLITS}
    total = sum(counts.values())
    return {
        "samples_per_split": counts,
        "samples_by_split_label": table,
        "groups": {label: sorted(members) for label, members in sorted(groups_by_label.items())},
        "group_count": len(all_groups),
        "insufficient_groups": {
            label: sorted(members)
            for label, members in sorted(groups_by_label.items())
            if len(members) < len(SPLITS)
        },
        "leakage_check": {
            "groups_spanning_splits": [
                group
                for group in {str(row["group"]) for row in rows}
                if len({assignment[str(row["group"])] for row in rows
                        if str(row["group"]) == group}) > 1
            ]
        },
        "fraction_of_samples": {
            split: (counts[split] / total if total else 0.0) for split in SPLITS
        },
    }
