"""Bounded, lossless-on-normal-exit chunks for long keyboard sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ChunkRecorder:
    """Persist control/contact records before a bounded live window discards them.

    Chunk writes happen in the GUI loop, never inside a physics callback. Each
    completed chunk is independently readable after interruption. The final
    partial chunk is flushed on exit; the manifest identifies its exact interval.
    """

    def __init__(self, directory: Path, dof_names: tuple[str, ...], frames: int) -> None:
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            raise ValueError("chunk frames must be a positive integer")
        self.directory, self.dof_names, self.frames = directory, dof_names, frames
        directory.mkdir(parents=True, exist_ok=True)
        self.pending: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.pending.append(row)

    def flush(self, *, final: bool = False) -> None:
        while len(self.pending) >= self.frames or (final and self.pending):
            rows = self.pending[: self.frames]
            index = len(self.chunks)
            control = self.directory / f"control_{index:05d}.npz"
            contacts = self.directory / f"contacts_{index:05d}.json"
            arrays = {
                key: np.asarray([r[key] for r in rows])
                for key in rows[0]
                if key not in {"contacts", "contact_detail", "floor_contact_slips_m_s"}
            }
            np.savez_compressed(control, dof_names=self.dof_names, **arrays)
            contacts.write_text(
                json.dumps(
                    [
                        {
                            "time_s": r["time_s"],
                            "mode": r["mode"],
                            "samples": r["contact_detail"],
                            "floor_contact_slips_m_s": r["floor_contact_slips_m_s"],
                        }
                        for r in rows
                    ]
                ),
                encoding="utf-8",
            )
            self.chunks.append(
                {
                    "control": control.name,
                    "contacts": contacts.name,
                    "frames": len(rows),
                    "start_s": rows[0]["time_s"],
                    "end_s": rows[-1]["time_s"],
                }
            )
            del self.pending[: len(rows)]
            payload = {
                "dof_names": self.dof_names,
                "chunks": self.chunks,
                "total_frames": sum(c["frames"] for c in self.chunks),
            }
            temporary = self.directory / "manifest.tmp"
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.directory / "manifest.json")
