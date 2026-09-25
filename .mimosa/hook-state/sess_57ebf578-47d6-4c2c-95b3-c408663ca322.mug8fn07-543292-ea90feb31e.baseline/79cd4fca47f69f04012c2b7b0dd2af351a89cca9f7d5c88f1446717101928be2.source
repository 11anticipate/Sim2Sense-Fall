"""Optional simulator integration boundaries with a dependency-free dry run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """Inputs needed to generate one aligned motion and channel sample."""

    scene_id: str
    subject_id: str
    activity: str
    seed: int
    duration_s: float
    sample_rate_hz: float


class SimulatorAdapter(Protocol):
    """Protocol implemented by Isaac Sim/Sionna runtime adapters."""

    def generate(self, request: SimulationRequest) -> object:
        """Generate an implementation-specific simulation artifact."""


def check_optional_runtime(*, dry_run: bool = True) -> str:
    """Describe runtime availability without importing heavyweight simulators."""

    if dry_run:
        return "dry-run"
    missing = []
    for module_name in ("omni.isaac", "sionna"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"optional simulator runtime is unavailable: {joined}")
    return "available"
