"""Stable data structures shared by simulators, preprocessing, and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Activity(str, Enum):
    """Activity labels accepted by the baseline pipeline."""

    FALL = "fall"
    ADL = "adl"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ChannelSample:
    """One time-aligned wireless observation sequence."""

    sample_id: str
    scene_id: str
    subject_id: str
    hardware_profile: str
    activity: Activity
    timestamp_s: np.ndarray
    channel: np.ndarray
    channel_representation: str
    sample_rate_hz: float
    simulator_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
