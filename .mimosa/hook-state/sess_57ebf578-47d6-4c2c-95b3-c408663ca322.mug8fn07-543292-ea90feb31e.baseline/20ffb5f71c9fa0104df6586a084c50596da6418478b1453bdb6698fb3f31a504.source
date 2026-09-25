"""Validation for the cross-runtime channel sample contract."""

from __future__ import annotations

import numpy as np

from .schema import ChannelSample


def validate_sample(sample: ChannelSample) -> None:
    """Raise ``ValueError`` when a sample violates the data contract."""

    required_text = {
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "subject_id": sample.subject_id,
        "hardware_profile": sample.hardware_profile,
        "simulator_version": sample.simulator_version,
    }
    for field_name, value in required_text.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if sample.channel_representation not in {"csi", "cir"}:
        raise ValueError("channel_representation must be 'csi' or 'cir'")
    if not np.isfinite(sample.sample_rate_hz) or sample.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    timestamps = np.asarray(sample.timestamp_s)
    channel = np.asarray(sample.channel)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamp_s must be a one-dimensional sequence with at least 2 values")
    if channel.shape[0] != timestamps.size:
        raise ValueError("channel and timestamp_s must have the same first dimension")
    if not np.isfinite(timestamps).all() or not np.isfinite(channel).all():
        raise ValueError("timestamp_s and channel must contain only finite values")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamp_s must be strictly increasing")
