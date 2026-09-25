"""CPU-only streaming window and alarm aggregation baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Parameters for converting positive windows into alarm events."""

    window_length_s: float = 2.0
    stride_s: float = 0.25
    threshold: float = 0.80
    min_consecutive_positive_windows: int = 2


def iter_window_starts(duration_s: float, config: WindowConfig) -> np.ndarray:
    """Return deterministic window start times for a stream duration."""

    if duration_s <= 0 or config.window_length_s <= 0 or config.stride_s <= 0:
        raise ValueError("duration_s, window_length_s and stride_s must be positive")
    if duration_s < config.window_length_s:
        return np.empty(0, dtype=float)
    count = int(np.floor((duration_s - config.window_length_s) / config.stride_s)) + 1
    return np.arange(count, dtype=float) * config.stride_s


def aggregate_alarm(probabilities: np.ndarray, config: WindowConfig) -> bool:
    """Return true once enough consecutive windows exceed the alarm threshold."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not 0 < config.threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if config.min_consecutive_positive_windows < 1:
        raise ValueError("min_consecutive_positive_windows must be positive")
    run = 0
    for probability in values:
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilities must be finite values in [0, 1]")
        run = run + 1 if probability >= config.threshold else 0
        if run >= config.min_consecutive_positive_windows:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--window-length-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.25)
    args = parser.parse_args()
    starts = iter_window_starts(
        args.duration_s,
        WindowConfig(window_length_s=args.window_length_s, stride_s=args.stride_s),
    )
    print(f"windows={starts.size}")


if __name__ == "__main__":
    main()
