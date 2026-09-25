"""Strict numeric checks shared by scene specifications and derived geometry."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def finite_number(value: Any, name: str) -> float:
    """Accept finite real numbers, excluding booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def finite_vector(value: Any, length: int, name: str, *, positive: bool = False) -> None:
    """Validate a fixed-size vector without coercing invalid inputs."""
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    for i, component in enumerate(value):
        number = finite_number(component, f"{name}[{i}]")
        if positive and number <= 0:
            raise ValueError(f"{name} extents must be positive, got {value!r}")


def strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def strict_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value
