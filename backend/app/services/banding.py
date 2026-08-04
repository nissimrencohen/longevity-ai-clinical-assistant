"""Probability -> risk band.

Thresholds are from data/DATA_DICTIONARY.md and are uniform across the five risks
(real instruments band per-outcome; these are illustrative).

Intervals are HALF-OPEN — ``[lower, upper)`` — so a probability of exactly 0.10 is
``borderline`` and exactly 0.35 is ``high``. Boundary behaviour is pinned by tests
because "which side does 0.35 fall on" is precisely the sort of off-by-one that
turns into a wrong clinical band.
"""

from __future__ import annotations

from math import isfinite

# (exclusive upper bound, band). Ordered ascending; first match wins.
BAND_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.10, "low"),
    (0.20, "borderline"),
    (0.35, "intermediate"),
    (float("inf"), "high"),
)

BANDS: tuple[str, ...] = tuple(name for _, name in BAND_THRESHOLDS)


def band(probability: float) -> str:
    """Map a probability in [0, 1] to its risk band."""
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be a finite value in [0, 1], got {probability!r}")

    for upper, name in BAND_THRESHOLDS:
        if probability < upper:
            return name
    # Unreachable: the final threshold is +inf.
    raise AssertionError("band thresholds must end with an infinite upper bound")
