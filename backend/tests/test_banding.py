"""Risk-band boundary tests.

Bands are half-open ``[lower, upper)``. The exact boundary values are the whole
point of this file: 0.10, 0.20 and 0.35 each sit at a band edge, and getting the
comparison operator backwards moves a patient a whole band — e.g. reporting a
0.35 probability as "intermediate" instead of "high".
"""

from __future__ import annotations

import pytest

from backend.app.services.banding import BANDS, band


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.05, "low"),
        (0.0999999, "low"),
        (0.10, "borderline"),      # boundary: inclusive lower edge
        (0.15, "borderline"),
        (0.1999999, "borderline"),
        (0.20, "intermediate"),    # boundary
        (0.30, "intermediate"),
        (0.3499999, "intermediate"),
        (0.35, "high"),            # boundary — the eval gold cases sit near here
        (0.50, "high"),
        (1.0, "high"),
    ],
)
def test_band_boundaries(probability: float, expected: str) -> None:
    assert band(probability) == expected


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_band_rejects_non_probabilities(bad: float) -> None:
    """A value outside [0, 1] is a bug upstream — refuse rather than clamp."""
    with pytest.raises(ValueError):
        band(bad)


def test_band_names_are_the_documented_four() -> None:
    assert BANDS == ("low", "borderline", "intermediate", "high")


def test_designed_patient_bands_from_data_dictionary() -> None:
    """Spot-check the bands the eval gold cases expect."""
    assert band(0.50) == "high"   # P004 CKD
    assert band(0.44) == "high"   # P002 CVD
    assert band(0.03) == "low"    # P001 CVD
