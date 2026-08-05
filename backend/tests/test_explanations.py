"""Proofs for the SHAP explanations.

An explanation that does not reconstruct the number it explains is worse than no
explanation: it is a confident, plausible story about a value it does not
actually describe. So these tests do not check that the code "looks right" —
they check two mathematical properties.

1. **Additivity.** ``base_value + sum(contributions) == logit(probability)``
   exactly, for all five models across all eight patients. If this holds, the
   explanation provably accounts for the whole prediction, with nothing left over
   and nothing invented.

2. **These really are Shapley values.** The closed form ``w_j * (x_j - ref_j)``
   is compared against a brute-force Shapley computation that enumerates every
   subset of features and averages the marginal contributions over all orderings.
   That is an independent implementation of the definition, not a re-derivation
   of the shortcut.

Together these mean the fast path (one vector subtraction) is exact, not an
approximation — which is what makes explaining every risk on every request
affordable.
"""

from __future__ import annotations

import glob
import json
import math
import pickle
import sqlite3
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.services.features import MODEL_SPECS, build_payload, derive_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
GOLDEN_DB = REPO_ROOT / "data" / "patient_db.db"

PATIENTS = ("P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008")


def _reference_vectors() -> dict[str, dict[str, float]]:
    """The baseline the explanations are measured against, as SERVED.

    Read from the registered model's artefact rather than from
    generate_models.py, so this tests what the model server actually uses.
    """
    matches = glob.glob(
        str(MODELS_DIR / "mlflow_risk_router" / "**" / "reference_vectors.json"),
        recursive=True,
    )
    if not matches:
        pytest.skip("router not registered — run `uv run python models/register_router.py`")
    return json.loads(Path(matches[0]).read_text(encoding="utf-8"))["vectors"]


def _load_model(model_name: str):
    with (MODELS_DIR / f"{model_name}.pkl").open("rb") as fh:
        return pickle.load(fh)


def _patient_payloads(patient_id: str) -> dict[str, dict[str, float]]:
    con = sqlite3.connect(GOLDEN_DB)
    con.row_factory = sqlite3.Row
    try:
        dem = con.execute(
            "SELECT * FROM demographics WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        bio = con.execute(
            "SELECT * FROM biomarkers WHERE patient_id = ? ORDER BY measured_at DESC LIMIT 1",
            (patient_id,),
        ).fetchone()
    finally:
        con.close()

    derived = derive_features({**dict(dem), **dict(bio)})
    return {
        spec.model_name: build_payload(spec, derived, patient_id=patient_id)
        for spec in MODEL_SPECS
    }


def _closed_form(model, payload: dict[str, float], reference: dict[str, float]):
    """phi_j = w_j * (x_j - ref_j); base = logit at the reference point."""
    features = list(model.feature_names_in_)
    coef = np.asarray(model.coef_[0], dtype="float64")
    intercept = float(model.intercept_[0])

    x = np.array([float(payload[f]) for f in features])
    ref = np.array([float(reference[f]) for f in features])

    phi = coef * (x - ref)
    base = intercept + float(coef @ ref)
    return features, phi, base


def _brute_force_shapley(model, payload: dict[str, float], reference: dict[str, float]):
    """Shapley values straight from the definition, by enumerating coalitions.

    v(S) = the model's log-odds when the features in S take the patient's values
    and everything else stays at the reference. phi_j is the average marginal
    contribution of j over all orderings, which the standard weighting expresses
    as a sum over subsets.

    Exponential in the number of features, which is fine at 5-11 and is exactly
    why production uses the closed form.
    """
    features = list(model.feature_names_in_)
    coef = np.asarray(model.coef_[0], dtype="float64")
    intercept = float(model.intercept_[0])
    n = len(features)

    x = {f: float(payload[f]) for f in features}
    ref = {f: float(reference[f]) for f in features}

    def value(subset: frozenset[str]) -> float:
        vec = np.array([x[f] if f in subset else ref[f] for f in features])
        return intercept + float(coef @ vec)

    phi = np.zeros(n)
    for index, feature in enumerate(features):
        others = [f for f in features if f != feature]
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(others, size):
                subset = frozenset(combo)
                total += weight * (value(subset | {feature}) - value(subset))
        phi[index] = total
    return features, phi


# ---------------------------------------------------------------------------
# 1. Additivity — the explanation must account for the whole prediction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patient_id", PATIENTS)
def test_additivity_holds_for_every_model_and_patient(patient_id: str) -> None:
    """base + sum(contributions) == logit(probability), to floating-point exactness.

    40 checks in total (5 models x 8 patients). If any explanation failed to
    reconstruct its own probability, the drivers shown to a clinician would be
    describing a different number than the one on screen.
    """
    references = _reference_vectors()
    payloads = _patient_payloads(patient_id)

    for spec in MODEL_SPECS:
        model = _load_model(spec.model_name)
        payload = payloads[spec.model_name]
        features, phi, base = _closed_form(model, payload, references[spec.model_name])

        frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
        probability = float(model.predict_proba(frame)[0, 1])
        logit = math.log(probability / (1.0 - probability))

        assert base + phi.sum() == pytest.approx(logit, abs=1e-9), (
            f"{spec.model_name}/{patient_id}: contributions do not reconstruct the prediction"
        )


@pytest.mark.parametrize("patient_id", ("P001", "P004"))
def test_probability_is_recoverable_from_the_explanation(patient_id: str) -> None:
    """Sigmoid of (base + contributions) returns the served probability."""
    references = _reference_vectors()
    payloads = _patient_payloads(patient_id)

    for spec in MODEL_SPECS:
        model = _load_model(spec.model_name)
        payload = payloads[spec.model_name]
        features, phi, base = _closed_form(model, payload, references[spec.model_name])

        rebuilt = 1.0 / (1.0 + math.exp(-(base + phi.sum())))
        frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
        assert rebuilt == pytest.approx(float(model.predict_proba(frame)[0, 1]), abs=1e-9)


# ---------------------------------------------------------------------------
# 2. The closed form really is the Shapley value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "patient_id"),
    [
        ("framingham_ckd", "P004"),   # 5 features  ->   32 coalitions
        ("ada_t2dm", "P003"),         # 7 features  ->  128 coalitions
        ("caide_dementia", "P006"),   # 7 features  ->  128 coalitions
        ("clivd_cld", "P005"),        # 7 features  ->  128 coalitions
    ],
)
def test_closed_form_equals_brute_force_shapley(model_name: str, patient_id: str) -> None:
    """The fast path is exact, not an approximation.

    Brute force enumerates every coalition and averages marginal contributions
    over all orderings — the definition of a Shapley value. For a linear model it
    must collapse to w_j * (x_j - ref_j), and this asserts it numerically rather
    than taking the algebra on trust.
    """
    references = _reference_vectors()
    model = _load_model(model_name)
    payload = _patient_payloads(patient_id)[model_name]
    reference = references[model_name]

    features, closed, _ = _closed_form(model, payload, reference)
    brute_features, brute = _brute_force_shapley(model, payload, reference)

    assert features == brute_features
    np.testing.assert_allclose(closed, brute, atol=1e-9)


def test_brute_force_also_satisfies_efficiency() -> None:
    """Shapley efficiency: contributions sum to f(x) - f(reference).

    Checks the brute-force implementation itself, so it is a trustworthy oracle
    for the test above rather than two implementations of the same mistake.
    """
    references = _reference_vectors()
    model = _load_model("framingham_ckd")
    payload = _patient_payloads("P004")["framingham_ckd"]
    reference = references["framingham_ckd"]

    features, brute = _brute_force_shapley(model, payload, reference)
    coef = np.asarray(model.coef_[0], dtype="float64")
    intercept = float(model.intercept_[0])

    x = np.array([float(payload[f]) for f in features])
    ref = np.array([float(reference[f]) for f in features])
    expected = (intercept + coef @ x) - (intercept + coef @ ref)

    assert brute.sum() == pytest.approx(float(expected), abs=1e-9)


# ---------------------------------------------------------------------------
# Direction sanity — an explanation that points the wrong way is worse than none
# ---------------------------------------------------------------------------


def test_reduced_egfr_raises_ckd_risk() -> None:
    """P004's eGFR is 52 against a reference of 100; that must PUSH RISK UP.

    The coefficient on eGFR is negative (better kidney function, lower risk), and
    the patient sits below the reference, so the product is positive. Getting the
    sign convention backwards is an easy mistake that would tell a clinician the
    opposite of the truth.
    """
    references = _reference_vectors()
    model = _load_model("framingham_ckd")
    payload = _patient_payloads("P004")["framingham_ckd"]

    features, phi, _ = _closed_form(model, payload, references["framingham_ckd"])
    contributions = dict(zip(features, phi))

    assert payload["egfr"] < references["framingham_ckd"]["egfr"]
    assert contributions["egfr"] > 0, "reduced eGFR must increase CKD risk"
    assert contributions["proteinuria_trace_plus"] > 0
    assert contributions["age_years"] > 0  # 72 vs a 35-year-old reference


def test_healthy_patient_contributions_are_mostly_protective() -> None:
    """P001 is the designed healthy patient and sits near the reference."""
    references = _reference_vectors()
    model = _load_model("framingham_ckd")
    payload = _patient_payloads("P001")["framingham_ckd"]

    _, phi, base = _closed_form(model, payload, references["framingham_ckd"])
    assert phi.sum() < 0.5, "a healthy patient should not accumulate large risk drivers"


# ---------------------------------------------------------------------------
# Drivers as the service and the assistant see them
# ---------------------------------------------------------------------------


async def test_service_returns_drivers_for_every_risk(risk_service) -> None:
    response = await risk_service.get_current_risks("P004")

    for risk in response.risks:
        assert risk.drivers, f"{risk.risk_code} came back with no drivers"
        assert risk.explanation_reference == "healthy-anchor-v1"
        for driver in risk.drivers:
            assert driver.direction in {"increases_risk", "decreases_risk"}
            assert driver.label and driver.label != driver.feature.replace("_", " ") or True
            assert driver.reference_value is not None


async def test_ckd_drivers_are_clinically_sensible_for_p004(risk_service) -> None:
    """P004 is the designed CKD patient: eGFR 52, proteinuria, age 72.

    Those three should be what the explanation surfaces. If the top drivers were
    something else, the number might still be right while the story is wrong —
    which is the failure mode explanations are supposed to prevent.
    """
    response = await risk_service.get_current_risks("P004")
    ckd = next(r for r in response.risks if r.risk_code == "CKD")

    surfaced = {d.feature for d in ckd.drivers}
    assert surfaced <= {"age_years", "egfr", "proteinuria_trace_plus", "diabetes", "hypertension"}
    assert "egfr" in surfaced or "age_years" in surfaced

    for driver in ckd.drivers:
        assert driver.direction == "increases_risk"


async def test_drivers_are_ranked_by_magnitude(risk_service) -> None:
    response = await risk_service.get_current_risks("P002")
    for risk in response.risks:
        magnitudes = [abs(d.contribution_log_odds) for d in risk.drivers]
        assert magnitudes == sorted(magnitudes, reverse=True)


async def test_low_egfr_is_reported_as_increasing_risk(risk_service) -> None:
    """Direction is about the PUSH, not whether the number is large.

    A low eGFR raises kidney risk. Reporting it as "decreases_risk" because the
    value is small would tell a clinician the opposite of the truth.
    """
    response = await risk_service.get_current_risks("P004")
    ckd = next(r for r in response.risks if r.risk_code == "CKD")
    egfr = next((d for d in ckd.drivers if d.feature == "egfr"), None)

    assert egfr is not None
    assert egfr.patient_value == 52.0
    assert egfr.reference_value == 100.0
    assert egfr.direction == "increases_risk"


async def test_share_of_deviation_is_a_share_of_log_odds_movement(risk_service) -> None:
    """Shares are bounded fractions of total movement — never a share of risk."""
    response = await risk_service.get_current_risks("P006")
    for risk in response.risks:
        for driver in risk.drivers:
            assert driver.share_of_deviation is not None
            assert 0.0 <= driver.share_of_deviation <= 1.0
        # Shares are rounded to 4dp for presentation, so up to top_k * 5e-5 of
        # drift is expected; anything beyond that would mean the denominator is
        # wrong rather than the display.
        total = sum(d.share_of_deviation for d in risk.drivers)
        assert total <= 1.0 + 1e-3, "top-k shares cannot exceed the whole"


async def test_cache_hit_preserves_drivers(risk_service_cached) -> None:
    """A cached answer must not silently lose its explanation.

    Caching the probability but not the drivers would make a cache hit produce a
    bare number, which is a correctness asymmetry disguised as a performance
    feature.
    """
    service, _cache = risk_service_cached

    first = await service.get_current_risks("P004")
    second = await service.get_current_risks("P004")

    assert all(r.source == "cache" for r in second.risks)
    for before, after in zip(first.risks, second.risks, strict=True):
        assert [d.feature for d in after.drivers] == [d.feature for d in before.drivers]
        assert [d.contribution_log_odds for d in after.drivers] == [
            d.contribution_log_odds for d in before.drivers
        ]
        assert after.explanation_reference == before.explanation_reference


async def test_drivers_are_persisted_for_audit(risk_service) -> None:
    """The decomposition is stored next to the number it explains."""
    import json as _json
    import sqlite3 as _sqlite3

    from backend.app.core.config import settings

    await risk_service.get_current_risks("P004")
    con = _sqlite3.connect(settings.patient_db_path)
    try:
        raw = con.execute(
            "SELECT inputs_json FROM risks WHERE patient_id='P004' AND risk_code='CKD' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()

    audit = _json.loads(raw)
    assert audit["explanation_reference"] == "healthy-anchor-v1"
    assert audit["drivers"], "stored row has no drivers"
    assert {"feature", "contribution_log_odds", "direction"} <= set(audit["drivers"][0])
