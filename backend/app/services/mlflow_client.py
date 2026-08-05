"""Async client for the MLflow model server (the RiskRouter pyfunc).

Wire format (``POST /invocations``)::

    {"dataframe_split": {"columns": [...], "data": [[...]]},
     "params": {"model": "framingham_ckd"}}
    -> {"predictions": [0.5003...]}

With ``params={"explain": true}`` each prediction becomes an object carrying the
probability plus its SHAP decomposition. The explanation arrives in the SAME
round trip as the number it explains, so the two can never disagree.

Everything that can go wrong upstream — connection refused, timeout, 4xx/5xx, a
malformed body, a probability outside [0, 1] — collapses into ``ModelServerError``
so the API layer has exactly one thing to turn into a 502. We never substitute a
default probability: an unavailable model must read as unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from ..core.errors import ModelServerError


@dataclass(frozen=True)
class ModelPrediction:
    """One model's answer, optionally with its explanation."""

    probability: float
    base_value: float | None = None
    reference_id: str | None = None
    contributions: dict[str, float] = field(default_factory=dict)
    reference_values: dict[str, float] = field(default_factory=dict)
    feature_values: dict[str, float] = field(default_factory=dict)

    @property
    def explained(self) -> bool:
        return bool(self.contributions)


class MLflowRiskClient:
    """Calls one routed model per request. Safe for concurrent use."""

    def __init__(self, client: httpx.AsyncClient, invocations_url: str) -> None:
        self._client = client
        self._url = invocations_url

    async def predict(
        self,
        model_name: str,
        payload: Mapping[str, float],
        *,
        explain: bool = False,
    ) -> ModelPrediction:
        """Score one patient under one model."""
        columns = list(payload)
        body = {
            "dataframe_split": {"columns": columns, "data": [[payload[c] for c in columns]]},
            "params": {"model": model_name, "explain": explain},
        }

        try:
            response = await self._client.post(self._url, json=body)
        except httpx.TimeoutException as exc:
            raise ModelServerError(
                f"Timed out calling the model server for {model_name!r} at {self._url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelServerError(
                f"Could not reach the model server for {model_name!r} at {self._url}: {exc}"
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise ModelServerError(
                f"Model server returned HTTP {response.status_code} for {model_name!r}: "
                f"{response.text[:500]}"
            )

        return self._parse(model_name, response)

    async def predict_proba(self, model_name: str, payload: Mapping[str, float]) -> float:
        """Probability only — the original contract, kept for callers that want it."""
        return (await self.predict(model_name, payload)).probability

    def _parse(self, model_name: str, response: httpx.Response) -> ModelPrediction:
        try:
            prediction = response.json()["predictions"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelServerError(
                f"Unparseable model-server response for {model_name!r}: "
                f"{response.text[:500]}"
            ) from exc

        if isinstance(prediction, Mapping):
            result = ModelPrediction(
                probability=self._as_probability(model_name, prediction.get("probability")),
                base_value=_optional_float(prediction.get("base_value")),
                reference_id=prediction.get("reference_id"),
                contributions=_float_map(prediction.get("contributions")),
                reference_values=_float_map(prediction.get("reference_values")),
                feature_values=_float_map(prediction.get("feature_values")),
            )
        else:
            result = ModelPrediction(
                probability=self._as_probability(model_name, prediction)
            )
        return result

    @staticmethod
    def _as_probability(model_name: str, value: object) -> float:
        try:
            probability = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ModelServerError(
                f"Model server returned a non-numeric prediction for {model_name!r}: {value!r}"
            ) from exc

        # A router misconfigured to call .predict() instead of .predict_proba()
        # returns class LABELS (0.0/1.0). We cannot distinguish a label 1.0 from a
        # genuine probability of 1.0, but anything outside [0, 1] is unambiguously
        # not a probability and must not reach a clinician.
        if not 0.0 <= probability <= 1.0:
            raise ModelServerError(
                f"Model server returned {probability!r} for {model_name!r}, which is not a "
                "probability in [0, 1] — is the router returning predict_proba?"
            )
        return probability


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return out
