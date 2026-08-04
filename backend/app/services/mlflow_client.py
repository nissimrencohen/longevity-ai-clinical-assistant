"""Async client for the MLflow model server (the RiskRouter pyfunc).

Wire format (``POST /invocations``)::

    {"dataframe_split": {"columns": [...], "data": [[...]]},
     "params": {"model": "framingham_ckd"}}
    -> {"predictions": [0.5003...]}

Everything that can go wrong upstream — connection refused, timeout, 4xx/5xx, a
malformed body, a probability outside [0, 1] — collapses into ``ModelServerError``
so the API layer has exactly one thing to turn into a 502. We never substitute a
default probability: an unavailable model must read as unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from ..core.errors import ModelServerError


class MLflowRiskClient:
    """Calls one routed model per request. Safe for concurrent use."""

    def __init__(self, client: httpx.AsyncClient, invocations_url: str) -> None:
        self._client = client
        self._url = invocations_url

    async def predict_proba(self, model_name: str, payload: Mapping[str, float]) -> float:
        """Return the positive-class probability for one patient under one model."""
        columns = list(payload)
        body = {
            "dataframe_split": {"columns": columns, "data": [[payload[c] for c in columns]]},
            "params": {"model": model_name},
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

        return self._extract_probability(model_name, response)

    @staticmethod
    def _extract_probability(model_name: str, response: httpx.Response) -> float:
        try:
            predictions = response.json()["predictions"]
            value = float(predictions[0])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelServerError(
                f"Unparseable model-server response for {model_name!r}: "
                f"{response.text[:500]}"
            ) from exc

        # A router misconfigured to call .predict() instead of .predict_proba()
        # returns class LABELS (0.0/1.0). We cannot distinguish a label 1.0 from a
        # genuine probability of 1.0, but anything outside [0, 1] is unambiguously
        # not a probability and must not reach a clinician.
        if not 0.0 <= value <= 1.0:
            raise ModelServerError(
                f"Model server returned {value!r} for {model_name!r}, which is not a "
                "probability in [0, 1] — is the router returning predict_proba?"
            )
        return value
