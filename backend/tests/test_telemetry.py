"""Traces must not become a PHI leak.

Auto-instrumentation records request URLs, query strings and exception messages
as span attributes, and those go to a trace backend. For this stack that would
mean patient identifiers and lab values sitting in a dashboard — undoing the
minimisation work in core/phi.py while looking like an improvement.

These tests assert the exporter cannot do that: allowlist filtering, URL query
stripping, patient-id pseudonymisation, and events dropped wholesale.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from backend.app.core.phi import pseudonym
from backend.app.core.telemetry import (
    ALLOWED_ATTRIBUTES,
    PhiScrubbingExporter,
    scrub_attributes,
    strip_url,
)


class _CapturingExporter:
    """Stands in for the OTLP exporter; keeps whatever it was handed."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _span(name: str = "GET /api/v1/get_current_risks", **attributes) -> ReadableSpan:
    return ReadableSpan(name=name, attributes=attributes)


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


def test_unknown_attributes_are_dropped() -> None:
    """Deny by default: a denylist is always one library release behind."""
    cleaned = scrub_attributes(
        {
            "http.method": "GET",
            "http.status_code": 200,
            # Exactly the sort of thing instrumentation invents.
            "db.statement": "SELECT * FROM biomarkers WHERE patient_id='P004'",
            "request.body": '{"patient_id": "P004"}',
            "patient.egfr": 52.0,
        }
    )
    assert cleaned == {"http.method": "GET", "http.status_code": 200}


def test_allowlist_covers_only_non_clinical_facts() -> None:
    """Nothing in the allowlist should be able to carry a measurement."""
    for key in ALLOWED_ATTRIBUTES:
        assert not key.startswith("clinic.patient_value")
        assert "egfr" not in key and "cholesterol" not in key


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "http://backend:8001/api/v1/get_current_risks?patient_id=P004",
            "http://backend:8001/api/v1/get_current_risks",
        ),
        ("/api/v1/find_patient?query=Maya%20Cohen", "/api/v1/find_patient"),
        ("http://mlflow:5001/invocations", "http://mlflow:5001/invocations"),
    ],
)
def test_query_strings_are_stripped(raw: str, expected: str) -> None:
    """`?patient_id=P004` is precisely what ends up in a dashboard otherwise."""
    assert strip_url(raw) == expected


def test_url_attribute_is_sanitised_in_place() -> None:
    cleaned = scrub_attributes(
        {"http.url": "http://backend:8001/api/v1/get_current_risks?patient_id=P004"}
    )
    assert "patient_id" not in cleaned["http.url"]
    assert "P004" not in cleaned["http.url"]


# ---------------------------------------------------------------------------
# Patient identifiers
# ---------------------------------------------------------------------------


def test_patient_ids_are_pseudonymised_anywhere_they_appear() -> None:
    """Pseudonymous alone, identifying beside clinical data — so never in clear."""
    cleaned = scrub_attributes({"clinic.patient_ref": "P004"})
    assert cleaned["clinic.patient_ref"] == pseudonym("P004")
    assert "P004" not in cleaned["clinic.patient_ref"]


def test_pseudonym_is_consistent_so_traces_remain_joinable() -> None:
    """Debugging still needs to follow one patient through a trace."""
    first = scrub_attributes({"clinic.patient_ref": "P004"})["clinic.patient_ref"]
    second = scrub_attributes({"http.route": "P004"})["http.route"]
    assert first == second


# ---------------------------------------------------------------------------
# End to end through the exporter
# ---------------------------------------------------------------------------


def test_exporter_strips_everything_sensitive() -> None:
    inner = _CapturingExporter()
    exporter = PhiScrubbingExporter(inner)

    exporter.export(
        [
            _span(
                **{
                    "http.method": "GET",
                    "http.url": "http://backend:8001/api/v1/get_current_risks?patient_id=P004",
                    "http.status_code": 200,
                    "db.statement": "SELECT egfr_ml_min_1_73m2 FROM biomarkers",
                    "clinic.patient_ref": "P004",
                }
            )
        ]
    )

    exported = inner.spans[0]
    blob = repr(dict(exported.attributes))

    assert "P004" not in blob
    assert "patient_id" not in blob
    assert "egfr" not in blob
    assert "db.statement" not in blob
    # Useful debugging signal survives.
    assert exported.attributes["http.method"] == "GET"
    assert exported.attributes["http.status_code"] == 200
    assert exported.name == "GET /api/v1/get_current_risks"


def test_known_lab_value_never_reaches_the_exporter() -> None:
    """The blunt version of the same check, against real values from the fixture."""
    inner = _CapturingExporter()
    PhiScrubbingExporter(inner).export(
        [
            _span(
                **{
                    "http.method": "GET",
                    "response.body": '{"egfr_ml_min_1_73m2": 52.0, "name": "Avraham Friedman"}',
                    "patient.name": "Avraham Friedman",
                }
            )
        ]
    )

    blob = repr(dict(inner.spans[0].attributes))
    for secret in ("52.0", "Avraham", "Friedman"):
        assert secret not in blob


def test_span_events_are_dropped() -> None:
    """Exception messages quote URLs and values; the type is enough to debug."""
    from opentelemetry.sdk.trace import Event

    inner = _CapturingExporter()
    span = ReadableSpan(
        name="db.query",
        attributes={"http.method": "GET"},
        events=(Event(name="exception", attributes={"exception.message": "P004 egfr 52.0"}),),
    )
    PhiScrubbingExporter(inner).export([span])

    assert inner.spans[0].events == ()


def test_telemetry_is_off_by_default() -> None:
    """The graded path must not depend on a collector being reachable."""
    from backend.app.core.config import settings
    from backend.app.core.telemetry import setup_telemetry

    assert settings.otel_enabled is False
    assert setup_telemetry(None) is False
