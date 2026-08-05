"""Distributed tracing, with PHI kept out of it.

Traces are how you prove the concurrency claim rather than asserting it: one
doctor's question becomes `mcp.tool -> backend -> cache -> 5x mlflow -> db.write`,
and the five model calls appear as sibling spans or they do not.

THE RISK THIS MODULE EXISTS TO MANAGE. Auto-instrumentation is enthusiastic. It
records request URLs, query strings and sometimes bodies as span attributes, and
those go to a trace backend — which for this stack means labs and patient
identifiers sitting in a third-party dashboard. That is the single easiest way to
leak PHI while believing you have improved things, and it would undo the
minimisation work in `core/phi.py`.

So the exporter is wrapped: span attributes are filtered against an ALLOWLIST
(deny by default), URLs are stripped of query strings, and patient identifiers
are replaced with the same pseudonym the researcher role sees. What reaches the
backend is names, timings, statuses and a `patient_ref` — enough to debug a slow
request, not enough to learn anything about a person.

OpenTelemetry rather than a vendor SDK because it is the neutral wire format: the
backend (Phoenix, Langfuse, Tempo) stays swappable without touching this code.

Off by default (`OTEL_ENABLED=false`), so the graded path carries no extra
dependency at runtime.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .phi import pseudonym

logger = logging.getLogger(__name__)

# Attributes worth keeping. Anything not listed is dropped, because a denylist of
# "sensitive" keys will always be one instrumentation release behind.
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "http.method", "http.request.method",
        "http.status_code", "http.response.status_code",
        "http.route", "http.scheme", "http.flavor",
        "net.peer.name", "net.peer.port", "server.address", "server.port",
        "db.system", "db.operation",
        "rpc.method", "rpc.service",
        "service.name", "service.version",
        "otel.status_code", "otel.status_description",
        "error.type", "exception.type",
        # Ours: safe by construction (see sanitise_value).
        "clinic.tool", "clinic.model_name", "clinic.model_version",
        "clinic.risk_code", "clinic.cache", "clinic.actor_role",
        "clinic.inputs_hash", "clinic.patient_ref",
    }
)

# Attributes carrying a URL: kept, but stripped of query and fragment, because
# `?patient_id=P004` is exactly the sort of thing that ends up in a dashboard.
URL_ATTRIBUTES: frozenset[str] = frozenset(
    {"http.url", "url.full", "http.target", "url.path", "url.query"}
)

_PATIENT_ID_RE = re.compile(r"\bP\d{3}\b")
_REDACTED = "[redacted]"


def strip_url(value: str) -> str:
    """Drop query and fragment; keep scheme/host/path for debugging."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return _REDACTED
    if not parts.scheme and not parts.netloc:
        # A bare path such as "/api/v1/get_current_risks?patient_id=P004".
        return value.split("?", 1)[0].split("#", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def sanitise_value(key: str, value: Any) -> Any:
    """Make one attribute safe to export."""
    if key in URL_ATTRIBUTES and isinstance(value, str):
        value = strip_url(value)
    if isinstance(value, str):
        # A patient id is pseudonymous on its own but identifying next to
        # clinical data, so it never travels in the clear.
        value = _PATIENT_ID_RE.sub(lambda m: pseudonym(m.group(0)), value)
    return value


def scrub_attributes(attributes: Any) -> dict[str, Any]:
    """Allowlist filter plus value sanitisation. Deny by default."""
    if not attributes:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in dict(attributes).items():
        if key in ALLOWED_ATTRIBUTES or key in URL_ATTRIBUTES:
            cleaned[key] = sanitise_value(key, value)
    return cleaned


class PhiScrubbingExporter(SpanExporter):
    """Wraps a real exporter and rewrites every span before it leaves.

    Rebuilds each span with filtered attributes rather than mutating it: a
    ReadableSpan is meant to be immutable, and reaching into private state to
    edit one is exactly the kind of shortcut that stops working silently on a
    library upgrade — with PHI leakage as the failure mode.
    """

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._clean(span) for span in spans])

    def _clean(self, span: ReadableSpan) -> ReadableSpan:
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=scrub_attributes(span.attributes),
            # Events carry their own attributes (exception messages can quote a
            # URL or a value), so they are dropped wholesale. Status and the
            # exception TYPE survive, which is what debugging actually needs.
            events=(),
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


def setup_telemetry(app: Any = None, *, service_name: str = "longevity-backend") -> bool:
    """Configure tracing if enabled. Returns True when instrumentation was set up.

    Deliberately forgiving: a missing instrumentation package or an unreachable
    collector must not stop the clinic working. Observability is for
    understanding the system, not a dependency of it.
    """
    from .config import settings

    if not settings.otel_enabled:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                PhiScrubbingExporter(
                    OTLPSpanExporter(endpoint=settings.otel_endpoint)
                )
            )
        )
        trace.set_tracer_provider(provider)

        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

        logger.info("telemetry enabled -> %s", settings.otel_endpoint)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry setup failed, continuing without it: %s", exc)
        return False
