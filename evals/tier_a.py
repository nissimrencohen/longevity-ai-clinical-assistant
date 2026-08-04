"""Tier A — deterministic evaluation against the MCP tools. No LLM, no API key.

This is the regression gate. It calls the same MCP tools the assistant calls and
checks the values the assistant would be repeating: exact biomarkers, probability
tolerances, bands, trend directions, time horizons, the unknown-patient contract,
and determinism.

It cannot judge prose — whether the assistant hedged appropriately or refused to
prescribe is Tier B's job. Those checks are marked SKIP here, never PASS.

Because it needs no API key it is free to run, fast, and safe for CI, which is
what makes it usable as a gate on every change.
"""

from __future__ import annotations

import time
from typing import Any

from evals.cases import Case
from evals.mcp_tools import ToolCallError, call_tool, open_client
from evals.results import FAIL, PASS, SKIP, CaseResult, Check
from evals.scoring import numbers_close, risk_entry, trend_series

TIER = "A"


async def run(cases: list[Case]) -> list[CaseResult]:
    results: list[CaseResult] = []
    async with open_client() as client:
        available = {t.name for t in await client.list_tools()}
        for case in cases:
            results.append(await _run_case(client, case, available))
    return results


async def _run_case(client: Any, case: Case, available: set[str]) -> CaseResult:
    started = time.perf_counter()
    result = CaseResult(
        case_id=case.id, category=case.category, source=case.source, tier=TIER
    )

    # Contract check: the tool the case expects must actually exist. Catches a
    # renamed or dropped tool before any value assertion muddies the picture.
    if case.expected_tool and case.expected_tool not in {"any", "none", None}:
        result.checks.append(
            Check(
                name=f"tool_exists:{case.expected_tool}",
                axis="tool_contract",
                status=PASS if case.expected_tool in available else FAIL,
                detail=""
                if case.expected_tool in available
                else f"server exposes {sorted(available)}",
                expected=case.expected_tool,
                actual=sorted(available),
            )
        )

    payloads: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for patient_id in case.target_patients:
        tool = _tool_for(case)
        if tool is None:
            continue
        try:
            payloads[patient_id] = await call_tool(
                client, tool, {"patient_id": patient_id}
            )
            result.tool_calls.append({"tool": tool, "patient_id": patient_id, "ok": True})
        except ToolCallError as exc:
            errors[patient_id] = exc.message
            result.tool_calls.append(
                {"tool": tool, "patient_id": patient_id, "ok": False, "error": exc.message}
            )

    for fact in case.expected_facts:
        result.checks.extend(await _check_fact(client, case, fact, payloads, errors))

    result.duration_s = round(time.perf_counter() - started, 3)
    return result


def _tool_for(case: Case) -> str | None:
    tool = case.expected_tool
    if tool in {None, "none"}:
        return None
    if tool == "any":
        # The case does not pin a tool; risks is the superset for our purposes.
        return "get_current_risks"
    return tool


async def _check_fact(
    client: Any,
    case: Case,
    fact: dict[str, Any],
    payloads: dict[str, Any],
    errors: dict[str, str],
) -> list[Check]:
    kind = fact.get("kind")
    handler = {
        "biomarker": _check_biomarker,
        "risk": _check_risk,
        "trend": _check_trend,
        "horizon": _check_horizon,
        "comparison": _check_comparison,
    }.get(kind)

    if handler is not None:
        return handler(case, fact, payloads, errors)
    if kind == "determinism":
        return await _check_determinism(client, case, fact)
    if kind == "no_fabrication":
        return _check_no_fabrication_contract(case, fact, payloads, errors)
    if kind in {"safety", "citation"}:
        return [
            Check(
                name=f"{kind}",
                axis=kind,
                status=SKIP,
                detail="behavioural — judged in Tier B, not assertable from tool output",
                expected=fact.get("must") or fact.get("expect"),
            )
        ]
    return [
        Check(name=str(kind), axis="unknown", status=SKIP, detail="unrecognised fact kind")
    ]


def _payload_for(case: Case, fact: dict[str, Any], payloads: dict[str, Any]) -> Any:
    patient_id = fact.get("patient_id") or case.patient_id
    if patient_id in payloads:
        return payloads[patient_id]
    return next(iter(payloads.values()), None)


def _check_biomarker(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    field = fact["field"]
    expected = fact["value"]
    payload = _payload_for(case, fact, payloads)
    name = f"biomarker:{field}"

    if payload is None:
        return [
            Check(name, "numeric_faithfulness", FAIL, f"no tool payload ({errors})", expected)
        ]

    actual = (payload.get("biomarkers") or {}).get(field)
    ok = actual is not None and numbers_close(float(actual), float(expected), 1e-9)
    return [
        Check(
            name,
            "numeric_faithfulness",
            PASS if ok else FAIL,
            "" if ok else f"expected exactly {expected} {fact.get('unit', '')}".strip(),
            expected,
            actual,
        )
    ]


def _check_risk(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    code = fact["risk_code"]
    patient_id = fact.get("patient_id") or case.patient_id
    payload = _payload_for(case, fact, payloads)
    checks: list[Check] = []
    suffix = f"{code}@{patient_id}"

    if payload is None:
        return [
            Check(f"risk:{suffix}", "numeric_faithfulness", FAIL, f"no tool payload ({errors})")
        ]

    entry = risk_entry(payload, code)
    if entry is None:
        return [Check(f"risk:{suffix}", "numeric_faithfulness", FAIL, "risk_code absent")]

    if "band" in fact:
        actual = entry.get("risk_band")
        ok = actual == fact["band"]
        checks.append(
            Check(
                f"band:{suffix}",
                "band_faithfulness",
                PASS if ok else FAIL,
                "" if ok else f"probability was {entry.get('probability')}",
                fact["band"],
                actual,
            )
        )

    if "approx_probability" in fact:
        tolerance = float(fact.get("tolerance", 0.05))
        expected = float(fact["approx_probability"])
        actual = float(entry.get("probability", float("nan")))
        ok = numbers_close(actual, expected, tolerance)
        checks.append(
            Check(
                f"probability:{suffix}",
                "numeric_faithfulness",
                PASS if ok else FAIL,
                "" if ok else f"|{actual:.4f} - {expected}| > {tolerance}",
                f"{expected} +/- {tolerance}",
                round(actual, 4),
            )
        )
    return checks


def _check_trend(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    code = fact["risk_code"]
    expected = fact["direction"]
    payload = _payload_for(case, fact, payloads)
    name = f"trend:{code}"

    if payload is None:
        return [Check(name, "trend", FAIL, f"no tool payload ({errors})", expected)]

    entry = risk_entry(payload, code) or {}
    actual = entry.get("trend_direction")
    series = [round(float(p["probability"]), 4) for p in trend_series(payload, code)]
    ok = actual == expected
    return [
        Check(
            name,
            "trend",
            PASS if ok else FAIL,
            f"series={series}",
            expected,
            actual,
        )
    ]


def _check_horizon(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    code = fact["risk_code"]
    expected = fact.get("value")
    payload = _payload_for(case, fact, payloads)
    name = f"horizon:{code}"

    if payload is None:
        return [Check(name, "numeric_faithfulness", FAIL, f"no tool payload ({errors})")]

    entry = risk_entry(payload, code) or {}
    actual = entry.get("time_horizon_years")
    ok = actual == expected
    return [Check(name, "numeric_faithfulness", PASS if ok else FAIL, "", expected, actual)]


def _check_comparison(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    code = fact["risk_code"]
    expected = fact["expect_higher"]
    name = f"comparison:{code}"

    scores: dict[str, float] = {}
    for patient_id in fact.get("patient_ids", case.target_patients):
        payload = payloads.get(patient_id)
        entry = risk_entry(payload, code) if payload else None
        if entry is None:
            return [Check(name, "comparison", FAIL, f"missing payload for {patient_id}")]
        scores[patient_id] = float(entry["probability"])

    actual = max(scores, key=lambda k: scores[k])
    ok = actual == expected
    return [
        Check(
            name,
            "comparison",
            PASS if ok else FAIL,
            ", ".join(f"{k}={v:.4f}" for k, v in scores.items()),
            expected,
            actual,
        )
    ]


async def _check_determinism(client: Any, case: Case, fact: dict[str, Any]) -> list[Check]:
    """Same inputs must produce bit-identical probabilities on a repeat call."""
    patient_id = case.patient_id
    name = f"determinism:{fact.get('risk_code', 'all')}"
    if not patient_id:
        return [Check(name, "determinism", SKIP, "no patient_id")]

    first = await call_tool(client, "get_current_risks", {"patient_id": patient_id})
    second = await call_tool(client, "get_current_risks", {"patient_id": patient_id})

    drift = {
        r["risk_code"]: (r["probability"], s["probability"])
        for r, s in zip(first["risks"], second["risks"])
        if r["probability"] != s["probability"]
    }
    ok = not drift
    return [
        Check(
            name,
            "determinism",
            PASS if ok else FAIL,
            "identical across two calls" if ok else f"drifted: {drift}",
        )
    ]


def _check_no_fabrication_contract(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    """Tier A's slice of a no-fabrication case: the TOOL must refuse.

    Whether the assistant *says* the patient was not found is behavioural and
    belongs to Tier B. What is assertable here is that the tool returns an error
    rather than an empty-but-successful object — an empty 200 would hand the
    model a template to hallucinate into.
    """
    patient_id = case.patient_id
    name = "tool_refuses_unknown_patient"

    if not patient_id:
        return [
            Check(
                name,
                "safety",
                SKIP,
                "no patient_id to probe — behavioural only, judged in Tier B",
                fact.get("expect"),
            )
        ]

    if patient_id in errors:
        message = errors[patient_id]
        mentions = patient_id in message or "not found" in message.lower()
        return [
            Check(
                name,
                "tool_contract",
                PASS if mentions else FAIL,
                message[:200],
                f"error naming {patient_id}",
                "error raised",
            )
        ]

    return [
        Check(
            name,
            "tool_contract",
            FAIL,
            "tool returned data for an unknown patient instead of erroring",
            "error",
            "success",
        )
    ]
