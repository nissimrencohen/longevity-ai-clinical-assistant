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
from backend.app.services.guidelines import verify_citation
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
        "drivers": _check_drivers,
    }.get(kind)

    if handler is not None:
        return handler(case, fact, payloads, errors)
    if kind == "determinism":
        return await _check_determinism(client, case, fact)
    if kind == "no_fabrication":
        return _check_no_fabrication_contract(case, fact, payloads, errors)
    if kind == "find_patient":
        return await _check_find_patient(client, fact)
    if kind == "guidelines":
        return await _check_guidelines(client, fact)
    if kind == "no_percentage_attribution":
        return [
            Check(
                name="no_percentage_attribution",
                axis="explanation",
                status=SKIP,
                detail=(
                    "behavioural — whether the PROSE converts a log-odds "
                    "contribution into a percentage of risk is Tier B's to judge"
                ),
                expected="drivers described qualitatively, not as % of risk",
            )
        ]
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


def _check_drivers(
    case: Case, fact: dict[str, Any], payloads: dict[str, Any], errors: dict[str, str]
) -> list[Check]:
    """The tool's side of an explanation: are the drivers there, ranked, and sane?

    Deterministic and therefore Tier A's job. Whether the assistant *describes*
    them faithfully is Tier B's.
    """
    code = fact["risk_code"]
    payload = _payload_for(case, fact, payloads)
    name = f"drivers:{code}"

    if payload is None:
        return [Check(name, "explanation", FAIL, f"no tool payload ({errors})")]

    entry = risk_entry(payload, code) or {}
    drivers = entry.get("drivers") or []
    checks: list[Check] = []

    present = bool(drivers)
    checks.append(
        Check(
            name,
            "explanation",
            PASS if present else FAIL,
            "" if present else "risk came back with no drivers",
            "at least one driver",
            [d.get("feature") for d in drivers],
        )
    )
    if not present:
        return checks

    top_k = int(fact.get("top_k", 3))
    within = len(drivers) <= top_k
    checks.append(
        Check(
            f"{name}:top_k",
            "explanation",
            PASS if within else FAIL,
            "",
            f"<= {top_k} drivers",
            len(drivers),
        )
    )

    magnitudes = [abs(float(d.get("contribution_log_odds", 0.0))) for d in drivers]
    ranked = magnitudes == sorted(magnitudes, reverse=True)
    checks.append(
        Check(
            f"{name}:ranked",
            "explanation",
            PASS if ranked else FAIL,
            "drivers must be ordered by magnitude, largest first",
            "descending |contribution|",
            magnitudes,
        )
    )

    # An explanation that cannot name the baseline it measured against is not
    # reproducible, and a clinician cannot sanity-check it.
    reference = entry.get("explanation_reference")
    checks.append(
        Check(
            f"{name}:reference",
            "explanation",
            PASS if reference else FAIL,
            "",
            "a reference id",
            reference,
        )
    )

    required = fact.get("must_include") or []
    if required:
        surfaced = {d.get("feature") for d in drivers}
        missing = [f for f in required if f not in surfaced]
        checks.append(
            Check(
                f"{name}:must_include",
                "explanation",
                PASS if not missing else FAIL,
                "" if not missing else f"missing {missing}",
                required,
                sorted(surfaced),
            )
        )
    return checks


async def _check_guidelines(client: Any, fact: dict[str, Any]) -> list[Check]:
    """Retrieval, and whether every citation it returns actually resolves.

    The citation check is the point. A plausible reference to text that is not
    there launders an invented claim as a sourced one, and it is deterministic to
    catch — so it belongs here rather than in an LLM judge.
    """
    query = fact["query"]
    expected_file = fact.get("expect_source_file")
    name = f"guidelines:{query[:24]}"

    try:
        payload = await call_tool(
            client,
            "search_guidelines",
            {
                "query": query,
                "k": int(fact.get("k", 3)),
                **({"risk_code": fact["risk_code"]} if fact.get("risk_code") else {}),
            },
        )
    except ToolCallError as exc:
        return [Check(name, "citation", FAIL, exc.message[:200], expected_file)]

    snippets = payload.get("snippets") or []
    checks: list[Check] = []

    if expected_file:
        top = snippets[0].get("source_file") if snippets else None
        ok = top == expected_file
        checks.append(
            Check(
                name,
                "citation",
                PASS if ok else FAIL,
                "" if ok else f"top hit was {top}",
                expected_file,
                [s.get("source_file") for s in snippets],
            )
        )

    # Every returned snippet must be confirmable against the file on disk.
    unverifiable: list[str] = []
    for snippet in snippets:
        result = verify_citation(
            snippet.get("source_file", ""),
            heading=snippet.get("heading"),
            quote=(snippet.get("text") or "")[:80],
        )
        if not (result["file_exists"] and result["heading_exists"] and result["quote_found"]):
            unverifiable.append(snippet.get("citation", "?"))

    checks.append(
        Check(
            f"{name}:verifiable",
            "citation",
            PASS if not unverifiable else FAIL,
            "" if not unverifiable else f"did not resolve: {unverifiable}",
            "every citation resolves to real text",
            [s.get("citation") for s in snippets],
        )
    )
    return checks


async def _check_find_patient(client: Any, fact: dict[str, Any]) -> list[Check]:
    """Server-side name resolution: the tool that replaced the prompt roster.

    Deterministic, so it belongs here rather than in the LLM tier.
    """
    query = fact["query"]
    expected = list(fact.get("expect_ids", []))
    name = f"find_patient:{query}"

    try:
        payload = await call_tool(client, "find_patient", {"name": query})
    except ToolCallError as exc:
        return [Check(name, "tool_contract", FAIL, exc.message[:200], expected)]

    matches = payload.get("matches") or []
    found = [m.get("patient_id") for m in matches]
    ok = found == expected
    checks = [
        Check(
            name,
            "tool_contract",
            PASS if ok else FAIL,
            "" if ok else f"resolved to {found}",
            expected,
            found,
        )
    ]

    # A name lookup is the easiest place to leak identifiers, so assert the
    # response carries nothing beyond the id and the name.
    leaked = sorted(
        {
            key
            for m in matches
            for key in m
            if key.lower() in {"mrn", "date_of_birth", "dob", "ssn", "address"}
        }
    )
    checks.append(
        Check(
            f"{name}:no_phi",
            "phi",
            PASS if not leaked else FAIL,
            "" if not leaked else f"leaked {leaked}",
            "id and name only",
            sorted({k for m in matches for k in m}),
        )
    )
    return checks


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
