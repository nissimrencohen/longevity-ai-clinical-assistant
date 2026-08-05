"""Custom routing agent — deterministic orchestration over the MCP tools.

WHY THIS EXISTS ALONGSIDE THE BUILT-IN AGENT. LibreChat's agent is fine at
conversation and picks tools well (Tier B shows it chaining find_patient into the
clinical tools). What it cannot do is guarantee an ORDER, or refuse to proceed
when a step fails, because every step is a model decision. For a routine that
should run identically every time — resolve the patient, pull the risks, ground
the high ones in cited guidance — a graph is the right shape and an LLM is not
needed at all.

So this is a real agent in the orchestration sense and deliberately not a
conversational one: it decides nothing that a rule can decide. That has three
consequences worth having.

  * It is FREE and DETERMINISTIC, so it can be tested and run in CI. Every
    assertion below costs nothing.
  * It CANNOT hallucinate, because it never generates prose. It assembles values
    the tools returned.
  * A failure is a failure. If the patient does not resolve, it stops, rather
    than continuing with a plausible guess the way a model might.

REUSE, DON'T REBUILD: it drives the same MCP server as the chat UI, so the tool
logic, the RBAC checks and the audit trail are shared. Nothing is duplicated.

Run:
    uv run python agent/router.py "Avraham Friedman"
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from evals.mcp_tools import ToolCallError, call_tool, open_client  # noqa: E402

HIGH_RISK_BANDS = {"high", "intermediate"}


@dataclass
class ReviewResult:
    """What the routine found. Every field traces to a tool response."""

    query: str
    patient_id: str | None = None
    patient_name: str | None = None
    risks: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.patient_id is not None

    def elevated(self) -> list[dict]:
        return [r for r in self.risks if r.get("risk_band") in HIGH_RISK_BANDS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "ok": self.ok,
            "error": self.error,
            "steps": self.steps,
            "elevated_risks": [
                {
                    "risk_code": r["risk_code"],
                    "probability": r["probability"],
                    "risk_band": r["risk_band"],
                    "drivers": [d["label"] for d in r.get("drivers", [])],
                }
                for r in self.elevated()
            ],
            "citations": self.citations,
        }


async def review_patient(name: str, *, top_risks: int = 2) -> ReviewResult:
    """resolve -> risks -> ground the elevated ones in cited guidance.

    Each step is a hard gate. Ambiguity stops the run rather than being resolved
    by picking the first match, which is the failure mode this whole project has
    been guarding against.
    """
    result = ReviewResult(query=name)

    async with open_client() as client:
        # 1. Resolve the name server-side.
        try:
            found = await call_tool(client, "find_patient", {"name": name})
        except ToolCallError as exc:
            result.error = f"lookup failed: {exc.message}"
            return result

        matches = found.get("matches") or []
        result.steps.append(f"find_patient({name!r}) -> {len(matches)} match(es)")

        if not matches:
            result.error = f"no patient matching {name!r}"
            return result
        if len(matches) > 1:
            # A human picks. Choosing for them is exactly the substitution error
            # the tools were designed to prevent.
            names = ", ".join(f"{m['name']} ({m['patient_id']})" for m in matches)
            result.error = f"{name!r} is ambiguous: {names}"
            return result

        result.patient_id = matches[0]["patient_id"]
        result.patient_name = matches[0]["name"]

        # 2. Compute the risk panel.
        try:
            risks = await call_tool(
                client, "get_current_risks", {"patient_id": result.patient_id}
            )
        except ToolCallError as exc:
            result.error = f"risk computation failed: {exc.message}"
            return result

        result.risks = risks.get("risks") or []
        result.steps.append(
            f"get_current_risks({result.patient_id}) -> {len(result.risks)} risks"
        )

        # 3. Ground only the elevated ones. Retrieving guidance for a low risk
        #    would pad the output and invite the model reading it to over-read.
        elevated = sorted(
            result.elevated(), key=lambda r: r["probability"], reverse=True
        )[:top_risks]

        for risk in elevated:
            drivers = ", ".join(d["label"] for d in risk.get("drivers", []))
            query = f"{risk['risk_code']} risk factors {drivers}".strip()
            try:
                found = await call_tool(
                    client,
                    "search_guidelines",
                    {"query": query, "k": 1, "risk_code": risk["risk_code"]},
                )
            except ToolCallError as exc:
                result.steps.append(f"search_guidelines({risk['risk_code']}) failed: {exc.message}")
                continue

            for snippet in found.get("snippets") or []:
                result.citations.append(
                    {
                        "risk_code": risk["risk_code"],
                        "citation": snippet["citation"],
                        "lines": snippet["lines"],
                        "text": snippet["text"],
                    }
                )
            result.steps.append(
                f"search_guidelines({risk['risk_code']}) -> "
                f"{len(found.get('snippets') or [])} snippet(s)"
            )

    return result


def _number(value: Any) -> str:
    """Trim float noise. A BMI of 28.73174689021093 is false precision."""
    if isinstance(value, float):
        return f"{value:g}" if value == int(value) else f"{value:.4g}"
    return str(value)


def _preview(text: str) -> str:
    """One readable line from a chunk.

    Citing a bullet list would otherwise preview as "- **Age**.", which tells the
    reader nothing about what was cited.
    """
    flat = " ".join(
        line.strip().lstrip("-").strip()
        for line in text.strip().splitlines()
        if line.strip()
    )
    flat = flat.replace("**", "")
    return flat[:140] + ("..." if len(flat) > 140 else "")


def render(result: ReviewResult) -> str:
    """Assemble a clinician-readable note from tool values only.

    No generation, so nothing here can be invented. Every number and every
    citation came from a tool response.
    """
    if not result.ok:
        return f"Could not complete the review: {result.error}"

    lines = [
        f"Risk review — {result.patient_name} ({result.patient_id})",
        "=" * 60,
    ]

    elevated = sorted(result.elevated(), key=lambda r: r["probability"], reverse=True)
    if not elevated:
        lines.append("No elevated risks: all five are in the low band.")
    else:
        lines.append("Elevated risks:")
        for risk in elevated:
            horizon = (
                f"{risk['time_horizon_years']}-year"
                if risk.get("time_horizon_years")
                else "screening score"
            )
            lines.append(
                f"  {risk['risk_code']:9} {risk['probability']:.3f}  "
                f"{risk['risk_band']:13} ({horizon}, trend {risk.get('trend_direction')})"
            )
            for driver in risk.get("drivers", []):
                lines.append(
                    f"      - {driver['label']}: {_number(driver['patient_value'])} "
                    f"(reference {_number(driver['reference_value'])}, "
                    f"{driver['direction']})"
                )

    if result.citations:
        lines += ["", "Guidance:"]
        for citation in result.citations:
            lines.append(f"  [{citation['risk_code']}] {citation['citation']}")
            lines.append(f'      "{_preview(citation["text"])}"')

    lines += [
        "",
        "Decision support only. Risk models are surrogates, not validated "
        "instruments; treatment decisions rest with the clinician.",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print('usage: python agent/router.py "<patient name>"', file=sys.stderr)
        return 2
    result = asyncio.run(review_patient(" ".join(argv)))
    print(render(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
