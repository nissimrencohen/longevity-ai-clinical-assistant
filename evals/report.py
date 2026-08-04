"""Result persistence and rendering.

Every run writes a timestamped JSON (the full record, including tool traces,
answers and judge reasoning — so a failure can be re-read months later) plus a
markdown summary for humans.

`results/latest.md` and `results/latest.json` always point at the most recent
run, which is what makes this usable as a diff in review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.results import CaseResult, summarise

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def build_payload(
    runs: dict[str, list[CaseResult]], meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta,
        "tiers": {
            tier: {
                "summary": summarise(cases),
                "cases": [c.to_dict() for c in cases],
            }
            for tier, cases in runs.items()
        },
    }


def _tier_markdown(tier: str, cases: list[CaseResult]) -> list[str]:
    summary = summarise(cases)
    lines = [f"## Tier {tier}", ""]

    label = {
        "A": "deterministic, MCP tools called directly, no LLM",
        "B": "agent in the loop via OpenRouter",
    }.get(tier, "")
    if label:
        lines += [f"_{label}_", ""]

    lines += [
        f"**Pass rate: {_pct(summary['pass_rate'])}** "
        f"({summary['cases_passed']} passed, {summary['cases_failed']} failed, "
        f"{summary['cases_skipped']} skipped, {summary.get('cases_errored', 0)} errored "
        f"of {summary['cases_total']} runs)",
        "",
        "| Category | Pass | Fail | Skip | Error | Pass rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, bucket in sorted(summary["by_category"].items()):
        lines.append(
            f"| {category} | {bucket['pass']} | {bucket['fail']} | "
            f"{bucket['skip']} | {bucket.get('error', 0)} | {_pct(bucket['pass_rate'])} |"
        )

    lines += [
        "",
        "| Axis | Pass | Fail | Skip | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis, bucket in sorted(summary["by_axis"].items()):
        lines.append(
            f"| {axis} | {bucket['pass']} | {bucket['fail']} | "
            f"{bucket['skip']} | {_pct(bucket['pass_rate'])} |"
        )

    errored = [c for c in cases if c.status == "error"]
    if errored:
        lines += [
            "",
            f"> **{len(errored)} run(s) errored before the model answered** — "
            "infrastructure, not model quality. Excluded from the pass rate.",
            "",
        ]
        for case in sorted({c.case_id for c in errored}):
            first = next(c for c in errored if c.case_id == case)
            lines.append(f"- `{case}`: {(first.error or '')[:160]}")
        lines.append("")

    # Per-case stability across repeats — a single run of a stochastic agent is
    # an anecdote, so show how often each case passed.
    grouped: dict[str, list[CaseResult]] = {}
    for case in cases:
        grouped.setdefault(case.case_id, []).append(case)
    if any(len(v) > 1 for v in grouped.values()):
        lines += ["", "### Stability across repeats", "", "| Case | Passed |", "|---|---|"]
        for case_id, runs in sorted(grouped.items()):
            passed = sum(1 for r in runs if r.status == "pass")
            lines.append(f"| `{case_id}` | {passed}/{len(runs)} |")

    failures = [c for c in cases if c.status == "fail"]
    if failures:
        lines += ["", "### Failures", ""]
        for case in failures:
            lines.append(f"**`{case.case_id}`** ({case.category})")
            if case.error:
                lines.append(f"- run error: {case.error}")
            for check in case.failures:
                detail = f" - {check.detail}" if check.detail else ""
                lines.append(
                    f"- `{check.name}` [{check.axis}] expected `{check.expected}`, "
                    f"got `{check.actual}`{detail}"
                )
            lines.append("")
    else:
        lines += ["", "No failures.", ""]

    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# Longevity Clinical AI — evaluation report",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key, value in meta.items():
        lines.append(f"| {key} | `{value}` |")
    lines.append("")

    for tier, block in payload["tiers"].items():
        cases = [_rehydrate(c) for c in block["cases"]]
        lines += _tier_markdown(tier, cases)

    return "\n".join(lines).rstrip() + "\n"


def _rehydrate(data: dict[str, Any]) -> CaseResult:
    from evals.results import Check

    case = CaseResult(
        case_id=data["case_id"],
        category=data["category"],
        source=data["source"],
        tier=data["tier"],
        tool_calls=data.get("tool_calls", []),
        answer=data.get("answer"),
        error=data.get("error"),
        duration_s=data.get("duration_s", 0.0),
        repeat=data.get("repeat", 0),
    )
    case.checks = [Check(**c) for c in data.get("checks", [])]
    return case


def write(payload: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = payload["generated_at"].replace(":", "").replace("-", "")
    json_path = RESULTS_DIR / f"{stamp}.json"
    md_path = RESULTS_DIR / f"{stamp}.md"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    markdown = render_markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")

    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")
    return json_path, md_path


def print_console(payload: dict[str, Any]) -> None:
    for tier, block in payload["tiers"].items():
        summary = block["summary"]
        print(f"\nTier {tier}: {_pct(summary['pass_rate'])} "
              f"({summary['cases_passed']}P / {summary['cases_failed']}F / "
              f"{summary['cases_skipped']}S / {summary.get('cases_errored', 0)}E "
              f"of {summary['cases_total']})")
        for category, bucket in sorted(summary["by_category"].items()):
            print(f"  {category:24} {_pct(bucket['pass_rate']):>7}  "
                  f"({bucket['pass']}P {bucket['fail']}F {bucket['skip']}S "
                  f"{bucket.get('error', 0)}E)")
        if summary["failed_case_ids"]:
            print(f"  failed:  {', '.join(sorted(set(summary['failed_case_ids'])))}")
        if summary.get("errored_case_ids"):
            print(f"  errored: {', '.join(sorted(set(summary['errored_case_ids'])))}"
                  "  <- infrastructure, excluded from pass rate")
