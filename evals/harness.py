"""Evaluation harness for the Longevity clinical assistant.

Two tiers, deliberately separated by what they need and what they can prove:

  Tier A  deterministic. Calls the MCP tools directly and checks the values the
          assistant would be repeating. No LLM, no API key, seconds to run - so
          it is the regression gate that can run on every commit.

  Tier B  agent in the loop. An OpenAI-compatible tool-calling loop against
          OpenRouter with the same MCP tools attached, scoring the trace and the
          prose. Needs a key and costs money, so it is opt-in.

Usage:
    uv run python evals/harness.py                  # Tier A only (default)
    uv run python evals/harness.py --tier b         # Tier B only
    uv run python evals/harness.py --tier both
    uv run python evals/harness.py --tier b --repeats 3 --model openai/gpt-4o-mini
    uv run python evals/harness.py --only safety    # filter by id, category or source

Exit code is non-zero when any case fails, so CI can gate on it.

Prerequisites: the MCP server, the FastAPI backend and the MLflow model server
must be running (scripts/run_stack.ps1). Tier B additionally needs OPENROUTER_KEY
in the environment, this repo's .env, or LibreChat's .env.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Runnable both as `python evals/harness.py` (the Makefile's `make eval`) and as
# `python -m evals.harness`. The first form has no package context, so put the
# repo root on the path before importing the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import mcp_tools, report, tier_a, tier_b  # noqa: E402
from evals.cases import load_cases  # noqa: E402
from evals.openrouter import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    find_api_key,
)
from evals.results import CaseResult  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="harness", description="Evaluate the clinical assistant."
    )
    parser.add_argument(
        "--tier", choices=["a", "b", "both"], default="a",
        help="which tier to run (default: a - deterministic, no API key)",
    )
    parser.add_argument(
        "--only", default=None,
        help="filter cases by id, category, or source (gold|extra)",
    )
    parser.add_argument(
        "--no-extra", action="store_true",
        help="run only the assignment's gold cases, skipping the added ones",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Tier B model")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Tier B runs per case; >1 reports stability across runs",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print results without writing files"
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(include_extra=not args.no_extra, only=args.only)
    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 2

    reachable, detail = await mcp_tools.probe()
    if not reachable:
        print(f"MCP server not reachable: {detail}", file=sys.stderr)
        print(
            "Start the host services first:\n"
            "  powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1",
            file=sys.stderr,
        )
        return 2
    print(f"MCP: {detail}")
    print(f"Cases: {len(cases)} ({sum(1 for c in cases if c.source == 'extra')} added)")

    runs: dict[str, list[CaseResult]] = {}
    meta: dict[str, object] = {
        "mcp_url": mcp_tools.MCP_URL,
        "cases": len(cases),
        "filter": args.only or "(none)",
    }

    if args.tier in {"a", "both"}:
        print("\nRunning Tier A (deterministic)...")
        runs["A"] = await tier_a.run(cases)

    if args.tier in {"b", "both"}:
        api_key = find_api_key()
        if not api_key:
            print(
                "\nTier B needs an OpenRouter key. Set OPENROUTER_KEY in the "
                "environment, this repo's .env, or LibreChat's .env.",
                file=sys.stderr,
            )
            if args.tier == "b":
                return 2
        else:
            print(f"\nRunning Tier B (model={args.model}, repeats={args.repeats})...")
            if ":free" in args.model:
                # Learned the hard way: a full sweep on the free default scored
                # 7.4% and looked like a broken system. It was not — the free
                # models are poor at tool calling, and a Tier B score is a
                # property of the MODEL at least as much as of the system under
                # test. The default stays free so the harness runs on an account
                # with no credit, but an unqualified low number invites exactly
                # the wrong conclusion, so say so before it is misread.
                print(
                    f"  NOTE: {args.model} is a free model with weak tool-calling.\n"
                    "  A low pass rate here measures the model, not the system.\n"
                    "  Reported results use: --model anthropic/claude-haiku-4.5",
                    file=sys.stderr,
                )
            meta.update(
                {
                    "tier_b_model": args.model,
                    "judge_model": args.judge_model,
                    "repeats": args.repeats,
                    "temperature": 0.0,
                }
            )
            runs["B"] = await tier_b.run(
                cases,
                model=args.model,
                judge_model=args.judge_model,
                api_key=api_key,
                repeats=args.repeats,
            )

    payload = report.build_payload(runs, meta)
    report.print_console(payload)

    if not args.no_write:
        json_path, md_path = report.write(payload)
        print(f"\nWrote {json_path.relative_to(json_path.parents[2])}")
        print(f"Wrote {md_path.relative_to(md_path.parents[2])}")

    failed = sum(block["summary"]["cases_failed"] for block in payload["tiers"].values())
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
