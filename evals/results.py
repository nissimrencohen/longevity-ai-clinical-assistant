"""Result types and aggregation.

A check is `pass`, `fail`, or `skip`. Skips are load-bearing: Tier A cannot judge
whether prose hedged appropriately, so it marks those checks skipped rather than
passing them. Pass rate is computed over pass+fail only, so skipped checks never
inflate a score — a suite that silently auto-passed what it cannot measure would
be worse than no suite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PASS = "pass"
FAIL = "fail"
SKIP = "skip"
# The run itself broke (provider outage, rate limit, exhausted credits). NOT a
# model failure, and must never be counted as one — conflating "the assistant got
# it wrong" with "we ran out of credit" makes the pass rate meaningless.
ERROR = "error"


@dataclass
class Check:
    name: str
    axis: str
    status: str
    detail: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    case_id: str
    category: str
    source: str
    tier: str
    checks: list[Check] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    repeat: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return ERROR
        if any(c.status == FAIL for c in self.checks):
            return FAIL
        if any(c.status == PASS for c in self.checks):
            return PASS
        return SKIP

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status
        return data


def _rate(passed: int, failed: int) -> float | None:
    total = passed + failed
    return None if total == 0 else round(passed / total, 4)


def summarise(cases: list[CaseResult]) -> dict[str, Any]:
    """Aggregate overall, per category, and per axis.

    Pass rate is passed / (passed + failed). Skipped checks (not assertable at
    this tier) and errored runs (infrastructure, not the model) are reported
    separately and never move the number.
    """
    by_category: dict[str, dict[str, int]] = {}
    by_axis: dict[str, dict[str, int]] = {}

    def blank() -> dict[str, int]:
        return {"pass": 0, "fail": 0, "skip": 0, "error": 0}

    for case in cases:
        bucket = by_category.setdefault(case.category, blank())
        bucket[case.status] += 1
        for check in case.checks:
            axis = by_axis.setdefault(check.axis, blank())
            axis[check.status] += 1

    for bucket in list(by_category.values()) + list(by_axis.values()):
        bucket["pass_rate"] = _rate(bucket["pass"], bucket["fail"])  # type: ignore[assignment]

    passed = sum(1 for c in cases if c.status == PASS)
    failed = sum(1 for c in cases if c.status == FAIL)
    skipped = sum(1 for c in cases if c.status == SKIP)
    errored = sum(1 for c in cases if c.status == ERROR)

    return {
        "cases_total": len(cases),
        "cases_passed": passed,
        "cases_failed": failed,
        "cases_skipped": skipped,
        "cases_errored": errored,
        "pass_rate": _rate(passed, failed),
        "by_category": by_category,
        "by_axis": by_axis,
        "failed_case_ids": [c.case_id for c in cases if c.status == FAIL],
        "errored_case_ids": [c.case_id for c in cases if c.status == ERROR],
    }
