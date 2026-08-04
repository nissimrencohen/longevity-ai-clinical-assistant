"""Loading and typing of the gold eval cases.

Two files feed the suite:

* ``cases.jsonl``       — the gold cases shipped with the assignment. Not edited.
* ``cases_extra.jsonl`` — cases added here, mostly for failure modes the gold set
  does not reach (mid-conversation hallucination, nearest-name substitution,
  out-of-scope refusal, the NULL-input regression).

The schema is the assignment's, with two backward-compatible additions:
``turns`` for multi-turn cases and ``patient_ids`` for cases spanning more than
one patient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
GOLD_CASES = EVALS_DIR / "cases.jsonl"
EXTRA_CASES = EVALS_DIR / "cases_extra.jsonl"

# Fact kinds that describe how the assistant should BEHAVE rather than a value
# that can be checked against a tool response. Tier A cannot judge these, and
# marks them skipped rather than passing them for free.
BEHAVIOURAL_KINDS = {"safety", "no_fabrication", "citation"}


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    expected_tool: str | None
    patient_id: str | None
    expected_facts: list[dict[str, Any]]
    notes: str = ""
    turns: list[str] = field(default_factory=list)
    patient_ids: list[str] = field(default_factory=list)
    source: str = "gold"

    @property
    def conversation(self) -> list[str]:
        """User messages in order — single-turn cases are a one-item list."""
        return self.turns or [self.question]

    @property
    def target_patients(self) -> list[str]:
        """Every patient this case touches."""
        if self.patient_ids:
            return list(self.patient_ids)
        return [self.patient_id] if self.patient_id else []

    @property
    def is_behavioural_only(self) -> bool:
        return all(f.get("kind") in BEHAVIOURAL_KINDS for f in self.expected_facts)


def _load_file(path: Path, source: str) -> list[Case]:
    if not path.exists():
        return []
    cases: list[Case] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no} is not valid JSON: {exc}") from exc
        cases.append(
            Case(
                id=raw["id"],
                category=raw["category"],
                question=raw["question"],
                expected_tool=raw.get("expected_tool"),
                patient_id=raw.get("patient_id"),
                expected_facts=raw.get("expected_facts", []),
                notes=raw.get("notes", ""),
                turns=raw.get("turns", []),
                patient_ids=raw.get("patient_ids", []),
                source=source,
            )
        )
    return cases


def load_cases(*, include_extra: bool = True, only: str | None = None) -> list[Case]:
    """Load gold cases, optionally plus the added ones. ``only`` filters by id or category."""
    cases = _load_file(GOLD_CASES, "gold")
    if include_extra:
        cases += _load_file(EXTRA_CASES, "extra")

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate case id: {case.id}")
        seen.add(case.id)

    if only:
        cases = [c for c in cases if only in (c.id, c.category, c.source)]
    return cases
