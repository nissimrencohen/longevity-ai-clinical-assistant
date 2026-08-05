"""Retrieval and — the part that matters — citation integrity.

A plausible-looking citation to text that does not exist is worse than no
citation: it launders an invented claim as a sourced one, and a human reviewer is
unlikely to open the file and check. So these tests care less about ranking
quality than about whether a citation can be mechanically confirmed against the
corpus on disk.
"""

from __future__ import annotations

import pytest

from backend.app.services.guidelines import (
    GUIDELINES_DIR,
    RISK_TO_FILE,
    LexicalRetriever,
    chunk_markdown,
    load_corpus,
    verify_citation,
)

# ---------------------------------------------------------------------------
# Chunking: heading-level, with line spans that must be exact
# ---------------------------------------------------------------------------


def test_corpus_loads_and_excludes_the_readme() -> None:
    chunks = load_corpus()
    assert chunks, "no guideline chunks were loaded"
    assert not any(c.source_file.lower() == "readme.md" for c in chunks), (
        "README describes the corpus; it is not citable material"
    )


def test_every_risk_document_is_represented() -> None:
    covered = {c.risk_code for c in load_corpus() if c.risk_code}
    assert covered == set(RISK_TO_FILE)


def test_chunks_are_heading_level() -> None:
    """Sections are the unit a clinician would cite; token windows are not."""
    chunks = chunk_markdown(GUIDELINES_DIR / "ckd_framingham.md")
    headings = [c.heading for c in chunks]
    assert "What it estimates" in headings
    assert "Risk factors used" in headings


def test_document_preamble_is_not_citable() -> None:
    """The block under the `#` title is provenance, not clinical content.

    It was outranking the real sections for "what drives dementia risk", which
    produced a citation pointing at a disclaimer.
    """
    for chunk in load_corpus():
        assert "Simplified educational summary" not in chunk.text
        assert not chunk.heading.endswith("(summary)")


def test_line_spans_point_at_the_real_text() -> None:
    """The span is part of the citation, so it has to be right.

    Re-reads each file and checks the recorded lines actually contain the chunk
    text — an off-by-one here would make every citation subtly wrong.
    """
    for chunk in load_corpus():
        lines = (GUIDELINES_DIR / chunk.source_file).read_text(
            encoding="utf-8"
        ).splitlines()
        span = "\n".join(lines[chunk.line_start - 1 : chunk.line_end]).strip()
        assert chunk.text in span or span in chunk.text, (
            f"{chunk.citation}: lines {chunk.line_start}-{chunk.line_end} "
            "do not contain the chunk text"
        )


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------


def test_real_citation_is_confirmed() -> None:
    result = verify_citation(
        "ckd_framingham.md",
        heading="Risk factors used",
        quote="a reduced eGFR raises risk sharply",
    )
    assert result["file_exists"]
    assert result["heading_exists"]
    assert result["quote_found"]
    assert result["lines"]


def test_invented_file_is_caught() -> None:
    result = verify_citation("aha_guidelines_2024.md", heading="Statins")
    assert result["file_exists"] is False


def test_invented_heading_in_a_real_file_is_caught() -> None:
    """The likelier failure: cite a real document, invent the section."""
    result = verify_citation("ckd_framingham.md", heading="Recommended Drug Therapy")
    assert result["file_exists"] is True
    assert result["heading_exists"] is False


def test_invented_quote_in_a_real_file_is_caught() -> None:
    """The likeliest failure: real document, real heading, paraphrased beyond it."""
    result = verify_citation(
        "ckd_framingham.md",
        heading="Risk factors used",
        quote="start an ACE inhibitor at 10 mg daily",
    )
    assert result["file_exists"] is True
    assert result["heading_exists"] is True
    assert result["quote_found"] is False


def test_verification_is_whitespace_insensitive() -> None:
    """A model re-wrapping a quote has not invented it."""
    result = verify_citation(
        "ckd_framingham.md", quote="a reduced eGFR\n   raises risk    sharply"
    )
    assert result["quote_found"] is True


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever() -> LexicalRetriever:
    return LexicalRetriever()


def test_dementia_question_retrieves_the_dementia_document(retriever) -> None:
    hits = retriever.search("what factors drive dementia risk", k=3)
    assert hits
    assert hits[0][0].source_file == "dementia_caide.md"


def test_kidney_question_retrieves_the_ckd_document(retriever) -> None:
    hits = retriever.search("chronic kidney disease proteinuria eGFR", k=3)
    assert hits[0][0].source_file == "ckd_framingham.md"


def test_risk_code_filter_keeps_a_question_on_topic(retriever) -> None:
    """Without this a dementia question can surface the liver page."""
    hits = retriever.search("risk factors", k=5, risk_code="DEMENTIA")
    assert hits
    assert {c.source_file for c, _ in hits} == {"dementia_caide.md"}


def test_k_is_respected(retriever) -> None:
    assert len(retriever.search("risk", k=2)) <= 2


def test_no_results_rather_than_a_bad_one(retriever) -> None:
    """Returning nothing is better than returning something irrelevant.

    An unrelated query should come back empty so the assistant says it has no
    guidance, rather than citing whatever scored least badly.
    """
    assert retriever.search("orbital mechanics of geostationary satellites") == []


def test_every_hit_is_independently_verifiable(retriever) -> None:
    """The contract: anything retrieval returns can be confirmed on disk."""
    for query in ("dementia risk factors", "liver disease alcohol", "diabetes screening"):
        for chunk, _score in retriever.search(query, k=3):
            result = verify_citation(
                chunk.source_file, heading=chunk.heading, quote=chunk.text[:80]
            )
            assert result["file_exists"]
            assert result["heading_exists"]
            assert result["quote_found"], f"{chunk.citation} did not verify"


def test_citation_string_is_stable_and_quotable(retriever) -> None:
    chunk, _ = retriever.search("dementia risk factors", k=1)[0]
    assert chunk.citation == f"{chunk.source_file} § {chunk.heading}"
    assert " § " in chunk.citation
