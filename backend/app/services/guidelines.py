"""Retrieval over the guideline corpus, with citations that can be checked.

THE POINT IS THE CITATION, NOT THE RETRIEVER. A plausible-looking citation to
text that does not exist is worse than no citation: it launders an invented claim
as a sourced one. So every chunk carries its source file, its heading and its
exact line span, and `verify_citation` re-reads the file on disk to confirm the
quoted text is really there. That check is deterministic and free, which means
the eval harness can run it on every case rather than asking a model whether a
citation "looks right".

CHUNKING is heading-level, not fixed-token. The corpus is five short documents
with clean `##` sections ("What it estimates", "Risk factors used", "Modifiable
levers"), and those sections are already the unit a clinician would cite.
Splitting them into 200-token windows would cut mid-sentence and produce
citations that point at fragments.

RETRIEVAL is behind a protocol with two implementations:

* ``LexicalRetriever`` (default) — TF-IDF over the chunks, using scikit-learn,
  which is already a dependency. Deterministic, instant, no model download, and
  therefore usable in CI and in the free eval tier.
* ``EmbeddingRetriever`` (opt-in, ``uv sync --extra rag``) — Chroma with its
  ONNX embedder, for semantic matches the lexical path misses.

Being honest about the trade: at five documents and a vocabulary this narrow,
lexical retrieval is competitive and has the large advantage of being
reproducible. Embeddings earn their keep when the corpus grows; the protocol
exists so that swap is a config change rather than a rewrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
GUIDELINES_DIR = REPO_ROOT / "data" / "guidelines"

# Which document belongs to which risk, so a dementia question does not retrieve
# the liver page. Derived from the filenames the corpus ships with.
RISK_TO_FILE: dict[str, str] = {
    "CVD": "cvd_prevent.md",
    "T2DM": "t2dm_ada.md",
    "CKD": "ckd_framingham.md",
    "CLD": "cld_clivd.md",
    "DEMENTIA": "dementia_caide.md",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class GuidelineChunk:
    """One citable section of one document."""

    source_file: str
    heading: str
    text: str
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    risk_code: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.source_file} § {self.heading}"

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "heading": self.heading,
            "citation": self.citation,
            "lines": [self.line_start, self.line_end],
            "risk_code": self.risk_code,
            "text": self.text,
        }


def _risk_for_file(filename: str) -> str | None:
    for risk_code, name in RISK_TO_FILE.items():
        if name == filename:
            return risk_code
    return None


def chunk_markdown(path: Path) -> list[GuidelineChunk]:
    """Split one document into heading-level chunks, tracking line spans."""
    lines = path.read_text(encoding="utf-8").splitlines()
    risk_code = _risk_for_file(path.name)

    chunks: list[GuidelineChunk] = []
    heading = path.stem
    level = 1
    body_start = 1
    body: list[str] = []

    def flush(end_line: int) -> None:
        # Skip the document preamble — the block under the `#` title is a
        # provenance note ("Simplified educational summary... Source: ..."), not
        # clinical content. It was outranking the real sections for "what drives
        # dementia risk", which is a citation pointing at metadata.
        if level < 2:
            return
        text = "\n".join(body).strip()
        if text:
            chunks.append(
                GuidelineChunk(
                    source_file=path.name,
                    heading=heading,
                    text=text,
                    line_start=body_start,
                    line_end=end_line,
                    risk_code=risk_code,
                )
            )

    for number, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            flush(number - 1)
            level = len(match.group(1))
            heading = match.group(2).strip()
            body_start = number + 1
            body = []
        else:
            body.append(line)
    flush(len(lines))
    return chunks


@lru_cache(maxsize=1)
def load_corpus() -> tuple[GuidelineChunk, ...]:
    """Every citable chunk. README is excluded — it describes the corpus."""
    if not GUIDELINES_DIR.exists():
        return ()
    chunks: list[GuidelineChunk] = []
    for path in sorted(GUIDELINES_DIR.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_markdown(path))
    return tuple(chunks)


# ---------------------------------------------------------------------------
# Citation verification — the part that makes a citation worth anything
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_citation(
    source_file: str, heading: str | None = None, quote: str | None = None
) -> dict:
    """Check a citation against the file on disk.

    Returns what was and was not confirmed, so a caller can distinguish "wrong
    file" from "right file, invented quote". A model that cites a real document
    and then paraphrases beyond it is the more likely failure, and the one a
    human reviewer is least likely to catch.
    """
    path = GUIDELINES_DIR / source_file
    result = {
        "source_file": source_file,
        "file_exists": path.is_file(),
        "heading_exists": None,
        "quote_found": None,
        "lines": None,
    }
    if not result["file_exists"]:
        return result

    chunks = [c for c in load_corpus() if c.source_file == source_file]

    if heading is not None:
        match = next(
            (c for c in chunks if _normalise(c.heading) == _normalise(heading)), None
        )
        result["heading_exists"] = match is not None
        if match:
            result["lines"] = [match.line_start, match.line_end]

    if quote is not None:
        haystack = _normalise(path.read_text(encoding="utf-8"))
        result["quote_found"] = _normalise(quote) in haystack

    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class Retriever(Protocol):
    def search(
        self, query: str, k: int = 3, risk_code: str | None = None
    ) -> list[tuple[GuidelineChunk, float]]:
        ...


class LexicalRetriever:
    """TF-IDF over the chunks. Deterministic, dependency-free, instant."""

    def __init__(self, chunks: tuple[GuidelineChunk, ...] | None = None) -> None:
        self._chunks = chunks if chunks is not None else load_corpus()
        self._vectorizer = None
        self._matrix = None
        if self._chunks:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                stop_words="english", sublinear_tf=True, ngram_range=(1, 2)
            )
            self._matrix = self._vectorizer.fit_transform(
                [f"{c.heading}\n{c.text}" for c in self._chunks]
            )

    def search(
        self, query: str, k: int = 3, risk_code: str | None = None
    ) -> list[tuple[GuidelineChunk, float]]:
        if not self._chunks or self._vectorizer is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(
            self._vectorizer.transform([query]), self._matrix
        )[0]

        candidates = [
            (chunk, float(score))
            for chunk, score in zip(self._chunks, scores)
            # Filtering by risk keeps a dementia question off the liver page.
            if risk_code is None or chunk.risk_code == risk_code
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in candidates[:k] if pair[1] > 0.0]


def build_retriever(backend: str = "lexical") -> Retriever:
    """Pick a retriever. Lexical is the default — see the module docstring."""
    if backend == "embedding":
        from .guidelines_embedding import EmbeddingRetriever

        return EmbeddingRetriever()
    return LexicalRetriever()
