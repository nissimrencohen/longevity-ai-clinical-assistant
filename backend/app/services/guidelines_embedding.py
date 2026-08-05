"""Embedding-based retrieval over the guideline corpus (opt-in).

    uv sync --extra rag        # installs chromadb

Kept in its own module so importing the default lexical path never pulls chromadb
in. `build_retriever("embedding")` is the only caller.

Chroma's default embedding function is ONNX-based — no torch, no GPU, no model
server — which is what makes this viable as an option rather than a second
deployment. It runs in-process against an ephemeral client, because the corpus is
five documents and rebuilding the index costs milliseconds; persisting it would
add a volume and a staleness question for no benefit at this size.

The chunk metadata (source file, heading, line span) travels with each embedded
document, so citations are verified exactly the same way as on the lexical path.
Retrieval strategy changes; citation integrity does not.
"""

from __future__ import annotations

from .guidelines import GuidelineChunk, load_corpus


class EmbeddingRetriever:
    """Semantic retrieval via Chroma. Requires the `rag` extra."""

    def __init__(self, chunks: tuple[GuidelineChunk, ...] | None = None) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Embedding retrieval needs chromadb. Install it with "
                "`uv sync --extra rag`, or use the default lexical retriever."
            ) from exc

        self._chunks = chunks if chunks is not None else load_corpus()
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection("guidelines")

        if self._chunks:
            self._collection.add(
                ids=[f"{c.source_file}:{c.line_start}" for c in self._chunks],
                documents=[f"{c.heading}\n{c.text}" for c in self._chunks],
                metadatas=[
                    {
                        "source_file": c.source_file,
                        "heading": c.heading,
                        "line_start": c.line_start,
                        "line_end": c.line_end,
                        "risk_code": c.risk_code or "",
                    }
                    for c in self._chunks
                ],
            )

    def search(
        self, query: str, k: int = 3, risk_code: str | None = None
    ) -> list[tuple[GuidelineChunk, float]]:
        if not self._chunks:
            return []

        where = {"risk_code": risk_code} if risk_code else None
        result = self._collection.query(
            query_texts=[query], n_results=min(k, len(self._chunks)), where=where
        )

        by_id = {f"{c.source_file}:{c.line_start}": c for c in self._chunks}
        hits: list[tuple[GuidelineChunk, float]] = []
        for doc_id, distance in zip(
            result.get("ids", [[]])[0], result.get("distances", [[]])[0]
        ):
            chunk = by_id.get(doc_id)
            if chunk is not None:
                # Chroma returns a distance; report similarity for parity with
                # the lexical retriever's cosine score.
                hits.append((chunk, max(0.0, 1.0 - float(distance))))
        return hits
