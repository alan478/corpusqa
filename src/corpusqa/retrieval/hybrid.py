"""A minimal hybrid retriever for the chunking eval (step C); the retrieval step extends it.

One question gives three rankings of documents:
- `bm25`: the best `top_chunks` chunks by BM25 over the chunk header and text;
- `dense`: the best `top_chunks` chunks by cosine to the question's embedding (query role);
- `hybrid`: both chunk lists fused by weighted reciprocal rank (score = sum over the two
  rankers of weight / (k + rank); weights `RRF_BM25_WEIGHT`, `RRF_DENSE_WEIGHT`, both 1.0 by
  default = plain RRF).
Each chunk list is mapped to its documents, keeping a document's best rank only.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from corpusqa import obs
from corpusqa.config import RETRIEVAL_TOP_CHUNKS, RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT, RRF_K
from corpusqa.llm.interfaces import Embedder
from corpusqa.store.interfaces import ChunkReader
from corpusqa.store.vectors import VectorIndex

type Mode = Literal["bm25", "dense", "hybrid"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid")


@dataclass(frozen=True, slots=True)
class Rankings:
    """Document ids per mode, best first, without repeats."""

    bm25: tuple[str, ...]
    dense: tuple[str, ...]
    hybrid: tuple[str, ...]

    def of(self, mode: Mode) -> tuple[str, ...]:
        return {"bm25": self.bm25, "dense": self.dense, "hybrid": self.hybrid}[mode]


def rrf(
    rankings: Sequence[Sequence[int]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[int]:
    """Weighted reciprocal rank fusion of ranked id lists (rank 1 = best).

    An id scores the sum of weight / (k + rank) over the lists that hold it; `weights` has
    one entry per list (None = 1.0 each, plain RRF). Ties go to the lower id, so an id found
    only by weight-0 lists still appears, after every id with a score above 0."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"rrf got {len(weights)} weights for {len(rankings)} rankings")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def to_documents(chunk_ids: Sequence[int], chunk_doc: Mapping[int, str]) -> tuple[str, ...]:
    """The documents of ranked chunks, each at the rank of its best chunk."""
    return tuple(dict.fromkeys(chunk_doc[chunk_id] for chunk_id in chunk_ids))


class HybridRetriever:
    """BM25 + dense over one chunk index. Without vectors or an embedder, dense is empty and
    hybrid equals bm25 (while `bm25_weight` > 0). A weight of 0 turns its ranker's vote off:
    its chunks then follow every chunk the other ranker scored."""

    def __init__(
        self,
        chunks: ChunkReader,
        vectors: VectorIndex | None = None,
        embedder: Embedder | None = None,
        *,
        top_chunks: int = RETRIEVAL_TOP_CHUNKS,
        rrf_k: int = RRF_K,
        bm25_weight: float = RRF_BM25_WEIGHT,
        dense_weight: float = RRF_DENSE_WEIGHT,
    ) -> None:
        if not all(math.isfinite(w) and w >= 0 for w in (bm25_weight, dense_weight)):
            raise ValueError(
                f"fusion weights must be finite and >= 0, got {bm25_weight}, {dense_weight}"
            )
        self._chunks = chunks
        self._vectors = vectors if vectors is not None and len(vectors) else None
        self._embedder = embedder
        self._top = top_chunks
        self._rrf_k = rrf_k
        self._weights = (bm25_weight, dense_weight)
        self._chunk_doc = chunks.chunk_doc_ids()

    @property
    def has_dense(self) -> bool:
        return self._vectors is not None and self._embedder is not None

    def search(self, question: str) -> Rankings:
        with obs.span(
            "retrieve",
            input={"question": question},
            metadata={
                "top_chunks": self._top,
                "rrf_k": self._rrf_k,
                "bm25_weight": self._weights[0],
                "dense_weight": self._weights[1],
                "dense": self.has_dense,
            },
        ) as span:
            keyword_query = {
                "fts_expression": self._chunks.keyword_expression(question),
                "exact_phrases": self._chunks.exact_phrases,
            }
            with obs.span("bm25", input={"query": question}, metadata=keyword_query) as step:
                bm25 = self._chunks.bm25(question, self._top)
                step.update(output=self._traced(bm25))
            dense: list[tuple[int, float]] = []
            if self._vectors is not None and self._embedder is not None:
                with obs.span("dense", input={"query": question}) as step:
                    [query_vector] = self._embedder.embed([question], "query")
                    dense = self._vectors.search(query_vector, self._top)
                    step.update(output=self._traced(dense))
            fused = rrf([[c for c, _ in bm25], [c for c, _ in dense]], self._rrf_k, self._weights)
            result = Rankings(
                bm25=to_documents([c for c, _ in bm25], self._chunk_doc),
                dense=to_documents([c for c, _ in dense], self._chunk_doc),
                hybrid=to_documents(fused, self._chunk_doc),
            )
            span.update(output={mode: list(result.of(mode)) for mode in MODES})
        return result

    def _traced(self, hits: list[tuple[int, float]]) -> list[dict[str, object]]:
        return [
            {"chunk_id": c, "doc_id": self._chunk_doc[c], "score": round(score, 4)}
            for c, score in hits
        ]
