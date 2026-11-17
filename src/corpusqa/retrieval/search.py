"""Search for the agent: hybrid ranking (BM25 + dense, fused by weighted RRF, as in
`hybrid.py`) inside a scope, as pages of documents with their best passages.

The scope (a source, a created range, a set of documents such as a filter result) is
applied before ranking: BM25 and dense both rank only the chunks of the documents in
scope, so a large set is searched whole (not "rank everything, then drop", where the top
cut would lose most of it).

Depth (user decision 2026-10-16, "Search deeper"). The first ranking keeps each ranker's
best `top_chunks` chunks (the setting the retrieval eval validated) and fuses them; its
documents, each at its best chunk, start the list, so page 1 is the validated hybrid. A
page past what the list holds re-ranks with SEARCH_DEPTH_GROWTH times more chunks per
ranker (same RRF, same weights), again until the page is filled or every chunk in scope
is ranked. A deeper ranking adds its new documents at the end of the list, in its own
order; the documents found before keep their places. Rankers break ties by chunk id, so
a shallow ranking is the start of a deep one: the list depends only on the query and the
scope, never on the page asked for. Pages never repeat or skip a document, and the same
call gives the same page. The total is exact once every chunk in scope is ranked; before
that a page says only that more follow.

A created range uses the document's own created date (always inside the date window: a
date outside counts as no date); the result says how many documents in scope have no
usable created date and were left out.

Keyword matches (exact mode, config SEARCH_EXACT_PHRASES, read from the chunk reader).
The keyword side matches quoted phrases and key-shaped tokens exactly, so it can find
nothing while the dense side still returns near neighbours. Each hit then says whether a
passage of its document matches the keyword query at any rank (not just in the ranked
top: past it, the page's documents are checked one by one), and the result says whether
any document in scope matched. Nothing is dropped: a document found by meaning only stays
in the list, marked. Off: hits carry no mark (None), as before.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from corpusqa import obs
from corpusqa.config import (
    RETRIEVAL_TOP_CHUNKS,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
    SEARCH_DEPTH_GROWTH,
    SEARCH_SNIPPET_CHARS,
    SEARCH_SNIPPETS_PER_DOC,
)
from corpusqa.retrieval.hybrid import rrf, to_documents
from corpusqa.retrieval.paging import Page, page_of
from corpusqa.store.interfaces import ChunkReader, DocSummary
from corpusqa.store.vectors import VectorIndex

_WORD = re.compile(r"[^\W_]+")  # a word as the keyword index sees one
_ELLIPSIS = "..."


@dataclass(frozen=True, slots=True)
class Scope:
    """Where to search; every part optional (none: the whole corpus)."""

    source: str | None = None
    created_from: str | None = None  # ISO date, inclusive
    created_before: str | None = None  # ISO date, exclusive
    within: tuple[str, ...] | None = None  # these documents only

    @property
    def is_everything(self) -> bool:
        return (
            self.source is None
            and self.created_from is None
            and self.created_before is None
            and self.within is None
        )


@dataclass(frozen=True, slots=True)
class Hit:
    doc_id: str
    snippets: tuple[str, ...]
    # Exact mode: a passage of the document matches the keyword query (False: found by
    # the dense side only). None when exact mode is off (not checked).
    keyword: bool | None = None


@dataclass(frozen=True, slots=True)
class Ranking:
    """The documents found for a query in a scope, as deep as one page needed."""

    docs: tuple[str, ...]  # best first, each once
    chunks: tuple[int, ...]  # every ranked chunk, in the order passages are picked
    depth: int  # chunks per ranker in the deepest ranking run
    rankings: int  # how many rankings were run (1 + the times it went deeper)
    complete: bool  # every chunk in scope is ranked: `docs` is the whole list
    keyword_docs: frozenset[str]  # documents with a chunk the keyword side ranked
    # The keyword side ranked every chunk in scope that it matches: a document outside
    # `keyword_docs` has no matching passage.
    keyword_complete: bool


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    page: Page[Hit]
    scope_docs: int | None  # documents in scope (None: the whole corpus)
    undated: int  # documents in scope a created range could not check (no usable date)
    dense: bool  # the dense ranker took part
    exact_phrases: bool = False  # exact mode: hits carry `keyword`
    keyword_found: bool = False  # some document in scope matches the keyword query


class CorpusSearch:
    """Scoped hybrid search over one index. Holds no per-request state."""

    def __init__(
        self,
        chunks: ChunkReader,
        summaries: Sequence[DocSummary],
        vectors: VectorIndex | None = None,
        *,
        top_chunks: int = RETRIEVAL_TOP_CHUNKS,
        rrf_k: int = RRF_K,
        bm25_weight: float = RRF_BM25_WEIGHT,
        dense_weight: float = RRF_DENSE_WEIGHT,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors if vectors is not None and len(vectors) else None
        self._top = top_chunks
        self._rrf_k = rrf_k
        self._weights = (bm25_weight, dense_weight)
        self._chunk_doc = chunks.chunk_doc_ids()
        self._doc_chunks: dict[str, list[int]] = {}
        for chunk_id, doc_id in sorted(self._chunk_doc.items()):
            self._doc_chunks.setdefault(doc_id, []).append(chunk_id)
        self._summaries: Mapping[str, DocSummary] = {s.doc_id: s for s in summaries}

    @property
    def has_vectors(self) -> bool:
        return self._vectors is not None

    def search(
        self,
        query: str,
        scope: Scope,
        offset: int,
        size: int,
        query_vector: Sequence[float] | None = None,
    ) -> SearchResult:
        """One page of the documents that best match `query` in `scope`, ranked as deep
        as the page needs; dense takes part when there are vectors and `query_vector` is
        given."""
        given = {
            "query": query,
            "scope": {
                "source": scope.source,
                "created_from": scope.created_from,
                "created_before": scope.created_before,
                "within": None if scope.within is None else len(scope.within),
            },
            "offset": offset,
        }
        with obs.span("search", input=given) as span:
            docs, undated = self._scope_docs(scope)
            allowed = None
            if docs is not None:
                allowed = {c for d in docs for c in self._doc_chunks.get(d, ())}
            vector = query_vector if self._vectors is not None else None
            ranking = self._rank(query, allowed, vector, offset + size)
            page = page_of(ranking.docs, offset, size, exact=ranking.complete)
            exact = self._chunks.exact_phrases
            keyword = self._keyword_flags(query, page.items, ranking) if exact else {}
            hits = self._hits(page.items, ranking.chunks, query, keyword)
            result = SearchResult(
                query=query,
                page=Page(tuple(hits), page.offset, page.total, page.exact),
                scope_docs=None if docs is None else len(docs),
                undated=undated,
                dense=vector is not None,
                exact_phrases=exact,
                keyword_found=bool(ranking.keyword_docs),
            )
            span.update(
                output={"total": page.total, "exact": page.exact, "page": list(page.items)},
                metadata={
                    "scope_docs": result.scope_docs,
                    "scope_chunks": None if allowed is None else len(allowed),
                    "undated": undated,
                    "dense": result.dense,
                    "top_chunks": self._top,
                    "depth": ranking.depth,
                    "rankings": ranking.rankings,
                    "complete": ranking.complete,
                    "exact_phrases": exact,
                    "keyword_found": result.keyword_found,
                    "meaning_only": [d for d, found in keyword.items() if not found]
                    if exact
                    else None,
                },
            )
        return result

    def _rank(
        self,
        query: str,
        allowed: set[int] | None,
        vector: Sequence[float] | None,
        needed: int,
    ) -> Ranking:
        """Rankings ever deeper (from `top_chunks`, times SEARCH_DEPTH_GROWTH each time)
        until more than `needed` documents are found (so a next page is known to exist) or
        every chunk in scope is ranked. Each ranking adds its new documents and chunks at
        the end, in its own fused order."""
        in_scope = len(self._chunk_doc) if allowed is None else len(allowed)
        docs: dict[str, None] = {}
        chunks: dict[int, None] = {}
        keyword_docs: set[str] = set()
        depth, rankings = self._top, 0
        # The keyword query as the store runs it, and how it read the query (traced).
        keyword_query = {
            "fts_expression": self._chunks.keyword_expression(query),
            "exact_phrases": self._chunks.exact_phrases,
        }
        while True:
            rankings += 1
            bm25: list[tuple[int, float]] = []
            dense: list[tuple[int, float]] = []
            if in_scope:
                with obs.span(
                    "bm25", input={"query": query, "depth": depth}, metadata=keyword_query
                ) as step:
                    bm25 = self._chunks.bm25(query, depth, allowed)
                    step.update(output=self._traced(bm25))
                if vector is not None and self._vectors is not None:
                    with obs.span("dense", input={"query": query, "depth": depth}) as step:
                        dense = self._vectors.search(vector, depth, allowed)
                        step.update(output=self._traced(dense))
            fused = rrf([[c for c, _ in bm25], [c for c, _ in dense]], self._rrf_k, self._weights)
            docs.update(dict.fromkeys(to_documents(fused, self._chunk_doc)))
            chunks.update(dict.fromkeys(fused))
            keyword_docs.update(self._chunk_doc[c] for c, _ in bm25)
            # A ranker that returned fewer chunks than asked has ranked all it can.
            keyword_complete = depth >= in_scope or len(bm25) < depth
            exhausted = len(bm25) < depth and (vector is None or len(dense) < depth)
            complete = depth >= in_scope or exhausted
            if complete or len(docs) > needed:
                return Ranking(
                    tuple(docs), tuple(chunks), depth, rankings, complete,
                    frozenset(keyword_docs), keyword_complete,
                )  # fmt: skip
            depth *= SEARCH_DEPTH_GROWTH

    def _keyword_flags(
        self, query: str, doc_ids: Sequence[str], ranking: Ranking
    ) -> dict[str, bool]:
        """doc_id -> a passage of it matches the keyword query, at any rank: the ranking
        tells for the documents it ranked by keyword; past its top, the rest are checked."""
        flags = {d: d in ranking.keyword_docs for d in doc_ids}
        unsure = [d for d, found in flags.items() if not found]
        if unsure and not ranking.keyword_complete:
            own = {d: self._doc_chunks.get(d, []) for d in unsure}
            matched = self._chunks.keyword_matches(query, {c for ids in own.values() for c in ids})
            flags.update({d: any(c in matched for c in ids) for d, ids in own.items()})
        return flags

    def _scope_docs(self, scope: Scope) -> tuple[list[str] | None, int]:
        """The documents in scope (None: every document), and how many a created range
        could not check."""
        if scope.is_everything:
            return None, 0
        ids = list(self._summaries) if scope.within is None else list(dict.fromkeys(scope.within))
        kept: list[str] = []
        undated = 0
        dated = scope.created_from is not None or scope.created_before is not None
        for doc_id in ids:
            summary = self._summaries.get(doc_id)
            if summary is None or (scope.source is not None and summary.source != scope.source):
                continue
            if dated:
                created = summary.created
                if created is None:
                    undated += 1
                    continue
                if scope.created_from is not None and created < scope.created_from:
                    continue
                if scope.created_before is not None and created >= scope.created_before:
                    continue
            kept.append(doc_id)
        return kept, undated

    def _hits(
        self,
        doc_ids: Sequence[str],
        ranked: Sequence[int],
        query: str,
        keyword: Mapping[str, bool],
    ) -> list[Hit]:
        """Each document with the passages of its best chunks (in ranked order) and its
        keyword flag (None when `keyword` does not have it)."""
        best: dict[str, list[int]] = {d: [] for d in doc_ids}
        for chunk_id in ranked:
            chosen = best.get(self._chunk_doc[chunk_id])
            if chosen is not None and len(chosen) < SEARCH_SNIPPETS_PER_DOC:
                chosen.append(chunk_id)
        texts = {
            t.chunk_id: t.text
            for t in self._chunks.chunk_texts([c for ids in best.values() for c in ids])
        }
        words = list(dict.fromkeys(w.casefold() for w in _WORD.findall(query)))
        return [
            Hit(
                d,
                tuple(
                    snippet(texts[c], words, SEARCH_SNIPPET_CHARS) for c in best[d] if c in texts
                ),
                keyword.get(d),
            )
            for d in doc_ids
        ]

    def _traced(self, hits: list[tuple[int, float]]) -> list[dict[str, object]]:
        return [
            {"chunk_id": c, "doc_id": self._chunk_doc[c], "score": round(score, 4)}
            for c, score in hits
        ]


def snippet(text: str, words: Sequence[str], size: int) -> str:
    """About `size` characters of `text` around the most query words, on one line, with
    '...' where it was cut; the start of the text when no word is in it."""
    flat = " ".join(text.split())
    if len(flat) <= size:
        return flat
    start = 0
    if words:
        pattern = re.compile(
            r"(?<![^\W_])(?:" + "|".join(re.escape(w) for w in words) + r")(?![^\W_])",
            re.IGNORECASE,
        )
        found = [(m.start(), m.group().casefold()) for m in pattern.finditer(flat)]
        best = -1
        for anchor, _ in found:
            begin = max(0, anchor - size // 4)
            count = len({w for at, w in found if begin <= at < begin + size})
            if count > best:
                best, start = count, begin
    start = min(start, len(flat) - size)
    if start > 0:
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < start + size // 4 else start
    end = start + size
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        end = space if space > start else end
    return (
        (_ELLIPSIS if start > 0 else "")
        + flat[start:end].strip()
        + (_ELLIPSIS if end < len(flat) else "")
    )
