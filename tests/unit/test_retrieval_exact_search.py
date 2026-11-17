"""Exact search (config SEARCH_EXACT_PHRASES, run 3) on the tool corpus, through the
search and the search tool's page: a key-shaped token matches only the documents with its
parts together; every hit says whether its document matches the keywords at any rank;
the page marks the documents only the dense side found and says when no document in scope
matched, and still lists every document; the trace records the keyword query. With the
switch off, hits carry no mark and the page is as before.

Facts of the tool corpus used here: each order's header names its bench team ("bay-north"
for every fourth order); no chunk has "north" right before "bay"."""

import asyncio
import re
from collections.abc import Sequence, Set

import pytest

from corpusqa.agent.corpus_tools import CorpusToolSession, corpus_session_factory
from corpusqa.config import RETRIEVAL_TOP_CHUNKS
from corpusqa.llm.errors import EmbeddingError
from corpusqa.llm.fakes import FakeEmbedder
from corpusqa.llm.interfaces import EmbedRole, ToolCall
from corpusqa.obs.testing import RecordingTracer
from corpusqa.retrieval.search import CorpusSearch, Hit, Scope
from corpusqa.store.sqlite import SqliteChunkReader, SqliteReader
from tests.unit.tool_corpus import DIMS, N_ORDERS, SOURCE, TEAMS, Built, build_tool_corpus, doc_id

ORDERS = range(1, N_ORDERS + 1)
NORTH = frozenset(doc_id(n) for n in ORDERS if TEAMS[n % len(TEAMS)] == "bay-north")
KEY = "bay-north"  # key-shaped: its parts are in every order ("bay") and elsewhere
NOWHERE = "north-bay"  # both parts are in the corpus, never together in this order
HANDLE = re.compile(r"^(D\d+) · ", re.MULTILINE)
NO_KEYWORD_MATCH = "No document in scope contains the searched words, phrases or keys"
MARK = " · meaning only"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_tool_corpus(tmp_path_factory.mktemp("exact"))


class CheckingReader(SqliteChunkReader):
    """A chunk reader that records each check past the ranked top: (chunks, matched)."""

    def __init__(self, built: Built, *, exact_phrases: bool) -> None:
        assert built.runtime.index_path is not None
        super().__init__(built.runtime.index_path, exact_phrases=exact_phrases)
        self.checked: list[tuple[set[int], set[int]]] = []

    def keyword_matches(self, query: str, chunk_ids: Set[int]) -> set[int]:
        matched = super().keyword_matches(query, chunk_ids)
        self.checked.append((set(chunk_ids), matched))
        return matched


def corpus_search(
    built: Built, chunks: SqliteChunkReader, top: int = RETRIEVAL_TOP_CHUNKS
) -> CorpusSearch:
    assert built.runtime.index_path is not None
    summaries = SqliteReader(built.runtime.index_path).document_summaries()
    return CorpusSearch(chunks, summaries, built.runtime.vectors, top_chunks=top)


def every_hit(search: CorpusSearch, query: str, vector: Sequence[float] | None) -> list[Hit]:
    """Every hit of the search in the benchwork source, page by page from offset 0."""
    hits: list[Hit] = []
    offset: int | None = 0
    while offset is not None:
        result = search.search(query, Scope(source=SOURCE), offset, 8, vector)
        hits += result.page.items
        offset = result.page.next_offset
    return hits


# ---------------------------------------------------------------- the search
def test_a_key_shaped_token_matches_only_documents_with_its_parts_together(built: Built) -> None:
    exact = CheckingReader(built, exact_phrases=True)
    chunk_doc = exact.chunk_doc_ids()
    assert {chunk_doc[c] for c, _ in exact.bm25(KEY, 10_000)} == NORTH
    off = CheckingReader(built, exact_phrases=False)
    every_order = {doc_id(n) for n in ORDERS}
    assert every_order < {chunk_doc[c] for c, _ in off.bm25(KEY, 10_000)}  # run 1: any word


@pytest.mark.parametrize("top", [4, RETRIEVAL_TOP_CHUNKS])
def test_every_hit_says_whether_its_document_matches_the_keywords_at_any_rank(
    built: Built, top: int
) -> None:
    [vector] = built.embedder.embed([KEY], "query")
    chunks = CheckingReader(built, exact_phrases=True)
    hits = every_hit(corpus_search(built, chunks, top), KEY, vector)
    assert sorted(h.doc_id for h in hits) == sorted(doc_id(n) for n in ORDERS)  # none dropped
    assert {h.doc_id for h in hits if h.keyword} == NORTH
    assert all(h.keyword is False for h in hits if h.doc_id not in NORTH)
    if top == 4:  # past the ranked top, page documents were checked, some matched there
        assert any(matched for _, matched in chunks.checked)


def test_a_key_written_nowhere_finds_no_keyword_match_and_meaning_still_lists(
    built: Built,
) -> None:
    [vector] = built.embedder.embed([NOWHERE], "query")
    search = corpus_search(built, CheckingReader(built, exact_phrases=True))
    result = search.search(NOWHERE, Scope(source=SOURCE), 0, 8, vector)
    assert result.exact_phrases and not result.keyword_found
    assert len(result.page.items) == 8  # dense neighbours, nothing hidden
    assert all(h.keyword is False for h in result.page.items)
    keyword_only = search.search(NOWHERE, Scope(source=SOURCE), 0, 8, None)
    assert keyword_only.page.total == 0 and not keyword_only.keyword_found


def test_with_the_switch_off_hits_carry_no_mark(built: Built) -> None:
    [vector] = built.embedder.embed([NOWHERE], "query")
    chunks = CheckingReader(built, exact_phrases=False)
    result = corpus_search(built, chunks).search(NOWHERE, Scope(source=SOURCE), 0, 8, vector)
    assert not result.exact_phrases and result.keyword_found  # "north" OR "bay"
    assert all(h.keyword is None for h in result.page.items)
    assert chunks.checked == []  # no check past the top: as before


def test_the_trace_records_the_keyword_query_and_the_mode(built: Built) -> None:
    [vector] = built.embedder.embed([NOWHERE], "query")
    for exact, expression in ((True, '"north bay"'), (False, '"north" OR "bay"')):
        tracer = RecordingTracer()
        search = corpus_search(built, CheckingReader(built, exact_phrases=exact))
        with tracer.trace("t"):
            result = search.search(NOWHERE, Scope(source=SOURCE), 0, 8, vector)
        for bm25 in tracer.named("bm25"):
            assert bm25.metadata["fts_expression"] == expression
            assert bm25.metadata["exact_phrases"] is exact
        [span] = tracer.named("search")
        assert span.metadata["exact_phrases"] is exact
        assert span.metadata["keyword_found"] is not exact
        meaning_only = [h.doc_id for h in result.page.items] if exact else None
        assert span.metadata["meaning_only"] == meaning_only


# ---------------------------------------------------------------- the page
class BrokenEmbedder(FakeEmbedder):
    """Every query embedding fails: the search is keyword only."""

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        if role == "query":
            raise EmbeddingError("the embedding service is down")
        return super().embed(texts, role)


def session(built: Built, *, exact: bool, broken: bool = False) -> CorpusToolSession:
    runtime = built.runtime
    assert runtime.index_path is not None
    embedder = BrokenEmbedder(DIMS) if broken else runtime.embedder
    chunks = CheckingReader(built, exact_phrases=exact)
    factory = corpus_session_factory(
        SqliteReader(runtime.index_path), chunks, runtime.vectors, embedder
    )
    return factory()


def call(session: CorpusToolSession, name: str, **arguments: object) -> str:
    [message] = asyncio.run(session.toolbox.run([ToolCall("c1", name, arguments)]))
    return message.content


def test_the_page_says_when_no_document_matched_and_marks_every_one(built: Built) -> None:
    text = call(session(built, exact=True), "search", query=NOWHERE, source=SOURCE)
    lines = text.splitlines()
    assert lines[1] == (
        f'{NO_KEYWORD_MATCH}: the documents below are only close in meaning (marked "meaning '
        'only").'
    )
    assert lines[2] == "documents 1-8; next: offset=8"  # every document still listed
    doc_lines = [line for line in lines if HANDLE.match(line)]
    assert len(doc_lines) == 8 and all(line.endswith(MARK) for line in doc_lines)


def test_without_dense_a_page_with_no_keyword_match_has_no_documents(built: Built) -> None:
    text = call(session(built, exact=True, broken=True), "search", query=NOWHERE, source=SOURCE)
    assert text.splitlines()[1:] == [f"{NO_KEYWORD_MATCH}.", "no documents"]


def test_the_page_marks_exactly_the_documents_with_no_keyword_match(built: Built) -> None:
    tools = session(built, exact=True)
    offset: int | None = 0
    marked: dict[str, bool] = {}
    while offset is not None:
        text = call(tools, "search", query=KEY, source=SOURCE, offset=offset)
        lines = text.splitlines()
        page: dict[str, bool] = {}  # doc_id -> its line has the mark
        for line in lines:
            if handle := HANDLE.match(line):
                page[tools.handles.resolve([handle.group(1)])[0]] = line.endswith(MARK)
        explained = any(
            line.startswith('Documents marked "meaning only" contain none') for line in lines
        )
        assert explained == any(page.values())  # the mark is explained on its page
        assert NO_KEYWORD_MATCH not in text
        marked.update(page)
        match = re.search(r"next: offset=(\d+)", text)
        offset = int(match.group(1)) if match else None
    assert len(marked) == N_ORDERS
    assert {d for d, is_marked in marked.items() if not is_marked} == NORTH


def test_with_the_switch_off_the_page_is_as_before(built: Built) -> None:
    text = call(session(built, exact=False), "search", query=NOWHERE, source=SOURCE)
    lines = text.splitlines()
    assert lines[0] == f'search "{NOWHERE}" (in source "{SOURCE}"): best first.'
    assert lines[1] == "documents 1-8; next: offset=8"
    assert "meaning only" not in text and NO_KEYWORD_MATCH not in text
