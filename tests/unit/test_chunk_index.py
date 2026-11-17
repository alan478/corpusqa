"""Chunk tables (store), chunk rows from stored documents, and the hybrid retriever on the
mini corpus with FakeEmbedder (no network)."""

import sqlite3
from pathlib import Path

import pytest

from corpusqa.config import SEARCH_EXACT_PHRASES, Settings
from corpusqa.ingest import run
from corpusqa.ingest.chunking import ChunkSpec, chunk_input, chunk_rows, embedding_text
from corpusqa.llm.fakes import FakeEmbedder
from corpusqa.obs.testing import RecordingTracer
from corpusqa.retrieval.hybrid import HybridRetriever
from corpusqa.store.interfaces import ChunkRow, DocumentText, EmbeddingRow, StoreError
from corpusqa.store.sqlite import (
    SqliteChunkReader,
    SqliteChunkWriter,
    SqliteVectorCache,
    building_path,
    match_expression,
)
from corpusqa.store.vectors import VectorIndex, from_blob, to_blob

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
DIMS = 64


def text_doc(
    created_rule: str | None = "order", created: str | None = "2024-03-04T10:00:00Z"
) -> DocumentText:
    body = "## alpha\nFirst part about boilers.\n\n## beta\nSecond part about ladders."
    return DocumentText(
        doc_id="d1",
        source="src",
        display_title="Plant notes",
        created=created,
        created_rule=created_rule,
        body=body,
        body_fields=("alpha", "beta"),
    )


def row(chunk_id: int, doc_id: str, text: str, header: str = "") -> ChunkRow:
    return ChunkRow(chunk_id, doc_id, "src", 0, "body", "whole", header, text, 1, 0, len(text))


# ---------------------------------------------------------------- chunk rows
def test_chunk_rows_are_numbered_slices_with_the_header() -> None:
    doc = text_doc()
    rows = chunk_rows(doc, ChunkSpec("recursive", max_tokens=8), first_id=7)
    assert [r.chunk_id for r in rows] == list(range(7, 7 + len(rows)))
    assert [r.ord for r in rows] == list(range(len(rows)))
    assert len(rows) >= 2
    for r in rows:
        assert r.char_start is not None and r.char_end is not None
        assert doc.body[r.char_start : r.char_end] == r.text
        assert r.header == "src · Plant notes · 2024-03-04"
        assert r.doc_id == "d1" and r.kind == "body"


def test_the_header_shows_any_created_date_and_none_without_one() -> None:
    # Every stored created date is inside the window (one outside counts as no date), so
    # the header shows it whatever rule picked it.
    for rule in ("order", "name_hint", "file_stem"):
        [first, *_] = chunk_rows(text_doc(created_rule=rule), ChunkSpec("whole"), first_id=1)
        assert first.header == "src · Plant notes · 2024-03-04", rule
    undated = text_doc(created_rule=None, created=None)
    assert chunk_input(undated).created is None
    [first, *_] = chunk_rows(undated, ChunkSpec("whole"), first_id=1)
    assert first.header == "src · Plant notes"


def test_headers_off_store_an_empty_header_and_embed_the_text_alone() -> None:
    [only] = chunk_rows(text_doc(), ChunkSpec("whole", header=False), first_id=1)
    assert only.header == ""
    assert embedding_text(only.header, only.text) == only.text
    assert embedding_text("h", "t") == "h\nt"


# ---------------------------------------------------------------- store
def test_chunk_index_round_trip_and_publish(tmp_path: Path) -> None:
    path = tmp_path / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks([row(1, "a", "the boiler hums", "src · Boiler log"),
                       row(2, "b", "ladders are stored in the shed")])  # fmt: skip
    writer.rebuild_fts()
    vector = to_blob([0.6, 0.8])
    writer.add_embeddings([EmbeddingRow(1, "m", 2, vector)])
    assert not path.exists() and building_path(path).exists()
    with pytest.raises(StoreError):
        SqliteChunkReader(path)
    writer.finish()
    assert path.exists() and not building_path(path).exists()

    reader = SqliteChunkReader(path)
    assert reader.chunk_doc_ids() == {1: "a", 2: "b"}
    assert [c for c, _ in reader.bm25("Where are the ladders?", 5)] == [2, 1]  # any word
    assert [c for c, _ in reader.bm25("ladders", 5)] == [2]
    assert [c for c, _ in reader.bm25("boiler log", 5)] == [1]  # header words count too
    assert reader.bm25("?!", 5) == []
    assert reader.embeddings() == [(1, vector)]
    assert from_blob(vector) == pytest.approx([0.6, 0.8])


def test_bm25_scores_are_higher_is_better(tmp_path: Path) -> None:
    path = tmp_path / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks([row(1, "a", "kettle kettle kettle descaling"),
                       row(2, "b", "kettle on the shelf next to many other things")])  # fmt: skip
    writer.rebuild_fts()
    writer.finish()
    hits = SqliteChunkReader(path).bm25("kettle descaling", 5)
    assert [c for c, _ in hits] == [1, 2]
    assert hits[0][1] > hits[1][1]


def test_chunk_abort_leaves_nothing(tmp_path: Path) -> None:
    path = tmp_path / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks([row(1, "a", "text")])
    writer.abort()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Who owns ZX-12?", '"who" OR "owns" OR "zx" OR "12"'),
        ('say "NEAR" OR AND NOT', '"say" OR "near" OR "or" OR "and" OR "not"'),
        ("snake_case word word", '"snake" OR "case" OR "word"'),
        ("Café déjà", '"café" OR "déjà"'),
        ("*^?", None),
    ],
)
def test_match_expression_quotes_every_word(query: str, expected: str | None) -> None:
    assert match_expression(query) == expected


def test_user_text_cannot_break_the_fts_query(tmp_path: Path) -> None:
    path = tmp_path / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks([row(1, "a", "near the end of the column")])
    writer.rebuild_fts()
    writer.finish()
    for query in ('"unbalanced', "NEAR(a b)", "column:* OR", "a AND (", "^start -x +y"):
        SqliteChunkReader(path).bm25(query, 5)  # must not raise


# ---------------------------------------------------------------- exact mode (run 3)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ('"ZX-12" OR "QV-7"', '"zx 12" OR "qv 7"'),
        ("Who owns ZX-12?", '"who" OR "owns" OR "zx 12"'),
        ('"night shift" lamp', '"night shift" OR "lamp"'),
        ("QV-7 OR qv-7", '"qv 7"'),  # one part once
        ('say "NEAR" OR AND NOT', '"say" OR "near" OR "not"'),  # capital OR / AND: not words
        ("salt and pepper or", '"salt" OR "and" OR "pepper" OR "or"'),  # lower case: words
        ('"unbalanced quote', '"unbalanced" OR "quote"'),
        ("snake_case a/b.txt v1.2 PR#7", '"snake case" OR "a b txt" OR "v1 2" OR "pr 7"'),
        ("“night shift”", '"night shift"'),  # curly quotes too
        ('"" *^? OR', None),
    ],
)
def test_exact_match_expression_keeps_phrases_and_keys_whole(query: str, expected: str) -> None:
    assert match_expression(query, exact=True) == expected


def exact_index(tmp_path: Path) -> Path:
    path = tmp_path / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks([
        row(1, "a", "the night shift log is on the wall"),
        row(2, "b", "shift the night lamp to the left"),
        row(3, "c", "night watch, then a shift change later"),
        row(4, "d", "ticket ZX-12 was closed on time"),
        row(5, "e", "zx parts and 12 bolts arrived"),
        row(6, "f", "see ZX-120 for the next step"),
        row(7, "g", "the boiler hums", "src · Boiler log"),
    ])  # fmt: skip
    writer.rebuild_fts()
    writer.finish()
    return path


def found(reader: SqliteChunkReader, query: str, chunk_ids: set[int] | None = None) -> set[int]:
    return {c for c, _ in reader.bm25(query, 100, chunk_ids)}


def test_exact_mode_matches_a_quoted_phrase_only_with_its_words_adjacent_and_in_order(
    tmp_path: Path,
) -> None:
    path = exact_index(tmp_path)
    exact = SqliteChunkReader(path, exact_phrases=True)
    assert found(exact, '"night shift"') == {1}  # 2: other order, 3: apart
    assert found(exact, '"Night  SHIFT"') == {1}  # case and spacing do not matter
    assert found(exact, '"shift night"') == set()
    assert found(exact, '"night shift"', {2, 3}) == set()  # a scope too
    assert found(exact, '"boiler log"') == {7}  # the header counts, as for words
    assert found(exact, '"night shift" lamp') == {1, 2}  # other words: any of them
    assert found(SqliteChunkReader(path, exact_phrases=False), '"night shift"') == {1, 2, 3}


def test_exact_mode_matches_a_key_shaped_token_as_its_parts_together(tmp_path: Path) -> None:
    path = exact_index(tmp_path)
    exact = SqliteChunkReader(path, exact_phrases=True)
    for written in ("ZX-12", "zx_12", "ZX/12", '"ZX 12"', "Who closed ZX-12?"):
        assert found(exact, written) == {4}, written  # 5: both parts apart, 6: ZX-120
    assert exact.keyword_expression("ZX-12") == '"zx 12"'
    # Off (run 1): the key falls apart into words that match the near misses too.
    off = SqliteChunkReader(path, exact_phrases=False)
    assert found(off, "ZX-12") == {4, 5, 6}
    assert off.keyword_expression("ZX-12") == '"zx" OR "12"'


def test_exact_mode_finds_nothing_for_a_key_that_is_written_nowhere(tmp_path: Path) -> None:
    path = exact_index(tmp_path)
    exact = SqliteChunkReader(path, exact_phrases=True)
    assert exact.bm25("ZX-99", 5) == []
    assert exact.bm25('"ZX-99" OR "QV-7"', 5) == []
    assert exact.bm25("ZX-99", 5, {4, 5, 6}) == []
    assert exact.keyword_matches("ZX-99", {4, 5, 6}) == set()
    assert found(SqliteChunkReader(path, exact_phrases=False), "ZX-99") == {4, 5, 6}


def test_keyword_matches_checks_given_chunks_at_any_rank(tmp_path: Path) -> None:
    path = exact_index(tmp_path)
    for exact, matched in ((True, {1}), (False, {1, 2, 3})):
        reader = SqliteChunkReader(path, exact_phrases=exact)
        assert reader.keyword_matches('"night shift"', {1, 2, 3, 4}) == matched
        assert reader.keyword_matches('"night shift"', {1}) == {1}
        assert reader.keyword_matches("night", set()) == set()
        assert reader.keyword_matches("?!", {1, 2}) == set()


@pytest.mark.parametrize(
    "query",
    [
        '"unbalanced', '""', '"a""b"', '"""', "NEAR(night shift)", 'text:"night"', "night*",
        '"night" NOT "shift"', "a AND (", "^start -x +y", "{header text}: night",
        "'); DROP TABLE chunks; --", "“night shift", "OR", "AND OR",
    ],
)  # fmt: skip
def test_exact_mode_never_reads_user_text_as_fts_syntax(tmp_path: Path, query: str) -> None:
    exact = SqliteChunkReader(exact_index(tmp_path), exact_phrases=True)
    exact.bm25(query, 5)  # must not raise
    exact.bm25(query, 5, {1, 2})
    exact.keyword_matches(query, {1, 2})


def test_exact_mode_operators_and_marks_are_only_words(tmp_path: Path) -> None:
    exact = SqliteChunkReader(exact_index(tmp_path), exact_phrases=True)
    assert found(exact, "boil*") == set()  # no prefix search: "boil" is no word here
    assert found(exact, '"boiler" NOT hums') == {7}  # NOT excludes nothing
    assert found(exact, "header:night") == {1, 2, 3}  # no column filter
    assert found(exact, "NEAR(night lamp)") == {1, 2, 3}  # no NEAR group: words
    assert exact.bm25("OR AND", 5) == []


def test_a_chunk_reader_takes_its_mode_from_the_config(tmp_path: Path) -> None:
    assert SqliteChunkReader(exact_index(tmp_path)).exact_phrases is SEARCH_EXACT_PHRASES


def test_vector_cache_keeps_vectors_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    cache = SqliteVectorCache(path)
    cache.put_many("m", 2, [("k1", b"\x01" * 8), ("k2", b"\x02" * 8)])
    cache.close()
    again = SqliteVectorCache(path)
    assert again.get_many(["k2", "k3", "k1"]) == {"k1": b"\x01" * 8, "k2": b"\x02" * 8}
    assert again.get_many([f"x{i}" for i in range(1200)]) == {}  # more keys than one query


def test_document_texts_reads_what_chunking_needs(tmp_path: Path) -> None:
    reader = run(Settings(corpus_dir=FIXTURE, port=8080, index_dir=tmp_path)).store
    assert reader is not None
    texts = list(reader.document_texts())
    assert {t.doc_id for t in texts} == reader.doc_ids()
    printer = next(t for t in texts if t.doc_id == "fx-tkt-101")
    assert "Monday" in printer.body and printer.body_fields
    assert printer.created is not None and printer.created_rule


# ---------------------------------------------------------------- retriever on the fixture
@pytest.fixture(scope="module")
def chunk_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, FakeEmbedder]:
    folder = tmp_path_factory.mktemp("index")
    reader = run(Settings(corpus_dir=FIXTURE, port=8080, index_dir=folder)).store
    assert reader is not None
    chunks: list[ChunkRow] = []
    for doc in reader.document_texts():
        chunks.extend(chunk_rows(doc, ChunkSpec("recursive", max_tokens=40), len(chunks) + 1))
    embedder = FakeEmbedder(dims=DIMS)
    vectors = embedder.embed([embedding_text(c.header, c.text) for c in chunks], "document")
    path = folder / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks(chunks)
    writer.rebuild_fts()
    writer.add_embeddings(
        [EmbeddingRow(c.chunk_id, embedder.model, DIMS, to_blob(v))
         for c, v in zip(chunks, vectors, strict=True)]
    )  # fmt: skip
    writer.finish()
    return path, embedder


def retriever(chunk_index: tuple[Path, FakeEmbedder]) -> HybridRetriever:
    path, embedder = chunk_index
    reader = SqliteChunkReader(path)
    return HybridRetriever(reader, VectorIndex.from_rows(reader.embeddings(), DIMS), embedder)


def test_every_mode_finds_the_document_that_matches(
    chunk_index: tuple[Path, FakeEmbedder],
) -> None:
    found = retriever(chunk_index).search("Why does the printer jam on Monday mornings?")
    for docs in (found.bm25, found.dense, found.hybrid):
        assert docs[0] == "fx-tkt-101"
        assert len(docs) == len(set(docs))  # one entry per document
    assert set(found.hybrid) == set(found.bm25) | set(found.dense)


def test_the_question_is_embedded_with_the_query_role(
    chunk_index: tuple[Path, FakeEmbedder],
) -> None:
    _, embedder = chunk_index
    retriever(chunk_index).search("printer roller")
    assert embedder.calls[-1] == ("query", ("printer roller",))


def test_without_vectors_dense_is_empty_and_hybrid_is_bm25(
    chunk_index: tuple[Path, FakeEmbedder],
) -> None:
    path, _ = chunk_index
    found = HybridRetriever(SqliteChunkReader(path)).search("printer Monday")
    assert found.dense == () and found.hybrid == found.bm25 and found.bm25


def test_retrieval_is_traced_with_a_span_per_ranker(
    chunk_index: tuple[Path, FakeEmbedder],
) -> None:
    tracer = RecordingTracer()
    with tracer.trace("eval_retrieval"):
        retriever(chunk_index).search("printer Monday")
    [retrieve] = tracer.named("retrieve")
    assert retrieve.input == {"question": "printer Monday"}
    assert set(retrieve.output) == {"bm25", "dense", "hybrid"}
    assert retrieve.metadata["bm25_weight"] == 1.0 and retrieve.metadata["dense_weight"] == 1.0
    children = {c.name: c for c in tracer.children(retrieve)}
    assert set(children) == {"bm25", "dense"}
    first = children["bm25"].output[0]
    assert set(first) == {"chunk_id", "doc_id", "score"}


def test_fusion_weights_reach_the_span_and_a_zero_weight_turns_a_ranker_off(
    chunk_index: tuple[Path, FakeEmbedder],
) -> None:
    path, embedder = chunk_index
    reader = SqliteChunkReader(path)
    vectors = VectorIndex.from_rows(reader.embeddings(), DIMS)
    tracer = RecordingTracer()
    with tracer.trace("eval_retrieval"):
        found = HybridRetriever(
            reader, vectors, embedder, bm25_weight=0.0, dense_weight=2.0
        ).search("printer Monday")
    [retrieve] = tracer.named("retrieve")
    assert retrieve.metadata["bm25_weight"] == 0.0 and retrieve.metadata["dense_weight"] == 2.0
    # bm25 has no vote: dense's documents lead in dense's order, bm25-only ones follow
    assert found.hybrid[: len(found.dense)] == found.dense
    assert set(found.hybrid) == set(found.bm25) | set(found.dense)
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(reader, bm25_weight=-1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_fusion_weights_must_be_finite(chunk_index: tuple[Path, FakeEmbedder], bad: float) -> None:
    # Codex review 2026-10-16 finding 1: NaN and infinity passed the `< 0` check; an
    # infinite weight ranks every hit equal (id order), NaN leaves the order undefined.
    reader = SqliteChunkReader(chunk_index[0])
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(reader, bm25_weight=bad)
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(reader, dense_weight=bad)


def test_chunk_index_has_no_foreign_key_to_documents_enforced(tmp_path: Path) -> None:
    """A standalone chunk file has no documents table; inserts must still work."""
    path = tmp_path / "chunks.sqlite"
    writer = SqliteChunkWriter(path)
    writer.add_chunks([row(1, "not-a-stored-doc", "text")])
    writer.finish()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
