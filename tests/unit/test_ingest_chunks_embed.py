"""S6 chunks, S9 embeddings and the vectors in the Runtime (design.md S6, S9, §6, §7).

Offline: FakeEmbedder (and fakes built on it) stand in for the model; a fake clock drives
the time box. Every test points the embedding cache at a temporary file.
"""

import json
import math
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from corpusqa import obs
from corpusqa.config import (
    CHUNK_HEADER_FIELDS,
    CHUNK_HEADER_SEPARATOR,
    CHUNK_MAX_TOKENS,
    EMBED_PROMPT_NAME,
    FINGERPRINT_CODE_PATHS,
    TOKEN_CHARS,
    Settings,
)
from corpusqa.ingest import Runtime, run
from corpusqa.ingest import chunking as chunking_module
from corpusqa.ingest import embed as embed_module
from corpusqa.ingest.embed import EmbedOptions, _Text, cache_key, round_robin
from corpusqa.ingest.fingerprint import code_hash
from corpusqa.llm.errors import EmbeddingError, SpendCapReached
from corpusqa.llm.fakes import FakeEmbedder
from corpusqa.llm.interfaces import EmbedRole
from corpusqa.obs.testing import RecordingTracer
from corpusqa.store.sqlite import SqliteEmbeddingWriter, building_path
from corpusqa.store.vectors import BadVector, blob_fits, to_blob, unit_blob
from tests.unit.test_ingest_pipeline import FIXTURE

DIMS = 16


# ---------------------------------------------------------------- helpers
def make_corpus(tmp_path: Path) -> Path:
    """The fixture plus a long page (several chunks) and an invented source whose one date
    field has a date inside the window (2024) and one far outside it (2099)."""
    target = tmp_path / "corpus"
    shutil.copytree(FIXTURE, target)
    paragraphs = [
        f"Section {i} of the lantern rota. " + "The keeper trims every wick and logs the oil. " * 9
        for i in range(12)
    ]
    page = {
        "pg_name": "Lantern rota handbook",
        "pg_text": "\n\n".join(paragraphs),
        "title_field_name": "pg_name",
        "content_field_names": ["pg_text"],
        "dataset_doc_uuid": "fx-wiki-long",
    }
    (target / "wiki" / "lantern-rota.json").write_text(json.dumps(page), encoding="utf-8")
    log = target / "logbook"
    log.mkdir()
    for n, stamp in enumerate(("2024-06-01", "2099-01-01")):
        entry = {
            "entry_name": f"Tide log {n}",
            "entry_text": f"High water at the north quay, reading {n}.",
            "stamp_at": stamp,
            "title_field_name": "entry_name",
            "content_field_names": ["entry_text"],
            "dataset_doc_uuid": f"fx-log-{n}",
        }
        (log / f"tide-{n}.json").write_text(json.dumps(entry), encoding="utf-8")
    return target


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return make_corpus(tmp_path)


def settings_for(corpus: Path, tmp_path: Path, index: str = "index") -> Settings:
    return Settings(
        corpus_dir=corpus.resolve(),
        port=8080,
        index_dir=tmp_path / index,
        embed_cache_db=tmp_path / "embed-cache.sqlite",
    )


def connect(runtime: Runtime) -> sqlite3.Connection:
    assert runtime.index_path is not None
    return sqlite3.connect(runtime.index_path)


def meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM meta").fetchall())


def n_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def sent_texts(embedder: FakeEmbedder) -> int:
    return sum(len(texts) for _, texts in embedder.calls)


def s9_output(tracer: RecordingTracer) -> dict[str, object]:
    [span] = tracer.named("S9_embed")
    assert isinstance(span.output, dict)
    return span.output


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SlowEmbedder(FakeEmbedder):
    """Each request takes `seconds` on the fake clock."""

    def __init__(self, clock: FakeClock, seconds: float) -> None:
        super().__init__(dims=DIMS)
        self.clock, self.seconds = clock, seconds

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        self.clock.now += self.seconds
        return super().embed(texts, role)


class FailsAfter(FakeEmbedder):
    """Answers `ok` requests, then raises `error` for every later one (or only the
    `only`-th request when `only` is set)."""

    def __init__(self, ok: int, error: BaseException, only: int | None = None) -> None:
        super().__init__(dims=DIMS)
        self.ok, self.error, self.only, self.requests = ok, error, only, 0

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        self.requests += 1
        failing = self.requests == self.only if self.only else self.requests > self.ok
        if failing:
            raise self.error
        return super().embed(texts, role)


class BadAnswers(FakeEmbedder):
    """Request 1: its second vector holds a NaN. Request 2: one vector too few."""

    def __init__(self) -> None:
        super().__init__(dims=DIMS)
        self.requests = 0

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        self.requests += 1
        vectors = super().embed(texts, role)
        if self.requests == 1:
            vectors[1] = [math.nan] * DIMS
        if self.requests == 2:
            vectors = vectors[:-1]
        return vectors


class SpendCap(SpendCapReached):
    def __init__(self) -> None:
        super().__init__("spend cap reached (test)")


class Crash(BaseException):
    """Stands for the process dying mid-request (not an Exception: nothing catches it)."""


def small(clock: FakeClock | None = None, **kwargs: object) -> EmbedOptions:
    """Two texts per request, one in flight, so the order of requests is fixed; cool-downs
    take no real time."""
    options: dict[str, object] = {
        "batch_size": 2, "concurrency": 1, "sleep": lambda seconds: None, **kwargs
    }  # fmt: skip
    if clock is not None:
        options["clock"] = clock
    return EmbedOptions(**options)  # type: ignore[arg-type]


# ---------------------------------------------------------------- S6
def test_chunks_are_slices_of_the_final_body_within_the_size_limit(
    corpus: Path, tmp_path: Path
) -> None:
    runtime = run(settings_for(corpus, tmp_path))
    conn = connect(runtime)
    bodies = dict(conn.execute("SELECT doc_id, body FROM documents"))
    rows = conn.execute(
        "SELECT chunk_id, doc_id, ord, text, n_tokens, char_start, char_end FROM chunks "
        "ORDER BY chunk_id"
    ).fetchall()
    assert [r[0] for r in rows] == list(range(1, len(rows) + 1))  # ids 1..N, no gaps
    by_doc: dict[str, list[int]] = {}
    for _, doc_id, ord_, text, tokens, start, end in rows:
        assert bodies[doc_id][start:end] == text and text == text.strip() and text
        assert tokens == math.ceil(len(text) / TOKEN_CHARS) <= CHUNK_MAX_TOKENS
        by_doc.setdefault(doc_id, []).append(ord_)
    assert all(ords == list(range(len(ords))) for ords in by_doc.values())
    assert set(by_doc) == set(bodies)  # every document has at least one chunk
    assert len(by_doc["fx-wiki-long"]) >= 2
    counted = dict(conn.execute("SELECT doc_id, n_chunks FROM documents"))
    assert counted == {doc_id: len(ords) for doc_id, ords in by_doc.items()}
    assert meta(conn)["n_chunks"] == str(len(rows))
    assert runtime.chunks is not None and runtime.chunks.count_chunks() == len(rows)


@pytest.mark.parametrize("fields_on", [False, True])
def test_headers_show_the_final_display_title_and_only_a_usable_date(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields_on: bool
) -> None:
    # Off, the header is exactly these parts; on, field values follow them (their content is
    # checked in test_header_fields.py).
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", fields_on)
    conn = connect(run(settings_for(corpus, tmp_path)))
    rows = conn.execute(
        "SELECT c.header, d.source, d.display_title, d.created, d.created_rule, d.title_is_weak "
        "FROM chunks c JOIN documents d USING (doc_id)"
    ).fetchall()
    weak = with_fields = 0
    for header, source, display_title, created, _rule, is_weak in rows:
        parts = (source, display_title, created[:10] if created else None)
        base = CHUNK_HEADER_SEPARATOR.join(p for p in parts if p)
        if fields_on:
            assert header == base or header.startswith(base + CHUNK_HEADER_SEPARATOR)
        else:
            assert header == base
        weak += is_weak
        with_fields += header != base
    assert weak >= 1  # titles S8 found weak carry the humanised file stem
    assert (with_fields > 0) is fields_on
    sql = "SELECT d.doc_id, c.header FROM chunks c JOIN documents d USING (doc_id) "
    logs = dict(conn.execute(sql + "WHERE d.source = 'logbook'").fetchall())
    # Each date on its own: the 2024 date is in the header although the other value of its
    # field is outside the window; the 2099 one counts as no date and is not shown.
    assert "2024-06-01" in logs["fx-log-0"]
    assert logs["fx-log-1"].startswith("logbook · Tide log 1") and "2099" not in logs["fx-log-1"]


def test_keyword_search_covers_every_chunk_without_vectors(corpus: Path, tmp_path: Path) -> None:
    runtime = run(settings_for(corpus, tmp_path))
    assert runtime.chunks is not None and runtime.vectors is None
    assert runtime.embed_coverage == 0.0 and runtime.embedder is None
    hits = runtime.chunks.bm25("lantern keeper wick", 5)
    doc_of = runtime.chunks.chunk_doc_ids()
    assert hits and doc_of[hits[0][0]] == "fx-wiki-long"
    conn = connect(runtime)
    assert (meta(conn)["embed_status"], meta(conn)["embed_stop"]) == ("skipped", "no_embedder")
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


# ---------------------------------------------------------------- S9 and the Runtime
def test_every_chunk_is_embedded_and_the_runtime_carries_the_vectors(
    corpus: Path, tmp_path: Path
) -> None:
    embedder = FakeEmbedder(dims=DIMS)
    runtime = run(settings_for(corpus, tmp_path), embedder)
    conn = connect(runtime)
    total = n_chunks(conn)
    assert runtime.vectors is not None and len(runtime.vectors) == total
    assert runtime.embed_coverage == 1.0 and runtime.embedder is embedder
    assert runtime.vectors.dims == DIMS
    m = meta(conn)
    assert (m["embed_status"], m["embed_stop"], m["n_embeddings"]) == ("complete", "", str(total))
    assert (m["embed_model"], m["embed_dims"], m["embed_coverage"]) == (
        embedder.model, str(DIMS), "1.0000",
    )  # fmt: skip
    for blob, dims in conn.execute("SELECT vector, dims FROM embeddings"):
        vector = np.frombuffer(blob, dtype="<f4")
        assert dims == DIMS and abs(float(np.linalg.norm(vector)) - 1) < 1e-3
    # The vector of a chunk is the embedding of its exact embedded text (header + text).
    header, text, chunk_id = conn.execute(
        "SELECT header, text, chunk_id FROM chunks WHERE doc_id = 'fx-wiki-long' AND ord = 1"
    ).fetchone()
    [query] = FakeEmbedder(dims=DIMS).embed([f"{header}\n{text}"], "query")
    assert runtime.vectors.search(query, 1)[0][0] == chunk_id
    assert {role for role, _ in embedder.calls} == {"document"}


def test_a_cached_vector_means_no_call(corpus: Path, tmp_path: Path) -> None:
    first = FakeEmbedder(dims=DIMS)
    run(settings_for(corpus, tmp_path, "index-a"), first)
    second = FakeEmbedder(dims=DIMS)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path, "index-b"), second)  # a new index
    assert not runtime.cache_hit and second.calls == []
    assert runtime.embed_coverage == 1.0
    out = s9_output(tracer)
    assert out["cache_hits"] == out["chunks"] == sent_texts(first)
    assert out.get("requests_sent", 0) == 0


def test_the_time_box_leaves_partial_coverage_and_startup_succeeds(
    corpus: Path, tmp_path: Path
) -> None:
    clock = FakeClock()
    embedder = SlowEmbedder(clock, seconds=10)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(
            settings_for(corpus, tmp_path), embedder, embed_options=small(clock, budget_s=25)
        )
    # Requests start at t=0, 10 and 20; at t=30 the 25 s budget is spent.
    assert len(embedder.calls) == 3 and sent_texts(embedder) == 6
    conn = connect(runtime)
    total = n_chunks(conn)
    assert runtime.vectors is not None and len(runtime.vectors) == 6
    assert runtime.embed_coverage == pytest.approx(6 / total)
    m = meta(conn)
    assert (m["embed_status"], m["embed_stop"], m["n_embeddings"]) == ("partial", "deadline", "6")
    out = s9_output(tracer)
    assert out["stop"] == "deadline" and out["not_attempted_chunks"] == total - 6
    # Round robin across sources by document: six sources got one document each.
    by_source = out["coverage_by_source"]
    assert isinstance(by_source, dict) and sum(v > 0 for v in by_source.values()) == 6


def test_the_spend_cap_leaves_partial_coverage_and_startup_succeeds(
    corpus: Path, tmp_path: Path
) -> None:
    embedder = FailsAfter(ok=2, error=SpendCap())
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), embedder, embed_options=small())
    assert embedder.requests == 3  # two answered, the third refused, then no more
    assert runtime.vectors is not None and len(runtime.vectors) == 4
    m = meta(connect(runtime))
    assert (m["embed_status"], m["embed_stop"]) == ("partial", "spend_cap")
    assert s9_output(tracer)["requests_refused:spend_cap"] == 1


def test_a_later_start_resumes_only_what_is_missing(corpus: Path, tmp_path: Path) -> None:
    settings = settings_for(corpus, tmp_path)
    first = run(settings, FailsAfter(ok=2, error=SpendCap()), embed_options=small())
    assert first.vectors is not None and len(first.vectors) == 4
    total = n_chunks(connect(first))
    resumer = FakeEmbedder(dims=DIMS)
    second = run(settings, resumer, embed_options=small())
    assert second.cache_hit and second.index_path == first.index_path
    assert sent_texts(resumer) == total - 4  # only the chunks without a vector
    assert second.embed_coverage == 1.0
    assert meta(connect(second))["embed_status"] == "complete"


def test_a_crash_mid_embedding_keeps_every_finished_request(corpus: Path, tmp_path: Path) -> None:
    settings = settings_for(corpus, tmp_path)
    with pytest.raises(Crash):
        run(settings, FailsAfter(ok=2, error=Crash()), embed_options=small())
    # The index was published before S9; the two finished requests are in it.
    [folder] = (tmp_path / "index").iterdir()
    db = folder / "corpus.sqlite"
    assert db.is_file() and not building_path(db).exists()
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 4
    total = n_chunks(conn)
    resumer = FakeEmbedder(dims=DIMS)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings, resumer, embed_options=small())
    assert runtime.cache_hit and runtime.embed_coverage == 1.0
    assert s9_output(tracer)["already_embedded"] == 4
    assert sent_texts(resumer) == total - 4


def test_bad_vectors_are_rejected_and_the_rest_is_kept(corpus: Path, tmp_path: Path) -> None:
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), BadAnswers(), embed_options=small())
    conn = connect(runtime)
    total = n_chunks(conn)
    # One vector with a NaN (its chunk only) and one short answer (both its chunks) lost.
    assert runtime.vectors is not None and len(runtime.vectors) == total - 3
    out = s9_output(tracer)
    assert out["vectors_rejected:not_finite"] == 1
    assert out["requests_failed:count_mismatch"] == 1
    assert (out["status"], out["stop"]) == ("partial", None)
    for (blob,) in conn.execute("SELECT vector FROM embeddings"):
        assert blob_fits(blob, DIMS)


def test_failing_requests_stop_after_too_many_in_a_row(corpus: Path, tmp_path: Path) -> None:
    embedder = FailsAfter(ok=0, error=EmbeddingError("quota exhausted (test)"))
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(
            settings_for(corpus, tmp_path),
            embedder,
            embed_options=small(max_failed_in_a_row=3, max_cooldowns=0),
        )
    assert embedder.requests == 3
    assert runtime.vectors is None and runtime.embed_coverage == 0.0
    assert len(runtime.known_doc_ids) > 0  # start-up still succeeded
    out = s9_output(tracer)
    assert (out["stop"], out["status"]) == ("failures", "partial")
    assert out["requests_failed:EmbeddingError"] == 3


def test_after_a_pause_a_throttled_api_recovers(corpus: Path, tmp_path: Path) -> None:
    # Requests 1-3 fail (a spent per-minute quota), then the API answers again.
    class Throttled(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__(dims=DIMS)
            self.requests = 0

        def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
            self.requests += 1
            if self.requests <= 3:
                raise EmbeddingError("429 quota (test)")
            return super().embed(texts, role)

    clock, pauses = FakeClock(), []

    def sleep(seconds: float) -> None:
        pauses.append(seconds)
        clock.now += seconds

    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(
            settings_for(corpus, tmp_path),
            Throttled(),
            embed_options=small(clock, max_failed_in_a_row=3, cooldown_s=60, sleep=sleep),
        )
    assert pauses == [60] and runtime.embed_coverage == 1.0
    out = s9_output(tracer)
    assert out["cooldowns"] == 1 and out["status"] == "complete"


def test_a_dead_api_gives_up_after_the_last_pause(corpus: Path, tmp_path: Path) -> None:
    embedder = FailsAfter(ok=0, error=EmbeddingError("503 (test)"))
    clock, pauses = FakeClock(), []

    def sleep(seconds: float) -> None:
        pauses.append(seconds)
        clock.now += seconds

    options = small(clock, max_failed_in_a_row=3, max_cooldowns=2, cooldown_s=60, sleep=sleep)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), embedder, embed_options=options)
    assert embedder.requests == 9 and pauses == [60, 60]  # three rounds of three failures
    assert runtime.embed_coverage == 0.0 and len(runtime.known_doc_ids) > 0
    assert s9_output(tracer)["stop"] == "failures"


def test_no_pause_runs_past_the_deadline(corpus: Path, tmp_path: Path) -> None:
    embedder = FailsAfter(ok=0, error=EmbeddingError("503 (test)"))
    clock, pauses = FakeClock(), []
    options = small(clock, budget_s=30, max_failed_in_a_row=3, cooldown_s=60, sleep=pauses.append)
    run(settings_for(corpus, tmp_path), embedder, embed_options=options)
    assert embedder.requests == 3 and pauses == []  # a 60 s pause would pass the 30 s budget


def test_a_request_that_fails_once_is_tried_again_at_the_end(corpus: Path, tmp_path: Path) -> None:
    embedder = FailsAfter(ok=0, error=EmbeddingError("timeout (test)"), only=1)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), embedder, embed_options=small())
    assert runtime.embed_coverage == 1.0
    out = s9_output(tracer)
    assert out["requests_requeued"] == 1 and out["status"] == "complete"


def test_vectors_of_another_model_are_never_mixed(corpus: Path, tmp_path: Path) -> None:
    settings = settings_for(corpus, tmp_path)
    run(settings, FakeEmbedder(dims=DIMS))

    class OtherModel(FakeEmbedder):
        model = "another-embedding"

    other = OtherModel(dims=DIMS)
    runtime = run(settings, other)
    assert other.calls == [] and runtime.vectors is None and runtime.embed_coverage == 0.0
    assert meta(connect(runtime))["embed_stop"] == "model_mismatch"
    again = run(settings, FakeEmbedder(dims=DIMS))
    assert again.vectors is not None and again.embed_coverage == 1.0


def test_an_unusable_cache_does_not_stop_embedding(corpus: Path, tmp_path: Path) -> None:
    settings = settings_for(corpus, tmp_path)
    settings.embed_cache_db.write_text("not a database", encoding="utf-8")
    runtime = run(settings, FakeEmbedder(dims=DIMS))
    assert runtime.embed_coverage == 1.0


# ---------------------------------------------------------------- tracing
def test_s6_s9_and_vector_loading_are_spans_with_their_counts(corpus: Path, tmp_path: Path) -> None:
    class TracedFake(FakeEmbedder):
        def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
            with obs.generation("embed", model=self.model, input={"texts": list(texts)}):
                return super().embed(texts, role)

    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), TracedFake(dims=DIMS), embed_options=small())
    total = n_chunks(connect(runtime))
    [s6] = tracer.named("S6_chunk")
    assert isinstance(s6.output, dict)
    assert s6.output["chunks"] == total and s6.output["documents_without_chunks"] == 0
    [dates] = tracer.named("S4_date_window")
    assert isinstance(dates.output, dict) and dates.output["dates_outside"] >= 1
    outside = dates.output["outside"]
    assert isinstance(outside, list)
    [logbook] = [s for s in outside if s["source"] == "logbook"]
    assert logbook["fields"] == [{"field": "stamp_at", "values": 2, "outside": 1}]
    assert {"documents", "text_tokens", "embed_tokens", "max_chunk_tokens", "levels"} <= set(
        s6.output
    )
    assert s6.metadata["strategy"] == "recursive" and s6.metadata["max_tokens"] == 512
    assert s6.metadata["header"] is True and "seconds" in s6.metadata
    # Header fields follow config.CHUNK_HEADER_FIELDS: the profile's picks and the count of
    # values shown are traced when on, absent / 0 when off (test_header_fields.py: both).
    assert s6.metadata["header_fields"] is CHUNK_HEADER_FIELDS
    assert ("header_field_sources" in s6.metadata) is CHUNK_HEADER_FIELDS
    assert (s6.output["header_field_values"] > 0) is CHUNK_HEADER_FIELDS
    [s9] = tracer.named("S9_embed")
    out = s9_output(tracer)
    for key in (
        "chunks", "already_embedded", "pending", "cache_hits", "to_embed_texts", "requests_sent",
        "requests_answered", "embedded_chunks", "tokens_est", "cost_est_usd", "status", "stop",
        "coverage", "coverage_by_source", "not_attempted_chunks", "seconds",
    ):  # fmt: skip
        assert key in out, key
    assert out["embedded_chunks"] == total and out["coverage"] == 1.0
    assert s9.metadata["prompt_name"] == EMBED_PROMPT_NAME and s9.metadata["batch_size"] == 2
    generations = [o for o in tracer.children(s9) if o.kind == "generation"]
    assert len(generations) == out["requests_sent"] == math.ceil(total / 2)
    [vectors] = tracer.named("S10_vectors")
    assert isinstance(vectors.output, dict)
    assert vectors.output["vectors"] == total and vectors.output["coverage"] == 1.0
    [ingest] = tracer.named("ingest")
    assert isinstance(ingest.output, dict)
    assert (ingest.output["chunks"], ingest.output["embedded"]) == (total, total)
    assert ingest.output["embed_status"] == "complete"


# ---------------------------------------------------------------- small parts
def test_round_robin_takes_one_document_per_source_in_turn() -> None:
    texts = [
        _Text("k1", "beta", "b1", [1]), _Text("k2", "beta", "b1", [2]),
        _Text("k3", "beta", "b2", [3]), _Text("k4", "alpha", "a1", [4]),
        _Text("k5", "alpha", "a2", [5]), _Text("k6", "alpha", "a3", [6]),
    ]  # fmt: skip
    assert [t.key for t in round_robin(texts)] == ["k4", "k1", "k2", "k5", "k3", "k6"]
    assert round_robin([]) == []


def test_the_cache_key_covers_everything_that_decides_the_vector() -> None:
    base = cache_key("m", 8, "document", "1", "text")
    assert base == cache_key("m", 8, "document", "1", "text")
    others = [
        cache_key("n", 8, "document", "1", "text"),
        cache_key("m", 9, "document", "1", "text"),
        cache_key("m", 8, "query", "1", "text"),
        cache_key("m", 8, "document", "2", "text"),
        cache_key("m", 8, "document", "1", "text "),
    ]
    assert len({base, *others}) == 6


def test_unit_blob_checks_and_normalises_every_vector() -> None:
    assert np.frombuffer(unit_blob([3.0, 4.0], 2), dtype="<f4").tolist() == pytest.approx(
        [0.6, 0.8]
    )
    for values, reason in (([1.0], "size"), ([1.0, math.inf], "not_finite"), ([0.0, 0.0], "zero")):
        with pytest.raises(BadVector) as caught:
            unit_blob(values, 2)
        assert caught.value.reason == reason
    assert blob_fits(to_blob([0.6, 0.8]), 2)
    assert not blob_fits(to_blob([0.6, 0.8]), 3) and not blob_fits(to_blob([math.nan, 1.0]), 2)


def test_the_embed_prompt_is_part_of_the_index_fingerprint(tmp_path: Path) -> None:
    prompt_path = f"prompts/{EMBED_PROMPT_NAME}.txt"
    assert prompt_path in FINGERPRINT_CODE_PATHS
    root = tmp_path / "src"
    (root / "prompts").mkdir(parents=True)
    prompt = root / prompt_path
    prompt.write_text("version: 1\ndocument: {text}\n", encoding="utf-8")
    before = code_hash((prompt_path,), root)
    prompt.write_text("version: 2\ndocument: passage: {text}\n", encoding="utf-8")
    assert code_hash((prompt_path,), root) != before


def test_the_embedding_writer_opens_only_a_complete_published_index(
    corpus: Path, tmp_path: Path
) -> None:
    runtime = run(settings_for(corpus, tmp_path))
    assert runtime.index_path is not None
    writer = SqliteEmbeddingWriter(runtime.index_path)
    try:
        coverage = writer.coverage()
        assert coverage.n_embedded == 0 and coverage.share == 0.0
        assert coverage.n_chunks == n_chunks(connect(runtime))
        assert [c.chunk_id for c in writer.unembedded(0, 3)] == [1, 2, 3]
        assert [c.chunk_id for c in writer.chunk_texts([3, 1])] == [3, 1]
    finally:
        writer.close()
    with pytest.raises(Exception, match="no published index"):
        SqliteEmbeddingWriter(building_path(runtime.index_path))


def test_an_unexpected_error_in_s9_stops_it_but_not_startup(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(texts: object) -> list[object]:
        raise RuntimeError("bug in the embed pass (test)")

    monkeypatch.setattr(embed_module, "round_robin", broken)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), FakeEmbedder(dims=DIMS))
    assert len(runtime.known_doc_ids) > 0 and runtime.vectors is None
    [s9] = tracer.named("S9_embed")
    assert s9.level == "ERROR" and "bug in the embed pass" in str(s9.status_message)
    assert meta(connect(runtime))["embed_stop"] == "error"


@pytest.mark.parametrize("failing", ["open", "set_meta", "close"])
def test_an_index_write_error_in_s9_stops_it_but_not_startup(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    # B5 review finding 5: the writer was opened, and its meta written and closed, outside
    # S9's error handler, so a locked or full disk failed start-up despite a keyword index.
    def locked(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked (test)")

    if failing == "open":
        monkeypatch.setattr(embed_module, "SqliteEmbeddingWriter", locked)
    else:
        monkeypatch.setattr(SqliteEmbeddingWriter, failing, locked)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), FakeEmbedder(dims=DIMS))
    assert len(runtime.known_doc_ids) > 0 and runtime.chunks is not None
    [s9] = tracer.named("S9_embed")
    assert s9.level == "ERROR" and "database is locked" in str(s9.status_message)
    assert isinstance(s9.output, dict) and "database is locked" in str(s9.output["error"])


def test_a_bad_answer_through_the_real_client_is_dropped_not_resent(
    corpus: Path, tmp_path: Path
) -> None:
    # B5 review finding 7: GeminiEmbedder rejects a bad answer itself, and S9 took that
    # for a transport error and paid for the same request again.
    from google.genai import types as genai_types

    from corpusqa.llm.client import GeminiEmbedder
    from corpusqa.llm.limits import RateLimiter, SpendLedger

    class ShortFirst:
        """Stands in for genai.Client: the first answer has one vector too few."""

        def __init__(self) -> None:
            self.models, self.calls = self, 0

        def embed_content(
            self, *, model: str, contents: list[genai_types.Content], config: object
        ) -> genai_types.EmbedContentResponse:
            self.calls += 1
            n = len(contents) - (1 if self.calls == 1 else 0)
            values = [[float(self.calls), 1.0] + [0.0] * (DIMS - 2) for _ in range(n)]
            return genai_types.EmbedContentResponse(
                embeddings=[genai_types.ContentEmbedding(values=v) for v in values]
            )

    client = ShortFirst()
    real = GeminiEmbedder(
        client,  # type: ignore[arg-type]
        SpendLedger(tmp_path / "ledger.json", 10.0),
        limiter=RateLimiter(1e9, sleep=lambda s: None),
        sleep=lambda s: None,
        dims=DIMS,
    )
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path), real, embed_options=small())
    total = n_chunks(connect(runtime))
    out = s9_output(tracer)
    assert out["requests_failed:bad_answer"] == 1
    assert "requests_requeued" not in out  # not sent again
    assert out["requests_sent"] == client.calls == math.ceil(total / 2)
    assert runtime.vectors is not None and len(runtime.vectors) == total - 2


def test_a_zero_or_non_unit_cached_vector_is_a_miss(corpus: Path, tmp_path: Path) -> None:
    # B5 review finding 8: cached vectors were checked for size and finite values only, so
    # a zero or non-unit entry became a finished embedding that skews the dot product.
    settings = settings_for(corpus, tmp_path, "index-a")
    run(settings, FakeEmbedder(dims=DIMS))
    with sqlite3.connect(settings.embed_cache_db) as cache:
        keys = [k for (k,) in cache.execute("SELECT key FROM vector_cache ORDER BY key LIMIT 2")]
        cache.execute("UPDATE vector_cache SET vector = ? WHERE key = ?",
                      (to_blob([0.0] * DIMS), keys[0]))  # fmt: skip
        cache.execute("UPDATE vector_cache SET vector = ? WHERE key = ?",
                      (to_blob([2.0] + [0.0] * (DIMS - 1)), keys[1]))  # fmt: skip
    again = FakeEmbedder(dims=DIMS)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings_for(corpus, tmp_path, "index-b"), again)
    out = s9_output(tracer)
    assert out["cache_entries_unusable"] == 2 and sent_texts(again) == 2
    assert runtime.embed_coverage == 1.0
    for (blob,) in connect(runtime).execute("SELECT vector FROM embeddings"):
        assert float(np.linalg.norm(np.frombuffer(blob, dtype="<f4"))) == pytest.approx(1.0)
    assert not blob_fits(to_blob([0.0, 0.0]), 2) and not blob_fits(to_blob([3.0, 4.0]), 2)
