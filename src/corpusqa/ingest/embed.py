"""S9 embed: vectors for the chunks of a published index (design.md S9 and §7).

S9 runs on the index after S10 has checked and published it, both after a fresh build
and when start-up reuses a cached index (the resume). The job is the same each time:
embed every chunk that has no vector yet.

1. Cache pass. Chunks without a vector are read in pages. Each one's embedded text
   (`chunking.embedding_text`: header, newline, text) gets a key (`cache_key`: model,
   dims, role, prompt version, text). Vectors found in the embedding cache go into the
   index at once, without a call.
2. Embed pass. The other texts, each distinct text once, in round-robin order across
   sources by document (a cut-off leaves every source partly covered), in requests of
   `batch_size` texts with `concurrency` in flight. Every vector is checked
   (`store.vectors.unit_blob`) before it is stored. A request whose answer has the wrong
   number of vectors is dropped whole (it cannot be aligned); one bad vector costs only
   its own chunks. An answer the client itself rejects (`BadEmbeddingAnswer`: the real
   client checks every answer) is dropped whole and never resent. Each answered request
   is written to the index and the cache at once (a checkpoint). A request that fails
   with an error is tried once more at the end.

What stops it early, none of which fails start-up: the deadline (no new request after
it; requests in flight finish), the spend cap (every later call would be refused too),
`max_failed_in_a_row` failed requests in a row once its `max_cooldowns` pauses of
`cooldown_s` are used up (API down or quota spent; a per-minute quota refills during a
pause), no embedder (no key), stored vectors of another model (never mixed), or an
unexpected error. What is
not embedded is still found by keyword search; `embed_coverage` says how much is
embedded, and the next start resumes the rest. Nothing embeds after /health turns 200.

`load_vectors` then puts the vectors that fit the embedder into one matrix.
"""

import hashlib
import json
import logging
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from corpusqa import obs
from corpusqa.config import (
    EMBED_BATCH_SIZE,
    EMBED_CACHE_PAGE_CHUNKS,
    EMBED_CONCURRENCY,
    EMBED_COOLDOWN_S,
    EMBED_MAX_COOLDOWNS,
    EMBED_MAX_FAILED_IN_A_ROW,
    EMBED_PRICE_PER_M_TOKENS_USD,
    EMBED_PROMPT_NAME,
    STARTUP_BUDGET_S,
    STARTUP_RESERVE_S,
)
from corpusqa.ingest.chunking import count_tokens, embedding_text
from corpusqa.llm.errors import BadEmbeddingAnswer, EmbedderUnavailable, SpendCapReached
from corpusqa.llm.interfaces import Embedder, EmbedRole
from corpusqa.prompts import load_prompt
from corpusqa.store.interfaces import (
    ChunkReader,
    EmbedCoverage,
    EmbeddingRow,
    EmbeddingWriter,
    StageRow,
    VectorCache,
)
from corpusqa.store.sqlite import SqliteChunkReader, SqliteEmbeddingWriter, SqliteVectorCache
from corpusqa.store.vectors import BadVector, VectorIndex, blob_fits, unit_blob

logger = logging.getLogger(__name__)

S9 = "S9_embed"
S10_VECTORS = "S10_vectors"
ROLE: EmbedRole = "document"
_TEXT_SLOT = "{text}"
_TOKENS_PER_MILLION = 1_000_000

# Why S9 stopped before every chunk had a vector (meta `embed_stop`, span `stop`).
NO_EMBEDDER = "no_embedder"
MODEL_MISMATCH = "model_mismatch"
DEADLINE = "deadline"
SPEND_CAP = "spend_cap"
UNAVAILABLE = "embedder_unavailable"
FAILURES = "failures"
ERROR = "error"
# Status: complete (every chunk has a vector), partial, or skipped (nothing was tried).
COMPLETE, PARTIAL, SKIPPED = "complete", "partial", "skipped"

# What one finished request means for the loop.
_OK, _FAILED, _RETRY = "ok", "failed", "retry"


@dataclass(frozen=True, slots=True)
class EmbedOptions:
    """S9's knobs. Production uses the config constants; tests shrink them."""

    # Seconds from the start of ingest.run after which no new request is sent.
    budget_s: float = STARTUP_BUDGET_S - STARTUP_RESERVE_S
    batch_size: int = EMBED_BATCH_SIZE
    concurrency: int = EMBED_CONCURRENCY
    max_failed_in_a_row: int = EMBED_MAX_FAILED_IN_A_ROW
    cooldown_s: float = EMBED_COOLDOWN_S
    max_cooldowns: int = EMBED_MAX_COOLDOWNS
    clock: Callable[[], float] = time.monotonic  # for the deadline only
    sleep: Callable[[float], None] = time.sleep  # for the cool-downs

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.concurrency < 1 or self.max_failed_in_a_row < 1:
            raise ValueError("batch_size, concurrency and max_failed_in_a_row must be >= 1")
        if self.cooldown_s < 0 or self.max_cooldowns < 0:
            raise ValueError("cooldown_s and max_cooldowns must be >= 0")


@dataclass(frozen=True, slots=True)
class EmbedResult:
    status: str  # COMPLETE | PARTIAL | SKIPPED
    stop: str | None  # why S9 stopped early; None when nothing stopped it
    coverage: EmbedCoverage
    counts: dict[str, object]


@dataclass
class _Text:
    """One distinct embedded text still without a vector, and the chunks that share it."""

    key: str
    source: str
    doc_id: str  # of its first chunk
    chunk_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Request:
    texts: tuple[_Text, ...]
    attempt: int = 0


def cache_key(model: str, dims: int, role: EmbedRole, prompt_version: str, text: str) -> str:
    """The embedding cache key: everything that decides the vector of `text`."""
    record = [model, dims, role, prompt_version, text]
    return hashlib.sha256(json.dumps(record, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the stage
def embed_stage(
    path: Path,
    embedder: Embedder | None,
    cache_db: Path,
    options: EmbedOptions,
    deadline: float,
) -> EmbedResult:
    """S9 on the published index at `path`: the work, its span, meta and stage row."""
    started, stage_started = time.perf_counter(), _now()
    logger.info("ingest: %s: started (vectors from the cache first)", S9)
    prompt = load_prompt(EMBED_PROMPT_NAME)
    with obs.span(
        S9,
        input={"index": str(path)},
        metadata={
            "model": embedder.model if embedder else None,
            "dims": embedder.dims if embedder else None,
            "prompt_name": prompt.name,
            "prompt_version": prompt.version,
            "batch_size": options.batch_size,
            "concurrency": options.concurrency,
            "max_failed_in_a_row": options.max_failed_in_a_row,
            "cooldown_s": options.cooldown_s,
            "max_cooldowns": options.max_cooldowns,
            "budget_left_s": round(deadline - options.clock(), 1),
            "cache": str(cache_db),
        },
    ) as span:
        result = _guarded(path, embedder, cache_db, options, deadline, stage_started, started)
        counts = {**result.counts, "seconds": round(time.perf_counter() - started, 3)}
        span.update(output=counts)
        if result.stop == ERROR:
            span.error(str(counts.get("error")))
    cov = result.coverage
    logger.info(
        "ingest: %s: done in %.1f s (%s: %d of %d chunks embedded, %.1f %%%s)",
        S9, time.perf_counter() - started, result.status, cov.n_embedded, cov.n_chunks,
        100 * cov.share, f", stopped: {result.stop}" if result.stop else "",
    )  # fmt: skip
    return result


def _guarded(
    path: Path,
    embedder: Embedder | None,
    cache_db: Path,
    options: EmbedOptions,
    deadline: float,
    stage_started: str,
    started: float,
) -> EmbedResult:
    """S9's work, meta and stage row. Every step, opening and closing the writer included,
    sits inside the handler: S9 never fails start-up. A broken index is still caught when
    `_open` reads it after S9."""
    writer: SqliteEmbeddingWriter | None = None
    result: EmbedResult | None = None
    try:
        writer = SqliteEmbeddingWriter(path)
        result = _embed_safely(writer, embedder, cache_db, options, deadline)
        counts = {**result.counts, "seconds": round(time.perf_counter() - started, 3)}
        writer.set_meta(
            _meta(result, writer.embedding_models(), load_prompt(EMBED_PROMPT_NAME).version)
        )
        writer.add_stage(StageRow(S9, stage_started, _now(), counts))
    except Exception as exc:
        logger.exception("ingest: S9 could not write to the index; it stops here")
        result = _write_failed(result, exc)
    except BaseException:  # the process is going down: close, then let it go
        if writer is not None:
            _close(writer)
        raise
    if writer is not None and (error := _close(writer)) is not None:
        result = _write_failed(result, error)
    return result


def _write_failed(result: EmbedResult | None, exc: Exception) -> EmbedResult:
    """The result after an index write error: what was done so far, stopped by an error."""
    error = f"{type(exc).__name__}: {exc}"
    if result is None:  # the writer never opened: nothing was tried, coverage unknown
        empty = EmbedCoverage(chunks={}, embedded={})
        return EmbedResult(SKIPPED, ERROR, empty, {"status": SKIPPED, "stop": ERROR,
                                                   "error": error})  # fmt: skip
    # The vectors written so far are committed: the status still says how many there are.
    counts = {**result.counts, "stop": ERROR, "error": error}
    return EmbedResult(result.status, ERROR, result.coverage, counts)


def _close(writer: SqliteEmbeddingWriter) -> Exception | None:
    """Close the writer; the error, if closing failed."""
    try:
        writer.close()
    except Exception as exc:
        logger.exception("ingest: S9 could not close the index writer")
        return exc
    return None


def _embed_safely(
    writer: EmbeddingWriter,
    embedder: Embedder | None,
    cache_db: Path,
    options: EmbedOptions,
    deadline: float,
) -> EmbedResult:
    """`embed_chunks`, with any error turned into a stop: a partial embed never stops
    start-up."""
    cache: VectorCache | None = None
    try:
        if embedder is not None:
            cache = _open_cache(cache_db)
        return embed_chunks(writer, embedder, cache, options, deadline)
    except Exception as exc:
        logger.exception("ingest: S9 stopped by an error; unembedded chunks stay keyword-only")
        counts: dict[str, object] = {"error": f"{type(exc).__name__}: {exc}"}
        return _result(writer, ERROR, counts)
    finally:
        if cache is not None:
            cache.close()


def _open_cache(path: Path) -> VectorCache | None:
    try:
        return SqliteVectorCache(path)
    except Exception:
        logger.exception("ingest: embedding cache %s cannot be opened; embedding without it", path)
        return None


def embed_chunks(
    writer: EmbeddingWriter,
    embedder: Embedder | None,
    cache: VectorCache | None,
    options: EmbedOptions,
    deadline: float,
) -> EmbedResult:
    """Give every chunk of the index without a vector one, until done or stopped."""
    counts: Counter[str] = Counter()
    before = writer.coverage()
    counts["chunks"], counts["already_embedded"] = before.n_chunks, before.n_embedded
    if embedder is None:
        return _result(writer, NO_EMBEDDER, dict(counts))
    stored = writer.embedding_models()
    if stored - {(embedder.model, embedder.dims)}:
        logger.warning(
            "ingest: the index holds vectors of %s, the embedder makes %s; not mixing them",
            sorted(stored), (embedder.model, embedder.dims),
        )  # fmt: skip
        return _result(writer, MODEL_MISMATCH, dict(counts))
    prompt = load_prompt(EMBED_PROMPT_NAME)
    todo = _cache_pass(writer, cache, embedder, prompt.version, counts)
    stop = _embed_pass(writer, embedder, cache, todo, prompt.get(ROLE), options, deadline, counts)
    return _result(writer, stop, dict(counts))


def _cache_pass(
    writer: EmbeddingWriter,
    cache: VectorCache | None,
    embedder: Embedder,
    prompt_version: str,
    counts: Counter[str],
) -> list[_Text]:
    """Store every cached vector; return the distinct texts still to embed, in chunk order."""
    model, dims = embedder.model, embedder.dims
    todo: dict[str, _Text] = {}
    after = 0
    while page := writer.unembedded(after, EMBED_CACHE_PAGE_CHUNKS):
        after = page[-1].chunk_id
        keys = [
            cache_key(model, dims, ROLE, prompt_version, embedding_text(c.header, c.text))
            for c in page
        ]
        found = cache.get_many(list(dict.fromkeys(keys))) if cache is not None else {}
        hits: list[EmbeddingRow] = []
        for chunk, key in zip(page, keys, strict=True):
            blob = found.get(key)
            if blob is not None and blob_fits(blob, dims):
                hits.append(EmbeddingRow(chunk.chunk_id, model, dims, blob))
                continue
            if blob is not None:
                counts["cache_entries_unusable"] += 1
            todo.setdefault(key, _Text(key, chunk.source, chunk.doc_id)).chunk_ids.append(
                chunk.chunk_id
            )
        writer.add_embeddings(hits)
        counts["pending"] += len(page)
        counts["cache_hits"] += len(hits)
    counts["to_embed_texts"] = len(todo)
    counts["to_embed_chunks"] = sum(len(t.chunk_ids) for t in todo.values())
    return list(todo.values())


def round_robin(texts: Sequence[_Text]) -> list[_Text]:
    """Texts reordered: the first document of each source, then the second of each, and so
    on (sources by name; documents and their texts keep their order)."""
    docs_by_source: dict[str, list[list[_Text]]] = {}
    for text in texts:  # chunk id order: a document's texts come together
        docs = docs_by_source.setdefault(text.source, [])
        if docs and docs[-1][0].doc_id == text.doc_id:
            docs[-1].append(text)
        else:
            docs.append([text])
    queues = [docs_by_source[source] for source in sorted(docs_by_source)]
    ordered: list[_Text] = []
    for i in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if i < len(queue):
                ordered.extend(queue[i])
    return ordered


def _embed_pass(
    writer: EmbeddingWriter,
    embedder: Embedder,
    cache: VectorCache | None,
    todo: Sequence[_Text],
    template: str,
    options: EmbedOptions,
    deadline: float,
    counts: Counter[str],
) -> str | None:
    """Embed `todo` in requests until done or stopped; returns why it stopped early."""
    ordered = round_robin(todo)
    size = options.batch_size
    queue = deque(_Request(tuple(ordered[i : i + size])) for i in range(0, len(ordered), size))
    call = obs.bind(embedder.embed)  # each request's generation nests under the S9 span
    stop: str | None = None
    failed_in_a_row = 0
    cooling = False  # too many failures in a row: let the requests in flight settle, then wait
    cooldowns_left = options.max_cooldowns
    in_flight: dict[Future[list[list[float]]], tuple[_Request, int]] = {}
    with ThreadPoolExecutor(max_workers=options.concurrency, thread_name_prefix="embed") as pool:
        while True:
            while stop is None and not cooling and queue and len(in_flight) < options.concurrency:
                if options.clock() >= deadline:
                    stop = DEADLINE
                    break
                request = queue.popleft()
                chunks = writer.chunk_texts([t.chunk_ids[0] for t in request.texts])
                texts = [embedding_text(c.header, c.text) for c in chunks]
                tokens = sum(count_tokens(template.replace(_TEXT_SLOT, t)) for t in texts)
                in_flight[pool.submit(call, texts, ROLE)] = (request, tokens)
                counts["requests_sent"] += 1
            if not in_flight:
                if not cooling or stop is not None or not queue:
                    break
                if cooldowns_left == 0 or options.clock() + options.cooldown_s >= deadline:
                    stop = FAILURES
                    break
                logger.warning(
                    "ingest: S9 %d requests failed in a row; waiting %.0f s before going on",
                    failed_in_a_row, options.cooldown_s,
                )  # fmt: skip
                options.sleep(options.cooldown_s)
                cooldowns_left -= 1
                counts["cooldowns"] += 1
                cooling, failed_in_a_row = False, 0
                continue
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                request, tokens = in_flight.pop(future)
                outcome = _settle(future, request, tokens, writer, cache, embedder, counts)
                if outcome == _OK:
                    failed_in_a_row, cooling = 0, False
                    continue
                if outcome in (_FAILED, _RETRY):
                    failed_in_a_row += 1
                    if outcome == _RETRY and request.attempt == 0:
                        queue.append(_Request(request.texts, attempt=1))
                        counts["requests_requeued"] += 1
                    cooling = cooling or failed_in_a_row >= options.max_failed_in_a_row
                    continue
                stop = stop or outcome  # spend cap or no embedder: nothing more can be sent
    # The loop drains the queue unless something stopped it.
    counts["not_attempted_chunks"] = sum(len(t.chunk_ids) for r in queue for t in r.texts)
    return stop


def _settle(
    future: Future[list[list[float]]],
    request: _Request,
    tokens: int,
    writer: EmbeddingWriter,
    cache: VectorCache | None,
    embedder: Embedder,
    counts: Counter[str],
) -> str:
    """Check and store one finished request; returns _OK, _FAILED, _RETRY or a stop reason."""
    texts = request.texts
    try:
        vectors = future.result()
    except SpendCapReached as exc:
        counts["requests_refused:spend_cap"] += 1
        logger.warning("ingest: S9 %s; no more embedding requests at this start", exc)
        return SPEND_CAP
    except EmbedderUnavailable as exc:
        counts["requests_refused:embedder_unavailable"] += 1
        logger.warning("ingest: S9 %s", exc)
        return UNAVAILABLE
    except BadEmbeddingAnswer as exc:  # answered (and billed) but unusable: never resent
        counts["tokens_est"] += tokens
        counts["requests_failed:bad_answer"] += 1
        logger.warning("ingest: S9 a request of %d texts got a bad answer: %s", len(texts), exc)
        return _FAILED
    except Exception as exc:  # API or transport error after the client's own retries
        counts[f"requests_failed:{type(exc).__name__}"] += 1
        logger.warning("ingest: S9 a request of %d texts failed: %s", len(texts), exc)
        return _RETRY
    counts["tokens_est"] += tokens  # an answered request is billed, usable or not
    if len(vectors) != len(texts):
        counts["requests_failed:count_mismatch"] += 1
        logger.warning(
            "ingest: S9 sent %d texts, got %d vectors; the request is dropped",
            len(texts), len(vectors),
        )  # fmt: skip
        return _FAILED
    model, dims = embedder.model, embedder.dims
    rows: list[EmbeddingRow] = []
    cached: list[tuple[str, bytes]] = []
    for text, values in zip(texts, vectors, strict=True):
        try:
            blob = unit_blob(values, dims)
        except BadVector as bad:
            counts[f"vectors_rejected:{bad.reason}"] += 1
            logger.warning("ingest: S9 rejected a vector for chunk %d: it %s",
                           text.chunk_ids[0], bad)  # fmt: skip
            continue
        rows.extend(EmbeddingRow(chunk_id, model, dims, blob) for chunk_id in text.chunk_ids)
        cached.append((text.key, blob))
    writer.add_embeddings(rows)  # checkpoint: survives a crash from here on
    if cache is not None and cached:
        cache.put_many(model, dims, cached)
    counts["requests_answered"] += 1
    counts["embedded_texts"] += len(cached)
    counts["embedded_chunks"] += len(rows)
    return _OK if rows else _FAILED


def _result(writer: EmbeddingWriter, stop: str | None, counts: dict[str, object]) -> EmbedResult:
    coverage = writer.coverage()
    skipped = stop in (NO_EMBEDDER, MODEL_MISMATCH)  # S9 did not run: say why
    if coverage.n_embedded == coverage.n_chunks:
        status, stop = COMPLETE, stop if skipped else None  # nothing is missing
    elif skipped:
        status = SKIPPED
    else:
        status = PARTIAL
    tokens = counts.get("tokens_est", 0)
    tokens = tokens if isinstance(tokens, int) else 0
    cost = tokens * EMBED_PRICE_PER_M_TOKENS_USD / _TOKENS_PER_MILLION
    out: dict[str, object] = {
        **counts,
        "status": status,
        "stop": stop,
        "embedded": coverage.n_embedded,
        "coverage": round(coverage.share, 4),
        "coverage_by_source": coverage.by_source(),
        "cost_est_usd": round(cost, 4),
    }
    return EmbedResult(status=status, stop=stop, coverage=coverage, counts=out)


def _meta(result: EmbedResult, stored: set[tuple[str, int]], prompt_version: str) -> dict[str, str]:
    """Embedding facts of the index (what is stored, whoever embedded it)."""
    models = sorted(stored)
    cov = result.coverage
    return {
        "embed_status": result.status,
        "embed_stop": result.stop or "",
        "embed_model": ",".join(model for model, _ in models),
        "embed_dims": ",".join(str(dims) for _, dims in models),
        "embed_prompt_version": prompt_version,
        "n_embeddings": str(cov.n_embedded),
        "embed_coverage": f"{cov.share:.4f}",
        "embed_coverage_by_source": json.dumps(cov.by_source(), ensure_ascii=False),
    }


# ---------------------------------------------------------------- vectors in memory
def load_vectors(
    path: Path, embedder: Embedder | None
) -> tuple[ChunkReader, VectorIndex | None, float]:
    """The index's chunk reader, its vectors as one matrix (only those the embedder can
    query: same model and dims), and the share of chunks they cover."""
    started = time.perf_counter()
    with obs.span(S10_VECTORS, input={"index": str(path)}) as span:
        reader = SqliteChunkReader(path)
        n_chunks = reader.count_chunks()
        stored = reader.embedding_models()
        vectors: VectorIndex | None = None
        not_loaded: str | None = None
        if embedder is None:
            not_loaded = NO_EMBEDDER
        elif not stored:
            not_loaded = "no_vectors"
        elif stored != {(embedder.model, embedder.dims)}:
            not_loaded = MODEL_MISMATCH
            logger.warning("ingest: stored vectors %s do not fit the embedder; dense search off",
                           sorted(stored))  # fmt: skip
        else:
            try:
                vectors = VectorIndex.from_rows(reader.embeddings(), embedder.dims)
            except ValueError:
                logger.exception("ingest: the stored vectors cannot be loaded; dense search off")
                not_loaded = "unreadable"
        coverage = len(vectors) / n_chunks if vectors is not None and n_chunks else 0.0
        span.update(
            output={
                "chunks": n_chunks,
                "vectors": len(vectors) if vectors is not None else 0,
                "dims": vectors.dims if vectors is not None else None,
                "coverage": round(coverage, 4),
                "not_loaded": not_loaded,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    return reader, vectors, coverage


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
