"""Chunking retrieval eval: `uv run eval-retrieval --chunking <strategy> [options]`.

Measures how well one chunking variant lets a fixed, simple retriever find the gold
documents of the tune questions (never held-out: questions come only through
`scripts/show_questions.load_questions`, which refuses every other set). One run is one
trace `eval_retrieval` with a span per phase:

1. `docs`: the document index of CORPUS_DIR built by the ingest pipeline (the corpus is
   only read) under .cache/eval/docs-promote-<on|off>/, reused while the corpus and the
   ingest/store/config code are unchanged.
2. `chunks`: every document chunked by the variant (`ingest.chunking.chunk_rows`) and
   written with its keyword index to .cache/eval/<variant>/chunks.sqlite.
3. `embed`: the exact text of each chunk (header, newline, text) embedded through
   `llm.client`, behind the embedding cache start-up also uses (`config.EMBED_CACHE_DB`,
   key `ingest.embed.cache_key`: sha256 of model, dims, role, prompt version and text),
   several requests in flight; each finished request is cached at once.
4. `queries`: each question with gold documents -> bm25, dense and hybrid document
   rankings (`retrieval.hybrid`).
5. `report`: metrics per mode to eval/results/<set>/retrieval/<time>-<variant>/
   (report.md, report.json; ranks only, never document ids) and one eval/history.csv row.

A variant index is reused when its manifest matches. `--bm25-only` skips embedding;
`--screen` keeps only every gold document plus a fixed random sample of others; `--probe N`
embeds N uncached chunks of the variant to measure throughput, then stops.
`--fields` puts field values in every chunk header (`ingest.chunking.select_header_fields`
on the document index's profile; variant name `-fields`): a new variant, so every chunk is
embedded again. Without it a variant follows `config.CHUNK_HEADER_FIELDS`.
`--bm25-weight`, `--dense-weight` and `--rrf-k` change only the fusion in phase 4: the
cached index is reused as is, and the run's name (report folder, history row) gains a
suffix such as `-w0.5x1-k20` (`Fusion.suffix`).

`--production` scores the index start-up itself builds instead (`ingest.run` with the
server's settings: .cache/index, the default chunking, final display titles and the
vectors S9 stored), so the number is the served index's; phases 1-3 are then start-up's.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpusqa import obs
from corpusqa.config import (
    CHUNK_DB_NAME,
    CHUNK_WRITE_BATCH_ROWS,
    EMBED_BATCH_SIZE,
    EMBED_CACHE_DB,
    EMBED_PRICE_PER_M_TOKENS_USD,
    EMBED_PROMPT_NAME,
    EMBED_SPEND_CAP_USD,
    EMBED_SPEND_LEDGER,
    EVAL_ALL_GOLD_K,
    EVAL_CACHE_DIR,
    EVAL_EMBED_WORKERS,
    EVAL_RECALL_KS,
    EVAL_SCREEN_SAMPLE_DOCS,
    EVAL_SCREEN_SEED,
    EVAL_TRACE_FLUSH_TIMEOUT_S,
    LOG_FORMAT,
    LOG_LEVEL,
    PROMOTE_PROSE_FIELDS,
    RETRIEVAL_TOP_CHUNKS,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
    Settings,
)
from corpusqa.ingest import core as ingest_core
from corpusqa.ingest import pipeline as ingest_pipeline
from corpusqa.ingest import run as run_ingest
from corpusqa.ingest.chunking import (
    STRATEGIES,
    ChunkSpec,
    chunk_rows,
    embedding_text,
    select_header_fields,
)
from corpusqa.ingest.embed import cache_key
from corpusqa.llm.client import GeminiEmbedder, estimate_tokens
from corpusqa.llm.errors import EmbedderUnavailable, LLMError
from corpusqa.llm.interfaces import Embedder, EmbedRole
from corpusqa.llm.limits import SpendLedger
from corpusqa.prompts import load_prompt
from corpusqa.retrieval.hybrid import MODES, HybridRetriever, Mode
from corpusqa.store.interfaces import ChunkReader, ChunkRow, EmbeddingRow
from corpusqa.store.sqlite import (
    SqliteChunkReader,
    SqliteChunkWriter,
    SqliteReader,
    SqliteVectorCache,
)
from corpusqa.store.vectors import VectorIndex, from_blob, to_blob
from eval.retrieval_metrics import ModeScores, QuestionRun, gold_ranks, score_all
from scripts.show_questions import ALLOWED_SETS, load_questions

logger = logging.getLogger("eval.retrieval")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"
HISTORY = ROOT / "eval" / "history.csv"
# eval/history.csv is shared by every eval kind; `metrics` holds the kind's own numbers.
HISTORY_HEADER = (
    "timestamp", "git_commit", "kind", "set", "variant", "questions", "headline", "value",
    "metrics", "cost_usd", "seconds", "report",
)  # fmt: skip
TRACE_NAME = "eval_retrieval"
MANIFEST = "manifest.json"
# Code whose change makes a cached document or chunk index stale.
CODE_PARTS = ("ingest", "store", "config.py")
TEXT_SLOT = "{text}"
PROGRESS_STEPS = 20  # progress lines per embedding run
# The history row's headline: per-question hybrid recall@10 (bm25 when a run has no dense
# side). Rows before 2026-10-16 carry the pooled hybrid_recall@10 under another name.
HEADLINE_MODE: Mode = "hybrid"
HEADLINE_METRIC = "q_recall"
HEADLINE_K = 10
# Reports written before fusion weights existed used plain RRF: weight 1.0 per ranker.
PLAIN_RRF_WEIGHT = 1.0


# ---------------------------------------------------------------- variant
@dataclass(frozen=True, slots=True)
class Variant:
    spec: ChunkSpec
    promote: bool  # PROMOTE_PROSE_FIELDS for the document build
    screen: bool  # gold documents + a fixed sample instead of the whole corpus

    @property
    def name(self) -> str:
        parts = [self.spec.strategy, str(self.spec.limit_tokens)]
        if self.spec.strategy == "fixed":
            parts.append(f"o{self.spec.overlap}")
        if not self.spec.header:
            parts.append("noheader")
        if self.spec.header_fields:
            parts.append("fields")
        if not self.promote:
            parts.append("nopromote")
        if self.screen:
            parts.append(f"screen{EVAL_SCREEN_SAMPLE_DOCS}s{EVAL_SCREEN_SEED}")
        return "-".join(parts)

    def describe(self) -> dict[str, object]:
        return {
            "strategy": self.spec.strategy,
            "max_tokens": self.spec.limit_tokens,
            "overlap_tokens": self.spec.overlap,
            "header": self.spec.header,
            "header_fields": self.spec.header_fields,
            "promote_prose_fields": self.promote,
            "screen": self.screen,
        }


@dataclass(frozen=True, slots=True)
class Fusion:
    """How the query step fuses the two rankers. Never part of a cached index: changing it
    reuses the index and only renames the run."""

    bm25_weight: float = RRF_BM25_WEIGHT
    dense_weight: float = RRF_DENSE_WEIGHT
    rrf_k: int = RRF_K

    @property
    def suffix(self) -> str:
        """'' at the config defaults; else `-w<bm25>x<dense>` and/or `-k<k>` for what differs."""
        out = ""
        if (self.bm25_weight, self.dense_weight) != (RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT):
            out += f"-w{_weight_name(self.bm25_weight)}x{_weight_name(self.dense_weight)}"
        if self.rrf_k != RRF_K:
            out += f"-k{self.rrf_k}"
        return out

    def describe(self) -> dict[str, object]:
        return {
            "rrf_k": self.rrf_k,
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
        }


def _weight_name(weight: float) -> str:
    """A weight in a run name: short when that reads back exactly (0.5, 1), else every digit
    (`repr`), so two weights never share a name; -0.0 is named 0."""
    weight += 0.0  # -0.0 + 0.0 == +0.0
    short = f"{weight:g}"
    return short if float(short) == weight else repr(weight)


# ---------------------------------------------------------------- embedding cache
class CachedEmbedder:
    """An `Embedder` with the vector cache in front.

    Misses go to `inner` in requests of `batch_size` texts, `workers` in flight; each
    finished request is cached at once, so an interrupted run loses at most the requests
    in flight. Use from one thread (the cache connection is single-threaded).
    """

    def __init__(
        self,
        inner: Embedder,
        cache: SqliteVectorCache,
        prompt_version: str,
        workers: int = EVAL_EMBED_WORKERS,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        self.model, self.dims = inner.model, inner.dims
        self.inner = inner  # the plain client (start-up's S9 caches on its own)
        self._inner = inner
        self._cache = cache
        self._prompt_version = prompt_version
        self.workers = workers
        self._batch_size = batch_size
        self.counts: Counter[str] = Counter()  # hits, misses, requests
        self.seconds = 0.0  # spent waiting for the model

    def key(self, text: str, role: EmbedRole) -> str:
        return cache_key(self.model, self.dims, role, self._prompt_version, text)

    def cached_keys(self, keys: Sequence[str]) -> set[str]:
        return set(self._cache.get_many(list(dict.fromkeys(keys))))

    def blobs(self, texts: Sequence[str], role: EmbedRole) -> list[bytes]:
        """Stored-form vectors of `texts`, in order."""
        keys = [self.key(text, role) for text in texts]
        found = self._cache.get_many(list(dict.fromkeys(keys)))
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
        self.counts["hits"] += len(set(keys)) - len(missing)
        self.counts["misses"] += len(missing)
        if missing:
            found.update(self._embed_missing(list(missing.items()), role))
        return [found[k] for k in keys]

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        return [from_blob(blob) for blob in self.blobs(texts, role)]

    def _embed_missing(self, items: list[tuple[str, str]], role: EmbedRole) -> dict[str, bytes]:
        groups = [items[i : i + self._batch_size] for i in range(0, len(items), self._batch_size)]
        step = max(1, len(groups) // PROGRESS_STEPS)
        done: dict[str, bytes] = {}
        started = time.monotonic()
        call = obs.bind(self._inner.embed)  # embedding generations nest under this span
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures: dict[Future[list[list[float]]], list[tuple[str, str]]] = {
                pool.submit(call, [text for _, text in group], role): group for group in groups
            }
            try:
                for n, future in enumerate(as_completed(futures), start=1):
                    group = futures[future]
                    pairs = [
                        (key, to_blob(vector))
                        for (key, _), vector in zip(group, future.result(), strict=True)
                    ]
                    self._cache.put_many(self.model, self.dims, pairs)
                    done.update(pairs)
                    self.counts["requests"] += 1
                    if n % step == 0 or n == len(groups):
                        elapsed = time.monotonic() - started
                        logger.info(
                            "embed %s: %d/%d requests, %d texts, %.0f s",
                            role, n, len(groups), len(done), elapsed,
                        )  # fmt: skip
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                self.seconds += time.monotonic() - started
        return done


# ---------------------------------------------------------------- caches on disk
def _sha_of_files(paths: Sequence[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def code_hash() -> str:
    src = ROOT / "src" / "corpusqa"
    files: list[Path] = []
    for part in CODE_PARTS:
        target = src / part
        files.extend(sorted(target.rglob("*.py")) if target.is_dir() else [target])
    return _sha_of_files(files, src)


def corpus_fingerprint(corpus: Path) -> str:
    """Paths, sizes and modification times of every file (cheap; the corpus is read-only)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        stat = path.stat()
        line = f"{path.relative_to(corpus).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()[:16]


def read_manifest(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")


@contextmanager
def prose_promotion(on: bool) -> Iterator[None]:
    """Build documents with PROMOTE_PROSE_FIELDS set to `on` (the flag B0 added for this)."""
    modules = (ingest_core, ingest_pipeline)
    name = "PROMOTE_PROSE_FIELDS"
    before = [getattr(module, name) for module in modules]
    try:
        for module in modules:
            setattr(module, name, on)
        yield
    finally:
        for module, value in zip(modules, before, strict=True):
            setattr(module, name, value)


def ensure_documents(settings: Settings, promote: bool) -> tuple[Path, dict[str, Any]]:
    """The document index for this promotion setting. The ingest pipeline reuses its own
    fingerprinted cache under `folder`, so an unchanged corpus and code are not rebuilt."""
    folder = EVAL_CACHE_DIR / f"docs-promote-{'on' if promote else 'off'}"
    wanted = {
        "corpus_dir": str(settings.corpus_dir),
        "corpus": corpus_fingerprint(settings.corpus_dir),
        "code": code_hash(),
        "promote_prose_fields": promote,
    }
    with obs.span("docs", input=wanted) as span:
        started = time.monotonic()
        with prose_promotion(promote):
            runtime = run_ingest(replace(settings, index_dir=folder))
        if runtime.index_path is None:
            raise RuntimeError("ingest returned no index path")
        seconds = round(time.monotonic() - started, 1)
        span.update(output={"reused": runtime.cache_hit, "documents": len(runtime.known_doc_ids),
                            "seconds": seconds, "index": str(runtime.index_path)})  # fmt: skip
    return runtime.index_path, wanted


# ---------------------------------------------------------------- build one variant
def make_chunks(docs_db: Path, spec: ChunkSpec, keep: set[str] | None) -> list[ChunkRow]:
    """Every kept document chunked by `spec`; with header fields, the fields are picked from
    the document index's own profile, as S6 picks them at start-up."""
    reader = SqliteReader(docs_db)
    header_fields = select_header_fields(reader.field_profiles()) if spec.header_fields else None
    rows: list[ChunkRow] = []
    for doc in reader.document_texts(header_fields=header_fields):
        if keep is None or doc.doc_id in keep:
            rows.extend(chunk_rows(doc, spec, len(rows) + 1))
    return rows


def chunk_stats(chunks: Sequence[ChunkRow], n_docs: int, document_template: str) -> dict[str, Any]:
    sizes = sorted(c.n_tokens for c in chunks)
    sent = sum(
        estimate_tokens(document_template.replace(TEXT_SLOT, embedding_text(c.header, c.text)))
        for c in chunks
    )
    return {
        "documents": n_docs,
        "documents_without_chunks": n_docs - len({c.doc_id for c in chunks}),
        "chunks": len(chunks),
        "levels": dict(Counter(c.level for c in chunks).most_common()),
        "text_tokens": sum(sizes),
        "chunk_tokens_p50": _percentile(sizes, 0.5),
        "chunk_tokens_p90": _percentile(sizes, 0.9),
        "chunk_tokens_max": sizes[-1] if sizes else 0,
        "embedded_tokens": sent,
        "embed_cost_full_usd": round(sent * EMBED_PRICE_PER_M_TOKENS_USD / 1_000_000, 4),
    }


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def build_index(
    path: Path, chunks: Sequence[ChunkRow], embedder: CachedEmbedder | None
) -> dict[str, float]:
    """Write chunks, their keyword index and (with an embedder) their vectors to `path`."""
    writer = SqliteChunkWriter(path)
    timings: dict[str, float] = {}
    try:
        started = time.monotonic()
        with obs.span("chunks_write", input={"chunks": len(chunks), "path": str(path)}):
            for i in range(0, len(chunks), CHUNK_WRITE_BATCH_ROWS):
                writer.add_chunks(chunks[i : i + CHUNK_WRITE_BATCH_ROWS])
            writer.rebuild_fts()
        timings["write_s"] = round(time.monotonic() - started, 1)
        if embedder is not None:
            started = time.monotonic()
            texts = [embedding_text(c.header, c.text) for c in chunks]
            with obs.span(
                "embed",
                input={"texts": len(texts), "model": embedder.model, "dims": embedder.dims},
            ) as span:
                blobs = embedder.blobs(texts, "document")
                span.update(output=dict(embedder.counts))
            rows = [
                EmbeddingRow(c.chunk_id, embedder.model, embedder.dims, blob)
                for c, blob in zip(chunks, blobs, strict=True)
            ]
            for i in range(0, len(rows), CHUNK_WRITE_BATCH_ROWS):
                writer.add_embeddings(rows[i : i + CHUNK_WRITE_BATCH_ROWS])
            timings["embed_s"] = round(time.monotonic() - started, 1)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return timings


# ---------------------------------------------------------------- run
@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    text: str
    gold: frozenset[str]


def gold_questions(set_name: str) -> tuple[list[Question], int]:
    """Questions of the set that have gold documents, and how many have none."""
    rows = load_questions(set_name)
    questions = [
        Question(r["question_id"], r["question"], frozenset(r["expected_doc_ids"]))
        for r in rows
        if r["expected_doc_ids"]
    ]
    return questions, len(rows) - len(questions)


def screen_subset(all_ids: frozenset[str], questions: Sequence[Question]) -> set[str]:
    gold = set().union(*(q.gold for q in questions)) & all_ids
    others = sorted(all_ids - gold)
    sample = random.Random(EVAL_SCREEN_SEED).sample(  # noqa: S311 (a fixed sample, not security)
        others, min(EVAL_SCREEN_SAMPLE_DOCS, len(others))
    )
    return gold | set(sample)


def make_embedder(settings: Settings, workers: int) -> CachedEmbedder | None:
    """The Gemini embedder behind the shared embedding cache (None without a key)."""
    try:
        inner = GeminiEmbedder.from_settings(settings)
    except EmbedderUnavailable as exc:
        logger.warning("%s; dense and hybrid are skipped (keyword-only results)", exc)
        return None
    version = load_prompt(EMBED_PROMPT_NAME).version
    return CachedEmbedder(inner, SqliteVectorCache(EMBED_CACHE_DB), version, workers)


def ledger_totals() -> tuple[int, float]:
    totals = SpendLedger(EMBED_SPEND_LEDGER, EMBED_SPEND_CAP_USD).totals
    return totals.tokens, totals.cost_usd


@dataclass(frozen=True, slots=True)
class Target:
    """The index one eval run scores."""

    name: str
    settings: dict[str, object]
    stats: dict[str, Any]
    reused: bool
    chunks: ChunkReader
    vectors: VectorIndex | None
    doc_ids: frozenset[str]


def run(args: argparse.Namespace, settings: Settings) -> int:
    started = time.monotonic()
    tokens_before, spent_before = ledger_totals()
    embedder = None if args.bm25_only else make_embedder(settings, args.workers)
    questions, without_gold = gold_questions(args.set)
    if args.production:
        target = production_index(settings, embedder)
    else:
        built = variant_index(args, settings, embedder, questions)
        if isinstance(built, int):  # --probe ran instead
            return built
        target = built
    dense = embedder is not None and target.vectors is not None and len(target.vectors) > 0
    fusion = Fusion(args.bm25_weight, args.dense_weight, args.rrf_k)
    name = target.name + fusion.suffix  # the index's own name (its cache folder) is unchanged

    with obs.span("queries", input={"set": args.set, "questions": len(questions)}) as span:
        retriever = HybridRetriever(
            target.chunks,
            target.vectors,
            embedder,
            rrf_k=fusion.rrf_k,
            bm25_weight=fusion.bm25_weight,
            dense_weight=fusion.dense_weight,
        )
        query_started = time.monotonic()
        runs = [QuestionRun(q.question_id, q.gold, retriever.search(q.text)) for q in questions]
        query_s = round((time.monotonic() - query_started) / max(1, len(questions)), 3)
        span.update(output={"questions": len(runs), "seconds_per_question": query_s})

    tokens_after, spent_after = ledger_totals()
    with obs.span("report") as span:
        scores = score_all(runs, EVAL_RECALL_KS, EVAL_ALL_GOLD_K)
        result = {
            "variant": name,
            "settings": target.settings,
            "set": args.set,
            "questions_with_gold": len(questions),
            "questions_without_gold_skipped": without_gold,
            "gold_documents": scores["bm25"].gold_documents,
            "gold_documents_not_in_index": sum(len(q.gold - target.doc_ids) for q in questions),
            "dense": dense,
            "retriever": {"top_chunks": RETRIEVAL_TOP_CHUNKS, **fusion.describe()},
            "scores": {m: _scores_json(scores[m]) if dense or m == "bm25" else None for m in MODES},
            "index": target.stats,
            "reused_index": target.reused,
            "embedding": {
                "model": embedder.model if embedder else None,
                "dims": embedder.dims if embedder else None,
                "cache": dict(embedder.counts) if embedder else {},
                "seconds_this_run": round(embedder.seconds, 1) if embedder else 0.0,
                "tokens_this_run": tokens_after - tokens_before,
                "spent_this_run_usd": round(spent_after - spent_before, 4),
                "ledger_total_usd": round(spent_after, 4),
            },
            "seconds_per_question": query_s,
            "seconds": round(time.monotonic() - started, 1),
            "per_question": [_question_json(r, dense) for r in runs],
        }
        folder_out = write_report(result, args.set, name)
        append_history(result, folder_out)
        span.update(output={**result["scores"], "report": str(folder_out)})
    print_summary(result, folder_out)
    return 0


def production_index(settings: Settings, embedder: CachedEmbedder | None) -> Target:
    """The index start-up builds and serves: `ingest.run` with the server's settings (index
    under .cache/index, default chunking, final display titles, the vectors S9 stored).
    It is reused when start-up already built it; S9 fills in any missing vector through
    the plain client (S9 keeps its own cache: the same file)."""
    with obs.span("production_index", input={"corpus_dir": str(settings.corpus_dir)}) as span:
        started = time.monotonic()
        runtime = run_ingest(settings, embedder.inner if embedder else None)
        seconds = round(time.monotonic() - started, 1)
        if runtime.chunks is None:
            raise RuntimeError("ingest returned no chunk reader")
        template = load_prompt(EMBED_PROMPT_NAME).get("document")
        chunks = list(_chunk_rows(runtime.chunks))
        stats = chunk_stats(chunks, len(runtime.known_doc_ids), template)
        stats |= {"build_s": seconds, "embed_coverage": round(runtime.embed_coverage, 4)}
        span.update(output={"reused": runtime.cache_hit, "index": str(runtime.index_path),
                            **stats})  # fmt: skip
    default = Variant(ChunkSpec(), promote=PROMOTE_PROSE_FIELDS, screen=False)
    return Target(
        name=f"production-{default.name}",
        settings={**default.describe(), "index": "production"},
        stats=stats,
        reused=runtime.cache_hit,
        chunks=runtime.chunks,
        vectors=runtime.vectors,
        doc_ids=runtime.known_doc_ids,
    )


def _chunk_rows(chunks: ChunkReader) -> Iterator[ChunkRow]:
    if not isinstance(chunks, SqliteChunkReader):
        raise TypeError("the production index is read through SqliteChunkReader")
    return chunks.chunks()


def variant_index(
    args: argparse.Namespace,
    settings: Settings,
    embedder: CachedEmbedder | None,
    questions: Sequence[Question],
) -> Target | int:
    """One chunking variant's own index under .cache/eval/<variant>/ (built or reused)."""
    # `--fields` turns header fields on; without it the spec follows CHUNK_HEADER_FIELDS.
    spec = ChunkSpec(args.chunking, args.max_tokens, args.overlap, header=not args.no_header,
                     fields=True if args.fields else None)  # fmt: skip
    variant = Variant(spec, promote=not args.no_promote, screen=args.screen)
    dense = embedder is not None
    docs_db, docs_manifest = ensure_documents(settings, variant.promote)
    all_ids = SqliteReader(docs_db).doc_ids()
    keep = screen_subset(all_ids, questions) if variant.screen else None
    template = load_prompt(EMBED_PROMPT_NAME).get("document")

    folder = EVAL_CACHE_DIR / variant.name
    index = folder / CHUNK_DB_NAME
    wanted = {
        "variant": variant.name,
        **variant.describe(),
        "documents": docs_manifest,
        "code": code_hash(),
        "model": embedder.model if embedder else None,
        "dims": embedder.dims if embedder else None,
        "prompt_version": load_prompt(EMBED_PROMPT_NAME).version,
        # A screened index holds only these documents: another question set needs its own.
        "screen_docs": _digest(keep) if keep is not None else None,
    }

    if args.probe:
        if embedder is None:
            logger.error("--probe needs GEMINI_API_KEY")
            return 2
        chunks = make_chunks(docs_db, spec, keep)
        return probe(chunks, embedder, args.probe)

    manifest = read_manifest(folder)
    reusable = (
        manifest is not None
        and index.is_file()
        and {k: manifest.get(k) for k in wanted if k not in ("model", "dims")}
        == {k: v for k, v in wanted.items() if k not in ("model", "dims")}
        and (not dense or (manifest.get("embedded") and manifest.get("model") == wanted["model"]
                           and manifest.get("dims") == wanted["dims"]))
    )  # fmt: skip
    with obs.span("chunks", input=wanted) as span:
        if reusable and manifest is not None:
            stats = manifest["stats"]
            span.update(output={"reused": True, **stats})
        else:
            build_started = time.monotonic()
            chunks = make_chunks(docs_db, spec, keep)
            stats = chunk_stats(chunks, len(keep) if keep is not None else len(all_ids), template)
            stats["chunk_s"] = round(time.monotonic() - build_started, 1)
            stats.update(build_index(index, chunks, embedder))
            stats["build_s"] = round(time.monotonic() - build_started - stats.get("embed_s", 0), 1)
            write_manifest(folder, {**wanted, "embedded": dense, "stats": stats})
            span.update(output={"reused": False, **stats})
            del chunks
    reader = SqliteChunkReader(index)
    vectors = VectorIndex.from_rows(reader.embeddings(), embedder.dims) if embedder else None
    return Target(
        name=variant.name,
        settings=variant.describe(),
        stats=stats,
        reused=bool(reusable),
        chunks=reader,
        vectors=vectors,
        doc_ids=all_ids,
    )


def _digest(doc_ids: set[str]) -> str:
    """sha256 of a set of document ids (order-free)."""
    return hashlib.sha256("\n".join(sorted(doc_ids)).encode("utf-8")).hexdigest()


def probe(chunks: Sequence[ChunkRow], embedder: CachedEmbedder, n: int) -> int:
    """Embed `n` uncached chunk texts and report the sustained throughput."""
    texts = [embedding_text(c.header, c.text) for c in chunks]
    cached = embedder.cached_keys([embedder.key(t, "document") for t in texts])
    fresh = [t for t in dict.fromkeys(texts) if embedder.key(t, "document") not in cached]
    sample = random.Random(EVAL_SCREEN_SEED).sample(fresh, min(n, len(fresh)))  # noqa: S311
    tokens_before, spent_before = ledger_totals()
    with obs.span("probe", input={"texts": len(sample)}) as span:
        started = time.monotonic()
        embedder.blobs(sample, "document")
        seconds = time.monotonic() - started
        tokens_after, spent_after = ledger_totals()
        tokens = tokens_after - tokens_before
        out = {
            "texts": len(sample),
            "requests": embedder.counts["requests"],
            "workers": embedder.workers,
            "tokens": tokens,
            "seconds": round(seconds, 1),
            "tokens_per_minute": round(tokens / seconds * 60) if seconds else None,
            "texts_per_minute": round(len(sample) / seconds * 60) if seconds else None,
            "spent_usd": round(spent_after - spent_before, 4),
        }
        span.update(output=out)
    print(json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- report
def _scores_json(s: ModeScores) -> dict[str, float]:
    out = {f"recall@{k}": round(v, 4) for k, v in s.recall.items()}
    out |= {f"q_recall@{k}": round(v, 4) for k, v in s.q_recall.items()}
    out[f"all_gold@{EVAL_ALL_GOLD_K}"] = round(s.all_gold, 4)
    out["mrr"] = round(s.mrr, 4)
    return out


def _question_json(run: QuestionRun, dense: bool) -> dict[str, object]:
    modes = MODES if dense else ("bm25",)
    return {
        "question_id": run.question_id,
        "gold": len(run.gold),
        "gold_ranks": {m: gold_ranks(run.rankings.of(m), run.gold) for m in modes},
    }


def write_report(result: dict[str, Any], set_name: str, variant: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = RESULTS_DIR / set_name / "retrieval" / f"{stamp}-{variant}"
    folder.mkdir(parents=True, exist_ok=True)
    result = {"timestamp": stamp, "git_commit": git_commit(), **result}
    (folder / "report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (folder / "report.md").write_text(render_markdown(result), encoding="utf-8")
    return folder


def render_markdown(r: dict[str, Any]) -> str:
    ks = [f"{metric}@{k}" for metric in ("recall", "q_recall") for k in EVAL_RECALL_KS]
    cols = [*ks, f"all_gold@{EVAL_ALL_GOLD_K}", "mrr"]
    fused = r["retriever"]
    lines = [
        f"# Retrieval eval: {r['variant']}",
        "",
        f"Set `{r['set']}`: {r['questions_with_gold']} questions with gold documents "
        f"({r['gold_documents']} gold documents; {r['questions_without_gold_skipped']} "
        f"questions without gold skipped). Commit `{r['git_commit']}`, {r['timestamp']}.",
        f"Settings: `{json.dumps(r['settings'])}`. Retriever: top {fused['top_chunks']} "
        f"chunks per ranker, RRF k = {fused['rrf_k']}, weights bm25 "
        f"{fused.get('bm25_weight', PLAIN_RRF_WEIGHT):g} / dense "
        f"{fused.get('dense_weight', PLAIN_RRF_WEIGHT):g}, chunks mapped to documents.",
        "",
        "| mode | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for mode in MODES:
        s = r["scores"][mode] or {}
        # n/a: no dense side, or a metric an older report did not have (q_recall).
        cells = [f"{s[c]:.3f}" if c in s else "n/a" for c in cols]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    ix, emb = r["index"], r["embedding"]
    lines += [
        "",
        "recall@k (pooled) = gold documents in the top k over all gold documents of all "
        "questions, so a question with many gold documents weighs more.",
        "q_recall@k (per question) = each question's share of its gold documents in the top k, "
        "averaged over the questions, so every question weighs the same.",
        f"all_gold@{EVAL_ALL_GOLD_K} = share of questions with every gold document in the top "
        f"{EVAL_ALL_GOLD_K}; MRR = mean 1/rank of the first gold document.",
        "",
        "## Index",
        f"- documents {ix['documents']} ({ix['documents_without_chunks']} without chunks), "
        f"chunks {ix['chunks']}, levels {ix['levels']}",
        f"- chunk tokens p50 {ix['chunk_tokens_p50']}, p90 {ix['chunk_tokens_p90']}, "
        f"max {ix['chunk_tokens_max']}; text tokens {ix['text_tokens']}",
        f"- embedded tokens (est., whole variant) {ix['embedded_tokens']}, cost to embed from "
        f"scratch ${ix['embed_cost_full_usd']:.2f}",
        f"- chunking {ix.get('chunk_s')} s, writing {ix.get('write_s')} s, embedding "
        f"{ix.get('embed_s', 0)} s (build {ix.get('build_s')} s); reused index: "
        f"{r['reused_index']}",
        f"- this run: embedding cache {emb['cache']}, {emb['tokens_this_run']} tokens, "
        f"${emb['spent_this_run_usd']:.4f} (ledger total ${emb['ledger_total_usd']:.2f}); "
        f"{r['seconds_per_question']} s per question; {r['seconds']} s in all",
        "",
        "## Per question (ranks of the gold documents; '-' = not in the top results)",
    ]
    modes = list(r["per_question"][0]["gold_ranks"]) if r["per_question"] else []
    lines += ["| question | gold | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for q in r["per_question"]:
        ranks = [
            ", ".join("-" if x is None else str(x) for x in q["gold_ranks"][m])
            for m in q["gold_ranks"]
        ]
        lines.append(f"| {q['question_id']} | {q['gold']} | " + " | ".join(ranks) + " |")
    return "\n".join(lines) + "\n"


def history_metrics(r: dict[str, Any]) -> str:
    parts = []
    for mode in MODES:
        s = r["scores"][mode]
        if s is not None:
            parts.append(f"{mode} " + " ".join(f"{k}={v:.3f}" for k, v in s.items()))
    ix = r["index"]
    parts.append(
        f"chunks={ix['chunks']} embedded_tokens={ix['embedded_tokens']} "
        f"embed_cost_full=${ix['embed_cost_full_usd']:.2f} build_s={ix.get('build_s')} "
        f"embed_s={ix.get('embed_s', 0)}"
    )
    return " | ".join(parts)


def headline(r: dict[str, Any]) -> tuple[str, float]:
    """The history headline's name and value (bm25 when the run had no dense side)."""
    mode = HEADLINE_MODE if r["dense"] else "bm25"
    key = f"{HEADLINE_METRIC}@{HEADLINE_K}"
    return f"{mode}_{key}", r["scores"][mode][key]


def append_history(r: dict[str, Any], report: Path) -> None:
    name, value = headline(r)
    new = not HISTORY.is_file()
    with HISTORY.open("a", encoding="utf-8", newline="") as handle:
        out = csv.writer(handle)
        if new:
            out.writerow(HISTORY_HEADER)
        out.writerow([
            datetime.now(UTC).isoformat(timespec="seconds"), git_commit(), "retrieval", r["set"],
            r["variant"], r["questions_with_gold"], name, f"{value:.4f}", history_metrics(r),
            f"{r['embedding']['spent_this_run_usd']:.4f}", r["seconds"],
            report.relative_to(ROOT).as_posix(),
        ])  # fmt: skip


def git_commit() -> str:
    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return done.stdout.strip()

    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return git("rev-parse", "--short", "HEAD") + ("+dirty" if dirty else "")


def print_summary(r: dict[str, Any], report: Path) -> None:
    chunks = r["index"]["chunks"]
    print(f"\n{r['variant']}  ({r['questions_with_gold']} questions, {chunks} chunks)")
    for mode in MODES:
        s = r["scores"][mode]
        shown = "n/a" if s is None else "  ".join(f"{k}={v:.3f}" for k, v in s.items())
        print(f"  {mode:<6} {shown}")
    print(f"  report: {report.relative_to(ROOT)}")


# ---------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--set", default="tune", choices=ALLOWED_SETS)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--chunking", choices=sorted(STRATEGIES), help="score this variant")
    which.add_argument(
        "--production",
        action="store_true",
        help="score the index start-up builds (.cache/index; built or resumed if needed)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="chunk size limit")
    parser.add_argument("--overlap", type=int, default=None, help="fixed only: overlap tokens")
    parser.add_argument("--no-header", action="store_true", help="chunks without the header")
    parser.add_argument("--fields", action="store_true", help="field values in chunk headers")
    parser.add_argument("--no-promote", action="store_true", help="PROMOTE_PROSE_FIELDS off")
    parser.add_argument("--bm25-only", action="store_true", help="no embeddings")
    parser.add_argument("--screen", action="store_true", help="gold docs + a fixed sample only")
    parser.add_argument("--workers", type=int, default=EVAL_EMBED_WORKERS)
    parser.add_argument("--probe", type=int, default=0, help="embed N chunks, report speed")
    # Fusion: these change only the query step; a cached index is reused, the run is renamed.
    parser.add_argument("--bm25-weight", type=float, default=RRF_BM25_WEIGHT, help="RRF weight")
    parser.add_argument("--dense-weight", type=float, default=RRF_DENSE_WEIGHT, help="RRF weight")
    parser.add_argument("--rrf-k", type=int, default=RRF_K, help="RRF k (rank offset)")
    args = parser.parse_args(argv)
    # Checked here, before any index is built or embedded, not after. NaN and infinity pass
    # a `< 0` test but break the ranking (Codex review 2026-10-16 finding 1).
    if not all(math.isfinite(w) and w >= 0 for w in (args.bm25_weight, args.dense_weight)):
        parser.error("--bm25-weight and --dense-weight must be finite and >= 0")
    if args.rrf_k < 0:
        parser.error("--rrf-k must be >= 0")
    if args.fields and (args.no_header or args.production):
        parser.error("--fields needs a --chunking variant with its header (the production "
                     "index follows CHUNK_HEADER_FIELDS)")  # fmt: skip
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    args = parse_args(argv)
    settings = Settings.from_env()
    tracer = obs.init_tracing(settings)
    try:
        with tracer.trace(TRACE_NAME, input=vars(args)) as trace:
            logger.info("trace id %s", trace.trace_id)
            return run(args, settings)
    except LLMError as exc:
        logger.error("embedding stopped: %s (finished requests are cached; rerun to resume)", exc)
        return 1
    finally:
        tracer.shutdown(EVAL_TRACE_FLUSH_TIMEOUT_S)


if __name__ == "__main__":
    sys.exit(main())
