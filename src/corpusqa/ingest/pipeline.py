"""Ingest orchestrator: S1 discover, the cache decision, then (on a miss) S2-S5 per file
streamed into batched S7 writes, the corpus-wide date pass (S4_date_window), S8 profile,
keys and links, S6 chunks (after S8: chunk headers show the final display titles) and
S10 finalize. Then, on a hit or a miss alike, S9 embeds the chunks that have no vector
yet (`ingest.embed`) and the vectors are loaded into memory (S10_vectors).

`run(settings, embedder)` returns the `Runtime`: every stored doc id, read-only readers
of the index, the vectors and their coverage, and the embedder. The index lives under
`settings.index_dir`, in a folder named by the fingerprint of the corpus, code, schema
and build settings (`ingest.fingerprint`): a restart on the same corpus reuses it, any
change builds a new one. A build is published only when complete and checked; any
exception in S1-S8, S6 or S10 propagates, so start-up fails and /health stays 503. S9 is
the exception: it runs on the published index and never fails start-up (a partial embed
leaves the rest to keyword search, and the next start resumes it).

Tracing: one `ingest` span with a child span per stage, plus `index_cache` (hit or miss,
and why). S2-S5 and S7 run interleaved per document, so their spans are written after
the loop, one per stage, with the stage's counts and the seconds spent in it
(`metadata.seconds`); a span per document would flood the trace.

Dates: per document, S4 uses the fixed window. After every document is stored, the date
pass learns the window from the corpus's ISO-format dates, recomputes `date_in_range`,
re-picks created/updated (only changed rows are written) and counts, per source, the
dates outside the window. Each date is judged on its own: one outside the window counts
as no date, whatever the rest of its field holds.
"""

import contextlib
import itertools
import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from corpusqa import obs
from corpusqa.config import (
    CACHE_KEEP,
    CHUNK_WRITE_BATCH_ROWS,
    CONDITIONAL_MIN_STRENGTH,
    DATE_WINDOW_MIN_VALUES,
    HUB_KEY_MAX_DOCS,
    INGEST_PROGRESS_EVERY_FILES,
    LEXICON_MIN_COUNT,
    PROMOTE_PROSE_FIELDS,
    STORE_BATCH_DOCS,
    TRACE_MAX_LISTED_ITEMS,
    WEAK_TITLE_DISTINCT_RATIO,
    ConfigError,
    Settings,
)
from corpusqa.ingest.chunking import (
    ChunkSpec,
    chunk_rows,
    count_header_values,
    count_tokens,
    embedding_text,
    select_header_fields,
)
from corpusqa.ingest.clean import clean_body
from corpusqa.ingest.core import build_core
from corpusqa.ingest.dates import (
    DateWindow,
    OutsideCount,
    fixed_window,
    iso_year,
    learn_window,
    outside_counts,
    pick_dates,
)
from corpusqa.ingest.discover import discover
from corpusqa.ingest.embed import EmbedOptions, EmbedResult, embed_stage, load_vectors
from corpusqa.ingest.fields import NO_DATE_FLAG, build_fields
from corpusqa.ingest.fingerprint import (
    META_CORPUS_DIR,
    CacheDecision,
    Fingerprint,
    decide,
    fingerprint,
    mark_used,
    prune,
)
from corpusqa.ingest.links import Resolver, build_keys, build_links
from corpusqa.ingest.models import Discovery, DocDates, FieldsResult, FileRef
from corpusqa.ingest.parse import parse
from corpusqa.ingest.profile import (
    OWN_ID,
    PERSON,
    REFERENCE,
    Kind,
    classify,
    conditions,
    field_profile_rows,
    gather,
    learn_lexicon,
    reused_values,
    shared_titles,
    source_profile_rows,
    synonym_groups,
)
from corpusqa.ingest.startup import Runtime
from corpusqa.llm.interfaces import Embedder
from corpusqa.store.interfaces import (
    META_DATE_WINDOW_MAX_YEAR,
    META_DATE_WINDOW_MIN_YEAR,
    BuildWriter,
    ChunkRow,
    DocumentDates,
    DocumentRow,
    FieldValue,
    QuarantineRow,
    StageRow,
    StoreError,
    ValueNormUpdate,
)
from corpusqa.store.norms import person_norm
from corpusqa.store.sqlite import SqliteBuildWriter, SqliteReader

logger = logging.getLogger(__name__)

S1, S2, S3, S4, S5, S7 = "S1_discover", "S2_parse", "S3_core", "S4_fields", "S5_clean", "S7_store"
S4_DATES = "S4_date_window"
S8_PROFILE, S8_KEYS, S8_LINKS = "S8_profile", "S8_keys", "S8_links"
S6 = "S6_chunk"
S10 = "S10_finalize"
CACHE_SPAN = "index_cache"


@dataclass
class _Stage:
    """Counts and time of one per-document stage, summed over the corpus."""

    name: str
    counts: Counter[str] = field(default_factory=Counter)
    seconds: float = 0.0

    def timed[**P, R](self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - started

    def count_flags(self, flags: Iterable[str]) -> None:
        self.counts.update(flag.split(":", 1)[0] for flag in flags)


@dataclass(frozen=True, slots=True)
class _DocDateInputs:
    """One stored document's date inputs, kept for the corpus-wide date pass."""

    doc_id: str
    source: str
    stem: str
    candidates: tuple[tuple[str, str], ...]
    dates: DocDates  # as stored (fixed window)
    flags: tuple[str, ...]  # as stored


@dataclass
class _DateStats:
    """What the date pass needs, gathered while documents stream by (small: no text)."""

    iso_years: Counter[int] = field(default_factory=Counter)  # years of ISO-format values
    field_years: dict[tuple[str, str], Counter[int]] = field(default_factory=dict)
    docs: list[_DocDateInputs] = field(default_factory=list)

    def add(self, doc: DocumentRow, fields: FieldsResult) -> None:
        for row in doc.fields:
            if row.value_iso is None:
                continue
            year = iso_year(row.value_iso)
            self.field_years.setdefault((doc.source, row.field), Counter())[year] += 1
            if row.date_format == "iso":
                self.iso_years[year] += 1
        self.docs.append(
            _DocDateInputs(
                doc.doc_id, doc.source, doc.file_stem, fields.candidates, fields.dates, doc.flags
            )
        )


def run(
    settings: Settings,
    embedder: Embedder | None = None,
    *,
    embed_options: EmbedOptions | None = None,
) -> Runtime:
    """Reuse or build the index of `settings.corpus_dir`, embed what has no vector yet, and
    return the read-only Runtime.

    `embedder` is what the wiring point built (None: keyword-only search); the Runtime
    carries it so queries are embedded the same way. S9 stops sending requests
    `embed_options.budget_s` after this call starts (design §7).
    """
    corpus = settings.corpus_dir
    if not corpus.is_dir():
        raise ConfigError(f"CORPUS_DIR {corpus} is not a readable directory.")
    options = embed_options or EmbedOptions()
    deadline = options.clock() + options.budget_s
    started = time.monotonic()
    this_year = datetime.now(UTC).year
    with obs.span(
        "ingest",
        input={"corpus_dir": str(corpus)},
        metadata={
            "index_dir": str(settings.index_dir),
            "promote_prose_fields": _promote(),
            "embedder": embedder.model if embedder else None,
        },
    ) as span:
        found, s1_row = _discover(corpus)
        fp = fingerprint(found, _build_settings(this_year))
        decision = _cache(settings.index_dir, fp, corpus)
        if decision.hit:
            path, totals = decision.db_path, {}
            mark_used(path)
        else:
            path, totals = _build(corpus, found, s1_row, decision.db_path, fp, this_year)
        embedded = embed_stage(path, embedder, settings.embed_cache_db, options, deadline)
        runtime = _open(path, decision.hit, embedder)
        seconds = round(time.monotonic() - started, 3)
        known = runtime.known_doc_ids
        span.update(
            output={
                **totals,
                "cache": "hit" if decision.hit else "miss",
                "known_doc_ids": len(known),
                "index": str(runtime.index_path),
                **_embed_summary(embedded, runtime),
                "seconds": seconds,
            }
        )
    if not known:
        logger.warning("ingest: no documents in %s; every answer will find nothing", corpus)
    how = "reused" if decision.hit else "built"
    logger.info(
        "ingest: %d documents ready in %.1f s (%s %s); %d chunks, %.1f %% with vectors "
        "in memory", len(known), seconds, how, runtime.index_path, embedded.coverage.n_chunks,
        100 * runtime.embed_coverage,
    )  # fmt: skip
    return runtime


def _embed_summary(embedded: EmbedResult, runtime: Runtime) -> dict[str, object]:
    return {
        "chunks": embedded.coverage.n_chunks,
        "embedded": embedded.coverage.n_embedded,
        "embed_status": embedded.status,
        "embed_stop": embedded.stop,
        "vectors": len(runtime.vectors) if runtime.vectors is not None else 0,
        "embed_coverage": round(runtime.embed_coverage, 4),
    }


def _promote() -> bool:
    # Read at call time: the retrieval eval switches this module's flag for one build.
    return PROMOTE_PROSE_FIELDS


def _build_settings(this_year: int) -> dict[str, object]:
    """Build inputs that are not in the fingerprinted code files."""
    return {"promote_prose_fields": _promote(), "this_year": this_year}


def _discover(corpus: Path) -> tuple[Discovery, StageRow]:
    started, timer = _now(), time.perf_counter()
    _log_start(S1, str(corpus))
    with obs.span(S1, input={"corpus_dir": str(corpus)}) as span:
        found = discover(corpus)
        counts = {
            "files": len(found.files),
            "files_by_source": dict(sorted(Counter(f.source for f in found.files).items())),
            "skipped": found.skipped,
            "quarantined": dict(Counter(q.reason for q in found.quarantined)),
        }
        span.update(output={**counts, "set_aside": _listed(found.quarantined)})
    _log_done(S1, counts, time.perf_counter() - timer)
    return found, StageRow(S1, started, _now(), counts)


def _cache(index_dir: Path, fp: Fingerprint, corpus: Path) -> CacheDecision:
    with obs.span(
        CACHE_SPAN,
        input={"index_dir": str(index_dir), "fingerprint": fp.full},
        metadata={"keep": CACHE_KEEP, "parts": fp.meta()},
    ) as span:
        pruned = prune(index_dir, keep=fp.folder, corpus_dir=corpus)
        decision = decide(index_dir, fp, corpus)
        span.update(
            output={
                "cache": "hit" if decision.hit else "miss",
                "reason": decision.reason,
                "changed": list(decision.changed),
                "previous": decision.previous,
                "fingerprint": fp.full,
                "index": str(decision.db_path),
                "pruned": pruned,
            }
        )
    if decision.hit:
        logger.info("ingest: reusing the index built for this corpus (%s)", decision.db_path)
    else:
        changed = f": {', '.join(decision.changed)}" if decision.changed else ""
        logger.info("ingest: building a new index (%s%s)", decision.reason, changed)
    return decision


def _open(path: Path, cache_hit: bool, embedder: Embedder | None) -> Runtime:
    """The Runtime over a published index: readers, vectors in memory, the embedder."""
    reader = SqliteReader(path)  # read-only; refuses an incomplete or other-schema file
    chunks, vectors, coverage = load_vectors(path, embedder)
    return Runtime(
        known_doc_ids=reader.doc_ids(),
        store=reader,
        index_path=path,
        built_at=reader.meta().get("built_at"),
        cache_hit=cache_hit,
        chunks=chunks,
        vectors=vectors,
        embed_coverage=coverage,
        embedder=embedder,
    )


def _build(
    corpus: Path, found: Discovery, s1_row: StageRow, db_path: Path, fp: Fingerprint, this_year: int
) -> tuple[Path, dict[str, int]]:
    """S2-S8, S6 and S10 into `db_path`; nothing is left behind if any stage fails."""
    writer = SqliteBuildWriter(db_path)
    try:
        writer.add_quarantine(found.quarantined)
        writer.add_stage(s1_row)
        totals = _load(found, writer, this_year)
        totals |= _corpus_wide(writer)
        totals |= _chunk(writer)
        path = _finalize(writer, corpus, totals, fp)
    except BaseException:
        writer.abort()
        with contextlib.suppress(OSError):
            db_path.parent.rmdir()  # only when empty
        raise
    return path, totals


def _chunk(writer: BuildWriter, spec: ChunkSpec | None = None) -> dict[str, int]:
    """S6: every document's body cut by the chunking strategy step C chose (the default
    spec), written with the keyword index. Runs after S8: headers use final titles and,
    when the spec shows field values, the fields the profile picked per source."""
    spec = spec or ChunkSpec()
    started, stage_started = time.perf_counter(), _now()
    _log_start(S6)
    with obs.span(
        S6,
        metadata={"strategy": spec.strategy, "max_tokens": spec.limit_tokens,
                  "overlap_tokens": spec.overlap, "header": spec.header,
                  "header_fields": spec.header_fields},
    ) as span:  # fmt: skip
        header_fields = None
        if spec.header_fields:
            header_fields = select_header_fields(writer.field_profiles())
            listed = itertools.islice(header_fields.items(), TRACE_MAX_LISTED_ITEMS)
            span.update(metadata={
                "header_field_sources": {source: list(fields) for source, fields in listed},
                "header_field_count": sum(len(fields) for fields in header_fields.values()),
            })  # fmt: skip
        counts: Counter[str] = Counter()
        levels: Counter[str] = Counter()
        batch: list[ChunkRow] = []
        next_id = 1
        for doc in writer.document_texts(header_fields=header_fields):
            rows = chunk_rows(doc, spec, next_id)
            next_id += len(rows)
            counts["documents"] += 1
            counts["documents_without_chunks"] += not rows
            # Values shown in the headers (a document without chunks shows none).
            counts["header_field_values"] += count_header_values(doc.fields) if rows else 0
            for row in rows:
                levels[row.level] += 1
                counts["text_tokens"] += row.n_tokens
                counts["embed_tokens"] += count_tokens(embedding_text(row.header, row.text))
                counts["max_chunk_tokens"] = max(counts["max_chunk_tokens"], row.n_tokens)
            batch.extend(rows)
            if len(batch) >= CHUNK_WRITE_BATCH_ROWS:
                writer.add_chunks(batch)
                batch = []
        writer.add_chunks(batch)
        writer.finish_chunks()
        counts["chunks"] = next_id - 1
        output = {**counts, "levels": dict(levels.most_common())}
        span.update(output=output, metadata={"seconds": round(time.perf_counter() - started, 3)})
    _record(writer, StageRow(S6, stage_started, _now(), _timed(output, started)))
    return {"chunks": counts["chunks"]}


def _load(found: Discovery, writer: BuildWriter, this_year: int) -> dict[str, int]:
    """S2-S5 per file, batched S7 writes, then the corpus-wide date pass."""
    stages = {name: _Stage(name) for name in (S2, S3, S4, S5, S7)}
    loop_started = _now()
    kept: dict[str, str] = {}  # doc_id -> rel_path of the file that holds it
    batch: list[DocumentRow] = []
    quarantined: list[QuarantineRow] = []
    dates = _DateStats()
    window = fixed_window(this_year)
    total = len(found.files)
    _log_start(", ".join(stages), f"{total:,} files, read and stored together")
    for n, file in enumerate(found.files, start=1):
        result = _process(file, stages, kept, window, dates)
        if n % INGEST_PROGRESS_EVERY_FILES == 0:
            logger.info("ingest: %s of %s files read", f"{n:,}", f"{total:,}")
        if isinstance(result, QuarantineRow):
            quarantined.append(result)
            continue
        batch.append(result)
        if len(batch) >= STORE_BATCH_DOCS:
            _write(writer, batch, stages[S7])
            batch = []
    _write(writer, batch, stages[S7])
    writer.add_quarantine(quarantined)
    loop_finished = _now()
    for stage in stages.values():
        with obs.span(stage.name, metadata={"seconds": round(stage.seconds, 3)}) as span:
            output: dict[str, object] = dict(stage.counts)
            if stage.name == S2:
                output["set_aside"] = _listed(quarantined)
            span.update(output=output)
        counts = {**stage.counts, "seconds": round(stage.seconds, 3)}
        _record(writer, StageRow(stage.name, loop_started, loop_finished, counts))
    _date_pass(writer, dates, this_year)
    return {
        "files": len(found.files) + len(found.quarantined),
        "documents": stages[S7].counts["documents"],
        "quarantined": len(found.quarantined) + len(quarantined),
        "field_rows": stages[S7].counts["field_rows"],
    }


def _process(
    file: FileRef,
    stages: dict[str, _Stage],
    kept: dict[str, str],
    window: DateWindow,
    dates: _DateStats,
) -> DocumentRow | QuarantineRow:
    """S2-S5 for one file. A file that breaks a stage is set aside, never fatal."""
    s2, s3, s4, s5 = stages[S2], stages[S3], stages[S4], stages[S5]
    try:
        parsed = s2.timed(parse, file)
        if isinstance(parsed, QuarantineRow):
            s2.counts[f"quarantined:{parsed.reason}"] += 1
            return parsed
        if parsed.doc_id in kept:
            s2.counts["quarantined:duplicate_uuid"] += 1
            logger.warning(
                "ingest: %s repeats the id of %s; set aside", file.rel_path, kept[parsed.doc_id]
            )
            detail = f"same id as {kept[parsed.doc_id]} (kept)"
            return QuarantineRow(
                file.rel_path, file.source, "duplicate_uuid", detail, parsed.doc_id
            )
        s2.counts["parsed"] += 1
        s2.count_flags(parsed.flags)
        core = s3.timed(build_core, parsed)
        s3.count_flags(core.flags)
        s3.counts["units"] += len(core.units)
        fields = s4.timed(build_fields, parsed, core, window)
        s4.count_flags(fields.flags)
        s4.counts["rows"] += len(fields.rows)
        s4.counts["noise_rows"] += sum(row.is_noise for row in fields.rows)
        body = s5.timed(clean_body, core)
        s5.count_flags(body.flags)
        s5.counts["body_chars"] += len(body.body)
    except Exception as exc:
        logger.exception("ingest: %s could not be processed; set aside", file.rel_path)
        s2.counts["quarantined:process_error"] += 1
        return QuarantineRow(
            file.rel_path, file.source, "process_error", f"{type(exc).__name__}: {exc}"
        )
    kept[parsed.doc_id] = file.rel_path
    picked = fields.dates
    row = DocumentRow(
        doc_id=parsed.doc_id,
        source=file.source,
        rel_path=file.rel_path,
        file_stem=file.stem,
        title=core.title,
        title_field=core.title_field,
        title_is_weak=core.title_is_weak,
        display_title=core.display_title,
        body=body.body,
        body_fields=body.labels,
        created=picked.created,
        created_field=picked.created_field,
        created_rule=picked.created_rule,
        updated=picked.updated,
        updated_field=picked.updated_field,
        updated_rule=picked.updated_rule,
        content_hash=parsed.content_hash,
        size_bytes=file.size,
        mtime_ns=file.mtime_ns,
        flags=(*file.flags, *parsed.flags, *core.flags, *fields.flags, *body.flags),
        raw=parsed.raw,
        fields=fields.rows,
    )
    dates.add(row, fields)
    return row


def _date_pass(writer: BuildWriter, stats: _DateStats, this_year: int) -> None:
    """Learn the date window, re-pick created/updated (each date on its own: one outside
    the window counts as no date), and count the dates outside the window per source."""
    started, timer = _now(), time.perf_counter()
    _log_start(S4_DATES)
    with obs.span(
        S4_DATES,
        input={"iso_values": stats.iso_years.total(), "date_fields": len(stats.field_years)},
        metadata={"min_iso_values": DATE_WINDOW_MIN_VALUES},
    ) as span:
        window = learn_window(stats.iso_years, this_year)
        changed: list[DocumentDates] = []
        rules: Counter[str] = Counter()
        undated: Counter[str] = Counter()  # documents without a usable date, per source
        for doc in stats.docs:
            picked = pick_dates(doc.candidates, doc.stem, window)
            flags = _with_no_date(doc.flags, picked.created is None)
            rules[f"created_rule:{picked.created_rule}"] += 1
            rules[f"updated_rule:{picked.updated_rule}"] += 1
            undated[doc.source] += picked.created is None
            if picked != doc.dates or flags != doc.flags:
                changed.append(_document_dates(doc.doc_id, picked, flags))
        writer.set_date_window(window.min_year, window.max_year)
        writer.update_dates(changed)
        outside = outside_counts(stats.field_years, window)
        listed = _dates_outside(stats.field_years, outside, undated)
        writer.set_meta(
            {
                META_DATE_WINDOW_MIN_YEAR: str(window.min_year),
                META_DATE_WINDOW_MAX_YEAR: str(window.max_year),
                "date_window_rule": window.rule,
                "date_window_iso_values": str(window.iso_values),
                "date_outside_window": json.dumps(listed, ensure_ascii=False),
            }
        )
        counts: dict[str, object] = {
            "window": [window.min_year, window.max_year],
            "window_rule": window.rule,
            "iso_values": window.iso_values,
            "date_values": sum(years.total() for years in stats.field_years.values()),
            "dates_outside": sum(c.outside for c in outside.values()),
            "sources_with_dates_outside": len(listed),
            "documents_changed": len(changed),
            "documents_without_date": rules["created_rule:None"],
            **{rule: n for rule, n in sorted(rules.items()) if not rule.endswith(":None")},
        }
        span.update(output={**counts, "outside": listed[:TRACE_MAX_LISTED_ITEMS]})
    _record(writer, StageRow(S4_DATES, started, _now(), counts), time.perf_counter() - timer)


def _dates_outside(
    field_years: Mapping[tuple[str, str], Counter[int]],
    outside: Mapping[tuple[str, str], OutsideCount],
    undated: Mapping[str, int],
) -> list[dict[str, object]]:
    """Per source with a date outside the window: its date values, how many are outside
    (each counts as no date), its documents without a usable date, and the fields (values,
    outside) that hold them. In source, then field order."""
    by_source: dict[str, list[tuple[str, int, int]]] = {}
    for (source, name), c in sorted(outside.items()):
        by_source.setdefault(source, []).append((name, c.values, c.outside))
    return [
        {
            "source": source,
            "values": sum(years.total() for (s, _), years in field_years.items() if s == source),
            "outside": sum(n for _, _, n in fields),
            "documents_without_date": undated.get(source, 0),
            "fields": [
                {"field": name, "values": values, "outside": n} for name, values, n in fields
            ],
        }
        for source, fields in by_source.items()
    ]


def _corpus_wide(writer: BuildWriter) -> dict[str, int]:
    """S8: profile, keys and links, from the stored rows (design.md S8)."""
    started = time.perf_counter()
    stage_started = _now()
    _log_start(S8_PROFILE)
    with obs.span(
        S8_PROFILE,
        metadata={
            "lexicon_min_count": LEXICON_MIN_COUNT,
            "conditional_min_strength": CONDITIONAL_MIN_STRENGTH,
            "weak_title_distinct_ratio": WEAK_TITLE_DISTINCT_RATIO,
        },
    ) as span:
        docs = list(writer.doc_facts())
        stats = gather(writer.field_values(), docs)
        source_docs = Counter(d.source for d in docs)
        lexicon = learn_lexicon(stats.values())
        reused = reused_values(stats.values())
        kinds = {
            key: classify(st, source_docs[st.source], lexicon, reused) for key, st in stats.items()
        }
        groups = synonym_groups(stats, kinds)
        conds = conditions(stats, kinds)
        writer.add_field_profiles(field_profile_rows(stats, kinds, groups, source_docs), conds)
        titles = shared_titles(docs)
        writer.update_titles(titles)
        rows = _rows_of_kinds(writer.field_values(), kinds, (PERSON, OWN_ID, REFERENCE))
        norms = _person_norms(rows[PERSON])
        writer.update_value_norms(norms)
        writer.set_meta(
            {"name_lexicon": json.dumps(sorted(lexicon), ensure_ascii=False),
             "name_lexicon_size": str(len(lexicon))}
        )  # fmt: skip
        by_source: dict[str, Counter[str]] = {}
        for (source, _), kind in kinds.items():
            by_source.setdefault(source, Counter())[kind.kind] += 1
        profile_counts: dict[str, object] = {
            "fields": len(stats),
            "lexicon": len(lexicon),
            "synonym_groups": len(set(groups.values())),
            "fields_in_synonym_groups": len(groups),
            "conditions": len(conds),
            "weak_titles_added": len(titles),
            "person_values_renormalised": len(norms),
        }
        span.update(
            output={
                **profile_counts,
                "kinds": {s: dict(sorted(c.items())) for s, c in sorted(by_source.items())},
                "strongest_conditions": [
                    {"source": c.source, "field": c.field, "given": c.given_field,
                     "strength": c.strength, "present_when": list(c.present_when)}
                    for c in sorted(conds, key=lambda c: -c.strength)[:TRACE_MAX_LISTED_ITEMS]
                ],
            },
            metadata={"seconds": round(time.perf_counter() - started, 3)},
        )  # fmt: skip
    _record(writer, StageRow(S8_PROFILE, stage_started, _now(), _timed(profile_counts, started)))

    started, stage_started = time.perf_counter(), _now()
    _log_start(S8_KEYS)
    with obs.span(S8_KEYS, metadata={"hub_key_max_docs": HUB_KEY_MAX_DOCS}) as span:
        docs = list(writer.doc_facts())  # with the weak titles S8_profile found
        keys = build_keys(docs, rows[OWN_ID])
        writer.add_doc_keys(keys.rows)
        writer.add_source_profiles(source_profile_rows(docs, kinds, keys.prefixes))
        own_ids: dict[str, list[str]] = {}
        for source, name in sorted(k for k, kind in kinds.items() if kind.kind == OWN_ID):
            own_ids.setdefault(source, []).append(name)
        key_counts: dict[str, object] = {
            "keys": len(keys.rows),
            **_prefixed("kind", Counter(_key_kind(r.kind) for r in keys.rows)),
            "hub_keys": len({r.key_norm for r in keys.rows if r.is_hub}),
            "documents_with_a_key": len({r.doc_id for r in keys.rows}),
            "prefixes": sum(len(p) for p in keys.prefixes.values()),
        }
        span.update(
            output={**key_counts, "own_id_fields": own_ids,
                    "prefixes_by_source": {s: list(p) for s, p in keys.prefixes.items()}},
            metadata={"seconds": round(time.perf_counter() - started, 3)},
        )  # fmt: skip
    _record(writer, StageRow(S8_KEYS, stage_started, _now(), _timed(key_counts, started)))

    started, stage_started = time.perf_counter(), _now()
    _log_start(S8_LINKS)
    with obs.span(S8_LINKS) as span:
        links = build_links(rows[REFERENCE], Resolver(keys.rows, docs))
        writer.add_links(links)
        values = {(link.doc_id, link.field, link.raw) for link in links}
        resolved = {(link.doc_id, link.field, link.raw) for link in links if link.target_doc_id}
        link_counts: dict[str, object] = {
            "reference_fields": sum(1 for k in kinds.values() if k.kind == REFERENCE),
            "values": len(values),
            "values_resolved": len(resolved),
            "rows": len(links),
            **_prefixed("method", Counter(r.method for r in links)),
            "hub_matches": sum(r.target_is_hub for r in links),
        }
        span.update(
            output=link_counts, metadata={"seconds": round(time.perf_counter() - started, 3)}
        )
    _record(writer, StageRow(S8_LINKS, stage_started, _now(), _timed(link_counts, started)))
    return {"doc_keys": len(keys.rows), "links": len(links)}


def _rows_of_kinds(
    values: Iterable[FieldValue], kinds: Mapping[tuple[str, str], Kind], wanted: Iterable[str]
) -> dict[str, list[FieldValue]]:
    """The rows of every field whose kind is one of `wanted`, grouped by kind."""
    rows: dict[str, list[FieldValue]] = {kind: [] for kind in wanted}
    for row in values:
        kind = kinds[(row.source, row.field)].kind
        if kind in rows:
            rows[kind].append(row)
    return rows


def _person_norms(rows: Iterable[FieldValue]) -> list[ValueNormUpdate]:
    """People's names get their own matching form (role and <email> dropped, `.` as space)."""
    updates = []
    for row in rows:
        if row.is_noise or row.value is None or row.value_type not in ("str", "int", "float"):
            continue
        norm = person_norm(row.value)
        if norm != row.value_norm:
            updates.append(ValueNormUpdate(row.doc_id, row.field, row.seq, norm))
    return updates


def _timed(counts: dict[str, object], started: float) -> dict[str, object]:
    """Stage counts plus the seconds since `started` (perf_counter), for ingest_stages."""
    return {**counts, "seconds": round(time.perf_counter() - started, 3)}


def _log_start(stages: str, detail: str = "") -> None:
    """The start-up log's line for stages that start now (the terminal's sign of progress)."""
    logger.info("ingest: %s: started%s", stages, f" ({detail})" if detail else "")


def _log_done(stage: str, counts: Mapping[str, object], seconds: float | None = None) -> None:
    """The start-up log's line for a finished stage: its time and its whole-number counts
    (the stage's span in the trace has the rest)."""
    took = counts.get("seconds", seconds)
    shown = ", ".join(
        f"{name}={value:,}"
        for name, value in counts.items()
        if isinstance(value, int) and not isinstance(value, bool)
    )
    logger.info(
        "ingest: %s: done%s%s",
        stage,
        f" in {took:.1f} s" if isinstance(took, int | float) else "",
        f" ({shown})" if shown else "",
    )


def _record(writer: BuildWriter, row: StageRow, seconds: float | None = None) -> None:
    """Store a finished stage's row and log its line."""
    writer.add_stage(row)
    _log_done(row.stage, row.counts, seconds)


def _key_kind(kind: str) -> str:
    return kind.split(":", 1)[0]


def _prefixed(prefix: str, counts: Counter[str]) -> dict[str, int]:
    return {f"{prefix}:{name}": n for name, n in sorted(counts.items())}


def _finalize(writer: BuildWriter, corpus: Path, totals: dict[str, int], fp: Fingerprint) -> Path:
    """S10: check the build, write its facts, publish it, check it opens read-only. The
    vectors come after (S9 on the published index), so their meta says `pending` here."""
    started, stage_started = time.perf_counter(), _now()
    _log_start(S10)
    with obs.span(S10, metadata={"fingerprint": fp.full}) as span:
        integrity = writer.integrity()
        if not integrity.ok:
            raise StoreError(
                f"index failed its checks: quick_check={integrity.quick_check!r}, "
                f"foreign key violations={integrity.foreign_key_violations}"
            )
        for table in ("documents", "chunks"):
            if integrity.table_rows[table] != totals[table]:
                raise StoreError(
                    f"index holds {integrity.table_rows[table]} {table}, "
                    f"the build wrote {totals[table]}"
                )
        counts = {f"n_{table}": n for table, n in integrity.table_rows.items()}
        writer.set_meta(
            {**_meta(corpus, totals), **fp.meta(), **{k: str(v) for k, v in counts.items()},
             "embed_status": "pending"}
        )  # fmt: skip
        writer.add_stage(StageRow(S10, stage_started, _now(), _timed(dict(counts), started)))
        path = writer.finish()
        reader = SqliteReader(path)  # read-only; refuses an incomplete or other-schema file
        known = reader.doc_ids()
        span.update(
            output={
                **counts,
                "quick_check": integrity.quick_check,
                "known_doc_ids": len(known),
                "index": str(path),
                "size_bytes": path.stat().st_size,
            },
            metadata={"seconds": round(time.perf_counter() - started, 3)},
        )
    _log_done(S10, counts, time.perf_counter() - started)
    return path


def _with_no_date(flags: tuple[str, ...], missing: bool) -> tuple[str, ...]:
    if missing == (NO_DATE_FLAG in flags):
        return flags
    return (*flags, NO_DATE_FLAG) if missing else tuple(f for f in flags if f != NO_DATE_FLAG)


def _document_dates(doc_id: str, d: DocDates, flags: tuple[str, ...]) -> DocumentDates:
    return DocumentDates(
        doc_id=doc_id,
        created=d.created,
        created_field=d.created_field,
        created_rule=d.created_rule,
        updated=d.updated,
        updated_field=d.updated_field,
        updated_rule=d.updated_rule,
        flags=flags,
    )


def _write(writer: BuildWriter, batch: list[DocumentRow], stage: _Stage) -> None:
    if not batch:
        return
    stage.timed(writer.add_documents, batch)
    stage.counts["batches"] += 1
    stage.counts["documents"] += len(batch)
    stage.counts["field_rows"] += sum(len(doc.fields) for doc in batch)


def _listed(rows: Iterable[QuarantineRow]) -> list[dict[str, str | None]]:
    """Files set aside, for a span: path, reason and detail (capped)."""
    listed = [{"path": r.rel_path, "reason": r.reason, "detail": r.detail} for r in rows]
    return listed[:TRACE_MAX_LISTED_ITEMS]


def _meta(corpus: Path, totals: dict[str, int]) -> dict[str, str]:
    return {
        "corpus_dir_name": corpus.name,
        META_CORPUS_DIR: str(corpus),
        "built_at": _now(),
        "promote_prose_fields": str(_promote()).lower(),
        "n_files": str(totals["files"]),
        "n_documents": str(totals["documents"]),
        "n_quarantined": str(totals["quarantined"]),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
