"""SQLite implementation of the store: the build writer and read-only access.

The writer builds `<db>.building` with one connection in WAL mode, one transaction per
batch, and renames it to `<db>` only in `finish`, so a crash never leaves a database
that looks complete. The corpus-wide stage (S8) and chunking (S6) read the rows back
through the same connection. Readers open the published file read-only, one connection
each; `read_meta` reads the build facts of any index file without judging it (the cache).

Vectors (S9) are added to the published index by `SqliteEmbeddingWriter`, one
transaction per request, so every finished request survives a crash and a later start
resumes from there. `SqliteVectorCache` keeps vectors by text hash outside any index.

The chunking eval: `SqliteChunkWriter` builds a file holding only the chunk tables (one
per variant) the same way; `SqliteChunkReader` reads the chunk tables of any published
file that has them (the eval's files and the document index).
"""

import json
import re
import sqlite3
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence, Set
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from corpusqa.config import (
    BM25_HEADER_WEIGHT,
    BM25_TEXT_WEIGHT,
    BUILDING_SUFFIX,
    CHUNK_READ_PAGE_DOCS,
    SCHEMA_VERSION,
    SEARCH_EXACT_PHRASES,
    SQL_KEYS_PER_QUERY,
)
from corpusqa.store.interfaces import (
    META_DATE_WINDOW_MAX_YEAR,
    META_DATE_WINDOW_MIN_YEAR,
    ChunkReader,
    ChunkRow,
    ChunkText,
    CorpusReader,
    DocFacts,
    DocKeyRow,
    DocSummary,
    DocumentDates,
    DocumentRow,
    DocumentText,
    EmbedCoverage,
    EmbeddingRow,
    FieldConditionRow,
    FieldCoverage,
    FieldMatch,
    FieldProfileRow,
    FieldRow,
    FieldText,
    FieldValue,
    Integrity,
    LinkRow,
    QuarantineRow,
    SourceDates,
    SourceProfileRow,
    StageRow,
    StoredDocument,
    StoreError,
    TitleUpdate,
    ValueCount,
    ValueNormUpdate,
)
from corpusqa.store.norms import fold_key
from corpusqa.store.schema import BUILD_PRAGMAS, CHUNK_DDL, DDL, VECTOR_CACHE_DDL

STATUS_BUILDING = "building"
STATUS_COMPLETE = "complete"
# SQLite's side files next to a WAL database.
_SIDE_SUFFIXES = ("-wal", "-shm")

_INSERT_DOCUMENT = """
INSERT INTO documents (doc_id, source, rel_path, file_stem, title, title_field,
  title_is_weak, display_title, body, body_fields, created, created_field, created_rule,
  updated, updated_field, updated_rule, content_hash, size_bytes, mtime_ns, flags, raw)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_FIELD = """
INSERT INTO fields (doc_id, source, field, field_norm, seq, role, value, value_type,
  value_norm, value_num, value_iso, date_precision, date_in_range, date_format, in_list,
  is_noise)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
# The year of value_iso is the text before its first dash.
_SET_DATE_IN_RANGE = """
UPDATE fields
SET date_in_range = CAST(substr(value_iso, 1, instr(value_iso, '-') - 1) AS INTEGER)
  BETWEEN ? AND ?
WHERE value_iso IS NOT NULL
"""
_UPDATE_DATES = """
UPDATE documents SET created = ?, created_field = ?, created_rule = ?, updated = ?,
  updated_field = ?, updated_rule = ?, flags = ?
WHERE doc_id = ?
"""
_SELECT_FIELD_VALUES = """
SELECT source, field, field_norm, doc_id, seq, role, value, value_type, value_norm,
  value_num, value_iso, in_list, is_noise
FROM fields ORDER BY source, field, doc_id, seq
"""
_SELECT_DOC_FACTS = """
SELECT doc_id, source, rel_path, file_stem, title, title_field, title_is_weak, display_title,
  body_fields, flags
FROM documents ORDER BY rowid
"""
_INSERT_FIELD_PROFILE = """
INSERT INTO field_profile (source, field, field_norm, synonym_group, n_docs, presence,
  n_values, n_distinct, list_ratio, type_mix, kind, kind_confidence, title_ratio,
  content_ratio, noise_ratio, avg_len, max_len, top_values)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_SOURCE_PROFILE = """
INSERT INTO source_profile (source, n_docs, title_fields, content_fields, own_id_fields,
  key_prefixes, date_fields, example_doc_ids, path_depths, stem_title_match)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_LINK = """
INSERT INTO links (doc_id, field, raw, key_norm, target_doc_id, method, target_is_hub)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""
# Tables whose row counts S10 reports (and checks against what the build wrote).
_COUNTED_TABLES = (
    "documents", "fields", "quarantine", "field_profile", "field_conditions",
    "source_profile", "doc_keys", "links", "chunks",
)  # fmt: skip
# Each document's chunk count, after the last chunk is in (S6).
_SET_N_CHUNKS = """
UPDATE documents SET n_chunks = (SELECT count(*) FROM chunks WHERE chunks.doc_id = documents.doc_id)
"""


def building_path(path: Path) -> Path:
    """Where the build of `path` is written until it is complete."""
    return path.with_name(path.name + BUILDING_SUFFIX)


def _remove_side_files(path: Path) -> None:
    for suffix in _SIDE_SUFFIXES:
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def _start_build(path: Path) -> Path:
    """The `.building` file for `path`, with any stale one from a crashed build removed."""
    building = building_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    building.unlink(missing_ok=True)
    _remove_side_files(building)
    return building


def _publish(conn: sqlite3.Connection, building: Path, path: Path) -> Path:
    """Close the build connection and rename the finished file into place."""
    # Fold the WAL into the main file so the rename moves the whole database.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    _remove_side_files(building)
    # Side files of the previous file belong to the file being replaced.
    _remove_side_files(path)
    building.replace(path)
    return path


def _discard(conn: sqlite3.Connection, building: Path) -> None:
    conn.close()
    building.unlink(missing_ok=True)
    _remove_side_files(building)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _document_params(doc: DocumentRow) -> tuple[object, ...]:
    return (
        doc.doc_id,
        doc.source,
        doc.rel_path,
        doc.file_stem,
        doc.title,
        doc.title_field,
        int(doc.title_is_weak),
        doc.display_title,
        doc.body,
        _json(list(doc.body_fields)),
        doc.created,
        doc.created_field,
        doc.created_rule,
        doc.updated,
        doc.updated_field,
        doc.updated_rule,
        doc.content_hash,
        doc.size_bytes,
        doc.mtime_ns,
        _json(list(doc.flags)),
        zlib.compress(doc.raw),
    )


def _field_params(doc: DocumentRow, row: FieldRow) -> tuple[object, ...]:
    in_range = None if row.date_in_range is None else int(row.date_in_range)
    return (
        doc.doc_id,
        doc.source,
        row.field,
        row.field_norm,
        row.seq,
        row.role,
        row.value,
        row.value_type,
        row.value_norm,
        row.value_num,
        row.value_iso,
        row.date_precision,
        in_range,
        row.date_format,
        int(row.in_list),
        int(row.is_noise),
    )


def _field_profile_params(r: FieldProfileRow) -> tuple[object, ...]:
    return (
        r.source,
        r.field,
        r.field_norm,
        r.synonym_group,
        r.n_docs,
        r.presence,
        r.n_values,
        r.n_distinct,
        r.list_ratio,
        _json(dict(r.type_mix)),
        r.kind,
        r.kind_confidence,
        r.title_ratio,
        r.content_ratio,
        r.noise_ratio,
        r.avg_len,
        r.max_len,
        _json([list(item) for item in r.top_values]),
    )


def _source_profile_params(r: SourceProfileRow) -> tuple[object, ...]:
    return (
        r.source,
        r.n_docs,
        _json(dict(r.title_fields)),
        _json(dict(r.content_fields)),
        _json(list(r.own_id_fields)),
        _json(list(r.key_prefixes)),
        _json(list(r.date_fields)),
        _json(list(r.example_doc_ids)),
        _json(dict(r.path_depths)),
        r.stem_title_match,
    )


class SqliteBuildWriter:
    """Builds a fresh index at `path` (see the module docstring). One thread only."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # A crashed earlier build leaves a stale file: start from nothing.
        self.building = _start_build(path)
        self._conn = sqlite3.connect(self.building)
        self._conn.executescript(BUILD_PRAGMAS + DDL + CHUNK_DDL)
        self.set_meta({"schema_version": str(SCHEMA_VERSION), "status": STATUS_BUILDING})

    def add_documents(self, docs: Sequence[DocumentRow]) -> None:
        with self._conn:
            self._conn.executemany(_INSERT_DOCUMENT, [_document_params(d) for d in docs])
            self._conn.executemany(
                _INSERT_FIELD, [_field_params(d, row) for d in docs for row in d.fields]
            )

    def set_date_window(self, min_year: int, max_year: int) -> None:
        with self._conn:
            self._conn.execute(_SET_DATE_IN_RANGE, (min_year, max_year))

    def update_dates(self, rows: Sequence[DocumentDates]) -> None:
        with self._conn:
            self._conn.executemany(
                _UPDATE_DATES,
                [
                    (
                        r.created,
                        r.created_field,
                        r.created_rule,
                        r.updated,
                        r.updated_field,
                        r.updated_rule,
                        _json(list(r.flags)),
                        r.doc_id,
                    )
                    for r in rows
                ],
            )

    def add_quarantine(self, rows: Sequence[QuarantineRow]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO quarantine (rel_path, source, reason, detail, doc_id) "
                "VALUES (?, ?, ?, ?, ?)",
                [(r.rel_path, r.source, r.reason, r.detail, r.doc_id) for r in rows],
            )

    def add_stage(self, row: StageRow) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO ingest_stages (stage, started_at, finished_at, counts) "
                "VALUES (?, ?, ?, ?)",
                (row.stage, row.started_at, row.finished_at, _json(dict(row.counts))),
            )

    def set_meta(self, items: Mapping[str, str]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", list(items.items())
            )

    # ------------------------------------------------------------ corpus-wide stage (S8)
    def field_values(self) -> Iterator[FieldValue]:
        for row in self._conn.execute(_SELECT_FIELD_VALUES):
            (source, field, field_norm, doc_id, seq, role, value, value_type, value_norm,
             value_num, value_iso, in_list, is_noise) = row  # fmt: skip
            yield FieldValue(
                source=source,
                field=field,
                field_norm=field_norm,
                doc_id=doc_id,
                seq=seq,
                role=role,
                value=value,
                value_type=value_type,
                value_norm=value_norm,
                value_num=value_num,
                value_iso=value_iso,
                in_list=bool(in_list),
                is_noise=bool(is_noise),
            )

    def doc_facts(self) -> Iterator[DocFacts]:
        for row in self._conn.execute(_SELECT_DOC_FACTS):
            (doc_id, source, rel_path, file_stem, title, title_field, title_is_weak,
             display_title, body_fields, flags) = row  # fmt: skip
            yield DocFacts(
                doc_id=doc_id,
                source=source,
                rel_path=rel_path,
                file_stem=file_stem,
                title=title,
                title_field=title_field,
                title_is_weak=bool(title_is_weak),
                display_title=display_title,
                body_fields=tuple(json.loads(body_fields)),
                flags=tuple(json.loads(flags)),
            )

    def update_titles(self, rows: Sequence[TitleUpdate]) -> None:
        with self._conn:
            self._conn.executemany(
                "UPDATE documents SET title_is_weak = 1, display_title = ?, flags = ? "
                "WHERE doc_id = ?",
                [(r.display_title, _json(list(r.flags)), r.doc_id) for r in rows],
            )

    def update_value_norms(self, rows: Sequence[ValueNormUpdate]) -> None:
        with self._conn:
            self._conn.executemany(
                "UPDATE fields SET value_norm = ? WHERE doc_id = ? AND field = ? AND seq = ?",
                [(r.value_norm, r.doc_id, r.field, r.seq) for r in rows],
            )

    def add_field_profiles(
        self, fields: Sequence[FieldProfileRow], conditions: Sequence[FieldConditionRow]
    ) -> None:
        with self._conn:
            self._conn.executemany(
                _INSERT_FIELD_PROFILE, [_field_profile_params(r) for r in fields]
            )
            self._conn.executemany(
                "INSERT INTO field_conditions (source, field, given_field, present_when, "
                "strength) VALUES (?, ?, ?, ?, ?)",
                [(r.source, r.field, r.given_field, _json(list(r.present_when)), r.strength)
                 for r in conditions],
            )  # fmt: skip

    def add_source_profiles(self, rows: Sequence[SourceProfileRow]) -> None:
        with self._conn:
            self._conn.executemany(
                _INSERT_SOURCE_PROFILE, [_source_profile_params(r) for r in rows]
            )

    def add_doc_keys(self, rows: Sequence[DocKeyRow]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO doc_keys (doc_id, key_norm, kind, is_hub) VALUES (?, ?, ?, ?)",
                [(r.doc_id, r.key_norm, r.kind, int(r.is_hub)) for r in rows],
            )

    def add_links(self, rows: Sequence[LinkRow]) -> None:
        with self._conn:
            self._conn.executemany(
                _INSERT_LINK,
                [(r.doc_id, r.field, r.raw, r.key_norm, r.target_doc_id, r.method,
                  int(r.target_is_hub)) for r in rows],
            )  # fmt: skip

    # ------------------------------------------------------------ chunks (S6)
    def field_profiles(self) -> list[FieldProfileRow]:
        return [_field_profile_row(row) for row in self._conn.execute(_SELECT_FIELD_PROFILES)]

    def document_texts(
        self, header_fields: Mapping[str, Sequence[str]] | None = None
    ) -> Iterator[DocumentText]:
        return _document_texts(self._conn, header_fields)

    def add_chunks(self, rows: Sequence[ChunkRow]) -> None:
        insert_chunks(self._conn, rows)

    def finish_chunks(self) -> None:
        rebuild_chunk_fts(self._conn)
        with self._conn:
            self._conn.execute(_SET_N_CHUNKS)

    # ------------------------------------------------------------ finalize (S10)
    def integrity(self) -> Integrity:
        with self._conn:
            quick = [row[0] for row in self._conn.execute("PRAGMA quick_check")]
            violations = len(self._conn.execute("PRAGMA foreign_key_check").fetchall())
            counts = {
                table: self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608
                for table in _COUNTED_TABLES
            }
        return Integrity(
            quick_check="ok" if quick == ["ok"] else "; ".join(quick),
            foreign_key_violations=violations,
            table_rows=counts,
        )

    def finish(self) -> Path:
        self.set_meta({"status": STATUS_COMPLETE})
        return _publish(self._conn, self.building, self.path)

    def abort(self) -> None:
        _discard(self._conn, self.building)


def open_readonly(path: Path) -> sqlite3.Connection:
    """A new read-only connection to a published index (one per request or task)."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def read_meta(path: Path) -> dict[str, str] | None:
    """The meta table of the index file at `path`, whatever its state; None when there is
    no such file or it cannot be read as an index."""
    if not path.is_file():
        return None
    try:
        with closing(open_readonly(path)) as conn:
            return dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        return None


class SqliteReader:
    """Read access to a published, complete index. Each call uses its own connection."""

    def __init__(self, path: Path) -> None:
        if path.name.endswith(BUILDING_SUFFIX):
            raise StoreError(f"{path} is an unfinished build; it is never opened for reading")
        if not path.is_file():
            raise StoreError(f"no index at {path}")
        self.path = path
        with self._connect() as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        if meta.get("status") != STATUS_COMPLETE:
            raise StoreError(f"index {path} is not complete (status={meta.get('status')!r})")
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise StoreError(
                f"index {path} has schema {meta.get('schema_version')!r}, "
                f"this code needs {SCHEMA_VERSION}"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(open_readonly(self.path)) as conn:
            yield conn

    def meta(self) -> dict[str, str]:
        with self._connect() as conn:
            return dict(conn.execute("SELECT key, value FROM meta").fetchall())

    def doc_ids(self) -> frozenset[str]:
        with self._connect() as conn:
            return frozenset(row[0] for row in conn.execute("SELECT doc_id FROM documents"))

    def field_profiles(self) -> list[FieldProfileRow]:
        with self._connect() as conn:
            return [_field_profile_row(row) for row in conn.execute(_SELECT_FIELD_PROFILES)]

    def document_texts(
        self, header_fields: Mapping[str, Sequence[str]] | None = None
    ) -> Iterator[DocumentText]:
        with self._connect() as conn:
            yield from _document_texts(conn, header_fields)

    # ------------------------------------------------------------ the agent's tools
    def field_conditions(self) -> list[FieldConditionRow]:
        with self._connect() as conn:
            return [
                FieldConditionRow(source, field, given, tuple(json.loads(when)), strength)
                for source, field, given, when, strength in conn.execute(_SELECT_CONDITIONS)
            ]

    def source_dates(self) -> list[SourceDates]:
        with self._connect() as conn:
            rows = conn.execute(_SOURCE_DATES).fetchall()
        return [SourceDates(*row) for row in rows]

    def date_window(self) -> tuple[int, int] | None:
        meta = self.meta()
        low, high = meta.get(META_DATE_WINDOW_MIN_YEAR), meta.get(META_DATE_WINDOW_MAX_YEAR)
        if low is None or high is None:
            return None
        return int(low), int(high)

    def document_summaries(self, doc_ids: Sequence[str] | None = None) -> list[DocSummary]:
        with self._connect() as conn:
            if doc_ids is None:
                query = f"SELECT {_SUMMARY_COLUMNS} FROM documents ORDER BY rowid"  # noqa: S608
                return [_summary(row) for row in conn.execute(query)]
            found: dict[str, DocSummary] = {}
            for part in _parts(doc_ids):
                query = f"SELECT {_SUMMARY_COLUMNS} FROM documents WHERE doc_id IN ({_marks(part)})"  # noqa: S608
                found.update((row[0], _summary(row)) for row in conn.execute(query, part))
        return [found[doc_id] for doc_id in dict.fromkeys(doc_ids) if doc_id in found]

    def document(self, doc_id: str) -> StoredDocument | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_DOCUMENT, (doc_id,)).fetchone()
            if row is None:
                return None
            fields = _field_rows(conn, [doc_id], None)
        title, title_field, body, flags = row[_SUMMARY_WIDTH:]
        return StoredDocument(
            summary=_summary(row[:_SUMMARY_WIDTH]),
            title=title,
            title_field=title_field,
            body=body,
            flags=tuple(json.loads(flags)),
            fields=tuple(fields),
        )

    def value_counts(self, source: str, field: str) -> list[ValueCount]:
        docs: dict[str, set[str]] = {}
        spellings: dict[str, Counter[str]] = {}
        with self._connect() as conn:
            for value_norm, value, doc_id in conn.execute(
                _FIELD_VALUE_ROWS, (source, fold_key(field), field)
            ):
                docs.setdefault(value_norm, set()).add(doc_id)
                spellings.setdefault(value_norm, Counter())[value] += 1
        counts = [
            ValueCount(
                value_norm=norm,
                n_docs=len(ids),
                spellings=tuple(sorted(spellings[norm].items(), key=lambda s: (-s[1], s[0]))),
            )
            for norm, ids in docs.items()
        ]
        return sorted(counts, key=lambda c: (-c.n_docs, c.value_norm))

    def field_coverage(self, source: str, field: str) -> FieldCoverage:
        rows_of = (source, fold_key(field), field)
        with self._connect() as conn:
            [n_docs] = conn.execute(_COUNT_SOURCE_DOCS, (source,)).fetchone()
            per_doc = conn.execute(_COVERAGE_ROWS, rows_of).fetchall()
            [in_body] = conn.execute(_COVERAGE_IN_BODY, (source, field, *rows_of)).fetchone()
            [outside] = conn.execute(_COVERAGE_DATES_OUTSIDE, rows_of).fetchone()
        with_value = sum(1 for _, valued in per_doc if valued)
        return FieldCoverage(
            source_docs=n_docs,
            with_value=with_value,
            empty=len(per_doc) - with_value,
            in_body=in_body,
            dates_outside=outside,
        )

    def filter_doc_ids(
        self,
        source: str,
        matches: Sequence[FieldMatch],
        created_from: str | None = None,
        created_before: str | None = None,
    ) -> list[str]:
        where: list[str] = ["d.source = ?"]
        params: list[object] = [source]
        if created_from is not None or created_before is not None:
            # A created date is always inside the window (one outside counts as no date).
            where.append("d.created IS NOT NULL")
            if created_from is not None:
                where.append("d.created >= ?")
                params.append(created_from)
            if created_before is not None:
                where.append("d.created < ?")
                params.append(created_before)
        for match in matches:
            clause, more = _match_clause(source, match)
            where.append(clause)
            params.extend(more)
        query = (
            "SELECT d.doc_id FROM documents d WHERE "  # noqa: S608
            + " AND ".join(f"({clause})" for clause in where)
            + " ORDER BY d.created IS NULL, d.created DESC, d.doc_id"
        )
        with self._connect() as conn:
            return [row[0] for row in conn.execute(query, params)]

    def document_fields(
        self, doc_ids: Sequence[str], fields: Sequence[str] | None = None
    ) -> list[FieldValue]:
        with self._connect() as conn:
            return _field_rows(conn, doc_ids, fields)

    def key_owners(self, key_norm: str) -> list[DocKeyRow]:
        with self._connect() as conn:
            rows = conn.execute(_KEY_OWNERS, (key_norm,)).fetchall()
        return [DocKeyRow(doc_id, key, kind, bool(hub)) for doc_id, key, kind, hub in rows]

    def links_to(self, key_norm: str, targets: Sequence[str]) -> list[LinkRow]:
        params = (key_norm, f"%{_like_escape(key_norm)}%", json.dumps(list(targets)))
        with self._connect() as conn:
            rows = conn.execute(_LINKS_TO, params).fetchall()
        return [
            LinkRow(doc_id, field, raw, key, target, method, bool(hub))
            for doc_id, field, raw, key, target, method, hub in rows
        ]


# ---------------------------------------------------------------- the agent's tools (SQL)
_SELECT_CONDITIONS = """
SELECT source, field, given_field, present_when, strength FROM field_conditions
ORDER BY source, field, given_field
"""
_SUMMARY_COLUMNS = (
    "doc_id, source, display_title, created, created_rule, updated, updated_rule, body_fields"
)
_SUMMARY_WIDTH = len(_SUMMARY_COLUMNS.split(","))
_SELECT_DOCUMENT = f"""
SELECT {_SUMMARY_COLUMNS}, title, title_field, body, flags
FROM documents WHERE doc_id = ?
"""  # noqa: S608
_COUNT_SOURCE_DOCS = "SELECT count(*) FROM documents WHERE source = ?"
# count(created) counts the documents with a created date (every one inside the window).
_SOURCE_DATES = """
SELECT source, count(*), count(created), min(created), max(created)
FROM documents GROUP BY source ORDER BY source
"""
# Rows of one field of one source, noise left out (parameters: source, field_norm, field).
# field_norm comes first so the (source, field_norm, ...) indexes are used.
_FIELD_ROWS_OF = "source = ? AND field_norm = ? AND field = ? AND is_noise = 0"
_VALUED = "value IS NOT NULL AND value_type NOT IN ('empty', 'null')"
_FIELD_VALUE_ROWS = f"""
SELECT value_norm, value, doc_id FROM fields
WHERE {_FIELD_ROWS_OF} AND {_VALUED} AND value_norm IS NOT NULL
"""  # noqa: S608
_COVERAGE_ROWS = f"""
SELECT doc_id, max(CASE WHEN {_VALUED} THEN 1 ELSE 0 END) FROM fields
WHERE {_FIELD_ROWS_OF} GROUP BY doc_id
"""  # noqa: S608
# The field's text is in the document's body (parameter: the field name).
_IN_BODY = "EXISTS (SELECT 1 FROM json_each(d.body_fields) WHERE value = ?)"
# Parameters: source, field, then those of _FIELD_ROWS_OF.
_COVERAGE_IN_BODY = f"""
SELECT count(*) FROM documents d WHERE d.source = ? AND {_IN_BODY}
  AND d.doc_id NOT IN (SELECT doc_id FROM fields WHERE {_FIELD_ROWS_OF})
"""  # noqa: S608
# Documents with a date value of the field outside the window (parameters: _FIELD_ROWS_OF).
_COVERAGE_DATES_OUTSIDE = f"""
SELECT count(DISTINCT doc_id) FROM fields
WHERE {_FIELD_ROWS_OF} AND value_iso IS NOT NULL AND date_in_range = 0
"""  # noqa: S608
_KEY_OWNERS = """
SELECT k.doc_id, k.key_norm, k.kind, k.is_hub FROM doc_keys k
JOIN documents d ON d.doc_id = k.doc_id WHERE k.key_norm = ?
ORDER BY d.rowid, k.kind
"""
_LINKS_TO = """
SELECT l.doc_id, l.field, l.raw, l.key_norm, l.target_doc_id, l.method, l.target_is_hub
FROM links l JOIN documents d ON d.doc_id = l.doc_id
WHERE l.key_norm = ? OR l.key_norm LIKE ? ESCAPE '\\'
  OR l.target_doc_id IN (SELECT value FROM json_each(?))
ORDER BY d.rowid, l.link_id
"""
# `{marks}`: one ? per doc id. rowid order is the file's key order (rows go in that way).
_SELECT_FIELD_ROWS = """
SELECT source, field, field_norm, doc_id, seq, role, value, value_type, value_norm,
  value_num, value_iso, in_list, is_noise
FROM fields WHERE doc_id IN ({marks}) ORDER BY rowid
"""


def _summary(row: Sequence[Any]) -> DocSummary:
    """A DocSummary from the _SUMMARY_COLUMNS of one row."""
    *facts, body_fields = row
    return DocSummary(*facts, body_fields=tuple(json.loads(body_fields)))


def _like_escape(text: str) -> str:
    """`text` as a literal inside a LIKE pattern with ESCAPE '\\'."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parts(keys: Sequence[str]) -> Iterator[list[str]]:
    """`keys` in parts of at most SQL_KEYS_PER_QUERY, so statements stay short."""
    for start in range(0, len(keys), SQL_KEYS_PER_QUERY):
        yield list(keys[start : start + SQL_KEYS_PER_QUERY])


def _marks(part: Sequence[object]) -> str:
    return ", ".join("?" * len(part))


def _match_clause(source: str, match: FieldMatch) -> tuple[str, list[object]]:
    """The WHERE clause (on documents `d`) of one filter condition, and its parameters."""
    rows_of: list[object] = [source, fold_key(match.field), match.field]
    # Only fixed SQL text goes in (column names chosen here); every value is a parameter.
    in_rows = f"d.doc_id IN (SELECT doc_id FROM fields WHERE {_FIELD_ROWS_OF} AND {{test}})"  # noqa: S608
    if match.kind in ("present", "missing"):
        present = f"({in_rows.format(test=_VALUED)} OR {_IN_BODY})"
        clause = present if match.kind == "present" else f"NOT {present}"
        return clause, [*rows_of, match.field]
    if match.kind in ("norm_in", "value_in"):
        column = "value_norm" if match.kind == "norm_in" else "value"
        test = f"{column} IN (SELECT value FROM json_each(?))"  # noqa: S608
        return in_rows.format(test=test), [*rows_of, json.dumps(list(match.values))]
    column = "value_num" if match.kind == "num_range" else "value_iso"
    tests: list[str] = [f"{column} IS NOT NULL"]
    if match.kind == "iso_range":
        tests.append("date_in_range = 1")  # a date outside the window counts as no date
    bounds: list[object] = []
    if match.low is not None:
        tests.append(f"{column} >= ?")
        bounds.append(match.low)
    if match.high is not None:
        tests.append(f"{column} <= ?" if match.kind == "num_range" else f"{column} < ?")
        bounds.append(match.high)
    return in_rows.format(test=" AND ".join(tests)), [*rows_of, *bounds]


def _field_rows(
    conn: sqlite3.Connection, doc_ids: Sequence[str], fields: Sequence[str] | None
) -> list[FieldValue]:
    """The field rows of these documents (only `fields` when given), by document in the
    order given, then in file key order."""
    wanted = None if fields is None else set(fields)
    by_doc: dict[str, list[FieldValue]] = {}
    order = list(dict.fromkeys(doc_ids))
    for part in _parts(order):
        for row in conn.execute(_SELECT_FIELD_ROWS.format(marks=_marks(part)), part):
            value = _field_value(row)
            if wanted is None or value.field in wanted:
                by_doc.setdefault(value.doc_id, []).append(value)
    return [row for doc_id in order for row in by_doc.get(doc_id, [])]


def _field_value(row: Sequence[Any]) -> FieldValue:
    (source, field, field_norm, doc_id, seq, role, value, value_type, value_norm, value_num,
     value_iso, in_list, is_noise) = row  # fmt: skip
    return FieldValue(
        source=source,
        field=field,
        field_norm=field_norm,
        doc_id=doc_id,
        seq=seq,
        role=role,
        value=value,
        value_type=value_type,
        value_norm=value_norm,
        value_num=value_num,
        value_iso=value_iso,
        in_list=bool(in_list),
        is_noise=bool(is_noise),
    )


_SELECT_FIELD_PROFILES = """
SELECT source, field, field_norm, synonym_group, n_docs, presence, n_values, n_distinct,
  list_ratio, type_mix, kind, kind_confidence, title_ratio, content_ratio, noise_ratio,
  avg_len, max_len, top_values
FROM field_profile ORDER BY source, field
"""


def _field_profile_row(row: Sequence[Any]) -> FieldProfileRow:
    """The inverse of `_field_profile_params`."""
    (source, field, field_norm, synonym_group, n_docs, presence, n_values, n_distinct,
     list_ratio, type_mix, kind, kind_confidence, title_ratio, content_ratio, noise_ratio,
     avg_len, max_len, top_values) = row  # fmt: skip
    return FieldProfileRow(
        source=source,
        field=field,
        field_norm=field_norm,
        synonym_group=synonym_group,
        n_docs=n_docs,
        presence=presence,
        n_values=n_values,
        n_distinct=n_distinct,
        list_ratio=list_ratio,
        type_mix=json.loads(type_mix),
        kind=kind,
        kind_confidence=kind_confidence,
        title_ratio=title_ratio,
        content_ratio=content_ratio,
        noise_ratio=noise_ratio,
        avg_len=avg_len,
        max_len=max_len,
        top_values=tuple((value, n) for value, n in json.loads(top_values)),
    )


# ---------------------------------------------------------------- chunking input (S6)
_PAGE_DOCUMENT_TEXTS = """
SELECT rowid, doc_id, source, display_title, created, created_rule, body, body_fields
FROM documents WHERE rowid > ? ORDER BY rowid LIMIT ?
"""
# Values a chunk header may show: metadata (the title is in the header already), not
# generator noise, not null, empty or a nested object. `{marks}`: one ? per doc id.
_SELECT_HEADER_FIELDS = """
SELECT doc_id, source, field, seq, value FROM fields
WHERE doc_id IN ({marks}) AND role = 'meta' AND is_noise = 0 AND value IS NOT NULL
  AND value_type NOT IN ('empty', 'json')
ORDER BY doc_id, field, seq
"""


def _document_texts(
    conn: sqlite3.Connection, header_fields: Mapping[str, Sequence[str]] | None
) -> Iterator[DocumentText]:
    """Every document in rowid order, read in pages of CHUNK_READ_PAGE_DOCS. Each page (and
    its field values) is fetched whole, so no read is open while the caller writes chunks."""
    after = 0
    while True:
        page = conn.execute(_PAGE_DOCUMENT_TEXTS, (after, CHUNK_READ_PAGE_DOCS)).fetchall()
        if not page:
            return
        after = page[-1][0]
        fields = {} if header_fields is None else _header_fields(conn, page, header_fields)
        for row in page:
            yield _document_text(row[1:], fields.get(row[1], ()))


def _header_fields(
    conn: sqlite3.Connection, page: Sequence[Sequence[Any]], wanted: Mapping[str, Sequence[str]]
) -> dict[str, tuple[FieldText, ...]]:
    """doc_id -> the values of its source's wanted fields, in the wanted order, list items
    in list order (documents with none are left out)."""
    rank = {
        source: {field: i for i, field in enumerate(fields)} for source, fields in wanted.items()
    }
    found: dict[str, list[tuple[int, int, FieldText]]] = {}
    doc_ids = [row[1] for row in page if row[2] in rank]
    for start in range(0, len(doc_ids), SQL_KEYS_PER_QUERY):
        part = doc_ids[start : start + SQL_KEYS_PER_QUERY]
        query = _SELECT_HEADER_FIELDS.format(marks=", ".join("?" * len(part)))
        for doc_id, source, field, seq, value in conn.execute(query, part):
            order = rank[source].get(field)
            if order is not None:
                found.setdefault(doc_id, []).append((order, seq, FieldText(field, str(value))))
    return {
        doc_id: tuple(text for _, _, text in sorted(items, key=lambda item: item[:2]))
        for doc_id, items in found.items()
    }


def _document_text(row: Sequence[object], fields: tuple[FieldText, ...] = ()) -> DocumentText:
    doc_id, source, display_title, created, created_rule, body, body_fields = row
    return DocumentText(
        doc_id=str(doc_id),
        source=str(source),
        display_title=str(display_title),
        created=None if created is None else str(created),
        created_rule=None if created_rule is None else str(created_rule),
        body=str(body),
        body_fields=tuple(json.loads(str(body_fields))),
        fields=fields,
    )


# ---------------------------------------------------------------- chunks
# Foreign keys stay off: in a standalone chunk file the documents table is absent.
_CHUNK_PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
"""
_INSERT_CHUNK = """
INSERT INTO chunks (chunk_id, doc_id, source, ord, kind, level, header, text, n_tokens,
  char_start, char_end)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
# Ties go to the lower chunk id (rowid), so the best `limit` is always the start of the
# best 2 x `limit` (deeper search pages rely on it).
_BM25 = """
SELECT rowid, bm25(chunks_fts, ?, ?) AS score
FROM chunks_fts WHERE chunks_fts MATCH ?
ORDER BY score, rowid LIMIT ?
"""
_BM25_ALL = """
SELECT rowid, bm25(chunks_fts, ?, ?) AS score
FROM chunks_fts WHERE chunks_fts MATCH ?
ORDER BY score, rowid
"""
# Every chunk the keyword query matches, unranked (to check given chunks at any rank).
_MATCHING = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
# A word as the FTS5 unicode61 tokenizer sees one: letters and digits; everything else
# (including "_") separates words.
_FTS_WORD = re.compile(r"[^\W_]+")
# Exact mode (SEARCH_EXACT_PHRASES). A double quote as the model writes one, straight or
# curly; text between a pair is one phrase.
_QUOTE = re.compile(r'["“”]')
# Outside quotes: a word, or a key-shaped token (two or more runs of letters or digits
# joined by - _ / . or #, like ABC-123, v1.2 or snake_case), matched as one phrase. Only
# the shape counts: no prefix or name is learned or listed.
_TOKEN = re.compile(r"[^\W_]+(?:[-_/.#]+[^\W_]+)*")
# Written in capitals outside quotes, these are the model's operators, not words to find:
# every part is already any-of (OR); AND is not honoured (FTS5 matches per chunk, so it
# would miss a document whose parts are in different passages).
_OPERATOR_WORDS = frozenset({"AND", "OR"})


def match_expression(query: str, *, exact: bool = False) -> str | None:
    """An FTS5 query that matches any part of `query` (None when it has no words).

    A part is one word; with `exact`, also a phrase (the words in double quotes, or of a
    key-shaped token) that matches only those words together, in that order. Each part
    is an FTS5 string built from the words alone (any quote in it doubled), so the user's
    text can never be read as FTS5 syntax.
    """
    if exact:
        parts = dict.fromkeys(_exact_parts(query))
    else:
        parts = dict.fromkeys(word.casefold() for word in _FTS_WORD.findall(query))
    if not parts:
        return None
    return " OR ".join(_fts_string(part) for part in parts)


def _exact_parts(query: str) -> Iterator[str]:
    """The parts of `query` in exact mode, in order: each the casefolded words of one
    phrase, joined by spaces (one word: a plain word)."""
    pieces = _QUOTE.split(query)
    if len(pieces) % 2 == 0:  # an unpaired last quote: the text after it is not quoted
        last = pieces.pop()
        pieces[-1] += " " + last
    for i, piece in enumerate(pieces):
        if i % 2:  # between a pair of quotes
            words = _FTS_WORD.findall(piece)
            if words:
                yield " ".join(word.casefold() for word in words)
            continue
        for token in _TOKEN.findall(piece):
            if token not in _OPERATOR_WORDS:
                yield " ".join(word.casefold() for word in _FTS_WORD.findall(token))


def _fts_string(text: str) -> str:
    """`text` as an FTS5 string (a phrase of its words): quoted, any quote doubled."""
    return '"' + text.replace('"', '""') + '"'


def insert_chunks(conn: sqlite3.Connection, rows: Sequence[ChunkRow]) -> None:
    """Insert chunk rows (the caller rebuilds the keyword index after the bulk load)."""
    with conn:
        conn.executemany(
            _INSERT_CHUNK,
            [_chunk_params(r) for r in rows],
        )


def _chunk_params(r: ChunkRow) -> tuple[object, ...]:
    return (r.chunk_id, r.doc_id, r.source, r.ord, r.kind, r.level, r.header, r.text,
            r.n_tokens, r.char_start, r.char_end)  # fmt: skip


def rebuild_chunk_fts(conn: sqlite3.Connection) -> None:
    """Rebuild chunks_fts from chunks once, after all chunks are in (design S7)."""
    with conn:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def insert_embeddings(conn: sqlite3.Connection, rows: Sequence[EmbeddingRow]) -> None:
    with conn:
        conn.executemany(
            "INSERT INTO embeddings (chunk_id, model, dims, vector) VALUES (?, ?, ?, ?)",
            [(r.chunk_id, r.model, r.dims, r.vector) for r in rows],
        )


class SqliteChunkWriter:
    """Builds a fresh file at `path` holding only the chunk tables. One thread only.

    Same life cycle as the document index: written as `<path>.building`, renamed by
    `finish`, deleted by `abort`.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.building = _start_build(path)
        self._conn = sqlite3.connect(self.building)
        self._conn.executescript(_CHUNK_PRAGMAS + CHUNK_DDL)

    def add_chunks(self, rows: Sequence[ChunkRow]) -> None:
        insert_chunks(self._conn, rows)

    def rebuild_fts(self) -> None:
        rebuild_chunk_fts(self._conn)

    def add_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        insert_embeddings(self._conn, rows)

    def finish(self) -> Path:
        return _publish(self._conn, self.building, self.path)

    def abort(self) -> None:
        _discard(self._conn, self.building)


class SqliteChunkReader:
    """Read access to the chunk tables of a published file (see `ChunkReader`).

    `exact_phrases` sets how the keyword side reads a query (`match_expression`).
    """

    def __init__(self, path: Path, *, exact_phrases: bool = SEARCH_EXACT_PHRASES) -> None:
        if path.name.endswith(BUILDING_SUFFIX) or not path.is_file():
            raise StoreError(f"no published chunk index at {path}")
        self.path = path
        self.exact_phrases = exact_phrases

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(open_readonly(self.path)) as conn:
            yield conn

    def chunk_doc_ids(self) -> dict[int, str]:
        with self._connect() as conn:
            return dict(conn.execute("SELECT chunk_id, doc_id FROM chunks"))

    def keyword_expression(self, query: str) -> str | None:
        return match_expression(query, exact=self.exact_phrases)

    def keyword_matches(self, query: str, chunk_ids: Set[int]) -> set[int]:
        match = self.keyword_expression(query)
        if match is None or not chunk_ids:
            return set()
        found: set[int] = set()
        with self._connect() as conn:
            for (chunk_id,) in conn.execute(_MATCHING, (match,)):
                if chunk_id in chunk_ids:
                    found.add(chunk_id)
                    if len(found) == len(chunk_ids):
                        break
        return found

    def bm25(
        self, query: str, limit: int, chunk_ids: Set[int] | None = None
    ) -> list[tuple[int, float]]:
        match = self.keyword_expression(query)
        if match is None or limit <= 0:
            return []
        with self._connect() as conn:
            if chunk_ids is None:
                rows = conn.execute(_BM25, (BM25_HEADER_WEIGHT, BM25_TEXT_WEIGHT, match, limit))
                # SQLite's bm25() is lower-is-better; callers get higher-is-better.
                return [(chunk_id, -score) for chunk_id, score in rows]
            # Every match, best first, keeping the allowed chunks: FTS5 cannot take a long
            # rowid list itself (`rowid IN (...)` re-runs the match per id: 23 s for 25,000
            # ids on data/, against 44 ms for all 62,000 matches of a common word).
            found: list[tuple[int, float]] = []
            rows = conn.execute(_BM25_ALL, (BM25_HEADER_WEIGHT, BM25_TEXT_WEIGHT, match))
            for chunk_id, score in rows:
                if chunk_id in chunk_ids:
                    found.append((chunk_id, -score))
                    if len(found) == limit:
                        break
            return found

    def chunk_texts(self, chunk_ids: Sequence[int]) -> list[ChunkText]:
        with self._connect() as conn:
            found = _chunk_texts(conn, chunk_ids)
        return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]

    def embeddings(self) -> list[tuple[int, bytes]]:
        with self._connect() as conn:
            return list(conn.execute("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"))

    def embedding_models(self) -> set[tuple[str, int]]:
        with self._connect() as conn:
            return _embedding_models(conn)

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def chunks(self) -> Iterator[ChunkRow]:
        """Every chunk row in id order (reports and the eval)."""
        with self._connect() as conn:
            for row in conn.execute(f"SELECT {_CHUNK_COLUMNS} FROM chunks ORDER BY chunk_id"):  # noqa: S608
                yield ChunkRow(*row)


_CHUNK_COLUMNS = (
    "chunk_id, doc_id, source, ord, kind, level, header, text, n_tokens, char_start, char_end"
)


def _embedding_models(conn: sqlite3.Connection) -> set[tuple[str, int]]:
    return {(model, dims) for model, dims in conn.execute(
        "SELECT DISTINCT model, dims FROM embeddings"
    )}  # fmt: skip


# ---------------------------------------------------------------- vectors (S9)
_SELECT_UNEMBEDDED = """
SELECT c.chunk_id, c.doc_id, c.source, c.header, c.text
FROM chunks c LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
WHERE e.chunk_id IS NULL AND c.chunk_id > ?
ORDER BY c.chunk_id LIMIT ?
"""
_COVERAGE = """
SELECT c.source, count(*), count(e.chunk_id)
FROM chunks c LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
GROUP BY c.source
"""


def _chunk_texts(conn: sqlite3.Connection, chunk_ids: Sequence[int]) -> dict[int, ChunkText]:
    """chunk_id -> its text, for the ids that have a chunk."""
    found: dict[int, ChunkText] = {}
    for start in range(0, len(chunk_ids), SQL_KEYS_PER_QUERY):
        part = list(chunk_ids[start : start + SQL_KEYS_PER_QUERY])
        marks = ", ".join("?" * len(part))
        query = (
            f"SELECT chunk_id, doc_id, source, header, text FROM chunks "  # noqa: S608
            f"WHERE chunk_id IN ({marks})"
        )
        found.update((row[0], ChunkText(*row)) for row in conn.execute(query, part))
    return found


class SqliteEmbeddingWriter:
    """Adds vectors to a published, complete index (see `EmbeddingWriter`). One thread only.

    The index stays complete throughout: every write is one transaction of whole rows, and
    chunks without a vector are simply keyword-only.
    """

    def __init__(self, path: Path) -> None:
        if path.name.endswith(BUILDING_SUFFIX) or not path.is_file():
            raise StoreError(f"no published index at {path}")
        meta = read_meta(path) or {}
        if meta.get("status") != STATUS_COMPLETE or meta.get("schema_version") != str(
            SCHEMA_VERSION
        ):
            raise StoreError(f"index {path} is not a complete index of schema {SCHEMA_VERSION}")
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(BUILD_PRAGMAS)

    def unembedded(self, after_chunk_id: int, limit: int) -> list[ChunkText]:
        rows = self._conn.execute(_SELECT_UNEMBEDDED, (after_chunk_id, limit)).fetchall()
        return [ChunkText(*row) for row in rows]

    def chunk_texts(self, chunk_ids: Sequence[int]) -> list[ChunkText]:
        found = _chunk_texts(self._conn, chunk_ids)
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in found]
        if missing:
            raise StoreError(f"no chunks with ids {missing[:5]}")
        return [found[chunk_id] for chunk_id in chunk_ids]

    def add_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        insert_embeddings(self._conn, rows)

    def embedding_models(self) -> set[tuple[str, int]]:
        return _embedding_models(self._conn)

    def coverage(self) -> EmbedCoverage:
        rows = self._conn.execute(_COVERAGE).fetchall()
        return EmbedCoverage(
            chunks={source: n for source, n, _ in rows},
            embedded={source: n for source, _, n in rows},
        )

    def set_meta(self, items: Mapping[str, str]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", list(items.items())
            )

    def add_stage(self, row: StageRow) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO ingest_stages (stage, started_at, finished_at, counts) "
                "VALUES (?, ?, ?, ?)",
                (row.stage, row.started_at, row.finished_at, _json(dict(row.counts))),
            )

    def close(self) -> None:
        """Fold the WAL into the file (readers then need no side files) and close."""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            self._conn.close()


class SqliteVectorCache:
    """Vectors by key (a hash of the exact embedded text), kept across runs. One thread only."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_CHUNK_PRAGMAS + VECTOR_CACHE_DDL)

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """The cached vectors of these keys (missing keys are left out)."""
        found: dict[str, bytes] = {}
        for start in range(0, len(keys), SQL_KEYS_PER_QUERY):
            part = keys[start : start + SQL_KEYS_PER_QUERY]
            marks = ", ".join("?" * len(part))
            query = f"SELECT key, vector FROM vector_cache WHERE key IN ({marks})"  # noqa: S608
            found.update(self._conn.execute(query, part))
        return found

    def put_many(self, model: str, dims: int, items: Sequence[tuple[str, bytes]]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vector_cache (key, model, dims, vector) "
                "VALUES (?, ?, ?, ?)",
                [(key, model, dims, vector) for key, vector in items],
            )

    def close(self) -> None:
        self._conn.close()


if TYPE_CHECKING:  # pyright checks that the readers keep the protocols' shape

    def _reader_is_a_corpus_reader(reader: SqliteReader) -> CorpusReader:
        return reader

    def _chunks_are_a_chunk_reader(reader: SqliteChunkReader) -> ChunkReader:
        return reader
