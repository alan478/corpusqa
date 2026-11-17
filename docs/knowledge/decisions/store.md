---
name: Store decisions
description: Decisions about the store layer (the only place with SQL): one BuildWriter protocol plus readers instead of seven repository protocols, row dataclasses live in store (store may not import ingest), the published DB stays in WAL and is read with mode=ro, raw file bytes kept zlib-compressed, VectorIndex moved to store/vectors.py so ingest can build it, and the header-field reads (`field_profiles()`, `document_texts(header_fields=...)` with `FieldText` values, one paged generator shared by the build writer and the reader); and the agent tools' reads (2026-10-16): scoped BM25 reads every match and keeps the allowed chunks (FTS5 with a long rowid list took 23 s), value tools leave out noise rows, and the reader's value list is `value_counts` (not `field_values`, which already names every field row).
source: runs/2026-10-14-ingestion, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-corpus-tools
---

# Store decisions

## 2026-10-16 Scoped BM25 reads every match and keeps the allowed chunks
- **Area:** store, search
- **Chose:** with `chunk_ids`, `SqliteChunkReader.bm25` reads all matches in score order
  and keeps the allowed ones up to the limit.
- **Over:** `rowid IN (json_each(?))` inside the FTS5 query.
- **Why:** measured on the production index: 23.7 s for 25,000 ids that way, 44 ms for all
  62,000 matches of a common word. The ranking is the same: the best `limit` among the
  allowed chunks (gotchas/retrieval.md).

## 2026-10-16 Value tools ignore noise rows
- **Area:** store, tools
- **Chose:** `value_counts`, `field_coverage` and the filter leave out `is_noise` rows;
  `read_document` shows the whole document.
- **Over:** treating noise rows as values.
- **Why:** the profile already leaves them out of `n_distinct` and the agent's field list; a
  value `field_values` never shows must not be what a filter matches. (Side effect found
  by Codex on 2026-10-17, open: ingest marks a JSON null as noise, so a null shows as
  "absent", not "empty"; `docs/backlog.md`.)

## 2026-10-16 The reader's value list is `value_counts`, not `field_values`
- **Area:** store
- **Chose:** `CorpusReader.value_counts(source, field)`.
- **Over:** `field_values`, the tool's name.
- **Why:** `BuildWriter.field_values()` already means "every field row" (S8); one name for
  two things in one module would mislead.

## 2026-10-16 Header-field reads: field_profiles() and document_texts(header_fields), one paged generator
- **Area:** store / ingestion
- **Chose:**
  - New row type `FieldText(field, value)` (one metadata value as text, one per list
    item) and `DocumentText.fields: tuple[FieldText, ...] = ()`.
  - New reads on both `SqliteBuildWriter` and `SqliteReader` (and the `BuildWriter` and
    `DocumentReader` protocols): `field_profiles() -> list[FieldProfileRow]` and
    `document_texts(header_fields: Mapping[str, Sequence[str]] | None = None)`.
  - Both read chunking input through one generator `_document_texts(conn, header_fields)`
    that pages documents by rowid (`CHUNK_READ_PAGE_DOCS`, 500). With a mapping (source ->
    field names), one query over `fields` per page (ids in `SQL_KEYS_PER_QUERY` batches,
    only documents of a mapped source) reads `role = 'meta'`, non-noise values that are
    not null, empty or json; it also returns `seq`, so values are sorted by (mapping
    order, seq) explicitly. The reader used to read all documents in one unpaged query; it
    now pages too, so the two cannot drift apart. The old `_SELECT_DOCUMENT_TEXTS` is gone.
- **Over:** a JOIN of documents and fields (one row per value; the body would repeat per
  value, and a page boundary would cut a document's values); one query per document (a
  round trip per document on a full corpus).
- **Why:** the writer must hold no open read while S6 writes chunks; per-page reads keep
  that and add one small query per 500 documents. Tests: `tests/unit/test_header_fields.py`.

## 2026-10-16 B4: VectorIndex moves from retrieval/ to store/vectors.py
- **Area:** code layout (design.md §10)
- **Chose:** `git mv retrieval/vectors.py store/vectors.py`; imports in `retrieval/hybrid.py`, `eval/retrieval.py` and two tests updated; AGENTS.md code map updated. It stays the only numpy module.
- **Over:** leaving it in retrieval (the "Ingest uses only the store" contract forbids ingest importing retrieval, so start-up could not build the matrix for the `Runtime`); handing raw rows to the server to build it (two places would then load vectors, and the cache-hit path lives in ingest).
- **Why:** design.md §10 already put `VectorIndex` in `store/vectors.py`; both ingest (builds it) and retrieval (searches it) may import store.

## 2026-10-16 B1: store shape
- **Area:** ingestion / store
- **Superseded in part by:** ingestion.md#2026-10-16 B3 (item (2): the S8 tables were added; item (4): the index moved into a fingerprint sub-folder, `.cache/index/<fp16>/corpus.sqlite`); ingestion.md#2026-10-16 B4 (item (2): chunk tables added; item (3): `documents.n_chunks` and the embedding meta keys added).
- **Chose:** (1) one `BuildWriter` and one `DocumentReader` Protocol in `store/interfaces.py`, with the row dataclasses (`DocumentRow`, `FieldRow`, `QuarantineRow`, `StageRow`) next to them; (2) only `meta`, `documents`, `fields`, `quarantine`, `ingest_stages` now; (3) `documents.n_chunks` and the chunk/embedding meta keys left out until the chunking step; (4) the index is `settings.index_dir / corpus.sqlite` (`index_dir` defaults to `<repo>/.cache/index`); (5) the published DB stays in WAL mode: a `mode=ro` connection on it works (checked on SQLite 3.53.1; test `test_readonly_connection_cannot_write`); (6) `raw` holds the file's bytes, zlib-compressed; (7) the committed JSON files in `tests/fixtures/mini_corpus/` are the fixture (its generator script was not kept).
- **Over:** seven per-table repository Protocols (design §10), which nothing needs yet; creating the S8 tables empty now.
- **Why:** build brief: lean, no extra layers. Row types must live in `store` because `store` may not import `ingest`.
