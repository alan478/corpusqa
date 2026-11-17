---
name: Corpus index schema (SQLite, schema 5) and the Runtime
description: The SQLite index that start-up builds (SCHEMA_VERSION 5): every table, its key columns and what each holds (meta keys included), the chunk and embedding tables (chunk header = source · title · date, then `field: value` parts; embedded text = header + newline + text), the embedding cache file, the eval-only chunk files, the invariants the tests check, and the shape of the Runtime the server receives. Read before writing SQL in store/, reading the index from retrieval or the agent, or changing the Runtime.
source: runs/2026-10-14-ingestion, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-corpus-tools, runs/2026-10-16-agent-tools
---

# Corpus index schema and Runtime

DDL: `src/corpusqa/store/schema.py` (`DDL` + `CHUNK_DDL` in one file per build;
`VECTOR_CACHE_DDL` in a separate cache file). `config.SCHEMA_VERSION` = 5 and changes
with any edit to that file. A reader refuses a file whose `meta.status` is not
`complete` or whose `schema_version` differs (no migrations: a new version rebuilds).
How the tables are filled: `../architecture/ingestion.md`.

## Files
| File | Holds |
|---|---|
| `.cache/index/<fp16>/corpus.sqlite` | The index of one corpus + code + schema + settings. WAL mode; read with `?mode=ro`. Written as `corpus.sqlite.building`, renamed when complete. |
| `.cache/embeddings/vectors.sqlite` | Embedding cache, shared by start-up and `eval-retrieval`; outside any index. |
| `.cache/spend/embeddings.json` | Spend ledger (cumulative across runs; cap `EMBED_SPEND_CAP_USD` = $60). |
| `.cache/eval/<variant>/chunks.sqlite` | Eval-only: chunk tables alone for one chunking variant, with a manifest. |

## Tables of `corpus.sqlite`
| Table | One row per | Key columns and meaning |
|---|---|---|
| `meta` | fact | `key`, `value` (text). Keys below. |
| `documents` | stored document | `doc_id` (PK, the `dataset_doc_uuid` trimmed, as given), `source`, `rel_path` (unique), `file_stem`, `title`, `title_field` (NULL = fallback), `title_is_weak`, `display_title` (title, or `title — humanised stem` when weak), `body` (cleaned, `## label` sections), `body_fields` (JSON list of labels), `created` / `updated` (ISO UTC) with `*_field` and `*_rule` (`order`, `name_hint`, `file_stem`; always a date inside the learned window, else NULL), `content_hash` (sha256 of bytes), `size_bytes`, `mtime_ns`, `flags` (JSON list), `raw` (zlib of the file bytes), `n_chunks`. |
| `fields` | metadata value (list items get their own row) | PK (`doc_id`, `field`, `seq`). `field` exactly as in the file; `field_norm` (folded key, grouping only); `role` (`title` or `meta`); `value` (text, NULL for JSON null); `value_type` (str, int, float, bool, null, empty, json); `value_norm` (matching form; person fields use `person_norm`); `value_num`; `value_iso` + `date_precision` + `date_in_range` (1 inside the learned window; 0 outside: no date range matches it) + `date_format` (iso, epoch, text); `in_list`; `is_noise`. Body fields are not repeated here. Indexes on (source, field_norm, value_norm / value_iso / value_num) and value_norm. |
| `quarantine` | file that did not become a document | `rel_path` (PK), `source`, `reason` (`unreadable`, `too_large`, `json_invalid`, `not_object`, `no_uuid`, `duplicate_uuid`, `process_error`), `detail`, `doc_id` (when known). |
| `ingest_stages` | stage | `stage` (S1_discover … S10_finalize, S9_embed), `started_at`, `finished_at`, `counts` (JSON). The place to read per-stage counts. |
| `field_profile` | (source, field) | `kind` (date, boolean, own_id, number, reference, url, email, person, enum, tags, dated_log, text, mixed, unknown) + `kind_confidence`, `synonym_group`, `n_docs`, `presence`, `n_values`, `n_distinct`, `list_ratio`, `type_mix`, `title_ratio`, `content_ratio` (share of its docs whose body holds its text), `noise_ratio`, `avg_len`, `max_len`, `top_values` (JSON). Present = has field rows or its text is in the body. |
| `field_conditions` | (source, field F, given enum field C) | `present_when` (JSON list of C values where F is nearly always there), `strength` (max − min presence, >= 0.8 kept). |
| `source_profile` | source | `n_docs`, `title_fields`, `content_fields`, `own_id_fields`, `key_prefixes`, `date_fields`, `example_doc_ids`, `path_depths`, `stem_title_match`. |
| `doc_keys` | (doc, key, kind) | `key_norm`, `kind` (`field:<name>`, `stem`, `stem_key`, `folder_number`), `is_hub` (key held by > 5 docs). Many-to-many: duplicate keys exist. |
| `links` | (doc, field, raw value, target) | `raw`, `key_norm`, `target_doc_id` (NULL = unresolved, self-match or hub-only), `method` (key, stem, folder_number, unique_title, self, none), `target_is_hub`. Reverse links = query on `target_doc_id`. |
| `chunks` | chunk | `chunk_id` (PK, 1..N in document order), `doc_id`, `source`, `ord` (0..n in the doc), `kind` (`body`), `level` (whole, section, paragraph, line, sentence, word, char), `header` (one line: `source · display_title · date`, then, with `CHUNK_HEADER_FIELDS` on (production), `field: value` parts for the fields the profile picked, list items joined by `, `), `text` (the slice, without header; the embedded text is `header + "\n" + text`, or the text alone when headers are off), `n_tokens` (chars/4, <= 512), `char_start` / `char_end` into `documents.body`. |
| `chunks_fts` | chunk | FTS5 external-content table over (`header`, `text`), `unicode61 remove_diacritics 2`; rebuilt once after the bulk load. BM25 weights: header 0.5, text 1.0. |
| `embeddings` | embedded chunk | `chunk_id` (PK), `model`, `dims`, `vector` (float32 little-endian, L2-normalised). A chunk without a row is keyword-only. |

## `meta` keys
- Build: `schema_version`, `status` (`building` → `complete`), `built_at`, `corpus_dir`,
  `corpus_dir_name`, `promote_prose_fields`, `n_files`, `n_documents`, `n_quarantined`,
  `n_<table>` for documents, fields, quarantine, field_profile, field_conditions,
  source_profile, doc_keys, links, chunks.
- Fingerprint: `corpus_fingerprint` (full), `corpus_hash`, `pipeline_version` (code
  hash), `schema_version`, `build_settings` (JSON).
- Dates: `date_window_min_year`, `date_window_max_year` (key names are the constants
  `META_DATE_WINDOW_MIN_YEAR` / `_MAX_YEAR` in `store/interfaces.py`), `date_window_rule`,
  `date_window_iso_values`, `date_outside_window` (JSON: per source with a date outside the
  window, its date values, how many are outside, its documents without a usable date, and
  per field the values and how many are outside). Replaced `date_low_trust_fields`
  (2026-10-16: each date is judged on its own; `../decisions/dates.md`).
- Profile: `name_lexicon` (JSON), `name_lexicon_size`.
- Embeddings (S10 writes `embed_status = pending`; S9 rewrites): `embed_status`
  (pending, complete, partial, skipped), `embed_stop`, `embed_model`, `embed_dims`,
  `embed_prompt_version`, `n_embeddings`, `embed_coverage`, `embed_coverage_by_source`.
  No timestamp key (two builds of one corpus must have equal meta).

## Embedding cache (`vectors.sqlite`)
`vector_cache(key PK, model, dims, vector)` WITHOUT ROWID. `key` = sha256 of JSON
`[model, dims, role, prompt version, text]` (`ingest.embed.cache_key`). The key does not
cover client code: if the client ever changes how vectors are shaped, bump the
`embed_roles` prompt version or delete the cache.

## Invariants (tested)
- `known_doc_ids` == every `documents.doc_id`; no quarantined file's id is in it.
- `documents.body[char_start:char_end] == chunks.text`; no chunk over its token limit.
- Every vector has the meta's dims and unit length (± `UNIT_NORM_TOLERANCE` 1e-3).
- `created <= updated` when both are set.
- Two builds of the same corpus give the same ids, rows and meta.

## Runtime (`ingest/startup.py`)
Frozen dataclass the server receives from start-up; requests share it, never change it.

| Field | Type | Meaning |
|---|---|---|
| `known_doc_ids` | `frozenset[str]` | Every ingested `dataset_doc_uuid`; `/ask` may cite only these. |
| `store` | `IndexReader \| None` | Read-only access to the index; a connection per call. `IndexReader` (`ingest/startup.py`) = `DocumentReader` (`meta`, `doc_ids`, `field_profiles`, `document_texts(header_fields=...)`, whose `DocumentText.fields` holds `FieldText(field, value)` rows) + `CorpusReader` (the agent tools' reads: `document_summaries`, `document`, `value_counts`, `field_coverage`, `filter_doc_ids`, `document_fields`, `key_owners`, `links_to`, `date_window`, `source_dates`, `field_conditions`; `../architecture/corpus-tools.md`). `SqliteReader` is both. None only in placeholder and server tests. |
| `index_path` | `Path \| None` | The published `corpus.sqlite`. |
| `built_at` | `str \| None` | ISO time the index was built. |
| `cache_hit` | `bool` | True when start-up reused an index. |
| `chunks` | `ChunkReader \| None` | BM25 over chunk header + text (optionally limited to a set of chunk ids: `bm25(..., chunk_ids=)`), `keyword_matches`, `chunk_texts`, chunk → doc map, stored vectors. |
| `vectors` | `VectorIndex \| None` | Chunk vectors in memory for dense search (`search(query, k, allowed=None) -> [(chunk_id, score)]`; ties go to the lower chunk id). None when there is no embedder, no vector, or the stored model/dims differ from the embedder's. |
| `embed_coverage` | `float` | Share of chunks whose vector is in `vectors` (0.0 when None). |
| `embedder` | `Embedder \| None` | The embedder start-up used; embed queries with it (query role). None = keyword-only. |
