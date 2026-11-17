"""SQLite schema of the corpus index (design.md §5).

`DDL` is the document index, with the tables the profile stage (S8) learns from it:
field and source profiles, conditional presence, own keys and links. `CHUNK_DDL` (chunks,
their keyword index and their vectors) is a separate group: the build writer adds it to
the document index (S6, S9), and the chunking eval builds it alone in a file per variant.
`VECTOR_CACHE_DDL` is the embedding cache (S9 and the eval), kept outside any index.
`config.SCHEMA_VERSION` changes with any edit to this file.
"""

# Per build connection: WAL lets readers work beside the writer; NORMAL sync is safe in
# WAL mode and much faster for a bulk load; foreign keys make deletes cascade.
BUILD_PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
"""

DDL = """
-- Build facts and status. status: building | complete.
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per ingested JSON file.
CREATE TABLE documents (
  doc_id          TEXT PRIMARY KEY,          -- dataset_doc_uuid, trimmed, as given
  source          TEXT NOT NULL,             -- first folder under CORPUS_DIR
  rel_path        TEXT NOT NULL UNIQUE,
  file_stem       TEXT NOT NULL,
  title           TEXT NOT NULL,
  title_field     TEXT,                      -- NULL when the fallback was used
  title_is_weak   INTEGER NOT NULL DEFAULT 0,
  display_title   TEXT NOT NULL,             -- title, or title + humanised stem
  body            TEXT NOT NULL,             -- cleaned, labelled, full text
  body_fields     TEXT NOT NULL,             -- JSON list of unit labels in order
  created         TEXT,                      -- ISO 8601 UTC ('2025-02-18' or '...T10:00:00Z')
  created_field   TEXT,
  created_rule    TEXT,                      -- order | name_hint | file_stem
  updated         TEXT,
  updated_field   TEXT,
  updated_rule    TEXT,
  content_hash    TEXT NOT NULL,             -- sha256 of the file bytes (change detection)
  size_bytes      INTEGER NOT NULL,
  mtime_ns        INTEGER NOT NULL,
  flags           TEXT NOT NULL DEFAULT '[]',-- JSON list of flag strings
  raw             BLOB NOT NULL,             -- zlib(original file bytes)
  n_chunks        INTEGER NOT NULL DEFAULT 0 -- chunks of its body (S6)
);
CREATE INDEX documents_source   ON documents(source);
CREATE INDEX documents_created  ON documents(created);
CREATE INDEX documents_updated  ON documents(updated);
CREATE INDEX documents_stem     ON documents(file_stem);
CREATE INDEX documents_hash     ON documents(content_hash);

-- One row per metadata value (list items get their own row).
CREATE TABLE fields (
  doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  source          TEXT NOT NULL,             -- denormalised for fast filters
  field           TEXT NOT NULL,             -- the key exactly as in the file
  field_norm      TEXT NOT NULL,             -- folded key, for grouping only
  seq             INTEGER NOT NULL,          -- 0 for scalars; list position otherwise
  role            TEXT NOT NULL,             -- title | meta
  value           TEXT,                      -- as text; NULL for JSON null
  value_type      TEXT NOT NULL,             -- str|int|float|bool|null|empty|json
  value_norm      TEXT,                      -- matching form, <= VALUE_NORM_MAX_CHARS
  value_num       REAL,
  value_iso       TEXT,                      -- parsed date/time, UTC
  date_precision  TEXT,                      -- date|datetime|month|week|quarter
  date_in_range   INTEGER,                   -- 1/0 when value_iso is set
  date_format     TEXT,                      -- iso|epoch|text when value_iso is set
  in_list         INTEGER NOT NULL DEFAULT 0,-- 1 when the field's value is a JSON list
  is_noise        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (doc_id, field, seq)
);
CREATE INDEX fields_filter ON fields(source, field_norm, value_norm);
CREATE INDEX fields_value  ON fields(value_norm);
CREATE INDEX fields_date   ON fields(source, field_norm, value_iso);
CREATE INDEX fields_num    ON fields(source, field_norm, value_num);

-- Files that did not become documents, with the reason.
CREATE TABLE quarantine (
  rel_path        TEXT PRIMARY KEY,
  source          TEXT NOT NULL,
  reason          TEXT NOT NULL,             -- S1/S2 reason codes
  detail          TEXT,
  doc_id          TEXT                       -- when known (duplicate_uuid)
);

-- Stage timings and counts of the build.
CREATE TABLE ingest_stages (
  stage           TEXT PRIMARY KEY,
  started_at      TEXT NOT NULL,
  finished_at     TEXT,
  counts          TEXT NOT NULL              -- JSON
);

-- S8: learned facts about each field of each source (the agent's field list). A field
-- counts as present in a document when it has field rows or its text is in the body.
-- Value statistics (n_values ... top_values) come from field rows only: a field whose
-- text went into the body has none there.
CREATE TABLE field_profile (
  source          TEXT NOT NULL,
  field           TEXT NOT NULL,
  field_norm      TEXT NOT NULL,
  synonym_group   INTEGER,                   -- same number = shown together (NULL: alone)
  n_docs          INTEGER NOT NULL,          -- docs having the key
  presence        REAL NOT NULL,             -- n_docs / docs in source
  n_values        INTEGER NOT NULL,          -- field rows
  n_distinct      INTEGER NOT NULL,          -- distinct value_norm of non-noise values
  list_ratio      REAL NOT NULL,             -- share of its docs with field rows holding a list
  type_mix        TEXT NOT NULL,             -- JSON {value_type: rows}
  kind            TEXT NOT NULL,             -- ingest.profile kind rules
  kind_confidence REAL NOT NULL,             -- 0..1: share of values that fit the kind
  title_ratio     REAL NOT NULL,             -- share of its docs where it is the title field
  content_ratio   REAL NOT NULL,             -- share of its docs where its text is in the body
  noise_ratio     REAL NOT NULL,             -- share of its rows marked is_noise
  avg_len         REAL NOT NULL,
  max_len         INTEGER NOT NULL,
  top_values      TEXT NOT NULL,             -- JSON [[value, rows], ...] (most common first)
  PRIMARY KEY (source, field)
);

-- S8: "field F applies when enum field C has one of these values".
CREATE TABLE field_conditions (
  source          TEXT NOT NULL,
  field           TEXT NOT NULL,             -- F
  given_field     TEXT NOT NULL,             -- C
  present_when    TEXT NOT NULL,             -- JSON list: C values where F is (nearly) always there
  strength        REAL NOT NULL,             -- 0..1: max - min presence of F over C's values
  PRIMARY KEY (source, field, given_field)
);

-- S8: per-source summary.
CREATE TABLE source_profile (
  source          TEXT PRIMARY KEY,
  n_docs          INTEGER NOT NULL,
  title_fields    TEXT NOT NULL,             -- JSON {field: docs} (fallback titles left out)
  content_fields  TEXT NOT NULL,             -- JSON {field: docs whose body holds its text}
  own_id_fields   TEXT NOT NULL,             -- JSON list
  key_prefixes    TEXT NOT NULL,             -- JSON list, learned from own keys
  date_fields     TEXT NOT NULL,             -- JSON list
  example_doc_ids TEXT NOT NULL,             -- JSON list (first in path order)
  path_depths     TEXT NOT NULL,             -- JSON {folders below the source: docs}
  stem_title_match REAL NOT NULL             -- share of docs whose file stem shares title words
);

-- S8: a document's own identifiers (many-to-many: duplicate keys exist).
CREATE TABLE doc_keys (
  doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  key_norm        TEXT NOT NULL,
  kind            TEXT NOT NULL,             -- field:<name> | stem | stem_key | folder_number
  is_hub          INTEGER NOT NULL DEFAULT 0,-- held by too many documents to identify one
  PRIMARY KEY (doc_id, key_norm, kind)
);
CREATE INDEX doc_keys_key ON doc_keys(key_norm);

-- S8: references from a document to others: the raw value and 0..n resolved targets.
CREATE TABLE links (
  link_id         INTEGER PRIMARY KEY,
  doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  field           TEXT,
  raw             TEXT NOT NULL,
  key_norm        TEXT NOT NULL,
  target_doc_id   TEXT REFERENCES documents(doc_id) ON DELETE SET NULL,
  method          TEXT NOT NULL,             -- key|stem|folder_number|unique_title|self|none
  target_is_hub   INTEGER NOT NULL DEFAULT 0 -- matched only a hub key (target left NULL)
);
CREATE INDEX links_doc    ON links(doc_id);
CREATE INDEX links_target ON links(target_doc_id);
CREATE UNIQUE INDEX links_unique ON links(doc_id, field, raw, target_doc_id);
"""

# Chunks, their keyword index and vectors (design.md §5). The columns label, speaker and
# block_date of the design belong to the structure-aware split, which was not chosen;
# `level` records how the recursive split cut the chunk. A chunk without a header stores
# an empty one. In a standalone chunk file (the eval) `documents` is absent and the
# reference to it is not enforced (foreign keys stay off there).
CHUNK_DDL = """
CREATE TABLE chunks (
  chunk_id        INTEGER PRIMARY KEY,       -- rowid; vectors refer to it
  doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  source          TEXT NOT NULL,
  ord             INTEGER NOT NULL,          -- 0..n within the document
  kind            TEXT NOT NULL,             -- body
  level           TEXT NOT NULL,             -- how it was cut (ingest.chunking)
  header          TEXT NOT NULL,             -- context header ('' when headers are off)
  text            TEXT NOT NULL,             -- the slice itself, without header
  n_tokens        INTEGER NOT NULL,
  char_start      INTEGER,                   -- offsets into documents.body (body kind)
  char_end        INTEGER,
  UNIQUE (doc_id, ord)
);
CREATE INDEX chunks_source ON chunks(source);
CREATE INDEX chunks_doc    ON chunks(doc_id);

-- Keyword index over header and text (external content; rebuilt once after the bulk load).
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  header, text,
  content = 'chunks', content_rowid = 'chunk_id',
  tokenize = 'unicode61 remove_diacritics 2'
);

-- One vector per embedded chunk; chunks without a row are keyword-only.
CREATE TABLE embeddings (
  chunk_id        INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  model           TEXT NOT NULL,
  dims            INTEGER NOT NULL,
  vector          BLOB NOT NULL              -- float32 little-endian, L2-normalised
);
"""

# Embedding cache (S9 and the eval): one vector per exact embedded text.
VECTOR_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS vector_cache (
  key             TEXT PRIMARY KEY,          -- sha256 of model, dims, role, prompt version, text
  model           TEXT NOT NULL,
  dims            INTEGER NOT NULL,
  vector          BLOB NOT NULL              -- float32 little-endian, L2-normalised
) WITHOUT ROWID;
"""
