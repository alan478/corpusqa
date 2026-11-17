---
name: Ingestion pipeline — how start-up builds the index (as built)
description: The ingestion pipeline as built on ingest/build: S1–S10 in their real order (S4_date_window after S7, S6 after S8, S9 after the S10 publish), S6 chunk headers with learned field values, the index cache and its fingerprint, pruning, the embedding cache and spend ledger, what the Runtime holds, start-up flow and measured times, failure behaviour, tracing spans, where the build differs from the design, and what the agent's tools read (dates judged one by one against the learned window, person_norm in store/norms.py, CorpusReader). Read before touching src/corpusqa/ingest/, store/ or llm/, or before using the Runtime.
source: runs/2026-10-14-ingestion, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-corpus-tools
---

# Ingestion pipeline (as built)

Start-up turns every JSON file under `CORPUS_DIR` into one stored document, then chunks,
indexes and embeds it, and hands the server a read-only `Runtime`. `/health` turns 200
when `ingest.run` returns.

**Design intent.** Code does only what code can do exactly: ids, dates, splitting,
counting, exact values. Everything else is stored as it came, labelled and searchable.
The meaning of a field is left to the agent at question time. Every rule is a procedure
learned from the loaded corpus (no source names, field names or values from `data/`).
Only three things are promised for the unseen corpus: the folder layout, the document
format, and `dataset_doc_uuid` in every document (the corpus format).

Schema and Runtime field by field: `../models/index-schema.md`. Why each choice was made:
`../decisions/ingestion.md` (plus `dates.md`, `chunking.md`, `embeddings.md`, `store.md`,
`boundaries.md`). Traps: `../gotchas/ingestion.md`, `../gotchas/embeddings.md`.
The full design proposal (with the parts not built) is in the run folder
`docs/runs/2026-10-14-ingestion/design.md` (archive).

## Parts
| File | Role |
|---|---|
| `server/__main__.py` | Wiring point. Builds `GeminiEmbedder.from_settings` (None, with a log line, when there is no key or the spend ledger is unreadable) and passes `functools.partial(ingest.run, embedder=...)` as the `StartupFn`. The only module outside `llm/` that imports `llm.client`. |
| `ingest/pipeline.py` | `run(settings, embedder, *, embed_options) -> Runtime`: the orchestrator (stages, spans, cache decision, publish). |
| `ingest/discover.py` | S1. `parse.py` S2. `core.py` S3. `fields.py` + `dates.py` S4 and the date pass. `clean.py` S5. `profile.py` + `links.py` S8. `chunking.py` S6. `embed.py` S9 and `load_vectors`. `fingerprint.py` the index cache. `models.py` stage dataclasses. `startup.py` `Runtime`, `StartupFn`, `placeholder_startup` (tests only). |
| `store/schema.py` | `DDL` (document index + S8 tables), `CHUNK_DDL` (chunks, FTS5, embeddings), `VECTOR_CACHE_DDL`. `config.SCHEMA_VERSION` = 5 changes with any edit. |
| `store/sqlite.py` | All SQL: `SqliteBuildWriter` (the `.building` file), `SqliteReader`, `SqliteChunkReader` (BM25, chunk → doc, vectors), `SqliteEmbeddingWriter` (S9 on the published file), `SqliteVectorCache`, `SqliteChunkWriter` (eval-only chunk files). |
| `store/interfaces.py` | Protocols `BuildWriter`, `DocumentReader`, `CorpusReader` (the agent tools' reads), `ChunkReader`, `EmbeddingWriter`, `VectorCache` and the row dataclasses. `store/norms.py`: `normalise`, `fold_key`, `person_norm` (shared by ingest and retrieval). |
| `store/vectors.py` | `VectorIndex` (the only numpy module): one float32 matrix, cosine top-k; `unit_blob` / `blob_fits` vector checks. |
| `llm/client.py` | `GeminiEmbedder` (gemini-embedding-2, 768 dims, one `Content` per text, answer checks, L2-norm, one `embed` generation per request). `llm/limits.py`: `RateLimiter`, `SpendLedger`. `llm/interfaces.py`: SDK-free `Embedder` protocol. `llm/fakes.py`: `FakeEmbedder`. |
| `prompts/embed_roles.txt` | Role prefixes (version 1): query `task: question answering \| query: {text}`, document `title: none \| text: {text}`. |

## Flow (real order)
```
./serve -> uvicorn binds PORT (503) -> lifespan daemon thread -> ingest.run(settings, embedder)
  S1_discover          sorted walk of CORPUS_DIR
  fingerprint          corpus listing + code + schema + build settings
  index_cache          prune old builds, then decide: hit or miss (with the reason)
  miss:  SqliteBuildWriter writes <index_dir>/<fp16>/corpus.sqlite.building
         S2_parse > S3_core > S4_fields > S5_clean   per file, streamed
         S7_store                                    batches of 500 documents
         S4_date_window                              corpus-wide date pass
         S8_profile > S8_keys > S8_links             from the stored rows
         S6_chunk                                    after S8: headers use final titles
                                                     and the S8 field profile
         S10_finalize                                checks, meta, rename -> corpus.sqlite
  hit:   mark_used (touch the folder: newest for pruning), nothing rebuilt
  both:  S9_embed      on the PUBLISHED index: embed every chunk without a vector
         S10_vectors   load the vectors that fit the embedder into a VectorIndex
  -> Runtime -> /health 200
```
A fresh build and a resume (cache hit with coverage < 1) are the same S9 code path.
Nothing embeds after `/health` turns 200: the Runtime is frozen and no writer touches the
index while requests read it.

## Stages
| Stage | What it does (as built) |
|---|---|
| S1 discover | Walks `CORPUS_DIR` in sorted order; linked folders only after every real folder (result never depends on listing order). Skips hidden entries, non-`.json` files, links leaving the corpus (counted). Sets aside `unreadable` and `too_large` (> 16 MB). `source` = first folder; a file at the root gets `_root`. |
| S2 parse | `utf-8-sig` decode (bad bytes replaced, flag), NUL removed, duplicate keys flagged, keys kept exactly as given. Lone surrogates from JSON escapes become U+FFFD (`bad_unicode`). Sets aside `json_invalid`, `not_object`, `no_uuid` (missing, blank, or holding a lone surrogate), `duplicate_uuid` (first in path order wins). Any exception in S2–S5 sets the file aside as `process_error`; the build goes on. |
| S3 core | Title chain (title field, else humanised file stem); weak-title rules b–d; body units from `content_field_names` in order (a list gives one unit per item); numbered field families joined; unlisted prose fields promoted (>= 150 chars, >= 15 words, not link-shaped, after a duplicate check; behind `PROMOTE_PROSE_FIELDS`, on). |
| S4 fields | One `fields` row per value and list item for every non-contract, non-body key (title field gets `role='title'`): `value_norm`, `value_num`, `value_iso` + precision + `date_format` (iso, epoch, text), `in_list`, `is_noise`. created/updated picked per document (order, English role-word hints, planned words excluded, file-stem fallback) with the fixed window first. |
| S5 clean | Literal escapes decoded only when they outnumber real newlines; CRLF; trailing spaces; long blank runs. Body = units joined by a blank line; with 2+ labels a `## <label>` line opens each run of one field. No text: body = title, flag `no_body` (still indexed and citable). |
| S7 store | Batched writes (`STORE_BATCH_DOCS` 500) into the `.building` file; `raw` = zlib of the file bytes. A write error fails start-up. |
| S4_date_window | After every document is stored. Learns the window from ISO-format years (1st–99th percentile, ±1 year; fixed window if < 100 values), recomputes `date_in_range`, re-picks created/updated with each date judged on its own (a date outside the window counts as no date: it is no candidate; its row keeps the value with `date_in_range = 0`), writes only changed rows. Facts in `meta` (window, rule, `date_outside_window`: per source its date values, how many are outside, its documents without a usable date, per field). |
| S8 profile | One streamed Python pass over the field rows: per-field kind + confidence (`field_profile`), first-name lexicon, synonym groups, conditional presence (`field_conditions`), weak-title rule (a) (shared titles → `display_title = title — stem`), person matching form (`profile.person_norm`) rewritten into `value_norm`. |
| S8 keys / links | `doc_keys` (own-id values, file stem, key-shaped stem token, digits-only ids as `folder_number`, hub keys held by > 5 docs), learned key prefixes, `source_profile`; `links` for every reference-kind value, resolved by key, last path segment, `folder_number`, unique title, or `self`; never by a bare number. Text-mention scan not built. |
| S6 chunk | Recursive split, 512 est. tokens (chars/4), no overlap. Header `source · display_title · YYYY-MM-DD · field: value · ...` (the date part only when the document has a created date inside the window). With `CHUNK_HEADER_FIELDS` on (production), S6 reads `writer.field_profiles()`, picks per source the fields to show (`chunking.select_header_fields`: presence >= 0.5, title and content ratio < 0.5, kind enum/person/tags/text/mixed/number/boolean), reads their values with `writer.document_texts(header_fields=...)` and renders them after the date (list items joined by `, `, at most 5; values cut at 60 chars; the fields part at most 400 chars, whole parts only). Chunk ids 1..N in document order; FTS5 rebuilt once; `documents.n_chunks` set. Rules and numbers: `../decisions/chunking.md`. |
| S10 finalize | `PRAGMA quick_check`, `foreign_key_check`, row counts of `documents` and `chunks` equal what the build wrote; meta (fingerprint parts, counts, `embed_status = pending`); `status = complete`; rename `.building` → `corpus.sqlite`; reopen read-only. |
| S9 embed | See below. |
| S10_vectors | `load_vectors`: vectors load only if the stored (model, dims) equal the embedder's; else dense search is off (`not_loaded` on the span). |

## Index cache
- Folder: `.cache/index/<first 16 hex of the fingerprint>/corpus.sqlite`.
- Fingerprint = sha256 of four parts, each also in `meta` so a miss can say what changed:
  corpus (sorted path, size, mtime_ns of each kept file; path + reason of each file set
  aside; contents are not read), code (bytes of `FINGERPRINT_CODE_PATHS` = `ingest/`,
  `store/`, `config.py`, `prompts/embed_roles.txt`), `SCHEMA_VERSION`, build settings
  (prose promotion, and this year). The corpus folder path is not in it.
- Miss reasons: `no_index`, `changed` (lists the parts), `incomplete` (a `.building` left
  behind), `unreadable`, `fingerprint_mismatch`.
- Pruning at each start: keep `CACHE_KEEP` (2) builds per corpus folder (the one in use
  counts); delete builds whose corpus folder is gone and crashed builds untouched for an
  hour; never touch other corpora. So the `serve` contract test (temp corpus) never
  deletes the `data/` index.
- `llm/` is not fingerprinted: a client change does not rebuild; the embedding cache
  would hand back the same vectors anyway.

## S9 embed (time-boxed, never fails start-up)
1. **Cache pass.** Each chunk's embedded text (`header + "\n" + text`) gets a key:
   sha256 of JSON `[model, dims, role, prompt version, text]`. Hits from
   `.cache/embeddings/vectors.sqlite` are written into the index at once. The same file
   and key serve `uv run eval-retrieval`. A cached vector must pass the unit-length check
   or it is a miss.
2. **Embed pass.** Each distinct text once, round-robin across sources by document, 100
   texts per request, 4 in flight, behind the client's limiter (120 requests/min). Every
   vector is checked (`unit_blob`: size, finite, non-zero, then L2-normalised). A wrong
   vector count drops the whole request (not resent); one bad vector costs only its
   chunks. Each answered request is written to the index and the cache at once (a
   checkpoint). A request that failed with an error is retried once at the end.
3. **Stops** (meta `embed_stop`): `deadline` (start of `ingest.run` + 3 h − 10 min; requests
   in flight finish), `spend_cap` (ledger `.cache/spend/embeddings.json`, cumulative, cap
   $60, calls in flight reserved), `failures` (8 failed requests in a row after 5 pauses of
   60 s), `no_embedder`, `model_mismatch`, `embedder_unavailable`, `error`. Status:
   `complete`, `partial` or `skipped`. Chunks without a vector stay keyword-searchable;
   the next start resumes.
4. The client retries only 429 and 5xx (backoff 1 s doubling, cap 60 s, up to 5 retries); it never
   retries an answer it rejects (`BadEmbeddingAnswer`), a transport error or a timeout.

## Start-up times (measured on data/, 21,612 docs, 2026-10-16)
| Case | /health 200 after |
|---|---|
| Warm start (index reused) | 1.5 s |
| Rebuild, every vector from the embedding cache | ~48 s (S1–S8 ~19 s, S6 ~4 s, S10 ~1 s, S9 cache pass ~23 s) |
| Cold, cache 64 % seeded | 169 s (24,274 texts in 243 requests, ~4.8M tokens/min) |
| Grader (fresh clone, empty cache), projected | ~6–7 min, ~$5.6 (680 requests at 120/min) |

Index file ~867 MB; server RSS ~480 MB after ready. Budget: the contract allows 4 h; S9's
deadline sits at 2 h 50 min after `ingest.run` starts.

## Failure behaviour
| Problem | Result |
|---|---|
| `CORPUS_DIR` missing | `ConfigError`; start-up fails; `/health` stays 503 (`failed`) |
| One bad file (any S1/S2 reason, or a stage exception) | Set aside in `quarantine` with a reason; build goes on |
| Write error, S8, S6, or an S10 check fails | Exception; the `.building` file is discarded; start-up fails (503) |
| Zero documents | Ready; warning; `known_doc_ids` empty |
| Any S9 problem (index writes included) | S9 stops with a reason; start-up still succeeds |
| No key, or stored vectors from another model | Keyword-only search (`vectors = None`, `embed_coverage = 0.0`) |

## Tracing
One `startup` trace → `ingest` span → `S1_discover`, `index_cache`, `S2_parse`, `S3_core`,
`S4_fields`, `S5_clean`, `S7_store`, `S4_date_window`, `S8_profile`, `S8_keys`, `S8_links`,
`S6_chunk`, `S10_finalize`, `S9_embed`, `S10_vectors`, each with counts. S2–S5 and S7 run
interleaved per file, so their spans are summaries written after the loop
(`metadata.seconds`). `S6_chunk` metadata also holds `header_fields` (on/off),
`header_field_sources` (source -> chosen field names, at most `TRACE_MAX_LISTED_ITEMS`
sources) and `header_field_count`; its output counts include `header_field_values`
(values actually shown after the caps; 0 when off). One `embed` generation per request. With tracing off, the log has
only summary lines; per-stage counts are in the index's `ingest_stages` table.

## Where the build differs from design.md
- S6 is the recursive split chosen in step C (512 tokens, body chunks only). Not built:
  code-fence/table masks, the email/turn/heading cascade, packing to 400, fields chunks
  (replaced by field values in every chunk header, 2026-10-16), dropping repeated email
  quotes. Chunk columns `label`, `speaker`, `block_date` dropped;
  `level` added.
- Order: date pass after S7; S6 after S8 (design: before S7); S9 after S10 publishes
  (design: before S10).
- Dates: a learned window, each date judged on its own (outside it = no date), instead of
  the fixed 1990..this year + 5. (A field-level low-trust rule was built first and removed
  on 2026-10-16: `../decisions/dates.md`.)
- Embeddings: `gemini-embedding-2` with text role prefixes (design: `-001` + task types).
- S8 runs as a streamed Python pass, not SQL aggregates; `links.method` adds `self`.
- No `vectors.f32` / `chunk_ids.i64` mmap files: vectors load from SQLite in ~0.2 s.
- `Runtime` has no `profile` field (the profile lives in `field_profile`,
  `field_conditions`, `source_profile`); it adds `chunks`, `embedder`, `index_path`,
  `cache_hit`. The store is one writer protocol plus readers, not seven repositories, and
  `SqliteReader` opens a connection per call (no pool).
- Fingerprint also covers set-aside files and build settings; pruning is per corpus folder.
- Not built: S1's `duplicate_content_of` flag (it could never fire), the text-mention link
  scan (`LINK_SCAN_TEXT` not added).

## What the agent's tools read (built 2026-10-16: `corpus-tools.md`)
- The Runtime gives `store` (an `IndexReader`: `DocumentReader` + `CorpusReader`),
  `chunks` (BM25 + chunk → doc), `vectors`, `embed_coverage` and `embedder`. Questions are
  embedded with `Runtime.embedder` and the query role, so they match the stored vectors.
- `retrieval/search.py` ranks like `retrieval/hybrid.py` (BM25 + dense, weighted RRF bm25
  1.0 / dense 1.0 / k 60, chunks → documents; it reuses `rrf` and `to_documents`), inside a
  scope. On the production index with header fields (tune set, 2026-10-16): hybrid
  q_recall@10 0.727 (pooled R@10 0.403, MRR 0.671), dense q_recall@10 0.696, BM25 0.603
  (`../decisions/retrieval.md`).
- Every tool read is in `store/` (SQL only there): `CorpusReader` on `SqliteReader`.
- Person values: the tools match through the same `person_norm` S8 used; it lives in
  `store/norms.py` (with `normalise`, `fold_key`), so ingest and retrieval share it.
- Dates: a created/updated date is always inside the window; a document without one has
  no usable date (the tools say how many such documents a date filter could not check).
- The field catalog in the system prompt reads `field_profile` (kind, presence, top
  values, synonym group) and `field_conditions` (when a field applies, so a missing value
  is never read as "no").
- In dev, restart the server after a code change: a second build of the same corpus can
  prune the index a running server reads.

## Tests
`tests/unit/test_ingest_stages.py`, `test_ingest_pipeline.py` (invariants, determinism,
generality with renamed fields and sources, tracing, failures), `test_ingest_profile.py`,
`test_ingest_corpus_wide.py`, `test_index_cache.py`, `test_ingest_chunks_embed.py`,
`test_chunking.py`, `test_header_fields.py`, `test_chunk_index.py`, `test_store_sqlite.py`, `test_llm_client.py`,
`tests/contract/test_server_ingest.py`. Fixture: `tests/fixtures/mini_corpus/` (invented
sources and field names, one edge case per file). Live: `uv run tests-live` (one real
embedding call; loads `.env`).
