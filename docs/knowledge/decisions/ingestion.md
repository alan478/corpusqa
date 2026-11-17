---
name: Ingestion decisions
description: Decisions about the ingestion pipeline (S1–S10, index cache, Runtime): every rule generic and learned from the loaded corpus; the design approved with its recommendations (parts later superseded); B0–B4 readings of the spec (stage order with S6 after S8 and S9 on the published index, S8 kind/key/link rules, fingerprint and per-corpus pruning, S9 stops, no embedding after ready); how the end-to-end run was done; the B5 Codex review answers; and the open questions still waiting for the user.
source: runs/2026-10-14-ingestion, runs/2026-10-16-corpus-tools
---

# Ingestion decisions

## Open questions (OPEN: waiting for the user)
Decided by default so the build could go on; the user may overrule any of them.
- **Non-UTF-8 file names (pending fix, recommended by the coordinator).** On Linux a file
  name whose bytes are not UTF-8 reaches SQLite as a surrogate-escaped `rel_path` and
  fails the whole build, the same way lone surrogates did (B5). Not fixed: APFS refuses
  such names, so it cannot be tested on this machine, and the corpus format makes it
  unlikely. Recommended: fix it anyway, with a test (set the file aside with a reason, or
  store the path safely).
- **B5 (Codex findings 2 and 3, rejected for now):** two servers on one checkout can break
  each other (two builds of one corpus delete each other's `.building`; a third build can
  prune an index a running server reads). The graded run has one server. Add a
  cross-process lock before running several servers or evals at once?
- **B5 (finding 4, rejected):** the corpus fingerprint stays path + size + mtime (no content
  hash). OK?
- **B5 (finding 7, partly fixed):** a bad answer from the real client drops the whole
  request (never resent) instead of saving its good vectors. OK?
- **B4:** (2) S9 runs after S10 publishes (the design drew it before S10); (4) no background
  embedding after `/health` 200; (5) after 8 failures in a row S9 pauses 60 s and goes on,
  up to 5 times (the design said only "skip the batch"). Confirm each.
- **B3:** (1) key prefixes: the rule learns 99 on data/ (about 90 from one CRM source's
  company ids; the deep dive counted 5 owned). Only the text-mention scan (not built) uses
  them: require several keys per prefix when it is built? (2) Six field conditions on
  data/ (strongest 0.97; the design said "exactly one"): keep all six? (3) Two rules
  tightened beyond the measured ones (own id needs one-token values; a person by shape
  needs 2+ capitalised words): confirm. (4) Pruning keeps 2 builds per corpus folder:
  confirm. (5) The person matching form `profile.person_norm` differs from `normalise`;
  the agent's filter tool must use it for person fields. (Done 2026-10-16, corpus tools: `person_norm` moved to
  `store/norms.py`; the filter matches person values through it.)
- **B2:** (3) confirm the 15 B2 readings below (e.g. `process_error` quarantine, summary
  spans for the per-document stages).
- Elsewhere: date window (`dates.md`), import contracts (`boundaries.md`), prose promotion
  and chunking tie-breaks (`chunking.md`), spend cap and dimensions (`embeddings.md`).

## 2026-10-16 B5 Codex review: answers to the 10 findings
- **Area:** review (ingestion build; review file `docs/reviews/2026-10-16-ingestion-build.md`)
- **Scope and how:** `git diff foundation..ingest/build` at 465e5377 (`foundation` is the merge base, so the foundation work already reviewed is left out). `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high"` (reviewer model and effort from `~/.agent-harness/team/team.json`).
- **Result: 7 fixed (one of them partly: 7), 3 rejected.** Each fix started with a test that failed before it.
  1. **High, lone surrogates abort ingestion — FIXED (887d7ae5).** A `\ud800` escape reached the batch write and raised UnicodeEncodeError, so the whole build failed. Parse now replaces lone surrogates in keys and values with U+FFFD (flag `bad_unicode`); a document whose id holds one is set aside as `no_uuid` (that id could never be cited as given). Only files containing a surrogate escape are walked. data/: 289 files hold escapes, all valid pairs; index unchanged.
  2. **High, concurrent builders delete each other's `.building` file — REJECTED (for now).** True, but only when two servers build the same corpus at the same moment in one checkout. CONTRACT runs exactly one `./serve` per fresh clone. A cross-process lock would add a blocking path to every graded start-up for a dev-only race. Open question.
  3. **High, pruning can delete an index a running server uses — REJECTED (for now).** Same reason: it needs a second server on the same checkout building a third index of the same corpus while the first still runs (possible in dev: a code change plus `eval-retrieval --production` while `serve-dev` runs; the server then answers with the internal-error text, still HTTP 200). Workaround: restart the server after a code change. Open question.
  4. **Medium, corpus fingerprint is path + size + mtime only — REJECTED.** Deliberate (fingerprint.py docstring, design §7): a content hash adds a full read of the corpus to every warm start. The grader never has a cache, and the dev corpus does not change; a stale hit needs a same-size rewrite with the old mtime kept.
  5. **Medium, S9's error boundary misses writer open, meta, stage row and close — FIXED (e9a59a68).** All of S9 runs in one guarded helper (`embed._guarded`); an error stops S9 with `stop=error` and never fails start-up; the index is still checked when `_open` reads it. A BaseException still propagates.
  6. **Medium, concurrent calls can pass the spend cap — FIXED (aadbb9da).** `SpendLedger.reserve` admits a call atomically (recorded + in-flight + this one <= cap); `add` swaps the reservation for the real cost; `release` frees it when the call fails.
  7. **Medium, the real client's answer checks bypass S9's handling — PARTLY FIXED (d70aaff4).** `checked_vectors` raises `BadEmbeddingAnswer` (subclass of EmbeddingError); S9 counts it as `requests_failed:bad_answer`, charges its tokens and does not resend it. Not done: per-vector salvage through the real client (the client never hands back an unchecked batch; a bad vector is rare and its chunks stay keyword-only until the next start).
  8. **Medium, cached vectors skip the zero and unit-length checks — FIXED (1557d2ee).** `blob_fits` needs unit length within `UNIT_NORM_TOLERANCE` (1e-3); anything else is a cache miss. All 393,206 dev cache vectors were within 1e-6, so nothing was re-embedded.
  9. **Medium, folder links make S1 depend on listing order — FIXED (be940351).** S1 sorts each folder's entries and walks linked folders only after every real one.
  10. **Medium, screened eval indexes reused across question sets — FIXED (43a797f4).** The variant manifest holds a digest of the kept document ids (`screen_docs`; None for full-corpus variants, so existing eval caches stay valid).
- **Also seen, not fixed:** non-UTF-8 file names on Linux (see Open questions).
- **After the fixes:** rebuild on data/ 49.1 s, 0 requests; documents, chunks and vectors byte-identical to the pre-fix index, so `eval-retrieval --production` was not rerun.

## 2026-10-16 B5: how the end-to-end run was done
- **Area:** ingestion / verification
- **Chose:** (1) cold start = move `.cache/index/` aside and keep `.cache/embeddings/`, so every vector comes from the cache ($0). No true empty-cache start (the B4 projection of ~6–7 min and ~$5.6 stands). (2) Per-stage counts read from the index's `ingest_stages` table, and once from a RecordingTracer run of `ingest.run` on data/, because tracing is off without Langfuse keys and the log has summary lines only. (3) Grader-style `./serve` by hand with exactly CORPUS_DIR, PORT, GEMINI_API_KEY, PATH, HOME (key loaded by the shell from `.env` the way `serve-dev` does, never printed), stopped with SIGTERM to its process group; plus the `serve` contract test inside `uv run check`.
- **Over:** a fresh empty-cache embed (money); a fingerprint-change rebuild for the cold start (moving the folder is simpler and leaves the code untouched).
- **Why:** B5 brief.

## 2026-10-16 B4: S6 chunks, S9 embeddings, the embedding cache and vectors in the Runtime (readings of the spec)
- **Area:** ingestion / S6, S9, §6, §7 (design.md; "Step C result")
- **Superseded in part by:** dates.md#2026-10-16 Dates are judged one by one (the low-trust date rule this entry names is gone).
- **Chose (order and flow):**
  1. Order: S1-S5, S7, date pass, S8, **S6**, S10 (checks, meta, publish), then **S9 on the published index**, then the vectors are loaded (`S10_vectors` span). design.md §4 draws S9 before S10. Why: a fresh build and a resume (cache hit, coverage < 1) are then one code path ("embed every chunk without a vector"), and each finished request is a checkpoint inside the published index, so a crash mid-S9 loses nothing. S10 writes `embed_status = pending`; S9 rewrites the embed meta. Any error in S1-S8, S6 or S10 still fails start-up; S9 never does.
  2. **No background embedding after /health turns 200.** The design is silent on it. Safer: none. The Runtime stays frozen, no writer touches the index while requests read it, and query embeddings never share the rate limit with S9. What is missing is resumed at the next start; keyword search covers it meanwhile.
  3. Deadline = start of `ingest.run` + `STARTUP_BUDGET_S` (3 h) - `STARTUP_RESERVE_S` (10 min), checked before each new request; requests in flight finish. Counted from `ingest.run`, not process start (uv sync and imports fit in the contract's spare hour). `EmbedOptions` carries budget, batch size, concurrency, failure limit and the clock (tests use a fake clock).
- **Chose (S6):**
  4. `ChunkSpec()` (recursive, 512, no overlap, header on) over every document after S8, read back in pages of 500 (`CHUNK_READ_PAGE_DOCS`), chunk ids 1..N in document order, FTS rebuilt once, `documents.n_chunks` set by one UPDATE. S10 checks the chunk count like the document count. The low-trust date rule is step C's (`chunking.chunk_input`). `SCHEMA_VERSION` 4 -> 5.
- **Chose (S9):**
  5. Cache pass first (hits written at once, no call), then the misses, one request per distinct embedded text (chunks with identical text share it), round-robin across sources (by name) by document.
  6. Every vector is checked in S9 whatever the embedder (`store.vectors.unit_blob`: size, finite, non-zero, then L2-normalised). A wrong count drops the whole request (it cannot be aligned; not retried, it would repeat); one bad vector costs only its chunks. A cached vector is checked too (`blob_fits`).
  7. Stops (none fails start-up): deadline; spend cap (`SpendCapReached`); 8 failed requests in a row (`EMBED_MAX_FAILED_IN_A_ROW`) once 5 pauses of 60 s are used up (`EMBED_MAX_COOLDOWNS`, `EMBED_COOLDOWN_S`; after each pause it goes on, so a per-minute quota refills; no pause runs past the deadline; the design said only "skip the batch", which on a dead API would spend ~1.5 h of start-up for nothing, while stopping at once would give up on a throttled key with hours left); no embedder; stored vectors of another model/dims (never mixed; not loaded either); any other exception (logged, span marked ERROR). A request that fails with an error is re-queued once at the end. A crash (BaseException) still propagates.
  8. `EMBED_CONCURRENCY` = 4 in flight (what the eval measured at ~4.9M tokens/min), batch 100.
  9. Spend cap kept at **$60 total** (`EMBED_SPEND_CAP_USD`, the ledger is cumulative). Not lowered: the dev ledger already holds $31.40, so a lower cap would refuse every dev embed; on the grader the ledger starts at 0 and one index is ~$6. The cap is a runaway guard, not a budget.
  10. Embed meta keys: `embed_status` (pending | complete | partial | skipped), `embed_stop`, `embed_model`, `embed_dims`, `embed_prompt_version`, `n_embeddings`, `embed_coverage`, `embed_coverage_by_source`. No timestamp key (the two-builds-are-identical test compares meta).
- **Chose (cache):**
  11. One embedding cache for start-up and the eval: `.cache/embeddings/vectors.sqlite` (`EMBED_CACHE_DB`, `Settings.embed_cache_db` so tests use a temp file), outside the fingerprinted index. Key = sha256 of JSON `[model, dims, role, prompt version, text]` (`ingest.embed.cache_key`); the eval calls the same function and uses the same file. Seeded by an APFS clone (`cp -c`) of the step C eval cache.
  12. The key does not cover client code (the client L2-normalises; S9 normalises again). If the client ever changes how vectors are shaped, bump the embed prompt version or delete the cache. `llm/` is not in the fingerprint: it would rebuild the index, and the cache would still hand back the same vectors.
- **Chose (fingerprint, Runtime):**
  13. `FINGERPRINT_CODE_PATHS` += `prompts/embed_roles.txt` (the role prefixes shape every vector). New ingest code (`ingest/embed.py`) is covered by `ingest`.
  14. `Runtime` += `chunks` (`SqliteChunkReader`: BM25, chunk -> doc, embeddings), `vectors` (`VectorIndex` or None), `embed_coverage` (share of chunks whose vector is in `vectors`; 0.0 when None), `embedder` (for query embedding). Vectors load only when the embedder can query them (same model and dims). No `vectors.f32` mmap files (design §6): loading from SQLite is fast enough.
- **Chose (eval):** `uv run eval-retrieval --production` scores the index start-up built instead of a variant index; `--chunking` and `--production` are mutually exclusive.
- **Over:** S9 inside the `.building` file (two code paths, and a crash mid-S9 throws the index's checkpoints away); embedding in the background after ready; lowering the cap; a cache keyed by text alone; mmap files.
- **Why:** B4 brief (conservative where the design is silent). Tests: `tests/unit/test_ingest_chunks_embed.py`, `tests/contract/test_server_ingest.py`.

## 2026-10-16 B3: S8 profile, keys and links, S10 finalize and the index cache (readings of the spec)
- **Area:** ingestion / S8, S10, cache (design.md S8, S10, §5, §7)
- **Chose (S8 profile):**
  1. Statistics come from one streamed Python pass over the field rows read back through the store (plus `documents.body_fields`), not SQL aggregates: the kind rules need value-shape tests SQL cannot do. A second pass collects the rows of person, own-id and reference fields. On data/ S8 takes ~5 s.
  2. A field is present in a document when it has field rows there or its text went into the body. Value statistics come from field rows only. `content_ratio` = share of its documents whose body holds its text (listed or promoted). New column `fields.in_list` (a one-item list looked like a scalar otherwise).
  3. Kind order: date, boolean, own_id, number, reference, url, email, person (lexicon), enum, person (shape + reuse), tags, dated_log, text, else mixed; `unknown` when a field has no value and no body text; `text` for a field whose text is only in bodies. Why: own_id before number (numeric ids are ids); person (lexicon) before enum (a small team's assignee field is not an enum); person (shape + reuse) after enum so status-like words never become people; url after reference = URLs without a path (homepages).
  4. `kind_confidence` = the share of values that fit the deciding rule (own_id: distinct ratio; enum: share of rows whose value repeats; body-only text: content ratio; mixed/unknown 0).
  5. Reference shape: after an optional `<label>: ` (<= 30 chars, the rest one token), a URL with a path, a path, an issue-style key, or a digit slug. Values over 500 chars skip the short-value shape tests (`SHAPE_MAX_CHARS`).
  6. Own id also needs >= 90 % one-token values (`OWN_ID_MIN_TOKEN_SHARE`): unique short titles that each hold a number passed the measured rule. A path needs a letter: a slash date `06/07/2024` was a path. On data/ the six own-id fields are unchanged.
  7. Lexicon: "seen >= 5 times" read as "starts >= 5 distinct names", so a company name repeated over many documents never becomes a first name. On data/: 126 first names.
  8. Person by shape + reuse: 2-4 capitalised words or a handle (`@x`, `first.last`; underscore handles left out), and >= 50 % of its distinct values also in another field. One capitalised word is not enough. Companies with two capitalised words can still pass (known limit of value-only rules).
  9. Person matching form (`profile.person_norm`): role in parentheses and `<email>` removed, `.` -> space unless the value is an email, then `normalise`. The later filter tool must apply the same function to its input for person fields.
  10. Synonym groups: same `field_norm`, or never in one document + same kind in {enum, tags, reference, url, email} + overlap >= 0.5 of the smaller field's distinct values. People left out: the same people fill many roles. On data/: 19 groups, 40 fields.
  11. Conditional presence: enum fields with <= 20 values (first value per document), values with >= 20 documents, at least two such values; F = every other field of the source; `present_when` = values whose presence is at or above the midpoint of max and min. On data/: 6 rows >= 0.8 (the design said "exactly one (0.93)"; our strongest, 0.97, is that same pair).
  12. Weak-title rule (a): per (source, title field), distinct case-folded titles / documents < 0.5 -> weak, display title = `title — humanised stem`. On data/: 11,111 documents (the chat source).
- **Chose (S8 keys and links):**
  13. A digits-only own-id value or file stem never stands alone as a key: it becomes `folder_number` `<parent folder>/<number>` (leading zeros dropped). A file directly in its source folder has the source as parent.
  14. Key prefixes come from own-id values that are exactly key-shaped and from stem keys. On data/: 99 prefixes. Kept the rule as written: prefixes feed only the text-mention scan, which is off. Note for that step: require several keys per prefix.
  15. Hub = a key held by more than 5 documents. A link that matches only hub keys gets no target (`target_is_hub` = 1). Hubs are never expanded into rows.
  16. Resolution: (1) key candidates = the normalised target and every issue-style token in the raw value; (2) last path segment (no scheme, host, query or fragment), also without a short extension, digits-only segments skipped; (3) `name#n` or path segments `.../name/<word>/n`, then `.../name/n`; (4) unique title among non-weak titles. A match on the referring document itself is method `self` with no target. One row set per distinct (document, field, value). On data/: 42,200 values, 1,122 resolved (2.7 %).
  17. Text-mention scan (`LINK_SCAN_TEXT`) not built and no constant added (decision 6: off).
- **Chose (S10 and cache):**
  18. S10 checks `PRAGMA quick_check`, `PRAGMA foreign_key_check` and that `documents` holds what S7 wrote; any failure raises `StoreError` (start-up fails, /health 503, nothing published). Meta gets the fingerprint parts, `corpus_dir` and `n_<table>` counts.
  19. Fingerprint = sha256 over: the corpus listing (path, size, mtime_ns of each file S1 kept; path and reason of each it set aside), a code hash (`FINGERPRINT_CODE_PATHS`), `SCHEMA_VERSION`, and build settings (prose promotion as the pipeline reads it at call time, and this year). Folder = first 16 hex chars. The corpus folder path is not in it, so identical copies share an index.
  20. Miss reasons: `no_index`, `changed` (with the parts that differ from the newest complete build of the same corpus folder), `incomplete`, `unreadable`, `fingerprint_mismatch`.
  21. Pruning at start keeps `CACHE_KEEP` (2) builds per corpus folder (the one in use counts), deletes builds whose corpus folder is gone and crashed builds untouched for an hour, and never touches other corpora or folders not named like a fingerprint. Why not "keep the newest 2" overall: the `serve` contract test runs `./serve` on a new temp corpus each run, so every `uv run check` would delete the dev index of data/.
  22. `Runtime` gains `store` (a `DocumentReader`; None only for the placeholder and server tests), `index_path`, `built_at`, `cache_hit`. The placeholder start-up stays (tests use it).
  23. `eval/retrieval.py` `ensure_documents` uses `runtime.index_path`; reuse is the pipeline's cache. The old flat `.cache/eval/docs-promote-*/corpus.sqlite` files are no longer used.
  24. The generality test helper `_renamed_copy` got `keep_empty_key` (default off): renaming the empty key gives it a name, and an empty key is noise by shape.
- **Over:** SQL-only aggregates; the design's table order for person vs enum; a cache that prunes other corpora; resolving digits-only keys.
- **Why:** B3 brief: follow the spec, conservative where unclear; each choice has a test in `tests/unit/test_ingest_profile.py`, `test_ingest_corpus_wide.py` or `test_index_cache.py`.

## 2026-10-16 B2: how S1-S5 + S7 read the spec where it was open or contradictory
- **Area:** ingestion / stages
- **Superseded in part by:** item 1's `S7_finalize` span became `S10_finalize` (ingestion.md#2026-10-16 B3); item 15 by ingestion.md#2026-10-16 B3 (item 22) and ingestion.md#2026-10-16 B4 (`ingest.run` is wired; Runtime extended).
- **Chose:**
  1. Tracing: S2-S5 and S7 run interleaved per document (streaming), so their spans are summaries written after the loop, one per stage, with counts and the seconds spent (`metadata.seconds`). S1/S2 spans also list the files set aside (capped by `TRACE_MAX_LISTED_ITEMS`).
  2. A file that makes a stage raise is set aside as `process_error` (new reason code) instead of failing start-up; a write error still fails start-up.
  3. The S1 flag `duplicate_content_of` is not built: identical bytes mean the same `dataset_doc_uuid`, so the second file is already `duplicate_uuid`; the flag could never fire.
  4. A fallback (file-stem) title is never weak; `display_title` does not repeat the stem when it equals the title.
  5. Family siblings shown as header lines also stay field rows (their dates and senders remain filterable and feed created/updated). `body_keys` = keys that produced at least one unit.
  6. A title field that is also listed as content is a body unit and still gets its `title` row.
  7. The promotion duplicate check compares with the body so far (listed + already promoted), ignoring literal `\n` escapes.
  8. A bare-string `content_field_names` is read as a one-item list (flag kept); repeated names are deduplicated.
  9. List items are typed one by one; only objects and lists inside lists are `json`.
  10. `value_norm` strips quotes and `.,;:!?` at both ends; brackets stay (`Name (role)` keeps its `)`).
  11. Role words: stems of 4+ letters match word starts, shorter ones only whole words; the rule says `name_hint` only when the vote changed the date; when votes contradict (created after updated), order decides both.
  12. `NN/NN/YYYY` with equal parts is parsed (not ambiguous).
  13. Body: a `## label` line opens each run of units from one field (not every list item); units are joined by one blank line; family header lines sit directly above their text; `no_body` is set in S5, where the body is built.
  14. A folder named `x.json` is walked like any folder (files inside are documents).
  15. `ingest.run` is not wired into `server/__main__.py` yet (B3), and `Runtime` is unchanged.
- **Over:** literal readings that were unreachable (3), noisy (13) or lost data (5).
- **Why:** build brief: follow the spec, conservative where unclear, lean. Each choice has a test in `tests/unit/test_ingest_stages.py` or `test_ingest_pipeline.py`.

## 2026-10-16 B0: numpy waits for the vector step; prose promotion behind a flag
- **Area:** ingestion / build
- **Superseded in part by:** retrieval.md#2026-10-16 C4 (numpy was added there, for the eval's vector matrix).
- **Chose:** (1) no `uv add numpy` in B0: nothing in B0–B3 uses vectors. (2) Promotion of unlisted prose fields (decision 1, reopened) is built behind `config.PROMOTE_PROSE_FIELDS`, default True, so the chunking eval can compare on and off without code changes.
- **Over:** adding numpy now (an unused dependency); leaving promotion out until step C.
- **Why:** build brief for B0–B2: lean, nothing speculative.

## 2026-10-16 Ingestion design approved with all recommended answers (user decision)
- **Area:** ingestion
- **Superseded in part by:** embeddings.md#2026-10-16 Embedding model (decision 3); chunking.md#2026-10-16 Chunking is not approved (decisions 1, 7, 9 and S6 reopened), then chunking.md#2026-10-16 Step C result (S6 = recursive 512; 1 stays on; 7 not built); dates.md#2026-10-16 Dates (the fixed date window); ingestion.md#2026-10-16 B4 (S9 after S10).
- **Chose:** build design.md as written (Claude Doc "Ingestion Design", rev 19, no edits). User: "implement this now", so the nine open decisions take their recommendations: (1) promote unlisted prose fields (>=150 chars, >=15 words, not link-shaped, after the duplicate check); (2) dates by order plus English role-word hints, planned words excluded, rule recorded per doc; (3) Gemini `gemini-embedding-001` at 768 dims through `llm.client`; (4) ready with partial embeddings at the deadline; (5) raw JSON kept in the DB, compressed; (6) own keys + field links now, text-mention scan off; (7) drop email quotes that repeat another message in the doc; (8) add `numpy` for the vector matrix; (9) chunk target 400 tokens, tune later.
- **Over:** the alternatives listed in design.md §14.
- **Why:** user approval; each recommendation's reason is in design.md §14 (run folder).

## 2026-10-15 Normalization build removed (user decision)
- **Area:** ingestion
- **Chose:** delete the whole team build of step 1a (normalization): handoff, run folder, worktree and every `team/2026-10-15-normalization*` branch, plus its plans. Only the code stays recoverable: tag `archive/normalization-2026-10-15` (dba98bd2).
- **Over:** merging it into `foundation`, or keeping the branches.
- **Why:** after a review of the build (about 5,900 source lines, 80 thresholds, 12,700 test lines; chat sources came out with almost no people or created times; own ids leaked into references). Kept on `foundation`: the split-guard hardening done during that planning (hash-based hard-coding guard, split test that never opens the held-out file). Lesson carried into the design: do not decide the meaning of every field at ingest.

## 2026-10-14 Every ingestion rule is generic; dataset examples only illustrate (user correction)
- **Area:** ingestion
- **Chose:** each rule is stated as a procedure computed from whatever corpus is loaded (value statistics per source), with no source names, field names or values from data/. Examples from data/ are labelled as "what the rule finds here". First case: "fields that apply only to some kinds of documents" is detected by measuring a field's presence per value of each category field (strength score 0–1), not written as a rule about one dataset field.
- **Over:** describing behaviour per field as observed in data/.
- **Why:** the grader runs on an unseen corpus, so a rule tied to a field of this dataset is overfitting.
