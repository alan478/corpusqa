---
name: Ingestion gotchas
description: Traps in the ingestion pipeline: lone surrogates from JSON escapes broke the SQLite write and failed the whole build (fixed; non-UTF-8 file names on Linux still open), folder links made S1 depend on listing order, two servers on one checkout can break each other's index, stage counts live in ingest_stages not the log, value-only kind rules need lookalike tests, a fixed chunk overlap broke small sizes, chat epoch timestamps are mostly fake, and short date role words matched inside longer words.
source: runs/2026-10-14-ingestion, runs/2026-10-16-corpus-tools
---

# Ingestion gotchas

## 2026-10-16 Lone surrogates from JSON escapes failed the whole build (B5 Codex finding 1)
- **Area:** ingestion / S2 parse
- **Symptom:** a document holding the JSON escape `\ud800` (a lone surrogate) raised UnicodeEncodeError at the batch write, and start-up failed for the whole corpus.
- **Cause:** `json.loads` accepts the escape and yields a lone surrogate, which is not valid Unicode; SQLite stores UTF-8 and refuses it.
- **Fix:** parse replaces lone surrogates in keys and values with U+FFFD (flag `bad_unicode`); a document whose id holds one is set aside as `no_uuid`. Test: `test_a_lone_surrogate_in_one_document_never_fails_the_build`.
- **Still open:** on Linux a file name with bytes that are not UTF-8 becomes a surrogate-escaped `rel_path` and fails the build the same way. Not fixed (cannot be tested on APFS). Recommended fix pending the user (decisions/ingestion.md, Open questions).
- **Avoid next time:** any text that reaches SQLite from outside (values, keys, paths) must be valid UTF-8; one bad document must be set aside, never fail the build.

## 2026-10-16 Folder links made S1 depend on the listing order (B5 Codex finding 9)
- **Area:** ingestion / S1 discover
- **Symptom:** a folder reachable both directly and through a link could be found under either path, so `rel_path` and `source` depended on the order the file system listed entries.
- **Cause:** the walk followed links as soon as it met them.
- **Fix:** sort each folder's entries and walk linked folders only after every real one. Test: `test_discover_prefers_real_folders_over_links_whatever_the_listing_order`.
- **Avoid next time:** never let a result depend on file-system listing order; sort, and test both orders.

## 2026-10-16 Two servers on one checkout can break each other's index (B5 Codex findings 2 and 3)
- **Area:** ingestion / index cache (dev only)
- **Symptom:** possible, not seen: two builds of one corpus at the same time delete each other's `.building` file; a third build of the same corpus (e.g. after a code change, `uv run eval-retrieval --production` while `serve-dev` runs) can prune the index a running server still reads, and `/ask` then gives the internal-error answer.
- **Cause:** no cross-process lock; pruning keeps 2 builds per corpus folder.
- **Fix:** none yet (rejected for now; the graded run has one server). Open question for the user.
- **Avoid next time:** restart the server after a code change; do not run several servers or production evals on one checkout at once.

## 2026-10-16 Stage counts are in `ingest_stages`, not in the log (B5)
- **Area:** ingestion / verification
- **Symptom:** with tracing off (no Langfuse keys) the log shows only summary lines (`building a new index`, `S9 complete`, `documents ready`).
- **Cause:** per-stage counts go to spans and to the index's `ingest_stages` table (one row per stage, JSON counts).
- **Fix:** read `ingest_stages`, or run `ingest.run` under a `RecordingTracer`.
- **Avoid next time:** read `ingest_stages` instead of grepping the log.

## 2026-10-16 Value-only kind rules had three gaps that tests and data/ showed (B3)
- **Area:** ingestion / S8 kinds
- **Symptom:** (1) a crafted title field ("Ticket 12: the gate needs work") came out `own_id`; (2) a slash date `06/07/2024` came out `reference`; (3) on data/, one-value fields holding a tool or priority word ("High") came out `person`.
- **Cause:** (1) the measured own-id rule never checks for spaces; (2) the path shape accepted any `x/y`; (3) the shape-and-reuse person rule accepted one capitalised word.
- **Fix:** own id needs >= 90 % one-token values; a path needs a letter; a person needs 2-4 capitalised words or a handle. Tests in `test_ingest_profile.py`.
- **Avoid next time:** test each value-only rule with a lookalike from another kind, not only with its own examples.

## 2026-10-16 A fixed default overlap broke small chunk sizes (C3)
- **Area:** ingestion / chunking
- **Symptom:** `ChunkSpec("fixed", max_tokens=10)` raised "overlap_tokens must be at least 0 and below max_tokens".
- **Cause:** the default overlap was an absolute 50 tokens, larger than any limit under 51.
- **Fix:** the default overlap is a share of the size (`CHUNK_FIXED_OVERLAP_SHARE` = 0.1). Test: `test_defaults_and_bad_specs`.
- **Avoid next time:** make a default that depends on another parameter a ratio of it, not a constant.

## 2026-10-16 Chat-source epoch timestamps are mostly fake (date check after B2)
- **Area:** ingestion / dates
- **Symptom:** with the fixed window (this year + 5), 4,842 chat docs got no date, and ~2,080 kept dates in 2027–2031 that looked newer than everything else.
- **Cause:** the corpus is generated; one chat source's epoch values spread from 2021 to 2286 (only ~4,675 of 11,606 in 2024–2026). ISO dates corpus-wide sit at p1 2024, p99 2029. The benchmark's own notes say some timestamps are fake-looking.
- **Fix:** a window learned from ISO-format years; each date is judged on its own against it, and one outside counts as no date (decisions/dates.md; a first rule of low-trust fields was removed on 2026-10-16).
- **Avoid next time:** check a date field's spread before trusting it; epoch numbers say nothing about plausibility.

## 2026-10-16 Short role words matched inside longer words (B2)
- **Area:** ingestion / dates (S4 created/updated)
- **Symptom:** on the fixture, a field named `d_slash_mdy` was dropped from the date candidates as a "planned" date; `updated` came out a week early.
- **Cause:** role words were matched as word prefixes, so the planned stem `sla` matched the word `slash` (`eta` and `due` have the same risk: `etag`, `duel`).
- **Fix:** stems shorter than `ROLE_WORD_MIN_PREFIX_CHARS` (4) must equal a whole word; longer stems still match word starts. Test: `test_short_role_words_must_be_whole_words`.
- **Avoid next time:** check every short heuristic word against ordinary English words before using it as a prefix.
