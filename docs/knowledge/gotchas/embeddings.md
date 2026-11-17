---
name: Embedding gotchas
description: Traps with Gemini embeddings and S9: a list of plain strings is embedded as ONE vector (each text must be its own Content); our own limiter (120 requests/min) sets the pace and 429s start a little above ~5M tokens/min; a cached S9 still spends ~23 s on the cache pass; and four fixed Codex findings: S9's error boundary missed the writer's open/close, concurrent calls could pass the spend cap, the real client's rejected answers bypassed S9's handling, cached vectors skipped the unit checks.
source: runs/2026-10-14-ingestion
---

# Embedding gotchas

## 2026-10-16 S9's error boundary missed opening, meta, stage row and closing (B5 Codex finding 5)
- **Area:** ingestion / S9
- **Symptom:** an error while opening the embedding writer, writing meta or the stage row, or closing would have failed start-up, although S9 must never do so.
- **Cause:** only the embed loop sat inside the handler.
- **Fix:** all of S9 runs in one guarded helper (`embed._guarded`); an error stops S9 with `stop=error`. Test: `test_an_index_write_error_in_s9_stops_it_but_not_startup`.
- **Avoid next time:** when a stage must never fail start-up, put setup and teardown inside the guard too.

## 2026-10-16 Concurrent calls could pass the spend cap (B5 Codex finding 6)
- **Area:** llm / spend ledger
- **Symptom:** with 4 requests in flight, 3 calls were admitted against room for 2 (overshoot at most a few cents, but the cap is the one guard on paid calls).
- **Cause:** the check read recorded spend only; calls in flight were not counted.
- **Fix:** `SpendLedger.reserve` admits a call atomically (recorded + in flight + this one <= cap); `add` swaps the reservation for the real cost; `release` frees it on failure. Tests: `test_calls_in_flight_count_against_the_cap`, `test_a_failed_call_frees_its_share_of_the_cap`.
- **Avoid next time:** any budget checked by parallel workers needs reservations, not a read-then-charge.

## 2026-10-16 The real client's rejected answers bypassed S9's handling (B5 Codex finding 7, partly fixed)
- **Area:** ingestion / S9, llm client
- **Symptom:** an answer the real client rejected (wrong count, bad vector) surfaced as a plain `EmbeddingError`, so S9 re-queued the whole request like a transport error: another billed request, and one bad vector cost the whole batch.
- **Cause:** S9's "drop, do not resend" and per-vector rules were tested only with fakes that return raw vectors, never through `GeminiEmbedder`.
- **Fix:** `checked_vectors` raises `BadEmbeddingAnswer`; S9 counts it as `requests_failed:bad_answer`, charges its tokens and never resends it. Not done: saving the good vectors of a rejected request. Test: `test_a_bad_answer_through_the_real_client_is_dropped_not_resent`.
- **Avoid next time:** test stage behaviour through the real client with a scripted SDK, not only with fakes.

## 2026-10-16 Cached vectors skipped the zero and unit-length checks (B5 Codex finding 8)
- **Area:** ingestion / embedding cache
- **Symptom:** a zero or non-unit vector in the cache would have been loaded without the checks a fresh vector gets.
- **Fix:** `blob_fits` needs unit length within `UNIT_NORM_TOLERANCE` (1e-3); anything else is a miss and is embedded again. Test: `test_a_zero_or_non_unit_cached_vector_is_a_miss`.
- **Avoid next time:** a cache hit must pass the same checks as a fresh result.

## 2026-10-16 A cached S9 still takes ~23 s (B4)
- **Area:** ingestion / S9 sizing
- **Symptom:** a rebuild with every vector in the cache still took ~48 s to /health 200, ~23 s of it in S9 with 0 requests.
- **Cause:** the cache pass hashes ~68K texts and looks them up in a ~1.7 GB, ~393K-row cache.
- **Fix:** none needed (warm starts reuse the index and skip it: 1.5 s).
- **Avoid next time:** do not read a slow cached rebuild as API trouble.

## 2026-10-16 Our own rate limiter, not the API quota, sets the embedding pace (C4)
- **Area:** llm / embeddings (S9 sizing)
- **Symptom:** none; measured: 10,000 texts in 50.7 s = 4.87M tokens/min, ~11,800 texts/min, 118 requests/min.
- **Cause:** `EMBED_REQUESTS_PER_MINUTE` = 120 is the bottleneck. In long runs a few 429s appeared near 5M tokens/min (bursts of up to ~600K tokens per request); the client's backoff (1 s, 2 s) recovered every one.
- **Fix:** none needed: a full data/ index (~28M tokens) takes ~6 min. Cost is estimated (chars/4; the API returns no token counts).
- **Avoid next time:** do not raise the limiter much without handling more 429s; measure before designing a time budget.

## 2026-10-16 A list of plain strings is embedded as ONE vector (C2)
- **Area:** llm / Gemini SDK
- **Symptom:** (would be) one vector back for a batch of texts, or vectors that do not line up with their texts.
- **Cause:** `google-genai`'s `_transformers.t_contents` merges a list of plain strings into ONE Content, and gemini-embedding-2 aggregates the parts of one Content into one vector.
- **Fix:** send `contents=[Content(parts=[Part(text=t)]) for t in texts]`, one Content per text; check vectors returned == texts sent. Test drives the real SDK over an httpx mock transport.
- **Avoid next time:** never pass plain strings as a batch; always check the count of vectors.
