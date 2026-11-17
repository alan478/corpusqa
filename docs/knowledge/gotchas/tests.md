---
name: Test gotchas
description: Traps when testing: TestClient never binds a socket and httpx ASGITransport skips the lifespan; the serve contract test greps a start-up log phrase; shutil.copytree keeps file times, so copied fixture corpora share one index fingerprint; and the generality test's renaming helper gives the empty key a name; and an invented test source with 100+ ISO dates moves the learned date window (use epochs for a date meant to fall outside it); a test name thought to be invented was in the corpus (grep data/corpus first).
source: runs/2026-10-14-fastapi-server, runs/2026-10-14-ingestion, runs/2026-10-16-corpus-tools, runs/2026-10-16-agent-tools
---

# Test gotchas

## 2026-10-16 A test name thought to be invented was in the corpus
- **Area:** tests
- **Symptom:** `test_agent.py` used "Larkspur" in its question; a grep of `data/corpus` found
  it in 3+ files (a company, a channel, an email).
- **Cause:** the name was picked without checking the corpus. `test_no_hardcoding.py` scans
  `src/` only, so it could not catch it.
- **Fix:** replaced it with "Quillmoor" after a grep of `data/corpus` found 0 files.
- **Avoid next time:** grep `data/corpus` for every invented name before using it in a test.

## 2026-10-16 An invented test source with ISO dates moves the learned date window
- **Area:** tests, ingest dates
- **Symptom:** a test source meant to hold a date outside the window (dates 2024 and 2099)
  got both as usable created dates.
- **Cause:** with 100+ ISO values in the corpus the window is learned from percentiles
  (`DATE_WINDOW_MIN_VALUES`); the 80 added orders made it 2023-2100, so 2099 was inside.
- **Fix:** that source uses epochs (2024 and 2128): epochs never feed the learned window,
  so the 2128 one stays outside.
- **Avoid next time:** check `meta.date_window_*` and `date_outside_window` of a test build
  before relying on a date being outside the window.

## 2026-10-16 A contract test greps a start-up log phrase (B4)
- **Area:** tests / serve script
- **Symptom:** `test_serve_starts_answers_and_exits_on_sigterm` failed after the start-up log line was reworded.
- **Cause:** the test checks `"ingest: 1 documents ready"` in the log to prove the real start-up ran.
- **Fix:** kept that phrase at the start of the line and added the chunk and vector counts after it.
- **Avoid next time:** grep the tests for a log line before rewording it.

## 2026-10-16 The renaming helper gives the empty key a name (B3)
- **Area:** tests / generality
- **Symptom:** the S8 generality test found one more `mixed` and one fewer `unknown` field after renaming.
- **Cause:** `_renamed_copy` turns the key `""` into a tag; an empty key is noise by shape, a named key is not.
- **Fix:** opt-in `keep_empty_key=True` for the S8 test; B2's test unchanged.
- **Avoid next time:** when a generality test fails, diff the two builds field by field before suspecting the code.

## 2026-10-16 copytree keeps file times, so copied corpora share one fingerprint (B3)
- **Area:** tests / index cache
- **Symptom:** a prune test expected three cache folders and got one.
- **Cause:** the fingerprint covers paths, sizes and mtimes; `shutil.copytree` copies mtimes, so copies of the fixture are the same corpus to the cache (by design).
- **Fix:** tests that need distinct corpora edit one file in each copy.
- **Avoid next time:** when a test needs "a different corpus", change a file, not only its folder.

## 2026-10-14 TestClient never binds a port; ASGITransport skips the lifespan
- **Area:** tests
- **Symptom:** "port binds before start-up finishes" cannot be shown with TestClient; with `httpx.ASGITransport` the start-up never runs and /health stays 503.
- **Cause:** both are in-process ASGI drivers: TestClient runs the lifespan but no socket; ASGITransport runs neither.
- **Fix:** a test runs real `uvicorn.Server` in a thread with a gated start-up and checks 503 over TCP; the async concurrency test wraps the client in `app.router.lifespan_context(app)`.
- **Avoid next time:** test socket-level claims with a real server; drive the lifespan yourself with ASGITransport.
