---
name: Test decisions
description: Decisions about test tooling and test conventions: httpx kept as the TestClient backend for now despite Starlette 1.8 naming httpx2 (pending the user's approval); a test whose behaviour depends on a config switch sets the switch itself (both values where it matters) instead of leaning on the default; the /ask eval runner is tested against a scripted httpx.MockTransport server with invented questions and every output path in tmp_path; the corpus tools' tests build a copy of the mini corpus plus invented sources (paging needs more than 50 matches) instead of growing the fixture; FakeLLM takes a script of replies, exceptions or functions of the request and raises ScriptExhausted (an AssertionError, not an LLMError) when it runs out.
source: runs/2026-10-14-fastapi-server, runs/2026-10-16-eval-weights-headers, runs/2026-10-16-ask-eval-runner, runs/2026-10-16-corpus-tools, runs/2026-10-16-agent-tools
---

# Test decisions

## 2026-10-16 FakeLLM script shape
- **Area:** tests / llm
- **Chose:** `FakeLLM(script)`: each entry is a `ModelReply`, an exception (raised), or a
  function of the `ChatRequest` (sync or async) that returns a reply or raises. Every call is
  recorded first, with the messages copied at call time. Running out raises
  `ScriptExhausted`, a subclass of `AssertionError`. `fake_reply(...)` builds assistant
  replies with zero usage.
- **Over:** a plain list of replies (the timeout and fallback tests need failures and slow
  calls); `ScriptExhausted` as an `LLMError` (the agent's fallbacks would catch it as a model
  failure and hide a script that is too short).
- **Why:** the loop tests need failures, slow replies and replies that depend on the request
  (e.g. on `tool_choice`).

## 2026-10-16 Tool tests build a copy of the mini corpus plus invented sources
- **Area:** tests
- **Chose:** `tests/unit/tool_corpus.py` copies `tests/fixtures/mini_corpus/` and adds
  `benchwork` (80 repair orders) and `logbook` (2 entries: one date inside the window, one
  far outside), then ingests it with `FakeEmbedder(dims=16)`.
- **Over:** adding files to `mini_corpus/` itself.
- **Why:** many ingest tests assert exact counts of the mini corpus (21 documents, 27
  files); the copy keeps them unchanged, the same way `_beacon_corpus` does. Paging needs
  more than 50 matching documents, which the mini corpus does not have.

## 2026-10-16 The /ask eval runner is tested against a fake server, on invented questions only
- **Area:** tests (`tests/unit/test_ask_eval.py`)
- **Chose:** `fake_server` builds an `httpx.MockTransport` that scripts each question's
  reply, delay or transport error and the /health status; `runner.run(..., transport=...)`
  uses it. The `workdir` fixture writes five invented questions and sets to tmp_path and
  points the runner's module-level `QUESTIONS_FILE`, `SETS_DIR`, `RESULTS_DIR`, `HISTORY`
  and `SPEND_DIR` at tmp_path. A `RecordingTracer` checks the `eval_ask` trace and its
  `question` spans, including a run on the protected stand-in set.
- **Over:** a live `serve-dev` server, or the real question file and sets.
- **Why:** unit tests never touch the network, and no test may open the real question
  file or the held-out set (AGENTS.md: Eval split). The fake server can also send what a
  real one rarely does (over-size bodies, deep JSON, timeouts).

## 2026-10-16 Tests set a config switch themselves; they never lean on its default
- **Area:** tests
- **Chose:** when `CHUNK_HEADER_FIELDS` flipped to True, the seven unit tests that had
  leaned on the old default were changed to set the switch themselves: both values where
  it matters (the S6 span off/on in `test_header_fields.py`, the header-parts test
  parametrized off/on, the eval variant-name test); the S6/S9 span test follows the
  config value.
- **Over:** updating the expected values to the new default.
- **Why:** the tests then test the rule, not the current default, and the next flip of a
  switch breaks nothing.

## 2026-10-14 Dev dependency: httpx (kept, despite Starlette's deprecation)
- **Area:** tests
- **Chose:** `httpx` (dev) for `fastapi.testclient.TestClient`, `httpx.AsyncClient` + `ASGITransport` (concurrency test), and real-HTTP calls in the ./serve test.
- **Over:** `httpx2`, which Starlette 1.8 names as the TestClient backend.
- **Why:** switching needs the user's approval (adding httpx2 was denied by a permission prompt). Cost: one StarletteDeprecationWarning per test run. Revisit before Starlette drops httpx support.
