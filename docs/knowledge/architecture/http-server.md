---
name: HTTP server (FastAPI) — how /health and /ask work
description: The contract-facing HTTP layer as built: ./serve, config from env, the FastAPI app factory, background start-up with /health 503 → 200, the answerer factory that builds the agent once the Runtime is ready (no key - UnavailableAnswerer, the internal-error answer), the always-200 /ask path (own body parsing, AskQueue with 4 answers at once and the rest waiting, a deadline counted from arrival with asyncio.timeout_at, id filtering, size caps; the body is only answer and document_ids, the trace id goes in the X-Trace-Id response header), and the seams (Answerer, StartupFn → Runtime). Read before touching src/corpusqa/server/, config.py or ./serve.
source: runs/2026-10-14-fastapi-server, runs/2026-10-15-langfuse-tracing, runs/2026-10-14-ingestion, runs/2026-10-16-agent-tools
---

# HTTP server (FastAPI)

## Parts
| File | Role |
|---|---|
| `serve` | Contract entry point: `uv sync --frozen --no-dev --inexact`, then `exec uv run --frozen --no-dev python -m corpusqa.server`. No args, foreground, stays in the grader's process group. |
| `src/corpusqa/config.py` | `Settings.from_env()` (CORPUS_DIR, PORT, GEMINI_API_KEY hidden from repr), `ConfigError`, and every constant of this layer: `ASK_DEADLINE_S = 150`, `ASK_MAX_CONCURRENT = 4`, `TRACE_ID_HEADER = "X-Trace-Id"`, `SHUTDOWN_GRACE_S = 5`, `MAX_ANSWER_CHARS = 64_000`, `MAX_DOCUMENT_IDS = 500`, the fixed answer texts. |
| `server/__main__.py` | The only place that picks concrete parts. Start-up: `functools.partial(ingest.run, embedder=build_embedder(settings))` (the Gemini embedder, or None: keyword-only search; see `ingestion.md`). Chat model: `build_chat(settings)` = `GeminiChat.from_settings` (`__main__.py:64`), or None and the reason (no key, unreadable ledger). Answerer: `functools.partial(build_answerer, chat=..., problem=..., settings=...)` passed as `answerer_factory` (`__main__.py:124`). `init_tracing(settings)`. Then `uvicorn.run(app, workers=1, log_config=None, access_log=False, timeout_graceful_shutdown=5)`. Bad env → one CRITICAL line, exit 2. |
| `server/__main__.py: build_answerer` | Called once by the start-up thread with the `Runtime` (`__main__.py:76`). No chat model: `UnavailableAnswerer(reason)` and a WARNING on the `answerer` span. Else `corpus_session_factory(runtime.store, runtime.chunks, runtime.vectors, runtime.embedder)` (field index, summaries, catalog: 0.4-0.6 s on data/) and `AgentAnswerer(chat, tools, catalog=tools.catalog, ask_deadline_s=settings.ask_deadline_s)`. A Runtime without `store` or `chunks` raises: start-up fails. The agent itself: `agent.md`. |
| `server/app.py` | `create_app(settings, startup, answerer=None, tracer=None, *, answerer_factory=None)` (in `app.py`): exactly one of `answerer` (ready now; tests) and `answerer_factory` (else `ValueError`). Lifespan, `/health`, `/ask`, `AskQueue`. Tracer defaults to off. FastAPI's own telemetry off. No module-level state. |
| `server/models.py` | `AskRequest(question)`, `AskResponse(answer, document_ids)` (exactly CONTRACT §3's body; user, 2026-10-17: "they don't ask for trace"), `HealthResponse(status)`. The trace id is sent in the `X-Trace-Id` response header. |
| `server/respond.py` | `build_response(...)`: the last check on every /ask body (non-empty answer, ids de-duplicated in order and filtered to known ids, at most `MAX_DOCUMENT_IDS`, a cut logged and put on the ask trace, lone surrogates cleaned). |
| `agent/answerer.py` | Seam: `Answerer` Protocol (`async answer(question, time_left_s=None) -> AnswerResult`); `AgentAnswerer` (the agent), `UnavailableAnswerer` (every call raises `AnswererUnavailable`), `PlaceholderAnswerer` (tests only). |
| `ingest/startup.py` | Seam: `type StartupFn = Callable[[Settings], Runtime]`, `Runtime` (known_doc_ids, store as an `IndexReader`, index_path, built_at, cache_hit, chunks, vectors, embed_coverage, embedder; see `../models/index-schema.md`), `placeholder_startup` (tests only). |

## Flow
```
./serve ─► uvicorn (1 process) ─► lifespan: daemon thread
             startup trace: StartupFn(settings) ─► Runtime
                            answerer span: answerer_factory(Runtime) ─► Answerer
             ─► Readiness.ready = Ready(runtime, answerer)   (one reference assignment)
GET  /health ─► 200 {"status":"ok"} once ready is set; else 503 {"status":"starting"|"failed"}
POST /ask    ─► deadline = arrival + ask_deadline_s (150 s)
             ─► parse body itself (one try block) ─► not ready? fixed answer
             ─► asyncio.timeout_at(deadline) around:
                  AskQueue.slot (at most 4 answers at once; the rest wait in line, never refused)
                  answerer.answer(q, time_left_s = deadline - now)
             ─► build_response ─► 200 {answer, document_ids}, header X-Trace-Id
             every failure path ─► fixed answer text, document_ids [], still HTTP 200
```
- The port binds immediately; start-up and the answerer factory run beside it in a **daemon thread**, so a SIGTERM never waits on a long ingest. /health is 200 only once the answerer exists; a factory that raises fails start-up exactly like the start-up job (logged `startup FAILED`, trace ERROR, /health 503 `failed`).
- **The queue** (`AskQueue` in `app.py`): an `asyncio.Semaphore(ASK_MAX_CONCURRENT)` plus two counters used only on the event loop. Only the answerer call is queued: bad requests and not-ready answers never wait. Waiters are woken in arrival order.
- **The deadline counts from arrival** (the first thing the `ask` handler does), so a wait in line is part of it. A request whose wait and answer pass it gets the timeout answer at its own deadline, still in line or answering. The answerer is told the time left; `AgentAnswerer` ends its own deadline `AGENT_DEADLINE_MARGIN_S` (10 s) before it, so the agent's answer normally beats the server's cut.
- **No key** (no chat model): every /ask raises `AnswererUnavailable`, which maps to the internal-error answer (HTTP 200, no ids), outcome `error`, one log line without a traceback. Not the placeholder text: that says answering "is not implemented", which is false now.
- One log line per /ask: trace_id, outcome (`ok`, `invalid_request`, `not_ready`, `timeout`, `empty_answer`, `error`), ms, id count, answer size, question prefix. A request that has to wait logs one more line.
- Internal errors and blank answers give the internal-error text, never a "not found" answer.
- The ask trace records the queue: `queue_limit`, `queue_running_on_arrival`, `queue_waiting_on_arrival`, `queue_wait_ms`, `queue_got_slot`.

## Known gaps (Codex review 2026-10-17, open; `docs/backlog.md`)
- The request body is read before `timeout_at` starts (`_answer` awaits `request.body()` before it enters `asyncio.timeout_at(deadline)`): only a stalled client waits; it holds no slot.
- Tool calls have no time limit of their own: a slow tool can run past the agent's deadline, and the server's timeout answer then replaces the agent's (the contract still holds).

## Tests
`tests/contract/test_server_contract.py` (health states, shapes, id filtering, errors, timeout, bad bodies incl. invalid UTF-8, size cap, 4 concurrent asks, the trace id header), `tests/contract/test_server_agent.py` (the answerer factory runs once with the Runtime and /health waits for it; a raising factory fails start-up; no chat model → internal-error answer; the real corpus tools through the server; six asks run four at a time; the wait counts against the deadline), `tests/contract/test_serve_script.py` (real ./serve: minimal env, no key, stdin /dev/null, own process group, health → ask → internal-error answer → SIGTERM), `tests/unit/test_server_queue.py`, `test_config.py`, `test_server_respond.py`, `test_placeholders.py`.
