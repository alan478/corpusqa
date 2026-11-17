---
name: Server decisions
description: Decisions about the HTTP server process: FastAPI + plain uvicorn as runtime deps, start-up in a daemon thread from the lifespan, uvicorn with no access log, logs on stderr and a 5 s graceful-shutdown cap; the agent wired in through an answerer factory called once the Runtime is ready (no key: UnavailableAnswerer, the internal-error answer; Runtime.store typed IndexReader); and AskQueue (at most 4 answers at once, the rest wait in line, the deadline counts from arrival with asyncio.timeout_at, the time left goes to the answerer).
source: runs/2026-10-14-fastapi-server, runs/2026-10-16-agent-tools
---

# Server decisions

## 2026-10-16 A queue for /ask: four answers at a time, the deadline from arrival
- **Area:** server / agent
- **Chose** (user: "we need to add queue right now for ask so that we can handle 4 request
  in one go"):
  - `ASK_MAX_CONCURRENT = 4` (CONTRACT §4). `server/app.py` `AskQueue`: an
    `asyncio.Semaphore(4)` and two counters (running, waiting; only on the event loop). One
    queue per app. Only the answerer call is queued: bad requests and not-ready answers never
    wait. A request that finds four running waits in line (asyncio wakes waiters in order);
    nobody is refused.
  - The deadline counts from arrival: the handler takes `loop.time() + ask_deadline_s`
    first, and `asyncio.timeout_at(that)` covers the wait and the answer (the old
    `asyncio.wait_for` counted from the answerer call). A request whose wait and answer pass
    it gets the timeout answer at its own deadline (HTTP 200).
  - The answerer is told the time left: `Answerer.answer(question, time_left_s=None)`.
    `AgentAnswerer` counts the agent's deadline from it; with no time left after the margin
    it raises `TimeoutError` at once.
  - Trace: `queue_limit`, `queue_running_on_arrival`, `queue_waiting_on_arrival`,
    `queue_wait_ms`, `queue_got_slot` on the `ask` trace; a request that waits logs a line.
  - Checked: nothing on the /ask path blocks the event loop for long (tool bodies and
    `resolve_citations` run in threads; the chat call is async). Still on the loop, small:
    building a session, `rough_tokens`, `build_response`, building each Gemini request, and
    `SpendLedger.add` writing its file (Codex #3, open).
- **Tests:** `tests/unit/test_server_queue.py`; contract tests: six asks at once run four at
  a time and all get answers; the wait counts against the deadline and ends in the timeout
  answer; the queue is on each ask trace.
- **Over:** refusing a fifth request (CONTRACT: queue or serve in parallel); a deadline from
  the slot (a queued request could pass the contract's 180 s); passing the time through a
  context variable (hidden coupling). The user also asked about one question at a time: not
  recommended (the 4th of 4 would wait ~3 answers, ~112 s with HIGH); one constant if wanted.

## 2026-10-16 The agent wired into the server through an answerer factory
- **Area:** server / agent / ingest
- **Chose:**
  - `create_app(settings, startup, answerer=None, tracer=None, *, answerer_factory=None)`:
    exactly one of `answerer` (ready now; tests) and `answerer_factory: Callable[[Runtime],
    Answerer]` (else `ValueError`). The start-up thread runs the start-up job, then the
    factory with its `Runtime`, inside the `startup` trace (an `answerer` span). Both land in
    one `Ready(runtime, answerer)` set by one reference assignment, so /health is 200 only once
    the answerer exists. A factory that raises fails start-up exactly like the job.
  - `server/__main__.py`: `build_chat(settings)` (`GeminiChat.from_settings`; no key or an
    unreadable ledger: None, logged `the agent is off: ...`) and `build_answerer(runtime, *,
    chat, problem, settings)`, handed over as a `functools.partial`. With a chat model it builds
    `corpus_session_factory(...)` once (0.4-0.6 s on data/, before /health is 200) and returns
    `AgentAnswerer(chat, tools, catalog=tools.catalog, ask_deadline_s=...)`. A Runtime without
    `store` or `chunks` raises.
  - No key: `UnavailableAnswerer(reason)`; every /ask raises `AnswererUnavailable`, mapped to
    the internal-error answer (HTTP 200, no ids), outcome `error`, one log line without a
    traceback. Not the placeholder: its text says answering "is not implemented yet", which is
    false; the internal-error text says no answer was produced, which is true (AGENTS.md: an
    internal error never gives a "not found").
  - `Runtime.store` is typed `IndexReader` (`ingest/startup.py`: `DocumentReader` +
    `CorpusReader`), so the factory gets a `CorpusReader` with no cast.
  - Concurrency: the shared parts are read-only or thread-safe; each question gets a new
    session and a new `Agent`.
- **Tests:** `tests/contract/test_server_agent.py` (factory once, /health waits; a raising
  factory fails start-up; neither/both refused; no chat model -> internal-error answer; the
  real corpus tools through the server; four asks keep their own handles);
  `test_serve_script.py` (no key) now expects the internal-error answer.
- **Over:** building the answerer lazily on the first /ask (CPU work on the loop; the first
  question pays); a /health 200 with an internal-error answerer when the factory fails (it
  hides a broken start-up); the placeholder answer without a key.

## 2026-10-14 uvicorn: no access log, logs via root, 5 s graceful-shutdown cap
- **Area:** server
- **Chose:** `uvicorn.run(app, workers=1, log_config=None, access_log=False, timeout_graceful_shutdown=SHUTDOWN_GRACE_S)`; `logging.basicConfig(stream=sys.stderr)`.
- **Over:** uvicorn's default log config (access log on stdout).
- **Why:** everything on stderr in one format; /health is polled every second for up to 4 h (access log = noise) and /ask logs its own line with trace_id, outcome and timing. Without the cap uvicorn waits for in-flight /ask (up to 150 s) and the grader's SIGKILL lands at 10 s.

## 2026-10-14 Start-up job runs in a daemon thread started by the lifespan
- **Area:** server
- **Chose:** lifespan starts `threading.Thread(daemon=True)` running `StartupFn(settings) -> Runtime`, then yields at once; a `Readiness` object holds the result (single reference writes, no lock).
- **Over:** `asyncio.to_thread` / an asyncio task; running start-up before binding.
- **Why:** the port must bind immediately (uvicorn binds after lifespan start-up). `asyncio.run` joins default-executor threads at exit (up to 300 s in 3.12), so a SIGTERM during a long ingest would hang until SIGKILL; a daemon thread dies with the process. Ingest is sync/CPU-bound anyway.

## 2026-10-14 Runtime dependencies: fastapi + uvicorn (plain, no extras)
- **Area:** server
- **Chose:** `fastapi` (0.143) and `uvicorn` (0.54) without `[standard]`.
- **Over:** a hand-written HTTP layer (stdlib `http.server`/asyncio); `uvicorn[standard]`.
- **Why:** user decision. Typed request/response models, lifespan, TestClient and a production ASGI server for ~0 code; matches the README's "library for plumbing". Extras add speed and reload we do not need at 4 concurrent requests.
