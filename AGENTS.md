# Corpus QA service — rules for coding agents

Applies to Claude Code and Codex. `CLAUDE.md` imports this file. The global
harness rules (handoffs, decision and error logs, close-out, code quality) also apply.

## What this is
An HTTP service that answers questions over a company's records. `./serve` ingests the corpus at
`CORPUS_DIR`; `POST /ask` returns an answer and the `dataset_doc_uuid`s it
relies on. Interface: `SOLUTION.md` section 2. Design as built: `docs/knowledge/architecture/`
(one file per part: server, ingestion, agent, eval, tracing).
It is graded on an UNSEEN corpus: everything must work for any corpus of the
same format.

<!-- split-rule:start -->
## Eval split — never break it (user decision)
The 79 dev questions are split once, in `eval/sets/` (made by
`scripts/make_eval_split.py`, seed 20261014):
- **tune (30)**: the ONLY questions you may read, run, debug and tune on. The
  smoke set (10) is inside it. Read them with `uv run questions-tune`
  (`--id pub_XXXX` for one).
- **heldout (49)**: the test set. Never read its questions, gold answers,
  expected ids, traces or per-question results. Never change code, prompts,
  tools or settings because of a held-out result.
- Run held-out only when the user names a milestone:
  `uv run eval-heldout --milestone <name>`. Report only its aggregate scores.
- Never edit, regenerate or reshuffle the split. Never copy a question, answer
  or expected id into code, prompts or tests.

Enforced by hooks: agent reads of `data/questions.jsonl`, `eval/sets/heldout.json`
and `eval/results/heldout/` are blocked; edits to `eval/sets/` are blocked;
this rule is shown again at every session start and after every compaction.
<!-- split-rule:end -->

## Commands
All commands live in `commands.json`, one key each. Run one with
`uv run <key> [extra args]`; `uv run commands` lists them with descriptions.
Use these keys, not raw tool commands. To add a command: add the key to
`commands.json` AND to `[project.scripts]` in `pyproject.toml` (a test checks
both match). Most used:

| Key | What it does |
|---|---|
| `setup` | First time: install deps and turn on git hooks |
| `check` | Every check in order; run before saying done |
| `fmt` | Format + safe lint fixes |
| `tests` / `tests-unit` | All offline tests / unit tests only |
| `serve-dev` | Run the service on `data/corpus` at port 8080 |
| `eval-smoke`, `eval-tune` | Eval against the running server (tuning sets) |
| `eval-heldout` | Held-out test set; only at a milestone the user names |
| `eval-judge-calibrate` | Check the answer judge against the hand-checked tune runs |

No command reads a trace back: open it in Langfuse
(`docs/knowledge/architecture/tracing.md`).

## Code map
```
src/corpusqa/
  server/      HTTP only (FastAPI + uvicorn); calls agent
    app.py       create_app(settings, startup, answerer | answerer_factory, tracer): lifespan,
                 /health, /ask; AskQueue: 4 answers at once, the rest wait in line
    respond.py   last-line guard: known ids only, deduped; answer non-empty, size-capped
    models.py    wire models (AskRequest, AskResponse: answer + document_ids only,
                 HealthResponse); the trace id goes in the X-Trace-Id header
    __main__.py  `python -m corpusqa.server` (what ./serve execs): the ONLY wiring point;
                 builds GeminiChat, then the AgentAnswerer once start-up returns
  agent/       the agent: loop, tools, handles, citations
    agent.py     Agent (one per question): history, first search, budget note, stop
                 (finish or a limit), forced finish, deadline, compaction
    tools.py     Tool, ToolBox (errors come back as text; a ToolInputError's message
                 reaches the model), finish
    corpus_tools.py  the 7 corpus tools + CorpusToolSession (one per request)
    handles.py   D# / R# handles; citations -> ids; handles in the answer text named
    session.py   ToolSession protocol; answerer.py: AgentAnswerer, UnavailableAnswerer
  answer/      empty (citations are resolved in agent/)
  retrieval/   the tools' logic over store interfaces: search (scoped hybrid), filtering,
               values, fields, keys, documents, paging, errors (ToolInputError);
               hybrid.py: BM25 + dense, RRF, chunks -> documents
  store/       repository interfaces + SQLite; the ONLY place with SQL
               vectors.py: the only numpy code (vector matrix, cosine top-k); ingest
               builds it at start-up, retrieval searches it
  ingest/      S1-S10 pipeline; writes through store interfaces
               startup.py: StartupFn (Settings -> Runtime) + placeholder_startup
               embed.py: S9 (embedding cache in .cache/embeddings, time box, spend cap,
               resume) and loading the vectors into the Runtime
  llm/         client interface, Gemini adapters, fakes, limiter, spend ledger
               client.py: GeminiEmbedder, GeminiChat (agent and judge chat model)
               interfaces.py: Embedder, ChatModel (SDK-free; other layers type against them)
  obs/         tracing: the only package that imports langfuse / opentelemetry
               tracer.py: facade (trace/span/generation, observe, bind, current), NoopTracer
               scrub.py: secrets out of every traced value; bootstrap.py: init_tracing
               _langfuse.py: Langfuse backend; testing.py: RecordingTracer for tests
  prompts/     versioned prompt files: agent.md (v7), question.md, compact.md, judge.md,
               embed_roles.txt, tools.json (v6); why each changed: docs/prompt-log.md
  config.py    Settings from env; models, caps, budgets, fixed answer texts
               (no magic numbers elsewhere)
serve          the CONTRACT start command: uv sync --frozen --no-dev --inexact, exec server
commands.json  every command, by key (`uv run <key>`); runner: src/corpusqa/devtasks.py
scripts/       scan_secrets.py, make_eval_split.py, hooks/ (agent hooks)
tests/         unit/, contract/, fixtures/
eval/          run.py (the /ask eval), judge.py (answer judge), judge_calibrate.py,
               sets/ (smoke, tune, heldout ids), results/, history.csv
               retrieval.py: `uv run eval-retrieval` (chunking variants, caches in .cache/eval/;
               `--production` scores the index start-up built)
```

## Boundaries (enforced by `lint-imports`)
- Imports go down only: server -> agent -> answer | retrieval -> store.
  `ingest` uses `store` only. `llm`, `obs` and `config` may be used by all.
- No SQL or `sqlite3` outside `store/`. No Gemini SDK (`google.genai`) outside `llm/`.
  No Langfuse SDK or OpenTelemetry (`langfuse`, `opentelemetry`) outside `obs/`.
- Every LLM and embedding call goes through `llm.client`: rate limiter, spend
  ledger and a Langfuse generation. Never call the SDK directly.
- Only `server/__main__.py` imports `llm.client`: it builds the Gemini clients and
  hands them down. Every other layer types against `llm.interfaces` (the contract
  ignores exactly that one import).

## Hard rules (the grader checks these)
- Never hard-code anything from `data/`: names, companies, issue keys, doc ids,
  source names, field names. Learn them at ingest.
  (`tests/unit/test_no_hardcoding.py` enforces the keys, ids and names.)
- `/ask` always returns HTTP 200 with a non-empty answer. An internal error
  gives the internal-error answer, never a "not found" answer.
  Enforced in `server/app.py`: `/ask` reads its own body (no FastAPI body
  param, whose 422/400 errors fire before the handler) and one try block
  maps bad input, not-ready, timeout (`asyncio.timeout_at`, counted from
  arrival) and any exception
  to a fixed answer from `config.py`. Every body goes through
  `server/respond.build_response`. `tests/contract/test_server_contract.py`
  checks each path.
- `document_ids` hold only ids from the ingest id map (`Runtime.known_doc_ids`;
  `build_response` drops and logs anything else).
- `/health` is 503 until the start-up job (a daemon thread started by the
  lifespan, so the port binds at once) returns a `Runtime`; it stays 503 if
  start-up raises.
- 150 s internal deadline per request; up to 4 requests at once; no state
  shared between requests.
- Read only `CORPUS_DIR`, `PORT`, `GEMINI_API_KEY`, `PATH`, `HOME`, `TMPDIR`,
  `LANG` (plus the optional dev-only `LANGFUSE_*` keys: unset at grading, so
  tracing is off). Never write to `CORPUS_DIR`. Caches go in `.cache/`.
- `serve` stays in the foreground: no daemon, no `setsid`; it `exec`s the server.

## Using the LLM
- Code does what code can do exactly: counting, filtering, id lookup,
  citation lists. The model decides what to look up and writes the answer.
- Prompts and tool schemas live in `src/corpusqa/prompts/` with a `version:`
  line. Any text change bumps the version (skill `prompt-change`).
- Structured output is parsed into Pydantic models. Invalid output: retry
  once, then the fallback. Never regex-parse free text for structure.
- Every call sets `max_output_tokens` and a thinking budget (or level) from `config.py`.

## Tests
- Unit tests never touch the network: use `FakeLLM` (scripted tool calls)
  and `FakeEmbedder`.
- Fixtures use `tests/fixtures/mini_corpus/` with invented names AND invented
  field names.
- Every bug fix adds a test that failed before the fix.
- Real-API tests are marked `@pytest.mark.live` and skipped by default.
- `tests/contract/test_serve_script.py` (marker `serve`) runs `./serve` with the
  grader's minimal env and SIGTERMs its process group; skip with `-m "not serve"`.
- Never delete, skip or loosen a test to make it pass.

## Evals
- Any change to prompts, tools, retrieval, chunking or caps: run the smoke
  eval (skill `run-eval`) and keep its row in `eval/history.csv`.
- Work only on the tune set (30). Held-out (49) follows **Eval split — never
  break it** at the top of this file.
- Keep a change only if the tune set does not regress. Name any regression in
  the handoff.
- Debug a wrong answer from its failure label and trace (skill `debug-trace`).
  Never special-case one question.

## Tracing — everything is traced, in Langfuse (user decision)
We use **Langfuse** for tracing; we do not build our own tracer. The rule:
**if it runs, it is traced.** Work that adds an LLM call, a tool or a stage
without its trace is not done.
- **One trace per `/ask`.** The response's `X-Trace-Id` header is the Langfuse
  trace id, so any answer can be opened in Langfuse. The body is only CONTRACT
  §3's `answer` and `document_ids` (user, 2026-10-17).
- **Every LLM and embedding call is a Langfuse generation:** model, the full
  input exactly as sent (system prompt, messages, tool definitions), the
  output and any tool calls, tokens, cost, latency, retries and errors.
- **Every agent tool call is a span:** arguments as given and as normalized,
  internals (search scores, filter matches, ids resolved), and the result
  exactly as the model saw it.
- **Every pipeline stage is a span:** the agent's turns, citation resolution (each
  citation kept or dropped, and why), and each ingestion stage (counts, files
  set aside with reasons, what the profile learned).
- **Prompts are named and versioned** (the `version:` line) and the version is
  recorded on every generation that uses them.
- **One place talks to Langfuse: `corpusqa.obs`.** Other code uses its
  helpers (decorator / context managers), never the Langfuse SDK directly.
- **Tracing never breaks or slows a request.** The grader sets no Langfuse
  keys (CONTRACT: only CORPUS_DIR, PORT, GEMINI_API_KEY): with no keys, tracing
  is off, makes no network calls and costs ~nothing. Sending is in the
  background; a tracing error is logged and swallowed, never raised.
- **Never put secrets in a trace** (API keys, auth headers). Corpus text is fine.
- **Tests check tracing:** a fake tracer records spans; each new LLM call,
  tool or stage gets a test that its span exists with the required fields.

How to trace (`from corpusqa import obs`): wrap work in `with obs.span(name,
input=..., metadata=...) as s:` and call `s.update(output=...)`; wrap every
LLM/embedding call in `with obs.generation(name, model=..., input=...,
model_parameters=..., prompt_name=..., prompt_version=...) as g:` then
`g.update(output=...)` and `g.record_usage(input_tokens=..., output_tokens=...,
cost_usd=...)`; or decorate a function (sync or async) with `@obs.observe()`.
New observations nest under the current one (contextvars: works in asyncio tasks
and `asyncio.to_thread`); wrap a plain thread's target in `obs.bind(fn)`. Mark a
failure with `s.error(msg_or_exc)`; an exception leaving a `with` marks it ERROR
and still propagates. Every value is scrubbed of secrets first. Outside a trace,
or with tracing off, the helpers are no-ops. In tests, pass
`RecordingTracer.for_settings(settings)` (from `corpusqa.obs.testing`) as
`create_app(..., tracer=...)` and assert on `tracer.traces("ask")` /
`tracer.children(...)`.

## Logs
- No `print()` in `src/` (ruff `T20`); use the logger. Never log the API key.

## Python conventions
- Python 3.12. Type hints on every function; `pyright` passes on `src/`.
- `ruff format` and `ruff check` pass. Line length 100.
- Data crossing a module boundary is a dataclass or Pydantic model, not a dict.
- Async server: no blocking I/O in the event loop (ruff `ASYNC`). SQLite and
  CPU work run in a thread pool.
- Open files with `encoding="utf-8"`. Use `pathlib` (ruff `PTH`).
- Ingest quarantines bad input with a reason and continues. Query code never
  raises to the client.

## Dependencies
- Add with `uv add <pkg>` (dev tools: `uv add --dev`). Commit `pyproject.toml`
  and `uv.lock` together. Never edit `uv.lock` by hand (a hook blocks it).
- Each new runtime dependency gets a `decisions.md` entry: why, and what it replaces.

## Hooks (what runs automatically)
- Claude Code (`.claude/settings.json`) and Codex (`.codex/hooks.json`) run the
  same scripts in `scripts/hooks/`: block hand edits to `data/`, `uv.lock`
  and existing eval sets/history; format + lint each edited Python file; run
  the unit tests when the agent stops (asks once to fix failures).
- Git pre-commit (`.githooks/`, on after `uv run setup`): runs `uv run precommit`
  (format, lint, import boundaries, unit tests, secret scan).
- If a hook blocks you, fix the cause. Never work around it.

## Before you say done
1. `uv run check` passes; paste its last lines into the handoff.
2. Every new LLM call, tool or stage is traced (see Tracing) and has a test
   that its span exists.
3. If behaviour changed: the eval row is in `eval/history.csv`.
4. Big change (new module, agent loop, ingest): Codex review (skill `codex-review`).

## Skills (`docs/skills/`)
`run-eval`, `debug-trace`, `prompt-change`, `add-agent-tool`, `codex-review`.
