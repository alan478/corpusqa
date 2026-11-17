---
name: Tracing (Langfuse) — how corpusqa.obs works
description: The tracing layer: obs facade with three backends (no-op, Langfuse, recording for tests), how /ask, the agent (agent, first_search, step, chat, tool, compact, citations, answer_handles) and start-up are traced, how to trace new code (trace, span, generation, observe), the safety rules (off without keys, never breaks a request, secrets scrubbed, bounded flush), the dev setup against Langfuse Cloud US (verified live 2026-10-16) and how to read a trace back through the v2 Observations API. Read before adding an LLM call, a tool or a pipeline stage, or when looking up a trace.
source: runs/2026-10-15-langfuse-tracing, runs/2026-10-14-ingestion, runs/2026-10-16-langfuse-cloud-us, runs/2026-10-16-agent-tools
---

# Tracing (Langfuse) — `corpusqa.obs`

User decision: Langfuse for all tracing; AGENTS.md rule "if it runs, it is traced".

## Parts
| File | Role |
|---|---|
| `obs/tracer.py` | The facade every module uses: `trace`, `span`, `generation`, `@observe` (sync + async), `bind`, `current`, `flush`, `shutdown`; `NoopTracer`. Guards every backend call (a failure is logged once per kind and swallowed; exceptions from the traced code still propagate and mark the observation ERROR). Keeps the current observation in a contextvar, so 4 parallel requests never mix. |
| `obs/scrub.py` | Removes configured keys, common key shapes and values of secret-named fields from everything before it is traced. |
| `obs/bootstrap.py` | `init_tracing(settings)`: Langfuse only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set (host: `LANGFUSE_BASE_URL` > `LANGFUSE_HOST` > `LANGFUSE_DEFAULT_HOST`, the US cloud); else the no-op tracer, which never imports the SDK. Logs one line: `tracing off (no Langfuse keys)` or `tracing on: Langfuse at <host>`. |
| `obs/_langfuse.py` | The only file importing `langfuse` / `opentelemetry` (import-linter contract). Own `TracerProvider` (never the global one). |
| `obs/testing.py` | `RecordingTracer`: in-memory, same facade, same scrub and guard path. |

## What is traced today
- **/ask:** one root observation per request, trace id = the response's `X-Trace-Id` header (uuid4 hex = valid OTel trace id; the body holds only `answer` and `document_ids`). Name `ask`; input = question; output = answer + ids; metadata = outcome, latency, counts, the queue (`queue_limit`, `queue_running_on_arrival`, `queue_waiting_on_arrival`, `queue_wait_ms`, `queue_got_slot`) and, when the id list was cut, `document_ids_cited/cut/cap` (WARNING). Level: ERROR for timeout / error / empty_answer; WARNING for invalid_request / not_ready; DEFAULT for ok. The answerer runs inside it, so its spans nest.
- **The agent, under `ask`** (`agent.md`, Tracing): `agent` (prompt versions, tools, limits, catalog size; at the end stop, forced_by, first_search, steps, tool calls, compactions, tokens, cost) > `first_search` (its `tool:search` under it), one `step` per model call (budget note, estimated tokens) with its `chat` generation (model, the contents and config exactly as sent, tool calls, usage, cost, retries, `thinking_level` or budget, prompt name and version) and its `tool:<name>` spans (arguments given and checked, the result as the model saw it, handles; inside `tool:search` the query's `embed` generation and `search` > `bm25`, `dense`; inside the others `filter`, `read_fields`, `lookup_key`, ...), and one `compact` per compaction. Then `citations` (citations given, ids resolved) and `answer_handles` (handles named in the answer text) under `ask`.
- **Start-up:** a `startup` trace (function name, duration, known-id count, answerer class, error). Ingestion nests under it: an `ingest` span with one span per stage and one `embed` generation per embedding request (see `ingestion.md`, Tracing); then an `answerer` span for building the answerer (catalog size, documents, dense search on/off).
- **The eval runner** (a client): one `eval_ask` trace per run, and the judge's `judge_all` (see `eval.md`, Tracing).
- **Not traced:** `/health` (polled every second). FastAPI's own OpenTelemetry is switched off in `create_app`.
- **Shutdown:** lifespan exit calls `shutdown(2.0 s)` only when tracing is on; it never waits longer.

## How to trace new code
- A tool or stage: `with tracer.span("name", input=..., metadata=...) as s: ...; s.update(output=...)`, or decorate with `@observe`.
- An LLM / embedding call: `tracer.generation(name, model=, input=, model_parameters=, prompt_name=, prompt_version=)` then `update(output=, usage_details={"input", "output", "total"}, cost_details={"total"})`. Prompt name/version go to metadata and the observation `version` (not the SDK's `prompt=`, which needs Langfuse-managed prompts).
- Tests: build the app or component with `RecordingTracer` and assert the span exists with its required fields.

## Verified against Langfuse Cloud (2026-10-16)
- `uv run tests-live tests/unit/test_obs_langfuse.py`: one trace delivered, no "Failed to export".
- `uv run serve-dev` + one `POST /ask`: log `tracing on: Langfuse at https://us.cloud.langfuse.com`,
  0 export failures, SIGTERM stop in 1 s. The `ask` observation read back from the API had
  input = question, output = answer + ids, metadata outcome / latency_ms / answer_chars /
  document_ids, `isRootObservation: true`, level DEFAULT. Its `parentObservationId` is a
  synthetic id the SDK derives from the trace id when given `trace_context`; harmless.
- Not set yet: Langfuse `environment` (shows `default`); observation types other than
  span / generation (`agent`, `tool`, `retriever`, wanted by Langfuse's best-practices page
  for the agent step).

## Dev setup (Langfuse Cloud, US region)
- The project's Langfuse Cloud project is in the US region; `config.LANGFUSE_DEFAULT_HOST`
  is `https://us.cloud.langfuse.com`, so no host line is needed in `.env`.
- Keys: `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` (git-ignored; agents can
  neither read nor write it, the user maintains it). `uv run serve-dev`, every `eval-*`
  key and `tests-live` load `.env` before running; `./serve` never reads it.
- The grader sets no keys: tracing off, SDK never imported.

## Reading a trace back (API)
`GET /api/public/traces/{id}` is a legacy API: it returns 410 for organisations created on or
after 2026-09-16 and goes away for everyone on 2026-11-16 (gotchas/tracing.md). Use the v2
Observations API, basic auth = public:secret key:
```
GET {host}/api/public/v2/observations?traceId=<trace_id>&fromStartTime=<iso>&toStartTime=<iso>
    &fields=core,basic,io,metadata&limit=50
```
Rows come newest first; `meta.cursor` pages. `input` / `output` arrive only with the `io`
field group, as JSON-encoded strings (decode them before reading keys); `metadata` only
with `metadata` (values cut at 200 chars unless named in `expandMetadata`). `name=` and
`type=` work without `traceId=`: `name=agent&type=SPAN&fields=core,basic,metadata` over a
run's time window gives every question's `agent` span (steps, stop, cost) in one call;
`fields=...,model,usage` adds model, `usageDetails` and `costDetails`.

No project command reads traces (a `uv run trace` key was planned, never built, and
removed on 2026-10-17). What worked (2026-10-16/17): a read-only pytest file in the session scratchpad,
run with `uv run tests-live <file> -p no:cacheprovider` (that key loads `.env`, which
agents may not read), writing a filtered summary without `metadata.scope`. Observation types the filter knows: GENERATION, SPAN, EVENT, AGENT, TOOL,
CHAIN, RETRIEVER, EVALUATOR, EMBEDDING, GUARDRAIL. The Hobby plan rate-limits the public
API: one request per 10 s already produced 429s. The SDK writes the public key into
`metadata.scope.attributes.public_key`: do not print metadata unfiltered.
