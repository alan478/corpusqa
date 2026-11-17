---
name: Tracing gotchas
description: Traps with Langfuse 5, Langfuse Cloud and OpenTelemetry: the legacy read API (/api/public/traces) returns 410 on new orgs and the Hobby plan rate-limits the public API, FastAPI picks up OTel by itself, one Langfuse client per public key per process, flush/shutdown have no timeout, and the SDK logs the public key at INFO; the v2 Observations API returns input and output as JSON strings; no project command reads a trace back yet (a read-only test file through uv run tests-live does).
source: runs/2026-10-15-langfuse-tracing, runs/2026-10-16-langfuse-cloud-us, runs/2026-10-16-agent-tools
---

# Tracing gotchas

## 2026-10-16 Langfuse's v2 Observations API returns `input` as a JSON string
- **Area:** tracing
- **Symptom:** checking that each `chat` generation's input held the system prompt and the
  tool definitions found no dict at all (`input_keys: None` on all 16 generations).
- **Cause:** the v2 API sends `input` (and `output`) as a JSON-encoded string, not an object.
- **Fix:** `json.loads` the string first; then `contents` and `config` (with
  `system_instruction`, `tools`, `tool_config`, `thinking_config`) were all there.
- **Avoid next time:** any trace reader must decode `input` / `output`. The v2 API also takes
  `name=` and `type=` without `traceId=` (architecture/tracing.md).

## 2026-10-16 No project command reads a trace back from Langfuse yet
- **Area:** tracing / tooling
- **Symptom:** the read-back needs the Langfuse keys, which live only in `.env` (agents may
  not read it); no project command reads traces (a `uv run trace` key was planned, never
  built, and removed on 2026-10-17).
- **Fix:** a read-only pytest file in the scratchpad, run through the project command that
  loads `.env` itself: `LF_TRACE_ID=... LF_FROM=... LF_TO=... uv run tests-live
  <scratchpad>/test_lf_readback.py -p no:cacheprovider`. It writes a filtered summary
  (metadata `scope` left out: it holds the public key), checked for key shapes after.
- **Avoid next time:** use this or the Langfuse UI.


- **Area:** tracing
- **Symptom:** fetching an `ask` trace by id from the US cloud gave HTTP 410 with
  `LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`; two of nine tries gave 429 (rate limit)
  at one request per 10 s.
- **Cause:** orgs created on or after 2026-09-16 have no legacy read APIs; Langfuse v4 reads
  span data only through `GET /api/public/v2/observations?fromStartTime=&toStartTime=`
  (filter by trace id). Legacy endpoints go away for everyone on 2026-11-16.
- **Fix:** query the v2 Observations API with a time window and the trace id.
- **Avoid next time:** any trace-reading tool, if one is built, must use the v2
  Observations API, never `/api/public/traces`; keep public-API calls well under one per
  few seconds on the Hobby plan.

## 2026-10-15 FastAPI 0.143 imports OpenTelemetry and can trace by itself
- **Area:** tracing
- **Symptom:** `opentelemetry` appeared in `sys.modules` on the no-key path, although obs imports it lazily.
- **Cause:** `fastapi.telemetry` imports it when installed and enables request spans once a global provider is set.
- **Fix:** native telemetry switched off in `create_app`; Langfuse uses its own provider, never the global.
- **Avoid next time:** after adding an OTel-based dependency, check what else picks it up.

## 2026-10-15 Langfuse keeps one client per public key for the whole process
- **Area:** tracing
- **Symptom:** (avoided) a second `Langfuse(...)` with the same public key silently reuses the first one's provider, exporter and host.
- **Cause:** `LangfuseResourceManager.__new__` is a singleton keyed by public key.
- **Fix:** each test tracer uses a fresh fake public key (`pk-lf-test-<uuid>`).
- **Avoid next time:** never build two Langfuse tracers with one key in a process.

## 2026-10-15 Langfuse flush/shutdown have no timeout
- **Area:** tracing
- **Symptom:** (from the source) `Langfuse.flush()` = `force_flush()` + unbounded queue joins; OTel's BatchProcessor ignores `timeout_millis`.
- **Cause:** SDK design.
- **Fix:** facade runs them in a daemon thread with a bounded wait; a test with a silent host shows /ask < 0.5 s and shutdown at the 2 s bound.
- **Avoid next time:** never call SDK flush on a request path or without a bound.

## 2026-10-15 Langfuse SDK logs the public key at INFO
- **Area:** tracing
- **Symptom:** "Startup: Langfuse tracer successfully initialized | public_key=..." in the server log.
- **Cause:** `resource_manager._initialize_instance` logs it.
- **Fix:** the `langfuse` logger is set to WARNING; our own line says "tracing on: Langfuse at <host>".
- **Avoid next time:** check third-party INFO logs for keys when adding an SDK.
