---
name: Tracing decisions
description: Decisions about Langfuse tracing: Langfuse Cloud (US region) for dev instead of a local stack, with the US host as the config default; obs facade with no-op/Langfuse/recording backends, own TracerProvider, bounded flush/shutdown, trace name and IO on the root observation, prompt version in metadata, trace level per outcome, the import contract with indirect imports allowed, FastAPI's native telemetry off.
source: runs/2026-10-15-langfuse-tracing, runs/2026-10-16-langfuse-cloud-us, runs/2026-10-14-ai-dev-setup
---

# Tracing decisions

## 2026-10-16 The US cloud region is the default Langfuse host
- **Area:** tracing
- **Chose:** `LANGFUSE_DEFAULT_HOST = "https://us.cloud.langfuse.com"` in `config.py`
  (user gave this URL); `LANGFUSE_BASE_URL` / `LANGFUSE_HOST` still override it.
- **Over:** a `LANGFUSE_BASE_URL` line in `.env`, or `env` defaults on each dev command in
  `commands.json`.
- **Why:** one place, every path (server, evals, live test) gets it, and agents cannot
  write `.env`. The grader sets no keys, so the default is dev only.

## 2026-10-16 Langfuse Cloud instead of a local self-hosted stack (user decision)
- **Area:** tracing
- **Chose:** the user's Langfuse Cloud project (Hobby plan, US region); keys in `.env`.
- **Over:** the official docker-compose stack (web, worker, Postgres, ClickHouse, Redis,
  MinIO) in the repo with `langfuse-up/down` commands.
- **Why:** nothing to build (the code already targets the cloud), no Docker RAM, the free
  tier covers this project's volume, the trace project can be shared with a grader, and
  grading is identical either way (no keys = tracing off). Local stays the option if the
  corpus text must not leave the machine.

## 2026-10-15 FastAPI's native OpenTelemetry is switched off
- **Area:** tracing
- **Chose:** `FastAPI(..., telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False})` in `create_app`.
- **Over:** FastAPI 0.143's default, which turns itself on when any global OTel provider exists.
- **Why:** obs must be the only tracing path and /health must never be traced. A test fails without the setting.

## 2026-10-15 Import contract allows indirect imports of langfuse
- **Area:** tracing
- **Chose:** forbidden contract "Langfuse SDK and OpenTelemetry only in obs" with `allow_indirect_imports = true`; sources include config and devtasks.
- **Over:** the default (indirect imports forbidden).
- **Why:** the default flags the allowed chain `server -> obs -> obs._langfuse -> langfuse`. Direct imports are still caught (checked with a probe file).

## 2026-10-15 ask trace level per outcome
- **Area:** tracing
- **Chose:** ERROR for timeout / error / empty_answer; WARNING for invalid_request / not_ready; DEFAULT for ok, with a message saying what went wrong.
- **Over:** ERROR for every non-ok outcome.
- **Why:** a bad request or an early question is the caller's problem; Langfuse error filters then show real failures only.

## 2026-10-15 Prompt name and version go in metadata and `version`
- **Area:** tracing
- **Chose:** `obs.generation(prompt_name=, prompt_version=)` writes `metadata.prompt_name/prompt_version` and the observation's `version`.
- **Over:** the SDK's `prompt=` argument.
- **Why:** `prompt=` expects a Langfuse-managed `PromptClient`; our prompts are versioned files.

## 2026-10-15 Trace name and IO on the root observation
- **Area:** tracing
- **Chose:** root = `start_as_current_observation(as_type="span", trace_context={"trace_id": ours})` + `propagate_attributes(trace_name=...)`; trace input/output = the root's input/output.
- **Over:** a trace object / `update_trace` (v3 API).
- **Why:** Langfuse 5 has no trace object; a trace's name and input/output live on its root observation (client.py docstring).

## 2026-10-15 Bounded flush and shutdown in a daemon thread
- **Area:** tracing
- **Chose:** `Tracer.flush(timeout_s)` / `shutdown(timeout_s)` run the SDK call in a daemon thread with a bounded wait; lifespan exit calls `shutdown(2.0)` only when tracing is on.
- **Over:** calling `Langfuse.flush()` / `shutdown()` directly.
- **Why:** both block with no timeout, which could break the grader's 10 s SIGTERM → SIGKILL window with a slow host.

## 2026-10-15 Langfuse gets its own TracerProvider
- **Area:** tracing
- **Chose:** `Langfuse(tracer_provider=TracerProvider(shutdown_on_exit=False))`.
- **Over:** the SDK default, which sets a process-global provider.
- **Why:** no global side effects (FastAPI and other libraries keep a no-op provider), no OTel atexit hook that could stall exit.

## 2026-10-15 obs = facade + backends; tests use the same facade
- **Area:** tracing
- **Chose:** `obs.Tracer` facade (guards every backend call, scrubs every value, contextvar for the current observation) with NoopTracer, LangfuseTracer (lazy, only with both keys) and RecordingTracer.
- **Over:** using the SDK's `@observe` / `get_client()` across the code.
- **Why:** one place for the safety and secret rules; tests exercise the same scrub/guard path as production; the no-key path never imports the SDK.

## 2026-10-15 Langfuse for all tracing (user decision)
- **Area:** tracing
- **Chose:** Langfuse 5 via `corpusqa.obs`; AGENTS.md rule "if it runs, it is traced".
- **Over:** an own JSON tracer with OpenTelemetry field names.
- **Why:** the user wants an existing tool and every prompt and tool call traced thoroughly.
