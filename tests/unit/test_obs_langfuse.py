"""The Langfuse backend and init_tracing, offline.

An in-memory OpenTelemetry exporter replaces the HTTP exporter, so these tests see
exactly what would be sent to Langfuse, with no network. Fake keys only. One live
test needs real keys in the environment (`uv run tests-live`); it is skipped otherwise.
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import langfuse
import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from corpusqa import obs
from corpusqa.config import Settings
from corpusqa.obs import _langfuse, init_tracing
from corpusqa.obs._langfuse import LangfuseTracer
from corpusqa.obs.scrub import Scrubber

FAKE_GEMINI = "fake-gemini-" + "g" * 24
FAKE_LF_SECRET = "sk-lf-" + "s" * 24
GOOGLE_SHAPED = "AIza" + "x" * 35
# Nothing is exported over HTTP in these tests; port 9 (discard) is closed anyway.
CLOSED_HOST = "http://127.0.0.1:9"
FLUSH_S = 5.0


def fake_public_key() -> str:
    """A fresh key per tracer: the SDK keeps one client (and exporter) per public key."""
    return "pk-lf-test-" + uuid.uuid4().hex


@pytest.fixture
def exported() -> Iterator[tuple[LangfuseTracer, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    tracer = LangfuseTracer(
        public_key=fake_public_key(),
        secret_key=FAKE_LF_SECRET,
        host=CLOSED_HOST,
        scrubber=Scrubber((FAKE_GEMINI, FAKE_LF_SECRET)),
        span_exporter=exporter,
    )
    yield tracer, exporter
    tracer.shutdown(timeout_s=FLUSH_S)


def finished(tracer: LangfuseTracer, exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    assert tracer.flush(timeout_s=FLUSH_S)
    return list(exporter.get_finished_spans())


def trace_hex(span: ReadableSpan) -> str:
    assert span.context is not None
    return format(span.context.trace_id, "032x")


def parent_of(span: ReadableSpan) -> int | None:
    return None if span.parent is None else span.parent.span_id


def span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def attr(span: ReadableSpan, key: str) -> object:
    return (span.attributes or {}).get(key)


def json_attr(span: ReadableSpan, key: str) -> object:
    value = attr(span, key)
    assert isinstance(value, str), (key, value)
    return json.loads(value)


# ---------------------------------------------------------------- what reaches Langfuse


def test_trace_id_name_io_and_nesting_reach_the_exporter(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported
    trace_id = obs.new_trace_id()
    with tracer.trace("ask", trace_id=trace_id) as root:
        assert root.trace_id == trace_id
        root.update(input="Why?")
        with obs.span("search", input={"q": "why"}, metadata={"hits": 1}) as search:
            search.update(output=["doc-a"])
            with obs.generation(
                "draft",
                model="fake-model",
                input=[{"role": "user", "content": "Why?"}],
                model_parameters={"temperature": 0.0},
                prompt_name="answer",
                prompt_version="3",
            ) as gen:
                gen.update(output="Because.")
                gen.record_usage(input_tokens=10, output_tokens=5, cost_usd=0.002)
        root.update(
            output={"answer": "Because.", "document_ids": ["doc-a"]}, metadata={"outcome": "ok"}
        )

    spans = {span.name: span for span in finished(tracer, exporter)}
    assert set(spans) == {"ask", "search", "draft"}
    ask, search_span, draft = spans["ask"], spans["search"], spans["draft"]
    assert {trace_hex(span) for span in spans.values()} == {trace_id}
    # The root: our trace id, the trace name, and the trace's input/output.
    assert attr(ask, "langfuse.trace.name") == "ask"
    assert attr(ask, "langfuse.internal.as_root") is True
    assert attr(ask, "langfuse.observation.input") == "Why?"
    assert json_attr(ask, "langfuse.observation.output") == {
        "answer": "Because.",
        "document_ids": ["doc-a"],
    }
    assert json_attr(ask, "langfuse.observation.metadata.outcome") == "ok"
    # Nesting follows the OpenTelemetry context.
    assert parent_of(search_span) == span_id(ask)
    assert parent_of(draft) == span_id(search_span)
    assert json_attr(search_span, "langfuse.observation.metadata.hits") == 1
    # The generation's model, parameters, prompt version, usage and cost.
    assert attr(draft, "langfuse.observation.type") == "generation"
    assert attr(draft, "langfuse.observation.model.name") == "fake-model"
    assert json_attr(draft, "langfuse.observation.model.parameters") == {"temperature": 0.0}
    assert attr(draft, "langfuse.version") == "3"
    assert json_attr(draft, "langfuse.observation.metadata.prompt_name") == "answer"
    assert json_attr(draft, "langfuse.observation.usage_details") == {
        "input": 10,
        "output": 5,
        "total": 15,
    }
    assert json_attr(draft, "langfuse.observation.cost_details") == {"total": 0.002}


def test_errors_set_level_message_and_otel_status(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported
    with pytest.raises(RuntimeError), tracer.trace("ask"):
        raise RuntimeError("model call failed")
    [ask] = finished(tracer, exporter)
    assert attr(ask, "langfuse.observation.level") == "ERROR"
    assert attr(ask, "langfuse.observation.status_message") == "RuntimeError: model call failed"
    assert ask.status.status_code is StatusCode.ERROR


def test_concurrent_traces_keep_their_own_spans_across_tasks_and_threads(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported

    def thread_work(i: int, name: str) -> None:
        with obs.span(name, input=f"q{i}"):
            pass

    async def one(i: int) -> str:
        trace_id = obs.new_trace_id()
        with (
            tracer.trace("ask", trace_id=trace_id, input=f"q{i}"),
            obs.span("work", input=f"q{i}"),
        ):
            await asyncio.sleep(0.01)  # the four interleave here
            await asyncio.create_task(asyncio.to_thread(thread_work, i, "to-thread"))
            plain = threading.Thread(target=obs.bind(thread_work), args=(i, "bound-thread"))
            plain.start()
            await asyncio.to_thread(plain.join)
        return trace_id

    async def scenario() -> list[str]:
        return list(await asyncio.gather(*(one(i) for i in range(4))))

    trace_ids = asyncio.run(scenario())
    spans = finished(tracer, exporter)
    assert len(spans) == 16
    for i, trace_id in enumerate(trace_ids):
        mine = {span.name: span for span in spans if trace_hex(span) == trace_id}
        assert set(mine) == {"ask", "work", "to-thread", "bound-thread"}
        assert {attr(span, "langfuse.observation.input") for span in mine.values()} == {f"q{i}"}
        assert parent_of(mine["work"]) == span_id(mine["ask"])
        assert parent_of(mine["to-thread"]) == span_id(mine["work"])
        assert parent_of(mine["bound-thread"]) == span_id(mine["work"])


def test_secrets_never_reach_the_exporter(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported
    with tracer.trace("ask", input=f"Is {FAKE_GEMINI} valid?") as root:
        with obs.generation(
            "draft",
            model="fake-model",
            input={"headers": {"x-goog-api-key": FAKE_GEMINI}, "text": GOOGLE_SHAPED},
            model_parameters={"api_key": FAKE_GEMINI},
            metadata={"note": FAKE_LF_SECRET},
        ) as gen:
            gen.update(output=f"echo {FAKE_LF_SECRET}")
        root.error(f"failed near {GOOGLE_SHAPED}")
    exported_text = repr([dict(span.attributes or {}) for span in finished(tracer, exporter)])
    for secret in (FAKE_GEMINI, FAKE_LF_SECRET, GOOGLE_SHAPED):
        assert secret not in exported_text
    assert obs.REDACTED in exported_text


# ---------------------------------------------------------------- init_tracing


def test_no_keys_means_tracing_off_and_no_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a Langfuse client was constructed without keys")

    monkeypatch.setattr(_langfuse, "Langfuse", no_client)
    monkeypatch.setattr(langfuse, "Langfuse", no_client)
    caplog.set_level(logging.INFO)
    tracer = init_tracing(Settings.from_env({"CORPUS_DIR": str(tmp_path), "PORT": "8080"}))
    assert type(tracer) is obs.NoopTracer
    assert not tracer.enabled
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("corpusqa.obs")]
    assert lines == ["tracing off (no Langfuse keys)"]


@pytest.mark.parametrize("key", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_one_key_alone_keeps_tracing_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, key: str
) -> None:
    def no_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a Langfuse client was constructed with one key")

    monkeypatch.setattr(_langfuse, "Langfuse", no_client)
    caplog.set_level(logging.INFO)
    env = {"CORPUS_DIR": str(tmp_path), "PORT": "8080", key: "only-this-one-key"}
    assert type(init_tracing(Settings.from_env(env))) is obs.NoopTracer
    assert "tracing off (set both" in caplog.text
    assert "only-this-one-key" not in caplog.text


def test_both_keys_turn_tracing_on_and_the_log_shows_no_secrets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    public_key = fake_public_key()
    env = {
        "CORPUS_DIR": str(tmp_path),
        "PORT": "8080",
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": FAKE_LF_SECRET,
        "LANGFUSE_BASE_URL": "http://someone:hunter2pass@127.0.0.1:9/",
    }
    caplog.set_level(logging.INFO)
    tracer = init_tracing(Settings.from_env(env))
    try:
        assert isinstance(tracer, LangfuseTracer)
        lines = [r.getMessage() for r in caplog.records if r.name.startswith("corpusqa.obs")]
        assert lines == ["tracing on: Langfuse at http://127.0.0.1:9"]
        for secret in (public_key, FAKE_LF_SECRET, "hunter2pass"):
            assert secret not in caplog.text  # neither ours nor the SDK's start-up line
    finally:
        assert tracer.shutdown(timeout_s=FLUSH_S)


def test_a_client_that_fails_to_start_leaves_tracing_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken_client(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("cannot start")

    monkeypatch.setattr(_langfuse, "Langfuse", broken_client)
    settings = Settings(
        corpus_dir=tmp_path,
        port=8080,
        langfuse_public_key=fake_public_key(),
        langfuse_secret_key=FAKE_LF_SECRET,
    )
    assert type(init_tracing(settings)) is obs.NoopTracer
    assert "tracing off (the Langfuse client failed to start: RuntimeError)" in caplog.text


# ---------------------------------------------------------------- live (real keys only)


@pytest.mark.live
@pytest.mark.skipif(
    not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")),
    reason="needs real LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
)
def test_live_one_trace_is_delivered_to_langfuse(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    env = {k: v for k, v in os.environ.items() if k.startswith("LANGFUSE_")}
    tracer = init_tracing(Settings.from_env({"CORPUS_DIR": str(tmp_path), "PORT": "8080", **env}))
    assert tracer.enabled
    with tracer.trace("live-check", trace_id=obs.new_trace_id(), input="ping") as root:
        with obs.span("child") as child:
            child.update(output="pong")
        root.update(output="done", metadata={"test": "test_obs_langfuse"})
    assert tracer.shutdown(timeout_s=30), "flush did not finish in time"
    # The OTLP exporter logs "Failed to export ..." on a refused or failed export.
    assert "Failed to export" not in caplog.text, caplog.text
