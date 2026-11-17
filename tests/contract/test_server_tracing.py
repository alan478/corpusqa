"""Tracing of the HTTP layer (AGENTS.md: Tracing), with the in-memory RecordingTracer.

One `ask` trace per /ask whose id is the response's `X-Trace-Id` header, every outcome traced
with its level, the answerer's spans nested inside it, a `startup` trace, no trace
for /health, a bounded flush on shutdown, no secrets in any trace, and a broken or
unreachable tracer never breaking /ask. Fake keys only.
"""

import asyncio
import dataclasses
import logging
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from corpusqa import obs
from corpusqa.agent import AnswerResult
from corpusqa.config import (
    ASK_MAX_CONCURRENT,
    INTERNAL_ERROR_ANSWER,
    INVALID_REQUEST_ANSWER,
    NOT_READY_ANSWER,
    TIMEOUT_ANSWER,
    TRACE_SHUTDOWN_TIMEOUT_S,
    Settings,
)
from corpusqa.ingest import Runtime, StartupFn
from corpusqa.obs._langfuse import LangfuseTracer
from corpusqa.obs.scrub import REDACTED, Scrubber
from corpusqa.obs.testing import RecordedObservation, RecordingTracer
from corpusqa.server import create_app
from corpusqa.server.app import ANSWERER_SPAN
from tests.contract.test_server_contract import (
    KNOWN,
    WAIT_S,
    FnAnswerer,
    ask_at_once,
    assert_valid_body,
    ready_with,
    returns,
    trace_id_of,
    wait_until_ready,
)
from tests.unit.test_obs import BrokenTracer

FAKE_GEMINI = "fake-gemini-" + "g" * 24
FAKE_LF_PUBLIC = "pk-lf-" + "p" * 24
FAKE_LF_SECRET = "sk-lf-" + "s" * 24
GOOGLE_SHAPED = "AIza" + "x" * 35


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        corpus_dir=tmp_path,
        port=8080,
        gemini_api_key=FAKE_GEMINI,
        langfuse_public_key=FAKE_LF_PUBLIC,
        langfuse_secret_key=FAKE_LF_SECRET,
    )


@pytest.fixture
def tracer(settings: Settings) -> RecordingTracer:
    return RecordingTracer.for_settings(settings)


@contextmanager
def serving(
    settings: Settings,
    answerer: FnAnswerer,
    tracer: obs.Tracer,
    startup: StartupFn | None = None,
    wait: bool = True,
) -> Iterator[TestClient]:
    app = create_app(settings, startup or ready_with(), answerer, tracer=tracer)
    with TestClient(app) as client:
        if wait:
            wait_until_ready(client)
        yield client


def only_ask(tracer: RecordingTracer) -> RecordedObservation:
    [ask] = tracer.traces("ask")
    return ask


# ---------------------------------------------------------------- /ask


def test_each_ask_is_one_trace_with_the_response_trace_id(
    settings: Settings, tracer: RecordingTracer
) -> None:
    with serving(settings, returns("It is blue.", ["doc-a", "dsid_unknown"]), tracer) as client:
        response = client.post("/ask", json={"question": "  What colour?  "})
    assert_valid_body(response)
    ask = only_ask(tracer)
    assert (ask.kind, ask.trace_id, ask.parent_id, ask.ended) == (
        "trace",
        trace_id_of(response),
        None,
        True,
    )
    assert ask.input == "What colour?"
    assert ask.output == {"answer": "It is blue.", "document_ids": ["doc-a"]}
    assert ask.metadata["outcome"] == "ok"
    assert ask.metadata["document_ids"] == 1
    assert ask.metadata["answer_chars"] == len("It is blue.")
    latency = ask.metadata["latency_ms"]
    assert isinstance(latency, int) and 0 <= latency < WAIT_S * 1000
    assert (ask.level, ask.status_message) == ("DEFAULT", None)


def test_answerer_spans_nest_under_the_ask_trace(
    settings: Settings, tracer: RecordingTracer
) -> None:
    async def traced(question: str) -> AnswerResult:
        with obs.span("lookup", input={"question": question}) as lookup:
            lookup.update(output=["doc-a"])
        with obs.generation(
            "draft", model="fake-model", input=question, prompt_name="answer", prompt_version="3"
        ) as gen:
            gen.record_usage(input_tokens=10, output_tokens=5)
        return AnswerResult(answer="Traced.", document_ids=["doc-a"])

    with serving(settings, FnAnswerer(traced), tracer) as client:
        response = client.post("/ask", json={"question": "Nest?"})
    assert_valid_body(response)
    trace_id = trace_id_of(response)
    ask = only_ask(tracer)
    lookup, draft = tracer.children(ask)
    assert (lookup.name, lookup.trace_id, lookup.output) == ("lookup", trace_id, ["doc-a"])
    assert (draft.name, draft.kind, draft.trace_id) == ("draft", "generation", trace_id)
    assert draft.metadata == {"prompt_name": "answer", "prompt_version": "3"}
    assert draft.usage == {"input": 10, "output": 5, "total": 15}


async def _boom(_question: str) -> AnswerResult:
    with obs.span("model-call"):
        raise RuntimeError("model call failed")


async def _too_slow(_question: str) -> AnswerResult:
    with obs.span("model-call"):
        await asyncio.sleep(WAIT_S)
    return AnswerResult(answer="Too late.")


async def _blank(_question: str) -> AnswerResult:
    return AnswerResult(answer="  ")


@pytest.mark.parametrize(
    ("answerer", "outcome", "answer", "message"),
    [
        (FnAnswerer(_boom), "error", INTERNAL_ERROR_ANSWER, "RuntimeError: model call failed"),
        (FnAnswerer(_too_slow), "timeout", TIMEOUT_ANSWER, "timed out after"),
        (
            FnAnswerer(_blank),
            "empty_answer",
            INTERNAL_ERROR_ANSWER,
            "the answerer returned a blank",
        ),
    ],
    ids=["exception", "timeout", "blank answer"],
)
def test_failed_asks_are_error_traces(
    settings: Settings,
    tracer: RecordingTracer,
    answerer: FnAnswerer,
    outcome: str,
    answer: str,
    message: str,
) -> None:
    quick = dataclasses.replace(settings, ask_deadline_s=0.05)
    with serving(quick, answerer, tracer) as client:
        response = client.post("/ask", json={"question": "Will this fail?"})
    assert assert_valid_body(response)["answer"] == answer
    ask = only_ask(tracer)
    assert ask.trace_id == trace_id_of(response)
    assert ask.input == "Will this fail?"
    assert ask.metadata["outcome"] == outcome
    assert ask.level == "ERROR"
    assert ask.status_message is not None and ask.status_message.startswith(message)
    assert ask.output == {"answer": answer, "document_ids": []}
    for span in tracer.children(ask):  # the answerer's span failed with it
        assert (span.name, span.level, span.ended) == ("model-call", "ERROR", True)


def test_invalid_request_is_a_warning_trace_without_input(
    settings: Settings, tracer: RecordingTracer
) -> None:
    with serving(settings, returns("Unused."), tracer) as client:
        response = client.post("/ask", content=b"not json")
    assert assert_valid_body(response)["answer"] == INVALID_REQUEST_ANSWER
    ask = only_ask(tracer)
    assert (ask.trace_id, ask.input, ask.level) == (trace_id_of(response), None, "WARNING")
    assert ask.metadata["outcome"] == "invalid_request"
    assert ask.status_message is not None and ask.status_message.startswith("invalid request: ")


def test_not_ready_is_a_warning_trace(settings: Settings, tracer: RecordingTracer) -> None:
    gate = threading.Event()

    def slow_startup(_settings: Settings) -> Runtime:
        gate.wait(WAIT_S)
        return Runtime(known_doc_ids=KNOWN)

    with serving(settings, returns("Unused."), tracer, slow_startup, wait=False) as client:
        response = client.post("/ask", json={"question": "Too early?"})
        gate.set()
    assert assert_valid_body(response)["answer"] == NOT_READY_ANSWER
    ask = only_ask(tracer)
    assert (ask.trace_id, ask.input, ask.level) == (trace_id_of(response), "Too early?", "WARNING")
    assert ask.metadata["outcome"] == "not_ready"


def test_four_concurrent_asks_get_four_separate_traces(
    settings: Settings, tracer: RecordingTracer
) -> None:
    async def scenario() -> list[httpx.Response]:
        barrier = asyncio.Barrier(4)

        async def echo(question: str) -> AnswerResult:
            with obs.span("work", input=question):
                await barrier.wait()  # all four are in flight at once
                await asyncio.to_thread(time.sleep, 0.001)
            return AnswerResult(answer=f"echo {question}", document_ids=[f"doc-{question}"])

        known = frozenset(f"doc-q{i}" for i in range(4))
        app = create_app(
            dataclasses.replace(settings, ask_deadline_s=WAIT_S),
            ready_with(known),
            FnAnswerer(echo),
            tracer=tracer,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                deadline = time.monotonic() + WAIT_S
                while (await client.get("/health")).status_code != 200:
                    assert time.monotonic() < deadline, "server never became ready"
                    await asyncio.sleep(0.01)
                asks = (client.post("/ask", json={"question": f"q{i}"}) for i in range(4))
                return list(await asyncio.gather(*asks))

    responses = asyncio.run(scenario())
    assert all(assert_valid_body(response) for response in responses)
    trace_ids = [trace_id_of(response) for response in responses]
    asks = {ask.trace_id: ask for ask in tracer.traces("ask")}
    assert set(asks) == set(trace_ids)
    for i, trace_id in enumerate(trace_ids):
        ask = asks[trace_id]
        assert ask.input == f"q{i}"
        assert ask.output == {"answer": f"echo q{i}", "document_ids": [f"doc-q{i}"]}
        [work] = tracer.children(ask)
        assert (work.name, work.input, work.trace_id) == ("work", f"q{i}", trace_id)


def test_the_queue_wait_and_load_are_on_each_ask_trace(
    settings: Settings, tracer: RecordingTracer
) -> None:
    async def slow(question: str) -> AnswerResult:
        await asyncio.sleep(0.2)
        return AnswerResult(answer=f"answer to {question}")

    app = create_app(
        dataclasses.replace(settings, ask_deadline_s=WAIT_S),
        ready_with(),
        FnAnswerer(slow),
        tracer=tracer,
    )
    results = asyncio.run(ask_at_once(app, [f"q{i}" for i in range(ASK_MAX_CONCURRENT + 1)]))
    assert all(assert_valid_body(response) for response, _seconds in results)
    asks = tracer.traces("ask")
    assert len(asks) == ASK_MAX_CONCURRENT + 1
    for ask in asks:
        assert ask.metadata["queue_limit"] == ASK_MAX_CONCURRENT
        assert ask.metadata["queue_got_slot"] is True
    by_wait = sorted(asks, key=lambda ask: ask.metadata["queue_wait_ms"])  # type: ignore[arg-type, return-value]
    first, last = by_wait[0], by_wait[-1]
    # The first four got a slot at once; the fifth found four running and waited for one.
    assert first.metadata["queue_wait_ms"] < 100  # type: ignore[operator]
    assert last.metadata["queue_wait_ms"] >= 150  # type: ignore[operator]
    assert last.metadata["queue_running_on_arrival"] == ASK_MAX_CONCURRENT
    assert last.metadata["queue_waiting_on_arrival"] == 0


# ---------------------------------------------------------------- startup, health, shutdown


def test_startup_is_a_trace_with_its_stages_inside(
    settings: Settings, tracer: RecordingTracer
) -> None:
    def staged_startup(_settings: Settings) -> Runtime:
        with obs.span("load") as stage:
            stage.update(output={"documents": 3})
        return Runtime(known_doc_ids=KNOWN)

    with serving(settings, returns("Ok."), tracer, staged_startup):
        pass
    [startup] = tracer.traces("startup")
    assert (startup.ended, startup.level) == (True, "DEFAULT")
    assert startup.metadata["startup_fn"] == staged_startup.__qualname__
    assert startup.metadata["outcome"] == "ready"
    duration = startup.metadata["duration_s"]
    assert isinstance(duration, float) and 0 <= duration < WAIT_S
    assert startup.output == {"known_doc_ids": len(KNOWN)}
    assert startup.metadata["answerer"] == "FnAnswerer"
    # The start-up job's stages, then building the answerer over its result.
    [stage, answerer] = tracer.children(startup)
    assert (stage.name, stage.output) == ("load", {"documents": 3})
    assert (answerer.name, answerer.output) == (ANSWERER_SPAN, {"answerer": "FnAnswerer"})
    assert answerer.ended and answerer.level == "DEFAULT"


def test_failed_startup_is_an_error_trace(settings: Settings, tracer: RecordingTracer) -> None:
    def broken_startup(_settings: Settings) -> Runtime:
        raise RuntimeError(f"index build exploded near {FAKE_GEMINI}")

    with serving(settings, returns("Unused."), tracer, broken_startup, wait=False) as client:
        deadline = time.monotonic() + WAIT_S
        while client.get("/health").json()["status"] != "failed":
            assert time.monotonic() < deadline, "start-up failure never surfaced"
            time.sleep(0.01)
    [startup] = tracer.traces("startup")
    assert (startup.ended, startup.level) == (True, "ERROR")
    assert startup.status_message == f"RuntimeError: index build exploded near {REDACTED}"
    assert startup.metadata["outcome"] == "failed"
    assert startup.output is None


def test_health_is_not_traced(settings: Settings, tracer: RecordingTracer) -> None:
    with serving(settings, returns("Ok."), tracer) as client:
        for _ in range(5):
            assert client.get("/health").status_code == 200
    # Only start-up (and its answerer span) is traced, never a /health poll.
    assert [o.name for o in tracer.observations] == ["startup", ANSWERER_SPAN]


def test_shutdown_flushes_the_tracer(settings: Settings, tracer: RecordingTracer) -> None:
    with serving(settings, returns("Ok."), tracer) as client:
        assert_valid_body(client.post("/ask", json={"question": "Flush me?"}))
        assert (tracer.flush_calls, tracer.shutdown_calls) == (0, 0)
    assert (tracer.flush_calls, tracer.shutdown_calls) == (1, 1)


# ---------------------------------------------------------------- secrets


def test_secrets_never_reach_a_trace(settings: Settings, tracer: RecordingTracer) -> None:
    bearer = "Bearer abc123" + "t" * 20

    async def leaky(question: str) -> AnswerResult:
        with obs.span(
            "call-llm",
            input={"api_key": FAKE_GEMINI, "headers": {"Authorization": bearer}},
            metadata={"config": f"secret={FAKE_LF_SECRET} public={FAKE_LF_PUBLIC}"},
        ) as call:
            call.update(output=f"the key is {GOOGLE_SHAPED}")
        with obs.generation(
            "draft",
            model="fake-model",
            input=[{"role": "user", "content": question}],
            model_parameters={"x-goog-api-key": FAKE_GEMINI},
        ):
            pass
        return AnswerResult(answer=f"Leaked {FAKE_GEMINI} and {FAKE_LF_SECRET}", document_ids=[])

    async def leaky_failure(_question: str) -> AnswerResult:
        raise RuntimeError(f"auth failed for {FAKE_GEMINI}")

    with serving(settings, FnAnswerer(leaky), tracer) as client:
        assert_valid_body(client.post("/ask", json={"question": f"Is {FAKE_LF_SECRET} ok?"}))
    with serving(settings, FnAnswerer(leaky_failure), tracer) as client:
        assert_valid_body(client.post("/ask", json={"question": f"Is {FAKE_LF_PUBLIC} ok?"}))

    recorded = repr(tracer.observations)
    for secret in (FAKE_GEMINI, FAKE_LF_PUBLIC, FAKE_LF_SECRET, GOOGLE_SHAPED, "abc123"):
        assert secret not in recorded
    assert REDACTED in recorded
    assert len(tracer.traces("ask")) == 2


# ---------------------------------------------------------------- tracing never breaks /ask


@pytest.mark.parametrize("fail", [{"open"}, {"close"}], ids=["open", "update+close"])
def test_a_broken_tracer_never_breaks_ask(
    settings: Settings, fail: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    async def traced(question: str) -> AnswerResult:
        with obs.span("inner") as inner:
            inner.update(output=question)
        return AnswerResult(answer=f"answered {question}", document_ids=["doc-a"])

    broken = BrokenTracer(frozenset(fail))
    with serving(settings, FnAnswerer(traced), broken) as client:
        for i in range(3):
            body = assert_valid_body(client.post("/ask", json={"question": f"q{i}"}))
            assert (body["answer"], body["document_ids"]) == (f"answered q{i}", ["doc-a"])
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    failed_ops = [m.split(" failed")[0] for m in warnings if m.startswith("tracing:")]
    assert failed_ops, "the tracing failure was not logged"
    assert len(failed_ops) == len(set(failed_ops)), failed_ops  # once per kind, not per request


@contextmanager
def silent_server() -> Iterator[int]:
    """A local port that accepts TCP connections and never answers (a hung Langfuse)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(64)  # the kernel completes handshakes; nobody ever reads
        yield sock.getsockname()[1]


def test_real_client_at_an_unresponsive_host_never_slows_ask(settings: Settings) -> None:
    with silent_server() as port:
        tracer = LangfuseTracer(
            public_key="pk-lf-test-" + uuid.uuid4().hex,  # the SDK keeps one client per key
            secret_key=FAKE_LF_SECRET,
            host=f"http://127.0.0.1:{port}",
            scrubber=Scrubber.for_settings(settings),
            timeout_s=1,
        )
        with serving(settings, returns("Fine.", ["doc-a"]), tracer) as client:
            assert_valid_body(client.post("/ask", json={"question": "First?"}))
            # Start an export that hangs on the silent host, in the background.
            assert tracer.flush(timeout_s=0.1) is False
            for i in range(3):
                started = time.monotonic()
                body = assert_valid_body(client.post("/ask", json={"question": f"Again {i}?"}))
                assert body["answer"] == "Fine."
                assert time.monotonic() - started < 0.5
            closing = time.monotonic()
        # Leaving the client ran the lifespan exit: a flush bounded by the constant.
        assert time.monotonic() - closing < TRACE_SHUTDOWN_TIMEOUT_S + 1.0


def test_fastapi_native_telemetry_stays_off_even_with_a_global_provider(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If any library set a global OpenTelemetry provider, FastAPI would start tracing
    # every request (/health included) by itself; create_app turns that off.
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)
    with serving(settings, returns("Ok."), obs.NoopTracer()) as client:
        assert client.get("/health").status_code == 200
        assert_valid_body(client.post("/ask", json={"question": "Traced twice?"}))
    assert exporter.get_finished_spans() == ()
