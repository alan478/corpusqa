"""HTTP contract (CONTRACT.md §2-4) against the real app with fake start-up and answerers.

Every /ask response, whatever went wrong, must be HTTP 200 with a valid body:
`assert_valid_body` checks that on every request these tests make.
"""

import asyncio
import dataclasses
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from corpusqa.agent import Answerer, AnswerResult, PlaceholderAnswerer
from corpusqa.config import (
    ANSWER_TRUNCATED_SUFFIX,
    ASK_MAX_CONCURRENT,
    INTERNAL_ERROR_ANSWER,
    INVALID_REQUEST_ANSWER,
    MAX_RESPONSE_BYTES,
    NOT_READY_ANSWER,
    TIMEOUT_ANSWER,
    TRACE_ID_HEADER,
    Settings,
)
from corpusqa.ingest import Runtime, StartupFn
from corpusqa.server import create_app

KNOWN = frozenset({"doc-a", "doc-b", "doc-c"})  # invented ids
WAIT_S = 5.0  # generous bound for anything the tests wait on


class FnAnswerer:
    """Test answerer whose behaviour is one async function; records the time left it got."""

    def __init__(self, fn: Callable[[str], Awaitable[AnswerResult]]) -> None:
        self._fn = fn
        self.time_left: dict[str, float | None] = {}  # question -> time_left_s

    async def answer(self, question: str, time_left_s: float | None = None) -> AnswerResult:
        self.time_left[question] = time_left_s
        return await self._fn(question)


def returns(answer: str, ids: list[str] | None = None) -> FnAnswerer:
    async def fn(_question: str) -> AnswerResult:
        return AnswerResult(answer=answer, document_ids=ids or [])

    return FnAnswerer(fn)


def ready_with(known: frozenset[str] = KNOWN) -> StartupFn:
    def startup(_settings: Settings) -> Runtime:
        return Runtime(known_doc_ids=known)

    return startup


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(corpus_dir=tmp_path, port=8080)


# CONTRACT §3: the body is exactly these two fields (user decision 2026-10-17: no extras).
CONTRACT_BODY_KEYS = {"answer", "document_ids"}


def assert_valid_body(response: httpx.Response) -> dict:
    """The contract for every /ask response: exactly its two fields; the trace id in a header."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert len(response.content) < MAX_RESPONSE_BYTES
    body = response.json()
    assert set(body) == CONTRACT_BODY_KEYS, body
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert isinstance(body["document_ids"], list)
    assert all(isinstance(doc_id, str) for doc_id in body["document_ids"])
    assert uuid.UUID(hex=trace_id_of(response)).version == 4
    return body


def trace_id_of(response: httpx.Response) -> str:
    """The /ask response's trace id: its `X-Trace-Id` header, never a body field."""
    return response.headers[TRACE_ID_HEADER]


def wait_until_ready(client: TestClient) -> None:
    deadline = time.monotonic() + WAIT_S
    while client.get("/health").status_code != 200:
        assert time.monotonic() < deadline, "server never became ready"
        time.sleep(0.01)


@contextmanager
def serving(
    settings: Settings, answerer: Answerer, startup: StartupFn | None = None
) -> Iterator[TestClient]:
    app = create_app(settings, startup or ready_with(), answerer)
    with TestClient(app) as client:  # `with` runs the lifespan, so start-up runs
        wait_until_ready(client)
        yield client


# ---------------------------------------------------------------- /health


def test_health_is_503_until_startup_finishes_then_200(settings: Settings) -> None:
    gate = threading.Event()

    def slow_startup(_settings: Settings) -> Runtime:
        gate.wait(WAIT_S)
        return Runtime(known_doc_ids=KNOWN)

    app = create_app(settings, slow_startup, PlaceholderAnswerer())
    with TestClient(app) as client:
        starting = client.get("/health")
        assert starting.status_code == 503
        assert starting.json() == {"status": "starting"}
        early = assert_valid_body(client.post("/ask", json={"question": "Too early?"}))
        assert early == {"answer": NOT_READY_ANSWER, "document_ids": []}

        gate.set()
        wait_until_ready(client)
        assert client.get("/health").json() == {"status": "ok"}


def test_failed_startup_keeps_health_503_and_logs_it(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    def broken_startup(_settings: Settings) -> Runtime:
        raise RuntimeError("index build exploded")

    app = create_app(settings, broken_startup, PlaceholderAnswerer())
    with TestClient(app) as client:
        deadline = time.monotonic() + WAIT_S
        while client.get("/health").json()["status"] != "failed":
            assert time.monotonic() < deadline, "start-up failure never surfaced"
            time.sleep(0.01)
        assert client.get("/health").status_code == 503
        body = assert_valid_body(client.post("/ask", json={"question": "Anything?"}))
        assert body["answer"] == NOT_READY_ANSWER
    assert "startup FAILED" in caplog.text
    assert "index build exploded" in caplog.text


def test_real_uvicorn_binds_the_port_before_startup_finishes(
    settings: Settings, free_port: int
) -> None:
    # TestClient never binds a port; this checks the grader's view over real TCP.
    gate = threading.Event()

    def slow_startup(_settings: Settings) -> Runtime:
        gate.wait(WAIT_S)
        return Runtime(known_doc_ids=KNOWN)

    app = create_app(settings, slow_startup, PlaceholderAnswerer())
    config = uvicorn.Config(app, host="127.0.0.1", port=free_port, log_config=None)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    health = f"http://127.0.0.1:{free_port}/health"
    try:
        deadline = time.monotonic() + WAIT_S
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn never started"
            time.sleep(0.01)
        assert httpx.get(health).status_code == 503
        gate.set()
        while httpx.get(health).status_code != 200:
            assert time.monotonic() < deadline, "server never became ready"
            time.sleep(0.01)
    finally:
        gate.set()
        server.should_exit = True
        thread.join(WAIT_S)


# ---------------------------------------------------------------- /ask happy path


def test_ask_returns_exactly_the_contract_body_and_the_trace_id_in_a_header(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    with serving(settings, returns("It is blue.", ["doc-a"])) as client, caplog.at_level("INFO"):
        response = client.post("/ask", json={"question": "What colour?"})
    assert assert_valid_body(response) == {"answer": "It is blue.", "document_ids": ["doc-a"]}
    assert "trace_id" not in response.json()
    # The server's one log line per /ask still names the trace id the header carries.
    assert f"ask trace_id={trace_id_of(response)} outcome=ok" in caplog.text


def test_answerer_gets_the_stripped_question(settings: Settings) -> None:
    seen: list[str] = []

    async def record(question: str) -> AnswerResult:
        seen.append(question)
        return AnswerResult(answer="ok")

    with serving(settings, FnAnswerer(record)) as client:
        assert_valid_body(client.post("/ask", json={"question": "  Why?\n", "extra": 1}))
    assert seen == ["Why?"]


def test_ids_are_deduplicated_in_order_and_unknown_ids_dropped(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    ids = ["doc-b", "doc-a", "dsid_not_in_corpus", "doc-b", "doc-a", "doc-c"]
    with serving(settings, returns("Cited.", ids)) as client:
        response = client.post("/ask", json={"question": "Which docs?"})
    body = assert_valid_body(response)
    assert body["document_ids"] == ["doc-b", "doc-a", "doc-c"]
    assert "dropped 1 unknown document ids" in caplog.text
    assert trace_id_of(response) in caplog.text


def test_every_response_gets_its_own_trace_id(settings: Settings) -> None:
    with serving(settings, returns("Same answer.")) as client:
        responses = [client.post("/ask", json={"question": "Again?"})]
        responses += [client.post("/ask", content=b"not json")]
        responses += [client.post("/ask", json={"question": "Again?"})]
    assert all(assert_valid_body(response) for response in responses)
    assert len({trace_id_of(response) for response in responses}) == len(responses)


# ---------------------------------------------------------------- /ask failures still answer 200

BAD_BODIES = {
    "not json": b"not json",
    "empty body": b"",
    "invalid utf-8": b'{"question": "\xff\xfe"}',
    "missing field": b"{}",
    "empty question": b'{"question": ""}',
    "blank question": b'{"question": "   \\n"}',
    "number": b'{"question": 42}',
    "null": b'{"question": null}',
    "list": b'{"question": ["Why?"]}',
    "not an object": b'["question"]',
}


@pytest.mark.parametrize("raw", BAD_BODIES.values(), ids=BAD_BODIES.keys())
def test_bad_request_bodies_get_200_with_a_valid_body(settings: Settings, raw: bytes) -> None:
    with serving(settings, returns("Should not be used.", ["doc-a"])) as client:
        response = client.post("/ask", content=raw, headers={"content-type": "application/json"})
    body = assert_valid_body(response)
    assert body["answer"] == INVALID_REQUEST_ANSWER
    assert body["document_ids"] == []


def test_answerer_exception_gives_internal_error_not_a_not_found(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    async def boom(_question: str) -> AnswerResult:
        raise RuntimeError("model call failed")

    with serving(settings, FnAnswerer(boom)) as client:
        body = assert_valid_body(client.post("/ask", json={"question": "Will this fail?"}))
    assert body["answer"] == INTERNAL_ERROR_ANSWER
    assert body["document_ids"] == []
    assert "model call failed" in caplog.text


def test_slow_answerer_gets_timeout_answer_within_the_deadline(settings: Settings) -> None:
    async def too_slow(_question: str) -> AnswerResult:
        await asyncio.sleep(WAIT_S)
        return AnswerResult(answer="Too late.", document_ids=["doc-a"])

    quick = dataclasses.replace(settings, ask_deadline_s=0.05)
    with serving(quick, FnAnswerer(too_slow)) as client:
        started = time.monotonic()
        body = assert_valid_body(client.post("/ask", json={"question": "Slow one?"}))
        assert time.monotonic() - started < WAIT_S / 2
    assert body["answer"] == TIMEOUT_ANSWER
    assert body["document_ids"] == []


@pytest.mark.parametrize("blank", ["", "  \n\t"])
def test_blank_answer_becomes_internal_error(settings: Settings, blank: str) -> None:
    with serving(settings, returns(blank, ["doc-a"])) as client:
        body = assert_valid_body(client.post("/ask", json={"question": "Say nothing?"}))
    assert body["answer"] == INTERNAL_ERROR_ANSWER
    assert body["document_ids"] == []


def test_huge_answer_is_truncated_and_body_stays_under_1_mib(settings: Settings) -> None:
    # \x01 is the worst case: JSON escapes it to 6 bytes (\u0001).
    with serving(settings, returns("\x01" * 2_000_000, ["doc-a"])) as client:
        response = client.post("/ask", json={"question": "Write a lot?"})
    body = assert_valid_body(response)  # includes the < 1 MiB check
    assert body["answer"].endswith(ANSWER_TRUNCATED_SUFFIX)
    assert body["document_ids"] == ["doc-a"]


# ---------------------------------------------------------------- concurrency


def test_four_concurrent_asks_run_in_parallel_without_cross_talk(settings: Settings) -> None:
    async def scenario() -> list[httpx.Response]:
        # Each answer waits until all four are in flight: serialised requests would
        # never pass the barrier and would end as timeout answers instead.
        barrier = asyncio.Barrier(4)

        async def echo(question: str) -> AnswerResult:
            await barrier.wait()
            return AnswerResult(answer=f"echo {question}", document_ids=[f"doc-{question}"])

        known = frozenset(f"doc-q{i}" for i in range(4))
        app: FastAPI = create_app(
            dataclasses.replace(settings, ask_deadline_s=WAIT_S),
            ready_with(known),
            FnAnswerer(echo),
        )
        async with app.router.lifespan_context(app):  # ASGITransport skips the lifespan
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                deadline = time.monotonic() + WAIT_S
                while (await client.get("/health")).status_code != 200:
                    assert time.monotonic() < deadline, "server never became ready"
                    await asyncio.sleep(0.01)
                asks = (client.post("/ask", json={"question": f"q{i}"}) for i in range(4))
                return list(await asyncio.gather(*asks))

    responses = asyncio.run(scenario())
    bodies = [assert_valid_body(response) for response in responses]
    for i, body in enumerate(bodies):
        assert body["answer"] == f"echo q{i}"
        assert body["document_ids"] == [f"doc-q{i}"]
    assert len({trace_id_of(response) for response in responses}) == 4


# ---------------------------------------------------------------- the /ask queue
async def ask_at_once(app: FastAPI, questions: list[str]) -> list[tuple[httpx.Response, float]]:
    """Send every question at the same moment; each response with its seconds taken."""
    async with app.router.lifespan_context(app):  # ASGITransport skips the lifespan
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            deadline = time.monotonic() + WAIT_S
            while (await client.get("/health")).status_code != 200:
                assert time.monotonic() < deadline, "server never became ready"
                await asyncio.sleep(0.01)

            async def one(question: str) -> tuple[httpx.Response, float]:
                sent = time.monotonic()
                response = await client.post("/ask", json={"question": question})
                return response, time.monotonic() - sent

            return list(await asyncio.gather(*(one(q) for q in questions)))


class Gauge:
    """Counts the answers running now and the most that ever ran at once."""

    def __init__(self) -> None:
        self.now = 0
        self.most = 0

    async def hold(self, seconds: float) -> None:
        self.now += 1
        self.most = max(self.most, self.now)
        try:
            await asyncio.sleep(seconds)
        finally:
            self.now -= 1


def test_six_asks_at_once_run_four_at_a_time_and_all_get_answers(settings: Settings) -> None:
    assert ASK_MAX_CONCURRENT == 4
    gauge = Gauge()

    async def slow(question: str) -> AnswerResult:
        await gauge.hold(0.2)
        return AnswerResult(answer=f"answer to {question}", document_ids=["doc-a"])

    app = create_app(
        dataclasses.replace(settings, ask_deadline_s=WAIT_S), ready_with(), FnAnswerer(slow)
    )
    questions = [f"q{i}" for i in range(6)]
    results = asyncio.run(ask_at_once(app, questions))
    assert gauge.most == ASK_MAX_CONCURRENT  # four ran side by side, never a fifth
    for question, (response, _seconds) in zip(questions, results, strict=True):
        body = assert_valid_body(response)
        assert body == {"answer": f"answer to {question}", "document_ids": ["doc-a"]}
    # Two waited for a slot: about one answer's time longer than the first four.
    seconds = sorted(s for _response, s in results)
    assert seconds[-1] >= seconds[0] + 0.15


def test_the_wait_counts_against_the_deadline_and_ends_in_the_timeout_answer(
    settings: Settings,
) -> None:
    # Each answer takes 0.6 s of a 1.0 s deadline. The fifth request waits 0.6 s for a
    # slot, so with 0.4 s left it gets the timeout answer at its deadline (1.0 s), not an
    # answer at 1.2 s (a deadline counted from the slot) and not a hang.
    answerer = FnAnswerer(lambda question: _sleep_then_answer(0.6, question))
    quick = dataclasses.replace(settings, ask_deadline_s=1.0)
    app = create_app(quick, ready_with(), answerer)
    questions = [f"q{i}" for i in range(5)]
    results = asyncio.run(ask_at_once(app, questions))
    bodies = [assert_valid_body(response) for response, _seconds in results]
    answered = [b for b in bodies if b["answer"].startswith("answer to")]
    timed_out = [
        (b, s) for b, (_r, s) in zip(bodies, results, strict=True) if b["answer"] == TIMEOUT_ANSWER
    ]
    assert len(answered) == 4 and len(timed_out) == 1
    [(body, seconds)] = timed_out
    assert body["document_ids"] == []
    assert 0.9 <= seconds < 1.15  # at its own deadline, counted from its arrival
    # The waiting request was told how much of its deadline was left.
    [late] = [q for q, left in answerer.time_left.items() if left is not None and left < 0.6]
    assert answerer.time_left[late] == pytest.approx(0.4, abs=0.1)
    assert all(
        left == pytest.approx(1.0, abs=0.1) for q, left in answerer.time_left.items() if q != late
    )


async def _sleep_then_answer(seconds: float, question: str) -> AnswerResult:
    await asyncio.sleep(seconds)
    return AnswerResult(answer=f"answer to {question}", document_ids=["doc-a"])
