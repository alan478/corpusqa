"""The agent answerer behind the real app: the HTTP contract still holds, and the wiring.

FakeLLM plays the model. The server's own guard keeps only known ids, and an agent that
ends with no answer gives the internal-error answer, never a "not found". The wiring:
`create_app` builds the answerer from the start-up `Runtime` (once, before /health says
200); `__main__.build_answerer` makes the agent over the real corpus tools of that index
(the field catalog in its system message), or, with no chat model, an answerer that gives
the internal-error answer.
"""

import asyncio
import functools
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from corpusqa.agent import Answerer, PlaceholderAnswerer
from corpusqa.agent.agent import CATALOG_SLOT, FIRST_SEARCH_CALL_ID, AgentLimits
from corpusqa.agent.answerer import AgentAnswerer, UnavailableAnswerer
from corpusqa.agent.corpus_tools import corpus_session_factory
from corpusqa.agent.handles import CleanedAnswer
from corpusqa.agent.tools import ToolBox
from corpusqa.config import INTERNAL_ERROR_ANSWER, NOT_READY_ANSWER, Settings
from corpusqa.ingest import Runtime
from corpusqa.llm.client import GeminiChat
from corpusqa.llm.errors import ChatError
from corpusqa.llm.fakes import ChatRequest, FakeLLM
from corpusqa.llm.interfaces import ModelReply, ToolCall
from corpusqa.obs.testing import RecordingTracer
from corpusqa.server import __main__ as server_main
from corpusqa.server import create_app
from corpusqa.server.app import ANSWERER_SPAN
from tests.contract.test_server_contract import (
    KNOWN,
    WAIT_S,
    assert_valid_body,
    ready_with,
    returns,
    serving,
    wait_until_ready,
)
from tests.contract.test_server_ingest import wired_main
from tests.unit.test_agent import echo, echo_box, finish, reply
from tests.unit.tool_corpus import N_ORDERS, SOURCE, Built, build_tool_corpus, doc_id, state_of

NO_KEY = "GEMINI_API_KEY is not set: the agent has no chat model"


class PassThroughSession:
    """Citations come back as ids unchanged: the server's guard must drop unknown ones."""

    def __init__(self) -> None:
        self._toolbox = echo_box()

    @property
    def toolbox(self) -> ToolBox:
        return self._toolbox

    def resolve_citations(self, citations: Sequence[str]) -> list[str]:
        return list(citations)

    def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
        return CleanedAnswer(answer)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(corpus_dir=tmp_path, port=8080)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_tool_corpus(tmp_path_factory.mktemp("agent-index"))


def in_state(state: str) -> set[str]:
    return {doc_id(n) for n in range(1, N_ORDERS + 1) if state_of(n) == state}


def state_filter(call_id: str, state: str) -> ToolCall:
    conditions = [{"field": "job_state", "op": "eq", "value": state}]
    return ToolCall(call_id, "filter", {"source": SOURCE, "conditions": conditions})


def agent_factory(model: FakeLLM, settings: Settings) -> functools.partial[Answerer]:
    return functools.partial(server_main.build_answerer, chat=model, problem="", settings=settings)


# ---------------------------------------------------------------- the HTTP contract
def test_an_agent_answer_comes_back_with_only_known_ids(settings: Settings) -> None:
    assert {"doc-a", "doc-b"} <= KNOWN
    model = FakeLLM([reply(echo("c1")), reply(finish(citations=["doc-b", "made-up", "doc-a"]))])
    with serving(settings, AgentAnswerer(model, PassThroughSession)) as client:
        body = assert_valid_body(client.post("/ask", json={"question": "When did it move?"}))
    assert body["answer"] == "It moved to May because the supplier was late."
    assert body["document_ids"] == ["doc-b", "doc-a"]


def test_an_agent_with_no_answer_gives_the_internal_error_answer(settings: Settings) -> None:
    model = FakeLLM([reply(echo("c1")), ChatError("down"), ChatError("down")])
    answerer = AgentAnswerer(model, PassThroughSession, limits=AgentLimits(max_steps=1))
    with serving(settings, answerer) as client:
        body = assert_valid_body(client.post("/ask", json={"question": "When did it move?"}))
    assert body["answer"] == INTERNAL_ERROR_ANSWER
    assert body["document_ids"] == []


# ---------------------------------------------------------------- the answerer factory
def test_the_answerer_is_built_once_from_the_startup_runtime(settings: Settings) -> None:
    runtime = Runtime(known_doc_ids=KNOWN)
    gate = threading.Event()
    seen: list[Runtime] = []

    def factory(got: Runtime) -> Answerer:
        gate.wait(WAIT_S)
        seen.append(got)
        return returns("Built after start-up.", ["doc-a"])

    app = create_app(settings, lambda _settings: runtime, answerer_factory=factory)
    with TestClient(app) as client:
        # Start-up has returned, but the answerer is not built yet: not ready.
        assert client.get("/health").status_code == 503
        early = assert_valid_body(client.post("/ask", json={"question": "Too early?"}))
        assert early["answer"] == NOT_READY_ANSWER
        gate.set()
        wait_until_ready(client)
        bodies = [
            assert_valid_body(client.post("/ask", json={"question": f"Question {i}?"}))
            for i in range(2)
        ]
    assert all(b["answer"] == "Built after start-up." for b in bodies)
    assert all(b["document_ids"] == ["doc-a"] for b in bodies)
    assert seen == [runtime]  # once, with what start-up returned


def test_a_failing_answerer_factory_fails_startup(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(_runtime: Runtime) -> Answerer:
        raise RuntimeError("catalog build exploded")

    tracer = RecordingTracer.for_settings(settings)
    app = create_app(settings, ready_with(), answerer_factory=broken, tracer=tracer)
    with TestClient(app) as client:
        deadline = time.monotonic() + WAIT_S
        while client.get("/health").json()["status"] != "failed":
            assert time.monotonic() < deadline, "the factory's failure never surfaced"
            time.sleep(0.01)
        assert client.get("/health").status_code == 503
        body = assert_valid_body(client.post("/ask", json={"question": "Anything?"}))
        assert body["answer"] == NOT_READY_ANSWER
    assert "startup FAILED" in caplog.text and "catalog build exploded" in caplog.text
    [startup] = tracer.traces("startup")
    [span] = tracer.named(ANSWERER_SPAN)
    assert span.parent_id == startup.id
    assert startup.level == span.level == "ERROR"


@pytest.mark.parametrize("both", [False, True], ids=["neither", "both"])
def test_create_app_takes_exactly_one_answerer_source(settings: Settings, both: bool) -> None:
    extra = {"answerer": PlaceholderAnswerer()} if both else {}
    with pytest.raises(ValueError, match="answerer"):
        if both:
            create_app(
                settings, ready_with(), answerer_factory=lambda _r: PlaceholderAnswerer(), **extra
            )
        else:
            create_app(settings, ready_with())


# ---------------------------------------------------------------- main's wiring
def test_main_builds_the_chat_model_and_hands_the_agent_factory_to_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Building the client makes no network call; nothing is asked here.
    seen = wired_main(tmp_path, monkeypatch, key="test-key-not-real")
    factory = seen["answerer_factory"]
    assert isinstance(factory, functools.partial) and factory.func is server_main.build_answerer
    assert isinstance(factory.keywords["chat"], GeminiChat)
    assert factory.keywords["settings"] is seen["settings"]


def test_main_without_a_key_has_no_chat_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = wired_main(tmp_path, monkeypatch, key=None)
    factory = seen["answerer_factory"]
    assert isinstance(factory, functools.partial)
    assert factory.keywords["chat"] is None
    assert factory.keywords["problem"] == NO_KEY


def test_without_a_chat_model_every_ask_gives_the_internal_error_answer(
    settings: Settings,
) -> None:
    tracer = RecordingTracer.for_settings(settings)
    factory = functools.partial(
        server_main.build_answerer, chat=None, problem=NO_KEY, settings=settings
    )
    app = create_app(settings, ready_with(), answerer_factory=factory, tracer=tracer)
    with TestClient(app) as client:
        wait_until_ready(client)  # the service runs; it cannot answer
        body = assert_valid_body(client.post("/ask", json={"question": "Who owns it?"}))
    assert body["answer"] == INTERNAL_ERROR_ANSWER  # never a "not found"
    assert body["document_ids"] == []
    reason = server_main.NO_CHAT_MODEL.format(problem=NO_KEY)
    [span] = tracer.named(ANSWERER_SPAN)
    assert span.output == {"answerer": UnavailableAnswerer.__name__}
    assert (span.level, span.status_message) == ("WARNING", reason)
    [ask] = tracer.traces("ask")
    assert ask.level == "ERROR"
    assert ask.status_message == f"AnswererUnavailable: {reason}"


def test_a_runtime_without_index_readers_cannot_build_the_agent(settings: Settings) -> None:
    with pytest.raises(ValueError, match="index readers"):
        server_main.build_answerer(
            Runtime(known_doc_ids=KNOWN), chat=FakeLLM([]), problem="", settings=settings
        )


# ---------------------------------------------------------------- the agent over a real index
def test_the_agent_answers_over_the_real_index_through_the_server(
    built: Built, settings: Settings
) -> None:
    model = FakeLLM(
        [
            reply(state_filter("c1", "in repair")),
            reply(finish(answer="55 orders are in repair.", citations=["R1"])),
        ]
    )
    tracer = RecordingTracer.for_settings(settings)
    app = create_app(
        settings,
        lambda _settings: built.runtime,
        answerer_factory=agent_factory(model, settings),
        tracer=tracer,
    )
    with TestClient(app) as client:
        wait_until_ready(client)
        question = {"question": "How many orders are in repair?"}
        body = assert_valid_body(client.post("/ask", json=question))
    assert body["answer"] == "55 orders are in repair."
    # R1 is the filter's whole result set: every matching id, each once.
    assert len(body["document_ids"]) == 55
    assert set(body["document_ids"]) == in_state("In Repair")
    # The field catalog the tools learned from the index is in the system message.
    runtime = built.runtime
    assert runtime.store is not None and runtime.chunks is not None
    catalog = corpus_session_factory(
        runtime.store, runtime.chunks, runtime.vectors, runtime.embedder
    ).catalog
    system = model.requests[0].messages[0]
    assert system.role == "system"
    assert catalog in system.content and CATALOG_SLOT not in system.content
    assert {schema.name for schema in model.requests[0].tools} >= {"filter", "search", "finish"}
    [span] = tracer.named(ANSWERER_SPAN)
    assert span.output == {"answerer": AgentAnswerer.__name__}
    assert span.metadata["catalog_chars"] == len(catalog)
    assert span.metadata["documents"] == len(runtime.known_doc_ids)
    assert span.metadata["dense_search"] is True


def test_four_agent_asks_at_once_share_no_tool_state(built: Built, settings: Settings) -> None:
    states = {"first": "In Repair", "second": "Waiting Parts", "third": "Done", "fourth": "Done"}

    async def scenario() -> list[httpx.Response]:
        barrier = asyncio.Barrier(len(states))  # every first call waits for all four

        async def play(request: ChatRequest) -> ModelReply:
            text = request.messages[1].content
            [word] = [w for w in states if f"the {w} question" in text]
            # The first call of this question: the code's first search is its last message.
            if request.messages[-1].tool_call_id == FIRST_SEARCH_CALL_ID:
                await barrier.wait()
                return reply(state_filter(f"{word}-1", states[word].lower()))
            return reply(finish(answer=f"Answer to the {word} question.", citations=["R1"]))

        model = FakeLLM([play] * (2 * len(states)))
        app = create_app(
            settings,
            lambda _settings: built.runtime,
            answerer_factory=agent_factory(model, settings),
        )
        async with app.router.lifespan_context(app):  # ASGITransport skips the lifespan
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                deadline = time.monotonic() + WAIT_S
                while (await client.get("/health")).status_code != 200:
                    assert time.monotonic() < deadline, "server never became ready"
                    await asyncio.sleep(0.01)
                asks = (
                    client.post("/ask", json={"question": f"What is in the {word} question?"})
                    for word in states
                )
                return list(await asyncio.gather(*asks))

    responses = asyncio.run(scenario())
    for word, response in zip(states, responses, strict=True):
        body = assert_valid_body(response)
        assert body["answer"] == f"Answer to the {word} question."
        # Each session's own R1: its own filter's ids, not another request's.
        assert set(body["document_ids"]) == in_state(states[word])
