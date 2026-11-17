"""The first search: before the first model call, code runs `search` with the question as
asked, through the request's own tool box (agent/agent.py, plan 4.2; user 2026-10-17: "build
the search as well first so we send the search result for the first question as well").

Offline: FakeLLM; a test-only search tool (the real schema and arguments model, a scripted
body) for the loop's paths; the invented tool corpus (tests/unit/tool_corpus.py) for the real
search, its handles and its citations.
"""

import asyncio
import itertools
import json
import re
import threading
from collections.abc import Callable
from typing import Any

import pytest
from google.genai import types as genai_types

from corpusqa.agent.agent import (
    AGENT_SPAN,
    FIRST_SEARCH_CALL_ID,
    FIRST_SEARCH_FAILED_NOTE,
    FIRST_SEARCH_SPAN,
    STEP_SPAN,
    Agent,
    AgentLimits,
    AgentResult,
    first_search_call,
)
from corpusqa.agent.answerer import AgentAnswerer
from corpusqa.agent.corpus_tools import SEARCH, CorpusTools, SearchArgs, corpus_session_factory
from corpusqa.agent.session import FinishOnlySession
from corpusqa.agent.tools import Tool, ToolBox, ToolRun, finish_tool
from corpusqa.config import (
    AGENT_FINISH_RESERVE_S,
    AGENT_FIRST_SEARCH,
    AGENT_FIRST_SEARCH_TIMEOUT_S,
    CHAT_STAND_IN_THOUGHT_SIGNATURE,
)
from corpusqa.llm.client import to_gemini
from corpusqa.llm.fakes import ChatRequest, FakeLLM, ScriptEntry, fake_reply
from corpusqa.llm.interfaces import Message, ModelReply, ToolCall
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_tool_schemas
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.store.sqlite import SqliteReader
from tests.unit.test_agent import (
    ECHO,
    FAR_DEADLINE,
    FORCED,
    EchoArgs,
    FakeClock,
    assert_history_rules,
    by_choice,
    checked,
    echo,
    finish,
    note,
    reply,
)
from tests.unit.tool_corpus import Built, build_tool_corpus

# Quotes and a double space: the question goes to search exactly as asked, never cleaned.
QUESTION = 'Which "kiln" order in bay-north  was signed off, and by whom?'
CORPUS_QUESTION = "Which repair order is about the kiln in bay-north?"
FAKE_RESULT = "search: 3 documents about the kiln"
SCHEMAS = load_tool_schemas()
HANDLE = re.compile(r"^(D\d+) · ", re.MULTILINE)
WAIT_S = 5.0  # upper bound on the slow tool's wait; the test releases it at once
MEMORY = "Searched for the kiln order; echoed twice."


# ---------------------------------------------------------------- a test-only search tool
def search_tools(
    search: Callable[[SearchArgs], str] | None = None, seen: list[SearchArgs] | None = None
) -> list[Tool[Any]]:
    """The real search schema and arguments model with a scripted body; echo and finish."""

    def scripted(args: SearchArgs) -> str:
        if seen is not None:
            seen.append(args)
        return FAKE_RESULT

    def echo_run(args: EchoArgs) -> str:
        return f"echo: {args.word}"

    return [
        Tool(SCHEMAS.get(SEARCH), SearchArgs, search or scripted),
        Tool(ECHO, EchoArgs, echo_run),
        finish_tool(SCHEMAS),
    ]


def search_box(
    search: Callable[[SearchArgs], str] | None = None, seen: list[SearchArgs] | None = None
) -> ToolBox:
    return ToolBox(search_tools(search, seen))


class RaisingBox(ToolBox):
    """A tool box whose run_one itself raises on the first search (a bug, not a tool error)."""

    async def run_one(self, call: ToolCall) -> ToolRun:
        if call.id == FIRST_SEARCH_CALL_ID:
            raise RuntimeError("a bug in the box")
        return await super().run_one(call)


def make(
    script: list[ScriptEntry],
    box: ToolBox | None = None,
    *,
    limits: AgentLimits | None = None,
    clock: FakeClock | None = None,
    deadline: float = FAR_DEADLINE,
    question: str = QUESTION,
    **options: bool,
) -> tuple[Agent, FakeLLM]:
    """An agent over `box` (default: the test search box) with the default switch."""
    fake = FakeLLM(script)
    agent = Agent(
        question,
        fake,
        box or search_box(),
        deadline=deadline,
        clock=clock or FakeClock(),
        limits=limits or AgentLimits(),
        **options,
    )
    return agent, fake


def run(agent: Agent) -> AgentResult:
    return asyncio.run(agent.run())


def the_call(question: str = QUESTION) -> ToolCall:
    return ToolCall(id=FIRST_SEARCH_CALL_ID, name=SEARCH, arguments={"query": question})


# ---------------------------------------------------------------- settings
def test_it_is_on_by_default_with_a_short_bound() -> None:
    assert AGENT_FIRST_SEARCH is True  # the user wants it in the runs (2026-10-17)
    assert 0 < AGENT_FIRST_SEARCH_TIMEOUT_S <= 10  # seconds, not the embedder's minutes
    assert AgentLimits().first_search_timeout_s == AGENT_FIRST_SEARCH_TIMEOUT_S
    assert first_search_call(QUESTION) == the_call()


# ---------------------------------------------------------------- the first request
def test_it_runs_before_the_first_model_call_and_is_sent_as_a_code_written_call() -> None:
    seen: list[SearchArgs] = []
    ref: list[Agent] = []

    def first_model_call(request: ChatRequest) -> ModelReply:
        assert seen == [SearchArgs(query=QUESTION)]  # it already ran, once, nothing but the query
        return checked(ref, reply(finish()))(request)

    agent, fake = make([first_model_call], search_box(seen=seen))
    ref.append(agent)
    result = run(agent)
    assert result.stop == "finish"
    [request] = fake.requests
    assert list(request.messages[2:]) == [
        Message(role="assistant", tool_calls=(the_call(),)),
        Message(role="tool", tool_call_id=FIRST_SEARCH_CALL_ID, name=SEARCH, content=FAKE_RESULT),
    ]  # no budget note before the first step
    assert agent.history[0].raw is None  # code wrote it: no provider reply behind it
    assert_history_rules(agent.history, agent.question_prompt)


def test_it_uses_no_step_and_counts_as_one_tool_call() -> None:
    ref: list[Agent] = []
    agent, fake = make([checked(ref, reply(echo("c1"))), checked(ref, reply(finish()))])
    ref.append(agent)
    result = run(agent)
    assert (result.stop, result.steps, len(fake.requests)) == ("finish", 2, 2)
    assert result.tool_calls == 3  # the first search, the echo and the finish
    first, second = fake.requests
    assert first.messages[-1].tool_call_id == FIRST_SEARCH_CALL_ID  # no note on step 1
    assert second.messages[-1] == note(2)  # "Step 2 of 10": the step numbers do not move
    assert_history_rules(agent.history, agent.question_prompt)


def test_it_counts_toward_the_tool_call_cap() -> None:
    model = by_choice(reply(echo("c1")), reply(finish()))
    agent, fake = make([model] * 2, limits=AgentLimits(max_tool_calls=2))
    result = run(agent)
    # The first search (1) and the echo (2) reach the cap: the next step may only finish.
    assert [r.tool_choice for r in fake.requests] == ["auto", FORCED]
    assert (result.forced_by, result.tool_calls, result.steps) == ("tool_call_limit", 3, 2)


def test_to_gemini_sends_it_as_a_valid_model_turn_and_the_prompt_stays_append_only() -> None:
    agent, fake = make([reply(echo("c1")), reply(finish())])
    run(agent)
    first, second = (to_gemini(r.messages) for r in fake.requests)
    system, contents = first
    assert system == agent.system_prompt
    question, model, results = contents
    assert question.role == "user" and model.role == "model" and results.role == "user"
    [part] = model.parts or []
    assert part.function_call == genai_types.FunctionCall(
        id=FIRST_SEARCH_CALL_ID, name=SEARCH, args={"query": QUESTION}
    )
    # The API refuses a function call without a thought signature (step 1 probe).
    assert part.thought_signature == CHAT_STAND_IN_THOUGHT_SIGNATURE
    [answer] = results.parts or []
    assert answer.function_response == genai_types.FunctionResponse(
        id=FIRST_SEARCH_CALL_ID, name=SEARCH, response={"result": FAKE_RESULT}
    )
    # The next prompt starts with this one, byte for byte (the implicit prompt cache).
    dumps = [[c.model_dump_json() for c in cs] for cs in (contents, second[1])]
    for sent, after in itertools.pairwise(dumps):
        assert after[: len(sent)] == sent


# ---------------------------------------------------------------- skipped
def test_switched_off_there_is_no_first_search() -> None:
    seen: list[SearchArgs] = []
    agent, fake = make([reply(finish())], search_box(seen=seen), first_search=False)
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = run(agent)
    [request] = fake.requests
    assert len(request.messages) == 2 and seen == []
    assert result.tool_calls == 1  # the finish only
    [span] = tracer.named(FIRST_SEARCH_SPAN)
    assert (span.metadata["outcome"], span.metadata["reason"]) == ("skipped", "off")
    assert tracer.named(f"tool:{SEARCH}") == []


def test_the_answerer_passes_the_switch_on() -> None:
    seen: list[SearchArgs] = []
    box = search_box(seen=seen)

    class Session(FinishOnlySession):
        @property
        def toolbox(self) -> ToolBox:
            return box

    for on in (False, True):
        model = FakeLLM([reply(finish())])
        asyncio.run(AgentAnswerer(model, Session, first_search=on).answer(QUESTION))
        [request] = model.requests
        assert len(request.messages) == (4 if on else 2)
    assert seen == [SearchArgs(query=QUESTION)]  # only the run with the switch on searched


def test_a_finish_only_session_skips_it() -> None:
    model = FakeLLM([reply(finish())])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        asyncio.run(AgentAnswerer(model, FinishOnlySession).answer(QUESTION))
    [request] = model.requests
    assert len(request.messages) == 2  # system and question: nothing could be searched
    [span] = tracer.named(FIRST_SEARCH_SPAN)
    assert (span.metadata["outcome"], span.metadata["reason"]) == ("skipped", "no_search_tool")
    [agent_span] = tracer.named(AGENT_SPAN)
    assert agent_span.metadata["first_search"] == "skipped"


def test_with_no_time_before_the_reserve_it_is_skipped() -> None:
    seen: list[SearchArgs] = []
    agent, fake = make([reply(finish())], search_box(seen=seen), deadline=AGENT_FINISH_RESERVE_S)
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = run(agent)
    assert seen == []
    assert (result.stop, result.forced_by) == ("forced_finish", "deadline")
    [span] = tracer.named(FIRST_SEARCH_SPAN)
    assert (span.metadata["outcome"], span.metadata["reason"]) == ("skipped", "no_time")
    [request] = fake.requests
    assert request.tool_choice == FORCED and len(request.messages) == 2  # nothing added


def test_its_bound_never_reaches_into_the_finish_reserve() -> None:
    agent, _ = make([reply(finish())], deadline=AGENT_FINISH_RESERVE_S + 5.0)
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        run(agent)
    [span] = tracer.named(FIRST_SEARCH_SPAN)
    assert span.metadata["timeout_s"] == 5.0  # not the full AGENT_FIRST_SEARCH_TIMEOUT_S
    assert span.metadata["outcome"] == "ok"


def test_a_blank_question_is_not_searched() -> None:
    seen: list[SearchArgs] = []
    agent, fake = make([reply(finish())], search_box(seen=seen), question="   ")
    run(agent)
    assert seen == [] and len(fake.requests[0].messages) == 2
    assert agent.first_search == "skipped"


# ---------------------------------------------------------------- failed or slow
def raise_runtime(args: SearchArgs) -> str:
    raise RuntimeError("the index is gone")


def raise_input(args: SearchArgs) -> str:
    raise ToolInputError("no such source")


def raising_box() -> ToolBox:
    return RaisingBox(search_tools())


@pytest.mark.parametrize(
    ("make_box", "reason"),
    [
        (lambda: search_box(raise_runtime), "the tool gave an error (tool_failed)"),
        (lambda: search_box(raise_input), "the tool gave an error (tool_input)"),
        (raising_box, "RuntimeError: a bug in the box"),
    ],
    ids=["tool_raises", "tool_input_error", "box_raises"],
)
def test_a_failed_first_search_gives_the_note_and_the_run_goes_on(
    make_box: Callable[[], ToolBox], reason: str
) -> None:
    ref: list[Agent] = []
    script = [checked(ref, reply(echo("c1"))), checked(ref, reply(finish()))]
    agent, fake = make(script, make_box())
    ref.append(agent)
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = run(agent)
    assert (result.stop, result.steps) == ("finish", 2)
    assert result.tool_calls == 2  # the echo and the finish: the failed search is not counted
    first = fake.requests[0]
    assert list(first.messages[2:]) == [Message(role="user", content=FIRST_SEARCH_FAILED_NOTE)]
    assert agent.history[0] == Message(role="user", content=FIRST_SEARCH_FAILED_NOTE)
    assert all(m.tool_call_id != FIRST_SEARCH_CALL_ID for m in agent.history)
    assert_history_rules(agent.history, agent.question_prompt)
    [span] = tracer.named(FIRST_SEARCH_SPAN)
    assert (span.metadata["outcome"], span.metadata["reason"]) == ("failed", reason)
    assert span.metadata["counted"] is False
    assert span.output == FIRST_SEARCH_FAILED_NOTE  # what the model saw instead
    assert span.level == "WARNING"


def test_a_slow_first_search_times_out_gives_the_note_and_its_late_result_is_dropped() -> None:
    release = threading.Event()

    def slow(args: SearchArgs) -> str:
        release.wait(WAIT_S)  # like an embedding call that retries for minutes
        return "too late"

    def first_model_call(request: ChatRequest) -> ModelReply:
        release.set()  # the stuck thread may end now (asyncio.run waits for it at the end)
        return reply(finish())

    agent, fake = make(
        [first_model_call], search_box(slow), limits=AgentLimits(first_search_timeout_s=0.05)
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = run(agent)
    assert result.stop == "finish" and result.tool_calls == 1
    [request] = fake.requests
    assert list(request.messages[2:]) == [Message(role="user", content=FIRST_SEARCH_FAILED_NOTE)]
    assert all("too late" not in m.content for m in agent.history)
    [span] = tracer.named(FIRST_SEARCH_SPAN)
    assert span.metadata["outcome"] == "timed_out"
    assert span.metadata["reason"] == "no result within 0.05 s"
    assert span.metadata["timeout_s"] == 0.05 and span.level == "WARNING"
    assert span.metadata["elapsed_s"] < WAIT_S  # it did not wait for the tool
    [tool_span] = tracer.children(span)
    assert (tool_span.name, tool_span.level) == (f"tool:{SEARCH}", "ERROR")  # cancelled


# ---------------------------------------------------------------- tracing
def test_the_first_search_span_carries_its_fields() -> None:
    agent, _ = make([reply(finish())])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        run(agent)
    [agent_span] = tracer.named(AGENT_SPAN)
    first, step = tracer.children(agent_span)
    assert (first.name, step.name) == (FIRST_SEARCH_SPAN, STEP_SPAN)
    assert first.input == {"tool": SEARCH, "arguments": {"query": QUESTION}}
    assert first.output == FAKE_RESULT  # the result exactly as the model saw it
    meta = first.metadata
    assert (meta["outcome"], meta["reason"], meta["counted"]) == ("ok", None, True)
    assert meta["tool_call_id"] == FIRST_SEARCH_CALL_ID
    assert meta["timeout_s"] == AGENT_FIRST_SEARCH_TIMEOUT_S
    assert isinstance(meta["elapsed_s"], float) and meta["elapsed_s"] >= 0
    assert meta["result_chars"] == len(FAKE_RESULT)
    assert first.level == "DEFAULT"
    [tool_span] = tracer.children(first)  # the tool's own span nests under it
    assert tool_span.name == f"tool:{SEARCH}" and tool_span.output == FAKE_RESULT
    assert tool_span.metadata["tool_call_id"] == FIRST_SEARCH_CALL_ID
    assert agent_span.metadata["first_search"] == "ok"
    limits = agent_span.metadata["limits"]
    assert isinstance(limits, dict) and limits["first_search_timeout_s"] == 8.0


# ---------------------------------------------------------------- compaction
def test_compaction_folds_it_in_but_never_counts_it_as_a_step() -> None:
    script: list[ScriptEntry] = [
        reply(echo("c1", "first")),
        reply(echo("c2", "second")),
        fake_reply(MEMORY),  # the compaction call
        reply(finish()),
    ]
    # Over the limit from the start: only the steps decide when there is something to fold.
    agent, fake = make(script, limits=AgentLimits(compact_at_tokens=1, compact_keep_steps=1))
    result = run(agent)
    assert result.stop == "finish" and result.steps == 3  # the compaction call is no step
    [event] = result.compactions
    assert (event.steps_replaced, event.steps_kept) == (1, 1)  # the first search is no step
    transcript = fake.requests[2].messages[1].content
    arguments = json.dumps({"query": QUESTION}, ensure_ascii=False)
    assert (
        "[before step 1: the service searched the question as asked]\n"
        f"call search({arguments}) (id {FIRST_SEARCH_CALL_ID})"
    ) in transcript
    assert f"[before step 1: result of search, call {FIRST_SEARCH_CALL_ID}]\n{FAKE_RESULT}" in (
        transcript
    )
    assert "[step 1: the agent]" in transcript and "[step 0" not in transcript
    assert all(m.tool_call_id != FIRST_SEARCH_CALL_ID for m in agent.history)  # folded in
    assert_history_rules(agent.history, agent.question_prompt)


# ---------------------------------------------------------------- the real corpus tools
@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_tool_corpus(tmp_path_factory.mktemp("first-search"))


@pytest.fixture(scope="module")
def tools(built: Built) -> CorpusTools:
    runtime = built.runtime
    assert runtime.index_path is not None and runtime.chunks is not None
    reader = SqliteReader(runtime.index_path)
    return corpus_session_factory(reader, runtime.chunks, runtime.vectors, runtime.embedder)


def searched(tools: CorpusTools, query: str) -> tuple[str, list[str]]:
    """What a new session's search shows for `query`, and the ids of its D1, D2, ..."""
    session = tools()
    [message] = asyncio.run(session.toolbox.run([ToolCall("c1", SEARCH, {"query": query})]))
    handles = HANDLE.findall(message.content)
    return message.content, session.resolve_citations(handles)


def test_with_the_corpus_tools_it_shows_the_search_tool_s_own_result(
    tools: CorpusTools, built: Built
) -> None:
    session = tools()
    model = FakeLLM([reply(finish(citations=["D1"]))])
    agent = Agent(CORPUS_QUESTION, model, session.toolbox, deadline=FAR_DEADLINE, clock=FakeClock())
    run(agent)
    [request] = model.requests
    assistant, tool = request.messages[2:]
    assert assistant.tool_calls == (the_call(CORPUS_QUESTION),)
    expected, ids = searched(tools, CORPUS_QUESTION)
    assert tool.content == expected  # word for word, with this request's own handles
    assert HANDLE.findall(tool.content)[:2] == ["D1", "D2"] and ids
    assert ("query", (CORPUS_QUESTION,)) in built.embedder.calls  # dense search too
    assert session.resolve_citations(["D1"]) == ids[:1]


def test_through_the_answerer_its_handles_resolve_as_citations_and_are_named(
    tools: CorpusTools,
) -> None:
    model = FakeLLM([reply(finish(answer="The kiln order is D1.", citations=["D1"]))])
    result = asyncio.run(AgentAnswerer(model, tools).answer(CORPUS_QUESTION))
    _, ids = searched(tools, CORPUS_QUESTION)
    assert result.document_ids == ids[:1]  # the first search's D1, resolved
    title = " ".join(tools.summaries[ids[0]].display_title.split())
    assert "D1" not in result.answer and title in result.answer  # named for the reader
