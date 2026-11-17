"""AgentAnswerer and tool sessions: a new session and agent per question, resolved ids,
nothing cited after a fallback, handles named in the answer text, the deadline margin,
errors. One live end-to-end test.
"""

import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from corpusqa.agent.agent import COMPACT_SPAN, AgentError, AgentLimits, system_text
from corpusqa.agent.answerer import (
    ANSWER_HANDLES_SPAN,
    CITATIONS_SPAN,
    AgentAnswerer,
    AnswerResult,
)
from corpusqa.agent.handles import CleanedAnswer, Replacement
from corpusqa.agent.session import FinishOnlySession, ToolSession
from corpusqa.agent.tools import FINISH_TOOL_NAME, Tool, ToolBox, finish_tool
from corpusqa.config import (
    AGENT_CALL_TIMEOUT_S,
    AGENT_DEADLINE_MARGIN_S,
    AGENT_FINISH_RESERVE_S,
    ASK_DEADLINE_S,
    Settings,
)
from corpusqa.llm.errors import ChatError
from corpusqa.llm.fakes import ChatRequest, FakeLLM
from corpusqa.llm.interfaces import ModelReply, ToolSchema
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_text_prompt
from tests.unit.test_agent import FakeClock, echo, echo_box, finish, reply

QUESTION = "Who signed off the Quillmoor change?"


class MappedSession:
    """A test session: echo and finish; citations resolve through a fixed map, and the
    answer text's handles are named through another (`names`: handle -> its name)."""

    def __init__(self, ids: dict[str, list[str]], names: dict[str, str] | None = None) -> None:
        self._toolbox = echo_box()
        self._ids = ids
        self._names = names or {}
        self.resolved: list[tuple[str, ...]] = []
        self.cleaned: list[tuple[str, str]] = []  # (answer, question) as given

    @property
    def toolbox(self) -> ToolBox:
        return self._toolbox

    def resolve_citations(self, citations: Sequence[str]) -> list[str]:
        self.resolved.append(tuple(citations))
        out: list[str] = []
        for citation in citations:
            for doc_id in self._ids.get(citation, []):
                if doc_id not in out:
                    out.append(doc_id)
        return out

    def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
        self.cleaned.append((answer, question))
        replacements = []
        for handle, name in self._names.items():
            if handle in answer:
                replacements.append(Replacement(handle, name, answer.count(handle)))
                answer = answer.replace(handle, name)
        return CleanedAnswer(answer, tuple(replacements))


class Factory:
    """Counts the sessions it makes."""

    def __init__(
        self, ids: dict[str, list[str]] | None = None, names: dict[str, str] | None = None
    ) -> None:
        self.sessions: list[MappedSession] = []
        self._ids = ids or {}
        self._names = names or {}

    def __call__(self) -> ToolSession:
        session = MappedSession(self._ids, self._names)
        self.sessions.append(session)
        return session


def answer(answerer: AgentAnswerer, question: str = QUESTION) -> AnswerResult:
    return asyncio.run(answerer.answer(question))


# ---------------------------------------------------------------- sessions
def test_finish_only_session_has_only_finish_and_resolves_nothing() -> None:
    session = FinishOnlySession()
    assert [s.name for s in session.toolbox.schemas()] == [FINISH_TOOL_NAME]
    assert session.resolve_citations(["doc-1", "D1"]) == []
    assert session.clean_answer("See D1 (R1).", "What is D1?") == CleanedAnswer("See D1 (R1).")
    assert FinishOnlySession().toolbox is not session.toolbox  # nothing shared


def test_finish_only_session_runs_a_whole_question() -> None:
    answerer = AgentAnswerer(FakeLLM([reply(finish(citations=["doc-1"]))]), FinishOnlySession)
    assert answer(answerer) == AnswerResult(
        answer="It moved to May because the supplier was late.", document_ids=[]
    )


# ---------------------------------------------------------------- answers and ids
def test_the_answer_comes_with_the_resolved_ids() -> None:
    factory = Factory({"R1": ["uuid-a", "uuid-b"], "D2": ["uuid-b", "uuid-c"]})
    model = FakeLLM([reply(echo("c1")), reply(finish(citations=["R1", "nope", "D2"]))])
    result = answer(AgentAnswerer(model, factory))
    assert result.answer == "It moved to May because the supplier was late."
    assert result.document_ids == ["uuid-a", "uuid-b", "uuid-c"]
    [session] = factory.sessions
    assert session.resolved == [("R1", "nope", "D2")]  # the model's citations, as given


def test_every_question_gets_the_catalog_in_its_system_message() -> None:
    catalog = "- source 'ledger': 4 documents\n  - stage: text, in 4 docs (100%), 2 values"
    model = FakeLLM([reply(finish())] * 2)
    answerer = AgentAnswerer(model, Factory(), catalog=catalog)
    answer(answerer, "First one?")
    answer(answerer, "Second one?")
    systems = [request.messages[0].content for request in model.requests]
    assert systems[0] == systems[1] == system_text(load_text_prompt("agent"), catalog)


def test_a_forced_finish_still_resolves_its_citations() -> None:
    factory = Factory({"D1": ["uuid-a"]})
    model = FakeLLM([reply(echo("c1")), reply(finish(citations=["D1"]))])
    answerer = AgentAnswerer(model, factory, limits=AgentLimits(max_steps=1))
    assert answer(answerer).document_ids == ["uuid-a"]


def test_a_fallback_answer_cites_nothing() -> None:
    factory = Factory({"D1": ["uuid-a"]})
    model = FakeLLM(
        [reply(echo("c1"), text="So far: D1 says May."), ChatError("x"), ChatError("x")]
    )
    answerer = AgentAnswerer(model, factory, limits=AgentLimits(max_steps=1))
    result = answer(answerer)
    assert result == AnswerResult(answer="So far: D1 says May.", document_ids=[])
    assert factory.sessions[0].resolved == []  # not even asked


def test_the_answer_leaves_with_its_handles_named_and_the_ids_unchanged() -> None:
    factory = Factory({"D1": ["uuid-a"]}, names={"D1": '"Belt order" (tracker, 2025-01-02)'})
    raw = "It moved to May (see D1)."
    model = FakeLLM([reply(finish(answer=raw, citations=["D1"]))])
    result = answer(AgentAnswerer(model, factory))
    assert result == AnswerResult(
        answer='It moved to May (see "Belt order" (tracker, 2025-01-02)).',
        document_ids=["uuid-a"],
    )
    [session] = factory.sessions
    assert session.cleaned == [(raw, QUESTION)]  # the model's answer and the question
    assert session.resolved == [("D1",)]  # citations resolved as given, not renamed


def test_a_fallback_answer_has_its_handles_named_too() -> None:
    factory = Factory({"D1": ["uuid-a"]}, names={"D1": "the belt order"})
    model = FakeLLM(
        [reply(echo("c1"), text="So far: D1 says May."), ChatError("x"), ChatError("x")]
    )
    answerer = AgentAnswerer(model, factory, limits=AgentLimits(max_steps=1))
    assert answer(answerer) == AnswerResult(
        answer="So far: the belt order says May.", document_ids=[]
    )


def test_a_failed_naming_sends_the_answer_as_written() -> None:
    class Broken(MappedSession):
        def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
            raise KeyError("doc-9")

    model = FakeLLM([reply(finish(answer="See D1.", citations=["D1"]))])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = answer(AgentAnswerer(model, lambda: Broken({"D1": ["uuid-a"]})))
    assert result == AnswerResult(answer="See D1.", document_ids=["uuid-a"])
    [span] = tracer.named(ANSWER_HANDLES_SPAN)
    assert span.level == "ERROR"
    assert span.output == {"answer": "See D1.", "replacements": []}


def test_an_agent_error_propagates() -> None:
    model = FakeLLM([reply(echo("c1")), ChatError("x"), ChatError("x")])
    answerer = AgentAnswerer(model, Factory(), limits=AgentLimits(max_steps=1))
    with pytest.raises(AgentError):
        answer(answerer)


def test_each_question_gets_a_new_session_and_a_new_agent() -> None:
    def own_question(request: ChatRequest) -> ModelReply:
        question = request.messages[1].content
        if len(request.messages) == 2:
            return reply(echo("c1"))
        return reply(finish(answer=f"About: {question}"))

    model = FakeLLM([own_question] * 4)
    factory = Factory()
    answerer = AgentAnswerer(model, factory)

    async def both() -> list[AnswerResult]:
        return await asyncio.gather(answerer.answer("First one?"), answerer.answer("Second one?"))

    first, second = asyncio.run(both())
    assert len(factory.sessions) == 2 and factory.sessions[0] is not factory.sessions[1]
    assert "First one?" in first.answer and "Second one?" not in first.answer
    assert "Second one?" in second.answer and "First one?" not in second.answer
    for request in model.requests:  # no request mixes the two questions
        texts = " ".join(m.content for m in request.messages)
        assert ("First one?" in texts) != ("Second one?" in texts)


# ---------------------------------------------------------------- the deadline
def test_the_agent_deadline_ends_the_margin_before_the_servers() -> None:
    model = FakeLLM([reply(finish())])
    clock = FakeClock(now=1_000.0)
    answerer = AgentAnswerer(
        model, Factory(), ask_deadline_s=50.0, deadline_margin_s=10.0, clock=clock
    )
    answer(answerer)
    # 40 s for the agent; a normal step must end before the 30 s finish reserve.
    assert model.requests[0].timeout_s == pytest.approx(min(40.0 - AGENT_FINISH_RESERVE_S, 40.0))


def test_the_agent_deadline_counts_from_the_time_left_the_server_passes() -> None:
    # The request waited 30 s for a slot: 120 s of its 150 s are left, so the agent gets
    # 110 s (10 s margin), and the forced finish comes at the deadline counted from there.
    clock = FakeClock()

    def at_the_reserve(request: ChatRequest) -> ModelReply:
        clock.now = 110.0 - AGENT_FINISH_RESERVE_S  # exactly the reserve left
        return reply(echo("c1"))

    model = FakeLLM([at_the_reserve, reply(finish())])
    answerer = AgentAnswerer(
        model, Factory(), ask_deadline_s=150.0, deadline_margin_s=10.0, clock=clock
    )
    asyncio.run(answerer.answer(QUESTION, time_left_s=120.0))
    assert [r.tool_choice for r in model.requests] == ["auto", (FINISH_TOOL_NAME,)]
    assert model.requests[1].timeout_s == pytest.approx(AGENT_FINISH_RESERVE_S)


@pytest.mark.parametrize("left", [10.0, 3.0, 0.0, -1.0])
def test_no_time_left_after_the_margin_is_a_timeout_at_once(left: float) -> None:
    model = FakeLLM([])
    answerer = AgentAnswerer(model, Factory(), deadline_margin_s=10.0, clock=FakeClock())
    with pytest.raises(TimeoutError, match="no time to answer"):
        asyncio.run(answerer.answer(QUESTION, time_left_s=left))
    assert model.requests == []  # no model call was made


def test_with_the_default_deadline_the_first_call_gets_its_full_timeout() -> None:
    model = FakeLLM([reply(finish())])
    answer(AgentAnswerer(model, Factory(), clock=FakeClock()))
    assert model.requests[0].timeout_s == AGENT_CALL_TIMEOUT_S


def test_the_default_budget_leaves_room_for_steps_and_the_forced_finish() -> None:
    budget = ASK_DEADLINE_S - AGENT_DEADLINE_MARGIN_S
    assert AGENT_DEADLINE_MARGIN_S > 0
    assert budget > AGENT_FINISH_RESERVE_S + AGENT_CALL_TIMEOUT_S


@pytest.mark.parametrize("margin", [-1.0, 50.0, 60.0])
def test_a_margin_outside_the_deadline_is_refused(margin: float) -> None:
    with pytest.raises(ValueError, match="deadline_margin_s"):
        AgentAnswerer(FakeLLM([]), Factory(), ask_deadline_s=50.0, deadline_margin_s=margin)


# ---------------------------------------------------------------- tracing
def test_the_citations_span_shows_what_was_cited_and_what_it_resolved_to() -> None:
    factory = Factory({"R1": ["uuid-a", "uuid-b"]})
    model = FakeLLM([reply(finish(citations=["R1", "nope"]))])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        answer(AgentAnswerer(model, factory))
    [ask] = tracer.traces("ask")
    assert [o.name for o in tracer.children(ask)] == ["agent", CITATIONS_SPAN, ANSWER_HANDLES_SPAN]
    [span] = tracer.named(CITATIONS_SPAN)
    assert span.input == {"citations": ["R1", "nope"]}
    assert span.output == {"document_ids": ["uuid-a", "uuid-b"]}
    assert span.metadata == {"stop": "finish", "citations": 2, "document_ids": 2}


def test_the_answer_handles_span_records_each_replacement_and_how_many() -> None:
    names = {"D1": '"Belt order" (tracker, 2025-01-02)', "R1": "the 2 matching tracker records"}
    raw = "R1 are open; D1 is the newest, and D1 names the fix."
    model = FakeLLM([reply(finish(answer=raw, citations=["R1"]))])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = answer(AgentAnswerer(model, Factory(names=names)))
    [span] = tracer.named(ANSWER_HANDLES_SPAN)
    assert span.input == {"answer": raw}
    assert span.output == {
        "answer": result.answer,
        "replacements": [
            {"handle": "D1", "text": '"Belt order" (tracker, 2025-01-02)', "count": 2},
            {"handle": "R1", "text": "the 2 matching tracker records", "count": 1},
        ],
    }
    assert span.metadata == {"replaced": 3, "kept": []}
    assert "D1" not in result.answer and "R1" not in result.answer


# ---------------------------------------------------------------- live (skipped by default)
class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str


LOOKUP = ToolSchema(
    name="lookup",
    description=(
        "Look up one word in a small test glossary. Returns the record id and the entry. "
        "Example: lookup(word='quokka')."
    ),
    parameters={
        "type": "object",
        "properties": {"word": {"type": "string", "description": "One word, lower case."}},
        "required": ["word"],
        "additionalProperties": False,
    },
)
# Invented test records. Each entry is padded with filler notes so that a compaction has
# something to shrink; the first one names the second word, so the model needs two steps.
FILLER = " ".join(
    f"Note {i}: this line is filler for the test and says nothing." for i in range(30)
)
GLOSSARY = {
    "quokka": (
        "rec-q17",
        "A quokka is a small wallaby. To finish the question, also look up the word 'axolotl'.",
    ),
    "axolotl": ("rec-a04", "An axolotl is a salamander that keeps its gills as an adult."),
}


class GlossarySession:
    """The live test's session: a glossary lookup and finish; ids resolve to test uuids."""

    def __init__(self) -> None:
        self._toolbox = ToolBox(
            [Tool(schema=LOOKUP, args=LookupArgs, run=self._lookup), finish_tool()]
        )

    @property
    def toolbox(self) -> ToolBox:
        return self._toolbox

    def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
        return CleanedAnswer(answer)  # it issues no handles

    def resolve_citations(self, citations: Sequence[str]) -> list[str]:
        known = {record: f"uuid-{record}" for record, _ in GLOSSARY.values()}
        out: list[str] = []
        for citation in citations:
            doc_id = known.get(citation)
            if doc_id is not None and doc_id not in out:
                out.append(doc_id)
        return out

    @staticmethod
    def _lookup(args: LookupArgs) -> str:
        entry = GLOSSARY.get(args.word.strip().lower())
        if entry is None:
            return f"No entry for {args.word!r}."
        record, text = entry
        return f"Record {record}: {text}\n{FILLER}"


@pytest.mark.live
def test_live_end_to_end_with_a_real_compaction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from corpusqa.llm.client import GeminiChat  # only this test needs the SDK adapter

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    model = GeminiChat.from_settings(Settings(corpus_dir=tmp_path, port=8080, gemini_api_key=key))
    # Any prompt is over this limit, so the agent compacts as soon as it holds two steps.
    answerer = AgentAnswerer(model, GlossarySession, limits=AgentLimits(compact_at_tokens=1))
    question = (
        "Use the lookup tool on the word 'quokka', and follow what its entry says. "
        "Then answer: what are the two animals? Cite the record ids."
    )
    tracer = RecordingTracer()
    with caplog.at_level(logging.INFO), tracer.trace("ask"):
        result = asyncio.run(answerer.answer(question))
    log = logging.getLogger(__name__)
    [agent_span] = tracer.named("agent")
    compacts = tracer.named(COMPACT_SPAN)
    log.info(
        "live agent: stop %s, steps %s, tool calls %s, compactions %s, $%.5f, answer %r, ids %s",
        agent_span.metadata.get("stop"),
        agent_span.metadata.get("steps"),
        agent_span.metadata.get("tool_calls"),
        [
            (
                c.metadata.get("path"),
                c.metadata.get("tokens_before"),
                c.metadata.get("tokens_after"),
            )
            for c in compacts
        ],
        agent_span.metadata.get("cost_usd"),
        result.answer,
        result.document_ids,
    )
    assert result.answer.strip()
    assert agent_span.metadata["stop"] in ("finish", "forced_finish")
    assert compacts, "no compaction happened"
    assert compacts[0].metadata["path"] == "model"  # the real compaction call answered
    assert compacts[0].metadata["no_gain"] is False
    assert any(g.metadata.get("prompt_name") == "compact" for g in tracer.named("chat"))
    assert set(result.document_ids) <= {"uuid-rec-q17", "uuid-rec-a04"}
