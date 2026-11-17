"""The Agent: prompt, history rules, the loop with its limits, forced finish, fallbacks, spans.

Offline with FakeLLM. A reply that depends on the request (its tool choice, or the time
on a fake clock) is a function of the request; time never really passes.
"""

import asyncio
import itertools
from collections.abc import Callable, Iterable, Sequence

import pytest
from pydantic import BaseModel, ConfigDict

from corpusqa.agent.agent import (
    AGENT_SPAN,
    BUDGET_CLOSE,
    BUDGET_OPEN,
    CATALOG_SLOT,
    FIRST_SEARCH_SPAN,
    FORCED_NOTE,
    NO_CATALOG,
    NOT_RUN_AFTER_FINISH,
    NOT_RUN_NOT_ALLOWED,
    NOT_RUN_OVER_CAP,
    NUDGE,
    STEP_SPAN,
    Agent,
    AgentError,
    AgentLimits,
    AgentResult,
    budget_text,
    question_text,
    system_text,
)
from corpusqa.agent.tools import (
    FINISH_RESULT,
    FINISH_TOOL_NAME,
    Tool,
    ToolBox,
    finish_tool,
    read_finish,
)
from corpusqa.llm.client import to_gemini
from corpusqa.llm.errors import ChatError, ChatTimeout
from corpusqa.llm.fakes import ChatRequest, FakeLLM, ScriptEntry, fake_reply
from corpusqa.llm.interfaces import Message, ModelReply, ToolCall, ToolSchema, Usage
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import PromptError, PromptText, load_text_prompt, load_tool_schemas

QUESTION = "When did the Quillmoor rollout move, and why?"
FORCED = (FINISH_TOOL_NAME,)
# Room enough that no test hits the deadline unless it moves the clock.
FAR_DEADLINE = 1_000.0


# ---------------------------------------------------------------- invented test tool
class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str


ECHO = ToolSchema(
    name="echo",
    description="Echo one word back.",
    parameters={
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
        "additionalProperties": False,
    },
)


def echo_box(seen: list[str] | None = None) -> ToolBox:
    def echo(args: EchoArgs) -> str:
        if seen is not None:
            seen.append(args.word)
        return f"echo: {args.word}"

    return ToolBox([Tool(schema=ECHO, args=EchoArgs, run=echo), finish_tool()])


# ---------------------------------------------------------------- helpers
class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def echo(call_id: str, word: str = "hi") -> ToolCall:
    return ToolCall(id=call_id, name="echo", arguments={"word": word})


def finish(
    call_id: str = "f1",
    answer: str = "It moved to May because the supplier was late.",
    citations: Sequence[str] = ("doc-1",),
    not_found: Sequence[str] = (),
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=FINISH_TOOL_NAME,
        arguments={"answer": answer, "citations": list(citations), "not_found": list(not_found)},
    )


BAD_FINISH = ToolCall(
    id="fb", name=FINISH_TOOL_NAME, arguments={"answer": "  ", "citations": [], "not_found": []}
)


def reply(*calls: ToolCall, text: str = "", usage: Usage | None = None) -> ModelReply:
    return fake_reply(text, calls, usage=usage)


def note(step: int, max_steps: int = AgentLimits().max_steps) -> Message:
    """The budget note the agent adds before model call `step` (every call after the first)."""
    return Message(role="user", content=budget_text(step, max_steps))


def forced_note() -> Message:
    """The budget note before a forced finish."""
    return Message(role="user", content=f"{BUDGET_OPEN}{FORCED_NOTE}{BUDGET_CLOSE}")


def by_choice(normal: ModelReply, forced: ModelReply) -> Callable[[ChatRequest], ModelReply]:
    """A reply that depends on whether the step may only call finish."""
    return lambda request: forced if request.tool_choice == FORCED else normal


def make_agent(
    script: Iterable[ScriptEntry],
    *,
    limits: AgentLimits | None = None,
    clock: FakeClock | None = None,
    deadline: float = FAR_DEADLINE,
    seen: list[str] | None = None,
    question: str = QUESTION,
) -> tuple[Agent, FakeLLM]:
    fake = FakeLLM(script)
    agent = Agent(
        question,
        fake,
        echo_box(seen),
        deadline=deadline,
        clock=clock or FakeClock(),
        limits=limits or AgentLimits(),
    )
    return agent, fake


def run(agent: Agent) -> AgentResult:
    return asyncio.run(agent.run())


def assert_history_rules(history: Sequence[Message], question_prompt: str) -> None:
    """Plan section 3: system and question not in history; every call answered in order."""
    assert all(m.role in ("assistant", "tool", "user") for m in history)
    assert all(m.content != question_prompt for m in history if m.role == "user")
    assert all(m.raw is None for m in history if m.role != "assistant")
    index = 0
    while index < len(history):
        message = history[index]
        assert message.role != "tool", f"tool message {index} does not follow its call"
        calls = message.tool_calls
        answers = history[index + 1 : index + 1 + len(calls)]
        assert [a.role for a in answers] == ["tool"] * len(calls)
        assert [a.tool_call_id for a in answers] == [c.id for c in calls]
        assert [a.name for a in answers] == [c.name for c in calls]
        index += 1 + len(calls)


def checked(agent_ref: list[Agent], entry: ModelReply) -> Callable[[ChatRequest], ModelReply]:
    """A script entry that first checks the prompt it was sent (= the state after the last step)."""

    def play(request: ChatRequest) -> ModelReply:
        agent = agent_ref[0]
        assert request.messages[0] == Message(role="system", content=agent.system_prompt)
        assert request.messages[1] == Message(role="user", content=agent.question_prompt)
        assert_history_rules(request.messages[2:], agent.question_prompt)
        return entry

    return play


# ---------------------------------------------------------------- prompts
def test_shipped_agent_and_question_prompts_load() -> None:
    system = load_text_prompt("agent")
    question = load_text_prompt("question")
    assert (system.version, question.version) == ("7", "1")
    assert "finish" in system.text
    assert system.text.count(CATALOG_SLOT) == 1
    assert question.text.count("{question}") == 1
    assert BUDGET_OPEN in system.text  # the model is told what the budget note is
    assert system.text.count("# When to stop\n") == 1  # the stop rules (user, 2026-10-17)


def test_the_system_prompt_has_one_guide_line_per_tool() -> None:
    lines = load_text_prompt("agent").text.splitlines()
    for schema in load_tool_schemas().tools:  # every tool of tools.json, finish included
        assert sum(line.startswith(f"{schema.name}: ") for line in lines) == 1, schema.name


def test_the_catalog_goes_into_its_one_slot_and_braces_in_it_are_kept() -> None:
    template = PromptText(name="s", version="1", text="Rules.\n{catalog}\nEnd.")
    catalog = "- source 'x': 3 documents\n  - a {b} field"
    assert system_text(template, catalog) == f"Rules.\n{catalog}\nEnd."
    assert system_text(template, "  \n") == f"Rules.\n{NO_CATALOG}\nEnd."


@pytest.mark.parametrize("text", ["No slot here.", "{catalog} and {catalog}"])
def test_a_system_template_without_exactly_one_catalog_slot_is_an_error(text: str) -> None:
    with pytest.raises(PromptError, match="exactly once"):
        system_text(PromptText(name="s", version="1", text=text), "catalog")


def test_the_agent_sends_its_catalog_in_the_system_message() -> None:
    catalog = "- source 'ledger': 4 documents\n  - stage: text, in 4 docs (100%), 2 values"
    fake = FakeLLM([reply(finish())])
    agent = Agent(
        QUESTION, fake, echo_box(), deadline=FAR_DEADLINE, clock=FakeClock(), catalog=catalog
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        run(agent)
    [request] = fake.requests
    system = request.messages[0]
    assert system == Message(role="system", content=system_text(load_text_prompt("agent"), catalog))
    assert catalog in system.content and CATALOG_SLOT not in system.content
    [span] = tracer.named(AGENT_SPAN)
    assert span.metadata["catalog_chars"] == len(catalog)
    assert span.metadata["system_chars"] == len(system.content)


def test_the_question_goes_into_its_one_slot_and_braces_in_it_are_kept() -> None:
    template = PromptText(name="q", version="1", text="Q: {question}\nGo.")
    assert question_text("a {b} c", template) == "Q: a {b} c\nGo."


@pytest.mark.parametrize("text", ["No slot here.", "{question} and {question}"])
def test_a_question_template_without_exactly_one_slot_is_an_error(text: str) -> None:
    with pytest.raises(PromptError, match="exactly once"):
        question_text("x", PromptText(name="q", version="1", text=text))


# ---------------------------------------------------------------- build_prompt
def test_build_prompt_is_system_question_then_history() -> None:
    agent, fake = make_agent([reply(echo("c1", "quokka")), reply(finish())])
    before = agent.build_prompt()
    assert before == [
        Message(role="system", content=system_text(load_text_prompt("agent"), "")),
        Message(role="user", content=question_text(QUESTION, load_text_prompt("question"))),
    ]
    assert QUESTION in before[1].content
    assert NO_CATALOG in before[0].content  # no corpus tools here: no catalog
    run(agent)
    second = fake.requests[1].messages
    assert list(second[:2]) == before
    # The model's action, then the tool result, in order.
    assert second[2] == Message(role="assistant", tool_calls=(echo("c1", "quokka"),))
    assert second[3] == Message(role="tool", tool_call_id="c1", name="echo", content="echo: quokka")
    assert list(second[4:]) == [note(2)]  # then the budget note, last
    assert agent.build_prompt() == [*before, *agent.history]


# ---------------------------------------------------------------- history rules
def test_history_rules_hold_after_every_step() -> None:
    ref: list[Agent] = []
    script = [
        reply(echo("c1"), echo("c2", "axolotl")),
        reply(text="Let me think."),  # no tool call: the nudge follows
        reply(echo("c3"), BAD_FINISH),
        reply(*(echo(f"d{i}") for i in range(6))),  # more than the cap
        reply(finish("f9"), echo("c4")),
    ]
    agent, fake = make_agent(
        [checked(ref, entry) for entry in script], limits=AgentLimits(max_calls_per_step=4)
    )
    ref.append(agent)
    result = run(agent)
    assert agent.finished is not None and len(fake.requests) == 5
    assert_history_rules(agent.history, agent.question_prompt)
    assert result.stop == "finish"


# ---------------------------------------------------------------- the budget note
# Run 1 (smoke 20261017T013336Z): three runs used all 10 steps although every cited
# record was read by step 3, and the model never saw how many steps it had left.
def test_the_budget_note_says_the_step_and_the_steps_left() -> None:
    assert budget_text(2, 10) == "<budget>Step 2 of 10. 8 steps left after this one.</budget>"
    assert budget_text(9, 10) == "<budget>Step 9 of 10. 1 step left after this one.</budget>"
    assert budget_text(10, 10) == "<budget>Step 10 of 10. No steps left after this one.</budget>"
    assert budget_text(11, 10, forced=True) == (
        "<budget>Out of budget. Only finish may be called now: "
        "answer from the records you have read.</budget>"
    )


def test_every_call_after_the_first_ends_with_its_note_and_extends_the_last_prompt() -> None:
    script = [
        reply(echo("c1")),
        reply(text="Let me think."),  # no tool call: the nudge follows
        reply(echo("c3"), echo("c4")),
        reply(echo("c5")),  # the last normal step
        reply(finish()),  # the forced finish (step limit)
    ]
    agent, fake = make_agent(script, limits=AgentLimits(max_steps=4))
    result = run(agent)
    assert (result.stop, result.forced_by, result.steps) == ("forced_finish", "step_limit", 5)
    requests = fake.requests
    assert len(requests[0].messages) == 2  # the first call: system and question only
    assert [r.messages[-1] for r in requests[1:]] == [
        note(2, 4),
        note(3, 4),
        note(4, 4),
        forced_note(),
    ]
    assert requests[2].messages[-2:] == (Message(role="user", content=NUDGE), note(3, 4))
    # Nothing already sent ever changes: each prompt is the last one plus new messages.
    for sent, after in itertools.pairwise(requests):
        assert after.messages[: len(sent.messages)] == sent.messages
        assert len(after.messages) > len(sent.messages)
    # The same, byte for byte, as Gemini is sent it (the implicit prompt cache keys on it).
    gemini = [to_gemini(r.messages) for r in requests]
    assert len({system for system, _ in gemini}) == 1
    dumps = [[content.model_dump_json() for content in contents] for _, contents in gemini]
    for sent, after in itertools.pairwise(dumps):
        assert after[: len(sent)] == sent
    assert_history_rules(agent.history, agent.question_prompt)
    assert [m for m in agent.history if m.role == "user"] == [
        note(2, 4),
        Message(role="user", content=NUDGE),
        note(3, 4),
        note(4, 4),
        forced_note(),
    ]


def test_a_failed_call_keeps_no_note_and_the_next_call_sends_its_own() -> None:
    agent, fake = make_agent([reply(echo("c1")), ChatError("down"), reply(finish())])
    result = run(agent)
    assert (result.stop, result.forced_by) == ("forced_finish", "model_error")
    _, failed, forced = fake.requests
    assert failed.messages[-1] == note(2)
    assert forced.messages == (*failed.messages[:-1], forced_note())
    assert [m for m in agent.history if m.role == "user"] == [forced_note()]
    assert_history_rules(agent.history, agent.question_prompt)


def test_a_retried_forced_finish_sends_the_same_prompt() -> None:
    agent, fake = make_agent(
        [reply(echo("c1")), ChatTimeout("slow"), reply(finish())], limits=AgentLimits(max_steps=1)
    )
    assert run(agent).stop == "forced_finish"
    _, failed, retry = fake.requests
    assert failed.messages == retry.messages
    assert retry.messages[-1] == forced_note()


# ---------------------------------------------------------------- finish
def test_a_good_finish_ends_the_run() -> None:
    agent, fake = make_agent([reply(finish(not_found=["who approved it"]))])
    result = asyncio.run(agent.run())
    assert result.stop == "finish" and result.forced_by is None
    assert result.answer == "It moved to May because the supplier was late."
    assert result.citations == ("doc-1",)
    assert result.not_found == ("who approved it",)
    assert (result.steps, result.tool_calls) == (1, 1)
    [request] = fake.requests
    assert request.tool_choice == "auto"
    assert request.tools == tuple(agent.tools.schemas())
    assert (request.prompt_name, request.prompt_version) == ("agent", "7")
    assert agent.history[-1].content == FINISH_RESULT


def test_other_calls_in_a_reply_with_a_good_finish_are_not_run() -> None:
    seen: list[str] = []
    agent, _ = make_agent([reply(echo("c1"), finish("f1"), echo("c2"))], seen=seen)
    result = asyncio.run(agent.run())
    assert result.stop == "finish"
    assert seen == []  # the echo tool never ran
    assert [m.content for m in agent.history[1:]] == [
        NOT_RUN_AFTER_FINISH,
        FINISH_RESULT,
        NOT_RUN_AFTER_FINISH,
    ]
    assert result.tool_calls == 1


def test_the_first_good_finish_of_a_reply_counts() -> None:
    agent, _ = make_agent([reply(BAD_FINISH, finish("f1", answer="First."), finish("f2"))])
    result = asyncio.run(agent.run())
    assert result.answer == "First."
    assert [m.content for m in agent.history[1:]] == [
        NOT_RUN_AFTER_FINISH,
        FINISH_RESULT,
        NOT_RUN_AFTER_FINISH,
    ]


def test_bad_finish_arguments_give_the_error_text_and_the_loop_goes_on() -> None:
    seen: list[str] = []
    agent, fake = make_agent([reply(echo("c1"), BAD_FINISH), reply(finish())], seen=seen)
    result = asyncio.run(agent.run())
    error = read_finish(BAD_FINISH)
    assert isinstance(error, str) and "must not be blank" in error
    assert seen == ["hi"]  # without a good finish, the other calls run
    assert agent.history[2].content == error
    assert fake.requests[1].messages[-2].content == error  # the model saw it
    assert fake.requests[1].messages[-1] == note(2)
    assert result.stop == "finish" and result.steps == 2


def test_at_most_the_cap_of_calls_of_one_reply_run() -> None:
    seen: list[str] = []
    calls = [echo(f"c{i}", f"w{i}") for i in range(6)]
    agent, _ = make_agent(
        [reply(*calls), reply(finish())], seen=seen, limits=AgentLimits(max_calls_per_step=4)
    )
    result = asyncio.run(agent.run())
    assert sorted(seen) == ["w0", "w1", "w2", "w3"]
    over = NOT_RUN_OVER_CAP.format(cap=4)
    assert [m.content for m in agent.history[1:7]] == [
        "echo: w0",
        "echo: w1",
        "echo: w2",
        "echo: w3",
        over,
        over,
    ]
    assert result.tool_calls == 5  # four echoes and the finish


# ---------------------------------------------------------------- forced finish
def test_the_step_limit_forces_a_finish_that_offers_only_finish() -> None:
    model = by_choice(reply(echo("c1")), reply(finish()))
    agent, fake = make_agent([model] * 3, limits=AgentLimits(max_steps=2))
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", FORCED]
    assert (result.stop, result.forced_by, result.steps) == ("forced_finish", "step_limit", 3)
    assert fake.requests[-1].tools == tuple(agent.tools.schemas())


def test_the_tool_call_limit_forces_a_finish() -> None:
    model = by_choice(reply(echo("c1"), echo("c2")), reply(finish()))
    agent, fake = make_agent([model] * 3, limits=AgentLimits(max_tool_calls=3))
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", FORCED]
    assert (result.stop, result.forced_by, result.tool_calls) == (
        "forced_finish",
        "tool_call_limit",
        5,
    )


def test_the_deadline_forces_a_finish_and_bounds_every_call() -> None:
    clock = FakeClock()

    def slow(request: ChatRequest) -> ModelReply:
        clock.now += 40.0  # each call takes 40 s of fake time
        return reply(finish()) if request.tool_choice == FORCED else reply(echo("c1"))

    limits = AgentLimits(finish_reserve_s=30.0, call_timeout_s=40.0)
    agent, fake = make_agent([slow] * 3, clock=clock, deadline=100.0, limits=limits)
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", FORCED]
    # Normal steps end before the reserve; the forced finish may use it.
    assert [r.timeout_s for r in fake.requests] == [40.0, 30.0, 20.0]
    assert (result.stop, result.forced_by) == ("forced_finish", "deadline")


def test_exactly_the_reserve_left_forces_a_finish() -> None:
    # A normal call must end before the reserve, so at exactly the reserve it would get 0 s.
    clock = FakeClock()

    def until_the_reserve(request: ChatRequest) -> ModelReply:
        clock.now = 70.0  # leaves exactly the 30 s reserve
        return reply(echo("c1"))

    limits = AgentLimits(finish_reserve_s=30.0)
    agent, fake = make_agent(
        [until_the_reserve, reply(finish())], clock=clock, deadline=100.0, limits=limits
    )
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", FORCED]
    assert fake.requests[1].timeout_s == 30.0
    assert (result.stop, result.forced_by) == ("forced_finish", "deadline")


def test_no_tool_call_gives_one_nudge_then_a_forced_finish() -> None:
    model = by_choice(reply(text="I am not sure yet."), reply(finish()))
    agent, fake = make_agent([model] * 3)
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", FORCED]
    assert fake.requests[1].messages[-2:] == (Message(role="user", content=NUDGE), note(2))
    assert [m for m in agent.history if m.role == "user"] == [
        Message(role="user", content=NUDGE),
        note(2),
        forced_note(),
    ]
    assert (result.stop, result.forced_by) == ("forced_finish", "no_tool_call")


def test_a_failed_model_call_forces_a_finish() -> None:
    agent, fake = make_agent([ChatError("the API is down"), reply(finish())])
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", FORCED]
    assert (result.stop, result.forced_by, result.steps) == ("forced_finish", "model_error", 2)


@pytest.mark.parametrize(
    "first_try",
    [ChatTimeout("slow"), reply(BAD_FINISH), reply(text="Here is my answer.")],
    ids=["model_error", "bad_arguments", "no_call"],
)
def test_a_failed_forced_finish_is_tried_once_more(first_try: ScriptEntry) -> None:
    agent, fake = make_agent(
        [reply(echo("c1")), first_try, reply(finish())], limits=AgentLimits(max_steps=1)
    )
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", FORCED, FORCED]
    assert result.stop == "forced_finish"


def test_a_forced_reply_runs_no_other_tool() -> None:
    seen: list[str] = []
    agent, _ = make_agent(
        [reply(echo("c1", "a")), reply(echo("c2", "b")), reply(finish())],
        limits=AgentLimits(max_steps=1),
        seen=seen,
    )
    result = asyncio.run(agent.run())
    assert seen == ["a"]
    assert agent.history[2] == forced_note()
    assert agent.history[4].content == NOT_RUN_NOT_ALLOWED.format(names="finish")
    assert result.stop == "forced_finish"


def test_two_failed_forced_finishes_fall_back_to_the_last_text() -> None:
    agent, _ = make_agent(
        [
            reply(echo("c1"), text="So far: it moved to May."),
            ChatError("down"),
            reply(BAD_FINISH),
        ],
        limits=AgentLimits(max_steps=1),
    )
    result = asyncio.run(agent.run())
    assert (result.stop, result.answer) == ("fallback", "So far: it moved to May.")
    assert (result.citations, result.not_found) == ((), ())


def test_no_good_finish_and_no_text_is_an_agent_error() -> None:
    agent, fake = make_agent(
        [reply(echo("c1")), ChatError("down"), ChatError("down")], limits=AgentLimits(max_steps=1)
    )
    with pytest.raises(AgentError, match="no answer after 3 steps"):
        asyncio.run(agent.run())
    assert len(fake.requests) == 3


def test_no_time_left_means_no_model_call_at_all() -> None:
    agent, fake = make_agent([], clock=FakeClock(now=10.0), deadline=10.0)
    with pytest.raises(AgentError):
        asyncio.run(agent.run())
    assert fake.requests == []


def test_a_bug_is_not_turned_into_a_fallback() -> None:
    agent, _ = make_agent([RuntimeError("a bug")])
    with pytest.raises(RuntimeError, match="a bug"):
        asyncio.run(agent.run())


# ---------------------------------------------------------------- setup and limits
def test_a_tool_box_without_finish_is_refused() -> None:
    box = ToolBox([Tool(schema=ECHO, args=EchoArgs, run=lambda args: args.word)])
    with pytest.raises(ValueError, match="no 'finish' tool"):
        Agent(QUESTION, FakeLLM([]), box, deadline=FAR_DEADLINE)


def test_an_agent_runs_once() -> None:
    agent, _ = make_agent([reply(finish())])
    asyncio.run(agent.run())
    with pytest.raises(RuntimeError, match="once"):
        asyncio.run(agent.run())


@pytest.mark.parametrize(
    "bad",
    [{"max_steps": 0}, {"max_calls_per_step": 0}, {"call_timeout_s": 0}, {"finish_reserve_s": -1}],
)
def test_limits_must_be_positive(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AgentLimits(**bad)  # type: ignore[arg-type]


def test_usage_is_summed_over_every_call() -> None:
    usage = Usage(
        input_tokens=100, output_tokens=10, thinking_tokens=5, cached_tokens=2, cost_usd=0.5
    )
    agent, _ = make_agent([reply(echo("c1"), usage=usage), reply(finish(), usage=usage)])
    result = asyncio.run(agent.run())
    assert result.usage == Usage(200, 20, 10, 4, 1.0)


# ---------------------------------------------------------------- isolation
def test_two_agents_share_nothing() -> None:
    def answer_own_question(request: ChatRequest) -> ModelReply:
        question = request.messages[1].content
        if len(request.messages) == 2:
            return reply(echo("c1", "first"))
        return reply(finish(answer=f"Answer to: {question}"))

    fake = FakeLLM([answer_own_question] * 4)
    box = echo_box()
    one = Agent("Question one?", fake, box, deadline=FAR_DEADLINE, clock=FakeClock())
    two = Agent("Question two?", fake, box, deadline=FAR_DEADLINE, clock=FakeClock())

    async def both() -> list[AgentResult]:
        return await asyncio.gather(one.run(), two.run())

    asyncio.run(both())
    assert one.history is not two.history and one.replies is not two.replies
    assert one.finished is not None and "Question one?" in one.finished.answer
    assert two.finished is not None and "Question two?" in two.finished.answer
    for agent, other in ((one, "Question two?"), (two, "Question one?")):
        assert all(other not in m.content for m in agent.build_prompt())
        assert agent.steps == 2 and agent.tool_calls == 2


# ---------------------------------------------------------------- tracing
def test_agent_and_step_spans_carry_their_fields() -> None:
    usage = Usage(
        input_tokens=100, output_tokens=10, thinking_tokens=5, cached_tokens=0, cost_usd=0.25
    )
    agent, _ = make_agent([reply(echo("c1"), usage=usage), reply(finish(), usage=usage)])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        asyncio.run(agent.run())
    [ask] = tracer.traces("ask")
    [span] = tracer.children(ask)
    assert span.name == AGENT_SPAN
    assert span.input == {"question": QUESTION}
    assert span.output == {
        "answer": "It moved to May because the supplier was late.",
        "citations": ["doc-1"],
        "not_found": [],
    }
    tools_version = load_tool_schemas().version
    assert span.metadata["prompts"] == {
        "agent": "7",
        "question": "1",
        "compact": "1",
        "tools": tools_version,
    }
    assert span.metadata["tools"] == ["echo", "finish"]
    assert span.metadata["stop"] == "finish" and span.metadata["forced_by"] is None
    assert (span.metadata["steps"], span.metadata["tool_calls"]) == (2, 2)
    assert (span.metadata["input_tokens"], span.metadata["output_tokens"]) == (200, 20)
    assert span.metadata["cost_usd"] == 0.5
    # The first search's span comes first; the echo box has no search tool, so it did not run.
    first, *steps = tracer.children(span)
    assert first.name == FIRST_SEARCH_SPAN
    assert (first.metadata["outcome"], first.metadata["reason"]) == ("skipped", "no_search_tool")
    assert span.metadata["first_search"] == "skipped"
    assert [s.name for s in steps] == [STEP_SPAN, STEP_SPAN]
    assert [s.metadata["step"] for s in steps] == [1, 2]
    assert [s.metadata["tool_choice"] for s in steps] == ["auto", "auto"]
    assert [s.metadata["budget_note"] for s in steps] == [None, note(2).content]
    assert all(isinstance(s.metadata["timeout_s"], float) for s in steps)
    assert steps[0].output == {"text": "", "tool_calls": [{"id": "c1", "name": "echo"}]}
    assert [s.metadata["finished"] for s in steps] == [False, True]
    assert [c.name for c in tracer.children(steps[0])] == ["tool:echo"]
    assert [c.name for c in tracer.children(steps[1])] == ["tool:finish"]


def test_a_forced_step_and_its_not_run_calls_are_on_the_step_span() -> None:
    agent, _ = make_agent(
        [reply(echo("c1")), reply(echo("c2"), finish())], limits=AgentLimits(max_steps=1)
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        asyncio.run(agent.run())
    [span] = tracer.named(AGENT_SPAN)
    assert (span.metadata["stop"], span.metadata["forced_by"]) == ("forced_finish", "step_limit")
    forced = tracer.named(STEP_SPAN)[1]
    assert forced.metadata["tool_choice"] == ["finish"]
    assert forced.metadata["not_run"] == [{"id": "c2", "name": "echo", "reason": "after_finish"}]


def test_a_failed_run_marks_the_agent_span_as_error() -> None:
    agent, _ = make_agent(
        [reply(echo("c1")), ChatError("x"), ChatError("x")], limits=AgentLimits(max_steps=1)
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"), pytest.raises(AgentError):
        asyncio.run(agent.run())
    [span] = tracer.named(AGENT_SPAN)
    assert span.level == "ERROR" and span.metadata["steps"] == 3
    assert [s.level for s in tracer.named(STEP_SPAN)] == ["DEFAULT", "ERROR", "ERROR"]


def test_a_fallback_marks_the_agent_span_as_warning() -> None:
    agent, _ = make_agent(
        [reply(echo("c1"), text="Partial."), ChatError("x"), ChatError("x")],
        limits=AgentLimits(max_steps=1),
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        asyncio.run(agent.run())
    [span] = tracer.named(AGENT_SPAN)
    assert span.level == "WARNING" and span.metadata["stop"] == "fallback"
