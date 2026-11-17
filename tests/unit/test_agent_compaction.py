"""The token estimate and compaction: reported counts, the split, the working memory, skips.

Offline with FakeLLM. The prompt's size is steered with the input tokens a fake reply
reports (the estimate trusts them), and long tool results make old steps worth shrinking.
"""

import asyncio
import json
import logging
import math
from collections.abc import Callable

import pytest

from corpusqa.agent.agent import (
    AGENT_SPAN,
    BUDGET_OPEN,
    COMPACT_SPAN,
    FIRST_SEARCH_SPAN,
    MEMORY_CLOSE,
    MEMORY_OPEN,
    SKIPPED_NOTE,
    STEP_SPAN,
    AgentLimits,
    CompactionEvent,
    rough_tokens,
)
from corpusqa.config import (
    AGENT_COMPACT_AT_TOKENS,
    COMPACT_CALL_MAX_OUTPUT_TOKENS,
    COMPACT_THINKING_BUDGET,
    TOKEN_CHARS,
)
from corpusqa.llm.errors import ChatError, ChatTimeout
from corpusqa.llm.fakes import ChatRequest, fake_reply
from corpusqa.llm.interfaces import Message, ModelReply, ToolCall, Usage
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_text_prompt
from tests.unit.test_agent import (
    FORCED,
    QUESTION,
    FakeClock,
    assert_history_rules,
    echo,
    finish,
    make_agent,
    note,
    reply,
)

LONG = "x" * 4_000  # a long word: its call and its echo are each ~1,000 rough tokens
OVER = AGENT_COMPACT_AT_TOKENS + 10_000  # a reported count above the compaction limit
SMALL = 1_000  # a reported count well below it
MEMORY = "Question: when and why the rollout moved. Facts found: it moved to May (doc-1)."


def tokens(input_tokens: int, cost_usd: float = 0.0) -> Usage:
    return Usage(input_tokens, 10, 0, 0, cost_usd)


def tool_message(call: ToolCall, content: str) -> Message:
    return Message(role="tool", tool_call_id=call.id, name=call.name, content=content)


def memory_message(text: str) -> Message:
    return Message(role="user", content=f"{MEMORY_OPEN}\n{text}\n{MEMORY_CLOSE}")


# ---------------------------------------------------------------- rough size
def test_rough_size_is_the_json_characters_over_token_chars() -> None:
    messages = [Message(role="user", content="héllo €"), Message(role="assistant", content="ok")]
    plain = [
        {"role": m.role, "content": m.content, "tool_calls": [], "tool_call_id": None, "name": None}
        for m in messages
    ]
    expected = math.ceil(len(json.dumps(plain, ensure_ascii=False)) / TOKEN_CHARS)
    assert rough_tokens(messages) == expected


def test_rough_size_counts_tool_calls_and_leaves_raw_out() -> None:
    plain = Message(role="assistant", tool_calls=(echo("c1", LONG),))
    with_raw = Message(role="assistant", tool_calls=(echo("c1", LONG),), raw=object())
    assert rough_tokens([plain]) == rough_tokens([with_raw]) > len(LONG) // TOKEN_CHARS


# ---------------------------------------------------------------- token estimate
def test_before_the_first_call_the_estimate_is_the_rough_size() -> None:
    agent, _ = make_agent([])
    assert agent.estimate_tokens() == rough_tokens(agent.build_prompt())


def test_after_a_call_the_estimate_is_the_reported_count_plus_what_was_added() -> None:
    agent, fake = make_agent([reply(echo("c1", LONG), usage=tokens(5_000))])
    asyncio.run(agent.step("auto"))
    sent = list(fake.requests[0].messages)
    added = rough_tokens(agent.build_prompt()) - rough_tokens(sent)
    assert added > 0
    assert agent.estimate_tokens() == 5_000 + added
    before = agent.estimate_tokens()
    agent.history.append(Message(role="user", content="y" * 400))
    assert agent.estimate_tokens() > before  # grows with each message added
    agent.history.append(Message(role="user", content="z" * 400))
    assert agent.estimate_tokens() == 5_000 + rough_tokens(agent.build_prompt()) - rough_tokens(
        sent
    )


def test_a_reported_count_of_zero_is_not_trusted() -> None:
    agent, _ = make_agent([reply(echo("c1"), usage=tokens(0))])
    asyncio.run(agent.step("auto"))
    assert agent.estimate_tokens() == rough_tokens(agent.build_prompt())


def test_the_step_span_carries_the_estimate_before_its_call() -> None:
    agent, _ = make_agent([reply(echo("c1"), usage=tokens(SMALL)), reply(finish())])
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        asyncio.run(agent.run())
    first, second = tracer.named(STEP_SPAN)
    assert first.metadata["tokens_estimated"] == rough_tokens(
        [agent.build_prompt()[0], agent.build_prompt()[1]]
    )
    assert isinstance(second.metadata["tokens_estimated"], int)
    assert second.metadata["tokens_estimated"] > SMALL
    assert first.metadata["input_tokens"] == SMALL


# ---------------------------------------------------------------- when compaction runs
def test_no_compaction_at_or_below_the_limit() -> None:
    script = [reply(echo(f"c{i}", LONG), usage=tokens(SMALL)) for i in range(4)]
    agent, fake = make_agent([*script, reply(finish())])
    result = asyncio.run(agent.run())
    assert result.compactions == () and agent.compactions == []
    assert {r.prompt_name for r in fake.requests} == {"agent"}


def test_no_compaction_when_history_holds_only_the_kept_steps() -> None:
    agent, fake = make_agent([reply(echo("c1", LONG), usage=tokens(OVER)), reply(finish())])
    result = asyncio.run(agent.run())
    assert result.compactions == ()
    assert [r.prompt_name for r in fake.requests] == ["agent", "agent"]


def test_compact_with_nothing_to_compact_is_an_error() -> None:
    agent, _ = make_agent([])
    with pytest.raises(ValueError, match="nothing to compact"):
        asyncio.run(agent.compact())


# ---------------------------------------------------------------- one compaction
def test_compaction_keeps_system_question_and_the_last_step_and_shrinks_the_prompt() -> None:
    first = reply(echo("c1", LONG), usage=tokens(SMALL, 0.01))
    second = reply(echo("c2", "b"), usage=tokens(OVER, 0.01))
    memory = fake_reply(MEMORY, usage=tokens(3_000, 0.02))
    agent, fake = make_agent([first, second, memory, reply(finish(), usage=tokens(SMALL, 0.01))])
    result = asyncio.run(agent.run())

    assert [r.prompt_name for r in fake.requests] == ["agent", "agent", "compact", "agent"]
    head = list(fake.requests[0].messages)  # system and question, as first sent
    step_one = [first.message, tool_message(echo("c1", LONG), f"echo: {LONG}")]
    step_two = [second.message, tool_message(echo("c2", "b"), "echo: b")]
    without = [*head, *step_one, note(2), *step_two, note(3)]
    after = list(fake.requests[3].messages)
    # System, question and the last complete step word for word; the old step replaced
    # (with the budget note before the kept step); then this call's budget note.
    assert after == [*head, memory_message(MEMORY), *step_two, note(3)]
    assert rough_tokens(after) < rough_tokens(without)  # the prompt really got smaller
    assert agent.build_prompt()[2] == memory_message(MEMORY)
    assert_history_rules(agent.history, agent.question_prompt)

    [event] = result.compactions
    assert event == agent.compactions[0]
    assert (event.step, event.steps_replaced, event.steps_kept) == (2, 1, 1)
    assert (event.path, event.no_gain) == ("model", False)
    assert event.tokens_after < event.tokens_before
    assert event.tokens_before > AGENT_COMPACT_AT_TOKENS
    assert result.steps == 3  # the compaction call is not a step
    assert result.usage.cost_usd == pytest.approx(0.05)  # compaction call included


def test_the_compaction_call_sends_the_old_part_as_text_without_tools() -> None:
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), text="Looking it up.", usage=tokens(SMALL)),
            reply(echo("c2", "b"), usage=tokens(OVER)),
            fake_reply(MEMORY),
            reply(finish()),
        ]
    )
    asyncio.run(agent.run())
    request: ChatRequest = fake.requests[2]
    assert (request.tools, request.tool_choice) == ((), "none")
    assert request.max_output_tokens == COMPACT_CALL_MAX_OUTPUT_TOKENS
    assert request.thinking_budget == COMPACT_THINKING_BUDGET
    assert (request.prompt_name, request.prompt_version) == ("compact", "1")
    system, user = request.messages
    assert system == Message(role="system", content=load_text_prompt("compact").text)
    assert user.role == "user"
    assert user.content.startswith(f"<question>\n{QUESTION}\n</question>")
    assert "[step 1: the agent]\nLooking it up." in user.content
    assert f'call echo({{"word": "{LONG}"}}) (id c1)' in user.content
    assert f"[step 1: result of echo, call c1]\necho: {LONG}" in user.content  # whole
    assert "echo: b" not in user.content  # the kept step is not summarised
    assert BUDGET_OPEN not in user.content  # the budget note before step 2 is left out


def test_a_step_is_never_split_and_keep_steps_is_honoured() -> None:
    kept_reply = reply(echo("k1", "p"), echo("k2", "q"), usage=tokens(OVER))
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", LONG), usage=tokens(SMALL)),
            kept_reply,
            fake_reply(MEMORY),
            reply(finish()),
        ],
        limits=AgentLimits(compact_keep_steps=2),
    )
    result = asyncio.run(agent.run())
    [event] = result.compactions
    assert (event.steps_replaced, event.steps_kept) == (1, 2)
    after = fake.requests[4].messages
    assert after[2] == memory_message(MEMORY)
    assert after[3].tool_calls == (echo("c2", LONG),)
    # The kept steps, each with all its tool messages; then this call's budget note.
    assert list(after[5:]) == [
        note(3),
        kept_reply.message,
        tool_message(echo("k1", "p"), "echo: p"),
        tool_message(echo("k2", "q"), "echo: q"),
        note(4),
    ]


# ---------------------------------------------------------------- two compactions
def test_a_second_compaction_folds_in_the_first_working_memory() -> None:
    second_memory = "Facts found: it moved to May (doc-1); the supplier was late (doc-2)."
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", LONG), usage=tokens(OVER)),
            fake_reply(MEMORY),
            reply(echo("c3", "c"), usage=tokens(OVER)),
            fake_reply(second_memory),
            reply(finish()),
        ]
    )
    result = asyncio.run(agent.run())
    names = [r.prompt_name for r in fake.requests]
    assert names == ["agent", "agent", "compact", "agent", "compact", "agent"]
    second_input = fake.requests[4].messages[1].content
    assert f"[earlier working memory]\n{MEMORY}" in second_input
    assert "[step 2: the agent]" in second_input  # step numbers go on across compactions
    assert f"echo: {LONG}" in second_input  # step 2, whole
    last = fake.requests[5].messages
    assert last[2] == memory_message(second_memory)
    assert all(MEMORY not in m.content for m in last)  # the first memory was replaced
    assert [e.steps_replaced for e in result.compactions] == [1, 1]
    assert_history_rules(agent.history, agent.question_prompt)


# ---------------------------------------------------------------- a failed compaction is skipped
# User, 2026-10-16: "Skip compaction instead". When the compaction call fails or writes
# nothing, the history stays whole (no code-written short memory) and the next step tries
# again; the event and the span record why.
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (ChatError("the API is down"), "the compaction call failed (ChatError: the API is down)"),
        (fake_reply("   "), "the compaction call wrote nothing"),
    ],
    ids=["error", "blank"],
)
def test_a_failed_compaction_keeps_the_history_whole(failure: object, reason: str) -> None:
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), text="Looking it up.", usage=tokens(SMALL)),
            reply(echo("c2", "b"), usage=tokens(OVER)),
            failure,  # type: ignore[list-item]
            reply(finish()),
        ]
    )
    result = asyncio.run(agent.run())
    assert result.stop == "finish"  # the loop goes on
    assert [r.prompt_name for r in fake.requests] == ["agent", "agent", "compact", "agent"]
    # The step after the failed call sees every earlier step word for word, no memory:
    # the last prompt, step 2 and this call's budget note.
    assert list(fake.requests[3].messages) == [
        *fake.requests[1].messages,
        *agent.history[3:5],
        note(3),
    ]
    assert f"echo: {LONG}" in [m.content for m in fake.requests[3].messages]
    assert all(MEMORY_OPEN not in m.content for m in agent.history)
    assert_history_rules(agent.history, agent.question_prompt)
    [event] = result.compactions
    assert (event.path, event.no_gain, event.reason) == ("skipped", False, reason)
    assert (event.step, event.steps_replaced, event.steps_kept) == (2, 0, 2)
    assert event.tokens_after == event.tokens_before > AGENT_COMPACT_AT_TOKENS


def test_a_skipped_compaction_is_tried_again_at_the_next_step() -> None:
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", "b"), usage=tokens(OVER)),
            ChatError("down"),  # skipped: the history stays whole
            reply(echo("c3", "c"), usage=tokens(OVER)),
            fake_reply(MEMORY),  # tried again before the next step, and it works
            reply(finish()),
        ]
    )
    result = asyncio.run(agent.run())
    names = [r.prompt_name for r in fake.requests]
    assert names == ["agent", "agent", "compact", "agent", "compact", "agent"]
    assert [e.path for e in result.compactions] == ["skipped", "model"]
    # The second call summarises every step before the kept one: steps 1 and 2.
    second_input = fake.requests[4].messages[1].content
    assert "[step 1: the agent]" in second_input and "[step 2: the agent]" in second_input
    assert f"echo: {LONG}" in second_input
    assert [e.steps_replaced for e in result.compactions] == [0, 2]
    last = fake.requests[5].messages
    assert last[2] == memory_message(MEMORY)
    assert_history_rules(agent.history, agent.question_prompt)


def test_a_skipped_compaction_keeps_an_earlier_working_memory() -> None:
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", LONG), usage=tokens(OVER)),
            fake_reply(MEMORY),
            reply(echo("c3", "c"), usage=tokens(OVER)),
            fake_reply(""),  # the second compaction writes nothing: skipped
            reply(finish()),
        ]
    )
    result = asyncio.run(agent.run())
    assert [e.path for e in result.compactions] == ["model", "skipped"]
    last = fake.requests[5].messages
    # The first memory, then the steps after it, all word for word.
    assert last[2] == memory_message(MEMORY)
    assert [m.tool_calls for m in last if m.role == "assistant"] == [
        (echo("c2", LONG),),
        (echo("c3", "c"),),
    ]
    assert f"echo: {LONG}" in [m.content for m in last]


def test_no_gain_is_flagged_and_the_new_history_is_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    long_memory = "m" * 20_000  # longer than the steps it replaces
    agent, fake = make_agent(
        [
            reply(echo("c1"), usage=tokens(SMALL)),
            reply(echo("c2"), usage=tokens(OVER)),
            fake_reply(long_memory),
            reply(finish()),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="corpusqa.agent.agent"):
        result = asyncio.run(agent.run())
    [event] = result.compactions
    assert event.no_gain is True
    assert "no_gain" in caplog.text
    assert fake.requests[3].messages[2] == memory_message(long_memory)


# ---------------------------------------------------------------- time
def arriving(
    clock: FakeClock,
) -> Callable[[float, ModelReply], Callable[[ChatRequest], ModelReply]]:
    """`at(t, entry)`: a script entry that sets the fake clock to `t`, then replies."""

    def at(seconds: float, entry: ModelReply) -> Callable[[ChatRequest], ModelReply]:
        def play(request: ChatRequest) -> ModelReply:
            clock.now = seconds
            return entry

        return play

    return at


def test_no_time_before_the_reserve_means_no_compaction_call() -> None:
    clock = FakeClock()
    at = arriving(clock)
    agent, fake = make_agent(
        [
            at(10.0, reply(echo("c1", LONG), usage=tokens(SMALL))),
            at(75.0, reply(echo("c2", "b"), usage=tokens(OVER))),  # 25 s left < 30 s reserve
            reply(finish()),
        ],
        clock=clock,
        deadline=100.0,
        limits=AgentLimits(finish_reserve_s=30.0),
    )
    result = asyncio.run(agent.run())
    assert [r.prompt_name for r in fake.requests] == ["agent", "agent", "agent"]
    assert (result.stop, result.forced_by, result.compactions) == ("forced_finish", "deadline", ())


def test_the_limits_are_checked_again_after_a_compaction() -> None:
    clock = FakeClock()
    at = arriving(clock)
    agent, fake = make_agent(
        [
            at(5.0, reply(echo("c1", LONG), usage=tokens(SMALL))),
            at(10.0, reply(echo("c2", "b"), usage=tokens(OVER))),
            at(75.0, fake_reply(MEMORY)),  # the compaction call leaves 25 s
            reply(finish()),
        ],
        clock=clock,
        deadline=100.0,
        limits=AgentLimits(finish_reserve_s=30.0),
    )
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", "none", FORCED]
    assert fake.requests[2].timeout_s == 40.0  # min(90 - 30, 40)
    assert (result.stop, result.forced_by) == ("forced_finish", "deadline")


def test_a_failed_compaction_near_the_deadline_never_blocks_the_forced_finish() -> None:
    clock = FakeClock()
    at = arriving(clock)

    def fails_late(request: ChatRequest) -> ModelReply:
        clock.now = 69.0 + request.timeout_s  # the call times out at the end of its time
        raise ChatTimeout("no answer")

    agent, fake = make_agent(
        [
            at(5.0, reply(echo("c1", LONG), usage=tokens(SMALL))),
            at(69.0, reply(echo("c2", "b"), usage=tokens(OVER))),  # 31 s left
            fails_late,  # the compaction call may use 1 s: it ends as the reserve starts
            reply(finish()),
        ],
        clock=clock,
        deadline=100.0,
        limits=AgentLimits(finish_reserve_s=30.0),
    )
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", "none", FORCED]
    assert fake.requests[2].timeout_s == pytest.approx(1.0)  # never into the reserve
    assert fake.requests[3].timeout_s == pytest.approx(30.0)  # the reserve is whole
    assert (result.stop, result.forced_by) == ("forced_finish", "deadline")
    assert [e.path for e in result.compactions] == ["skipped"]
    assert f"echo: {LONG}" in [m.content for m in fake.requests[3].messages]  # whole


def test_a_skipped_compaction_before_a_forced_finish_still_lets_it_answer() -> None:
    agent, fake = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", "b"), usage=tokens(OVER)),
            ChatError("down"),  # before the forced finish (step limit): skipped
            reply(finish()),
        ],
        limits=AgentLimits(max_steps=2),
    )
    result = asyncio.run(agent.run())
    assert [r.tool_choice for r in fake.requests] == ["auto", "auto", "none", FORCED]
    assert (result.stop, result.forced_by) == ("forced_finish", "step_limit")
    assert [e.path for e in result.compactions] == ["skipped"]


# ---------------------------------------------------------------- tracing
def test_the_compact_span_carries_its_fields() -> None:
    agent, _ = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", "b"), usage=tokens(OVER)),
            fake_reply(MEMORY),
            reply(finish()),
        ]
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = asyncio.run(agent.run())
    [agent_span] = tracer.named(AGENT_SPAN)
    [span] = tracer.named(COMPACT_SPAN)
    assert span.parent_id == agent_span.id
    [event] = result.compactions
    assert isinstance(event, CompactionEvent)
    for field in ("step", "tokens_before", "tokens_after", "steps_replaced", "steps_kept"):
        assert span.metadata[field] == getattr(event, field)
    assert (span.metadata["path"], span.metadata["no_gain"]) == ("model", False)
    assert span.output == MEMORY
    assert span.input == {"steps_replaced": 1, "steps_kept": 1, "chars": span.input["chars"]}  # type: ignore[index]
    assert (agent_span.metadata["compactions"], agent_span.metadata["compactions_skipped"]) == (
        1,
        0,
    )
    names = [o.name for o in tracer.children(agent_span)]
    # The first search's span comes first (skipped here: the echo box has no search tool).
    assert names == [FIRST_SEARCH_SPAN, STEP_SPAN, STEP_SPAN, COMPACT_SPAN, STEP_SPAN]


def test_a_skipped_compaction_is_a_warning_on_its_span(caplog: pytest.LogCaptureFixture) -> None:
    agent, _ = make_agent(
        [
            reply(echo("c1", LONG), usage=tokens(SMALL)),
            reply(echo("c2", "b"), usage=tokens(OVER)),
            ChatError("down"),
            reply(finish()),
        ]
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"), caplog.at_level(logging.WARNING, logger="corpusqa.agent.agent"):
        asyncio.run(agent.run())
    [agent_span] = tracer.named(AGENT_SPAN)
    [span] = tracer.named(COMPACT_SPAN)
    assert span.level == "WARNING"
    assert span.metadata["path"] == "skipped"
    assert span.metadata["reason"] == "the compaction call failed (ChatError: down)"
    assert (span.metadata["steps_replaced"], span.metadata["steps_kept"]) == (0, 2)
    assert span.output is None  # no working memory was written
    message = f"compaction skipped: the compaction call failed (ChatError: down); {SKIPPED_NOTE}"
    assert span.status_message == message
    assert message in caplog.text
    assert (agent_span.metadata["compactions"], agent_span.metadata["compactions_skipped"]) == (
        0,
        1,
    )
