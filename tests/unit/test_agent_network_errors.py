"""A dropped connection at each point the agent recovers from (Codex review #5).

The real GeminiChat on the real SDK; only the network is faked (tests/unit/sdk_wire.py).
The SDK raises httpx's own errors for a dropped connection. The agent catches LLMError at
a step, at a forced finish attempt and at compaction; an httpx error that got past the
client skipped all three and the request got the internal-error answer.
"""

import asyncio
from pathlib import Path

import httpx

from corpusqa.agent.agent import FINISH, FORCED_FINISH, Agent, AgentLimits
from corpusqa.agent.tools import FINISH_TOOL_NAME
from corpusqa.config import AGENT_COMPACT_AT_TOKENS
from tests.unit.sdk_wire import DROP, Wire, chat_on_wire, chat_reply, scripted
from tests.unit.test_agent import FAR_DEADLINE, QUESTION, FakeClock, echo_box

ANSWER = "It moved to May because the supplier was late."
FINISH_ARGS = {"answer": ANSWER, "citations": ["doc-1"], "not_found": []}
LONG = "x" * 4_000  # a long tool result makes the first step worth compacting
OVER = AGENT_COMPACT_AT_TOKENS + 10_000  # a reported prompt size above the compaction limit


def agent_on_wire(answers: list[Wire], tmp_path: Path) -> tuple[Agent, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    model = chat_on_wire(scripted(answers, seen), tmp_path)
    agent = Agent(
        QUESTION,
        model,
        echo_box(),
        deadline=FAR_DEADLINE,
        clock=FakeClock(),
        limits=AgentLimits(),
    )
    return agent, seen


def test_a_dropped_step_and_a_dropped_forced_finish_still_give_the_answer(
    tmp_path: Path,
) -> None:
    finish = chat_reply([("f1", FINISH_TOOL_NAME, FINISH_ARGS)])
    # step 1 drops -> forced finish; its first attempt drops; the second answers
    agent, seen = agent_on_wire([httpx.ConnectError, httpx.ReadError, finish], tmp_path)
    result = asyncio.run(agent.run())
    assert (result.stop, agent.forced_by, result.answer) == (FORCED_FINISH, "model_error", ANSWER)
    assert len(seen) == 3


def test_a_dropped_compaction_call_keeps_the_history_and_the_loop_goes_on(
    tmp_path: Path,
) -> None:
    answers: list[Wire] = [
        chat_reply([("c1", "echo", {"word": LONG})], prompt_tokens=1_000),
        chat_reply([("c2", "echo", {"word": "b"})], prompt_tokens=OVER),
        httpx.ConnectError,  # the compaction call
        chat_reply([("f1", FINISH_TOOL_NAME, FINISH_ARGS)]),
    ]
    agent, seen = agent_on_wire(answers, tmp_path)
    result = asyncio.run(agent.run())
    assert (result.stop, result.answer) == (FINISH, ANSWER)
    [event] = result.compactions
    assert event.path == "skipped"
    assert "ChatError" in event.reason and DROP in event.reason
    assert len(seen) == 4
