"""Chat message types, chat errors and the scripted FakeLLM (offline)."""

import asyncio
from collections.abc import Sequence

import pytest

from corpusqa.llm.errors import BadChatAnswer, ChatError, ChatTimeout, ChatUnavailable, LLMError
from corpusqa.llm.fakes import ChatRequest, FakeLLM, ScriptExhausted, fake_reply
from corpusqa.llm.interfaces import (
    ChatModel,
    Message,
    ModelReply,
    ToolCall,
    ToolChoice,
    ToolSchema,
    Usage,
)

SCHEMA = ToolSchema(
    name="lookup",
    description="Look something up.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
PROMPT: list[Message] = [
    Message(role="system", content="You answer questions."),
    Message(role="user", content="What is due on Friday?"),
]


async def ask(
    model: ChatModel,
    messages: Sequence[Message] = PROMPT,
    tool_choice: ToolChoice = "auto",
) -> ModelReply:
    """One call with fixed settings, through the protocol the agent will use."""
    return await model.complete(
        messages,
        [SCHEMA],
        tool_choice=tool_choice,
        max_output_tokens=512,
        thinking_budget=128,
        timeout_s=12.5,
        prompt_name="agent",
        prompt_version="3",
    )


# ---------------------------------------------------------------- message types
def test_message_defaults_are_empty() -> None:
    message = Message(role="user")
    assert message.content == ""
    assert message.tool_calls == ()
    assert message.tool_call_id is None
    assert message.name is None
    assert message.raw is None


def test_messages_are_frozen_values() -> None:
    call = ToolCall(id="call_1", name="lookup", arguments={"q": "x"})
    message = Message(role="assistant", tool_calls=(call,))
    assert message == Message(
        role="assistant", tool_calls=(ToolCall("call_1", "lookup", {"q": "x"}),)
    )
    with pytest.raises(AttributeError):
        message.content = "changed"  # type: ignore[misc]


def test_fake_reply_builds_an_assistant_message_with_zero_usage() -> None:
    call = ToolCall(id="call_1", name="lookup", arguments={"q": "x"})
    reply = fake_reply("thinking aloud", [call])
    assert reply.message == Message(role="assistant", content="thinking aloud", tool_calls=(call,))
    assert reply.usage == Usage(0, 0, 0, 0, 0.0)
    assert reply.stop_reason == "STOP"
    usage = Usage(
        input_tokens=10, output_tokens=2, thinking_tokens=5, cached_tokens=0, cost_usd=0.1
    )
    assert fake_reply(usage=usage, stop_reason="MAX_TOKENS").usage == usage


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("error", [ChatTimeout, BadChatAnswer, ChatUnavailable])
def test_chat_errors_are_llm_errors(error: type[ChatError]) -> None:
    """Callers that already catch LLMError (or ChatError) catch every chat failure."""
    assert issubclass(error, ChatError)
    assert issubclass(error, LLMError)


# ---------------------------------------------------------------- FakeLLM
def test_fake_llm_plays_its_script_in_order() -> None:
    first = fake_reply("one")
    second = fake_reply("", [ToolCall(id="call_1", name="lookup", arguments={"q": "x"})])
    fake = FakeLLM([first, second])
    assert fake.remaining == 2
    assert asyncio.run(ask(fake)) is first
    assert asyncio.run(ask(fake)) is second
    assert fake.remaining == 0


def test_fake_llm_records_every_argument_of_every_call() -> None:
    fake = FakeLLM([fake_reply("ok")])
    asyncio.run(ask(fake, tool_choice=("lookup",)))
    assert fake.requests == [
        ChatRequest(
            messages=tuple(PROMPT),
            tools=(SCHEMA,),
            tool_choice=("lookup",),
            max_output_tokens=512,
            thinking_budget=128,
            timeout_s=12.5,
            prompt_name="agent",
            prompt_version="3",
        )
    ]


def test_fake_llm_copies_the_messages_at_call_time() -> None:
    """The agent's history list grows after a call; the record must show what was sent."""
    history = list(PROMPT)
    fake = FakeLLM([fake_reply("ok")])
    asyncio.run(ask(fake, history))
    history.append(Message(role="user", content="added later"))
    assert fake.requests[0].messages == tuple(PROMPT)


def test_a_callable_entry_gets_the_request() -> None:
    def echo_last(request: ChatRequest) -> ModelReply:
        return fake_reply(f"you said: {request.messages[-1].content}")

    fake = FakeLLM([echo_last])
    assert asyncio.run(ask(fake)).message.content == "you said: What is due on Friday?"


def test_an_async_callable_entry_is_awaited() -> None:
    async def slow(request: ChatRequest) -> ModelReply:
        await asyncio.sleep(0)
        return fake_reply(f"choice {request.tool_choice}")

    fake = FakeLLM([slow])
    assert asyncio.run(ask(fake, tool_choice="none")).message.content == "choice none"


def test_an_exception_entry_is_raised_after_the_call_is_recorded() -> None:
    fake = FakeLLM([ChatTimeout("scripted timeout"), fake_reply("recovered")])
    with pytest.raises(ChatTimeout, match="scripted timeout"):
        asyncio.run(ask(fake))
    assert len(fake.requests) == 1
    assert asyncio.run(ask(fake)).message.content == "recovered"


def test_a_callable_entry_may_raise() -> None:
    def refuse(request: ChatRequest) -> ModelReply:
        raise BadChatAnswer(f"blocked at {request.prompt_name}")

    fake = FakeLLM([refuse])
    with pytest.raises(BadChatAnswer, match="blocked at agent"):
        asyncio.run(ask(fake))


def test_a_call_after_the_script_ends_fails_clearly() -> None:
    fake = FakeLLM([fake_reply("only one")])
    asyncio.run(ask(fake))
    with pytest.raises(ScriptExhausted, match=r"call 2 came after all 1 scripted entries"):
        asyncio.run(ask(fake, tool_choice=("lookup",)))
    assert len(fake.requests) == 2  # the extra call is still recorded


def test_an_empty_script_fails_on_the_first_call() -> None:
    with pytest.raises(ScriptExhausted, match="tool_choice: 'auto'"):
        asyncio.run(ask(FakeLLM([])))
