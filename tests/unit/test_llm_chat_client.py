"""GeminiChat: message conversion, reply parsing, retries, timeout, spend cap, tracing.

Offline: a scripted stand-in for the SDK's async client, or the real SDK over an httpx
mock transport. One live test (marked) makes real calls.
"""

import asyncio
import base64
import dataclasses
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from corpusqa.config import (
    AGENT_MAX_OUTPUT_TOKENS,
    AGENT_MODEL,
    AGENT_THINKING_BUDGET,
    AGENT_THINKING_LEVEL,
    CHAT_HTTP_TIMEOUT_MS,
    CHAT_PRICE_CACHED_INPUT_PER_M_TOKENS_USD,
    CHAT_PRICE_INPUT_PER_M_TOKENS_USD,
    CHAT_PRICE_OUTPUT_PER_M_TOKENS_USD,
    CHAT_STAND_IN_THOUGHT_SIGNATURE,
    COMPACT_CALL_MAX_OUTPUT_TOKENS,
    COMPACT_THINKING_BUDGET,
    Settings,
)
from corpusqa.llm.client import (
    GeminiChat,
    chat_cost,
    from_gemini,
    gemini_config,
    to_gemini,
    usage_of,
)
from corpusqa.llm.errors import (
    BadChatAnswer,
    ChatError,
    ChatTimeout,
    ChatUnavailable,
    SpendCapReached,
)
from corpusqa.llm.interfaces import Message, ModelReply, ToolCall, ToolChoice, ToolSchema, Usage
from corpusqa.llm.limits import AsyncRateLimiter, SpendLedger
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_tool_schemas
from tests.unit.sdk_wire import DROP, chat_on_wire, scripted

FAKE_KEY = "fake-key-for-tests"  # never a real key
FINISH = load_tool_schemas().get("finish")
LOOKUP = ToolSchema(
    name="lookup",
    description="Look up the meaning of one word in a small dictionary. Returns the meaning.",
    parameters={
        "type": "object",
        "properties": {"word": {"type": "string", "description": "One word, lower case."}},
        "required": ["word"],
        "additionalProperties": False,
    },
)
TOOLS = (LOOKUP, FINISH)
SYSTEM = Message(role="system", content="You answer questions with the tools.")
QUESTION = Message(role="user", content="What do 'quokka' and 'axolotl' mean?")
STOP = genai_types.FinishReason.STOP


# ---------------------------------------------------------------- builders
def text_part(text: str, *, thought: bool = False, sig: bytes | None = None) -> genai_types.Part:
    return genai_types.Part(text=text, thought=thought or None, thought_signature=sig)


def call_part(
    name: str, args: dict[str, object], call_id: str | None = None, sig: bytes | None = None
) -> genai_types.Part:
    call = genai_types.FunctionCall(id=call_id, name=name, args=args)
    return genai_types.Part(function_call=call, thought_signature=sig)


def response(
    *parts: genai_types.Part,
    finish: genai_types.FinishReason | None = STOP,
    prompt: int | None = 100,
    output: int | None = 20,
    thoughts: int | None = None,
    cached: int | None = None,
) -> genai_types.GenerateContentResponse:
    usage = genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=output,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached,
    )
    content = genai_types.Content(role="model", parts=list(parts))
    candidate = genai_types.Candidate(content=content, finish_reason=finish)
    return genai_types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)


def two_calls() -> genai_types.GenerateContentResponse:
    return response(
        call_part("lookup", {"word": "quokka"}, "c1", sig=b"real-signature"),
        call_part("lookup", {"word": "axolotl"}, "c2"),
    )


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": "scripted"}})


type Answer = (
    genai_types.GenerateContentResponse
    | BaseException
    | Callable[[], Awaitable[genai_types.GenerateContentResponse]]
)


class ScriptedChatClient:
    """Stands in for genai.Client: `aio.models.generate_content` plays the next answer."""

    def __init__(self, answers: Sequence[Answer]) -> None:
        self.aio = self
        self.models = self
        self.answers = list(answers)
        self.requests: list[dict[str, object]] = []

    async def generate_content(
        self, *, model: str, contents: list[genai_types.Content], config: object
    ) -> genai_types.GenerateContentResponse:
        self.requests.append({"model": model, "contents": contents, "config": config})
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        return answer


def chat(
    tmp_path: Path, answers: Sequence[Answer], cap: float = 10.0, **kwargs: object
) -> tuple[GeminiChat, ScriptedChatClient, list[float]]:
    client = ScriptedChatClient(answers)
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def no_wait(seconds: float) -> None:
        return None

    built = GeminiChat(
        client,  # type: ignore[arg-type]
        SpendLedger(tmp_path / "ledger.json", cap),
        limiter=AsyncRateLimiter(1e9, sleep=no_wait),
        sleep=record_sleep,
        **kwargs,  # type: ignore[arg-type]
    )
    return built, client, slept


def ask(
    model: GeminiChat,
    messages: Sequence[Message] = (SYSTEM, QUESTION),
    *,
    tools: Sequence[ToolSchema] = TOOLS,
    tool_choice: ToolChoice = "auto",
    max_output_tokens: int = 1_000,
    thinking_budget: int = 256,
    timeout_s: float = 5.0,
) -> ModelReply:
    return asyncio.run(
        model.complete(
            messages,
            tools,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            timeout_s=timeout_s,
            prompt_name="agent",
            prompt_version="7",
        )
    )


# ---------------------------------------------------------------- to_gemini
def test_system_becomes_the_instruction_and_the_question_a_user_content() -> None:
    system, contents = to_gemini([SYSTEM, QUESTION])
    assert system == SYSTEM.content
    assert contents == [
        genai_types.Content(role="user", parts=[genai_types.Part(text=QUESTION.content)])
    ]


def test_a_reply_gemini_wrote_is_sent_back_unchanged() -> None:
    raw = two_calls().candidates[0].content  # type: ignore[index]
    calls = (ToolCall("c1", "lookup", {"word": "quokka"}), ToolCall("c2", "lookup", {"word": "x"}))
    assistant = Message(role="assistant", tool_calls=calls, raw=raw)
    _, contents = to_gemini([SYSTEM, QUESTION, assistant])
    assert contents[1] is raw


def test_a_turn_code_wrote_gets_the_stand_in_signature_on_its_first_call() -> None:
    """Without a signature the API answers 400 (step 1 probe)."""
    calls = (ToolCall("c1", "lookup", {"word": "quokka"}), ToolCall("c2", "lookup", {"word": "x"}))
    assistant = Message(role="assistant", content="Looking both up.", tool_calls=calls)
    _, contents = to_gemini([QUESTION, assistant])
    model = contents[1]
    assert model.role == "model"
    text, first, second = model.parts or []
    assert text.text == "Looking both up." and text.thought_signature is None
    assert first.function_call == genai_types.FunctionCall(
        id="c1", name="lookup", args={"word": "quokka"}
    )
    assert first.thought_signature == CHAT_STAND_IN_THOUGHT_SIGNATURE
    assert second.function_call is not None and second.thought_signature is None


def test_tool_results_after_one_reply_become_one_user_content_in_call_order() -> None:
    raw = two_calls().candidates[0].content  # type: ignore[index]
    reply = from_gemini(two_calls())
    results = [
        Message(role="tool", content="a small wallaby", tool_call_id="c1", name="lookup"),
        Message(role="tool", content="a salamander", tool_call_id="c2", name="lookup"),
    ]
    _, contents = to_gemini([QUESTION, dataclasses.replace(reply.message, raw=raw), *results])
    assert len(contents) == 3
    answer = contents[2]
    assert answer.role == "user"
    assert [p.function_response for p in answer.parts or []] == [
        genai_types.FunctionResponse(
            id="c1", name="lookup", response={"result": "a small wallaby"}
        ),
        genai_types.FunctionResponse(id="c2", name="lookup", response={"result": "a salamander"}),
    ]


def test_an_id_we_filled_in_is_not_sent_back() -> None:
    no_ids = response(call_part("lookup", {"word": "quokka"}, sig=b"s"))
    reply = from_gemini(no_ids)
    [call] = reply.message.tool_calls
    assert call.id == "call_1"
    result = Message(role="tool", content="a wallaby", tool_call_id=call.id, name="lookup")
    _, contents = to_gemini([QUESTION, reply.message, result])
    [part] = contents[2].parts or []
    assert part.function_response is not None and part.function_response.id is None


def test_user_messages_in_a_row_share_one_content() -> None:
    memory = Message(role="user", content="<working_memory>\nkiwi: a bird\n</working_memory>")
    _, contents = to_gemini([SYSTEM, QUESTION, memory])
    assert [[p.text for p in c.parts or []] for c in contents] == [
        [QUESTION.content, memory.content]
    ]


def test_a_user_message_after_tool_results_is_its_own_content() -> None:
    reply = from_gemini(two_calls())
    result = Message(role="tool", content="a wallaby", tool_call_id="c1", name="lookup")
    nudge = Message(role="user", content="Call a tool, or call finish with your answer.")
    _, contents = to_gemini([QUESTION, reply.message, result, nudge])
    assert [c.role for c in contents] == ["user", "model", "user", "user"]
    assert contents[3].parts == [genai_types.Part(text=nudge.content)]


def test_an_empty_assistant_message_is_left_out() -> None:
    _, contents = to_gemini([QUESTION, Message(role="assistant"), QUESTION])
    assert len(contents) == 1 and len(contents[0].parts or []) == 2


@pytest.mark.parametrize(
    ("choice", "mode", "allowed"),
    [
        ("auto", genai_types.FunctionCallingConfigMode.AUTO, None),
        ("none", genai_types.FunctionCallingConfigMode.NONE, None),
        (("finish",), genai_types.FunctionCallingConfigMode.ANY, ["finish"]),
    ],
)
def test_tool_choice_maps_to_a_function_calling_mode(
    choice: ToolChoice, mode: genai_types.FunctionCallingConfigMode, allowed: list[str] | None
) -> None:
    config = gemini_config("S", TOOLS, choice, max_output_tokens=900, thinking_budget=300)
    assert config.tool_config is not None
    calling = config.tool_config.function_calling_config
    assert calling is not None
    assert (calling.mode, calling.allowed_function_names) == (mode, allowed)


@pytest.mark.parametrize("choice", [(), ("absent",)])
def test_tool_choice_must_name_tools_that_were_sent(choice: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="must name tools that were sent"):
        gemini_config(None, TOOLS, choice, max_output_tokens=900, thinking_budget=300)


def test_config_carries_the_caps_and_the_schemas_as_given() -> None:
    config = gemini_config("S", TOOLS, "auto", max_output_tokens=900, thinking_budget=300)
    assert config.system_instruction == "S"
    assert config.max_output_tokens == 900
    assert config.thinking_config == genai_types.ThinkingConfig(thinking_budget=300)
    [tool] = config.tools or []
    assert isinstance(tool, genai_types.Tool)
    declarations = tool.function_declarations or []
    assert [(d.name, d.description) for d in declarations] == [
        (t.name, t.description) for t in TOOLS
    ]
    assert declarations[1].parameters_json_schema == dict(FINISH.parameters)


def test_no_tools_means_no_tool_config() -> None:
    config = gemini_config(None, [], "none", max_output_tokens=900, thinking_budget=0)
    assert config.tools is None and config.tool_config is None


# ---------------------------------------------------------------- from_gemini
def test_a_reply_becomes_text_without_thoughts_tool_calls_and_its_raw_content() -> None:
    answer = response(
        text_part("private reasoning", thought=True),
        text_part("Looking ", sig=b"s"),
        text_part("both up."),
        call_part("lookup", {"word": "quokka"}, "c1"),
    )
    reply = from_gemini(answer)
    assert reply.message.role == "assistant"
    assert reply.message.content == "Looking both up."
    assert reply.message.tool_calls == (ToolCall("c1", "lookup", {"word": "quokka"}),)
    assert reply.message.raw is answer.candidates[0].content  # type: ignore[index]
    assert reply.stop_reason == "STOP"


def test_missing_call_ids_are_filled_in_order() -> None:
    reply = from_gemini(
        response(
            text_part("x"), call_part("a", {}), call_part("b", {}, "given"), call_part("c", {})
        )
    )
    assert [c.id for c in reply.message.tool_calls] == ["call_1", "given", "call_3"]


def test_usage_maps_every_count_and_prices_cached_input_apart() -> None:
    usage = usage_of(response(prompt=1_000, output=100, thoughts=50, cached=400))
    expected = (
        600 * CHAT_PRICE_INPUT_PER_M_TOKENS_USD
        + 400 * CHAT_PRICE_CACHED_INPUT_PER_M_TOKENS_USD
        + 150 * CHAT_PRICE_OUTPUT_PER_M_TOKENS_USD
    ) / 1e6
    assert usage == Usage(
        input_tokens=1_000,
        output_tokens=100,
        thinking_tokens=50,
        cached_tokens=400,
        cost_usd=pytest.approx(expected),  # type: ignore[arg-type]
    )


def test_absent_counts_are_zero() -> None:
    """The API leaves thoughts_token_count out when the model did not think (step 1)."""
    usage = usage_of(response(prompt=None, output=None))
    assert usage == Usage(0, 0, 0, 0, 0.0)
    assert from_gemini(response(text_part("x"))).usage.thinking_tokens == 0


def test_an_empty_reply_is_an_empty_message_without_raw() -> None:
    reply = from_gemini(response())
    assert reply.message == Message(role="assistant")


def test_no_candidate_is_a_bad_answer() -> None:
    blocked = genai_types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
            block_reason=genai_types.BlockedReason.SAFETY
        ),
    )
    with pytest.raises(BadChatAnswer, match=r"no candidate.*SAFETY"):
        from_gemini(blocked)


@pytest.mark.parametrize(
    "finish",
    [genai_types.FinishReason.SAFETY, genai_types.FinishReason.MALFORMED_FUNCTION_CALL],
)
def test_a_blocked_or_malformed_reply_is_a_bad_answer(finish: genai_types.FinishReason) -> None:
    with pytest.raises(BadChatAnswer, match=f"stopped with {finish.name}"):
        from_gemini(response(text_part("partial"), finish=finish))


def test_a_call_without_a_name_is_a_bad_answer() -> None:
    nameless = genai_types.Part(function_call=genai_types.FunctionCall(args={}))
    with pytest.raises(BadChatAnswer, match="without a name"):
        from_gemini(response(nameless))


# ---------------------------------------------------------------- complete()
def test_a_call_sends_the_request_and_records_its_spend(tmp_path: Path) -> None:
    model, client, _ = chat(
        tmp_path, [response(text_part("hi"), prompt=100, output=20, thoughts=5)]
    )
    reply = ask(model)
    assert reply.message.content == "hi" and reply.usage.thinking_tokens == 5
    [request] = client.requests
    assert request["model"] == AGENT_MODEL
    config = request["config"]
    assert isinstance(config, genai_types.GenerateContentConfig)
    assert config.max_output_tokens == 1_000
    assert config.thinking_config == genai_types.ThinkingConfig(thinking_budget=256)
    totals = SpendLedger(tmp_path / "ledger.json", 10.0).totals
    assert (totals.calls, totals.tokens) == (1, 125)
    assert totals.cost_usd == pytest.approx(chat_cost(100, 0, 25))


def test_429_and_5xx_are_retried_with_exponential_backoff(tmp_path: Path) -> None:
    model, client, slept = chat(tmp_path, [api_error(429), api_error(503), two_calls()])
    assert len(ask(model).message.tool_calls) == 2
    assert len(client.requests) == 3 and slept == [1.0, 2.0]


def test_other_errors_are_not_retried(tmp_path: Path) -> None:
    model, client, slept = chat(tmp_path, [api_error(400), two_calls()])
    with pytest.raises(ChatError, match="400"):
        ask(model)
    assert len(client.requests) == 1 and slept == []


def test_retries_stop_at_the_limit(tmp_path: Path) -> None:
    model, client, slept = chat(tmp_path, [api_error(500)] * 3, max_retries=2)
    with pytest.raises(ChatError, match="500"):
        ask(model)
    assert len(client.requests) == 3 and slept == [1.0, 2.0]


def test_a_slow_answer_is_a_timeout_and_frees_its_share_of_the_cap(tmp_path: Path) -> None:
    async def slow() -> genai_types.GenerateContentResponse:
        await asyncio.sleep(5)
        return two_calls()

    # One call reserves ~$0.0375 (10,000 output tokens at $3.75/M): two do not fit in $0.05.
    model, client, _ = chat(tmp_path, [slow, two_calls()], cap=0.05)
    with pytest.raises(ChatTimeout, match=r"within 0\.1 s"):
        ask(model, max_output_tokens=10_000, timeout_s=0.1)
    assert len(ask(model, max_output_tokens=10_000).message.tool_calls) == 2
    assert len(client.requests) == 2


def test_the_cap_refuses_a_call_before_it_is_made(tmp_path: Path) -> None:
    model, client, _ = chat(tmp_path, [two_calls()], cap=0.01)
    with pytest.raises(SpendCapReached, match="spend cap"):
        ask(model, max_output_tokens=10_000)  # could cost ~$0.0375
    assert client.requests == []


def test_a_bad_answer_is_still_charged(tmp_path: Path) -> None:
    blocked = response(text_part("x"), finish=genai_types.FinishReason.SAFETY)
    model, _, _ = chat(tmp_path, [blocked])
    with pytest.raises(BadChatAnswer):
        ask(model)
    assert SpendLedger(tmp_path / "ledger.json", 10.0).totals.calls == 1


def test_a_reply_cut_at_max_output_tokens_is_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Thinking counts toward max_output_tokens (step 1 probe): the answer can be cut."""
    tracer = RecordingTracer()
    cut = response(
        text_part("41 (No"), output=6, thoughts=118, finish=genai_types.FinishReason.MAX_TOKENS
    )
    model, _, _ = chat(tmp_path, [cut])
    with caplog.at_level(logging.WARNING), tracer.trace("t"):
        reply = ask(model, max_output_tokens=128)
    assert reply.stop_reason == "MAX_TOKENS" and reply.message.content == "41 (No"
    assert "max_output_tokens=128 (thinking 118, output 6 tokens)" in caplog.text
    [generation] = tracer.named("chat")
    assert generation.level == "WARNING"
    assert "max_output_tokens=128" in (generation.status_message or "")


def test_no_key_means_no_chat_model(tmp_path: Path) -> None:
    with pytest.raises(ChatUnavailable, match="GEMINI_API_KEY is not set"):
        GeminiChat.from_settings(Settings(corpus_dir=tmp_path, port=8080))


def test_from_settings_builds_the_production_model_without_calling(tmp_path: Path) -> None:
    built = GeminiChat.from_settings(
        Settings(corpus_dir=tmp_path, port=8080, gemini_api_key=FAKE_KEY)
    )
    assert built.model == AGENT_MODEL


def test_the_async_rate_limiter_spaces_calls_evenly() -> None:
    now = [0.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = AsyncRateLimiter(60, clock=lambda: now[0], sleep=sleep)

    async def three() -> None:
        for _ in range(3):
            await limiter.wait()

    asyncio.run(three())
    assert slept == [1.0, 1.0]


# ---------------------------------------------------------------- tracing
def test_each_call_is_one_chat_generation_with_the_required_fields(tmp_path: Path) -> None:
    settings = Settings(corpus_dir=tmp_path, port=8080)
    tracer = RecordingTracer.for_settings(settings)
    answer = response(
        text_part("Looking up."), call_part("lookup", {"word": "quokka"}, "c1"), thoughts=7
    )
    model, _, _ = chat(tmp_path, [api_error(429), answer])
    with tracer.trace("t"):
        ask(model, tool_choice=("lookup",))
    [generation] = tracer.named("chat")
    assert (generation.kind, generation.model, generation.version) == (
        "generation",
        AGENT_MODEL,
        "7",
    )
    assert generation.metadata["prompt_name"] == "agent"
    assert generation.metadata["retries"] == 1
    assert generation.metadata["stop_reason"] == "STOP"
    assert generation.metadata["tools"] == ["lookup", "finish"]
    assert isinstance(generation.metadata["latency_s"], float)
    assert generation.model_parameters == {
        "max_output_tokens": 1_000,
        "thinking_budget": 256,
        "tool_choice": ["lookup"],
    }
    sent = generation.input
    assert isinstance(sent, dict)
    assert sent["contents"] == [{"role": "user", "parts": [{"text": QUESTION.content}]}]
    config = sent["config"]
    assert isinstance(config, dict)
    assert config["system_instruction"] == SYSTEM.content
    assert config["tool_config"] == {
        "function_calling_config": {"mode": "ANY", "allowed_function_names": ["lookup"]}
    }
    names = [d["name"] for d in config["tools"][0]["function_declarations"]]  # type: ignore[index]
    assert names == ["lookup", "finish"]
    assert generation.output == {
        "text": "Looking up.",
        "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"word": "quokka"}}],
    }
    assert generation.usage == {
        "input": 100,
        "output": 27,
        "total": 127,
        "thinking": 7,
        "cached_input": 0,
    }
    assert generation.cost == {"total": pytest.approx(chat_cost(100, 0, 27))}
    assert generation.level == "DEFAULT"


def test_a_failed_call_is_an_error_generation(tmp_path: Path) -> None:
    tracer = RecordingTracer()
    model, _, _ = chat(tmp_path, [api_error(400)])
    with tracer.trace("t"), pytest.raises(ChatError):
        ask(model)
    [generation] = tracer.named("chat")
    assert generation.level == "ERROR" and "400" in (generation.status_message or "")


# ---------------------------------------------------------------- network errors (Codex #5)
# The SDK raises httpx's own errors for a dropped connection. They must come out as our
# errors, so the agent's recovery (catches LLMError) runs instead of the internal-error answer.
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError, ChatError),
        (httpx.ReadError, ChatError),
        (httpx.RemoteProtocolError, ChatError),
        (httpx.ReadTimeout, ChatTimeout),
    ],
)
def test_a_network_error_is_a_chat_error_through_the_real_sdk(
    tmp_path: Path, error: type[httpx.TransportError], expected: type[ChatError]
) -> None:
    tracer = RecordingTracer()
    seen: list[httpx.Request] = []
    model = chat_on_wire(scripted([error], seen), tmp_path)
    with tracer.trace("t"), pytest.raises(ChatError, match=DROP) as caught:
        ask(model)
    assert type(caught.value) is expected
    assert len(seen) == 1  # not retried: only 429 and 5xx are
    assert SpendLedger(tmp_path / "chat-ledger.json", 10.0).totals.calls == 0
    [generation] = tracer.named("chat")
    assert generation.level == "ERROR" and DROP in (generation.status_message or "")


# ---------------------------------------------------------------- the real SDK, offline
def test_the_wire_request_and_reply_through_the_real_sdk(tmp_path: Path) -> None:
    """What google-genai actually sends and parses: signatures, ids, schemas, tool config."""
    seen: list[dict[str, object]] = []
    reply_json = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "c9",
                                "name": "finish",
                                "args": {"answer": "ok"},
                            },
                            "thoughtSignature": base64.b64encode(b"sig-from-api").decode(),
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 9, "totalTokenCount": 59},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json=reply_json)

    client = genai.Client(
        api_key=FAKE_KEY,
        http_options=genai_types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    model = GeminiChat(client, SpendLedger(tmp_path / "ledger.json", 1.0), sleep=asyncio.sleep)
    built_turn = Message(
        role="assistant",
        tool_calls=(
            ToolCall("c1", "lookup", {"word": "quokka"}),
            ToolCall("c2", "lookup", {"word": "x"}),
        ),
    )
    results = [
        Message(role="tool", content="a wallaby", tool_call_id="c1", name="lookup"),
        Message(role="tool", content="unknown", tool_call_id="c2", name="lookup"),
    ]
    reply = ask(model, [SYSTEM, QUESTION, built_turn, *results], tool_choice=("finish",))

    [call] = seen
    assert str(call["url"]).endswith(f"models/{AGENT_MODEL}:generateContent")
    body = call["body"]
    assert isinstance(body, dict)
    assert body["systemInstruction"]["parts"] == [{"text": SYSTEM.content}]
    model_turn = body["contents"][1]
    assert model_turn["role"] == "model"
    first, second = model_turn["parts"]
    assert base64.b64decode(first["thoughtSignature"]) == CHAT_STAND_IN_THOUGHT_SIGNATURE
    assert first["functionCall"] == {"id": "c1", "name": "lookup", "args": {"word": "quokka"}}
    assert "thoughtSignature" not in second
    assert [p["functionResponse"] for p in body["contents"][2]["parts"]] == [
        {"id": "c1", "name": "lookup", "response": {"result": "a wallaby"}},
        {"id": "c2", "name": "lookup", "response": {"result": "unknown"}},
    ]
    declarations = body["tools"][0]["functionDeclarations"]
    # google-genai 2.31 sends this one field in snake_case; the API accepts it (step 1 probe).
    assert declarations[1]["parameters_json_schema"] == dict(FINISH.parameters)
    assert body["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["finish"]}
    }
    assert body["generationConfig"]["maxOutputTokens"] == 1_000
    # Also snake_case on the wire; the probe showed the API honours the budget.
    assert body["generationConfig"]["thinkingConfig"] == {"thinking_budget": 256}

    assert reply.message.tool_calls == (ToolCall("c9", "finish", {"answer": "ok"}),)
    raw = reply.message.raw
    assert isinstance(raw, genai_types.Content)
    assert (raw.parts or [])[0].thought_signature == b"sig-from-api"
    assert (reply.usage.input_tokens, reply.usage.output_tokens, reply.usage.thinking_tokens) == (
        50,
        9,
        0,
    )


# ---------------------------------------------------------------- thinking level (run 4)
def chat_config(client: ScriptedChatClient) -> genai_types.GenerateContentConfig:
    [request] = client.requests
    config = request["config"]
    assert isinstance(config, genai_types.GenerateContentConfig)
    return config


@pytest.mark.parametrize(
    ("level", "sent"),
    [("low", genai_types.ThinkingLevel.LOW), ("high", genai_types.ThinkingLevel.HIGH)],
)
def test_with_a_level_an_agent_step_sends_the_level_instead_of_the_budget(
    tmp_path: Path, level: str, sent: genai_types.ThinkingLevel
) -> None:
    model, client, _ = chat(tmp_path, [two_calls()], thinking_level=level)
    ask(model, max_output_tokens=AGENT_MAX_OUTPUT_TOKENS, thinking_budget=AGENT_THINKING_BUDGET)
    # The level alone: no budget beside it.
    assert chat_config(client).thinking_config == genai_types.ThinkingConfig(thinking_level=sent)


def test_without_a_level_an_agent_step_sends_the_budget_as_in_run_1(tmp_path: Path) -> None:
    model, client, _ = chat(tmp_path, [two_calls()], thinking_level=None)
    ask(model, max_output_tokens=AGENT_MAX_OUTPUT_TOKENS, thinking_budget=AGENT_THINKING_BUDGET)
    assert chat_config(client).thinking_config == genai_types.ThinkingConfig(
        thinking_budget=AGENT_THINKING_BUDGET
    )


def test_with_a_level_the_compaction_call_keeps_its_own_budget(tmp_path: Path) -> None:
    model, client, _ = chat(tmp_path, [response(text_part("memory"))], thinking_level="high")
    ask(
        model,
        tools=[],
        tool_choice="none",
        max_output_tokens=COMPACT_CALL_MAX_OUTPUT_TOKENS,
        thinking_budget=COMPACT_THINKING_BUDGET,
    )
    assert chat_config(client).thinking_config == genai_types.ThinkingConfig(
        thinking_budget=COMPACT_THINKING_BUDGET
    )


def test_the_budgets_tell_an_agent_step_from_compaction() -> None:
    """GeminiChat.thinking_sent tells the two calls apart by their budget alone."""
    assert AGENT_THINKING_BUDGET != COMPACT_THINKING_BUDGET


@pytest.mark.parametrize("level", ["medium", "minimal", "HIGH", "", "max"])
def test_an_unknown_level_is_refused_at_construction(tmp_path: Path, level: str) -> None:
    with pytest.raises(ChatError, match=f"thinking level '{level}' is not one of"):
        chat(tmp_path, [], thinking_level=level)


def test_gemini_config_refuses_an_unknown_level() -> None:
    with pytest.raises(ValueError, match="thinking_level 'medium'"):
        gemini_config(
            None, [], "none", max_output_tokens=900, thinking_budget=0, thinking_level="medium"
        )


def test_the_shipped_level_is_accepted(tmp_path: Path) -> None:
    built = GeminiChat.from_settings(
        Settings(corpus_dir=tmp_path, port=8080, gemini_api_key=FAKE_KEY)
    )
    assert built.thinking_level == AGENT_THINKING_LEVEL


def test_the_generation_records_the_thinking_actually_sent(tmp_path: Path) -> None:
    tracer = RecordingTracer()
    answers = [two_calls(), response(text_part("memory"))]
    model, _, _ = chat(tmp_path, answers, thinking_level="high")
    with tracer.trace("t"):
        ask(model, thinking_budget=AGENT_THINKING_BUDGET)
        ask(model, tools=[], tool_choice="none", thinking_budget=COMPACT_THINKING_BUDGET)
    step, compaction = tracer.named("chat")
    assert step.model_parameters == {
        "max_output_tokens": 1_000,
        "thinking_level": "HIGH",
        "tool_choice": "auto",
    }
    assert compaction.model_parameters == {
        "max_output_tokens": 1_000,
        "thinking_budget": COMPACT_THINKING_BUDGET,
        "tool_choice": "none",
    }
    sent = step.input
    assert isinstance(sent, dict)
    assert sent["config"]["thinking_config"] == {"thinking_level": "HIGH"}  # type: ignore[index]


def test_the_level_on_the_wire_through_the_real_sdk(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []
    reply_json = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply_json)

    client = genai.Client(
        api_key=FAKE_KEY,
        http_options=genai_types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    ledger = SpendLedger(tmp_path / "ledger.json", 1.0)
    model = GeminiChat(client, ledger, sleep=asyncio.sleep, thinking_level="high")
    assert ask(model, thinking_budget=AGENT_THINKING_BUDGET).message.content == "ok"
    [body] = seen
    # google-genai 2.31 sends the field in snake_case, as it does thinking_budget.
    assert body["generationConfig"] == {"thinkingConfig": {"thinking_level": "HIGH"}} | {
        "maxOutputTokens": 1_000
    }


# ---------------------------------------------------------------- live (skipped by default)
@pytest.mark.live
def test_live_two_tool_calls_then_the_raw_reply_and_a_built_turn_are_accepted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    model = GeminiChat.from_settings(Settings(corpus_dir=tmp_path, port=8080, gemini_api_key=key))
    tracer = RecordingTracer()
    question = Message(
        role="user",
        content="Look up the words 'quokka' and 'axolotl' with the lookup tool. "
        "Make both calls in this one reply, then answer with finish.",
    )
    dictionary = {"quokka": "a small wallaby", "axolotl": "a salamander that keeps its gills"}

    async def complete(messages: list[Message]) -> ModelReply:
        return await model.complete(
            messages,
            TOOLS,
            tool_choice="auto",
            max_output_tokens=AGENT_MAX_OUTPUT_TOKENS,
            thinking_budget=AGENT_THINKING_BUDGET,
            timeout_s=60.0,
            prompt_name="live_test",
            prompt_version="1",
        )

    async def run() -> list[ModelReply]:
        first = await complete([SYSTEM, question])
        results = [
            Message(
                role="tool",
                content=dictionary.get(str(call.arguments.get("word")), "unknown"),
                tool_call_id=call.id,
                name=call.name,
            )
            for call in first.message.tool_calls
        ]
        resent = await complete([SYSTEM, question, first.message, *results])
        built = dataclasses.replace(first.message, raw=None)  # the stand-in signature path
        rebuilt = await complete([SYSTEM, question, built, *results])
        return [first, resent, rebuilt]

    with caplog.at_level(logging.INFO), tracer.trace("live"):
        first, resent, rebuilt = asyncio.run(run())
    assert len(first.message.tool_calls) >= 2 and first.message.raw is not None
    for reply in (resent, rebuilt):
        assert reply.message.tool_calls or reply.message.content
    assert len(tracer.named("chat")) == 3
    cost = sum(r.usage.cost_usd for r in (first, resent, rebuilt))
    logging.getLogger(__name__).info(
        "live chat: 3 calls, $%.5f, stops %s, calls %s",
        cost,
        [r.stop_reason for r in (first, resent, rebuilt)],
        [[c.name for c in r.message.tool_calls] for r in (first, resent, rebuilt)],
    )


# The live level test's hard spend cap (its own ledger) and output cap per call: two calls of
# at most ~$0.016 each (4,096 output tokens at $3.75/M plus a few hundred input tokens).
LIVE_LEVEL_CAP_USD = 0.05
LIVE_LEVEL_MAX_OUTPUT_TOKENS = 4_096


@pytest.mark.live
def test_live_thinking_level_high_on_a_small_tool_prompt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One agent step at thinking level HIGH, then the same step with run 1's budget.

    Asserts that the API takes the level and that the trace records it. Whether the model
    thinks is its choice: the thinking tokens, latency and cost of both calls are logged
    (run with `-o log_cli=true --log-cli-level=INFO` to see them), not asserted.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    client = genai.Client(
        vertexai=False,
        api_key=key,
        http_options=genai_types.HttpOptions(timeout=CHAT_HTTP_TIMEOUT_MS),
    )
    ledger = SpendLedger(tmp_path / "ledger.json", LIVE_LEVEL_CAP_USD)  # refuses past $0.05
    high = GeminiChat(client, ledger, thinking_level="high")
    budget = GeminiChat(client, ledger, thinking_level=None)
    tracer = RecordingTracer()
    question = Message(
        role="user",
        content="Which of 'quokka' and 'axolotl' names an animal that keeps its gills as an "
        "adult? Look both words up with the lookup tool, making both calls in this one reply.",
    )

    async def step(model: GeminiChat) -> tuple[ModelReply, float]:
        started = time.monotonic()
        reply = await model.complete(
            [SYSTEM, question],
            TOOLS,
            tool_choice="auto",
            max_output_tokens=LIVE_LEVEL_MAX_OUTPUT_TOKENS,
            thinking_budget=AGENT_THINKING_BUDGET,
            timeout_s=60.0,
            prompt_name="live_test",
            prompt_version="1",
        )
        return reply, time.monotonic() - started

    async def run() -> list[tuple[ModelReply, float]]:
        return [await step(high), await step(budget)]

    with caplog.at_level(logging.INFO), tracer.trace("live"):
        results = asyncio.run(run())
    high_generation, budget_generation = tracer.named("chat")
    assert high_generation.model_parameters["thinking_level"] == "HIGH"  # type: ignore[index]
    assert "thinking_budget" not in (high_generation.model_parameters or {})
    assert budget_generation.model_parameters["thinking_budget"] == AGENT_THINKING_BUDGET  # type: ignore[index]
    for reply, _ in results:
        assert reply.message.tool_calls or reply.message.content
    assert ledger.totals.cost_usd < LIVE_LEVEL_CAP_USD
    for name, (reply, latency) in zip(("level HIGH", "budget 1024"), results, strict=True):
        logging.getLogger(__name__).info(
            "live %s: thinking %d, output %d, input %d tokens, %.2f s, $%.5f, stop %s, calls %s",
            name,
            reply.usage.thinking_tokens,
            reply.usage.output_tokens,
            reply.usage.input_tokens,
            latency,
            reply.usage.cost_usd,
            reply.stop_reason,
            [(c.name, dict(c.arguments)) for c in reply.message.tool_calls],
        )
