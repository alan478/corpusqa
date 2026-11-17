"""Model clients: the only way code embeds text or calls a chat model (AGENTS.md: Using the LLM).

`Embedder` and `ChatModel` (in `llm.interfaces`) are what callers use. `GeminiEmbedder`
talks to gemini-embedding-2 through the google-genai SDK:

- Each text is sent as its own `Content`. A plain list of strings would become ONE
  content with many parts, and the model returns one aggregated vector for it.
- The model takes no `task_type`; the role (document or query) is a short text prefix
  from the prompt file `EMBED_PROMPT_NAME`.
- Texts go in batches of `EMBED_BATCH_SIZE`, paced by a rate limiter, retried with
  exponential backoff on 429 and 5xx answers, and charged to a spend ledger with a hard
  cap (tokens from the API when it reports them, else chars / TOKEN_CHARS).
- Every answer is checked (one vector per text, the right size, finite, non-zero) before
  anything is returned: a bad answer fails the batch, it never shifts vectors onto the
  wrong texts. Vectors are L2-normalised.
- One Langfuse generation per batch (model, texts as sent, tokens, cost, latency,
  retries, prompt name and version).

Network errors: for a dropped connection or an HTTP timeout the SDK raises httpx's own
error (`httpx.TransportError`), not an `APIError`. Both clients turn it into their own
error (`EmbeddingError`; `ChatError`, or `ChatTimeout` for a timeout) without a retry, so
callers that catch `LLMError` recover: search falls back to keywords, the agent forces a
finish or skips compaction (Codex review #5).

`GeminiChat` is the agent's chat model (plan section 4), on the SDK's async client:

- `to_gemini` turns our messages into Gemini contents. An assistant message that Gemini
  wrote is re-sent as its own `raw` Content, unchanged: its parts carry thought signatures
  the API checks. A model turn code wrote gets the stand-in signature on its first call.
- `from_gemini` turns the reply back: text (thought parts left out), tool calls (an empty
  id becomes `call_<n>`), the raw Content, usage (an absent count is 0) and cost.
- Each call is paced by a shared async rate limiter, reserved against the generation spend
  ledger, retried on 429 and 5xx inside the caller's timeout, and traced as one `chat`
  generation (full input as sent, output, tool calls, usage, cost, latency, retries,
  prompt name and version). A reply cut at max_output_tokens is logged as a warning.
- Thinking: the call's `thinking_budget` is sent, unless the instance has a thinking level
  (`AGENT_THINKING_LEVEL`) and the call asks for `AGENT_THINKING_BUDGET`: then the level is
  sent instead (see `GeminiChat.thinking_sent`). The generation records what was sent.
"""

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from corpusqa import obs
from corpusqa.config import (
    AGENT_MODEL,
    AGENT_THINKING_BUDGET,
    AGENT_THINKING_LEVEL,
    CHAT_BACKOFF_BASE_S,
    CHAT_BACKOFF_MAX_S,
    CHAT_HTTP_TIMEOUT_MS,
    CHAT_MAX_RETRIES,
    CHAT_PRICE_CACHED_INPUT_PER_M_TOKENS_USD,
    CHAT_PRICE_INPUT_PER_M_TOKENS_USD,
    CHAT_PRICE_OUTPUT_PER_M_TOKENS_USD,
    CHAT_REQUESTS_PER_MINUTE,
    CHAT_STAND_IN_THOUGHT_SIGNATURE,
    EMBED_BACKOFF_BASE_S,
    EMBED_BACKOFF_MAX_S,
    EMBED_BATCH_SIZE,
    EMBED_DIMS,
    EMBED_MAX_INPUT_TOKENS,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    EMBED_PRICE_PER_M_TOKENS_USD,
    EMBED_PROMPT_NAME,
    EMBED_REQUESTS_PER_MINUTE,
    EMBED_SPEND_CAP_USD,
    EMBED_SPEND_LEDGER,
    EMBED_TIMEOUT_MS,
    GEN_SPEND_CAP_USD,
    GEN_SPEND_LEDGER,
    TOKEN_CHARS,
    Settings,
)
from corpusqa.llm.errors import (
    BadChatAnswer,
    BadEmbeddingAnswer,
    ChatError,
    ChatTimeout,
    ChatUnavailable,
    EmbedderUnavailable,
    EmbeddingError,
)
from corpusqa.llm.interfaces import (
    ChatModel,
    EmbedRole,
    Message,
    ModelReply,
    ToolCall,
    ToolChoice,
    ToolSchema,
    Usage,
)
from corpusqa.llm.limits import AsyncRateLimiter, RateLimiter, SpendLedger
from corpusqa.prompts import PromptFile, load_prompt

logger = logging.getLogger(__name__)

_TEXT_SLOT = "{text}"
_TOKENS_PER_MILLION = 1_000_000
_TOO_MANY_REQUESTS = 429
_SERVER_ERRORS = range(500, 600)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / TOKEN_CHARS)


def checked_vectors(raw: Sequence[Sequence[float] | None], n: int, dims: int) -> list[list[float]]:
    """The answer's vectors, L2-normalised, or BadEmbeddingAnswer if any does not fit."""
    if len(raw) != n:
        raise BadEmbeddingAnswer(
            f"sent {n} texts, got {len(raw)} vectors; the batch is dropped (never misaligned)"
        )
    vectors = []
    for i, values in enumerate(raw):
        if values is None or len(values) != dims:
            size = None if values is None else len(values)
            raise BadEmbeddingAnswer(f"vector {i} has {size} values, expected {dims}")
        if not all(math.isfinite(v) for v in values):
            raise BadEmbeddingAnswer(f"vector {i} has a value that is not finite")
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            raise BadEmbeddingAnswer(f"vector {i} is all zeros")
        vectors.append([v / norm for v in values])
    return vectors


class GeminiEmbedder:
    """gemini-embedding-2 through google-genai (see the module docstring)."""

    def __init__(
        self,
        client: genai.Client,
        ledger: SpendLedger,
        *,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        model: str = EMBED_MODEL,
        dims: int = EMBED_DIMS,
        batch_size: int = EMBED_BATCH_SIZE,
        max_retries: int = EMBED_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.dims = dims
        self._client = client
        self._ledger = ledger
        self._limiter = limiter or RateLimiter(EMBED_REQUESTS_PER_MINUTE)
        self._sleep = sleep
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._prompt: PromptFile = load_prompt(EMBED_PROMPT_NAME)
        for role in ("document", "query"):
            if _TEXT_SLOT not in self._prompt.get(role):
                raise EmbeddingError(f"prompt {EMBED_PROMPT_NAME!r} entry {role!r} lacks {{text}}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiEmbedder":
        """The production embedder. No key: EmbedderUnavailable (callers go keyword-only)."""
        if not settings.gemini_api_key:
            raise EmbedderUnavailable(
                "GEMINI_API_KEY is not set: embeddings are off and search uses keywords only"
            )
        client = genai.Client(
            vertexai=False,  # the Gemini API with this key, whatever the environment says
            api_key=settings.gemini_api_key,
            http_options=genai_types.HttpOptions(timeout=EMBED_TIMEOUT_MS),
        )
        return cls(client, SpendLedger(EMBED_SPEND_LEDGER, EMBED_SPEND_CAP_USD))

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size], role))
        return vectors

    def _embed_batch(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        template = self._prompt.get(role)
        sent = [template.replace(_TEXT_SLOT, text) for text in texts]
        estimated = sum(estimate_tokens(text) for text in sent)
        # Counted against the cap while in flight: calls running at once cannot pass it.
        reserved = self._cost(estimated)
        self._ledger.reserve(reserved)
        try:
            started = time.monotonic()
            with obs.generation(
                "embed",
                model=self.model,
                input={"role": role, "texts": sent},
                model_parameters={"output_dimensionality": self.dims},
                prompt_name=self._prompt.name,
                prompt_version=self._prompt.version,
                metadata={
                    "n_texts": len(sent),
                    "chars": sum(len(text) for text in sent),
                    "texts_over_input_limit": sum(
                        estimate_tokens(text) > EMBED_MAX_INPUT_TOKENS for text in sent
                    ),
                },
            ) as generation:
                response, retries = self._call(sent, generation)
                embeddings = response.embeddings or []
                counted = [
                    e.statistics.token_count
                    for e in embeddings
                    if e.statistics is not None and e.statistics.token_count is not None
                ]
                from_api = bool(embeddings) and len(counted) == len(embeddings)
                tokens = round(sum(counted)) if from_api else estimated
                cost = self._cost(tokens)
                settled, reserved = reserved, 0.0
                # Billed even if the answer turns out unusable; frees the reservation.
                self._ledger.add(tokens, cost, reserved_usd=settled)
                generation.record_usage(input_tokens=tokens, output_tokens=0, cost_usd=cost)
                generation.update(
                    metadata={
                        "retries": retries,
                        "latency_s": round(time.monotonic() - started, 3),
                        "tokens_source": "api" if from_api else "estimate",
                    }
                )
                vectors = checked_vectors([e.values for e in embeddings], len(sent), self.dims)
                generation.update(output={"vectors": len(vectors), "dims": self.dims})
        finally:
            if reserved:  # no answer was recorded: free this call's share of the cap
                self._ledger.release(reserved)
        return vectors

    def _call(
        self, sent: list[str], generation: obs.Observation
    ) -> tuple[genai_types.EmbedContentResponse, int]:
        """One request, retried on 429 / 5xx; returns the response and the retries used."""
        contents = [genai_types.Content(parts=[genai_types.Part(text=text)]) for text in sent]
        config = genai_types.EmbedContentConfig(output_dimensionality=self.dims)
        for attempt in range(self._max_retries + 1):
            self._limiter.wait()
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=contents, config=config
                )
                return response, attempt
            except genai_errors.APIError as exc:
                retryable = exc.code == _TOO_MANY_REQUESTS or exc.code in _SERVER_ERRORS
                if not retryable or attempt == self._max_retries:
                    generation.update(metadata={"retries": attempt})
                    raise EmbeddingError(f"embedding request failed: {exc}") from exc
                delay = min(EMBED_BACKOFF_MAX_S, EMBED_BACKOFF_BASE_S * 2**attempt)
                logger.warning("embed: %s; retry %d in %.1f s", exc.code, attempt + 1, delay)
                self._sleep(delay)
            except httpx.TransportError as exc:  # the SDK raises these as they are
                generation.update(metadata={"retries": attempt})
                raise EmbeddingError(f"embedding request failed: {_network(exc)}") from exc
        raise AssertionError("unreachable: the loop returns or raises")

    @staticmethod
    def _cost(tokens: int) -> float:
        return tokens * EMBED_PRICE_PER_M_TOKENS_USD / _TOKENS_PER_MILLION


# ---------------------------------------------------------------- chat (plan section 4)
_GEN_NAME = "chat"
_MAX_TOKENS = genai_types.FinishReason.MAX_TOKENS
# A reply that stopped for any other reason carries nothing we may use (blocked, malformed
# call, ...): BadChatAnswer. No reason at all is accepted when the reply has content.
_USABLE_REASONS = frozenset(
    {
        genai_types.FinishReason.STOP,
        genai_types.FinishReason.MAX_TOKENS,
        genai_types.FinishReason.FINISH_REASON_UNSPECIFIED,
    }
)
_UNKNOWN_REASON = "UNKNOWN"
_FILLED_ID_PREFIX = "call_"
_RESULT_KEY = "result"
# The thinking levels a GeminiChat may send (AGENT_THINKING_LEVEL). The SDK also has MINIMAL
# and MEDIUM; the step 1 probe tried only LOW and HIGH, so only those are allowed.
_THINKING_LEVELS = {
    "low": genai_types.ThinkingLevel.LOW,
    "high": genai_types.ThinkingLevel.HIGH,
}


@dataclass(frozen=True, slots=True)
class ChatPrices:
    """USD per 1M tokens of one chat model: fresh input, cached input, output (incl. thinking)."""

    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


# AGENT_MODEL's prices: the default of every chat call. The eval's judge passes its own.
AGENT_PRICES = ChatPrices(
    CHAT_PRICE_INPUT_PER_M_TOKENS_USD,
    CHAT_PRICE_CACHED_INPUT_PER_M_TOKENS_USD,
    CHAT_PRICE_OUTPUT_PER_M_TOKENS_USD,
)


def chat_cost(
    input_tokens: int, cached_tokens: int, output_tokens: int, prices: ChatPrices = AGENT_PRICES
) -> float:
    """USD for one call: `input_tokens` includes the cached ones (Gemini's prompt count);
    `output_tokens` includes thinking (billed at the output price)."""
    fresh = max(0, input_tokens - cached_tokens)
    return (
        fresh * prices.input_per_m
        + cached_tokens * prices.cached_input_per_m
        + output_tokens * prices.output_per_m
    ) / _TOKENS_PER_MILLION


def to_gemini(messages: Sequence[Message]) -> tuple[str | None, list[genai_types.Content]]:
    """Our messages as Gemini's system instruction and contents (plan section 4.1).

    - `system` messages become the system instruction (joined if there are several).
    - `user` messages become a user Content; user messages in a row share one Content.
    - An `assistant` message with a Gemini `raw` Content is that Content, unchanged.
      Without one it is built from its text and tool calls, with the stand-in thought
      signature on the first call (the API refuses calls without one). An assistant
      message with nothing in it is left out.
    - `tool` messages in a row become one user Content of function_response parts, in
      order. A response carries its call's id only when the model turn before it has a
      call with that id (an id we filled in is not sent back).
    """
    system: list[str] = []
    contents: list[genai_types.Content] = []
    last: str | None = None  # the role that made contents[-1]
    call_ids: set[str] = set()  # the ids of the last model turn's calls
    for message in messages:
        if message.role == "system":
            system.append(message.content)
            continue
        if message.role == "assistant":
            content = _model_content(message)
            if content is None:
                continue
            contents.append(content)
            call_ids = {
                part.function_call.id
                for part in content.parts or []
                if part.function_call and part.function_call.id
            }
        elif message.role == "user":
            part = genai_types.Part(text=message.content)
            if last == "user":
                _parts(contents[-1]).append(part)
            else:
                contents.append(genai_types.Content(role="user", parts=[part]))
        else:  # tool
            call_id = message.tool_call_id if message.tool_call_id in call_ids else None
            response = genai_types.FunctionResponse(
                id=call_id, name=message.name, response={_RESULT_KEY: message.content}
            )
            part = genai_types.Part(function_response=response)
            if last == "tool":
                _parts(contents[-1]).append(part)
            else:
                contents.append(genai_types.Content(role="user", parts=[part]))
        last = message.role
    return ("\n\n".join(system) if system else None), contents


def _parts(content: genai_types.Content) -> list[genai_types.Part]:
    if content.parts is None:
        content.parts = []
    return content.parts


def _model_content(message: Message) -> genai_types.Content | None:
    if isinstance(message.raw, genai_types.Content) and message.raw.parts:
        return message.raw
    parts: list[genai_types.Part] = []
    if message.content:
        parts.append(genai_types.Part(text=message.content))
    for index, call in enumerate(message.tool_calls):
        parts.append(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    id=call.id, name=call.name, args=dict(call.arguments)
                ),
                thought_signature=CHAT_STAND_IN_THOUGHT_SIGNATURE if index == 0 else None,
            )
        )
    return genai_types.Content(role="model", parts=parts) if parts else None


def gemini_config(
    system: str | None,
    tools: Sequence[ToolSchema],
    tool_choice: ToolChoice,
    max_output_tokens: int,
    thinking_budget: int,
    thinking_level: str | None = None,
) -> genai_types.GenerateContentConfig:
    """The request config: system instruction, tools, tool choice, output and thinking caps.

    Thinking is the budget, or the level when one is given (in its place: only one is sent).
    """
    if thinking_level is None:
        thinking = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
    elif thinking_level in _THINKING_LEVELS:
        thinking = genai_types.ThinkingConfig(thinking_level=_THINKING_LEVELS[thinking_level])
    else:
        raise ValueError(f"thinking_level {thinking_level!r} is not one of {_level_names()}")
    declarations = [
        genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=dict(tool.parameters),
        )
        for tool in tools
    ]
    calling = _calling_config(tool_choice, {tool.name for tool in tools})
    return genai_types.GenerateContentConfig(
        system_instruction=system,
        tools=[genai_types.Tool(function_declarations=declarations)] if declarations else None,
        tool_config=genai_types.ToolConfig(function_calling_config=calling)
        if declarations
        else None,
        max_output_tokens=max_output_tokens,
        thinking_config=thinking,
    )


def _level_names() -> list[str]:
    return sorted(_THINKING_LEVELS)


def _calling_config(choice: ToolChoice, names: set[str]) -> genai_types.FunctionCallingConfig:
    mode = genai_types.FunctionCallingConfigMode
    if choice == "auto":
        return genai_types.FunctionCallingConfig(mode=mode.AUTO)
    if choice == "none":
        return genai_types.FunctionCallingConfig(mode=mode.NONE)
    unknown = sorted(set(choice) - names)
    if not choice or unknown:
        raise ValueError(f"tool_choice {choice!r} must name tools that were sent ({unknown})")
    return genai_types.FunctionCallingConfig(mode=mode.ANY, allowed_function_names=list(choice))


def usage_of(
    response: genai_types.GenerateContentResponse, prices: ChatPrices = AGENT_PRICES
) -> Usage:
    """Token counts of a reply (a count the API leaves out is 0) and their cost."""
    meta = response.usage_metadata or genai_types.GenerateContentResponseUsageMetadata()
    prompt = meta.prompt_token_count or 0
    cached = meta.cached_content_token_count or 0
    output = meta.candidates_token_count or 0
    thinking = meta.thoughts_token_count or 0  # absent, not 0, when the model did not think
    return Usage(
        input_tokens=prompt,
        output_tokens=output,
        thinking_tokens=thinking,
        cached_tokens=cached,
        cost_usd=chat_cost(prompt, cached, output + thinking, prices),
    )


def from_gemini(
    response: genai_types.GenerateContentResponse, prices: ChatPrices = AGENT_PRICES
) -> ModelReply:
    """The model's reply as our message (plan section 4.2); BadChatAnswer if unusable."""
    usage = usage_of(response, prices)
    if not response.candidates:
        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback else None
        raise BadChatAnswer(f"the reply has no candidate (prompt block reason: {reason})")
    candidate = response.candidates[0]
    reason = candidate.finish_reason
    stop = getattr(reason, "name", None) or (str(reason) if reason else _UNKNOWN_REASON)
    if reason is not None and reason not in _USABLE_REASONS:
        detail = f": {candidate.finish_message}" if candidate.finish_message else ""
        raise BadChatAnswer(f"the reply stopped with {stop}{detail}")
    raw = candidate.content
    parts = (raw.parts if raw else None) or []
    calls: list[ToolCall] = []
    for part in parts:
        if part.function_call is None:
            continue
        call = part.function_call
        if not call.name:
            raise BadChatAnswer("the reply has a function call without a name")
        call_id = call.id or f"{_FILLED_ID_PREFIX}{len(calls) + 1}"
        calls.append(ToolCall(id=call_id, name=call.name, arguments=dict(call.args or {})))
    text = "".join(part.text for part in parts if part.text and not part.thought)
    message = Message(
        role="assistant", content=text, tool_calls=tuple(calls), raw=raw if parts else None
    )
    return ModelReply(message=message, usage=usage, stop_reason=stop)


class GeminiChat:
    """The agent's chat model on google-genai's async client (see the module docstring).

    One instance serves every request of the process: the rate limiter and the spend
    ledger are shared; nothing else is kept between calls. It holds the SDK client for its
    whole life (a `genai.Client` that nothing references closes its HTTP client).

    `thinking_level` ("low", "high" or None) replaces the agent steps' thinking budget; see
    `thinking_sent`. Any other value is refused here, so a bad setting fails at start-up
    (a ChatError: the server turns the agent off and says why), never on a call.

    `prices` cost every call (usage, the ledger, the generation): the model's own prices.
    The default is AGENT_MODEL's; the eval's judge passes JUDGE_MODEL's.
    """

    def __init__(
        self,
        client: genai.Client,
        ledger: SpendLedger,
        *,
        limiter: AsyncRateLimiter | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        model: str = AGENT_MODEL,
        max_retries: int = CHAT_MAX_RETRIES,
        thinking_level: str | None = AGENT_THINKING_LEVEL,
        prices: ChatPrices = AGENT_PRICES,
    ) -> None:
        if thinking_level is not None and thinking_level not in _THINKING_LEVELS:
            raise ChatError(
                f"thinking level {thinking_level!r} is not one of {_level_names()} or None "
                "(AGENT_THINKING_LEVEL in config.py)"
            )
        self.model = model
        self.thinking_level = thinking_level
        self.prices = prices
        self._client = client
        self._ledger = ledger
        self._limiter = limiter or AsyncRateLimiter(CHAT_REQUESTS_PER_MINUTE)
        self._sleep = sleep
        self._max_retries = max_retries

    def thinking_sent(self, thinking_budget: int) -> str | None:
        """The thinking level to send in place of `thinking_budget`, or None to send the budget.

        The rule: with a level set, a call that asks for AGENT_THINKING_BUDGET gets the level;
        any other budget is sent as asked. `ChatModel.complete` carries a budget and nothing
        that says which call it is, and every agent step (the forced finish included) asks for
        AGENT_THINKING_BUDGET while the compaction call asks for COMPACT_THINKING_BUDGET, so
        the budget is what tells them apart without changing the interface or the agent.
        Compaction keeps its small budget on purpose: it summarises a transcript and its
        output cap (COMPACT_CALL_MAX_OUTPUT_TOKENS) is sized for that budget, not for HIGH.
        A unit test checks the two budgets differ, so this rule cannot catch compaction.
        """
        if self.thinking_level is not None and thinking_budget == AGENT_THINKING_BUDGET:
            return self.thinking_level
        return None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: str = AGENT_MODEL,
        prices: ChatPrices = AGENT_PRICES,
        thinking_level: str | None = AGENT_THINKING_LEVEL,
        ledger_path: Path = GEN_SPEND_LEDGER,
        cap_usd: float = GEN_SPEND_CAP_USD,
    ) -> "GeminiChat":
        """The production chat model (the defaults). The eval's judge passes its own model,
        prices, thinking and ledger. No key: ChatUnavailable."""
        if not settings.gemini_api_key:
            raise ChatUnavailable("GEMINI_API_KEY is not set: the agent has no chat model")
        client = genai.Client(
            vertexai=False,  # the Gemini API with this key, whatever the environment says
            api_key=settings.gemini_api_key,
            http_options=genai_types.HttpOptions(timeout=CHAT_HTTP_TIMEOUT_MS),
        )
        return cls(
            client,
            SpendLedger(ledger_path, cap_usd),
            model=model,
            thinking_level=thinking_level,
            prices=prices,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        tool_choice: ToolChoice,
        max_output_tokens: int,
        thinking_budget: int,
        timeout_s: float,
        prompt_name: str,
        prompt_version: str,
    ) -> ModelReply:
        system, contents = to_gemini(messages)
        config = gemini_config(
            system,
            tools,
            tool_choice,
            max_output_tokens,
            thinking_budget,
            thinking_level=self.thinking_sent(thinking_budget),
        )
        # What the request carries: {"thinking_budget": n} or {"thinking_level": "HIGH"}.
        thinking = (config.thinking_config or genai_types.ThinkingConfig()).model_dump(
            mode="json", exclude_none=True
        )
        sent = {
            "contents": [c.model_dump(mode="json", exclude_none=True) for c in contents],
            "config": config.model_dump(mode="json", exclude_none=True),
        }
        # Counted against the cap while in flight, at the most the call could cost.
        reserved = chat_cost(estimate_tokens(json.dumps(sent)), 0, max_output_tokens, self.prices)
        self._ledger.reserve(reserved)
        try:
            started = time.monotonic()
            with obs.generation(
                _GEN_NAME,
                model=self.model,
                input=sent,
                model_parameters={
                    "max_output_tokens": max_output_tokens,
                    **thinking,
                    "tool_choice": _choice_label(tool_choice),
                },
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                metadata={
                    "messages": len(messages),
                    "tools": [tool.name for tool in tools],
                    "timeout_s": timeout_s,
                },
            ) as generation:
                try:
                    async with asyncio.timeout(timeout_s):
                        response = await self._call(contents, config, generation)
                except TimeoutError as exc:
                    raise ChatTimeout(
                        f"no answer from {self.model} within {timeout_s:.1f} s (retries included)"
                    ) from exc
                usage = usage_of(response, self.prices)
                settled, reserved = reserved, 0.0
                # Billed even if the answer turns out unusable; frees the reservation.
                self._ledger.add(
                    usage.input_tokens + usage.output_tokens + usage.thinking_tokens,
                    usage.cost_usd,
                    reserved_usd=settled,
                )
                generation.record_usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens + usage.thinking_tokens,
                    cost_usd=usage.cost_usd,
                    details={
                        "thinking": usage.thinking_tokens,
                        "cached_input": usage.cached_tokens,
                    },
                )
                generation.update(metadata={"latency_s": round(time.monotonic() - started, 3)})
                reply = from_gemini(response, self.prices)
                generation.update(
                    output=_trace_output(reply), metadata={"stop_reason": reply.stop_reason}
                )
                if reply.stop_reason == _MAX_TOKENS.name:
                    problem = (
                        f"the reply stopped at max_output_tokens={max_output_tokens} "
                        f"(thinking {usage.thinking_tokens}, output {usage.output_tokens} "
                        "tokens): its text or tool call may be cut off"
                    )
                    logger.warning("chat: %s", problem)
                    generation.warning(problem)
        finally:
            if reserved:  # no answer was recorded: free this call's share of the cap
                self._ledger.release(reserved)
        return reply

    async def _call(
        self,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        generation: obs.Observation,
    ) -> genai_types.GenerateContentResponse:
        """One request, retried on 429 / 5xx (other errors at once); records the retries."""
        for attempt in range(self._max_retries + 1):
            await self._limiter.wait()
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except genai_errors.APIError as exc:
                retryable = exc.code == _TOO_MANY_REQUESTS or exc.code in _SERVER_ERRORS
                if not retryable or attempt == self._max_retries:
                    generation.update(metadata={"retries": attempt})
                    raise ChatError(f"chat request failed: {exc}") from exc
                delay = min(CHAT_BACKOFF_MAX_S, CHAT_BACKOFF_BASE_S * 2**attempt)
                logger.warning("chat: %s; retry %d in %.1f s", exc.code, attempt + 1, delay)
                await self._sleep(delay)
                continue
            except httpx.TimeoutException as exc:  # the SDK raises these as they are
                generation.update(metadata={"retries": attempt})
                raise ChatTimeout(f"chat request timed out: {_network(exc)}") from exc
            except httpx.TransportError as exc:
                generation.update(metadata={"retries": attempt})
                raise ChatError(f"chat request failed: {_network(exc)}") from exc
            generation.update(metadata={"retries": attempt})
            return response
        raise AssertionError("unreachable: the loop returns or raises")


def _network(exc: httpx.TransportError) -> str:
    """A network error as text: its kind (ConnectError, ReadTimeout, ...) and message."""
    return f"{type(exc).__name__}: {exc}"


def _choice_label(choice: ToolChoice) -> str | list[str]:
    return choice if isinstance(choice, str) else list(choice)


def _trace_output(reply: ModelReply) -> Mapping[str, object]:
    calls = [
        {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
        for call in reply.message.tool_calls
    ]
    return {"text": reply.message.content, "tool_calls": calls}


if TYPE_CHECKING:  # pyright checks that GeminiChat keeps the ChatModel shape

    def _gemini_chat_is_a_chat_model(chat: GeminiChat) -> ChatModel:
        return chat
