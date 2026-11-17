"""Offline stand-ins for tests (AGENTS.md: unit tests never touch the network)."""

import collections
import hashlib
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from corpusqa.config import EMBED_DIMS
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

_WORD = re.compile(r"\w+")
_FAKE_STOP_REASON = "STOP"


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One `complete` call as FakeLLM received it (the messages copied at call time)."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...]
    tool_choice: ToolChoice
    max_output_tokens: int
    thinking_budget: int
    timeout_s: float
    prompt_name: str
    prompt_version: str


# A scripted step: a reply to return, an exception to raise, or a function of the request
# (sync or async) that returns a reply or raises.
type ScriptEntry = (
    ModelReply | BaseException | Callable[[ChatRequest], ModelReply | Awaitable[ModelReply]]
)


class ScriptExhausted(AssertionError):
    """FakeLLM got more calls than its script has entries: the test's script is too short."""


def fake_reply(
    content: str = "",
    tool_calls: Sequence[ToolCall] = (),
    *,
    usage: Usage | None = None,
    stop_reason: str = _FAKE_STOP_REASON,
) -> ModelReply:
    """An assistant reply for a script; usage is all zeros unless given."""
    return ModelReply(
        message=Message(role="assistant", content=content, tool_calls=tuple(tool_calls)),
        usage=usage or Usage(0, 0, 0, 0, 0.0),
        stop_reason=stop_reason,
    )


class FakeLLM:
    """A `ChatModel` that plays a script, one entry per `complete` call, in order.

    Every call is recorded in `requests` before its entry plays, so a test can check what
    was sent even when the entry raises. A call after the last entry raises
    ScriptExhausted. Unlike the real client it keeps state (the script): use one per test.
    """

    model = "fake-chat"

    def __init__(self, script: Iterable[ScriptEntry]) -> None:
        self._script: collections.deque[ScriptEntry] = collections.deque(script)
        self._played = 0
        self.requests: list[ChatRequest] = []

    @property
    def remaining(self) -> int:
        """Script entries not played yet."""
        return len(self._script)

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
        request = ChatRequest(
            messages=tuple(messages),
            tools=tuple(tools),
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            timeout_s=timeout_s,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )
        self.requests.append(request)
        if not self._script:
            last = messages[-1].role if messages else "none"
            raise ScriptExhausted(
                f"FakeLLM script ran out: call {len(self.requests)} came after all "
                f"{self._played} scripted entries were played (last message role: {last}, "
                f"tool_choice: {tool_choice!r}); add entries to the script"
            )
        entry = self._script.popleft()
        self._played += 1
        if isinstance(entry, ModelReply):
            return entry
        if isinstance(entry, BaseException):
            raise entry
        result = entry(request)
        if inspect.isawaitable(result):
            result = await result
        return result


if TYPE_CHECKING:  # pyright checks that FakeLLM keeps the ChatModel shape

    def _fake_llm_is_a_chat_model(fake: FakeLLM) -> ChatModel:
        return fake


class FakeEmbedder:
    """Deterministic bag-of-words vectors: each word adds +-1 to a hashed slot, then L2-norm.

    Texts that share words get similar vectors, so retrieval tests behave like the real
    thing; the role does not change the vector. Every call is recorded in `calls`.
    """

    model = "fake-embedding"

    def __init__(self, dims: int = EMBED_DIMS) -> None:
        self.dims = dims
        self.calls: list[tuple[EmbedRole, tuple[str, ...]]] = []

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        self.calls.append((role, tuple(texts)))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dims
        for word in _WORD.findall(text.casefold()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dims
            values[slot] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:  # no words: a fixed unit vector
            return [1.0] + [0.0] * (self.dims - 1)
        return [v / norm for v in values]
