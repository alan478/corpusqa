"""What callers of the model clients depend on, without the SDK behind them.

Other layers type against these (e.g. retrieval takes an `Embedder`, the agent takes a
`ChatModel`); only the wiring code builds the Gemini adapters from `llm.client`. Keeping
the protocols and message types here keeps the Gemini SDK out of every layer that only
uses a model.

Chat messages follow one shape for every provider (plan section 3):
- `system` and `user` messages hold text only.
- An `assistant` message holds the model's text and/or its tool calls. `raw` is the
  provider's own reply object, kept opaque so the client can send it back unchanged.
- A `tool` message answers one tool call: `tool_call_id` names the call, `name` the tool.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

type EmbedRole = Literal["document", "query"]

type Role = Literal["system", "user", "assistant", "tool"]
# "auto": the model may call any tool or none; "none": no tool calls; a tuple of names:
# the model must call one of these tools.
type ToolChoice = Literal["auto", "none"] | tuple[str, ...]


class Embedder(Protocol):
    """Turns texts into unit vectors, one per text, in order."""

    model: str
    dims: int

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call the model asked for."""

    id: str  # the client fills one in when the model gives none
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Message:
    """One message of a chat request or reply (see the module docstring for the roles)."""

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant only
    tool_call_id: str | None = None  # tool only: the call it answers
    name: str | None = None  # tool only: the tool's name
    raw: object | None = None  # assistant only: the provider's own reply, opaque


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the model sees it: name, description and a JSON schema of its arguments."""

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens and cost of one model call, as the provider reported them."""

    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cached_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class ModelReply:
    """The model's answer to one chat request."""

    message: Message  # role "assistant"
    usage: Usage
    stop_reason: str


class ChatModel(Protocol):
    """Sends one chat request and returns the model's reply.

    Called concurrently by requests running at once: keep no per-request state on self.
    `prompt_name` and `prompt_version` name the system prompt, for the trace.
    """

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
    ) -> ModelReply: ...
