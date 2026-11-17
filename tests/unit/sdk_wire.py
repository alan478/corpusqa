"""The real google-genai SDK over a faked network (httpx.MockTransport), for tests.

Each request goes to `handler`, which answers or raises an httpx error exactly as a
dropped connection would. Used where the SDK's own behaviour matters (Codex review #5:
which exceptions a network failure raises through it).
"""

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import httpx
from google import genai
from google.genai import types as genai_types

from corpusqa.llm.client import GeminiChat, GeminiEmbedder
from corpusqa.llm.limits import AsyncRateLimiter, RateLimiter, SpendLedger

FAKE_KEY = "fake-key-for-tests"  # never a real key
DROP = "scripted network failure"

type Handler = Callable[[httpx.Request], httpx.Response]
type Wire = httpx.Response | type[httpx.TransportError]


def scripted(answers: Iterable[Wire], seen: list[httpx.Request]) -> Handler:
    """A handler that records each request and plays the next answer (or raises it)."""
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = queue.pop(0)
        if isinstance(answer, httpx.Response):
            return answer
        raise answer(DROP, request=request)

    return handler


def chat_on_wire(handler: Handler, tmp_path: Path, cap: float = 10.0) -> GeminiChat:
    async def no_wait(seconds: float) -> None:
        return None

    client = genai.Client(
        api_key=FAKE_KEY,
        http_options=genai_types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    return GeminiChat(
        client,
        SpendLedger(tmp_path / "chat-ledger.json", cap),
        limiter=AsyncRateLimiter(1e9, sleep=no_wait),
        sleep=asyncio.sleep,
    )


def embedder_on_wire(handler: Handler, tmp_path: Path, dims: int) -> GeminiEmbedder:
    client = genai.Client(
        api_key=FAKE_KEY,
        http_options=genai_types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    return GeminiEmbedder(
        client,
        SpendLedger(tmp_path / "embed-ledger.json", 10.0),
        limiter=RateLimiter(1e9, sleep=lambda s: None),
        sleep=lambda s: None,
        dims=dims,
    )


def chat_reply(
    calls: Sequence[tuple[str, str, Mapping[str, object]]] = (),
    text: str = "",
    prompt_tokens: int = 100,
) -> httpx.Response:
    """A generateContent answer: optional text, then function calls (id, name, args)."""
    parts: list[dict[str, object]] = [{"text": text}] if text else []
    parts += [
        {"functionCall": {"id": call_id, "name": name, "args": dict(args)}}
        for call_id, name, args in calls
    ]
    body = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": 10,
            "totalTokenCount": prompt_tokens + 10,
        },
    }
    return httpx.Response(200, json=body)
