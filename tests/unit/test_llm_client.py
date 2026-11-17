"""Embedding client: per-text contents, validation, retries, spend cap, tracing, fakes.

Offline: a scripted stand-in for the SDK client, or the real SDK over an httpx mock
transport. One live test (marked) calls the real API.
"""

import json
import math
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from corpusqa.config import EMBED_DIMS, EMBED_PRICE_PER_M_TOKENS_USD, Settings
from corpusqa.llm.client import GeminiEmbedder, checked_vectors
from corpusqa.llm.errors import EmbedderUnavailable, EmbeddingError, LLMError, SpendCapReached
from corpusqa.llm.fakes import FakeEmbedder
from corpusqa.llm.limits import RateLimiter, SpendLedger
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_prompt
from tests.unit.sdk_wire import DROP, embedder_on_wire, scripted

DIMS = 4
FAKE_KEY = "fake-key-for-tests"  # never a real key


def response(n: int, dims: int = DIMS, value: float = 3.0) -> genai_types.EmbedContentResponse:
    return genai_types.EmbedContentResponse(
        embeddings=[genai_types.ContentEmbedding(values=[value] * dims) for _ in range(n)]
    )


class ScriptedClient:
    """Stands in for genai.Client: each call pops the next answer (a response or an error)."""

    def __init__(self, answers: Sequence[object]) -> None:
        self.models = self
        self.answers = list(answers)
        self.requests: list[list[genai_types.Content]] = []

    def embed_content(
        self, *, model: str, contents: list[genai_types.Content], config: object
    ) -> genai_types.EmbedContentResponse:
        self.requests.append(contents)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(contents)
        assert isinstance(answer, genai_types.EmbedContentResponse)
        return answer


def one_per_text(contents: list[genai_types.Content]) -> genai_types.EmbedContentResponse:
    return response(len(contents))


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": "scripted"}})


def embedder(
    tmp_path: Path, answers: Sequence[object], cap: float = 10.0, **kwargs: object
) -> tuple[GeminiEmbedder, ScriptedClient, list[float]]:
    client = ScriptedClient(answers)
    slept: list[float] = []
    built = GeminiEmbedder(
        client,  # type: ignore[arg-type]
        SpendLedger(tmp_path / "ledger.json", cap),
        limiter=RateLimiter(1e9, sleep=lambda s: None),
        sleep=slept.append,
        dims=DIMS,
        **kwargs,  # type: ignore[arg-type]
    )
    return built, client, slept


# ---------------------------------------------------------------- request shape
def test_each_text_is_its_own_content_on_the_real_sdk_path(tmp_path: Path) -> None:
    """Through the real SDK: one batchEmbedContents request entry per text, role prefixed."""
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "body": body})
        vectors = [{"values": [1.0, 0.0, 0.0, 0.0]} for _ in body["requests"]]
        return httpx.Response(200, json={"embeddings": vectors})

    client = genai.Client(
        api_key=FAKE_KEY,
        http_options=genai_types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    real = GeminiEmbedder(
        client, SpendLedger(tmp_path / "ledger.json", 1.0), dims=DIMS, sleep=lambda s: None
    )
    vectors = real.embed(["alpha", "beta", "gamma"], "document")
    assert vectors == [[1.0, 0.0, 0.0, 0.0]] * 3
    [call] = seen
    assert str(call["url"]).endswith("models/gemini-embedding-2:batchEmbedContents")
    body = call["body"]
    assert isinstance(body, dict)
    template = load_prompt("embed_roles").get("document")
    assert [r["content"]["parts"] for r in body["requests"]] == [
        [{"text": template.replace("{text}", word)}] for word in ("alpha", "beta", "gamma")
    ]
    assert all(r["outputDimensionality"] == DIMS for r in body["requests"])
    assert not any("taskType" in r for r in body["requests"])


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout])
def test_a_network_error_is_an_embedding_error_through_the_real_sdk(
    tmp_path: Path, error: type[httpx.TransportError]
) -> None:
    """Codex review #5: search catches LLMError to fall back to keywords; httpx's own
    errors (what the SDK raises for a dropped connection) would get past it."""
    seen: list[httpx.Request] = []
    real = embedder_on_wire(scripted([error], seen), tmp_path, DIMS)
    with pytest.raises(EmbeddingError, match=DROP):
        real.embed(["alpha"], "query")
    assert len(seen) == 1  # not retried: only 429 and 5xx are
    assert SpendLedger(tmp_path / "embed-ledger.json", 10.0).totals.calls == 0


def test_plain_strings_would_be_one_content_which_is_why_we_wrap_each_text() -> None:
    from google.genai import _transformers

    assert len(_transformers.t_contents(["alpha", "beta"])) == 1  # aggregated by the SDK


def test_the_role_prefix_comes_from_the_versioned_prompt_file(tmp_path: Path) -> None:
    built, client, _ = embedder(tmp_path, [one_per_text])
    built.embed(["when is the drill?"], "query")
    prompt = load_prompt("embed_roles")
    assert prompt.version and "{text}" in prompt.get("query")
    [[content]] = client.requests
    assert content.parts is not None
    assert content.parts[0].text == prompt.get("query").replace("{text}", "when is the drill?")


def test_texts_go_in_batches_and_come_back_in_order(tmp_path: Path) -> None:
    def numbered(contents: list[genai_types.Content]) -> genai_types.EmbedContentResponse:
        values = [[float(len(c.parts[0].text or "")), 1.0, 0.0, 0.0] for c in contents]  # type: ignore[index]
        return genai_types.EmbedContentResponse(
            embeddings=[genai_types.ContentEmbedding(values=v) for v in values]
        )

    built, client, _ = embedder(tmp_path, [numbered] * 3, batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = built.embed(texts, "document")
    assert [len(r) for r in client.requests] == [2, 2, 1]
    lengths = [round(v[0] / v[1]) for v in vectors]  # un-normalise: first value is the length
    assert lengths == sorted(lengths) and len(vectors) == 5
    assert built.embed([], "document") == [] and len(client.requests) == 3


# ---------------------------------------------------------------- validation
def test_an_aggregated_answer_fails_the_batch(tmp_path: Path) -> None:
    built, _, _ = embedder(tmp_path, [response(1)])
    with pytest.raises(EmbeddingError, match="sent 3 texts, got 1 vectors"):
        built.embed(["a", "b", "c"], "document")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([[1.0, 2.0, 3.0]], "has 3 values"),
        ([None], "has None values"),
        ([[1.0, math.nan, 0.0, 0.0]], "not finite"),
        ([[0.0, 0.0, 0.0, 0.0]], "all zeros"),
    ],
)
def test_a_vector_that_does_not_fit_fails_the_batch(
    raw: list[list[float] | None], message: str
) -> None:
    with pytest.raises(EmbeddingError, match=message):
        checked_vectors(raw, 1, DIMS)


def test_vectors_are_l2_normalised() -> None:
    [vector] = checked_vectors([[3.0, 4.0, 0.0, 0.0]], 1, DIMS)
    assert vector == [0.6, 0.8, 0.0, 0.0]


# ---------------------------------------------------------------- retries
def test_429_and_5xx_are_retried_with_exponential_backoff(tmp_path: Path) -> None:
    built, client, slept = embedder(tmp_path, [api_error(429), api_error(503), one_per_text])
    assert len(built.embed(["a"], "document")) == 1
    assert len(client.requests) == 3 and slept == [1.0, 2.0]


def test_other_errors_are_not_retried(tmp_path: Path) -> None:
    built, client, slept = embedder(tmp_path, [api_error(400), one_per_text])
    with pytest.raises(EmbeddingError, match="400"):
        built.embed(["a"], "document")
    assert len(client.requests) == 1 and slept == []


def test_retries_stop_at_the_limit(tmp_path: Path) -> None:
    built, client, slept = embedder(tmp_path, [api_error(500)] * 3, max_retries=2)
    with pytest.raises(EmbeddingError, match="500"):
        built.embed(["a"], "document")
    assert len(client.requests) == 3 and slept == [1.0, 2.0]


def test_the_rate_limiter_spaces_calls_evenly() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert slept == [1.0, 1.0]


# ---------------------------------------------------------------- spend
def test_spend_is_recorded_and_persisted(tmp_path: Path) -> None:
    built, _, _ = embedder(tmp_path, [one_per_text])
    built.embed(["x" * 400], "document")  # with its prefix: a little over 100 tokens
    totals = SpendLedger(tmp_path / "ledger.json", 10.0).totals
    sent = load_prompt("embed_roles").get("document").replace("{text}", "x" * 400)
    tokens = math.ceil(len(sent) / 4)
    assert (totals.calls, totals.tokens) == (1, tokens)
    assert totals.cost_usd == pytest.approx(tokens * EMBED_PRICE_PER_M_TOKENS_USD / 1e6)


def test_token_counts_from_the_api_win_over_the_estimate(tmp_path: Path) -> None:
    stats = genai_types.ContentEmbeddingStatistics(token_count=7)
    answer = genai_types.EmbedContentResponse(
        embeddings=[genai_types.ContentEmbedding(values=[1.0] * DIMS, statistics=stats)] * 2
    )
    built, _, _ = embedder(tmp_path, [answer])
    built.embed(["a", "b"], "document")
    assert SpendLedger(tmp_path / "ledger.json", 10.0).totals.tokens == 14


def test_the_cap_refuses_calls_before_they_are_made(tmp_path: Path) -> None:
    built, client, _ = embedder(tmp_path, [one_per_text, one_per_text], cap=3e-5)
    built.embed(["x" * 400], "document")  # ~$0.00002: under the cap
    with pytest.raises(SpendCapReached, match="spend cap"):
        built.embed(["x" * 400], "document")  # would pass it
    assert len(client.requests) == 1


def test_calls_in_flight_count_against_the_cap(tmp_path: Path) -> None:
    # B5 review finding 6: the cap was checked against finished calls only, so four calls
    # in flight could all pass a cap that had room for one.
    entered: list[int] = []
    go = threading.Event()

    def held(contents: list[genai_types.Content]) -> genai_types.EmbedContentResponse:
        entered.append(1)
        go.wait(timeout=5)
        return response(len(contents))

    sent = load_prompt("embed_roles").get("document").replace("{text}", "x" * 400)
    cost = math.ceil(len(sent) / 4) * EMBED_PRICE_PER_M_TOKENS_USD / 1e6
    built, _client, _ = embedder(tmp_path, [held, held, held], cap=2.5 * cost)
    refused: list[BaseException] = []

    def call() -> None:
        try:
            built.embed(["x" * 400], "document")
        except SpendCapReached as exc:
            refused.append(exc)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while len(entered) + len(refused) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    in_flight = len(entered)
    go.set()
    for thread in threads:
        thread.join(timeout=5)
    assert (in_flight, len(refused)) == (2, 1)
    assert SpendLedger(tmp_path / "ledger.json", 1.0).totals.calls == 2


def test_a_failed_call_frees_its_share_of_the_cap(tmp_path: Path) -> None:
    sent = load_prompt("embed_roles").get("document").replace("{text}", "x" * 400)
    cost = math.ceil(len(sent) / 4) * EMBED_PRICE_PER_M_TOKENS_USD / 1e6
    built, client, _ = embedder(tmp_path, [api_error(400), one_per_text], cap=1.5 * cost)
    with pytest.raises(EmbeddingError, match="failed"):
        built.embed(["x" * 400], "document")
    assert built.embed(["x" * 400], "document")  # the failed call no longer holds the cap
    assert len(client.requests) == 2


def test_an_unreadable_ledger_refuses_instead_of_starting_from_zero(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(LLMError, match="unreadable"):
        SpendLedger(path, 10.0)


def test_a_bad_answer_is_still_charged(tmp_path: Path) -> None:
    built, _, _ = embedder(tmp_path, [response(1)])
    with pytest.raises(EmbeddingError):
        built.embed(["a", "b"], "document")
    assert SpendLedger(tmp_path / "ledger.json", 10.0).totals.calls == 1


# ---------------------------------------------------------------- key, tracing
def test_no_key_means_no_embedder_with_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EmbedderUnavailable, match="GEMINI_API_KEY is not set"):
        GeminiEmbedder.from_settings(Settings(corpus_dir=tmp_path, port=8080))


def test_each_batch_is_one_generation_with_usage_and_prompt_version(tmp_path: Path) -> None:
    settings = Settings(corpus_dir=tmp_path, port=8080)
    tracer = RecordingTracer.for_settings(settings)
    built, _, _ = embedder(tmp_path, [api_error(429), one_per_text, one_per_text], batch_size=2)
    with tracer.trace("t"):
        built.embed(["a", "b", "c"], "query")
    first, second = tracer.named("embed")
    prompt = load_prompt("embed_roles")
    assert (first.kind, first.model, first.version) == ("generation", built.model, prompt.version)
    assert first.metadata["prompt_name"] == "embed_roles" and first.metadata["n_texts"] == 2
    assert (first.metadata["retries"], second.metadata["retries"]) == (1, 0)
    assert first.metadata["tokens_source"] == "estimate" and "latency_s" in first.metadata
    assert first.model_parameters == {"output_dimensionality": DIMS}
    assert isinstance(first.input, dict) and first.input["role"] == "query"
    assert first.input["texts"] == [prompt.get("query").replace("{text}", t) for t in "ab"]
    assert first.output == {"vectors": 2, "dims": DIMS}
    assert first.usage is not None and first.usage["input"] > 0
    assert first.cost is not None and first.cost["total"] > 0


def test_a_failed_batch_is_an_error_generation(tmp_path: Path) -> None:
    settings = Settings(corpus_dir=tmp_path, port=8080)
    tracer = RecordingTracer.for_settings(settings)
    built, _, _ = embedder(tmp_path, [response(1)])
    with tracer.trace("t"), pytest.raises(EmbeddingError):
        built.embed(["a", "b"], "document")
    [generation] = tracer.named("embed")
    assert generation.level == "ERROR" and "got 1 vectors" in (generation.status_message or "")


# ---------------------------------------------------------------- fake
def test_fake_embedder_is_deterministic_unit_length_and_word_aware() -> None:
    fake = FakeEmbedder(dims=64)
    a, b, c = fake.embed(["boiler engineer at ten", "the boiler engineer", "fire drill"], "query")
    assert fake.embed(["boiler engineer at ten"], "document")[0] == a
    assert all(math.isclose(sum(v * v for v in vec), 1.0) for vec in (a, b, c))

    def dot(x: list[float], y: list[float]) -> float:
        return sum(p * q for p, q in zip(x, y, strict=True))

    assert dot(a, b) > dot(a, c)
    assert fake.calls[0] == (
        "query",
        ("boiler engineer at ten", "the boiler engineer", "fire drill"),
    )
    assert fake.embed([""], "query")[0][0] == 1.0  # no words: still a unit vector


# ---------------------------------------------------------------- live (skipped by default)
@pytest.mark.live
def test_live_gemini_returns_one_unit_vector_per_text(tmp_path: Path) -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    real = GeminiEmbedder.from_settings(
        Settings(corpus_dir=tmp_path, port=8080, gemini_api_key=key)
    )
    docs = [
        "The boiler engineer arrives at ten on Monday.",
        "The quarterly fire drill is on Thursday afternoon.",
        "The night shift handover note is in the blue folder.",
    ]
    vectors = real.embed(docs, "document")
    assert len(vectors) == 3 and all(len(v) == EMBED_DIMS for v in vectors)
    assert all(math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-6) for v in vectors)
    assert vectors[0] != vectors[1]  # not one aggregated vector repeated
    [query] = real.embed(["When does the boiler engineer arrive?"], "query")
    scores = [sum(q * d for q, d in zip(query, v, strict=True)) for v in vectors]
    assert max(range(3), key=scores.__getitem__) == 0
