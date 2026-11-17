"""The server with the real start-up job (ingest): /health is 503 until the index is
built or reused, then 200; it stays 503 when ingest fails. `__main__` wires ingest."""

import functools
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from corpusqa.agent import PlaceholderAnswerer
from corpusqa.config import NOT_READY_ANSWER, PLACEHOLDER_ANSWER, Settings
from corpusqa.ingest import Runtime, run
from corpusqa.llm.client import GeminiEmbedder
from corpusqa.llm.errors import SpendCapReached
from corpusqa.llm.fakes import FakeEmbedder
from corpusqa.llm.interfaces import EmbedRole
from corpusqa.obs.testing import RecordingTracer
from corpusqa.server import __main__ as server_main
from corpusqa.server import create_app

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
WAIT_S = 10.0


def wait_for_health(client: TestClient, status: str) -> None:
    deadline = time.monotonic() + WAIT_S
    while client.get("/health").json()["status"] != status:
        assert time.monotonic() < deadline, f"/health never said {status!r}"
        time.sleep(0.01)


def test_health_is_503_until_ingest_returns_then_200(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    settings = Settings(corpus_dir=corpus, port=8080, index_dir=tmp_path / "index")
    gate = threading.Event()

    def gated_ingest(s: Settings) -> Runtime:
        gate.wait(WAIT_S)
        return run(s)

    tracer = RecordingTracer.for_settings(settings)
    app = create_app(settings, gated_ingest, PlaceholderAnswerer(), tracer=tracer)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        early = client.post("/ask", json={"question": "Anything yet?"}).json()
        assert early["answer"] == NOT_READY_ANSWER
        gate.set()
        wait_for_health(client, "ok")
        assert client.get("/health").status_code == 200
        body = client.post("/ask", json={"question": "Anything now?"}).json()
        assert body["answer"] == PLACEHOLDER_ANSWER
    [startup] = tracer.traces("startup")
    assert startup.output == {"known_doc_ids": 21}
    [ingest] = tracer.named("ingest")
    assert ingest.trace_id == startup.trace_id  # ingest spans live in the startup trace
    assert [s.name for s in tracer.named("index_cache")] == ["index_cache"]


def test_health_turns_200_even_when_every_embedding_request_fails(tmp_path: Path) -> None:
    """A partial embed never stops start-up: search falls back to keywords."""

    class Refused(FakeEmbedder):
        def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
            raise SpendCapReached("spend cap reached (test)")

    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    settings = Settings(corpus_dir=corpus, port=8080, index_dir=tmp_path / "index",
                        embed_cache_db=tmp_path / "embed-cache.sqlite")  # fmt: skip
    seen: list[Runtime] = []

    def startup(s: Settings) -> Runtime:
        seen.append(run(s, Refused()))
        return seen[-1]

    with TestClient(create_app(settings, startup, PlaceholderAnswerer())) as client:
        wait_for_health(client, "ok")
        assert client.get("/health").status_code == 200
    [runtime] = seen
    assert runtime.vectors is None and runtime.embed_coverage == 0.0
    assert runtime.chunks is not None and runtime.chunks.count_chunks() == 21


def test_health_stays_503_when_ingest_fails(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    blocked = tmp_path / "index"
    blocked.write_text("a file where the index folder should be", encoding="utf-8")
    settings = Settings(corpus_dir=corpus, port=8080, index_dir=blocked)
    with TestClient(create_app(settings, run, PlaceholderAnswerer())) as client:
        wait_for_health(client, "failed")
        assert client.get("/health").status_code == 503


def wired_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str | None
) -> dict[str, object]:
    """Run `main` with the server parts faked; returns what it handed to `create_app`."""
    seen: dict[str, object] = {}

    def fake_create_app(settings: Settings, startup: object, **kwargs: object) -> str:
        seen.update(kwargs, settings=settings, startup=startup)
        return "app"

    monkeypatch.setattr(server_main, "create_app", fake_create_app)
    monkeypatch.setattr(server_main.uvicorn, "run", lambda app, **_: seen.update(app=app))
    for name in ("GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    if key is not None:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    monkeypatch.setenv("PORT", "8080")
    assert server_main.main() == 0
    assert seen["app"] == "app"
    # The agent needs the index, so main hands over a factory, not a ready answerer.
    assert "answerer" not in seen and callable(seen["answerer_factory"])
    return seen


def wired_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str | None) -> object:
    """Run `main` with the server parts faked; returns the start-up job it wired."""
    return wired_main(tmp_path, monkeypatch, key)["startup"]


def test_main_wires_the_real_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = wired_startup(tmp_path, monkeypatch, key=None)
    assert isinstance(startup, functools.partial) and startup.func is run
    assert startup.keywords == {"embedder": None}  # no key: keyword-only search


def test_main_hands_the_gemini_embedder_to_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Building the client makes no network call; nothing is embedded here.
    startup = wired_startup(tmp_path, monkeypatch, key="test-key-not-real")
    assert isinstance(startup, functools.partial) and startup.func is run
    assert isinstance(startup.keywords["embedder"], GeminiEmbedder)
