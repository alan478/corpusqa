"""Entry point that ./serve execs: settings from env, wire the parts, run one uvicorn process.

This module is the only place that picks concrete parts, and the only module outside `llm/`
that imports `llm.client` (the import contract allows exactly this one edge); everything
below types against `llm.interfaces`.

- Start-up is the real ingest pipeline (`corpusqa.ingest.run`: build the index of
  CORPUS_DIR, or reuse the one cached for the same corpus and code), embedding with the
  Gemini embedder built here. Without a usable key the embedder is None and search is
  keyword-only.
- The answerer is the agent: the Gemini chat model built here, and the corpus tools over
  the index start-up returns. The tools need that index, so the answerer is built by
  `build_answerer`, which the server calls once in the start-up thread when the
  `Runtime` is ready (before /health says 200): it learns the field catalog for the system
  prompt and makes one tool session per question. Without a chat model (no key) every
  /ask gives the internal-error answer (`UnavailableAnswerer`): nothing can answer, and
  that answer never claims the records lack one.
- The tracer is Langfuse when both LANGFUSE_* keys are set, otherwise off (the grader's
  case).
"""

import functools
import logging
import sys

import uvicorn

from corpusqa import obs
from corpusqa.agent.answerer import AgentAnswerer, Answerer, UnavailableAnswerer
from corpusqa.agent.corpus_tools import corpus_session_factory
from corpusqa.config import (
    HOST,
    LOG_FORMAT,
    LOG_LEVEL,
    SHUTDOWN_GRACE_S,
    ConfigError,
    Settings,
)
from corpusqa.ingest import Runtime
from corpusqa.ingest import run as ingest_corpus
from corpusqa.llm.client import GeminiChat, GeminiEmbedder
from corpusqa.llm.errors import ChatUnavailable, EmbedderUnavailable, LLMError
from corpusqa.llm.interfaces import ChatModel, Embedder
from corpusqa.obs import init_tracing
from corpusqa.server.app import create_app

logger = logging.getLogger("corpusqa.server")

# Why every /ask gives the internal-error answer when the chat model could not be built.
NO_CHAT_MODEL = "the agent has no chat model ({problem}), so no question can be answered"


def build_embedder(settings: Settings) -> Embedder | None:
    """The Gemini embedder, or None (keyword-only search) when it cannot be built."""
    try:
        return GeminiEmbedder.from_settings(settings)
    except EmbedderUnavailable as exc:  # no key
        logger.warning("%s", exc)
    except LLMError as exc:  # e.g. an unreadable spend ledger
        logger.error("embeddings are off: %s", exc)
    return None


def build_chat(settings: Settings) -> tuple[ChatModel | None, str]:
    """The Gemini chat model, or None and why it could not be built."""
    try:
        return GeminiChat.from_settings(settings), ""
    except ChatUnavailable as exc:  # no key
        logger.warning("the agent is off: %s", exc)
        return None, str(exc)
    except LLMError as exc:  # e.g. an unreadable spend ledger
        logger.error("the agent is off: %s", exc)
        return None, str(exc)


def build_answerer(
    runtime: Runtime, *, chat: ChatModel | None, problem: str, settings: Settings
) -> Answerer:
    """The agent over the index start-up returned (called once, in the start-up thread).

    No chat model: `UnavailableAnswerer`, so every /ask gives the internal-error answer.
    A Runtime without the index readers cannot serve the agent: that fails start-up.
    """
    if chat is None:
        reason = NO_CHAT_MODEL.format(problem=problem)
        obs.current().warning(reason)
        return UnavailableAnswerer(reason)
    if runtime.store is None or runtime.chunks is None:
        raise ValueError("start-up returned no index readers: the agent's tools need them")
    tools = corpus_session_factory(runtime.store, runtime.chunks, runtime.vectors, runtime.embedder)
    obs.current().update(
        metadata={
            "catalog_chars": len(tools.catalog),
            "documents": len(tools.known_doc_ids),
            "dense_search": tools.embedder is not None,
        }
    )
    logger.info(
        "agent: ready with %d documents, a %d-character field catalog, %s search",
        len(tools.known_doc_ids),
        len(tools.catalog),
        "hybrid" if tools.embedder is not None else "keyword-only",
    )
    return AgentAnswerer(chat, tools, catalog=tools.catalog, ask_deadline_s=settings.ask_deadline_s)


def main() -> int:
    # stderr only; uvicorn's own loggers propagate here because log_config=None.
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stderr)
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        logger.critical("cannot start: %s", exc)
        return 2
    logger.info(
        "starting: corpus_dir=%s port=%d gemini_api_key=%s",
        settings.corpus_dir,
        settings.port,
        "set" if settings.gemini_api_key else "unset",
    )
    tracer = init_tracing(settings)  # logs one line: tracing on or off
    startup = functools.partial(ingest_corpus, embedder=build_embedder(settings))
    chat, problem = build_chat(settings)
    answerer_factory = functools.partial(
        build_answerer, chat=chat, problem=problem, settings=settings
    )
    app = create_app(settings, startup=startup, tracer=tracer, answerer_factory=answerer_factory)
    # One process, no reload or worker pool: it stays in ./serve's process group, so the
    # grader's SIGTERM reaches it, and uvicorn turns SIGTERM into a graceful shutdown.
    # Access logs are off: /health is polled every second; /ask logs its own line.
    uvicorn.run(
        app,
        host=HOST,
        port=settings.port,
        workers=1,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=SHUTDOWN_GRACE_S,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
