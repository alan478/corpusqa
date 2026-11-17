"""The answering contract the server calls, the agent behind it, a placeholder, and a
stand-in for when the agent cannot run (`UnavailableAnswerer`).

The server owns HTTP concerns (deadline, id filtering, size cap, error answers); an
`Answerer` only turns one question into an answer. Keeping that seam a Protocol lets
the real agent loop, a one-shot baseline or a test fake plug in unchanged.

`AgentAnswerer` answers each question with a new tool session and a new `Agent`, so no
state is shared between requests. Its deadline ends `AGENT_DEADLINE_MARGIN_S` before the
server's cut, so the server's timeout answer never replaces an answer the agent could
give. The server's cut counts from the request's arrival (it may wait for a slot), so the
server passes the time left (`time_left_s`); the agent's deadline is counted from that.
An `AgentError` (no answer at all) propagates: the server maps it to its internal-error
answer.

Before the answer leaves, the session names in its text every handle it issued
(`ToolSession.clean_answer`, the `answer_handles` span): a handle such as D3 means nothing
to the reader (tune run T3, 2026-10-17). The prompt says so too (agent.md R8); this is the
safety net. It changes only the answer text, never the document ids, and if it fails
the answer leaves as the model wrote it (logged, and the span marked ERROR).
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from corpusqa import obs
from corpusqa.agent.agent import DEFAULT_LIMITS, FALLBACK, Agent, AgentLimits
from corpusqa.agent.handles import CleanedAnswer
from corpusqa.agent.session import ToolSessionFactory
from corpusqa.config import (
    AGENT_DEADLINE_MARGIN_S,
    AGENT_FIRST_SEARCH,
    ASK_DEADLINE_S,
    PLACEHOLDER_ANSWER,
)
from corpusqa.llm.interfaces import ChatModel

logger = logging.getLogger(__name__)

CITATIONS_SPAN = "citations"
ANSWER_HANDLES_SPAN = "answer_handles"


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """One answer and the dataset_doc_uuids it relies on (CONTRACT.md §3)."""

    answer: str
    document_ids: list[str] = field(default_factory=list)


class Answerer(Protocol):
    """Answers one question. Called concurrently: keep no per-request state on self.

    `time_left_s` is how much of the request's own deadline is left as the answer starts
    (the server's cut minus the time the request waited for a slot); None: the whole
    `ASK_DEADLINE_S`.
    """

    async def answer(self, question: str, time_left_s: float | None = None) -> AnswerResult: ...


class PlaceholderAnswerer:
    """Says plainly that answering is not built yet, and cites nothing."""

    async def answer(self, question: str, time_left_s: float | None = None) -> AnswerResult:
        return AnswerResult(answer=PLACEHOLDER_ANSWER, document_ids=[])


class AnswererUnavailable(RuntimeError):
    """The agent cannot run at all (e.g. no chat model); the message says why."""


class UnavailableAnswerer:
    """Stands in when the agent cannot run (e.g. no GEMINI_API_KEY, so no chat model).

    Every question raises `AnswererUnavailable`: the server logs and traces it as an error
    and gives its internal-error answer, which never claims the records lack the answer.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def answer(self, question: str, time_left_s: float | None = None) -> AnswerResult:
        raise AnswererUnavailable(self.reason)


class AgentAnswerer:
    """Answers with the agent: per question a new session (its tools) and a new Agent.

    `ask_deadline_s` is the server's cut for one /ask (`Settings.ask_deadline_s`). The
    agent gets the time left of it (`time_left_s`, or all of it when not given) minus
    `deadline_margin_s`, counted from the start of `answer`. With no time left after the
    margin, `answer` raises `TimeoutError` at once (the server's timeout answer).
    `catalog` is the field catalog every agent's system message holds (the corpus tools
    learn it once at start-up: `CorpusTools.catalog`). `first_search` turns each agent's
    first search on or off (`AGENT_FIRST_SEARCH`). Holds only what every request shares:
    the model, the factory, the catalog, the limits, the switch and the clock.
    """

    def __init__(
        self,
        model: ChatModel,
        session_factory: ToolSessionFactory,
        *,
        ask_deadline_s: float = ASK_DEADLINE_S,
        deadline_margin_s: float = AGENT_DEADLINE_MARGIN_S,
        limits: AgentLimits = DEFAULT_LIMITS,
        clock: Callable[[], float] = time.monotonic,
        catalog: str = "",
        first_search: bool = AGENT_FIRST_SEARCH,
    ) -> None:
        if not 0 <= deadline_margin_s < ask_deadline_s:
            raise ValueError(
                f"deadline_margin_s ({deadline_margin_s}) must be at least 0 and below "
                f"ask_deadline_s ({ask_deadline_s})"
            )
        self._model = model
        self._session_factory = session_factory
        self._ask_deadline_s = ask_deadline_s
        self._margin_s = deadline_margin_s
        self._limits = limits
        self._clock = clock
        self._catalog = catalog
        self._first_search = first_search

    async def answer(self, question: str, time_left_s: float | None = None) -> AnswerResult:
        started = self._clock()
        left = self._ask_deadline_s if time_left_s is None else time_left_s
        budget = left - self._margin_s
        if budget <= 0:
            raise TimeoutError(
                f"only {left:.1f} s of the request's deadline are left, not more than the "
                f"{self._margin_s:.1f} s margin: no time to answer"
            )
        deadline = started + budget
        session = self._session_factory()
        agent = Agent(
            question,
            self._model,
            session.toolbox,
            deadline=deadline,
            clock=self._clock,
            limits=self._limits,
            catalog=self._catalog,
            first_search=self._first_search,
        )
        result = await agent.run()
        with obs.span(
            CITATIONS_SPAN,
            input={"citations": list(result.citations)},
            metadata={"stop": result.stop},
        ) as span:
            if result.stop == FALLBACK:  # no good finish: the model cited nothing checked
                document_ids: list[str] = []
            else:  # a thread: a session may read the index (SQLite) to resolve them
                document_ids = await asyncio.to_thread(session.resolve_citations, result.citations)
            span.update(
                output={"document_ids": document_ids},
                metadata={"citations": len(result.citations), "document_ids": len(document_ids)},
            )
        with obs.span(ANSWER_HANDLES_SPAN, input={"answer": result.answer}) as span:
            try:
                cleaned = await asyncio.to_thread(session.clean_answer, result.answer, question)
            except Exception as exc:  # a readability net must never cost the answer
                logger.exception("naming the handles in the answer failed; sent as written")
                span.error(exc)
                cleaned = CleanedAnswer(result.answer)
            span.update(
                output={
                    "answer": cleaned.text,
                    "replacements": [
                        {"handle": r.handle, "text": r.text, "count": r.count}
                        for r in cleaned.replacements
                    ],
                },
                metadata={"replaced": cleaned.replaced, "kept": list(cleaned.kept)},
            )
        return AnswerResult(answer=cleaned.text, document_ids=document_ids)


if TYPE_CHECKING:  # pyright checks that the answerers keep the Answerer shape

    def _agent_answerer_is_an_answerer(answerer: AgentAnswerer) -> Answerer:
        return answerer

    def _unavailable_answerer_is_an_answerer(answerer: UnavailableAnswerer) -> Answerer:
        return answerer
