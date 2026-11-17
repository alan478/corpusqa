"""One request's tools, and the state they share (agent-tools plan, step 7).

The answerer makes a new session for every question (`ToolSessionFactory`), so nothing a
tool remembers during a run (e.g. the handles it gave out) is ever shared between
requests. After the run, the session turns the model's citations into dataset_doc_uuids,
and names in the answer text any handle it issued (`clean_answer`): handles mean nothing
to the reader.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

from corpusqa.agent.handles import CleanedAnswer
from corpusqa.agent.tools import ToolBox, finish_tool


class ToolSession(Protocol):
    """One request's tools and the state they share (e.g. handles).

    Never shared between requests.
    """

    @property
    def toolbox(self) -> ToolBox: ...

    def resolve_citations(self, citations: Sequence[str]) -> list[str]:
        """The dataset_doc_uuids behind the model's citations: handles expanded (every id of a
        result-set handle), unknown citations dropped, deduped, in first-seen order."""
        ...

    def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
        """The answer with each handle this session issued, written in its text, replaced
        by what it is (a record's title, source and date; a result set's description).
        Any other text, and every token the question holds, stays as written."""
        ...


type ToolSessionFactory = Callable[[], ToolSession]


class FinishOnlySession:
    """A session whose only tool is `finish`: nothing can be looked up, so nothing resolves."""

    def __init__(self) -> None:
        self._toolbox = ToolBox([finish_tool()])

    @property
    def toolbox(self) -> ToolBox:
        return self._toolbox

    def resolve_citations(self, citations: Sequence[str]) -> list[str]:
        return []

    def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
        return CleanedAnswer(answer)  # it issues no handles
