"""Last line of defence: shape any answerer output into a body CONTRACT.md accepts.

Upstream layers should already cite well and stay short; this module does not trust
them. A body built here always has a non-empty answer, only known document ids (each
once, in the answerer's order, at most `MAX_DOCUMENT_IDS`) and a size well under the 1 MiB
limit. A cut of the id list is logged and recorded on the current trace (the /ask trace):
how many ids were cited and dropped, and a WARNING status.
"""

import logging
from collections.abc import Iterable

from corpusqa import obs
from corpusqa.config import (
    ANSWER_TRUNCATED_SUFFIX,
    INTERNAL_ERROR_ANSWER,
    LOG_DROPPED_ID_SAMPLE,
    MAX_ANSWER_CHARS,
    MAX_DOCUMENT_IDS,
)
from corpusqa.server.models import AskResponse

logger = logging.getLogger(__name__)


def build_response(
    answer: str, document_ids: Iterable[str], known_doc_ids: frozenset[str], trace_id: str
) -> AskResponse:
    """Return a contract-valid /ask body; blank answers become the internal-error answer.

    `trace_id` only labels this module's log lines; the body never carries it.
    """
    if not answer.strip():
        return AskResponse(answer=INTERNAL_ERROR_ANSWER, document_ids=[])
    return AskResponse(
        answer=_bounded(answer, trace_id),
        document_ids=_cited_ids(document_ids, known_doc_ids, trace_id),
    )


def _bounded(answer: str, trace_id: str) -> str:
    # Lone surrogates cannot be encoded as UTF-8 and would crash JSON rendering (a 500).
    text = answer.encode("utf-8", errors="replace").decode("utf-8")
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    logger.warning("ask trace_id=%s answer truncated from %d chars", trace_id, len(text))
    return text[: MAX_ANSWER_CHARS - len(ANSWER_TRUNCATED_SUFFIX)] + ANSWER_TRUNCATED_SUFFIX


def _cited_ids(document_ids: Iterable[str], known: frozenset[str], trace_id: str) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for doc_id in document_ids:
        # isinstance first: an unhashable item would make the `in` check raise.
        if not isinstance(doc_id, str) or doc_id not in known:
            dropped.append(repr(doc_id))
        elif doc_id not in seen:
            seen.add(doc_id)
            kept.append(doc_id)
    if dropped:
        logger.warning(
            "ask trace_id=%s dropped %d unknown document ids, e.g. %s",
            trace_id,
            len(dropped),
            ", ".join(dropped[:LOG_DROPPED_ID_SAMPLE]),
        )
    if len(kept) > MAX_DOCUMENT_IDS:
        cut = len(kept) - MAX_DOCUMENT_IDS
        problem = (
            f"cut {cut} of {len(kept)} cited document ids: at most {MAX_DOCUMENT_IDS} are "
            "returned (the first ones, in the answerer's order)"
        )
        logger.warning("ask trace_id=%s %s", trace_id, problem)
        trace = obs.current()
        trace.update(
            metadata={
                "document_ids_cited": len(kept),
                "document_ids_cut": cut,
                "document_ids_cap": MAX_DOCUMENT_IDS,
            }
        )
        trace.warning(problem)
        kept = kept[:MAX_DOCUMENT_IDS]
    return kept
