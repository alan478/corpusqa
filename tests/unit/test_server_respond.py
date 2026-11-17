"""The /ask response guard: known ids only, once each, in order; bounded, renderable answer."""

from typing import Any

import pytest
from fastapi.responses import JSONResponse

from corpusqa.config import (
    ANSWER_TRUNCATED_SUFFIX,
    INTERNAL_ERROR_ANSWER,
    MAX_ANSWER_CHARS,
    MAX_DOCUMENT_IDS,
    MAX_RESPONSE_BYTES,
)
from corpusqa.obs.testing import RecordingTracer
from corpusqa.server.respond import build_response

KNOWN = frozenset({"doc-a", "doc-b", "doc-c"})  # invented ids
TRACE = "0" * 32


def rendered_size(answer: str, ids: list[Any], known: frozenset[str] = KNOWN) -> int:
    """Body size exactly as the server renders it."""
    return len(JSONResponse(build_response(answer, ids, known, TRACE).model_dump()).body)


@pytest.mark.parametrize("answer", ["An answer.", "  "], ids=["answer", "blank answer"])
def test_the_body_has_exactly_the_contract_fields(answer: str) -> None:
    # CONTRACT §3 (user decision 2026-10-17): no trace_id or any other extra field;
    # the trace id travels in the X-Trace-Id header instead.
    body = build_response(answer, ["doc-a"], KNOWN, TRACE).model_dump()
    assert set(body) == {"answer", "document_ids"}


def test_ids_keep_first_occurrence_order_and_drop_unknown() -> None:
    ids = ["doc-c", "doc-a", "ghost", "doc-c", "doc-b", "doc-a"]
    assert build_response("ok", ids, KNOWN, TRACE).document_ids == ["doc-c", "doc-a", "doc-b"]


def test_non_string_ids_are_dropped_not_crashing() -> None:
    ids: list[Any] = [None, ["doc-a"], 7, "doc-b"]
    assert build_response("ok", ids, KNOWN, TRACE).document_ids == ["doc-b"]


def test_id_list_is_capped() -> None:
    known = frozenset(f"doc-{i}" for i in range(MAX_DOCUMENT_IDS * 2))
    ids = sorted(known)
    assert build_response("ok", ids, known, TRACE).document_ids == ids[:MAX_DOCUMENT_IDS]


def test_a_cut_of_the_id_list_is_logged_and_recorded_on_the_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    known = frozenset(f"doc-{i:04d}" for i in range(MAX_DOCUMENT_IDS + 7))
    ids = [*sorted(known), "ghost"]  # an unknown id is dropped first, not counted as cut
    tracer = RecordingTracer()
    with tracer.trace("ask"), caplog.at_level("WARNING", logger="corpusqa.server.respond"):
        body = build_response("ok", ids, known, TRACE)
    assert body.document_ids == sorted(known)[:MAX_DOCUMENT_IDS]
    [ask] = tracer.traces("ask")
    assert ask.metadata["document_ids_cited"] == MAX_DOCUMENT_IDS + 7
    assert ask.metadata["document_ids_cut"] == 7
    assert ask.metadata["document_ids_cap"] == MAX_DOCUMENT_IDS
    assert ask.level == "WARNING"
    problem = (
        f"cut 7 of {MAX_DOCUMENT_IDS + 7} cited document ids: at most {MAX_DOCUMENT_IDS} are "
        "returned (the first ones, in the answerer's order)"
    )
    assert ask.status_message == problem
    assert f"ask trace_id={TRACE} {problem}" in caplog.text


def test_no_cut_leaves_the_trace_alone() -> None:
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        build_response("ok", ["doc-a", "doc-b"], KNOWN, TRACE)
    [ask] = tracer.traces("ask")
    assert "document_ids_cut" not in (ask.metadata or {})
    assert ask.level != "WARNING"


@pytest.mark.parametrize("blank", ["", " ", "\n\t "])
def test_blank_answer_becomes_internal_error_without_ids(blank: str) -> None:
    response = build_response(blank, ["doc-a"], KNOWN, TRACE)
    assert response.answer == INTERNAL_ERROR_ANSWER
    assert response.document_ids == []


def test_short_answer_is_unchanged() -> None:
    assert build_response("Short **Markdown**.", [], KNOWN, TRACE).answer == "Short **Markdown**."


def test_long_answer_is_truncated_with_a_marker() -> None:
    answer = build_response("x" * (MAX_ANSWER_CHARS + 1), [], KNOWN, TRACE).answer
    assert len(answer) == MAX_ANSWER_CHARS
    assert answer.endswith(ANSWER_TRUNCATED_SUFFIX)


def test_worst_case_body_stays_under_1_mib() -> None:
    # Control chars escape to 6 bytes each; max-length ids at the id cap.
    known = frozenset(f"dsid_{i:032x}" for i in range(MAX_DOCUMENT_IDS * 2))
    assert rendered_size("\x01" * 5_000_000, sorted(known), known) < MAX_RESPONSE_BYTES


def test_lone_surrogate_does_not_break_rendering() -> None:
    assert rendered_size("bad \ud800 text", ["doc-a"]) > 0
