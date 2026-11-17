"""The corpus tools as the model sees them: texts with handles and paging lines, errors,
citations, handles named in the answer text, sessions that share nothing, spans, and agent
runs through the answerer."""

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from corpusqa import obs
from corpusqa.agent.agent import FIRST_SEARCH_CALL_ID
from corpusqa.agent.answerer import ANSWER_HANDLES_SPAN, AgentAnswerer
from corpusqa.agent.corpus_tools import (
    CORPUS_TOOL_NAMES,
    CorpusTools,
    CorpusToolSession,
    corpus_session_factory,
)
from corpusqa.agent.handles import CleanedAnswer, DocName, Handles, Replacement
from corpusqa.agent.session import ToolSession
from corpusqa.agent.tools import FINISH_TOOL_NAME
from corpusqa.llm.errors import EmbeddingError
from corpusqa.llm.fakes import FakeEmbedder, FakeLLM, fake_reply
from corpusqa.llm.interfaces import EmbedRole, ToolCall
from corpusqa.obs.testing import RecordingTracer
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.store.sqlite import SqliteReader
from tests.unit.sdk_wire import embedder_on_wire, scripted
from tests.unit.tool_corpus import (
    N_ORDERS,
    SOURCE,
    Built,
    build_tool_corpus,
    doc_id,
    opened_of,
    state_of,
    waiver_of,
)

ORDERS = range(1, N_ORDERS + 1)
IN_REPAIR = [n for n in ORDERS if state_of(n) == "In Repair"]
HANDLE = re.compile(r"^(D\d+) · ", re.MULTILINE)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_tool_corpus(tmp_path_factory.mktemp("tools"))


@pytest.fixture(scope="module")
def tools(built: Built) -> CorpusTools:
    runtime = built.runtime
    assert runtime.index_path is not None and runtime.chunks is not None
    reader = SqliteReader(runtime.index_path)
    return corpus_session_factory(reader, runtime.chunks, runtime.vectors, runtime.embedder)


def call(session: CorpusToolSession, name: str, **arguments: object) -> str:
    [message] = asyncio.run(session.toolbox.run([ToolCall("c1", name, arguments)]))
    return message.content


def newest_first(numbers: Sequence[int]) -> list[str]:
    return [doc_id(n) for n in sorted(numbers, key=opened_of, reverse=True)]


def shown_ids(session: CorpusToolSession, text: str) -> list[str]:
    """The documents a result listed, by their D# handles, in order."""
    return [session.handles.resolve([h])[0] for h in HANDLE.findall(text)]


def repair_filter(**extra: object) -> dict[str, object]:
    conditions = [{"field": "job_state", "op": "eq", "value": "in repair"}]
    return {"source": SOURCE, "conditions": conditions, **extra}


# ---------------------------------------------------------------- the box
def test_a_session_has_every_corpus_tool_and_finish(tools: CorpusTools) -> None:
    session = tools()
    names = [s.name for s in session.toolbox.schemas()]
    assert names == [*CORPUS_TOOL_NAMES, FINISH_TOOL_NAME]
    as_protocol: ToolSession = session
    assert as_protocol.resolve_citations([]) == []


# ---------------------------------------------------------------- filter
def test_filter_gives_the_exact_count_a_handle_and_pages_reaching_every_row(
    tools: CorpusTools,
) -> None:
    session = tools()
    first = call(session, "filter", **repair_filter())
    assert f"55 of its {N_ORDERS} documents match." in first
    assert "All 55 are saved as R1 (newest first)" in first
    assert '"in repair" -> "In Repair" (55 docs)' in first
    assert f"a value in {N_ORDERS} of {N_ORDERS} documents, empty in 0, absent in 0" in first
    assert "rows 1-50 of 55; next: offset=50" in first
    second = call(session, "filter", **repair_filter(offset=50))
    assert "rows 51-55 of 55 (last page)" in second
    assert "saved as R1" in second  # the same set, the same handle
    rows = shown_ids(session, first) + shown_ids(session, second)
    assert rows == newest_first(IN_REPAIR)  # every row, once, in order
    assert session.resolve_citations(["R1"]) == newest_first(IN_REPAIR)


def test_filter_rows_show_the_values_of_in_range_and_contains(tools: CorpusTools) -> None:
    session = tools()
    conditions = [{"field": "job_waiver", "op": "in", "values": ["granted"]}]
    text = call(session, "filter", source=SOURCE, conditions=conditions)
    assert text.count('job_waiver: "granted"') == sum(1 for n in ORDERS if waiver_of(n))
    missing = call(
        session, "filter", source=SOURCE, conditions=[{"field": "job_waiver", "op": "missing"}]
    )
    empty = sum(1 for n in ORDERS if waiver_of(n) == "")
    assert f"empty in {empty}" in missing and "job_waiver missing" in missing


def test_filter_errors_name_the_real_values_and_fields(tools: CorpusTools) -> None:
    session = tools()
    several = call(
        session, "filter", source=SOURCE,
        conditions=[{"field": "budget_band", "op": "eq", "value": "10K 20K"}],
    )  # fmt: skip
    assert several.startswith("<tool_error>") and '"$10k-$20k" (4 docs)' in several
    field = call(
        session, "filter", source=SOURCE, conditions=[{"field": "job_stat", "op": "present"}]
    )
    assert field.startswith("<tool_error>no field") and '"job_state"' in field
    source = call(session, "filter", source="nowhere", conditions=[])
    assert source.startswith("<tool_error>no source") and f'"{SOURCE}"' in source
    shape = call(session, "filter", source=SOURCE, conditions=[{"field": "job_state", "op": "eq"}])
    assert "takes exactly one value" in shape


def test_numbers_from_the_model_are_read_as_text(tools: CorpusTools) -> None:
    session = tools()
    conditions = [{"field": "job_cost", "op": "range", "from": 100, "to": 150}]
    text = call(session, "filter", source=SOURCE, conditions=conditions)
    assert not text.startswith("<tool_error>") and "job_cost range 100 to 150" in text


# ---------------------------------------------------------------- reads
def test_read_fields_of_a_result_set_pages_every_document(tools: CorpusTools) -> None:
    session = tools()
    call(session, "filter", **repair_filter())
    first = call(session, "read_fields", documents=["R1"], fields=["crew_lead", "job_waiver"])
    assert 'read_fields "crew_lead", "job_waiver" of 55 documents:' in first
    assert "rows 1-50 of 55; next: offset=50" in first
    assert "job_waiver: (absent)" in first and "job_waiver: (empty)" in first
    rest = call(session, "read_fields", documents=["R1"], fields=["crew_lead"], offset=50)
    assert shown_ids(session, first) + shown_ids(session, rest) == newest_first(IN_REPAIR)
    unknown = call(session, "read_fields", documents=["R9"], fields=["crew_lead"])
    assert unknown.startswith("<tool_error>not a handle of this run")


def test_read_document_shows_the_whole_document(tools: CorpusTools) -> None:
    session = tools()
    text = call(session, "lookup_key", key="WRK-1")
    [owner] = [h for h in HANDLE.findall(text) if session.handles.resolve([h]) == [doc_id(1)]]
    doc = call(session, "read_document", document=owner)
    assert doc.startswith(f"{owner} · {SOURCE} · ")
    assert '- job_tags: "hydraulics"' in doc and "text:\nThe " in doc and "new belt" in doc
    a_set = call(session, "read_document", document="R1")
    assert a_set.startswith("<tool_error>") and "reads one" in a_set


def test_lookup_key_lists_owners_then_referrers(tools: CorpusTools) -> None:
    session = tools()
    text = call(session, "lookup_key", key="wrk-1")
    assert "owned by 1 document(s), referred to by 6 other document(s)." in text
    assert "All 7 are saved as R1" in text and "rows 1-7 of 7 (last page)" in text
    assert text.count("owns it") == 1
    assert 'refers to it (blocked_by: "https://tracker.example/browse/WRK-1")' in text
    none = call(session, "lookup_key", key="ZZZ-404")
    assert none.startswith("no document owns or refers to the key")


def test_list_fields_and_field_values_page_with_exact_totals(tools: CorpusTools) -> None:
    session = tools()
    fields = call(session, "list_fields", source=SOURCE, near="state")
    assert f'source "{SOURCE}": {N_ORDERS} documents, created ' in fields
    assert '1 field(s) near "state" by name come first.' in fields
    first_field = fields.split("(last page)\n", 1)[1].splitlines()[0]
    assert first_field.startswith("- job_state: enum, in 80 docs (100%), 3 values; top")
    assert "job_label: text" in fields and "the title field" in fields
    values = call(session, "field_values", source=SOURCE, field="crew_lead", near="ottilie b")
    assert "30 distinct values" in values and "values 1-30 of 30 (last page)" in values
    assert '- "Ottilie Brandvold": 3 docs (also spelled ' in values
    past = call(session, "field_values", source=SOURCE, field="job_state", offset=3)
    assert past.startswith("<tool_error>offset=3 is past the end")


# ---------------------------------------------------------------- search
def test_search_pages_of_eight_and_within_a_result_set(tools: CorpusTools) -> None:
    session = tools()
    text = call(session, "search", query="kiln belt service", source=SOURCE)
    assert "documents 1-8; next: offset=8" in text
    assert len(HANDLE.findall(text)) == 8 and "\n  > " in text
    call(
        session,
        "filter",
        source=SOURCE,
        conditions=[{"field": "job_rush", "op": "eq", "value": "R1"}],
    )
    inside = call(session, "search", query="kiln belt service", within="R1", offset=0)
    allowed = set(session.resolve_citations(["R1"]))
    assert allowed and set(shown_ids(session, inside)) <= allowed


def test_search_page_lines_never_claim_a_total_that_is_not_exact(tools: CorpusTools) -> None:
    session = tools()
    first = call(session, "search", query="kiln belt service", source=SOURCE)
    assert first.startswith('search "kiln belt service" (in source "benchwork"): best first.')
    # The first ranking has not reached every order: more follow, the total is not given.
    assert "\ndocuments 1-8; next: offset=8\n" in first
    last = call(session, "search", query="kiln belt service", source=SOURCE, offset=72)
    assert f"\ndocuments 73-{N_ORDERS} of {N_ORDERS} (last page)\n" in last  # ranked whole
    assert len(HANDLE.findall(last)) == N_ORDERS - 72
    past = call(session, "search", query="kiln belt service", source=SOURCE, offset=N_ORDERS)
    assert past.startswith(f"<tool_error>offset={N_ORDERS} is past the end: there are ")
    rush = [{"field": "job_rush", "op": "eq", "value": "R1"}]
    call(session, "filter", source=SOURCE, conditions=rush)
    inside = call(session, "search", query="kiln belt service", within="R1")
    n_rush = sum(1 for n in ORDERS if n % 3 == 0)
    assert f"\ndocuments 1-8 of {n_rush}; next: offset=8\n" in inside  # a small set: whole


class BrokenEmbedder(FakeEmbedder):
    """Embeds documents at build time, but every query call fails."""

    def embed(self, texts: Sequence[str], role: EmbedRole) -> list[list[float]]:
        if role == "query":
            raise EmbeddingError("the embedding service is down")
        return super().embed(texts, role)


def test_search_falls_back_to_keywords_when_the_query_cannot_be_embedded(built: Built) -> None:
    runtime = built.runtime
    assert runtime.index_path is not None and runtime.chunks is not None
    reader = SqliteReader(runtime.index_path)
    broken = corpus_session_factory(reader, runtime.chunks, runtime.vectors, BrokenEmbedder(16))
    text = call(broken(), "search", query="kiln belt service")
    assert "(keyword ranking only)" in text and HANDLE.findall(text)
    nothing = call(broken(), "search", query="zqxv")  # no word matches, no dense ranker
    assert nothing.splitlines()[2] == "no documents"


def test_search_falls_back_to_keywords_when_the_network_drops(built: Built, tmp_path: Path) -> None:
    """Codex review #5: the real embedder on the real SDK, the connection dropped."""
    runtime = built.runtime
    assert runtime.index_path is not None and runtime.chunks is not None
    seen: list[httpx.Request] = []
    embedder = embedder_on_wire(scripted([httpx.ConnectError], seen), tmp_path, dims=16)
    reader = SqliteReader(runtime.index_path)
    dropped = corpus_session_factory(reader, runtime.chunks, runtime.vectors, embedder)
    text = call(dropped(), "search", query="kiln belt service")
    assert "(keyword ranking only)" in text and HANDLE.findall(text)
    assert len(seen) == 1


# ---------------------------------------------------------------- citations and sessions
def test_handles_name_each_document_and_each_set_once() -> None:
    handles = Handles(frozenset({"a", "b", "c"}))
    assert [handles.doc("b"), handles.doc("a"), handles.doc("b")] == ["D1", "D2", "D1"]
    assert handles.result_set(["c", "a"]) == handles.result_set(["c", "a"]) == "R1"
    assert handles.result_set(["a", "c"]) == "R2"  # another order is another list
    assert [handles.expand(t) for t in ("d2", " [R1] ", "c", "D9", "R0", "x")] == [
        ("a",),
        ("c", "a"),
        ("c",),
        None,
        None,
        None,
    ]
    with pytest.raises(ToolInputError, match='not a handle of this run: "D9", "x"'):
        handles.documents(["D1", "D9", "x"])


def test_citations_expand_handles_keep_known_ids_and_drop_the_rest(tools: CorpusTools) -> None:
    session = tools()
    text = call(session, "filter", **repair_filter())
    [first, second] = HANDLE.findall(text)[:2]
    got = session.resolve_citations([second, "r1", f"[{first}]", "nope", doc_id(80), second])
    [second_id] = session.handles.resolve([second])
    expected = list(dict.fromkeys([second_id, *newest_first(IN_REPAIR), doc_id(80)]))
    assert got == expected  # D#, every id of R1, a known id; unknown dropped; deduped


def test_sessions_share_nothing(tools: CorpusTools) -> None:
    one, two = tools(), tools()
    call(one, "filter", **repair_filter())
    assert one.resolve_citations(["R1"]) and two.resolve_citations(["R1", "D1"]) == []
    text = call(two, "lookup_key", key="WRK-1")
    assert "saved as R1" in text  # numbering starts again in each session


# ---------------------------------------------------------------- handles in the answer text
NAMES = {
    "a": DocName("Kiln relay review", "mailbox", "2025-02-21"),
    "b": DocName("Notes on D3", "chat", "2025-03-01"),  # a title that holds a handle shape
    "c": DocName("Belt order", "tracker", None),  # no usable date
}
KILN = '"Kiln relay review" (mailbox, 2025-02-21)'
KILN_IN_BRACKETS = '"Kiln relay review", mailbox, 2025-02-21'


def named(text: str, question: str = "") -> CleanedAnswer:
    """`text` through a registry that issued D1-D3 (a, b, c) and R1 (a, c)."""
    handles = Handles(frozenset(NAMES))
    assert [handles.doc(d) for d in NAMES] == ["D1", "D2", "D3"]
    assert handles.result_set(["a", "c"]) == "R1"
    return handles.name_in_text(
        text, NAMES.__getitem__, lambda ids: f"the {len(ids)} matching records", question=question
    )


def test_a_doc_handle_in_running_text_is_named_by_title_source_and_date() -> None:
    got = named("Shown in email thread D1, then closed.")
    assert got.text == f"Shown in email thread {KILN}, then closed."
    assert got.replacements == (Replacement("D1", KILN, 1),) and got.replaced == 1
    assert named("See D3.").text == 'See "Belt order" (tracker).'  # no date: none shown


def test_a_result_set_handle_is_named_by_a_description_of_its_set() -> None:
    got = named("All of R1 are open.")
    assert got.text == "All of the 2 matching records are open."
    assert got.replacements == (Replacement("R1", "the 2 matching records", 1),)


def test_a_handle_in_brackets_keeps_one_pair_of_brackets() -> None:
    assert named("It was approved (D1).").text == f"It was approved ({KILN_IN_BRACKETS})."
    assert named("[D3]").text == '["Belt order", tracker]'
    # A handle closing a bracket its sentence opened (tune run T3's "email thread D3").
    assert named("(documented in email thread D1)").text == (
        f"(documented in email thread {KILN_IN_BRACKETS})"
    )


def test_several_handles_are_each_named_and_counted() -> None:
    got = named("D1 and D3 agree (D1, D3; R1).")
    belt, belt_in_brackets = '"Belt order" (tracker)', '"Belt order", tracker'
    assert got.text == (
        f"{KILN} and {belt} agree ({KILN_IN_BRACKETS}; {belt_in_brackets}; the 2 matching records)."
    )
    assert got.replacements == (
        Replacement("D1", KILN, 1),
        Replacement("D3", belt, 1),
        Replacement("D1", KILN_IN_BRACKETS, 1),
        Replacement("D3", belt_in_brackets, 1),
        Replacement("R1", "the 2 matching records", 1),
    )
    assert got.replaced == 5


def test_tokens_that_are_not_handles_of_this_run_are_untouched() -> None:
    text = "D9, R5, R0, D1x, X-D1, D1-2, D1.5, d1, r1 and D01 stay as written."
    assert named(text) == CleanedAnswer(text)
    # In a bracket with a token the run never issued, only the issued handle is named.
    assert named("(D1, D9)").text == f"({KILN}, D9)"


def test_inserted_names_are_not_scanned_again() -> None:
    got = named("D2 then D3.")
    assert got.text == '"Notes on D3" (chat, 2025-03-01) then "Belt order" (tracker).'
    assert [r.handle for r in got.replacements] == ["D2", "D3"]


def test_a_token_the_question_holds_is_kept_as_written() -> None:
    # The asker never saw a handle: a "D1" in the question names something real.
    got = named("The D1 rollout moved; see D2.", question="What changed in the D1 rollout?")
    assert got.text == 'The D1 rollout moved; see "Notes on D3" (chat, 2025-03-01).'
    assert got.kept == ("D1",)


def test_the_session_names_its_handles_from_the_index(tools: CorpusTools) -> None:
    session = tools()
    text = call(session, "filter", **repair_filter())
    newest = HANDLE.findall(text)[0]
    [doc] = session.handles.resolve([newest])
    s = tools.summaries[doc]
    assert s.created is not None
    got = session.clean_answer(
        f"55 orders are in repair (R1); the newest is {newest}.", "How many are in repair?"
    )
    assert got.text == (
        "55 orders are in repair (the 55 matching benchwork records); the newest is "
        f'"{s.display_title}" (benchwork, {s.created[:10]}).'
    )
    # The corpus's own "R1" (a job_rush value), asked about by name, is not this run's R1.
    real = session.clean_answer("12 orders are marked R1.", "How many orders are marked R1?")
    assert real == CleanedAnswer("12 orders are marked R1.", kept=("R1",))
    assert tools().clean_answer("See D1 and R1.", "") == CleanedAnswer("See D1 and R1.")


def test_the_catalog_lists_sources_and_their_common_fields(tools: CorpusTools) -> None:
    catalog = tools.catalog
    assert f'- source "{SOURCE}": {N_ORDERS} documents' in catalog
    assert "  - job_state: enum" in catalog
    assert "job_label" not in catalog and "job_notes" not in catalog
    window = tools.index.date_window
    assert window is not None
    assert catalog.startswith(f"Dates outside the date window {window[0]}-{window[1]} count as no ")
    assert (
        '- source "logbook": 2 documents, created 2024-06-01 to 2024-06-01 '
        "(1 without a usable created date)"
    ) in catalog


def test_date_texts_say_how_many_documents_have_no_usable_date(tools: CorpusTools) -> None:
    session = tools()
    window = tools.index.date_window
    assert window is not None
    outside = f"outside the date window {window[0]}-{window[1]}"
    created = call(session, "filter", source="logbook", conditions=[], created_from="2000")
    assert "1 of its 2 documents match." in created
    assert (
        "- created 2000 to ...; 1 documents of the source have no usable created date (none, "
        f"or only dates {outside}, which count as no date) and were not matched"
    ) in created
    ranged = call(
        session, "filter", source="logbook",
        conditions=[{"field": "stamp_at", "op": "range", "from": "1990", "to": "2999"}],
    )  # fmt: skip
    assert "1 of its 2 documents match." in ranged
    assert f"; 1 of them hold a date {outside} (it counts as no date" in ranged
    values = call(session, "field_values", source="logbook", field="stamp_at")
    assert f"1 of them hold a date {outside}" in values
    searched = call(session, "search", query="high water", source="logbook", created_from="2000")
    assert f"1 documents in scope have no usable created date (none, or only dates {outside}" in (
        searched
    )
    assert len(HANDLE.findall(searched)) == 1
    dated, undated = (call(session, "read_document", document=h) for h in ("lg-0", "lg-1"))
    assert {dated.split(" · ")[2], undated.split(" · ")[2]} == {"2024-06-01T10:00:00Z", "no date"}


# ---------------------------------------------------------------- tracing
def test_each_tool_call_has_its_span_and_the_tool_internals_nest_under_it(
    tools: CorpusTools,
) -> None:
    tracer = RecordingTracer()
    session = tools()
    calls = [
        ToolCall("c1", "filter", repair_filter()),
        ToolCall("c2", "search", {"query": "kiln", "source": SOURCE}),
        ToolCall("c3", "lookup_key", {"key": "WRK-1"}),
        ToolCall("c4", "list_fields", {"source": SOURCE}),
        ToolCall("c5", "field_values", {"source": SOURCE, "field": "job_state"}),
    ]

    async def step() -> None:
        with tracer.trace("t"), obs.span("step"):
            await session.toolbox.run(calls)

    asyncio.run(step())
    inner = {"filter": "filter", "search": "search", "lookup_key": "lookup_key",
             "list_fields": "list_fields", "field_values": "field_values"}  # fmt: skip
    for c in calls:
        [span] = [o for o in tracer.named(f"tool:{c.name}") if o.metadata["tool_call_id"] == c.id]
        assert span.metadata["error"] is None
        [child] = [o for o in tracer.children(span) if o.name == inner[c.name]]
        assert child.input is not None and child.output is not None
    [tool_span] = tracer.named("tool:filter")
    # The calls ran at the same time, so which call numbered its handles first varies.
    handles = tool_span.metadata["handles"]
    assert isinstance(handles, dict) and handles["count"] == 55
    result_set, page = handles["result_set"], handles["page"]
    assert isinstance(result_set, str) and isinstance(page, list)
    assert session.resolve_citations([result_set]) == newest_first(IN_REPAIR)
    assert session.resolve_citations([str(h) for h in page]) == newest_first(IN_REPAIR)[:50]
    [inside] = tracer.named("filter")
    [normalised] = inside.metadata["normalised"]  # type: ignore[misc]
    assert normalised["resolved"][0]["matched"] == [["In Repair", 55]]  # type: ignore[index]


# ---------------------------------------------------------------- one agent run
def test_an_agent_run_filters_and_cites_the_whole_set(tools: CorpusTools) -> None:
    model = FakeLLM(
        [
            fake_reply(tool_calls=[ToolCall("a1", "filter", repair_filter())]),
            fake_reply(
                tool_calls=[
                    ToolCall(
                        "a2",
                        FINISH_TOOL_NAME,
                        {
                            "answer": "55 orders are in repair.",
                            "citations": ["R1"],
                            "not_found": [],
                        },
                    )
                ]
            ),
        ]
    )
    result = asyncio.run(AgentAnswerer(model, tools).answer("How many orders are in repair?"))
    assert result.answer == "55 orders are in repair."
    assert result.document_ids == newest_first(IN_REPAIR)
    [_, second] = model.requests
    # The first search's result comes first, then the filter's; last: the budget note.
    first_search, tool_message = [m for m in second.messages if m.role == "tool"]
    assert (first_search.name, first_search.tool_call_id) == ("search", FIRST_SEARCH_CALL_ID)
    assert tool_message.tool_call_id == "a1"
    assert "55 of its 80 documents match." in tool_message.content


def test_an_agent_run_names_the_handles_its_answer_text_holds(tools: CorpusTools) -> None:
    finish = {"answer": "55 orders are in repair (R1).", "citations": ["R1"], "not_found": []}
    model = FakeLLM(
        [
            fake_reply(tool_calls=[ToolCall("a1", "filter", repair_filter())]),
            fake_reply(tool_calls=[ToolCall("a2", FINISH_TOOL_NAME, finish)]),
        ]
    )
    tracer = RecordingTracer()
    with tracer.trace("ask"):
        result = asyncio.run(AgentAnswerer(model, tools).answer("How many are in repair?"))
    assert result.answer == "55 orders are in repair (the 55 matching benchwork records)."
    assert result.document_ids == newest_first(IN_REPAIR)  # the ids are not touched
    [span] = tracer.named(ANSWER_HANDLES_SPAN)
    assert span.output == {
        "answer": result.answer,
        "replacements": [{"handle": "R1", "text": "the 55 matching benchwork records", "count": 1}],
    }
    assert span.metadata == {"replaced": 1, "kept": []}
