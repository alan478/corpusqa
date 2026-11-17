"""The retrieval logic of the agent's tools on the tool corpus: value matching, the field
index, the filter, field reads, key lookup and scoped search."""

from collections.abc import Sequence

import pytest

from corpusqa.config import RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT, RRF_K, SEARCH_DEPTH_GROWTH
from corpusqa.obs.testing import RecordingTracer
from corpusqa.retrieval.documents import read_document, read_fields
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.fields import FieldIndex, field_values, list_fields
from corpusqa.retrieval.filtering import Condition, run_filter
from corpusqa.retrieval.hybrid import rrf, to_documents
from corpusqa.retrieval.keys import lookup_key
from corpusqa.retrieval.paging import Page, page_of
from corpusqa.retrieval.search import CorpusSearch, Hit, Scope, snippet
from corpusqa.retrieval.values import day_range, loose, number, resolve_value
from corpusqa.store.interfaces import ValueCount
from corpusqa.store.sqlite import SqliteReader
from tests.unit.tool_corpus import (
    MIXED_DATES_SOURCE,
    N_ORDERS,
    PEOPLE,
    SOURCE,
    Built,
    build_tool_corpus,
    cost_of,
    doc_id,
    lead_of,
    opened_of,
    rush_of,
    state_of,
    waiver_of,
)

ORDERS = range(1, N_ORDERS + 1)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_tool_corpus(tmp_path_factory.mktemp("tools"))


@pytest.fixture(scope="module")
def reader(built: Built) -> SqliteReader:
    assert built.runtime.index_path is not None
    return SqliteReader(built.runtime.index_path)


@pytest.fixture(scope="module")
def index(reader: SqliteReader) -> FieldIndex:
    return FieldIndex.from_reader(reader)


@pytest.fixture(scope="module")
def search(built: Built, reader: SqliteReader) -> CorpusSearch:
    assert built.runtime.chunks is not None
    return CorpusSearch(built.runtime.chunks, reader.document_summaries(), built.runtime.vectors)


def newest_first(numbers: list[int]) -> tuple[str, ...]:
    return tuple(doc_id(n) for n in sorted(numbers, key=opened_of, reverse=True))


def counts(*items: tuple[str, int]) -> list[ValueCount]:
    return [ValueCount(value_norm=v, n_docs=n, spellings=((v, n),)) for v, n in items]


# ---------------------------------------------------------------- values
def test_the_loose_form_folds_case_and_punctuation() -> None:
    assert loose("ENG sre") == loose("eng-sre") == loose("Eng_SRE") == "engsre"
    assert loose("$10k-$20k") == loose("10k-20k") == loose("10K 20K")


def test_an_exact_value_is_used_and_close_spellings_are_named() -> None:
    known = counts(("10k-20k", 6), ("$10k-$20k", 4), ("40k-60k", 4))
    got = resolve_value("10k-20k", known, person=False, where="f")
    assert (got.how, [c.value_norm for c in got.matched]) == ("exact", ["10k-20k"])
    assert [c.value_norm for c in got.left_out] == ["$10k-$20k"]  # named, not merged


def test_one_loose_match_is_used_several_are_all_listed_none_shows_the_nearest() -> None:
    teams = counts(("bay-north", 20), ("bay-south", 20))
    got = resolve_value("BAY north", teams, person=False, where="f")
    assert (got.how, [c.value_norm for c in got.matched]) == ("loose", ["bay-north"])
    bands = counts(("10k-20k", 6), ("$10k-$20k", 4))
    with pytest.raises(ToolInputError) as several:
        resolve_value("10K 20K", bands, person=False, where="f")
    assert '"10k-20k" (6 docs)' in several.value.message
    assert '"$10k-$20k" (4 docs)' in several.value.message
    assert "Nothing was run" in several.value.message
    with pytest.raises(ToolInputError) as none:
        resolve_value("bay-centre", teams, person=False, where="field x")
    assert "no value of field x" in none.value.message
    assert '"bay-north" (20 docs)' in none.value.message  # the nearest, never a substitute


def test_dates_and_numbers_from_the_model() -> None:
    assert day_range("2024-02-29", "d") == ("2024-02-29", "2024-03-01")
    assert day_range("2024-12", "d") == ("2024-12-01", "2025-01-01")
    assert day_range("2024", "d") == ("2024-01-01", "2025-01-01")
    for bad in ("2024-13", "03/05/2024", "soon"):
        with pytest.raises(ToolInputError, match="is not a date"):
            day_range(bad, "d")
    assert (number("1,200"), number("$40"), number("-3.5"), number("40k")) == (
        1200.0,
        40.0,
        -3.5,
        None,
    )


def test_page_of_gives_exact_totals_and_refuses_an_offset_past_the_end() -> None:
    page = page_of(list(range(7)), 5, 5)
    assert (page.items, page.total, page.next_offset) == ((5, 6), 7, None)
    assert page_of(list(range(7)), 0, 5).next_offset == 5
    with pytest.raises(ToolInputError, match="past the end"):
        page_of(list(range(7)), 7, 5)


# ---------------------------------------------------------------- the field index
def test_sources_and_fields_resolve_loosely_or_name_the_nearest(index: FieldIndex) -> None:
    bench = index.source("BenchWork")
    assert bench.name == SOURCE and bench.n_docs == N_ORDERS
    assert index.field(bench, "Job State").field == "job_state"
    assert index.field(bench, "jobstate").field == "job_state"
    with pytest.raises(ToolInputError) as missing:
        index.field(bench, "job_stat")
    assert '"job_state"' in missing.value.message and "list_fields" in missing.value.message
    with pytest.raises(ToolInputError, match="no source"):
        index.source("nowhere")


def test_the_catalog_keeps_common_fields_and_leaves_out_title_body_and_rare_ones(
    index: FieldIndex,
) -> None:
    catalog = {source.name: [r.field for r in rows] for source, rows in index.catalog()}
    bench = catalog[SOURCE]
    assert {"job_state", "job_rush", "crew_lead", "job_tags", "job_cost"} <= set(bench)
    assert "job_label" not in bench  # the title
    assert "job_notes" not in bench  # in the body
    assert "job_waiver" not in bench  # in 20% of the orders


def test_list_fields_pages_every_field_and_puts_near_names_first(index: FieldIndex) -> None:
    every = [r.field for r in index.source(SOURCE).fields]
    seen: list[str] = []
    offset: int | None = 0
    while offset is not None:
        got = list_fields(index, SOURCE, None, offset, 4)
        assert got.page.total == len(every)
        seen += [r.field for r in got.page.items]
        offset = got.page.next_offset
    assert seen == every  # every field reachable, in order, once
    near = list_fields(index, SOURCE, "state", 0, 3)
    assert near.page.items[0].field == "job_state" and near.n_near == 1


def test_field_values_page_every_value_with_counts_and_coverage(
    reader: SqliteReader, index: FieldIndex
) -> None:
    got = field_values(reader, index, SOURCE, "job_state", None, 0, 50)
    assert [(c.shown, c.n_docs) for c in got.page.items] == [
        ("In Repair", 55),
        ("Waiting Parts", 15),
        ("Done", 10),
    ]
    assert got.coverage.with_value == N_ORDERS
    near = field_values(reader, index, SOURCE, "bench_team", "north", 0, 1)
    assert [c.shown for c in near.page.items] == ["bay-north"] and near.page.next_offset == 1


# ---------------------------------------------------------------- the filter
def test_a_loose_value_is_resolved_reported_and_counted_exactly(
    reader: SqliteReader, index: FieldIndex
) -> None:
    result = run_filter(reader, index, SOURCE, [Condition("job_state", "eq", ("in-repair",))])
    assert result.doc_ids == newest_first([n for n in ORDERS if state_of(n) == "In Repair"])
    [applied] = result.conditions
    [resolved] = applied.resolved
    assert (resolved.how, resolved.matched[0].shown) == ("loose", "In Repair")


def test_conditions_are_joined_by_and(reader: SqliteReader, index: FieldIndex) -> None:
    conditions = [
        Condition("job_state", "eq", ("In Repair",)),
        Condition("job_rush", "eq", ("r1",)),
    ]
    result = run_filter(reader, index, SOURCE, conditions)
    expected = [n for n in ORDERS if state_of(n) == "In Repair" and rush_of(n) == "R1"]
    assert result.doc_ids == newest_first(expected)


def test_several_close_values_stop_the_filter_and_in_takes_them_both(
    reader: SqliteReader, index: FieldIndex
) -> None:
    with pytest.raises(ToolInputError, match="matches 2 values"):
        run_filter(reader, index, SOURCE, [Condition("budget_band", "eq", ("10K 20K",))])
    both = Condition("budget_band", "in", ("10k-20k", "$10k-$20k"))
    assert len(run_filter(reader, index, SOURCE, [both]).doc_ids) == 10
    exact = run_filter(reader, index, SOURCE, [Condition("budget_band", "eq", ("10k-20k",))])
    assert len(exact.doc_ids) == 6
    assert [c.shown for c in exact.conditions[0].resolved[0].left_out] == ["$10k-$20k"]


def test_people_match_however_they_are_written(reader: SqliteReader, index: FieldIndex) -> None:
    # The first person leads orders 1, 31 and 61, written three ways.
    expected = newest_first([n for n in ORDERS if (n - 1) % len(PEOPLE) == 0])
    assert len({lead_of(n) for n in (1, 31, 61)}) == 3
    for asked in ("Ottilie Brandvold", "ottilie.brandvold", "OTTILIE BRANDVOLD (lead)"):
        result = run_filter(reader, index, SOURCE, [Condition("crew_lead", "eq", (asked,))])
        assert result.doc_ids == expected, asked


def test_ranges_on_numbers_and_dates_and_eq_on_a_day(
    reader: SqliteReader, index: FieldIndex
) -> None:
    cheap = Condition("job_cost", "range", low="100", high="150")
    got = run_filter(reader, index, SOURCE, [cheap]).doc_ids
    assert got == newest_first([n for n in ORDERS if 100 <= cost_of(n) <= 150])
    opened = opened_of(5).isoformat()
    on_day = run_filter(reader, index, SOURCE, [Condition("job_opened", "eq", (opened,))])
    assert on_day.doc_ids == (doc_id(5),)
    month = Condition("job_opened", "range", low="2024-02", high="2024-02")
    in_feb = [n for n in ORDERS if opened_of(n).month == 2 and opened_of(n).year == 2024]
    assert run_filter(reader, index, SOURCE, [month]).doc_ids == newest_first(in_feb)
    with pytest.raises(ToolInputError, match="holds numbers"):
        run_filter(reader, index, SOURCE, [Condition("job_cost", "range", low="cheap")])


def test_missing_counts_empty_and_absent_and_coverage_says_which(
    reader: SqliteReader, index: FieldIndex
) -> None:
    present = run_filter(reader, index, SOURCE, [Condition("job_waiver", "present")])
    missing = run_filter(reader, index, SOURCE, [Condition("job_waiver", "missing")])
    assert present.doc_ids == newest_first([n for n in ORDERS if waiver_of(n)])
    assert missing.doc_ids == newest_first([n for n in ORDERS if not waiver_of(n)])
    coverage = missing.conditions[0].coverage
    empty = sum(1 for n in ORDERS if waiver_of(n) == "")
    assert (coverage.with_value, coverage.empty) == (len(present.doc_ids), empty)
    assert coverage.absent == N_ORDERS - len(present.doc_ids) - empty


def test_the_title_field_and_contains(reader: SqliteReader, index: FieldIndex) -> None:
    kiln = run_filter(reader, index, SOURCE, [Condition("job_label", "contains", ("KILN",))])
    assert len(kiln.doc_ids) == N_ORDERS // 8 and kiln.conditions[0].resolved[0].how == "contains"
    tags = run_filter(reader, index, SOURCE, [Condition("job_tags", "eq", ("night shift",))])
    assert tags.doc_ids == newest_first([n for n in ORDERS if n % 5 == 0])  # a list item


def test_the_created_range_and_documents_without_a_usable_date(
    reader: SqliteReader, index: FieldIndex
) -> None:
    low, high = opened_of(10).isoformat(), opened_of(19).isoformat()
    got = run_filter(reader, index, SOURCE, [], created_from=low, created_to=high)
    assert got.doc_ids == newest_first(list(range(10, 20))) and got.undated == 0
    # lg-0's 2024 date is used although half of its field's values are outside the window;
    # lg-1's 2128 date counts as no date, and the result counts it.
    logs = run_filter(reader, index, MIXED_DATES_SOURCE, [], created_from="2000")
    assert (logs.doc_ids, logs.undated) == (("lg-0",), 1)


def test_a_date_outside_the_window_matches_no_range_and_is_counted(
    reader: SqliteReader, index: FieldIndex
) -> None:
    wide = Condition("stamp_at", "range", low="1990", high="2999")
    got = run_filter(reader, index, MIXED_DATES_SOURCE, [wide])
    assert got.doc_ids == ("lg-0",) and got.conditions[0].coverage.dates_outside == 1
    on_day = Condition("stamp_at", "eq", ("2128-06-11",))
    assert run_filter(reader, index, MIXED_DATES_SOURCE, [on_day]).doc_ids == ()
    assert index.date_window is not None and index.date_window[1] < 2128


def test_an_op_with_the_wrong_arguments_is_refused(reader: SqliteReader, index: FieldIndex) -> None:
    for condition in (
        Condition("job_state", "eq", ("Done", "In Repair")),
        Condition("job_state", "in"),
        Condition("job_cost", "range"),
        Condition("job_state", "present", ("Done",)),
    ):
        with pytest.raises(ToolInputError, match="takes"):
            run_filter(reader, index, SOURCE, [condition])


def test_the_filter_span_records_what_ran(reader: SqliteReader, index: FieldIndex) -> None:
    tracer = RecordingTracer()
    with tracer.trace("t"):
        run_filter(reader, index, SOURCE, [Condition("job_state", "eq", ("in repair",))])
    [span] = tracer.named("filter")
    assert span.input == {
        "source": SOURCE,
        "conditions": [
            {"field": "job_state", "op": "eq", "values": ["in repair"], "from": None, "to": None}
        ],
        "created_from": None,
        "created_to": None,
    }
    assert isinstance(span.output, dict) and span.output["count"] == 55
    [normalised] = span.metadata["normalised"]  # type: ignore[misc]
    assert normalised["match"] == {
        "kind": "norm_in",
        "values": ["in repair"],
        "low": None,
        "high": None,
    }  # type: ignore[index]


# ---------------------------------------------------------------- reads and keys
def test_read_fields_says_why_a_field_has_no_value(reader: SqliteReader, index: FieldIndex) -> None:
    ids = [doc_id(10), doc_id(11), doc_id(12), "lg-0"]
    got = read_fields(reader, index, ids, ["job_waiver", "job_notes"], 0, 50)
    states = [[(f.state, f.values) for f in row.fields] for row in got.page.items]
    assert states == [
        [("value", ("granted",)), ("in_body", ())],
        [("empty", ()), ("in_body", ())],
        [("absent", ()), ("in_body", ())],
        [("not_in_source", ()), ("not_in_source", ())],
    ]
    with pytest.raises(ToolInputError, match="no field"):
        read_fields(reader, index, ids, ["no_such_thing"], 0, 50)


def test_read_document_is_whole_or_an_error(reader: SqliteReader) -> None:
    doc = read_document(reader, doc_id(3))
    assert "new belt" in doc.body and any(r.field == "job_code" for r in doc.fields)
    with pytest.raises(ToolInputError, match="no document"):
        read_document(reader, "nope")


def test_a_key_finds_its_owner_then_every_document_that_refers_to_it(
    reader: SqliteReader,
) -> None:
    got = lookup_key(reader, "wrk-1")
    assert got.owners == (doc_id(1),)
    assert [r.doc_id for r in got.referrers] == [doc_id(n) for n in range(2, 9) if n != 7]
    assert all(r.resolved for r in got.referrers)
    twelve = lookup_key(reader, "WRK-12")  # its links are not order 1's
    assert (twelve.owners, [r.doc_id for r in twelve.referrers]) == ((doc_id(12),), [doc_id(9)])
    nobody = lookup_key(reader, "WRK-999")
    assert nobody.owners == () and [(r.doc_id, r.resolved) for r in nobody.referrers] == [
        (doc_id(7), False)
    ]


# ---------------------------------------------------------------- search
def test_search_in_a_scope_returns_nothing_outside_it(search: CorpusSearch, built: Built) -> None:
    [vector] = built.embedder.embed(["kiln belt service"], "query")
    within = tuple(doc_id(n) for n in (3, 11, 19, 40))
    for scope, allowed in (
        (Scope(source=SOURCE), {doc_id(n) for n in ORDERS}),
        (Scope(within=within), set(within)),
        (
            Scope(
                source=SOURCE,
                created_from=opened_of(10).isoformat(),
                created_before=opened_of(20).isoformat(),
            ),
            {doc_id(n) for n in range(10, 20)},
        ),
    ):
        result = search.search("kiln belt service", scope, 0, 100, vector)
        found = {hit.doc_id for hit in result.page.items}
        assert found and found <= allowed, scope
        assert result.dense


def test_search_pages_reach_every_ranked_document_once(search: CorpusSearch) -> None:
    seen: list[str] = []
    offset: int | None = 0
    total = None
    while offset is not None:
        result = search.search("belt service order", Scope(source=SOURCE), offset, 8)
        total = result.page.total
        seen += [hit.doc_id for hit in result.page.items]
        offset = result.page.next_offset
    assert total is not None and len(seen) == total == len(set(seen)) > 8


# ---------------------------------------------------------------- deeper pages
QUERY = "kiln belt service"
SHALLOW = 4  # chunks per ranker in the first ranking of these tests (production: 50)


def shallow_search(built: Built, reader: SqliteReader, top: int = SHALLOW) -> CorpusSearch:
    assert built.runtime.chunks is not None
    return CorpusSearch(
        built.runtime.chunks, reader.document_summaries(), built.runtime.vectors, top_chunks=top
    )


def walk(
    search: CorpusSearch, scope: Scope, size: int, vector: Sequence[float]
) -> tuple[list[str], list[Page[Hit]]]:
    """Every page from offset 0, following next_offset to the end."""
    seen: list[str] = []
    pages: list[Page[Hit]] = []
    offset: int | None = 0
    while offset is not None:
        page = search.search(QUERY, scope, offset, size, vector).page
        pages.append(page)
        seen += [hit.doc_id for hit in page.items]
        offset = page.next_offset
    return seen, pages


def test_later_pages_search_deeper_with_no_repeats_and_a_stable_order(
    built: Built, reader: SqliteReader
) -> None:
    [vector] = built.embedder.embed([QUERY], "query")
    search = shallow_search(built, reader)
    scope = Scope(source=SOURCE)
    seen, pages = walk(search, scope, 3, vector)
    # Far past the first ranking's 4 chunks per ranker: every order, each once (dense
    # ranks every chunk in scope once the search is deep enough).
    assert len(seen) == len(set(seen)) == N_ORDERS
    assert not pages[0].exact and pages[-1].exact and pages[-1].total == N_ORDERS
    assert all(p.next_offset is not None for p in pages if not p.exact)  # never a dead end
    again, _ = walk(search, scope, 5, vector)  # another page size: the same list
    assert again == seen
    alone = search.search(QUERY, scope, 30, 3, vector).page  # a page asked for on its own
    assert [hit.doc_id for hit in alone.items] == seen[30:33]


def test_the_list_is_the_first_ranking_then_what_each_deeper_ranking_adds(
    built: Built, reader: SqliteReader
) -> None:
    chunks, vectors = built.runtime.chunks, built.runtime.vectors
    assert chunks is not None and vectors is not None
    [vector] = built.embedder.embed([QUERY], "query")
    chunk_doc = chunks.chunk_doc_ids()
    orders = {doc_id(n) for n in ORDERS}
    allowed = {c for c, d in chunk_doc.items() if d in orders}
    bm25 = [c for c, _ in chunks.bm25(QUERY, len(allowed), allowed)]
    dense = [c for c, _ in vectors.search(vector, len(allowed), allowed)]
    expected: dict[str, None] = {}
    depth = SHALLOW
    while True:  # RRF and weights as configured, on each ranker's best `depth` chunks
        fused = rrf([bm25[:depth], dense[:depth]], RRF_K, (RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT))
        expected.update(dict.fromkeys(to_documents(fused, chunk_doc)))
        if depth >= len(allowed):
            break
        depth *= SEARCH_DEPTH_GROWTH
    seen, _ = walk(shallow_search(built, reader), Scope(source=SOURCE), 8, vector)
    assert seen == list(expected)
    first = to_documents(rrf([bm25[:SHALLOW], dense[:SHALLOW]], RRF_K), chunk_doc)
    assert seen[: len(first)] == list(first)  # page 1 is the first (validated) ranking


def test_within_a_set_the_search_stops_at_the_set_with_its_exact_total(
    built: Built, reader: SqliteReader
) -> None:
    [vector] = built.embedder.embed([QUERY], "query")
    search = shallow_search(built, reader, top=2)
    within = tuple(doc_id(n) for n in (3, 11, 19, 40, 77))
    assert built.runtime.chunks is not None
    n_chunks = sum(1 for d in built.runtime.chunks.chunk_doc_ids().values() if d in within)
    tracer = RecordingTracer()
    with tracer.trace("t"):
        result = search.search(QUERY, Scope(within=within), 0, 8, vector)
    assert {hit.doc_id for hit in result.page.items} == set(within)
    assert (result.page.total, result.page.exact, result.page.next_offset) == (5, True, None)
    [span] = tracer.named("search")
    depth = span.metadata["depth"]
    assert isinstance(depth, int) and depth // SEARCH_DEPTH_GROWTH < n_chunks <= depth
    assert span.metadata["complete"] is True
    depths = [o.input["depth"] for o in tracer.named("bm25") if isinstance(o.input, dict)]
    assert depths == [2 * SEARCH_DEPTH_GROWTH**i for i in range(len(depths))]
    assert depths[-1] == depth and span.metadata["rankings"] == len(depths)
    with pytest.raises(ToolInputError, match="past the end: there are 5 in all"):
        search.search(QUERY, Scope(within=within), 5, 8, vector)


def test_a_page_with_an_open_total_always_has_a_next_offset() -> None:
    page = page_of(list(range(12)), 8, 3, exact=False)
    assert (page.items, page.total, page.exact, page.next_offset) == ((8, 9, 10), 12, False, 11)
    assert page_of(list(range(12)), 0, 3).exact


def test_snippets_show_the_query_words() -> None:
    text = "intro " * 80 + "the kiln belt snapped during the night " + "outro " * 80
    got = snippet(text, ["kiln", "belt"], 60)
    assert "kiln belt" in got and got.startswith("...") and got.endswith("...")
    assert len(got) <= 60 + 6
    assert snippet("short text", ["x"], 60) == "short text"
