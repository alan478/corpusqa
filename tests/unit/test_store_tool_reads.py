"""The store reads behind the agent's tools, on the tool corpus (mini corpus + benchwork)."""

from collections import Counter

import pytest

from corpusqa.store.interfaces import ChunkReader, CorpusReader, FieldMatch
from corpusqa.store.sqlite import SqliteChunkReader, SqliteReader
from corpusqa.store.vectors import VectorIndex, unit_blob
from tests.unit.tool_corpus import (
    MIXED_DATES_SOURCE,
    N_ORDERS,
    SOURCE,
    Built,
    build_tool_corpus,
    cost_of,
    doc_id,
    opened_of,
    state_of,
    tags_of,
    waiver_of,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build_tool_corpus(tmp_path_factory.mktemp("tools"))


@pytest.fixture(scope="module")
def reader(built: Built) -> SqliteReader:
    assert built.runtime.index_path is not None
    return SqliteReader(built.runtime.index_path)


@pytest.fixture(scope="module")
def chunks(built: Built) -> SqliteChunkReader:
    assert built.runtime.index_path is not None
    return SqliteChunkReader(built.runtime.index_path)


def newest_first(numbers: list[int]) -> list[str]:
    """Order ids the way the filter does: created descending (every order has one)."""
    return [doc_id(n) for n in sorted(numbers, key=opened_of, reverse=True)]


def test_the_sqlite_reader_is_a_corpus_reader(reader: SqliteReader) -> None:
    as_protocol: CorpusReader = reader  # pyright checks the shape; this checks it runs
    assert doc_id(1) in as_protocol.doc_ids()


def test_source_dates_count_the_usable_created_dates(reader: SqliteReader) -> None:
    dates = {d.source: d for d in reader.source_dates()}
    bench = dates[SOURCE]
    assert (bench.n_docs, bench.n_dated) == (N_ORDERS, N_ORDERS)
    assert (bench.first, bench.last) == (opened_of(1).isoformat(), opened_of(N_ORDERS).isoformat())
    # Each date on its own: the logbook's 2024 date is used, its 2128 one is no date.
    logbook = dates[MIXED_DATES_SOURCE]
    assert (logbook.n_docs, logbook.n_dated) == (2, 1)
    assert logbook.first == logbook.last == "2024-06-01T10:00:00Z"
    assert list(dates) == sorted(dates)


def test_the_date_window_is_read_from_the_index(reader: SqliteReader) -> None:
    meta = reader.meta()
    window = (int(meta["date_window_min_year"]), int(meta["date_window_max_year"]))
    assert reader.date_window() == window
    assert window[0] <= 2024 <= window[1] < 2128


def test_document_summaries_keep_the_order_given_and_drop_unknown_ids(
    reader: SqliteReader,
) -> None:
    got = reader.document_summaries([doc_id(3), "no-such-id", doc_id(1), doc_id(3)])
    assert [s.doc_id for s in got] == [doc_id(3), doc_id(1)]
    assert got[0].source == SOURCE and got[0].created == opened_of(3).isoformat()
    assert got[0].created_rule == "order"
    assert len(reader.document_summaries()) == len(reader.doc_ids())


def test_a_document_comes_whole_with_its_fields_in_file_order(reader: SqliteReader) -> None:
    doc = reader.document(doc_id(10))
    assert doc is not None
    assert "new belt" in doc.body and doc.title_field == "job_label"
    fields = [row.field for row in doc.fields]
    assert fields[:3] == ["job_label", "job_code", "job_state"]
    assert [row.value for row in doc.fields if row.field == "job_tags"] == [
        "hydraulics",
        "night-shift",
    ]
    assert reader.document("no-such-id") is None


def test_value_counts_count_documents_per_matching_form(reader: SqliteReader) -> None:
    states = {v.shown: v.n_docs for v in reader.value_counts(SOURCE, "job_state")}
    assert states == {"In Repair": 55, "Waiting Parts": 15, "Done": 10}
    leads = {v.value_norm: v for v in reader.value_counts(SOURCE, "crew_lead")}
    first = leads["ottilie brandvold"]  # one person, written three ways
    assert first.n_docs == 3
    assert sorted(s for s, _ in first.spellings) == [
        "Ottilie Brandvold",
        "Ottilie Brandvold (Lead)",
        "ottilie.brandvold",
    ]
    tags = {v.shown: v.n_docs for v in reader.value_counts(SOURCE, "job_tags")}
    expected = Counter(tag for n in range(1, N_ORDERS + 1) for tag in tags_of(n))
    assert tags == dict(expected)  # list items one by one
    assert reader.value_counts(SOURCE, "no_such_field") == []


def test_field_coverage_tells_empty_from_absent_and_body(reader: SqliteReader) -> None:
    waiver = reader.field_coverage(SOURCE, "job_waiver")
    granted = sum(1 for n in range(1, N_ORDERS + 1) if waiver_of(n))
    empty = sum(1 for n in range(1, N_ORDERS + 1) if waiver_of(n) == "")
    assert (waiver.source_docs, waiver.with_value, waiver.empty, waiver.in_body) == (
        N_ORDERS,
        granted,
        empty,
        0,
    )
    assert waiver.absent == N_ORDERS - granted - empty
    notes = reader.field_coverage(SOURCE, "job_notes")  # a content field: text in the body
    assert (notes.with_value, notes.in_body, notes.absent) == (0, N_ORDERS, 0)


def test_filter_matches_and_orders_newest_first(reader: SqliteReader) -> None:
    repair = FieldMatch("job_state", "norm_in", ("in repair",))
    assert reader.filter_doc_ids(SOURCE, [repair]) == newest_first(list(range(1, 56)))
    rush = FieldMatch("job_rush", "value_in", ("R1",))
    both = reader.filter_doc_ids(SOURCE, [repair, rush])  # AND
    assert both == newest_first([n for n in range(1, 56) if n % 3 == 0])


def test_filter_ranges_are_inclusive_for_numbers_and_half_open_for_dates(
    reader: SqliteReader,
) -> None:
    cheap = FieldMatch("job_cost", "num_range", low=100, high=150)
    expected = [n for n in range(1, N_ORDERS + 1) if 100 <= cost_of(n) <= 150]
    assert reader.filter_doc_ids(SOURCE, [cheap]) == newest_first(expected)
    low, high = opened_of(10).isoformat(), opened_of(20).isoformat()
    opened = FieldMatch("job_opened", "iso_range", low=low, high=high)
    assert reader.filter_doc_ids(SOURCE, [opened]) == newest_first(list(range(10, 20)))
    by_created = reader.filter_doc_ids(SOURCE, [], created_from=low, created_before=high)
    assert by_created == newest_first(list(range(10, 20)))


def test_present_and_missing_split_every_document(reader: SqliteReader) -> None:
    present = reader.filter_doc_ids(SOURCE, [FieldMatch("job_waiver", "present")])
    missing = reader.filter_doc_ids(SOURCE, [FieldMatch("job_waiver", "missing")])
    assert sorted(present) == sorted(doc_id(n) for n in range(1, N_ORDERS + 1) if waiver_of(n))
    assert len(present) + len(missing) == N_ORDERS and not set(present) & set(missing)
    in_body = reader.filter_doc_ids(SOURCE, [FieldMatch("job_notes", "present")])
    assert len(in_body) == N_ORDERS  # text in the body counts as present


def test_a_date_outside_the_window_is_matched_by_no_date_range(reader: SqliteReader) -> None:
    # lg-0's date (2024) is inside the window, lg-1's (2128) outside: it counts as no date.
    assert reader.filter_doc_ids(MIXED_DATES_SOURCE, [], created_from="1990-01-01") == ["lg-0"]
    assert reader.filter_doc_ids(MIXED_DATES_SOURCE, [], created_from="2100-01-01") == []
    every_year = FieldMatch("stamp_at", "iso_range", low="1990-01-01", high="3000-01-01")
    assert reader.filter_doc_ids(MIXED_DATES_SOURCE, [every_year]) == ["lg-0"]
    in_2128 = FieldMatch("stamp_at", "iso_range", low="2128-01-01", high="2129-01-01")
    assert reader.filter_doc_ids(MIXED_DATES_SOURCE, [in_2128]) == []
    assert len(reader.filter_doc_ids(MIXED_DATES_SOURCE, [])) == 2
    # The row keeps its value and says it is outside: nothing is lost from the index.
    rows = {r.doc_id: r for r in reader.document_fields(["lg-0", "lg-1"], ["stamp_at"])}
    assert rows["lg-1"].value == "5000000000" and rows["lg-1"].value_iso is not None
    coverage = reader.field_coverage(MIXED_DATES_SOURCE, "stamp_at")
    assert (coverage.with_value, coverage.dates_outside) == (2, 1)
    assert reader.field_coverage(SOURCE, "job_opened").dates_outside == 0


def test_document_fields_by_document_in_the_order_given(reader: SqliteReader) -> None:
    rows = reader.document_fields([doc_id(2), doc_id(1)], ["job_state", "job_tags"])
    assert [(r.doc_id, r.field, r.value) for r in rows] == [
        (doc_id(2), "job_state", state_of(2)),
        (doc_id(2), "job_tags", "electrics"),
        (doc_id(1), "job_state", state_of(1)),
        (doc_id(1), "job_tags", "hydraulics"),
    ]


def test_key_owners_and_the_links_that_may_refer_to_a_key(reader: SqliteReader) -> None:
    owners = reader.key_owners("wrk-1")
    assert [(o.doc_id, o.kind) for o in owners] == [(doc_id(1), "field:job_code")]
    links = reader.links_to("wrk-1", [doc_id(1)])
    referrers = {(link.doc_id, link.raw) for link in links}
    assert {(doc_id(n), "WRK-1") for n in range(2, 7)} <= referrers
    assert (doc_id(8), "https://tracker.example/browse/WRK-1") in referrers
    unresolved = reader.links_to("wrk-999", [])
    assert [(link.doc_id, link.target_doc_id) for link in unresolved] == [(doc_id(7), None)]


def test_bm25_in_a_set_ranks_only_that_set(chunks: SqliteChunkReader, reader: SqliteReader) -> None:
    as_protocol: ChunkReader = chunks
    chunk_doc = as_protocol.chunk_doc_ids()
    everything = as_protocol.bm25("belt service order", 5)
    assert everything
    allowed = {c for c, d in chunk_doc.items() if d in {doc_id(77), doc_id(78)}}
    inside = as_protocol.bm25("belt service order", 5, allowed)
    assert {chunk_doc[c] for c, _ in inside} == {doc_id(77), doc_id(78)}
    assert all(c in allowed for c, _ in inside)
    assert as_protocol.bm25("belt", 5, set()) == []
    [text] = as_protocol.chunk_texts([inside[0][0]])
    assert text.chunk_id == inside[0][0] and "belt" in text.text
    assert as_protocol.chunk_texts([-1]) == []


def test_vector_search_in_a_set_scores_only_that_set() -> None:
    vectors = VectorIndex([1, 2, 3], [_unit(1, 0), _unit(0.9, 0.1), _unit(0, 1)], dims=2)
    assert [c for c, _ in vectors.search([1.0, 0.0], 2)] == [1, 2]
    assert [c for c, _ in vectors.search([1.0, 0.0], 2, allowed={2, 3})] == [2, 3]
    assert vectors.search([1.0, 0.0], 2, allowed=set()) == []
    assert vectors.search([1.0, 0.0], 2, allowed={9}) == []


def test_vector_ties_go_to_the_lower_chunk_id_so_a_short_list_starts_a_long_one() -> None:
    same = _unit(1, 1)
    ids = [7, 3, 5, 1, 9]
    vectors = VectorIndex(ids, [same, _unit(1, 0), same, _unit(0, 1), same], dims=2)
    every = [c for c, _ in vectors.search([1.0, 1.0], len(ids))]
    assert every == [5, 7, 9, 1, 3]  # equal scores by chunk id
    for k in range(1, len(ids) + 1):
        assert [c for c, _ in vectors.search([1.0, 1.0], k)] == every[:k], k
    assert [c for c, _ in vectors.search([1.0, 1.0], 2, allowed={9, 7, 3})] == [7, 9]


def test_bm25_ties_go_to_the_lower_chunk_id_so_a_short_list_starts_a_long_one(
    chunks: SqliteChunkReader,
) -> None:
    every = [c for c, _ in chunks.bm25("belt service", 10_000)]
    for limit in (1, 2, 5, 13, 50):
        assert [c for c, _ in chunks.bm25("belt service", limit)] == every[:limit], limit
    allowed = set(every[::2])
    in_set = [c for c, _ in chunks.bm25("belt service", 10_000, allowed)]
    assert [c for c, _ in chunks.bm25("belt service", 7, allowed)] == in_set[:7]


def _unit(x: float, y: float) -> bytes:
    return unit_blob([x, y], 2)
