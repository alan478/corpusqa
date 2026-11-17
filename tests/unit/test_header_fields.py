"""Field values in chunk headers (backlog item 3): which fields the profile picks, how the
store reads their values, how the header renders them, and S6 / the eval using them."""

import sqlite3
from pathlib import Path

import pytest

from corpusqa.config import (
    CHUNK_HEADER_SEPARATOR,
    HEADER_FIELD_KINDS,
    HEADER_FIELD_LIST_ITEMS,
    HEADER_FIELD_MAX_CONTENT_RATIO,
    HEADER_FIELD_MAX_TITLE_RATIO,
    HEADER_FIELD_MIN_PRESENCE,
    HEADER_FIELD_VALUE_CHARS,
    HEADER_FIELDS_MAX_CHARS,
)
from corpusqa.ingest import Runtime
from corpusqa.ingest import chunking as chunking_module
from corpusqa.ingest.chunking import (
    ChunkInput,
    ChunkSpec,
    chunk_input,
    chunk_rows,
    count_header_values,
    make_header,
    render_header_fields,
    select_header_fields,
)
from corpusqa.obs.testing import RecordingTracer
from corpusqa.store import sqlite as store_sqlite
from corpusqa.store.interfaces import DocumentText, FieldProfileRow, FieldText
from corpusqa.store.sqlite import SqliteBuildWriter, SqliteReader
from eval.retrieval import make_chunks
from tests.unit.test_ingest_pipeline import FIXTURE, build

# Fields of the fixture's invented tracker source (tests/fixtures/mini_corpus/tracker/).
PRINTER = "fx-tkt-101"
ASKED = ("tkt_urgency", "tkt_phase", "tkt_tags", "_gen_marker", "tkt_empty_text", "tkt_heading")


def profile(field: str, source: str = "src", **changes: object) -> FieldProfileRow:
    """A field profile row that passes every header-field test unless `changes` say not."""
    values: dict[str, object] = {
        "source": source, "field": field, "field_norm": field.casefold(), "synonym_group": None,
        "n_docs": 10, "presence": 1.0, "n_values": 10, "n_distinct": 3, "list_ratio": 0.0,
        "type_mix": {"str": 10}, "kind": "enum", "kind_confidence": 1.0, "title_ratio": 0.0,
        "content_ratio": 0.0, "noise_ratio": 0.0, "avg_len": 5.0, "max_len": 9,
        "top_values": (("open", 6), ("shut", 4)),
    } | changes  # fmt: skip
    return FieldProfileRow(**values)  # type: ignore[arg-type]


def in_store_order(rows: list[FieldProfileRow]) -> list[FieldProfileRow]:
    return sorted(rows, key=lambda r: (r.source, r.field))


# ---------------------------------------------------------------- which fields
def test_each_threshold_keeps_or_drops_a_field() -> None:
    rows = [
        profile("at_min_presence", presence=HEADER_FIELD_MIN_PRESENCE),
        profile("below_min_presence", presence=HEADER_FIELD_MIN_PRESENCE - 0.01),
        profile("below_title_ratio", title_ratio=HEADER_FIELD_MAX_TITLE_RATIO - 0.01),
        profile("at_title_ratio", title_ratio=HEADER_FIELD_MAX_TITLE_RATIO),
        profile("below_content_ratio", content_ratio=HEADER_FIELD_MAX_CONTENT_RATIO - 0.01),
        profile("at_content_ratio", content_ratio=HEADER_FIELD_MAX_CONTENT_RATIO),
    ]
    assert set(select_header_fields(rows)["src"]) == {
        "at_min_presence", "below_title_ratio", "below_content_ratio",
    }  # fmt: skip


def test_only_kinds_whose_values_mean_something_are_kept() -> None:
    dropped = ("date", "own_id", "reference", "url", "email", "dated_log", "unknown")
    assert not HEADER_FIELD_KINDS & set(dropped)
    rows = [profile(f"k_{kind}", kind=kind) for kind in (*sorted(HEADER_FIELD_KINDS), *dropped)]
    assert set(select_header_fields(rows)["src"]) == {f"k_{kind}" for kind in HEADER_FIELD_KINDS}


def test_fields_are_ordered_by_presence_then_name_per_source() -> None:
    rows = [
        profile("zeta", source="beta", presence=0.9),
        profile("alpha", source="beta", presence=0.6),
        profile("gamma", source="beta", presence=0.9),
        profile("solo", source="alpha", presence=0.7),
        profile("hidden", source="empty", kind="url"),
    ]
    chosen = select_header_fields(rows)
    assert chosen == {"alpha": ("solo",), "beta": ("gamma", "zeta", "alpha")}
    assert list(chosen) == ["alpha", "beta"]  # sources sorted; one with no field left out
    assert select_header_fields(list(reversed(rows))) == chosen  # input order does not matter


# ---------------------------------------------------------------- store reads
@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Runtime:
    runtime, conn = build(FIXTURE, tmp_path_factory.mktemp("index"))
    conn.close()
    return runtime


def reader_of(runtime: Runtime) -> SqliteReader:
    assert runtime.index_path is not None
    return SqliteReader(runtime.index_path)


def test_field_profiles_round_trip_through_the_build_writer(tmp_path: Path) -> None:
    rows = [
        profile("phase", source="a", top_values=(("open", 6), ("shut", 4))),
        profile("owner", source="a", synonym_group=2, type_mix={"str": 8, "null": 2}),
        profile("phase", source="b", top_values=()),
    ]
    writer = SqliteBuildWriter(tmp_path / "corpus.sqlite")
    writer.add_field_profiles(rows, [])
    assert writer.field_profiles() == in_store_order(rows)
    assert SqliteReader(writer.finish()).field_profiles() == in_store_order(rows)


def test_the_fixture_index_has_field_profiles(built: Runtime) -> None:
    rows = reader_of(built).field_profiles()
    assert rows and rows == in_store_order(rows)
    phase = next(r for r in rows if (r.source, r.field) == ("tracker", "tkt_phase"))
    assert ("Waiting_On_Vendor", 1) in phase.top_values
    assert all(isinstance(item, tuple) for item in phase.top_values)


def test_document_texts_bring_the_asked_field_values_in_the_asked_order(built: Runtime) -> None:
    texts = list(reader_of(built).document_texts(header_fields={"tracker": ASKED}))
    printer = next(t for t in texts if t.doc_id == PRINTER)
    # Noise (`_gen_marker`), empty values and the title field are never header values.
    assert printer.fields == (
        FieldText("tkt_urgency", "2"),
        FieldText("tkt_phase", "Waiting_On_Vendor"),
        FieldText("tkt_tags", "hardware"),
        FieldText("tkt_tags", "floor-3"),
    )
    assert all(t.fields == () for t in texts if t.source != "tracker")


def test_without_a_mapping_no_field_values_are_read(built: Runtime) -> None:
    texts = list(reader_of(built).document_texts())
    assert texts and all(t.fields == () for t in texts)
    assert {t.doc_id for t in texts} == built.known_doc_ids


def test_small_pages_and_small_id_batches_read_the_same(
    built: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = {"tracker": ASKED}
    whole = list(reader_of(built).document_texts(header_fields=mapping))
    monkeypatch.setattr(store_sqlite, "CHUNK_READ_PAGE_DOCS", 2)
    monkeypatch.setattr(store_sqlite, "SQL_KEYS_PER_QUERY", 1)
    assert list(reader_of(built).document_texts(header_fields=mapping)) == whole


# ---------------------------------------------------------------- rendering
def one(field: str, *values: str) -> tuple[FieldText, ...]:
    return tuple(FieldText(field, value) for value in values)


def header_of(*fields: FieldText, created: str | None = "2024-06-01") -> str:
    return make_header(ChunkInput("src", "A Title", created, "Body.", fields=fields))


def test_field_values_follow_the_date_as_field_value_parts() -> None:
    fields = (*one("tags", "a"), *one("phase", "open"), *one("tags", "b"))
    # Grouped by field in input order; list items joined by ", ".
    assert header_of(*fields) == "src · A Title · 2024-06-01 · tags: a, b · phase: open"
    assert header_of(*fields, created=None) == "src · A Title · tags: a, b · phase: open"
    assert header_of() == "src · A Title · 2024-06-01"  # no fields: the header as before
    assert render_header_fields(()) == "" and count_header_values(()) == 0


def test_a_list_shows_its_first_items_only() -> None:
    items = [f"t{i}" for i in range(HEADER_FIELD_LIST_ITEMS + 2)]
    fields = one("tags", *items)
    assert render_header_fields(fields) == "tags: " + ", ".join(items[:HEADER_FIELD_LIST_ITEMS])
    assert count_header_values(fields) == HEADER_FIELD_LIST_ITEMS


def test_values_are_one_line_and_cut_and_blank_values_are_dropped() -> None:
    long = "w" * (HEADER_FIELD_VALUE_CHARS - 1) + " tail"
    fields = (*one("note", "line one\n\n  line\ttwo "), *one("long", long), *one("blank", "  \n"))
    rendered = render_header_fields(fields)
    assert rendered == f"note: line one line two · long: {'w' * (HEADER_FIELD_VALUE_CHARS - 1)}"
    assert render_header_fields(one("odd\nname", "x")) == "odd name: x"
    assert count_header_values(fields) == 2


def test_the_fields_part_is_cut_at_a_part_boundary() -> None:
    fields = tuple(FieldText(f"f{i:02d}", "v" * 40) for i in range(20))
    parts = [f"f{i:02d}: {'v' * 40}" for i in range(20)]
    rendered = render_header_fields(fields)
    assert len(rendered) <= HEADER_FIELDS_MAX_CHARS
    kept = rendered.split(CHUNK_HEADER_SEPARATOR)
    assert kept == parts[: len(kept)] and 1 < len(kept) < len(parts)
    # One more whole part would not have fit.
    assert len(CHUNK_HEADER_SEPARATOR.join(parts[: len(kept) + 1])) > HEADER_FIELDS_MAX_CHARS
    assert count_header_values(fields) == len(kept)
    # A part that alone is over the cap is left out whole, never cut.
    assert render_header_fields(one("n" * HEADER_FIELDS_MAX_CHARS, "x")) == ""


# ---------------------------------------------------------------- spec and input
def stored(fields: tuple[FieldText, ...]) -> DocumentText:
    return DocumentText("d1", "src", "A Title", "2024-06-01", "order", "Body text.", (), fields)


def test_header_fields_need_the_header() -> None:
    with pytest.raises(ValueError, match="need the header"):
        ChunkSpec(header=False, fields=True)
    assert ChunkSpec(header=False, fields=False).header_fields is False


def test_the_spec_follows_the_config_switch_unless_told(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", False)
    assert ChunkSpec().header_fields is False  # config.CHUNK_HEADER_FIELDS
    assert ChunkSpec(fields=True).header_fields is True
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", True)
    assert ChunkSpec().header_fields is True
    assert ChunkSpec(fields=False).header_fields is False
    assert ChunkSpec(header=False).header_fields is False  # no header, no fields


def test_chunk_input_copies_field_values_only_when_the_spec_shows_them() -> None:
    doc = stored(one("phase", "open"))
    assert chunk_input(doc, ChunkSpec(fields=False)).fields == ()
    assert chunk_input(doc, ChunkSpec(fields=True)).fields == doc.fields
    [row] = chunk_rows(doc, ChunkSpec("whole", fields=True), first_id=1)
    assert row.header == "src · A Title · 2024-06-01 · phase: open"
    [plain] = chunk_rows(doc, ChunkSpec("whole", fields=False), first_id=1)
    assert plain.header == "src · A Title · 2024-06-01"


# ---------------------------------------------------------------- S6 and the eval
def headers(conn: sqlite3.Connection, doc_id: str | None = None) -> dict[int, str]:
    sql, args = "SELECT chunk_id, header FROM chunks", ()
    if doc_id is not None:
        sql, args = sql + " WHERE doc_id = ?", (doc_id,)
    return dict(conn.execute(sql, args).fetchall())


def test_s6_puts_the_profile_fields_in_every_header_when_switched_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", False)
    off_tracer = RecordingTracer()
    with off_tracer.trace("startup"):
        _, off = build(FIXTURE, tmp_path / "off")
    before = headers(off, PRINTER)
    # Switched off: no values read or shown, and the span says so.
    [s6_off] = off_tracer.named("S6_chunk")
    assert s6_off.metadata["header_fields"] is False
    assert "header_field_sources" not in s6_off.metadata
    assert isinstance(s6_off.output, dict) and s6_off.output["header_field_values"] == 0
    assert not any("tkt_phase:" in header for header in headers(off).values())
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", True)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        _, on = build(FIXTURE, tmp_path / "a")
    _, again = build(FIXTURE, tmp_path / "b")
    after = headers(on, PRINTER)
    assert before and after.keys() == before.keys()
    for chunk_id, header in after.items():
        assert header.startswith(before[chunk_id] + CHUNK_HEADER_SEPARATOR)
        assert "tkt_phase: Waiting_On_Vendor" in header and "tkt_tags: hardware, floor-3" in header
    every = headers(on)
    assert not any("\n" in header for header in every.values())
    assert headers(again) == every  # two builds, the same headers

    [s6] = tracer.named("S6_chunk")
    assert s6.metadata["header_fields"] is True
    sources = s6.metadata["header_field_sources"]
    assert isinstance(sources, dict) and {"tkt_phase", "tkt_tags"} <= set(sources["tracker"])
    assert s6.metadata["header_field_count"] == sum(len(fields) for fields in sources.values())
    assert isinstance(s6.output, dict) and s6.output["header_field_values"] >= 3


def test_the_eval_variant_picks_header_fields_from_the_document_index(built: Runtime) -> None:
    assert built.index_path is not None
    plain = make_chunks(built.index_path, ChunkSpec("recursive", fields=False), {PRINTER})
    with_fields = make_chunks(built.index_path, ChunkSpec("recursive", fields=True), {PRINTER})
    assert [c.text for c in with_fields] == [c.text for c in plain]
    for old, new in zip(plain, with_fields, strict=True):
        assert new.header.startswith(old.header + CHUNK_HEADER_SEPARATOR)
        assert "tkt_phase: Waiting_On_Vendor" in new.header
