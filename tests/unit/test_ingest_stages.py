"""Ingest stages S1-S5, one behaviour per test, on the invented mini corpus."""

import functools
from collections import Counter
from pathlib import Path

import pytest

from corpusqa.config import DATE_WINDOW_MIN_VALUES
from corpusqa.ingest import clean, core, discover, fields
from corpusqa.ingest.clean import clean_body, clean_text
from corpusqa.ingest.core import build_core
from corpusqa.ingest.dates import (
    fixed_window,
    is_ambiguous_date,
    learn_window,
    outside_counts,
    parse_date,
    pick_dates,
)
from corpusqa.ingest.discover import discover as walk
from corpusqa.ingest.fields import build_fields, normalise
from corpusqa.ingest.models import CoreDoc, FileRef, ParsedDoc, Unit
from corpusqa.ingest.parse import parse
from corpusqa.store.interfaces import QuarantineRow

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
YEAR = 2026
WINDOW = fixed_window(YEAR)


@functools.cache
def _files() -> dict[str, FileRef]:
    return {f.rel_path: f for f in walk(FIXTURE).files}


def parsed(rel_path: str) -> ParsedDoc:
    result = parse(_files()[rel_path])
    assert isinstance(result, ParsedDoc), result
    return result


def cored(rel_path: str) -> CoreDoc:
    return build_core(parsed(rel_path))


def body_of(rel_path: str) -> str:
    return clean_body(cored(rel_path)).body


# ---------------------------------------------------------------- S1 discover
def test_discover_lists_files_sorted_with_sources_and_skips() -> None:
    found = walk(FIXTURE)
    paths = [f.rel_path for f in found.files]
    assert paths == sorted(paths) and len(paths) == 27
    assert "tracker/alpha/beta/gamma/TKT-103.json" in paths  # depth 4
    root = _files()["loose-note.json"]
    assert (root.source, root.flags) == ("_root", ("no_source_folder",))
    assert _files()["tracker/alpha/TKT-102.json"].source == "tracker"
    assert found.skipped == {"hidden": 2, "not_json": 1}  # .trash/, .draft.json, readme.txt
    assert found.quarantined == ()


def test_discover_follows_links_only_inside_the_corpus_and_caps_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, outside = tmp_path / "corpus", tmp_path / "outside.json"
    (corpus / "src").mkdir(parents=True)
    (corpus / "src" / "real.json").write_text("{}", encoding="utf-8")
    (corpus / "src" / "big.json").write_text("x" * 50, encoding="utf-8")
    outside.write_text("{}", encoding="utf-8")
    (corpus / "src" / "inside-link.json").symlink_to(corpus / "src" / "real.json")
    (corpus / "src" / "outside-link.json").symlink_to(outside)
    (corpus / "src" / "broken-link.json").symlink_to(corpus / "src" / "gone.json")
    (corpus / "src" / "loop").symlink_to(corpus / "src")
    monkeypatch.setattr(discover, "MAX_DOC_BYTES", 10)
    found = walk(corpus)
    assert [f.rel_path for f in found.files] == ["src/inside-link.json", "src/real.json"]
    reasons = {q.rel_path: q.reason for q in found.quarantined}
    assert reasons == {"src/big.json": "too_large", "src/broken-link.json": "unreadable"}
    assert found.skipped == {"folder_seen_twice": 1, "symlink_outside_corpus": 1}


@pytest.mark.parametrize("reverse", [False, True])
def test_discover_prefers_real_folders_over_links_whatever_the_listing_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reverse: bool
) -> None:
    # B5 review finding 9: a link to a folder inside the corpus won or lost against the
    # real folder depending on the order the file system listed them.
    corpus = tmp_path / "corpus"
    (corpus / "zsrc").mkdir(parents=True)
    (corpus / "zsrc" / "note.json").write_text("{}", encoding="utf-8")
    (corpus / "alias").symlink_to(corpus / "zsrc")  # sorts before the real folder
    listed = Path.iterdir

    def iterdir(self: Path):  # type: ignore[no-untyped-def]
        return iter(sorted(listed(self), reverse=reverse))

    monkeypatch.setattr(Path, "iterdir", iterdir)
    found = walk(corpus)
    assert [(f.rel_path, f.source) for f in found.files] == [("zsrc/note.json", "zsrc")]
    assert found.skipped == {"folder_seen_twice": 1}


# ---------------------------------------------------------------- S2 parse
@pytest.mark.parametrize(
    ("rel_path", "reason", "detail"),
    [
        ("crm/empty.json", "json_invalid", "empty file"),
        ("crm/broken.json", "json_invalid", "Expecting"),
        ("crm/listing.json", "not_object", "list"),
        ("wiki/no-id.json", "no_uuid", "missing or blank"),
        ("wiki/blank-id.json", "no_uuid", "missing or blank"),
    ],
)
def test_parse_sets_bad_files_aside_with_a_reason(rel_path: str, reason: str, detail: str) -> None:
    result = parse(_files()[rel_path])
    assert isinstance(result, QuarantineRow)
    assert result.reason == reason and detail in (result.detail or "")


def test_parse_accepts_any_id_format_and_flags_repairs() -> None:
    doc = parsed("crm/bom-account.json")  # BOM, padded id, duplicate key
    assert doc.doc_id == "acct:77"
    assert doc.obj["acct_label"] == "Tidewater Looms account"  # last value wins
    assert doc.flags == ("duplicate_key:acct_label", "uuid_trimmed")
    bad = parsed("crm/bad-bytes.json")
    assert bad.flags == ("bad_encoding", "nul_bytes")
    assert "�" in bad.obj["acct_summary"] and "\x00" not in bad.obj["acct_summary"]


def test_parse_reads_damaged_contract_keys_leniently(tmp_path: Path) -> None:
    nested = parsed("crm/nested.json")  # content list is a bare string; nested values
    assert nested.content_fields == ("acct_summary",)
    assert nested.flags == ("bad_content_field_names", "nested_value")
    assert parsed("wiki/untitled-page.json").flags == ("no_title_field",)
    path = tmp_path / "x.json"
    path.write_text('{"dataset_doc_uuid": "a", "content_field_names": ["b", 3, "b", "c"]}', "utf-8")
    doc = parse(FileRef("x.json", path, "_root", 1, 1))
    assert isinstance(doc, ParsedDoc)
    assert doc.content_fields == ("b", "c")
    assert doc.flags == ("no_title_field", "bad_content_field_names")


def test_parse_replaces_lone_surrogates_and_sets_aside_an_id_with_one(tmp_path: Path) -> None:
    # B5 review finding 1: JSON escapes can decode to a lone surrogate, which SQLite cannot
    # store (UTF-8). Escaped pairs (an emoji) are fine and must stay as they are.
    path = tmp_path / "x.json"
    path.write_text(
        '{"dataset_doc_uuid": "a", "content_field_names": ["no\\ud800te"],'
        ' "no\\ud800te": ["ok \\ud83d\\ude00", "bad \\udc00 end"], "plain": "x"}',
        encoding="utf-8",
    )
    doc = parse(FileRef("x.json", path, "_root", 1, 1))
    assert isinstance(doc, ParsedDoc)
    assert "bad_unicode" in doc.flags
    assert doc.obj["no�te"] == ["ok \U0001f600", "bad � end"]
    assert doc.content_fields == ("no�te",)
    assert doc.obj["plain"] == "x"
    path.write_text('{"dataset_doc_uuid": "a\\udfffb"}', encoding="utf-8")
    bad = parse(FileRef("x.json", path, "_root", 1, 1))
    assert isinstance(bad, QuarantineRow)
    assert bad.reason == "no_uuid" and "invalid Unicode" in (bad.detail or "")


# ---------------------------------------------------------------- S3 core: titles
@pytest.mark.parametrize(
    ("rel_path", "title", "display", "weak"),
    [
        ("tracker/alpha/beta/gamma/TKT-103.json", "Desk move / east wing", None, False),
        ("tracker/alpha/beta/TKT-104.json", "4471", "4471 — TKT 104", True),  # (c) digits
        ("calls/standup.json", "standup.json", "standup.json — standup", True),  # (b) file
        ("calls/q3-review.json", "Q3", "Q3 — q3 review", True),  # (d) short
        ("wiki/untitled-page.json", "untitled page", None, False),  # fallback: stem
    ],
)
def test_title_chain_and_weak_titles(
    rel_path: str, title: str, display: str | None, weak: bool
) -> None:
    doc = cored(rel_path)
    assert (doc.title, doc.display_title, doc.title_is_weak) == (title, display or title, weak)
    assert ("weak_title" in doc.flags) == weak
    fallback = rel_path == "wiki/untitled-page.json"
    assert ("title_fallback" in doc.flags) == fallback and (doc.title_field is None) == fallback


# ---------------------------------------------------------------- S3 core: body units
def test_units_follow_the_content_list_one_per_list_item() -> None:
    doc = cored("tracker/TKT-101-printer-jam.json")
    assert [u.label for u in doc.units] == ["tkt_story", "tkt_followups", "tkt_followups"]
    assert doc.body_keys == {"tkt_story", "tkt_followups"}  # the blank item is dropped
    missing = cored("tracker/alpha/beta/TKT-104.json")
    assert "content_field_missing:tkt_missing_part" in missing.flags


def test_numbered_family_reads_in_order_with_sibling_headers() -> None:
    doc = cored("mailbox/2024/03/thread-quarterly.json")  # only ltr_1_text is listed
    assert [u.label for u in doc.units] == ["ltr_1_text", "ltr_2_text"]
    assert doc.units[1].part_headers == (
        "author: Piet Calloway <piet@example.org>",
        "stamp: Wed, 06 Mar 2024 11:40:00 +0000",
    )
    assert "family_completed" in doc.flags
    assert doc.body_keys == {"ltr_1_text", "ltr_2_text"}  # siblings stay field rows
    chat = cored("chatter/1700000000-room-sync.json")  # bare key + _2 + _3
    assert [u.label for u in chat.units][:3] == ["said", "said_2", "said_3"]


def test_unlisted_prose_is_promoted_but_not_links_short_text_or_copies() -> None:
    call = cored("calls/kickoff-call.json")
    assert [u.label for u in call.units] == [
        "call_name", "call_recap", "call_actions", "call_actions", "call_followup_plan",
    ]  # fmt: skip
    assert call.flags == ("promoted:call_followup_plan",)  # links and short text stay out
    assert "promoted:," in cored("chatter/1700000000-room-sync.json").flags  # odd key, as given
    dupe = cored("chatter/room-dupe.json")
    assert dupe.flags == ("dup_field_skipped:said_copy",) and dupe.body_keys == {"said"}


def test_promotion_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "PROMOTE_PROSE_FIELDS", False)
    call = cored("calls/kickoff-call.json")
    assert "call_followup_plan" not in call.body_keys and call.flags == ()


# ---------------------------------------------------------------- S4 fields
def test_field_rows_one_per_value_typed_from_the_value() -> None:
    doc = parsed("tracker/TKT-101-printer-jam.json")
    result = build_fields(doc, build_core(doc), WINDOW)
    rows = {(r.field, r.seq): r for r in result.rows}
    assert not {"tkt_story", "tkt_followups"} & {r.field for r in result.rows}  # body
    assert rows[("tkt_heading", 0)].role == "title"
    assert [rows[("tkt_tags", i)].value for i in (0, 1)] == ["hardware", "floor-3"]
    types = {f: rows[(f, 0)].value_type for f in ("tkt_urgency", "tkt_blocked")}
    assert types == {"tkt_urgency": "int", "tkt_blocked": "bool"}
    assert (rows[("tkt_estimate_hours", 0)].value_num, rows[("tkt_owner_note", 0)].value) == (
        1.5,
        None,
    )
    assert rows[("tkt_empty_text", 0)].value_type == rows[("tkt_empty_list", 0)].value_type
    assert rows[("tkt_empty_list", 0)].value_type == "empty"  # present but empty, not absent
    noise = {r.field for r in result.rows if r.is_noise}
    assert noise == {"_gen_marker", "tkt_source_file", "tkt_owner_note"}
    assert len(result.rows) == 16


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ('"Open"', "open"),
        ("Waiting_On_Vendor", "waiting on vendor"),
        ("Yes", "true"),
        ("no", "false"),
        ("\uff21\uff23\uff2d-\uff17", "acm-7"),
        ("  two   words. ", "two words"),
        ("x" * 300, "x" * 256),
    ],
)
def test_value_norm(raw: str, norm: str) -> None:
    assert normalise(raw) == norm


def test_value_num_and_nested_values() -> None:
    doc = parsed("crm/acct-copy-a.json")
    rows = {r.field: r for r in build_fields(doc, build_core(doc), WINDOW).rows}
    assert rows["acct_value"].value_num == 12500.0
    nested = parsed("crm/nested.json")
    types = [(r.field, r.value_type) for r in build_fields(nested, build_core(nested), WINDOW).rows]
    assert ("acct_address", "json") in types and ("acct_contacts", "json") in types


def test_keys_are_kept_exactly_and_folded_only_for_grouping() -> None:
    doc = parsed("calls/kickoff-call.json")
    rows = build_fields(doc, build_core(doc), WINDOW).rows
    cyrillic = next(r for r in rows if r.field == "n\u043etes")
    assert cyrillic.field_norm == "n\u043etes"
    assert fields.fold_key(" Due-Date Field ") == "due_date_field"


# ---------------------------------------------------------------- S4 dates
@pytest.mark.parametrize(
    ("raw", "iso", "precision"),
    [
        ("2024-06-01", "2024-06-01", "date"),
        ("2024-06-02T10:30:00Z", "2024-06-02T10:30:00Z", "datetime"),
        ("2024-06-03 10:30", "2024-06-03T10:30:00Z", "datetime"),
        ("2024-06-04T10:30:00+02:00", "2024-06-04T08:30:00Z", "datetime"),
        ("2024-06", "2024-06-01", "month"),
        ("2024-W23", "2024-06-03", "week"),
        ("2024-Q2", "2024-04-01", "quarter"),
        ("1717236000", "2024-06-01T10:00:00Z", "datetime"),
        ("1717236000.5", "2024-06-01T10:00:00Z", "datetime"),
        ("1717236000000", "2024-06-01T10:00:00Z", "datetime"),
        ("Mon, 03 Jun 2024 09:00:00 +0000", "2024-06-03T09:00:00Z", "datetime"),
        ("June 5, 2024", "2024-06-05", "date"),
        ("6 Jun 2024", "2024-06-06", "date"),
        ("06/25/2024", "2024-06-25", "date"),
        ("25/06/2024", "2024-06-25", "date"),
    ],
)
def test_parse_date_formats(raw: str, iso: str, precision: str) -> None:
    parsed_date = parse_date(raw)
    assert parsed_date is not None
    assert (parsed_date.iso, parsed_date.precision) == (iso, precision)


@pytest.mark.parametrize("raw", ["2024-13-45", "06/07/2024", "2024", "hello", "", "Smarch 3, 2024"])
def test_parse_date_rejects_non_dates_and_ambiguous_ones(raw: str) -> None:
    assert parse_date(raw) is None


def test_out_of_range_dates_are_kept_but_never_used() -> None:
    doc = parsed("calls/date-zoo.json")
    result = build_fields(doc, build_core(doc), WINDOW)
    rows = {r.field: r for r in result.rows}
    assert (rows["d_far_future"].value_iso, rows["d_far_future"].date_in_range) == (
        "2999-01-01",
        False,
    )
    assert rows["d_too_old"].date_in_range is False
    assert rows["d_not_a_date"].value_iso is None
    dates = result.dates
    assert (dates.created, dates.created_field) == ("2024-04-01", "d_quarter")
    assert (dates.updated, dates.updated_field) == ("2024-06-25", "d_slash_mdy")
    assert "ambiguous_date" in result.flags and is_ambiguous_date("06/07/2024")


def test_created_and_updated_by_order_name_hint_and_file_stem() -> None:
    def dates_of(rel_path: str) -> tuple[object, ...]:
        doc = parsed(rel_path)
        d = build_fields(doc, build_core(doc), WINDOW).dates
        return (
            d.created,
            d.created_field,
            d.created_rule,
            d.updated,
            d.updated_field,
            d.updated_rule,
        )

    # tkt_logged_at is earlier, but "opened" names the creation; tkt_due_by is planned.
    assert dates_of("tracker/TKT-101-printer-jam.json") == (
        "2024-01-05", "tkt_opened_on", "name_hint", "2024-01-09", "tkt_touched_on", "order",
    )  # fmt: skip
    assert dates_of("mailbox/20240412-standup-notes.json") == (
        "2024-04-12", None, "file_stem", "2024-04-12", None, "file_stem",
    )  # fmt: skip
    assert dates_of("tracker/alpha/beta/gamma/TKT-103.json") == (None,) * 6


def test_votes_that_contradict_fall_back_to_order_and_equal_dates_keep_order() -> None:
    contradicting = pick_dates([("ended", "2024-01-01"), ("created_at", "2024-05-01"),
                                ("last_edit", "2024-03-01")], "x", WINDOW)  # fmt: skip
    assert (contradicting.created_field, contradicting.updated_field) == ("ended", "created_at")
    same = pick_dates([("thread", "2024-01-01"), ("first_seen", "2024-01-01")], "x", WINDOW)
    assert (same.created_field, same.created_rule) == ("first_seen", "order")


def test_short_role_words_must_be_whole_words() -> None:
    # "sla" (planned) must not match "slash"; "eta" must not match inside "metadata".
    kept = pick_dates([("d_slash_mdy", "2024-01-01"), ("metadata_ts", "2024-02-01")], "x", WINDOW)
    assert (kept.created_field, kept.updated_field) == ("d_slash_mdy", "metadata_ts")
    planned = pick_dates([("sla_due_at", "2024-09-01"), ("seen", "2024-01-01")], "x", WINDOW)
    assert planned.updated_field == "seen"


def test_each_date_value_records_how_it_was_written() -> None:
    doc = parsed("calls/date-zoo.json")
    rows = {r.field: r for r in build_fields(doc, build_core(doc), WINDOW).rows}
    written = {
        name: rows[name].date_format
        for name in ("d_plain", "d_week", "d_epoch_ms", "d_rfc", "d_long", "d_not_a_date")
    }
    assert written == {"d_plain": "iso", "d_week": "iso", "d_epoch_ms": "epoch",
                       "d_rfc": "text", "d_long": "text", "d_not_a_date": None}  # fmt: skip


def test_learned_window_is_the_1st_to_99th_percentile_of_years_plus_a_margin() -> None:
    years = Counter({1850: 1, 2024: 100, 2025: 95, 2026: 3, 2999: 1})
    window = learn_window(years, YEAR)
    assert (window.min_year, window.max_year, window.rule, window.iso_values) == (
        2023, 2027, "learned", 200,
    )  # fmt: skip
    assert window.contains(2023) and window.contains(2027) and not window.contains(2028)


def test_too_few_iso_values_fall_back_to_the_fixed_window() -> None:
    window = learn_window(Counter({2024: DATE_WINDOW_MIN_VALUES - 1}), YEAR)
    assert window == fixed_window(YEAR, DATE_WINDOW_MIN_VALUES - 1)
    assert (window.min_year, window.max_year, window.rule) == (1990, YEAR + 5, "fixed")


def test_dates_outside_the_window_are_counted_per_field() -> None:
    window = learn_window(Counter({2024: 50, 2025: 50}), YEAR)  # 2023..2026
    field_years = {
        "pulse": Counter({2025: 3, 2150: 7}),  # 70 % outside
        "edge": Counter({2025: 8, 2290: 2}),
        "clean": Counter({2024: 10}),  # none outside: not listed
    }
    counts = outside_counts(field_years, window)
    assert {key: (c.values, c.outside) for key, c in counts.items()} == {
        "pulse": (10, 7),
        "edge": (10, 2),
    }


def test_each_date_is_judged_on_its_own() -> None:
    # User, 2026-10-16: no field is trusted or distrusted as a whole. "pulse" may have most
    # of its values outside the window; its in-window dates count like any other.
    window = learn_window(Counter({2024: 50, 2025: 50}), YEAR)  # 2023..2026
    both = pick_dates([("pulse", "2025-03-01"), ("logged", "2025-01-01")], "x", window)
    assert (both.created_field, both.updated_field, both.created_rule, both.updated_rule) == (
        "logged", "pulse", "order", "order",
    )  # fmt: skip
    alone = pick_dates([("pulse", "2025-03-01"), ("pulse_end", "2025-04-01")], "x", window)
    assert (alone.created, alone.created_rule, alone.updated, alone.updated_rule) == (
        "2025-03-01", "order", "2025-04-01", "order",
    )  # fmt: skip
    # A planned date never counts; a value outside the window counts as no date.
    planned = pick_dates([("due_on", "2025-06-01"), ("pulse", "2025-02-01")], "x", window)
    assert (planned.created_field, planned.created_rule) == ("pulse", "order")
    one_out = pick_dates([("pulse", "2150-01-01"), ("logged", "2025-05-01")], "x", window)
    assert (one_out.created_field, one_out.updated_field) == ("logged", "logged")
    outside = pick_dates([("pulse", "2150-01-01"), ("logged", "2029-01-01")], "x", window)
    assert outside.created is None and outside.created_rule is None
    from_stem = pick_dates([("pulse", "2150-01-01")], "20250102-notes", window)
    assert (from_stem.created, from_stem.created_rule) == ("2025-01-02", "file_stem")


# ---------------------------------------------------------------- S5 clean
def test_clean_decodes_escapes_only_when_they_are_the_majority() -> None:
    assert clean_text('a\\nb \\"q\\"\\n\\tc') == ('a\nb "q"\n\tc', True)
    code = 'line one\nline two\nprint("x\\n")'
    assert clean_text(code) == (code, False)


def test_clean_fixes_line_endings_trailing_spaces_and_blank_runs() -> None:
    assert clean_text("a  \r\nb\r\n\r\n\r\n\r\n\r\nc\rd\n") == ("a\nb\n\n\nc\nd", False)
    assert body_of("wiki/release-checklist.json").count("\n\n\n\n") == 0


def test_body_labels_each_field_once_and_only_with_several_fields() -> None:
    assert body_of("tracker/TKT-101-printer-jam.json") == (
        "## tkt_story\nThe large printer near the quiet room jams on Monday mornings.\n"
        "Brennagh Voss reported it twice.\n\n## tkt_followups\nChecked the paper tray; "
        "it was overfilled.\n\nOrdered a new roller from Halvorsen Supply."
    )
    assert body_of("wiki/onboarding-guide.json").startswith("Welcome aboard.\n\nStep one")
    mail = body_of("mailbox/2024/03/thread-quarterly.json")
    assert "## ltr_2_text\nauthor: Piet Calloway <piet@example.org>\nstamp: Wed" in mail


def test_no_body_falls_back_to_the_title_and_long_bodies_are_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = clean_body(cored("mailbox/empty-shell.json"))
    assert (shell.body, shell.labels, shell.flags) == ("Empty shell", (), ("no_body",))
    monkeypatch.setattr(clean, "MAX_BODY_CHARS", 30)
    units = (Unit("a", "x" * 20, "a"), Unit("b", "y" * 20, "b"))
    doc = CoreDoc("T", "t", False, "T", units, frozenset({"a", "b"}), ())
    cut = clean_body(doc)
    assert (cut.body, cut.labels, cut.flags) == ("## a\n" + "x" * 20, ("a",), ("body_truncated",))
