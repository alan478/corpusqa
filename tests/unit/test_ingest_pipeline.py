"""The whole build (S1-S5 + S7) on the mini corpus: invariants, determinism, generality,
tracing, and failure handling."""

import json
import logging
import random
import re
import shutil
import sqlite3
import string
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from corpusqa.config import CONTENT_KEY, CONTRACT_KEYS, TITLE_KEY, ConfigError, Settings
from corpusqa.ingest import Runtime, pipeline, run
from corpusqa.ingest.models import CoreDoc, ParsedDoc
from corpusqa.obs.testing import RecordingTracer
from corpusqa.store.sqlite import SqliteBuildWriter

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"


def build(corpus: Path, index_dir: Path) -> tuple[Runtime, sqlite3.Connection]:
    runtime = run(Settings(corpus_dir=corpus.resolve(), port=8080, index_dir=index_dir))
    assert runtime.index_path is not None and runtime.index_path.parent.parent == index_dir
    return runtime, sqlite3.connect(runtime.index_path)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Runtime, sqlite3.Connection]:
    return build(FIXTURE, tmp_path_factory.mktemp("index"))


def rows(conn: sqlite3.Connection, sql: str) -> list[tuple[object, ...]]:
    return conn.execute(sql).fetchall()


# ---------------------------------------------------------------- invariants (design §5)
def test_counts_on_the_mini_corpus(built: tuple[Runtime, sqlite3.Connection]) -> None:
    runtime, conn = built
    assert len(runtime.known_doc_ids) == 21
    reasons = dict(rows(conn, "SELECT reason, count(*) FROM quarantine GROUP BY reason"))
    assert reasons == {"json_invalid": 2, "not_object": 1, "no_uuid": 2, "duplicate_uuid": 1}
    meta = dict(rows(conn, "SELECT key, value FROM meta"))
    assert (meta["status"], meta["n_files"], meta["n_documents"], meta["n_quarantined"]) == (
        "complete", "27", "21", "6",
    )  # fmt: skip
    stages = [r[0] for r in rows(conn, "SELECT stage FROM ingest_stages ORDER BY rowid")]
    assert stages == [
        "S1_discover", "S2_parse", "S3_core", "S4_fields", "S5_clean", "S7_store", "S4_date_window",
        "S8_profile", "S8_keys", "S8_links", "S6_chunk", "S10_finalize", "S9_embed",
    ]  # fmt: skip


def test_known_ids_are_exactly_the_stored_documents(
    built: tuple[Runtime, sqlite3.Connection],
) -> None:
    runtime, conn = built
    assert runtime.known_doc_ids == {r[0] for r in rows(conn, "SELECT doc_id FROM documents")}
    # The only quarantined file with an id is the duplicate; its id belongs to the kept file.
    [(path, doc_id)] = rows(
        conn, "SELECT rel_path, doc_id FROM quarantine WHERE doc_id IS NOT NULL"
    )
    kept = conn.execute("SELECT rel_path FROM documents WHERE doc_id = ?", (doc_id,)).fetchall()
    assert kept == [("crm/acct-copy-a.json",)] and path == "crm/acct-copy-b.json"


def test_every_non_body_key_has_field_rows(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    stored = Counter(r[0] for r in rows(conn, "SELECT doc_id FROM fields"))
    for doc_id, raw, body_fields, title_field in rows(
        conn, "SELECT doc_id, raw, body_fields, title_field FROM documents"
    ):
        assert isinstance(raw, bytes) and isinstance(body_fields, str)
        text = zlib.decompress(raw).decode("utf-8-sig", errors="replace").replace("\x00", "")
        obj = json.loads(text)
        body_keys = set(json.loads(body_fields)) - {title_field}
        expected = sum(
            len(v) if isinstance(v, list) and v else 1
            for k, v in obj.items()
            if k not in CONTRACT_KEYS and k not in body_keys
        )
        assert stored[doc_id] == expected, doc_id


def test_created_is_never_after_updated(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    assert rows(conn, "SELECT count(*) FROM documents WHERE created > updated") == [(0,)]
    assert rows(conn, "SELECT count(*) FROM documents WHERE created IS NOT NULL") == [(10,)]


# ---------------------------------------------------------------- learned date window
def test_a_small_corpus_keeps_the_fixed_window_and_counts_the_dates_outside_it(
    built: tuple[Runtime, sqlite3.Connection],
) -> None:
    _, conn = built
    meta = dict(rows(conn, "SELECT key, value FROM meta WHERE key LIKE 'date%'"))
    this_year = datetime.now(UTC).year
    assert (meta["date_window_rule"], meta["date_window_iso_values"]) == ("fixed", "17")
    assert (meta["date_window_min_year"], meta["date_window_max_year"]) == (
        "1990", str(this_year + 5),
    )  # fmt: skip
    outside = {
        (s["source"], f["field"]): (f["values"], f["outside"])
        for s in json.loads(meta["date_outside_window"])
        for f in s["fields"]
    }
    assert outside == {("calls", "d_far_future"): (1, 1), ("calls", "d_too_old"): (1, 1),
                       ("chatter", "ended_ts"): (2, 1)}  # fmt: skip
    # Each date on its own (user, 2026-10-16): half of ended_ts's values are outside the
    # window, yet its in-window value of this chat is used, so the update comes from it.
    [chat] = rows(conn, "SELECT created_field, updated_field, updated_rule FROM documents "
                        "WHERE doc_id = 'fx-chat-01'")  # fmt: skip
    assert chat == ("began_ts", "ended_ts", "order")
    assert rows(conn, "SELECT count(*) FROM documents WHERE created_rule LIKE '%trust%'") == [(0,)]


def _beacon_corpus(target: Path) -> None:
    """The mini corpus plus an invented source with enough ISO dates to learn a window.

    300 files carry an ISO date (2024-2026) and an epoch whose year is usually absurd;
    20 carry only an in-window epoch, 5 only an absurd one, and 1 only an ISO date in
    2030 (inside the fixed window, outside the learned one).
    """
    shutil.copytree(FIXTURE, target)
    folder = target / "beacons"
    folder.mkdir()
    in_window_epoch, absurd_epoch = 1735689600, 5_000_000_000  # 2025-01-01, year 2128

    def write(name: str, extra: dict[str, object]) -> None:
        doc = {"tag": name, "note": f"Beacon {name} log.", **extra, TITLE_KEY: "tag",
               CONTENT_KEY: ["note"], "dataset_doc_uuid": f"bcn-{name}"}  # fmt: skip
        (folder / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    for i in range(300):
        pulse = absurd_epoch + i if i % 10 < 7 else in_window_epoch + i
        write(f"a{i}", {"logged_on": f"{2024 + i % 3}-0{1 + i % 9}-15", "pulse_ts": str(pulse)})
    for i in range(20):
        write(f"b{i}", {"pulse_ts": str(in_window_epoch + i)})
    for i in range(5):
        write(f"c{i}", {"pulse_ts": str(absurd_epoch + i)})
    write("late", {"logged_on": "2030-05-01"})


def test_the_window_is_learned_from_iso_dates_and_each_epoch_is_judged_on_its_own(
    tmp_path: Path,
) -> None:
    _beacon_corpus(tmp_path / "corpus")
    _, conn = build(tmp_path / "corpus", tmp_path / "index")
    meta = dict(rows(conn, "SELECT key, value FROM meta WHERE key LIKE 'date%'"))
    # 318 ISO values (300 + 17 in the mini corpus + 1): p1 = 2024, p99 = 2026, +-1 year.
    assert (meta["date_window_rule"], meta["date_window_iso_values"]) == ("learned", "318")
    assert (meta["date_window_min_year"], meta["date_window_max_year"]) == ("2023", "2027")
    [beacons] = [s for s in json.loads(meta["date_outside_window"]) if s["source"] == "beacons"]
    fields = {f["field"]: (f["values"], f["outside"]) for f in beacons["fields"]}
    # pulse_ts: 210 of the a-files and the 5 c-files are absurd; logged_on: only "late".
    assert fields == {"pulse_ts": (325, 215), "logged_on": (301, 1)}
    assert (beacons["values"], beacons["outside"]) == (626, 216)
    assert beacons["documents_without_date"] == 6  # the 5 c-files and "late"

    def rule_counts(prefix: str) -> dict[object, object]:
        sql = "SELECT created_rule, count(*) FROM documents WHERE doc_id LIKE ? GROUP BY 1"
        return dict(conn.execute(sql, (f"bcn-{prefix}%",)).fetchall())

    def created_fields(prefix: str) -> dict[object, object]:
        sql = "SELECT created_field, count(*) FROM documents WHERE doc_id LIKE ? GROUP BY 1"
        return dict(conn.execute(sql, (f"bcn-{prefix}%",)).fetchall())

    assert rule_counts("a") == {"order": 300}
    # pulse_ts's in-window values (90 a-files) compete by order like any date: those from
    # 2025-01-01 come before a logged_on of 2025 or 2026 (60 files), not before 2024 (30).
    assert created_fields("a") == {"logged_on": 240, "pulse_ts": 60}
    assert rule_counts("b") == {"order": 20}  # an in-window epoch is a date like any other
    assert rule_counts("c") == {None: 5}  # outside the window: never a date
    [late] = rows(conn, "SELECT created, flags FROM documents WHERE doc_id = 'bcn-late'")
    assert late[0] is None and "no_date" in json.loads(str(late[1]))
    in_range = dict(rows(conn, "SELECT value, date_in_range FROM fields "
                               "WHERE doc_id IN ('bcn-late', 'bcn-a0', 'bcn-a7') "
                               "AND value_iso IS NOT NULL"))  # fmt: skip
    assert in_range == {"2030-05-01": 0, "2024-01-15": 1, str(5_000_000_000): 0,
                        "2025-08-15": 1, str(1735689600 + 7): 1}  # fmt: skip


def test_only_the_finished_index_remains(tmp_path: Path) -> None:
    runtime, _ = build(FIXTURE, tmp_path)
    assert runtime.index_path is not None and runtime.index_path.is_file()
    [folder] = tmp_path.iterdir()
    assert (folder / "corpus.sqlite").is_file()
    assert not [p.name for p in folder.iterdir() if ".building" in p.name]


# ---------------------------------------------------------------- determinism, generality
def test_two_builds_are_identical(tmp_path: Path) -> None:
    _, first = build(FIXTURE, tmp_path / "a")
    _, second = build(FIXTURE, tmp_path / "b")
    for sql in (
        "SELECT * FROM documents ORDER BY doc_id",
        "SELECT * FROM fields ORDER BY doc_id, field, seq",
        "SELECT * FROM quarantine ORDER BY rel_path",
        "SELECT * FROM chunks ORDER BY chunk_id",
    ):
        assert rows(first, sql) == rows(second, sql), sql
    meta = "SELECT key, value FROM meta WHERE key != 'built_at' ORDER BY key"
    assert rows(first, meta) == rows(second, meta)


def _renamed_copy(source: Path, target: Path, seed: int, *, keep_empty_key: bool = False) -> None:
    """Copy the corpus with every source folder and field name changed by random tags.

    Folders get a suffix. In field names each word (run of letters) gets a suffix, the
    same one wherever the word appears. That keeps the generic shapes the design relies
    on (numbered families, a leading `_`, English role words as word starts) while no
    name stays the same. `keep_empty_key` leaves the empty key empty: it has no name to
    change, and renaming it gives it one (an empty key is noise by its shape).
    """
    rng = random.Random(seed)  # noqa: S311 - test data, not a secret

    def tag() -> str:
        return "".join(rng.choices(string.ascii_lowercase, k=4))

    folders = {p.name: p.name + "_" + tag() for p in source.iterdir() if p.is_dir()}
    words: dict[str, str] = {}

    def word(match: re.Match[str]) -> str:
        return words.setdefault(match.group(0), match.group(0) + tag())

    def rename(key: str) -> str:
        if keep_empty_key and key == "":
            return key
        new = re.sub(r"[^\W\d_]+", word, key)
        return new if new != key else key + words.setdefault("", tag())

    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = path.relative_to(source).parts
        out = target.joinpath(folders.get(rel[0], rel[0]), *rel[1:])
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        try:
            obj = json.loads(raw.decode("utf-8-sig", errors="replace").replace("\x00", ""))
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            out.write_bytes(raw)
            continue
        renamed = {k if k in CONTRACT_KEYS else rename(k): v for k, v in obj.items()}
        if isinstance(renamed.get(TITLE_KEY), str):
            renamed[TITLE_KEY] = rename(renamed[TITLE_KEY])
        if isinstance(renamed.get(CONTENT_KEY), list):
            renamed[CONTENT_KEY] = [
                rename(n) if isinstance(n, str) else n for n in obj[CONTENT_KEY]
            ]
        elif isinstance(renamed.get(CONTENT_KEY), str):
            renamed[CONTENT_KEY] = rename(renamed[CONTENT_KEY])
        out.write_text(json.dumps(renamed, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("seed", [1, 2])
def test_renaming_every_field_and_source_changes_no_count(tmp_path: Path, seed: int) -> None:
    _renamed_copy(FIXTURE, tmp_path / "renamed", seed)
    _, original = build(FIXTURE, tmp_path / "a")
    _, renamed = build(tmp_path / "renamed", tmp_path / "b")
    for sql in (
        "SELECT count(*) FROM documents",
        "SELECT count(*) FROM fields",
        "SELECT reason, count(*) FROM quarantine GROUP BY reason ORDER BY reason",
        "SELECT count(*) FROM documents WHERE created IS NOT NULL",
    ):
        assert rows(original, sql) == rows(renamed, sql), sql
    sources = {r[0] for r in rows(renamed, "SELECT source FROM documents")}
    assert sources.isdisjoint({"tracker", "wiki", "crm", "mailbox", "chatter", "calls"})


# ---------------------------------------------------------------- tracing
def test_every_stage_is_a_span_with_counts(tmp_path: Path) -> None:
    settings = Settings(corpus_dir=FIXTURE, port=8080, index_dir=tmp_path)
    tracer = RecordingTracer.for_settings(settings)
    with tracer.trace("startup"):
        run(settings)
    [ingest] = tracer.named("ingest")
    assert [s.name for s in tracer.children(ingest)] == [
        "S1_discover", "index_cache", "S2_parse", "S3_core", "S4_fields", "S5_clean", "S7_store",
        "S4_date_window", "S8_profile", "S8_keys", "S8_links", "S6_chunk", "S10_finalize",
        "S9_embed", "S10_vectors",
    ]  # fmt: skip
    by_name = {s.name: s for s in tracer.children(ingest)}
    assert isinstance(ingest.output, dict) and ingest.output["documents"] == 21
    assert ingest.output["known_doc_ids"] == 21
    assert by_name["S1_discover"].output == {
        "files": 27,
        "files_by_source": {
            "_root": 1, "calls": 4, "chatter": 2, "crm": 8, "mailbox": 3, "tracker": 4, "wiki": 5,
        },
        "skipped": {"hidden": 2, "not_json": 1},
        "quarantined": {},
        "set_aside": [],
    }  # fmt: skip
    s2 = by_name["S2_parse"].output
    assert isinstance(s2, dict) and s2["parsed"] == 21 and s2["quarantined:no_uuid"] == 2
    set_aside = {item["path"]: item["reason"] for item in s2["set_aside"]}
    assert set_aside["crm/listing.json"] == "not_object" and len(set_aside) == 6
    s3 = by_name["S3_core"].output
    assert isinstance(s3, dict) and s3["promoted"] == 2 and s3["weak_title"] == 3
    assert by_name["S7_store"].output == {"batches": 1, "documents": 21, "field_rows": 102}
    assert all("seconds" in by_name[n].metadata for n in ("S2_parse", "S5_clean", "S7_store"))
    dates = by_name["S4_date_window"].output
    assert isinstance(dates, dict) and dates["window_rule"] == "fixed"
    assert (dates["sources_with_dates_outside"], dates["documents_without_date"]) == (2, 11)
    assert (dates["date_values"], dates["dates_outside"]) == (33, 3)
    assert [
        (item["source"], [f["field"] for f in item["fields"]]) for item in dates["outside"]
    ] == [("calls", ["d_far_future", "d_too_old"]), ("chatter", ["ended_ts"])]


# ---------------------------------------------------------------- the start-up log
# User, 2026-10-17: start-up printed nothing between "building a new index" and S9, so the
# terminal gave no sign the build was running. Now each stage logs when it starts and ends.
BUILD_STAGES = [
    "S1_discover", "S2_parse", "S3_core", "S4_fields", "S5_clean", "S7_store", "S4_date_window",
    "S8_profile", "S8_keys", "S8_links", "S6_chunk", "S10_finalize", "S9_embed",
]  # fmt: skip


def ingest_log(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("corpusqa.ingest")]


def done_stages(lines: list[str]) -> list[str]:
    return [line.split()[1].rstrip(":") for line in lines if ": done" in line]


def test_a_build_logs_each_stage_when_it_starts_and_when_it_is_done(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="corpusqa.ingest"):
        build(FIXTURE, tmp_path / "index")
    lines = ingest_log(caplog)
    assert done_stages(lines) == BUILD_STAGES
    for stage in BUILD_STAGES:
        start = next(i for i, line in enumerate(lines) if stage in line and ": started" in line)
        end = next(i for i, line in enumerate(lines) if line.startswith(f"ingest: {stage}: done"))
        assert start < end, stage
    done = {line.split()[1].rstrip(":"): line for line in lines if ": done" in line}
    assert "files=27" in done["S1_discover"]
    assert "parsed=21" in done["S2_parse"] and "documents=21" in done["S7_store"]
    assert re.search(r"done in \d+\.\d s", done["S6_chunk"])


def test_a_reused_index_logs_only_the_stages_that_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    build(FIXTURE, tmp_path / "index")
    with caplog.at_level(logging.INFO, logger="corpusqa.ingest"):
        build(FIXTURE, tmp_path / "index")
    assert done_stages(ingest_log(caplog)) == ["S1_discover", "S9_embed"]


def test_the_file_loop_logs_its_progress(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "INGEST_PROGRESS_EVERY_FILES", 10)
    with caplog.at_level(logging.INFO, logger="corpusqa.ingest"):
        build(FIXTURE, tmp_path / "index")
    assert [line for line in ingest_log(caplog) if "files read" in line] == [
        "ingest: 10 of 27 files read",
        "ingest: 20 of 27 files read",
    ]


# ---------------------------------------------------------------- failures
def test_a_file_that_breaks_a_stage_is_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pipeline.build_core

    def flaky(doc: ParsedDoc) -> CoreDoc:
        if doc.doc_id == "fx-tkt-102":
            raise RuntimeError("boom")
        return real(doc)

    monkeypatch.setattr(pipeline, "build_core", flaky)
    runtime, conn = build(FIXTURE, tmp_path)
    assert len(runtime.known_doc_ids) == 20 and "fx-tkt-102" not in runtime.known_doc_ids
    assert rows(conn, "SELECT reason, detail FROM quarantine WHERE rel_path LIKE '%TKT-102%'") == [
        ("process_error", "RuntimeError: boom")
    ]


def test_a_lone_surrogate_in_one_document_never_fails_the_build(tmp_path: Path) -> None:
    # B5 review finding 1: before the fix, the batch write raised UnicodeEncodeError and
    # start-up failed for the whole corpus.
    corpus = tmp_path / "corpus"
    (corpus / "notes").mkdir(parents=True)
    (corpus / "notes" / "good.json").write_text(
        '{"dataset_doc_uuid": "good-1", "title": "Fine", "content_field_names": ["text"],'
        ' "text": "plain words"}',
        encoding="utf-8",
    )
    (corpus / "notes" / "odd.json").write_text(
        '{"dataset_doc_uuid": "odd-1", "title_field_name": "name", "name": "Half \\ud83d emoji",'
        ' "content_field_names": ["text"], "text": "cut \\udc00 here", "x\\ud800": 1}',
        encoding="utf-8",
    )
    (corpus / "notes" / "bad-id.json").write_text(
        '{"dataset_doc_uuid": "bad\\ud800", "text": "never stored"}', encoding="utf-8"
    )
    runtime, conn = build(corpus, tmp_path / "index")
    assert runtime.known_doc_ids == frozenset({"good-1", "odd-1"})
    [(title, flags)] = rows(conn, "SELECT title, flags FROM documents WHERE doc_id = 'odd-1'")
    assert title == "Half � emoji" and "bad_unicode" in str(flags)
    assert rows(conn, "SELECT rel_path, reason FROM quarantine") == [
        ("notes/bad-id.json", "no_uuid")
    ]


def test_a_write_error_fails_startup_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: SqliteBuildWriter, docs: object) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(SqliteBuildWriter, "add_documents", broken)
    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        build(FIXTURE, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_empty_corpus_is_ready_with_no_documents(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    runtime, _ = build(tmp_path / "corpus", tmp_path / "index")
    assert runtime.known_doc_ids == frozenset() and runtime.store is not None


def test_a_missing_corpus_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a readable directory"):
        run(Settings(corpus_dir=tmp_path / "gone", port=8080, index_dir=tmp_path))
