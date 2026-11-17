"""S8 (profile, keys, links) and S10 (finalize) in the real build, on the mini corpus plus
invented sources made for them: own keys, a shared key, a hub key, every link method, a
field that depends on an enum, shared titles, and people with a learnable lexicon."""

import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from corpusqa.config import CONTENT_KEY, TITLE_KEY, UUID_KEY, Settings
from corpusqa.ingest import Runtime, run
from corpusqa.obs.testing import RecordingTracer
from corpusqa.store.interfaces import Integrity, StoreError
from corpusqa.store.sqlite import SqliteBuildWriter
from tests.unit.test_ingest_pipeline import FIXTURE, _renamed_copy

GIVEN = ("Odo", "Brix", "Tamsa", "Wenna", "Corvin")
FAMILY = ("Quarrow", "Pellam", "Strand", "Hollis", "Vane", "Ostrik")
N_TICKETS = 40  # 20 faults and 20 requests: both values reach CONDITIONAL_MIN_SUPPORT


def _write(path: Path, doc: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def linked_corpus(target: Path) -> None:
    """Copy the mini corpus and add four invented sources (see the module docstring)."""
    shutil.copytree(FIXTURE, target)
    pages = {"rota-plan": "Rota plan for the frost week", "kiln-checks": "Kiln checks"}
    for stem, title in pages.items():
        _write(target / "pages" / "ops" / f"{stem}.json", {
            "page_title": title, "page_text": f"{title}: what to do and when.",
            TITLE_KEY: "page_title", CONTENT_KEY: ["page_text"], UUID_KEY: f"pg-{stem}",
        })  # fmt: skip
    for i in range(6):  # six documents share the stem "readme": a hub key
        _write(target / "pages" / f"space-{i}" / "readme.json", {
            "page_title": f"Readme of space {i}", "page_text": "Start here.",
            TITLE_KEY: "page_title", CONTENT_KEY: ["page_text"], UUID_KEY: f"pg-readme-{i}",
        })  # fmt: skip
    for n in (7, 8, 9):  # numeric own ids: folder_number keys
        _write(target / "code" / "repo-a" / f"change-{n}.json", {
            "change_title": f"Change {n} to the kiln timer", "change_body": "Adjusts the timer.",
            "change_no": n, TITLE_KEY: "change_title", CONTENT_KEY: ["change_body"],
            UUID_KEY: f"cd-{n}",
        })  # fmt: skip
    for i in range(8):  # two room names over eight threads: shared, weak titles
        _write(target / "rooms" / f"thread-{i}.json", {
            "room": ["kiln", "yard"][i % 2], "said": f"Message {i} about the yard.",
            TITLE_KEY: "room", CONTENT_KEY: ["said"], UUID_KEY: f"rm-{i}",
        })  # fmt: skip
    refs = [
        "QZ-3", "Tracker: QZ-4", "QZ-999", "https://wiki.example/ops/rota-plan.html",
        "repo-a#7", "https://code.example/org/repo-a/pull/8", "#9", "Kiln checks",
        "https://wiki.example/space-9/readme",
    ]  # fmt: skip
    for n in range(1, N_TICKETS + 1):
        fault = n % 2 == 1
        doc: dict[str, object] = {
            "ticket_ref": "QZ-5" if n == N_TICKETS else f"QZ-{n}",  # QZ-5 is held twice
            "heading": f"Ticket {n}: the {['gate', 'kiln', 'yard', 'loom'][n % 4]} needs work",
            "story": f"Story of ticket {n}.",
            "ticket_kind": "fault" if fault else "request",
            "handler": f"{GIVEN[n % 5]} {FAMILY[n % 6]}" + (" (night lead)" if n == 1 else ""),
            "opened_at": f"2025-03-{1 + n % 28:02d}",
            "blocks": [refs[n % len(refs)], refs[(n + 3) % len(refs)]],
            TITLE_KEY: "heading",
            CONTENT_KEY: ["story"],
            UUID_KEY: f"tk-{n}",
        }
        if fault:
            doc["impact"] = ["high", "low"][n % 4 // 2]
        _write(target / "desk" / ["east", "west"][n % 2] / f"QZ-{n}-ticket.json", doc)


def build(corpus: Path, index_dir: Path, tracer: RecordingTracer | None = None) -> Runtime:
    settings = Settings(corpus_dir=corpus.resolve(), port=8080, index_dir=index_dir)
    if tracer is None:
        return run(settings)
    with tracer.trace("startup"):
        return run(settings)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Runtime, sqlite3.Connection]:
    corpus = tmp_path_factory.mktemp("linked") / "corpus"
    linked_corpus(corpus)
    runtime = build(corpus, tmp_path_factory.mktemp("index"))
    assert runtime.index_path is not None
    return runtime, sqlite3.connect(runtime.index_path)


def q(conn: sqlite3.Connection, sql: str, *args: object) -> list[tuple[object, ...]]:
    return conn.execute(sql, args).fetchall()


# ---------------------------------------------------------------- profile
def test_kinds_are_learned_from_values(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    kinds = dict(q(conn, "SELECT field, kind FROM field_profile WHERE source = 'desk'"))
    assert kinds == {
        "ticket_ref": "own_id", "heading": "mixed", "story": "text", "ticket_kind": "enum",
        "handler": "person", "opened_at": "date", "blocks": "reference", "impact": "enum",
    }  # fmt: skip
    [(presence, content, title, top)] = q(
        conn, "SELECT presence, content_ratio, title_ratio, top_values FROM field_profile "
              "WHERE source = 'desk' AND field = 'impact'"
    )  # fmt: skip
    assert (presence, content, title) == (0.5, 0.0, 0.0)
    assert json.loads(str(top)) == [["high", 10], ["low", 10]]
    assert q(conn, "SELECT content_ratio FROM field_profile WHERE field = 'story'") == [(1.0,)]


def test_every_field_of_every_source_has_a_profile(
    built: tuple[Runtime, sqlite3.Connection],
) -> None:
    _, conn = built
    stored = set(q(conn, "SELECT DISTINCT source, field FROM fields"))
    for source, body_fields in q(conn, "SELECT source, body_fields FROM documents"):
        stored |= {(source, name) for name in json.loads(str(body_fields))}
    assert set(q(conn, "SELECT source, field FROM field_profile")) == stored


def test_the_lexicon_and_people_matching_form(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    meta = dict(q(conn, "SELECT key, value FROM meta"))
    assert json.loads(meta["name_lexicon"]) == sorted(g.casefold() for g in GIVEN)
    [(raw, norm)] = q(conn, "SELECT value, value_norm FROM fields WHERE doc_id = 'tk-1' "
                            "AND field = 'handler'")  # fmt: skip
    assert (raw, norm) == ("Brix Pellam (night lead)", "brix pellam")


def test_a_field_that_applies_to_one_enum_value_is_found(
    built: tuple[Runtime, sqlite3.Connection],
) -> None:
    _, conn = built
    rows = q(conn, "SELECT source, field, given_field, present_when, strength "
                   "FROM field_conditions WHERE source = 'desk'")  # fmt: skip
    assert rows == [("desk", "impact", "ticket_kind", '["fault"]', 1.0)]


def test_shared_titles_get_the_file_stem(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    rooms = q(conn, "SELECT display_title, title_is_weak, flags FROM documents "
                    "WHERE source = 'rooms' ORDER BY rel_path")  # fmt: skip
    assert rooms[0][:2] == ("kiln — thread 0", 1)
    assert all(weak == 1 and "weak_title" in json.loads(str(flags)) for _, weak, flags in rooms)
    # Unique titles stay as they are.
    assert q(conn, "SELECT count(*) FROM documents WHERE source = 'desk' AND title_is_weak") == [
        (0,)
    ]


def test_source_profiles(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    [row] = q(conn, "SELECT n_docs, title_fields, own_id_fields, key_prefixes, date_fields, "
                    "path_depths, example_doc_ids FROM source_profile "
                    "WHERE source = 'desk'")  # fmt: skip
    n_docs, titles, own_ids, prefixes, dates, depths, examples = row
    assert n_docs == N_TICKETS and json.loads(str(titles)) == {"heading": N_TICKETS}
    assert json.loads(str(own_ids)) == ["ticket_ref"] and json.loads(str(prefixes)) == ["QZ"]
    assert json.loads(str(dates)) == ["opened_at"] and json.loads(str(depths)) == {"1": N_TICKETS}
    assert len(json.loads(str(examples))) == 3
    sources = {r[0] for r in q(conn, "SELECT source FROM source_profile")}
    assert sources == {r[0] for r in q(conn, "SELECT DISTINCT source FROM documents")}


# ---------------------------------------------------------------- keys and links
def test_doc_keys(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    shared = q(conn, "SELECT doc_id FROM doc_keys WHERE key_norm = 'qz-5' AND kind LIKE 'field:%' "
                     "ORDER BY doc_id")  # fmt: skip
    assert shared == [(f"tk-{N_TICKETS}",), ("tk-5",)]  # many-to-many
    assert q(conn, "SELECT DISTINCT is_hub FROM doc_keys WHERE key_norm = 'readme'") == [(1,)]
    folders = q(conn, "SELECT key_norm FROM doc_keys WHERE kind = 'folder_number' "
                      "AND doc_id LIKE 'cd-%' ORDER BY key_norm")  # fmt: skip
    assert folders == [("repo-a/7",), ("repo-a/8",), ("repo-a/9",)]
    assert q(conn, "SELECT count(*) FROM doc_keys WHERE key_norm GLOB '[0-9]*' "
                   "AND key_norm NOT GLOB '*[^0-9]*'") == [(0,)]  # fmt: skip
    without_key = q(conn, "SELECT count(*) FROM documents WHERE doc_id NOT IN "
                          "(SELECT doc_id FROM doc_keys)")  # fmt: skip
    assert without_key == [(0,)]


def test_links_by_method(built: tuple[Runtime, sqlite3.Connection]) -> None:
    _, conn = built
    by_raw = {
        raw: (method, target)
        for raw, method, target in q(
            conn, "SELECT raw, method, target_doc_id FROM links WHERE doc_id = 'tk-6'"
        )
    }
    assert by_raw == {"#9": ("none", None), "QZ-3": ("key", "tk-3")}
    methods = dict(q(conn, "SELECT raw, group_concat(DISTINCT method) FROM links "
                           "WHERE doc_id LIKE 'tk-%' GROUP BY raw"))  # fmt: skip
    assert methods == {
        "QZ-3": "key", "Tracker: QZ-4": "key", "QZ-999": "none",
        "https://wiki.example/ops/rota-plan.html": "stem", "repo-a#7": "folder_number",
        "https://code.example/org/repo-a/pull/8": "folder_number", "#9": "none",
        "Kiln checks": "unique_title", "https://wiki.example/space-9/readme": "stem",
    }  # fmt: skip
    hub = q(conn, "SELECT DISTINCT target_doc_id, target_is_hub FROM links "
                  "WHERE raw = 'https://wiki.example/space-9/readme'")  # fmt: skip
    assert hub == [(None, 1)]
    dangling = q(conn, "SELECT count(*) FROM links WHERE target_doc_id IS NOT NULL "
                       "AND target_doc_id NOT IN (SELECT doc_id FROM documents)")  # fmt: skip
    assert dangling == [(0,)]


# ---------------------------------------------------------------- finalize
def test_finalize_records_the_build_and_publishes_read_only(
    built: tuple[Runtime, sqlite3.Connection],
) -> None:
    runtime, conn = built
    meta = dict(q(conn, "SELECT key, value FROM meta"))
    assert meta["status"] == "complete" and len(meta["corpus_fingerprint"]) == 64
    for key in ("corpus_hash", "pipeline_version", "build_settings", "corpus_dir", "built_at"):
        assert meta[key], key
    tables = ("documents", "fields", "field_profile", "doc_keys", "links", "source_profile")
    for table in tables:
        count = q(conn, f"SELECT count(*) FROM {table}")[0][0]  # noqa: S608 - fixed names
        assert meta[f"n_{table}"] == str(count), table
    assert q(conn, "PRAGMA foreign_key_check") == []
    timed = q(conn, "SELECT stage FROM ingest_stages "
                    "WHERE json_extract(counts, '$.seconds') IS NOT NULL "
                    "AND (stage LIKE 'S8%' OR stage = 'S10_finalize') ORDER BY rowid")  # fmt: skip
    assert timed == [("S8_profile",), ("S8_keys",), ("S8_links",), ("S10_finalize",)]
    assert runtime.store is not None and runtime.store.doc_ids() == runtime.known_doc_ids
    assert runtime.built_at == meta["built_at"] and not runtime.cache_hit


def test_a_failed_check_stops_startup_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: SqliteBuildWriter) -> Integrity:
        return Integrity(quick_check="page 7: btree damaged", foreign_key_violations=0,
                         table_rows={"documents": 21})  # fmt: skip

    monkeypatch.setattr(SqliteBuildWriter, "integrity", broken)
    with pytest.raises(StoreError, match="btree damaged"):
        build(FIXTURE, tmp_path / "index")
    assert list((tmp_path / "index").iterdir()) == []


# ---------------------------------------------------------------- generality, tracing
S8_COUNTS = (
    "SELECT count(*) FROM field_profile",
    "SELECT kind, count(*) FROM field_profile GROUP BY kind ORDER BY kind",
    "SELECT count(*), sum(synonym_group IS NOT NULL) FROM field_profile",
    "SELECT count(*) FROM field_conditions",
    "SELECT substr(kind, 1, instr(kind || ':', ':') - 1) k, count(*), sum(is_hub) "
    "FROM doc_keys GROUP BY k ORDER BY k",
    "SELECT method, count(*), sum(target_is_hub) FROM links GROUP BY method ORDER BY method",
    "SELECT count(*) FROM documents WHERE title_is_weak",
    "SELECT value FROM meta WHERE key = 'name_lexicon'",
)


@pytest.mark.parametrize("seed", [3, 4])
def test_renaming_every_field_and_source_changes_no_s8_count(tmp_path: Path, seed: int) -> None:
    linked_corpus(tmp_path / "corpus")
    _renamed_copy(tmp_path / "corpus", tmp_path / "renamed", seed, keep_empty_key=True)
    original = build(tmp_path / "corpus", tmp_path / "a")
    renamed = build(tmp_path / "renamed", tmp_path / "b")
    assert original.index_path is not None and renamed.index_path is not None
    a, b = sqlite3.connect(original.index_path), sqlite3.connect(renamed.index_path)
    for sql in S8_COUNTS:
        assert q(a, sql) == q(b, sql), sql
    assert q(a, "SELECT count(*) FROM links WHERE target_doc_id IS NOT NULL")[0][0] > 0


def test_s8_and_s10_are_spans_with_their_counts(tmp_path: Path) -> None:
    linked_corpus(tmp_path / "corpus")
    tracer = RecordingTracer()
    build(tmp_path / "corpus", tmp_path / "index", tracer)
    spans = {s.name: s for s in tracer.observations}
    profile, keys, links, final = (
        spans[n].output for n in ("S8_profile", "S8_keys", "S8_links", "S10_finalize")
    )
    assert isinstance(profile, dict) and isinstance(keys, dict)
    assert isinstance(links, dict) and isinstance(final, dict)
    assert profile["conditions"] >= 1 and profile["weak_titles_added"] == 8
    assert profile["lexicon"] == len(GIVEN)
    kinds = profile["kinds"]
    assert isinstance(kinds, dict) and kinds["desk"] == dict(
        sorted(Counter(["own_id", "mixed", "text", "enum", "person", "date", "reference",
                        "enum"]).items())
    )  # fmt: skip
    assert keys["own_id_fields"] == {"code": ["change_no"], "desk": ["ticket_ref"]}
    assert keys["hub_keys"] == 1
    assert keys["prefixes_by_source"] == {"desk": ["QZ"], "tracker": ["TKT"]}  # from file stems
    assert links["values_resolved"] > 0 and links["method:none"] > 0
    assert links["method:unique_title"] >= 1 and links["hub_matches"] >= 1
    assert final["quick_check"] == "ok" and final["n_documents"] == final["known_doc_ids"]
    assert all("seconds" in spans[n].metadata for n in ("S8_profile", "S8_keys", "S10_finalize"))
