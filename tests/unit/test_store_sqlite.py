"""The SQLite store: batch writes, publish by rename, read-only access, cascades."""

import sqlite3
import zlib
from pathlib import Path

import pytest

from corpusqa.config import SCHEMA_VERSION
from corpusqa.store.interfaces import DocumentRow, FieldRow, QuarantineRow, StageRow, StoreError
from corpusqa.store.sqlite import (
    SqliteBuildWriter,
    SqliteReader,
    building_path,
    open_readonly,
    read_meta,
)


def make_doc(doc_id: str, n_fields: int = 2) -> DocumentRow:
    fields = tuple(
        FieldRow(field="colour", field_norm="colour", seq=i, role="meta", value=f"v{i}",
                 value_type="str", value_norm=f"v{i}")
        for i in range(n_fields)
    )  # fmt: skip
    return DocumentRow(
        doc_id=doc_id, source="src", rel_path=f"src/{doc_id}.json", file_stem=doc_id,
        title="T", title_field="name", title_is_weak=False, display_title="T", body="B",
        body_fields=("text",), created=None, created_field=None, created_rule=None,
        updated=None, updated_field=None, updated_rule=None, content_hash="h",
        size_bytes=3, mtime_ns=1, flags=("x",), raw=b"{}", fields=fields,
    )  # fmt: skip


def build(path: Path, docs: list[DocumentRow]) -> Path:
    writer = SqliteBuildWriter(path)
    writer.add_documents(docs)
    writer.add_quarantine([QuarantineRow("src/bad.json", "src", "json_invalid", "empty file")])
    writer.add_stage(StageRow("S1_discover", "t0", "t1", {"files": 3}))
    return writer.finish()


def test_build_is_invisible_until_finish_then_renamed(tmp_path: Path) -> None:
    db = tmp_path / "idx" / "corpus.sqlite"
    writer = SqliteBuildWriter(db)
    writer.add_documents([make_doc("a")])
    assert building_path(db).exists() and not db.exists()
    with pytest.raises(StoreError, match="unfinished build"):
        SqliteReader(building_path(db))
    assert writer.finish() == db
    assert db.exists() and not building_path(db).exists()
    assert sorted(p.name for p in db.parent.iterdir()) == ["corpus.sqlite"]


def test_rows_round_trip(tmp_path: Path) -> None:
    db = build(tmp_path / "corpus.sqlite", [make_doc("a"), make_doc("b", n_fields=3)])
    assert SqliteReader(db).doc_ids() == frozenset({"a", "b"})
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM fields").fetchone() == (5,)
        raw, flags = conn.execute("SELECT raw, flags FROM documents WHERE doc_id='a'").fetchone()
        assert zlib.decompress(raw) == b"{}" and flags == '["x"]'
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        assert meta == {"schema_version": str(SCHEMA_VERSION), "status": "complete"}
        assert conn.execute("SELECT reason FROM quarantine").fetchall() == [("json_invalid",)]
        assert conn.execute("SELECT counts FROM ingest_stages").fetchone() == ('{"files": 3}',)


def test_a_new_build_replaces_the_old_index(tmp_path: Path) -> None:
    db = build(tmp_path / "corpus.sqlite", [make_doc("old")])
    build(db, [make_doc("new")])
    assert SqliteReader(db).doc_ids() == frozenset({"new"})


def test_abort_leaves_nothing(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite"
    writer = SqliteBuildWriter(db)
    writer.add_documents([make_doc("a")])
    writer.abort()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_batch_writes_none_of_its_rows(tmp_path: Path) -> None:
    writer = SqliteBuildWriter(tmp_path / "corpus.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        writer.add_documents([make_doc("a"), make_doc("a")])
    writer.add_documents([make_doc("b")])
    db = writer.finish()
    assert SqliteReader(db).doc_ids() == frozenset({"b"})


def test_reader_refuses_incomplete_or_other_schema(tmp_path: Path) -> None:
    db = build(tmp_path / "corpus.sqlite", [make_doc("a")])
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value='building' WHERE key='status'")
    with pytest.raises(StoreError, match="not complete"):
        SqliteReader(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value='complete' WHERE key='status'")
        conn.execute("UPDATE meta SET value='0' WHERE key='schema_version'")
    with pytest.raises(StoreError, match="schema"):
        SqliteReader(db)
    with pytest.raises(StoreError, match="no index"):
        SqliteReader(tmp_path / "missing.sqlite")


def test_readonly_connection_cannot_write(tmp_path: Path) -> None:
    db = build(tmp_path / "corpus.sqlite", [make_doc("a")])
    conn = open_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM documents")
    finally:
        conn.close()


def test_deleting_a_document_cascades_to_its_fields(tmp_path: Path) -> None:
    db = build(tmp_path / "corpus.sqlite", [make_doc("a"), make_doc("b")])
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM documents WHERE doc_id='a'")
        assert conn.execute("SELECT DISTINCT doc_id FROM fields").fetchall() == [("b",)]


def test_read_meta_reads_any_index_file_and_never_raises(tmp_path: Path) -> None:
    db = build(tmp_path / "corpus.sqlite", [make_doc("a")])
    assert read_meta(db) == SqliteReader(db).meta() == {
        "schema_version": str(SCHEMA_VERSION), "status": "complete",
    }  # fmt: skip
    writer = SqliteBuildWriter(tmp_path / "other.sqlite")
    assert read_meta(writer.building) == {"schema_version": str(SCHEMA_VERSION),
                                          "status": "building"}  # fmt: skip
    writer.abort()
    (tmp_path / "junk.sqlite").write_bytes(b"not a database at all, just bytes" * 10)
    assert read_meta(tmp_path / "junk.sqlite") is None
    assert read_meta(tmp_path / "missing.sqlite") is None


def test_integrity_counts_rows_and_finds_no_problem_in_a_sound_build(tmp_path: Path) -> None:
    writer = SqliteBuildWriter(tmp_path / "corpus.sqlite")
    writer.add_documents([make_doc("a", n_fields=3)])
    found = writer.integrity()
    writer.abort()
    assert found.ok and found.quick_check == "ok" and found.foreign_key_violations == 0
    assert found.table_rows["documents"] == 1 and found.table_rows["fields"] == 3
    assert found.table_rows["links"] == 0
