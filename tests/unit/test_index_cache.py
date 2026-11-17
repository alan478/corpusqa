"""The index cache (design.md §7): reuse on the same corpus and code, rebuild on any
change, prune old builds, never write into CORPUS_DIR."""

import os
import shutil
import time
from pathlib import Path

import pytest

from corpusqa.config import CACHE_KEEP, CACHE_STALE_BUILD_S, Settings
from corpusqa.ingest import Runtime, fingerprint, pipeline, run
from corpusqa.obs.testing import RecordedObservation, RecordingTracer
from corpusqa.store import sqlite as store_sqlite
from tests.unit.test_ingest_pipeline import FIXTURE


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    target = tmp_path / "corpus"
    shutil.copytree(FIXTURE, target)
    return target


def start(corpus: Path, index_dir: Path) -> tuple[Runtime, RecordedObservation]:
    """One start-up; returns the Runtime and the cache decision's span."""
    settings = Settings(corpus_dir=corpus.resolve(), port=8080, index_dir=index_dir)
    tracer = RecordingTracer()
    with tracer.trace("startup"):
        runtime = run(settings)
    [span] = tracer.named("index_cache")
    assert not tracer.named("S2_parse") if runtime.cache_hit else tracer.named("S2_parse")
    return runtime, span


def cache_of(span: RecordedObservation) -> tuple[object, object, object]:
    assert isinstance(span.output, dict)
    return span.output["cache"], span.output["reason"], span.output["changed"]


def folders(index_dir: Path) -> list[str]:
    return sorted(p.name for p in index_dir.iterdir())


def test_a_restart_on_the_same_corpus_reuses_the_index(corpus: Path, tmp_path: Path) -> None:
    first, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "no_index", [])
    second, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("hit", "hit", [])
    assert isinstance(span.output, dict) and span.output["fingerprint"]
    assert second.cache_hit and not first.cache_hit
    assert second.index_path == first.index_path and second.built_at == first.built_at
    assert second.known_doc_ids == first.known_doc_ids and len(second.known_doc_ids) == 21
    assert second.store is not None and second.store.doc_ids() == second.known_doc_ids


def test_a_changed_file_builds_a_new_index(corpus: Path, tmp_path: Path) -> None:
    first, _ = start(corpus, tmp_path / "index")
    note = corpus / "loose-note.json"
    note.write_text(note.read_text(encoding="utf-8").replace("frost", "thaw"), encoding="utf-8")
    second, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "changed", ["corpus"])
    assert second.index_path != first.index_path and not second.cache_hit
    assert len(folders(tmp_path / "index")) == 2  # the older build is kept (CACHE_KEEP)


def test_an_added_file_builds_a_new_index(corpus: Path, tmp_path: Path) -> None:
    start(corpus, tmp_path / "index")
    shutil.copy(corpus / "loose-note.json", corpus / "wiki" / "copy.json")
    second, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "changed", ["corpus"])
    assert len(second.known_doc_ids) == 21  # same id twice: the copy is set aside


def test_a_schema_version_bump_builds_a_new_index(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = start(corpus, tmp_path / "index")
    for module in (fingerprint, store_sqlite):
        monkeypatch.setattr(module, "SCHEMA_VERSION", module.SCHEMA_VERSION + 1)
    second, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "changed", ["schema"])
    assert second.index_path != first.index_path
    assert second.store is not None and second.store.meta()["schema_version"] == str(
        store_sqlite.SCHEMA_VERSION
    )


def test_a_code_change_builds_a_new_index(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start(corpus, tmp_path / "index")
    real = fingerprint.code_hash
    monkeypatch.setattr(fingerprint, "code_hash", lambda: "edited-" + real())
    _, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "changed", ["code"])


def test_a_build_setting_change_builds_a_new_index(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start(corpus, tmp_path / "index")
    monkeypatch.setattr(pipeline, "PROMOTE_PROSE_FIELDS", not pipeline.PROMOTE_PROSE_FIELDS)
    _, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "changed", ["settings"])


def test_code_hash_follows_the_bytes_of_the_code(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "conf.py").write_text("y = 2\n", encoding="utf-8")
    before = fingerprint.code_hash(("pkg", "conf.py"), root=tmp_path)
    assert before == fingerprint.code_hash(("pkg", "conf.py"), root=tmp_path)
    (tmp_path / "pkg" / "a.py").write_text("x = 3\n", encoding="utf-8")
    assert fingerprint.code_hash(("pkg", "conf.py"), root=tmp_path) != before


def test_a_crashed_build_is_rebuilt_not_reused(corpus: Path, tmp_path: Path) -> None:
    first, _ = start(corpus, tmp_path / "index")
    assert first.index_path is not None
    first.index_path.unlink()
    (first.index_path.parent / "corpus.sqlite.building").write_bytes(b"half")
    second, span = start(corpus, tmp_path / "index")
    assert cache_of(span) == ("miss", "incomplete", [])
    assert second.index_path == first.index_path and len(second.known_doc_ids) == 21
    assert [p.name for p in first.index_path.parent.iterdir() if "building" in p.name] == []


def test_old_builds_of_the_same_corpus_are_pruned(corpus: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    built: list[str] = []
    pruned: list[str] = []
    for _ in range(4):
        runtime, span = start(corpus, index)
        assert runtime.index_path is not None and isinstance(span.output, dict)
        built.append(runtime.index_path.parent.name)
        pruned.extend(span.output["pruned"])
        assert len(folders(index)) <= CACHE_KEEP
        note = corpus / "loose-note.json"
        note.write_text(note.read_text(encoding="utf-8") + " ", encoding="utf-8")
        time.sleep(0.01)  # distinct folder times
    assert len(set(built)) == 4
    assert folders(index) == sorted(built[-CACHE_KEEP:])
    assert pruned == built[:-CACHE_KEEP]


def _edited_copy(target: Path) -> Path:
    """A copy of the fixture with one file changed, so its fingerprint differs."""
    shutil.copytree(FIXTURE, target)
    note = target / "loose-note.json"
    note.write_text(note.read_text(encoding="utf-8") + f" {target.name}", encoding="utf-8")
    return target


def test_builds_of_other_corpora_are_kept_unless_their_folder_is_gone(
    corpus: Path, tmp_path: Path
) -> None:
    index = tmp_path / "index"
    other_rt, _ = start(_edited_copy(tmp_path / "other"), index)
    gone_rt, _ = start(_edited_copy(tmp_path / "gone"), index)
    shutil.rmtree(tmp_path / "gone")
    assert other_rt.index_path is not None and gone_rt.index_path is not None
    _, span = start(corpus, index)
    assert isinstance(span.output, dict)
    assert span.output["pruned"] == [gone_rt.index_path.parent.name]
    assert other_rt.index_path.is_file()


def test_crashed_builds_are_pruned_only_when_stale(corpus: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    fresh, stale, alien = index / ("a" * 16), index / ("b" * 16), index / "notes"
    for folder in (fresh, stale, alien):
        folder.mkdir(parents=True)
        (folder / "corpus.sqlite.building").write_bytes(b"half")
    old = time.time() - CACHE_STALE_BUILD_S - 60
    os.utime(stale / "corpus.sqlite.building", (old, old))
    os.utime(stale, (old, old))
    os.utime(alien, (old, old))
    _, span = start(corpus, index)
    assert isinstance(span.output, dict) and span.output["pruned"] == ["b" * 16]
    assert fresh.is_dir() and alien.is_dir() and not stale.exists()


def test_corpus_dir_is_never_written(corpus: Path, tmp_path: Path) -> None:
    def listing() -> list[tuple[str, int, int]]:
        return sorted(
            (str(p.relative_to(corpus)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in corpus.rglob("*")
        )

    before = listing()
    start(corpus, tmp_path / "index")
    start(corpus, tmp_path / "index")
    assert listing() == before
