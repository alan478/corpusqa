"""Index cache (design.md §7): which built index fits this corpus and this code.

Every build lives in `<index_dir>/<fingerprint[:FINGERPRINT_DIR_CHARS]>/corpus.sqlite`.
The fingerprint is a sha256 over four parts, each also stored in the index's meta so a
miss can say what changed:
- corpus: the sorted (path, size, mtime_ns) of every document file S1 found, plus the
  files it set aside (path, reason). File contents are not read: cheap, and CORPUS_DIR
  is read-only;
- code: the bytes of the source files that shape the index (FINGERPRINT_CODE_PATHS);
- schema: SCHEMA_VERSION;
- settings: build settings that are not in those files (prose promotion as the build
  sees it, and the year the fixed date window counts from).

A restart on the same corpus and code reuses the index; any change builds a new one.
Old builds are pruned at start (`prune`): beyond CACHE_KEEP per corpus folder, any whose
corpus folder is gone, and crashed builds. Nothing is ever written into CORPUS_DIR.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from corpusqa.config import (
    BUILDING_SUFFIX,
    CACHE_KEEP,
    CACHE_STALE_BUILD_S,
    FINGERPRINT_CODE_PATHS,
    FINGERPRINT_DIR_CHARS,
    INDEX_DB_NAME,
    SCHEMA_VERSION,
)
from corpusqa.ingest.models import Discovery
from corpusqa.store.sqlite import STATUS_COMPLETE, read_meta

logger = logging.getLogger(__name__)

# src/corpusqa: the code paths are relative to it.
CODE_ROOT = Path(__file__).resolve().parents[1]
_CACHE_FOLDER = re.compile(rf"^[0-9a-f]{{{FINGERPRINT_DIR_CHARS}}}$")

# meta keys of the fingerprint and its parts.
META_FINGERPRINT = "corpus_fingerprint"
META_CORPUS = "corpus_hash"
META_CODE = "pipeline_version"
META_SCHEMA = "schema_version"
META_SETTINGS = "build_settings"
META_CORPUS_DIR = "corpus_dir"
# What a miss reports as changed, per meta key.
_PARTS = {META_CORPUS: "corpus", META_CODE: "code", META_SCHEMA: "schema",
          META_SETTINGS: "settings"}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Fingerprint:
    corpus: str
    code: str
    schema: str
    settings: str  # JSON, keys sorted
    full: str

    @property
    def folder(self) -> str:
        return self.full[:FINGERPRINT_DIR_CHARS]

    def meta(self) -> dict[str, str]:
        return {
            META_FINGERPRINT: self.full,
            META_CORPUS: self.corpus,
            META_CODE: self.code,
            META_SCHEMA: self.schema,
            META_SETTINGS: self.settings,
        }


@dataclass(frozen=True, slots=True)
class CacheDecision:
    """Reuse the index at `db_path` (hit) or build it there (miss), and why."""

    hit: bool
    db_path: Path
    reason: str  # hit | no_index | changed | incomplete | unreadable | fingerprint_mismatch
    changed: tuple[str, ...] = ()  # parts that differ from the last build of this corpus
    previous: str | None = None  # folder of that last build


def code_hash(paths: Sequence[str] = FINGERPRINT_CODE_PATHS, root: Path = CODE_ROOT) -> str:
    """sha256 of the code that shapes the index: each file's path and bytes, in order."""
    files: list[Path] = []
    for part in paths:
        target = root / part
        if target.is_dir():
            files.extend(sorted(p for p in target.rglob("*") if _is_code(p)))
        elif target.is_file():
            files.append(target)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _is_code(path: Path) -> bool:
    return path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def corpus_hash(found: Discovery) -> str:
    """sha256 of the corpus listing S1 made (the files it kept and the ones it set aside)."""
    digest = hashlib.sha256()
    for file in found.files:
        digest.update(f"{file.rel_path}\0{file.size}\0{file.mtime_ns}\n".encode())
    for row in found.quarantined:
        digest.update(f"{row.rel_path}\0{row.reason}\n".encode())
    return digest.hexdigest()


def fingerprint(found: Discovery, settings: Mapping[str, object]) -> Fingerprint:
    parts = {
        "corpus": corpus_hash(found),
        "code": code_hash(),
        "schema": str(SCHEMA_VERSION),
        "settings": json.dumps(dict(settings), sort_keys=True),
    }
    full = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
    return Fingerprint(full=full, **parts)


def decide(index_dir: Path, fp: Fingerprint, corpus_dir: Path) -> CacheDecision:
    """Hit when this fingerprint's folder holds a complete index built for it."""
    db = index_dir / fp.folder / INDEX_DB_NAME
    meta = read_meta(db)
    if meta is not None and meta.get("status") == STATUS_COMPLETE:
        if meta.get(META_FINGERPRINT) == fp.full and meta.get(META_SCHEMA) == fp.schema:
            return CacheDecision(hit=True, db_path=db, reason="hit")
        return CacheDecision(hit=False, db_path=db, reason="fingerprint_mismatch")
    if meta is not None or db.with_name(db.name + BUILDING_SUFFIX).exists():
        return CacheDecision(hit=False, db_path=db, reason="incomplete")
    if db.exists():
        return CacheDecision(hit=False, db_path=db, reason="unreadable")
    last = _last_build(index_dir, corpus_dir, skip=fp.folder)
    if last is None:
        return CacheDecision(hit=False, db_path=db, reason="no_index")
    folder, old = last
    ours = fp.meta()
    changed = tuple(part for key, part in _PARTS.items() if old.get(key) != ours[key])
    return CacheDecision(hit=False, db_path=db, reason="changed", changed=changed, previous=folder)


def prune(index_dir: Path, keep: str, corpus_dir: Path) -> list[str]:
    """Delete old builds; returns the folder names deleted. Never touches `keep`, folders
    not named like a fingerprint, other corpora that still exist, or a build in progress."""
    if not index_dir.is_dir():
        return []
    ours: list[tuple[float, str]] = []
    doomed: list[str] = []
    now = time.time()
    for folder in index_dir.iterdir():
        if folder.name == keep or not folder.is_dir() or not _CACHE_FOLDER.match(folder.name):
            continue
        meta = read_meta(folder / INDEX_DB_NAME)
        if meta is None or meta.get("status") != STATUS_COMPLETE:
            if now - _last_change(folder) > CACHE_STALE_BUILD_S:
                doomed.append(folder.name)  # a crashed build
            continue
        built_for = meta.get(META_CORPUS_DIR)
        if built_for is None or not Path(built_for).is_dir():
            doomed.append(folder.name)  # its corpus is gone
        elif built_for == str(corpus_dir):
            ours.append((folder.stat().st_mtime, folder.name))
    ours.sort(reverse=True)
    doomed.extend(name for _, name in ours[max(CACHE_KEEP - 1, 0) :])
    for name in sorted(doomed):
        shutil.rmtree(index_dir / name, ignore_errors=True)
        logger.info("ingest: deleted old index %s", name)
    return sorted(doomed)


def mark_used(db_path: Path) -> None:
    """Make this build the newest for pruning (a reused index counts as fresh)."""
    os.utime(db_path.parent)


def _last_build(index_dir: Path, corpus_dir: Path, skip: str) -> tuple[str, dict[str, str]] | None:
    """The newest complete build of this corpus folder (other than `skip`), with its meta."""
    if not index_dir.is_dir():
        return None
    best: tuple[float, str, dict[str, str]] | None = None
    for folder in index_dir.iterdir():
        if folder.name == skip or not folder.is_dir() or not _CACHE_FOLDER.match(folder.name):
            continue
        meta = read_meta(folder / INDEX_DB_NAME)
        if meta is None or meta.get("status") != STATUS_COMPLETE:
            continue
        if meta.get(META_CORPUS_DIR) != str(corpus_dir):
            continue
        mtime = folder.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, folder.name, meta)
    return None if best is None else (best[1], best[2])


def _last_change(folder: Path) -> float:
    """Newest modification time of the folder and its files (now, if it cannot be read:
    a folder another process is changing is not stale)."""
    try:
        times = [folder.stat().st_mtime]
        times.extend(p.stat().st_mtime for p in folder.iterdir() if p.is_file())
    except OSError:
        return time.time()
    return max(times)
