"""S1 discover: list the candidate document files under CORPUS_DIR, in sorted path order."""

import logging
from collections import Counter
from pathlib import Path

from corpusqa.config import DOC_SUFFIX, FOLLOW_SYMLINKS_INSIDE_CORPUS, MAX_DOC_BYTES, ROOT_SOURCE
from corpusqa.ingest.models import Discovery, FileRef
from corpusqa.store.interfaces import QuarantineRow

logger = logging.getLogger(__name__)


def discover(corpus_dir: Path) -> Discovery:
    """Walk `corpus_dir`. Hidden entries, non-.json files and links leaving the corpus are
    skipped (counted by kind); unreadable and oversized files are set aside with a reason.

    The walk does not depend on the order the file system lists entries: entries are
    sorted, and linked folders are walked only after every real folder, so a folder
    reachable both ways is always found under its real path.
    """
    root = corpus_dir.resolve()
    files: list[FileRef] = []
    quarantined: list[QuarantineRow] = []
    skipped: Counter[str] = Counter()
    visited = {root}  # real folders already walked: stops symlink loops
    folders = [root]
    links: list[Path] = []  # linked folders, in the order met; walked after real ones
    while folders or links:
        folder = folders.pop() if folders else links.pop(0)
        if folder.is_symlink():
            real = folder.resolve()
            if real in visited:
                skipped["folder_seen_twice"] += 1
                continue
            visited.add(real)
        try:
            entries = sorted(folder.iterdir())
        except OSError as exc:
            logger.warning("ingest: cannot list %s (%s); skipped", folder, exc)
            skipped["unreadable_folder"] += 1
            continue
        for entry in entries:
            kind = _skip_kind(entry, root)
            if kind is not None:
                skipped[kind] += 1
            elif entry.is_dir():
                if entry.is_symlink():
                    links.append(entry)
                    continue
                real = entry.resolve()
                if real in visited:
                    skipped["folder_seen_twice"] += 1
                else:
                    visited.add(real)
                    folders.append(entry)
            elif entry.suffix.lower() != DOC_SUFFIX:
                skipped["not_json"] += 1
            else:
                found = _file_ref(entry, root)
                if isinstance(found, QuarantineRow):
                    quarantined.append(found)
                else:
                    files.append(found)
    return Discovery(
        files=tuple(sorted(files, key=lambda f: f.rel_path)),
        skipped=dict(sorted(skipped.items())),
        quarantined=tuple(sorted(quarantined, key=lambda q: q.rel_path)),
    )


def _skip_kind(entry: Path, root: Path) -> str | None:
    if entry.name.startswith("."):
        return "hidden"
    if entry.is_symlink():
        if not FOLLOW_SYMLINKS_INSIDE_CORPUS:
            return "symlink"
        if not entry.resolve().is_relative_to(root):
            return "symlink_outside_corpus"
    return None


def _file_ref(path: Path, root: Path) -> FileRef | QuarantineRow:
    rel_path = path.relative_to(root).as_posix()
    parts = rel_path.split("/")
    source, flags = (ROOT_SOURCE, ("no_source_folder",)) if len(parts) == 1 else (parts[0], ())
    try:
        stat = path.stat()  # follows a link; a broken one raises
    except OSError as exc:
        return QuarantineRow(rel_path, source, "unreadable", f"{type(exc).__name__}: {exc}")
    if stat.st_size > MAX_DOC_BYTES:
        return QuarantineRow(rel_path, source, "too_large", f"{stat.st_size} bytes")
    return FileRef(rel_path, path, source, stat.st_size, stat.st_mtime_ns, flags)
