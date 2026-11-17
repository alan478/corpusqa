"""Data passed between the ingest stages (design.md §4). Rows the store writes live in
`corpusqa.store.interfaces`; these are the stages' own inputs and outputs.

Each stage returns its own `flags` (what happened to the document); the pipeline joins
them into the stored document's flag list.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpusqa.store.interfaces import FieldRow, QuarantineRow


@dataclass(frozen=True, slots=True)
class FileRef:
    """S1: one candidate document file."""

    rel_path: str  # POSIX path under CORPUS_DIR
    path: Path  # where to read it
    source: str
    size: int
    mtime_ns: int
    flags: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.rel_path.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return Path(self.name).stem


@dataclass(frozen=True, slots=True)
class Discovery:
    """S1 output: files in sorted path order, skips counted by kind, files set aside."""

    files: tuple[FileRef, ...]
    skipped: dict[str, int]
    quarantined: tuple[QuarantineRow, ...]


@dataclass(frozen=True, slots=True)
class ParsedDoc:
    """S2 output: the decoded object and its contract keys."""

    file: FileRef
    doc_id: str
    obj: dict[str, Any]
    title_field: str | None
    content_fields: tuple[str, ...]
    content_hash: str
    raw: bytes
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Unit:
    """One piece of body text, labelled with the field it came from."""

    label: str
    text: str
    origin_field: str
    # "label: value" lines from sibling fields of a numbered family, shown above the text.
    part_headers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CoreDoc:
    """S3 output: title and body units."""

    title: str
    title_field: str | None  # None when the fallback title was used
    title_is_weak: bool
    display_title: str
    units: tuple[Unit, ...]
    # Keys whose values became body text; they get no field rows.
    body_keys: frozenset[str]
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocDates:
    """created / updated with where they came from.

    rule: order | name_hint (from the usable dates, each inside the window), or file_stem.
    """

    created: str | None = None
    created_field: str | None = None
    created_rule: str | None = None
    updated: str | None = None
    updated_field: str | None = None
    updated_rule: str | None = None


@dataclass(frozen=True, slots=True)
class FieldsResult:
    """S4 output: field rows and the document's dates."""

    rows: tuple[FieldRow, ...]
    dates: DocDates
    flags: tuple[str, ...]
    # (field, iso) of every scalar date value in file key order, inside the window or not.
    candidates: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CleanBody:
    """S5 output: the labelled body text and its unit labels in order."""

    body: str
    labels: tuple[str, ...]
    flags: tuple[str, ...]
