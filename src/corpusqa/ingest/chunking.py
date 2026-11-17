"""Chunking strategies (step C): pure functions from a stored document to chunk drafts.

Every chunk is a contiguous slice of the body: `body[char_start:char_end] == text`, with
no whitespace at either end. The header (source, display title, created date and, when
the spec asks, field values) is kept apart from the text; whoever embeds a chunk joins
the two. Sizes are estimated as
chars / TOKEN_CHARS, so a limit of N tokens is N * TOKEN_CHARS characters.

Strategies (`STRATEGIES`, parameters in `ChunkSpec`):
- `recursive`: the whole body if it fits; else split at body sections, then blank-line
  paragraphs, lines, sentences and words, going deeper only for pieces still over the
  limit, and merge neighbouring pieces of one level back up to the limit. No overlap.
- `fixed`: windows of the limit with an overlap, cut at whitespace.
- `whole`: one chunk per document; a body over the limit falls back to `recursive`.

Nothing here knows a source or field name: sections are the body's own `## <label>`
lines (S5), found from the labels stored with the document.

`chunk_rows` turns a stored document into store rows; the header shows the document's
created date when it has one (always inside the date window: a date outside counts as no
date, so it is never shown).
`embedding_text` is the exact text embedded for a chunk: header, newline, text.

Header fields (`ChunkSpec.header_fields`, default CHUNK_HEADER_FIELDS): per source, the
fields `select_header_fields` picks from the profile (S8); the store reads their values
and `render_header_fields` appends them to the header as `field: value` parts, one line,
within the size caps of config.
"""

import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from corpusqa.config import (
    BODY_LABEL_PREFIX,
    CHUNK_FIXED_OVERLAP_SHARE,
    CHUNK_HEADER_FIELDS,
    CHUNK_HEADER_SEPARATOR,
    CHUNK_MAX_EMBED_TOKENS,
    CHUNK_MAX_TOKENS,
    HEADER_FIELD_KINDS,
    HEADER_FIELD_LIST_ITEMS,
    HEADER_FIELD_MAX_CONTENT_RATIO,
    HEADER_FIELD_MAX_TITLE_RATIO,
    HEADER_FIELD_MIN_PRESENCE,
    HEADER_FIELD_VALUE_CHARS,
    HEADER_FIELDS_MAX_CHARS,
    TOKEN_CHARS,
)
from corpusqa.store.interfaces import ChunkRow, DocumentText, FieldProfileRow, FieldText
from corpusqa.store.profiles import is_common_field

_DATE_CHARS = len("YYYY-MM-DD")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_LINE_BREAK = re.compile(r"\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """What chunking reads from a stored document."""

    source: str
    display_title: str
    created: str | None  # ISO date or date-time; only the date goes into the header
    body: str
    labels: tuple[str, ...] = ()  # documents.body_fields: the `## <label>` sections in order
    fields: tuple[FieldText, ...] = ()  # values shown in the header, in order (header fields)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    text: str
    header: str | None  # None when the spec turns headers off
    char_start: int
    char_end: int
    n_tokens: int  # of `text`
    # How the chunk was cut: whole | section | paragraph | line | sentence | word | char
    # (recursive and whole) or fixed.
    level: str


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """A strategy and its parameters. `max_tokens=None` takes the strategy's default;
    `fields=None` takes CHUNK_HEADER_FIELDS (when the header is on)."""

    strategy: str = "recursive"
    max_tokens: int | None = None
    overlap_tokens: int | None = None  # fixed only; None: CHUNK_FIXED_OVERLAP_SHARE of the size
    header: bool = True
    fields: bool | None = None  # field values in the header (needs the header)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown chunking strategy {self.strategy!r}; use {list(STRATEGIES)}")
        if not 0 < self.limit_tokens <= CHUNK_MAX_EMBED_TOKENS:
            raise ValueError(f"max_tokens must be 1..{CHUNK_MAX_EMBED_TOKENS}")
        if not 0 <= self.overlap < self.limit_tokens:
            raise ValueError("overlap_tokens must be at least 0 and below max_tokens")
        if self.fields and not self.header:
            raise ValueError("header fields need the header: fields=True with header=False")

    @property
    def header_fields(self) -> bool:
        """Whether chunk headers show field values."""
        if self.fields is None:
            return self.header and CHUNK_HEADER_FIELDS
        return self.fields

    @property
    def limit_tokens(self) -> int:
        if self.max_tokens is not None:
            return self.max_tokens
        return CHUNK_MAX_EMBED_TOKENS if self.strategy == "whole" else CHUNK_MAX_TOKENS

    @property
    def overlap(self) -> int:
        if self.strategy != "fixed":
            return 0
        if self.overlap_tokens is None:
            return int(self.limit_tokens * CHUNK_FIXED_OVERLAP_SHARE)
        return self.overlap_tokens


type _Piece = tuple[int, int, str]  # char_start, char_end, level
type Strategy = Callable[[str, tuple[str, ...], ChunkSpec], list[_Piece]]


def chunk_document(doc: ChunkInput, spec: ChunkSpec | None = None) -> list[ChunkDraft]:
    """The document's chunks in body order. An empty body gives no chunks."""
    spec = spec or ChunkSpec()
    header = make_header(doc) if spec.header else None
    return [
        ChunkDraft(
            text=doc.body[start:end],
            header=header,
            char_start=start,
            char_end=end,
            n_tokens=count_tokens(doc.body[start:end]),
            level=level,
        )
        for start, end, level in STRATEGIES[spec.strategy](doc.body, doc.labels, spec)
    ]


def chunk_input(doc: DocumentText, spec: ChunkSpec | None = None) -> ChunkInput:
    """A stored document as chunking input; its field values only when the spec puts them
    in the header."""
    spec = spec or ChunkSpec()
    fields = doc.fields if spec.header_fields else ()
    return ChunkInput(doc.source, doc.display_title, doc.created, doc.body, doc.body_fields, fields)


def chunk_rows(doc: DocumentText, spec: ChunkSpec, first_id: int) -> list[ChunkRow]:
    """The document's chunks as store rows, numbered from `first_id`."""
    return [
        ChunkRow(
            chunk_id=first_id + ord_,
            doc_id=doc.doc_id,
            source=doc.source,
            ord=ord_,
            kind="body",
            level=draft.level,
            header=draft.header or "",
            text=draft.text,
            n_tokens=draft.n_tokens,
            char_start=draft.char_start,
            char_end=draft.char_end,
        )
        for ord_, draft in enumerate(chunk_document(chunk_input(doc, spec), spec))
    ]


def embedding_text(header: str, text: str) -> str:
    """The text embedded for a chunk (design S9): the header line, then the text."""
    return f"{header}\n{text}" if header else text


def make_header(doc: ChunkInput) -> str:
    """One line: source, display title, created date, then the field values (if any)."""
    date = doc.created[:_DATE_CHARS] if doc.created else None
    fields = render_header_fields(doc.fields)
    return CHUNK_HEADER_SEPARATOR.join(
        p for p in (doc.source, doc.display_title, date, fields) if p
    )


def count_tokens(text: str) -> int:
    return math.ceil(len(text) / TOKEN_CHARS)


# ---------------------------------------------------------------- header fields
def select_header_fields(rows: Sequence[FieldProfileRow]) -> dict[str, tuple[str, ...]]:
    """Per source, the fields whose values go into its chunk headers: common in the source,
    rarely its title or body text, of a kind whose values say what a document is about
    (thresholds and kinds in config). Ordered by presence (most common first), then name,
    so two builds agree; sources in name order, a source with no such field left out."""
    chosen: dict[str, list[FieldProfileRow]] = {}
    for row in rows:
        common = is_common_field(
            row,
            min_presence=HEADER_FIELD_MIN_PRESENCE,
            max_title_ratio=HEADER_FIELD_MAX_TITLE_RATIO,
            max_content_ratio=HEADER_FIELD_MAX_CONTENT_RATIO,
        )
        if common and row.kind in HEADER_FIELD_KINDS:
            chosen.setdefault(row.source, []).append(row)
    return {
        source: tuple(r.field for r in sorted(chosen[source], key=lambda r: (-r.presence, r.field)))
        for source in sorted(chosen)
    }


def render_header_fields(fields: Sequence[FieldText]) -> str:
    """The header's fields part: `field: value` per field in input order (list items joined
    by ", "), parts joined by CHUNK_HEADER_SEPARATOR, one line, within the caps of config;
    "" when there are none."""
    return CHUNK_HEADER_SEPARATOR.join(part for part, _ in _field_parts(fields))


def count_header_values(fields: Sequence[FieldText]) -> int:
    """How many of these values `render_header_fields` shows (the caps drop the rest)."""
    return sum(shown for _, shown in _field_parts(fields))


def _field_parts(fields: Sequence[FieldText]) -> list[tuple[str, int]]:
    """(part, values in it) per field: at most HEADER_FIELD_LIST_ITEMS values, each made one
    line and cut at HEADER_FIELD_VALUE_CHARS; whole parts only, while the joined parts stay
    within HEADER_FIELDS_MAX_CHARS."""
    grouped: dict[str, list[str]] = {}
    for item in fields:
        value = _one_line(item.value)[:HEADER_FIELD_VALUE_CHARS].rstrip()
        if value:
            grouped.setdefault(item.field, []).append(value)
    parts: list[tuple[str, int]] = []
    size = 0
    for field, values in grouped.items():
        shown = values[:HEADER_FIELD_LIST_ITEMS]
        part = f"{_one_line(field)}: {', '.join(shown)}"
        size += len(part) + (len(CHUNK_HEADER_SEPARATOR) if parts else 0)
        if size > HEADER_FIELDS_MAX_CHARS:
            break
        parts.append((part, len(shown)))
    return parts


def _one_line(text: str) -> str:
    """Every run of whitespace (newlines too) as one space, none at either end."""
    return _WHITESPACE.sub(" ", text).strip()


# ---------------------------------------------------------------- strategies
def _recursive(body: str, labels: tuple[str, ...], spec: ChunkSpec) -> list[_Piece]:
    """Also `whole`: the same rule with the large default limit, so a body that fits is one
    chunk and only a body over the embedding limit is split."""
    return _recursive_at(body, labels, spec.limit_tokens * TOKEN_CHARS)


def _fixed(body: str, labels: tuple[str, ...], spec: ChunkSpec) -> list[_Piece]:
    size, overlap = spec.limit_tokens * TOKEN_CHARS, spec.overlap * TOKEN_CHARS
    start, stop = _trim(body, 0, len(body))
    pieces: list[_Piece] = []
    while start < stop:
        end = min(start + size, stop)
        if end < stop:  # cut after the last whitespace that keeps the window in size
            cut = next((i for i in range(end, start, -1) if body[i].isspace()), end)
            end = cut
        chunk_start, chunk_end = _trim(body, start, end)
        pieces.append((chunk_start, chunk_end, "fixed"))
        if end >= stop:
            break
        start = _next_word(body, max(chunk_end - overlap, start + 1), end)
    return pieces


STRATEGIES: Mapping[str, Strategy] = {
    "recursive": _recursive,
    "fixed": _fixed,
    "whole": _recursive,
}


# ---------------------------------------------------------------- recursive splitting
def _recursive_at(body: str, labels: tuple[str, ...], limit: int) -> list[_Piece]:
    start, end = _trim(body, 0, len(body))
    if start == end:
        return []
    if end - start <= limit:
        return [(start, end, "whole")]
    return _split(body, start, end, 0, limit, labels)


def _split(
    body: str, start: int, end: int, depth: int, limit: int, labels: tuple[str, ...]
) -> list[_Piece]:
    """Pieces of body[start:end] (over the limit) at the first level, from `depth`, that
    cuts it in two or more; pieces still too big go one level deeper; small neighbours
    are merged up to the limit."""
    for level_index in range(depth, len(_LEVELS)):
        level, cutter = _LEVELS[level_index]
        spans = cutter(body, start, end, labels)
        if len(spans) > 1:
            break
    else:  # one word longer than the limit: cut it into pieces of the limit
        return [(s, min(s + limit, end), "char") for s in range(start, end, limit)]
    pieces: list[_Piece] = []
    open_start = open_end = -1  # the merge buffer: body[open_start:open_end]
    for s, e in spans:
        if e - s > limit:
            if open_start >= 0:
                pieces.append((open_start, open_end, level))
                open_start = -1
            pieces.extend(_split(body, s, e, level_index + 1, limit, labels))
        elif open_start >= 0 and e - open_start <= limit:
            open_end = e
        else:
            if open_start >= 0:
                pieces.append((open_start, open_end, level))
            open_start, open_end = s, e
    if open_start >= 0:
        pieces.append((open_start, open_end, level))
    return pieces


def _sections(body: str, start: int, end: int, labels: tuple[str, ...]) -> list[tuple[int, int]]:
    """Cut before each line that is exactly `## <label>` for one of the document's labels."""
    heads = {BODY_LABEL_PREFIX + label for label in labels}
    cuts = [start]
    pos = start
    while pos < end:
        line_end = body.find("\n", pos, end)
        line_end = end if line_end == -1 else line_end
        if pos > start and body[pos:line_end] in heads:
            cuts.append(pos)
        pos = line_end + 1
    cuts.append(end)
    return _trimmed(body, list(itertools.pairwise(cuts)))


def _at(
    pattern: re.Pattern[str],
) -> Callable[[str, int, int, tuple[str, ...]], list[tuple[int, int]]]:
    """A cutter that splits at every match of `pattern` (the match itself is dropped)."""

    def cut(body: str, start: int, end: int, labels: tuple[str, ...]) -> list[tuple[int, int]]:
        spans, pos = [], start
        for match in pattern.finditer(body, start, end):
            spans.append((pos, match.start()))
            pos = match.end()
        spans.append((pos, end))
        return _trimmed(body, spans)

    return cut


_LEVELS: tuple[
    tuple[str, Callable[[str, int, int, tuple[str, ...]], list[tuple[int, int]]]], ...
] = (
    ("section", _sections),
    ("paragraph", _at(_PARAGRAPH_BREAK)),
    ("line", _at(_LINE_BREAK)),
    ("sentence", _at(_SENTENCE_END)),
    ("word", _at(_WHITESPACE)),
)


# ---------------------------------------------------------------- helpers
def _trim(body: str, start: int, end: int) -> tuple[int, int]:
    """The span without whitespace at either end (empty spans collapse to start == end)."""
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


def _trimmed(body: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    trimmed = (_trim(body, s, e) for s, e in spans)
    return [(s, e) for s, e in trimmed if s < e]


def _next_word(body: str, pos: int, limit: int) -> int:
    """The start of the first whole word at or after `pos` (never past `limit`)."""
    if 0 < pos < limit and not body[pos - 1].isspace():
        while pos < limit and not body[pos].isspace():
            pos += 1
    while pos < limit and body[pos].isspace():
        pos += 1
    return pos
