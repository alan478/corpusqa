"""What the store accepts and offers, independent of the database engine.

Ingest builds these rows and writes them through `BuildWriter`; nothing outside
`store/` sees SQL. The row types live here, not in `ingest`, because imports go down
only: `ingest` uses `store`, never the reverse.
"""

from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


class StoreError(RuntimeError):
    """The index on disk cannot be used (incomplete build, wrong schema version)."""


# The years a date must fall in to count as a date (the build's window), in `meta`.
META_DATE_WINDOW_MIN_YEAR = "date_window_min_year"
META_DATE_WINDOW_MAX_YEAR = "date_window_max_year"


@dataclass(frozen=True, slots=True)
class FieldRow:
    """One metadata value of a document (one row per list item)."""

    field: str
    field_norm: str
    seq: int
    role: str  # title | meta
    value: str | None
    value_type: str  # str | int | float | bool | null | empty | json
    value_norm: str | None = None
    value_num: float | None = None
    value_iso: str | None = None
    date_precision: str | None = None
    # Inside the date window, when value_iso is set; a date outside counts as no date.
    date_in_range: bool | None = None
    date_format: str | None = None  # iso | epoch | text, when value_iso is set
    in_list: bool = False  # the field's value is a JSON list (one row per item)
    is_noise: bool = False


@dataclass(frozen=True, slots=True)
class DocumentRow:
    """One ingested file, with its field rows."""

    doc_id: str
    source: str
    rel_path: str
    file_stem: str
    title: str
    title_field: str | None
    title_is_weak: bool
    display_title: str
    body: str
    body_fields: tuple[str, ...]
    created: str | None
    created_field: str | None
    created_rule: str | None
    updated: str | None
    updated_field: str | None
    updated_rule: str | None
    content_hash: str
    size_bytes: int
    mtime_ns: int
    flags: tuple[str, ...]
    raw: bytes  # the file's bytes; the store compresses them
    fields: tuple[FieldRow, ...]


@dataclass(frozen=True, slots=True)
class DocumentDates:
    """created / updated of one document as decided after the whole corpus was read."""

    doc_id: str
    created: str | None
    created_field: str | None
    created_rule: str | None
    updated: str | None
    updated_field: str | None
    updated_rule: str | None
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuarantineRow:
    """A file that did not become a document, and why."""

    rel_path: str
    source: str
    reason: str
    detail: str | None = None
    doc_id: str | None = None


@dataclass(frozen=True, slots=True)
class StageRow:
    """Timing and counts of one ingest stage."""

    stage: str
    started_at: str
    finished_at: str
    counts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FieldValue:
    """One stored field row, as the corpus-wide stage (S8) reads it back."""

    source: str
    field: str
    field_norm: str
    doc_id: str
    seq: int
    role: str
    value: str | None
    value_type: str
    value_norm: str | None
    value_num: float | None
    value_iso: str | None
    in_list: bool
    is_noise: bool


@dataclass(frozen=True, slots=True)
class DocFacts:
    """What the corpus-wide stage (S8) reads about each stored document (no text)."""

    doc_id: str
    source: str
    rel_path: str
    file_stem: str
    title: str
    title_field: str | None
    title_is_weak: bool
    display_title: str
    body_fields: tuple[str, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TitleUpdate:
    """A title found weak once the whole corpus was read (S8, weak-title rule a)."""

    doc_id: str
    display_title: str
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValueNormUpdate:
    """A new matching form for one field row (S8: people's names)."""

    doc_id: str
    field: str
    seq: int
    value_norm: str | None


@dataclass(frozen=True, slots=True)
class FieldProfileRow:
    """What S8 learned about one field of one source (see schema.field_profile)."""

    source: str
    field: str
    field_norm: str
    synonym_group: int | None
    n_docs: int
    presence: float
    n_values: int
    n_distinct: int
    list_ratio: float
    type_mix: Mapping[str, int]
    kind: str
    kind_confidence: float
    title_ratio: float
    content_ratio: float
    noise_ratio: float
    avg_len: float
    max_len: int
    top_values: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class FieldConditionRow:
    """Field `field` is (near) always present when `given_field` has one of `present_when`."""

    source: str
    field: str
    given_field: str
    present_when: tuple[str, ...]
    strength: float


@dataclass(frozen=True, slots=True)
class SourceProfileRow:
    """Per-source summary (see schema.source_profile)."""

    source: str
    n_docs: int
    title_fields: Mapping[str, int]
    content_fields: Mapping[str, int]
    own_id_fields: tuple[str, ...]
    key_prefixes: tuple[str, ...]
    date_fields: tuple[str, ...]
    example_doc_ids: tuple[str, ...]
    path_depths: Mapping[str, int]
    stem_title_match: float


@dataclass(frozen=True, slots=True)
class DocKeyRow:
    """One identifier a document owns."""

    doc_id: str
    key_norm: str
    kind: str  # field:<name> | stem | stem_key | folder_number
    is_hub: bool


@dataclass(frozen=True, slots=True)
class LinkRow:
    """One reference value of a document and the target it resolved to (if any)."""

    doc_id: str
    field: str
    raw: str
    key_norm: str
    target_doc_id: str | None
    method: str  # key | stem | folder_number | unique_title | self | none
    target_is_hub: bool


@dataclass(frozen=True, slots=True)
class Integrity:
    """What the finalize stage (S10) checks before publishing."""

    quick_check: str  # "ok" when the database file is sound
    foreign_key_violations: int
    table_rows: Mapping[str, int]

    @property
    def ok(self) -> bool:
        return self.quick_check == "ok" and self.foreign_key_violations == 0


@dataclass(frozen=True, slots=True)
class FieldText:
    """One metadata value of a document as text (one per list item), for a chunk header."""

    field: str
    value: str


@dataclass(frozen=True, slots=True)
class DocumentText:
    """What chunking reads from a stored document."""

    doc_id: str
    source: str
    display_title: str
    created: str | None  # a date inside the window, or None
    created_rule: str | None
    body: str
    body_fields: tuple[str, ...]
    # Values of the fields asked for (`document_texts(header_fields=...)`), in the order
    # asked, list items in list order; () when none were asked for.
    fields: tuple[FieldText, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """One chunk: a slice of a document's body, with its context header."""

    chunk_id: int
    doc_id: str
    source: str
    ord: int  # 0..n within the document
    kind: str  # body
    level: str  # how the split cut it (see ingest.chunking)
    header: str  # '' when headers are off
    text: str
    n_tokens: int
    char_start: int | None
    char_end: int | None


@dataclass(frozen=True, slots=True)
class EmbeddingRow:
    """The vector of one chunk."""

    chunk_id: int
    model: str
    dims: int
    vector: bytes  # float32 little-endian, L2-normalised


@dataclass(frozen=True, slots=True)
class ChunkText:
    """What the embedding stage (S9) reads about one chunk."""

    chunk_id: int
    doc_id: str
    source: str
    header: str
    text: str


@dataclass(frozen=True, slots=True)
class EmbedCoverage:
    """Chunks and embedded chunks per source."""

    chunks: Mapping[str, int]
    embedded: Mapping[str, int]

    @property
    def n_chunks(self) -> int:
        return sum(self.chunks.values())

    @property
    def n_embedded(self) -> int:
        return sum(self.embedded.values())

    @property
    def share(self) -> float:
        """Embedded share of all chunks (1.0 when there are none: nothing is missing)."""
        return self.n_embedded / self.n_chunks if self.n_chunks else 1.0

    def by_source(self) -> dict[str, float]:
        return {
            source: round(self.embedded.get(source, 0) / n, 4) if n else 1.0
            for source, n in sorted(self.chunks.items())
        }


class BuildWriter(Protocol):
    """Writes one index build. Nothing is visible to readers until `finish`."""

    def add_documents(self, docs: Sequence[DocumentRow]) -> None:
        """Write documents and their field rows in one transaction."""
        ...

    def set_date_window(self, min_year: int, max_year: int) -> None:
        """Recompute `date_in_range` of every dated field row for this window of years."""
        ...

    def update_dates(self, rows: Sequence[DocumentDates]) -> None:
        """Replace created / updated (with provenance) and flags of these documents."""
        ...

    def add_quarantine(self, rows: Sequence[QuarantineRow]) -> None: ...

    def add_stage(self, row: StageRow) -> None: ...

    def set_meta(self, items: Mapping[str, str]) -> None: ...

    # Corpus-wide stage (S8): read back what was written, then add what it learned.
    def field_values(self) -> Iterator[FieldValue]:
        """Every field row, ordered by source, field, doc_id, seq. Do not write while
        iterating."""
        ...

    def doc_facts(self) -> Iterator[DocFacts]:
        """Every document (no text), in the order they were stored."""
        ...

    def update_titles(self, rows: Sequence[TitleUpdate]) -> None:
        """Mark these titles weak, with their new display title and flags."""
        ...

    def update_value_norms(self, rows: Sequence[ValueNormUpdate]) -> None: ...

    def add_field_profiles(
        self, fields: Sequence[FieldProfileRow], conditions: Sequence[FieldConditionRow]
    ) -> None: ...

    def add_source_profiles(self, rows: Sequence[SourceProfileRow]) -> None: ...

    def add_doc_keys(self, rows: Sequence[DocKeyRow]) -> None: ...

    def add_links(self, rows: Sequence[LinkRow]) -> None: ...

    # Chunks (S6): read the final documents back, then write their chunks.
    def field_profiles(self) -> list[FieldProfileRow]:
        """Every field profile row S8 wrote, ordered by source, field."""
        ...

    def document_texts(
        self, header_fields: Mapping[str, Sequence[str]] | None = None
    ) -> Iterator[DocumentText]:
        """Every document's chunking input, in the order they were stored. Safe to write
        chunks between items (the documents are read in pages). With `header_fields`
        (source -> field names), each document's `fields` holds the values of its source's
        fields, in that order: metadata only, no noise, no empty or nested values."""
        ...

    def add_chunks(self, rows: Sequence[ChunkRow]) -> None: ...

    def finish_chunks(self) -> None:
        """After the last chunk: build the keyword index and each document's n_chunks."""
        ...

    def integrity(self) -> Integrity:
        """Check the database before it is published (S10)."""
        ...

    def finish(self) -> Path:
        """Mark the build complete and publish it; returns the published path."""
        ...

    def abort(self) -> None:
        """Close and delete the unfinished build."""
        ...


class DocumentReader(Protocol):
    """Read access to a published index."""

    def meta(self) -> dict[str, str]:
        """The build facts (schema and pipeline version, fingerprint, counts, built_at)."""
        ...

    def doc_ids(self) -> frozenset[str]: ...

    def field_profiles(self) -> list[FieldProfileRow]:
        """Every field profile row, ordered by source, field."""
        ...

    def document_texts(
        self, header_fields: Mapping[str, Sequence[str]] | None = None
    ) -> Iterator[DocumentText]:
        """Every document's chunking input, in the order they were stored (`header_fields`
        as in `BuildWriter.document_texts`)."""
        ...


# ---------------------------------------------------------------- the agent's tools
@dataclass(frozen=True, slots=True)
class DocSummary:
    """What the agent's tools show about a document besides its text."""

    doc_id: str
    source: str
    display_title: str
    created: str | None  # a date inside the window, or None (no usable date)
    created_rule: str | None
    updated: str | None
    updated_rule: str | None
    body_fields: tuple[str, ...]  # the fields whose text is in the body (they have no rows)


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """One whole document as the agent reads it: its facts, body and every field row."""

    summary: DocSummary
    title: str
    title_field: str | None
    body: str
    flags: tuple[str, ...]
    fields: tuple[FieldValue, ...]  # every field row, noise included, in file key order


@dataclass(frozen=True, slots=True)
class ValueCount:
    """One distinct value of a field (by its matching form `value_norm`) and how many
    documents hold it, with each spelling as stored (most documents first)."""

    value_norm: str
    n_docs: int
    spellings: tuple[tuple[str, int], ...]

    @property
    def shown(self) -> str:
        """The spelling most documents use."""
        return self.spellings[0][0]


@dataclass(frozen=True, slots=True)
class FieldCoverage:
    """How the documents of one source hold one field; each document is in one group."""

    source_docs: int
    with_value: int  # a row with a value (not empty or null; noise rows ignored)
    empty: int  # rows, but none with a value (an empty string or list, or null)
    in_body: int  # no row, but the field's text is in the body
    # Of `with_value`: documents with a date value outside the window (it counts as no
    # date, so date ranges do not match it). Not a group of its own.
    dates_outside: int = 0

    @property
    def absent(self) -> int:
        return self.source_docs - self.with_value - self.empty - self.in_body


@dataclass(frozen=True, slots=True)
class SourceDates:
    """Each source's documents and the range of their created dates (every created date is
    inside the window: a date outside counts as no date)."""

    source: str
    n_docs: int
    n_dated: int  # created is set
    first: str | None  # earliest created
    last: str | None


type MatchKind = Literal["norm_in", "value_in", "num_range", "iso_range", "present", "missing"]


@dataclass(frozen=True, slots=True)
class FieldMatch:
    """One resolved filter condition on `field`: the rows that make a document match.

    norm_in / value_in: a row's value_norm / value is one of `values`; num_range:
    low <= value_num <= high; iso_range: low <= value_iso < high (ISO text) and the date is
    inside the window (one outside counts as no date); a bound of None is open. present: a
    row with a value, or the field's text in the body; missing: neither. Noise rows never
    match.
    """

    field: str
    kind: MatchKind
    values: tuple[str, ...] = ()
    low: float | str | None = None
    high: float | str | None = None


class CorpusReader(Protocol):
    """What the agent's tools read from a published index (each call its own connection)."""

    def doc_ids(self) -> frozenset[str]: ...

    def field_profiles(self) -> list[FieldProfileRow]: ...

    def field_conditions(self) -> list[FieldConditionRow]:
        """Every conditional-presence row, ordered by source, field, given_field."""
        ...

    def source_dates(self) -> list[SourceDates]:
        """Every source in name order."""
        ...

    def date_window(self) -> tuple[int, int] | None:
        """The (first, last) year a date had to fall in to count, as the build learned it;
        None when the index does not say."""
        ...

    def document_summaries(self, doc_ids: Sequence[str] | None = None) -> list[DocSummary]:
        """Those documents in the order given (unknown ids left out); None: every document,
        in the order they were stored."""
        ...

    def document(self, doc_id: str) -> StoredDocument | None: ...

    def value_counts(self, source: str, field: str) -> list[ValueCount]:
        """Every distinct value of the field in the source (rows with a value, no noise),
        most documents first, then by value."""
        ...

    def field_coverage(self, source: str, field: str) -> FieldCoverage: ...

    def filter_doc_ids(
        self,
        source: str,
        matches: Sequence[FieldMatch],
        created_from: str | None = None,
        created_before: str | None = None,
    ) -> list[str]:
        """Every document of the source that meets all matches (AND) and, when a bound is
        given, has a created date in [created_from, created_before); newest first
        (no date last), then by doc_id."""
        ...

    def document_fields(
        self, doc_ids: Sequence[str], fields: Sequence[str] | None = None
    ) -> list[FieldValue]:
        """The field rows of these documents (only `fields` when given), noise included,
        by document in the order given, then in file key order."""
        ...

    def key_owners(self, key_norm: str) -> list[DocKeyRow]:
        """The documents that own this key (any kind), in stored order."""
        ...

    def links_to(self, key_norm: str, targets: Sequence[str]) -> list[LinkRow]:
        """Links whose key is this key, or holds it as a substring, or that resolved to one
        of `targets`; by referring document in stored order."""
        ...


class ChunkReader(Protocol):
    """Read access to the chunk tables of a published index."""

    def chunk_doc_ids(self) -> dict[int, str]:
        """chunk_id -> doc_id of every chunk."""
        ...

    @property
    def exact_phrases(self) -> bool:
        """The keyword side matches quoted text and key-shaped tokens as phrases (config
        SEARCH_EXACT_PHRASES); False: every word on its own."""
        ...

    def keyword_expression(self, query: str) -> str | None:
        """The keyword query `bm25` runs for `query` (for traces); None when it has no
        words (then nothing matches)."""
        ...

    def keyword_matches(self, query: str, chunk_ids: Set[int]) -> set[int]:
        """Which of `chunk_ids` the keyword query of `query` matches, at any rank."""
        ...

    def bm25(
        self, query: str, limit: int, chunk_ids: Set[int] | None = None
    ) -> list[tuple[int, float]]:
        """The best `limit` chunks for the parts of `query` by BM25 (any part may match: a
        word, or with `exact_phrases` a phrase), as (chunk_id, score), best first; higher
        scores are better, equal scores go to the lower chunk id (so the best `limit` is
        the start of any longer list). With `chunk_ids`, only those chunks are ranked (the
        best `limit` among them, not a cut of the global list)."""
        ...

    def chunk_texts(self, chunk_ids: Sequence[int]) -> list[ChunkText]:
        """These chunks, in the order given (ids with no chunk are left out)."""
        ...

    def embeddings(self) -> list[tuple[int, bytes]]:
        """(chunk_id, vector bytes) of every embedded chunk, in chunk_id order."""
        ...

    def embedding_models(self) -> set[tuple[str, int]]:
        """The (model, dims) pairs of the stored vectors."""
        ...

    def count_chunks(self) -> int: ...


class EmbeddingWriter(Protocol):
    """Adds vectors to a published index (S9, and its resume on a later start)."""

    def unembedded(self, after_chunk_id: int, limit: int) -> list[ChunkText]:
        """Up to `limit` chunks without a vector whose id is above `after_chunk_id`, in id
        order (so documents and their chunks come in order)."""
        ...

    def chunk_texts(self, chunk_ids: Sequence[int]) -> list[ChunkText]:
        """These chunks, in the order given."""
        ...

    def add_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        """Write vectors in one transaction (a checkpoint: they survive a crash)."""
        ...

    def embedding_models(self) -> set[tuple[str, int]]: ...

    def coverage(self) -> EmbedCoverage: ...

    def set_meta(self, items: Mapping[str, str]) -> None: ...

    def add_stage(self, row: StageRow) -> None: ...

    def close(self) -> None: ...


class VectorCache(Protocol):
    """Vectors by key (a hash of everything that decides the vector), kept across builds."""

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """The cached vectors of these keys (missing keys are left out)."""
        ...

    def put_many(self, model: str, dims: int, items: Sequence[tuple[str, bytes]]) -> None: ...

    def close(self) -> None: ...
