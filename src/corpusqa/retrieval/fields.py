"""What the profile learned about each source and field, for the agent's tools.

`FieldIndex` is built once from the index (immutable, shared by every request). It
resolves the model's source and field names (exact, else the one folded or loose match,
else ToolInputError with the nearest names), picks each source's common fields for the
system prompt's catalog (the same rule as the chunk header fields), and serves
`list_fields` and `field_values`: every field or value, paged, the ones near the model's
words first.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from corpusqa import obs
from corpusqa.config import (
    CATALOG_FIELD_MAX_CONTENT_RATIO,
    CATALOG_FIELD_MAX_NOISE_RATIO,
    CATALOG_FIELD_MAX_TITLE_RATIO,
    CATALOG_FIELD_MIN_PRESENCE,
    NEAREST_SHOWN,
)
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.paging import Page, page_of
from corpusqa.retrieval.values import is_near, loose, nearest, nearness, quote, words
from corpusqa.store.interfaces import (
    CorpusReader,
    FieldConditionRow,
    FieldCoverage,
    FieldProfileRow,
    SourceDates,
    ValueCount,
)
from corpusqa.store.norms import fold_key
from corpusqa.store.profiles import is_common_field

# A field whose kind the asked words name counts as near, like one shared word.
_KIND_WORD_SCORE = 0.8


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """One source: its documents, created range and fields (most documents first)."""

    name: str
    n_docs: int
    n_dated: int  # documents with a created date (inside the date window)
    first: str | None
    last: str | None
    fields: tuple[FieldProfileRow, ...]


class FieldIndex:
    """Sources and fields as the profile learned them. Holds no per-request state."""

    def __init__(
        self,
        profiles: Sequence[FieldProfileRow],
        dates: Sequence[SourceDates],
        conditions: Sequence[FieldConditionRow] = (),
        date_window: tuple[int, int] | None = None,
    ) -> None:
        # The (first, last) year a date must fall in to count as a date; None: not known.
        self.date_window = date_window
        by_source: dict[str, list[FieldProfileRow]] = {}
        for row in profiles:
            by_source.setdefault(row.source, []).append(row)
        self._sources = {
            d.source: SourceInfo(
                name=d.source,
                n_docs=d.n_docs,
                n_dated=d.n_dated,
                first=d.first,
                last=d.last,
                fields=tuple(
                    sorted(by_source.get(d.source, []), key=lambda r: (-r.n_docs, r.field))
                ),
            )
            for d in sorted(dates, key=lambda d: d.source)
        }
        self._conditions: dict[tuple[str, str], list[FieldConditionRow]] = {}
        for c in conditions:
            self._conditions.setdefault((c.source, c.field), []).append(c)
        self._groups: dict[tuple[str, int], list[str]] = {}
        for row in profiles:
            if row.synonym_group is not None:
                self._groups.setdefault((row.source, row.synonym_group), []).append(row.field)

    @classmethod
    def from_reader(cls, reader: CorpusReader) -> "FieldIndex":
        return cls(
            reader.field_profiles(),
            reader.source_dates(),
            reader.field_conditions(),
            reader.date_window(),
        )

    @property
    def sources(self) -> tuple[SourceInfo, ...]:
        return tuple(self._sources.values())

    def source(self, asked: str) -> SourceInfo:
        """The source the model named: exact, else the one with the same loose form."""
        if asked in self._sources:
            return self._sources[asked]
        same = [s for name, s in self._sources.items() if loose(name) == loose(asked)]
        if len(same) == 1:
            return same[0]
        names = ", ".join(quote(s) for s in self._sources)
        raise ToolInputError(f"no source {quote(asked)}. Sources: {names}.")

    def field(self, source: SourceInfo, asked: str) -> FieldProfileRow:
        """The field of `source` the model named: exact, else the one with the same folded
        name (case, `-`, spaces), else the one with the same loose form."""
        found = self.find_field(source, asked)
        if found is not None:
            return found
        close = nearest(asked, source.fields, _field_name)[:NEAREST_SHOWN]
        raise ToolInputError(
            f"no field {quote(asked)} in source {quote(source.name)}; nearest: "
            f"{', '.join(quote(r.field) for r in close)}. list_fields(source="
            f"{quote(source.name)}, near={quote(asked)}) lists every field."
        )

    def find_field(self, source: SourceInfo, asked: str) -> FieldProfileRow | None:
        """As `field`, but None when no field (or more than one) fits."""
        for same in (
            lambda r: r.field == asked,
            lambda r: fold_key(r.field) == fold_key(asked),
            lambda r: loose(r.field) == loose(asked),
        ):
            hits = [r for r in source.fields if same(r)]
            if len(hits) == 1:
                return hits[0]
            if hits:
                return None
        return None

    def conditions_of(self, row: FieldProfileRow) -> tuple[FieldConditionRow, ...]:
        """When the field applies: 'present when <given_field> is one of ...'."""
        return tuple(self._conditions.get((row.source, row.field), ()))

    def synonyms_of(self, row: FieldProfileRow) -> tuple[str, ...]:
        """Other fields of its source the profile groups with it (renamed copies)."""
        if row.synonym_group is None:
            return ()
        group = self._groups.get((row.source, row.synonym_group), [])
        return tuple(name for name in group if name != row.field)

    def catalog(self) -> list[tuple[SourceInfo, tuple[FieldProfileRow, ...]]]:
        """Per source, its common fields (store/profiles.py rule, CATALOG_* thresholds),
        generator noise left out, most present first, then by name."""
        chosen = []
        for source in self._sources.values():
            rows = [
                r
                for r in source.fields
                if r.noise_ratio < CATALOG_FIELD_MAX_NOISE_RATIO
                and is_common_field(
                    r,
                    min_presence=CATALOG_FIELD_MIN_PRESENCE,
                    max_title_ratio=CATALOG_FIELD_MAX_TITLE_RATIO,
                    max_content_ratio=CATALOG_FIELD_MAX_CONTENT_RATIO,
                )
            ]
            rows.sort(key=lambda r: (-r.presence, r.field))
            chosen.append((source, tuple(rows)))
        return chosen


@dataclass(frozen=True, slots=True)
class FieldsPage:
    """One page of a source's fields; `n_near` of them are near the asked words (those come
    first), None when no words were asked."""

    source: SourceInfo
    page: Page[FieldProfileRow]
    near: str | None
    n_near: int | None


def list_fields(
    index: FieldIndex, source: str, near: str | None, offset: int, size: int
) -> FieldsPage:
    """Every field of the source: near the words first (when given), then by documents."""
    with obs.span("list_fields", input={"source": source, "near": near, "offset": offset}) as span:
        info = index.source(source)
        rows: Sequence[FieldProfileRow] = info.fields
        n_near = None
        if near is not None and near.strip():
            rows, n_near = near_first(info.fields, [_field_nearness(near, r) for r in info.fields])
        page = page_of(rows, offset, size)
        span.update(
            output={"fields": [r.field for r in page.items]},
            metadata={"source": info.name, "total": page.total, "n_near": n_near},
        )
    return FieldsPage(source=info, page=page, near=near, n_near=n_near)


@dataclass(frozen=True, slots=True)
class ValuesPage:
    """One page of a field's distinct values, with how its documents hold the field."""

    source: SourceInfo
    field: FieldProfileRow
    coverage: FieldCoverage
    page: Page[ValueCount]
    near: str | None
    n_near: int | None


def field_values(
    reader: CorpusReader,
    index: FieldIndex,
    source: str,
    field: str,
    near: str | None,
    offset: int,
    size: int,
) -> ValuesPage:
    """Every distinct value of the field with its document count: near the words first
    (when given), then by documents."""
    given = {"source": source, "field": field, "near": near, "offset": offset}
    with obs.span("field_values", input=given) as span:
        info = index.source(source)
        row = index.field(info, field)
        counts: Sequence[ValueCount] = reader.value_counts(info.name, row.field)
        n_near = None
        if near is not None and near.strip():
            counts, n_near = near_first(counts, [nearness(near, c.shown) for c in counts])
        page = page_of(counts, offset, size)
        coverage = reader.field_coverage(info.name, row.field)
        span.update(
            output={"values": [c.shown for c in page.items]},
            metadata={
                "source": info.name,
                "field": row.field,
                "total": page.total,
                "n_near": n_near,
                "coverage": coverage_dict(coverage),
            },
        )
    return ValuesPage(info, row, coverage, page, near, n_near)


def coverage_dict(c: FieldCoverage) -> dict[str, int]:
    """A coverage as span metadata."""
    return {
        "source_docs": c.source_docs,
        "with_value": c.with_value,
        "empty": c.empty,
        "in_body": c.in_body,
        "absent": c.absent,
    }


def near_first[T](items: Sequence[T], scores: Sequence[float]) -> tuple[list[T], int]:
    """The near items first (nearest first), then the rest in their given order; and how
    many are near."""
    order = sorted(
        range(len(items)),
        key=lambda i: (0, -scores[i], i) if is_near(scores[i]) else (1, 0.0, i),
    )
    return [items[i] for i in order], sum(1 for s in scores if is_near(s))


def _field_nearness(near: str, row: FieldProfileRow) -> float:
    """How near a field is to the asked words: by its name, or its kind named as a word
    ("due date" brings date fields)."""
    by_kind = _KIND_WORD_SCORE if row.kind in words(near) else 0.0
    return max(nearness(near, row.field), by_kind)


def _field_name(row: FieldProfileRow) -> str:
    return row.field
