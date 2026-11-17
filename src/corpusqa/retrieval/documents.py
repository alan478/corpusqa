"""Field values of chosen documents (`read_fields`) and whole documents (`read_document`).

`read_fields` reads straight from the field rows (exact; no model in the loop): for every
document, each asked field's values, or why it has none (empty, its text is in the body,
absent, not a field of that document's source). Field names are resolved per source;
a name no source of the documents has is an error that names the nearest fields.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from corpusqa import obs
from corpusqa.config import NEAREST_SHOWN
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.fields import FieldIndex
from corpusqa.retrieval.paging import Page, page_of
from corpusqa.retrieval.values import nearest, quote
from corpusqa.store.interfaces import CorpusReader, DocSummary, FieldValue, StoredDocument

type State = Literal["value", "empty", "in_body", "absent", "not_in_source"]
_EMPTY_TYPES = frozenset({"empty", "null"})


@dataclass(frozen=True, slots=True)
class FieldState:
    """One asked field of one document: its values (list items in order), or why none."""

    asked: str
    field: str | None  # the field's name in the document's source (None: it has none)
    state: State
    values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocRow:
    summary: DocSummary
    fields: tuple[FieldState, ...]


@dataclass(frozen=True, slots=True)
class FieldsRead:
    """One page of documents with their asked fields; `names` maps each asked field to its
    name per source (sources without it are left out)."""

    page: Page[DocRow]
    names: Mapping[str, Mapping[str, str]]


def read_fields(
    reader: CorpusReader,
    index: FieldIndex,
    doc_ids: Sequence[str],
    fields: Sequence[str],
    offset: int,
    size: int,
) -> FieldsRead:
    """The asked fields of every document in `doc_ids` (in that order), one page."""
    given = {"documents": len(doc_ids), "fields": list(fields), "offset": offset}
    with obs.span("read_fields", input=given) as span:
        if not fields:
            raise ToolInputError("name at least one field to read.")
        summaries = reader.document_summaries(doc_ids)
        names = _names(index, sorted({s.source for s in summaries}), fields)
        page = page_of(summaries, offset, size)
        rows = doc_rows(reader, page.items, fields, names)
        result = FieldsRead(Page(tuple(rows), page.offset, page.total), names)
        span.update(
            output={"rows": len(rows), "total": page.total},
            metadata={"names": {a: dict(n) for a, n in names.items()}},
        )
    return result


def doc_rows(
    reader: CorpusReader,
    summaries: Sequence[DocSummary],
    asked: Sequence[str],
    names: Mapping[str, Mapping[str, str]],
) -> list[DocRow]:
    """Each document with the state of each asked field (`names`: asked -> source -> name)."""
    wanted = sorted({name for per_source in names.values() for name in per_source.values()})
    rows: dict[tuple[str, str], list[FieldValue]] = {}
    if wanted and summaries:
        for row in reader.document_fields([s.doc_id for s in summaries], wanted):
            if not row.is_noise:
                rows.setdefault((row.doc_id, row.field), []).append(row)
    return [
        DocRow(s, tuple(_state(s, a, names.get(a, {}).get(s.source), rows) for a in asked))
        for s in summaries
    ]


def _state(
    summary: DocSummary,
    asked: str,
    field: str | None,
    rows: Mapping[tuple[str, str], list[FieldValue]],
) -> FieldState:
    if field is None:
        return FieldState(asked, None, "not_in_source")
    found = rows.get((summary.doc_id, field), [])
    values = tuple(
        r.value for r in found if r.value is not None and r.value_type not in _EMPTY_TYPES
    )
    if values:
        return FieldState(asked, field, "value", values)
    if found:
        return FieldState(asked, field, "empty")
    if field in summary.body_fields:
        return FieldState(asked, field, "in_body")
    return FieldState(asked, field, "absent")


def _names(
    index: FieldIndex, sources: Sequence[str], fields: Sequence[str]
) -> dict[str, dict[str, str]]:
    """asked field -> {source: its name there}; ToolInputError for a field no source has."""
    infos = [index.source(s) for s in sources]
    names: dict[str, dict[str, str]] = {}
    for asked in fields:
        per_source = {}
        for info in infos:
            row = index.find_field(info, asked)
            if row is not None:
                per_source[info.name] = row.field
        if infos and not per_source:
            close = nearest(asked, [r for i in infos for r in i.fields], lambda r: r.field)
            shown = ", ".join(quote(r.field) for r in close[:NEAREST_SHOWN])
            where = ", ".join(quote(i.name) for i in infos)
            raise ToolInputError(
                f"no field {quote(asked)} in the documents' sources ({where}); nearest: "
                f"{shown}. list_fields lists every field of a source."
            )
        names[asked] = per_source
    return names


def read_document(reader: CorpusReader, doc_id: str) -> StoredDocument:
    """The whole document: its facts, every field row and the body. Raises ToolInputError
    for an id the index does not have."""
    with obs.span("read_document", input={"doc_id": doc_id}) as span:
        doc = reader.document(doc_id)
        if doc is None:
            raise ToolInputError(f"no document {quote(doc_id)} in the index.")
        span.update(
            output={"source": doc.summary.source, "title": doc.summary.display_title},
            metadata={"body_chars": len(doc.body), "field_rows": len(doc.fields)},
        )
    return doc
