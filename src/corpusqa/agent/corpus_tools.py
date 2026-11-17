"""The agent's corpus tools: what each takes, the text each returns, and the session.

The logic lives in `retrieval/` (deterministic code over the store); here are the Tool
adapters: Pydantic argument models (their JSON schemas are in `prompts/tools.json`) and
compact text with handles (D# for a document, R# for a result set) and a paging line on
every list ("rows 1-50 of 278; next: offset=50"). Nothing is cut: each tool pages. A
search page whose ranking has not reached every chunk in scope gives no total
("documents 1-8; next: offset=8"): the next page searches deeper.

`CorpusTools` holds what every request shares (the index readers, the field index, the
search, the field catalog for the system prompt) and makes one `CorpusToolSession` per
request: its own handles, so nothing is shared between requests. A session matches
`agent.session.ToolSession`: `toolbox` (the corpus tools and `finish`),
`resolve_citations` (D# -> its id, R# -> every id of the set, a known id kept) and
`clean_answer` (each issued D# in the answer text -> its title, source and date; each R#
-> "the 14 matching hubspot records").
"""

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from corpusqa import obs
from corpusqa.agent.handles import CleanedAnswer, DocName, Handles
from corpusqa.agent.tools import Tool, ToolBox, finish_tool
from corpusqa.config import (
    CATALOG_FIELD_MAX_CONTENT_RATIO,
    CATALOG_FIELD_MAX_TITLE_RATIO,
    CATALOG_TOP_VALUES,
    CONTAINS_VALUES_SHOWN,
    FIELD_VALUES_PAGE_ROWS,
    FILTER_PAGE_ROWS,
    KEY_LOOKUP_PAGE_ROWS,
    LIST_FIELDS_PAGE_ROWS,
    LIST_FIELDS_TOP_VALUES,
    PREVIEW_MAX_AVG_CHARS,
    READ_FIELDS_PAGE_ROWS,
    SEARCH_PAGE_DOCS,
)
from corpusqa.llm.errors import LLMError
from corpusqa.llm.interfaces import Embedder
from corpusqa.prompts import ToolSchemas, load_tool_schemas
from corpusqa.retrieval.documents import DocRow, FieldState, doc_rows, read_document, read_fields
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.fields import FieldIndex, SourceInfo, field_values, list_fields
from corpusqa.retrieval.filtering import SHOWS_VALUES, Applied, Condition, Op, run_filter
from corpusqa.retrieval.keys import lookup_key
from corpusqa.retrieval.paging import Page, page_of
from corpusqa.retrieval.search import CorpusSearch, Scope, SearchResult
from corpusqa.retrieval.values import Resolved, date_bounds, quote, show_counts
from corpusqa.store.interfaces import (
    ChunkReader,
    CorpusReader,
    DocSummary,
    FieldCoverage,
    FieldProfileRow,
)
from corpusqa.store.vectors import VectorIndex

if TYPE_CHECKING:
    from corpusqa.agent.session import ToolSession

logger = logging.getLogger(__name__)

SEARCH, FILTER, READ_FIELDS, READ_DOCUMENT = "search", "filter", "read_fields", "read_document"
LOOKUP_KEY, LIST_FIELDS, FIELD_VALUES = "lookup_key", "list_fields", "field_values"
CORPUS_TOOL_NAMES = (
    SEARCH, FILTER, READ_FIELDS, READ_DOCUMENT, LOOKUP_KEY, LIST_FIELDS, FIELD_VALUES,
)  # fmt: skip
_SEP = " · "
# Exact search: the mark of a document only the dense side found (no keyword match).
_MEANING_ONLY = "meaning only"


# ---------------------------------------------------------------- arguments
def _as_text(value: object) -> object:
    """A JSON number or boolean the model sent where text is expected, as text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return value


type Text = Annotated[str, BeforeValidator(_as_text)]
type Offset = Annotated[int, Field(ge=0)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class SearchArgs(_Args):
    query: Annotated[str, Field(min_length=1)]
    source: str | None = None
    created_from: Text | None = None
    created_to: Text | None = None
    within: str | None = None
    offset: Offset = 0


class ConditionArgs(_Args):
    field: str
    op: Op
    value: Text | None = None
    values: list[Text] | None = None
    from_: Text | None = Field(default=None, alias="from")
    to: Text | None = None

    def condition(self) -> Condition:
        values = ([] if self.value is None else [self.value]) + list(self.values or [])
        return Condition(self.field, self.op, tuple(values), self.from_, self.to)


class FilterArgs(_Args):
    source: str
    conditions: list[ConditionArgs] = Field(default_factory=list)
    created_from: Text | None = None
    created_to: Text | None = None
    offset: Offset = 0


class ReadFieldsArgs(_Args):
    documents: Annotated[list[str], Field(min_length=1)]
    fields: Annotated[list[str], Field(min_length=1)]
    offset: Offset = 0


class ReadDocumentArgs(_Args):
    document: str


class LookupKeyArgs(_Args):
    key: Annotated[str, Field(min_length=1)]
    offset: Offset = 0


class ListFieldsArgs(_Args):
    source: str
    near: str | None = None
    offset: Offset = 0


class FieldValuesArgs(_Args):
    source: str
    field: str
    near: str | None = None
    offset: Offset = 0


# ---------------------------------------------------------------- shared parts, sessions
class CorpusTools:
    """What every request's tools share, read-only: the index readers, the field index,
    the search, the known ids and the field catalog. Calling it makes a new session."""

    def __init__(
        self,
        reader: CorpusReader,
        chunks: ChunkReader,
        vectors: VectorIndex | None = None,
        embedder: Embedder | None = None,
        *,
        schemas: ToolSchemas | None = None,
    ) -> None:
        self.reader = reader
        self.index = FieldIndex.from_reader(reader)
        summaries = reader.document_summaries()
        self.summaries: Mapping[str, DocSummary] = {s.doc_id: s for s in summaries}
        self.known_doc_ids = frozenset(self.summaries)
        self.search = CorpusSearch(chunks, summaries, vectors)
        # Dense search needs both: the vectors, and the embedder that made them.
        self.embedder = embedder if self.search.has_vectors else None
        self.schemas = schemas or load_tool_schemas()
        self.catalog = field_catalog(self.index)

    def __call__(self) -> "CorpusToolSession":
        return CorpusToolSession(self)


def corpus_session_factory(
    reader: CorpusReader,
    chunks: ChunkReader,
    vectors: VectorIndex | None = None,
    embedder: Embedder | None = None,
    *,
    schemas: ToolSchemas | None = None,
) -> CorpusTools:
    """The session factory over one index: `factory()` is one request's session, and
    `factory.catalog` the field catalog for the system prompt."""
    return CorpusTools(reader, chunks, vectors, embedder, schemas=schemas)


class CorpusToolSession:
    """One request's corpus tools and handles (never shared between requests)."""

    def __init__(self, tools: CorpusTools) -> None:
        self._tools = tools
        self.handles = Handles(tools.known_doc_ids)
        self._vectors: dict[str, list[float] | None] = {}
        self._vectors_lock = threading.Lock()
        schemas = tools.schemas
        self._toolbox = ToolBox(
            [
                Tool(schemas.get(SEARCH), SearchArgs, self._search),
                Tool(schemas.get(FILTER), FilterArgs, self._filter),
                Tool(schemas.get(READ_FIELDS), ReadFieldsArgs, self._read_fields),
                Tool(schemas.get(READ_DOCUMENT), ReadDocumentArgs, self._read_document),
                Tool(schemas.get(LOOKUP_KEY), LookupKeyArgs, self._lookup_key),
                Tool(schemas.get(LIST_FIELDS), ListFieldsArgs, self._list_fields),
                Tool(schemas.get(FIELD_VALUES), FieldValuesArgs, self._field_values),
                finish_tool(schemas),
            ]
        )

    @property
    def toolbox(self) -> ToolBox:
        return self._toolbox

    def resolve_citations(self, citations: Sequence[str]) -> list[str]:
        """dataset_doc_uuids behind the model's citations: D# -> its id, R# -> every id of
        the set, a known id kept, anything else dropped; deduped, first-seen order."""
        return self.handles.resolve(citations)

    def clean_answer(self, answer: str, question: str) -> CleanedAnswer:
        """The answer with each D# and R# this session issued named by what it is (see
        `Handles.name_in_text`); tokens it never issued, or that the question holds, stay."""
        return self.handles.name_in_text(answer, self._doc_name, self._set_name, question=question)

    def _doc_name(self, doc_id: str) -> DocName:
        s = self._tools.summaries[doc_id]
        return DocName(s.display_title, s.source, s.created[:10] if s.created else None)

    def _set_name(self, doc_ids: tuple[str, ...]) -> str:
        """'the 14 matching hubspot records' (the source named when they share one)."""
        sources = {self._tools.summaries[i].source for i in doc_ids}
        source = f"{next(iter(sources))} " if len(sources) == 1 else ""
        noun = "record" if len(doc_ids) == 1 else "records"
        return f"the {len(doc_ids)} matching {source}{noun}"

    # ------------------------------------------------------------ the tools
    def _search(self, args: SearchArgs) -> str:
        t = self._tools
        source = t.index.source(args.source).name if args.source else None
        low, high = date_bounds(args.created_from, args.created_to, "created")
        within = tuple(self.handles.documents([args.within])) if args.within else None
        scope = Scope(source=source, created_from=low, created_before=high, within=within)
        vector = self._query_vector(args.query)
        result = t.search.search(args.query, scope, args.offset, SEARCH_PAGE_DOCS, vector)
        window = t.index.date_window
        lines = [_search_head(args, source, result.undated, result.dense, window)]
        if result.scope_docs == 0:
            lines.append("No document is in this scope.")
        elif result.exact_phrases:
            lines.extend(_keyword_text(result))
        lines.append(_paging(result.page, "documents"))
        for hit in result.page.items:
            line = _doc_line(self.handles, t.summaries[hit.doc_id])
            lines.append(line + _SEP + _MEANING_ONLY if hit.keyword is False else line)
            lines.extend(f"  > {text}" for text in hit.snippets)
        _record(page=[self.handles.doc(h.doc_id) for h in result.page.items])
        return "\n".join(lines)

    def _filter(self, args: FilterArgs) -> str:
        t = self._tools
        conditions = [c.condition() for c in args.conditions]
        result = run_filter(
            t.reader, t.index, args.source, conditions, args.created_from, args.created_to
        )
        ids = result.doc_ids
        page = page_of(ids, args.offset, FILTER_PAGE_ROWS)
        shown = [a for a in result.conditions if a.asked.op in SHOWS_VALUES]
        names = {a.field.field: {result.source.name: a.field.field} for a in shown}
        rows = doc_rows(
            t.reader, [t.summaries[i] for i in page.items], [a.field.field for a in shown], names
        )
        n, total = len(ids), result.source.n_docs
        lines = [
            f"filter on source {quote(result.source.name)}: {n} of its {total} documents match."
        ]
        handle = None
        if ids:
            handle = self.handles.result_set(ids)
            lines.append(_saved(handle, n, "newest first"))
        window = t.index.date_window
        lines.extend(f"- {_applied_text(a, window)}" for a in result.conditions)
        if args.created_from or args.created_to:
            created = _created_text(args.created_from, args.created_to, result.undated, window)
            lines.append(f"- {created}")
        lines.append(_paging(page, "rows"))
        lines.extend(_row_line(self.handles, row, source=False) for row in rows)
        _record(result_set=handle, count=n, page=[self.handles.doc(i) for i in page.items])
        return "\n".join(lines)

    def _read_fields(self, args: ReadFieldsArgs) -> str:
        t = self._tools
        ids = self.handles.documents(args.documents)
        got = read_fields(t.reader, t.index, ids, args.fields, args.offset, READ_FIELDS_PAGE_ROWS)
        fields = ", ".join(quote(f) for f in args.fields)
        lines = [f"read_fields {fields} of {got.page.total} documents:"]
        for asked, per_source in got.names.items():
            renamed = {s: name for s, name in per_source.items() if name != asked}
            if renamed:
                where = ", ".join(f"{quote(name)} in {quote(s)}" for s, name in renamed.items())
                lines.append(f"- {quote(asked)} is {where}")
        lines.append(_paging(got.page, "rows"))
        lines.extend(_row_line(self.handles, row, source=True) for row in got.page.items)
        _record(
            expanded=len(ids), page=[self.handles.doc(r.summary.doc_id) for r in got.page.items]
        )
        return "\n".join(lines)

    def _read_document(self, args: ReadDocumentArgs) -> str:
        ids = self.handles.documents([args.document])
        if len(ids) != 1:
            raise ToolInputError(
                f"{quote(args.document)} is a set of {len(ids)} documents; read_document "
                "reads one: give a D# handle (read_fields reads fields of a whole set)."
            )
        [doc_id] = ids
        doc = read_document(self._tools.reader, doc_id)
        s = doc.summary
        lines = [_doc_line(self.handles, s)]
        if s.updated is not None and s.updated != s.created:
            lines.append(f"updated: {_date_text(s.updated)}")
        meta = [r for r in doc.fields if r.role != "title"]
        if meta:
            lines.append("fields:")
            by_field: dict[str, list[str]] = {}
            for r in meta:
                values = by_field.setdefault(r.field, [])
                if r.value is not None and r.value_type not in ("empty", "null"):
                    values.append(r.value)
            for name, values in by_field.items():
                text = _values_text(tuple(values)) if values else "(empty)"
                lines.append(f"- {name}: {text}")
        lines.append("text:")
        lines.append(doc.body)
        _record(document=self.handles.doc(doc_id), body_chars=len(doc.body))
        return "\n".join(lines)

    def _lookup_key(self, args: LookupKeyArgs) -> str:
        t = self._tools
        got = lookup_key(t.reader, args.key)
        owners, referrers = got.owners, got.referrers
        if not owners and not referrers:
            return (
                f"no document owns or refers to the key {quote(args.key)}. Its text may still "
                "be in documents: search finds that."
            )
        ids = [*owners, *(r.doc_id for r in referrers)]
        head = f"key {quote(args.key)}: owned by {len(owners)} document(s)"
        lines = [f"{head}, referred to by {len(referrers)} other document(s)."]
        if got.hub:
            lines.append("The key is shared by many documents: it does not name one record.")
        handle = self.handles.result_set(ids)
        lines.append(_saved(handle, len(ids), "owners first, then referrers"))
        page = page_of(ids, args.offset, KEY_LOOKUP_PAGE_ROWS)
        refs = {r.doc_id: r for r in referrers}
        lines.append(_paging(page, "rows"))
        for doc_id in page.items:
            line = _doc_line(self.handles, t.summaries[doc_id])
            if doc_id in refs:
                how = "; ".join(f"{field}: {quote(raw)}" for field, raw in refs[doc_id].refs)
                lines.append(f"{line}{_SEP}refers to it ({how})")
            else:
                lines.append(f"{line}{_SEP}owns it")
        _record(result_set=handle, owners=len(owners), referrers=len(referrers))
        return "\n".join(lines)

    def _list_fields(self, args: ListFieldsArgs) -> str:
        t = self._tools
        got = list_fields(t.index, args.source, args.near, args.offset, LIST_FIELDS_PAGE_ROWS)
        lines = [_source_text(got.source) + f"; {got.page.total} fields."]
        if got.n_near is not None:
            lines.append(f"{got.n_near} field(s) near {quote(args.near or '')} by name come first.")
        lines.append(_paging(got.page, "fields"))
        lines.extend(
            f"- {_field_text(t.index, row, LIST_FIELDS_TOP_VALUES)}" for row in got.page.items
        )
        return "\n".join(lines)

    def _field_values(self, args: FieldValuesArgs) -> str:
        t = self._tools
        got = field_values(
            t.reader, t.index, args.source, args.field, args.near, args.offset,
            FIELD_VALUES_PAGE_ROWS,
        )  # fmt: skip
        row = got.field
        head = f"field {quote(row.field)} of source {quote(got.source.name)} ({row.kind})"
        coverage = _coverage_text(got.coverage, t.index.date_window)
        lines = [f"{head}: {got.page.total} distinct values; {coverage}."]
        if got.n_near is not None:
            lines.append(f"{got.n_near} value(s) near {quote(args.near or '')} come first.")
        lines.append(_paging(got.page, "values"))
        for count in got.page.items:
            line = f"- {quote(count.shown)}: {count.n_docs} docs"
            others = [s for s, _ in count.spellings[1:]]
            if others:
                line += f" (also spelled {', '.join(quote(s) for s in others)})"
            lines.append(line)
        return "\n".join(lines)

    def _query_vector(self, query: str) -> list[float] | None:
        """The query's vector (one embedding call per distinct query of this session), or
        None when there is no dense search or the call failed (keyword ranking then)."""
        embedder = self._tools.embedder
        if embedder is None:
            return None
        with self._vectors_lock:
            if query in self._vectors:
                return self._vectors[query]
        try:
            [vector] = embedder.embed([query], "query")
        except LLMError as exc:
            logger.warning("search: the query could not be embedded (%s); keyword only", exc)
            obs.current().warning(f"query not embedded: {type(exc).__name__}")
            vector = None
        with self._vectors_lock:
            self._vectors[query] = vector
        return vector


def _record(**metadata: Any) -> None:
    """Handles and counts of this call, on its `tool:<name>` span."""
    obs.current().update(metadata={"handles": metadata})


# ---------------------------------------------------------------- text
def _paging(page: Page[Any], noun: str) -> str:
    """'rows 1-50 of 278; next: offset=50', or '(last page)', or 'no rows'. Without the
    'of N' when the total is not known yet (a search page: more follow, deeper)."""
    if page.total == 0:
        return f"no {noun}"
    first, last = page.offset + 1, page.offset + len(page.items)
    line = f"{noun} {first}-{last}" + (f" of {page.total}" if page.exact else "")
    if page.next_offset is None:
        return f"{line} (last page)"
    return f"{line}; next: offset={page.next_offset}"


def _saved(handle: str, n: int, order: str) -> str:
    return (
        f"All {n} are saved as {handle} ({order}): cite {handle} to cite every one; "
        f'read_fields(documents=["{handle}"], fields=[...]) reads fields of each; '
        f'search(..., within="{handle}") searches only them.'
    )


def _date_text(value: str | None) -> str:
    return "no date" if value is None else value


def _window_text(window: tuple[int, int] | None) -> str:
    """The corpus's date window as the texts name it."""
    return "the date window" if window is None else f"the date window {window[0]}-{window[1]}"


def _no_date_text(n: int, where: str, window: tuple[int, int] | None) -> str:
    return (
        f"{n} documents {where} have no usable created date (none, or only dates outside "
        f"{_window_text(window)}, which count as no date)"
    )


def _doc_line(handles: Handles, s: DocSummary, *, source: bool = True) -> str:
    parts = [handles.doc(s.doc_id)]
    if source:
        parts.append(s.source)
    parts += [_date_text(s.created), quote(s.display_title)]
    return _SEP.join(parts)


def _values_text(values: tuple[str, ...]) -> str:
    if len(values) == 1:
        return quote(values[0])
    return "[" + ", ".join(quote(v) for v in values) + "]"


def _state_text(state: FieldState, source: str) -> str:
    match state.state:
        case "value":
            return _values_text(state.values)
        case "empty":
            return "(empty)"
        case "in_body":
            return "(in the document text)"
        case "absent":
            return "(absent)"
        case _:
            return f"(not a field of source {quote(source)})"


def _row_line(handles: Handles, row: DocRow, *, source: bool) -> str:
    line = _doc_line(handles, row.summary, source=source)
    for state in row.fields:
        name = state.field or state.asked
        line += f"{_SEP}{name}: {_state_text(state, row.summary.source)}"
    return line


def _coverage_text(c: FieldCoverage, window: tuple[int, int] | None) -> str:
    text = f"a value in {c.with_value} of {c.source_docs} documents, empty in {c.empty}"
    if c.in_body:
        text += f", in the document text in {c.in_body}"
    text += f", absent in {c.absent}"
    if c.dates_outside:
        text += (
            f"; {c.dates_outside} of them hold a date outside {_window_text(window)} "
            "(it counts as no date: date ranges do not match it)"
        )
    return text


def _applied_text(a: Applied, window: tuple[int, int] | None) -> str:
    c, name = a.asked, a.field.field
    if c.op in ("present", "missing"):
        what = f"{name} {c.op}"
    elif c.op == "range":
        what = f"{name} range {c.low or '...'} to {c.high or '...'}"
    elif not a.resolved:  # eq on a date or a number, run as a range
        what = f"{name} eq {quote(c.values[0])}"
    else:
        what = f"{name} {c.op} " + "; ".join(_resolved_text(r) for r in a.resolved)
    return f"{what} [{name}: {_coverage_text(a.coverage, window)}]"


def _resolved_text(r: Resolved) -> str:
    if r.how == "contains":
        listed = show_counts(r.matched[:CONTAINS_VALUES_SHOWN])
        more = len(r.matched) - CONTAINS_VALUES_SHOWN
        text = f"{quote(r.asked)}: {len(r.matched)} values match: {listed}"
        return text + (f" and {more} more (each row shows its value)" if more > 0 else "")
    text = f"{quote(r.asked)} -> {show_counts(r.matched)}"
    if r.how == "loose":
        text += " (spelled differently)"
    if r.left_out:
        text += (
            f"; also spelled {show_counts(r.left_out)}: not included (op 'in' with both "
            "includes it)"
        )
    return text


def _created_text(
    low: str | None, high: str | None, undated: int, window: tuple[int, int] | None
) -> str:
    text = f"created {low or '...'} to {high or '...'}"
    if undated:
        text += f"; {_no_date_text(undated, 'of the source', window)} and were not matched"
    return text


def _search_head(
    args: SearchArgs,
    source: str | None,
    undated: int,
    dense: bool,
    window: tuple[int, int] | None,
) -> str:
    scope = []
    if source is not None:
        scope.append(f"in source {quote(source)}")
    if args.created_from or args.created_to:
        scope.append(f"created {args.created_from or '...'} to {args.created_to or '...'}")
    if args.within:
        scope.append(f"within {args.within}")
    head = f"search {quote(args.query)}" + (f" ({', '.join(scope)})" if scope else "")
    head += ": best first"
    if not dense:
        head += " (keyword ranking only)"
    if undated:
        head += f"; {_no_date_text(undated, 'in scope', window)} and were left out"
    return head + "."


def _keyword_text(result: SearchResult) -> list[str]:
    """Exact search: the line saying no document in scope matched the keywords, or what
    the "meaning only" mark on this page means (no line when nothing is marked)."""
    if not result.keyword_found:
        line = "No document in scope contains the searched words, phrases or keys"
        if result.dense:
            line += f': the documents below are only close in meaning (marked "{_MEANING_ONLY}")'
        return [line + "."]
    if any(hit.keyword is False for hit in result.page.items):
        return [
            f'Documents marked "{_MEANING_ONLY}" contain none of the searched words, phrases '
            "or keys: they are only close in meaning."
        ]
    return []


def _source_text(s: SourceInfo) -> str:
    text = f"source {quote(s.name)}: {s.n_docs} documents"
    if s.first is not None:
        text += f", created {s.first[:10]} to {(s.last or s.first)[:10]}"
    if s.n_docs - s.n_dated:
        text += f" ({s.n_docs - s.n_dated} without a usable created date)"
    return text


def _field_text(index: FieldIndex, row: FieldProfileRow, top: int) -> str:
    """One field as list_fields and the catalog show it."""
    share = round(100 * row.presence)
    shown = "<1" if share == 0 and row.n_docs else str(share)
    text = f"{row.field}: {row.kind}, in {row.n_docs} docs ({shown}%), {row.n_distinct} values"
    if row.title_ratio >= CATALOG_FIELD_MAX_TITLE_RATIO:
        text += "; the title field"
    if row.content_ratio >= CATALOG_FIELD_MAX_CONTENT_RATIO:
        text += "; its text is in the document text (search finds it, filter cannot)"
    if row.top_values and row.avg_len <= PREVIEW_MAX_AVG_CHARS:
        text += "; top " + ", ".join(f"{quote(v)} ({n})" for v, n in row.top_values[:top])
    elif row.top_values:
        text += f"; free text (about {round(row.avg_len)} characters)"
    if synonyms := index.synonyms_of(row):
        text += "; same as " + ", ".join(synonyms)
    for c in index.conditions_of(row):
        text += f"; present when {c.given_field} is {', '.join(quote(v) for v in c.present_when)}"
    return text


def field_catalog(index: FieldIndex) -> str:
    """The sources and their common fields, for the system prompt (the profile picks the
    fields; list_fields shows every field), after one line on which dates count."""
    lines = [
        f"Dates outside {_window_text(index.date_window)} count as no date (they are not "
        "created dates and date ranges do not match them)."
    ]
    for source, rows in index.catalog():
        lines.append(f"- {_source_text(source)}")
        lines.extend(f"  - {_field_text(index, row, CATALOG_TOP_VALUES)}" for row in rows)
    return "\n".join(lines)


if TYPE_CHECKING:  # pyright checks that a session keeps track A's ToolSession shape

    def _session_is_a_tool_session(session: CorpusToolSession) -> "ToolSession":
        return session
