"""The filter: which documents of a source meet every condition (AND), exactly.

Each condition names a field of the source (any field: title fields too) and an op:
- eq / in: the value(s) are first resolved against the field's real values
  (`values.resolve_value`); then the query matches their stored forms exactly. On a date
  field, eq with a date means that day (or month, or year); on a number field, eq with a
  number means that number.
- contains: every value whose loose form contains the asked words.
- range: from..to, both inclusive, either may be left out: numbers on a number field,
  dates (YYYY-MM-DD, YYYY-MM, YYYY) on a date field; on other fields, numbers when both
  bounds are numbers, else dates. A date outside the corpus's date window counts as no
  date: no date range (nor eq on a day) matches it; the field's coverage counts them.
- present / missing: a value (or the field's text in the body) or not; an empty value is
  missing. Each condition reports how the source's documents hold its field (with a
  value, empty, in the body, absent), so a missing field is never read as "no".
`created_from` / `created_to` match the document's own created date (always inside the
window); the result says how many documents of the source have no usable created date.
The result is the complete ordered id list (newest first): the agent pages it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from corpusqa import obs
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.fields import FieldIndex, SourceInfo, coverage_dict
from corpusqa.retrieval.values import (
    DATE_KIND,
    NUMBER_KIND,
    PERSON_KIND,
    Resolved,
    date_bounds,
    day_range,
    number,
    quote,
    resolve_contains,
    resolve_value,
)
from corpusqa.store.interfaces import CorpusReader, FieldCoverage, FieldMatch, FieldProfileRow

type Op = Literal["eq", "in", "range", "present", "missing", "contains"]
OPS: tuple[Op, ...] = ("eq", "in", "range", "present", "missing", "contains")
# Ops whose rows show the field's values (eq shows nothing new: every row has the value).
SHOWS_VALUES: frozenset[Op] = frozenset({"in", "range", "contains"})


@dataclass(frozen=True, slots=True)
class Condition:
    """One condition as the model gave it (`values` for eq / in / contains, `low` / `high`
    for range)."""

    field: str
    op: Op
    values: tuple[str, ...] = ()
    low: str | None = None
    high: str | None = None


@dataclass(frozen=True, slots=True)
class Applied:
    """One condition as it ran: the field it named, what each value matched, the store
    match, and how the source's documents hold the field."""

    asked: Condition
    field: FieldProfileRow
    match: FieldMatch
    resolved: tuple[Resolved, ...]  # eq / in / contains on text, one per asked value
    coverage: FieldCoverage


@dataclass(frozen=True, slots=True)
class FilterResult:
    source: SourceInfo
    conditions: tuple[Applied, ...]
    created_from: str | None
    created_to: str | None
    # Documents of the source a created range could not check (no usable created date).
    undated: int
    doc_ids: tuple[str, ...]  # every match, newest first (no date last), then by id


def run_filter(
    reader: CorpusReader,
    index: FieldIndex,
    source: str,
    conditions: Sequence[Condition],
    created_from: str | None = None,
    created_to: str | None = None,
) -> FilterResult:
    """The documents of `source` that meet every condition. Raises ToolInputError when a
    source, field or value is not there (nothing is run then)."""
    given = {
        "source": source,
        "conditions": [_condition_dict(c) for c in conditions],
        "created_from": created_from,
        "created_to": created_to,
    }
    with obs.span("filter", input=given) as span:
        info = index.source(source)
        applied = tuple(_apply(reader, index, info, c) for c in conditions)
        low, high = date_bounds(created_from, created_to, "created")
        dated = low is not None or high is not None
        ids = reader.filter_doc_ids(info.name, [a.match for a in applied], low, high)
        result = FilterResult(
            source=info,
            conditions=applied,
            created_from=created_from,
            created_to=created_to,
            undated=info.n_docs - info.n_dated if dated else 0,
            doc_ids=tuple(ids),
        )
        span.update(
            output={"count": len(ids), "first_ids": list(ids[:10])},
            metadata={
                "source": info.name,
                "normalised": [_applied_dict(a) for a in applied],
                "created_bounds": [low, high],
                "undated": result.undated,
            },
        )
    return result


def _apply(
    reader: CorpusReader, index: FieldIndex, source: SourceInfo, condition: Condition
) -> Applied:
    row = index.field(source, condition.field)
    where = f"field {quote(row.field)} of source {quote(source.name)}"
    _check_shape(condition)
    resolved: tuple[Resolved, ...] = ()
    op = condition.op
    if op in ("present", "missing"):
        match = FieldMatch(row.field, op)
    elif op == "range":
        match = _range(row, condition.low, condition.high, where)
    elif op == "eq" and (special := _eq_as_range(row, condition.values[0])) is not None:
        match = special
    else:
        counts = reader.value_counts(source.name, row.field)
        if op == "contains":
            resolved = (resolve_contains(condition.values[0], counts, where=where),)
        else:
            person = row.kind == PERSON_KIND
            resolved = tuple(
                resolve_value(v, counts, person=person, where=where) for v in condition.values
            )
        norms = dict.fromkeys(c.value_norm for r in resolved for c in r.matched)
        match = FieldMatch(row.field, "norm_in", tuple(norms))
    coverage = reader.field_coverage(source.name, row.field)
    return Applied(condition, row, match, resolved, coverage)


def _check_shape(c: Condition) -> None:
    """Each op's arguments: what it takes and what it must not get."""
    n = len(c.values)
    has_bounds = c.low is not None or c.high is not None
    problem = None
    if c.op in ("eq", "contains") and n != 1:
        problem = f"op {quote(c.op)} takes exactly one value (got {n}); op 'in' takes several"
    elif c.op == "in" and n == 0:
        problem = "op 'in' takes one or more values"
    elif c.op == "range" and (n or not has_bounds):
        problem = "op 'range' takes 'from' and/or 'to', and no values"
    elif c.op in ("present", "missing") and (n or has_bounds):
        problem = f"op {quote(c.op)} takes no values and no bounds"
    elif c.op != "range" and has_bounds:
        problem = f"op {quote(c.op)} takes no 'from' / 'to' (they belong to op 'range')"
    if problem is not None:
        raise ToolInputError(f"condition on {quote(c.field)}: {problem}.")


def _eq_as_range(row: FieldProfileRow, value: str) -> FieldMatch | None:
    """eq on a date field with a date: that day (month, year); on a number field with a
    number: that number. None: match the value as text."""
    if row.kind == DATE_KIND:
        try:
            low, high = day_range(value, "value")
        except ToolInputError:
            return None
        return FieldMatch(row.field, "iso_range", low=low, high=high)
    if row.kind == NUMBER_KIND and (x := number(value)) is not None:
        return FieldMatch(row.field, "num_range", low=x, high=x)
    return None


def _range(row: FieldProfileRow, low: str | None, high: str | None, where: str) -> FieldMatch:
    numbers = [number(b) for b in (low, high) if b is not None]
    as_numbers = row.kind == NUMBER_KIND or (
        row.kind != DATE_KIND and all(x is not None for x in numbers)
    )
    if as_numbers:
        if any(x is None for x in numbers):
            raise ToolInputError(f"{where} holds numbers: give 'from' / 'to' as numbers.")
        lo = number(low) if low is not None else None
        hi = number(high) if high is not None else None
        if lo is not None and hi is not None and lo > hi:
            raise ToolInputError(f"range on {where}: from {low} is above to {high}.")
        return FieldMatch(row.field, "num_range", low=lo, high=hi)
    start, end = date_bounds(low, high, f"range on {where}")
    return FieldMatch(row.field, "iso_range", low=start, high=end)


def _condition_dict(c: Condition) -> dict[str, object]:
    return {"field": c.field, "op": c.op, "values": list(c.values), "from": c.low, "to": c.high}


def _applied_dict(a: Applied) -> dict[str, object]:
    return {
        "field": a.field.field,
        "kind": a.field.kind,
        "match": {
            "kind": a.match.kind,
            "values": list(a.match.values),
            "low": a.match.low,
            "high": a.match.high,
        },
        "resolved": [
            {
                "asked": r.asked,
                "how": r.how,
                "matched": [[c.shown, c.n_docs] for c in r.matched],
                "left_out": [[c.shown, c.n_docs] for c in r.left_out],
            }
            for r in a.resolved
        ],
        "coverage": coverage_dict(a.coverage),
    }
