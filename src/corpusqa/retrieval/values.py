"""Matching the model's words against the real names and values in the index.

- `loose`: the loose form (case, `-`, `_`, `$`, punctuation and spaces folded away), so
  "ENG sre" finds `eng-sre`.
- `nearness`: how close a name or value is to the model's words, for "did you mean" lists
  and for ranking by `near`.
- `resolve_value` / `resolve_contains`: the stored values one asked value stands for. One
  exact match is used (close spellings it leaves out are named); else one loose match is
  used and reported; several loose matches are all listed and nothing runs (never merged
  silently); none names the nearest values. Persons match through `person_norm`, as
  ingest stored them.
- `day_range`, `number`: the model's dates and numbers, for ranges.
"""

import difflib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from corpusqa.config import NEAR_MIN_RATIO, NEAREST_SHOWN
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.store.interfaces import ValueCount
from corpusqa.store.norms import normalise, person_norm

# Kind names the profile stage stores in field_profile.kind (ingest.profile).
DATE_KIND, NUMBER_KIND, PERSON_KIND = "date", "number", "person"

_WORD = re.compile(r"[^\W_]+")
# Containment counts as near only for words this long ("a" is inside almost anything).
_CONTAIN_MIN_CHARS = 3
# Containment and shared words rank near, below an exact loose match.
_CONTAINED_SCORE = 0.9
_SHARED_BASE, _SHARED_SPAN = 0.6, 0.3
_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ].*)?$")
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR = re.compile(r"^(\d{4})$")
_NUMBER_NOISE = str.maketrans("", "", ",$€£ ")
_DATE_FORMS = "YYYY-MM-DD, YYYY-MM or YYYY"


def loose(text: str) -> str:
    """The loose form: `normalise` (case, NFKC, `_`, yes/no), then letters and digits only."""
    return "".join(ch for ch in normalise(text) if ch.isalnum())


def words(text: str) -> frozenset[str]:
    """The words of a name or value, case-folded (`_` and `-` split words)."""
    return frozenset(w.casefold() for w in _WORD.findall(text))


def quote(text: str) -> str:
    """A name or value as the tools show it: in double quotes, JSON-escaped."""
    return json.dumps(text, ensure_ascii=False)


def nearness(asked: str, candidate: str) -> float:
    """0..1: how near `candidate` is to the model's words (1.0: the same loose form)."""
    a, b = loose(asked), loose(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    score = difflib.SequenceMatcher(None, a, b).ratio()
    if min(len(a), len(b)) >= _CONTAIN_MIN_CHARS and (a in b or b in a):
        score = max(score, _CONTAINED_SCORE)
    asked_words = words(asked)
    if shared := asked_words & words(candidate):
        score = max(score, _SHARED_BASE + _SHARED_SPAN * len(shared) / len(asked_words))
    return score


def is_near(score: float) -> bool:
    return score >= NEAR_MIN_RATIO


def nearest[T](asked: str, items: Sequence[T], name: Callable[[T], str]) -> list[T]:
    """`items` nearest to `asked` first (stable: ties keep the given order)."""
    scored = [(nearness(asked, name(item)), i) for i, item in enumerate(items)]
    return [items[i] for _, i in sorted(scored, key=lambda s: (-s[0], s[1]))]


def show_counts(counts: Sequence[ValueCount]) -> str:
    """'"In Repair" (55 docs), "Done" (10 docs)'."""
    return ", ".join(f"{quote(c.shown)} ({c.n_docs} docs)" for c in counts)


@dataclass(frozen=True, slots=True)
class Resolved:
    """The stored values one asked value stands for, and how it matched."""

    asked: str
    matched: tuple[ValueCount, ...]
    how: Literal["exact", "loose", "contains"]
    # Other spellings with the same loose form, left out because one matched exactly.
    left_out: tuple[ValueCount, ...] = ()


def resolve_value(
    asked: str, counts: Sequence[ValueCount], *, person: bool, where: str
) -> Resolved:
    """The value of `counts` that `asked` names (see the module docstring). `where` names
    the field for messages ('field "x" of source "y"'). Raises ToolInputError."""
    norm = person_norm(asked) if person else normalise(asked)
    folded = loose(norm)
    if not folded:
        raise ToolInputError(f"the value {quote(asked)} has no letters or digits to match.")
    _require_values(counts, where)
    exact = [c for c in counts if c.value_norm == norm]
    close = [c for c in counts if c.value_norm != norm and loose(c.value_norm) == folded]
    if exact:
        return Resolved(asked, tuple(exact), "exact", tuple(close))
    if len(close) == 1:
        return Resolved(asked, tuple(close), "loose")
    if close:
        raise ToolInputError(
            f"{quote(asked)} matches {len(close)} values of {where}, spelled differently: "
            f"{show_counts(close)}. Nothing was run. Use op 'in' with the ones you mean, "
            "or 'eq' with one of them."
        )
    raise ToolInputError(
        f"no value of {where} matches {quote(asked)}. Nearest: "
        f"{show_counts(nearest(asked, counts, _shown)[:NEAREST_SHOWN])}. Nothing was run. "
        f"field_values lists all {len(counts)} values; op 'contains' matches part of a value."
    )


def resolve_contains(asked: str, counts: Sequence[ValueCount], *, where: str) -> Resolved:
    """Every value of `counts` whose loose form contains the loose form of `asked` (each
    spelling checked whole, not only its stored matching form). Raises ToolInputError."""
    folded = loose(asked)
    if not folded:
        raise ToolInputError(f"the value {quote(asked)} has no letters or digits to match.")
    _require_values(counts, where)
    matched = [c for c in counts if any(folded in loose(s) for s, _ in c.spellings)]
    if not matched:
        raise ToolInputError(
            f"no value of {where} contains {quote(asked)}. Nearest: "
            f"{show_counts(nearest(asked, counts, _shown)[:NEAREST_SHOWN])}. Nothing was run."
        )
    return Resolved(asked, tuple(matched), "contains")


def _require_values(counts: Sequence[ValueCount], where: str) -> None:
    if not counts:
        raise ToolInputError(
            f"{where} holds no values to compare; use op 'present' or 'missing' with it."
        )


def _shown(count: ValueCount) -> str:
    return count.shown


def day_range(text: str, what: str) -> tuple[str, str]:
    """[first day, the day after the last day) of a date the model wrote (YYYY-MM-DD,
    YYYY-MM or YYYY; a date-time counts as its day), as ISO dates. Raises ToolInputError."""
    value = text.strip()
    try:
        if m := _DAY.match(value):
            first = date(int(m[1]), int(m[2]), int(m[3]))
            return first.isoformat(), (first + timedelta(days=1)).isoformat()
        if m := _MONTH.match(value):
            first = date(int(m[1]), int(m[2]), 1)
            after = date(first.year + first.month // 12, first.month % 12 + 1, 1)
            return first.isoformat(), after.isoformat()
        if m := _YEAR.match(value):
            year = int(m[1])
            return date(year, 1, 1).isoformat(), date(year + 1, 1, 1).isoformat()
    except ValueError:
        pass  # e.g. month 13: the message below
    raise ToolInputError(f"{what} {quote(text)} is not a date: write it as {_DATE_FORMS}.")


def date_bounds(low: str | None, high: str | None, what: str) -> tuple[str | None, str | None]:
    """ISO bounds [low, high) for a date range the model gave as an inclusive from..to."""
    start = day_range(low, f"{what} from")[0] if low is not None else None
    end = day_range(high, f"{what} to")[1] if high is not None else None
    if start is not None and end is not None and start >= end:
        raise ToolInputError(f"{what}: from {quote(low or '')} is after to {quote(high or '')}.")
    return start, end


def number(text: str) -> float | None:
    """The model's number ("1,200", "$40", "-3.5"), or None when it is not one."""
    try:
        value = float(text.translate(_NUMBER_NOISE))
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None
