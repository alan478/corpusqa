"""S4 dates: read a value as a date (from the value alone), and pick created/updated.

Formats, in order: YYYY-MM-DD; ISO date-time (T or space, optional seconds, fraction and
zone); YYYY-MM, YYYY-Www, YYYY-Qn; epoch seconds (9-10 digits) and milliseconds (12-13);
RFC 2822; "Month D, YYYY" and "D Month YYYY" (English); MM/DD/YYYY or DD/MM/YYYY only
when one part is over 12. Times are stored in UTC; a time without a zone is taken as UTC.

Which dates count: a window of years, and each date value on its own (user, 2026-10-16:
no field is trusted or distrusted as a whole). A value inside the window is a usable date;
one outside counts as no date. The per-document stage uses the fixed window
(DATE_MIN_YEAR .. this year + DATE_MAX_FUTURE_YEARS). Once every value is read, the
pipeline learns the window from the ISO-format values (`learn_window`), counts the values
outside it (`outside_counts`), and picks created/updated again (`pick_dates`).
"""

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime

from corpusqa.config import (
    CREATED_WORDS,
    DATE_MAX_FUTURE_YEARS,
    DATE_MIN_YEAR,
    DATE_WINDOW_HIGH_QUANTILE,
    DATE_WINDOW_LOW_QUANTILE,
    DATE_WINDOW_MARGIN_YEARS,
    DATE_WINDOW_MIN_VALUES,
    PLANNED_WORDS,
    ROLE_WORD_MIN_PREFIX_CHARS,
    UPDATED_WORDS,
)
from corpusqa.ingest.models import DocDates

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_WEEK = re.compile(r"^(\d{4})-W(\d{2})$")
_QUARTER = re.compile(r"^(\d{4})-Q([1-4])$")
_EPOCH_S = re.compile(r"^\d{9,10}(?:\.\d+)?$")
_EPOCH_MS = re.compile(r"^\d{12,13}$")
_RFC2822 = re.compile(r"^(?:[A-Za-z]{3},\s*)?\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{1,2}:\d{2}")
_LONG_MDY = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")
_LONG_DMY = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})$")
_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
# A date at the start of a file name: YYYY-MM-DD, YYYYMMDD or a 10-digit epoch.
_STEM_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{8}|\d{10})(?!\d)")
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_MONTHS_IN_YEAR = len(_MONTHS)
_QUARTER_MONTHS = 3
_MONTH_ABBREV_CHARS = 3  # "jun" is the shortest month name accepted
_MS_PER_S = 1000


@dataclass(frozen=True, slots=True)
class ParsedDate:
    iso: str  # "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"
    precision: str  # date | datetime | month | week | quarter
    year: int
    # How the value was written: iso (YYYY-MM-DD, ISO date-time, YYYY-MM, -Www, -Qn),
    # epoch (a number of seconds or milliseconds) or text (RFC 2822, month names, slashes).
    kind: str


@dataclass(frozen=True, slots=True)
class DateWindow:
    """The years a date may fall in to feed created/updated, and where the bounds came from."""

    min_year: int
    max_year: int
    rule: str  # learned | fixed
    iso_values: int  # ISO-format date values seen (0 when not learned from a corpus)

    def contains(self, year: int) -> bool:
        return self.min_year <= year <= self.max_year


def parse_date(text: str) -> ParsedDate | None:
    """The date `text` holds, or None. Never raises."""
    s = text.strip()
    if not s or not s[0].isalnum():
        return None
    try:
        return _parse(s)
    except (ValueError, OverflowError, OSError, TypeError):
        return None


def is_ambiguous_date(text: str) -> bool:
    """A NN/NN/YYYY value whose day and month could be swapped (both 1-12, different)."""
    m = _SLASH.match(text.strip())
    if not m:
        return False
    a, b = int(m[1]), int(m[2])
    return a != b and 1 <= a <= _MONTHS_IN_YEAR and 1 <= b <= _MONTHS_IN_YEAR


def iso_year(iso: str) -> int:
    """The year of a stored `value_iso` (the part before the first dash)."""
    return int(iso.split("-", 1)[0])


def fixed_window(this_year: int, iso_values: int = 0) -> DateWindow:
    """The window used before (or instead of) learning one from the corpus."""
    return DateWindow(DATE_MIN_YEAR, this_year + DATE_MAX_FUTURE_YEARS, "fixed", iso_values)


def learn_window(iso_years: Counter[int], this_year: int) -> DateWindow:
    """The 1st-99th percentile of the years of ISO-format values, widened by a margin.

    Fewer than DATE_WINDOW_MIN_VALUES values: the fixed window (percentiles of a handful
    of values are just their extremes).
    """
    n = iso_years.total()
    if n < DATE_WINDOW_MIN_VALUES:
        return fixed_window(this_year, n)
    low = _percentile(iso_years, DATE_WINDOW_LOW_QUANTILE) - DATE_WINDOW_MARGIN_YEARS
    high = _percentile(iso_years, DATE_WINDOW_HIGH_QUANTILE) + DATE_WINDOW_MARGIN_YEARS
    return DateWindow(low, high, "learned", n)


@dataclass(frozen=True, slots=True)
class OutsideCount:
    """How many date values a key (a field) has, and how many of them are outside the window."""

    values: int
    outside: int


def outside_counts[K](
    field_years: Mapping[K, Counter[int]], window: DateWindow
) -> dict[K, OutsideCount]:
    """Per key, its date values and those outside the window (each counts as no date); only
    keys with at least one value outside. A key is whatever the caller groups by (the
    pipeline uses (source, field))."""
    counts = {
        key: OutsideCount(
            years.total(), sum(n for year, n in years.items() if not window.contains(year))
        )
        for key, years in field_years.items()
    }
    return {key: c for key, c in counts.items() if c.outside}


def pick_dates(candidates: Sequence[tuple[str, str]], stem: str, window: DateWindow) -> DocDates:
    """created / updated from (field, iso) candidates in file key order.

    Each date is judged on its own: one inside `window` counts, one outside counts as no
    date (whatever the rest of its field holds), and planned-date fields never count. No
    date at all: a date at the start of the file name.

    By order: earliest is created, latest is updated. Role words in a field name vote
    (open decision 2) and win when they disagree. When the votes contradict each other
    (created after updated), order decides both. The rule is `name_hint` only when the
    vote changed the date; when both give the same date, the voted (named) field is
    recorded with rule `order`.
    """
    usable = [
        c
        for c in candidates
        if window.contains(iso_year(c[1])) and not _has_word(c[0], PLANNED_WORDS)
    ]
    if usable:
        return _by_order_and_votes(usable)
    from_stem = _stem_date(stem, window)
    if from_stem is None:
        return DocDates()
    return DocDates(from_stem, None, "file_stem", from_stem, None, "file_stem")


def _by_order_and_votes(dated: Sequence[tuple[str, str]]) -> DocDates:
    first = min(dated, key=lambda c: c[1])
    last = max(dated, key=lambda c: c[1])
    created_votes = [c for c in dated if _role(c[0]) == "created"]
    updated_votes = [c for c in dated if _role(c[0]) == "updated"]
    created = min(created_votes, key=lambda c: c[1]) if created_votes else first
    updated = max(updated_votes, key=lambda c: c[1]) if updated_votes else last
    if created[1] > updated[1]:
        created, updated = first, last
    return DocDates(
        created=created[1],
        created_field=created[0],
        created_rule="order" if created[1] == first[1] else "name_hint",
        updated=updated[1],
        updated_field=updated[0],
        updated_rule="order" if updated[1] == last[1] else "name_hint",
    )


def _percentile(counts: Counter[int], q: float) -> int:
    """Nearest-rank percentile of a multiset of years."""
    rank = max(1, math.ceil(q * counts.total()))
    seen = 0
    for year in sorted(counts):
        seen += counts[year]
        if seen >= rank:
            return year
    return max(counts)


def _parse(s: str) -> ParsedDate | None:
    if m := _DATE.match(s):
        return _day(date(int(m[1]), int(m[2]), int(m[3])), "date", "iso")
    if _DATETIME.match(s):
        return _moment(datetime.fromisoformat(s), "iso")
    if m := _MONTH.match(s):
        return _day(date(int(m[1]), int(m[2]), 1), "month", "iso")
    if m := _WEEK.match(s):
        return _day(date.fromisocalendar(int(m[1]), int(m[2]), 1), "week", "iso")
    if m := _QUARTER.match(s):
        month = _QUARTER_MONTHS * (int(m[2]) - 1) + 1
        return _day(date(int(m[1]), month, 1), "quarter", "iso")
    if _EPOCH_S.match(s):
        return _moment(datetime.fromtimestamp(float(s), UTC), "epoch")
    if _EPOCH_MS.match(s):
        return _moment(datetime.fromtimestamp(int(s) / _MS_PER_S, UTC), "epoch")
    if _RFC2822.match(s):
        return _moment(parsedate_to_datetime(s), "text")
    if m := _LONG_MDY.match(s):
        return _day(date(int(m[3]), _month(m[1]), int(m[2])), "date", "text")
    if m := _LONG_DMY.match(s):
        return _day(date(int(m[3]), _month(m[2]), int(m[1])), "date", "text")
    if (m := _SLASH.match(s)) and not is_ambiguous_date(s):
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        month, day = (b, a) if a > _MONTHS_IN_YEAR else (a, b)
        return _day(date(year, month, day), "date", "text")
    return None


def _day(d: date, precision: str, kind: str) -> ParsedDate:
    return ParsedDate(d.isoformat(), precision, d.year, kind)


def _moment(dt: datetime, kind: str) -> ParsedDate:
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    # isoformat pads the year to 4 digits on every platform (strftime's %Y may not).
    stamp = dt.replace(tzinfo=None, microsecond=0).isoformat()
    return ParsedDate(f"{stamp}Z", "datetime", dt.year, kind)


def _month(name: str) -> int:
    folded = name.casefold()
    for number, month in enumerate(_MONTHS, start=1):
        if len(folded) >= _MONTH_ABBREV_CHARS and month.startswith(folded):  # "jun", "sept"
            return number
    raise ValueError(f"not an English month name: {name!r}")


def _stem_date(stem: str, window: DateWindow) -> str | None:
    m = _STEM_DATE.match(stem)
    if not m:
        return None
    token = m[1]
    if len(token) == len("YYYYMMDD"):
        token = f"{token[:4]}-{token[4:6]}-{token[6:]}"
    parsed = parse_date(token)
    return parsed.iso if parsed and window.contains(parsed.year) else None


def _words(field: str) -> list[str]:
    """The words of a field name: split at non-letters and at camelCase humps."""
    return re.findall(r"[^\W\d_]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", field).casefold())


def _has_word(field: str, stems: Sequence[str]) -> bool:
    return any(_matches(word, stem) for word in _words(field) for stem in stems)


def _matches(word: str, stem: str) -> bool:
    if len(stem) < ROLE_WORD_MIN_PREFIX_CHARS:
        return word == stem
    return word.startswith(stem)


def _role(field: str) -> str | None:
    created, updated = _has_word(field, CREATED_WORDS), _has_word(field, UPDATED_WORDS)
    if created == updated:
        return None
    return "created" if created else "updated"
