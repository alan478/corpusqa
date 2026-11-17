"""S4 fields: one row per metadata value, typed from the value alone, plus the dates.

Every key that is not a contract key and did not become body text gets rows; the
title field always gets a `title` row. A list gives one row per item. Present-but-empty
values get an `empty` row, so a filter never reads "absent" as "no".
"""

import json
import re

from corpusqa.config import (
    CONTRACT_KEYS,
    DOC_SUFFIX,
    NOISE_KEY_PREFIXES,
    NOISE_MAX_VALUE_CHARS,
)
from corpusqa.ingest.dates import DateWindow, is_ambiguous_date, parse_date, pick_dates
from corpusqa.ingest.models import CoreDoc, FieldsResult, FileRef, ParsedDoc
from corpusqa.store.interfaces import FieldRow
from corpusqa.store.norms import fold_key, normalise

# A plain number, optionally with thousands separators: "42", "-1.5", "12,500".
_NUMBER = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
NO_DATE_FLAG = "no_date"


def build_fields(doc: ParsedDoc, core: CoreDoc, window: DateWindow) -> FieldsResult:
    """Field rows, and created/updated from the scalar dates inside `window`.

    `candidates` keeps every scalar date (any year), so the pipeline can pick the dates
    again once it has learned the corpus's window.
    """
    rows: list[FieldRow] = []
    candidates: list[tuple[str, str]] = []  # (field, iso) of scalar dates, file key order
    ambiguous = False
    for key, value in doc.obj.items():
        if key in CONTRACT_KEYS or (key in core.body_keys and key != core.title_field):
            continue
        role = "title" if key == core.title_field else "meta"
        in_list = isinstance(value, list)
        items = list(enumerate(value)) if in_list and value else [(0, value)]
        for seq, item in items:
            row = _row(key, seq, role, item, doc.file, window, in_list=in_list)
            rows.append(row)
            ambiguous |= row.value_type == "str" and is_ambiguous_date(row.value or "")
            if row.value_iso and not isinstance(value, list):
                candidates.append((key, row.value_iso))
    dates = pick_dates(candidates, doc.file.stem, window)
    flags = [NO_DATE_FLAG] if dates.created is None else []
    if ambiguous:
        flags.append("ambiguous_date")
    return FieldsResult(
        rows=tuple(rows), dates=dates, flags=tuple(flags), candidates=tuple(candidates)
    )


def _row(
    key: str, seq: int, role: str, item: object, file: FileRef, window: DateWindow, *, in_list: bool
) -> FieldRow:
    value_type, value = _typed(item)
    norm = num = iso = precision = dated = written = None
    if value is not None and value_type not in ("empty", "json"):
        norm = normalise(value)
        if value_type != "bool":
            num = float(value.replace(",", "")) if _NUMBER.match(value.strip()) else None
            parsed = parse_date(value)
            if parsed is not None:
                iso, precision, written = parsed.iso, parsed.precision, parsed.kind
                dated = window.contains(parsed.year)
    return FieldRow(
        field=key,
        field_norm=fold_key(key),
        seq=seq,
        role=role,
        value=value,
        value_type=value_type,
        value_norm=norm,
        value_num=num,
        value_iso=iso,
        date_precision=precision,
        date_in_range=dated,
        date_format=written,
        in_list=in_list,
        is_noise=_is_noise(key, value, file),
    )


def _typed(item: object) -> tuple[str, str | None]:
    """(value_type, value as text) of one JSON value."""
    if item is None:
        return "null", None
    if isinstance(item, bool):
        return "bool", json.dumps(item)
    if isinstance(item, int):
        return "int", str(item)
    if isinstance(item, float):
        return "float", json.dumps(item)
    if isinstance(item, str):
        return ("empty" if not item.strip() else "str"), item
    if item == []:
        return "empty", ""
    return "json", json.dumps(item, ensure_ascii=False)


def _is_noise(key: str, value: str | None, file: FileRef) -> bool:
    """Generator residue: hidden from the agent's field list by default, still stored."""
    if key == "" or value is None:
        return True
    if key.startswith(NOISE_KEY_PREFIXES) and len(value) < NOISE_MAX_VALUE_CHARS:
        return True
    text = value.strip()
    if text in (file.name, file.stem, file.rel_path):
        return True
    return text.endswith(DOC_SUFFIX) and not any(ch.isspace() for ch in text)
