"""S2 parse: file bytes to a JSON object with its contract keys, or a quarantine row.

Keys are kept exactly as given (empty, punctuation, non-ASCII). The id format is not
checked: the contract promises the `dataset_doc_uuid` field, not its shape.

A JSON escape such as `\\ud800` decodes to a lone surrogate, which is not valid Unicode and
cannot be stored (SQLite takes UTF-8). Such characters in keys and values become U+FFFD
(flag `bad_unicode`); an id holding one is set aside (it could never be cited as given).
Escaped pairs (an emoji) decode to one real character and are not touched.
"""

import hashlib
import json
import logging
import re
from typing import Any

from corpusqa.config import (
    CONTENT_KEY,
    CONTRACT_KEYS,
    MAX_FIELDS_PER_DOC,
    TITLE_KEY,
    UUID_KEY,
    UUID_MAX_CHARS,
)
from corpusqa.ingest.models import FileRef, ParsedDoc
from corpusqa.store.interfaces import QuarantineRow

logger = logging.getLogger(__name__)

# A lone surrogate can only come from a JSON escape: valid UTF-8 never decodes to one.
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
_REPLACEMENT = "\ufffd"


def parse(file: FileRef) -> ParsedDoc | QuarantineRow:
    def reject(reason: str, detail: str) -> QuarantineRow:
        return QuarantineRow(file.rel_path, file.source, reason, detail)

    try:
        raw = file.path.read_bytes()
    except OSError as exc:
        return reject("unreadable", f"{type(exc).__name__}: {exc}")
    flags: list[str] = []
    try:
        text = raw.decode("utf-8-sig")  # also drops a byte-order mark
    except UnicodeDecodeError:
        text = raw.decode("utf-8-sig", errors="replace")
        flags.append("bad_encoding")
    if "\x00" in text:
        text = text.replace("\x00", "")
        flags.append("nul_bytes")

    duplicates: list[str] = []

    def keep_last(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                duplicates.append(key)
            obj[key] = value
        return obj

    try:
        obj = json.loads(text, object_pairs_hook=keep_last)
    except json.JSONDecodeError as exc:
        empty = not text.strip()
        return reject("json_invalid", "empty file" if empty else f"{exc.msg} at {exc.pos}")
    except (ValueError, RecursionError) as exc:
        return reject("json_invalid", f"{type(exc).__name__}: {exc}")
    if not isinstance(obj, dict):
        return reject("not_object", f"top level is {type(obj).__name__}")
    raw_id = obj.get(UUID_KEY)
    if _SURROGATE_ESCAPE.search(text):
        if isinstance(raw_id, str) and _LONE_SURROGATE.search(raw_id):
            return reject("no_uuid", f"{UUID_KEY} holds invalid Unicode (a lone surrogate)")
        obj, repaired = _valid_unicode(obj, duplicates)
        if repaired:
            flags.append("bad_unicode")
    flags.extend(f"duplicate_key:{key}" for key in dict.fromkeys(duplicates))

    if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
        return reject("no_uuid", f"{UUID_KEY} is missing or blank")
    if not isinstance(raw_id, str):
        return reject("no_uuid", f"{UUID_KEY} is a {type(raw_id).__name__}, not a string")
    doc_id = raw_id.strip()
    if len(doc_id) > UUID_MAX_CHARS:
        return reject("no_uuid", f"{UUID_KEY} is longer than {UUID_MAX_CHARS} characters")
    if doc_id != raw_id:
        flags.append("uuid_trimmed")

    title_field = obj.get(TITLE_KEY)
    if not isinstance(title_field, str):
        title_field = None
        flags.append("no_title_field")

    listed = obj.get(CONTENT_KEY)
    # A bare string is read as a one-item list; non-string items are dropped.
    items = [listed] if isinstance(listed, str) else listed if isinstance(listed, list) else []
    if not isinstance(listed, list) or not all(isinstance(n, str) for n in listed):
        flags.append("bad_content_field_names")
    content_fields = tuple(dict.fromkeys(n for n in items if isinstance(n, str)))

    if any(_is_nested(v) for k, v in obj.items() if k not in CONTRACT_KEYS):
        flags.append("nested_value")
    if len(obj) > MAX_FIELDS_PER_DOC:
        logger.warning("ingest: %s has %d keys (kept)", file.rel_path, len(obj))
    return ParsedDoc(
        file=file,
        doc_id=doc_id,
        obj=obj,
        title_field=title_field,
        content_fields=content_fields,
        content_hash=hashlib.sha256(raw).hexdigest(),
        raw=raw,
        flags=tuple(flags),
    )


def _valid_unicode(value: Any, duplicates: list[str]) -> tuple[Any, bool]:
    """`value` with every lone surrogate in its strings and keys replaced by U+FFFD, and
    whether anything changed. Keys that become equal keep the last value (a duplicate)."""
    if isinstance(value, str):
        fixed = _LONE_SURROGATE.sub(_REPLACEMENT, value)
        return fixed, fixed != value
    if isinstance(value, list):
        items = [_valid_unicode(item, duplicates) for item in value]
        return [item for item, _ in items], any(changed for _, changed in items)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        repaired = False
        for key, item in value.items():
            new_key, key_changed = _valid_unicode(key, duplicates)
            new_item, item_changed = _valid_unicode(item, duplicates)
            if new_key in out:
                duplicates.append(new_key)
            out[new_key] = new_item
            repaired = repaired or key_changed or item_changed
        return out, repaired
    return value, False


def _is_nested(value: object) -> bool:
    if isinstance(value, dict):
        return True
    return isinstance(value, list) and any(isinstance(i, dict | list) for i in value)
