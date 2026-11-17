"""S5 clean: fix text-encoding residue in each body unit, then build the labelled body.

Only three things change: literal escape sequences (when they are the majority), line
endings, and trailing spaces / long blank runs. Everything else stays byte for byte;
matching tolerance belongs to `value_norm` and the search tokenizer.
"""

import re

from corpusqa.config import BODY_LABEL_PREFIX, MAX_BODY_CHARS
from corpusqa.ingest.models import CleanBody, CoreDoc

_BLANK_RUNS = re.compile(r"\n{4,}")  # more than two blank lines in a row
_TWO_BLANK_LINES = "\n\n\n"
_UNIT_GAP = "\n\n"


def clean_text(text: str) -> tuple[str, bool]:
    """The cleaned text, and whether literal escapes were decoded.

    Literal `\\n` sequences are decoded only when they outnumber real newlines: then the
    text was escaped twice. A minority is left alone (it is mostly inside code).
    """
    escaped = text.count("\\n") > text.count("\n")
    if escaped:
        text = text.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    return _BLANK_RUNS.sub(_TWO_BLANK_LINES, text).strip("\n"), escaped


def clean_body(core: CoreDoc) -> CleanBody:
    """Join the cleaned units. With 2+ labels, a "## <label>" line opens each run of
    units from one field. No text at all: the body is the title (flag `no_body`).
    """
    flags: list[str] = []
    cleaned = []
    for unit in core.units:
        text, escaped = clean_text(unit.text)
        if escaped and "escaped_newlines" not in flags:
            flags.append("escaped_newlines")
        if text.strip():
            cleaned.append((unit, text))
    labels = tuple(dict.fromkeys(unit.label for unit, _ in cleaned))
    parts: list[str] = []
    size = 0
    previous: str | None = None
    for unit, text in cleaned:
        lines = (
            [BODY_LABEL_PREFIX + unit.label] if len(labels) > 1 and unit.label != previous else []
        )
        lines += [clean_text(header)[0] for header in unit.part_headers]
        part = "\n".join([*lines, text])
        if size + len(part) > MAX_BODY_CHARS:
            if not parts:  # one unit alone is too long: keep its start
                parts.append(part[:MAX_BODY_CHARS])
            flags.append("body_truncated")
            break
        parts.append(part)
        size += len(part) + len(_UNIT_GAP)
        previous = unit.label
    if not parts:
        return CleanBody(body=core.title, labels=(), flags=(*flags, "no_body"))
    used = tuple(dict.fromkeys(unit.label for unit, _ in cleaned[: len(parts)]))
    return CleanBody(body=_UNIT_GAP.join(parts), labels=used, flags=tuple(flags))
