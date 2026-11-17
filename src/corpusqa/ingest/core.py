"""S3 core: the title (weak-title rules b-d) and the body units, in reading order.

Body units come from the listed content fields, in list order. Numbered field
families (`x`, `x_2`, `x_3` or `x_1_text`, `x_1_from`, ...) are read as one ordered
thread; with prose promotion on, unlisted prose fields follow, unless they repeat the
body. Rule (a) of the weak-title test needs corpus-wide counts and runs in S8.
"""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from corpusqa.config import (
    CONTRACT_KEYS,
    DISPLAY_TITLE_SEPARATOR,
    DOC_SUFFIX,
    DUP_JACCARD,
    DUP_SHINGLE_WORDS,
    LINK_SHAPED_MIN_SHARE,
    PROMOTE_PROSE_FIELDS,
    PROSE_MIN_CHARS,
    PROSE_MIN_WORDS,
    TITLE_LIST_SEPARATOR,
    WEAK_TITLE_MIN_CHARS,
)
from corpusqa.ingest.models import CoreDoc, FileRef, ParsedDoc, Unit

# Family members: "<base>_<n>_<suffix>" and "<base>_<n>" or "<base><n>".
_SUFFIXED = re.compile(r"^(?P<base>.+?)_(?P<n>\d+)_(?P<suffix>\D.*)$")
_PLAIN = re.compile(r"^(?P<base>.+?)_?(?P<n>\d+)$")
# One item that is a URL, a path or an issue-style key, not prose.
_LINKISH = re.compile(r"^(?:\S*://\S+|\S*/\S*|[A-Z][A-Z0-9]{1,9}-\d+)$")
_LITERAL_ESCAPES = re.compile(r"\\[nrt]")

type Member = tuple[int, str | None, str]  # (n, suffix or None, key)

WEAK_TITLE_FLAG = "weak_title"


def build_core(doc: ParsedDoc) -> CoreDoc:
    flags: list[str] = []
    title, title_field = _title(doc, flags)
    weak = title_field is not None and _is_weak(title, doc.file)
    display = display_title(title, doc.file.stem) if weak else title
    if weak:
        flags.append(WEAK_TITLE_FLAG)
    body = _Body(doc.obj)
    _read_listed(doc, body, flags)
    if PROMOTE_PROSE_FIELDS:
        _promote(doc, body, title_field, flags)
    return CoreDoc(
        title=title,
        title_field=title_field,
        title_is_weak=weak,
        display_title=display,
        units=tuple(body.units),
        body_keys=frozenset(body.keys),
        flags=tuple(flags),
    )


def humanise(stem: str) -> str:
    """A file stem as words: dashes and underscores become spaces."""
    return " ".join(re.sub(r"[-_]+", " ", stem).split())


# ---------------------------------------------------------------- title
def _title(doc: ParsedDoc, flags: list[str]) -> tuple[str, str | None]:
    if doc.title_field is not None and doc.title_field in doc.obj:
        text = _title_text(doc.obj[doc.title_field])
        if text:
            return text, doc.title_field
    flags.append("title_fallback")
    return humanise(doc.file.stem) or doc.file.stem, None


def _title_text(value: object) -> str | None:
    if isinstance(value, list):
        parts = [_title_text(item) for item in value]
        text = TITLE_LIST_SEPARATOR.join(p for p in parts if p)
    elif isinstance(value, str) or (isinstance(value, int | float) and not isinstance(value, bool)):
        text = str(value)
    else:
        return None
    return " ".join(text.split()) or None


def _is_weak(title: str, file: FileRef) -> bool:
    folded = title.casefold()
    return (
        folded in (file.name.casefold(), file.stem.casefold())  # (b) the file's own name
        or folded.endswith(DOC_SUFFIX)  # (b) a path to a document file
        or not any(ch.isalpha() for ch in title)  # (c) only digits and punctuation
        or len(title) < WEAK_TITLE_MIN_CHARS  # (d) too short to mean anything
    )


def display_title(title: str, file_stem: str) -> str:
    """How a weak title is shown: followed by the humanised file stem (unless they match)."""
    stem = humanise(file_stem)
    if not stem or stem.casefold() == title.casefold():
        return title
    return f"{title}{DISPLAY_TITLE_SEPARATOR}{stem}"


# ---------------------------------------------------------------- body units
@dataclass
class _Body:
    obj: Mapping[str, Any]
    units: list[Unit] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)  # fields whose values became units
    header_keys: set[str] = field(default_factory=set)  # family siblings shown as headers

    def add(self, key: str, headers: tuple[str, ...] = ()) -> bool:
        texts = unit_texts(self.obj[key])
        for i, text in enumerate(texts):
            self.units.append(Unit(key, text, key, headers if i == 0 else ()))
        if texts:
            self.keys.add(key)
        return bool(texts)


def _read_listed(doc: ParsedDoc, body: _Body, flags: list[str]) -> None:
    listed = set(doc.content_fields)
    family_of, families = _families([k for k in doc.obj if k not in CONTRACT_KEYS])
    done: set[str] = set()
    completed = False
    for name in doc.content_fields:
        if name in CONTRACT_KEYS:
            continue
        if name not in doc.obj:
            flags.append(f"content_field_missing:{name}")
        elif name not in family_of:
            body.add(name)
        elif family_of[name] not in done:
            done.add(family_of[name])
            completed |= _add_family(families[family_of[name]], listed, body)
    if completed:
        flags.append("family_completed")


def _families(keys: list[str]) -> tuple[dict[str, str], dict[str, list[Member]]]:
    """Group keys that differ only by a number. Returns key -> family id, id -> members."""
    groups: dict[str, list[Member]] = {}
    for key in keys:
        if (m := _SUFFIXED.match(key)) and _has_letter(m["base"]):
            groups.setdefault(f"s:{m['base']}", []).append((int(m["n"]), m["suffix"], key))
        elif (m := _PLAIN.match(key)) and _has_letter(m["base"]):
            groups.setdefault(f"p:{m['base']}", []).append((int(m["n"]), None, key))
    present = set(keys)
    for family_id, members in groups.items():
        base = family_id[2:]
        if family_id.startswith("p:") and base in present:
            members.append((0, None, base))  # the bare key opens the thread
    position = {key: i for i, key in enumerate(keys)}
    families = {
        family_id: sorted(members, key=lambda m: (m[0], position[m[2]]))
        for family_id, members in groups.items()
        if len(members) >= 2
    }
    family_of = {m[2]: family_id for family_id, members in families.items() for m in members}
    return family_of, families


def _add_family(members: list[Member], listed: set[str], body: _Body) -> bool:
    """Add a family's members in number order; True when an unlisted member was added."""
    completed = False
    # Suffixed families: the listed suffixes are the text of each part; the other
    # suffixes of the same number become "suffix: value" header lines. Plain families
    # have no suffixes: every member is text.
    texts = {s for _, s, key in members if key in listed and s is not None}
    by_n: dict[int, list[Member]] = {}
    for member in members:
        by_n.setdefault(member[0], []).append(member)
    for part in by_n.values():
        siblings = [m for m in part if m[1] is not None and m[1] not in texts]
        headers = tuple(h for _, s, key in siblings if (h := _header(s or key, body.obj[key])))
        for member in part:
            if member in siblings:
                continue
            if body.add(member[2], headers):
                body.header_keys.update(key for _, _, key in siblings)
                headers = ()  # only the part's first unit carries them
                completed |= member[2] not in listed
    return completed


def _header(label: str, value: object) -> str:
    text = ", ".join(unit_texts(value)) if isinstance(value, list) else _scalar(value)
    text = " ".join((text or "").split())
    return f"{label}: {text}" if text else ""


# ---------------------------------------------------------------- promotion
def _promote(doc: ParsedDoc, body: _Body, title_field: str | None, flags: list[str]) -> None:
    """Append unlisted prose fields (file key order) that do not repeat the body."""
    seen_text: str | None = None
    seen_shingles: set[tuple[str, ...]] = set()
    for key in doc.obj:
        if key in CONTRACT_KEYS or key in body.keys or key in body.header_keys:
            continue
        if key == title_field:
            continue
        items = unit_texts(doc.obj[key])
        text = "\n".join(items)
        if not _is_prose(text, items):
            continue
        if seen_text is None:  # built only when a candidate exists
            seen_text = _dup_norm(" ".join(u.text for u in body.units))
            seen_shingles = _shingles(seen_text)
        candidate = _dup_norm(text)
        shingles = _shingles(candidate)
        if candidate in seen_text or _jaccard(shingles, seen_shingles) >= DUP_JACCARD:
            flags.append(f"dup_field_skipped:{key}")
            continue
        body.add(key)
        flags.append(f"promoted:{key}")
        seen_text += " " + candidate
        seen_shingles |= shingles


def _is_prose(text: str, items: list[str]) -> bool:
    if len(text) < PROSE_MIN_CHARS or len(text.split()) < PROSE_MIN_WORDS:
        return False
    links = sum(1 for item in items if _LINKISH.match(item.strip()))
    return links / len(items) < LINK_SHAPED_MIN_SHARE


def _dup_norm(text: str) -> str:
    """Comparison form: literal escapes and all whitespace runs become one space."""
    return " ".join(_LITERAL_ESCAPES.sub(" ", text).casefold().split())


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = text.split()
    if len(words) <= DUP_SHINGLE_WORDS:
        return {tuple(words)} if words else set()
    return {
        tuple(words[i : i + DUP_SHINGLE_WORDS]) for i in range(len(words) - DUP_SHINGLE_WORDS + 1)
    }


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------- values as text
def unit_texts(value: object) -> list[str]:
    """Body texts of one value: a string is one, a list gives one per item; blanks dropped.

    Strings are never split on separators. An object renders as "key: value" lines.
    """
    items: Iterable[object] = value if isinstance(value, list) else [value]
    texts = []
    for item in items:
        if isinstance(item, dict):
            text = "\n".join(f"{k}: {_scalar(v) or ''}" for k, v in item.items())
        else:
            text = _scalar(item)
        if text and text.strip():
            texts.append(text)
    return texts


def _scalar(value: object) -> str | None:
    """A JSON value as text: strings as they are, others as JSON (`true`, `42`); null -> None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)
