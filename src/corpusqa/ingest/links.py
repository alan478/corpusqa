"""S8 keys and links: each document's own identifiers, and its references resolved to
other documents when that is safe (design.md S8 "Keys", "Links").

Keys (many-to-many: duplicates exist): every value of the document's own-id fields, its
file stem, and the issue-style key a stem starts with. A key made only of digits never
stands alone: it becomes "<parent folder>/<number>" (kind folder_number), so the file
system gives the namespace and no reference resolves by a bare number. A key held by
more than HUB_KEY_MAX_DOCS documents is a hub.

Links: each value of a reference-kind field. Resolution, in order, first that finds a
document other than the referring one through a key that is not a hub: (1) the value, or
an issue-style key inside it, equals an owned key; (2) its last path segment equals a
file stem; (3) "name#123" or ".../name/<word>/123" equals a folder_number key; (4) the
whole value equals a title held by exactly one document (weak titles excluded).
Nothing resolves by number alone. A value that only finds itself is method "self"; one
that only finds hub keys keeps no target and is marked target_is_hub.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from corpusqa.config import HUB_KEY_MAX_DOCS
from corpusqa.ingest.profile import reference_target
from corpusqa.store.interfaces import DocFacts, DocKeyRow, FieldValue, LinkRow
from corpusqa.store.norms import normalise

STEM, STEM_KEY, FOLDER_NUMBER = "stem", "stem_key", "folder_number"
FIELD_KIND_PREFIX = "field:"
KEY, UNIQUE_TITLE, SELF, NONE = "key", "unique_title", "self", "none"

# An issue-style key at the start of a file stem, and anywhere inside a value.
_STEM_KEY = re.compile(r"^([A-Z][A-Z0-9]{1,9}-\d+)(?!\d)")
_OWN_KEY = re.compile(r"^([A-Z][A-Z0-9]{1,9})-\d+$")
_KEY_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,9}-\d+(?!\d)")
_HASH_NUMBER = re.compile(r"(?:^|/)(?P<name>[^/#\s]+)#(?P<n>\d+)$")
_SCHEME_HOST = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/]*")
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,5}$")


@dataclass(frozen=True, slots=True)
class Keys:
    """Keys of the corpus and the prefixes learned from them."""

    rows: tuple[DocKeyRow, ...]
    prefixes: Mapping[str, tuple[str, ...]]  # source -> sorted key prefixes


def parent_folder(doc: DocFacts) -> str:
    """The folder that holds the file (the source folder for a file directly in it)."""
    parts = doc.rel_path.split("/")
    return parts[-2] if len(parts) >= 2 else doc.source


def folder_number(folder: str, digits: str) -> str:
    return f"{normalise(folder)}/{int(digits)}"


def build_keys(docs: Sequence[DocFacts], own_id_rows: Iterable[FieldValue]) -> Keys:
    """Doc keys from own-id field values and file stems; prefixes from own keys only."""
    by_id = {d.doc_id: d for d in docs}
    found: dict[tuple[str, str, str], None] = {}  # (doc_id, key_norm, kind), in order
    prefixes: dict[str, set[str]] = {}
    for row in own_id_rows:
        if row.is_noise or not row.value or not row.value_norm:
            continue
        doc = by_id[row.doc_id]
        value = row.value.strip()
        if row.value_norm.isdigit():
            found[
                (doc.doc_id, folder_number(parent_folder(doc), row.value_norm), FOLDER_NUMBER)
            ] = None
        else:
            found[(doc.doc_id, row.value_norm, FIELD_KIND_PREFIX + row.field)] = None
        if (m := _OWN_KEY.match(value)) is not None:
            prefixes.setdefault(doc.source, set()).add(m[1])
    for doc in docs:
        stem = doc.file_stem.strip()
        if stem.isdigit():
            found[(doc.doc_id, folder_number(parent_folder(doc), stem), FOLDER_NUMBER)] = None
        elif norm := normalise(stem):
            found[(doc.doc_id, norm, STEM)] = None
        if (m := _STEM_KEY.match(stem)) is not None:
            found[(doc.doc_id, normalise(m[1]), STEM_KEY)] = None
            prefixes.setdefault(doc.source, set()).add(m[1].rsplit("-", 1)[0])
    held = {(doc_id, key) for doc_id, key, _ in found}
    docs_per_key = Counter(key for _, key in held)
    rows = tuple(
        DocKeyRow(doc_id, key, kind, docs_per_key[key] > HUB_KEY_MAX_DOCS)
        for doc_id, key, kind in found
    )
    return Keys(rows, {s: tuple(sorted(p)) for s, p in sorted(prefixes.items())})


class Resolver:
    """Finds the documents a reference value points to (see the module docstring)."""

    def __init__(self, keys: Sequence[DocKeyRow], docs: Sequence[DocFacts]) -> None:
        self.owned: dict[str, set[str]] = {}
        self.stems: dict[str, set[str]] = {}
        self.folders: dict[str, set[str]] = {}
        self.hubs: set[str] = set()
        for row in keys:
            index = {STEM: self.stems, FOLDER_NUMBER: self.folders}.get(row.kind, self.owned)
            index.setdefault(row.key_norm, set()).add(row.doc_id)
            if row.is_hub:
                self.hubs.add(row.key_norm)
        titles: dict[str, set[str]] = {}
        for doc in docs:
            if not doc.title_is_weak and (norm := normalise(doc.title)):
                titles.setdefault(norm, set()).add(doc.doc_id)
        self.titles = {t: ids for t, ids in titles.items() if len(ids) == 1}

    def resolve(self, doc_id: str, field: str, raw: str) -> list[LinkRow]:
        target = reference_target(raw)
        key_norm = normalise(target)
        steps: list[tuple[str, list[str], Mapping[str, set[str]]]] = [
            (KEY, _key_candidates(raw, key_norm), self.owned),
            (STEM, _stem_candidates(target), self.stems),
            (FOLDER_NUMBER, _folder_candidates(target), self.folders),
            (UNIQUE_TITLE, [key_norm], self.titles),
        ]
        hub_method: str | None = None
        found_self = False
        for method, candidates, index in steps:
            targets: set[str] = set()
            for candidate in dict.fromkeys(c for c in candidates if c):
                ids = index.get(candidate)
                if not ids:
                    continue
                if candidate in self.hubs and method != UNIQUE_TITLE:
                    hub_method = hub_method or method
                    continue
                targets |= ids
            found_self |= doc_id in targets
            targets.discard(doc_id)
            if targets:
                return [
                    LinkRow(doc_id, field, raw, key_norm, t, method, target_is_hub=False)
                    for t in sorted(targets)
                ]
        if hub_method is not None:
            return [LinkRow(doc_id, field, raw, key_norm, None, hub_method, target_is_hub=True)]
        method = SELF if found_self else NONE
        return [LinkRow(doc_id, field, raw, key_norm, None, method, target_is_hub=False)]


def build_links(reference_rows: Iterable[FieldValue], resolver: Resolver) -> list[LinkRow]:
    """One or more link rows per distinct (document, field, value) of reference fields."""
    seen: set[tuple[str, str, str]] = set()
    links: list[LinkRow] = []
    for row in reference_rows:
        if row.is_noise or not row.value or not row.value.strip():
            continue
        key = (row.doc_id, row.field, row.value)
        if key in seen:
            continue
        seen.add(key)
        links.extend(resolver.resolve(row.doc_id, row.field, row.value))
    return links


def _key_candidates(raw: str, key_norm: str) -> list[str]:
    return [key_norm, *(normalise(token) for token in _KEY_TOKEN.findall(raw))]


def _path_of(target: str) -> str:
    """The path part of a URL or path: no scheme or host, query or fragment."""
    path = _SCHEME_HOST.sub("", target)
    return path.split("?", 1)[0].split("#", 1)[0].rstrip("/")


def _stem_candidates(target: str) -> list[str]:
    segment = _path_of(target).rsplit("/", 1)[-1]
    if not segment or segment.isdigit():
        return []
    return [normalise(segment), normalise(_EXTENSION.sub("", segment))]


def _folder_candidates(target: str) -> list[str]:
    candidates = []
    if (m := _HASH_NUMBER.search(target.split("?", 1)[0])) is not None:
        candidates.append(folder_number(m["name"], m["n"]))
    segments = [s for s in _path_of(target).split("/") if s]
    if segments and segments[-1].isdigit():
        number = segments[-1]
        for back in (3, 2):  # ".../name/<word>/123", then ".../name/123"
            if len(segments) >= back:
                candidates.append(folder_number(segments[-back], number))
    return candidates
