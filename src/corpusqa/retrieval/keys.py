"""Key lookup: a document's own identifier (an issue key and the like) -> the documents
that own it, then the documents that refer to it.

Owners come from `doc_keys` (own-id field values, file stems, keys a stem starts with).
Referrers come from `links`, resolved or not (on data/ under 4% resolve): a link whose
key is the asked key, whose key holds it as a whole token (e.g. at the end of a URL), or
that resolved to an owner. Mentions in free text are search's job.
"""

from dataclasses import dataclass

from corpusqa import obs
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.values import quote
from corpusqa.store.interfaces import CorpusReader, LinkRow
from corpusqa.store.norms import normalise


@dataclass(frozen=True, slots=True)
class Referrer:
    """A document that refers to the key: each (field, value as written) that does."""

    doc_id: str
    refs: tuple[tuple[str, str], ...]
    resolved: bool  # at least one of its links was resolved to an owner at ingest


@dataclass(frozen=True, slots=True)
class KeyLookup:
    asked: str
    key_norm: str
    owners: tuple[str, ...]
    hub: bool  # the key is held by too many documents to name one (ingest's hub rule)
    referrers: tuple[Referrer, ...]  # owners left out, in stored order


def lookup_key(reader: CorpusReader, asked: str) -> KeyLookup:
    with obs.span("lookup_key", input={"key": asked}) as span:
        key = normalise(asked)
        if not any(ch.isalnum() for ch in key):
            raise ToolInputError(f"the key {quote(asked)} has no letters or digits.")
        owned = reader.key_owners(key)
        owners = tuple(dict.fromkeys(row.doc_id for row in owned))
        by_doc: dict[str, list[LinkRow]] = {}
        for link in reader.links_to(key, owners):
            if link.doc_id not in owners and _refers(link, key, owners):
                by_doc.setdefault(link.doc_id, []).append(link)
        referrers = tuple(
            Referrer(
                doc_id=doc_id,
                refs=tuple(dict.fromkeys((link.field, link.raw) for link in links)),
                resolved=any(link.target_doc_id in owners for link in links),
            )
            for doc_id, links in by_doc.items()
        )
        result = KeyLookup(asked, key, owners, any(row.is_hub for row in owned), referrers)
        span.update(
            output={"owners": list(owners), "referrers": [r.doc_id for r in referrers]},
            metadata={"key_norm": key, "hub": result.hub, "owner_kinds": [r.kind for r in owned]},
        )
    return result


def _refers(link: LinkRow, key: str, owners: tuple[str, ...]) -> bool:
    """The link names the key: resolved to an owner, the same key, or the key as a whole
    token inside it (not "abc-1" inside "abc-12")."""
    if link.target_doc_id is not None and link.target_doc_id in owners:
        return True
    text = link.key_norm
    start = text.find(key)
    while start != -1:
        end = start + len(key)
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        start = text.find(key, start + 1)
    return False
