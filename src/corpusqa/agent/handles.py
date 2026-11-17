"""Handles: the short names one request's tools give to what they show the model.

- `D<n>`: one document a tool result showed (the same document always gets the same D#).
- `R<n>`: a result set, the complete ordered list of ids behind a filter or a key lookup
  (the same list always gets the same R#). The model cites it instead of retyping ids,
  and citing it cites every one of its documents (user decision 2026-10-16: no cap).

A registry lives in one tool session, so handles are never shared between requests. The
tool calls of one model step run at the same time in threads: every method takes a lock.

Handles mean nothing to the reader of an answer, so `name_in_text` replaces each one this
run issued that the answer text holds by what it is (tune run T3, 2026-10-17: "email
thread D3", "(D4)"): a D# by its record's title, source and date, an R# by a short
description of its set. A token the run never issued, or one the question itself holds
(the asker never saw a handle, so it names something real), is left as written.
"""

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from corpusqa.config import DOC_HANDLE_PREFIX, RESULT_HANDLE_PREFIX
from corpusqa.retrieval.errors import ToolInputError
from corpusqa.retrieval.values import quote

# What may surround a handle the model wrote: brackets, quotes, spaces.
_WRAPPING = " \t\n[]()<>{}\"'`"
_HANDLE = re.compile(
    rf"^(?P<kind>{re.escape(DOC_HANDLE_PREFIX)}|{re.escape(RESULT_HANDLE_PREFIX)})(?P<n>\d+)$",
    re.IGNORECASE,
)
# A handle as answer text writes it: the issued spelling, as a whole token (not part of a
# longer word, key, or number such as "D3x", "X-D3", "D3-1" or "D3.5"). Case-sensitive:
# "d3" is left alone.
_TOKEN = rf"(?:{re.escape(DOC_HANDLE_PREFIX)}|{re.escape(RESULT_HANDLE_PREFIX)})\d+"
_SEPARATOR = r"(?:\s*[,;/&]\s*(?:and\s+)?|\s+and\s+)"
_IN_TEXT = re.compile(
    rf"(?P<group>[(\[]\s*{_TOKEN}(?:{_SEPARATOR}{_TOKEN})*\s*[)\]])"  # "(D4)", "[D3, R1]"
    rf"|(?<![\w-])(?P<token>{_TOKEN})(?![\w-]|\.\d)"
)
_WHOLE_TOKEN = re.compile(rf"(?<![\w-]){_TOKEN}(?![\w-]|\.\d)")
_ISSUED = re.compile(
    rf"(?P<kind>{re.escape(DOC_HANDLE_PREFIX)}|{re.escape(RESULT_HANDLE_PREFIX)})(?P<n>\d+)"
)
_CLOSES_HERE = re.compile(r"\s*\)")


@dataclass(frozen=True, slots=True)
class DocName:
    """What one document is, for answer text: its display title, source and date."""

    title: str
    source: str
    date: str | None  # YYYY-MM-DD, or None (no usable date)

    def running(self) -> str:
        """In running text: '"Title" (gmail, 2025-02-21)'."""
        title = " ".join(self.title.split())
        if not title:
            return f"a {self.source} record" + (f" ({self.date})" if self.date else "")
        details = ", ".join(p for p in (self.source, self.date) if p)
        return f'"{title}" ({details})'

    def bracketed(self) -> str:
        """Inside brackets the text already has: '"Title", gmail, 2025-02-21'."""
        title = " ".join(self.title.split())
        first = f'"{title}"' if title else f"a {self.source} record"
        return ", ".join(p for p in (first, None if not title else self.source, self.date) if p)


@dataclass(frozen=True, slots=True)
class Replacement:
    """One issued handle the answer text held, what replaced it, and how many times."""

    handle: str
    text: str
    count: int


@dataclass(frozen=True, slots=True)
class CleanedAnswer:
    """The answer text with this run's handles named (`replacements`); `kept`: issued
    handles left as written because the question holds them."""

    text: str
    replacements: tuple[Replacement, ...] = ()
    kept: tuple[str, ...] = ()

    @property
    def replaced(self) -> int:
        return sum(r.count for r in self.replacements)


class Handles:
    """One request's D# and R# handles, over the index's known document ids."""

    def __init__(self, known_doc_ids: frozenset[str]) -> None:
        self._known = known_doc_ids
        self._lock = threading.Lock()
        self._docs: list[str] = []  # D<n> is self._docs[n - 1]
        self._doc_handles: dict[str, str] = {}
        self._sets: list[tuple[str, ...]] = []  # R<n> is self._sets[n - 1]
        self._set_handles: dict[tuple[str, ...], str] = {}

    def doc(self, doc_id: str) -> str:
        """The D# of a document (a new one the first time it is shown)."""
        with self._lock:
            handle = self._doc_handles.get(doc_id)
            if handle is None:
                self._docs.append(doc_id)
                handle = self._doc_handles[doc_id] = f"{DOC_HANDLE_PREFIX}{len(self._docs)}"
            return handle

    def result_set(self, doc_ids: Sequence[str]) -> str:
        """The R# of an ordered id list (a new one the first time it is seen)."""
        key = tuple(doc_ids)
        with self._lock:
            handle = self._set_handles.get(key)
            if handle is None:
                self._sets.append(key)
                handle = self._set_handles[key] = f"{RESULT_HANDLE_PREFIX}{len(self._sets)}"
            return handle

    def expand(self, token: str) -> tuple[str, ...] | None:
        """The ids behind a citation: D# -> its id, R# -> every id of the set in order, a
        known dataset_doc_uuid -> itself; None for anything else."""
        text = token.strip(_WRAPPING)
        if text in self._known:
            return (text,)
        match = _HANDLE.match(text)
        if match is None:
            return None
        n = int(match["n"])
        with self._lock:
            if match["kind"].upper() == DOC_HANDLE_PREFIX.upper():
                return (self._docs[n - 1],) if 1 <= n <= len(self._docs) else None
            return self._sets[n - 1] if 1 <= n <= len(self._sets) else None

    def documents(self, tokens: Sequence[str]) -> list[str]:
        """The ids behind the handles a tool was given, in order, deduped. Raises
        ToolInputError naming every token that is not a handle or id of this run."""
        ids: dict[str, None] = {}
        unknown = []
        for token in tokens:
            expanded = self.expand(token)
            if expanded is None:
                unknown.append(token)
            else:
                ids.update(dict.fromkeys(expanded))
        if unknown:
            names = ", ".join(quote(t) for t in unknown)
            raise ToolInputError(
                f"not a handle of this run: {names}. Use the D# and R# handles that tool "
                "results showed (or a document id exactly as shown)."
            )
        return list(ids)

    def resolve(self, citations: Sequence[str]) -> list[str]:
        """The dataset_doc_uuids behind the model's citations: every id of each handle,
        known ids kept, anything else dropped; deduped, first-seen order."""
        ids: dict[str, None] = {}
        for citation in citations:
            ids.update(dict.fromkeys(self.expand(citation) or ()))
        return list(ids)

    def name_in_text(
        self,
        text: str,
        doc_name: Callable[[str], DocName],
        set_name: Callable[[tuple[str, ...]], str],
        *,
        question: str = "",
    ) -> CleanedAnswer:
        """`text` with each handle this run issued, written as a whole token, replaced by
        what it is: a D# by `doc_name(id)`, an R# by `set_name(ids)`.

        A bracket holding only issued handles keeps its one pair of brackets: "(D4)" ->
        '("Title", gmail, 2025-02-21)', "(D3, D4)" -> '("A", ...; "B", ...)'. A D# just
        before the ")" of an open bracket takes the bracketed form too ("(see D3)" ->
        '(see "Title", gmail, 2025-02-21)'); anywhere else, the running form ("thread D3"
        -> 'thread "Title" (gmail, 2025-02-21)'). Inserted names are never scanned again.
        Untouched: tokens the run did not issue, and every token the question holds.
        """
        with self._lock:
            namer = _Namer(list(self._docs), list(self._sets), doc_name, set_name, question)
        cleaned = _IN_TEXT.sub(namer.replace, text)
        replacements = tuple(Replacement(h, shown, n) for (h, shown), n in namer.counts.items())
        return CleanedAnswer(cleaned, replacements, tuple(namer.kept))


class _Namer:
    """One `name_in_text` pass: what each issued handle is, and what was replaced."""

    def __init__(
        self,
        docs: list[str],
        sets: list[tuple[str, ...]],
        doc_name: Callable[[str], DocName],
        set_name: Callable[[tuple[str, ...]], str],
        question: str,
    ) -> None:
        self._docs, self._sets = docs, sets
        self._doc_name, self._set_name = doc_name, set_name
        self._in_question = set(_WHOLE_TOKEN.findall(question))
        self.counts: dict[tuple[str, str], int] = {}  # (handle, what replaced it) -> times
        self.kept: dict[str, None] = {}

    def replace(self, match: re.Match[str]) -> str:
        """`_IN_TEXT.sub`'s function: a whole bracket of handles, or one handle."""
        if match["group"] is not None:
            group = match["group"]
            tokens = re.findall(_TOKEN, group)
            targets = [self._target(t) for t in tokens]
            found = [f for f in targets if f is not None]
            if len(found) == len(tokens):
                names = (
                    self._named(t, f, bracketed=True) for t, f in zip(tokens, found, strict=True)
                )
                return group[0] + "; ".join(names) + group[-1]
            # Not every token is an issued handle: name the ones that are, in running form.
            return _WHOLE_TOKEN.sub(lambda m: self._one(m[0], bracketed=False), group)
        text, (start, end) = match.string, match.span()
        in_bracket = _CLOSES_HERE.match(text, end) is not None and (
            text.rfind("(", 0, start) > text.rfind(")", 0, start)
        )
        return self._one(match["token"], bracketed=in_bracket)

    def _one(self, token: str, *, bracketed: bool) -> str:
        found = self._target(token)
        return token if found is None else self._named(token, found, bracketed=bracketed)

    def _target(self, token: str) -> str | tuple[str, ...] | None:
        """The id (D#) or ids (R#) behind a handle this run issued, in its issued spelling
        ("D03" is not D3); None for any other token and for a token the question holds."""
        match = _ISSUED.fullmatch(token)
        if match is None or match["n"] != str(int(match["n"])):
            return None
        n = int(match["n"])
        if match["kind"] == DOC_HANDLE_PREFIX:
            found: str | tuple[str, ...] | None = (
                self._docs[n - 1] if 1 <= n <= len(self._docs) else None
            )
        else:
            found = self._sets[n - 1] if 1 <= n <= len(self._sets) else None
        if found is not None and token in self._in_question:
            self.kept[token] = None
            return None
        return found

    def _named(self, token: str, found: str | tuple[str, ...], *, bracketed: bool) -> str:
        if isinstance(found, str):
            name = self._doc_name(found)
            shown = name.bracketed() if bracketed else name.running()
        else:
            shown = self._set_name(found)
        self.counts[token, shown] = self.counts.get((token, shown), 0) + 1
        return shown
