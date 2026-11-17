"""S8 profile: what each field of each source holds, learned from the stored rows.

Runs once per corpus, after every document is stored (design.md S8). One pass over the
field rows gathers statistics per (source, field); then, from values alone:

- a first-name lexicon: first tokens of two-token capitalised names, from fields made
  almost only of such names, kept when they start several distinct names;
- a kind per field with a confidence (the share of its values that fit the kind);
- synonym groups: fields shown together in the agent's field list;
- conditional presence: "field F is present when enum field C is one of ...";
- weak-title rule (a): a title field whose values repeat across its source;
- per-source summaries.

A field is present in a document when it has field rows there or its text went into
the body (`documents.body_fields`). Nothing here names a source, field or value of any
corpus; the field name is shown to the agent, never read by these rules.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from corpusqa.config import (
    CONDITIONAL_MAX_VALUES,
    CONDITIONAL_MIN_STRENGTH,
    CONDITIONAL_MIN_SUPPORT,
    DATED_LOG_MIN_SHARE,
    ENUM_MAX_DISTINCT,
    ENUM_MIN_DOCS_PER_VALUE,
    KIND_BOOLEAN_MAX_DISTINCT,
    KIND_BOOLEAN_MIN_SHARE,
    KIND_DATE_MIN_SHARE,
    KIND_EMAIL_MIN_SHARE,
    KIND_LIST_MIN_RATIO,
    KIND_NUMBER_MIN_SHARE,
    KIND_REFERENCE_MIN_SHARE,
    KIND_SCALAR_MAX_LIST_RATIO,
    KIND_URL_MIN_SHARE,
    LEXICON_FIELD_MIN_SHARE,
    LEXICON_MIN_COUNT,
    OWN_ID_MAX_MEDIAN_LEN,
    OWN_ID_MIN_DIGIT_SHARE,
    OWN_ID_MIN_DISTINCT_RATIO,
    OWN_ID_MIN_PRESENCE,
    OWN_ID_MIN_TOKEN_SHARE,
    PERSON_LEXICON_MIN_SHARE,
    PERSON_MAX_TOKENS,
    PERSON_MIN_TOKENS,
    PERSON_REUSED_MIN_SHARE,
    PERSON_SHAPED_MIN_SHARE,
    PROFILE_EXAMPLE_DOCS,
    PROFILE_TOP_VALUE_CHARS,
    PROFILE_TOP_VALUES,
    REFERENCE_LABEL_MAX_CHARS,
    ROOT_SOURCE,
    SHAPE_MAX_CHARS,
    STEM_TITLE_MIN_OVERLAP,
    SYNONYM_MIN_VALUE_OVERLAP,
    SYNONYM_VALUE_KINDS,
    TAG_MAX_CHARS,
    TAGS_MIN_DOCS_PER_VALUE,
    TAGS_MIN_SINGLE_TOKEN_SHARE,
    TEXT_MIN_CHARS,
    TEXT_MIN_SHARE,
    TEXT_MIN_WORDS,
    WEAK_TITLE_DISTINCT_RATIO,
)
from corpusqa.ingest.core import WEAK_TITLE_FLAG, display_title
from corpusqa.ingest.dates import parse_date
from corpusqa.store.interfaces import (
    DocFacts,
    FieldConditionRow,
    FieldProfileRow,
    FieldValue,
    SourceProfileRow,
    TitleUpdate,
)
from corpusqa.store.norms import fold_key, strip_person

# Kinds, in the order they are tried (see `classify`).
DATE, BOOLEAN, OWN_ID, NUMBER = "date", "boolean", "own_id", "number"
REFERENCE, URL, EMAIL, PERSON, ENUM = "reference", "url", "email", "person", "enum"
TAGS, DATED_LOG, TEXT, MIXED, UNKNOWN = "tags", "dated_log", "text", "mixed", "unknown"

# Value types that carry a value (not null, empty or a nested object).
_VALUED_TYPES = frozenset({"str", "int", "float", "bool"})
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^\s/?#]+(?P<path>[/?#]\S*)?$")
_PATH = re.compile(r"^[^\s/]*[^\s/]/[^\s/]\S*$")  # a slash between two characters, no spaces
_PATH_MIN_CHARS = 4  # "n/a" is not a path
_KEY = re.compile(r"^[A-Z][A-Z0-9]{1,9}-\d+$")  # an issue-style key
# Letter/digit runs joined by separators, with a letter and a digit somewhere.
_SLUG = re.compile(r"^[^\W_]+(?:[-_.:#][^\W_]+)+$")
_LABEL = re.compile(rf"^[^:\s][^:]{{0,{REFERENCE_LABEL_MAX_CHARS - 1}}}:\s+(?P<rest>\S+)$")
_EMAIL = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")
_NAMED_EMAIL = re.compile(r"^[^<>]*<[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+>$")
_DOTTED_HANDLE = re.compile(r"^@?[^\W\d_]+(?:\.[^\W\d_]+)+$")  # first.last
_AT_HANDLE = re.compile(r"^@[^\W_][\w.-]*$")
_NAME_PUNCT = frozenset("'\u2019-.")  # apostrophes, hyphen, dot
_DATE_EDGE = "[](),:;- \t"
_WORD = re.compile(r"[^\W_]+")


# ---------------------------------------------------------------- value shapes
def reference_target(value: str) -> str:
    """The part of a reference value that names the target: "<label>: <target>" loses
    its label when the rest is one token (a label tells the target's kind)."""
    text = value.strip()
    match = _LABEL.match(text)
    return match["rest"] if match else text


def is_path(text: str) -> bool:
    """Parts joined by slashes, no spaces, with a letter (a slash date is not a path)."""
    return len(text) >= _PATH_MIN_CHARS and bool(_PATH.match(text)) and _has_letter(text)


def is_url(text: str) -> bool:
    return bool(_URL.match(text)) or is_path(text)


def is_reference(value: str) -> bool:
    """A URL with a path, a path, an issue-style key or a digit slug (after the label)."""
    text = reference_target(value)
    if (m := _URL.match(text)) is not None:
        return (m["path"] or "/") != "/"
    if _KEY.match(text) or is_path(text):
        return True
    return bool(_SLUG.match(text)) and _has_digit(text) and _has_letter(text)


def is_email(text: str) -> bool:
    return bool(_EMAIL.match(text) or _NAMED_EMAIL.match(text))


def _is_name_token(token: str) -> bool:
    return (
        token[0].isalpha()
        and token[0].isupper()
        and all(ch.isalpha() or ch in _NAME_PUNCT for ch in token)
        and any(ch.islower() for ch in token)
    )


def two_token_name(value: str) -> str | None:
    """'First Last' (capitalised) as a case-folded name, else None."""
    tokens = strip_person(value).split()
    if len(tokens) == 2 and all(_is_name_token(t) for t in tokens):
        return " ".join(t.casefold() for t in tokens)
    return None


def is_person_shaped(value: str) -> bool:
    """2-4 capitalised name words, or a handle (`@name`, `first.last`)."""
    text = strip_person(value)
    tokens = text.split()
    if PERSON_MIN_TOKENS <= len(tokens) <= PERSON_MAX_TOKENS and all(
        _is_name_token(t) for t in tokens
    ):
        return True
    return len(tokens) == 1 and bool(_DOTTED_HANDLE.match(text) or _AT_HANDLE.match(text))


def starts_with_date(text: str) -> bool:
    tokens = text.split(maxsplit=3)
    if not tokens:
        return False
    if parse_date(tokens[0].strip(_DATE_EDGE)) is not None:
        return True
    return len(tokens) >= 3 and parse_date(" ".join(tokens[:3]).strip(_DATE_EDGE)) is not None


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


# ---------------------------------------------------------------- statistics (one pass)
@dataclass
class FieldStats:
    """What one pass over a field's rows learned. Docs are indexes into the doc list."""

    source: str
    name: str  # the field key
    field_norm: str
    rows: int = 0
    noise: int = 0
    valued: int = 0  # non-noise rows holding a value
    docs: set[int] = field(default_factory=set)  # docs with field rows
    body_docs: set[int] = field(default_factory=set)  # docs whose body holds its text
    list_docs: set[int] = field(default_factory=set)
    title_docs: set[int] = field(default_factory=set)
    valued_docs: set[int] = field(default_factory=set)
    types: Counter[str] = field(default_factory=Counter)
    norms: Counter[str] = field(default_factory=Counter)  # value_norm -> valued rows
    spelling: dict[str, str] = field(default_factory=dict)  # value_norm -> first raw value
    lengths: Counter[int] = field(default_factory=Counter)
    shapes: Counter[str] = field(default_factory=Counter)
    first_tokens: Counter[str] = field(default_factory=Counter)  # of two-token names
    names: set[str] = field(default_factory=set)  # distinct two-token names
    # doc -> value_norm, kept only while the field has few distinct values (conditions).
    doc_value: dict[int, str] | None = field(default_factory=dict)

    @property
    def all_docs(self) -> set[int]:
        return self.docs | self.body_docs

    def share(self, shape: str) -> float:
        return self.shapes[shape] / self.valued if self.valued else 0.0

    def add(self, row: FieldValue, doc: int) -> None:
        self.rows += 1
        self.types[row.value_type] += 1
        self.docs.add(doc)
        if row.in_list:
            self.list_docs.add(doc)
        if row.role == "title":
            self.title_docs.add(doc)
        if row.is_noise:
            self.noise += 1
            return
        if row.value is None or row.value_type not in _VALUED_TYPES:
            return
        self.valued += 1
        self.valued_docs.add(doc)
        norm = row.value_norm or ""
        self.norms[norm] += 1
        self.spelling.setdefault(norm, row.value)
        if self.doc_value is not None:
            self.doc_value.setdefault(doc, norm)
            if len(self.norms) > CONDITIONAL_MAX_VALUES:
                self.doc_value = None
        self.lengths[len(row.value)] += 1
        self.shapes.update(_shapes(row))
        if (name := two_token_name(row.value)) is not None:
            self.first_tokens[name.split()[0]] += 1
            self.names.add(name)


def _shapes(row: FieldValue) -> list[str]:
    value = row.value or ""
    text = value.strip()
    found = []
    if row.value_iso is not None:
        found.append("iso")
    if row.value_num is not None:
        found.append("number")
    if row.value_norm in ("true", "false"):
        found.append("boolean")
    if _has_digit(text):
        found.append("digit")
    if len(text.split()) > TEXT_MIN_WORDS or len(text) > TEXT_MIN_CHARS:
        found.append("long")
    if text and len(text) <= TAG_MAX_CHARS and not any(ch.isspace() for ch in text):
        found.append("token")
    if len(text) <= SHAPE_MAX_CHARS:
        if is_reference(text):
            found.append("reference")
        if is_url(text):
            found.append("url")
        if is_email(text):
            found.append("email")
        if two_token_name(text) is not None:
            found.append("name2")
        if is_person_shaped(text):
            found.append("person_shaped")
    if row.in_list and starts_with_date(text):
        found.append("dated")
    return found


def gather(
    values: Iterable[FieldValue], docs: Sequence[DocFacts]
) -> dict[tuple[str, str], FieldStats]:
    """Statistics per (source, field) from every field row and every body field."""
    index = {d.doc_id: i for i, d in enumerate(docs)}
    stats: dict[tuple[str, str], FieldStats] = {}
    for row in values:
        key = (row.source, row.field)
        st = stats.get(key)
        if st is None:
            st = stats[key] = FieldStats(row.source, row.field, row.field_norm)
        st.add(row, index[row.doc_id])
    for i, doc in enumerate(docs):
        for name in dict.fromkeys(doc.body_fields):
            key = (doc.source, name)
            st = stats.get(key)
            if st is None:
                st = stats[key] = FieldStats(doc.source, name, fold_key(name))
            st.body_docs.add(i)
    return dict(sorted(stats.items()))


# ---------------------------------------------------------------- lexicon and kinds
def learn_lexicon(stats: Iterable[FieldStats]) -> frozenset[str]:
    """First names: first tokens that start at least LEXICON_MIN_COUNT distinct names in
    fields where nearly every value is a two-token capitalised name."""
    names_by_first: dict[str, set[str]] = {}
    for st in stats:
        if st.valued and st.share("name2") >= LEXICON_FIELD_MIN_SHARE:
            for name in st.names:
                names_by_first.setdefault(name.split()[0], set()).add(name)
    return frozenset(t for t, names in names_by_first.items() if len(names) >= LEXICON_MIN_COUNT)


def reused_values(stats: Iterable[FieldStats]) -> frozenset[str]:
    """value_norms held by two or more (source, field) pairs."""
    holders: Counter[str] = Counter()
    for st in stats:
        holders.update(st.norms.keys())
    return frozenset(v for v, n in holders.items() if n >= 2)


@dataclass(frozen=True, slots=True)
class Kind:
    kind: str
    confidence: float


def classify(
    st: FieldStats, source_docs: int, lexicon: frozenset[str], reused: frozenset[str]
) -> Kind:
    """The kind of one field, from its values alone. Rules are tried in this order:
    date, boolean, own_id, number, reference, url, email, person (lexicon), enum,
    person (shape + reuse), tags, dated_log, text; else mixed. A field with no value of
    its own is text when its text went into bodies, else unknown."""
    if not st.valued:
        n_docs = len(st.all_docs)
        content = len(st.body_docs) / n_docs if n_docs else 0.0
        return Kind(TEXT, round(content, 4)) if st.body_docs else Kind(UNKNOWN, 0.0)
    list_ratio = len(st.list_docs) / len(st.docs) if st.docs else 0.0
    scalar = list_ratio <= KIND_SCALAR_MAX_LIST_RATIO
    is_list = list_ratio >= KIND_LIST_MIN_RATIO
    distinct = len(st.norms)
    per_value = st.valued / distinct
    for kind, share in _share_kinds(st, scalar, source_docs, lexicon):
        if share is not None:
            return Kind(kind, round(share, 4))
    if scalar and distinct <= ENUM_MAX_DISTINCT and per_value >= ENUM_MIN_DOCS_PER_VALUE:
        repeated = sum(n for n in st.norms.values() if n >= 2) / st.valued
        return Kind(ENUM, round(repeated, 4))
    shaped = st.share("person_shaped")
    if shaped >= PERSON_SHAPED_MIN_SHARE:
        reuse = sum(1 for v in st.norms if v in reused) / distinct
        if reuse >= PERSON_REUSED_MIN_SHARE:
            return Kind(PERSON, round(min(shaped, reuse), 4))
    tokens = st.share("token")
    if is_list and tokens >= TAGS_MIN_SINGLE_TOKEN_SHARE and per_value >= TAGS_MIN_DOCS_PER_VALUE:
        return Kind(TAGS, round(tokens, 4))
    if is_list and (dated := st.share("dated")) >= DATED_LOG_MIN_SHARE:
        return Kind(DATED_LOG, round(dated, 4))
    if (long := st.share("long")) >= TEXT_MIN_SHARE:
        return Kind(TEXT, round(long, 4))
    return Kind(MIXED, 0.0)


def _share_kinds(
    st: FieldStats, scalar: bool, source_docs: int, lexicon: frozenset[str]
) -> Iterable[tuple[str, float | None]]:
    """(kind, confidence or None when the rule fails) for the rules before enum, in order."""
    yield DATE, _at_least(st.share("iso"), KIND_DATE_MIN_SHARE)
    boolean = st.share("boolean")
    few = len(st.norms) <= KIND_BOOLEAN_MAX_DISTINCT
    yield BOOLEAN, _at_least(boolean, KIND_BOOLEAN_MIN_SHARE) if few else None
    yield OWN_ID, _own_id(st, scalar, source_docs)
    yield NUMBER, _at_least(st.share("number"), KIND_NUMBER_MIN_SHARE)
    yield REFERENCE, _at_least(st.share("reference"), KIND_REFERENCE_MIN_SHARE)
    yield URL, _at_least(st.share("url"), KIND_URL_MIN_SHARE)
    yield EMAIL, _at_least(st.share("email"), KIND_EMAIL_MIN_SHARE)
    in_lexicon = sum(n for t, n in st.first_tokens.items() if t in lexicon) / st.valued
    yield PERSON, _at_least(in_lexicon, PERSON_LEXICON_MIN_SHARE)


def _own_id(st: FieldStats, scalar: bool, source_docs: int) -> float | None:
    """Scalar, nearly always present, nearly unique, digits inside, short, one token."""
    presence = len(st.valued_docs) / source_docs if source_docs else 0.0
    distinct_ratio = len(st.norms) / st.valued
    if (
        scalar
        and presence >= OWN_ID_MIN_PRESENCE
        and distinct_ratio >= OWN_ID_MIN_DISTINCT_RATIO
        and st.share("digit") >= OWN_ID_MIN_DIGIT_SHARE
        and st.share("token") >= OWN_ID_MIN_TOKEN_SHARE
        and _median(st.lengths) <= OWN_ID_MAX_MEDIAN_LEN
    ):
        return distinct_ratio
    return None


def _at_least(share: float, minimum: float) -> float | None:
    return share if share >= minimum else None


def _median(counts: Counter[int]) -> int:
    """Lower median of a {value: count} table."""
    total = counts.total()
    seen = 0
    for value in sorted(counts):
        seen += counts[value]
        if 2 * seen >= total:
            return value
    return 0


# ---------------------------------------------------------------- synonyms, conditions
def synonym_groups(
    stats: Mapping[tuple[str, str], FieldStats], kinds: Mapping[tuple[str, str], Kind]
) -> dict[tuple[str, str], int]:
    """Group number per field that shares a group (fields alone get none).

    Same source, and either the same folded name, or never in the same document with the
    same kind (one whose values name things) and overlapping values.
    """
    by_source: dict[str, list[FieldStats]] = {}
    for st in stats.values():
        by_source.setdefault(st.source, []).append(st)
    groups: dict[tuple[str, str], int] = {}
    next_group = 1
    for source in sorted(by_source):
        members = sorted(by_source[source], key=lambda s: s.name)
        parent = {st.name: st.name for st in members}

        def find(name: str, parent: dict[str, str] = parent) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(a: str, b: str, parent: dict[str, str] = parent) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for a, b in combinations(members, 2):
            if a.field_norm == b.field_norm:
                union(a.name, b.name)
                continue
            kind = kinds[(source, a.name)].kind
            if kind != kinds[(source, b.name)].kind or kind not in SYNONYM_VALUE_KINDS:
                continue
            if a.all_docs.isdisjoint(b.all_docs) and _overlap(a.norms, b.norms):
                union(a.name, b.name)
        clusters: dict[str, list[str]] = {}
        for st in members:
            clusters.setdefault(find(st.name), []).append(st.name)
        for root in sorted(clusters):
            if len(clusters[root]) >= 2:
                for name in clusters[root]:
                    groups[(source, name)] = next_group
                next_group += 1
    return groups


def _overlap(a: Counter[str], b: Counter[str]) -> bool:
    if not a or not b:
        return False
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    shared = sum(1 for v in small if v in big)
    return shared / len(small) >= SYNONYM_MIN_VALUE_OVERLAP


def conditions(
    stats: Mapping[tuple[str, str], FieldStats], kinds: Mapping[tuple[str, str], Kind]
) -> list[FieldConditionRow]:
    """For each enum field C with few values and each other field F of its source: how
    much F's presence depends on C's value (max - min over C's well-supported values)."""
    by_source: dict[str, list[FieldStats]] = {}
    for st in stats.values():
        by_source.setdefault(st.source, []).append(st)
    rows: list[FieldConditionRow] = []
    for (source, name), given in stats.items():
        if kinds[(source, name)].kind != ENUM or given.doc_value is None:
            continue
        docs_by_value: dict[str, set[int]] = {}
        for doc, value in given.doc_value.items():
            docs_by_value.setdefault(value, set()).add(doc)
        supported = {
            v: docs for v, docs in docs_by_value.items() if len(docs) >= CONDITIONAL_MIN_SUPPORT
        }
        if len(supported) < 2:
            continue
        for other in by_source[source]:
            if other is given:
                continue
            present = other.all_docs
            presence = {v: len(docs & present) / len(docs) for v, docs in supported.items()}
            high, low = max(presence.values()), min(presence.values())
            if high - low < CONDITIONAL_MIN_STRENGTH:
                continue
            middle = (high + low) / 2
            when = sorted(
                (v for v, p in presence.items() if p >= middle), key=lambda v: (-presence[v], v)
            )
            rows.append(
                FieldConditionRow(
                    source=source,
                    field=other.name,
                    given_field=name,
                    present_when=tuple(given.spelling[v] for v in when),
                    strength=round(high - low, 4),
                )
            )
    return rows


# ---------------------------------------------------------------- titles, rows
def shared_titles(docs: Sequence[DocFacts]) -> list[TitleUpdate]:
    """Weak-title rule (a): documents whose title field repeats its values across the
    source (distinct ratio below WEAK_TITLE_DISTINCT_RATIO) get the file stem shown too."""
    groups: dict[tuple[str, str], list[DocFacts]] = {}
    for doc in docs:
        if doc.title_field is not None:
            groups.setdefault((doc.source, doc.title_field), []).append(doc)
    updates: list[TitleUpdate] = []
    for members in groups.values():
        distinct = len({" ".join(d.title.casefold().split()) for d in members})
        if distinct / len(members) >= WEAK_TITLE_DISTINCT_RATIO:
            continue
        for d in members:
            if not d.title_is_weak:
                flags = d.flags if WEAK_TITLE_FLAG in d.flags else (*d.flags, WEAK_TITLE_FLAG)
                updates.append(TitleUpdate(d.doc_id, display_title(d.title, d.file_stem), flags))
    return updates


def field_profile_rows(
    stats: Mapping[tuple[str, str], FieldStats],
    kinds: Mapping[tuple[str, str], Kind],
    groups: Mapping[tuple[str, str], int],
    source_docs: Mapping[str, int],
) -> list[FieldProfileRow]:
    rows = []
    for key, st in stats.items():
        n_docs = len(st.all_docs)
        top = tuple(
            (st.spelling[v][:PROFILE_TOP_VALUE_CHARS], n)
            for v, n in st.norms.most_common(PROFILE_TOP_VALUES)
        )
        total_len = sum(length * n for length, n in st.lengths.items())
        rows.append(
            FieldProfileRow(
                source=st.source,
                field=st.name,
                field_norm=st.field_norm,
                synonym_group=groups.get(key),
                n_docs=n_docs,
                presence=_ratio(n_docs, source_docs[st.source]),
                n_values=st.rows,
                n_distinct=len(st.norms),
                list_ratio=_ratio(len(st.list_docs), len(st.docs)),
                type_mix=dict(sorted(st.types.items())),
                kind=kinds[key].kind,
                kind_confidence=kinds[key].confidence,
                title_ratio=_ratio(len(st.title_docs), n_docs),
                content_ratio=_ratio(len(st.body_docs), n_docs),
                noise_ratio=_ratio(st.noise, st.rows),
                avg_len=_ratio(total_len, st.valued),
                max_len=max(st.lengths, default=0),
                top_values=top,
            )
        )
    return rows


def source_profile_rows(
    docs: Sequence[DocFacts],
    kinds: Mapping[tuple[str, str], Kind],
    prefixes: Mapping[str, Sequence[str]],
) -> list[SourceProfileRow]:
    by_source: dict[str, list[DocFacts]] = {}
    for doc in docs:
        by_source.setdefault(doc.source, []).append(doc)
    rows = []
    for source, members in sorted(by_source.items()):
        titles = Counter(d.title_field for d in members if d.title_field is not None)
        content = Counter(name for d in members for name in dict.fromkeys(d.body_fields))
        depths = Counter(str(_depth(d)) for d in members)
        matches = sum(_stem_matches_title(d) for d in members)

        def of_kind(kind: str, source: str = source) -> tuple[str, ...]:
            return tuple(sorted(f for (s, f), k in kinds.items() if s == source and k.kind == kind))

        rows.append(
            SourceProfileRow(
                source=source,
                n_docs=len(members),
                title_fields=dict(titles.most_common()),
                content_fields=dict(content.most_common()),
                own_id_fields=of_kind(OWN_ID),
                key_prefixes=tuple(prefixes.get(source, ())),
                date_fields=of_kind(DATE),
                example_doc_ids=tuple(d.doc_id for d in members[:PROFILE_EXAMPLE_DOCS]),
                path_depths=dict(sorted(depths.items(), key=lambda kv: int(kv[0]))),
                stem_title_match=_ratio(matches, len(members)),
            )
        )
    return rows


def _depth(doc: DocFacts) -> int:
    """Folders between the source folder and the file."""
    parts = doc.rel_path.split("/")
    return 0 if doc.source == ROOT_SOURCE else max(0, len(parts) - 2)


def _stem_matches_title(doc: DocFacts) -> bool:
    stem = {w.casefold() for w in _WORD.findall(doc.file_stem)}
    title = {w.casefold() for w in _WORD.findall(doc.title)}
    if not stem or not title:
        return False
    return len(stem & title) / min(len(stem), len(title)) >= STEM_TITLE_MIN_OVERLAP


def _ratio(part: float, whole: float) -> float:
    return round(part / whole, 4) if whole else 0.0
