"""S8 profile, keys and links as pure functions, on crafted rows (invented names only)."""

import re
from collections.abc import Sequence

import pytest

from corpusqa.config import (
    CONDITIONAL_MIN_SUPPORT,
    HUB_KEY_MAX_DOCS,
    LEXICON_MIN_COUNT,
)
from corpusqa.ingest.dates import parse_date
from corpusqa.ingest.fields import fold_key, normalise
from corpusqa.ingest.links import Resolver, build_keys, build_links
from corpusqa.ingest.profile import (
    BOOLEAN,
    DATE,
    DATED_LOG,
    EMAIL,
    ENUM,
    MIXED,
    NUMBER,
    OWN_ID,
    PERSON,
    REFERENCE,
    TAGS,
    TEXT,
    UNKNOWN,
    URL,
    FieldStats,
    Kind,
    classify,
    conditions,
    gather,
    learn_lexicon,
    reused_values,
    shared_titles,
    synonym_groups,
)
from corpusqa.store.interfaces import DocFacts, FieldValue
from corpusqa.store.norms import person_norm

_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
GIVEN = ("Odo", "Brix", "Tamsa", "Wenna", "Corvin", "Ilse")
FAMILY = ("Quarrow", "Pellam", "Strand", "Hollis", "Vane", "Ostrik")


def doc(i: int, source: str = "src", **kw: object) -> DocFacts:
    values: dict[str, object] = {
        "doc_id": f"d{i}", "source": source, "rel_path": f"{source}/f{i}.json",
        "file_stem": f"f{i}", "title": f"Title number {i}", "title_field": "label",
        "title_is_weak": False, "display_title": f"Title number {i}", "body_fields": (),
        "flags": (),
    }  # fmt: skip
    values.update(kw)
    return DocFacts(**values)  # type: ignore[arg-type]


def value(doc_id: str, field: str, text: object, seq: int = 0, **kw: object) -> FieldValue:
    """A field row as S4 would store it (value typed, normalised, dated from the text)."""
    if text is None:
        typed, raw = "null", None
    elif isinstance(text, bool):
        typed, raw = "bool", str(text).lower()
    else:
        raw = str(text)
        typed = "str" if raw.strip() else "empty"
    parsed = parse_date(raw) if raw and typed == "str" else None
    values: dict[str, object] = {
        "source": "src", "field": field, "field_norm": fold_key(field), "doc_id": doc_id,
        "seq": seq, "role": "meta", "value": raw, "value_type": typed,
        "value_norm": normalise(raw) if raw is not None and typed != "empty" else None,
        "value_num": float(raw) if raw and _NUMBER.match(raw) else None,
        "value_iso": parsed.iso if parsed else None, "in_list": False, "is_noise": False,
    }  # fmt: skip
    values.update(kw)
    return FieldValue(**values)  # type: ignore[arg-type]


def stats_for(
    columns: dict[str, Sequence[object]], n_docs: int | None = None, lists: Sequence[str] = ()
) -> tuple[dict[tuple[str, str], FieldStats], list[DocFacts]]:
    """Statistics of one source: each column is the field's value per document (None =
    absent; a list = a list value). Fields named in `lists` hold one-item lists."""
    width = n_docs or max(len(v) for v in columns.values())
    docs = [doc(i) for i in range(width)]
    rows = []
    for name, per_doc in columns.items():
        for i, item in enumerate(per_doc):
            if item is None:
                continue
            items = item if isinstance(item, list) else [item]
            in_list = isinstance(item, list) or name in lists
            for seq, text in enumerate(items):
                rows.append(value(f"d{i}", name, text, seq, in_list=in_list))
    rows.sort(key=lambda r: (r.source, r.field, r.doc_id, r.seq))
    return gather(rows, docs), docs


def kind_of(
    column: Sequence[object],
    n_docs: int | None = None,
    lexicon: frozenset[str] = frozenset(),
    reused: frozenset[str] = frozenset(),
    lists: bool = False,
) -> Kind:
    stats, docs = stats_for({"x": column}, n_docs, lists=["x"] if lists else [])
    return classify(stats[("src", "x")], len(docs), lexicon, reused)


def names(n: int) -> list[str]:
    return [f"{GIVEN[i % len(GIVEN)]} {FAMILY[(i // len(GIVEN)) % len(FAMILY)]}" for i in range(n)]


# ---------------------------------------------------------------- kinds
def test_dates_numbers_and_booleans() -> None:
    assert kind_of([f"2025-03-{d:02d}" for d in range(1, 29)]).kind == DATE
    assert kind_of([str(n * 7 % 5) for n in range(40)]).kind == NUMBER
    assert kind_of(["yes", "no", "Yes", "true"] * 5) == Kind(BOOLEAN, 1.0)
    # One odd value in twenty still leaves 95 % dates.
    assert kind_of([*(f"2025-01-{d:02d}" for d in range(1, 20)), "soon"]) == Kind(DATE, 0.95)


def test_own_id_needs_presence_uniqueness_digits_and_short_values() -> None:
    ids = [f"QZ-{100 + i}" for i in range(60)]
    assert kind_of(ids) == Kind(OWN_ID, 1.0)
    # Outliers are tolerated: two docs without it, one duplicate, one value without digits.
    tolerant = [*ids[:56], None, None, ids[0], "pending"]
    assert kind_of(tolerant).kind == OWN_ID
    assert kind_of([*ids[:50], *([None] * 10)]).kind != OWN_ID  # present in 83 % only
    assert kind_of([f"note {i} " + "x" * 50 for i in range(60)]).kind != OWN_ID  # long
    # Numeric ids are own ids, not numbers.
    assert kind_of([str(5000 + i) for i in range(60)]).kind == OWN_ID
    # Unique short titles that each hold a number are not ids (they have spaces).
    assert kind_of([f"Order {i} for the kiln" for i in range(60)]).kind != OWN_ID


def test_references_urls_and_emails() -> None:
    refs = ["QZ-12", "Tracker: QZ-40", "https://wiki.example/space/page-9", "team/notes/plan"]
    assert kind_of(refs * 5).kind == REFERENCE
    assert kind_of(["https://quarrow.example", "https://pellam.example/"] * 10).kind == URL
    # A slash date is neither a path nor a reference.
    assert kind_of([f"0{m}/1{m}/2024" for m in range(1, 10)] * 3).kind not in (REFERENCE, URL)
    mails = ["odo@quarrow.example", "Brix Pellam <brix@pellam.example>"]
    assert kind_of(mails * 10).kind == EMAIL


def test_people_from_the_lexicon_or_from_shape_and_reuse() -> None:
    people = names(30)
    lexicon = frozenset(g.casefold() for g in GIVEN)
    assert kind_of(people, lexicon=lexicon).kind == PERSON
    # Without the lexicon, the same scalar field with few names is an enum...
    assert kind_of(people[:6] * 5).kind == ENUM
    # ...and a list of full names reused in another field is still people.
    reused = frozenset(normalise(p) for p in people)
    assert kind_of([[p] for p in people], reused=reused).kind == PERSON
    # One capitalised word is not a name, even when reused.
    words = ["Lantern", "Kettle", "Harbour"] * 10
    assert kind_of([[w] for w in words], reused=frozenset(normalise(w) for w in words)).kind == TAGS


def test_enum_tags_dated_logs_text_and_the_rest() -> None:
    assert kind_of(["open", "closed", "waiting"] * 10).kind == ENUM
    assert kind_of([["alpha", "beta"], ["beta"], ["gamma", "alpha"]] * 5).kind == TAGS
    log = [["2025-01-02: filed", "2025-01-05 fixed"], ["[2025-02-01] reopened"]] * 5
    assert kind_of(log).kind == DATED_LOG
    prose = [f"The loom on line {i} stopped twice during the night shift again." for i in range(9)]
    assert kind_of(prose).kind == TEXT
    assert kind_of(["a b", "7 x", "?"] * 2).kind == MIXED
    # Only null and empty values: nothing to type.
    stats = gather([value("d0", "x", None), value("d1", "x", " ")], [doc(0), doc(1)])
    assert classify(stats[("src", "x")], 2, frozenset(), frozenset()) == Kind(UNKNOWN, 0.0)


def test_a_field_whose_text_went_into_the_body_is_text() -> None:
    docs = [doc(i, body_fields=("story",)) for i in range(4)]
    stats = gather([], docs)
    assert classify(stats[("src", "story")], 4, frozenset(), frozenset()) == Kind(TEXT, 1.0)


def test_noise_rows_are_counted_but_never_typed() -> None:
    rows = [value(f"d{i}", "x", f"f{i}.json", is_noise=True) for i in range(5)]
    stats = gather(rows, [doc(i) for i in range(5)])
    st = stats[("src", "x")]
    assert (st.rows, st.noise, st.valued) == (5, 5, 0)
    assert classify(st, 5, frozenset(), frozenset()).kind == UNKNOWN


# ---------------------------------------------------------------- lexicon
def test_the_lexicon_keeps_first_tokens_that_start_many_distinct_names() -> None:
    people = names(36)  # every given name with six family names
    firms = [f"Kestrel {w}" for w in ("Mills", "Mills", "Mills")] + ["Ardent Works"] * 3
    stats, _ = stats_for({"owner": people, "firm": firms * 6})
    lexicon = learn_lexicon(stats.values())
    assert lexicon == frozenset(g.casefold() for g in GIVEN)
    assert "kestrel" not in lexicon  # one distinct name, however often it repeats
    assert LEXICON_MIN_COUNT == 5


def test_person_values_get_their_own_matching_form() -> None:
    assert person_norm("Odo Quarrow (Night lead)") == "odo quarrow"
    assert person_norm("Brix Pellam <brix@pellam.example>") == "brix pellam"
    assert person_norm("odo.quarrow") == "odo quarrow"
    assert person_norm("odo_quarrow@quarrow.example") == "odo quarrow@quarrow.example"


# ---------------------------------------------------------------- synonyms, conditions
def test_synonyms_need_the_same_folded_name_or_never_sharing_a_document() -> None:
    half = 20
    columns: dict[str, Sequence[object]] = {
        # renamed: never in the same document, same kind, same vocabulary
        "phase": [["open", "done", "waiting"][i % 3] if i < half else None for i in range(40)],
        "stage": [None if i < half else ["open", "done", "waiting"][i % 3] for i in range(40)],
        # same vocabulary but in the same documents: not a rename
        "mood": [["open", "done", "waiting"][i % 3] for i in range(40)],
        # names that fold to the same key
        "Owner Team": ["red"] * 40,
        "owner-team": ["blue"] * 40,
    }
    stats, docs = stats_for(columns)
    kinds = {k: classify(st, len(docs), frozenset(), frozenset()) for k, st in stats.items()}
    groups = synonym_groups(stats, kinds)
    assert groups[("src", "phase")] == groups[("src", "stage")]
    assert ("src", "mood") not in groups
    assert groups[("src", "Owner Team")] == groups[("src", "owner-team")]
    assert len(set(groups.values())) == 2


def test_conditional_presence_measures_how_much_a_field_depends_on_an_enum() -> None:
    n = CONDITIONAL_MIN_SUPPORT
    kinds_of_doc = ["incident"] * n + ["task"] * n + ["rare"] * 3
    columns: dict[str, Sequence[object]] = {
        "doc_kind": kinds_of_doc,
        # present for incidents only (one incident lacks it)
        "impact": ["high" if k == "incident" and i else None for i, k in enumerate(kinds_of_doc)],
        "colour": ["red"] * len(kinds_of_doc),  # everywhere: no condition
    }
    stats, docs = stats_for(columns)
    kinds = {k: classify(st, len(docs), frozenset(), frozenset()) for k, st in stats.items()}
    assert kinds[("src", "doc_kind")].kind == ENUM
    [row] = conditions(stats, kinds)
    assert (row.field, row.given_field, row.present_when) == ("impact", "doc_kind", ("incident",))
    assert row.strength == pytest.approx((n - 1) / n, abs=1e-4)


# ---------------------------------------------------------------- weak titles (rule a)
def test_titles_shared_across_a_source_become_weak() -> None:
    shared = [doc(i, title="general", file_stem=f"thread-{i}") for i in range(6)]
    unique = [doc(10 + i, source="other") for i in range(6)]
    already = doc(20, title="general", file_stem="x-20", title_is_weak=True, flags=("weak_title",))
    updates = shared_titles([*shared, *unique, already])
    assert [u.doc_id for u in updates] == [d.doc_id for d in shared]
    assert updates[0].display_title == "general — thread 0"
    assert updates[0].flags == ("weak_title",)


# ---------------------------------------------------------------- keys
def own(doc_id: str, text: str, field: str = "ticket") -> FieldValue:
    return value(doc_id, field, text)


def test_keys_own_ids_stems_stem_keys_and_folder_numbers() -> None:
    docs = [
        doc(1, rel_path="src/QZ-7-broken-gate.json", file_stem="QZ-7-broken-gate"),
        doc(2, rel_path="src/repo-a/42.json", file_stem="42"),
        doc(3, rel_path="src/repo-b/note.json", file_stem="note"),
    ]
    keys = build_keys(docs, [own("d1", "QZ-7"), own("d2", "0042"), own("d3", "AB-3")])
    got = {(r.doc_id, r.key_norm, r.kind) for r in keys.rows}
    assert got == {
        ("d1", "qz-7", "field:ticket"), ("d1", "qz-7-broken-gate", "stem"),
        ("d1", "qz-7", "stem_key"), ("d2", "repo-a/42", "folder_number"),
        ("d3", "ab-3", "field:ticket"), ("d3", "note", "stem"),
    }  # fmt: skip
    assert not any(r.key_norm.isdigit() for r in keys.rows)  # never a bare number
    assert keys.prefixes == {"src": ("AB", "QZ")}


def test_a_key_held_by_too_many_documents_is_a_hub() -> None:
    n = HUB_KEY_MAX_DOCS + 1
    docs = [doc(i) for i in range(n + 1)]
    rows = [own(f"d{i}", "QZ-1") for i in range(n)] + [own(f"d{n}", "QZ-2")]
    keys = build_keys(docs, rows)
    hubs = {(r.key_norm, r.is_hub) for r in keys.rows if r.kind == "field:ticket"}
    assert hubs == {("qz-1", True), ("qz-2", False)}


# ---------------------------------------------------------------- links
@pytest.fixture
def resolver() -> Resolver:
    docs = [
        doc(1, rel_path="tracker/QZ-7-gate.json", file_stem="QZ-7-gate", title="Gate is stuck"),
        doc(2, rel_path="wiki/space/rota-plan.json", file_stem="rota-plan", title="Rota plan"),
        doc(3, rel_path="code/repo-a/42.json", file_stem="42", title="Fix the rota export"),
        doc(4, rel_path="tracker/QZ-8.json", file_stem="QZ-8", title="Shared"),
        doc(5, rel_path="tracker/QZ-9.json", file_stem="QZ-9", title="Shared"),
        doc(
            6, rel_path="chat/general/t1.json", file_stem="t1", title="general", title_is_weak=True
        ),
        *(doc(10 + i, rel_path=f"misc/{i}/readme.json", file_stem="readme") for i in range(6)),
    ]
    rows = [own("d1", "QZ-7"), own("d3", "42", field="number")]
    return Resolver(build_keys(docs, rows).rows, docs)


def resolve(resolver: Resolver, raw: str, doc_id: str = "d99") -> list[tuple[str | None, str]]:
    return [(r.target_doc_id, r.method) for r in resolver.resolve(doc_id, "refs", raw)]


def test_links_resolve_by_key_stem_folder_number_and_unique_title(resolver: Resolver) -> None:
    assert resolve(resolver, "QZ-7") == [("d1", "key")]
    assert resolve(resolver, "Tracker: QZ-7") == [("d1", "key")]
    assert resolve(resolver, "https://tracker.example/browse/QZ-7") == [("d1", "key")]
    assert resolve(resolver, "https://wiki.example/space/rota-plan.html?x=1") == [("d2", "stem")]
    assert resolve(resolver, "repo-a#42") == [("d3", "folder_number")]
    assert resolve(resolver, "https://code.example/org/repo-a/pull/42") == [("d3", "folder_number")]
    assert resolve(resolver, "Rota plan") == [("d2", "unique_title")]
    assert resolve(resolver, "fix the rota export") == [("d3", "unique_title")]


def test_links_never_resolve_by_number_alone(resolver: Resolver) -> None:
    for raw in ("42", "#42", "PR 42", "https://code.example/org/repo-z/pull/42"):
        assert resolve(resolver, raw) == [(None, "none")], raw


def test_shared_weak_and_self_targets_are_not_links(resolver: Resolver) -> None:
    assert resolve(resolver, "Shared") == [(None, "none")]  # title held by two documents
    assert resolve(resolver, "general") == [(None, "none")]  # weak title
    assert resolve(resolver, "QZ-7", doc_id="d1") == [(None, "self")]
    rows = resolver.resolve("d99", "refs", "https://wiki.example/readme")
    assert [(r.target_doc_id, r.method, r.target_is_hub) for r in rows] == [(None, "stem", True)]


def test_a_key_shared_by_a_few_documents_links_to_each() -> None:
    docs = [doc(1), doc(2), doc(3)]
    resolver = Resolver(build_keys(docs, [own("d1", "QZ-5"), own("d2", "QZ-5")]).rows, docs)
    links = build_links([value("d3", "refs", "QZ-5"), value("d3", "refs", "QZ-5", seq=1)], resolver)
    assert [(link.target_doc_id, link.method) for link in links] == [("d1", "key"), ("d2", "key")]


def test_reused_values_are_values_of_two_fields() -> None:
    stats, _ = stats_for({"a": ["x", "y"], "b": ["y", "z"]})
    assert reused_values(stats.values()) == frozenset({"y"})
