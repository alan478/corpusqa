"""Chunking strategies: slices of the body, size limits, each recursion level, headers."""

import itertools
import math
from pathlib import Path

import pytest

from corpusqa.config import CHUNK_MAX_EMBED_TOKENS, CHUNK_MAX_TOKENS, TOKEN_CHARS
from corpusqa.ingest.chunking import STRATEGIES, ChunkDraft, ChunkInput, ChunkSpec, chunk_document
from corpusqa.ingest.clean import clean_body
from corpusqa.ingest.core import build_core
from corpusqa.ingest.discover import discover
from corpusqa.ingest.models import ParsedDoc
from corpusqa.ingest.parse import parse

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
SMALL = 10  # tokens: 40 characters, so crafted text stays short


def doc(body: str, labels: tuple[str, ...] = (), created: str | None = "2024-06-01") -> ChunkInput:
    return ChunkInput(source="src", display_title="A Title", created=created, body=body,
                      labels=labels)  # fmt: skip


def fixture_docs() -> list[ChunkInput]:
    docs = []
    for file in discover(FIXTURE).files:
        parsed = parse(file)
        if not isinstance(parsed, ParsedDoc):
            continue
        core = build_core(parsed)
        body = clean_body(core)
        docs.append(ChunkInput(file.source, core.display_title, None, body.body, body.labels))
    return docs


def assert_valid(source: ChunkInput, chunks: list[ChunkDraft], limit: int) -> None:
    for chunk in chunks:
        assert source.body[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.text and chunk.text == chunk.text.strip()
        assert chunk.n_tokens == math.ceil(len(chunk.text) / TOKEN_CHARS) <= limit


def levels(chunks: list[ChunkDraft]) -> list[str]:
    return [c.level for c in chunks]


# ---------------------------------------------------------------- invariants
CRAFTED = [
    "short body",
    "## alpha\nfirst section text. " * 12 + "\n\n## beta\n" + "second. " * 40,
    "\n\n".join(f"Paragraph {i}. " + "word " * (5 + i * 7) for i in range(12)),
    "one-line " * 300,
    "x" * 5000,
    "  \n\n  leading and trailing space around a body  \n\n ",
]


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
@pytest.mark.parametrize("max_tokens", [SMALL, 50, None])
def test_every_chunk_is_a_slice_of_the_body_and_within_its_limit(
    strategy: str, max_tokens: int | None
) -> None:
    spec = ChunkSpec(strategy, max_tokens=max_tokens, overlap_tokens=0 if max_tokens else None)
    for source in [*fixture_docs(), *(doc(text, ("alpha", "beta")) for text in CRAFTED)]:
        chunks = chunk_document(source, spec)
        assert_valid(source, chunks, spec.limit_tokens)
        assert chunks, source.body[:40]
        if strategy != "fixed":  # no overlap: pieces are in order and disjoint
            ends = [(c.char_start, c.char_end) for c in chunks]
            assert all(a[1] <= b[0] for a, b in itertools.pairwise(ends))


def test_the_same_document_always_gives_the_same_chunks() -> None:
    for strategy in STRATEGIES:
        spec = ChunkSpec(strategy, max_tokens=SMALL)
        for source in fixture_docs():
            assert chunk_document(source, spec) == chunk_document(source, spec)


@pytest.mark.parametrize("body", ["", "   \n\n\t "])
def test_an_empty_body_gives_no_chunks(body: str) -> None:
    for strategy in STRATEGIES:
        assert chunk_document(doc(body), ChunkSpec(strategy)) == []


# ---------------------------------------------------------------- recursive levels
def test_a_body_that_fits_is_one_chunk() -> None:
    [chunk] = chunk_document(doc("\n  A short note.\n\nSecond line.  \n"))
    assert (chunk.text, chunk.level) == ("A short note.\n\nSecond line.", "whole")


def test_sections_split_at_the_documents_own_label_lines_only() -> None:
    body = "## alpha\nfirst\n## not a label\nmore\n## beta\nsecond part here"  # 35 + 1 + 24
    chunks = chunk_document(doc(body, ("alpha", "beta")), ChunkSpec(max_tokens=SMALL))
    assert levels(chunks) == ["section", "section"]
    assert [c.text for c in chunks] == [
        "## alpha\nfirst\n## not a label\nmore",
        "## beta\nsecond part here",
    ]


def test_paragraphs_then_merge_small_neighbours_up_to_the_limit() -> None:
    body = "\n\n".join(["aaaaaaaaa."] * 5)  # 5 x 10 chars, 2-char breaks: 34 fits, 46 not
    chunks = chunk_document(doc(body), ChunkSpec(max_tokens=SMALL))
    assert levels(chunks) == ["paragraph", "paragraph"]
    assert [c.text.count(".") for c in chunks] == [3, 2]


@pytest.mark.parametrize(
    ("body", "level"),
    [
        ("line one is long-ish\nline two is long-ish\nline three", "line"),
        ("Sentence one is here. Sentence two is here! And a third?", "sentence"),
        ("words without any sentence end go on and on here", "word"),
        ("x" * 100, "char"),
    ],
)
def test_each_deeper_level_is_used_when_the_one_above_cannot_cut(body: str, level: str) -> None:
    chunks = chunk_document(doc(body), ChunkSpec(max_tokens=SMALL))
    assert len(chunks) > 1 and set(levels(chunks)) == {level}
    assert_valid(doc(body), chunks, SMALL)


def test_only_pieces_over_the_limit_go_deeper() -> None:
    small = "Short paragraph."
    big = "First long sentence here. Second long sentence here."
    chunks = chunk_document(doc(f"{small}\n\n{big}"), ChunkSpec(max_tokens=SMALL))
    assert levels(chunks) == ["paragraph", "sentence", "sentence"]
    assert chunks[0].text == small


# ---------------------------------------------------------------- fixed and whole
def test_fixed_windows_overlap_and_cut_at_whitespace() -> None:
    body = " ".join(f"w{i:03d}" for i in range(100))  # 100 words of 4 chars
    source = doc(body)
    chunks = chunk_document(source, ChunkSpec("fixed", max_tokens=SMALL, overlap_tokens=3))
    assert_valid(source, chunks, SMALL)
    assert set(levels(chunks)) == {"fixed"} and chunks[-1].char_end == len(body)
    for before, after in itertools.pairwise(chunks):
        assert before.char_start < after.char_start < before.char_end  # they overlap
    for chunk in chunks:  # whole words only
        assert chunk.char_start == 0 or body[chunk.char_start - 1] == " "
        assert chunk.char_end == len(body) or body[chunk.char_end] == " "


def test_fixed_without_overlap_tiles_the_body() -> None:
    body = " ".join(["word"] * 50)
    chunks = chunk_document(doc(body), ChunkSpec("fixed", max_tokens=SMALL, overlap_tokens=0))
    assert " ".join(c.text for c in chunks) == body


def test_whole_is_one_chunk_up_to_the_embedding_limit_then_falls_back_to_recursive() -> None:
    long_body = "\n\n".join(["A paragraph of some length here."] * 400)  # ~3,400 tokens
    [one] = chunk_document(doc(long_body), ChunkSpec("whole"))
    assert one.level == "whole" and one.text == long_body
    assert ChunkSpec("whole").limit_tokens == CHUNK_MAX_EMBED_TOKENS
    pieces = chunk_document(doc(long_body), ChunkSpec("whole", max_tokens=1000))
    assert len(pieces) > 1 and set(levels(pieces)) == {"paragraph"}
    assert_valid(doc(long_body), pieces, 1000)


# ---------------------------------------------------------------- header and spec
def test_the_header_is_kept_apart_from_the_text_and_can_be_switched_off() -> None:
    [on] = chunk_document(doc("Body text.", created="2024-06-01T10:00:00Z"))
    assert on.header == "src · A Title · 2024-06-01" and on.text == "Body text."
    [undated] = chunk_document(doc("Body text.", created=None))
    assert undated.header == "src · A Title"
    [off] = chunk_document(doc("Body text."), ChunkSpec(header=False))
    assert off.header is None and off.text == "Body text."


def test_defaults_and_bad_specs() -> None:
    assert ChunkSpec().limit_tokens == CHUNK_MAX_TOKENS == 512
    assert (ChunkSpec().overlap, ChunkSpec("fixed").overlap) == (0, 51)  # 10 % of 512
    assert ChunkSpec("fixed", max_tokens=SMALL).overlap == 1
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        ChunkSpec("semantic")
    with pytest.raises(ValueError, match="max_tokens"):
        ChunkSpec(max_tokens=CHUNK_MAX_EMBED_TOKENS + 1)
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkSpec("fixed", max_tokens=SMALL, overlap_tokens=SMALL)
