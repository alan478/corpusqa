"""Placeholders keep the server runnable before ingest and the agent exist."""

import asyncio
from pathlib import Path

import pytest

from corpusqa.agent import PlaceholderAnswerer
from corpusqa.config import PLACEHOLDER_ANSWER, ConfigError, Settings
from corpusqa.ingest import Runtime, placeholder_startup


def test_placeholder_startup_knows_no_documents(tmp_path: Path) -> None:
    runtime = placeholder_startup(Settings(corpus_dir=tmp_path, port=8080))
    assert runtime == Runtime(known_doc_ids=frozenset())


def test_placeholder_startup_fails_if_the_corpus_vanished(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a readable directory"):
        placeholder_startup(Settings(corpus_dir=tmp_path / "gone", port=8080))


def test_placeholder_answerer_says_it_is_a_placeholder_and_cites_nothing() -> None:
    result = asyncio.run(PlaceholderAnswerer().answer("Anything?"))
    assert result.answer == PLACEHOLDER_ANSWER
    assert result.document_ids == []
