"""The start-up job contract: what the server waits for before /health says 200.

The server runs one `StartupFn` in the background at boot. Whatever it returns is the
read-only `Runtime` every request uses afterwards. The real ingest pipeline replaces
`placeholder_startup` by being passed in its place; the server does not change.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from corpusqa.config import ConfigError, Settings
from corpusqa.llm.interfaces import Embedder
from corpusqa.store.interfaces import ChunkReader, CorpusReader, DocumentReader
from corpusqa.store.vectors import VectorIndex


class IndexReader(DocumentReader, CorpusReader, Protocol):
    """Both reads of a published index: the build's (`DocumentReader`) and the agent tools'
    (`CorpusReader`). `store.sqlite.SqliteReader` is both."""


@dataclass(frozen=True, slots=True)
class Runtime:
    """Ready state published by start-up. Frozen: requests share it but never change it."""

    # Every dataset_doc_uuid ingested from CORPUS_DIR. /ask cites only these ids.
    known_doc_ids: frozenset[str]
    # Read-only access to the published index (a new connection per call); the agent's
    # tools read it as a `CorpusReader`. None only for the placeholder and server tests
    # that never read the corpus.
    store: IndexReader | None = None
    index_path: Path | None = None
    built_at: str | None = None  # when the index was built (ISO 8601, UTC)
    cache_hit: bool = False  # True when start-up reused an index built earlier
    # Keyword search (BM25 over chunk header and text) and chunk -> document, over every
    # chunk of the index. None only where `store` is None.
    chunks: ChunkReader | None = None
    # The chunk vectors in memory, for dense search; None when there is no embedder, no
    # vector, or the stored vectors come from another model than `embedder`.
    vectors: VectorIndex | None = None
    # Share of all chunks whose vector is in `vectors` (0.0 when it is None). Chunks
    # without one are still found by keyword search.
    embed_coverage: float = 0.0
    # The embedder start-up was given (queries must be embedded the same way); None means
    # keyword-only search.
    embedder: Embedder | None = None


# Runs in a worker thread (it may block for hours); raising means start-up failed.
type StartupFn = Callable[[Settings], Runtime]


def placeholder_startup(settings: Settings) -> Runtime:
    """Stand-in until ingest exists: re-check the corpus is readable, know no documents."""
    corpus = settings.corpus_dir
    if not corpus.is_dir() or not os.access(corpus, os.R_OK | os.X_OK):
        raise ConfigError(f"CORPUS_DIR {corpus} is not a readable directory.")
    return Runtime(known_doc_ids=frozenset())
