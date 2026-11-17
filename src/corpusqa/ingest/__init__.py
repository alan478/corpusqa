"""ingest layer. See AGENTS.md code map."""

from corpusqa.ingest.pipeline import run
from corpusqa.ingest.startup import Runtime, StartupFn, placeholder_startup

__all__ = ["Runtime", "StartupFn", "placeholder_startup", "run"]
