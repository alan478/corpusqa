"""HTTP layer (FastAPI). Run with `python -m corpusqa.server`; see AGENTS.md code map."""

from corpusqa.server.app import create_app

__all__ = ["create_app"]
