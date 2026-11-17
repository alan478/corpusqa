"""Shared helpers for contract tests."""

import socket

import pytest


@pytest.fixture
def free_port() -> int:
    """A TCP port that was free a moment ago (good enough for local tests)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
