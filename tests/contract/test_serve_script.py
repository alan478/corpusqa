"""./serve end to end, run the way CONTRACT.md §1 says the grader runs it.

Minimal environment (PATH, HOME, TMPDIR, LANG, CORPUS_DIR, PORT), stdin closed, its own
process group, SIGTERM to the whole group, and SIGKILL would follow after 10 s.
Skip with `pytest -m "not serve"`; skipped automatically when uv is not on PATH.
"""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from corpusqa.config import INTERNAL_ERROR_ANSWER, TRACE_ID_HEADER

ROOT = Path(__file__).resolve().parents[2]
READY_TIMEOUT_S = 60.0  # includes `uv sync` on a cold cache
EXIT_TIMEOUT_S = 10.0  # CONTRACT: SIGKILL follows SIGTERM after 10 s
POLL_S = 0.2

pytestmark = [
    pytest.mark.serve,
    pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH"),
]


def grader_env(corpus: Path, port: int) -> dict[str, str]:
    """Only the variables the grader sets; PATH without this repo's .venv (uv run adds it)."""
    venv = str(ROOT / ".venv")
    path = os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep) if venv not in p)
    return {
        "PATH": path,
        "HOME": os.environ["HOME"],
        "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "CORPUS_DIR": str(corpus),
        "PORT": str(port),
    }


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def test_serve_starts_answers_and_exits_on_sigterm(tmp_path: Path, free_port: int) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "notes").mkdir(parents=True)
    doc = {"dataset_doc_uuid": "dsid_" + "0" * 32, "title": "Invented note", "body": "Hello."}
    (corpus / "notes" / "one.json").write_text(json.dumps(doc), encoding="utf-8")
    port = free_port
    base = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "serve.log"

    def log_tail() -> str:
        return log_path.read_text(encoding="utf-8", errors="replace")[-4000:]

    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            ["./serve"],
            cwd=ROOT,
            env=grader_env(corpus, port),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            process_group=0,  # like the grader: we signal the whole group
        )
    try:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            assert proc.poll() is None, f"./serve exited early:\n{log_tail()}"
            assert time.monotonic() < deadline, f"/health never returned 200:\n{log_tail()}"
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass  # port not bound yet
            time.sleep(POLL_S)

        response = httpx.post(f"{base}/ask", json={"question": "What is here?"}, timeout=10)
        assert response.status_code == 200, response.text
        body = response.json()
        # No GEMINI_API_KEY here: the agent has no chat model, so every /ask gives the
        # internal-error answer (never a "not found"), with no traceback in the log.
        # CONTRACT §3: exactly these two fields; the trace id is in a header.
        assert body == {"answer": INTERNAL_ERROR_ANSWER, "document_ids": []}
        assert response.headers[TRACE_ID_HEADER]

        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=EXIT_TIMEOUT_S)
        deadline = time.monotonic() + 1
        while group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not group_alive(proc.pid), "a process in ./serve's group outlived SIGTERM"
    finally:
        if proc.poll() is None or group_alive(proc.pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    # The checks read the whole log; only the failure message shows its tail. (They read the
    # tail before, so a longer start-up log pushed a start-up line out of their view.)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    assert "Application shutdown complete" in log_text, log_tail()
    assert "Traceback" not in log_text, log_tail()
    assert "ingest: 1 documents ready" in log_text, log_tail()  # the real start-up job ran
    assert "the agent is off: GEMINI_API_KEY is not set" in log_text, log_tail()
    assert "cannot be answered: the agent has no chat model" in log_text, log_tail()
