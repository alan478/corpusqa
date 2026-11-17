"""Guard: shipped code must not contain keys, ids or names taken from the dev questions.

The grader runs on an unseen corpus (AGENTS.md, hard rules). Anything copied from
the dev questions into src/ is overfitting. The test never reads the question file
(AGENTS.md: Eval split): it compares one-way hashes of tokens found in src/ with
tests/fixtures/forbidden_tokens.sha256 (made by scripts/make_forbidden_hashes.py).
If a flagged phrase is truly generic, add it to ALLOW with a reason instead of
weakening the patterns.
"""

import hashlib
from pathlib import Path

from tests.forbidden_tokens import candidate_tokens

ROOT = Path(__file__).resolve().parents[2]
HASHES = ROOT / "tests" / "fixtures" / "forbidden_tokens.sha256"
SCANNED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml"}
ALLOW: dict[str, str] = {}  # phrase -> reason it is generic


def sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def forbidden_hashes() -> set[str]:
    lines = HASHES.read_text(encoding="utf-8").splitlines()
    hashes = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    return hashes - {sha(phrase) for phrase in ALLOW}


def shipped_files() -> list[Path]:
    return [p for p in (ROOT / "src").rglob("*") if p.is_file() and p.suffix in SCANNED_SUFFIXES]


def test_hash_file_is_present_and_plausible() -> None:
    hashes = forbidden_hashes()
    assert len(hashes) > 100
    assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes)


def test_candidate_tokens_finds_keys_ids_and_names() -> None:
    text = "See ZQX-4821 and dsid_" + "ab" * 16 + " for Quillon Varrow Industries."
    tokens = candidate_tokens(text)
    assert "ZQX-4821" in tokens
    assert "dsid_" + "ab" * 16 in tokens
    assert "Quillon Varrow Industries" in tokens


def test_src_has_no_dev_question_content() -> None:
    forbidden = forbidden_hashes()
    hits = []
    for path in shipped_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in sorted(candidate_tokens(text)):
            if sha(token) in forbidden:
                hits.append(f"{path.relative_to(ROOT)}: {token!r}")
    assert not hits, "dev-question content in src/:\n" + "\n".join(hits)
