"""Write one-way hashes of tokens that must never appear in shipped code.

Reads the dev-question file once (coordinator only) and writes
tests/fixtures/forbidden_tokens.sha256: sorted SHA-256 hashes of every issue
key, doc id and multi-word name found in the questions, gold answers, facts and
expected ids. Tests compare against the hashes, so no test ever reads the
question file or prints a token from it (AGENTS.md: Eval split).

Run again only if the dev-question file changes.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from forbidden_tokens import candidate_tokens

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "questions.jsonl"
OUT = ROOT / "tests" / "fixtures" / "forbidden_tokens.sha256"


def main() -> int:
    tokens: set[str] = set()
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        text = " ".join([q["question"], q["gold_answer"], *q["answer_facts"]])
        tokens |= candidate_tokens(text) | set(q["expected_doc_ids"])
    hashes = sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in tokens)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# SHA-256 of tokens from the dev questions that must never appear in src/.\n"
        "# Made by scripts/make_forbidden_hashes.py; checked by tests/unit/test_no_hardcoding.py.\n"
    )
    OUT.write_text(header + "\n".join(hashes) + "\n", encoding="utf-8")
    print(f"wrote {len(hashes)} hashes to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
