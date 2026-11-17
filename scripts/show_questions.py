"""Show eval questions from the tune or smoke set only: `uv run questions-tune [--id ID]`.

The held-out set is refused (AGENTS.md: Eval split). This is the only way an
agent should read dev questions; eval code reads them through `load_questions`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SETS = ("tune", "smoke")
QUESTIONS_FILE = ROOT / "data" / "questions.jsonl"
SETS_DIR = ROOT / "eval" / "sets"


def load_set(sets_dir: Path, name: str) -> list[str]:
    path = sets_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["question_ids"]


def load_questions(
    set_name: str, questions_file: Path = QUESTIONS_FILE, sets_dir: Path = SETS_DIR
) -> list[dict[str, Any]]:
    """The questions of an allowed set, in file order. Any other set raises ValueError."""
    if set_name not in ALLOWED_SETS:
        raise ValueError(f"Refused: only {ALLOWED_SETS} may be read (AGENTS.md: Eval split).")
    allowed = set(load_set(sets_dir, set_name))
    lines = questions_file.read_text(encoding="utf-8").splitlines()
    return [q for q in (json.loads(line) for line in lines) if q["question_id"] in allowed]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="tune", help="tune (default) or smoke")
    parser.add_argument("--id", action="append", help="only these question ids")
    parser.add_argument("--brief", action="store_true", help="question text only")
    # Test-only overrides, so tests run on invented files and never open the real ones.
    parser.add_argument("--questions-file", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--sets-dir", type=Path, default=SETS_DIR)
    args = parser.parse_args()
    if args.set not in ALLOWED_SETS:
        print(
            f"Refused: only {ALLOWED_SETS} may be shown (AGENTS.md: Eval split).", file=sys.stderr
        )
        return 2
    allowed = set(load_set(args.sets_dir, args.set))
    wanted = set(args.id or allowed)
    if wanted - allowed:
        outside = ", ".join(sorted(wanted - allowed))
        print(
            f"Refused: {outside} not in the {args.set} set (AGENTS.md: Eval split).",
            file=sys.stderr,
        )
        return 2
    for question in load_questions(args.set, args.questions_file, args.sets_dir):
        if question["question_id"] not in wanted:
            continue
        print(f"## {question['question_id']}\n{question['question']}")
        if not args.brief:
            print(f"\ngold: {question['gold_answer']}")
            print("facts:\n" + "\n".join(f"- {fact}" for fact in question["answer_facts"]))
            print(f"expected: {', '.join(question['expected_doc_ids']) or '(none: unanswerable)'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
