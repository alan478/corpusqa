"""The question loader of the end-to-end eval (`eval.run`) only.

Unlike `scripts.show_questions.load_questions` (tune and smoke only, on purpose: it is how
agents read questions), this loader also reads the held-out set and `all` (every dev
question), because `uv run eval-heldout` and `uv run eval-all` must send them to the server.
Nothing it returns is ever printed: the runner shows aggregates only for those sets
(AGENTS.md: Eval split). No other module may use it.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.make_eval_split import kind as split_kind
from scripts.show_questions import QUESTIONS_FILE, SETS_DIR, load_set

ALL_SET = "all"  # every question in the question file
RUN_SETS = ("smoke", "tune", "heldout", ALL_SET)
# Sets whose per-question results agents must never read (AGENTS.md: Eval split).
PROTECTED_SETS = frozenset({"heldout", ALL_SET})

Kind = Literal["single", "multi", "unanswerable"]
KINDS: tuple[Kind, ...] = ("single", "multi", "unanswerable")


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    question_id: str
    text: str
    expected: frozenset[str]  # the expected dataset_doc_uuids (empty: unanswerable)
    kind: Kind  # the split's rule: 0 expected ids unanswerable, 1 single, 2+ multi
    # For the answer judge (eval/judge.py) only; never printed or written to a report.
    gold_answer: str = ""
    facts: tuple[str, ...] = ()  # the question's answer_facts, in order


def load_run_questions(
    set_name: str, questions_file: Path = QUESTIONS_FILE, sets_dir: Path = SETS_DIR
) -> list[EvalQuestion]:
    """The questions of `set_name` (one of RUN_SETS), in file order.

    Raises ValueError for an unknown set, or when the set names a question the file lacks.
    """
    if set_name not in RUN_SETS:
        raise ValueError(f"unknown set {set_name!r}; one of {RUN_SETS}")
    lines = questions_file.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if set_name != ALL_SET:
        wanted = set(load_set(sets_dir, set_name))
        missing = wanted - {row["question_id"] for row in rows}
        if missing:
            raise ValueError(f"set {set_name!r} names {len(missing)} ids the question file lacks")
        rows = [row for row in rows if row["question_id"] in wanted]
    return [
        EvalQuestion(
            question_id=row["question_id"],
            text=row["question"],
            expected=frozenset(row["expected_doc_ids"]),
            kind=split_kind(row),
            gold_answer=row.get("gold_answer", ""),
            facts=tuple(row.get("answer_facts", ())),
        )
        for row in rows
    ]
