"""The eval split is fixed: 30 tune questions, 49 held-out, smoke inside tune (user decision).

Never opens the held-out set file or the question file (AGENTS.md: Eval split).
Held-out is, by construction, every dev id that is not in tune
(scripts/make_eval_split.py), so tune and smoke are enough to check the split.
The held-out file itself is protected from edits by the edit guard hook.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / "eval" / "sets"
DEV_IDS = {f"pub_{n:04d}" for n in range(1, 80)}  # the 79 dev question ids
QUESTION_ID = re.compile(r"pub_\d{4}")


def ids(name: str) -> set[str]:
    return set(json.loads((SETS / f"{name}.json").read_text(encoding="utf-8"))["question_ids"])


def test_split_sizes_and_coverage() -> None:
    tune, smoke = ids("tune"), ids("smoke")
    assert len(tune) == 30
    assert tune <= DEV_IDS
    assert len(DEV_IDS - tune) == 49  # the held-out set
    assert all(QUESTION_ID.fullmatch(q) for q in tune)
    assert smoke <= tune
