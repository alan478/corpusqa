"""Split the dev questions once into tune (30) and held-out (49) sets.

User decision (2026-10-14): tune and debug the agent on 30 questions only; keep
the other 49 to test it. The split is stratified by kind (single-doc,
multi-doc, unanswerable) and seeded, so it is reproducible.

Questions that were read in detail while planning (the 10-question smoke set
and three worked examples) are forced into tune, so the held-out set holds
only questions nobody has studied.

Writes eval/sets/{smoke,tune,heldout}.json (question ids only) and refuses to
overwrite them: the split is fixed once made.
"""

import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "questions.jsonl"
OUT = ROOT / "eval" / "sets"
SEED = 20261014
TUNE_SIZE = 30
SMOKE = [
    "pub_0002", "pub_0010", "pub_0023", "pub_0028", "pub_0040",
    "pub_0041", "pub_0046", "pub_0052", "pub_0059", "pub_0075",
]  # fmt: skip
SEEN_WHILE_PLANNING = ["pub_0009", "pub_0017", "pub_0026"]  # worked examples in the plan docs


def kind(question: dict) -> str:
    n = len(question["expected_doc_ids"])
    return "unanswerable" if n == 0 else "single" if n == 1 else "multi"


def split(questions: list[dict]) -> tuple[list[str], list[str]]:
    by_id = {q["question_id"]: q for q in questions}
    kinds = {qid: kind(q) for qid, q in by_id.items()}
    totals = Counter(kinds.values())
    # Tune quota per kind, proportional to the full set (largest remainder).
    raw = {k: TUNE_SIZE * n / len(by_id) for k, n in totals.items()}
    quota = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - quota[k], reverse=True)[
        : TUNE_SIZE - sum(quota.values())
    ]:
        quota[k] += 1

    forced = SMOKE + SEEN_WHILE_PLANNING
    tune = list(forced)
    rng = random.Random(SEED)  # noqa: S311 - reproducible sampling, not security
    for k in sorted(quota):
        have = sum(1 for qid in tune if kinds[qid] == k)
        pool = sorted(qid for qid in by_id if kinds[qid] == k and qid not in tune)
        need = quota[k] - have
        if need < 0:
            raise SystemExit(f"forced questions exceed the {k} quota ({have} > {quota[k]})")
        tune += rng.sample(pool, need)
    heldout = sorted(qid for qid in by_id if qid not in tune)
    return sorted(tune), heldout


def write(name: str, ids: list[str], questions: list[dict], note: str) -> None:
    kinds = Counter(kind(q) for q in questions if q["question_id"] in ids)
    payload = {
        "name": name,
        "created": date.today().isoformat(),
        "seed": SEED,
        "source": "data/questions.jsonl",
        "note": note,
        "counts": {"total": len(ids), **dict(sorted(kinds.items()))},
        "question_ids": ids,
    }
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    existing = [p for p in (OUT / f"{n}.json" for n in ("smoke", "tune", "heldout")) if p.exists()]
    if existing:
        print(
            f"Refusing to overwrite the fixed split: {[str(p) for p in existing]}", file=sys.stderr
        )
        return 1
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    tune, heldout = split(questions)
    if not set(SMOKE) <= set(tune) or set(tune) & set(heldout):
        raise SystemExit("split invariant broken: smoke must be in tune; tune and heldout disjoint")
    OUT.mkdir(parents=True, exist_ok=True)
    write("smoke", sorted(SMOKE), questions, "Fast loop. Subset of tune.")
    write("tune", tune, questions, "Tune and debug on these only. Includes smoke and the "
          "questions read while planning.")  # fmt: skip
    write("heldout", heldout, questions, "Test set. Run only at milestones the user names; "
          "never tune on it or open its questions or traces while developing.")  # fmt: skip
    print(f"tune={len(tune)} heldout={len(heldout)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
