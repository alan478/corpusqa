"""Document-level retrieval metrics for the chunking eval.

Per mode (bm25, dense, hybrid), over the questions that have gold documents:
- recall@k: gold documents found in the top k, over ALL gold documents of all questions
  (pooled, so a question with three gold documents weighs three);
- q_recall@k: per question, the share of its gold documents found in the top k, averaged
  over the questions (each question weighs one, however many gold documents it has);
- all_gold@k: share of questions with every gold document in the top k;
- MRR: mean of 1 / rank of the first gold document (0 when none was retrieved).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from corpusqa.retrieval.hybrid import MODES, Mode, Rankings


@dataclass(frozen=True, slots=True)
class QuestionRun:
    """One question's gold documents and what each mode retrieved."""

    question_id: str
    gold: frozenset[str]
    rankings: Rankings


@dataclass(frozen=True, slots=True)
class ModeScores:
    recall: dict[int, float]  # k -> pooled recall@k
    q_recall: dict[int, float]  # k -> per-question recall@k (mean of found / gold)
    all_gold: float  # share of questions with every gold document in the top all_gold_k
    mrr: float
    questions: int
    gold_documents: int


def gold_ranks(ranked: Sequence[str], gold: frozenset[str]) -> list[int | None]:
    """The 1-based rank of each gold document (best first; None for those not retrieved).

    Ranks only: the result never names a document."""
    position = {doc_id: rank for rank, doc_id in enumerate(ranked, start=1)}
    found = sorted(position[doc_id] for doc_id in gold if doc_id in position)
    return [*found, *([None] * (len(gold) - len(found)))]


def score(
    runs: Sequence[QuestionRun], mode: Mode, ks: Sequence[int], all_gold_k: int
) -> ModeScores:
    runs = [run for run in runs if run.gold]
    total_gold = sum(len(run.gold) for run in runs)
    recall = {k: _share(sum(_found(run, mode, k) for run in runs), total_gold) for k in ks}
    q_recall = {
        k: _share(sum(_found(run, mode, k) / len(run.gold) for run in runs), len(runs)) for k in ks
    }
    complete = sum(run.gold <= set(run.rankings.of(mode)[:all_gold_k]) for run in runs)
    reciprocal = []
    for run in runs:
        first = gold_ranks(run.rankings.of(mode), run.gold)[0]
        reciprocal.append(0.0 if first is None else 1.0 / first)
    return ModeScores(
        recall=recall,
        q_recall=q_recall,
        all_gold=_share(complete, len(runs)),
        mrr=_share(sum(reciprocal), len(runs)),
        questions=len(runs),
        gold_documents=total_gold,
    )


def score_all(
    runs: Sequence[QuestionRun], ks: Sequence[int], all_gold_k: int
) -> dict[Mode, ModeScores]:
    return {mode: score(runs, mode, ks, all_gold_k) for mode in MODES}


def _found(run: QuestionRun, mode: Mode, k: int) -> int:
    """How many of the run's gold documents `mode` put in its top k."""
    return len(run.gold & set(run.rankings.of(mode)[:k]))


def _share(part: float, whole: int) -> float:
    return part / whole if whole else 0.0
