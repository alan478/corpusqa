"""Set-based citation metrics of the end-to-end eval (`eval.run`).

The `document_ids` of a response have no real order (user decision 2026-10-16: no MRR, no
nDCG), so every metric compares sets. Per question:
- answerable (1+ expected ids): precision = hits / ids returned (0 when none came back),
  recall = hits / expected ids, F1 = 2PR / (P + R) (0 when both are 0), hit = any expected
  id returned, all_gold = every expected id returned, exact = returned set == expected set;
- unanswerable (no expected ids): abstained = no ids returned, which is the correct reply.
A failed request (no response, timeout, non-200, a body off the contract) is wrong: it
returns nothing, so it scores 0 and does not count as abstaining.

Aggregates: the mean of each metric per group (single, multi, and both together as
"answerable"); the abstain rate of the unanswerable questions; latency percentiles (nearest
rank) and the maximum over every request; error counts.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from eval.questions import EvalQuestion, Kind

METRICS = ("precision", "recall", "f1", "hit", "all_gold", "exact")
ANSWERABLE = "answerable"  # single + multi
GROUPS = ("single", "multi", ANSWERABLE)


@dataclass(frozen=True, slots=True)
class Reply:
    """What one POST /ask gave back."""

    status: int | None  # HTTP status; None when no response came back
    latency_s: float
    answer: str | None
    document_ids: tuple[str, ...]  # deduplicated, first occurrence kept
    trace_id: str | None  # the server's trace of this request: its X-Trace-Id header
    error: str | None = None  # why the reply failed; None for a good reply


@dataclass(frozen=True, slots=True)
class Scores:
    """Set metrics of one answerable question."""

    expected: int
    returned: int
    hits: int
    precision: float
    recall: float
    f1: float
    hit: bool
    all_gold: bool
    exact: bool

    def values(self) -> dict[str, float]:
        """Every metric as a number (True = 1.0), keyed as in METRICS."""
        return {name: float(getattr(self, name)) for name in METRICS}


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question_id: str
    kind: Kind
    reply: Reply
    scores: Scores | None  # answerable questions only
    abstained: bool | None  # unanswerable questions only

    @property
    def failed(self) -> bool:
        return self.reply.error is not None


def dedupe(ids: Sequence[str]) -> tuple[str, ...]:
    """The ids without repeats, each at its first position."""
    return tuple(dict.fromkeys(ids))


def score_ids(returned: Sequence[str], expected: frozenset[str]) -> Scores:
    """Set metrics of the ids one answer returned against its expected ids (1 or more)."""
    if not expected:
        raise ValueError("an unanswerable question has no set metrics; see judge()")
    got = set(returned)
    hits = len(got & expected)
    precision = hits / len(got) if got else 0.0
    recall = hits / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(
        expected=len(expected),
        returned=len(got),
        hits=hits,
        precision=precision,
        recall=recall,
        f1=f1,
        hit=hits > 0,
        all_gold=expected <= got,
        exact=got == expected,
    )


def judge(question: EvalQuestion, reply: Reply) -> QuestionResult:
    """Score one reply. A failed reply returned nothing and never abstains."""
    returned = () if reply.error is not None else reply.document_ids
    if question.expected:
        return QuestionResult(
            question.question_id, question.kind, reply, score_ids(returned, question.expected), None
        )
    abstained = reply.error is None and not returned
    return QuestionResult(question.question_id, question.kind, reply, None, abstained)


# ---------------------------------------------------------------- aggregates
@dataclass(frozen=True, slots=True)
class GroupScores:
    n: int
    errors: int
    means: dict[str, float] | None  # metric -> mean over the group; None when it is empty


@dataclass(frozen=True, slots=True)
class AbstainScores:
    n: int
    errors: int
    abstained: int
    rate: float | None  # abstained / n; None when there are no unanswerable questions


@dataclass(frozen=True, slots=True)
class Summary:
    questions: int
    counts: dict[str, int]  # kind -> questions
    groups: dict[str, GroupScores]  # single, multi, answerable
    unanswerable: AbstainScores
    latency_s: dict[str, float | None]  # "p50", "p95", ..., "max"
    errors: int


def group_scores(results: Sequence[QuestionResult]) -> GroupScores:
    scored = [r.scores for r in results if r.scores is not None]
    errors = sum(r.failed for r in results)
    if not scored:
        return GroupScores(0, errors, None)
    rows = [s.values() for s in scored]
    means = {name: sum(row[name] for row in rows) / len(rows) for name in METRICS}
    return GroupScores(len(scored), errors, means)


def summarize(results: Sequence[QuestionResult], percentiles: Sequence[int]) -> Summary:
    by_kind = {kind: [r for r in results if r.kind == kind] for kind in ("single", "multi")}
    answerable = [r for r in results if r.scores is not None]
    unanswerable = [r for r in results if r.abstained is not None]
    abstained = sum(bool(r.abstained) for r in unanswerable)
    latencies = [r.reply.latency_s for r in results]
    latency: dict[str, float | None] = {f"p{p}": percentile(latencies, p) for p in percentiles}
    latency["max"] = max(latencies) if latencies else None
    return Summary(
        questions=len(results),
        counts={
            "single": len(by_kind["single"]),
            "multi": len(by_kind["multi"]),
            "unanswerable": len(unanswerable),
        },
        groups={
            "single": group_scores(by_kind["single"]),
            "multi": group_scores(by_kind["multi"]),
            ANSWERABLE: group_scores(answerable),
        },
        unanswerable=AbstainScores(
            n=len(unanswerable),
            errors=sum(r.failed for r in unanswerable),
            abstained=abstained,
            rate=abstained / len(unanswerable) if unanswerable else None,
        ),
        latency_s=latency,
        errors=sum(r.failed for r in results),
    )


def percentile(values: Sequence[float], p: int) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p % of values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = -(-p * len(ordered) // 100)  # ceil(p * n / 100) in integers: no float rounding
    return ordered[min(max(rank, 1), len(ordered)) - 1]
