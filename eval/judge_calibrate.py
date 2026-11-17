"""Calibrate the answer judge against the hand checks: `uv run eval-judge-calibrate`.

Re-judges the answers of hand-checked tune runs (each run folder's per_question.jsonl, as the
runner wrote it) with the answer judge (eval/judge.py), and compares the result with the hand
check in the same folder (answer_check.json: judged by an agent against answer_facts):
- fact status, exactly (right / partly / missing / wrong), over every fact of every answer
  the judge judged;
- the question verdict (the hand check's "partly correct" is the judge's "partly");
- for the unanswerable questions, says_not_found against "the hand check marked the 'says it
  is not found' fact right".
For facts and verdicts it gives the agreement, a confusion matrix and Cohen's kappa, and it
lists every disagreement. It also reports the judge's own consistency (do its first two
judgments agree) and the cost. The bar agreed with the user: fact agreement of at least
JUDGE_CALIBRATION_BAR (0.90).

Tune data only (AGENTS.md: Eval split). Questions come from scripts.show_questions'
load_questions (it refuses held-out), and runs only from eval/results/tune/ask/ or
eval/results/smoke/ask/. It never calls /ask. Judgments are kept in JUDGE_CACHE_DIR, so a
re-run of the same prompt version pays only for what it has not judged yet (--no-cache: judge
everything again). Output: eval/results/judge-calibration/<time>/report.md and report.json.
"""

import argparse
import asyncio
import json
import logging
import math
import os
import shlex
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpusqa import obs
from corpusqa.config import (
    EVAL_ASK_URL,
    EVAL_RESULTS_DIR,
    EVAL_TRACE_FLUSH_TIMEOUT_S,
    JUDGE_CACHE_DIR,
    JUDGE_CALIBRATION_BAR,
    JUDGE_CONCURRENCY,
    JUDGE_JUDGMENTS,
    JUDGE_MODEL,
    JUDGE_SPEND_LEDGER,
    JUDGE_THINKING_BUDGET,
    LOG_FORMAT,
    LOG_LEVEL,
)
from corpusqa.llm.errors import ChatUnavailable, LLMError
from corpusqa.llm.interfaces import ChatModel
from corpusqa.llm.limits import SpendLedger
from eval.judge import (
    LABELS,
    STATUSES,
    UNJUDGED,
    Judge,
    JudgeItem,
    JudgmentCache,
    Label,
    QuestionJudgment,
    Status,
    build_judge_chat,
)
from eval.retrieval import git_commit
from eval.run import display, fresh_folder, tracing_settings
from scripts.show_questions import QUESTIONS_FILE, SETS_DIR, load_questions

logger = logging.getLogger("eval.judge_calibrate")

# Module-level so tests can point them at tmp_path.
RESULTS_DIR = EVAL_RESULTS_DIR
CACHE_DIR = JUDGE_CACHE_DIR
LEDGER = JUDGE_SPEND_LEDGER

TRACE_NAME = "judge_calibration"
OUTPUT_SUBDIR = "judge-calibration"
REPORT_MD = "report.md"
REPORT_JSON = "report.json"
PER_QUESTION = "per_question.jsonl"
HAND_CHECK = "answer_check.json"
# The 5 hand-checked tune runs (eval/results/tune/ask/<name>), 150 answers in all.
DEFAULT_RUNS = (
    "20261017T033026Z",
    "20261017T033904Z",
    "20261017T034320Z",
    "20261017T072027Z",
    "20261017T072902Z",
)
# Run folders may only come from these sets' /ask results (never held-out).
ALLOWED_SETS = ("tune", "smoke")
DEFAULT_SET = "tune"
HAND_VERDICTS: dict[str, Label] = {
    "correct": "correct",
    "partly correct": "partly",
    "wrong": "wrong",
}
SHOWN_CHARS = 220  # a fact, quote or reason in report.md is cut to this length


class CalibrationError(ValueError):
    """The runs or the hand checks cannot be compared (the message says why)."""


# ---------------------------------------------------------------- inputs
@dataclass(frozen=True, slots=True)
class HandCheck:
    verdict: Label
    facts: tuple[Status, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class Case:
    """One answer of one run, with its hand check."""

    run: str
    item: JudgeItem
    fact_texts: tuple[str, ...]
    hand: HandCheck


def allowed(path: Path, results_dir: Path) -> bool:
    """Is `path`, symlinks resolved, inside (not at) an allowed set's /ask results?"""
    resolved = path.resolve()
    roots = [(results_dir / s / "ask").resolve() for s in ALLOWED_SETS]
    return any(resolved.is_relative_to(root) and resolved != root for root in roots)


def run_folder(name: str, results_dir: Path) -> Path:
    """The folder of a run: a bare name is under <results>/tune/ask/. Refused unless it is
    under the /ask results of an allowed set."""
    path = Path(name)
    folder = path if len(path.parts) > 1 else results_dir / DEFAULT_SET / "ask" / name
    if not allowed(folder, results_dir):
        raise CalibrationError(
            f"{name}: only runs under the /ask results of {ALLOWED_SETS} may be calibrated "
            "against (AGENTS.md: Eval split)"
        )
    return folder.resolve()


def input_file(folder: Path, name: str, results_dir: Path) -> Path:
    """A run's input file, refused when it (a symlink, say) leads outside the allowed runs."""
    path = folder / name
    if not allowed(path, results_dir):
        raise CalibrationError(
            f"{folder.name}/{name} leads outside the /ask results of {ALLOWED_SETS} "
            "(AGENTS.md: Eval split)"
        )
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_cases(
    runs: Sequence[str], results_dir: Path, questions_file: Path, sets_dir: Path
) -> list[Case]:
    """Every answer of the runs that has a hand check, in run then file order."""
    questions = {
        q["question_id"]: q
        for name in ALLOWED_SETS
        for q in load_questions(name, questions_file, sets_dir)
    }
    cases: list[Case] = []
    for name in runs:
        folder = run_folder(name, results_dir)
        answers_file = input_file(folder, PER_QUESTION, results_dir)
        answers = {row["question_id"]: row for row in read_jsonl(answers_file)}
        check_file = input_file(folder, HAND_CHECK, results_dir)
        check = json.loads(check_file.read_text(encoding="utf-8"))
        for hand in check["questions"]:
            qid = hand["id"]
            if qid not in questions or qid not in answers:
                # Not named: an id outside the tune and smoke sets may be a held-out one.
                raise CalibrationError(
                    f"{folder.name}: a hand-checked question is not in the tune or smoke "
                    "set, or has no answer in the run"
                )
            question, row = questions[qid], answers[qid]
            facts = tuple(question["answer_facts"])
            statuses = tuple(hand["fact_status"])
            if len(statuses) != len(facts) or not set(statuses) <= set(STATUSES):
                raise CalibrationError(
                    f"{folder.name}: {qid}: hand statuses {statuses} do not fit its "
                    f"{len(facts)} facts"
                )
            if hand["verdict"] not in HAND_VERDICTS:
                raise CalibrationError(f"{folder.name}: {qid}: unknown verdict {hand['verdict']}")
            item = JudgeItem(
                question_id=qid,
                question=question["question"],
                gold_answer=question["gold_answer"],
                facts=facts,
                answerable=bool(question["expected_doc_ids"]),
                answer=row["answer"] if row.get("error") is None else None,
            )
            cases.append(
                Case(
                    run=folder.name,
                    item=item,
                    fact_texts=facts,
                    hand=HandCheck(HAND_VERDICTS[hand["verdict"]], statuses, hand["reason"]),
                )
            )
    return cases


# ---------------------------------------------------------------- comparison
def cohen_kappa(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> float | None:
    """Agreement beyond chance: (observed - expected) / (1 - expected), where expected is the
    agreement two raters with these label shares would reach by chance. None when undefined
    (no pairs, or both raters always give the same one label)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[label] * second[label] for label in labels) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def confusion(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Rows: the hand check's label; columns: the judge's."""
    counts = Counter(pairs)
    return {hand: {judge: counts[(hand, judge)] for judge in labels} for hand in labels}


def share(part: int, whole: int) -> float | None:
    return None if whole == 0 else part / whole


@dataclass(frozen=True, slots=True)
class Agreement:
    agree: int
    total: int
    kappa: float | None
    matrix: dict[str, dict[str, int]]

    @property
    def rate(self) -> float | None:
        return share(self.agree, self.total)

    def to_json(self) -> dict[str, Any]:
        return {
            "agree": self.agree,
            "total": self.total,
            "rate": _round(self.rate),
            "kappa": _round(self.kappa),
            "confusion_hand_rows_judge_columns": self.matrix,
        }


def agreement(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> Agreement:
    return Agreement(
        agree=sum(a == b for a, b in pairs),
        total=len(pairs),
        kappa=cohen_kappa(pairs, labels),
        matrix=confusion(pairs, labels),
    )


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def compare(cases: Sequence[Case], results: Sequence[QuestionJudgment]) -> dict[str, Any]:
    """Every number of the report (see the module docstring)."""
    fact_pairs: list[tuple[str, str]] = []
    verdict_pairs: list[tuple[str, str]] = []
    said_pairs: list[tuple[bool, bool]] = []
    fact_disagreements: list[dict[str, Any]] = []
    verdict_disagreements: list[dict[str, Any]] = []
    unjudged: list[dict[str, Any]] = []
    per_run: dict[str, dict[str, int]] = {}
    for case, result in zip(cases, results, strict=True):
        run = per_run.setdefault(
            case.run, {"answers": 0, "facts": 0, "facts_agree": 0, "verdicts_agree": 0}
        )
        run["answers"] += 1
        where = {"run": case.run, "question_id": case.item.question_id}
        if result.verdict is None:
            unjudged.append({**where, "problems": list(result.problems)})
            continue
        for n, (hand, judged) in enumerate(zip(case.hand.facts, result.facts, strict=True), 1):
            fact_pairs.append((hand, judged))
            run["facts"] += 1
            run["facts_agree"] += hand == judged
            if hand != judged:
                fact_disagreements.append({
                    **where, "fact": n, "fact_text": case.fact_texts[n - 1], "hand": hand,
                    "judge": judged, "judge_evidence": result.evidence[n - 1],
                    "hand_reason": case.hand.reason,
                })  # fmt: skip
        verdict_pairs.append((case.hand.verdict, result.verdict))
        run["verdicts_agree"] += case.hand.verdict == result.verdict
        if case.hand.verdict != result.verdict:
            verdict_disagreements.append({
                **where, "hand": case.hand.verdict, "judge": result.verdict,
                "judge_verdicts": list(result.verdicts), "judge_reason": result.reason,
                "hand_reason": case.hand.reason,
                "contradictions": [c.model_dump() for c in result.contradictions],
            })  # fmt: skip
        if not case.item.answerable and result.says_not_found is not None:
            said_pairs.append((case.hand.facts[0] == "right", result.says_not_found))
    return {
        "answers": len(cases),
        "judged": len(cases) - len(unjudged),
        "facts": agreement(fact_pairs, STATUSES).to_json(),
        "verdicts": agreement(verdict_pairs, LABELS).to_json(),
        "said_not_found": {
            "unanswerable_judged": len(said_pairs),
            "agree": sum(a == b for a, b in said_pairs),
            "hand_said": sum(a for a, _ in said_pairs),
            "judge_said": sum(b for _, b in said_pairs),
        },
        "per_run": per_run,
        "consistency": consistency(results),
        "unjudged": unjudged,
        "fact_disagreements": fact_disagreements,
        "verdict_disagreements": verdict_disagreements,
    }


def consistency(results: Sequence[QuestionJudgment]) -> dict[str, Any]:
    """The judge against itself: its first two judgments of each answer."""
    two = [r for r in results if len(r.judgments) >= 2]
    same_verdict = sum(r.judgments[0].verdict == r.judgments[1].verdict for r in two)
    fact_pairs = [
        (a.status, b.status)
        for r in two
        for a, b in zip(r.judgments[0].facts, r.judgments[1].facts, strict=True)
    ]
    same_fact = sum(a == b for a, b in fact_pairs)
    return {
        "answers_with_two_judgments": len(two),
        "first_two_same_verdict": same_verdict,
        "verdict_rate": _round(share(same_verdict, len(two))),
        "facts_compared": len(fact_pairs),
        "first_two_same_fact_status": same_fact,
        "fact_rate": _round(share(same_fact, len(fact_pairs))),
        "third_judgments": sum(len(r.judgments) > JUDGE_JUDGMENTS for r in results),
        "unjudged": sum(r.verdict is None for r in results),
    }


# ---------------------------------------------------------------- report
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cell(text: object) -> str:
    shown = " ".join(str(text).split())
    if len(shown) > SHOWN_CHARS:
        shown = shown[: SHOWN_CHARS - 3] + "..."
    return shown.replace("|", "\\|")


def matrix_lines(matrix: dict[str, dict[str, int]], labels: Sequence[str]) -> list[str]:
    lines = [
        "| hand \\ judge | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for hand in labels:
        lines.append(f"| {hand} | " + " | ".join(str(matrix[hand][j]) for j in labels) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    c = report["comparison"]
    facts, verdicts, said, cons = c["facts"], c["verdicts"], c["said_not_found"], c["consistency"]
    passed = "PASSES" if report["passed"] else "MISSES"
    lines = [
        f"# Judge calibration: prompt `judge` v{report['prompt_version']} ({report['model']})",
        "",
        f"{report['timestamp']}, commit `{report['git_commit']}`. "
        f"{c['answers']} answers from {len(report['runs'])} hand-checked tune runs, "
        f"{c['judged']} judged. Label: {report['label'] or 'none'}.",
        "",
        "Each answer was judged again by the automatic judge (eval/judge.py) and compared with "
        "the hand check in its run folder (answer_check.json, checked by an agent against "
        "answer_facts). The judge saw the question, the gold answer, the facts and the answer "
        "text, not the run or its setting.",
        "",
        "## Result",
        "",
        f"- **Fact status agreement: {_pct(facts['rate'])}** ({facts['agree']} of "
        f"{facts['total']} facts have exactly the hand check's status). The bar is "
        f"{_pct(report['bar'])}: the judge **{passed}** it.",
        f"- Question verdict agreement: {_pct(verdicts['rate'])} ({verdicts['agree']} of "
        f"{verdicts['total']}).",
        f"- Cohen's kappa: facts {_num(facts['kappa'])}, verdicts {_num(verdicts['kappa'])}.",
        f'- Unanswerable questions, "says it is not in the records": the judge agrees with the '
        f"hand check on {said['agree']} of {said['unanswerable_judged']} (hand: "
        f"{said['hand_said']} said so; judge: {said['judge_said']}).",
        f"- Unjudged answers: {len(c['unjudged'])}.",
        "",
        "How to read it. Agreement = the share of facts (or questions) where the judge gave "
        "exactly the hand check's label. Cohen's kappa = agreement beyond what chance would give: "
        "(observed agreement - chance agreement) / (1 - chance agreement), where chance agreement "
        "is what two raters with these label shares would reach by guessing. 1 = perfect, 0 = no "
        "better than chance. Most facts are `right`, so plain agreement is high even for a judge "
        "that says `right` to everything; kappa shows whether the rare labels match too.",
        "",
        "## Facts: confusion matrix (rows: hand check, columns: judge)",
        "",
        *matrix_lines(facts["confusion_hand_rows_judge_columns"], STATUSES),
        "",
        "## Verdicts: confusion matrix (rows: hand check, columns: judge)",
        "",
        *matrix_lines(verdicts["confusion_hand_rows_judge_columns"], LABELS),
        "",
        "## Per run",
        "",
        "| run | answers | fact agreement | verdict agreement |",
        "|---|---|---|---|",
    ]
    for run, n in c["per_run"].items():
        judged = n["answers"] - sum(u["run"] == run for u in c["unjudged"])
        lines.append(
            f"| {run} | {n['answers']} | {n['facts_agree']} of {n['facts']} "
            f"({_pct(share(n['facts_agree'], n['facts']))}) | {n['verdicts_agree']} of "
            f"{judged} ({_pct(share(n['verdicts_agree'], judged))}) |"
        )
    lines += [
        "",
        "## The judge's own consistency",
        "",
        f"Each answer was judged {JUDGE_JUDGMENTS} times (a third time when the two verdicts "
        "differed; the majority counts).",
        f"- Same question verdict in the first two judgments: {cons['first_two_same_verdict']} "
        f"of {cons['answers_with_two_judgments']} ({_pct(cons['verdict_rate'])}).",
        f"- Same fact status in the first two judgments: {cons['first_two_same_fact_status']} "
        f"of {cons['facts_compared']} ({_pct(cons['fact_rate'])}).",
        f"- Third judgments: {cons['third_judgments']}. Unjudged: {cons['unjudged']}.",
        "",
        "## Cost",
        "",
        f"- Model calls this run: {report['calls']}; judgments reused from the cache: "
        f"{report['cached']}.",
        f"- Cost of this run's calls: ${report['cost_usd']:.4f}. Judge ledger total "
        f"(`.cache/spend/judge.json`): "
        + ("n/a" if report["ledger_usd"] is None else f"${report['ledger_usd']:.4f}")
        + ".",
        "",
        f"## Fact disagreements ({len(c['fact_disagreements'])})",
        "",
        "| run | question | # | fact | hand | judge | judge's evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in c["fact_disagreements"]:
        lines.append(
            f"| {d['run']} | {d['question_id']} | {d['fact']} | {_cell(d['fact_text'])} | "
            f"{d['hand']} | {d['judge']} | {_cell(d['judge_evidence']) or '-'} |"
        )
    lines += [
        "",
        f"## Verdict disagreements ({len(c['verdict_disagreements'])})",
        "",
        "| run | question | hand | judge (all) | judge's reason | hand check's reason |",
        "|---|---|---|---|---|---|",
    ]
    for d in c["verdict_disagreements"]:
        lines.append(
            f"| {d['run']} | {d['question_id']} | {d['hand']} | {d['judge']} "
            f"({', '.join(d['judge_verdicts'])}) | {_cell(d['judge_reason'])} | "
            f"{_cell(d['hand_reason'])} |"
        )
    if c["unjudged"]:
        lines += ["", "## Unjudged", ""]
        lines += [
            f"- {u['run']} {u['question_id']}: {_cell('; '.join(u['problems']))}"
            for u in c["unjudged"]
        ]
    lines += [
        "",
        "## Run",
        "",
        f"- command: `{report['command']}`",
        f"- model `{report['model']}`, thinking budget {report['thinking_budget']}, "
        f"{JUDGE_JUDGMENTS} judgments per answer, prompt `judge` v{report['prompt_version']}",
        "- runs: " + ", ".join(f"`{r}`" for r in report["runs"]),
        f"- wall time: {report['seconds']} s",
    ]
    return "\n".join(lines) + "\n"


def question_rows(cases: Sequence[Case], results: Sequence[QuestionJudgment]) -> list[dict]:
    return [
        {
            "run": case.run,
            "question_id": case.item.question_id,
            "hand": {"verdict": case.hand.verdict, "facts": list(case.hand.facts)},
            "judge": {
                "outcome": result.outcome,
                "verdict": result.verdict,
                "facts": list(result.facts),
                "evidence": list(result.evidence),
                "contradictions": [c.model_dump() for c in result.contradictions],
                "says_not_found": result.says_not_found,
                "reason": result.reason,
                "verdicts": list(result.verdicts),
                "problems": list(result.problems),
                "calls": result.calls,
                "cost_usd": round(result.cost_usd, 6),
            },
        }
        for case, result in zip(cases, results, strict=True)
    ]


# ---------------------------------------------------------------- run
def ledger_total() -> float | None:
    try:
        return SpendLedger(LEDGER, math.inf).totals.cost_usd
    except LLMError as exc:
        logger.warning("judge ledger unreadable (%s)", exc)
        return None


def calibrate(
    args: argparse.Namespace,
    chat: ChatModel,
    command: str,
    questions_file: Path = QUESTIONS_FILE,
    sets_dir: Path = SETS_DIR,
) -> Path:
    """Judge, compare, write the report; returns its folder."""
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    cases = load_cases(args.runs, RESULTS_DIR, questions_file, sets_dir)
    cache = None if args.no_cache else JudgmentCache(CACHE_DIR)
    judge = Judge(chat, model_name=JUDGE_MODEL, cache=cache, concurrency=args.concurrency)
    print(f"calibrating prompt judge v{judge.prompt.version} ({JUDGE_MODEL}) on {len(cases)} "
          f"answers from {len(args.runs)} runs", flush=True)  # fmt: skip
    done = Counter[str]()

    def progress(result: QuestionJudgment) -> None:
        done[result.outcome] += 1
        print(f"  {sum(done.values())}/{len(cases)} judged ({done[UNJUDGED]} unjudged)",
              flush=True)  # fmt: skip

    results = asyncio.run(judge.judge_all([c.item for c in cases], progress))
    report: dict[str, Any] = {
        "kind": "judge_calibration",
        "timestamp": stamp,
        "git_commit": git_commit(),
        "command": command,
        "label": args.label,
        "model": JUDGE_MODEL,
        "prompt_name": judge.prompt.name,
        "prompt_version": judge.prompt.version,
        "thinking_budget": JUDGE_THINKING_BUDGET,
        "judgments_per_answer": JUDGE_JUDGMENTS,
        "runs": list(dict.fromkeys(case.run for case in cases)),
        "bar": JUDGE_CALIBRATION_BAR,
        "comparison": compare(cases, results),
        "calls": sum(r.calls for r in results),
        "cached": sum(r.cached for r in results),
        "cost_usd": round(sum(r.cost_usd for r in results), 4),
        "ledger_usd": _round(ledger_total()),
    }
    rate = report["comparison"]["facts"]["rate"]
    report["passed"] = rate is not None and rate >= JUDGE_CALIBRATION_BAR
    report["seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    report["questions"] = question_rows(cases, results)
    folder = fresh_folder(RESULTS_DIR / OUTPUT_SUBDIR, stamp)
    (folder / REPORT_JSON).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (folder / REPORT_MD).write_text(render_markdown(report), encoding="utf-8")
    obs.current().update(
        output={k: v for k, v in report.items() if k not in {"questions", "comparison"}}
    )
    c = report["comparison"]
    print(f"\nfact agreement {_pct(c['facts']['rate'])} (bar {_pct(JUDGE_CALIBRATION_BAR)}: "
          f"{'passes' if report['passed'] else 'misses'}), verdict agreement "
          f"{_pct(c['verdicts']['rate'])}, kappa facts {_num(c['facts']['kappa'])} verdicts "
          f"{_num(c['verdicts']['kappa'])}, cost ${report['cost_usd']:.4f}")  # fmt: skip
    print(f"report: {display(folder / REPORT_MD)}")
    return folder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="hand-checked run folders (a bare name is under eval/results/tune/ask/)",
    )
    parser.add_argument("--concurrency", type=int, default=JUDGE_CONCURRENCY)
    parser.add_argument("--no-cache", action="store_true", help="judge everything again")
    parser.add_argument("--label", default=None, help="a note for the report (e.g. round 2)")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    command = shlex.join(["python", "-m", "eval.judge_calibrate", *argv])
    port = int(EVAL_ASK_URL.rsplit(":", 1)[-1])
    settings = tracing_settings(port, os.environ)
    try:
        chat = build_judge_chat(settings)
    except ChatUnavailable as exc:
        logger.error("%s", exc)
        return 2
    tracer = obs.init_tracing(settings)
    try:
        with tracer.trace(TRACE_NAME, input={"runs": args.runs, "label": args.label}) as trace:
            logger.info("trace id %s", trace.trace_id)
            calibrate(args, chat, command)
            return 0
    except CalibrationError as exc:
        logger.error("%s", exc)
        return 2
    finally:
        tracer.shutdown(EVAL_TRACE_FLUSH_TIMEOUT_S)


if __name__ == "__main__":
    sys.exit(main())
