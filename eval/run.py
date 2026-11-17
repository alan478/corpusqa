"""End-to-end eval: `uv run eval-smoke|eval-tune|eval-heldout|eval-all` (`python -m eval.run`).

Asks the RUNNING server every question of a set through POST /ask, the way the grader does
(JSON `{"question": ...}`, at most --concurrency requests in flight, --timeout per question
from sending to the last byte), and compares the returned `document_ids` with each
question's expected ids as sets (eval/ask_metrics.py: precision, recall, F1, hit, all_gold,
exact; unanswerable = no ids returned; no rank metrics, user decision 2026-10-16). It is the
final test shown to the graders, not a tool to tune the system with.

Before the first question it checks GET /health once and stops (exit 3, no report) when the
server is not ready. Output, in eval/results/<set>/ask/<time>/:
- summary.md: for graders. Set metrics per group (single-doc, multi-doc, all answerable),
  the unanswerable abstain rate, latency, errors, the command, commit, set and milestone.
- summary.json: the same aggregates with the settings.
- per_question.jsonl: one row per question (id, kind, number of expected ids, HTTP status,
  latency, answer, returned ids, the server's trace id, error, metrics). Never the question
  text or the expected ids.
For the held-out set and `all` (both hold held-out questions), per_question.jsonl goes under
eval/results/heldout/ask/<time>/ (a path the agent guard blocks), and the readable summary,
the printed lines and the trace output hold aggregates only (AGENTS.md: Eval split).
One row goes to eval/history.csv (kind `ask`, headline `recall`: the mean per-question
recall over the answerable questions).

After the requests, the answer judge (eval/judge.py, JUDGE_MODEL) judges every answer text
fact by fact against its `answer_facts` (off with --no-judge; not run without a Gemini key).
Its aggregates go into summary.md, summary.json, the printed lines and the history row:
shares of questions correct / partly / wrong / unjudged, shares of facts right / partly /
missing / wrong, contradictions, the unanswerable "said not found" rate, judge agreement and
cost. Its per-question results go to judge.jsonl beside summary.md, for smoke and tune only:
for the held-out set and `all` nothing per question is written or traced. Citations and
"an unanswerable question returned no ids" stay as the runner computes them. The judge has
its own spend ledger (.cache/spend/judge.json), so `spend_total` counts it from then on; the
report's spend line is the system's (measured before judging) and the judge's cost is shown
apart.

One run is one trace `eval_ask` with a span `question` per request (id, kind, status,
latency, metrics, the server's trace id; no question text, answer or expected ids). For the
held-out set and `all` a span keeps only its status and latency. Without Langfuse keys
tracing is off and costs nothing.
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, StrictStr, ValidationError

from corpusqa import obs
from corpusqa.config import (
    ASK_PATH,
    EVAL_ASK_CONCURRENCY,
    EVAL_ASK_LATENCY_PERCENTILES,
    EVAL_ASK_MAX_CONCURRENCY,
    EVAL_ASK_TIMEOUT_S,
    EVAL_ASK_URL,
    EVAL_HISTORY,
    EVAL_RESULTS_DIR,
    EVAL_SPEND_DIR,
    EVAL_TRACE_FLUSH_TIMEOUT_S,
    HEALTH_PATH,
    JUDGE_JUDGMENTS,
    JUDGE_MODEL,
    JUDGE_PROMPT_NAME,
    JUDGE_THINKING_BUDGET,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_RESPONSE_BYTES,
    TRACE_ID_HEADER,
    Settings,
)
from corpusqa.llm.errors import ChatUnavailable, LLMError, SpendCapReached
from corpusqa.llm.interfaces import ChatModel
from corpusqa.llm.limits import SpendLedger
from eval import judge as answer_judge
from eval.ask_metrics import (
    ANSWERABLE,
    GROUPS,
    METRICS,
    QuestionResult,
    Reply,
    Summary,
    dedupe,
    judge,
    summarize,
)
from eval.questions import PROTECTED_SETS, RUN_SETS, EvalQuestion, load_run_questions
from eval.retrieval import HISTORY_HEADER, ROOT, git_commit
from scripts.show_questions import QUESTIONS_FILE, SETS_DIR

logger = logging.getLogger("eval.run")

# Module-level so tests can point them at tmp_path.
RESULTS_DIR = EVAL_RESULTS_DIR
HISTORY = EVAL_HISTORY
SPEND_DIR = EVAL_SPEND_DIR

TRACE_NAME = "eval_ask"
SPAN_NAME = "question"
HISTORY_KIND = "ask"
HEADLINE = "recall"  # mean per-question recall over the answerable questions
SYSTEMS = ("full", "baseline")
BUILT_SYSTEMS = ("full",)  # the one-shot baseline is not built yet
# The results folder the agent guard blocks (scripts/hooks/guard_split.py).
PROTECTED_RESULTS_SUBDIR = "heldout"
SUMMARY_MD = "summary.md"
SUMMARY_JSON = "summary.json"
PER_QUESTION = "per_question.jsonl"
JUDGE_FILE = "judge.jsonl"  # per-question judge results: smoke and tune only
HTTP_DEFAULT_PORTS = {"http": 80, "https": 443}
EXIT_NOT_READY = 3
# A failed span of a protected set says only this; the reason is in the guarded file.
PROTECTED_SPAN_ERROR = "request failed (the reason is in the guarded per-question file)"
ERROR_DETAILS_SHOWN = 3  # validation problems named in one error message

GROUP_LABELS = {
    "single": "single-doc (1 expected id)",
    "multi": "multi-doc (2+ expected ids)",
    ANSWERABLE: "all answerable",
    "unanswerable": "unanswerable (0 expected ids)",
}
METRIC_LABELS = {
    "precision": "precision",
    "recall": "recall",
    "f1": "F1",
    "hit": "hit",
    "all_gold": "all_gold",
    "exact": "exact",
}


# ---------------------------------------------------------------- one request
class WireReply(BaseModel):
    """The CONTRACT §3 body: an answer string and a list of id strings (extra fields ignored)."""

    answer: StrictStr
    document_ids: list[StrictStr]


def read_reply(
    status: int, content: bytes, latency_s: float, *, trace_id: str | None = None
) -> Reply:
    """Turn one HTTP response into a Reply; anything off the contract is a failed reply.

    `trace_id` is the response's `TRACE_ID_HEADER` header (our server sends it; others may
    not). It is kept on a failed reply too, so a bad reply can be opened in Langfuse.
    """
    trace_id = trace_id or None  # an empty header is no trace id

    def failed(error: str) -> Reply:
        return Reply(status, latency_s, None, (), trace_id, error)

    if status != 200:
        return failed(f"HTTP {status}")
    if len(content) > MAX_RESPONSE_BYTES:
        return failed(f"body of {len(content)} bytes is over the {MAX_RESPONSE_BYTES} limit")
    try:
        body = json.loads(content)
    except (ValueError, RecursionError):  # deep nesting raises RecursionError (Codex finding 3)
        return failed("body is not JSON")
    try:
        wire = WireReply.model_validate(body)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['msg']}"
            for e in exc.errors()[:ERROR_DETAILS_SHOWN]
        ]
        return failed("body is off the contract: " + "; ".join(problems))
    if not wire.answer.strip():
        return failed("empty answer")
    return Reply(status, latency_s, wire.answer, dedupe(wire.document_ids), trace_id)


class NotReady(Exception):
    """GET /health did not answer 200 before the run."""


@dataclass(slots=True)
class Progress:
    total: int
    done: int = 0
    failed: int = 0

    def add(self, result: QuestionResult) -> None:
        self.done += 1
        self.failed += result.failed
        # Counts only: per-question lines would show held-out results.
        print(f"  {self.done}/{self.total} answered, {self.failed} failed", flush=True)


async def check_health(client: httpx.AsyncClient, timeout_s: float) -> None:
    try:
        response = await asyncio.wait_for(client.get(HEALTH_PATH), timeout_s)
    except (TimeoutError, httpx.HTTPError) as exc:
        raise NotReady(f"GET {HEALTH_PATH} failed ({type(exc).__name__}: {exc})") from exc
    if response.status_code != 200:
        raise NotReady(f"GET {HEALTH_PATH} answered {response.status_code}")


async def ask_one(
    client: httpx.AsyncClient,
    gate: asyncio.Semaphore,
    question: EvalQuestion,
    timeout_s: float,
    progress: Progress,
    protected: bool,
) -> QuestionResult:
    async with gate:
        # A protected set's span names neither the question nor its scores (Codex finding 1).
        meta = {} if protected else {"question_id": question.question_id, "kind": question.kind}
        with obs.span(SPAN_NAME, metadata=meta) as span:
            started = time.perf_counter()
            try:
                # post() reads the whole body, so the limit runs to the last byte.
                response = await asyncio.wait_for(
                    client.post(ASK_PATH, json={"question": question.text}), timeout_s
                )
            except TimeoutError:
                error = f"timeout: no full reply within {timeout_s:g} s"
                reply = Reply(None, time.perf_counter() - started, None, (), None, error)
            except httpx.HTTPError as exc:
                error = f"request failed: {type(exc).__name__}: {exc}"
                reply = Reply(None, time.perf_counter() - started, None, (), None, error)
            else:
                elapsed = time.perf_counter() - started
                reply = read_reply(
                    response.status_code,
                    response.content,
                    elapsed,
                    trace_id=response.headers.get(TRACE_ID_HEADER),
                )
            result = judge(question, reply)
            span.update(metadata=span_metadata(result, protected))
            if result.reply.error is not None:
                span.error(PROTECTED_SPAN_ERROR if protected else result.reply.error)
    progress.add(result)
    return result


async def ask_all(
    questions: Sequence[EvalQuestion],
    url: str,
    concurrency: int,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
    protected: bool = False,
) -> list[QuestionResult]:
    """Every question, at most `concurrency` in flight; results in question order.

    `protected`: the set holds held-out questions, so each span keeps only its status and
    latency."""
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=url, timeout=timeout_s, limits=limits, transport=transport
    ) as client:
        await check_health(client, timeout_s)
        gate = asyncio.Semaphore(concurrency)
        progress = Progress(len(questions))
        return list(
            await asyncio.gather(
                *(ask_one(client, gate, q, timeout_s, progress, protected) for q in questions)
            )
        )


def span_metadata(result: QuestionResult, protected: bool) -> dict[str, object]:
    """A question span's fields. For a protected set only the status and the latency: no
    question id, server trace id, error text or scores (those are per-question results)."""
    reply = result.reply
    meta: dict[str, object] = {"status": reply.status, "latency_s": round(reply.latency_s, 3)}
    if protected:
        return meta | {"failed": result.failed}
    meta |= {
        "returned": len(reply.document_ids),
        "server_trace_id": reply.trace_id,
        "error": reply.error,
    }
    if result.scores is not None:
        meta |= {"expected": result.scores.expected, "hits": result.scores.hits}
        meta |= result.scores.values()
    else:
        meta["abstained"] = result.abstained
    return meta


# ---------------------------------------------------------------- run
def spend_total(folder: Path) -> float | None:
    """Recorded spend over every ledger in `folder` (0 when there is none); None if unreadable."""
    try:
        paths = sorted(folder.glob("*.json"))
        return sum(SpendLedger(path, math.inf).totals.cost_usd for path in paths)
    except (OSError, LLMError) as exc:
        logger.warning("spend ledger unreadable (%s); spend is reported as n/a", exc)
        return None


@dataclass(frozen=True, slots=True)
class JudgeRun:
    """What the answer judge did in one run: its results, or why it did not run."""

    note: str  # "done", "off (--no-judge)" or "not run: <why>"
    model: str = JUDGE_MODEL
    prompt_version: str | None = None
    totals: answer_judge.JudgeTotals | None = None
    results: tuple[answer_judge.QuestionJudgment, ...] = ()
    # Growth of every ledger while the judge ran: its own ledger only, as nothing else runs
    # then. It also counts calls billed but unusable, which the replies' sum misses.
    spend_usd: float | None = None


@dataclass(slots=True)
class JudgeProgress:
    total: int
    done: int = 0
    unjudged: int = 0

    def add(self, result: answer_judge.QuestionJudgment) -> None:
        self.done += 1
        self.unjudged += result.outcome == answer_judge.UNJUDGED
        # Counts only: per-question lines would show held-out results.
        print(f"  {self.done}/{self.total} judged, {self.unjudged} unjudged", flush=True)


def judge_item(question: EvalQuestion, result: QuestionResult) -> answer_judge.JudgeItem:
    return answer_judge.JudgeItem(
        question_id=question.question_id,
        question=question.text,
        gold_answer=question.gold_answer,
        facts=question.facts,
        answerable=bool(question.expected),
        answer=None if result.failed else result.reply.answer,
    )


def judge_answers(
    args: argparse.Namespace,
    questions: Sequence[EvalQuestion],
    results: Sequence[QuestionResult],
    chat: ChatModel | None,
    protected: bool,
) -> JudgeRun:
    """Judge every answer (see the module docstring). Never raises: a judge that stops
    leaves the /ask results reported, with the reason (for a protected set only the error's
    type, unless it is the spend cap)."""
    if not args.judge:
        return JudgeRun("off (--no-judge)")
    if chat is None:
        return JudgeRun("not run: no judge model (GEMINI_API_KEY is not set)")
    judge_model = answer_judge.Judge(chat, model_name=JUDGE_MODEL, traced=not protected)
    version = judge_model.prompt.version
    items = [judge_item(q, r) for q, r in zip(questions, results, strict=True)]
    print(f"judging {len(items)} answers with {JUDGE_MODEL} (prompt judge v{version})",
          flush=True)  # fmt: skip
    progress = JudgeProgress(len(items))
    before = spend_total(SPEND_DIR)
    try:
        judged = asyncio.run(judge_model.judge_all(items, progress.add))
    except Exception as exc:
        # JudgeFailed's message is only a type; the spend cap's names no question.
        shown = isinstance(exc, (SpendCapReached, answer_judge.JudgeFailed)) or not protected
        reason = f"{type(exc).__name__}: {exc}" if shown else type(exc).__name__
        logger.error("the answer judge stopped (%s); the /ask results are still reported", reason)
        return JudgeRun(f"not run: the judge stopped ({reason})", prompt_version=version)
    after = spend_total(SPEND_DIR)
    spend = None if before is None or after is None else after - before
    totals = answer_judge.summarize(judged)
    return JudgeRun("done", JUDGE_MODEL, version, totals, tuple(judged), spend)


def run(
    args: argparse.Namespace,
    command: str,
    transport: httpx.AsyncBaseTransport | None = None,
    judge_chat: ChatModel | None = None,
) -> int:
    """One eval run. `judge_chat` is the answer judge's model (main() builds it from the
    environment); without one the judge does not run and the report says so."""
    started = time.monotonic()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    questions = load_run_questions(args.set, QUESTIONS_FILE, SETS_DIR)
    protected = args.set in PROTECTED_SETS
    spend_before = spend_total(SPEND_DIR)
    print(
        f"eval {args.set}: {len(questions)} questions -> {args.url} "
        f"({args.concurrency} at once, {args.timeout:g} s limit each)",
        flush=True,
    )
    try:
        results = asyncio.run(
            ask_all(questions, args.url, args.concurrency, args.timeout, transport, protected)
        )
    except NotReady as exc:
        logger.error("server at %s is not ready: %s. Start it (uv run serve-dev) and wait for "
                     "GET /health 200; no report was written.", args.url, exc)  # fmt: skip
        obs.current().error(str(exc))
        return EXIT_NOT_READY
    spend_after = spend_total(SPEND_DIR)
    spend = None if spend_before is None or spend_after is None else spend_after - spend_before
    judging = judge_answers(args, questions, results, judge_chat, protected)
    summary = summarize(results, EVAL_ASK_LATENCY_PERCENTILES)
    folder, per_question_folder = output_folders(args.set, stamp)
    report = build_report(args, command, stamp, summary, results, spend, protected)
    report["judge"] = judge_report(judging, protected)
    report["seconds"] = round(time.monotonic() - started, 1)
    report["per_question_file"] = display(per_question_folder / PER_QUESTION)
    write_outputs(report, results, folder, per_question_folder, judging)
    append_history(report, folder, judging)
    hidden = {"error_list", "judge_not_correct"}
    obs.current().update(output={k: v for k, v in report.items() if k not in hidden})
    print_summary(report, folder, judging)
    return 0


def output_folders(set_name: str, stamp: str) -> tuple[Path, Path]:
    """The readable report folder and the folder of per_question.jsonl.

    For a protected set the per-question file goes under the guarded results folder; the
    held-out set's own folder is already there."""
    readable = fresh_folder(RESULTS_DIR / set_name / HISTORY_KIND, stamp)
    if set_name not in PROTECTED_SETS:
        return readable, readable
    guarded = RESULTS_DIR / PROTECTED_RESULTS_SUBDIR
    if readable.is_relative_to(guarded):
        return readable, readable
    return readable, fresh_folder(guarded / HISTORY_KIND, stamp)


def fresh_folder(parent: Path, stamp: str) -> Path:
    """parent/<stamp>, or parent/<stamp>-2, -3... when a run in the same second has it."""
    parent.mkdir(parents=True, exist_ok=True)
    for n in count(1):
        folder = parent / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise AssertionError("unreachable")


def display(path: Path) -> str:
    """A path relative to the repo when it is inside it."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


# ---------------------------------------------------------------- report
def build_report(
    args: argparse.Namespace,
    command: str,
    stamp: str,
    summary: Summary,
    results: Sequence[QuestionResult],
    spend: float | None,
    protected: bool,
) -> dict[str, Any]:
    """summary.json: aggregates and settings. Names failed questions only for readable sets."""
    report: dict[str, Any] = {
        "kind": HISTORY_KIND,
        "timestamp": stamp,
        "git_commit": git_commit(),
        "set": args.set,
        "milestone": args.milestone,
        "system": args.system,
        "url": args.url,
        "concurrency": args.concurrency,
        "timeout_s": args.timeout,
        "command": command,
        "questions": summary.questions,
        "counts": summary.counts,
        "groups": {
            name: {
                "n": g.n,
                "errors": g.errors,
                "means": None if g.means is None else {k: round(v, 4) for k, v in g.means.items()},
            }
            for name, g in summary.groups.items()
        },
        "unanswerable": {
            "n": summary.unanswerable.n,
            "errors": summary.unanswerable.errors,
            "abstained": summary.unanswerable.abstained,
            "abstain_rate": _round(summary.unanswerable.rate),
        },
        "latency_s": {k: _round(v, 3) for k, v in summary.latency_s.items()},
        "errors": summary.errors,
        "spend_usd": _round(spend, 4),
        "per_question_hidden": protected,
    }
    if not protected:
        report["error_list"] = [
            {"question_id": r.question_id, "status": r.reply.status, "error": r.reply.error}
            for r in results
            if r.failed
        ]
    return report


def judge_report(judging: JudgeRun, protected: bool) -> dict[str, Any]:
    """summary.json's `judge`: the judge's settings and aggregates. Names the questions it did
    not mark correct only for readable sets (like `error_list`)."""
    report: dict[str, Any] = {
        "status": judging.note,
        "model": judging.model,
        "prompt_name": JUDGE_PROMPT_NAME,
        "prompt_version": judging.prompt_version,
        "judgments_per_answer": JUDGE_JUDGMENTS,
        "thinking_budget": JUDGE_THINKING_BUDGET,
    }
    if judging.totals is not None:
        report |= judging.totals.to_json()
        report["spend_usd"] = _round(judging.spend_usd)
    if not protected and judging.totals is not None:
        report["not_correct"] = [
            {"question_id": r.question_id, "verdict": r.verdict, "verdicts": list(r.verdicts)}
            for r in judging.results
            if r.verdict != "correct"
        ]
    return report


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def question_row(result: QuestionResult) -> dict[str, object]:
    """One line of per_question.jsonl: what came back and how it scored (never the question
    text or the expected ids)."""
    reply = result.reply
    row: dict[str, object] = {
        "question_id": result.question_id,
        "kind": result.kind,
        "expected": result.scores.expected if result.scores is not None else 0,
        "status": reply.status,
        "latency_s": round(reply.latency_s, 3),
        "answer": reply.answer,
        "document_ids": list(reply.document_ids),
        "server_trace_id": reply.trace_id,
        "error": reply.error,
    }
    if result.scores is not None:
        s = result.scores
        row["metrics"] = {"hits": s.hits, "returned": s.returned, **s.values()}
    else:
        row["abstained"] = result.abstained
    return row


def write_outputs(
    report: dict[str, Any],
    results: Sequence[QuestionResult],
    folder: Path,
    per_question_folder: Path,
    judging: JudgeRun | None = None,
) -> None:
    """summary.json, summary.md and per_question.jsonl; and judge.jsonl beside summary.md for
    a readable set the judge ran on (never for a protected set: AGENTS.md, Eval split)."""
    judging = judging or JudgeRun("off (--no-judge)")
    (folder / SUMMARY_JSON).write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = render_markdown(report, judge_section(judging, report["per_question_hidden"]))
    (folder / SUMMARY_MD).write_text(md, encoding="utf-8")
    lines = [json.dumps(question_row(r), ensure_ascii=False) for r in results]
    (per_question_folder / PER_QUESTION).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if judging.totals is not None and not report["per_question_hidden"]:
        kinds = {r.question_id: r.kind for r in results}
        rows = [
            json.dumps(answer_judge.question_row(j, kinds[j.question_id]), ensure_ascii=False)
            for j in judging.results
        ]
        (folder / JUDGE_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")


def judge_section(judging: JudgeRun, protected: bool) -> list[str]:
    """The judge's lines of summary.md: aggregates, and for a readable set the questions it
    did not mark correct."""
    if judging.totals is None:
        return ["## Answer judge", "", f"The answer judge: {judging.note}."]
    not_correct = None if protected else [r for r in judging.results if r.verdict != "correct"]
    return answer_judge.render_markdown(
        judging.totals, judging.model, judging.prompt_version or "?", not_correct,
        judging.spend_usd,
    )  # fmt: skip


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(r: dict[str, Any], judge_lines: Sequence[str] = ()) -> str:
    counts = r["counts"]
    milestone = r["milestone"] or "none"
    lines = [
        f"# /ask eval: {r['set']} ({r['system']})",
        "",
        f"Set `{r['set']}`: {r['questions']} questions ({counts['single']} single-doc, "
        f"{counts['multi']} multi-doc, {counts['unanswerable']} unanswerable). "
        f"Milestone: {milestone}. Commit `{r['git_commit']}`, {r['timestamp']}.",
        f"Server {r['url']}, {r['concurrency']} requests at once, {r['timeout_s']:g} s limit "
        "per question.",
        "",
        "## Answerable questions (mean over the questions of each group)",
        "",
        "| group | n | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " | errors |",
        "|---|---|" + "---|" * len(METRICS) + "---|",
    ]
    for name in GROUPS:
        g = r["groups"][name]
        cells = [_num(None if g["means"] is None else g["means"][m]) for m in METRICS]
        lines.append(
            f"| {GROUP_LABELS[name]} | {g['n']} | " + " | ".join(cells) + f" | {g['errors']} |"
        )
    u = r["unanswerable"]
    lines += [
        "",
        "## Unanswerable questions",
        "",
        "| group | n | abstain rate | abstained | errors |",
        "|---|---|---|---|---|",
        f"| {GROUP_LABELS['unanswerable']} | {u['n']} | {_num(u['abstain_rate'])} | "
        f"{u['abstained']} | {u['errors']} |",
        "",
        'Unanswerable means "no ids returned": a question with no expected ids counts as '
        "correct when its response returns no ids, and it is left out of the means above.",
        "",
        "How to read: the ids are compared as sets (their order in the response is not used). "
        "precision = expected ids among the ids returned / ids returned (0 when none came "
        "back); recall = expected ids returned / expected ids; F1 = 2 x precision x recall / "
        "(precision + recall); hit = share of questions with at least one expected id "
        "returned; all_gold = share with every expected id returned; exact = share whose "
        "returned set equals the expected set. A failed request scores 0.",
        "",
        "## Latency (seconds, every request)",
        "",
        ", ".join(f"{k} {_num(v)}" for k, v in r["latency_s"].items()) + " (nearest rank).",
        "",
        "## Errors",
        "",
        f"{r['errors']} failed requests (no response, timeout, non-200 or a body off the "
        "contract); each counts as wrong.",
    ]
    if r["per_question_hidden"]:
        lines.append(
            f"Which questions failed is in the per-question file `{r['per_question_file']}`, "
            "which agents never read (AGENTS.md: Eval split); this summary holds aggregates only."
        )
    elif r["error_list"]:
        lines += ["", "| question | status | error |", "|---|---|---|"]
        for e in r["error_list"]:
            error = str(e["error"]).replace("|", "\\|")
            lines.append(f"| {e['question_id']} | {e['status'] or '-'} | {error} |")
    if judge_lines:
        lines += ["", *judge_lines]
    spend = "n/a" if r["spend_usd"] is None else f"${r['spend_usd']:.4f}"
    lines += [
        "",
        "## Run",
        "",
        f"- command: `{r['command']}`",
        f"- commit: `{r['git_commit']}`",
        f"- set: `{r['set']}`; milestone: {milestone}; system: {r['system']}",
        f"- spend during the run: {spend} (growth of .cache/spend/*.json in this checkout "
        "while the questions were asked; the answer judge's cost is in its own section)",
        f"- wall time: {r['seconds']} s",
        f"- per-question file: `{r['per_question_file']}`",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- history
def headline(r: dict[str, Any]) -> float | None:
    means = r["groups"][ANSWERABLE]["means"]
    return None if means is None else means[HEADLINE]


def history_metrics(r: dict[str, Any]) -> str:
    parts = []
    for name in GROUPS:
        means = r["groups"][name]["means"]
        shown = "n/a" if means is None else " ".join(f"{m}={means[m]:.3f}" for m in METRICS)
        parts.append(f"{name}(n={r['groups'][name]['n']}) {shown}")
    u, lat = r["unanswerable"], r["latency_s"]
    parts.append(f"unanswerable(n={u['n']}) abstain={_num(u['abstain_rate'])}")
    parts.append(f"errors={r['errors']} " + " ".join(f"{k}_s={_num(v)}" for k, v in lat.items()))
    return " | ".join(parts)


def judge_history(judging: JudgeRun) -> str:
    if judging.totals is None:
        return f"judge {judging.note}"
    return answer_judge.history_text(
        judging.totals, judging.model, judging.prompt_version or "?", judging.spend_usd
    )


def append_history(r: dict[str, Any], folder: Path, judging: JudgeRun | None = None) -> None:
    judging = judging or JudgeRun("off (--no-judge)")
    value = headline(r)
    variant = r["system"] if not r["milestone"] else f"{r['system']}@{r['milestone']}"
    new = not HISTORY.is_file()
    with HISTORY.open("a", encoding="utf-8", newline="") as handle:
        out = csv.writer(handle)
        if new:
            out.writerow(HISTORY_HEADER)
        out.writerow([
            datetime.now(UTC).isoformat(timespec="seconds"), r["git_commit"], HISTORY_KIND,
            r["set"], variant, r["questions"], HEADLINE,
            "n/a" if value is None else f"{value:.4f}",
            history_metrics(r) + " | " + judge_history(judging),
            "n/a" if r["spend_usd"] is None else f"{r['spend_usd']:.4f}", r["seconds"],
            display(folder),
        ])  # fmt: skip


def print_summary(r: dict[str, Any], folder: Path, judging: JudgeRun | None = None) -> None:
    print(f"\n/ask eval {r['set']} ({r['questions']} questions, milestone "
          f"{r['milestone'] or 'none'})")  # fmt: skip
    for name in GROUPS:
        g = r["groups"][name]
        shown = (
            "n/a" if g["means"] is None else "  ".join(f"{m}={g['means'][m]:.3f}" for m in METRICS)
        )
        print(f"  {name:<10} n={g['n']:<3} {shown}")
    u = r["unanswerable"]
    print(f"  {'unanswer.':<10} n={u['n']:<3} abstain_rate={_num(u['abstain_rate'])}")
    latency = "  ".join(f"{k}={_num(v)}s" for k, v in r["latency_s"].items())
    spend = "n/a" if r["spend_usd"] is None else f"${r['spend_usd']:.4f}"
    print(f"  errors={r['errors']}  {latency}  spend={spend}")
    if judging is not None:
        print(f"  {judge_history(judging)}")
    print(f"  summary: {display(folder / SUMMARY_MD)}")


# ---------------------------------------------------------------- CLI
class SetOnce(argparse.Action):
    """Refuses a second `--set`. Each `uv run eval-<set>` key fixes its set and extra args
    come after it, so a repeat would switch an allowlisted smoke or tune key to a protected
    set without asking the user (Codex review 2026-10-16 finding 2)."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest) is not None:
            parser.error(f"{option_string} given twice: each `uv run eval-<set>` key fixes its "
                         "set; run the key of the set you want")  # fmt: skip
        setattr(namespace, self.dest, values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--set", choices=RUN_SETS, default=None, action=SetOnce, help="question set (required)"
    )
    parser.add_argument("--url", default=EVAL_ASK_URL, help="the running server")
    parser.add_argument("--system", choices=SYSTEMS, default=BUILT_SYSTEMS[0])
    parser.add_argument(
        "--milestone", default=None, help="milestone name (required for heldout and all)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=EVAL_ASK_CONCURRENCY,
        help=f"requests in flight, 1-{EVAL_ASK_MAX_CONCURRENCY}",
    )
    parser.add_argument(
        "--timeout", type=float, default=EVAL_ASK_TIMEOUT_S, help="seconds per question"
    )
    parser.add_argument(
        "--no-judge",
        dest="judge",
        action="store_false",
        help=f"do not judge the answer texts (the judge, {JUDGE_MODEL}, costs money)",
    )
    args = parser.parse_args(argv)
    if args.system not in BUILT_SYSTEMS:
        parser.error(f"--system {args.system} is not built yet; use --system full")
    if args.set is None:
        parser.error(f"--set is required (one of {', '.join(RUN_SETS)})")
    args.milestone = (args.milestone or "").strip() or None
    if args.set in PROTECTED_SETS and args.milestone is None:
        parser.error(
            f"--set {args.set} needs --milestone <name>: it holds held-out questions, "
            "so run it only at a milestone the user names (AGENTS.md: Eval split)"
        )
    if not 1 <= args.concurrency <= EVAL_ASK_MAX_CONCURRENCY:
        parser.error(f"--concurrency must be 1-{EVAL_ASK_MAX_CONCURRENCY} (the server serves "
                     f"{EVAL_ASK_MAX_CONCURRENCY} at once, CONTRACT §4)")  # fmt: skip
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("--timeout must be a finite number of seconds > 0")
    try:
        parts = urlsplit(args.url)
        port = parts.port
    except ValueError:
        parser.error(f"--url {args.url!r} has an invalid port")
    if parts.scheme not in HTTP_DEFAULT_PORTS or not parts.hostname:
        parser.error(f"--url {args.url!r} must be http(s)://host[:port]")
    args.port = port or HTTP_DEFAULT_PORTS[parts.scheme]
    return args


def tracing_settings(port: int, environ: Mapping[str, str]) -> Settings:
    """Settings for the tracer: the optional LANGFUSE_* keys, and every key for the scrubber.

    The runner is a client: CORPUS_DIR and PORT belong to the server, so the repo root and
    the server's port stand in for them (Settings.from_env requires both)."""
    return Settings.from_env({**environ, "CORPUS_DIR": str(ROOT), "PORT": str(port)})


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    # httpx logs every request at INFO; the progress lines already count them.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    command = shlex.join(["python", "-m", "eval.run", *argv])
    settings = tracing_settings(args.port, os.environ)
    judge_chat: ChatModel | None = None
    if args.judge:
        try:
            judge_chat = answer_judge.build_judge_chat(settings)
        except ChatUnavailable as exc:
            logger.warning("the answer judge will not run: %s", exc)
    tracer = obs.init_tracing(settings)
    trace_input = {k: v for k, v in vars(args).items() if k != "port"}
    try:
        with tracer.trace(TRACE_NAME, input=trace_input) as trace:
            logger.info("trace id %s", trace.trace_id)
            return run(args, command, judge_chat=judge_chat)
    finally:
        tracer.shutdown(EVAL_TRACE_FLUSH_TIMEOUT_S)


if __name__ == "__main__":
    sys.exit(main())
