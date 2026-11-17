"""The end-to-end /ask eval (`eval.run`): its loader, set-based metrics, report and history.

Invented data only: never a real question, answer or expected id. The question file and the
sets are written to tmp_path, so no test opens the real ones (AGENTS.md: Eval split).
"""

import asyncio
import csv
import json
import re
import shlex
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest

from corpusqa.config import (
    EVAL_ASK_CONCURRENCY,
    EVAL_ASK_MAX_CONCURRENCY,
    EVAL_ASK_TIMEOUT_S,
    EVAL_ASK_URL,
    JUDGE_MODEL,
    MAX_RESPONSE_BYTES,
    TRACE_ID_HEADER,
)
from corpusqa.llm.errors import SpendCapReached
from corpusqa.llm.fakes import ChatRequest, FakeLLM, fake_reply
from corpusqa.llm.interfaces import ChatModel, ModelReply, ToolCall, Usage
from corpusqa.obs import Tracer
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import load_text_prompt
from eval import run as runner
from eval.ask_metrics import Reply, dedupe, judge, percentile, score_ids, summarize
from eval.questions import PROTECTED_SETS, RUN_SETS, EvalQuestion, load_run_questions
from eval.retrieval import HISTORY_HEADER, ROOT
from scripts.show_questions import load_questions


# ---------------------------------------------------------------- invented question files
def write_questions(tmp_path: Path) -> tuple[Path, Path]:
    """Five invented questions: q_0001 single, q_0002 and q_0003 multi, q_0004 unanswerable,
    q_0005 single. tune = 1-3, smoke = 1, the held-out stand-in = 4-5."""
    expected = {
        "q_0001": ["doc-a"],
        "q_0002": ["doc-b", "doc-c"],
        "q_0003": ["doc-d", "doc-e", "doc-f"],
        "q_0004": [],
        "q_0005": ["doc-g"],
    }
    rows = [
        {
            "question_id": qid,
            "question": f"Invented question {qid}?",
            "gold_answer": f"Invented answer {qid}.",
            "answer_facts": [f"Invented fact of {qid}."],
            "expected_doc_ids": ids,
        }
        for qid, ids in expected.items()
    ]
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    sets = tmp_path / "sets"
    sets.mkdir()
    for name, ids in {
        "tune": ["q_0001", "q_0002", "q_0003"],
        "smoke": ["q_0001"],
        "heldout": ["q_0004", "q_0005"],
    }.items():
        (sets / f"{name}.json").write_text(json.dumps({"question_ids": ids}), encoding="utf-8")
    return questions, sets


# ---------------------------------------------------------------- loader
def test_loader_reads_every_run_set_and_all_is_the_whole_file(tmp_path: Path) -> None:
    questions, sets = write_questions(tmp_path)

    def ids(name: str) -> list[str]:
        return [q.question_id for q in load_run_questions(name, questions, sets)]

    assert RUN_SETS == ("smoke", "tune", "heldout", "all")
    assert ids("smoke") == ["q_0001"]
    assert ids("tune") == ["q_0001", "q_0002", "q_0003"]
    assert ids("heldout") == ["q_0004", "q_0005"]
    assert ids("all") == ["q_0001", "q_0002", "q_0003", "q_0004", "q_0005"]
    assert {"heldout", "all"} == PROTECTED_SETS


def test_loader_groups_by_the_split_kind_rule(tmp_path: Path) -> None:
    questions, sets = write_questions(tmp_path)
    loaded = {q.question_id: q for q in load_run_questions("all", questions, sets)}
    # The answer judge's inputs ride along (never printed or written to a report).
    assert loaded["q_0002"].gold_answer == "Invented answer q_0002."
    assert loaded["q_0002"].facts == ("Invented fact of q_0002.",)
    assert {qid: q.kind for qid, q in loaded.items()} == {
        "q_0001": "single",
        "q_0002": "multi",
        "q_0003": "multi",
        "q_0004": "unanswerable",
        "q_0005": "single",
    }
    assert loaded["q_0002"].expected == frozenset({"doc-b", "doc-c"})
    assert loaded["q_0004"].expected == frozenset()
    assert loaded["q_0001"].text == "Invented question q_0001?"


def test_loader_refuses_unknown_sets_and_ids_the_file_lacks(tmp_path: Path) -> None:
    questions, sets = write_questions(tmp_path)
    with pytest.raises(ValueError, match="unknown set"):
        load_run_questions("dev", questions, sets)
    (sets / "tune.json").write_text(json.dumps({"question_ids": ["q_9999"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        load_run_questions("tune", questions, sets)


def test_the_agents_question_reader_still_refuses_heldout_and_all(tmp_path: Path) -> None:
    questions, sets = write_questions(tmp_path)
    assert [q["question_id"] for q in load_questions("smoke", questions, sets)] == ["q_0001"]
    for name in ("heldout", "all"):
        with pytest.raises(ValueError, match="Refused"):
            load_questions(name, questions, sets)


# ---------------------------------------------------------------- per-question metrics
def reply(ids: tuple[str, ...] = (), *, error: str | None = None, latency: float = 1.0) -> Reply:
    status = None if error else 200
    return Reply(status, latency, None if error else "An invented answer.", ids, None, error)


def question(qid: str, expected: set[str]) -> EvalQuestion:
    n = len(expected)
    kind = "unanswerable" if n == 0 else "single" if n == 1 else "multi"
    return EvalQuestion(qid, f"Invented question {qid}?", frozenset(expected), kind)


def test_dedupe_keeps_the_first_position_of_each_id() -> None:
    assert dedupe(["b", "a", "b", "c", "a"]) == ("b", "a", "c")
    assert dedupe([]) == ()


def test_set_metrics_by_hand() -> None:
    both = score_ids(["b", "a"], frozenset({"a", "b"}))
    assert (both.precision, both.recall, both.f1) == (1.0, 1.0, 1.0)
    assert both.hit and both.all_gold and both.exact

    half = score_ids(["a", "x"], frozenset({"a", "b"}))  # 1 hit of 2 returned, of 2 expected
    assert (half.hits, half.returned, half.expected) == (1, 2, 2)
    assert (half.precision, half.recall, half.f1) == (0.5, 0.5, 0.5)
    assert half.hit and not half.all_gold and not half.exact

    narrow = score_ids(["a"], frozenset({"a", "b", "c"}))  # P 1, R 1/3: F1 = (2/3) / (4/3)
    assert narrow.precision == 1.0 and narrow.recall == pytest.approx(1 / 3)
    assert narrow.f1 == pytest.approx(0.5)

    wide = score_ids(["a", "b", "c", "d"], frozenset({"a"}))  # P 1/4, R 1: F1 = 0.5 / 1.25
    assert wide.precision == 0.25 and wide.recall == 1.0 and wide.f1 == pytest.approx(0.4)
    assert wide.all_gold and not wide.exact


def test_nothing_returned_scores_zero_and_order_does_not_matter() -> None:
    empty = score_ids([], frozenset({"a"}))
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)
    assert not (empty.hit or empty.all_gold or empty.exact)
    assert score_ids(["x", "y", "a"], frozenset({"a", "b"})) == score_ids(
        ["a", "y", "x"], frozenset({"a", "b"})
    )
    miss = score_ids(["x"], frozenset({"a"}))
    assert (miss.precision, miss.recall, miss.f1, miss.hit) == (0.0, 0.0, 0.0, False)


def test_duplicate_ids_count_once() -> None:
    repeated = score_ids(["a", "a", "a"], frozenset({"a", "b"}))
    assert (repeated.returned, repeated.hits, repeated.precision) == (1, 1, 1.0)
    assert repeated.recall == 0.5


def test_unanswerable_is_correct_only_when_no_ids_come_back() -> None:
    none_expected = question("q_0004", set())
    assert judge(none_expected, reply()).abstained is True
    assert judge(none_expected, reply(("doc-x",))).abstained is False
    assert judge(none_expected, reply(error="HTTP 500")).abstained is False  # a failure is wrong
    assert judge(none_expected, reply()).scores is None
    with pytest.raises(ValueError, match="unanswerable"):
        score_ids([], frozenset())


def test_a_failed_reply_scores_zero_even_with_ids() -> None:
    result = judge(question("q_0001", {"doc-a"}), reply(("doc-a",), error="timeout"))
    assert result.failed and result.abstained is None
    assert result.scores is not None and result.scores.recall == 0.0 and not result.scores.hit


# ---------------------------------------------------------------- aggregates
def test_summary_means_per_group_abstain_rate_latency_and_errors() -> None:
    results = [
        judge(question("q_0001", {"a"}), reply(("a",), latency=1.0)),  # single: all 1
        judge(question("q_0005", {"g"}), reply(error="HTTP 503", latency=3.0)),  # single: all 0
        judge(question("q_0002", {"b", "c"}), reply(("b", "x"), latency=2.0)),  # multi: 0.5s
        judge(question("q_0004", set()), reply(latency=4.0)),  # abstained
        judge(question("q_0006", set()), reply(("y",), latency=5.0)),  # cited: wrong
    ]
    s = summarize(results, (50, 95))
    assert s.questions == 5 and s.errors == 1
    assert s.counts == {"single": 2, "multi": 1, "unanswerable": 2}
    single = s.groups["single"]
    assert (single.n, single.errors) == (2, 1) and single.means is not None
    assert single.means == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "hit": 0.5,
                            "all_gold": 0.5, "exact": 0.5}  # fmt: skip
    multi = s.groups["multi"].means
    assert multi is not None and multi["precision"] == 0.5 and multi["recall"] == 0.5
    assert multi["all_gold"] == 0.0 and multi["hit"] == 1.0
    answerable = s.groups["answerable"]
    assert answerable.n == 3 and answerable.means is not None
    assert answerable.means["recall"] == pytest.approx((1 + 0 + 0.5) / 3)
    assert answerable.means["exact"] == pytest.approx(1 / 3)
    assert (s.unanswerable.n, s.unanswerable.abstained, s.unanswerable.rate) == (2, 1, 0.5)
    assert s.latency_s == {"p50": 3.0, "p95": 5.0, "max": 5.0}


def test_empty_groups_have_no_means_instead_of_zeros() -> None:
    s = summarize([judge(question("q_0001", {"a"}), reply(("a",)))], (50,))
    assert s.groups["multi"].means is None and s.groups["multi"].n == 0
    assert s.unanswerable.rate is None and s.unanswerable.n == 0
    assert summarize([], (50, 95)).latency_s == {"p50": None, "p95": None, "max": None}


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(20, 0, -1)]  # 1..20, unsorted
    assert percentile(values, 50) == 10.0 and percentile(values, 95) == 19.0
    assert percentile(values, 100) == 20.0 and percentile(values, 1) == 1.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0 and percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None


# ---------------------------------------------------------------- one reply
def test_read_reply_accepts_the_contract_shape_and_dedupes_ids() -> None:
    body = json.dumps({"answer": ANSWER, "document_ids": ["doc-b", "doc-a", "doc-b"]}).encode()
    got = runner.read_reply(200, body, 0.5, trace_id="t1")
    assert got.error is None and got.status == 200 and got.latency_s == 0.5
    assert got.document_ids == ("doc-b", "doc-a") and got.trace_id == "t1"
    assert got.answer == ANSWER
    assert runner.read_reply(200, body, 0.1, trace_id="").trace_id is None  # an empty header


def test_the_trace_id_comes_from_the_header_never_from_the_body() -> None:
    # User decision 2026-10-17: the body is exactly CONTRACT §3's two fields and the server's
    # trace id is in the X-Trace-Id header; a body field of that name is ignored like any extra.
    body = json.dumps({"answer": ANSWER, "document_ids": [], "trace_id": "from-the-body"}).encode()
    assert runner.read_reply(200, body, 0.1).trace_id is None
    assert runner.read_reply(200, body, 0.1, trace_id="h1").trace_id == "h1"


def test_a_failed_reply_keeps_the_header_trace_id() -> None:
    # The header does not depend on the body, so an off-contract reply can still be opened.
    got = runner.read_reply(200, b'{"answer": "  ", "document_ids": []}', 0.1, trace_id="h1")
    assert (got.error, got.trace_id, got.answer) == ("empty answer", "h1", None)


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (404, b"{}", "HTTP 404"),
        (200, b"not json", "body is not JSON"),
        (200, b'{"answer": "a"}', "body is off the contract: document_ids"),
        (200, b'{"answer": "a", "document_ids": ["x", 3]}', "body is off the contract"),
        (200, b'{"answer": 5, "document_ids": []}', "body is off the contract: answer"),
        (200, b'["a"]', "body is off the contract"),
        (200, b'{"answer": "  ", "document_ids": []}', "empty answer"),
        (200, b" " * (MAX_RESPONSE_BYTES + 1), "over the"),
    ],
)
def test_read_reply_fails_anything_off_the_contract(status: int, body: bytes, error: str) -> None:
    got = runner.read_reply(status, body, 0.1)
    assert got.error is not None and error in got.error
    assert got.status == status and got.document_ids == () and got.answer is None


# ---------------------------------------------------------------- runs against a fake server
ANSWER = "An invented answer."
Step = httpx.Response | Exception | Callable[[], Awaitable[httpx.Response]]


def ok(ids: list[object], trace_id: str | None = None) -> httpx.Response:
    """A contract reply; the server's trace id, if given, in the X-Trace-Id header."""
    headers = {} if trace_id is None else {TRACE_ID_HEADER: trace_id}
    return httpx.Response(200, json={"answer": ANSWER, "document_ids": ids}, headers=headers)


def fake_server(script: dict[str, Step], health: int = 200) -> httpx.MockTransport:
    """Answers each invented question as scripted, keyed by the id inside its text."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(health, json={"status": "ok" if health == 200 else "starting"})
        assert (request.method, request.url.path) == ("POST", "/ask")
        text = json.loads(request.content)["question"]
        step = script[text.removeprefix("Invented question ").removesuffix("?")]
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return await step()
        return step

    return httpx.MockTransport(handler)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Invented questions; results, history and spend ledgers under tmp_path."""
    questions, sets = write_questions(tmp_path)
    monkeypatch.setattr(runner, "QUESTIONS_FILE", questions)
    monkeypatch.setattr(runner, "SETS_DIR", sets)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "HISTORY", tmp_path / "history.csv")
    monkeypatch.setattr(runner, "SPEND_DIR", tmp_path / "spend")
    return tmp_path


def run_set(
    argv: list[str],
    script: dict[str, Step],
    health: int = 200,
    tracer: Tracer | None = None,
    judge_chat: ChatModel | None = None,
) -> int:
    args = runner.parse_args(argv)
    with (tracer or RecordingTracer()).trace(runner.TRACE_NAME):
        command = "python -m eval.run " + " ".join(argv)
        return runner.run(args, command, fake_server(script, health), judge_chat=judge_chat)


def only_child(folder: Path) -> Path:
    [child] = list(folder.iterdir())
    return child


def history_rows(workdir: Path) -> list[list[str]]:
    with (workdir / "history.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_a_run_writes_the_summary_numbers_files_and_one_history_row(workdir: Path) -> None:
    script: dict[str, Step] = {
        "q_0001": ok(["doc-a", "doc-a", "doc-z"], trace_id="t" * 32),  # P 1/2, R 1
        "q_0002": httpx.Response(500, json={"detail": "boom"}),  # failed: 0
        "q_0003": ok(["doc-f", "doc-e", "doc-d"]),  # exact
    }
    assert run_set(["--set", "tune"], script) == 0
    folder = only_child(workdir / "results" / "tune" / "ask")
    assert sorted(p.name for p in folder.iterdir()) == [
        "per_question.jsonl", "summary.json", "summary.md",
    ]  # fmt: skip
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert (s["set"], s["milestone"], s["system"], s["questions"]) == ("tune", None, "full", 3)
    assert s["counts"] == {"single": 1, "multi": 2, "unanswerable": 0}
    single = {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "hit": 1.0, "all_gold": 1.0,
              "exact": 0.0}  # fmt: skip
    assert s["groups"]["single"]["means"] == single
    multi = s["groups"]["multi"]
    assert (multi["n"], multi["errors"]) == (2, 1)
    assert multi["means"]["recall"] == 0.5 and multi["means"]["exact"] == 0.5
    assert s["groups"]["answerable"]["means"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["unanswerable"] == {"n": 0, "errors": 0, "abstained": 0, "abstain_rate": None}
    assert s["errors"] == 1
    assert s["error_list"] == [{"question_id": "q_0002", "status": 500, "error": "HTTP 500"}]
    assert s["spend_usd"] == 0.0 and s["per_question_hidden"] is False
    assert set(s["latency_s"]) == {"p50", "p95", "max"}
    assert s["command"] == "python -m eval.run --set tune"

    md = (folder / "summary.md").read_text(encoding="utf-8")
    assert (
        "| single-doc (1 expected id) | 1 | 0.500 | 1.000 | 0.667 | 1.000 | 1.000 | 0.000 | 0 |"
        in md
    )
    assert "| all answerable | 3 |" in md and "| unanswerable (0 expected ids) | 0 | n/a |" in md
    assert "| q_0002 | 500 | HTTP 500 |" in md
    assert 'Unanswerable means "no ids returned"' in md
    assert "`python -m eval.run --set tune`" in md and "milestone: none" in md
    assert "MRR" not in md and "nDCG" not in md and "ranking" not in md.lower()

    rows = [
        json.loads(line) for line in (folder / "per_question.jsonl").read_text("utf-8").splitlines()
    ]
    assert [r["question_id"] for r in rows] == ["q_0001", "q_0002", "q_0003"]
    assert rows[0]["document_ids"] == ["doc-a", "doc-z"] and rows[0]["server_trace_id"] == "t" * 32
    assert rows[0]["metrics"]["hits"] == 1 and rows[0]["metrics"]["returned"] == 2
    assert rows[1]["status"] == 500 and rows[1]["error"] == "HTTP 500"
    assert rows[1]["metrics"]["recall"] == 0.0 and rows[1]["expected"] == 2
    text = (folder / "per_question.jsonl").read_text(encoding="utf-8")
    assert "Invented question" not in text and "doc-b" not in text  # never question or gold

    header, row = history_rows(workdir)
    assert header == list(HISTORY_HEADER)
    record = dict(zip(header, row, strict=True))
    assert (record["kind"], record["set"], record["variant"]) == ("ask", "tune", "full")
    assert (record["questions"], record["headline"], record["value"]) == ("3", "recall", "0.6667")
    assert record["metrics"].startswith("single(n=1) precision=0.500 recall=1.000")
    assert "unanswerable(n=0) abstain=n/a" in record["metrics"] and "errors=1" in record["metrics"]
    assert record["cost_usd"] == "0.0000" and record["report"].endswith(folder.name)


def test_protected_sets_keep_per_question_results_out_of_every_readable_output(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script: dict[str, Step] = {
        "q_0001": ok(["doc-a"]),
        "q_0002": ok(["doc-b"]),
        "q_0003": ok([]),
        "q_0004": ok([]),  # unanswerable: abstained
        "q_0005": httpx.ConnectError("refused"),
    }
    assert run_set(["--set", "all", "--milestone", "m1"], script) == 0
    readable = only_child(workdir / "results" / "all" / "ask")
    assert sorted(p.name for p in readable.iterdir()) == ["summary.json", "summary.md"]
    guarded = only_child(workdir / "results" / "heldout" / "ask")
    assert [p.name for p in guarded.iterdir()] == ["per_question.jsonl"]
    rows = (guarded / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5 and "ConnectError" in rows[4]

    md = (readable / "summary.md").read_text(encoding="utf-8")
    summary = (readable / "summary.json").read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    for text in (md, summary, printed):
        assert "q_000" not in text and "ConnectError" not in text
    s = json.loads(summary)
    assert "error_list" not in s and s["per_question_hidden"] is True
    assert s["errors"] == 1 and s["milestone"] == "m1"
    assert s["unanswerable"]["abstain_rate"] == 1.0
    assert s["per_question_file"].endswith(f"heldout/ask/{guarded.name}/per_question.jsonl")
    assert s["per_question_file"] in md
    record = dict(zip(*history_rows(workdir), strict=True))
    assert (record["set"], record["variant"], record["questions"]) == ("all", "full@m1", "5")


def test_the_heldout_set_writes_everything_in_its_own_guarded_folder(workdir: Path) -> None:
    assert (
        run_set(
            ["--set", "heldout", "--milestone", "m1"], {"q_0004": ok([]), "q_0005": ok(["doc-g"])}
        )
        == 0
    )
    folder = only_child(workdir / "results" / "heldout" / "ask")
    assert sorted(p.name for p in folder.iterdir()) == [
        "per_question.jsonl", "summary.json", "summary.md",
    ]  # fmt: skip
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert s["groups"]["answerable"]["means"]["exact"] == 1.0 and "error_list" not in s


def test_each_question_is_a_span_under_the_eval_trace_without_its_text(workdir: Path) -> None:
    tracer = RecordingTracer()
    script: dict[str, Step] = {
        "q_0001": ok(["doc-a"], trace_id="s" * 32),
        "q_0002": httpx.Response(503),
        "q_0003": ok(["doc-d"]),
    }
    assert run_set(["--set", "tune"], script, tracer=tracer) == 0
    [trace] = tracer.traces(runner.TRACE_NAME)
    spans = {s.metadata["question_id"]: s for s in tracer.children(trace)}
    assert sorted(spans) == ["q_0001", "q_0002", "q_0003"]
    assert all(s.name == "question" and s.ended for s in spans.values())
    first = spans["q_0001"].metadata
    assert (first["status"], first["recall"], first["hits"]) == (200, 1.0, 1)
    assert first["server_trace_id"] == "s" * 32 and isinstance(first["latency_s"], float)
    assert spans["q_0002"].level == "ERROR" and spans["q_0002"].metadata["status"] == 503
    assert (
        spans["q_0003"].metadata["precision"] == 1.0 and spans["q_0003"].metadata["recall"] == 1 / 3
    )
    assert isinstance(trace.output, dict) and trace.output["groups"]["answerable"]["n"] == 3
    recorded = json.dumps([[o.input, o.output, o.metadata] for o in tracer.observations])
    assert "Invented question" not in recorded and ANSWER not in recorded


def test_at_most_concurrency_requests_are_in_flight(workdir: Path) -> None:
    in_flight, peak = 0, 0

    async def slow() -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return ok([])

    script: dict[str, Step] = {f"q_000{n}": slow for n in range(1, 6)}
    assert run_set(["--set", "all", "--milestone", "m1", "--concurrency", "2"], script) == 0
    assert peak == 2


def test_a_question_over_the_time_limit_fails_and_the_rest_go_on(workdir: Path) -> None:
    async def stuck() -> httpx.Response:
        await asyncio.sleep(5)
        return ok(["doc-a"])

    script: dict[str, Step] = {"q_0001": stuck, "q_0002": ok(["doc-b"]), "q_0003": ok([])}
    assert run_set(["--set", "tune", "--timeout", "0.05"], script) == 0
    folder = only_child(workdir / "results" / "tune" / "ask")
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    [failure] = s["error_list"]
    assert failure["question_id"] == "q_0001" and failure["error"].startswith("timeout")
    assert s["groups"]["single"]["means"]["recall"] == 0.0
    assert s["groups"]["multi"]["means"]["recall"] == 0.25  # q_0002 1/2, q_0003 0


def test_a_server_that_is_not_ready_stops_the_run_before_any_report(workdir: Path) -> None:
    assert run_set(["--set", "tune"], {}, health=503) == runner.EXIT_NOT_READY
    assert not (workdir / "results").exists() and not (workdir / "history.csv").exists()


def test_spend_is_the_growth_of_every_ledger_and_na_when_one_is_unreadable(workdir: Path) -> None:
    spend = workdir / "spend"
    spend.mkdir()
    (spend / "embeddings.json").write_text(
        json.dumps({"calls": 1, "tokens": 9, "cost_usd": 0.1}), encoding="utf-8"
    )

    async def spending() -> httpx.Response:  # the server records a call while answering
        (spend / "llm.json").write_text(
            json.dumps({"calls": 1, "tokens": 9, "cost_usd": 0.25}), encoding="utf-8"
        )
        return ok(["doc-a"])

    assert runner.spend_total(spend) == pytest.approx(0.1)
    assert runner.spend_total(workdir / "missing") == 0.0
    assert run_set(["--set", "smoke"], {"q_0001": spending}) == 0
    folder = only_child(workdir / "results" / "smoke" / "ask")
    assert json.loads((folder / "summary.json").read_text(encoding="utf-8"))["spend_usd"] == 0.25
    (spend / "llm.json").write_text("{broken", encoding="utf-8")
    assert runner.spend_total(spend) is None


def test_a_second_run_in_the_same_second_gets_its_own_folder(tmp_path: Path) -> None:
    assert runner.fresh_folder(tmp_path / "x", "S").name == "S"
    assert runner.fresh_folder(tmp_path / "x", "S").name == "S-2"


# ---------------------------------------------------------------- CLI
def test_cli_defaults_come_from_config() -> None:
    args = runner.parse_args(["--set", "smoke"])
    assert (args.concurrency, args.timeout) == (EVAL_ASK_CONCURRENCY, EVAL_ASK_TIMEOUT_S)
    assert (args.concurrency, args.timeout, EVAL_ASK_MAX_CONCURRENCY) == (2, 180.0, 4)
    assert (args.system, args.url, args.milestone) == ("full", EVAL_ASK_URL, None)
    assert args.port == 8080
    assert runner.parse_args(["--set", "tune", "--url", "https://example.test"]).port == 443


@pytest.mark.parametrize("name", ["heldout", "all"])
def test_held_out_sets_need_a_milestone(name: str) -> None:
    for argv in (["--set", name], ["--set", name, "--milestone", "  "]):
        with pytest.raises(SystemExit):
            runner.parse_args(argv)
    assert runner.parse_args(["--set", name, "--milestone", " m1 "]).milestone == "m1"
    assert runner.parse_args(["--set", "tune", "--milestone", "m1"]).milestone == "m1"


@pytest.mark.parametrize(
    "bad",
    [
        ["--concurrency", "0"],
        ["--concurrency", "5"],
        ["--timeout", "0"],
        ["--timeout", "-1"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--url", "localhost:8080"],
        ["--url", "ftp://example.test"],
        ["--url", "http://example.test:port"],
    ],
)
def test_cli_rejects_values_outside_the_contract(bad: list[str]) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--set", "smoke", *bad])
    assert runner.parse_args(["--set", "smoke", "--concurrency", "4"]).concurrency == 4


def test_baseline_exits_with_a_clear_message(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["--system", "baseline", "--set", "tune"], ["--system", "baseline"]):
        with pytest.raises(SystemExit) as exited:
            runner.parse_args(argv)
        assert exited.value.code == 2
        assert "--system baseline is not built yet" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        runner.parse_args([])
    assert "--set is required" in capsys.readouterr().err


def test_tracing_settings_need_no_server_variables() -> None:
    environ = {"LANGFUSE_PUBLIC_KEY": "pk-invented", "LANGFUSE_SECRET_KEY": "sk-invented",
               "CORPUS_DIR": "/no/such/dir"}  # fmt: skip
    settings = runner.tracing_settings(8080, environ)
    assert settings.langfuse_public_key == "pk-invented" and settings.port == 8080
    assert runner.tracing_settings(443, {}).langfuse_public_key is None


def test_every_eval_key_runs_the_runner_on_its_own_set() -> None:
    commands = json.loads((ROOT / "commands.json").read_text(encoding="utf-8"))["commands"]
    for name in ("smoke", "tune", "heldout", "all"):
        line = commands[f"eval-{name}"]["run"][-1]
        assert f"python -m eval.run --set {name} --system full" in line
    assert "--system baseline" in commands["eval-baseline"]["run"][-1]


# ---------------------------------------------------------------- Codex review 2026-10-16
def test_protected_runs_trace_no_per_question_results(workdir: Path) -> None:
    # Finding 1: question spans of a protected set named the question, its server trace and
    # its scores, so the trace held per-question held-out results outside the guarded folder.
    tracer = RecordingTracer()
    script: dict[str, Step] = {f"q_000{n}": ok(["doc-a"], trace_id="s" * 32) for n in range(1, 6)}
    script["q_0005"] = httpx.Response(500)
    assert run_set(["--set", "all", "--milestone", "m1"], script, tracer=tracer) == 0
    [trace] = tracer.traces(runner.TRACE_NAME)
    spans = tracer.children(trace)
    assert len(spans) == 5 and all(s.name == "question" for s in spans)
    assert sorted(s.metadata["status"] for s in spans) == [200, 200, 200, 200, 500]
    for span in spans:
        assert not {"question_id", "server_trace_id", "recall", "precision", "hits"} & set(
            span.metadata
        )
    recorded = json.dumps(
        [[o.input, o.output, o.metadata, o.status_message] for o in tracer.observations]
    )
    assert "q_000" not in recorded and "s" * 32 not in recorded
    assert isinstance(trace.output, dict) and trace.output["groups"]["answerable"]["n"] == 4


@pytest.mark.parametrize("key", ["eval-smoke", "eval-tune"])
@pytest.mark.parametrize("override", [["--set", "all"], ["--set=all"], ["--set", "tune"]])
def test_a_key_cannot_switch_to_another_set(key: str, override: list[str]) -> None:
    # Finding 2: extra args go after the key's own `--set`, and argparse kept the last one,
    # so an allowlisted smoke/tune key could run a protected set without asking the user.
    commands = json.loads((ROOT / "commands.json").read_text(encoding="utf-8"))["commands"]
    argv = shlex.split(commands[key]["run"][-1].split("python -m eval.run", 1)[1])
    with pytest.raises(SystemExit):
        runner.parse_args([*argv, *override, "--milestone", "m1"])


def test_deeply_nested_json_is_a_failed_reply_and_the_run_goes_on(workdir: Path) -> None:
    # Finding 3: json.loads raises RecursionError (not ValueError) on deep nesting, which
    # escaped gather and lost the whole run.
    nested = b"[" * 100_000 + b"]" * 100_000
    assert len(nested) < MAX_RESPONSE_BYTES
    assert runner.read_reply(200, nested, 0.1).error == "body is not JSON"
    script: dict[str, Step] = {
        "q_0001": httpx.Response(200, content=nested),
        "q_0002": ok(["doc-b", "doc-c"]),
        "q_0003": ok([]),
    }
    assert run_set(["--set", "tune"], script) == 0
    folder = only_child(workdir / "results" / "tune" / "ask")
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert s["error_list"] == [
        {"question_id": "q_0001", "status": 200, "error": "body is not JSON"}
    ]
    assert s["groups"]["multi"]["means"]["exact"] == 0.5


# ---------------------------------------------------------------- the answer judge
STATUS_OF = {"correct": "right", "partly": "partly", "wrong": "missing"}


def judge_model(verdicts: dict[str, list[str]], calls: int = 20) -> FakeLLM:
    """A judge that answers each invented question's judgments with its verdicts in turn
    (each invented question has one fact)."""
    seen: Counter[str] = Counter()

    def answer(request: ChatRequest) -> ModelReply:
        found = re.search(r"Invented question (q_\d+)\?", request.messages[-1].content)
        assert found is not None
        qid = found.group(1)
        verdict = verdicts[qid][seen[qid]]
        seen[qid] += 1
        arguments = {
            "facts": [{"fact": 1, "status": STATUS_OF[verdict], "evidence": "invented words"}],
            "contradictions": [],
            "says_not_found": False,
            "verdict": verdict,
            "reason": "Invented reason.",
        }
        return fake_reply(
            tool_calls=[ToolCall("c1", "verdict", arguments)], usage=Usage(100, 20, 0, 0, 0.002)
        )

    return FakeLLM([answer] * calls)


def test_the_judge_reports_aggregates_and_per_question_rows_for_a_tune_run(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tracer = RecordingTracer()
    script: dict[str, Step] = {
        "q_0001": ok(["doc-a"]),
        "q_0002": httpx.Response(500),  # no answer: judged by code as wrong
        "q_0003": ok(["doc-d"]),
    }
    judge = judge_model({"q_0001": ["correct", "correct"], "q_0003": ["correct", "partly",
                                                                      "partly"]})  # fmt: skip
    assert run_set(["--set", "tune"], script, tracer=tracer, judge_chat=judge) == 0
    folder = only_child(workdir / "results" / "tune" / "ask")
    assert sorted(p.name for p in folder.iterdir()) == [
        "judge.jsonl", "per_question.jsonl", "summary.json", "summary.md",
    ]  # fmt: skip
    j = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["judge"]
    assert (j["status"], j["model"], j["prompt_name"]) == ("done", JUDGE_MODEL, "judge")
    assert j["prompt_version"] == load_text_prompt("judge").version
    assert j["verdicts"] == {"correct": 1, "partly": 1, "wrong": 1, "unjudged": 0}
    assert j["verdict_shares"]["correct"] == round(1 / 3, 4) and j["no_answer"] == 1
    assert j["fact_counts"] == {"right": 1, "partly": 1, "missing": 1, "wrong": 0}
    assert j["agreement"] == {"two_judgments": 2, "first_two_agree": 1, "rate": 0.5,
                              "third_judgments": 1, "no_majority": 0}  # fmt: skip
    assert (j["calls"], j["cost_usd"]) == (5, 0.01) and len(judge.requests) == 5
    assert [n["question_id"] for n in j["not_correct"]] == ["q_0002", "q_0003"]

    lines = (folder / "judge.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["question_id"] for r in rows] == ["q_0001", "q_0002", "q_0003"]
    assert rows[1]["outcome"] == "no_answer" and rows[1]["verdict"] == "wrong"
    assert rows[2]["verdicts"] == ["correct", "partly", "partly"]
    assert rows[2]["facts"] == [{"fact": 1, "status": "partly", "evidence": "invented words"}]
    text = "\n".join(lines)
    assert "Invented question" not in text and "Invented answer" not in text
    assert "Invented fact" not in text

    md = (folder / "summary.md").read_text(encoding="utf-8")
    assert "## Answer judge" in md and "| q_0003 | partly | correct, partly, partly |" in md
    record = dict(zip(*history_rows(workdir), strict=True))
    assert (
        f"judge({JUDGE_MODEL} v{j['prompt_version']}) correct=0.333 partly=0.333"
        in (record["metrics"])
    )
    assert "judge_cost=$0.0100" in record["metrics"]
    assert "judge(" in capsys.readouterr().out
    spans = tracer.named("judge")
    assert sorted(s.input["question_id"] for s in spans) == ["q_0001", "q_0003"]  # type: ignore[index]
    [set_span] = tracer.named("judge_all")
    [trace] = tracer.traces(runner.TRACE_NAME)
    assert set_span.parent_id == trace.id and all(s.parent_id == set_span.id for s in spans)


def test_protected_sets_get_judge_aggregates_only(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tracer = RecordingTracer()
    script: dict[str, Step] = {f"q_000{n}": ok([]) for n in range(1, 6)}
    judge = judge_model({f"q_000{n}": ["correct", "correct"] for n in range(1, 6)})
    argv = ["--set", "all", "--milestone", "m1"]
    assert run_set(argv, script, tracer=tracer, judge_chat=judge) == 0
    readable = only_child(workdir / "results" / "all" / "ask")
    assert sorted(p.name for p in readable.iterdir()) == ["summary.json", "summary.md"]
    guarded = only_child(workdir / "results" / "heldout" / "ask")
    assert [p.name for p in guarded.iterdir()] == ["per_question.jsonl"]
    summary = (readable / "summary.json").read_text(encoding="utf-8")
    j = json.loads(summary)["judge"]
    assert j["status"] == "done" and j["verdicts"]["correct"] == 5 and "not_correct" not in j
    md = (readable / "summary.md").read_text(encoding="utf-8")
    assert "## Answer judge" in md and "aggregates only" in md
    printed = capsys.readouterr().out
    history = (workdir / "history.csv").read_text(encoding="utf-8")
    for text in (md, summary, printed, history):
        assert "q_000" not in text and "Invented" not in text
    assert tracer.named("judge") == [] and tracer.named("chat") == []
    [set_span] = tracer.named("judge_all")
    assert isinstance(set_span.output, dict) and set_span.output["verdicts"]["correct"] == 5
    recorded = json.dumps([[o.input, o.output, o.metadata] for o in tracer.observations])
    assert "q_000" not in recorded and "Invented" not in recorded


def test_no_judge_and_no_judge_model_are_reported_not_hidden(workdir: Path) -> None:
    unused = FakeLLM([])
    assert run_set(["--set", "smoke", "--no-judge"], {"q_0001": ok([])}, judge_chat=unused) == 0
    assert unused.requests == []
    assert run_set(["--set", "smoke"], {"q_0001": ok([])}) == 0
    statuses = []
    for folder in sorted((workdir / "results" / "smoke" / "ask").iterdir()):
        assert not (folder / "judge.jsonl").exists()
        s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        statuses.append(s["judge"]["status"])
        assert "## Answer judge" in (folder / "summary.md").read_text(encoding="utf-8")
    assert statuses == ["off (--no-judge)", "not run: no judge model (GEMINI_API_KEY is not set)"]
    rows = history_rows(workdir)[1:]
    assert rows[0][8].endswith("| judge off (--no-judge)")


def test_a_judge_that_stops_still_leaves_the_ask_report(workdir: Path) -> None:
    capped = FakeLLM([SpendCapReached("spend cap $10.00 reached")] * 4)
    assert run_set(["--set", "smoke"], {"q_0001": ok(["doc-a"])}, judge_chat=capped) == 0
    folder = only_child(workdir / "results" / "smoke" / "ask")
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert s["judge"]["status"].startswith("not run: the judge stopped (SpendCapReached: spend")
    assert s["groups"]["single"]["means"]["recall"] == 1.0


def test_a_protected_run_names_only_the_type_of_a_judge_failure(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: runner.JudgeProgress, result: object) -> None:
        raise RuntimeError("detail about q_0004")

    monkeypatch.setattr(runner.JudgeProgress, "add", broken)
    tracer = RecordingTracer()
    judge = judge_model({"q_0004": ["correct"] * 2, "q_0005": ["correct"] * 2})
    argv = ["--set", "heldout", "--milestone", "m1"]
    script: dict[str, Step] = {"q_0004": ok([]), "q_0005": ok([])}
    assert run_set(argv, script, tracer=tracer, judge_chat=judge) == 0
    folder = only_child(workdir / "results" / "heldout" / "ask")
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert s["judge"]["status"] == "not run: the judge stopped (JudgeFailed: RuntimeError)"
    recorded = json.dumps([[o.output, o.metadata, o.status_message] for o in tracer.observations])
    assert "q_0004" not in recorded and "detail" not in recorded


def test_the_judge_spend_is_the_growth_of_its_ledger(workdir: Path) -> None:
    # Codex review 2026-10-17, finding 5: a call billed but unusable added no cost to the
    # replies' sum, so the reported judge cost could be short. The ledger's growth is not.
    spend = workdir / "spend"
    spend.mkdir()
    replies = judge_model({"q_0001": ["correct", "correct"]})

    async def billed(request: ChatRequest) -> ModelReply:
        (spend / "judge.json").write_text(
            json.dumps({"calls": 1, "tokens": 9, "cost_usd": 0.05}), encoding="utf-8"
        )
        return await replies.complete(
            request.messages, request.tools, tool_choice=request.tool_choice,
            max_output_tokens=request.max_output_tokens, thinking_budget=request.thinking_budget,
            timeout_s=request.timeout_s, prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
        )  # fmt: skip

    assert (
        run_set(["--set", "smoke"], {"q_0001": ok(["doc-a"])}, judge_chat=FakeLLM([billed] * 2))
        == 0
    )
    folder = only_child(workdir / "results" / "smoke" / "ask")
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert s["judge"]["spend_usd"] == 0.05 and s["judge"]["cost_usd"] == 0.004
    assert s["spend_usd"] == 0.0  # the system's spend, measured before judging
    assert "judge ledger growth: $0.0500" in (folder / "summary.md").read_text("utf-8")
    assert "judge_spend=$0.0500" in history_rows(workdir)[1][8]


def test_the_calibration_key_runs_the_calibration_module() -> None:
    commands = json.loads((ROOT / "commands.json").read_text(encoding="utf-8"))["commands"]
    assert "exec python -m eval.judge_calibrate" in commands["eval-judge-calibrate"]["run"][-1]
    assert runner.parse_args(["--set", "tune"]).judge is True
    assert runner.parse_args(["--set", "tune", "--no-judge"]).judge is False
