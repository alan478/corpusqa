"""The answer judge (eval/judge.py) and its calibration (eval/judge_calibrate.py).

Offline: a scripted FakeLLM (or GeminiChat over a scripted SDK stand-in for the generation
fields). Invented questions, answers and facts only: never a real question, gold answer or id.
"""

import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from corpusqa.config import (
    JUDGE_JUDGMENTS,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_MODEL,
    JUDGE_PROMPT_NAME,
    JUDGE_SPEND_CAP_USD,
    JUDGE_SPEND_LEDGER,
    JUDGE_THINKING_BUDGET,
    JUDGE_TOOL_NAME,
    Settings,
)
from corpusqa.llm.client import AGENT_PRICES, GeminiChat, chat_cost
from corpusqa.llm.errors import ChatError, SpendCapReached
from corpusqa.llm.fakes import ChatRequest, FakeLLM, fake_reply
from corpusqa.llm.interfaces import ModelReply, ToolCall, Usage
from corpusqa.llm.limits import AsyncRateLimiter, SpendLedger
from corpusqa.obs.testing import RecordingTracer
from corpusqa.prompts import PromptText, load_text_prompt
from eval import judge_calibrate as calibration
from eval.judge import (
    JUDGE_PRICES,
    LABELS,
    NO_MAJORITY,
    QUESTION_SPAN,
    SET_SPAN,
    STATUSES,
    VERDICT_TOOL,
    InvalidVerdict,
    Judge,
    JudgeFailed,
    JudgeItem,
    JudgmentCache,
    QuestionJudgment,
    Verdict,
    build_judge_chat,
    parse_verdict,
    render_input,
    summarize,
)
from tests.unit.test_llm_chat_client import ScriptedChatClient, call_part, response

FACTS = ("The invented widget costs 3 coins.", "The invented widget is blue.")
ANSWER = "The invented widget costs 3 coins and it is blue."
PROMPT = load_text_prompt(JUDGE_PROMPT_NAME)


def item(
    qid: str = "q_0001",
    *,
    answer: str | None = ANSWER,
    facts: tuple[str, ...] = FACTS,
    answerable: bool = True,
) -> JudgeItem:
    return JudgeItem(
        question_id=qid,
        question=f"Invented question {qid}?",
        gold_answer=f"Invented gold answer {qid}.",
        facts=facts,
        answerable=answerable,
        answer=answer,
    )


def args(
    statuses: Sequence[str] = ("right", "right"),
    verdict: str = "correct",
    *,
    says: bool = False,
    contradictions: Sequence[dict[str, str]] = (),
    numbers: Sequence[int] | None = None,
) -> dict[str, object]:
    numbers = list(numbers) if numbers is not None else list(range(1, len(statuses) + 1))
    return {
        "facts": [
            {"fact": n, "status": s, "evidence": f"words {n}" if s != "missing" else ""}
            for n, s in zip(numbers, statuses, strict=True)
        ],
        "contradictions": list(contradictions),
        "says_not_found": says,
        "verdict": verdict,
        "reason": f"Invented reason: {verdict}.",
    }


def reply(arguments: dict[str, object] | None = None, cost: float = 0.001) -> ModelReply:
    calls = [] if arguments is None else [ToolCall("c1", JUDGE_TOOL_NAME, arguments)]
    return fake_reply(tool_calls=calls, usage=Usage(100, 20, 0, 0, cost))


def judge_with(script: Sequence[object], **kwargs: object) -> tuple[Judge, FakeLLM]:
    fake = FakeLLM(script)  # type: ignore[arg-type]
    return Judge(fake, model_name="fake-judge", **kwargs), fake  # type: ignore[arg-type]


def judge_all(judge: Judge, items: Sequence[JudgeItem]) -> list[QuestionJudgment]:
    return asyncio.run(judge.judge_all(items))


# ---------------------------------------------------------------- the verdict tool
def test_the_tool_schema_is_the_verdict_model_inlined() -> None:
    schema = VERDICT_TOOL.parameters
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == set(Verdict.model_fields)
    assert set(schema["required"]) == set(Verdict.model_fields)  # type: ignore[arg-type]
    fact = properties["facts"]["items"]
    assert fact["properties"]["status"]["enum"] == list(STATUSES)
    assert properties["verdict"]["enum"] == list(LABELS)
    assert VERDICT_TOOL.name == JUDGE_TOOL_NAME == "verdict"


def test_the_prompt_is_versioned_and_names_every_label() -> None:
    assert PROMPT.name == "judge" and PROMPT.version == "1"
    for word in (*STATUSES, *LABELS, "says_not_found", "Contradictions", "verdict"):
        assert word in PROMPT.text


def test_the_judge_sees_the_question_gold_facts_and_answer_but_not_the_run() -> None:
    text = render_input(item(answerable=False))
    assert "<question>\nInvented question q_0001?\n</question>" in text
    assert "<answerable>no</answerable>" in text
    assert "<gold_answer>\nInvented gold answer q_0001.\n</gold_answer>" in text
    assert f"1. {FACTS[0]}\n2. {FACTS[1]}" in text
    assert text.endswith(f"<answer>\n{ANSWER}\n</answer>")
    # Only these five parts: nothing names the run or the setting that wrote the answer.
    tags = re.findall(r"<(\w+)>", text)
    assert tags == ["question", "answerable", "gold_answer", "answer_facts", "answer"]
    assert set(JudgeItem.__dataclass_fields__) == {
        "question_id", "question", "gold_answer", "facts", "answerable", "answer",
    }  # fmt: skip
    assert "q_0001" not in text.replace("Invented question q_0001?", "").replace(
        "Invented gold answer q_0001.", ""
    )


def test_a_verdict_is_parsed_with_its_facts_in_order() -> None:
    parsed = parse_verdict(reply(args(("partly", "right"), "partly", numbers=[2, 1])), 2)
    assert [c.fact for c in parsed.facts] == [1, 2]
    assert [c.status for c in parsed.facts] == ["right", "partly"]
    assert parsed.verdict == "partly" and parsed.says_not_found is False


@pytest.mark.parametrize(
    ("arguments", "problem"),
    [
        (None, "no verdict call"),
        (args(("right",)), "expected 1 to 2"),  # a fact missing
        (args(("right", "right", "right")), "expected 1 to 2"),  # one too many
        (args(("right", "right"), numbers=[1, 1]), "expected 1 to 2"),  # one twice
        (args(("right", "fine")), "off the schema"),  # not a status
        (args(("right", "right"), "mostly"), "off the schema"),  # not a verdict
        ({**args(), "extra": 1}, "off the schema"),
        ({k: v for k, v in args().items() if k != "reason"}, "off the schema"),
    ],
)
def test_anything_off_the_schema_or_the_facts_is_invalid(
    arguments: dict[str, object] | None, problem: str
) -> None:
    with pytest.raises(InvalidVerdict, match=problem):
        parse_verdict(reply(arguments), 2)


# ---------------------------------------------------------------- judging one answer
def test_two_agreeing_judgments_are_enough_and_each_call_is_the_forced_tool() -> None:
    judge, fake = judge_with([reply(args()), reply(args())])
    [result] = judge_all(judge, [item()])
    assert (result.outcome, result.verdict, result.facts) == ("judged", "correct", ("right",) * 2)
    assert result.verdicts == ("correct", "correct") and result.first_two_agree is True
    assert (result.calls, result.cost_usd) == (2, pytest.approx(0.002))
    assert result.evidence == ("words 1", "words 2")
    assert fake.remaining == 0 and len(fake.requests) == JUDGE_JUDGMENTS
    request: ChatRequest = fake.requests[0]
    assert request.tools == (VERDICT_TOOL,) and request.tool_choice == (JUDGE_TOOL_NAME,)
    assert request.max_output_tokens == JUDGE_MAX_OUTPUT_TOKENS
    assert request.thinking_budget == JUDGE_THINKING_BUDGET
    assert (request.prompt_name, request.prompt_version) == (PROMPT.name, PROMPT.version)
    system, user = request.messages
    assert (system.role, system.content) == ("system", PROMPT.text)
    assert (user.role, user.content) == ("user", render_input(item()))


def test_a_third_judgment_breaks_a_tie_and_the_majority_gives_the_facts() -> None:
    script = [
        reply(args(("right", "right"), "correct")),
        reply(args(("right", "missing"), "partly")),
        reply(args(("right", "partly"), "partly")),
    ]
    judge, fake = judge_with(script)
    [result] = judge_all(judge, [item()])
    assert result.verdict == "partly" and result.verdicts == ("correct", "partly", "partly")
    assert result.facts == ("right", "missing")  # the first judgment with the majority verdict
    assert result.first_two_agree is False and result.calls == 3 and fake.remaining == 0


def test_three_different_verdicts_leave_the_question_unjudged() -> None:
    script = [reply(args(verdict=v)) for v in ("correct", "partly", "wrong")]
    judge, _ = judge_with(script)
    [result] = judge_all(judge, [item()])
    assert (result.outcome, result.verdict, result.facts) == ("unjudged", None, ())
    assert NO_MAJORITY in result.problems and result.calls == 3


def test_an_invalid_output_is_asked_for_once_more() -> None:
    judge, fake = judge_with([reply(args(("right",))), reply(args()), reply(args())])
    [result] = judge_all(judge, [item()])
    assert result.verdict == "correct" and result.calls == 3 and fake.remaining == 0
    assert len(result.problems) == 1 and "judgment 1" in result.problems[0]
    assert fake.requests[1].messages == fake.requests[0].messages  # the same request again


def test_two_invalid_outputs_leave_the_question_unjudged_and_stop_its_calls() -> None:
    judge, fake = judge_with([reply(None), ChatError("scripted failure"), reply(args())])
    [result] = judge_all(judge, [item()])
    assert (result.outcome, result.verdict, result.says_not_found) == ("unjudged", None, None)
    assert result.calls == 2 and fake.remaining == 1  # never a third try, never a guess
    assert "no verdict call" in result.problems[0] and "scripted failure" in result.problems[1]


def test_a_request_without_an_answer_is_judged_by_code() -> None:
    judge, fake = judge_with([])
    [result] = judge_all(judge, [item(answer=None)])
    assert (result.outcome, result.verdict, result.facts) == (
        "no_answer",
        "wrong",
        ("missing",) * 2,
    )
    assert result.calls == 0 and fake.requests == []


def test_the_spend_cap_stops_the_judging() -> None:
    judge, _ = judge_with([SpendCapReached("spend cap $10.00 reached")])
    with pytest.raises(SpendCapReached):
        judge_all(judge, [item()])


# ---------------------------------------------------------------- Codex review 2026-10-17
def test_the_spend_cap_stops_the_other_answers_too() -> None:
    # Finding 3: gather did not stop the other answers' tasks, so they went on calling.
    script = [SpendCapReached("spend cap $10.00 reached"), reply(args()), reply(args())]
    judge, fake = judge_with(script, concurrency=2)
    with pytest.raises(SpendCapReached):
        judge_all(judge, [item("q_0001"), item("q_0002")])
    assert len(fake.requests) == 1  # no call after the refusal


def test_a_transport_error_is_a_failed_call_not_a_crash() -> None:
    # Finding 2: only LLMError was caught; a transport error (not an SDK APIError) escaped
    # GeminiChat and stopped the whole judging.
    judge, _ = judge_with([ConnectionError("connection reset"), reply(args()), reply(args())])
    [result] = judge_all(judge, [item()])
    assert result.verdict == "correct" and result.calls == 3
    assert "connection reset" in result.problems[0]


def test_a_reply_must_hold_exactly_one_verdict_call_with_strict_types() -> None:
    # Finding 4: a second, contradicting verdict call was ignored, and "false" passed as a
    # boolean.
    two = fake_reply(
        tool_calls=[
            ToolCall("c1", JUDGE_TOOL_NAME, args(verdict="correct")),
            ToolCall("c2", JUDGE_TOOL_NAME, args(verdict="wrong")),
        ]
    )
    with pytest.raises(InvalidVerdict, match="exactly one"):
        parse_verdict(two, 2)
    with pytest.raises(InvalidVerdict, match="off the schema"):
        parse_verdict(reply({**args(), "says_not_found": "false"}), 2)
    with pytest.raises(InvalidVerdict, match="off the schema"):
        parse_verdict(reply({**args(), "reason": 7}), 2)


def test_a_protected_judge_failure_reaches_the_trace_as_its_type_only() -> None:
    # Finding 1: an exception from the untraced work crossed the `judge_all` span, whose
    # error status kept its full text (per-question held-out detail).
    tracer = RecordingTracer()

    def fail(result: QuestionJudgment) -> None:
        raise RuntimeError(f"detail about {result.question_id}")

    judge, _ = judge_with([reply(args()), reply(args())], traced=False)
    with tracer.trace("t"), pytest.raises(JudgeFailed, match=r"^RuntimeError$"):
        asyncio.run(judge.judge_all([item()], fail))
    recorded = json.dumps([[o.output, o.metadata, o.status_message] for o in tracer.observations])
    assert "q_0001" not in recorded and "detail" not in recorded
    [set_span] = tracer.named(SET_SPAN)
    assert set_span.level == "ERROR"


def test_the_cache_reuses_judgments_of_the_same_prompt_only(tmp_path: Path) -> None:
    cache = JudgmentCache(tmp_path / "cache")
    first, _ = judge_with([reply(args()), reply(args())], cache=cache)
    judge_all(first, [item()])
    again, fake = judge_with([], cache=cache)
    [result] = judge_all(again, [item()])
    assert (result.verdict, result.calls, result.cached) == ("correct", 0, 2)
    assert fake.requests == []
    changed = PromptText(PROMPT.name, "test-other", PROMPT.text)
    other, fake = judge_with([reply(args()), reply(args())], cache=cache, prompt=changed)
    assert judge_all(other, [item()])[0].cached == 0 and fake.remaining == 0
    another_answer, fake = judge_with([reply(args()), reply(args())], cache=cache)
    assert judge_all(another_answer, [item(answer="Other words.")])[0].calls == 2


def test_answers_are_judged_at_most_concurrency_at_once() -> None:
    in_flight, peak = 0, 0

    async def slow(request: ChatRequest) -> ModelReply:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return reply(args())

    judge, _ = judge_with([slow] * 10, concurrency=2)
    results = judge_all(judge, [item(f"q_{n}") for n in range(5)])
    assert [r.question_id for r in results] == [f"q_{n}" for n in range(5)]
    assert peak == 2


# ---------------------------------------------------------------- aggregates
def judged(
    verdicts: Sequence[str],
    statuses: Sequence[str] = ("right",),
    *,
    answerable: bool = True,
    says: bool = False,
    contradictions: int = 0,
) -> QuestionJudgment:
    judgments = tuple(
        Verdict.model_validate(
            args(
                statuses,
                v,
                says=says,
                contradictions=[{"claim": "c", "gold": "g"}] * contradictions,
            )
        )
        for v in verdicts
    )
    label = Counter(verdicts).most_common(1)[0][0]
    chosen = next(j for j in judgments if j.verdict == label)
    return QuestionJudgment(
        question_id="q",
        answerable=answerable,
        outcome="judged",
        verdict=chosen.verdict,
        facts=tuple(c.status for c in chosen.facts),
        evidence=tuple(c.evidence for c in chosen.facts),
        contradictions=tuple(chosen.contradictions),
        says_not_found=chosen.says_not_found,
        reason=chosen.reason,
        judgments=judgments,
        calls=len(verdicts),
        cost_usd=0.01 * len(verdicts),
    )


def test_summarize_counts_shares_agreement_and_cost() -> None:
    unjudged = QuestionJudgment(
        "q", True, "unjudged", None, (), (), (), None, "", problems=(NO_MAJORITY,), calls=3
    )
    no_answer = QuestionJudgment(
        "q", True, "no_answer", "wrong", ("missing", "missing"), ("", ""), (), False, ""
    )
    results = [
        judged(["correct", "correct"], ("right", "right")),
        judged(["partly", "correct", "partly"], ("right", "partly"), contradictions=1),
        judged(["correct", "correct"], ("right",), answerable=False, says=True),
        judged(["wrong", "wrong"], ("missing",), answerable=False),
        unjudged,
        no_answer,
    ]
    t = summarize(results)
    assert t.questions == 6
    assert t.verdict_counts == {"correct": 2, "partly": 1, "wrong": 2, "unjudged": 1}
    assert t.verdict_shares()["correct"] == pytest.approx(2 / 6)
    assert t.fact_counts == {"right": 4, "partly": 1, "missing": 3, "wrong": 0}
    assert t.facts == 8 and t.fact_shares()["missing"] == pytest.approx(3 / 8)
    assert t.contradictions == 1 and t.no_answer == 1
    assert (t.unanswerable, t.unanswerable_judged, t.said_not_found) == (2, 2, 1)
    assert t.said_not_found_rate == 0.5
    assert (t.two_judgments, t.first_two_agree, t.agreement) == (4, 3, 0.75)
    assert (t.third_judgments, t.no_majority) == (1, 1)
    assert t.calls == 2 + 3 + 2 + 2 + 3 and t.cost_usd == pytest.approx(0.09)
    out = t.to_json()
    assert out["verdict_shares"]["unjudged"] == round(1 / 6, 4)
    assert out["agreement"]["rate"] == 0.75 and out["unanswerable"]["rate"] == 0.5
    assert summarize([]).agreement is None and summarize([]).verdict_shares()["correct"] is None


# ---------------------------------------------------------------- tracing and the real client
def scripted_gemini(
    tmp_path: Path, answers: Sequence[object], cap: float = JUDGE_SPEND_CAP_USD
) -> tuple[GeminiChat, ScriptedChatClient, SpendLedger]:
    client = ScriptedChatClient(answers)  # type: ignore[arg-type]

    async def no_wait(seconds: float) -> None:
        return None

    ledger = SpendLedger(tmp_path / "judge.json", cap)
    chat = GeminiChat(
        client,  # type: ignore[arg-type]
        ledger,
        limiter=AsyncRateLimiter(1e9, sleep=no_wait),
        model=JUDGE_MODEL,
        thinking_level=None,
        prices=JUDGE_PRICES,
    )
    return chat, client, ledger


def verdict_response() -> object:
    return response(call_part(JUDGE_TOOL_NAME, args(), "c1"), prompt=1_000, output=200)


def test_each_answer_is_a_judge_span_with_its_chat_generations(tmp_path: Path) -> None:
    tracer = RecordingTracer()
    chat, client, ledger = scripted_gemini(tmp_path, [verdict_response(), verdict_response()])
    judge = Judge(chat)
    with tracer.trace("t") as trace:
        [result] = asyncio.run(judge.judge_all([item()]))
    assert result.verdict == "correct"
    [root] = tracer.traces("t")
    [set_span] = tracer.named(SET_SPAN)
    assert set_span.parent_id == root.id and trace.trace_id == root.trace_id
    assert set_span.metadata["prompt_version"] == PROMPT.version
    assert isinstance(set_span.output, dict) and set_span.output["verdicts"]["correct"] == 1
    [span] = tracer.named(QUESTION_SPAN)
    assert span.parent_id == set_span.id
    assert span.input == {"question_id": "q_0001", "answerable": True}
    assert span.output == {"outcome": "judged", "verdict": "correct", "facts": ["right", "right"],
                           "contradictions": 0, "says_not_found": False}  # fmt: skip
    assert span.metadata["verdicts"] == ["correct", "correct"]
    assert span.metadata["first_two_agree"] is True
    generations = tracer.children(span)
    assert [g.name for g in generations] == ["chat", "chat"]
    g = generations[0]
    assert (g.kind, g.model, g.version) == ("generation", JUDGE_MODEL, PROMPT.version)
    assert g.metadata["prompt_name"] == "judge" and g.metadata["prompt_version"] == PROMPT.version
    assert g.model_parameters == {
        "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
        "thinking_budget": JUDGE_THINKING_BUDGET,
        "tool_choice": [JUDGE_TOOL_NAME],
    }
    cost = chat_cost(1_000, 0, 200, JUDGE_PRICES)
    assert g.cost == {"total": pytest.approx(cost)}
    assert ledger.totals.cost_usd == pytest.approx(2 * cost)
    assert result.cost_usd == pytest.approx(2 * cost)
    config = client.requests[0]["config"]
    assert config.thinking_config.thinking_budget == JUDGE_THINKING_BUDGET  # type: ignore[attr-defined]
    assert client.requests[0]["model"] == JUDGE_MODEL


def test_untraced_judging_traces_only_the_aggregates(tmp_path: Path) -> None:
    tracer = RecordingTracer()
    chat, _, _ = scripted_gemini(tmp_path, [verdict_response(), verdict_response()])
    with tracer.trace("t"):
        [result] = asyncio.run(Judge(chat, traced=False).judge_all([item()]))
    assert result.verdict == "correct"
    assert [o.name for o in tracer.observations] == ["t", SET_SPAN]
    recorded = json.dumps([[o.input, o.output, o.metadata] for o in tracer.observations])
    assert "q_0001" not in recorded and "Invented" not in recorded and ANSWER not in recorded


def test_a_ledger_at_its_cap_refuses_the_judge(tmp_path: Path) -> None:
    chat, client, _ = scripted_gemini(tmp_path, [verdict_response()], cap=0.0001)
    with pytest.raises(SpendCapReached):
        asyncio.run(Judge(chat).judge_all([item()]))
    assert client.requests == []  # refused before any call


def test_the_judge_chat_uses_the_judge_model_prices_and_its_own_ledger(tmp_path: Path) -> None:
    chat = build_judge_chat(Settings(corpus_dir=tmp_path, port=8080, gemini_api_key="fake-key"))
    assert (chat.model, chat.prices, chat.thinking_level) == (JUDGE_MODEL, JUDGE_PRICES, None)
    assert chat._ledger.path == JUDGE_SPEND_LEDGER and chat._ledger.cap_usd == JUDGE_SPEND_CAP_USD
    assert JUDGE_SPEND_LEDGER.name == "judge.json" and JUDGE_SPEND_CAP_USD == 10.0
    assert (JUDGE_PRICES.input_per_m, JUDGE_PRICES.cached_input_per_m,
            JUDGE_PRICES.output_per_m) == (2.00, 0.20, 12.00)  # fmt: skip
    assert JUDGE_MODEL == "gemini-3.1-pro-preview"


def test_a_chat_model_bills_at_its_own_prices(tmp_path: Path) -> None:
    chat, _, ledger = scripted_gemini(tmp_path, [verdict_response()])
    out = asyncio.run(chat.complete(
        [], [VERDICT_TOOL], tool_choice="auto", max_output_tokens=10, thinking_budget=0,
        timeout_s=5.0, prompt_name="judge", prompt_version="1",
    ))  # fmt: skip
    judged_cost = chat_cost(1_000, 0, 200, JUDGE_PRICES)
    assert out.usage.cost_usd == pytest.approx(judged_cost)
    assert judged_cost > chat_cost(1_000, 0, 200, AGENT_PRICES)
    assert ledger.totals.cost_usd == pytest.approx(judged_cost)


# ---------------------------------------------------------------- calibration
def test_cohen_kappa_by_hand() -> None:
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]
    # observed 3/4; chance: first a 2, b 2; second a 1, b 3 -> (2*1 + 2*3) / 16 = 1/2
    assert calibration.cohen_kappa(pairs, ["a", "b"]) == pytest.approx(0.5)
    assert calibration.cohen_kappa([("a", "a")] * 3, ["a", "b"]) is None  # chance is 1
    assert calibration.cohen_kappa([], ["a"]) is None
    assert calibration.confusion(pairs, ["a", "b"]) == {
        "a": {"a": 1, "b": 1},
        "b": {"a": 0, "b": 2},
    }


def test_calibration_reads_only_tune_or_smoke_runs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    assert calibration.run_folder("R1", results) == (results / "tune" / "ask" / "R1").resolve()
    smoke = results / "smoke" / "ask" / "R2"
    assert calibration.run_folder(str(smoke), results) == smoke.resolve()
    for refused in (
        str(results / "heldout" / "ask" / "R3"),
        str(results / "all" / "ask" / "R4"),
        str(results / "tune" / "ask" / ".." / ".." / "heldout" / "ask" / "R5"),
        str(results / "tune" / "ask"),
        str(tmp_path / "elsewhere" / "R6"),
    ):
        with pytest.raises(calibration.CalibrationError, match="Eval split"):
            calibration.run_folder(refused, results)


def write_calibration_data(tmp_path: Path) -> tuple[Path, Path]:
    """Two invented tune questions (one unanswerable) and one hand-checked run R1."""
    rows = [
        {"question_id": "q_0001", "question": "Invented question q_0001?",
         "gold_answer": "Invented gold q_0001.", "answer_facts": list(FACTS),
         "expected_doc_ids": ["doc-a"]},
        {"question_id": "q_0002", "question": "Invented question q_0002?",
         "gold_answer": "The records do not say.", "answer_facts": ["Says it is not found."],
         "expected_doc_ids": []},
    ]  # fmt: skip
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    sets = tmp_path / "sets"
    sets.mkdir()
    for name, ids in {"tune": ["q_0001", "q_0002"], "smoke": ["q_0001"]}.items():
        (sets / f"{name}.json").write_text(json.dumps({"question_ids": ids}), encoding="utf-8")
    run = tmp_path / "results" / "tune" / "ask" / "R1"
    run.mkdir(parents=True)
    answers = [
        {"question_id": "q_0001", "answer": ANSWER, "error": None},
        {"question_id": "q_0002", "answer": "The records do not contain it.", "error": None},
    ]
    (run / "per_question.jsonl").write_text(
        "\n".join(json.dumps(a) for a in answers) + "\n", encoding="utf-8"
    )
    check = {"questions": [
        {"id": "q_0001", "verdict": "partly correct", "fact_status": ["right", "partly"],
         "reason": "Invented hand reason."},
        {"id": "q_0002", "verdict": "correct", "fact_status": ["right"], "reason": "Says so."},
    ]}  # fmt: skip
    (run / "answer_check.json").write_text(json.dumps(check), encoding="utf-8")
    return questions, sets


def responder(table: dict[str, ModelReply]) -> Callable[[ChatRequest], ModelReply]:
    def answer(request: ChatRequest) -> ModelReply:
        text = request.messages[-1].content
        return next(r for qid, r in table.items() if f"Invented question {qid}?" in text)

    return answer


def test_calibration_compares_facts_verdicts_and_not_found_with_the_hand_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    questions, sets = write_calibration_data(tmp_path)
    monkeypatch.setattr(calibration, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(calibration, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(calibration, "LEDGER", tmp_path / "judge.json")
    table = {
        "q_0001": reply(args(("right", "right"), "correct")),  # hand: right, partly -> 1 of 2
        "q_0002": reply(args(("right",), "correct", says=True)),
    }
    fake = FakeLLM([responder(table)] * 4)
    parsed = calibration.parse_args(["--runs", "R1", "--label", "test round"])
    folder = calibration.calibrate(parsed, fake, "python -m eval.judge_calibrate", questions, sets)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    c = report["comparison"]
    assert (c["answers"], c["judged"]) == (2, 2)
    assert (c["facts"]["agree"], c["facts"]["total"]) == (2, 3)
    assert c["facts"]["confusion_hand_rows_judge_columns"]["partly"]["right"] == 1
    assert (c["verdicts"]["agree"], c["verdicts"]["total"]) == (1, 2)
    assert c["said_not_found"] == {"unanswerable_judged": 1, "agree": 1, "hand_said": 1,
                                   "judge_said": 1}  # fmt: skip
    [fact] = c["fact_disagreements"]
    assert (fact["question_id"], fact["fact"], fact["hand"], fact["judge"]) == (
        "q_0001", 2, "partly", "right",
    )  # fmt: skip
    [verdict] = c["verdict_disagreements"]
    assert (verdict["hand"], verdict["judge"]) == ("partly", "correct")
    assert c["consistency"]["first_two_same_verdict"] == 2 and c["unjudged"] == []
    assert report["passed"] is False and report["bar"] == 0.9
    assert report["prompt_version"] == PROMPT.version and report["label"] == "test round"
    assert report["calls"] == 4 and report["cached"] == 0
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "Fact status agreement: 66.7%" in md and "MISSES" in md
    assert "| hand \\ judge | right | partly | missing | wrong |" in md
    assert "Cohen's kappa" in md and "q_0001" in md
    assert "fact agreement 66.7%" in capsys.readouterr().out
    again = calibration.calibrate(parsed, FakeLLM([]), "again", questions, sets)
    assert json.loads((again / "report.json").read_text("utf-8"))["cached"] == 4


def test_calibration_refuses_hand_checks_that_do_not_fit_the_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions, sets = write_calibration_data(tmp_path)
    monkeypatch.setattr(calibration, "RESULTS_DIR", tmp_path / "results")
    check = tmp_path / "results" / "tune" / "ask" / "R1" / "answer_check.json"
    data = json.loads(check.read_text(encoding="utf-8"))
    data["questions"][0]["fact_status"] = ["right"]
    check.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(calibration.CalibrationError, match="do not fit"):
        calibration.load_cases(["R1"], tmp_path / "results", questions, sets)


def test_calibration_refuses_input_files_that_lead_outside_the_tune_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Finding 6: only the folder was checked, so a symlinked file could read held-out
    # material; and an unknown id was named in the error message.
    questions, sets = write_calibration_data(tmp_path)
    results = tmp_path / "results"
    run = results / "tune" / "ask" / "R1"
    outside = results / "heldout" / "ask" / "X"
    outside.mkdir(parents=True)
    (outside / "per_question.jsonl").write_bytes((run / "per_question.jsonl").read_bytes())
    (run / "per_question.jsonl").unlink()
    (run / "per_question.jsonl").symlink_to(outside / "per_question.jsonl")
    with pytest.raises(calibration.CalibrationError, match="Eval split"):
        calibration.load_cases(["R1"], results, questions, sets)
    (run / "per_question.jsonl").unlink()
    (run / "per_question.jsonl").write_bytes((outside / "per_question.jsonl").read_bytes())
    check = run / "answer_check.json"
    data = json.loads(check.read_text(encoding="utf-8"))
    data["questions"][0]["id"] = "q_9999"
    check.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(calibration.CalibrationError) as refused:
        calibration.load_cases(["R1"], results, questions, sets)
    assert "q_9999" not in str(refused.value)


def test_the_default_runs_are_the_five_hand_checked_tune_runs() -> None:
    assert len(calibration.DEFAULT_RUNS) == 5
    assert calibration.parse_args([]).runs == list(calibration.DEFAULT_RUNS)
