"""The eval's automatic answer judge: each answer, fact by fact, against its `answer_facts`.

User, 2026-10-17: "okay build it that way and add the second one as the model". For one
answer, JUDGE_MODEL (config.py) gets the question, the gold answer, the answer facts, whether
the records hold an answer at all, and the system's answer. It is never told which run or
setting wrote the answer. It must call one tool, `verdict`, whose arguments are `Verdict`:
- per fact: right | partly | missing | wrong, and the answer's words it rests on;
- the claims in the answer that contradict the gold answer;
- says_not_found: does the answer say plainly that the records do not hold (part of) it;
- the question verdict, correct | partly | wrong, and one sentence on why.
The rules are the hand checks' (eval/results/tune/ask/*/answer_check.md), written in the prompt
src/corpusqa/prompts/judge.md. The field descriptions of `Verdict` are sent with the tool, so
they are part of that prompt: changing them bumps its version.

Steady: every answer is judged JUDGE_JUDGMENTS (2) times. When the two question verdicts differ
it is judged once more, and the verdict that more than half of the judgments share counts
(three different verdicts: no majority, the question is unjudged). Its facts, contradictions
and says_not_found come from the first judgment with that verdict. An invalid output (no
call, a value off the schema, a missing or extra fact, a failed call) is asked for again
once (also a reply with more than one call); a second invalid one leaves the question
unjudged. Strings and booleans are strict ("false" is not a boolean). Unjudged questions
are counted, never guessed. A request that brought back no answer is judged by code: every
fact missing, verdict wrong, no model call.

Code does what code can do: citations and "an unanswerable question returned no ids" stay as
eval/ask_metrics.py computes them. The judge judges only the answer text.

Spend: the model client's ledger (JUDGE_SPEND_LEDGER, cap JUDGE_SPEND_CAP_USD). A refused
call (SpendCapReached) stops the judging: it propagates. Tracing: a `judge_all` span with the
aggregates; under it one `judge` span per answer (question id, verdicts, agreement, the final
verdict), with that answer's `chat` generations (prompt `judge` and its version). With
`traced=False` (held-out sets) only `judge_all` is traced: per-question inputs (gold answers)
and results stay out of traces, as the runner keeps them out for those sets.
"""

import asyncio
import contextvars
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

from corpusqa import obs
from corpusqa.config import (
    JUDGE_CALL_TIMEOUT_S,
    JUDGE_CONCURRENCY,
    JUDGE_JUDGMENTS,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_MODEL,
    JUDGE_PRICE_CACHED_INPUT_PER_M_TOKENS_USD,
    JUDGE_PRICE_INPUT_PER_M_TOKENS_USD,
    JUDGE_PRICE_OUTPUT_PER_M_TOKENS_USD,
    JUDGE_PROMPT_NAME,
    JUDGE_SPEND_CAP_USD,
    JUDGE_SPEND_LEDGER,
    JUDGE_THINKING_BUDGET,
    JUDGE_TIEBREAK_JUDGMENTS,
    JUDGE_TOOL_NAME,
    JUDGE_TRIES_PER_JUDGMENT,
    Settings,
)
from corpusqa.llm.client import ChatPrices, GeminiChat
from corpusqa.llm.errors import SpendCapReached
from corpusqa.llm.interfaces import ChatModel, Message, ModelReply, ToolSchema
from corpusqa.prompts import PromptText, load_text_prompt

logger = logging.getLogger("eval.judge")

Status = Literal["right", "partly", "missing", "wrong"]
STATUSES: tuple[Status, ...] = ("right", "partly", "missing", "wrong")
Label = Literal["correct", "partly", "wrong"]
LABELS: tuple[Label, ...] = ("correct", "partly", "wrong")
Outcome = Literal["judged", "unjudged", "no_answer"]
UNJUDGED = "unjudged"

SET_SPAN = "judge_all"
QUESTION_SPAN = "judge"
NO_MAJORITY = "no majority: the judgments gave three different verdicts"
ERRORS_SHOWN = 3  # validation problems named in one invalid-output message

JUDGE_PRICES = ChatPrices(
    JUDGE_PRICE_INPUT_PER_M_TOKENS_USD,
    JUDGE_PRICE_CACHED_INPUT_PER_M_TOKENS_USD,
    JUDGE_PRICE_OUTPUT_PER_M_TOKENS_USD,
)


# ---------------------------------------------------------------- the verdict tool
class FactCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Lax on purpose: a JSON number may come back as 2.0; 2.5 is still refused.
    fact: int = Field(description="The fact's number, from 1, in the order given.")
    status: Status = Field(description="right, partly, missing or wrong, by the rules.")
    evidence: StrictStr = Field(
        description="The answer's words this status rests on, copied; empty when missing."
    )


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: StrictStr = Field(description="The answer's words, copied.")
    gold: StrictStr = Field(description="What the gold answer says instead.")


class Verdict(BaseModel):
    """The `verdict` tool's arguments: one judgment of one answer."""

    model_config = ConfigDict(extra="forbid")

    facts: list[FactCheck] = Field(description="One entry per answer fact, in the order given.")
    contradictions: list[Contradiction] = Field(
        description="Claims in the answer that contradict the gold answer; empty when none."
    )
    says_not_found: StrictBool = Field(
        description="True when the answer says plainly that the records do not contain the "
        "answer, or the answer to a part of the question."
    )
    verdict: Label = Field(description="correct, partly or wrong, by the rules.")
    reason: StrictStr = Field(description="One sentence: what decided the verdict.")


def tool_parameters(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema with every `$ref` inlined and no `title`s: one plain object
    schema, the shape the agent's tools already send (`parameters_json_schema`)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[str(node["$ref"]).rsplit("/", 1)[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    inlined = inline(schema)
    if not isinstance(inlined, dict):
        raise TypeError(f"{model.__name__}'s schema is not an object schema")
    return inlined


VERDICT_TOOL = ToolSchema(
    name=JUDGE_TOOL_NAME,
    description="Record your judgment of the answer: call it once, with one entry per answer fact.",
    parameters=tool_parameters(Verdict),
)


class InvalidVerdict(ValueError):
    """The model's output is not a usable verdict for this question."""


class JudgeFailed(RuntimeError):
    """Untraced (held-out) judging failed. The message is only the original error's type:
    its text may hold per-question detail, and the `judge_all` span records this error."""


def parse_verdict(reply: ModelReply, n_facts: int) -> Verdict:
    """The reply's verdict call, checked: exactly one call and it is `verdict`, the schema
    (strict strings and booleans), and exactly the facts 1..n_facts (put in order). Anything
    else raises InvalidVerdict, naming the problem."""
    calls = reply.message.tool_calls
    if not any(call.name == JUDGE_TOOL_NAME for call in calls):
        raise InvalidVerdict(f"no {JUDGE_TOOL_NAME} call (stop reason {reply.stop_reason})")
    if len(calls) != 1:
        names = [call.name for call in calls]
        raise InvalidVerdict(f"{len(calls)} calls {names}: expected exactly one {JUDGE_TOOL_NAME}")
    try:
        verdict = Verdict.model_validate(dict(calls[0].arguments))
    except ValidationError as exc:
        problems = [
            f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
            for e in exc.errors()[:ERRORS_SHOWN]
        ]
        raise InvalidVerdict("off the schema: " + "; ".join(problems)) from exc
    numbers = sorted(check.fact for check in verdict.facts)
    if numbers != list(range(1, n_facts + 1)):
        raise InvalidVerdict(f"facts numbered {numbers}, expected 1 to {n_facts} once each")
    return verdict.model_copy(update={"facts": sorted(verdict.facts, key=lambda c: c.fact)})


# ---------------------------------------------------------------- one answer
@dataclass(frozen=True, slots=True)
class JudgeItem:
    """What the judge gets for one answer (and the id and kind, for the reports only)."""

    question_id: str
    question: str
    gold_answer: str
    facts: tuple[str, ...]
    answerable: bool  # the question has expected ids
    answer: str | None  # None: the request brought back no answer


def render_input(item: JudgeItem) -> str:
    """The user message: the question, whether it is answerable, the gold answer, the
    numbered facts and the answer. Nothing about the run or setting that wrote the answer."""
    facts = "\n".join(f"{n}. {fact}" for n, fact in enumerate(item.facts, start=1)) or "(none)"
    return (
        f"<question>\n{item.question}\n</question>\n"
        f"<answerable>{'yes' if item.answerable else 'no'}</answerable>\n"
        f"<gold_answer>\n{item.gold_answer}\n</gold_answer>\n"
        f"<answer_facts>\n{facts}\n</answer_facts>\n"
        f"<answer>\n{item.answer}\n</answer>"
    )


@dataclass(frozen=True, slots=True)
class QuestionJudgment:
    """The judge's result for one answer."""

    question_id: str
    answerable: bool
    outcome: Outcome
    verdict: Label | None  # None when unjudged
    facts: tuple[Status, ...]  # the chosen judgment's statuses, one per fact (empty: unjudged)
    evidence: tuple[str, ...]
    contradictions: tuple[Contradiction, ...]
    says_not_found: bool | None
    reason: str
    judgments: tuple[Verdict, ...] = ()  # every valid judgment, in order
    problems: tuple[str, ...] = ()  # why outputs were invalid, and why it is unjudged
    calls: int = 0  # model calls made (a cached judgment makes none)
    cached: int = 0  # judgments taken from the calibration cache
    cost_usd: float = 0.0

    @property
    def verdicts(self) -> tuple[Label, ...]:
        return tuple(j.verdict for j in self.judgments)

    @property
    def first_two_agree(self) -> bool | None:
        """Did the first two judgments give the same question verdict (None: fewer than two)."""
        v = self.verdicts
        return None if len(v) < 2 else v[0] == v[1]


def no_answer_judgment(item: JudgeItem) -> QuestionJudgment:
    """A request without an answer states nothing: every fact missing, verdict wrong."""
    return QuestionJudgment(
        question_id=item.question_id,
        answerable=item.answerable,
        outcome="no_answer",
        verdict="wrong",
        facts=("missing",) * len(item.facts),
        evidence=("",) * len(item.facts),
        contradictions=(),
        says_not_found=False,
        reason="The request brought back no answer.",
    )


def majority(labels: Sequence[Label]) -> Label | None:
    """The verdict more than half of the judgments share, or None."""
    if not labels:
        return None
    label, count = Counter(labels).most_common(1)[0]
    return label if count * 2 > len(labels) else None


# ---------------------------------------------------------------- calibration cache
class JudgmentCache:
    """Valid judgments on disk, one JSON file per key (calibration only; see config.py)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def get(self, key: str) -> tuple[Verdict, float] | None:
        path = self.folder / f"{key}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Verdict.model_validate(data["verdict"]), float(data["cost_usd"])
        except (OSError, ValueError, KeyError, TypeError):  # missing or unreadable: a miss
            return None

    def put(self, key: str, verdict: Verdict, cost_usd: float) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{key}.json"
        partial = path.with_name(path.name + ".tmp")
        record = {"verdict": verdict.model_dump(mode="json"), "cost_usd": cost_usd}
        partial.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        partial.replace(path)


@dataclass(slots=True)
class _Tally:
    calls: int = 0
    cached: int = 0
    cost_usd: float = 0.0
    problems: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- the judge
class Judge:
    """Judges answers with a ChatModel (see the module docstring)."""

    def __init__(
        self,
        chat: ChatModel,
        *,
        model_name: str = JUDGE_MODEL,
        traced: bool = True,
        cache: JudgmentCache | None = None,
        prompt: PromptText | None = None,
        concurrency: int = JUDGE_CONCURRENCY,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt or load_text_prompt(JUDGE_PROMPT_NAME)
        self.traced = traced
        self._chat = chat
        self._cache = cache
        self._concurrency = concurrency
        # Set by the first refused call: every later call of this judge is refused too,
        # without being made (the other answers' tasks stop at their next call).
        self._stopped: SpendCapReached | None = None

    async def judge_all(
        self,
        items: Sequence[JudgeItem],
        on_done: Callable[[QuestionJudgment], None] | None = None,
    ) -> list[QuestionJudgment]:
        """Every answer, at most `concurrency` at once; results in item order."""
        meta = {
            "questions": len(items),
            "model": self.model_name,
            "prompt_name": self.prompt.name,
            "prompt_version": self.prompt.version,
            "per_question_traced": self.traced,
        }
        with obs.span(SET_SPAN, metadata=meta) as span:
            work = self._all(items, on_done)
            if self.traced:
                results = await work
            else:
                # An empty context: no observation is current, so nothing below is traced.
                # An error leaves it as its type only (JudgeFailed), as this span records it;
                # the spend cap's message names no question.
                try:
                    results = await asyncio.create_task(work, context=contextvars.Context())
                except SpendCapReached:
                    raise
                except Exception as exc:
                    raise JudgeFailed(type(exc).__name__) from None
            span.update(output=summarize(results).to_json())
        return results

    async def _all(
        self, items: Sequence[JudgeItem], on_done: Callable[[QuestionJudgment], None] | None
    ) -> list[QuestionJudgment]:
        gate = asyncio.Semaphore(self._concurrency)

        async def one(item: JudgeItem) -> QuestionJudgment:
            async with gate:
                result = await self.judge_one(item)
            if on_done is not None:
                on_done(result)
            return result

        return list(await asyncio.gather(*(one(item) for item in items)))

    async def judge_one(self, item: JudgeItem) -> QuestionJudgment:
        if item.answer is None:
            return no_answer_judgment(item)
        with obs.span(
            QUESTION_SPAN,
            input={"question_id": item.question_id, "answerable": item.answerable},
            metadata={"facts": len(item.facts), "prompt_version": self.prompt.version},
        ) as span:
            result = await self._judge(item)
            span.update(
                output={
                    "outcome": result.outcome,
                    "verdict": result.verdict,
                    "facts": list(result.facts),
                    "contradictions": len(result.contradictions),
                    "says_not_found": result.says_not_found,
                },
                metadata={
                    "verdicts": list(result.verdicts),
                    "first_two_agree": result.first_two_agree,
                    "calls": result.calls,
                    "cached": result.cached,
                    "cost_usd": round(result.cost_usd, 6),
                    "problems": list(result.problems),
                },
            )
            if result.outcome == UNJUDGED:
                span.warning("; ".join(result.problems))
        return result

    async def _judge(self, item: JudgeItem) -> QuestionJudgment:
        tally = _Tally()
        valid: list[Verdict] = []
        for index in range(1, JUDGE_JUDGMENTS + JUDGE_TIEBREAK_JUDGMENTS + 1):
            if index > JUDGE_JUDGMENTS and majority([v.verdict for v in valid]) is not None:
                break
            verdict = await self._judgment(item, index, tally)
            if verdict is None:  # two invalid outputs: never guessed
                return self._unjudged(item, valid, tally)
            valid.append(verdict)
        label = majority([v.verdict for v in valid])
        if label is None:
            tally.problems.append(NO_MAJORITY)
            return self._unjudged(item, valid, tally)
        chosen = next(v for v in valid if v.verdict == label)
        return QuestionJudgment(
            question_id=item.question_id,
            answerable=item.answerable,
            outcome="judged",
            verdict=label,
            facts=tuple(check.status for check in chosen.facts),
            evidence=tuple(check.evidence for check in chosen.facts),
            contradictions=tuple(chosen.contradictions),
            says_not_found=chosen.says_not_found,
            reason=chosen.reason,
            judgments=tuple(valid),
            problems=tuple(tally.problems),
            calls=tally.calls,
            cached=tally.cached,
            cost_usd=tally.cost_usd,
        )

    def _unjudged(
        self, item: JudgeItem, valid: Sequence[Verdict], tally: _Tally
    ) -> QuestionJudgment:
        return QuestionJudgment(
            question_id=item.question_id,
            answerable=item.answerable,
            outcome="unjudged",
            verdict=None,
            facts=(),
            evidence=(),
            contradictions=(),
            says_not_found=None,
            reason="",
            judgments=tuple(valid),
            problems=tuple(tally.problems),
            calls=tally.calls,
            cached=tally.cached,
            cost_usd=tally.cost_usd,
        )

    async def _judgment(self, item: JudgeItem, index: int, tally: _Tally) -> Verdict | None:
        """Judgment number `index` of one answer: from the cache, or a call that is made again
        once when its output is invalid. None after JUDGE_TRIES_PER_JUDGMENT invalid ones."""
        text = render_input(item)
        key = self._key(text, index)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                tally.cached += 1
                return hit[0]
        messages = [Message(role="system", content=self.prompt.text), Message("user", text)]
        for _ in range(JUDGE_TRIES_PER_JUDGMENT):
            if self._stopped is not None:
                raise self._stopped
            try:
                reply = await self._chat.complete(
                    messages,
                    [VERDICT_TOOL],
                    tool_choice=(JUDGE_TOOL_NAME,),
                    max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                    thinking_budget=JUDGE_THINKING_BUDGET,
                    timeout_s=JUDGE_CALL_TIMEOUT_S,
                    prompt_name=self.prompt.name,
                    prompt_version=self.prompt.version,
                )
            except SpendCapReached as exc:
                self._stopped = exc
                raise
            except Exception as exc:  # an LLMError, or a transport error the client let out
                tally.calls += 1
                failure = f"{type(exc).__name__}: {exc}"
                tally.problems.append(f"judgment {index}: call failed: {failure}")
                continue
            tally.calls += 1
            tally.cost_usd += reply.usage.cost_usd
            try:
                verdict = parse_verdict(reply, len(item.facts))
            except InvalidVerdict as exc:
                tally.problems.append(f"judgment {index}: {exc}")
                continue
            if self._cache is not None:
                self._cache.put(key, verdict, reply.usage.cost_usd)
            return verdict
        return None

    def _key(self, text: str, index: int) -> str:
        """Everything that shapes one judgment: model, prompt, tool, caps, input, number."""
        parts = [
            self.model_name,
            self.prompt.name,
            self.prompt.version,
            self.prompt.text,
            json.dumps(dict(VERDICT_TOOL.parameters), sort_keys=True),
            VERDICT_TOOL.description,
            JUDGE_THINKING_BUDGET,
            JUDGE_MAX_OUTPUT_TOKENS,
            text,
            index,
        ]
        return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()


def build_judge_chat(settings: Settings) -> GeminiChat:
    """The judge's model client: JUDGE_MODEL at its prices, its thinking budget as sent (no
    thinking level), its own ledger and cap. No key: ChatUnavailable."""
    return GeminiChat.from_settings(
        settings,
        model=JUDGE_MODEL,
        prices=JUDGE_PRICES,
        thinking_level=None,
        ledger_path=JUDGE_SPEND_LEDGER,
        cap_usd=JUDGE_SPEND_CAP_USD,
    )


# ---------------------------------------------------------------- aggregates
def _share(part: int, whole: int) -> float | None:
    return None if whole == 0 else part / whole


@dataclass(frozen=True, slots=True)
class JudgeTotals:
    """The judge's aggregates over one set of answers (all a held-out report may show)."""

    questions: int
    verdict_counts: dict[str, int]  # correct, partly, wrong, unjudged
    no_answer: int  # requests without an answer, judged by code (counted as wrong)
    fact_counts: dict[str, int]  # over the questions that have a verdict
    contradictions: int
    unanswerable: int
    unanswerable_judged: int
    said_not_found: int  # of the judged unanswerable questions
    two_judgments: int  # questions with at least two valid judgments
    first_two_agree: int
    third_judgments: int
    no_majority: int
    calls: int
    cached: int
    cost_usd: float

    @property
    def facts(self) -> int:
        return sum(self.fact_counts.values())

    def verdict_shares(self) -> dict[str, float | None]:
        return {k: _share(v, self.questions) for k, v in self.verdict_counts.items()}

    def fact_shares(self) -> dict[str, float | None]:
        return {k: _share(v, self.facts) for k, v in self.fact_counts.items()}

    @property
    def said_not_found_rate(self) -> float | None:
        return _share(self.said_not_found, self.unanswerable_judged)

    @property
    def agreement(self) -> float | None:
        return _share(self.first_two_agree, self.two_judgments)

    def to_json(self) -> dict[str, Any]:
        def rounded(values: Mapping[str, float | None]) -> dict[str, float | None]:
            return {k: None if v is None else round(v, 4) for k, v in values.items()}

        return {
            "questions": self.questions,
            "verdicts": dict(self.verdict_counts),
            "verdict_shares": rounded(self.verdict_shares()),
            "no_answer": self.no_answer,
            "facts": self.facts,
            "fact_counts": dict(self.fact_counts),
            "fact_shares": rounded(self.fact_shares()),
            "contradictions": self.contradictions,
            "unanswerable": {
                "n": self.unanswerable,
                "judged": self.unanswerable_judged,
                "said_not_found": self.said_not_found,
                "rate": None
                if self.said_not_found_rate is None
                else round(self.said_not_found_rate, 4),
            },
            "agreement": {
                "two_judgments": self.two_judgments,
                "first_two_agree": self.first_two_agree,
                "rate": None if self.agreement is None else round(self.agreement, 4),
                "third_judgments": self.third_judgments,
                "no_majority": self.no_majority,
            },
            "calls": self.calls,
            "cached": self.cached,
            "cost_usd": round(self.cost_usd, 4),
        }


def summarize(results: Sequence[QuestionJudgment]) -> JudgeTotals:
    verdicts = Counter(r.verdict or UNJUDGED for r in results)
    facts = Counter(status for r in results for status in r.facts)
    unanswerable = [r for r in results if not r.answerable]
    judged_unanswerable = [r for r in unanswerable if r.says_not_found is not None]
    two = [r for r in results if r.first_two_agree is not None]
    return JudgeTotals(
        questions=len(results),
        verdict_counts={label: verdicts[label] for label in (*LABELS, UNJUDGED)},
        no_answer=sum(r.outcome == "no_answer" for r in results),
        fact_counts={status: facts[status] for status in STATUSES},
        contradictions=sum(len(r.contradictions) for r in results),
        unanswerable=len(unanswerable),
        unanswerable_judged=len(judged_unanswerable),
        said_not_found=sum(bool(r.says_not_found) for r in judged_unanswerable),
        two_judgments=len(two),
        first_two_agree=sum(bool(r.first_two_agree) for r in two),
        third_judgments=sum(len(r.judgments) > JUDGE_JUDGMENTS for r in results),
        no_majority=sum(NO_MAJORITY in r.problems for r in results),
        calls=sum(r.calls for r in results),
        cached=sum(r.cached for r in results),
        cost_usd=sum(r.cost_usd for r in results),
    )


def question_row(result: QuestionJudgment, kind: str) -> dict[str, object]:
    """One line of judge.jsonl (tune and smoke only): the judge's result for one answer, by
    fact number (never the question, gold answer or fact texts)."""
    return {
        "question_id": result.question_id,
        "kind": kind,
        "outcome": result.outcome,
        "verdict": result.verdict,
        "verdicts": list(result.verdicts),
        "first_two_agree": result.first_two_agree,
        "facts": [
            {"fact": n, "status": status, "evidence": evidence}
            for n, (status, evidence) in enumerate(
                zip(result.facts, result.evidence, strict=True), start=1
            )
        ],
        "contradictions": [c.model_dump() for c in result.contradictions],
        "says_not_found": result.says_not_found,
        "reason": result.reason,
        "problems": list(result.problems),
        "calls": result.calls,
        "cost_usd": round(result.cost_usd, 6),
    }


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def history_text(
    totals: JudgeTotals, model: str, prompt_version: str, spend_usd: float | None
) -> str:
    """The judge's part of the eval/history.csv `metrics` cell. `spend_usd`: the growth of the
    judge's ledger while it ran (it also counts calls billed but unusable)."""
    v, f = totals.verdict_shares(), totals.fact_shares()
    return (
        f"judge({model} v{prompt_version}) "
        + " ".join(f"{k}={_num(v[k])}" for k in (*LABELS, UNJUDGED))
        + f" facts(n={totals.facts}) "
        + " ".join(f"{k}={_num(f[k])}" for k in STATUSES)
        + f" contradictions={totals.contradictions}"
        + f" said_not_found={_num(totals.said_not_found_rate)}"
        + f" agreement={_num(totals.agreement)}"
        + f" judge_cost=${totals.cost_usd:.4f}"
        + f" judge_spend={_usd(spend_usd)}"
    )


def render_markdown(
    totals: JudgeTotals,
    model: str,
    prompt_version: str,
    not_correct: Sequence[QuestionJudgment] | None,
    spend_usd: float | None = None,
) -> list[str]:
    """The judge's section of summary.md. `not_correct` is None for a protected set (the
    section then holds aggregates only)."""
    v, f = totals.verdict_shares(), totals.fact_shares()
    lines = [
        "## Answer judge",
        "",
        f"`{model}`, prompt `judge` v{prompt_version}, {JUDGE_JUDGMENTS} judgments per "
        "answer (a third when their verdicts differ; the majority counts). It judges the "
        "answer text only, fact by fact against `answer_facts`; it is not told which run "
        "wrote the answer.",
        "",
        "| questions | correct | partly | wrong | unjudged |",
        "|---|---|---|---|---|",
        f"| {totals.questions} | "
        + " | ".join(f"{_num(v[k])} ({totals.verdict_counts[k]})" for k in (*LABELS, UNJUDGED))
        + " |",
        "",
        "| facts | right | partly | missing | wrong |",
        "|---|---|---|---|---|",
        f"| {totals.facts} | "
        + " | ".join(f"{_num(f[k])} ({totals.fact_counts[k]})" for k in STATUSES)
        + " |",
        "",
        f"- Contradictions of the gold answer: {totals.contradictions}.",
        f"- Unanswerable questions whose answer says plainly the records do not hold it: "
        f"{totals.said_not_found} of {totals.unanswerable_judged} judged "
        f"({_num(totals.said_not_found_rate)}).",
        f"- Judge agreement (the first two judgments gave the same verdict): "
        f"{totals.first_two_agree} of {totals.two_judgments} ({_num(totals.agreement)}); "
        f"third judgments: {totals.third_judgments}; no majority: {totals.no_majority}.",
        f"- Requests without an answer (judged by code as wrong): {totals.no_answer}.",
        f"- Judge calls: {totals.calls}; cost of the replies: ${totals.cost_usd:.4f}; judge "
        f"ledger growth: {_usd(spend_usd)} (`.cache/spend/judge.json`; it also counts calls "
        "billed but unusable).",
        "",
        "Verdict: correct = every fact right and no contradiction; partly = a fact partly "
        "stated or missing (or a contradiction about something not asked); wrong = an asked "
        "part wrong or missing, or most facts missing. Unjudged = the judge's output was "
        "invalid twice, or three judgments disagreed: counted, never guessed.",
    ]
    if not_correct is None:
        lines.append("Per-question judge results are not written for this set (AGENTS.md: "
                     "Eval split); this section holds aggregates only.")  # fmt: skip
    elif not_correct:
        lines += ["", "| question | verdict | verdicts | why |", "|---|---|---|---|"]
        for r in not_correct:
            why = (r.reason or "; ".join(r.problems)).replace("|", "\\|")
            shown = ", ".join(r.verdicts) or "-"
            lines.append(f"| {r.question_id} | {r.verdict or UNJUDGED} | {shown} | {why} |")
    return lines
