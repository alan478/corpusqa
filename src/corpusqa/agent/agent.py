"""The agent: one question, one object (agent-tools plan, section 6).

The prompt is always `[system, question, *history]` (`build_prompt`). The system message is
`prompts/agent.md` with the field catalog (the sources and their common fields) in its
`{catalog}` slot. `history` holds,
in order, every model turn and the tool messages that answer its calls, plus the
agent's own user messages (the one-time nudge, the budget notes, a working memory).
History rules, true after every step:

- The system and question messages are not in `history`.
- Each assistant message with tool calls is followed by exactly one tool message per
  call, with the same `tool_call_id`, in call order (a call that was not run gets a text
  that says so and why).
- `raw` is set only on assistant messages the model client returned.

Before the first step, the first search (plan 4.2, the "seed step"; `AGENT_FIRST_SEARCH`):
code runs the `search` tool once with the question exactly as asked and no other argument,
through the request's own tool box (`ToolBox.run_one`), so its result has this request's
D# handles: the model can read and cite them like any other. It enters `history` as one
code-written assistant message with that one call (id `FIRST_SEARCH_CALL_ID`, no `raw`:
the client sends it with the stand-in thought signature) and its tool message, so the
history rules hold. The first model call sends it, with no budget note. It is not a step;
it is one tool call (it counts toward `max_tool_calls`, see config.py). The whole call is
bounded by `first_search_timeout_s`, and it never runs into the finish reserve. If it
times out, raises, or the tool gives an error, none of it enters `history`: one user
message (`FIRST_SEARCH_FAILED_NOTE`) tells the model instead, and the run goes on. It is
skipped, with nothing added, when it is switched off, when the tool box has no `search`
tool (a finish-only session), or when no time is left before the reserve. Compaction
folds it in with the old history but never counts it as a step; its transcript labels it.

`run` repeats `step` until a `finish` call with good arguments:

- A normal step offers every tool (`tool_choice="auto"`). Its model call must end
  `finish_reserve_s` before the deadline, and lasts at most `call_timeout_s`.
- A good `finish` ends the step: the other calls of that reply are not run. Bad `finish`
  arguments give `read_finish`'s error text as that call's tool message, and the loop
  goes on. At most `max_calls_per_step` calls of one reply run.
- The next step may only call `finish` (a forced finish, `tool_choice=("finish",)`) after
  `max_steps` steps, after `max_tool_calls` tool calls, with `finish_reserve_s` or less
  left, after a failed model call, or when a reply has no tool call for the second time
  (the first time, the agent adds a nudge and goes on).
- A forced finish that fails is tried once more. If that fails too, the answer is the
  model's last text (`stop="fallback"`, no citations), or there is none and `run`
  raises `AgentError` (the server then gives its internal-error answer).

Before every model call after the first, the agent adds a budget note (run 2): one user
message in `<budget>` tags with the step count (`budget_text`: "Step 6 of 10. 4 steps left
after this one."; a forced finish gets `FORCED_NOTE` instead). It is the last message of
that prompt, and it is kept in `history` once the call returns (a failed call keeps none;
the next call adds its own). Nothing already in the prompt changes: each prompt is the
previous one, byte for byte, plus new messages, so the provider's implicit prompt cache
keeps working. (`to_gemini` sends a note after tool results as its own user Content, and
merges it with a nudge right before it.) Compaction folds old notes in like any message;
its transcript leaves them out, since it numbers the steps itself.

Before each step, the prompt's size is estimated (plan 6.4): the input tokens the model
reported for the last call, plus the rough size (`rough_tokens`) of what was added since.
Above `compact_at_tokens`, and with more than `compact_keep_steps` complete steps in
`history`, the agent compacts (plan 6.5): the last `compact_keep_steps` complete steps stay
word for word (a step is never split); everything before them (an earlier working memory
included) is written out as plain text for one model call (`prompts/compact.md`, no tools),
and replaced in `history` by one user message `<working_memory>...</working_memory>`. If
that call fails or writes nothing, the compaction is skipped: `history` stays whole (nothing
is dropped or shortened), the event says why, and compaction is tried again before the next
step. A compaction call always ends before the finish reserve, so a failed one never takes
the forced finish's time. A compaction that does not shrink the prompt is flagged `no_gain`.
Compaction calls are not steps.

Tracing: one `agent` span per run; under it one `first_search` span (its `tool:search`
span nests under it; it is there when skipped too, with the reason), one `step` span per
model call (the model client's `chat` generation and the `tool:<name>` spans nest under it)
and one `compact` span per compaction.
"""

import asyncio
import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from corpusqa import obs
from corpusqa.agent.corpus_tools import SEARCH
from corpusqa.agent.tools import FINISH_TOOL_NAME, FinishArgs, ToolBox, ToolRun, read_finish
from corpusqa.config import (
    AGENT_CALL_TIMEOUT_S,
    AGENT_COMPACT_AT_TOKENS,
    AGENT_COMPACT_KEEP_STEPS,
    AGENT_FINISH_RESERVE_S,
    AGENT_FIRST_SEARCH,
    AGENT_FIRST_SEARCH_TIMEOUT_S,
    AGENT_FORCED_FINISH_TRIES,
    AGENT_MAX_CALLS_PER_STEP,
    AGENT_MAX_OUTPUT_TOKENS,
    AGENT_MAX_STEPS,
    AGENT_MAX_TOOL_CALLS,
    AGENT_THINKING_BUDGET,
    COMPACT_CALL_MAX_OUTPUT_TOKENS,
    COMPACT_THINKING_BUDGET,
    TOKEN_CHARS,
)
from corpusqa.llm.errors import LLMError
from corpusqa.llm.interfaces import ChatModel, Message, ModelReply, ToolCall, ToolChoice, Usage
from corpusqa.prompts import PromptError, PromptText, load_text_prompt, load_tool_schemas

logger = logging.getLogger(__name__)

AGENT_PROMPT = "agent"  # prompts/agent.md: the system message
QUESTION_PROMPT = "question"  # prompts/question.md: the question message
COMPACT_PROMPT = "compact"  # prompts/compact.md: the compaction call's system message
QUESTION_SLOT = "{question}"
# agent.md's slot for the field catalog: the sources and their common fields, which the
# corpus tools learn once at start-up from the index (`CorpusTools.catalog`).
CATALOG_SLOT = "{catalog}"
# What the slot holds when the agent has no catalog (no corpus tools: tests, a finish-only
# session).
NO_CATALOG = "(none: no records are loaded)"
AGENT_SPAN = "agent"
STEP_SPAN = "step"
COMPACT_SPAN = "compact"
# The one-time nudge after a reply with no tool call (plan 6.3).
NUDGE = "Call a tool, or call finish with your answer."
# Tool messages for calls the agent did not run. Code-made texts like the ToolBox's errors,
# in the same tags, so the model reads them the same way.
NOT_RUN_AFTER_FINISH = (
    "<tool_error>Not run: finish was called in the same reply, and it ends the run.</tool_error>"
)
NOT_RUN_OVER_CAP = (
    "<tool_error>Not run: at most {cap} tool calls run per reply. Make this call again in "
    "your next reply if you still need it.</tool_error>"
)
NOT_RUN_NOT_ALLOWED = "<tool_error>Not run: only {names} may be called now.</tool_error>"
# The budget note before every model call after the first (agent.md tells the model what
# it is). Run 1 (smoke 20261017T013336Z): three runs used all 10 steps with every cited
# record read by step 3, and the model was never told how many steps it had left.
BUDGET_OPEN = "<budget>"
BUDGET_CLOSE = "</budget>"
STEP_NOTE = "Step {step} of {max_steps}. {left} left after this one."
FORCED_NOTE = "Out of budget. Only finish may be called now: answer from the records you have read."
# The working memory that replaces the old steps (agent.md tells the model what it is).
MEMORY_OPEN = "<working_memory>"
MEMORY_CLOSE = "</working_memory>"
# Said with every skipped compaction (log line and span status).
SKIPPED_NOTE = "the history is kept whole and compaction is tried again at the next step"
# The first search (see the module docstring).
FIRST_SEARCH_SPAN = "first_search"
# The id of its code-written call. The API's ids look like "call_1383909" (step 1 probe) and
# the client fills in "call_<n>", so the model's calls never have this id.
FIRST_SEARCH_CALL_ID = "first_search"
# Its one argument: the search tool's query (tools.json, `SearchArgs`).
FIRST_SEARCH_QUERY_ARG = "query"
# The user message that replaces it when it fails or times out.
FIRST_SEARCH_FAILED_NOTE = "The first search of your question could not run; search yourself."
# Where the compaction transcript puts it (it is not a step).
FIRST_SEARCH_PLACE = "before step 1"

ZERO_USAGE = Usage(input_tokens=0, output_tokens=0, thinking_tokens=0, cached_tokens=0, cost_usd=0)

# How the run ended: a good finish in a normal step, a good finish in a forced step, or no
# good finish at all (the answer is the model's last text).
type StopReason = Literal["finish", "forced_finish", "fallback"]
FINISH: StopReason = "finish"
FORCED_FINISH: StopReason = "forced_finish"
FALLBACK: StopReason = "fallback"
# Why the agent forced a finish.
type ForcedBy = Literal["step_limit", "tool_call_limit", "deadline", "model_error", "no_tool_call"]
# How a compaction ended: the compaction call wrote the working memory, or the call failed
# or was blank and the compaction was skipped (history kept whole, tried again next step).
type CompactPath = Literal["model", "skipped"]
MODEL_PATH: CompactPath = "model"
SKIPPED_PATH: CompactPath = "skipped"
# How the first search ended: its result is in the history, it failed (an error, or the
# tool's error text) or timed out (the note is in the history instead), or it did not run.
type FirstSearchOutcome = Literal["ok", "failed", "timed_out", "skipped"]
FIRST_SEARCH_OK: FirstSearchOutcome = "ok"
FIRST_SEARCH_FAILED: FirstSearchOutcome = "failed"
FIRST_SEARCH_TIMED_OUT: FirstSearchOutcome = "timed_out"
FIRST_SEARCH_SKIPPED: FirstSearchOutcome = "skipped"
# Why it was skipped (the span's `reason`).
type FirstSearchSkip = Literal["off", "no_search_tool", "blank_question", "no_time"]


class AgentError(RuntimeError):
    """The run ended without an answer: no good finish, and the model wrote no text."""


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """The limits of one run (defaults from config.py; tests and the live test shrink them)."""

    max_steps: int = AGENT_MAX_STEPS
    max_calls_per_step: int = AGENT_MAX_CALLS_PER_STEP
    max_tool_calls: int = AGENT_MAX_TOOL_CALLS
    finish_reserve_s: float = AGENT_FINISH_RESERVE_S
    call_timeout_s: float = AGENT_CALL_TIMEOUT_S
    forced_finish_tries: int = AGENT_FORCED_FINISH_TRIES
    compact_at_tokens: int = AGENT_COMPACT_AT_TOKENS
    compact_keep_steps: int = AGENT_COMPACT_KEEP_STEPS
    first_search_timeout_s: float = AGENT_FIRST_SEARCH_TIMEOUT_S

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if value < 0 or (value == 0 and name != "finish_reserve_s"):
                raise ValueError(f"AgentLimits.{name} must be positive, got {value!r}")


DEFAULT_LIMITS = AgentLimits()


@dataclass(frozen=True, slots=True)
class CompactionEvent:
    """One compaction: when, how much it saved, what it replaced, and how it ended.

    A skipped one (`path="skipped"`) replaced nothing: every step is kept word for word,
    and `reason` says why the compaction call gave no working memory.
    """

    step: int  # steps taken before it
    tokens_before: int  # the estimate that triggered it
    tokens_after: int  # the estimate right after it
    steps_replaced: int  # complete steps folded into the working memory (0 when skipped)
    steps_kept: int  # complete steps kept word for word
    path: CompactPath
    no_gain: bool  # the prompt did not get smaller (rough size); the new history is kept
    reason: str | None = None  # why it was skipped (None when the model wrote the memory)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """How one run ended. `citations` are the model's, as it gave them (not resolved)."""

    answer: str
    citations: tuple[str, ...]
    not_found: tuple[str, ...]
    stop: StopReason
    forced_by: ForcedBy | None  # why a finish was forced (None: never forced)
    steps: int  # model calls made for steps (compaction calls not counted)
    tool_calls: int  # tool calls run (calls that were not run are not counted)
    usage: Usage  # totals over every model call of the run, compaction calls included
    compactions: tuple[CompactionEvent, ...]


def question_text(question: str, template: PromptText) -> str:
    """The question message: the template with the question in its one slot."""
    if template.text.count(QUESTION_SLOT) != 1:
        raise PromptError(f"prompt {template.name!r} must hold {QUESTION_SLOT} exactly once")
    return template.text.replace(QUESTION_SLOT, question)


def system_text(template: PromptText, catalog: str) -> str:
    """The system message: the template with the field catalog in its one slot.

    A blank catalog puts `NO_CATALOG` there, so the prompt never shows an empty list.
    """
    if template.text.count(CATALOG_SLOT) != 1:
        raise PromptError(f"prompt {template.name!r} must hold {CATALOG_SLOT} exactly once")
    return template.text.replace(CATALOG_SLOT, catalog.strip() or NO_CATALOG)


def first_search_call(question: str) -> ToolCall:
    """The first search's code-written call: `search` with the question exactly as asked."""
    return ToolCall(
        id=FIRST_SEARCH_CALL_ID, name=SEARCH, arguments={FIRST_SEARCH_QUERY_ARG: question}
    )


def budget_text(step: int, max_steps: int, *, forced: bool = False) -> str:
    """The budget note before model call `step`: the step count, or (forced) `FORCED_NOTE`."""
    if forced:
        return f"{BUDGET_OPEN}{FORCED_NOTE}{BUDGET_CLOSE}"
    left = max(0, max_steps - step)
    count = "No steps" if left == 0 else "1 step" if left == 1 else f"{left} steps"
    note = STEP_NOTE.format(step=step, max_steps=max_steps, left=count)
    return f"{BUDGET_OPEN}{note}{BUDGET_CLOSE}"


def rough_tokens(messages: Sequence[Message]) -> int:
    """The rough size of messages (plan 6.4): characters of their JSON / TOKEN_CHARS.

    `raw` is left out (the provider's own copy of a turn whose content is already counted).
    """
    plain = [
        {
            "role": m.role,
            "content": m.content,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": dict(c.arguments)} for c in m.tool_calls
            ],
            "tool_call_id": m.tool_call_id,
            "name": m.name,
        }
        for m in messages
    ]
    return math.ceil(len(json.dumps(plain, ensure_ascii=False, default=str)) / TOKEN_CHARS)


class Agent:
    """One question, one object. Nothing is shared between questions.

    `deadline` is a time on `clock` (seconds): no model call runs past it. The model and
    the tool box may serve many agents at once; everything that changes lives here.
    `catalog` is the field catalog for the system message (`system_text`); blank when
    there are no corpus tools. `first_search` turns the first search on or off.
    """

    def __init__(
        self,
        question: str,
        model: ChatModel,
        tools: ToolBox,
        *,
        deadline: float,
        clock: Callable[[], float] = time.monotonic,
        limits: AgentLimits = DEFAULT_LIMITS,
        catalog: str = "",
        first_search: bool = AGENT_FIRST_SEARCH,
    ) -> None:
        self._schemas = tools.schemas()
        if FINISH_TOOL_NAME not in {schema.name for schema in self._schemas}:
            raise ValueError(f"the tool box has no {FINISH_TOOL_NAME!r} tool")
        self.question = question
        self.model = model
        self.tools = tools
        self.deadline = deadline
        self.limits = limits
        self._clock = clock
        self._system = load_text_prompt(AGENT_PROMPT)
        self._question = load_text_prompt(QUESTION_PROMPT)
        self._compact = load_text_prompt(COMPACT_PROMPT)
        self.catalog = catalog
        self.system_prompt = system_text(self._system, catalog)
        self.question_prompt = question_text(question, self._question)
        self.history: list[Message] = []
        self.replies: list[ModelReply] = []
        self.compactions: list[CompactionEvent] = []
        self.finished: FinishArgs | None = None
        self.steps = 0
        self.tool_calls = 0
        self.usage = ZERO_USAGE
        self.forced_by: ForcedBy | None = None
        self._first_search_on = first_search
        self.first_search: FirstSearchOutcome | None = None  # set before the first step
        # The token estimate's base: the input tokens the model last reported, and the rough
        # size of the prompt they were reported for (None: no count reported yet).
        self._base_tokens: int | None = None
        self._base_rough = 0
        self._steps_compacted = 0  # steps already folded into a working memory
        self._ran = False

    # ------------------------------------------------------------ prompt
    def build_prompt(self) -> list[Message]:
        """What the model is sent: system, question, then the history in order."""
        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self.question_prompt),
            *self.history,
        ]

    def time_left(self) -> float:
        return self.deadline - self._clock()

    def estimate_tokens(self) -> int:
        """The next prompt's size: the last reported input tokens + the rough size added since.

        Before any reported count, the rough size of the whole prompt.
        """
        return self._estimate(self.build_prompt())

    def _estimate(self, prompt: Sequence[Message]) -> int:
        rough = rough_tokens(prompt)
        if self._base_tokens is None:
            return rough
        return self._base_tokens + max(0, rough - self._base_rough)

    # ------------------------------------------------------------ one step
    async def step(self, tool_choice: ToolChoice) -> None:
        """One model call, then the calls of its reply (see the module docstring).

        A failed model call raises (an `LLMError`); it still counts as a step.
        """
        self.steps += 1
        number = self.steps
        timeout_s = self._call_timeout(tool_choice)
        note = self._budget_note(number, tool_choice)
        prompt = [*self.build_prompt(), *([note] if note is not None else [])]
        with obs.span(
            STEP_SPAN,
            metadata={
                "step": number,
                "tool_choice": _choice_label(tool_choice),
                "timeout_s": round(timeout_s, 3),
                "tokens_estimated": self._estimate(prompt),
                "budget_note": note.content if note is not None else None,
            },
        ) as span:
            reply = await self.model.complete(
                prompt,
                self._schemas,
                tool_choice=tool_choice,
                max_output_tokens=AGENT_MAX_OUTPUT_TOKENS,
                thinking_budget=AGENT_THINKING_BUDGET,
                timeout_s=timeout_s,
                prompt_name=self._system.name,
                prompt_version=self._system.version,
            )
            self.usage = _add(self.usage, reply.usage)
            if reply.usage.input_tokens > 0:  # 0: the provider gave no count
                self._base_tokens = reply.usage.input_tokens
                self._base_rough = rough_tokens(prompt)
            self.replies.append(reply)
            if note is not None:  # kept only now: a failed call leaves no note behind
                self.history.append(note)
            self.history.append(reply.message)
            results, not_run = await self._dispatch(reply.message.tool_calls, tool_choice)
            self.history.extend(results)
            span.update(
                output={
                    "text": reply.message.content,
                    "tool_calls": [
                        {"id": call.id, "name": call.name} for call in reply.message.tool_calls
                    ],
                },
                metadata={
                    "stop_reason": reply.stop_reason,
                    "input_tokens": reply.usage.input_tokens,
                    "finished": self.finished is not None,
                    "not_run": not_run,
                },
            )

    def _budget_note(self, number: int, tool_choice: ToolChoice) -> Message | None:
        """The note for model call `number`: none on the first call (see the docstring)."""
        if number <= 1:
            return None
        text = budget_text(number, self.limits.max_steps, forced=_is_forced(tool_choice))
        return Message(role="user", content=text)

    async def _dispatch(
        self, calls: Sequence[ToolCall], tool_choice: ToolChoice
    ) -> tuple[list[Message], list[dict[str, str]]]:
        """One tool message per call, in call order; and which calls were not run, and why."""
        texts: dict[int, str] = {}  # the calls not run: their text
        reasons: dict[int, str] = {}
        to_run: list[int] = []
        finish = _good_finish(calls)
        allowed = set(tool_choice) if isinstance(tool_choice, tuple) else None
        for index, call in enumerate(calls):
            if finish is not None:
                if index != finish[0]:
                    texts[index], reasons[index] = NOT_RUN_AFTER_FINISH, "after_finish"
                    continue
            elif allowed is not None and call.name not in allowed:
                names = ", ".join(sorted(allowed))
                texts[index], reasons[index] = (
                    NOT_RUN_NOT_ALLOWED.format(names=names),
                    "not_allowed",
                )
                continue
            elif len(to_run) >= self.limits.max_calls_per_step:
                cap = self.limits.max_calls_per_step
                texts[index], reasons[index] = NOT_RUN_OVER_CAP.format(cap=cap), "over_cap"
                continue
            to_run.append(index)
        ran = dict(zip(to_run, await self.tools.run([calls[i] for i in to_run]), strict=True))
        self.tool_calls += len(to_run)
        if finish is not None:
            self.finished = finish[1]
        messages = [
            ran[index]
            if index in ran
            else Message(role="tool", tool_call_id=call.id, name=call.name, content=texts[index])
            for index, call in enumerate(calls)
        ]
        not_run = [
            {"id": calls[index].id, "name": calls[index].name, "reason": reason}
            for index, reason in reasons.items()
        ]
        return messages, not_run

    def _call_timeout(self, tool_choice: ToolChoice) -> float:
        """A forced finish may use the reserve; any other call must end before it."""
        left = self.time_left()
        if not _is_forced(tool_choice):
            left -= self.limits.finish_reserve_s
        return max(0.0, min(left, self.limits.call_timeout_s))

    # ------------------------------------------------------------ compaction
    async def maybe_compact(self) -> bool:
        """Compact when the prompt is over the limit and there are old steps to fold in.

        True when the history was compacted. Not tried (with a log line) when no time is
        left before the finish reserve: the prompt then keeps every step, which the model
        can still read whole. A skipped compaction (`compact` got no working memory) is
        tried again at the next call, since the prompt is still over the limit.
        """
        if self.estimate_tokens() <= self.limits.compact_at_tokens:
            return False
        if _count_steps(self.history) <= self.limits.compact_keep_steps:
            return False
        if self._call_timeout("none") <= 0:
            logger.info("agent: the prompt is over the compaction limit, but no time is left")
            return False
        event = await self.compact()
        return event.path == MODEL_PATH

    async def compact(self) -> CompactionEvent:
        """Replace every step before the last `compact_keep_steps` by one working memory.

        When the compaction call fails or writes nothing, nothing is replaced: the history
        stays whole and the event (path `skipped`) says why.
        """
        keep = self.limits.compact_keep_steps
        if _count_steps(self.history) <= keep:
            raise ValueError(f"nothing to compact: at most {keep} complete steps in history")
        cut = _start_of_last_steps(self.history, keep)
        old, kept = self.history[:cut], self.history[cut:]
        first_step = self._steps_compacted + 1
        replaced, kept_steps = _count_steps(old), _count_steps(kept)
        tokens_before = self.estimate_tokens()
        rough_before = rough_tokens(self.build_prompt())
        text = transcript(old, first_step)
        with obs.span(
            COMPACT_SPAN,
            input={"steps_replaced": replaced, "steps_kept": kept_steps, "chars": len(text)},
            metadata={"step": self.steps, "tokens_before": tokens_before},
        ) as span:
            memory, problem = await self._working_memory(text)
            if memory is None:
                return self._skipped(span, tokens_before, replaced + kept_steps, problem)
            self.history[:cut] = [
                Message(role="user", content=f"{MEMORY_OPEN}\n{memory}\n{MEMORY_CLOSE}")
            ]
            self._steps_compacted += replaced
            rough_after = rough_tokens(self.build_prompt())
            if self._base_tokens is not None:
                # The reported count was for the longer prompt: take off what was removed.
                removed = rough_before - rough_after
                self._base_tokens = max(rough_after, tokens_before - removed)
                self._base_rough = rough_after
            event = CompactionEvent(
                step=self.steps,
                tokens_before=tokens_before,
                tokens_after=self.estimate_tokens(),
                steps_replaced=replaced,
                steps_kept=kept_steps,
                path=MODEL_PATH,
                no_gain=rough_after >= rough_before,
            )
            self.compactions.append(event)
            span.update(output=memory, metadata=dataclasses.asdict(event))
            if event.no_gain:
                problem = f"no_gain: the prompt did not shrink ({rough_before} -> {rough_after})"
                logger.warning("agent: compaction %s", problem)
                span.warning(problem)
        logger.info(
            "agent: compacted %d steps, %d -> %d tokens",
            replaced,
            event.tokens_before,
            event.tokens_after,
        )
        return event

    def _skipped(
        self, span: obs.Observation, tokens_before: int, steps: int, problem: str
    ) -> CompactionEvent:
        """No working memory: the history is kept whole; the event and the span say why."""
        event = CompactionEvent(
            step=self.steps,
            tokens_before=tokens_before,
            tokens_after=self.estimate_tokens(),
            steps_replaced=0,
            steps_kept=steps,
            path=SKIPPED_PATH,
            no_gain=False,
            reason=problem,
        )
        self.compactions.append(event)
        message = f"compaction skipped: {problem}; {SKIPPED_NOTE}"
        logger.warning("agent: %s", message)
        span.update(metadata=dataclasses.asdict(event))
        span.warning(message)
        return event

    async def _working_memory(self, text: str) -> tuple[str | None, str]:
        """The compaction call's text, or None and why there is none (failed, or blank)."""
        prompt = [
            Message(role="system", content=self._compact.text),
            Message(
                role="user",
                content=(
                    f"<question>\n{self.question}\n</question>\n\n"
                    f"<transcript>\n{text}\n</transcript>"
                ),
            ),
        ]
        try:
            reply = await self.model.complete(
                prompt,
                [],
                tool_choice="none",
                max_output_tokens=COMPACT_CALL_MAX_OUTPUT_TOKENS,
                thinking_budget=COMPACT_THINKING_BUDGET,
                timeout_s=self._call_timeout("none"),
                prompt_name=self._compact.name,
                prompt_version=self._compact.version,
            )
        except LLMError as exc:
            return None, f"the compaction call failed ({_what(exc)})"
        self.usage = _add(self.usage, reply.usage)
        memory = reply.message.content.strip()
        if not memory:
            return None, "the compaction call wrote nothing"
        return memory, ""

    # ------------------------------------------------------------ the first search
    async def first_search_step(self) -> FirstSearchOutcome:
        """Before the first step: one `search` with the question as asked (module docstring).

        Never raises (a cancel of the whole request still propagates): whatever goes wrong,
        the run goes on without it.
        """
        call = first_search_call(self.question)
        bound = min(
            self.limits.first_search_timeout_s, self.time_left() - self.limits.finish_reserve_s
        )
        with obs.span(
            FIRST_SEARCH_SPAN,
            input={"tool": call.name, "arguments": dict(call.arguments)},
            metadata={"tool_call_id": call.id, "timeout_s": round(max(0.0, bound), 3)},
        ) as span:
            skip = self._first_search_skip(bound)
            if skip is not None:
                logger.info("agent: no first search (%s)", skip)
                span.update(metadata={"outcome": FIRST_SEARCH_SKIPPED, "reason": skip})
                self.first_search = FIRST_SEARCH_SKIPPED
                return FIRST_SEARCH_SKIPPED
            started = time.monotonic()
            outcome, problem, ran = await self._run_first_search(call, bound)
            elapsed_s = round(time.monotonic() - started, 3)
            if ran is not None:
                self.history += [Message(role="assistant", tool_calls=(call,)), ran.message]
                self.tool_calls += 1
                shown = ran.message.content
            else:
                self.history.append(Message(role="user", content=FIRST_SEARCH_FAILED_NOTE))
                shown = FIRST_SEARCH_FAILED_NOTE
                logger.warning(
                    "agent: the first search %s (%s); the run goes on without it", outcome, problem
                )
                span.warning(f"first search {outcome}: {problem}")
            span.update(
                output=shown,
                metadata={
                    "outcome": outcome,
                    "reason": problem,
                    "elapsed_s": elapsed_s,
                    "result_chars": len(shown),
                    "counted": ran is not None,  # toward max_tool_calls
                },
            )
        self.first_search = outcome
        return outcome

    def _first_search_skip(self, bound: float) -> FirstSearchSkip | None:
        """Why the first search does not run (None: it runs)."""
        if not self._first_search_on:
            return "off"
        if SEARCH not in {schema.name for schema in self._schemas}:
            return "no_search_tool"
        if not self.question.strip():
            return "blank_question"
        if bound <= 0:
            return "no_time"
        return None

    async def _run_first_search(
        self, call: ToolCall, bound: float
    ) -> tuple[FirstSearchOutcome, str | None, ToolRun | None]:
        """How it ended, what went wrong, and the run (only when its result may be used)."""
        try:
            # The tool runs in a thread, which a timeout cannot stop: it finishes on its own,
            # and its result is dropped.
            ran = await asyncio.wait_for(self.tools.run_one(call), timeout=bound)
        except TimeoutError:
            return FIRST_SEARCH_TIMED_OUT, f"no result within {bound:g} s", None
        except Exception as exc:  # it must never cost the request its answer
            logger.exception("agent: the first search raised")
            return FIRST_SEARCH_FAILED, _what(exc), None
        if ran.error is not None:
            return FIRST_SEARCH_FAILED, f"the tool gave an error ({ran.error})", None
        return FIRST_SEARCH_OK, None, ran

    # ------------------------------------------------------------ the loop
    async def run(self) -> AgentResult:
        """Steps until a good finish; then the answer (or `AgentError`, see the docstring)."""
        if self._ran:
            raise RuntimeError("an Agent answers one question once: make a new Agent")
        self._ran = True
        with obs.span(
            AGENT_SPAN,
            input={"question": self.question},
            metadata={
                "prompts": {
                    AGENT_PROMPT: self._system.version,
                    QUESTION_PROMPT: self._question.version,
                    COMPACT_PROMPT: self._compact.version,
                    "tools": load_tool_schemas().version,
                },
                "tools": [schema.name for schema in self._schemas],
                "limits": dataclasses.asdict(self.limits),
                "time_left_s": round(self.time_left(), 3),
                # The catalog itself is in every step's chat generation (the system message).
                "catalog_chars": len(self.catalog),
                "system_chars": len(self.system_prompt),
            },
        ) as span:
            try:
                await self.first_search_step()
                result = await self._loop()
            except AgentError:
                span.update(metadata=self._totals())
                raise
            span.update(
                output={
                    "answer": result.answer,
                    "citations": list(result.citations),
                    "not_found": list(result.not_found),
                },
                metadata={"stop": result.stop, **self._totals()},
            )
            if result.stop == FALLBACK:
                span.warning("no good finish call: the answer is the model's last text")
        return result

    async def _loop(self) -> AgentResult:
        nudged = False
        forced: ForcedBy | None = None
        while True:
            forced = forced or self._limit_reached()
            if forced is not None:
                self.forced_by = forced
                return await self._forced_finish()
            await self.maybe_compact()
            forced = self._limit_reached()  # a compaction call takes time
            if forced is not None:
                continue
            try:
                await self.step("auto")
            except LLMError as exc:
                logger.warning("agent step %d: the model call failed (%s)", self.steps, _what(exc))
                forced = "model_error"
                continue
            if self.finished is not None:
                return self._result(FINISH)
            if not self.replies[-1].message.tool_calls:
                if nudged:
                    forced = "no_tool_call"
                else:
                    self.history.append(Message(role="user", content=NUDGE))
                    nudged = True

    def _limit_reached(self) -> ForcedBy | None:
        if self.steps >= self.limits.max_steps:
            return "step_limit"
        if self.tool_calls >= self.limits.max_tool_calls:
            return "tool_call_limit"
        # At exactly the reserve a normal call would get no time at all (its call must end
        # before the reserve), so it is forced then too.
        if self.time_left() <= self.limits.finish_reserve_s:
            return "deadline"
        return None

    async def _forced_finish(self) -> AgentResult:
        logger.info("agent: forcing finish after %d steps (%s)", self.steps, self.forced_by)
        for attempt in range(1, self.limits.forced_finish_tries + 1):
            if self.time_left() <= 0:
                logger.warning("agent: no time left for forced finish attempt %d", attempt)
                break
            await self.maybe_compact()
            try:
                await self.step((FINISH_TOOL_NAME,))
            except LLMError as exc:
                logger.warning("agent: forced finish attempt %d failed (%s)", attempt, _what(exc))
                continue
            if self.finished is not None:
                return self._result(FORCED_FINISH)
            logger.warning("agent: forced finish attempt %d gave no good finish call", attempt)
        return self._fallback()

    def _fallback(self) -> AgentResult:
        """No good finish: the model's last text, if it wrote any; else AgentError."""
        for reply in reversed(self.replies):
            if reply.message.content.strip():
                logger.warning("agent: no good finish; answering with the model's last text")
                return self._result(FALLBACK, answer=reply.message.content)
        raise AgentError(
            f"no answer after {self.steps} steps: no good finish call and no text from the model"
        )

    def _result(self, stop: StopReason, answer: str | None = None) -> AgentResult:
        finished = self.finished if stop != FALLBACK else None
        return AgentResult(
            answer=finished.answer if finished is not None else answer or "",
            citations=tuple(finished.citations) if finished is not None else (),
            not_found=tuple(finished.not_found) if finished is not None else (),
            stop=stop,
            forced_by=self.forced_by,
            steps=self.steps,
            tool_calls=self.tool_calls,
            usage=self.usage,
            compactions=tuple(self.compactions),
        )

    def _totals(self) -> dict[str, object]:
        return {
            "forced_by": self.forced_by,
            "first_search": self.first_search,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "compactions": sum(1 for e in self.compactions if e.path == MODEL_PATH),
            "compactions_skipped": sum(1 for e in self.compactions if e.path == SKIPPED_PATH),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "thinking_tokens": self.usage.thinking_tokens,
            "cached_tokens": self.usage.cached_tokens,
            "cost_usd": self.usage.cost_usd,
        }


# ---------------------------------------------------------------- compaction text
def transcript(messages: Sequence[Message], first_step: int) -> str:
    """Old history as plain text for the compaction call: one block per message.

    Plain text, not live tool messages, so the compaction request can never break the
    provider's rule that a tool result follows its call. Every result is kept whole. The
    first search is not a step: it is placed "before step 1".
    """
    blocks: list[str] = []
    step = first_step - 1
    place = f"step {step}"  # where the next tool result belongs
    for message in messages:
        if _is_first_search(message):
            place = FIRST_SEARCH_PLACE
            lines = [f"[{place}: the service searched the question as asked]"]
            lines += [f"call {_call_text(call)} (id {call.id})" for call in message.tool_calls]
        elif message.role == "assistant":
            step += 1
            place = f"step {step}"
            lines = [f"[{place}: the agent]"]
            if message.content.strip():
                lines.append(message.content.strip())
            lines += [f"call {_call_text(call)} (id {call.id})" for call in message.tool_calls]
        elif message.role == "tool":
            lines = [f"[{place}: result of {message.name}, call {message.tool_call_id}]"]
            lines.append(message.content)
        elif _is_memory(message):
            lines = ["[earlier working memory]", _memory_text(message)]
        elif _is_budget_note(message):
            continue  # bookkeeping: the transcript numbers the steps itself
        else:
            lines = ["[note to the agent]", message.content]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _call_text(call: ToolCall) -> str:
    arguments = json.dumps(dict(call.arguments), ensure_ascii=False, default=str)
    return f"{call.name}({arguments})"


def _is_memory(message: Message) -> bool:
    return (
        message.role == "user"
        and message.content.startswith(MEMORY_OPEN)
        and message.content.endswith(MEMORY_CLOSE)
    )


def _is_budget_note(message: Message) -> bool:
    return (
        message.role == "user"
        and message.content.startswith(BUDGET_OPEN)
        and message.content.endswith(BUDGET_CLOSE)
    )


def _memory_text(message: Message) -> str:
    return message.content[len(MEMORY_OPEN) : -len(MEMORY_CLOSE)].strip()


def _is_first_search(message: Message) -> bool:
    """The first search's code-written turn (see the module docstring)."""
    return (
        message.role == "assistant"
        and message.raw is None
        and [call.id for call in message.tool_calls] == [FIRST_SEARCH_CALL_ID]
    )


def _is_step(message: Message) -> bool:
    """A step's model turn: an assistant message the first search did not write."""
    return message.role == "assistant" and not _is_first_search(message)


def _count_steps(messages: Sequence[Message]) -> int:
    """Complete steps: each step's model turn (its tool messages always follow it)."""
    return sum(1 for message in messages if _is_step(message))


def _start_of_last_steps(messages: Sequence[Message], steps: int) -> int:
    """Where the last `steps` complete steps start (the index of their first model turn)."""
    seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if _is_step(messages[index]):
            seen += 1
            if seen == steps:
                return index
    return 0


# ---------------------------------------------------------------- helpers
def _good_finish(calls: Sequence[ToolCall]) -> tuple[int, FinishArgs] | None:
    """The first `finish` call with good arguments, and its place in the reply."""
    for index, call in enumerate(calls):
        if call.name == FINISH_TOOL_NAME:
            checked = read_finish(call)
            if isinstance(checked, FinishArgs):
                return index, checked
    return None


def _is_forced(tool_choice: ToolChoice) -> bool:
    return isinstance(tool_choice, tuple)


def _choice_label(tool_choice: ToolChoice) -> str | list[str]:
    return list(tool_choice) if isinstance(tool_choice, tuple) else tool_choice


def _add(total: Usage, usage: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + usage.input_tokens,
        output_tokens=total.output_tokens + usage.output_tokens,
        thinking_tokens=total.thinking_tokens + usage.thinking_tokens,
        cached_tokens=total.cached_tokens + usage.cached_tokens,
        cost_usd=total.cost_usd + usage.cost_usd,
    )


def _what(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
