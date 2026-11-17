---
name: Codex review — the automatic answer judge (eval/judge.py, eval-judge-calibrate)
description: Independent read-only Codex review (gpt-6-astra, effort high, 2026-10-17) of commit 576870ca, the eval's answer judge and its calibration command. Six findings, all fixed with tests in the next commit — 1 high (an error from held-out judging kept its full text on the judge_all span), 5 medium (a transport error crashed the whole judging; the spend cap did not stop the other answers' calls; a reply with two verdict calls and "false" as a boolean were accepted; billed-but-unusable calls missing from the judge cost; calibration checked folders but not symlinked input files, and named unknown ids). No defect found in the majority / tie / unjudged logic, the aggregates, kappa, the cache key or the price change. Read before changing eval/judge.py, the runner's judge wiring or the calibration.
---

# Codex review: the automatic answer judge (2026-10-17)

Command: `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high" -o <file> -`
with the prompt below on stdin, run from the worktree `.claude/worktrees/agent-tools` at
`576870ca`. Codex used 104,730 tokens. It only read files (`git show`, `nl`, `sed`, `rg`,
`cat`); it ran no tests, no eval and no model call. It was told not to open `.env`, the
question file, the held-out set or results, or `docs/runs/`; its command log shows it kept to
that (one `git status --short` lists folder names only).

The `-o` file held only Codex's last message (a note that it changed nothing). The findings
below are its answer, copied unchanged from the Codex log.

Triage (each finding's entry is in the run's `decisions.md`, Area: review): all six
accepted and fixed, each with a test that failed before the fix.

<details><summary>Prompt (abridged: the off-limits files are named generally, and the design
points are summarised in brackets)</summary>

```text
You are reviewing one commit in this repository (read-only). Make NO code changes, run no
model calls, start no server, and do not run the eval.

Commit: 576870ca ("Add an automatic answer judge to the eval, and a calibration command").
See it with `git show 576870ca` (and `git show 576870ca --stat`). Read in place:
eval/judge.py, eval/judge_calibrate.py, eval/run.py, eval/questions.py,
src/corpusqa/prompts/judge.md, src/corpusqa/llm/client.py (the ChatPrices / prices /
from_settings change), src/corpusqa/config.py (the JUDGE_* block), tests/unit/test_judge.py,
tests/unit/test_ask_eval.py (the "answer judge" section at the end), AGENTS.md (rules; above
all "Eval split" and "Tracing").

Off limits: `.env`, the question file, the held-out set file, the held-out results folder,
and `docs/runs/`. You may read one or two files under
`eval/results/tune/ask/20261017T072027Z/` (answer_check.md) to see the hand-check rules the
judge prompt is meant to follow.

[The prompt then restated the seven points of the approved design (fact by fact; the Pro
model and its prices; one forced `verdict` call matching a Pydantic model, retry once then
unjudged; two judgments and a third on a split; citations stay with the runner; runner
wiring with aggregates only for held-out and `all`, own ledger and cap, `chat` generations
under `judge` spans; the calibration command on 5 tune runs) and asked for numbered
findings, most severe first, each with severity, file:line evidence, a concrete failing
path and a fix, looking hardest at: held-out per-question material reaching any readable
file, printed line, log, history row or trace (generations, exception messages, the
calibration cache); the majority / tie-break / unjudged logic and aggregate denominators;
the retry logic and the spend cap; whether the empty contextvars.Context really stops
per-question tracing; the cache key; the calibration's folder restriction, comparison and
kappa; callers billing at the wrong price after the GeminiChat change; tests that do not
test what they claim.]
```

</details>

## Codex's findings (unchanged)

1. **High — Held-out exception text leaks into traces.** [eval/judge.py:325](.claude/worktrees/agent-tools/eval/judge.py:325): an exception from the empty-context task crosses the outer `judge_all` span, whose exit handler records its full message. For example, `RuntimeError("detail about q_0004")` reaches the trace before the runner sanitizes it. **Fix:** sanitize exceptions inside the protected tracing boundary. The test at `tests/unit/test_ask_eval.py:777` checks only summary JSON; tracing tests also omit `status_message`.

2. **Medium — Failed transport calls can abort the entire batch without retry.** [eval/judge.py:454](.claude/worktrees/agent-tools/eval/judge.py:454) catches only `LLMError`; `GeminiChat._call` catches only SDK `APIError`. A transport exception such as `httpx.ConnectError` escapes, discarding completed judgments instead of retrying once and marking that question unjudged. **Fix:** normalize transport failures into `ChatError`, preserving the separate `SpendCapReached` handling; test this path.

3. **Medium — Spend-cap failure does not stop sibling tasks.** [eval/judge.py:347](.claude/worktrees/agent-tools/eval/judge.py:347): `asyncio.gather` propagates an exception without cancelling siblings. After one reservation exceeds the cap, another queued, smaller request can still start before `asyncio.run` shuts down. **Fix:** set a shared stop flag immediately on `SpendCapReached`, prevent further calls, and cancel/drain outstanding tasks. Existing cap tests use one item and miss this.

4. **Medium — Invalid multi-call replies are accepted.** [eval/judge.py:163](.claude/worktrees/agent-tools/eval/judge.py:163) filters calls and validates only the first `verdict`. A reply containing contradictory `correct` and `wrong` calls is silently scored from the first. Pydantic also accepts coercions such as `"false"` for the schema's boolean. **Fix:** require exactly one total tool call named `verdict`, enforce strict field types, and route violations through retry/unjudged.

5. **Medium — Judge cost excludes some billed failures.** [eval/judge.py:456](.claude/worktrees/agent-tools/eval/judge.py:456): costs accumulate only when `complete()` returns. However, `client.py:642` charges the ledger before `from_gemini()` can raise `BadChatAnswer`. Two unusable, billed responses therefore produce an unjudged result reporting zero cost. **Fix:** retain usage on billed exceptions or report ledger deltas; test an unusable response carrying usage.

6. **Medium — Calibration checks directory containment, but not input-file targets.** [eval/judge_calibrate.py:158](.claude/worktrees/agent-tools/eval/judge_calibrate.py:158): an allowed tune directory containing a symlinked input file can read held-out material. An unknown hand-check ID is then included verbatim in `CalibrationError`, reaching logs and the enclosing trace. **Fix:** validate resolved input-file paths and use generic errors for rejected IDs. The restriction test checks directory names only.

No additional defects found in majority/tie/unjudged selection, normal aggregate denominators, confusion matrices, kappa, current cache-key inputs, or price propagation. The empty context suppresses ordinary child spans/generations; the exception boundary is the gap.

Static review only. No changes, model calls, server, tests, or evals run; prohibited files were not read.

## What was done with each finding

| # | decision | fix |
|---|---|---|
| 1 | accepted | Untraced judging turns any error (other than the spend cap, whose text names no question) into `JudgeFailed(<type name>)` inside the `judge_all` span; the runner shows that type. Tests: `test_a_protected_judge_failure_reaches_the_trace_as_its_type_only`, and the runner test now checks every recorded `status_message`. |
| 2 | accepted, fixed in the judge | A call that raises anything but `SpendCapReached` counts as a failed attempt (retried once, then unjudged), so a transport error no longer ends the judging. Not normalized in `GeminiChat` itself: that would change the agent's client outside this task. Test: `test_a_transport_error_is_a_failed_call_not_a_crash`. |
| 3 | accepted | The first `SpendCapReached` sets a stop on the judge; every later call of that judge raises it without being made. Test: `test_the_spend_cap_stops_the_other_answers_too` (1 call, not 3). |
| 4 | accepted | Exactly one tool call, named `verdict`; `StrictStr` and `StrictBool` fields (the JSON schema sent is unchanged). `fact` stays a lax int on purpose: a JSON number may come back as 2.0. Test: `test_a_reply_must_hold_exactly_one_verdict_call_with_strict_types`. |
| 5 | accepted | The runner reports the judge's spend as the growth of the ledgers while it ran (`judge.spend_usd`, "judge ledger growth" in summary.md, `judge_spend=` in the history row), next to the replies' sum. Test: `test_the_judge_spend_is_the_growth_of_its_ledger`. |
| 6 | accepted | Each input file is checked after resolving symlinks; an id outside the tune and smoke sets is not named in the error. Test: `test_calibration_refuses_input_files_that_lead_outside_the_tune_runs`. |
