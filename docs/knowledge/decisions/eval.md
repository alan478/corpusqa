---
name: Eval decisions
description: Decisions about the eval split (tune 30 / held-out 49, seeded once by scripts/make_eval_split.py; the split rule written once in AGENTS.md, re-shown by a SessionStart hook and enforced by a read guard; both user decisions) and the end-to-end /ask eval (eval/run.py, `uv run eval-smoke|eval-tune|eval-heldout|eval-all`): set-based metrics only (precision, recall, F1, hit, all_gold, exact; no MRR or nDCG, user decision); groups single / multi / unanswerable by expected-id count; unanswerable = correct when no ids come back, left out of the means; a reply off the contract (or a timeout) is a failed reply that never abstains; --milestone required for heldout and all; protected runs write per_question.jsonl only to the guarded results folder and keep aggregates everywhere else (summary, stdout, history row, trace spans); --set may be given once; /health checked once (exit 3); concurrency 1-4 refused, not clamped; httpx.AsyncClient + semaphore; its own question loader; defaults in config.py; spend = growth of every ledger; .env loading and stand-in tracer settings; the guard blocks a raw `--set all`; the Codex review answers. One open question: eval/results/ is tracked by git. The retrieval eval's decisions are in retrieval.md; the automatic answer judge (gemini-3.1-pro-preview, prompt judge.md v1, fact by fact, two judgments per answer and a third on disagreement, strict schema, invalid output retried once then unjudged, held-out judged untraced with aggregates only, its own ledger and cap; calibrated at 99.3% fact agreement, kappa 0.90) and the six fixed Codex findings on it.
source: runs/2026-10-16-ask-eval-runner, runs/2026-10-16-agent-tools, runs/2026-10-14-ai-dev-setup
---

# Eval decisions

How the end-to-end eval of `POST /ask` works is in `../architecture/eval.md`. Decisions
about the retrieval-only eval (`uv run eval-retrieval`) are in `retrieval.md`.

## Open for the user (not decided)
- **eval/results/ is tracked by git.** If a held-out or `all` run is committed, its
  per-question file enters git history, where `git show` can print it without naming the
  held-out word (so the guard cannot see it). Option: git-ignore the guarded results
  folder (`eval/results/heldout/`). Raised 2026-10-16, ask-eval-runner task.

## 2026-10-17 Codex review of the answer judge: six findings, all fixed
- **Area:** review
- **Review:** `docs/reviews/2026-10-17-answer-judge.md` (gpt-6-astra, read-only, commit
  `576870ca`). Fixes in `ef258a2a`, each with a test that failed before it.
  1. High, accepted: an error from held-out (untraced) judging kept its full text on the
     `judge_all` span. Now it leaves as `JudgeFailed(<type name>)` inside the span.
  2. Medium, accepted (in the judge, not in `GeminiChat`): a transport error that is not an
     SDK `APIError` ended all judging. Now any call failure but the spend cap is a failed
     attempt: retried once, then the question is unjudged.
  3. Medium, accepted: the spend cap did not stop the other answers' calls. The first
     refusal now stops every later call of that judge.
  4. Medium, accepted: a reply with two verdict calls, or "false" as a boolean, was
     accepted. Now exactly one call, and `StrictStr` / `StrictBool` (the schema sent is the
     same). `fact` stays a lax int (a JSON number may come back as 2.0).
  5. Medium, accepted: calls billed but unusable were missing from the judge's cost. The
     runner now also reports the ledgers' growth while the judge ran (`judge.spend_usd`).
  6. Medium, accepted: calibration checked folders but not symlinked input files, and named
     unknown ids in its error. Now each file is checked after resolving; ids are not named.

## 2026-10-17 An automatic answer judge for the eval (fact by fact, gemini-3.1-pro-preview)
- **Area:** eval / llm
- **Chose:** `eval/judge.py` with the prompt `src/corpusqa/prompts/judge.md` (v1), the design
  the user approved ("okay build it that way and add the second one as the model": the
  Pro-class model of the probe, `gemini-3.1-pro-preview`, $2.00 / $0.20 / $12.00 per 1M,
  `JUDGE_*` in `config.py`).
  - Per answer the judge gets the question, the gold answer, the numbered `answer_facts`,
    whether the records hold an answer (code knows: no expected ids), and the answer text;
    never the run or the setting that wrote it.
  - One forced `verdict` call whose schema is the Pydantic `Verdict`, inlined: per fact
    right / partly / missing / wrong with the answer's words; contradicting claims;
    says_not_found; the verdict correct / partly / wrong. The rules are the hand checks',
    written generally; no question, answer or id is in the prompt.
  - An invalid output is asked for again once; a second one leaves the question unjudged
    (counted, never guessed). Two judgments per answer, a third when the verdicts differ, the
    majority counts. A request with no answer is judged by code. Thinking budget 1,024 (no
    level); output cap 1,024 + 4,096.
  - The runner judges after the /ask calls (`--no-judge` skips it; no key: "not run").
    Tune and smoke: `judge.jsonl` and a not-correct table. Held-out and `all`: aggregates
    only; the per-question work runs in an empty contextvars context, so no `judge` span and
    no `chat` generation (which carry gold answers) is traced, only `judge_all`.
  - Its own ledger `.cache/spend/judge.json`, cap $10. `GeminiChat` takes `prices`;
    `from_settings` takes model, prices, thinking level and ledger, so the SDK stays in
    `llm/`. Calibration only: a cache in `.cache/judge/` (the runner never uses it: it would
    keep held-out results outside the guarded folder).
- **Over:** a judge on the agent's own model (it would grade its own family's style); one
  judgment per answer (not steady); parsing free text; showing the judge the citations (code
  scores those); tracing held-out judge calls.
- **Calibration:** v1 passed the bar (fact agreement >= 90%, agreed with the user) on the
  first round: 586 of 590 facts (99.3%), 149 of 150 verdicts, kappa 0.900 / 0.968, 0
  unjudged, $2.78 (`eval/results/judge-calibration/20261017T090030Z/`). `JUDGE_CONCURRENCY`
  raised from 4 to 6 after round 1 ran 6 at once with no 429.
- **Why:** answer text was checked only by hand; this puts it in every eval run, at about
  $0.01 a call (~$0.65 a tune run, ~$1 a held-out run).

## 2026-10-16 Codex review of the /ask eval runner: triage
- **Area:** review (review file `docs/reviews/2026-10-16-ask-eval-runner.md`)
- **Review:** gpt-6-astra, effort high, diff main...HEAD at 92196ea5. 2 High, 1 Medium.
  All three fixed or partly fixed, tests first (each new test failed before its fix).
  1. **High, question spans of protected sets held per-question results — PARTLY ACCEPTED,
     fixed.** The brief asked for one span per question with its id and metrics, and that
     stays for smoke and tune. For heldout and all a span now keeps only status, latency
     and a failed flag (no question id, server trace id, error text or scores; a failed
     span says only that the reason is in the guarded file). Why partly: the spans stay
     (the timeline and failures are still visible), only their per-question content goes,
     because Langfuse is readable by agents and AGENTS.md keeps held-out per-question
     results in the guarded folder only. Test:
     `test_protected_runs_trace_no_per_question_results`.
  2. **High, an allowlisted smoke/tune key could select a protected set — ACCEPTED,
     fixed in the runner.** `uv run eval-smoke --set all --milestone m1` passed the guard
     (a key) and the permission allowlist, and argparse kept the last `--set`. `--set` now
     refuses a second value (`SetOnce`), so every key runs only its own set. The suggested
     guard-side check is not needed: the keys that run protected sets (eval-heldout,
     eval-all) are not allowlisted, so they always ask the user. Test:
     `test_a_key_cannot_switch_to_another_set` (smoke and tune keys, three overrides).
  3. **Medium, deeply nested JSON aborted the whole run — ACCEPTED, fixed.** `json.loads`
     raises RecursionError, which escaped `gather`; it now reads as "body is not JSON" and
     the run goes on. Test: `test_deeply_nested_json_is_a_failed_reply_and_the_run_goes_on`.

## 2026-10-16 The guard also blocks a raw run of the runner on the `all` set
- **Area:** eval / split guard (`scripts/hooks/guard_split.py`)
- **Chose:** a Bash pattern `eval[./]run ... --set all` next to the existing "heldout" word
  pattern; `uv run eval-all` (a key, not in the permission allowlist, so it asks the user)
  stays the only way. Tests in `test_split_guard.py`.
- **Over:** leaving it open (a raw `python -m eval.run --set heldout` was already blocked,
  but `--set all` sends the same held-out questions and was not).
- **Why:** the new set holds held-out questions; the rule "run held-out only at a milestone
  the user names" must cover it the same way.

## 2026-10-16 eval-* keys load .env; the tracer gets stand-in server settings
- **Area:** eval / tracing
- **Chose:** eval-smoke/tune/heldout/all load `.env` like eval-retrieval and serve-dev, so
  dev Langfuse keys reach the `eval_ask` trace. `tracing_settings` builds Settings with
  `Settings.from_env` and stand-ins for CORPUS_DIR (repo root) and PORT (the URL's port):
  the runner is a client and needs neither.
- **Over:** `env` defaults in commands.json (CORPUS_DIR=data/corpus) and a plain
  `Settings.from_env()` (fails without them, and names a corpus the client never reads).
- **Why:** tracing must work for every run; the client must not depend on server variables.

## 2026-10-16 Readable outputs of heldout and all hold aggregates only
- **Area:** eval / split
- **Chose:** for heldout and all, per_question.jsonl goes to the guarded
  `eval/results/heldout/ask/<time>/`; summary.md, summary.json, the printed lines, the trace
  output and the history row hold aggregates only (no failed-question list). Progress lines
  show counts only for every set. per_question rows never hold the question text or the
  expected ids (only how many). A second run in the same second gets `<time>-2`.
- **Over:** listing failed questions in every summary; per-question progress lines.
- **Why:** AGENTS.md: agents never read held-out per-question results, and agents may read
  the readable summary and the run's stdout.

## 2026-10-16 --milestone is required for heldout and all
- **Area:** eval / CLI
- **Chose:** `--set heldout` and `--set all` (the protected sets, `PROTECTED_SETS` in
  `eval/questions.py`) are argparse errors without a non-blank `--milestone <name>`. The
  milestone goes into the report and the history row's variant (`full@<milestone>`).
- **Over:** letting any run of those sets go ahead.
- **Why:** AGENTS.md: held-out runs happen only at a milestone the user names, and the
  history must say which milestone each row belongs to.

## 2026-10-16 Check GET /health once before the first question
- **Area:** eval
- **Chose:** a non-200 or failed /health stops the run with exit 3 and a clear message; no
  report and no history row.
- **Over:** asking every question and reporting all of them as connection errors.
- **Why:** a run against a server that is not up says nothing about the system and would
  add a misleading zero row to the history.

## 2026-10-16 A reply off the contract is a failed reply
- **Area:** eval
- **Chose:** like the grader (CONTRACT §3-4): non-200, a body over 1 MiB, not JSON, no
  `answer` string or a blank one, or `document_ids` not a list of strings is a failure; it
  scores 0 and does not abstain. A timeout (`asyncio.wait_for` around the whole request,
  180 s) and a transport error are failures too.
- **Over:** reading whatever ids a malformed body holds.
- **Why:** the final test must count exactly what the grader counts as a failed answer.

## 2026-10-16 Concurrency outside 1-4 is refused, not clamped
- **Area:** eval / CLI
- **Chose:** `--concurrency` must be 1-4 (`EVAL_ASK_MAX_CONCURRENCY`, CONTRACT §4); a bad
  value is an argparse error. Also: `--timeout` finite and > 0; `--url` http(s) with a host.
- **Over:** silently clamping to 4.
- **Why:** a silent change of a setting hides a typo; the report must show what was asked.

## 2026-10-16 httpx.AsyncClient with a semaphore; no new dependency
- **Area:** eval / HTTP client
- **Chose:** `httpx.AsyncClient` (already a dev dependency; eval/ is dev tooling, the
  service never imports it), one task per question behind an `asyncio.Semaphore`, and a
  connection limit equal to the concurrency. Tests use `httpx.MockTransport`.
- **Over:** threads with urllib (no test transport), FastAPI TestClient (sync, one at a time).
- **Why:** I/O-bound requests; MockTransport lets tests script replies, delays and errors,
  and contextvars carry the trace into each task, so question spans nest under `eval_ask`.

## 2026-10-16 A loader of its own; kind and history helpers reused
- **Area:** eval
- **Chose:** `eval/questions.py` reads smoke, tune, heldout and all (every row of the
  question file) and is used only by `eval.run`; `scripts.show_questions.load_questions`
  is unchanged (still refuses all but tune/smoke). The group rule is
  `scripts/make_eval_split.kind` itself. `HISTORY_HEADER`, `ROOT` and `git_commit` come from
  `eval.retrieval`, so both writers share one history header.
- **Over:** widening `load_questions` (the agents' reader); copying the kind rule.
- **Why:** user instruction; one rule for groups and one header for the shared history.

## 2026-10-16 Defaults live in config.py, which rebuilds the index once
- **Area:** eval / config
- **Chose:** `EVAL_ASK_URL`, `EVAL_ASK_CONCURRENCY` (2), `EVAL_ASK_MAX_CONCURRENCY` (4),
  `EVAL_ASK_TIMEOUT_S` (180), `EVAL_ASK_LATENCY_PERCENTILES` (50, 95), `EVAL_DIR`,
  `EVAL_RESULTS_DIR`, `EVAL_HISTORY`, `EVAL_SPEND_DIR` in config.py.
- **Over:** constants in eval/run.py.
- **Why:** user instruction (no magic numbers in eval/run.py). config.py is in the index
  fingerprint, so the next start rebuilds the index (~48 s with every vector cached).

## 2026-10-16 Spend = growth of every ledger during the run
- **Area:** eval / spend
- **Chose:** sum `cost_usd` of every `.cache/spend/*.json` (read through `SpendLedger`)
  before and after; the delta is the run's spend. No ledger = $0; an unreadable one = n/a.
- **Over:** only the embedding ledger.
- **Why:** the agent will add an LLM ledger next to it; any spend of this checkout while
  the run is going (the server's) counts.

## 2026-10-16 Groups by expected-id count; the unanswerable rule
- **Area:** eval / metrics
- **Chose:** each question's group comes from the split's rule
  (`scripts/make_eval_split.kind`): 0 expected ids = unanswerable, 1 = single, 2+ = multi.
  The report shows means for single, multi and all answerable (single + multi). An
  unanswerable question is correct when the reply returns no ids (`abstained`); its group
  shows the abstain rate and it is left out of the precision/recall/F1 means. A failed
  reply never abstains (it scores 0 when answerable and counts as not abstained when
  unanswerable). The headline (history `value`) is the mean per-question recall over the
  answerable questions.
- **Over:** scoring unanswerable questions with precision and recall (recall has no
  expected ids to divide by).
- **Why:** user decision 2026-10-16 (the task brief): unanswerable questions are correct
  when no ids come back and stay out of the means; one group rule for the split and the
  eval.

## 2026-10-16 Set-based metrics only: no MRR, no nDCG (user decision)
- **Area:** eval
- **Chose:** precision, recall, F1, hit (any expected id returned), all_gold (every expected
  id returned), exact set match; for unanswerable questions the abstain rate (no ids
  returned = correct). No rank-based metric and no per-question gold ranks; summary.md has
  no "ranking uses the response's id order" sentence.
- **Over:** the first brief's MRR and nDCG over the order of `document_ids`.
- **Why:** user decision 2026-10-16: the response's id list has no real order (the answer
  cites what it relies on; CONTRACT does not rank citations), so a rank metric would measure
  noise. The change came before any metric code was written, so nothing had to be removed.

## 2026-10-14 Eval split rule: written once, re-shown, enforced (user request)
- **Area:** eval
- **Chose:** one rule section in `AGENTS.md` (between `split-rule` markers) printed by a
  SessionStart hook on start, resume, clear and compaction for Claude Code and Codex; a read
  guard (hook + permission deny rules + `.ignore`) on the full question file and all held-out
  material; `uv run questions-tune` as the only question reader. Built as
  `scripts/hooks/split_reminder.py` and `guard_split.py` (`../architecture/dev-harness.md`).
- **Over:** a rule line in `AGENTS.md` only.
- **Why:** a rule read once can be lost after compaction or in a long session. Re-showing it
  at every session start and blocking the files makes forgetting hard. Shell access is best
  effort: a recursive grep over the whole repo can still reach the file; the rule and the
  reminder cover that. (The guard's shell patterns were widened later: `../gotchas/tooling.md`,
  2026-10-16.)

## 2026-10-14 Eval split: tune 30 / held-out 49 (user decision)
- **Area:** eval
- **Chose:** 30 tune questions (the 10-question smoke set + the 3 worked examples read while
  planning + 17 sampled, stratified by kind), 49 held-out; a seeded script
  (`scripts/make_eval_split.py`, seed 20261014) that refuses a second run; the set files
  locked by the edit guard; held-out run only at user-named milestones.
- **Over:** plan v4's ~50 tune / ~29 held-out.
- **Why:** the user wants a small tuning set and a large untouched test set, so the held-out
  score is a fair estimate before the grader's unseen corpus. Questions already studied go
  to tune so held-out stays unseen.
