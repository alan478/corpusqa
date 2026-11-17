---
name: Eval tooling — how the /ask question eval works
description: The end-to-end question eval as built (eval/run.py, eval/ask_metrics.py, eval/questions.py; keys eval-smoke, eval-tune, eval-heldout, eval-all): its inputs (a running server, a question set, CLI options with defaults in config.py), the flow (one /health check, POST /ask per question with a semaphore and a per-question limit, contract checks, set-based scoring of the ids by group, then the answer judge), its outputs (summary.md, summary.json, per_question.jsonl, judge.jsonl, and where protected runs put them), the eval/history.csv row it adds (with a real example), its trace, and what is not built yet. Also the answer judge (eval/judge.py, prompt judge.md v1, gemini-3.1-pro-preview, fact by fact against answer_facts, two judgments per answer) and its calibration (eval/judge_calibrate.py, key eval-judge-calibrate; 99.3% fact agreement, kappa 0.90 on 150 hand-checked tune answers). Read before changing the eval runner, the judge, its report, its history row or the eval keys. The retrieval-only eval is in ../decisions/retrieval.md.
source: runs/2026-10-16-ask-eval-runner, runs/2026-10-16-agent-tools
---

# Eval tooling

Two evals write rows to the same `eval/history.csv`:
- **Retrieval eval** (`uv run eval-retrieval`, `eval/retrieval.py`): ranks chunks with no
  answer model; tune and smoke sets only. Its design and metrics are in
  `../decisions/retrieval.md`.
- **Question eval** (this file): asks the running server every question, scores the ids it
  returns, and has the answer judge check each answer text. It is the final test shown to
  graders, not a tuning tool.

## Question eval (/ask)

### Parts
| File | Role |
|---|---|
| `eval/run.py` | CLI, HTTP client, report files, history row, trace. `python -m eval.run`. |
| `eval/ask_metrics.py` | Pure functions: `score_ids` (set metrics), `judge` (one reply), `summarize` (group means, abstain rate, latency percentiles, error count). |
| `eval/questions.py` | `load_run_questions(set)`: the runner's own loader (smoke, tune, heldout, all). Used only by `eval.run`. Groups come from `scripts/make_eval_split.kind`. |
| `eval/judge.py` | The answer judge (below): `Judge.judge_all`, the `Verdict` tool schema, `summarize` (`JudgeTotals`), its summary.md section and history text. |
| `eval/judge_calibrate.py` | `uv run eval-judge-calibrate`: re-judges hand-checked tune runs and compares with their `answer_check.json` (below). |
| `src/corpusqa/prompts/judge.md` | The judge's prompt (version 1). The `Verdict` field descriptions are sent with the tool, so they are part of it. |
| `src/corpusqa/config.py` | `EVAL_ASK_*` defaults (URL, concurrency 2 of max 4, 180 s per question, latency percentiles 50 and 95), `EVAL_*` paths (results, history, spend ledgers) and `JUDGE_*` (model, prices, caps, judgments, concurrency 6, ledger, cache, calibration bar 0.90). |
| `commands.json` | Keys `eval-smoke`, `eval-tune`, `eval-heldout`, `eval-all`, `eval-judge-calibrate`. Each fixes its `--set` and loads `.env` (dev Langfuse keys, the Gemini key for the judge). |
| `scripts/hooks/guard_split.py` | Blocks agent shell commands that name the held-out set, and a raw `eval.run --set all`. |

### Inputs
- **The running server** at `--url` (default `http://localhost:8080`; start it with
  `uv run serve-dev`).
- **A question set** (`--set`, given once): smoke 10, tune 30, heldout 49, all 79. The
  questions come from `data/questions.jsonl` and the id lists in `eval/sets/`. heldout
  and all are the protected sets: they need `--milestone <name>` (only at a milestone the
  user names).
- **Options:** `--concurrency` 1-4 (refused outside), `--timeout` seconds > 0, `--system
  full` (`baseline` is not built and exits with a message), `--no-judge` (skip the answer
  judge; it costs money).

### Flow
1. Parse and check the options. Load the questions. Read the spend ledgers
   (`.cache/spend/*.json`) for the "before" total.
2. `GET /health` once. Not 200, or no answer: exit 3, no report, no history row.
3. `POST /ask {"question": ...}` per question. An `asyncio.Semaphore` keeps at most
   `--concurrency` in flight; `asyncio.wait_for` limits each request to `--timeout`, up to
   the last byte of the body.
4. `read_reply` checks each body against CONTRACT §3-4. Non-200, over 1 MiB, not JSON, no
   `answer` string or a blank one, `document_ids` not a list of strings, a timeout or a
   transport error: a failed reply. It scores 0 and never abstains. Returned ids are
   deduplicated.
5. `judge` scores each reply by its group:
   - **single** (1 expected id) and **multi** (2+): precision, recall, F1, hit (any
     expected id returned), all_gold (every one returned), exact (same set). No rank
     metrics: the id list has no real order (user decision).
   - **unanswerable** (0 expected ids): correct when no ids come back (abstained). Left out
     of the means; the report shows the abstain rate.
6. `summarize` gives means for single, multi and all answerable, the abstain rate,
   latency p50, p95 and max, and the error count. Spend is the growth of the ledgers,
   measured before the judge runs.
7. The answer judge judges every answer text (below). Without a Gemini key it does not run
   and the report says so; if it stops, the /ask results are still reported.
8. Write the report files and one history row. Print a short summary.

### Outputs
In `eval/results/<set>/ask/<UTC time>/` (a second run in the same second gets `-2`):
- `summary.md`: for graders. A table of the metrics per group, the abstain rate,
  latency, errors, spend, the command, commit, set and milestone. For smoke and tune it
  also lists the failed questions with their status and error.
- `summary.json`: the same aggregates and the settings (and, for smoke and tune, the
  failed-question list `error_list`).
- `per_question.jsonl`: one row per question: id, kind, how many expected ids, HTTP
  status, latency, answer, returned ids, the server's trace id (from the `X-Trace-Id`
  response header), error, metrics. Never the
  question text or the expected ids themselves.
- `judge.jsonl` (smoke and tune only): one row per question: the verdict and every
  judgment's verdict, whether the first two agree, each fact's status and evidence (by fact
  number, never the fact text), contradictions, says_not_found, calls and cost.
  summary.md and summary.json add the judge's aggregates and, for smoke and tune, the
  questions it did not mark correct.

For the protected sets (heldout, all), `per_question.jsonl` goes to the guarded
`eval/results/heldout/ask/<time>/`, and nothing per question of the judge is written.
Everything an agent can read (summary.md, summary.json, stdout, the history row, the trace)
holds aggregates only. Progress lines show counts only, for every set.

### The history row
One row in `eval/history.csv`, same columns as the retrieval eval (`HISTORY_HEADER` in
`eval/retrieval.py`): `timestamp, git_commit, kind, set, variant, questions, headline,
value, metrics, cost_usd, seconds, report`. `kind` is `ask`; `variant` is `full`, or
`full@<milestone>`; the headline is `recall` (mean per-question recall over the answerable
questions). The real smoke row from the placeholder server (it returns no ids, so every
score is 0; the repeated multi and answerable metrics are cut to `...` here):

```
2026-10-16T22:46:15+00:00,92196ea5,ask,smoke,full,10,recall,0.0000,single(n=3) precision=0.000 recall=0.000 f1=0.000 hit=0.000 all_gold=0.000 exact=0.000 | multi(n=6) ... | answerable(n=9) ... | unanswerable(n=1) abstain=1.000 | errors=0 p50_s=0.001 p95_s=0.003 max_s=0.003,0.0000,0.1,eval/results/smoke/ask/20261016T224615Z
```

The history row's `metrics` cell ends with the judge's part once the judge has run:
`judge(<model> v<prompt version>) correct=... partly=... wrong=... unjudged=... facts(n=...)
right=... partly=... missing=... wrong=... contradictions=... said_not_found=...
agreement=... judge_cost=$... judge_spend=$...` (`eval/judge.py: history_text`), or
`judge <why it did not run>`.

### Tracing
One trace `eval_ask` per run (its input is the options; its output is summary.json
without `error_list`), with one `question` span per request. For smoke and tune a span
holds the question id, kind, status, latency, returned and hit counts, the server's trace
id, the error and the scores. For heldout and all it holds only status, latency and a
failed flag. The judge adds a `judge_all` span with its aggregates; for smoke and tune one
`judge` span per answer with its `chat` generations under it; for heldout and all nothing
per question (the judge's inputs hold gold answers). Without Langfuse keys tracing is off.
The runner is a client: its tracer settings use stand-ins for `CORPUS_DIR` and `PORT`.

## The answer judge (`eval/judge.py`)
Built and calibrated 2026-10-17 (user: "okay build it that way and add the second one as
the model").
- **Model:** `JUDGE_MODEL = "gemini-3.1-pro-preview"` (a Pro-class model, not the agent's
  Flash, so it does not grade its own family's style), through `GeminiChat` with the
  judge's prices, its own ledger `.cache/spend/judge.json` (cap `JUDGE_SPEND_CAP_USD` $10)
  and a thinking budget of 1,024 (no level).
- **Input per answer:** the question, the gold answer, the numbered `answer_facts`, whether
  the records hold an answer at all (code knows: no expected ids), and the answer text.
  Never which run or setting wrote it, and never the citations (code scores those).
- **Output:** one forced `verdict` tool call (Pydantic `Verdict`, strict strings and
  booleans): per fact right / partly / missing / wrong with the answer's words; claims that
  contradict the gold answer; `says_not_found`; the verdict correct / partly / wrong.
- **Steady:** two judgments per answer (`JUDGE_JUDGMENTS`); a third when the verdicts
  differ; the majority counts (three different: unjudged). An invalid output is asked for
  again once, then the question is unjudged: counted, never guessed. A request with no
  answer is judged by code (every fact missing, verdict wrong). The first spend-cap refusal
  stops every later call of that judge.
- **Calibration** (`uv run eval-judge-calibrate`, tune runs only, never calls /ask):
  re-judges each hand-checked run's `per_question.jsonl` and compares with its
  `answer_check.json`: fact-status and verdict agreement, confusion matrices, Cohen's
  kappa, every disagreement, the judge's own consistency, cost. Bar: fact agreement at least
  `JUDGE_CALIBRATION_BAR` (0.90). Judgments are cached in `.cache/judge/` (key: model,
  prompt text and version, tool schema, caps, input, judgment number); the runner never
  uses that cache. Report in `eval/results/judge-calibration/<time>/` (report.md,
  report.json).
- **Result** (round 1, prompt `judge` v1, the 5 hand-checked tune runs T1-T5: 150 answers,
  590 facts): fact agreement 99.3% (586 of 590), verdicts 149 of 150, kappa 0.900 on facts
  and 0.968 on verdicts, 0 unjudged, $2.78
  (`eval/results/judge-calibration/20261017T090030Z/report.md`; `docs/eval-log.md` "Answer
  judge calibration"). Known weakness: it can call a fact `missing` where the answer only
  names the thing. About $0.01 per call: ~$0.65 per tune run, ~$1 per held-out run.
- Decisions: `../decisions/eval.md` (the judge and its Codex review).

### Not built yet
- `--system baseline` (the one-shot RAG baseline; `uv run eval-baseline` exits with a
  message).
- A trace reader (a `uv run trace` key was planned and removed on 2026-10-17). Open
  a server trace id in Langfuse, or read it back through the v2 Observations API
  (`tracing.md`).
- An option to point the runner at another question file and its sets (a new corpus):
  `eval/run.py` reads `QUESTIONS_FILE` and `SETS_DIR` from `scripts/show_questions.py`
  (`docs/backlog.md`).
- No judged /ask run yet: the judge was calibrated on saved answers only.

Decisions behind this design: `../decisions/eval.md`. Traps found while building it:
`../gotchas/eval.md`.
