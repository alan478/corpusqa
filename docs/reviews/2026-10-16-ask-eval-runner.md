---
name: Codex review — end-to-end /ask eval runner
description: Independent read-only Codex review (gpt-6-astra, effort high, 2026-10-16) of git diff main...eval/ask-runner at 92196ea5 (eval/run.py, eval/ask_metrics.py, eval/questions.py, eval keys, split guard): 2 High findings (per-question spans of protected sets, a smoke/tune key could select a protected set through an extra --set) and 1 Medium (deeply nested JSON aborted the run). Triage in the run's decisions.md (Area: review). Read before changing the /ask eval runner, its report or the split guard.
---

# Codex review: end-to-end /ask eval runner (2026-10-16)

Command: `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high" -o <file> -` with the prompt below on stdin (scope: `git diff main...HEAD` at 92196ea5, steps 1-6). Output pasted unchanged.

<details><summary>Prompt</summary>

You are an independent senior reviewer. Review one branch of this repo. This is a READ-ONLY review: make NO code changes, write NO files, run no command that changes the repo, its caches or git state. Do not run `uv run eval-*` commands, do not start the server, and never open data/questions.jsonl, eval/sets/heldout.json or anything under eval/results/heldout/ (AGENTS.md: Eval split; a hook blocks them). Running `uv run tests-unit` or reading files is fine.

## What to review
The diff `git diff main...HEAD` on the checked-out branch `eval/ask-runner`: a new end-to-end eval runner that asks the RUNNING server every question of a set through `POST /ask` and scores the returned `document_ids` against each question's expected ids. Files:
- `eval/run.py` (new): CLI (`--set smoke|tune|heldout|all --url --system full|baseline --milestone --concurrency --timeout`), a GET /health pre-check, `httpx.AsyncClient` + `asyncio.Semaphore` + `asyncio.wait_for` per question, `read_reply` (contract checks of the body), report files (summary.md, summary.json, per_question.jsonl), the `eval/history.csv` row, tracing (`eval_ask` trace, one `question` span per request), spend from `.cache/spend/*.json`.
- `eval/ask_metrics.py` (new): set-based metrics (precision, recall, F1, hit, all_gold, exact; unanswerable = no ids returned), groups, aggregates, nearest-rank percentile.
- `eval/questions.py` (new): the runner's own question loader (allows heldout and all; used only by eval.run).
- `src/corpusqa/config.py` (new EVAL_ASK_* and EVAL_* path constants).
- `commands.json`, `pyproject.toml` (new key `eval-all`, descriptions, .env loading).
- `scripts/hooks/guard_split.py` (new pattern: a raw `eval.run --set all` is blocked) and `tests/unit/test_split_guard.py`.
- `tests/unit/test_ask_eval.py` (new), `docs/skills/run-eval/SKILL.md`.

## Rules and intent (read first)
- `AGENTS.md`: the eval split rule at the top (agents never read held-out questions, gold answers, expected ids or per-question results; readable outputs of held-out runs must hold aggregates only), hard rules, tracing rules, tests rules.
- `CONTRACT.md` §3-4: the /ask body shape, 180 s per question, up to 4 requests at once, 1 MiB body; what counts as a failed answer.
- User decisions: rank-based metrics (MRR, nDCG) are OUT because the response's id list has no real order; only set metrics. Unanswerable questions (no expected ids) are correct when no ids come back and are left out of the precision/recall/F1 means. Groups use the split's kind rule (`scripts/make_eval_split.kind`). `scripts/show_questions.load_questions` must keep refusing everything but tune and smoke. `--milestone` is required for heldout and all. For heldout and all, per_question.jsonl goes under `eval/results/heldout/ask/<time>/` (guarded), and the readable summary keeps aggregates only.
- The runner's design choices and their reasons: `docs/runs/2026-10-16-ask-eval-runner/decisions.md` and `brief.md`.

## Look especially for
1. Metric bugs: wrong precision/recall/F1/all_gold/exact on edge cases (duplicates, empty returns, failed replies, empty groups), wrong group means or abstain rate, percentile off by one.
2. Leaks of held-out per-question information into anything an agent can read: summary.md, summary.json, stdout/stderr, log lines, exception messages, the history row, the trace output, error messages from the loader.
3. Concurrency and timeout correctness: is the limit really to the last byte; can more than `--concurrency` requests be in flight; does one failure or a cancelled task lose the whole run; is the health check sound.
4. Contract checks in `read_reply`: anything the grader counts as a failed answer that the runner accepts, or the reverse.
5. Report and history correctness: file paths, folder collisions, relative paths, the headline value, CSV columns matching `HISTORY_HEADER`.
6. Tracing: spans nested under the trace, no question text/answer/expected ids/secrets in observations, flush at exit.
7. Test gaps: a behaviour above with no test, or a test that would pass with the bug present.
8. Rule breaks from AGENTS.md (magic numbers, hard-coded data, pathlib/encoding, boundaries).

## Output
Numbered findings, most severe first. For each: severity (High / Medium / Low), the evidence (file:line), what goes wrong (a concrete input or scenario), and a suggested fix. Say plainly when something is fine; do not pad. End with a one-line overall verdict.

</details>

## Output

1. **High — Held-out results leak through trace spans.**  
   Evidence: [eval/run.py:199](eval/run.py:199), [eval/run.py:246](eval/run.py:246). Every question span records its ID, individual scores, expected/hit counts, errors, and server trace ID—including for `heldout` and `all`. Removing `error_list` from the parent trace does not remove these child observations. This exposes protected per-question results outside the guarded files.  
   **Fix:** propagate the protected-set policy into tracing and export only aggregate results for those sets. Add a protected-set test inspecting **every observation**; the existing privacy test checks files/stdout, while the tracing test uses tune only.

2. **High — Allowlisted commands can select protected sets without the intended approval boundary.**  
   Evidence: [eval/run.py:569](eval/run.py:569), [scripts/hooks/guard_split.py:74](scripts/hooks/guard_split.py:74). Extra arguments are appended to command defaults, and argparse accepts the last `--set`. Thus `uv run eval-smoke --set all --milestone m1` selects all questions. The guard exempts recognized project keys, and the smoke command is already permission-allowlisted. A supplied milestone string does not establish user authorization.  
   **Fix:** enforce each named eval command’s fixed set and reject conflicting overrides; inspect protected-set selection before granting the guard exemption. Add regression tests for protected-set overrides through smoke/tune commands.

3. **Medium — One malformed response can abort the entire run.**  
   Evidence: [eval/run.py:141](eval/run.py:141), [eval/run.py:215](eval/run.py:215), [eval/run.py:240](eval/run.py:240). A response containing several thousand nested JSON arrays is below 1 MiB but makes `json.loads()` raise `RecursionError`. Only `ValueError` is caught. The exception escapes through `gather`, losing the report and history row instead of counting that answer as failed.  
   **Fix:** handle JSON recursion failures as invalid replies. Add a mixed-response test proving that the malformed question fails while other results and the report survive.

The set metrics, duplicate handling, abstention rules, empty-group means, nearest-rank percentiles, CSV column mapping, folder collision handling, trace nesting, and shutdown path look correct by inspection.

Validation: read-only review completed; targeted unit tests could not start because `uv` required cache write access. No evals or server were run.

**Overall verdict: request changes for held-out isolation and per-response failure containment.**