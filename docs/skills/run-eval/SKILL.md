---
name: run-eval
description: Run the eval (retrieval, smoke, tune, held-out or the whole golden set) against a running server and read the report. Use after any change to prompts, tools, retrieval, chunking or caps, and before calling such a change done.
---
# Run the eval
1. Start the server on the dev corpus if it is not up:
   `uv run serve-dev`; wait for `curl -fs localhost:8080/health`. The runner checks
   `/health` once and stops (exit 3, no report) when the server is not ready.
2. Pick the set. Retrieval-only change: `uv run eval-retrieval`.
   Prompt, tool or agent change: `uv run eval-smoke` first, then `uv run eval-tune` if smoke holds.
   `uv run eval-heldout --milestone <name>` only at a milestone the user names.
   `uv run eval-all --milestone <name>`: the whole golden set (79) for the graders, only
   when the user names the milestone. Never use it to tune.
   Options: `--concurrency N` (1-4, default 2), `--timeout S` (default 180 s per question),
   `--no-judge` (skip the answer judge: it costs about $0.65 per tune run).
3. The one-shot baseline (`uv run eval-baseline --set <set>`) is not built yet: it exits
   with a message.
4. Read `eval/results/<set>/ask/<time>/summary.md`: the table per group (single-doc,
   multi-doc, all answerable: precision, recall, F1, hit, all_gold, exact; ids compared as
   sets), the unanswerable abstain rate (no ids returned = correct), errors, latency, spend,
   and the answer judge's section (verdicts correct / partly / wrong / unjudged, fact
   statuses, contradictions, "said not found" rate, judge agreement and cost;
   `docs/knowledge/architecture/eval.md`).
   Smoke and tune: `per_question.jsonl` beside it has each question's status, answer,
   returned ids, server trace id, error and metrics; `judge.jsonl` the judge's verdict and
   fact statuses per question. Open a wrong one by its server trace id in Langfuse (`uv run
   trace` is not built; skill `debug-trace`).
   Held-out and all: read only the printed lines and the `eval/history.csv` row. Their
   per-question results live under the guarded held-out results folder: never read them.
5. Compare with the last `eval/history.csv` row of kind `ask` for the same set (headline
   `recall`: mean per-question recall over the answerable questions). A difference of one
   question or less: run once more before deciding.
6. Write set, scores, cost and regressed question ids (tune and smoke only) in the
   handoff, and an entry in `docs/eval-log.md`. Keep or revert by the keep-rule in
   AGENTS.md.

Never: tune on heldout or all; copy gold answers or expected ids anywhere; edit history rows.
