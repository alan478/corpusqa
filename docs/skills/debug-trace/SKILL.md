---
name: debug-trace
description: Find why one answer was wrong, starting from its failure label and trace, and choose what to fix. Use when an eval question fails or an /ask answer looks wrong.
---
# Debug one answer from its trace
0. Tune questions only (AGENTS.md: Eval split). See the question and its gold
   answer with `uv run questions-tune --id <question_id>`. Never debug a
   held-out question.
1. Get the server's trace id: it is in that question's row of
   `eval/results/<set>/ask/<time>/per_question.jsonl`, or in the `X-Trace-Id` header
   of the /ask response (the body holds only `answer` and `document_ids`). The
   judge's view of the answer is the same question's row in `judge.jsonl` there.
2. Read the trace. No project command reads it (`uv run trace` was never built).
   - Langfuse UI: open the trace id in the project (dev runs with the `LANGFUSE_*` keys
     in `.env`; without keys nothing was traced).
   - Read-back: the v2 Observations API (`docs/knowledge/architecture/tracing.md`,
     "Reading a trace back"), from a read-only test file run with `uv run tests-live
     <file>` (it loads `.env`; never print metadata unfiltered: it holds the public key).
     `input`/`output` arrive as JSON strings: decode them first.
   What to look at: `ask` (outcome, queue) > `agent` (steps, stop, forced_by,
   first_search, prompt versions, cost) > `first_search`, `step` (budget note, tokens),
   `chat` (the full prompt as sent, tool calls, thinking), `tool:<name>` (arguments given
   and checked, handles, the text the model saw), `compact`; then `citations` and
   `answer_handles` under `ask`.
3. Name the failure (these labels are for the diagnosis; the eval does not assign them):

| Label | Look at | Usual fix |
|---|---|---|
| retrieval_miss | `search` spans (`bm25`, `dense`): is the doc in either ranking? | chunk text, header fields, query handling, tool guide |
| ranked_out | fused ranks vs page size and depth | fusion weights / k, passages |
| shown_not_used, read_not_cited | steps after the doc's D# appeared | tool guide or rules in `agent.md`, the tool's text |
| citation_dropped | `citations` span: citations given vs ids resolved | handle expansion in `agent/handles.py` (test first) |
| wrong_fact | claim vs doc text | answer rules (status, versions, newest) |
| over_cited | cited records that state nothing asked | the citing rule (R7), result-set use |
| stopped_early, too_many_steps | stop, forced_by, budget notes, time per step | caps, deadline reserve, stop rule, first search |
| wrong_abstain, hallucinated | what the tools showed vs the answer | not-found rules |
| error | the span marked ERROR | code fix + regression test |

4. Log symptom, cause and fix in the run's `learnings.md`. Code at fault:
   write a failing test first. Then `run-eval` on smoke.

A fix must make sense for any corpus. Never special-case one question.
