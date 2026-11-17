---
name: Backlog — agreed work waiting to be built
description: Work the user agreed to but asked to build later, together with related work. Each item says why, what to build, where in the code, and what must stay the same. Open - 4 choose RRF k (60 or 20); 5 add --no-fields to eval-retrieval when a plain variant is needed again; 6 the Codex review fixes of the agent harness and corpus tools (#5 network errors fixed, #1 tool time, #4 contains, #9 failed calls traced thinly, the Lows); 7 check the learned date window against the evals and the created-date wording; 8 search passage size (open question); 9 accept quoted filter argument names; 10 stop sooner when the records do not hold the answer (and the step-10 note); 11 a code check of the citing rule; 13 eval a new corpus (question file option); 14 one judged eval run; 15 duplicate-call checks; 16 AI dev setup questions (ship the harness, planning records, a prompt-version lock test); 17 shared own-id values (a check that two documents do not carry the same id: flag-and-tell or strict quarantine, user decides). Done - 1 tunable fusion weights + per-question recall; 2 the agent's corpus tools; 3 field values in every chunk header; 12 removed the trace and fresh-clone-check keys. Read before starting work on retrieval, the agent or the evals. Never start an item on your own; the user says when.
---

# Backlog

Work the user agreed to but asked to build later, in one go with related work
(user, 2026-10-16: "add it as one of the task to do lets build further and than we
will build the entire thing in one go"). Do not start an item on your own.

## 4. Choose RRF k: 60 (default) or 20 (added 2026-10-16)
The fusion weight grid (item 1, tune set, 26 questions with right documents) found no
setting that clearly beats the defaults (bm25 1, dense 1, k 60). k 20 is the only
candidate: with plain headers hybrid q_recall@10 +0.019 (0.673 -> 0.692, half a
question; one question's worth is 0.038), pooled +0.013, MRR +0.008, all_gold@10 +1
question; with header fields on (production now) the same q_recall@10 (0.727) and MRR
+0.022 (0.671 -> 0.693). The user decides. To change: `RRF_K` in `config.py`, then
`uv run eval-retrieval --production` and keep its row in `eval/history.csv`. Details:
`docs/knowledge/decisions/retrieval.md` ("Fusion defaults stay bm25 1 / dense 1 / k 60").

## 5. Add `--no-fields` to `eval-retrieval` when a plain variant is needed again (added 2026-10-16)
With `CHUNK_HEADER_FIELDS = True`, `uv run eval-retrieval --chunking ...` follows the
switch and always adds header fields; there is no flag to turn them off, so the plain
variant cannot be rebuilt from the CLI without editing `config.py`. Not needed now (the
plain run is recorded in `eval/history.csv`). When it is: a `--no-fields` flag that sets
`ChunkSpec(fields=False)`, refused with `--production` (like `--fields`), variant name
without `-fields`, and a test (`tests/unit/test_retrieval_eval.py`).

## 6. Codex review of the agent harness and corpus tools: the fixes (added 2026-10-17)
Review `docs/reviews/2026-10-17-agent-harness-and-tools.md` (commit range
`21000f7c..cdc765eb`); its triage is at the end of that file and in
`docs/knowledge/decisions/agent.md`. The user chose (2026-10-17, submission prep) to fix #5
and #1 before submitting and to list the rest in SOLUTION.md. Suggested order for the rest:
#4, then #9. Each fix gets a test that fails before it; then a smoke and a tune run.
- **#5 — FIXED 2026-10-17** (submission prep): `llm/client.py` maps `httpx.TransportError`
  to `EmbeddingError` / `ChatError` (`ChatTimeout` for a timeout), not retried; tests
  through the real SDK for chat, embed, an agent step, a forced finish, a compaction and a
  search. The original note:
- **#5 [Medium] network errors skip the agent's recovery.** google-genai (httpx, no retry
  options) raises `httpx.ConnectError` / `ReadError` as they are; `GeminiChat._call` catches
  only `APIError`, the agent only `LLMError`, so one dropped connection at any step gives
  the internal-error answer, and in `search` (`_query_vector`) it fails the call instead of
  falling back to keywords. Fix: map httpx transport errors to `ChatError` /
  `EmbeddingError` in `llm/client.py`; tests for a step, a compaction, a forced finish and a
  search.
- **#1 [High] tool time is not bounded by the deadline.** Only model calls get a timeout;
  tool bodies run in `asyncio.to_thread` with no limit, and the query embedder uses the
  ingest settings (60 s HTTP timeout, 5 retries). /ask still answers 200 at 150 s, but the
  agent's answer is lost and the thread keeps running after its queue slot is freed. Fix:
  give tool calls the time left before the finish reserve; give the query embedder its own
  short timeout and at most one retry.
- **#4 [Medium] `contains` misses text and pulls in other spellings.** `loose()` uses
  `normalise()`, which cuts a value at 256 characters (`VALUE_NORM_MAX_CHARS`), silently;
  and a match selects its whole `value_norm` group, so person values that differ only by a
  bracketed role come back too (wrong count). Fix: match on the full value; select by the
  spellings that matched.
- **#9 [Medium] failed model calls are traced thinly.** A billed reply that is rejected
  (e.g. `MALFORMED_FUNCTION_CALL`) has no output on its generation; retries are counted only
  at the end. Fix: record the raw candidate and finish reason before parsing, and each retry
  as it happens.
- **Lows:** #2 the request body is read before `asyncio.timeout_at` starts (read it inside
  the deadline); #3 `SpendLedger.add` writes its JSON file on the event loop (save in a
  thread); #6 filled-in call ids `call_<n>` can collide with a provider id (skip used ids);
  #8 a JSON null shows as "absent", not "empty" (keep null rows out of noise; tests for
  null, empty string, empty list, absent); #10 the `citations` span has no per-citation
  reason (record each citation's expansion and whether it was kept, unknown or a repeat);
  and (not Codex's) the `bm25` / `dense` spans list every ranked chunk (up to ~68,000 on a
  deep page, built even with tracing off).
- **#7 rejected** (user decision 2026-10-16, "Search deeper"): search pages give no total
  until the ranking is complete.

## 7. Dates: check the learned window against the evals, and the created-date wording (added 2026-10-17)
- The user kept the learned window (data/: 2023-2030) with "lets try with the current range
  and if needed we can increase the range" (2026-10-16). No eval has checked it yet.
- The pub_0077 investigation (`eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md`)
  found a record's created date (a Slack thread's timestamp) a year off the date in its
  text, and 53% of dated Slack threads drift by more than 180 days; the model's created
  range then left the record out. Codex's ranking (`docs/reviews/2026-10-17-pub0077-fix-options.md`):
  after the first search (built), word the date as "created:" in results so the model
  does not read it as the event date; honest "no match" wording.
- Each change measured on its own (smoke, then tune), against the current tune baseline.

## 8. Search passages: 2 per document, about 300 characters (open question, added 2026-10-17)
Put to the user on 2026-10-16 (corpus tools, open question 6); no answer is recorded. Every
eval ran with it (`SEARCH_SNIPPETS_PER_DOC`, `SEARCH_SNIPPET_CHARS`). Change only on the
user's word, then a smoke and a tune run.

## 9. Accept `filter` argument names written with quotes (added 2026-10-17)
In every tune run the model wrote 5-6 of about 13 `filter` calls with its argument names
wrapped in literal quotes (`"\"field\""`, also on held-out: 6 per run, counted from the
server logs). `ConditionArgs` / `FilterArgs` (`extra="forbid"`, `agent/corpus_tools.py`)
refuse them, the model gets "Bad arguments", and a step is lost; it recovers on the next
step, so it costs time and money, not answers. Fix candidates: strip one pair of quotes
from argument names in code before checking them (in `ToolBox` or the arguments models), or
a stricter schema; a test with a quoted name; then a smoke and a tune run.

## 10. Stop sooner when the records do not hold the answer (added 2026-10-17)
The unanswerable questions (and pub_0039) search until the step limit: 17-23% of tune
questions, about 25% of the spend (T1-T5). Answers are right; only time and cost suffer.
Options: let code count repeated empty searches and say so in the budget note, or a stop
rule for repeated "nothing found". Related: at step 10 the note says "No steps left after
this one", but a forced-finish step follows (`agent/agent.py` `budget_text`): reword it
(found in the pub_0077 investigation). Each change measured on its own.

## 11. A code check of the citing rule (added 2026-10-17)
R7 (cite nothing when the whole answer is "not found"; no background records) is in the
prompt only. HIGH follows it (tune 4 of 4, held-out 7 of 7 unanswerable cite nothing); LOW
does not (2 of 4, 1 of 7). Worth it only if a cheaper setting ships. Options: drop every
citation when `not_found` covers the whole question; or a cheap relevance check per cited
record (a model call: cost).

## 13. Evaluate a new corpus: point the runner at another question file (added 2026-10-17)
`eval/run.py` reads `QUESTIONS_FILE` and `SETS_DIR` from `scripts/show_questions.py`, so it
can only run the dev questions. `scripts/show_questions.py` already takes `--questions-file`
and `--sets-dir`; the runner (and the judge calibration) need the same options. Someone must
still write gold answers and `answer_facts` for that corpus. The service itself is generic.

## 14. One judged eval run, to see the judge live (added 2026-10-17)
The answer judge was calibrated on saved tune answers only (`docs/eval-log.md`); no `uv run
eval-smoke` has run with it yet. Run one smoke eval with the judge on and check its section
in `summary.md`, `judge.jsonl` and the history row (about $0.25).

## 15. Duplicate-call checks (added 2026-10-17)
Only the prompt (R9) stops the model from repeating a call that already ran with the same
arguments. A code check that answers a repeat from the earlier result, or flags it, was
left for later by the user (2026-10-17; the run's notes call it a future improvement).

## 16. AI dev setup: three open questions for the user (added 2026-10-17)
From the AI-dev-setup task (handoff step 6, the draft doc's section 8; draft:
https://claude.ai/code/artifact/a88f2aa3-2921-4a78-b60c-4677ce9c040b).
- **How to ship the global harness** (`~/.agent-harness/`: rules, hooks such as
  `guard.py`, skills, templates) with the submission: copy it into the repo, describe it,
  or leave it live only. Never copy `~/.agent-harness/state/`: it names other job
  applications. The global harness changes were not committed in the harness's own repo
  (the user's call). `SOLUTION.md` describes the AI dev setup in the meantime.
- **Planning records:** keep today's practice (Codex reviews tracked in `docs/reviews/`,
  lasting facts promoted into `docs/knowledge/`, raw run folders git-ignored), or ship raw
  runs too.
- **A prompt-version lock test:** a test that fails when a prompt file's text changes but
  its `version:` line does not (today only the skill `prompt-change` and version pins in
  `tests/unit/test_agent.py` guard it). The `--milestone` flag from the same question is
  built (`eval-heldout`, `eval-all`).

## 17. Shared own-id values: a check that two documents do not carry the same id (added 2026-10-24)
**User's proposal (2026-10-24, ingestion walkthrough):** in a real system an id such as a Jira
ticket code is unique, so two documents with the same own-id value is a data error, and the
pipeline should check for it the way it already checks `dataset_doc_uuid` (S2 sets the second
file aside as `duplicate_uuid`). The user asked for this to be recorded for a future improvement.

**What the built index shows.** Seven Jira codes are carried by more than one file:
`SUP-18421` by 5 files (in three folders), `SUP-314159` and `SUP-4821` by 4, `SUP-209876`,
`SUP-271828`, `SUP-29438`, `SUP-863999` by 2. The two `SUP-271828` files are different tickets
(different reporter, assignee, customer, dates, summary, uuid); the generator reused the
number. Linear's `ENG-314159` is on 6 files, the only hub. The Jira own-id confidence of
0.9665 is this effect (1 - distinct ratio).

**Why the pipeline does not reject one of them today (the point to decide).** The only id the
contract promises is `dataset_doc_uuid`, and that one is checked. An own-id field is a learned
guess about a field, and each duplicate file is a real, distinct document with its own uuid
that the grader may ask about. Setting one aside would lose a citable document, and there is
no rule for which one to keep. So today S8 records the fact instead of acting on it: `doc_keys`
is many-to-many, a link to a shared key resolves to every owner (two `links` rows for
`SUP-271828` from `SUP-1000001`), a key held by more than `HUB_KEY_MAX_DOCS` (5) is a hub and
resolves to nothing, and the agent's `lookup_key` tool returns every owner plus `hub`.

**What to build (two variants; the user decides which).**
- *Flag and tell, no rejection (recommended):* in S8 keys, after `build_keys`, add a document
  flag `shared_own_id:<n>` (n = other documents with the same own-id value) and a
  `source_profile` count of shared own-id values; put the count in the `S8_keys` span output
  and in the ingest log; show "this id is held by N documents" in the `lookup_key`
  observation and in the agent's field catalog line for own_id fields, so the agent knows a
  reference is ambiguous before it cites. Documents stay; nothing resolves differently.
- *Strict check (the user's reading):* treat a shared own-id value like `duplicate_uuid`:
  keep the first file in path order, quarantine the rest with reason `duplicate_own_id`.
  Cost on data/: 7 Jira codes = at least 11 files set aside, 6 Linear files for the hub key;
  any eval question whose expected ids include a set-aside file fails. Only sound if the
  unseen corpus is known to be a clean export, which the contract does not promise.

**Measured on data/ (2026-10-24, user asked "does this break anything").** Documents whose own-id
value is shared: 223 (linear 161 across 70 codes, hubspot 29, jira 21, github 8, gmail 4). The
strict variant with "first file in path order wins" would set aside 128 of them. Checked against
the 30 tune questions (77 expected ids): two questions expect a document that would be dropped.
pub_0046 expects a HubSpot company file whose `company_id` is shared by several files and is not
first in path order. pub_0059 names `ENG-314159` in the question itself and expects one specific
file among the six that carry that code (the hub); first-wins would keep the `linear/design/`
file and drop the one the question is about. So the dataset's own questions are written against
the duplicated ids and expect the agent to pick the right file by content. The strict variant
breaks at least 2 of 30 tune questions; the held-out set was not checked (eval split rule).
`eval/history.csv` holds aggregates only and cannot show this; the check was the tune set's
expected ids against `doc_keys`. No file stem is shared by two documents (0), so the stem key
is unique today; only own-id values collide.

**Where:** `src/corpusqa/ingest/links.py` (`build_keys`: the per-key document count is already
computed as `docs_per_key`), `src/corpusqa/ingest/pipeline.py` (`_corpus_wide`, S8_keys span and
`StageRow` counts; a flag write needs a writer method next to `update_titles`),
`src/corpusqa/retrieval/keys.py` (`KeyLookup` gets an `owner_count` or the agent-facing text
says it), the field catalog builder for the own_id line, `config.py` for any new constant.
Tests: `tests/unit/test_ingest_profile.py` / `test_ingest_pipeline.py` with two mini-corpus
files sharing an invented id (never a real code from data/), and a tracing test for the span
fields. Run the smoke eval and keep its row.

**What must stay the same:** `duplicate_uuid` handling in S2; `document_ids` only from the
ingest id map; nothing hard-coded from data/ (no source or prefix names in the rule); the hub
rule and `HUB_KEY_MAX_DOCS`; every outcome still traced.

## Discussed, not added
- A reranker chosen by the eval (from the 2026-10-16 Q&A notes: a perfect one lifts
  per-question hybrid recall@10 only 0.673 -> 0.706, because it can only re-order what
  search found). User, 2026-10-16: "we will take a deciosn on it after wards". Decide
  after items 1-3: measure how often a right document is in the top 50 but not the top 10
  (the most a reranker can win). Only a Gemini key exists at grading: a Gemini rerank call
  per search, or a local model (new dependency, model download).
- A search switch that looks only at field text (`only_fields=True`): rejected. The
  filter finds the same documents exactly (all of them, with a count); a field-only
  search returns at most the top 50, with near-misses and no count. Item 3 also leaves no
  separate field chunks to point such a switch at.

## Done
- **1. Tunable fusion weights and per-question recall in the retrieval eval** — done
  2026-10-16, commits 1b6c1207..389a814e, grid bfb0e636, review fixes aac0c168,
  678373cf, ba616c5a; task `docs/runs/2026-10-16-eval-weights-headers/`. Weighted RRF
  (`RRF_BM25_WEIGHT`, `RRF_DENSE_WEIGHT` = 1.0 in `config.py`), `eval-retrieval`
  flags `--bm25-weight --dense-weight --rrf-k` with a run-name suffix, and per-question
  recall `q_recall@k` as the headline (`hybrid_q_recall@10`). Production (plain headers)
  reproduced pooled 0.377 and gave per-question 0.673. The weight grid kept the
  defaults (item 4 above). Knowledge: `docs/knowledge/decisions/retrieval.md`,
  `docs/knowledge/gotchas/eval.md`.
- **2. Agent tools: filter on fields, match values loosely, look up keys, search inside a
  result set** — done 2026-10-16, commits 8bb6f4f4..5fab1a9e (task
  `docs/runs/2026-10-16-corpus-tools/`); wired into the agent and the server the same day
  (`dcbe13e1`, task `docs/runs/2026-10-16-agent-tools/`). Built as planned, with the user's
  later answers: every list pages with its total and next offset (no `FIELD_READ_MAX_DOCS`
  cap); citing an R# cites every id (no `CITE_MAX_DOCS`; the server keeps at most 500 ids,
  and logs and traces a cut); the field list is the catalog in the system prompt;
  `person_norm` moved to `store/norms.py`; each date judged on its own. Knowledge:
  `docs/knowledge/architecture/corpus-tools.md`, `decisions/retrieval.md`,
  `decisions/dates.md`, `decisions/store.md`, `decisions/agent.md`.
- **3. Field values in every chunk's header** — done 2026-10-16, commits 5a8d9633,
  a00296b2 (build), e47e9021 (eval), e78ffcc9 (switched on); same task. On in
  production (`CHUNK_HEADER_FIELDS = True`): tune set, hybrid q_recall@10 0.673 -> 0.727, pooled recall@10 0.377 -> 0.403, MRR 0.588 ->
  0.671, all_gold@10 0.615 -> 0.692 (bm25 0.556 -> 0.603, dense 0.635 -> 0.696); the
  variant cost $6.35 to embed. Knowledge: `docs/knowledge/decisions/chunking.md`,
  `decisions/store.md`, `architecture/ingestion.md`, `models/index-schema.md`.
- **12. `uv run trace` and `uv run fresh-clone-check`** — removed 2026-10-17 (user's call,
  submission prep): both keys were never built, so they failed. Read a trace in the Langfuse
  UI or through the v2 Observations API (`docs/knowledge/architecture/tracing.md`); the
  fresh-clone contract check is covered by `tests/contract/test_serve_script.py`.
