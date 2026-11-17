---
name: Codex opinion — why pub_0077 is a false "not found", and which fix to build
description: Independent read-only Codex opinion (gpt-6-astra, effort high, 2026-10-17) on tune question pub_0077, a false "not found" in all three tune runs. Codex agrees with the diagnosis (the model's own query words and exact phrases; then shown but not read; not chunking) with two corrections: a date fix alone would not have surfaced the record (without the range it ranks 92-114), and step 10 was not a forced finish. Its ranking: seed search with the question text first; then created-date wording; then honest "no match" wording plus reading a fitting hit before a not-found; batching; the tool-side date warning, per-source date drift and more steps last. Measure on a fresh v5/v6 baseline, one change at a time, paired tune runs. Claude's notes at the end (checked against the code; adds the misleading step-10 budget note and how to build the seed). Read before building the seed step, or changing search wording, the created-date rule or the step limit.
---

# Codex opinion: pub_0077 false "not found", and which fix to build (2026-10-17)

The user asked for one Codex opinion on this question ("idk ask codex once see what it thinks
for this"). It is not a code review: nothing was built, and no code, prompt or setting was
changed.

Command: `codex exec --sandbox read-only -m gpt-6-astra -c model_reasoning_effort="high" -o <file> -`
with the prompt below on stdin, run from the worktree `.claude/worktrees/agent-tools` at
`598b543d` (no other agent was editing it). Codex used 85,035 tokens. It only read files
(`rg`, `sed`, `cat`); it ran no tests, no eval and no model call.
The output is pasted unchanged. This time the `-o` file held the whole answer: it matches the
last message in the Codex log word for word (the Stop hook ran after it but added no message).

Files given, read in place: `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md`;
`docs/eval-log.md` (sections "Tune runs T1-T3" and "Runs 1-4 at a glance");
`src/corpusqa/prompts/agent.md` (v5), `tools.json` (v6) and `question.md`;
`src/corpusqa/agent/agent.py`; `src/corpusqa/agent/corpus_tools.py`;
`src/corpusqa/retrieval/search.py`; `src/corpusqa/store/sqlite.py` (`match_expression`);
`src/corpusqa/ingest/dates.py`; `src/corpusqa/config.py`; `AGENTS.md`; and the design plan
`docs/runs/2026-10-14-rag-system-plan/plan.md` §4.2 (git-ignored, so Codex read a copy in the
scratchpad; §4.2 is also quoted in the prompt). Codex was told not to open `.env`, the question
file, the held-out set or results, or `docs/runs/`, and to read at most one corpus record. The
log shows it kept to that: the only `data/` file it read is the pub_0077 record.

<details><summary>Prompt</summary>

You are an independent senior reviewer of a retrieval-augmented QA agent. Give your opinion on ONE design question: why the agent answered "not found" on one tuning question in all three runs, and which fix to build. This is a READ-ONLY task: make NO code changes, write NO files, run no command that changes anything (no `uv`, no tests, no server, no git commands that change state, no network). Read files with `rg`, `sed -n`, `cat`; a small `python3 -c` on a string is fine.

Never open: `.env` or any secret file; `data/questions.jsonl`; `eval/sets/heldout.json`; anything under `eval/results/heldout/`; anything under `docs/runs/`. (The project's eval split: only the 30 "tune" questions may be read or tuned on; the 49 held-out questions must never be seen.) You may read one corpus record if you want to see it: `data/corpus/slack/finance/1712349999-neptunehealth-metering-discrepancy-credit-approval.json`. Do not search or scan `data/` beyond that file.

Your working directory is the repo (a git worktree on branch `agent/tools`, HEAD `598b543d`). No one else is editing it. Cite evidence as repo-relative `path:line`.

## The system in short
An HTTP service answers questions about one company's internal records (Slack, Gmail, Jira-like tickets, docs, CRM, transcripts, PRs). Ingest builds an SQLite index: chunks with a header, an FTS5 keyword index (BM25, unicode61 tokenizer, no stemming), and one embedding per chunk. A ReAct agent (Gemini flash model, max 10 steps, up to 4 tool calls per step, 25 tool calls in all) calls tools: `search` (hybrid BM25 + dense, fused by RRF, 8 documents per page, optional source / created range / within-set), `filter`, `field_values`, `read_fields`, `read_document`, `lookup_key`, `list_fields`, `finish`. Each tool result names documents by per-request handles (D#, R#). The answer cites documents; the grader scores the answer text and the cited ids.

## The question and the record
Tune question pub_0077 (tune set: allowed to read):
"In March 2025 our scaling system briefly tracked both a customer's old and new machine groups while their workloads were being moved, so they got billed twice for a few days of reserved compute. How big was the refund, and what did the account team offer on top as a make-good?"
The answer is in one Slack thread: a $14,200 credit to the customer, and one month of free dedicated capacity offered as goodwill by Customer Success. The thread's own timestamp (its created date) is 2024-04-05; its text talks about a March 2025 invoice. All three runs (thinking HIGH, LOW, off; they differ only in the thinking setting) answered that the records hold no such incident.

## Files to read (all in the working directory unless noted)
1. `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md` - the full investigation: the record as indexed, the word gap, all 25 searches of the three runs with the record's BM25 / dense / hybrid ranks, the created-date trap measured across sources, what was ruled out, and the fix options. Read this first and in full.
2. `docs/eval-log.md` - section "Tune runs T1-T3" (scores, steps, latency and cost per setting, the problems every setting has) and the table "Runs 1-4 at a glance" (smoke runs).
3. `src/corpusqa/prompts/agent.md` (v5) and `src/corpusqa/prompts/tools.json` (v6): the system prompt and the tool schemas. NOTE: the three tune runs used agent.md v4 and tools.json v5. v5 / v6 changed only the citing rule (cite only records that state what was asked; a whole-question not-found cites nothing) and "no D#/R# handles in the answer text". `src/corpusqa/prompts/question.md` is the question message.
4. `src/corpusqa/agent/agent.py` - the loop: prompt is always [system, question, *history]; budget note each step; limits; forced finish.
5. `src/corpusqa/agent/corpus_tools.py` - the tools; `_search` (around line 244), `_query_vector` (around line 406: the WHOLE query string, quotes and OR included, is embedded), `_search_head`, `_keyword_text`, `_doc_line` (each hit shows only the created date).
6. `src/corpusqa/retrieval/search.py` - scoped hybrid search, paging, the "meaning only" mark, how a created range filters on the document's own created date.
7. `src/corpusqa/store/sqlite.py` - `match_expression` and `_exact_parts` (around line 924-977): exact mode, quoted text is one phrase, OR/AND dropped, every other word matches on its own.
8. `src/corpusqa/ingest/dates.py` - how created/updated are picked from a record's date FIELDS (`pick_dates`); dates inside the text play no part.
9. `src/corpusqa/config.py` - the limits and settings (`AGENT_*`, `SEARCH_*`, `RRF_*`, `RETRIEVAL_TOP_CHUNKS`).
10. The design plan (not in git; a copy): (a local scratch copy), section 4.2 "Seed step (code, no LLM)": "Before the first model turn, code runs `search(question)` and `lookup_id` on any key-like tokens in the question. The results are the first observation." It was planned but never built. Section 4.4's tool guide says "start from the seed".
Also useful: `AGENTS.md` (the project rules).

## Project rules that constrain any fix
- Graded on an UNSEEN corpus of the same format. Nothing learned from `data/` may be hard-coded: no synonym lists of corpus words, no source names, field names, company names. Anything per-corpus must be learned at ingest.
- 150 s internal deadline per request; up to 4 requests at once; no state shared between requests.
- Cost per question matters (now $0.031 thinking off to $0.058 HIGH; about $0.005-0.008 per model step).
- Nothing may be dropped silently: when a tool leaves something out (a scope, a cap, a dropped citation) it must tell the model.
- Eval split: tune on the 30 tune questions only (10 of them are the smoke set). Never tune on held-out results.
- Code does what code can do exactly; the model decides what to look up and writes the answer. Prompt changes bump a version.

## What I want from you
1. **Diagnosis.** Do you agree with the investigation's diagnosis: (a) the model's own search words (quoted phrases in the question's vocabulary, while the record uses other words), (b) the created-date trap (every created range the model used was March-April 2025; the record's created date is 2024-04-05), (c) shown but not read at the end of the step budget (off run, step 8 of 10), and NOT chunking? Anything missed or wrong? Check claims against the code (for example: what the dense side embeds when the query is full of quotes and ORs; what each hit line shows the model; how the tool and prompt word "no document contains").
2. **Rank the fix options** from the investigation, with risk and expected effect for each:
   - seed search with the question text before step 1 (code, plan 4.2), or a prompt rule to search the question in its own plain words first;
   - a prompt rule about created dates vs dates in the text (2a);
   - a tool-side warning when out-of-range documents rank above the in-range ones (2b);
   - ingest-side learning of per-source drift between created dates and dates in the text, shown in the source list / dated results (2c);
   - "not found" wording that says the records may use other words, plus a rule to read the best fitting hit before a whole-question "not found" (3);
   - more steps or more batching (4).
   Then propose anything better that stays generic. Consider, among others: query rewriting by the model (e.g. asking for a plain-words query separate from exact phrases); searching without quotes as a fallback (the investigation tested this on all 25 queries: it would not have surfaced the record); embedding a cleaned query (quotes and ORs stripped) or combining the question text with the model's query on the dense side; showing a second date on a hit when its text's dates differ from its created date; anything else.
3. **Measurement.** How to measure the chosen fix on the tune set (30 questions, run-to-run variation of about one question per run, ~$0.9-1.7 per run) without overfitting to this one question. What to track besides the score (steps, cost, latency, abstains on the 4 unanswerable questions, citation precision).
4. **Risks.** How the fix could hurt other questions: anchoring on look-alikes from a seeded page, more steps, more tokens and cost, the 150 s deadline with 4 requests at once, prompt-cache hit rate, unanswerable questions citing a seeded look-alike.

## Output
Plain Markdown, no preamble, no closing remarks.
`## 1. Diagnosis` (agree / partly / disagree, then the points, each with `path:line` evidence where it rests on code).
`## 2. Fix options ranked` (a numbered list, best first; for each: what to build in one or two sentences, expected effect on this question and on others, risk, cost, and confidence).
`## 3. Better or additional ideas` (only ones that stay generic under the rules above).
`## 4. Recommendation` (what to build first, second, and what not to build; 3-6 sentences).
`## 5. How to measure it` (concrete).
`## 6. Risks to other questions` (concrete).
Be direct. Do not pad. Say where you are unsure.

</details>

## Output

## 1. Diagnosis

**Agree with the main diagnosis, with two important corrections: removing the date filter alone would not recover this record, and step 10 did not force the model to finish.** I checked the code and the permitted corpus record; the retrieval ranks below are the investigation’s reported measurements, not independently rerun measurements.

- **The main failure is loss of useful context when constructing queries.** The complete question retrieves the record at hybrid rank 1, while shortened, phrase-heavy searches usually miss it. Exact matching makes this worse: quoted text becomes one tokenized phrase; unquoted words are alternatives; uppercase `OR` and `AND` outside quotes are discarded. Even unquoted hyphenated tokens remain phrases, so removing quotation marks is not necessarily full relaxation. Evidence: `src/corpusqa/store/sqlite.py:940`, `src/corpusqa/store/sqlite.py:957`; full-question result: `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:172`.

- **Dense search receives the entire model-written query, including quotes and operators.** `_query_vector` passes `[query]` directly to the embedder. However, this establishes the input, not that punctuation caused the poor dense ranks: fragmentary concepts and missing relationships could matter more. The unquoted off-run opening query also failed, and the investigation’s quote-relaxation experiment did not recover the record. Evidence: `src/corpusqa/agent/corpus_tools.py:406`; `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:157`, `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:193`.

- **The created-date trap is real and affects both rankers.** Ingest excludes body fields from date candidates and chooses created/updated from scalar field dates, with a filename fallback. Search filters by `summary.created` before ranking. Each hit then displays that date without an explicit `created:` label. The record’s body explicitly discusses March 2025 despite its older timestamp fields. Evidence: `src/corpusqa/ingest/fields.py:27`, `src/corpusqa/ingest/dates.py:153`, `src/corpusqa/retrieval/search.py:162`, `src/corpusqa/retrieval/search.py:264`, `src/corpusqa/agent/corpus_tools.py:468`, `data/corpus/slack/finance/1712349999-neptunehealth-metering-discrepancy-credit-approval.json:14`.

- **The date counterfactual overstates what date-only fixes would achieve.** Adding the target record to a small date-filtered population produces ranks 3, 5 and 3. Removing the filter restores all competing documents; those three queries then rank it 114, 103 and 92. Thus, “repeat without the range” or “show two excluded hits” is useful general protection, but not a demonstrated rescue here. Evidence: `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:130`, `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:142`, `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:145`.

- **The off run has a candidate-selection failure under budget pressure.** D53 was shown at step 8; the model read a look-alike at step 9. Its date may explain that choice, but the trace does not establish why. Crucially, step 10 still allows ordinary tools; forced finish happens after ten ordinary steps. It could also have read both candidates in step 9. The recorded runs had zero forced finishes. Evidence: `src/corpusqa/agent/agent.py:608`, `src/corpusqa/agent/agent.py:635`, `src/corpusqa/config.py:429`, `src/corpusqa/agent/tools.py:96`, `docs/eval-log.md:28`.

- **The negative-search wording encourages an unjustified conclusion.** “No document … contains the searched words, phrases or keys” reports failure of the parsed keyword expression, not absence of an incident. A document can contain individual words from an unmatched quoted phrase. “Only close in meaning” can also make an exact semantic answer sound inferior. The schema repeats this framing, while the prompt permits corpus-wide absence claims after looking where the answer should be. Evidence: `src/corpusqa/agent/corpus_tools.py:579`, `src/corpusqa/prompts/tools.json:11`, `src/corpusqa/prompts/agent.md:45`.

- **Chunking is not the demonstrated problem; presentation still contributes.** The investigation reports one complete indexed chunk with a vector. But search shows at most 300 characters per selected chunk, so one intact chunk does not mean the model sees the complete evidence. Here the shown amount was initially a customer estimate; reading the thread establishes approval and the goodwill offer. Evidence: `eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:190`, `src/corpusqa/retrieval/search.py:298`, `src/corpusqa/config.py:540`, permitted corpus record at `:14`.

## 2. Fix options ranked

1. **Code-run full-question seed search before step 1.** Build one unscoped search using the complete question, with the normal eight-document page, request-local handles and tracing; append the observation after the question, leaving the static system prefix unchanged. This has **high confidence for surfacing this record, medium confidence for improving answers generally**: it preserves context and can save a model turn, but adds one embedding/search and repeated input tokens, with anchoring risk; the prompt-only equivalent is a cheaper implementation experiment but consumes a model turn and depends on compliance.

2. **Clarify created dates versus dates discussed in text (2a).** State that created filters constrain record timestamps, use them for explicit creation-time requirements, and label hit dates `created:`; event periods normally belong in the semantic query. **High confidence in correcting the semantics, medium confidence in behavioral improvement:** it may help the model read this record once shown, but does not independently recover it with the old queries; cost is a little prompt/output text, and the main risk is discouraging legitimate date filters.

3. **Correct negative-match wording and require inspection of a specifically relevant unread candidate (3).** Say “No indexed passage in this scope matches the keyword expression; relevant records may use different wording,” and before whole-question abstention read an unread candidate whose passage matches the event’s distinguishing details, if one exists. **Medium confidence:** this directly addresses the off-run failure and other false abstentions, at the possible cost of an extra read and reasoning turn; do not mandate reading whichever irrelevant hit happens to rank first.

4. **Use existing batching for competing candidates (part of 4).** Encourage reading two plausible candidates together when their relevance cannot be resolved from snippets; batching is already implemented, and the prompt already encourages independent calls. **Medium confidence for this question, uncertain incremental benefit elsewhere:** it could have checked D51 and D53 within step 9, saving turns compared with serial reads, but increases context and concurrent tool work; another generic “batch more” sentence alone may change little (`src/corpusqa/prompts/agent.md:17`, `src/corpusqa/agent/tools.py:96`).

5. **Out-of-range result warning (2b), deferred.** If built, compare candidates within one common date-unrestricted ranking that preserves source/within restrictions, and display excluded candidates separately with their exclusion reason; do not compare scoped RRF scores against unscoped RRF scores. **Low confidence for this question, medium potential elsewhere:** its old queries rank the target too low for a two-hit warning, while every dated search incurs extra ranking and distracting text; no new embedding is needed.

6. **Learn per-source created/text-date separation at ingest (2c), deferred.** Treat this as a descriptive statistic with coverage and sample size, never as a timestamp correction or probability that a source is wrong. **Low-to-medium confidence:** it might discourage the date mistake, but the reported 53% concerns only dated threads containing ISO dates, and legitimate retrospective or future-looking text also produces large gaps; extraction, storage, rebuilds and catalog tokens cost more than the immediate prompt clarification (`eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md:182`).

7. **Increase the step limit (remaining part of 4), last.** Consider this only after traces show useful, unfinished evidence gathering at the cap. **Low confidence now:** it permits another chance to read this record but equally permits more ineffective searches, costing roughly another model step per affected question and worsening tail latency; the current failures include voluntary finishes and repeated unproductive exploration (`docs/eval-log.md:28`, `docs/eval-log.md:67`).

## 3. Better or additional ideas

- **Separate semantic query text from lexical constraints if failures remain.** An optional plain-language `semantic_query` could describe the event and relationships, while the existing query retains exact keys and phrases for BM25. The model can supply both in the same call, avoiding a dedicated rewriting call. Trace both strings and their ranking contributions; the risk is extra schema complexity and invented details in rewrites.

- **Test dense-query cleanup as an ablation, not the primary fix.** Removing query-syntax quotes/operators from the embedding input is inexpensive, but cannot restore context the model omitted. Preserve identifier characters and distinguish operators from literal text. The investigation does not isolate the benefit of changing only the dense input while preserving exact BM25 behavior.

- **Prefer a separate full-question retrieval signal over concatenating the question onto every follow-up.** The seed already provides that signal once. If later experiments warrant reusing it, keep question-based and follow-up rankings distinguishable; blindly concatenating the original question can dilute a precise lookup for an entity discovered later.

- **Show text-date evidence with provenance, not a guessed second “event date.”** A bounded annotation such as `text mentions: 2025-03-22 (“Invoice … generated …”)` could help resolve apparent discrepancies. It must retain the created date and disclose truncation; multiple dates may concern plans, examples or unrelated events. Explicitly labeling the existing date is the cheaper first step.

- **Make snippet selection an experiment only if reading still fails.** The current snippet chooses a window containing the most distinct query words, rather than the most useful answer evidence (`src/corpusqa/retrieval/search.py:326`). A second non-overlapping passage from the same chunk might expose an approval or resolution, but adds tokens and requires its own evaluation; it does not justify changing chunk boundaries.

## 4. Recommendation

Build the **code-run full-question seed first**, matching the intent of the supplied plan’s §4.2, but leave automatic key lookups out of this initial experiment. The current loop starts with empty history and goes straight to the first model step, so that planned retrieval observation is absent (`src/corpusqa/agent/agent.py:287`, `src/corpusqa/agent/agent.py:608`). Second, clarify created-date semantics and label the displayed date; then separately test the negative-match wording and conditional read-before-abstaining rule. Keep exact matching for keys and genuine quotations. Do not initially raise limits, change chunking, build source-drift machinery, or assume a date-filter warning will rescue this case.

## 5. How to measure it

1. **Establish a fresh baseline.** Use current agent v5/tools v6 for both arms: the older T1–T3 results predate the citation correction. Freeze the index, model, thinking setting, limits and concurrency at four. Keep thinking selection separate; “off” here means the 1,024-token budget setting, not guaranteed zero thinking (`docs/eval-log.md:17`).

2. **Change one factor at a time.** Compare baseline against seed-only first. Run the required smoke check, then all 30 tune questions; smoke is a subset, so do not count it as independent evidence. Test date clarification and abstention wording as subsequent additions.

3. **Use three interleaved baseline/candidate pairs.** Six tune runs cost approximately **$5.4–10.2**, excluding smoke. Alternate arm order and rotate question order to reduce cache/load confounding. Inspect paired question outcomes, not just totals; one extra correct answer in one run is insufficient evidence.

4. **Track the mechanism and final outcome separately.**

   | Area | Measures |
   |---|---|
   | Retrieval | Expected-document recall@8; first relevant-document appearance; seed contribution |
   | Evidence use | Shown-but-unread relevant documents; first relevant read; shown/read/cited transitions |
   | Answer | Complete/partial/wrong answers; fact coverage; unsupported claims; false abstentions |
   | Citations | Precision, recall, exact sets; irrelevant seed-derived citations |
   | Unanswerable | Each of the four questions: correct textual abstention **and** empty citations |
   | Efficiency | Mean/p95 steps, searches, reads, tool calls, cap hits, forced finishes |
   | Runtime/cost | Full-request p50/p95/max latency, queue/limiter waits, timeouts, total cost including embeddings |
   | Cache | Input/cached/output/thinking tokens by step; cached-input fraction and billed cost |

5. **Set acceptance criteria before running.** Require repeatable improvement in the targeted retrieval/read behavior, no repeatable regression on other questions, and no deterioration in the four unanswerable cases or unsupported claims. Report outcomes both including and excluding pub_0077; improvement only on this question supports this repair, not broad generalization. A reasonable initial efficiency guardrail is no more than a 10% increase in mean cost or p95 latency without an explicit accuracy tradeoff, and no new deadline failures.

6. **Verify implementation behavior separately.** Use invented fixtures for timestamp/event-date disagreement, a misleading seed hit, an unanswerable question with plausible neighbors, and seed failure/deadline handling. Check tracing, request-local handles, disclosed exclusions and unchanged explicit filter semantics. Held-out evaluation remains reserved for a user-named milestone.

## 6. Risks to other questions

- **Anchoring and wrong citations:** Seeded look-alikes may be treated as answers, especially on the four unanswerable questions. Mark the seed as candidate evidence, preserve identity/claim checks, and retain the current rule that wholly unfound answers cite nothing (`src/corpusqa/prompts/agent.md:44`).

- **Wasted work on structured questions:** Counts and field-based lists may need `filter`, not narrative retrieval. The seed must not oblige the model to read it or substitute its eight hits for a complete set.

- **Tokens and cache costs:** Seed output persists in later prompts. Append it after the question and keep it stable within the request; do not put request-specific results into the system catalog. Existing append-only history supports prefix reuse, but “mostly cached” is an assumption to measure, especially on early turns (`src/corpusqa/agent/agent.py:31`; `docs/eval-log.md:48`).

- **Deadline contention:** Four simultaneous seeds add embedding and retrieval work; batching four tools per request can create larger bursts. Charge seed work against the existing request deadline and tool accounting, preserve finish time, and disclose seed failure or omission. The server deadline is 150 seconds, including queue time (`src/corpusqa/config.py:22`).

- **Incorrect date relaxation:** Some questions genuinely constrain creation dates. Preserve explicit filters and keep any out-of-range evidence separate; never silently merge it into counts or scoped results.

- **More exploration on genuinely missing answers:** An unconditional read-before-abstaining rule can prolong every unanswerable run. Require a specifically relevant unread candidate, not merely a nonempty semantic-search page.

## Claude's notes (2026-10-17)

Each point was checked against the code at `598b543d`. Nothing was changed or built.

**Where I agree with Codex.**
1. **Step 10 was not a forced finish.** `_limit_reached` forces a finish only when
   `steps >= max_steps` before a step (`agent.py:635-637`). So step 10 runs with every tool
   offered, and a finish-only step 11 would follow it. The off run's model chose to finish at
   step 10. The investigation's line "Step 10 was the last one, so it finished" is loose.
2. **A date fix alone would not have rescued pub_0077.** The investigation's own tables show it
   (lines 130, 142, 145): without the date range, those three queries rank the record 114, 103
   and 92 (pages 15, 13 and 12). The ranks 3, 5 and 3 hold only inside a range the record is
   not in. So 2a, 2b and 2c are general protection, not the fix for this question. The
   investigation's point 3 overstated what the range cost here.
3. **Seed first, without the key lookups for now.** One change at a time; pub_0077 has no keys.
4. **A fresh baseline is needed.** No tune run has used `agent.md` v5 or `tools.json` v6 yet.
   The last tune row in `eval/history.csv` is at `f96e8326`, before `bf224a40` and `598b543d`.
5. **The "no document contains" line overclaims.** `_keyword_text` (`corpus_tools.py:579-592`)
   prints it when the keyword expression matched nothing. With quoted phrases that only means
   "no text holds these exact phrases". All three answers turned it into "no such incident".

**What I add.**
1. **The model ignores the tool text on query syntax.** The `tools.json` v5 the runs used
   already said "OR and AND are not needed: they are not searched for" (checked with
   `git show 598b543d~1:src/corpusqa/prompts/tools.json`). Still, 16 of the 25 searches used
   OR. So a prompt-only rule on how to write queries ("search the question in plain words
   first") has a weak record here. This favours the code-run seed over its prompt version.
2. **The step-10 budget note misleads a little.** At step 10 it says "Step 10 of 10. No steps
   left after this one." (`STEP_NOTE`, `agent.py:121`; `budget_text`, `agent.py:222-229`). But a
   finish-only step still follows, and it can use what step 10 read. A model that takes the
   note at its word finishes instead of reading. A generic wording fix: say that step 10 is the
   last step for tools and that a finish step follows. In the off run, the model could then
   have read D53 at step 10 and answered at step 11. Cost: one more model call (about
   $0.005-0.008) on questions that hit the cap. Neither Codex nor the investigation names this.
3. **How to build the seed in this code** (a sketch, not tested):
   - Run it through the request's own tool session (`CorpusToolSession.toolbox()`,
     `ToolBox.run`). Then its D# handles can be cited, its `tool:search` span exists (the
     tracing rule), and its query vector is cached for the session (`_query_vector`,
     `corpus_tools.py:406-424`).
   - It can enter the history as a code-written search call plus its tool message. The client
     already sends code-written model turns with a stand-in thought signature
     (`llm/client.py:300-364`). The model then sees the seed like its own searches, and the
     prompt stays append-only, so the prompt cache works as now.
   - The seed's query embedding uses the ingest retry policy: 5 retries, backoff up to 60 s,
     60 s per request (`config.py:318-326`, `llm/client.py:233-250`). That is High finding 1 of
     the 2026-10-17 harness review, still open. A seed at the start of every request makes it
     matter more. Give the seed a short time box. If it is slow or fails, go on without it and
     tell the model so.
   - Count the seed as a tool call (against the 25-call cap), not as a model step.
4. **A cheap first look at the seed already exists.** `uv run eval-retrieval --production`
   ranks the served index with each tune question's own text: that is the seed's ranking. Its
   last row (`eval/history.csv`, 2026-10-16T21:30, before later index changes) gives hybrid
   per-question recall@5 0.662 and recall@10 0.727, with all gold documents in the top 10 for
   69% of the questions. So a seed page of 8 would hold most gold documents on most tune
   questions, and about a third would still need the model's own searches. Rerun it on today's
   index before any agent run. It needs only query embeddings while the index cache is current.

**Where I differ.**
1. **Measurement cost.** Three paired tune runs (6 runs) is sound. But it costs about $5.4 at
   thinking off to $10.4 at HIGH, plus the smoke runs. The thinking setting is still the
   user's open decision, and it changes this cost by about 2x. A cheaper path: the retrieval
   check above, then one baseline and one seed run, and more pairs only if the paired results
   are mixed. The user decides.
2. **2a at rank 2.** Fine as the next cheap change, but Codex itself says it does not recover
   this record. Measure it as its own change after the seed, with its own reason (the Slack
   date gap), not as part of the pub_0077 fix.
