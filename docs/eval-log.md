---
name: Eval log — runs, findings and the changes they led to
description: Every end-to-end eval of the agent (2026-10-17), newest first, for the final report - the final run (milestone final, all 79 questions, 4 at once: answerable F1 0.945, 10 of 11 no-answer questions cite nothing, 0 errors, slowest 59 s); smoke runs 1-4 (10 questions; run 1 found search that never says "nothing found", no stop rule or visible budget and no thinking; runs 2-4 kept the stop rule + step count, exact search and thinking HIGH; a table of all four), tune runs T1-T3 (30 questions, thinking HIGH vs LOW vs off, and the problems every setting shares) and T4-T5 (agent.md v7 + the first search, HIGH vs LOW; HIGH 28 of 30 correct and 4 of 4 abstains, chosen), the held-out milestones v7-first-search-high and -low (aggregates only; HIGH answerable F1 0.948 and 7 of 7 abstains), and the answer judge's calibration (prompt judge v1 agrees with the hand checks on 99.3% of 590 facts, kappa 0.90, $2.78). Each entry gives the setup, the scores, what went wrong and why with the evidence, and the changes it led to. Read before changing search, the agent prompt, the loop limits or the thinking settings, and when writing the report.
---

# Eval log

Newest first. Only smoke and tune runs are recorded here per question; held-out runs give
aggregate scores only (AGENTS.md, eval split).

## Final run, milestone `final` — 2026-10-17, all 79 dev questions (aggregates only)

**Why.** The user's last check before submission, run by the user from SOLUTION.md 2.3:
`uv run eval-all --milestone final --concurrency 4` (79 questions, 4 at a time, as the
grader sends them), against the service started as in 2.1 from a cache rebuilt from scratch.
It chooses and changes nothing. Only the runner's printed aggregates are used.

**Setup.** As T4: thinking HIGH, `agent.md` v7, `tools.json` v6, first search on; the code of
`d9eb700a` (start-up stage logging; nothing in the answer path changed since the held-out
milestones, apart from Codex #5, network errors). Judge `gemini-3.1-pro-preview`, prompt v1.

| All 79 (68 answerable, 11 no answer) | |
|---|---|
| Answerable ids: precision / recall / F1 | 0.944 / 0.961 / 0.945 |
| Single-record (31): precision / recall / F1 / exact | 0.935 / 0.968 / 0.946 / 0.903 |
| Multi-record (37): precision / recall / F1 / all gold / exact | 0.951 / 0.955 / 0.943 / 0.892 / 0.757 |
| Answerable: hit / all gold / exact | 0.985 / 0.926 / 0.824 |
| No answer: cite nothing / say "not in the records" | 0.909 (10 of 11) / 1.000 |
| Answers correct / partly / wrong / unjudged | 0.810 / 0.127 / 0.038 / 0.025 |
| Facts (308) right / partly / missing / wrong; contradictions | 0.942 / 0.023 / 0.032 / 0.003; 3 |
| Judge agreement (its two judgments) | 0.987 |
| Errors (incl. timeouts) | 0 |
| Latency p50 / p95 / max | 36.6 / 55.3 / 59.3 s |
| Spend: agent / judge | $3.33 / $2.04 |

**What it says.** No failed or slow answer with 4 at a time: the slowest took 59 s of the
180 s limit. Citations hold the held-out milestone's pattern (precision below recall: it
over-cites a little, most on one-record questions). One no-answer question cited a record
although its answer said the records lack it. Summary:
`eval/results/all/ask/20261017T123129Z/summary.md` (in the main checkout).

## Answer judge calibration — 2026-10-17, the 5 hand-checked tune runs (150 answers, 590 facts)

**What.** The eval now has an automatic answer judge (`eval/judge.py`, prompt
`src/corpusqa/prompts/judge.md`). `gemini-3.1-pro-preview` reads the question, the gold answer,
the numbered `answer_facts` and the answer text. It gives each fact a status (right, partly,
missing, wrong), lists claims that contradict the gold answer, says whether the answer says
"not in the records", and gives a verdict (correct, partly, wrong). It judges each answer twice,
and a third time if the two verdicts differ. The user approved this design: "okay build it
that way and add the second one as the model". `uv run eval-judge-calibrate` re-judged the
answers of the five tune runs that were checked by hand (T1-T5 below) and compared the two.

**Result: prompt `judge` v1 passed the bar on the first round. No prompt change was needed.**

| | round 1 (v1) | the bar |
|---|---|---|
| Report | `eval/results/judge-calibration/20261017T090030Z/` | |
| Fact status agreement | **99.3%** (586 of 590 facts) | at least 90% |
| Question verdict agreement | 99.3% (149 of 150) | |
| Cohen's kappa, facts / verdicts | 0.900 / 0.968 | |
| Unanswerable, "says not in the records" | 20 of 20 agree | |
| Unjudged answers | 0 | |
| Judge against itself (first two judgments) | same verdict 150 of 150; same fact status 588 of 590 | |
| Third judgments | 0 | |
| Calls, cost | 258 new calls, $2.78 (61 judgments reused from the pilot) | stop at $8 |

How to read it. "Agreement" is the share of facts (or answers) where the judge gave exactly
the label the hand check gave. Most facts are `right` (569 of 590), so a judge that always said
`right` would already agree on 96%. Cohen's kappa corrects for that: it is (observed agreement
- chance agreement) / (1 - chance agreement), where chance agreement is what two raters with
these label shares would reach by guessing. 1 means perfect, 0 means no better than chance.
0.90 on facts means the rare labels (partly, missing) match too.

**The 4 fact disagreements** (all tune data):
- pub_0002 fact 2, in 3 runs: hand `partly`, judge `missing`. The answer only names "the
  finalization-retry guard" (the question's own words) and does not say what it does. The hand
  check counted the name as the core of the fact; the judge counted nothing as stated. The
  prompt's rule F3 says this case is `partly`, so the judge broke its own rule here. The verdict
  is `partly` either way.
- pub_0017 fact 2, run T3: hand `partly`, judge `right`. The answer gives "$400 from 2026-04-01,
  up from $350" but does not name the old ticket INT-324. The T4 and T5 hand checks made a rule
  that a bracketed source note like "(from INT-324)" is not a detail; the T3 hand check was made
  before that rule. The judge follows the later rule (prompt rule F4). This is the one verdict
  disagreement too (hand `partly`, judge `correct`). It is a hand-check inconsistency, not a
  judge error.

**Pilot first.** One run (T4, 30 answers) was judged first to check cost and parsing:
`eval/results/judge-calibration/20261017T084850Z/`, 112 of 112 facts and 29 of 29 verdicts
agreed. One answer was unjudged there, after two API failures in a row (a 499 "cancelled",
then the 120 s call limit). Round 1 judged it again from the cache plus one new call, with no
failure.

**Cost and time.** About $0.010 per judge call; about 2 calls per answer. The judge's own
ledger (`.cache/spend/judge.json`) holds $3.46 for the pilot and round 1 together. Round 1
took 20 minutes at 6 answers at once; the only API errors were 504s, each passed on its retry.
So judging a tune run costs about $0.65 and adds 5-10 minutes; a held-out run about $1.

**What it means.** The judge reproduces the hand checks closely enough to use in every eval
run. `uv run eval-tune` and `eval-smoke` now report its aggregates in `summary.md` and the
history row, and `judge.jsonl` per question. Held-out and `all` runs get its aggregates only.
Known weakness: it can call a fact `missing` where the answer only names the thing (pub_0002).
Not yet done: a judged /ask run (no eval was run in this task).

## Held-out milestones `v7-first-search-high` and `v7-first-search-low` — 2026-10-17 (49 questions, aggregates only)

**Why.** The user asked to run the held-out set with both settings ("run the high and low
as well on the rest of the question", "just run the test first"), to see whether the tune
gains of T4-T5 hold on questions never looked at. The shipped setting (thinking HIGH) was
chosen from the tune set and recorded BEFORE these runs (`docs/runs/2026-10-16-agent-tools/decisions.md`);
these results choose nothing and change nothing (AGENTS.md eval split). Only the runner's
aggregate lines and history rows are used; no per-question held-out output was read, and the
server logs (which hold the question text) were deleted after counting.

**Setup.** As T4-T5: `agent.md` v7, `tools.json` v6, first search on, exact search on, the
handle safety net; 4 at a time. The rows show `b08a9661+dirty`: the handoff file was being
edited during the HIGH run, and LOW was set in the working tree; no code changed.

| | Tune T4: HIGH | **Held-out: HIGH** | Tune T5: LOW | **Held-out: LOW** |
|---|---|---|---|---|
| Questions (answerable / unanswerable) | 30 (26 / 4) | 49 (42 / 7) | 30 (26 / 4) | 49 (42 / 7) |
| Errors, timeouts | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| Answerable ids: precision / recall / F1 | 0.974 / 0.949 / 0.954 | **0.935 / 0.968 / 0.948** | 0.987 / 0.962 / 0.967 | 0.932 / 0.925 / 0.926 |
| Answerable: all gold / exact set | 0.885 / 0.846 | 0.952 / 0.857 | 0.923 / 0.885 | 0.857 / 0.857 |
| Single-document (held-out n=19): F1 / exact | 1.000 / 1.000 | 0.930 / 0.895 | 1.000 / 1.000 | 0.947 / 0.947 |
| Multi-document (held-out n=23): F1 / all gold | 0.914 / 0.786 | **0.962 / 0.957** | 0.938 / 0.857 | 0.909 / 0.783 |
| Unanswerable: cited nothing | 4 of 4 | **7 of 7** | 2 of 4 | **1 of 7** |
| Latency p50 / p95 / max | 32.8 / 51.0 / 71.0 s | 37.0 / 65.1 / 72.1 s | 9.8 / 34.7 / 37.1 s | 12.4 / 29.0 / 32.5 s |
| Spend; per question | $1.17; $0.039 | $2.26; $0.046 | $0.69; $0.023 | $1.32; $0.027 |

**What it says.**
- **The tune gains hold on unseen questions with HIGH:** answerable citation F1 0.948 on
  held-out against 0.954 on tune, recall even higher (0.968), and every unanswerable
  question answered with no citation (7 of 7). The decision made from tune (HIGH) is
  confirmed, not changed.
- **LOW drops on held-out:** F1 0.926 (tune 0.967), multi-document all-gold 0.783, and only 1
  of 7 unanswerable questions with no citation: LOW follows the citing rule much less often.
- **Time and cost carry over:** HIGH 37 s median and $0.046 a question, all under 73 s of
  the 150 s deadline; LOW about 3x faster and cheaper.
- **Answer text is not scored yet:** the runner scores ids only. The automatic answer judge
  (being built, calibrated against the tune hand checks first) will add held-out answer
  totals.
- The model's quoted `filter` argument names were rejected 6 times in each run (counts from
  the server logs, as on tune).

## Tune runs T4-T5 — 2026-10-17, tune set (30 questions): agent.md v7 + the first search, HIGH vs LOW

**What changed since T1-T3.** `agent.md` v5 (cite a record only if it states something
the question asks for; a "not found" answer cites nothing; no handles in the answer text),
v6 ("# How to search": to avoid anchoring bias and searching only in the question's own
wording), v7 (the line about the first search); `tools.json` v6 (the `finish` citations
text matches the rule); the handle safety net in code; and the code-side first search:
before step 1, `search` runs with the question exactly as asked and is shown as the
agent's first tool result (`AGENT_FIRST_SEARCH`, 8 s bound). The user asked for both runs:
"build the search as well first so we send the search result for the first question as
well", then "run it twice once with high and the other with low". Same as T1-T3 otherwise:
exact search on, 4 at a time, 180 s limit. A live check first confirmed the code-written
first-search call is accepted by Gemini. Answers checked by hand by two agents in parallel.

| | T1: HIGH, v4 | T2: LOW, v4 | **T4: HIGH, v7** | **T5: LOW, v7** |
|---|---|---|---|---|
| Folder | `eval/results/tune/ask/20261017T033026Z` | `.../20261017T033904Z` | `.../20261017T072027Z` | `.../20261017T072902Z` |
| Errors, timeouts, forced finishes | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| Answers correct / partly / wrong | 27 / 2 / 1 | 26 / 3 / 1 | **28 / 2 / 0** | 25 / 5 / 0 |
| Facts right / partly / missing / wrong (of 118) | 114 / 2 / 2 / 0 | 113 / 1 / 4 / 0 | **116 / 2 / 0 / 0** | 113 / 3 / 2 / 0 |
| Unsupported claims (that matter) | 5 (1) | 2 (1) | 4 (0) | 3 (0) |
| Every asked part answered | 28 of 30 | 29 of 30 | **30 of 30** | **30 of 30** |
| Answerable ids: precision / recall / F1 | 0.923 / 0.910 / 0.903 | 0.939 / 0.923 / 0.923 | 0.974 / 0.949 / 0.954 | **0.987 / 0.962 / 0.967** |
| Answerable ids: exact set (of 26) | 20 | 21 | 22 | **23** |
| Unanswerable abstain (cited nothing) | 0 of 4 | 1 of 4 | **4 of 4** | 2 of 4 |
| Steps per question (mean) | 7.4 | 5.7 | 6.8 | **5.1** |
| Latency p50 / max | 41.3 s / 72.3 s | 10.3 s / 26.4 s | 32.8 s / 71.0 s | **9.8 s** / 37.1 s |
| Cost per question; run | $0.058; $1.74 | $0.0445; $1.33 | $0.039; $1.17 | **$0.023; $0.69** |
| Share of input from the prompt cache | 71% | 57% | 86% | 83% |

**What the numbers say.**
- **v7 with the first search improved both settings.** No wrong answer in either run (T1,
  T2: one each), no unsupported claim that matters, every asked part answered, better ids.
- **pub_0077 is fixed in both:** the first search ranks the right record first (D1); the
  answers now give the $14,200 credit and the month of free capacity (T1-T3: a false "not
  found" after 10 steps). Across T5, 18 of 26 answerable questions cite only records from
  the first search; the 6 that cite only later records are list and count questions,
  answered with `filter`.
- **HIGH now has the best answers:** 28 correct, 116 of 118 facts, and all 4 unanswerable
  questions cite nothing. LOW still cites the record the question names on 2 of them
  (pub_0028, pub_0030: each is D1 of the first search, quoted for context); the citing rule
  is in the prompt only, no code checks it.
- **LOW is cheapest, fastest and has the best ids,** at $0.023 a question and a 9.8 s
  median; HIGH costs $0.039 and takes 32.8 s, with every question well under the 150 s
  deadline.
- **Most of the cost drop is the prompt cache** (83-86% of input served from it, against
  57-71%): about $0.51 of HIGH's $0.57 saving and $0.61 of LOW's $0.64. Nothing in our
  code changed that; the provider's cache is best effort, so part of it may not repeat.
- **The two new partial answers are omissions in the writing, not in the search:** pub_0010
  (the release name) and pub_0039 (a pair of dates) cite the records that hold them. HIGH
  fixed pub_0002 (it now describes the guard).
- Not re-run: thinking off with v7.

**Still open.**
1. The 4 unanswerable questions still use all 10 steps (about 25% of the spend).
2. The model writes the `filter` argument names with literal quotes (`"\"field\""`) in 5 of
   about 13 `filter` calls per run; the tool rejects them and a step is lost.
3. Where the citing rule is not followed (LOW), nothing in code catches it.

**Decision.** Waiting for the user. HIGH gives the best answers and all four abstains for
$0.039 a question; LOW is 40% cheaper and 3x faster with three fewer correct answers and two
missed abstains. The answer text and citations are what is graded, so HIGH is recommended
now (this replaces the T1-T3 recommendation of off, which was made before v7 and the first
search, when the three settings answered equally).

## Tune runs T1-T3 — 2026-10-17, tune set (30 questions), thinking HIGH vs LOW vs off

**Why.** Smoke run 4 showed thinking HIGH gave the best answers but cost +52% and 3.6x the
latency, on 10 questions. The user asked to compare the three settings on the full tune set:
"do three runs, one with thinking off, one with high thinking and one with low".

**Setup.** Everything as smoke run 4 (`agent.md` v4 with the stop rule and step note,
`tools.json` v5, exact search on), except `AGENT_THINKING_LEVEL`: "high" (committed), "low"
and None (= the 1,024-token budget; working tree only). 30 questions (26 answerable: 12
single-document, 14 multi-document; 4 unanswerable), 4 at a time (`--concurrency 4`, the
contract's limit), 180 s limit per question. The setting actually sent was confirmed in one
`chat` generation of each run. Answers checked by hand against `answer_facts` by three
agents in parallel, one per run.

| | T1: HIGH | T2: LOW | T3: off (budget) |
|---|---|---|---|
| Folder | `eval/results/tune/ask/20261017T033026Z` | `.../20261017T033904Z` | `.../20261017T034320Z` |
| Errors, timeouts, forced finishes | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| Answers correct / partly / wrong | **27** / 2 / 1 | 26 / 3 / 1 | 26 / 3 / 1 |
| Facts right / partly / missing / wrong (of 118) | **114** / 2 / 2 / 0 | 113 / 1 / 4 / 0 | 113 / 2 / 3 / 0 |
| Unsupported claims (that matter) | 5 (1) | **2** (1) | 3 (1) |
| Answerable ids: precision / recall / F1 | 0.923 / 0.910 / 0.903 | 0.939 / 0.923 / 0.923 | **0.949 / 0.923 / 0.928** |
| Answerable ids: exact set (of 26); all gold | 20; 0.846 | 21; 0.885 | **22; 0.885** |
| Unanswerable abstain (cited nothing) | 0 of 4 | **1 of 4** | 0 of 4 |
| Steps per question (mean) | 7.4 | 5.7 | 6.0 |
| Latency p50 / max | 41.3 s / 72.3 s | **10.3 s / 26.4 s** | 12.1 s / 39.8 s |
| Cost per question; per call | $0.058; $0.0079 | $0.0445; $0.0078 | **$0.031; $0.0051** |
| Thinking tokens | 17% of cost (~$0.010 a question) | 425 in the whole run | 660 in the whole run |
| Run cost (30 questions) | $1.74 | $1.33 | **$0.92** |

**What the numbers say.**
- **The three answer about equally well.** HIGH has one more correct answer and one more fact
  than LOW and off; the questions that differ (pub_0009, pub_0039, pub_0052, pub_0017) are
  ones whose verdict has flipped between runs before, so one question is within run-to-run
  variation. HIGH also has the most unsupported claims (5) and the worst ids.
- **Off is the cheapest and has the best citations;** HIGH costs 1.9x and takes about 3.4x
  as long per question. LOW thinks almost not at all (425 tokens in 30 questions), so it is
  close to off, but it cost 44% more than off: its prompts hit Gemini's cache less (85% of
  its cost is uncached input). Why the cache hit rate differs is not checked.
- **LOW's one abstain is not the setting's doing:** pub_0026 happened to finish with no
  citation; the other three unanswerable answers cite one record, as in HIGH and off.

**Problems every setting has (fixes are setting-independent).**
1. **Unanswerable questions cite a background record.** In all three runs, every one of the
   4 unanswerable answers says plainly that the records lack the answer, and then cites one
   record that states nothing the question asks for (a Q&A that mentions "Series C" once,
   the ticket the question names, a ticket that only plans the test, a process page with no
   counts). The runner counts these as not abstained. A sharper citing rule ("cite a record
   only if it states something the question asks for; a 'not found' answer cites nothing")
   would turn all 4 into correct abstains.
2. **pub_0077 is a false "not found" in all three runs.** The record exists
   (`data/corpus/slack/finance/...neptunehealth-metering-discrepancy-credit-approval.json`:
   a $14,200 credit and one month of dedicated capacity), but the question's words differ
   from the record's ("machine groups / refund / make-good" against "node groups / credit /
   goodwill"); HIGH thought 6,134 tokens on it and still did not find it. A retrieval
   problem (vocabulary), not a thinking one.
3. **The answers that use all 10 steps are the questions with no answer.** 5-7 questions per
   run (the 4 unanswerable, pub_0077, pub_0039) search until the step note says none are
   left: 17-23% of the questions, 25-30% of the spend.
4. **Internal handles leak into answer text** (off run: "email thread D3", "(D4)"): the
   reader cannot use them.
5. **Recurring partial answers:** pub_0002 names the fix without describing it (every run);
   pub_0052's early-termination rule comes and goes.

**Decision.** Waiting for the user. Recommended: thinking off (`AGENT_THINKING_LEVEL = None`):
the same answer quality within one question, the best citations, the lowest cost and close
to the lowest latency; HIGH's single extra correct answer does not repay 1.9x the cost and
3.4x the time. The fixes above (citing rule, no handles in answer text) apply to any setting
and are measured next on this same tune set.

## Runs 1-4 at a glance (smoke, 10 questions each, 2026-10-17)

| | Run 1 (base) | Run 2 (+L) | Run 3 (+S) | Run 4 (+K) |
|---|---|---|---|---|
| Answerable (9): precision / recall / F1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Unanswerable (1): abstain | 0.000 | 0.000 | 0.000 | 0.000 |
| Answers correct / partly / wrong | 6 / 4 / 0 | 9 / 1 / 0 | 9 / 1 / 0 | 10 / 0 / 0 |
| Facts right / partly / missing / wrong (of 46) | 43 / 2 / 1 / 0 | 45 / 1 / 0 / 0 | 45 / 0 / 1 / 0 | 46 / 0 / 0 / 0 |
| Unsupported claims (that matter) | 3 (1) | 1 (0) | 1 (0) | 0 (0) |
| Forced finishes; questions at 10 steps | 3; 3 | 0; 1 | 0; 1 | 0; 1 |
| Model calls per question | 6.8 | 5.0 | 4.3 | 6.9 |
| Questions that thought | 6 | 2 | 0 | 10 |
| Latency p50 / p95 | 15.6 s / 35.4 s | 8.4 s / 18.3 s | 7.7 s / 20.0 s | 27.7 s / 49.0 s |
| Cost per question | $0.0596 | $0.0410 | $0.0383 | $0.0583 |
| Decision | - | keep | keep | keep (by the keep-rule; costs more, see run 4) |

The one question still at 10 steps in runs 2-4 is pub_0028, the unanswerable one.

## Run 4 — 2026-10-17, smoke set (10 questions), L + S + K: thinking level HIGH

**Setup (what differs from run 3).** Commit `49069dc8+dirty`: the only dirty file was the
switch, `AGENT_THINKING_LEVEL = "high"` in `config.py` (committed after the run). Every
agent step sends `thinking_level: HIGH` instead of `thinking_budget: 1024` (all 69 chat
generations confirm it). Prompts, system prompt (28,121 characters), search and limits as
in run 3. Full record: `eval/results/smoke/ask/20261017T031540Z/` (`run_setup.md`,
`answer_check.md` / `.json`, `cost_analysis.md`, `step_limit_traces.md`).

### Scores

| | Run 3 (L + S) | Run 4 (L + S + K) |
|---|---|---|
| Answerable (9): precision / recall / F1 / exact | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| Unanswerable (1): abstain rate | 0.000 | 0.000 (cites the same ticket) |
| Errors, timeouts | 0, 0 | 0, 0 |
| Answers (by hand): correct / partly / wrong | 9 / 1 / 0 | **10 / 0 / 0** |
| Facts: right / partly / missing / wrong (of 46) | 45 / 0 / 1 / 0 | **46 / 0 / 0 / 0** |
| Unsupported claims (that matter) | 1 (0) | **0** |
| Forced finishes; questions at 10 steps | 0; 1 | 0; 1 (pub_0028) |
| Model calls: total, per question; tool calls | 43, 4.3; 58 | 69, 6.9; 97 |
| Thinking tokens (calls that thought) | 0 (0 of 43) | 24,955 (69 of 69) |
| Searches; searches that found no keyword match | 19; 4 | 38; 13 |
| Latency p50 / p95; slowest single call | 7.7 s / 20.0 s; 4.4 s | **27.7 s / 49.0 s; 15.4 s** |
| Cost: total, per question, per call | $0.383, $0.0383, $0.0089 | **$0.583, $0.0583**, $0.0085 |

### Cost

| | Questions | Model calls | Average per question | Average per call |
|---|---|---|---|---|
| 8-10 steps: pub_0023, pub_0028, pub_0040, pub_0052 | 4 | 36 | $0.0693 | $0.0077 |
| The other six (4-7 steps) | 6 | 33 | $0.0511 | $0.0093 |
| All | 10 | 69 | $0.0583 | $0.0085 |

+$0.201 against run 3 (+52%): about half is output, mostly thinking (24,955 tokens, $0.094,
16% of the run), and half is input from the extra steps, each re-sending the history (1.62M
input tokens against 0.86M). 9 of 10 questions cost more and all 10 took longer. The cost
per question is back at run 1's level.

### What changed against run 3, and why

- **Every call now thinks** (156-404 tokens on step 1, up to 1,785 on a finish), where
  run 3 thought on none.
- **The answers are the best of the four runs:** 10 of 10 correct, 46 of 46 facts, no
  unsupported claim. The two gains: pub_0010 drops the made-up "ET" time zone that runs 1-3
  all wrote; pub_0052 states the early-termination rule. The second fact has flipped in
  every run (missing, right, missing, right), so it is not a reliable gain. The answers are
  longer and carry more detail; every extra detail was checked and is in the records.
- **Thinking made the runs longer, not shorter.** The model checks more: pub_0023 had all
  six companies and owners from one `filter` + `read_fields` by step 2, then read each of
  the six records whole, one per step, and finished in step 9 (run 3: step 3). pub_0075 and
  pub_0052 searched wider. Steps went from 4.3 to 6.9 a question.
- **The looping question is unchanged.** pub_0028 (unanswerable) thought 4,550 tokens, made
  17 tool calls (run 3: 10), read two unrelated tickets, and still finished only when the
  step count said "No steps left". Its answer was right from step 2 in every run.
- The deadline is not at risk on these questions: the slowest took 49.0 s of 150 s, and the
  slowest single call 15.4 s of the 40 s call limit.

### Keep or revert

**Keep K by the keep-rule**, committed (`AGENT_THINKING_LEVEL = "high"`, `df4b2ea7`): citation precision,
recall and F1 the same, abstain the same, and the hand-checked answers did not get worse
(one more fact right, one fewer unsupported claim). **But it regresses on everything it was
meant to improve**: steps +60%, p50 latency 3.6x, p95 2.4x, cost +52%, and pub_0028 still
runs to its last step. The quality gain is small and partly noise. Whether that trade is
worth it is the user's call: setting `AGENT_THINKING_LEVEL = None` restores run 3 exactly
(one line; the index rebuilds at the next start).

## Run 3 — 2026-10-17, smoke set (10 questions), L + S: exact search for quoted text and keys

**Setup (what differs from run 2).** Commit `b1026c8a+dirty`: the dirty files were the
switch itself (`SEARCH_EXACT_PHRASES = True` in `config.py`), the one test line it moves
(`tests/unit/test_corpus_tools.py`: "no documents" is now on line 3) and the generated
handoffs index; the flip and the test line were committed after the run. `tools.json` v5
(the committed text: how quotes and keys match, "meaning only", the "no document in scope
contains ..." line). `agent.md` v4 and the system prompt (28,121 characters) unchanged.
`AGENT_THINKING_LEVEL = None`. Full record: `eval/results/smoke/ask/20261017T030436Z/`
(`run_setup.md`, `answer_check.md` / `.json`, `cost_analysis.md`, `step_limit_traces.md`).

### Scores

| | Run 2 (L) | Run 3 (L + S) |
|---|---|---|
| Answerable (9): precision / recall / F1 / exact | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| Unanswerable (1): abstain rate | 0.000 | 0.000 (cites the same ticket) |
| Errors, timeouts | 0, 0 | 0, 0 |
| Answers (by hand): correct / partly / wrong | 9 / 1 / 0 | 9 / 1 / 0 |
| Facts: right / partly / missing / wrong (of 46) | 45 / 1 / 0 / 0 | 45 / 0 / 1 / 0 |
| Unsupported claims (that matter) | 1 (0) | 1 (0) |
| Forced finishes; questions at 10 steps | 0; 1 (pub_0028) | 0; 1 (pub_0028) |
| Model calls: total, per question; tool calls | 50, 5.0; 52 | **43, 4.3; 58** |
| Searches; searches that found no keyword match | 13; 0 | 19; **4** |
| Latency p50 / p95 | 8.4 s / 18.3 s | **7.7 s** / 20.0 s |
| Cost: total, per question, per call | $0.410, $0.0410, $0.0082 | **$0.383, $0.0383**, $0.0089 |
| Questions that thought | 2 | 0 |

### Cost

| | Questions | Model calls | Average per question | Average per call |
|---|---|---|---|---|
| Long: pub_0028 (10 steps) | 1 | 10 | $0.0691 | $0.0069 |
| The other nine (3-5 steps) | 9 | 33 | $0.0348 | $0.0095 |
| All | 10 | 43 | $0.0383 | $0.0089 |

The cost per call went up a little because the model now runs several tool calls per step
(pub_0010: 4 searches in step 1, 4 reads in step 2; 13 tool calls in 5 steps), which leaves
fewer cheap cached middle steps; fewer steps still means less history re-sent, so the cost
per question fell 7%.

### What changed against run 2, and why

- **Search can now say "nothing found", and did 4 times** (3 in pub_0028, 1 in pub_0052);
  in run 2 no search ever came back without keyword matches. Quoted keys run as phrases
  (pub_0028's `"SUP-66233" OR "SUP-83210"` became `"sup 66233" OR "sup 83210"`).
- **pub_0028 did not stop sooner.** It read the only relevant record by step 2, then got
  three explicit "nothing found" pages and still searched new leads until its last step
  (10 steps, $0.069, as in run 2). The signal reaches the model; the model does not act on
  it. With 0 thinking tokens on every call, change K is the next thing to try.
- **The other questions got shorter:** pub_0010 9 -> 5 steps, pub_0040 5 -> 4, pub_0046
  4 -> 3, pub_0052 5 -> 4. Part of this is the model batching calls in one step, which
  `tools.json` v5 does not ask for; it may be the new text or chance.
- **Answers: same totals, one fact each way.** pub_0002 now describes the fix (fact 2
  partly -> right); pub_0052 again leaves out the early-termination rule (right -> missing;
  run 1 missed it too). Both read the same records as in run 2 (traces checked), so both are
  answer-writing variation, not search.
- **No thinking at all** (run 2: 2 questions). Not a score change; K tests it.

### Keep or revert

**Keep S** (`SEARCH_EXACT_PHRASES = True`, committed with the test line as `ed587baa`). The keep-rule
holds: citation precision, recall and F1 the same, abstain the same, facts right the same
(45) with no more wrong or unsupported claims. Steps, latency p50 and cost went down; p95
went up 1.7 s (it is pub_0028 in both runs). The one-fact swap is within what one rerun of
the same setup could change; recall is identical, so the run-twice rule does not apply.
Next: run 4 adds K (`AGENT_THINKING_LEVEL = "high"`).

## Run 2 — 2026-10-17, smoke set (10 questions), change L: a stop rule and a visible step count

**Setup (what differs from run 1).** Commit `9c2692e3+dirty`: the only dirty file was
`tools.json`, set back to v4 in the working tree for this run (v5 describes exact search,
which is off here; restored right after). `agent.md` v4: a "# When to stop" section (finish
once every part is answered from a record read or shown missing; follow leads only for
open parts; reword a search that brought nothing new once, then finish) and a
`<budget>Step N of 10. K steps left after this one.</budget>` message before every model
call after the first. `SEARCH_EXACT_PHRASES = False`, `AGENT_THINKING_LEVEL = None` (budget
1,024, as in run 1). System prompt 28,121 characters (run 1: 26,777). Everything else as
run 1. Full record: `eval/results/smoke/ask/20261017T025126Z/run_setup.md` (+ `setup/`),
`answer_check.md` / `.json`, `cost_analysis.md`, `step_limit_traces.md`.

### Scores

| | Run 1 | Run 2 |
|---|---|---|
| Answerable (9): precision / recall / F1 / exact | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| Unanswerable (1): abstain rate | 0.000 (cited 1 record) | 0.000 (cited the same record) |
| Errors, timeouts | 0, 0 | 0, 0 |
| Answers (by hand): correct / partly / wrong | 6 / 4 / 0 | **9 / 1 / 0** |
| Facts: right / partly / missing / wrong (of 46) | 43 / 2 / 1 / 0 | **45 / 1 / 0 / 0** |
| Unsupported claims (that matter) | 3 (1) | **1 (0)** |
| Forced finishes (step limit) | 3 | **0** (pub_0028 used all 10 steps and finished itself) |
| Model calls: total, per question | 68, 6.8 | **50, 5.0** |
| Latency p50 / p95 | 15.6 s / 35.4 s | **8.4 s / 18.3 s** |
| Cost: total, per question, per call | $0.596, $0.0596, $0.0088 | **$0.410, $0.0410, $0.0082** |
| Questions that thought (thinking tokens > 0) | 6 | 2 |

### Cost

| | Questions | Model calls | Average per question | Average per call |
|---|---|---|---|---|
| Long (9-10 steps): pub_0010, pub_0028 | 2 | 19 | $0.0636 | $0.0067 |
| The other eight (3-5 steps) | 8 | 31 | $0.0353 | $0.0091 |
| All | 10 | 50 | $0.0410 | $0.0082 |

Every call was read back from Langfuse this time (measured, not derived). The forced finish,
run 1's most expensive call ($0.023-0.027, uncached), is gone: the finishing call now
averages $0.0060 because it keeps the cache. Steps 1 and 2 are never cached (about $0.012
each) and are now most of a short question's cost.

### What changed against run 1, and why

- **The answers got better.** pub_0010 now gives the provider ticket's target release
  (v0.19.0); pub_0052 now states the early-termination rule and drops run 1's "400 days"
  (taken from a snippet of an unrelated record seen in a wasted search); pub_0028 now says
  the records do not show whether the rollback ran, instead of run 1's unsupported "did not
  execute". In all three, fewer wasted searches meant less stray text to pick up.
- **Fewer steps, no forced finish.** The three step-limit questions of run 1 took 5, 9 and
  10 steps (pub_0052, pub_0010, pub_0028), against 11 each. pub_0028 still searched 7 times
  after it had read the only relevant record (step 2): quoted keys joined by OR, which our
  search still splits into loose words, so every page came back full. The step count made
  it finish itself in its last step instead of being forced.
- **Unchanged:** the citation scores (already perfect) and the unanswerable question's
  citation (it still cites the ticket it read, so the runner counts it "not abstained").
- **Less thinking:** 2 questions thought (run 1: 6). Not a regression in any score; change
  K is what tests thinking.
- Small sample: 10 questions, one run each. Recall is the same (1.000), so the run-twice
  rule for a one-question difference does not apply; the answer-text gains are three
  questions, all in the direction the change targets.

### Keep or revert

**Keep L.** Nothing the keep-rule watches dropped: citation precision, recall and F1 the
same, abstain the same, and the hand-checked answers improved (3 more correct, 2 more facts
right, 2 fewer unsupported claims). Steps, latency and cost all went down. (L has no switch;
it stays in the code either way.) Next: run 3 adds S (`SEARCH_EXACT_PHRASES = True`,
`tools.json` v5).

## Run 1 — 2026-10-17, smoke set (10 questions), the first run of the full agent

**Setup.** Commit `d101107f` (branch `agent/tools`, now on `main`). Model `gemini-3.8-flash`,
thinking budget 1,024 tokens, at most 10 steps then a forced `finish`, compaction above 80,000
tokens (never triggered). Prompts: `agent.md` v3, `question.md` v1, `tools.json` v4. Eight
tools: `search`, `filter`, `field_values`, `read_fields`, `read_document`, `lookup_key`,
`list_fields`, `finish`. Tracing on (Langfuse US cloud); every question has a trace.

**Everything that decided the run** (every setting, the exact system prompt as sent, the
tool list) is in `eval/results/smoke/ask/20261017T013336Z/run_setup.md` and its `setup/`
folder, taken from commit `d101107f`. The ones most likely to be tuned:

| Setting | Value |
|---|---|
| `AGENT_MAX_STEPS` | 10, then a forced `finish` |
| `AGENT_THINKING_BUDGET` | 1,024 tokens (a maximum) |
| `AGENT_MAX_CALLS_PER_STEP` / `AGENT_MAX_TOOL_CALLS` | 4 / 25 |
| `AGENT_COMPACT_AT_TOKENS` | 80,000 |
| `RETRIEVAL_TOP_CHUNKS`, `RRF_K`, RRF weights | 50, 60, 1.0 / 1.0 |
| Keyword query | words OR-ed (`match_expression`) |
| `SEARCH_PAGE_DOCS`, snippets | 8 documents, 2 passages of 300 characters |
| `FILTER_PAGE_ROWS` and the other list pages | 50 |
| Prompts | `agent.md` v3 (26,777-character system message with the field catalog), `question.md` v1, `compact.md` v1, `tools.json` v4 |

**Files.** `eval/results/smoke/ask/20261017T013336Z/`: `summary.md` (runner's scores),
`answer_check.md` and `answer_check.json` (answer text checked by hand),
`step_limit_traces.md` (step-by-step timelines of the three long runs, with Langfuse links).
Row in `eval/history.csv`.

### Scores

Citations (the runner scores the returned `document_ids` as sets):

| Group | n | Precision | Recall | F1 | All gold |
|---|---|---|---|---|---|
| Answerable | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| Unanswerable | 1 | abstain rate 0.000 (it cited 1 record; it should cite none) | | | |

Errors 0, timeouts 0. Latency p50 15.6 s, p95 35.4 s. Cost $0.596 for 10 questions. The
row before this one (the placeholder answerer) had recall 0.000.

Answer text (checked by hand against each question's `answer_facts`; the runner has no answer
judge yet):
- 6 correct, 4 partly correct, 0 wrong.
- Facts: 43 of 46 right, 2 partly right, 1 missing, 0 wrong.
- 3 claims with no support in the records; one matters: pub_0028 states as fact that a
  rollback "did not execute", which no record says.

### Cost

Full analysis: `eval/results/smoke/ask/20261017T013336Z/cost_analysis.md`.

| | Questions | Model calls | Average per question | Average per call |
|---|---|---|---|---|
| Hit the step limit (outliers) | 3 | 33 | $0.0882 | $0.0080 |
| Finished on their own | 7 | 35 | $0.0474 | $0.0095 |
| All | 10 | 68 | $0.0596 | $0.0088 |

Total $0.5964. The three step-limit questions took 44% of the spend. Within a long run, a
middle step costs $0.004-0.005 (most of the prompt is cached), the first two steps about
$0.012 each (not cached yet), and the forced finish $0.023-0.027: about 6x a middle step
and the most expensive call of every long run, because it paid full price on about 30,000
input tokens. Input tokens, re-sent every step and growing from about 15,600 to 34,000, are
the cost; output is small. The seven short questions' figures are the total minus the three,
not measured call by call.

### What went wrong, and why

**1. Three runs kept going after they had everything (pub_0010, pub_0028, pub_0052).** Each
used all 10 steps and was then forced to finish. In all three, every record finally cited had
been read by step 3; the 22 tool calls after that added nothing. They cost about $0.09 each
against about $0.04 for the others, and took 22-35 s. Three causes, each seen in the code or
the traces:

- **Search never says "nothing found".** `match_expression` in `src/corpusqa/store/sqlite.py`
  splits every query into single words and joins them with OR. A search for `"INT-9412"`
  runs as "int" OR "9412", so any record with either word matches. Every search comes back
  with a full page of near misses, never a clear "this does not exist". 20 of the 22 wasted
  calls were searches with quoted keys or phrases, often joined with OR, that our search
  turned into loose word matches.
- **The prompt says to keep following, and never when to stop.** `agent.md` (v3) step 2:
  "Follow the names, keys and dates that results reveal." pub_0010 chased the dependency
  tickets named inside the one it had already read; pub_0052 chased the links inside a draft
  it had already read. No rule says to finish once every part is answered from a record read.
- **The model cannot see its budget.** `agent.md` step 5 says "Stay within budget ... Finish
  on your own before you are forced to", but the loop never tells the model which step it is
  on or how many are left.

**2. No thinking.** All 33 model calls of the three long runs used 0 thinking tokens. Six of
the seven runs that finished on their own did think. `AGENT_THINKING_BUDGET = 1024` is a
maximum, not a minimum: this model decides when to think, and on these tool steps it chose
not to. The step-1 probe (2026-10-16) showed it thinks more when asked through its thinking
level (HIGH) than through the budget. A link to the looping, not a proven cause.

**3. The unanswerable question cited a record.** pub_0028's answer rightly says the p95 number
is not logged, but it cites the ticket it looked at, so the citation check counts it wrong
(an unanswerable question is right only with no ids).

**4. One malformed tool call** (in a made-up question before the run): the model wrapped a
filter's argument names in literal quotes. The tool box rejected it as bad arguments and the
loop went on, as designed.

### Changes this run leads to, and why

Each change is made and measured on its own against this run (same 10 questions, about $0.60
a run), and kept only if the smoke scores do not drop (AGENTS.md keep-rule). The results go in
the next entries here.

1. **Search: exact matching for quoted text and keys.** Why: problem 1, first cause. A quoted
   phrase or a key-shaped token must match as a phrase, so a search for something that does
   not exist comes back empty and tells the model so.
2. **Loop and prompt: a stop rule and a visible budget.** Why: problem 1, second and third
   causes. The prompt says to call `finish` once every part of the question is answered from
   a record already read (following leads is for parts still open), and each step shows the
   model its step count ("step 6 of 10").
3. **Thinking: ask for it through the thinking level.** Why: problem 2. Set the agent steps'
   thinking level to HIGH (or raise the budget), then compare thinking tokens, steps and
   scores.

### Limits of this run
- 10 questions, all inside the tune set, and some of them guided how the tools were designed.
  A perfect citation score here is a good sign, not proof; the held-out set has not been run.
- The answer text was judged by hand, once. An automated judge does not exist yet.
