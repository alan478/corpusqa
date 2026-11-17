---
name: Prompt log — why each part of the agent's prompts changed
description: One entry per change to the agent's prompt files (`src/corpusqa/prompts/agent.md`, `tools.json`): what changed, the reason in principle, and how it is measured. agent.md v7 tells the agent that the service searches the question as written before its first step (the code-side first search); v6 adds "# How to search" to avoid anchoring bias and searching only in the question's own wording; v5 tightened citing and banned handles in the answer text; v4 added the stop rule and the step budget. v5-v7 were measured together in tune runs T4-T5 and kept. Read before changing a prompt, and when writing the report. Evidence and scores are in `docs/eval-log.md`.
---

# Prompt log

Newest first. Each entry says what changed and why, in principle. The runs that led to a
change and the runs that measured it are in `docs/eval-log.md`; every text change bumps
the file's `version:` (skill `prompt-change`).

## agent.md v7 — 2026-10-17: the service searches the question first

**What changed.** One line at the top of "# How the conversation works": before the agent's
first step, the service searches the question as written and shows that search as the
agent's first tool result. Nothing else changed. The first line of "# How to search" ("The
first search results you see are a starting point, not the answer.") stays as it is: it now
names exactly those results.

**Why.** The question's own wording is the most reliable first search: it holds every
detail the asker gave, in the asker's words, and searched whole it keeps those details
together. Code runs that search every time. The model does not reliably search that way on
its own: it shortens the question, quotes phrases and adds operators, and advice in the
prompt about how to write queries has not changed how it searches. So the search moves into
code, and the prompt only tells the model what it is looking at. A ready-made first page
brings the risk of anchoring on it; v6's rules handle that (check each record's details
against the question; a look-alike is not used).

**Context.** Built together with the code-side first search (`AGENT_FIRST_SEARCH` in
`config.py`, `agent/agent.py`): one `search` call with the question exactly as asked, before
step 1, shown as a call and its result. If it fails or times out, a short note tells the
agent to search itself. With the switch off the new line is not true, so a comparison with
it off should use v6.

**Status.** Measured together with v5-v7, `tools.json` v6, the handle net and the first search in tune runs T4-T5 against T1-T2 (`docs/eval-log.md`): no wrong answer in either setting, better ids, pub_0077 fixed; with thinking HIGH all 4 unanswerable questions cite nothing. Kept. The held-out milestones (aggregates only) show the same with HIGH.

## agent.md v6 — 2026-10-17: a "# How to search" section

**What changed.** A new section between "# How to think" and "# When to stop", with four
rules:
1. The first search results the agent sees are a starting point, not the answer.
2. Before reading or citing a record, check its details against the question (who, when,
   which system, which amounts); a record on the same topic with different details is a
   look-alike and is not used.
3. When a search in the question's own words finds nothing, search again in the words a
   record would use (a more general term, another common name), and do not quote phrases
   that have not been seen in a record.
4. When a search does not help, change the words or the tool instead of searching around
   the same guess.

**Why.** In order to avoid anchoring bias: the agent must judge each candidate record
against the question itself, not by how early or how high it appeared in the results. A
record that shows up first is not more likely to be right. The same principle covers the
search words: the question's wording is the goal, not a fixed search expression, because
the records may describe the same thing in other words; searching only in the question's
words, and in exact quotes, can miss the record that holds the answer. The rules are
written as checks the agent can apply to each record and each search, because general
advice about searching well has not changed how the agent searches.

**Context.** It is the safety net for a planned change in code: one search with the
question exactly as asked, run before the agent's first step. A first page of results
raises the risk of anchoring on it, so the two are measured together on the tune set,
against a fresh baseline of v5 / `tools.json` v6.

**Status.** Measured together with v5-v7, `tools.json` v6, the handle net and the first search in tune runs T4-T5 against T1-T2 (`docs/eval-log.md`): no wrong answer in either setting, better ids, pub_0077 fixed; with thinking HIGH all 4 unanswerable questions cite nothing. Kept. The held-out milestones (aggregates only) show the same with HIGH.

## agent.md v5 and tools.json v6 — 2026-10-17: citing and handles

**What changed.** A record is cited only if it states something the question asks for and
the answer uses it; background records, records read while looking, and the record a
question names when it does not hold the asked fact are not citations; a "not found"
answer cites nothing. The answer text never contains handles (D3, R1); a record is named
by its title, source and date (and code replaces any handle that slips through).
`tools.json` v6 says the same in the `finish` tool's `citations` text.

**Why.** Citations are scored: a cited record that does not support the answer counts
against it, and a question the records cannot answer is right only with no citation.
Handles mean nothing outside one run, so they must not reach the reader.

**Status.** Measured together with v5-v7, `tools.json` v6, the handle net and the first search in tune runs T4-T5 against T1-T2 (`docs/eval-log.md`): no wrong answer in either setting, better ids, pub_0077 fixed; with thinking HIGH all 4 unanswerable questions cite nothing. Kept. The held-out milestones (aggregates only) show the same with HIGH.

## agent.md v4 — 2026-10-17: the stop rule and the step budget

**What changed.** A "# When to stop" section (finish once every part of the question is
answered from a record read or shown missing; search only for parts still missing; one
newer-version check for current or latest questions), and the agent sees its step count
and the steps left before each call.

**Why.** Without a rule for when to stop, and without seeing its budget, the agent kept
searching after it had what it needed. Measured on the smoke set (`docs/eval-log.md`, run
2): kept.
