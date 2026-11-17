---
name: Prompt decisions
description: Decisions about the agent's prompt files - agent.md v6 "# How to search" (to avoid anchoring bias, written as checks, not general advice); the stop rule of v4 (the user's completion policy with three tweaks); v3 with the field catalog in the system message and one guide line per tool; v1 of agent.md and question.md; the fixed prompt shape (role, responsibilities, how to think, numbered rules); the finish tool text; strict loaders for .md prompts and tools.json. The citing rule (v5) and the first search line (v7) are in agent.md (decisions); why each version changed, in principle, is docs/prompt-log.md.
source: runs/2026-10-16-agent-tools
---

# Prompt decisions

## 2026-10-17 agent.md v6: "# How to search", to avoid anchoring bias
- **Area:** prompts / agent
- **Chose:** a short section of checks: the first search results are a starting point; check
  a record's details against the question before reading or citing it (a look-alike is not
  used); after a search in the question's words finds nothing, search in the words a record
  would use and do not quote unseen phrases; change words or tool instead of circling one
  guess. The user proposed a "SEARCH REASONING POLICY", asked "in order to avoid anchoring
  bias should we add this in our prompt?", then "yes and also add it in the docs why we
  changed this part of the prompt, do not give example, tell it like in order to avoid
  anchoring bias" (`docs/prompt-log.md`).
- **Over:** the policy as pasted (its stop lines repeat "# When to stop"; "maximize the
  likelihood" is a goal the model cannot check itself against; general search advice is not
  followed: `tools.json` said OR is not needed and 16 of 25 searches on pub_0077 used OR).
  Its first line says "the first search results you see", so it stayed true before the
  code-side first search existed.
- **Why:** pub_0077 (the model searched in the question's quoted words, then read a
  look-alike). Measured with v5, v7 and the first search in tune runs T4-T5: kept.

## 2026-10-17 The stop rule in agent.md v4: the user's completion policy, with three tweaks
- **Area:** prompts / agent
- **Chose:** the user's "AGENT COMPLETION POLICY" as a `# When to stop` section, with three
  tweaks the user accepted ("you need to incorporate the changes you suggested, feels like
  good ones. add them"): (1) "sufficient evidence" is tied to the question's parts (each
  answered from a record read, or shown missing), not to the model's confidence; (2) one
  search for a newer version is allowed before finishing when two records contradict or the
  question asks for the current or latest state; (3) it replaces v3's "Follow the names, keys
  and dates that results reveal" (now only for parts still open) and "Stay within budget"
  instead of sitting beside them.
- **Over:** the policy as pasted (stopping on confidence could drop facts; it would forbid the
  newer-version check; beside the old lines it would contradict them); per-call duplicate
  checks (user: a future improvement).
- **Why:** run 1's long runs had every cited record by step 3 and no rule said to stop.
  Measured with the budget note as change L (agent.md decisions): kept.

## 2026-10-16 agent.md v3: the field catalog and one guide line per tool
- **Area:** prompts / agent
- **Chose** (user: the catalog into the system message; one guide line per tool; "where a
  value lives decides the tool"; cite an R# for every document of a filter; cite only what
  the answer relies on; cite nothing when nothing was found): v2 kept word for word; added in
  `# Tools` one "where a value lives" line, one line per corpus tool, a `Citing:` paragraph,
  and a last section `# Source list` with the `{catalog}` slot. `system_text(template,
  catalog)` fills the one slot (a template without exactly one is a `PromptError`); a blank
  catalog puts "(none: no records are loaded)" there. The catalog goes last: it is the same
  for every question, so the whole system message is one fixed prefix (implicit cache), and
  the rules come before the long reference list.
- **Tests:** `test_the_system_prompt_has_one_guide_line_per_tool` (every tool of
  `tools.json` has exactly one `name: ` line), the catalog-slot tests, version pins.
- **Over:** the catalog in the question message; one combined tool paragraph.

## 2026-10-16 agent.md and question.md, version 1
- **Area:** prompts
- **Chose:** `agent.md` v1: the role (answer one question from what tools return) and
  numbered rules (only tool results; answer every part or say plainly what is missing and
  list it in `not_found`; cite only what the answer relies on; copy citations exactly; call
  `finish` once; `<tool_error>` and `<working_memory>` explained). `question.md` v1: the
  question in `<question>` tags, then "Find the answer with your tools, then call finish.";
  the `{question}` slot must appear exactly once and is filled with `str.replace`, so braces
  in a question are kept.
- **Why:** no eval could run yet (the agent was not wired); the first eval came with
  `agent.md` v3. No corpus names, keys, fields or answers in them.

## 2026-10-16 Prompt shape: role, responsibilities, how to think, numbered rules
- **Area:** prompts
- **Chose:** `agent.md` v2 and `compact.md` v1 in one fixed shape: `# Role` (what the system
  is, who reads its output, how it behaves), `# Responsibilities` (numbered duties), `# How
  to think`, `# Rules` (numbered `R1..` / `C1..`, so a tool or a fix adds a rule under its
  own number), the harness protocol (`<tool_error>`, `<working_memory>`) and a `# Tools`
  anchor for one line per tool. The system prompt holds judgment only (what the records are
  like, how to weigh a result); mechanics stay where code or a tool description states them.
- **Over:** v1's single paragraph plus a flat list (user: "In the prompt we need to mention
  what that system behaves like, what are the responsibilities that it needs to follow, how
  to think about the problem and certain rules which we can build upon"); a draft that
  repeated the loop's and `finish`'s mechanics (rejected by the user).
- **Why:** every later prompt edit is a small, placed change (a new rule, a new tool line).

## 2026-10-16 The `finish` tool text, version 1
- **Area:** prompts / agent
- **Chose:** `tools.json` v1 with `finish(answer, citations, not_found)`, all three required,
  `additionalProperties: false`; the description says what it does, when to call it (once,
  last), that other calls in the same reply are not run, an example, and what comes back
  ("Answer received.", or an error to fix). Later versions: v2 (the corpus tools; citations
  take D#/R#), v3 (dates), v4 (search depth), v5 (exact search), v6 (citations text matches
  R7).
- **Over:** a one-line description (the plan asks for when / example / result).

## 2026-10-16 Strict prompt and tool-schema loaders
- **Area:** prompts
- **Chose:** `load_text_prompt(name, folder)` for `.md` (first line exactly `version: N`, no
  comment lines because `#` is a heading, blank lines around the text dropped, empty text is
  an error). `load_tool_schemas(folder)` checks `tools.json` hard: exactly `version` and a
  non-empty `tools` list; each tool exactly `name`, `description`, `parameters`; names of
  letters, digits and `_`, at most `TOOL_NAME_MAX_CHARS` (64, Gemini's limit); no duplicates;
  an object schema whose `required` names are all in `properties`. Both cached and taking a
  folder argument (tests use tmp folders). `load_prompt` (`.txt`) unchanged.
- **Over:** one loader that guesses the format; loose checks that fail later as an API 400.
- **Why:** a malformed prompt file should fail at start, with the file and the reason.
